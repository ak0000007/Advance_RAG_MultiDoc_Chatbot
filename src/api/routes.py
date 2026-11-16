"""
Route handlers for the RAG API.

SRP: only HTTP → graph → HTTP translation.
OCP: add new routers in new files, include them in app.py.
"""

import hashlib
import json
import logging
import re
import secrets
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any
import jwt

logger = logging.getLogger(__name__)
from fastapi import APIRouter, Depends, HTTPException, Request, Response, Security, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from src.config import settings
from src.telemetry import generate_trace_id, get_tracer_callbacks
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ResumeRequest,
    HealthResponse,
    Citation,
    Usage,
    StageUsage,
    ThreadSummary,
    ThreadMessage,
    ThreadDetail,
)
from src.retrieval.citations import extract_citations_from_messages
from src.api.dependencies import get_compiled_graph, get_active_graph, get_db_pool, get_graph_for_tenant
from src.api.rate_limiter import get_limiter
from src.llm.provider import is_rate_limit_error, ModelRateLimitError
from src.llm.usage import (
    TokenUsageCallbackHandler,
    record_turn_usage,
    check_tenant_quota,
    collect_from_events,
)

router = APIRouter()

security_bearer = HTTPBearer(auto_error=False)
security_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)

# In-memory JTI replay store: maps jti -> exp (unix timestamp)
# Production note: In multi-worker deployments, replace with distributed Redis store (SET key 1 EX lifetime NX).
_seen_jtis: dict[str, float] = {}


def _get_sf_jwt_public_key() -> str:
    """Load the configured RS256 public key (direct string or file path)."""
    if settings.sf_jwt_public_key:
        val = (
            settings.sf_jwt_public_key.get_secret_value()
            if hasattr(settings.sf_jwt_public_key, "get_secret_value")
            else str(settings.sf_jwt_public_key)
        )
        if val.strip():
            return val.strip()

    if settings.sf_jwt_public_key_path:
        p = Path(settings.sf_jwt_public_key_path)
        if p.exists():
            return p.read_text(encoding="utf-8").strip()

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Salesforce JWT public key is not configured.",
    )


def _check_and_store_jti(jti: str, exp: float) -> bool:
    """
    Check if JTI has already been seen. Returns False if replayed, True if new.
    Evicts expired JTIs lazily.
    """
    now = time.time()
    if len(_seen_jtis) > 1000:
        expired = [k for k, v in _seen_jtis.items() if v <= now]
        for k in expired:
            _seen_jtis.pop(k, None)

    if jti in _seen_jtis:
        return False
    _seen_jtis[jti] = exp
    return True


class AuthUser(str):
    """
    Authenticated user wrapper carrying username, tenant_id, and Salesforce context.
    Inherits from str so existing string consumers continue to function seamlessly.
    """
    username: str
    tenant_id: str
    sf_user_id: Optional[str]
    sf_role: Optional[str]
    sf_org_id: Optional[str]
    api_key_id: Optional[str]

    def __new__(
        cls,
        username: str,
        tenant_id: str = "default",
        sf_user_id: Optional[str] = None,
        sf_role: Optional[str] = None,
        sf_org_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
    ):
        obj = super().__new__(cls, username)
        obj.username = username
        obj.tenant_id = tenant_id
        obj.sf_user_id = sf_user_id
        obj.sf_role = sf_role
        obj.sf_org_id = sf_org_id
        obj.api_key_id = api_key_id
        return obj


def _verify_salesforce_jwt(token: str) -> AuthUser:
    """
    Verify short-lived signed RS256 JWT for Salesforce user identity.
    Enforces issuer, audience, max lifetime <= 120s, clock skew <= 30s,
    and in-memory JTI replay protection.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Reject alg=none or symmetric algs immediately if header specifies them
    try:
        unverified_headers = jwt.get_unverified_header(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed JWT header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    alg = unverified_headers.get("alg")
    if alg != "RS256":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid algorithm '{alg}'. Only RS256 is permitted.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    public_key = _get_sf_jwt_public_key()

    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=settings.sf_jwt_audience or None,
            issuer=settings.sf_jwt_issuer or None,
            leeway=30,  # <= 30s clock skew
            options={
                "require": ["exp", "iat", "jti", "iss", "aud"],
                "verify_exp": True,
                "verify_iat": True,
                "verify_aud": bool(settings.sf_jwt_audience),
                "verify_iss": bool(settings.sf_jwt_issuer),
            },
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid JWT: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Enforce maximum lifetime (exp - iat) <= max allowed lifetime (default 120s)
    exp = payload.get("exp")
    iat = payload.get("iat")
    if exp is None or iat is None or (exp - iat) > settings.sf_jwt_max_lifetime_seconds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token lifetime exceeds maximum allowed ({settings.sf_jwt_max_lifetime_seconds}s).",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Missing sf_user_id check
    sf_user_id = payload.get("sf_user_id")
    if not sf_user_id or not str(sf_user_id).strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required 'sf_user_id' claim.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Replay protection on jti
    jti = payload.get("jti")
    if not jti or not _check_and_store_jti(str(jti), float(exp)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token replay detected or invalid jti.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = str(sf_user_id)
    tenant_id = str(payload.get("tenant_id") or "default")
    sf_role = str(payload.get("sf_role") or "") if payload.get("sf_role") is not None else None
    sf_org_id = str(payload.get("sf_org_id") or "") if payload.get("sf_org_id") is not None else None

    return AuthUser(
        username=username,
        tenant_id=tenant_id,
        sf_user_id=str(sf_user_id),
        sf_role=sf_role,
        sf_org_id=sf_org_id,
    )


def _verify_auth_and_get_user(
    auth_bearer: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    api_key: Optional[str] = Security(security_api_key),
) -> Optional[AuthUser]:
    """
    Enforce API authentication, constant-time token comparison, and user/tenant mapping.
    When SF_AUTH_ENABLED=true, verifies RS256 Salesforce JWT and returns AuthUser with SF context.
    When SF_AUTH_ENABLED=false, follows existing API-key / secret auth rules.
    """
    token = None
    if isinstance(auth_bearer, HTTPAuthorizationCredentials):
        token = auth_bearer.credentials
    elif isinstance(api_key, str):
        token = api_key

    if getattr(settings, "sf_auth_enabled", False) is True:
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required. Please provide a valid Bearer JWT.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return _verify_salesforce_jwt(token)

    # 1. Check api_key_mapping first (maps specific tokens to {username, tenant_id})
    if settings.api_key_mapping and token:
        for mapped_key, user_info in settings.api_key_mapping.items():
            if secrets.compare_digest(token, mapped_key):
                key_id = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
                if isinstance(user_info, dict):
                    uname = user_info.get("username") or settings.sf_default_username or f"key_{key_id}"
                    tid = user_info.get("tenant_id") or "default"
                else:
                    uname = str(user_info)
                    tid = "default"
                return AuthUser(uname, tid, api_key_id=key_id)

    # 2. Check global api_secret_key with constant-time comparison
    required_key = (
        settings.api_secret_key.get_secret_value()
        if hasattr(settings.api_secret_key, "get_secret_value")
        else settings.api_secret_key
    )
    if required_key:
        if not token or not secrets.compare_digest(token, required_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing authentication credentials.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        key_id = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
        if settings.sf_default_username:
            return AuthUser(settings.sf_default_username, "default", api_key_id=key_id)
        return AuthUser(f"key_{key_id}", "default", api_key_id=key_id)

    # 3. Dev mode / unauthenticated fallback
    if settings.sf_default_username:
        return AuthUser(settings.sf_default_username, "default")
    return None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_caller_identity(auth_user: Optional[str | AuthUser]) -> tuple[str, Optional[str]]:
    """
    Resolve (caller_tenant_id, caller_user_id) from auth_user.
    caller_user_id is sf_user_id when SF auth is active, else username,
    else api_key_id in shared-key mode, else None.
    """
    if auth_user is None:
        return "default", None

    tenant_id = getattr(auth_user, "tenant_id", "default") or "default"
    sf_user_id = getattr(auth_user, "sf_user_id", None)
    if sf_user_id:
        return tenant_id, str(sf_user_id)

    username = getattr(auth_user, "username", None) or (str(auth_user) if str(auth_user).strip() else None)
    if username:
        return tenant_id, str(username)

    api_key_id = getattr(auth_user, "api_key_id", None)
    if api_key_id:
        return tenant_id, f"key_{api_key_id}"

    return tenant_id, None


async def get_graph_for_request(
    request: Request,
    auth_user: Optional[str | AuthUser] = Depends(_verify_auth_and_get_user),
):
    """
    Resolve tenant identity and return cached compiled graph instance for tenant.
    Honors test dependency overrides on get_active_graph, and falls back to default.
    """
    if hasattr(request, "app") and hasattr(request.app, "dependency_overrides"):
        if get_active_graph in request.app.dependency_overrides:
            override = request.app.dependency_overrides[get_active_graph]
            return override() if callable(override) else override

    from src.llm.credentials import resolve_credential
    caller_tenant_id, _ = _resolve_caller_identity(auth_user)
    api_key_id = getattr(auth_user, "api_key_id", None) if auth_user else None

    try:
        cred = resolve_credential(caller_tenant_id, api_key_id)
        return get_graph_for_tenant(
            tenant_id=cred.tenant_id,
            provider=cred.provider,
            model=cred.model,
            fingerprint=cred.fingerprint,
            api_key_id=api_key_id,
            use_multi_agent=settings.use_multi_agent_architecture,
        )
    except Exception:
        return get_active_graph()


async def check_rate_limit(
    request: Request,
    auth_user: Optional[str | AuthUser] = Depends(_verify_auth_and_get_user),
) -> None:
    """
    Enforce tenant-aware sliding-window rate limiting on LLM/chat operations.
    Disabled by default (RATE_LIMIT_ENABLED=false).
    """
    if not settings.rate_limit_enabled:
        return

    caller_tenant_id, caller_user_id = _resolve_caller_identity(auth_user)
    if caller_user_id:
        key = f"{caller_tenant_id}:{caller_user_id}"
    elif request.client and request.client.host:
        key = f"{caller_tenant_id}:{request.client.host}"
    else:
        key = f"{caller_tenant_id}:anonymous"

    limiter = get_limiter()
    await limiter.acquire(key)



def _resolve_identity(auth_user: Optional[str | AuthUser]) -> tuple[Optional[str], str]:
    """
    Centralized identity resolution used across /chat, /chat/stream, /chat/resume, /chat/resume/stream.
    Returns (sf_username, tenant_id).
    When DEV_MODE is false, sf_username comes strictly from auth_user or sf_default_username.
    """
    sf_user_id = getattr(auth_user, "sf_user_id", None)
    sf_auth_active = (getattr(settings, "sf_auth_enabled", False) is True) or bool(sf_user_id)

    if sf_auth_active:
        sf_username = getattr(auth_user, "username", None) or (str(auth_user) if auth_user else None)
        tenant_id = getattr(auth_user, "tenant_id", "default")
        return sf_username, tenant_id

    # Non-SF-auth mode: server-side auth identity only
    if auth_user:
        raw_username = getattr(auth_user, "username", None) or str(auth_user)
        tenant_id = getattr(auth_user, "tenant_id", "default")
    else:
        raw_username = settings.sf_default_username
        tenant_id = "default"

    sf_username = None
    if raw_username:
        sf_username = re.sub(r"[\x00-\x1f\x7f]", "", str(raw_username)).strip()[:100]
    return sf_username, tenant_id

def _build_config(
    thread_id: str,
    sf_username: str | None = None,
    trace_id: str | None = None,
    endpoint: str = "chat",
    tenant_id: str | None = None,
    sf_user_id: str | None = None,
    sf_role: str | None = None,
    sf_org_id: str | None = None,
    sf_auth_active: bool = False,
) -> dict:
    tid = trace_id or generate_trace_id()
    metadata = {
        "thread_id": thread_id,
        "trace_id": tid,
        "endpoint": endpoint,
    }
    if sf_username:
        metadata["sf_username"] = sf_username
    if tenant_id:
        metadata["tenant_id"] = tenant_id
    if sf_user_id:
        metadata["sf_user_id"] = sf_user_id
    if sf_role:
        metadata["sf_role"] = sf_role
    if sf_org_id:
        metadata["sf_org_id"] = sf_org_id
    if sf_auth_active:
        metadata["sf_auth_active"] = True

    tags = ["advance-rag", "langgraph", endpoint]
    callbacks = get_tracer_callbacks(
        thread_id=thread_id,
        run_name=f"rag-turn-{thread_id[:8]}",
        tags=tags,
        metadata=metadata,
        trace_id=tid,
    )
    usage_handler = TokenUsageCallbackHandler(
        default_provider=getattr(settings, "llm_provider", ""),
        default_model=getattr(settings, "llm_model", ""),
    )
    callbacks.append(usage_handler)

    configurable: dict = {
        "thread_id": thread_id,
        "trace_id": tid,
        "usage_handler": usage_handler,
    }
    if sf_username:
        configurable["sf_username"] = sf_username
    if tenant_id:
        configurable["tenant_id"] = tenant_id
    if sf_user_id:
        configurable["sf_user_id"] = sf_user_id
    if sf_role:
        configurable["sf_role"] = sf_role
    if sf_org_id:
        configurable["sf_org_id"] = sf_org_id
    if sf_auth_active:
        configurable["sf_auth_active"] = True

    return {
        "configurable": configurable,
        "callbacks": callbacks,
        "metadata": metadata,
        "tags": tags,
        "run_name": f"rag-turn-{thread_id[:8]}",
    }


async def _assert_thread_access(
    graph,
    thread_id: str,
    auth_user: Optional[str | AuthUser],
    endpoint: str = "chat",
    snapshot: Any = None,
    trace_id: Optional[str] = None,
    pool: Any = None,
) -> Any:
    """
    Uniform thread ownership guard for /chat, /chat/stream, /chat/resume, /chat/resume/stream.
    Runs BEFORE any graph invocation or history read.
    """
    if pool is None:
        pool = get_db_pool()
    is_resume = endpoint in ("resume", "resume_stream")
    not_found_detail = (
        f"Thread '{thread_id}' is not paused or does not exist."
        if is_resume
        else f"Thread '{thread_id}' not found."
    )

    if snapshot is None:
        if not hasattr(graph, "aget_state"):
            return None
        check_config = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(check_config)

    # 1. Thread does not exist yet
    if not snapshot or not snapshot.values:
        if is_resume:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=not_found_detail)
        return snapshot

    # 2. Thread exists: inspect stored owner
    stored_owner_tenant = snapshot.values.get("owner_tenant_id")
    stored_owner_user = snapshot.values.get("owner_user_id")

    caller_tenant_id, caller_user_id = _resolve_caller_identity(auth_user)
    thread_hash = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:16]

    # Open mode with no key: threads are unowned/anonymous; ownership cannot be enforced
    has_auth_configured = bool(
        settings.sf_auth_enabled
        or settings.api_key_mapping
        or (
            settings.api_secret_key.get_secret_value()
            if hasattr(settings.api_secret_key, "get_secret_value")
            else settings.api_secret_key
        )
    )
    if auth_user is None and not has_auth_configured:
        logger.warning(
            "Unauthenticated request in open mode on thread %s; thread ownership cannot be enforced.",
            thread_hash,
        )
        return snapshot

    # Handle unowned legacy threads
    if not stored_owner_tenant and not stored_owner_user:
        if settings.allow_unowned_threads:
            logger.warning(
                "Access granted to legacy unowned thread %s because ALLOW_UNOWNED_THREADS=true. trace_id=%s",
                thread_hash, trace_id
            )
            return snapshot
        else:
            logger.warning(
                "Access denied to legacy unowned thread %s (ALLOW_UNOWNED_THREADS=false). trace_id=%s",
                thread_hash, trace_id
            )
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=not_found_detail)

    # Ownership comparison
    is_owner = (stored_owner_tenant == caller_tenant_id) and (stored_owner_user == caller_user_id)

    # Check approver role if configured:
    # Approvers may inspect (get_thread) and decide (resume/resume_stream) on colleagues' threads
    # ONLY when the thread currently has a pending approval interrupt.
    # Approvers may NEVER chat (chat/chat_stream) on another colleague's thread.
    is_approver_action = endpoint in ("resume", "resume_stream", "get_thread")
    is_approver = False
    if is_approver_action and not is_owner and settings.approver_roles:
        caller_role = getattr(auth_user, "sf_role", None)
        if caller_role and caller_role in settings.approver_roles and stored_owner_tenant == caller_tenant_id:
            interrupt_values = [
                intr.value
                for task in (getattr(snapshot, "tasks", None) or [])
                for intr in (getattr(task, "interrupts", None) or [])
            ]
            if interrupt_values:
                is_approver = True
            elif endpoint == "get_thread":
                # Self-healing: approver accessed thread but interrupt is no longer pending in checkpointer.
                # Heal threads.interrupted = FALSE to clear phantom from approver's list.
                await _sync_thread_interrupted(pool, thread_id, False)

    if not is_owner and not is_approver:
        logger.warning(
            "Thread access denied. trace_id=%s caller_user=%s caller_tenant=%s thread_hash=%s endpoint=%s",
            trace_id, caller_user_id, caller_tenant_id, thread_hash, endpoint
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=not_found_detail)

    return snapshot


def _prepare_chat_context(
    request: ChatRequest,
    auth_user: Optional[str | AuthUser],
    trace_id: Optional[str] = None,
) -> tuple[dict, dict, str]:
    """
    Standardize message sequence, authenticated sf_username, tenant_id, owner context, and thread config.
    """
    messages = []

    # Stateless history (only used when no thread_id is provided)
    if not request.thread_id and request.history:
        for msg in request.history:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("type", ""))
                content = msg.get("content", "")
                if role in ["human", "user"]:
                    messages.append(HumanMessage(content=content))
                elif role in ["ai", "assistant"]:
                    messages.append(AIMessage(content=content))
            else:
                messages.append(msg)

    messages.append(HumanMessage(content=request.question))

    sf_user_id = getattr(auth_user, "sf_user_id", None)
    sf_role = getattr(auth_user, "sf_role", None)
    sf_org_id = getattr(auth_user, "sf_org_id", None)
    sf_auth_active = (getattr(settings, "sf_auth_enabled", False) is True) or bool(sf_user_id)

    sf_username, tenant_id = _resolve_identity(auth_user)

    # Body-supplied sf_username is honoured ONLY when DEV_MODE=true
    if settings.dev_mode and request.sf_username and not sf_auth_active:
        raw_body_user = re.sub(r"[\x00-\x1f\x7f]", "", str(request.sf_username)).strip()[:100]
        if raw_body_user:
            logger.warning("DEV_MODE: using body-supplied sf_username '%s'", raw_body_user)
            sf_username = raw_body_user
    elif request.sf_username:
        logger.debug("Non-DEV_MODE: request.sf_username ignored.")

    owner_tenant_id, owner_user_id = _resolve_caller_identity(auth_user)

    thread_id = request.thread_id or str(uuid.uuid4())
    tid = trace_id or getattr(request, "trace_id", None) or generate_trace_id()
    config = _build_config(
        thread_id,
        sf_username,
        trace_id=tid,
        tenant_id=tenant_id,
        sf_user_id=sf_user_id,
        sf_role=sf_role,
        sf_org_id=sf_org_id,
        sf_auth_active=sf_auth_active,
    )
    inputs = {
        "question": request.question,
        "messages": messages,
        "citations": [],
        "tenant_id": tenant_id,
    }
    if sf_username:
        inputs["sf_username"] = sf_username
    if sf_user_id:
        inputs["sf_user_id"] = sf_user_id
    if sf_role:
        inputs["sf_role"] = sf_role
    if sf_org_id:
        inputs["sf_org_id"] = sf_org_id

    # Stamp immutable thread owner (stripped of any client-influenced fields)
    if owner_tenant_id:
        inputs["owner_tenant_id"] = owner_tenant_id
    if owner_user_id:
        inputs["owner_user_id"] = owner_user_id

    return inputs, config, thread_id


async def upsert_thread_record(
    pool: Any,
    thread_id: str,
    tenant_id: str,
    owner_user_id: Optional[str],
    question: Optional[str],
    turn_usage: Any,
    interrupted: bool = False,
) -> None:
    """
    Upsert metadata into the queryable `threads` index table.
    - Sets initial title = first 60 chars of user query (only on insert)
    - Updates last_activity = now(), increments message_count
    - Accumulates input_tokens, output_tokens, cost_usd
    - Updates interrupted status (True if awaiting HITL approval, False on completion)
    Best-effort: DB failure logs warning and NEVER crashes chat execution.
    """
    if not pool:
        return

    raw_title = question.strip() if question and isinstance(question, str) else "Conversation"
    title = " ".join(raw_title.split())[:60] or "Conversation"

    inp = getattr(turn_usage, "input_tokens", 0) or 0
    out = getattr(turn_usage, "output_tokens", 0) or 0
    cost = getattr(turn_usage, "cost_usd", 0.0) or 0.0

    query = """
        INSERT INTO threads (
            thread_id, tenant_id, owner_user_id, title, last_activity,
            input_tokens, output_tokens, cost_usd, message_count, interrupted, created_at
        ) VALUES (
            %(thread_id)s, %(tenant_id)s, %(owner_user_id)s, %(title)s, now(),
            %(input_tokens)s, %(output_tokens)s, %(cost_usd)s, 1, %(interrupted)s, now()
        )
        ON CONFLICT (thread_id) DO UPDATE SET
            last_activity = now(),
            message_count = threads.message_count + 1,
            input_tokens = threads.input_tokens + EXCLUDED.input_tokens,
            output_tokens = threads.output_tokens + EXCLUDED.output_tokens,
            cost_usd = threads.cost_usd + EXCLUDED.cost_usd,
            interrupted = EXCLUDED.interrupted;
    """
    try:
        from src.audit.log import _get_conn

        async with _get_conn(pool) as conn:
            await conn.execute(
                query,
                {
                    "thread_id": thread_id,
                    "tenant_id": tenant_id,
                    "owner_user_id": owner_user_id,
                    "title": title,
                    "input_tokens": inp,
                    "output_tokens": out,
                    "cost_usd": cost,
                    "interrupted": interrupted,
                },
            )
    except Exception as exc:
        logger.warning("Failed to upsert thread record for %s: %s", thread_id, exc)


async def _sync_thread_interrupted(
    pool: Any,
    thread_id: str,
    interrupted: bool,
) -> None:
    """
    Surgically update the `interrupted` status in the queryable `threads` index table.
    Ensures threads.interrupted accurately reflects live checkpointer status.
    Runs best-effort; logs warning and never crashes caller.
    """
    if not pool:
        return
    try:
        from src.audit.log import _get_conn

        async with _get_conn(pool) as conn:
            await conn.execute(
                """
                UPDATE threads
                SET interrupted = %(interrupted)s, last_activity = now()
                WHERE thread_id = %(thread_id)s;
                """,
                {"thread_id": thread_id, "interrupted": interrupted},
            )
    except Exception as exc:
        logger.warning("Failed to sync interrupted status for thread %s: %s", thread_id, exc)


def _extract_clean_thread_messages(raw_messages: list) -> list[ThreadMessage]:
    """
    Extract human-readable conversation dialogue from LangGraph checkpoint messages.
    Strictly strips:
      - Tool call execution messages / function outputs
      - System prompts
      - Intermediate CRAG grading JSONs
      - ApprovalRequest internal state dicts
    """
    clean: list[ThreadMessage] = []
    for m in raw_messages:
        mtype = getattr(m, "type", "")
        if mtype in ("human", "user") or isinstance(m, HumanMessage):
            content = getattr(m, "content", "")
            if isinstance(content, list):
                text = " ".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in content])
            else:
                text = str(content)
            text = text.strip()
            if text:
                clean.append(ThreadMessage(role="user", content=text))

        elif mtype in ("ai", "assistant") or isinstance(m, AIMessage):
            # Skip messages that are pure tool calls without text content
            if getattr(m, "tool_calls", None) and not getattr(m, "content", None):
                continue

            content = getattr(m, "content", "")
            if isinstance(content, list):
                text = " ".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in content])
            else:
                text = str(content)
            text = text.strip()

            # Filter out CRAG structured output JSON dumps or internal approval JSONs
            if text.startswith('{"is_relevant":') or text.startswith('{"relevant":') or text.startswith('{"domain":'):
                continue
            if text.startswith('{"__requires_approval__":'):
                continue

            if text:
                clean.append(ThreadMessage(role="assistant", content=text))

    return clean


async def _run_and_respond(graph, inputs, config: dict, thread_id: str, pool: Any = None) -> ChatResponse:
    """
    Stream the graph to completion (or interrupt), then inspect state.

    - Normal completion → ChatResponse(answer=...)
    - interrupt() fired → ChatResponse(interrupted=True, approval_request=...)
    """
    if pool is None:
        pool = get_db_pool()
    trace_id = config.get("configurable", {}).get("trace_id") or generate_trace_id()

    usage_handler = config.get("configurable", {}).get("usage_handler")
    if not usage_handler:
        usage_handler = TokenUsageCallbackHandler(
            default_provider=getattr(settings, "llm_provider", ""),
            default_model=getattr(settings, "llm_model", ""),
        )
        config.setdefault("callbacks", []).append(usage_handler)
        config.setdefault("configurable", {})["usage_handler"] = usage_handler

    # Run the graph; stream_mode="values" yields full state after each step.
    # astream finishes even when interrupted (GraphInterrupt is caught internally).
    try:
        try:
            async for _ in graph.astream(inputs, config, stream_mode="values"):
                pass
        except Exception as exc:
            if is_rate_limit_error(exc):
                raise HTTPException(
                    status_code=429,
                    detail="Upstream model rate limit or quota exceeded. Please retry later.",
                    headers={"Retry-After": "60"},
                )
            raise
    finally:
        # Guarantee interrupted status synchronization even on error/rejection
        try:
            cur_snapshot = await graph.aget_state(config)
            has_interrupt = bool([
                intr.value
                for task in (getattr(cur_snapshot, "tasks", None) or [])
                for intr in (getattr(task, "interrupts", None) or [])
            ])
            await _sync_thread_interrupted(pool, thread_id, has_interrupt)
        except Exception as cleanup_exc:
            logger.warning("Runner cleanup failed to sync interrupted for %s: %s", thread_id, cleanup_exc)

    # Extract multi-stage token accounting
    turn_usage = usage_handler.get_turn_usage()
    usage_schema = Usage(
        input_tokens=turn_usage.input_tokens,
        output_tokens=turn_usage.output_tokens,
        total_tokens=turn_usage.total_tokens,
        cost_usd=turn_usage.cost_usd,
        estimated=turn_usage.estimated,
        by_stage=[
            StageUsage(
                node=s.node,
                model=s.model,
                provider=s.provider,
                input_tokens=s.input_tokens,
                output_tokens=s.output_tokens,
                cost_usd=s.cost_usd,
                estimated=s.estimated,
            )
            for s in turn_usage.by_stage
        ],
    )

    # Inspect final / paused state
    snapshot = await graph.aget_state(config)

    # Collect interrupt payloads from all pending tasks
    interrupt_values = [
        intr.value
        for task in (snapshot.tasks or [])
        for intr in (getattr(task, "interrupts", None) or [])
    ]
    is_interrupted = bool(interrupt_values)

    # Persist usage to database (best-effort)
    tenant_id = config.get("configurable", {}).get("tenant_id") or "default"
    user_id = (
        config.get("configurable", {}).get("sf_username")
        or config.get("configurable", {}).get("sf_user_id")
    )
    await record_turn_usage(pool, thread_id, tenant_id, user_id, turn_usage, trace_id=trace_id)
    question = inputs.get("question") if isinstance(inputs, dict) else None
    await upsert_thread_record(pool, thread_id, tenant_id, user_id, question, turn_usage, interrupted=is_interrupted)

    if interrupt_values:
        return ChatResponse(
            interrupted=True,
            thread_id=thread_id,
            approval_request=interrupt_values[0],
            citations=[],
            trace_id=trace_id,
            usage=usage_schema,
        )

    # Citations must strictly belong to the current turn (never leaked from prior turns or CRM actions)
    messages = snapshot.values.get("messages", []) if snapshot and snapshot.values else []
    raw_citations = extract_citations_from_messages(messages, current_turn_only=True)

    citations_list: list[Citation] = []
    for c in raw_citations:
        try:
            citations_list.append(Citation(**c) if isinstance(c, dict) else c)
        except Exception:
            continue

    citation_verification = snapshot.values.get("citation_verification") if snapshot and snapshot.values else None

    return ChatResponse(
        answer=snapshot.values.get("answer", "") if snapshot and snapshot.values else "",
        citations=citations_list,
        citation_verification=citation_verification,
        trace_id=trace_id,
        usage=usage_schema,
    )


async def _stream_graph_events(graph, inputs, config: dict, thread_id: str, pool: Any = None):
    """
    Generator yielding Server-Sent Events (SSE).
    Follows text/event-stream spec:
      event: <type>\\n
      data: <json>\\n\\n

    Events emitted:
      - status: Intermediate progress (intent routing, tool start/end).
      - token: Incremental LLM text tokens.
      - interrupt: Graph paused waiting for human approval.
      - done: Graph execution completed with final answer and thread_id.
      - error: Unhandled exception during graph execution.
    """
    if pool is None:
        pool = get_db_pool()
    accumulated_answer: list[str] = []
    streamed_tokens: int = 0
    trace_id = config.get("configurable", {}).get("trace_id") or generate_trace_id()

    usage_handler = config.get("configurable", {}).get("usage_handler")
    if not usage_handler:
        usage_handler = TokenUsageCallbackHandler(
            default_provider=getattr(settings, "llm_provider", ""),
            default_model=getattr(settings, "llm_model", ""),
        )
        config.setdefault("callbacks", []).append(usage_handler)
        config.setdefault("configurable", {})["usage_handler"] = usage_handler

    captured_events: list[dict[str, Any]] = []

    try:
        async for event in graph.astream_events(inputs, config, version="v2"):
            captured_events.append(event)
            kind = event.get("event")
            node = event.get("metadata", {}).get("langgraph_node", "")

            # 1. Incremental Chat Model Tokens
            if kind == "on_chat_model_stream":
                # Only stream tokens from the conversational agent node (ignore router, tools, graders, rewriters)
                if node != "agent":
                    continue

                chunk = event.get("data", {}).get("chunk")
                if chunk is not None:
                    # Skip tool call chunks (function calling fragments)
                    if getattr(chunk, "tool_call_chunks", None):
                        continue

                    content = getattr(chunk, "content", None)
                    text = ""
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        texts = [
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in content
                        ]
                        text = "".join(texts)

                    if text:
                        accumulated_answer.append(text)
                        streamed_tokens += 1
                        payload = json.dumps({"event": "token", "chunk": text})
                        yield f"event: token\ndata: {payload}\n\n"

                        if streamed_tokens >= settings.max_streaming_tokens:
                            payload = json.dumps({
                                "event": "status",
                                "stage": "token_limit",
                                "message": f"Token streaming safeguard reached ({settings.max_streaming_tokens} tokens).",
                            })
                            yield f"event: status\ndata: {payload}\n\n"
                            break

            # 2. Tool Start Progress
            elif kind == "on_tool_start":
                tool_name = event.get("name", "tool")
                accumulated_answer.clear()
                payload = json.dumps({
                    "event": "status",
                    "stage": "tool_start",
                    "tool": tool_name,
                    "message": f"Calling {tool_name}...",
                })
                yield f"event: status\ndata: {payload}\n\n"

            # 3. Tool End Progress
            elif kind == "on_tool_end":
                tool_name = event.get("name", "tool")
                payload = json.dumps({
                    "event": "status",
                    "stage": "tool_end",
                    "tool": tool_name,
                    "message": f"Completed {tool_name}",
                })
                yield f"event: status\ndata: {payload}\n\n"

            # 4. Intent Routing Status
            elif kind == "on_chain_start" and node == "classify_intent":
                payload = json.dumps({
                    "event": "status",
                    "stage": "routing",
                    "message": "Classifying request intent...",
                })
                yield f"event: status\ndata: {payload}\n\n"

        # After astream_events completes, calculate multi-stage usage and persist
        turn_usage = usage_handler.get_turn_usage()
        if not turn_usage.by_stage and captured_events:
            fallback_usage = collect_from_events(
                captured_events,
                default_model=getattr(settings, "llm_model", "gemini-3.6-flash"),
                default_provider=getattr(settings, "llm_provider", ""),
            )
            if fallback_usage.by_stage:
                turn_usage = fallback_usage

        usage_dict = turn_usage.to_dict()

        # Inspect checkpointer state
        snapshot = await graph.aget_state(config)
        interrupt_values = [
            intr.value
            for task in (snapshot.tasks or [])
            for intr in (getattr(task, "interrupts", None) or [])
        ]
        is_interrupted = bool(interrupt_values)

        tenant_id = config.get("configurable", {}).get("tenant_id") or "default"
        user_id = (
            config.get("configurable", {}).get("sf_username")
            or config.get("configurable", {}).get("sf_user_id")
        )
        if pool is None:
            pool = get_db_pool()
        await record_turn_usage(pool, thread_id, tenant_id, user_id, turn_usage, trace_id=trace_id)
        question = inputs.get("question") if isinstance(inputs, dict) else None
        await upsert_thread_record(pool, thread_id, tenant_id, user_id, question, turn_usage, interrupted=is_interrupted)

        # Emit dedicated usage event prior to terminal event
        payload = json.dumps({"event": "usage", **usage_dict})
        yield f"event: usage\ndata: {payload}\n\n"

        if interrupt_values:
            payload = json.dumps({
                "event": "interrupt",
                "interrupted": True,
                "thread_id": thread_id,
                "approval_request": interrupt_values[0],
                "trace_id": trace_id,
                "usage": usage_dict,
            })
            yield f"event: interrupt\ndata: {payload}\n\n"
        else:
            final_answer = (
                snapshot.values.get("answer") if snapshot and snapshot.values else ""
            ) or "".join(accumulated_answer)

            # Citations must strictly belong to the current turn (never leaked from prior turns or CRM actions)
            messages = snapshot.values.get("messages", []) if snapshot and snapshot.values else []
            raw_citations = extract_citations_from_messages(messages, current_turn_only=True)

            citation_verification = snapshot.values.get("citation_verification") if snapshot and snapshot.values else None

            payload = json.dumps({
                "event": "done",
                "interrupted": False,
                "thread_id": thread_id,
                "answer": final_answer,
                "citations": raw_citations,
                "citation_verification": citation_verification,
                "trace_id": trace_id,
                "usage": usage_dict,
            })
            yield f"event: done\ndata: {payload}\n\n"

    except Exception as exc:
        if is_rate_limit_error(exc):
            payload = json.dumps({
                "event": "error",
                "detail": "Upstream model rate limit or quota exceeded. Please retry later.",
                "retry_after": 60,
            })
            yield f"event: error\ndata: {payload}\n\n"
        else:
            payload = json.dumps({"event": "error", "detail": str(exc)})
            yield f"event: error\ndata: {payload}\n\n"
    finally:
        # Guarantee interrupted status synchronization even on client disconnect (GeneratorExit) or stream error
        try:
            latest_snapshot = await graph.aget_state(config)
            has_intr = bool([
                intr.value
                for task in (getattr(latest_snapshot, "tasks", None) or [])
                for intr in (getattr(task, "interrupts", None) or [])
            ])
            await _sync_thread_interrupted(pool, thread_id, has_intr)
        except Exception as cleanup_exc:
            logger.warning("Stream cleanup failed to sync interrupted for %s: %s", thread_id, cleanup_exc)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    auth_user: Optional[str] = Depends(_verify_auth_and_get_user),
    _rate_limit: None = Depends(check_rate_limit),
    graph=Depends(get_graph_for_request),
    pool: Any = Depends(get_db_pool),
):
    """
    Invoke the RAG graph with user question + history.

    Returns:
        - ChatResponse(answer=...) on normal completion.
        - ChatResponse(interrupted=True, thread_id=..., approval_request=...) when
          the graph pauses for human approval (Salesforce update confirmation).
          The frontend must display the approval UI and POST to /chat/resume.
    """
    trace_id = getattr(request, "trace_id", None) or generate_trace_id()
    if request.thread_id:
        await _assert_thread_access(graph, request.thread_id, auth_user, endpoint="chat", trace_id=trace_id, pool=pool)
    inputs, config, thread_id = _prepare_chat_context(request, auth_user, trace_id=trace_id)

    # Check tenant token quota
    tenant_id = config.get("configurable", {}).get("tenant_id") or "default"
    allowed, reason = await check_tenant_quota(
        pool,
        tenant_id=tenant_id,
        quota_daily=settings.tenant_quota_daily_tokens,
        quota_monthly=settings.tenant_quota_monthly_tokens,
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=reason or "Tenant token quota exceeded. Please contact your administrator.",
        )

    return await _run_and_respond(graph, inputs, config, thread_id, pool=pool)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    auth_user: Optional[str] = Depends(_verify_auth_and_get_user),
    _rate_limit: None = Depends(check_rate_limit),
    graph=Depends(get_graph_for_request),
    pool: Any = Depends(get_db_pool),
):
    """
    Stream the RAG graph execution via Server-Sent Events (SSE).
    Yields 'status', 'token', 'interrupt', and 'done' events.
    """
    trace_id = getattr(request, "trace_id", None) or generate_trace_id()
    if request.thread_id:
        await _assert_thread_access(graph, request.thread_id, auth_user, endpoint="chat_stream", trace_id=trace_id, pool=pool)
    inputs, config, thread_id = _prepare_chat_context(request, auth_user, trace_id=trace_id)

    # Check tenant token quota
    tenant_id = config.get("configurable", {}).get("tenant_id") or "default"
    allowed, reason = await check_tenant_quota(
        pool,
        tenant_id=tenant_id,
        quota_daily=settings.tenant_quota_daily_tokens,
        quota_monthly=settings.tenant_quota_monthly_tokens,
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=reason or "Tenant token quota exceeded. Please contact your administrator.",
        )

    return StreamingResponse(
        _stream_graph_events(graph, inputs, config, thread_id, pool=pool),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/resume", response_model=ChatResponse)
async def resume_chat(
    request: ResumeRequest,
    auth_user: Optional[str] = Depends(_verify_auth_and_get_user),
    _rate_limit: None = Depends(check_rate_limit),
    graph=Depends(get_graph_for_request),
    pool: Any = Depends(get_db_pool),
):
    """
    Resume a graph that was interrupted for human approval.

    The frontend posts the thread_id (from the interrupted ChatResponse)
    and the human's decision ('Approve' or 'Reject').

    Returns the same ChatResponse envelope as /chat.
    """
    trace_id = getattr(request, "trace_id", None) or generate_trace_id()
    base_config = _build_config(request.thread_id, trace_id=trace_id, endpoint="resume")

    # Verify the thread actually exists and is interrupted
    snapshot = await graph.aget_state(base_config)
    if not snapshot or not snapshot.next:
        raise HTTPException(
            status_code=404,
            detail=f"Thread '{request.thread_id}' is not paused or does not exist.",
        )

    await _assert_thread_access(graph, request.thread_id, auth_user, endpoint="resume", snapshot=snapshot, trace_id=trace_id, pool=pool)

    # Resume execution config built strictly from CALLER's verified identity
    caller_sf_username, caller_tenant_id = _resolve_identity(auth_user)
    caller_sf_user_id = getattr(auth_user, "sf_user_id", None)
    caller_sf_role = getattr(auth_user, "sf_role", None)
    caller_sf_org_id = getattr(auth_user, "sf_org_id", None)
    sf_auth_active = bool(settings.sf_auth_enabled or caller_sf_user_id)

    # Check tenant token quota
    allowed, reason = await check_tenant_quota(
        pool,
        tenant_id=caller_tenant_id or "default",
        quota_daily=settings.tenant_quota_daily_tokens,
        quota_monthly=settings.tenant_quota_monthly_tokens,
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=reason or "Tenant token quota exceeded. Please contact your administrator.",
        )

    config = _build_config(
        request.thread_id,
        caller_sf_username,
        trace_id=trace_id,
        endpoint="resume",
        tenant_id=caller_tenant_id,
        sf_user_id=caller_sf_user_id,
        sf_role=caller_sf_role,
        sf_org_id=caller_sf_org_id,
        sf_auth_active=sf_auth_active,
    )

    try:
        return await _run_and_respond(
            graph,
            Command(resume=request.decision),
            config,
            request.thread_id,
            pool=pool,
        )
    finally:
        try:
            latest_snapshot = await graph.aget_state(config)
            still_interrupted = bool([
                intr.value
                for task in (getattr(latest_snapshot, "tasks", None) or [])
                for intr in (getattr(task, "interrupts", None) or [])
            ])
            await _sync_thread_interrupted(pool, request.thread_id, still_interrupted)
        except Exception as exc:
            logger.warning("Resume cleanup failed to inspect snapshot for %s: %s", request.thread_id, exc)


@router.post("/chat/resume/stream")
async def resume_chat_stream(
    request: ResumeRequest,
    auth_user: Optional[str] = Depends(_verify_auth_and_get_user),
    _rate_limit: None = Depends(check_rate_limit),
    graph=Depends(get_graph_for_request),
    pool: Any = Depends(get_db_pool),
):
    """
    Resume an interrupted graph and stream results via Server-Sent Events (SSE).
    """
    trace_id = getattr(request, "trace_id", None) or generate_trace_id()
    base_config = _build_config(request.thread_id, trace_id=trace_id, endpoint="resume_stream")

    snapshot = await graph.aget_state(base_config)
    if not snapshot or not snapshot.next:
        raise HTTPException(
            status_code=404,
            detail=f"Thread '{request.thread_id}' is not paused or does not exist.",
        )

    await _assert_thread_access(graph, request.thread_id, auth_user, endpoint="resume_stream", snapshot=snapshot, trace_id=trace_id, pool=pool)

    # Resume execution config built strictly from CALLER's verified identity
    caller_sf_username, caller_tenant_id = _resolve_identity(auth_user)
    caller_sf_user_id = getattr(auth_user, "sf_user_id", None)
    caller_sf_role = getattr(auth_user, "sf_role", None)
    caller_sf_org_id = getattr(auth_user, "sf_org_id", None)
    sf_auth_active = bool(settings.sf_auth_enabled or caller_sf_user_id)

    # Check tenant token quota
    allowed, reason = await check_tenant_quota(
        pool,
        tenant_id=caller_tenant_id or "default",
        quota_daily=settings.tenant_quota_daily_tokens,
        quota_monthly=settings.tenant_quota_monthly_tokens,
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=reason or "Tenant token quota exceeded. Please contact your administrator.",
        )

    config = _build_config(
        request.thread_id,
        caller_sf_username,
        trace_id=trace_id,
        endpoint="resume_stream",
        tenant_id=caller_tenant_id,
        sf_user_id=caller_sf_user_id,
        sf_role=caller_sf_role,
        sf_org_id=caller_sf_org_id,
        sf_auth_active=sf_auth_active,
    )

    inputs = Command(resume=request.decision)
    return StreamingResponse(
        _stream_graph_events(graph, inputs, config, request.thread_id, pool=pool),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/audit/{thread_id}")
async def get_thread_audit_log(
    thread_id: str,
    auth_user: Optional[str] = Depends(_verify_auth_and_get_user),
    pool=Depends(get_db_pool),
):
    """
    Fetch write audit log entries for a thread scoped strictly to caller's sf_username.
    """
    if not pool:
        return []
    from src.audit.log import get_audit_records_by_thread

    sf_username = auth_user or settings.sf_default_username or ""
    records = await get_audit_records_by_thread(pool, thread_id, sf_username)
    return records


@router.get("/metrics")
def get_metrics(auth_user: Optional[str] = Depends(_verify_auth_and_get_user)):
    """
    Prometheus scraper endpoint. Protected via API key when api_secret_key is configured.
    """
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/threads", response_model=list[ThreadSummary])
async def list_threads(
    limit: int = 50,
    offset: int = 0,
    auth_user: Optional[str | AuthUser] = Depends(_verify_auth_and_get_user),
    pool=Depends(get_db_pool),
):
    """
    List conversation threads scoped strictly to caller's tenant and user.
    Cross-tenant isolation: A tenant can never view another tenant's threads.
    """
    if not pool:
        return []

    caller_tenant_id, caller_user_id = _resolve_caller_identity(auth_user)

    is_approver = False
    if settings.approver_roles:
        caller_role = getattr(auth_user, "sf_role", None)
        if caller_role and caller_role in settings.approver_roles:
            is_approver = True

    has_auth_configured = bool(
        settings.sf_auth_enabled
        or settings.api_key_mapping
        or (
            settings.api_secret_key.get_secret_value()
            if hasattr(settings.api_secret_key, "get_secret_value")
            else settings.api_secret_key
        )
    )
    allow_unowned = bool(
        settings.allow_unowned_threads
        or (auth_user is None and not has_auth_configured)
    )

    try:
        from src.audit.log import _get_conn

        async with _get_conn(pool) as conn:
            if is_approver:
                if caller_user_id:
                    query = """
                        SELECT thread_id, tenant_id, owner_user_id, title, last_activity,
                               input_tokens, output_tokens, cost_usd, message_count, interrupted
                        FROM threads
                        WHERE tenant_id = %(tenant_id)s
                          AND (owner_user_id = %(owner_user_id)s OR owner_user_id IS NULL OR interrupted = TRUE)
                          AND (%(allow_unowned)s OR owner_user_id IS NOT NULL)
                        ORDER BY last_activity DESC
                        LIMIT %(limit)s OFFSET %(offset)s;
                    """
                    params = {
                        "tenant_id": caller_tenant_id,
                        "owner_user_id": caller_user_id,
                        "allow_unowned": allow_unowned,
                        "limit": limit,
                        "offset": offset,
                    }
                else:
                    query = """
                        SELECT thread_id, tenant_id, owner_user_id, title, last_activity,
                               input_tokens, output_tokens, cost_usd, message_count, interrupted
                        FROM threads
                        WHERE tenant_id = %(tenant_id)s
                          AND (owner_user_id IS NULL OR interrupted = TRUE)
                          AND (%(allow_unowned)s OR owner_user_id IS NOT NULL)
                        ORDER BY last_activity DESC
                        LIMIT %(limit)s OFFSET %(offset)s;
                    """
                    params = {
                        "tenant_id": caller_tenant_id,
                        "allow_unowned": allow_unowned,
                        "limit": limit,
                        "offset": offset,
                    }
            else:
                if caller_user_id:
                    query = """
                        SELECT thread_id, tenant_id, owner_user_id, title, last_activity,
                               input_tokens, output_tokens, cost_usd, message_count, interrupted
                        FROM threads
                        WHERE tenant_id = %(tenant_id)s
                          AND (owner_user_id = %(owner_user_id)s OR owner_user_id IS NULL)
                          AND (%(allow_unowned)s OR owner_user_id IS NOT NULL)
                        ORDER BY last_activity DESC
                        LIMIT %(limit)s OFFSET %(offset)s;
                    """
                    params = {
                        "tenant_id": caller_tenant_id,
                        "owner_user_id": caller_user_id,
                        "allow_unowned": allow_unowned,
                        "limit": limit,
                        "offset": offset,
                    }
                else:
                    query = """
                        SELECT thread_id, tenant_id, owner_user_id, title, last_activity,
                               input_tokens, output_tokens, cost_usd, message_count, interrupted
                        FROM threads
                        WHERE tenant_id = %(tenant_id)s
                          AND owner_user_id IS NULL
                          AND (%(allow_unowned)s OR owner_user_id IS NOT NULL)
                        ORDER BY last_activity DESC
                        LIMIT %(limit)s OFFSET %(offset)s;
                    """
                    params = {
                        "tenant_id": caller_tenant_id,
                        "allow_unowned": allow_unowned,
                        "limit": limit,
                        "offset": offset,
                    }

            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            summaries = []
            for r in rows:
                summaries.append(
                    ThreadSummary(
                        thread_id=r[0],
                        tenant_id=r[1],
                        owner_user_id=r[2],
                        title=r[3],
                        last_activity=r[4],
                        input_tokens=r[5],
                        output_tokens=r[6],
                        cost_usd=float(r[7]),
                        message_count=r[8],
                        interrupted=bool(r[9]) if len(r) > 9 else False,
                    )
                )
            return summaries

    except Exception as exc:
        logger.warning("Failed to list threads for tenant %s: %s", caller_tenant_id, exc)
        return []


@router.get("/threads/{thread_id}", response_model=ThreadDetail)
async def get_thread(
    thread_id: str,
    auth_user: Optional[str | AuthUser] = Depends(_verify_auth_and_get_user),
    graph=Depends(get_graph_for_request),
    pool=Depends(get_db_pool),
):
    """
    Retrieve full conversation history for a thread.
    - Uniform security guard: Enforces _assert_thread_access (404 on cross-tenant IDOR).
    - Message scrubbing: Strips intermediate tool calls, system prompts, and CRAG JSONs.
    - Paused state detection: Reports interrupted=True and approval_request if awaiting approval.
    """
    trace_id = generate_trace_id()
    # 1. Uniform thread access verification (404 on cross-tenant breach)
    snapshot = await _assert_thread_access(
        graph,
        thread_id,
        auth_user,
        endpoint="get_thread",
        trace_id=trace_id,
    )

    if not snapshot or not snapshot.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread '{thread_id}' not found.",
        )

    # 2. Extract and scrub conversation history
    raw_messages = snapshot.values.get("messages", [])
    clean_messages = _extract_clean_thread_messages(raw_messages)

    # 3. Check for pending approval interrupts
    interrupt_values = [
        intr.value
        for task in (snapshot.tasks or [])
        for intr in (getattr(task, "interrupts", None) or [])
    ]
    interrupted = bool(interrupt_values)
    approval_request = interrupt_values[0] if interrupted else None

    # 4. Fetch thread metadata from `threads` table
    caller_tenant_id, caller_user_id = _resolve_caller_identity(auth_user)
    tenant_id = snapshot.values.get("owner_tenant_id") or caller_tenant_id or "default"
    title = "Conversation"
    last_act = datetime.now(timezone.utc)
    inp_tokens = 0
    out_tokens = 0
    cost_val = 0.0
    msg_count = len(clean_messages)

    if pool:
        try:
            from src.audit.log import _get_conn

            async with _get_conn(pool) as conn:
                cursor = await conn.execute(
                    """
                    SELECT title, last_activity, input_tokens, output_tokens, cost_usd, message_count, interrupted
                    FROM threads
                    WHERE thread_id = %s;
                    """,
                    (thread_id,),
                )
                row = await cursor.fetchone()
                if row:
                    title, last_act, inp_tokens, out_tokens, cost_val, msg_count, db_interrupted = (
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                        float(row[4]),
                        row[5],
                        bool(row[6]),
                    )
                    # Self-healing: if checkpointer interrupt state diverged from DB, reconcile it now
                    if db_interrupted != interrupted:
                        await _sync_thread_interrupted(pool, thread_id, interrupted)
                else:
                    # Self-healing: if thread exists in checkpointer but missing in threads table, upsert it now
                    first_q = clean_messages[0].content if clean_messages else "Conversation"
                    title = first_q[:60]
                    owner_id = (
                        snapshot.values.get("owner_user_id")
                        if snapshot and snapshot.values
                        else caller_user_id
                    )
                    await upsert_thread_record(
                        pool,
                        thread_id,
                        tenant_id,
                        owner_id,
                        title,
                        TurnUsage(),
                        interrupted=interrupted,
                    )
        except Exception as exc:
            logger.warning("Failed to query threads metadata for %s: %s", thread_id, exc)

    return ThreadDetail(
        thread_id=thread_id,
        tenant_id=tenant_id,
        title=title,
        messages=clean_messages,
        interrupted=interrupted,
        approval_request=approval_request,
        input_tokens=inp_tokens,
        output_tokens=out_tokens,
        cost_usd=cost_val,
        message_count=msg_count,
        last_activity=last_act,
    )

