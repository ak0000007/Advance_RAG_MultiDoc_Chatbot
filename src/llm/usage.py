"""
Multi-stage token accounting and database persistence.

Captures token usage across all graph execution stages (intent classification,
multi-chunk grading, query rewriting, and final synthesis), computes costs,
and provides best-effort persistence to PostgreSQL for auditing and quota enforcement.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from src.llm.pricing import cost_usd

logger = logging.getLogger(__name__)


@dataclass
class StageUsage:
    node: str
    model: str
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    estimated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "estimated": self.estimated,
        }


@dataclass
class TurnUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    estimated: bool = False
    by_stage: list[StageUsage] = field(default_factory=list)

    def add_stage(self, stage: StageUsage) -> None:
        self.by_stage.append(stage)
        self.input_tokens += stage.input_tokens
        self.output_tokens += stage.output_tokens
        self.total_tokens += stage.input_tokens + stage.output_tokens
        self.cost_usd = round(self.cost_usd + stage.cost_usd, 6)
        if stage.estimated:
            self.estimated = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "estimated": self.estimated,
            "by_stage": [s.to_dict() for s in self.by_stage],
        }


def _extract_tokens_from_llm_result(
    response: LLMResult,
    default_input_chars: int = 0,
    default_output_chars: int = 0,
) -> tuple[int, int, bool]:
    """
    Extract (input_tokens, output_tokens, estimated) from an LLMResult.
    Checks:
      1. generation.message.usage_metadata
      2. response.llm_output['token_usage'] or ['usage']
      3. generation.message.response_metadata
      4. Fallback character heuristic (approx 4 chars / token).
    """
    input_tokens = 0
    output_tokens = 0
    found_metadata = False

    # 1. Check generation message usage_metadata (Standard LangChain 0.2+)
    for gen_list in response.generations:
        for gen in gen_list:
            msg = getattr(gen, "message", None)
            if msg:
                usage_meta = getattr(msg, "usage_metadata", None)
                if isinstance(usage_meta, dict):
                    inp = usage_meta.get("input_tokens", 0) or 0
                    out = usage_meta.get("output_tokens", 0) or 0
                    if inp or out:
                        input_tokens += inp
                        output_tokens += out
                        found_metadata = True

                # Check response_metadata
                resp_meta = getattr(msg, "response_metadata", {}) or {}
                if not found_metadata:
                    token_usage = resp_meta.get("token_usage") or resp_meta.get("usage")
                    if isinstance(token_usage, dict):
                        inp = (
                            token_usage.get("prompt_tokens")
                            or token_usage.get("input_tokens")
                            or 0
                        )
                        out = (
                            token_usage.get("completion_tokens")
                            or token_usage.get("output_tokens")
                            or 0
                        )
                        if inp or out:
                            input_tokens += inp
                            output_tokens += out
                            found_metadata = True

            # Also check generation_info on Generation objects
            info = getattr(gen, "generation_info", None)
            if isinstance(info, dict) and not found_metadata:
                token_usage = info.get("usage_metadata") or info.get("token_usage") or info.get("usage")
                if isinstance(token_usage, dict):
                    inp = (
                        token_usage.get("prompt_tokens")
                        or token_usage.get("input_tokens")
                        or 0
                    )
                    out = (
                        token_usage.get("completion_tokens")
                        or token_usage.get("output_tokens")
                        or 0
                    )
                    if inp or out:
                        input_tokens += inp
                        output_tokens += out
                        found_metadata = True

    # 2. Check response.llm_output
    if not found_metadata and response.llm_output:
        token_usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")
        if isinstance(token_usage, dict):
            inp = (
                token_usage.get("prompt_tokens")
                or token_usage.get("input_tokens")
                or 0
            )
            out = (
                token_usage.get("completion_tokens")
                or token_usage.get("output_tokens")
                or 0
            )
            if inp or out:
                input_tokens += inp
                output_tokens += out
                found_metadata = True

    if found_metadata:
        return input_tokens, output_tokens, False

    # 3. Fallback character estimation (approx 4 chars per token)
    inp_est = max(1, default_input_chars // 4) if default_input_chars > 0 else 0
    out_est = max(1, default_output_chars // 4) if default_output_chars > 0 else 0
    return inp_est, out_est, True


class TokenUsageCallbackHandler(AsyncCallbackHandler):
    """
    Async callback handler capturing token usage and calculating costs
    across all nodes in a LangGraph execution.
    """

    def __init__(self, default_provider: str = "", default_model: str = ""):
        super().__init__()
        self.default_provider = default_provider
        self.default_model = default_model
        self.turn_usage = TurnUsage()

    def get_turn_usage(self) -> TurnUsage:
        return self.turn_usage

    def _process_llm_result(
        self,
        response: LLMResult,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> None:
        meta = metadata or {}
        node = meta.get("langgraph_node") or (tags[0] if tags else "unknown")

        # Determine model
        model = self.default_model
        if response.llm_output and isinstance(response.llm_output, dict):
            model = response.llm_output.get("model_name") or model

        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg:
                    resp_meta = getattr(msg, "response_metadata", {}) or {}
                    if "model_name" in resp_meta:
                        model = resp_meta["model_name"]
                    elif "model" in resp_meta:
                        model = resp_meta["model"]

        if not model:
            model = "gemini-3.6-flash"

        # Content length heuristic fallback
        out_chars = 0
        for gen_list in response.generations:
            for gen in gen_list:
                text = getattr(gen, "text", "") or ""
                out_chars += len(text)

        input_tokens, output_tokens, is_estimated = _extract_tokens_from_llm_result(
            response,
            default_input_chars=out_chars * 2,  # rough prompt estimate if missing
            default_output_chars=out_chars,
        )

        cost, pricing_estimated = cost_usd(model, input_tokens, output_tokens)
        stage = StageUsage(
            node=str(node),
            model=str(model),
            provider=str(self.default_provider),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            estimated=is_estimated or pricing_estimated,
        )
        self.turn_usage.add_stage(stage)

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._process_llm_result(response, metadata=metadata, tags=tags)

    async def on_chat_model_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._process_llm_result(response, metadata=metadata, tags=tags)


def collect_from_events(
    events: list[dict[str, Any]],
    default_model: str = "gemini-3.6-flash",
    default_provider: str = "",
) -> TurnUsage:
    """
    Extract TurnUsage from a list of LangGraph astream_events v2 dictionaries.
    """
    usage = TurnUsage()
    for event in events:
        if event.get("event") == "on_chat_model_end":
            node = event.get("metadata", {}).get("langgraph_node", "unknown")
            output = event.get("data", {}).get("output")
            model = default_model
            provider = default_provider
            input_tokens = 0
            output_tokens = 0
            is_estimated = False

            if output:
                # Check usage_metadata on AIMessage
                usage_meta = getattr(output, "usage_metadata", None)
                if isinstance(usage_meta, dict):
                    input_tokens = usage_meta.get("input_tokens", 0) or 0
                    output_tokens = usage_meta.get("output_tokens", 0) or 0
                else:
                    resp_meta = getattr(output, "response_metadata", {}) or {}
                    token_usage = resp_meta.get("token_usage") or resp_meta.get("usage")
                    if isinstance(token_usage, dict):
                        input_tokens = (
                            token_usage.get("prompt_tokens")
                            or token_usage.get("input_tokens")
                            or 0
                        )
                        output_tokens = (
                            token_usage.get("completion_tokens")
                            or token_usage.get("output_tokens")
                            or 0
                        )
                    else:
                        is_estimated = True
                        content = getattr(output, "content", "")
                        text = str(content) if content else ""
                        output_tokens = max(1, len(text) // 4)
                        input_tokens = max(1, len(text) // 2)

                resp_meta = getattr(output, "response_metadata", {}) or {}
                model = resp_meta.get("model_name") or resp_meta.get("model") or default_model

            cost, pricing_estimated = cost_usd(model, input_tokens, output_tokens)
            stage = StageUsage(
                node=str(node),
                model=str(model),
                provider=str(provider),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                estimated=is_estimated or pricing_estimated,
            )
            usage.add_stage(stage)

    return usage


async def record_turn_usage(
    pool: Any,
    thread_id: str,
    tenant_id: str,
    user_id: Optional[str],
    turn_usage: TurnUsage,
    trace_id: Optional[str] = None,
) -> None:
    """
    Persist multi-stage token usage to PostgreSQL token_usage table.
    Best-effort: DB failure logs warning and NEVER breaks the chat response.
    """
    if not pool:
        return

    # If no stages captured, record aggregate turn row if tokens > 0
    stages = turn_usage.by_stage
    if not stages and (turn_usage.input_tokens > 0 or turn_usage.output_tokens > 0):
        stages = [
            StageUsage(
                node="turn_aggregate",
                model="unknown",
                provider="",
                input_tokens=turn_usage.input_tokens,
                output_tokens=turn_usage.output_tokens,
                cost_usd=turn_usage.cost_usd,
                estimated=turn_usage.estimated,
            )
        ]

    if not stages:
        return

    query = """
        INSERT INTO token_usage (
            thread_id, tenant_id, user_id, node, model, provider,
            input_tokens, output_tokens, cost_usd, estimated, trace_id, created_at
        ) VALUES (
            %(thread_id)s, %(tenant_id)s, %(user_id)s, %(node)s, %(model)s, %(provider)s,
            %(input_tokens)s, %(output_tokens)s, %(cost_usd)s, %(estimated)s, %(trace_id)s, now()
        );
    """

    try:
        from src.audit.log import _get_conn

        async with _get_conn(pool) as conn:
            for s in stages:
                params = {
                    "thread_id": thread_id,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "node": s.node,
                    "model": s.model,
                    "provider": s.provider,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "cost_usd": s.cost_usd,
                    "estimated": s.estimated,
                    "trace_id": trace_id,
                }
                await conn.execute(query, params)
    except Exception as exc:
        logger.warning(
            "Failed to record token usage to database for thread %s: %s",
            thread_id,
            exc,
        )


async def check_tenant_quota(
    pool: Any,
    tenant_id: str,
    quota_daily: Optional[int],
    quota_monthly: Optional[int],
) -> tuple[bool, Optional[str]]:
    """
    Check if a tenant has exceeded their daily or monthly token quota.
    Returns (allowed, error_reason).
    """
    if not pool or not tenant_id:
        return True, None

    if quota_daily is None and quota_monthly is None:
        return True, None

    try:
        from src.audit.log import _get_conn

        now_utc = datetime.now(timezone.utc)

        async with _get_conn(pool) as conn:
            # 1. Check daily quota (from midnight UTC today)
            if quota_daily is not None:
                start_of_day = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
                cursor = await conn.execute(
                    """
                    SELECT COALESCE(SUM(input_tokens + output_tokens), 0)
                    FROM token_usage
                    WHERE tenant_id = %s AND created_at >= %s;
                    """,
                    (tenant_id, start_of_day),
                )
                row = await cursor.fetchone()
                daily_tokens = row[0] if row else 0
                if daily_tokens >= quota_daily:
                    return False, f"Daily token quota exceeded ({daily_tokens:,} / {quota_daily:,} tokens)."

            # 2. Check monthly quota (from 1st of month UTC)
            if quota_monthly is not None:
                start_of_month = now_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                cursor = await conn.execute(
                    """
                    SELECT COALESCE(SUM(input_tokens + output_tokens), 0)
                    FROM token_usage
                    WHERE tenant_id = %s AND created_at >= %s;
                    """,
                    (tenant_id, start_of_month),
                )
                row = await cursor.fetchone()
                monthly_tokens = row[0] if row else 0
                if monthly_tokens >= quota_monthly:
                    return False, f"Monthly token quota exceeded ({monthly_tokens:,} / {quota_monthly:,} tokens)."

        return True, None

    except Exception as exc:
        logger.warning("Quota verification failed for tenant %s: %s. Permitting request.", tenant_id, exc)
        return True, None
