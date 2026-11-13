"""
Route handlers for the RAG API.

SRP: only HTTP → graph → HTTP translation.
OCP: add new routers in new files, include them in app.py.
"""

import json
import re
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Response, Security, status
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
)
from src.retrieval.citations import extract_citations_from_messages
from src.api.dependencies import get_compiled_graph, get_active_graph, get_db_pool

router = APIRouter()

security_bearer = HTTPBearer(auto_error=False)
security_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_auth_and_get_user(
    auth_bearer: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    api_key: Optional[str] = Security(security_api_key),
) -> Optional[str]:
    """
    Enforce API authentication and resolve authenticated user identity.
    If settings.api_secret_key is set, rejects unauthenticated calls with 401.
    If not set (dev mode), permits access.
    """
    required_key = (
        settings.api_secret_key.get_secret_value()
        if hasattr(settings.api_secret_key, "get_secret_value")
        else settings.api_secret_key
    )
    if required_key:
        token = auth_bearer.credentials if auth_bearer else api_key
        if not token or token != required_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing authentication credentials.",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return settings.sf_default_username


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_config(
    thread_id: str,
    sf_username: str | None = None,
    trace_id: str | None = None,
    endpoint: str = "chat",
) -> dict:
    tid = trace_id or generate_trace_id()
    metadata = {
        "thread_id": thread_id,
        "trace_id": tid,
        "endpoint": endpoint,
    }
    if sf_username:
        metadata["sf_username"] = sf_username

    tags = ["advance-rag", "langgraph", endpoint]
    callbacks = get_tracer_callbacks(
        thread_id=thread_id,
        run_name=f"rag-turn-{thread_id[:8]}",
        tags=tags,
        metadata=metadata,
        trace_id=tid,
    )

    configurable: dict = {
        "thread_id": thread_id,
        "trace_id": tid,
    }
    if sf_username:
        configurable["sf_username"] = sf_username

    return {
        "configurable": configurable,
        "callbacks": callbacks,
        "metadata": metadata,
        "tags": tags,
        "run_name": f"rag-turn-{thread_id[:8]}",
    }


def _prepare_chat_context(
    request: ChatRequest,
    auth_user: Optional[str],
) -> tuple[dict, dict, str]:
    """
    Standardize message sequence, authenticated sf_username, and thread config.
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

    # Resolve sf_username: authenticated server context takes precedence
    raw_username = auth_user or request.sf_username
    sf_username = None
    if raw_username:
        sf_username = re.sub(r"[\x00-\x1f\x7f]", "", raw_username).strip()[:100]

    thread_id = request.thread_id or str(uuid.uuid4())
    trace_id = getattr(request, "trace_id", None) or generate_trace_id()
    config = _build_config(thread_id, sf_username, trace_id=trace_id)
    inputs = {
        "question": request.question,
        "messages": messages,
        "citations": [],
    }
    if sf_username:
        inputs["sf_username"] = sf_username

    return inputs, config, thread_id


async def _run_and_respond(graph, inputs, config: dict, thread_id: str) -> ChatResponse:
    """
    Stream the graph to completion (or interrupt), then inspect state.

    - Normal completion → ChatResponse(answer=...)
    - interrupt() fired → ChatResponse(interrupted=True, approval_request=...)
    """
    trace_id = config.get("configurable", {}).get("trace_id") or generate_trace_id()

    # Run the graph; stream_mode="values" yields full state after each step.
    # astream finishes even when interrupted (GraphInterrupt is caught internally).
    async for _ in graph.astream(inputs, config, stream_mode="values"):
        pass

    # Inspect final / paused state
    snapshot = await graph.aget_state(config)

    # Collect interrupt payloads from all pending tasks
    interrupt_values = [
        intr.value
        for task in (snapshot.tasks or [])
        for intr in (getattr(task, "interrupts", None) or [])
    ]

    if interrupt_values:
        return ChatResponse(
            interrupted=True,
            thread_id=thread_id,
            approval_request=interrupt_values[0],
            citations=[],
            trace_id=trace_id,
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

    return ChatResponse(
        answer=snapshot.values.get("answer", "") if snapshot and snapshot.values else "",
        citations=citations_list,
        trace_id=trace_id,
    )


async def _stream_graph_events(graph, inputs, config: dict, thread_id: str):
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
    accumulated_answer: list[str] = []
    trace_id = config.get("configurable", {}).get("trace_id") or generate_trace_id()
    try:
        async for event in graph.astream_events(inputs, config, version="v2"):
            kind = event.get("event")
            node = event.get("metadata", {}).get("langgraph_node", "")

            # 1. Incremental Chat Model Tokens
            if kind == "on_chat_model_stream":
                # Skip structured router outputs (e.g. classify_intent JSON)
                if node == "classify_intent":
                    continue

                chunk = event.get("data", {}).get("chunk")
                if chunk is not None:
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
                        payload = json.dumps({"event": "token", "chunk": text})
                        yield f"event: token\ndata: {payload}\n\n"

            # 2. Tool Start Progress
            elif kind == "on_tool_start":
                tool_name = event.get("name", "tool")
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

        # After astream_events completes, inspect checkpointer state
        snapshot = await graph.aget_state(config)
        interrupt_values = [
            intr.value
            for task in (snapshot.tasks or [])
            for intr in (getattr(task, "interrupts", None) or [])
        ]

        if interrupt_values:
            payload = json.dumps({
                "event": "interrupt",
                "interrupted": True,
                "thread_id": thread_id,
                "approval_request": interrupt_values[0],
                "trace_id": trace_id,
            })
            yield f"event: interrupt\ndata: {payload}\n\n"
        else:
            final_answer = (
                snapshot.values.get("answer") if snapshot and snapshot.values else ""
            ) or "".join(accumulated_answer)

            # Citations must strictly belong to the current turn (never leaked from prior turns or CRM actions)
            messages = snapshot.values.get("messages", []) if snapshot and snapshot.values else []
            raw_citations = extract_citations_from_messages(messages, current_turn_only=True)

            payload = json.dumps({
                "event": "done",
                "interrupted": False,
                "thread_id": thread_id,
                "answer": final_answer,
                "citations": raw_citations,
                "trace_id": trace_id,
            })
            yield f"event: done\ndata: {payload}\n\n"

    except Exception as exc:
        payload = json.dumps({"event": "error", "detail": str(exc)})
        yield f"event: error\ndata: {payload}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    auth_user: Optional[str] = Depends(_verify_auth_and_get_user),
    graph=Depends(get_active_graph),
):
    """
    Invoke the RAG graph with user question + history.

    Returns:
        - ChatResponse(answer=...) on normal completion.
        - ChatResponse(interrupted=True, thread_id=..., approval_request=...) when
          the graph pauses for human approval (Salesforce update confirmation).
          The frontend must display the approval UI and POST to /chat/resume.
    """
    inputs, config, thread_id = _prepare_chat_context(request, auth_user)
    return await _run_and_respond(graph, inputs, config, thread_id)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    auth_user: Optional[str] = Depends(_verify_auth_and_get_user),
    graph=Depends(get_active_graph),
):
    """
    Stream the RAG graph execution via Server-Sent Events (SSE).
    Yields 'status', 'token', 'interrupt', and 'done' events.
    """
    inputs, config, thread_id = _prepare_chat_context(request, auth_user)
    return StreamingResponse(
        _stream_graph_events(graph, inputs, config, thread_id),
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
    graph=Depends(get_active_graph),
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

    # Preserve sf_username across resume: state checkpoint or server auth
    thread_sf_username = (
        snapshot.values.get("sf_username")
        if snapshot.values
        else None
    ) or auth_user
    config = _build_config(request.thread_id, thread_sf_username, trace_id=trace_id, endpoint="resume")

    return await _run_and_respond(
        graph,
        Command(resume=request.decision),
        config,
        request.thread_id,
    )


@router.post("/chat/resume/stream")
async def resume_chat_stream(
    request: ResumeRequest,
    auth_user: Optional[str] = Depends(_verify_auth_and_get_user),
    graph=Depends(get_active_graph),
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

    thread_sf_username = (
        snapshot.values.get("sf_username")
        if snapshot.values
        else None
    ) or auth_user
    config = _build_config(request.thread_id, thread_sf_username, trace_id=trace_id, endpoint="resume_stream")

    inputs = Command(resume=request.decision)
    return StreamingResponse(
        _stream_graph_events(graph, inputs, config, request.thread_id),
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

