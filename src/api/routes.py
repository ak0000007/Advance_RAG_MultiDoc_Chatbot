"""
Route handlers for the RAG API.

SRP: only HTTP → graph → HTTP translation.
OCP: add new routers in new files, include them in app.py.
"""

import uuid
from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ResumeRequest,
    HealthResponse,
)
from src.api.dependencies import get_compiled_graph

router = APIRouter()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_config(thread_id: str, sf_username: str | None = None) -> dict:
    config: dict = {"configurable": {"thread_id": thread_id}}
    if sf_username:
        config["configurable"]["sf_username"] = sf_username
    return config


async def _run_and_respond(graph, inputs, config: dict, thread_id: str) -> ChatResponse:
    """
    Stream the graph to completion (or interrupt), then inspect state.

    - Normal completion → ChatResponse(answer=...)
    - interrupt() fired → ChatResponse(interrupted=True, approval_request=...)
    """
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
        )

    return ChatResponse(answer=snapshot.values.get("answer", ""))


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    graph=Depends(get_compiled_graph),
):
    """
    Invoke the RAG graph with user question + history.

    Returns:
        - ChatResponse(answer=...) on normal completion.
        - ChatResponse(interrupted=True, thread_id=..., approval_request=...) when
          the graph pauses for human approval (Salesforce update confirmation).
          The frontend must display the approval UI and POST to /chat/resume.
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

    thread_id = request.thread_id or str(uuid.uuid4())
    config = _build_config(thread_id, request.sf_username)
    inputs = {"question": request.question, "messages": messages}

    return await _run_and_respond(graph, inputs, config, thread_id)


@router.post("/chat/resume", response_model=ChatResponse)
async def resume_chat(
    request: ResumeRequest,
    graph=Depends(get_compiled_graph),
):
    """
    Resume a graph that was interrupted for human approval.

    The frontend posts the thread_id (from the interrupted ChatResponse)
    and the human's decision ('Approve' or 'Reject').

    Returns the same ChatResponse envelope as /chat.
    """
    config = _build_config(request.thread_id)

    # Verify the thread actually exists and is interrupted
    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.next:
        raise HTTPException(
            status_code=404,
            detail=f"Thread '{request.thread_id}' is not paused or does not exist.",
        )

    return await _run_and_respond(
        graph,
        Command(resume=request.decision),
        config,
        request.thread_id,
    )
