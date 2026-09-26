"""
Route handlers for the RAG API.

SRP: only HTTP → graph → HTTP translation.
OCP: add new routers in new files, include them in app.py.
"""

from fastapi import APIRouter, Depends

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
)
from src.api.dependencies import get_compiled_graph

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    return HealthResponse()


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    graph=Depends(get_compiled_graph),
):
    """
    Invoke the RAG graph with user question + history.

    Returns answer and grading metadata.
    """

    # Support both stateless (history) and stateful (thread_id) modes
    inputs = {"question": request.question}
    config = {}

    if request.thread_id:
        config = {"configurable": {"thread_id": request.thread_id}}
    else:
        # Legacy stateless mode
        inputs["history"] = request.history

    result = await graph.ainvoke(inputs, config)

    return ChatResponse(
        answer=result.get("answer", ""),
        answer_supported=result.get("answer_supported"),
        answer_grade_reason=result.get(
            "answer_grade_reason"
        ),
        retrieval_relevant=result.get(
            "retrieval_relevant"
        ),
        retrieval_grade_reason=result.get(
            "retrieval_grade_reason"
        ),
    )
