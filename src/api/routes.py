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
    from langchain_core.messages import HumanMessage, AIMessage
    import uuid
    
    messages = []
    
    # Process stateless history if provided
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

    inputs = {
        "question": request.question,
        "messages": messages
    }
    
    config = {"configurable": {}}

    if request.thread_id:
        config["configurable"]["thread_id"] = request.thread_id
    else:
        # Generate a one-off thread_id to satisfy the checkpointer
        config["configurable"]["thread_id"] = str(uuid.uuid4())
    if request.sf_username:
        config["configurable"]["sf_username"] = request.sf_username

    result = await graph.ainvoke(inputs, config)

    return ChatResponse(
        answer=result.get("answer", ""),
    )
