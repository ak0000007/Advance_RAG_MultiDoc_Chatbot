"""
State definition for the LangGraph RAG workflow.
"""

from typing import TypedDict, Any, Annotated
from langgraph.graph.message import add_messages


class RAGState(TypedDict, total=False):
    """
    Shared state carried through the LangGraph RAG workflow.
    """

    # -----------------------------------------------------
    # Agent Conversation History
    # -----------------------------------------------------
    messages: Annotated[list[Any], add_messages]

    # -----------------------------------------------------
    # User input
    # -----------------------------------------------------

    question: str

    # Conversation history
    history: list[Any]

    # -----------------------------------------------------
    # Query processing
    # -----------------------------------------------------

    rewritten_query: str

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    documents: list[Any]

    retrieval_attempts: int

    # -----------------------------------------------------
    # Generation
    # -----------------------------------------------------

    answer: str

    # -----------------------------------------------------
    # Answer evaluation
    # -----------------------------------------------------