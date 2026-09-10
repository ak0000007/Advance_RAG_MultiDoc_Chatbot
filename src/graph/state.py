"""
State definition for the LangGraph RAG workflow.
"""

from typing import TypedDict, Any


class RAGState(TypedDict, total=False):
    """
    Shared state carried through the LangGraph RAG workflow.
    """

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

    retrieval_relevant: bool

    retrieval_grade_reason: str

    # -----------------------------------------------------
    # Generation
    # -----------------------------------------------------

    answer: str

    # -----------------------------------------------------
    # Answer evaluation
    # -----------------------------------------------------

    answer_supported: bool

    answer_grade_reason: str