"""
State definition for the LangGraph RAG workflow.

The state represents the information that travels through
the graph as different nodes execute.
"""

from typing import TypedDict, Any


class RAGState(TypedDict, total=False):
    """
    Shared state carried through the LangGraph RAG workflow.

    Each node reads information from this state and returns
    updates to one or more fields.
    """

    # Original user question
    question: str

    # Conversation history used for follow-up questions
    history: list[Any]

    # Query used for retrieval
    rewritten_query: str

    # Retrieved LangChain Documents
    documents: list[Any]

    # Result of retrieval-quality evaluation
    retrieval_relevant: bool

    # Number of retrieval attempts
    retrieval_attempts: int

    # Final generated answer
    answer: str

    # Whether the generated answer is supported
    # Whether the generated answer is supported
    answer_supported: bool