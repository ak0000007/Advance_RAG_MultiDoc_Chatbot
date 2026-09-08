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

    # Search-friendly standalone query
    rewritten_query: str

    # previous Conversational history
    history: list[Any]

    # Documents retrieved by the retrieval node
    documents: list[Any]

    # Final generated answer
    answer: str