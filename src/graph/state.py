"""
State definition for the LangGraph RAG workflow.

The state represents the information that travels through
the graph as different nodes execute.
"""

from typing import TypedDict


class RAGState(TypedDict,total=False):
    """
    Shared state carried through the LangGraph RAG workflow.
    Each node reads information from this state and returns

    updates to one or more fields.
    """

    question: str
    rewritten_query: str
    Documents: list[dict]
    answer: str