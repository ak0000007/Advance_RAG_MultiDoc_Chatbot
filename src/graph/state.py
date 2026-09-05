"""
State definition for the LangGraph RAG workflow.

The state represents the information that travels through
the graph as different nodes execute.
"""

from typing import TypedDict


class RAGState(TypedDict):
    """
    Shared state carried through the LangGraph RAG workflow.

    The state will gradually grow as we introduce:
    - retrieval
    - reranking
    - evaluation
    - retries
    - generation
    """

    question: str
    rewritten_query: str
    answer: str