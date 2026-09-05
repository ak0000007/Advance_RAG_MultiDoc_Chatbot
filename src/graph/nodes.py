"""
LangGraph nodes for the RAG workflow.

Nodes are responsible for performing individual operations
using the shared RAGState.
"""

from src.graph.state import RAGState
from src.rag.conversational import build_query_rewriter


def create_query_rewriter_node(llm):
    """
    Create a LangGraph node that rewrites the user's question
    using our existing LangChain query-rewriting chain.
    """

    query_rewriter = build_query_rewriter(llm)

    def query_rewriter_node(state: RAGState):
        """
        Read the current question from state and produce
        a standalone retrieval query.
        """

        question = state["question"]

        # For this first node, we use the existing conversation
        # history mechanism later when we connect the full graph.
        rewritten_query = query_rewriter.invoke(
            {
                "history": [],
                "question": question,
            }
        )

        return {
            "rewritten_query": rewritten_query
        }

    return query_rewriter_node