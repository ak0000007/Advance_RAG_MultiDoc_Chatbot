"""
LangGraph nodes for the RAG workflow.

Nodes are responsible for performing individual operations
using the shared RAGState.
"""

from src.graph.state import RAGState
from src.rag.conversational import build_query_rewriter
from src.rag.chain import format_docs


def create_query_rewriter_node(llm):
    """
    Create a LangGraph node that rewrites the user's question
    into a standalone retrieval query.

    NOTE:
    Conditional rewriting will be introduced in the next step.
    For now this node always performs rewriting.
    """

    query_rewriter = build_query_rewriter(llm)

    def query_rewriter_node(state: RAGState):

        question = state["question"]

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


def create_retrieval_node(retriever):
    """
    Create a LangGraph retrieval node.

    The node:

        rewritten query
              ↓
          retriever
              ↓
          documents
              ↓
        graph state
    """

    def retrieval_node(state: RAGState):

        query = state["rewritten_query"]

        documents = retriever.invoke(
            {
                "question": query,
                "metadata_filter": None,
            }
        )

        return {
            "documents": documents
        }

    return retrieval_node


def create_generation_node(generation_chain):
    """
    Create a generation node that consumes documents
    already retrieved by the retrieval node.

    IMPORTANT:

    This node does NOT retrieve documents.

    It only performs:

        documents
            ↓
        format
            ↓
        prompt
            ↓
        LLM
            ↓
        answer
    """

    def generation_node(state: RAGState):

        question = state["question"]

        documents = state.get("documents", [])

        answer = generation_chain.invoke(
            {
                "context": documents,
                "question": question,
            }
        )

        return {
            "answer": answer
        }

    return generation_node