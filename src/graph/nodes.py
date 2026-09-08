"""
LangGraph nodes for the RAG workflow.
"""

from src.graph.state import RAGState
from src.rag.conversational import build_query_rewriter


def route_question(state: RAGState) -> str:
    """
    Decide whether the question needs query rewriting.

    No conversation history:
        retrieve directly.

    Existing conversation history:
        rewrite first.
    """

    history = state.get("history", [])

    if history:
        return "rewrite"

    return "retrieve"


def create_query_rewriter_node(llm):

    query_rewriter = build_query_rewriter(llm)

    def query_rewriter_node(state: RAGState):

        question = state["question"]
        history = state.get("history", [])

        rewritten_query = query_rewriter.invoke(
            {
                "history": history,
                "question": question,
            }
        )

        return {
            "rewritten_query": rewritten_query
        }

    return query_rewriter_node


def create_retrieval_node(retriever):

    def retrieval_node(state: RAGState):

        # If rewrite was skipped, use the original question.
        query = state.get(
            "rewritten_query",
            state["question"],
        )

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