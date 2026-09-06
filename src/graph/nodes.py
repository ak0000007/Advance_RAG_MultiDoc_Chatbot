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



def create_retrieval_node(retriever):
    """
    Create a LangGraph retrieval node.

    The node reads the rewritten query from the graph state,
    invokes the supplied retriever, and stores the resulting
    documents back into the state.
    """

    def retrieval_node(state: RAGState):
        query = state["rewritten_query"]

        documents = retriever.invoke(query)

        return {
            "documents": documents
        }

    return retrieval_node


def create_generation_node(rag_chain):
    """
    Create a LangGraph generation node.

    The node reads the question and retrieved documents,
    then uses the existing RAG chain to generate the answer.

    Note:
    The current RAG chain performs retrieval internally,
    so this node is mainly useful as a transitional
    architecture while we progressively expose the
    individual RAG components.
    """

    def generation_node(state: RAGState):
        question = state["question"]

        answer = rag_chain.invoke({
            "question": question
        })

        return {
            "answer": answer
        }

    return generation_node
