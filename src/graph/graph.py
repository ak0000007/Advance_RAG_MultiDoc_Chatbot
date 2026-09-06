"""
LangGraph workflow construction.

Builds the first fixed RAG graph:

START
  ↓
Query Rewriter
  ↓
Retrieval
  ↓
Generation
  ↓
END
"""

from langgraph.graph import StateGraph, START, END

from src.graph.state import RAGState
from src.graph.nodes import (
    create_query_rewriter_node,
    create_retrieval_node,
    create_generation_node,
)


def build_rag_graph(
    llm,
    retriever,
    rag_chain,
):
    """
    Build the first fixed LangGraph RAG workflow.

    Workflow:

        START
          ↓
        rewrite
          ↓
        retrieve
          ↓
        generate
          ↓
         END
    """

    graph_builder = StateGraph(RAGState)

    # Create nodes
    query_rewriter_node = create_query_rewriter_node(llm)
    retrieval_node = create_retrieval_node(retriever)
    generation_node = create_generation_node(rag_chain)

    # Register nodes
    graph_builder.add_node(
        "rewrite",
        query_rewriter_node,
    )

    graph_builder.add_node(
        "retrieve",
        retrieval_node,
    )

    graph_builder.add_node(
        "generate",
        generation_node,
    )

    # Fixed edges
    graph_builder.add_edge(
        START,
        "rewrite",
    )

    graph_builder.add_edge(
        "rewrite",
        "retrieve",
    )

    graph_builder.add_edge(
        "retrieve",
        "generate",
    )

    graph_builder.add_edge(
        "generate",
        END,
    )

    return graph_builder.compile()