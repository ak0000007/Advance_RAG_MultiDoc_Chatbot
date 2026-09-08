"""
LangGraph workflow construction.

Workflow:

                    ┌── rewrite ──┐
                    │             │
START ── routing ───┤             ▼
                    │          retrieve
                    │             │
                    └─────────────┘
                                  │
                                  ▼
                               generate
                                  │
                                  ▼
                                 END
"""

from langgraph.graph import StateGraph, START, END

from src.graph.state import RAGState

from src.graph.nodes import (
    create_query_rewriter_node,
    create_retrieval_node,
    create_generation_node,
    route_question,
)


def build_rag_graph(
    llm,
    retriever,
    generation_chain,
):
    """
    Build the conditional LangGraph RAG workflow.

    Routing logic:

        No conversation history
            ↓
        retrieve directly

        Existing conversation history
            ↓
        rewrite
            ↓
        retrieve

        Both paths
            ↓
        generate
            ↓
        END
    """

    graph_builder = StateGraph(RAGState)

    # --------------------------------------------------
    # Create nodes
    # --------------------------------------------------

    query_rewriter_node = create_query_rewriter_node(
        llm
    )

    retrieval_node = create_retrieval_node(
        retriever
    )

    generation_node = create_generation_node(
        generation_chain
    )

    # --------------------------------------------------
    # Register nodes
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Conditional routing
    # --------------------------------------------------

    graph_builder.add_conditional_edges(
        START,
        route_question,
        {
            "rewrite": "rewrite",
            "retrieve": "retrieve",
        },
    )

    # --------------------------------------------------
    # Fixed edges after routing
    # --------------------------------------------------

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