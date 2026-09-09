"""
LangGraph workflow construction.

Current architecture:

START
  ↓
Question Router
  ├── standalone ────────────────┐
  │                              ↓
  │                           Retrieval
  │                              ↓
  └── follow-up → Rewrite → Retrieval
                                 ↓
                           Retrieval Grader
                           /             \
                       GOOD              BAD
                        ↓                 ↓
                    Generate       Retry available?
                        ↓             /        \
                       END          YES        NO
                                    ↓           ↓
                                 Rewrite     Fallback
                                    ↓           ↓
                                Retrieval      END
                                    ↓
                                  Grade
"""

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from src.graph.state import RAGState

from src.graph.nodes import (
    create_query_rewriter_node,
    create_retrieval_node,
    create_retrieval_grader_node,
    create_generation_node,
    create_fallback_node,
    route_question,
    route_after_grading,
)


def build_rag_graph(
    llm,
    retriever,
    generation_chain,
):
    """
    Build the corrective RAG LangGraph workflow.

    Components are injected into the graph rather than
    created inside the graph itself.

    This keeps the graph loosely coupled to:

        - LLM
        - Retriever
        - Generation chain
    """

    graph_builder = StateGraph(RAGState)

    # -----------------------------------------------------
    # Create nodes
    # -----------------------------------------------------

    query_rewriter_node = create_query_rewriter_node(
        llm
    )

    retrieval_node = create_retrieval_node(
        retriever
    )

    retrieval_grader_node = create_retrieval_grader_node(
        llm
    )

    generation_node = create_generation_node(
        generation_chain
    )

    fallback_node = create_fallback_node()

    # -----------------------------------------------------
    # Register nodes
    # -----------------------------------------------------

    graph_builder.add_node(
        "rewrite",
        query_rewriter_node,
    )

    graph_builder.add_node(
        "retrieve",
        retrieval_node,
    )

    graph_builder.add_node(
        "grade_retrieval",
        retrieval_grader_node,
    )

    graph_builder.add_node(
        "generate",
        generation_node,
    )

    graph_builder.add_node(
        "fallback",
        fallback_node,
    )

    # -----------------------------------------------------
    # START → Router
    # -----------------------------------------------------

    graph_builder.add_conditional_edges(
        START,
        route_question,
        {
            "rewrite": "rewrite",
            "retrieve": "retrieve",
        },
    )

    # -----------------------------------------------------
    # Rewrite → Retrieve
    # -----------------------------------------------------

    graph_builder.add_edge(
        "rewrite",
        "retrieve",
    )

    # -----------------------------------------------------
    # Retrieve → Grade
    # -----------------------------------------------------

    graph_builder.add_edge(
        "retrieve",
        "grade_retrieval",
    )

    # -----------------------------------------------------
    # Grade → Generate / Rewrite / Fallback
    # -----------------------------------------------------

    graph_builder.add_conditional_edges(
        "grade_retrieval",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "fallback": "fallback",
        },
    )

    # -----------------------------------------------------
    # End states
    # -----------------------------------------------------

    graph_builder.add_edge(
        "generate",
        END,
    )

    graph_builder.add_edge(
        "fallback",
        END,
    )

    return graph_builder.compile()