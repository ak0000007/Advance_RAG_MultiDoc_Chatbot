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
                    Generate          Rewrite
                        ↓                 ↑
                  Answer Grader           │
                        ↓                 │
                       END            Retrieve
                                         ↓
                                       Grade
                                         ↓
                                      Fallback
                                         ↓
                                        END
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
    create_answer_grader_node,
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
    Build the corrective RAG workflow with
    answer-quality evaluation.
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

    answer_grader_node = create_answer_grader_node(
        llm
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
        "grade_answer",
        answer_grader_node,
    )

    graph_builder.add_node(
        "fallback",
        fallback_node,
    )

    # -----------------------------------------------------
    # START → Question Router
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
    # Retrieve → Retrieval Grader
    # -----------------------------------------------------

    graph_builder.add_edge(
        "retrieve",
        "grade_retrieval",
    )

    # -----------------------------------------------------
    # Retrieval Grader → Decision
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
    # Generate → Answer Grader
    # -----------------------------------------------------

    graph_builder.add_edge(
        "generate",
        "grade_answer",
    )

    # -----------------------------------------------------
    # Terminal edges
    # -----------------------------------------------------

    graph_builder.add_edge(
        "grade_answer",
        END,
    )

    graph_builder.add_edge(
        "fallback",
        END,
    )

    return graph_builder.compile()