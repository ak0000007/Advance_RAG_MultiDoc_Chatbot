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
    create_retrieval_confidence_router,
    create_update_memory_node,
    route_question,
    route_after_grading,
)


def build_rag_graph(
    llm,
    retriever,
    generation_chain,
    score_threshold: float = 0.7,
    min_confident_docs: int = 3,
    checkpointer=None,
):
    """
    Build the corrective RAG workflow with
    answer-quality evaluation.

    score_threshold / min_confident_docs control when the
    LLM retrieval grader is skipped. Set score_threshold=1.0
    to always grade (original behavior).
    
    checkpointer allows injecting persistent database memory 
    (like PostgresSaver) or MemorySaver without changing logic.
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

    graph_builder.add_node(
        "update_memory",
        create_update_memory_node(),
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
    # Retrieve → Confidence Check
    #
    # High reranker score + enough docs → skip grader
    # Low score or few docs → grade with LLM
    # No docs → fallback
    # -----------------------------------------------------

    retrieval_confidence_router = (
        create_retrieval_confidence_router(
            score_threshold=score_threshold,
            min_docs=min_confident_docs,
        )
    )

    graph_builder.add_conditional_edges(
        "retrieve",
        retrieval_confidence_router,
        {
            "generate": "generate",
            "grade_retrieval": "grade_retrieval",
            "fallback": "fallback",
        },
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
        "update_memory",
    )

    graph_builder.add_edge(
        "fallback",
        "update_memory",
    )

    graph_builder.add_edge(
        "update_memory",
        END,
    )

    return graph_builder.compile(checkpointer=checkpointer)