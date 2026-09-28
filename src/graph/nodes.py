"""
LangGraph nodes for the RAG workflow.

Nodes are responsible for performing individual operations
using the shared RAGState.
"""

from pydantic import BaseModel, Field

from src.graph.state import RAGState
from src.rag.rewriter import build_query_rewriter
from src.rag.chain import format_docs

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser,
)

from langchain_core.runnables import RunnableLambda


# =========================================================
# Structured Output Schemas
# =========================================================


class RetrievalGrade(BaseModel):
    """
    Structured result produced by the retrieval grader.
    """

    relevant: bool = Field(
        description=(
            "Whether the retrieved documents contain "
            "useful information for answering the question."
        )
    )

    reason: str = Field(
        description=(
            "Brief explanation of why the retrieved "
            "documents are or are not relevant."
        )
    )


class AnswerGrade(BaseModel):
    """
    Structured result produced by the answer grader.
    """

    supported: bool = Field(
        description=(
            "Whether the generated answer is fully "
            "supported by the retrieved documents."
        )
    )

    reason: str = Field(
        description=(
            "Brief explanation of why the generated "
            "answer is or is not supported by the context."
        )
    )


# =========================================================
# Question Router
# =========================================================


def route_question(state: RAGState) -> str:
    """
    Decide whether the incoming question needs query rewriting.

    First question:
        history == []
        -> retrieve directly

    Follow-up question:
        history != []
        -> rewrite first
    """

    history = state.get("history", [])

    if history:
        return "rewrite"

    return "retrieve"


# =========================================================
# Retrieval Confidence Router
# =========================================================


def create_retrieval_confidence_router(
    score_threshold: float = 0.7,
    min_docs: int = 3,
):
    """
    Factory that returns a routing function to conditionally
    skip the LLM-based retrieval grader.

    Decision logic:

        No documents retrieved
            → fallback (no point grading nothing)

        Top reranker score >= threshold AND doc count >= min_docs
            → generate (high confidence, skip grader)

        Otherwise
            → grade_retrieval (uncertain, let LLM judge)

    Parameters are configurable via settings or .env:

        RETRIEVAL_CONFIDENCE_THRESHOLD=0.7
        RETRIEVAL_MIN_CONFIDENT_DOCS=3

    To disable this optimization entirely, set threshold=1.0.
    All requests will then flow through the grader as before.
    """

    def route_retrieval_confidence(state: RAGState) -> str:

        documents = state.get("documents", [])

        if not documents:
            return "fallback"

        top_score = documents[0].metadata.get(
            "rerank_score", 0.0,
        )

        if (
            top_score >= score_threshold
            and len(documents) >= min_docs
        ):
            return "generate"

        return "grade_retrieval"

    return route_retrieval_confidence


# =========================================================
# Query Rewriter
# =========================================================


def create_query_rewriter_node(llm):
    """
    Create a LangGraph node that rewrites the user's question.

    Two situations are handled:

    1. Normal conversational rewrite
       - Resolve references using conversation history.

    2. Corrective rewrite
       - Previous retrieval was judged irrelevant.
       - Create a better retrieval query.
    """

    query_rewriter = build_query_rewriter(llm)

    retry_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a query rewriting component inside a
Retrieval-Augmented Generation system.

The previous retrieval attempt was judged NOT RELEVANT
to the user's question.

Your task is to create a better search query for the
next retrieval attempt.

STRICT RULES:

1. Output ONLY one search query.

2. NEVER answer the user's question.

3. Do not explain your reasoning.

4. Preserve the user's actual intent.

5. Use conversation history only when it helps resolve
   references in the question.

6. Make the query more explicit and retrieval-friendly.

7. Include important entities, concepts, relationships,
   or constraints that are already known.

8. Do NOT invent facts.

9. Do NOT invent entities, dates, sections, documents,
   events, or explanations.

10. Do NOT output multiple alternatives.

11. Return exactly ONE search query.
                """,
            ),
            MessagesPlaceholder(
                variable_name="history"
            ),
            (
                "human",
                """
Original question:

{question}

Previous retrieval query:

{previous_query}

Create a better retrieval query.
                """,
            ),
        ]
    )

    corrective_rewriter = (
        retry_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(
            lambda text: text.strip()
        )
    )

    def query_rewriter_node(state: RAGState):

        question = state["question"]

        history = state.get(
            "history",
            [],
        )

        attempts = state.get(
            "retrieval_attempts",
            0,
        )

        # -------------------------------------------------
        # Corrective rewrite
        # -------------------------------------------------

        if attempts > 0:

            previous_query = state.get(
                "rewritten_query",
                question,
            )

            rewritten_query = corrective_rewriter.invoke(
                {
                    "history": history,
                    "question": question,
                    "previous_query": previous_query,
                }
            )

            return {
                "rewritten_query": rewritten_query
            }

        # -------------------------------------------------
        # Normal conversational rewrite
        # -------------------------------------------------

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


# =========================================================
# Retrieval Node
# =========================================================


def create_retrieval_node(retriever):
    """
    Create a LangGraph retrieval node.

    The node:

    1. Determines the retrieval query.
    2. Invokes the existing hybrid + reranking retriever.
    3. Stores documents in graph state.
    4. Increments retrieval_attempts.
    """

    def retrieval_node(state: RAGState):

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

        attempts = state.get(
            "retrieval_attempts",
            0,
        )

        return {
            "documents": documents,
            "retrieval_attempts": attempts + 1,
        }

    return retrieval_node


# =========================================================
# Retrieval Grader
# =========================================================



# =========================================================
# Retrieval Decision Router
# =========================================================


MAX_RETRIEVAL_ATTEMPTS = 2


def route_after_grading(state: RAGState) -> str:
    """
    Decide what happens after retrieval evaluation.

    Relevant:
        -> generate

    Not relevant + retries available:
        -> rewrite

    Not relevant + retry limit reached:
        -> fallback
    """

    retrieval_relevant = state.get(
        "retrieval_relevant",
        False,
    )

    attempts = state.get(
        "retrieval_attempts",
        0,
    )

    if retrieval_relevant:
        return "generate"

    if attempts < MAX_RETRIEVAL_ATTEMPTS:
        return "rewrite"

    return "fallback"


# =========================================================
# Generation Node
# =========================================================


def create_generation_node(generation_chain):
    """
    Create the LangGraph generation node.

    IMPORTANT:

    This node receives already retrieved documents.

    Therefore it uses the generation-only chain and
    DOES NOT perform retrieval again.
    """

    def generation_node(state: RAGState):

        question = state["question"]

        documents = state.get(
            "documents",
            [],
        )

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


# =========================================================
# Answer Grader
# =========================================================



# =========================================================
# Safe Fallback Node
# =========================================================


def create_fallback_node():
    """
    Return a safe response when retrieval remains
    irrelevant after the maximum number of attempts.
    """

    def fallback_node(state: RAGState):

        return {
            "answer": (
                "I couldn't find enough relevant information "
                "in the available documents to answer this question."
            )
        }

    return fallback_node


# =========================================================
# Memory Update Node
# =========================================================


def create_update_memory_node():
    """
    Append the current question and answer to the history
    so the checkpointer (database) saves the conversation.
    """
    from langchain_core.messages import HumanMessage, AIMessage

    def update_memory_node(state: RAGState):
        question = state["question"]
        answer = state.get("answer", "")
        
        # Read existing history
        history = state.get("history", [])[:]
        
        # Append the new interaction
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
        
        return {"history": history}

    return update_memory_node