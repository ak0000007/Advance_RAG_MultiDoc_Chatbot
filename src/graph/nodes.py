"""
LangGraph nodes for the RAG workflow.

Nodes are responsible for performing individual operations
using the shared RAGState.
"""

from pydantic import BaseModel, Field

from src.graph.state import RAGState
from src.rag.conversational import build_query_rewriter
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


def create_retrieval_grader_node(llm):
    """
    Create an LLM-based retrieval relevance grader.

    Instead of returning raw YES / NO text, the grader
    returns a validated RetrievalGrade Pydantic object.

    Example:

        RetrievalGrade(
            relevant=True,
            reason="The documents discuss..."
        )
    """

    parser = PydanticOutputParser(
        pydantic_object=RetrievalGrade
    )

    grader_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a retrieval relevance grader inside a
Retrieval-Augmented Generation system.

Your task is to determine whether the retrieved
documents contain information relevant to answering
the user's question.

IMPORTANT:

- Judge the retrieved context.
- Do not answer the question yourself.
- The documents do not need to contain the complete answer.
- They must contain useful information that can help answer
  the question.
- Do not use outside knowledge.
- Do not invent information.

Return the required structured format.

{format_instructions}
                """,
            ),
            (
                "human",
                """
Question:

{question}

Retrieved context:

{context}

Determine whether the retrieved context is relevant.
                """,
            ),
        ]
    )

    grader_chain = (
        grader_prompt
        | llm
        | parser
    )

    def retrieval_grader_node(state: RAGState):

        question = state["question"]

        documents = state.get(
            "documents",
            [],
        )

        # -------------------------------------------------
        # No documents
        # -------------------------------------------------

        if not documents:

            return {
                "retrieval_relevant": False,
                "retrieval_grade_reason": (
                    "No documents were retrieved."
                ),
            }

        # -------------------------------------------------
        # Format retrieved documents
        # -------------------------------------------------

        context = format_docs(
            documents
        )

        # -------------------------------------------------
        # Structured grading
        # -------------------------------------------------

        grade = grader_chain.invoke(
            {
                "question": question,
                "context": context,
                "format_instructions": (
                    parser.get_format_instructions()
                ),
            }
        )

        return {
            "retrieval_relevant": grade.relevant,
            "retrieval_grade_reason": grade.reason,
        }

    return retrieval_grader_node


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


def create_answer_grader_node(llm):
    """
    Evaluate whether the generated answer is supported
    by the retrieved context.

    Instead of raw YES / NO text, the grader returns:

        AnswerGrade(
            supported=True/False,
            reason="..."
        )
    """

    parser = PydanticOutputParser(
        pydantic_object=AnswerGrade
    )

    grader_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an answer quality evaluator inside a
Retrieval-Augmented Generation system.

Your task is to determine whether the generated answer
is supported by the provided retrieved context.

A supported answer must be grounded in information
contained in the context.

IMPORTANT RULES:

1. Do not use outside knowledge.

2. Judge only whether the answer is supported by
   the provided context.

3. The answer does not need to use the exact wording
   of the context.

4. Reasonable summarization or paraphrasing is allowed.

5. If the answer contains claims that are not supported
   by the context, return supported=false.

6. If the context does not provide enough information
   to support the answer, return supported=false.

7. Do not rewrite the answer.

8. Do not answer the original question.

Return the required structured format.

{format_instructions}
                """,
            ),
            (
                "human",
                """
Question:

{question}

Retrieved context:

{context}

Generated answer:

{answer}

Determine whether the generated answer is fully
supported by the retrieved context.
                """,
            ),
        ]
    )

    grader_chain = (
        grader_prompt
        | llm
        | parser
    )

    def answer_grader_node(state: RAGState):

        question = state["question"]

        documents = state.get(
            "documents",
            [],
        )

        answer = state.get(
            "answer",
            "",
        )

        # -------------------------------------------------
        # No answer
        # -------------------------------------------------

        if not answer:

            return {
                "answer_supported": False,
                "answer_grade_reason": (
                    "No answer was generated."
                ),
            }

        # -------------------------------------------------
        # Format context
        # -------------------------------------------------

        context = format_docs(
            documents
        )

        # -------------------------------------------------
        # Structured grading
        # -------------------------------------------------

        grade = grader_chain.invoke(
            {
                "question": question,
                "context": context,
                "answer": answer,
                "format_instructions": (
                    parser.get_format_instructions()
                ),
            }
        )

        return {
            "answer_supported": grade.supported,
            "answer_grade_reason": grade.reason,
        }

    return answer_grader_node


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