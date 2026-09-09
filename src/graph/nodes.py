"""
LangGraph nodes for the RAG workflow.

Nodes are responsible for performing individual operations
using the shared RAGState.
"""

from src.graph.state import RAGState
from src.rag.conversational import build_query_rewriter

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableLambda

from src.rag.chain import format_docs


# ---------------------------------------------------------
# Question Router
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Query Rewriter
# ---------------------------------------------------------

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
            MessagesPlaceholder(variable_name="history"),
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
        | RunnableLambda(lambda text: text.strip())
    )

    def query_rewriter_node(state: RAGState):
        question = state["question"]
        history = state.get("history", [])
        attempts = state.get("retrieval_attempts", 0)

        # -------------------------------------------------
        # Corrective rewrite
        # -------------------------------------------------
        #
        # If retrieval has already been attempted, this
        # means the previous retrieval was judged poor.
        #
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


# ---------------------------------------------------------
# Retrieval Node
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Retrieval Grader
# ---------------------------------------------------------

def create_retrieval_grader_node(llm):
    """
    Create an LLM-based retrieval relevance grader.

    The grader receives:

        question
        +
        retrieved documents

    and returns:

        retrieval_relevant = True / False
    """

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

- Judge the retrieved context, not the quality of the question.
- The documents do not need to contain the complete answer.
- They must contain useful information that can help answer
  the question.
- Do not answer the question yourself.

Return EXACTLY one of:

YES
NO

Do not provide explanations.
Do not provide additional text.
                """,
            ),
            (
                "human",
                """
Question:
{question}

Retrieved context:
{context}

Is the retrieved context relevant to the question?
                """,
            ),
        ]
    )

    grader_chain = (
        grader_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(
            lambda text: text.strip().upper()
        )
    )

    def retrieval_grader_node(state: RAGState):

        question = state["question"]

        documents = state.get(
            "documents",
            [],
        )

        # If nothing was retrieved, retrieval is
        # automatically considered unsuccessful.
        if not documents:
            return {
                "retrieval_relevant": False
            }

        context = format_docs(documents)

        result = grader_chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )

        # -------------------------------------------------
        # Parse the model's YES / NO response.
        #
        # Fail closed:
        # Anything other than YES is treated as NO.
        # -------------------------------------------------

        if result.startswith("YES"):
            relevant = True
        else:
            relevant = False

        return {
            "retrieval_relevant": relevant
        }

    return retrieval_grader_node


# ---------------------------------------------------------
# Retrieval Decision Router
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Generation Node
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Safe Fallback Node
# ---------------------------------------------------------

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