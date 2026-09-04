"""
Conversational RAG.

This module adds conversation awareness on top of the existing
RAG chain without changing the retrieval or generation pipeline.

Architecture:

User Question
      ↓
Conversation History
      ↓
Query Rewriter
      ↓
Standalone Retrieval Question
      ↓
Existing RAG Chain
      ↓
Final Answer
      ↓
Conversation History
"""

from typing import Dict

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import (
    RunnableLambda,
    RunnableWithMessageHistory,
)


class ConversationStore:
    """
    Simple in-memory conversation store.

    Each session_id gets its own chat history.

    Example:

        demo_user
            ├── HumanMessage
            ├── AIMessage
            ├── HumanMessage
            └── AIMessage
    """

    def __init__(self):
        self._store: Dict[str, BaseChatMessageHistory] = {}

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Return the chat history for a session.

        Creates a new history if the session does not exist.
        """

        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()

        return self._store[session_id]

    def clear_history(self, session_id: str) -> None:
        """
        Clear the conversation for a session.
        """

        if session_id in self._store:
            self._store[session_id].clear()


def build_query_rewriter(llm):
    """
    Build an LCEL query-rewriting chain.

    The chain converts a follow-up question into a standalone
    question using the conversation history.

    Example:

        History:
            User: What is RAG?
            AI: RAG combines retrieval with generation.

        Follow-up:
            "Why is it useful?"

        Rewritten:
            "Why is retrieval-augmented generation useful?"
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a query transformation component for a RAG system.

Your job is to rewrite the user's latest question into a
standalone search query that can be understood without the
conversation history.

Rules:

1. Preserve the user's original intent.
2. Resolve references such as:
   - it
   - this
   - that
   - previous section
   - above
   - earlier discussion
3. Do not answer the question.
4. Do not explain your reasoning.
5. Output ONLY the rewritten standalone question.
6. If the question is already standalone, return it unchanged.
""",
            ),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "{question}",
            ),
        ]
    )

    return (
        prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda text: text.strip())
    )


def build_conversational_rag(
    rag_chain,
    llm,
    history_store: ConversationStore,
):
    """
    Build the complete conversational RAG chain.

    The existing RAG chain remains responsible for:

        retrieval
        hybrid search
        reranking
        prompt construction
        answer generation

    This layer adds:

        conversation history
        query rewriting
        session-based memory
    """

    query_rewriter = build_query_rewriter(llm)

    def run_conversational_rag(inputs):
        """
        Execute conversational RAG.

        Input:

            {
                "question": "...",
                "history": [...],
                "metadata_filter": {...}
            }

        Output:

            clean final answer string
        """

        question = inputs["question"]

        history = inputs.get("history", [])

        metadata_filter = inputs.get(
            "metadata_filter"
        )

        # --------------------------------------------------
        # STEP 1
        # --------------------------------------------------
        # If there is no previous conversation, the question
        # is already standalone.
        #
        # Do NOT waste an LLM call rewriting it.
        # --------------------------------------------------

        if not history:

            retrieval_question = question

        # --------------------------------------------------
        # STEP 2
        # --------------------------------------------------
        # If conversation history exists, rewrite the
        # follow-up question into a standalone query.
        # --------------------------------------------------

        else:

            retrieval_question = query_rewriter.invoke(
                {
                    "history": history,
                    "question": question,
                }
            )

        # --------------------------------------------------
        # STEP 3
        # --------------------------------------------------
        # Pass the rewritten question into our EXISTING
        # RAG pipeline.
        #
        # We do not duplicate retrieval logic here.
        # --------------------------------------------------

        answer = rag_chain.invoke(
            {
                "question": retrieval_question,
                "metadata_filter": metadata_filter,
            }
        )

        # --------------------------------------------------
        # STEP 4
        # --------------------------------------------------
        # Return ONLY the final answer.
        #
        # RunnableWithMessageHistory will store:
        #
        # HumanMessage(question)
        # AIMessage(answer)
        #
        # It will NOT store the rewritten query.
        # --------------------------------------------------

        return answer

    conversational_chain = RunnableLambda(
        run_conversational_rag
    )

    # ------------------------------------------------------
    # Add LangChain-managed conversation history.
    # ------------------------------------------------------

    conversational_chain_with_history = (
        RunnableWithMessageHistory(
            conversational_chain,
            history_store.get_history,

            # The user's question is stored as HumanMessage.
            input_messages_key="question",

            # History is injected into the "history" input.
            history_messages_key="history",
        )
    )

    return conversational_chain_with_history