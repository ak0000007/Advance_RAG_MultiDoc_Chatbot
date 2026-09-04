"""
Conversational RAG application layer.

Responsibilities
----------------
1. Maintain short-term conversation history.
2. Rewrite follow-up questions into standalone retrieval queries.
3. Preserve metadata filters.
4. Execute the existing RAG chain.
5. Expose a LangChain RunnableWithMessageHistory interface.

Architecture

User Question
      |
      v
Conversation History
      |
      v
Query Rewriter
      |
      v
Standalone Retrieval Question
      |
      v
Existing RAG Chain
      |
      v
Final Answer
      |
      v
Conversation History
"""

from typing import Dict, Optional, Any

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

from langchain_core.messages import (
    BaseMessage,
)

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.runnables import (
    RunnableLambda,
    RunnableWithMessageHistory,
)


# ============================================================
# 1. CONVERSATION STORE
# ============================================================

class ConversationStore:
    """
    In-memory short-term conversation store.

    Each session_id gets an independent conversation history.

    Example:

        session_1
            HumanMessage
            AIMessage
            HumanMessage
            AIMessage

        session_2
            HumanMessage
            AIMessage

    This is intentionally simple for Phase 4.

    Later, the same interface can be backed by:
        - Redis
        - PostgreSQL
        - MongoDB
        - another persistent store
    """

    def __init__(self):
        self._store: Dict[
            str,
            BaseChatMessageHistory
        ] = {}

    def get_history(
        self,
        session_id: str
    ) -> BaseChatMessageHistory:
        """
        Get the history associated with a session.

        Creates a new history if the session does not exist.
        """

        if session_id not in self._store:
            self._store[
                session_id
            ] = InMemoryChatMessageHistory()

        return self._store[session_id]

    def clear_history(
        self,
        session_id: str
    ) -> None:
        """
        Delete all messages from a session.
        """

        if session_id in self._store:
            self._store[session_id].clear()

    def list_sessions(self):
        """
        Return all active session IDs.

        Useful later for application/session management.
        """

        return list(self._store.keys())


# ============================================================
# 2. QUERY REWRITER
# ============================================================

def build_query_rewriter(llm):
    """
    Build the history-aware query rewriting chain.

    Input:

        history
        question

    Output:

        standalone retrieval query

    Example:

        History:
            User: What is RAG?
            AI: RAG combines retrieval and generation.

        Question:
            Why is it useful?

        Output:
            Why is Retrieval-Augmented Generation useful?
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the query rewriting component of a RAG system.

Your ONLY job is to convert the user's latest question
into a standalone search query.

Rules:

1. Preserve the user's original intent.
2. Use conversation history only to resolve references.
3. Resolve references such as:
   - it
   - this
   - that
   - they
   - previous section
   - above
   - earlier discussion
4. Include important entities from the conversation when needed.
5. Do NOT answer the question.
6. Do NOT explain your reasoning.
7. Do NOT add information that is not present in the conversation.
8. Output ONLY the standalone search query.
9. If the question is already standalone, return it unchanged.
""",
            ),
            MessagesPlaceholder(
                variable_name="history"
            ),
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
        | RunnableLambda(
            lambda text: text.strip()
        )
    )


# ============================================================
# 3. CONVERSATIONAL RAG
# ============================================================

def build_conversational_rag(
    rag_chain,
    llm,
    history_store: ConversationStore,
):
    """
    Build the complete conversational RAG application.

    The existing RAG chain remains responsible for:

        retrieval
        hybrid retrieval
        reranking
        prompt construction
        answer generation

    This layer is responsible for:

        conversation history
        query rewriting
        session management
        metadata filter propagation
    """

    query_rewriter = build_query_rewriter(llm)

    def run_conversational_rag(
        inputs: Dict[str, Any]
    ):
        """
        Execute one conversational RAG turn.

        Expected input:

        {
            "question": "...",
            "history": [...],
            "metadata_filter": {...}
        }

        Returns:

            clean answer string
        """

        question = inputs["question"]

        history = inputs.get(
            "history",
            []
        )

        metadata_filter = inputs.get(
            "metadata_filter"
        )

        # ----------------------------------------------------
        # STEP 1
        # Determine retrieval question
        # ----------------------------------------------------

        if history:

            retrieval_question = (
                query_rewriter.invoke(
                    {
                        "history": history,
                        "question": question,
                    }
                )
            )

        else:

            retrieval_question = question

        # ----------------------------------------------------
        # STEP 2
        # Run the existing RAG chain
        # ----------------------------------------------------

        answer = rag_chain.invoke(
            {
                "question": retrieval_question,
                "metadata_filter": metadata_filter,
            }
        )

        return answer

    # --------------------------------------------------------
    # STEP 3
    # Convert function into LangChain Runnable
    # --------------------------------------------------------

    conversational_chain = RunnableLambda(
        run_conversational_rag
    )

    # --------------------------------------------------------
    # STEP 4
    # Attach session-based message history
    # --------------------------------------------------------

    conversational_chain_with_history = (
        RunnableWithMessageHistory(
            conversational_chain,
            history_store.get_history,

            input_messages_key="question",

            history_messages_key="history",
        )
    )

    return conversational_chain_with_history