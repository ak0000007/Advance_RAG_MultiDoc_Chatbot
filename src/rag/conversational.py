from typing import Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory


class InMemoryChatHistory(BaseChatMessageHistory):
    """
    Simple in-memory chat history.

    Each session owns an independent list of messages.
    """

    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def clear(self):
        self.messages = []


class ConversationStore:
    """
    Stores chat histories by session_id.

    Example:

        session_1 -> history
        session_2 -> history
    """

    def __init__(self):
        self._store = {}

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatHistory()

        return self._store[session_id]

    def clear_history(self, session_id: str):
        if session_id in self._store:
            self._store[session_id].clear()


def build_query_rewriter(llm):
    """
    Creates an LCEL runnable that converts a conversational
    question into a standalone retrieval question.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a query transformation component for a
document question-answering system.

Your job is to rewrite the user's latest question into a
standalone question that can be understood without the
conversation history.

Rules:
1. Resolve references such as "it", "they", "that", "previous year",
   "this company", etc. using the conversation history.
2. Preserve the user's original intent.
3. Do not answer the question.
4. Do not add information that is not present in the conversation.
5. If the question is already standalone, return it unchanged.
6. Return only the rewritten question.""",
            ),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "Current question:\n{question}",
            ),
        ]
    )

    return prompt | llm | RunnableLambda(
        lambda message: message.content
        if hasattr(message, "content")
        else str(message)
    )


def build_conversational_rag(
    rag_chain,
    llm,
    history_store: ConversationStore,
):
    """
    Wrap an existing RAG chain with conversational query rewriting
    and session-based message history.

    The existing rag_chain is expected to accept:

        {
            "question": str,
            "metadata_filter": dict | None
        }
    """

    query_rewriter = build_query_rewriter(llm)

    def _invoke(inputs):
        question = inputs["question"]
        history = inputs.get("history", [])
        metadata_filter = inputs.get("metadata_filter")

        # --------------------------------------------------
        # 1. Transform conversational question
        # --------------------------------------------------
        rewritten_question = query_rewriter.invoke(
            {
                "history": history,
                "question": question,
            }
        )

        # --------------------------------------------------
        # 2. Run the existing RAG pipeline
        # --------------------------------------------------
        answer = rag_chain.invoke(
            {
                "question": rewritten_question,
                "metadata_filter": metadata_filter,
            }
        )

        return answer

    conversational_chain = RunnableLambda(_invoke)

    return RunnableWithMessageHistory(
        conversational_chain,
        history_store.get_history,
        input_messages_key="question",
        history_messages_key="history",
    )