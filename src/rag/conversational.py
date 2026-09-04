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

    The rewriter converts the latest user question into a
    standalone search query using conversation history.

    IMPORTANT:
    The rewriter does NOT answer the question.

    It only transforms the question for retrieval.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a query rewriting component inside a Retrieval-Augmented
Generation (RAG) system.

Your ONLY task is to rewrite the user's latest question into a
standalone SEARCH QUERY that can be sent to a document retriever.

You are NOT the answer generator.

STRICT RULES:

1. Output ONLY a search query.

2. NEVER answer the user's question.

3. NEVER provide facts, explanations, conclusions, or background
   information that are not part of the conversation history.

4. Use the conversation history ONLY to resolve references in the
   latest question.

5. Resolve references such as:
   - it
   - this
   - that
   - they
   - them
   - previous section
   - above
   - earlier discussion
   - the first one
   - the second one
   - this document
   - that document

6. When resolving a reference, use ONLY information explicitly
   established in the conversation history.

7. NEVER invent:
   - sections
   - documents
   - entities
   - events
   - topics
   - facts
   - dates
   - explanations

8. If a reference cannot be resolved from the conversation
   history, preserve the user's original wording instead of
   guessing.

   IMPORTANT REFERENCE RULE:

    If the latest question contains a reference whose target cannot
    be identified with high confidence from the conversation history,
    DO NOT rewrite that reference.

    Return the user's latest question unchanged.

    Do not use the general topic of the conversation as a substitute
    for an unresolved reference.

    For example:

    Conversation:
    User: What is the document about?
    Assistant: The document is about the history of the United States.

    Latest question:
    What happened in the previous section?

    Correct output:
    What happened in the previous section?

    Incorrect output:
    The previous section discussed the history of the United States.


9. Do NOT add an answer to the rewritten query.

10. Do NOT add a date, fact, explanation, or conclusion unless
    that information is explicitly needed to resolve the reference
    AND is already present in the conversation.

11. If the user's question is already a standalone question,
    return it unchanged.

12. Keep the rewritten query concise and retrieval-friendly.

13. Do not use phrases such as:
    "The answer is..."
    "The document says..."
    "It happened because..."
    "The answer to your question is..."

14. Do not output multiple alternatives.

15. Do not explain what you changed.

16. Return exactly ONE standalone search query.

EXAMPLES:

Example 1:

Conversation:
User: When was the United States founded?
Assistant: The United States was founded after thirteen British
colonies declared independence in 1776.

Latest question:
When did it happen?

Correct output:
When was the United States founded?

Incorrect output:
The United States was founded on July 4, 1776.

Example 2:

Conversation:
User: What are the three branches created by the Constitution?
Assistant: The three branches are the legislative, executive,
and judicial branches.

Latest question:
What does the second one do?

Correct output:
What does the executive branch do?

Example 3:

Conversation:
User: What is the document about?
Assistant: The document is about the history of the United States.

Latest question:
What happened in the previous section?

Correct output:
What happened in the previous section?

Incorrect output:
The previous section discussed early exploration and settlement.

Example 4:

Conversation:
User: What is the capital of France?
Assistant: Paris is the capital of France.

Latest question:
What is the population of Germany?

Correct output:
What is the population of Germany?
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

    Flow:

        User Question
              ↓
        Conversation History
              ↓
        Query Rewriter
              ↓
        Existing RAG Chain
              ↓
        Final Answer
    """

    query_rewriter = build_query_rewriter(llm)

    def run_conversational_rag(inputs):
        question = inputs["question"]
        history = inputs.get("history", [])
        metadata_filter = inputs.get("metadata_filter")

        # First question in a conversation:
        # no rewriting is necessary.
        if not history:
            retrieval_question = question

        # Follow-up question:
        # rewrite it using the conversation history.
        else:
            retrieval_question = query_rewriter.invoke(
                {
                    "history": history,
                    "question": question,
                }
            )

        # Send the rewritten/standalone question to the
        # existing RAG pipeline.
        answer = rag_chain.invoke(
            {
                "question": retrieval_question,
                "metadata_filter": metadata_filter,
            }
        )

        return answer

    conversational_chain = RunnableLambda(
        run_conversational_rag
    )

    conversational_chain_with_history = (
        RunnableWithMessageHistory(
            conversational_chain,
            history_store.get_history,
            input_messages_key="question",
            history_messages_key="history",
        )
    )

    return conversational_chain_with_history