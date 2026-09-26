"""
Query Rewriter Module.

This module provides the query rewriter which takes the user's latest
question and the conversation history, and rewrites it into a standalone
search query.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


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
