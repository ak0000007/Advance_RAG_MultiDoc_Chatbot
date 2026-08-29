from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    """Convert retrieved LangChain Documents into plain text."""
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def build_rag_chain(retriever, llm):
    """
    Build a modern LangChain RAG pipeline.

    Flow:
        Question
            ↓
        Retriever
            ↓
        Documents
            ↓
        Context formatting
            ↓
        Prompt
            ↓
        LLM
    """

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful CRM assistant.

Answer the user's question using only the information
provided in the context.

If the answer cannot be found in the context,
clearly say that you do not have enough information.

Context:
{context}
"""
        ),
        (
            "human",
            "{question}"
        ),
    ])

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    return rag_chain