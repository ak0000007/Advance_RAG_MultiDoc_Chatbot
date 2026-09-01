from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    """Convert retrieved Documents into plain text."""

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def build_rag_chain(retriever_runnable, llm):
    """
    Builds the RAG chain.
    Depends on abstractions (retriever_runnable), not concrete implementations (QdrantStore).
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            """Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer using only the information in the context.

If the answer cannot be found in the context,
say that you do not have enough information."""
        )
    ])

    rag_chain = (
        {
            # retriever_runnable is responsible for handling the input dict
            # (including metadata_filter) and returning Documents.
            "context": retriever_runnable | format_docs,
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
    )

    return rag_chain