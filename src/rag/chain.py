"""
RAG generation chain module.

Provides two levels of LCEL composition:

1. build_rag_chain()
   Complete RAG pipeline:
   retriever -> formatting -> prompt -> LLM -> parser

2. build_generation_chain()
   Generation-only pipeline:
   context -> prompt -> LLM -> parser

The generation-only chain is useful when LangGraph
has already performed retrieval and placed documents
into graph state.
"""

from typing import Callable, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


DEFAULT_PROMPT_TEMPLATE = """Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer using only the information in the context.

If the answer cannot be found in the context,
say that you do not have enough information."""


def format_docs(docs) -> str:
    """
    Convert retrieved LangChain Documents into plain text
    with source attribution.
    """

    if not docs:
        return ""

    formatted = []

    for i, doc in enumerate(docs, 1):
        source = (
            doc.metadata.get("source")
            or doc.metadata.get("document_id")
            or "unknown"
        )

        formatted.append(
            f"[{i}] (Source: {source})\n{doc.page_content}"
        )

    return "\n\n".join(formatted)


def build_generation_chain(
    llm,
    prompt_template: Optional[str] = None,
    output_parser=None,
    format_docs_fn: Optional[Callable] = None,
):
    """
    Build a generation-only LCEL chain.

    Expected input:

        {
            "context": list[Document],
            "question": str
        }

    Workflow:

        Documents
            ↓
        format_docs
            ↓
        Prompt
            ↓
        LLM
            ↓
        Output parser

    IMPORTANT:
    This chain does NOT perform retrieval.

    LangGraph uses this chain after its retrieval node
    has already obtained the relevant documents.
    """

    template = prompt_template or DEFAULT_PROMPT_TEMPLATE

    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", template)
        ]
    )

    parser = output_parser or StrOutputParser()

    doc_formatter = format_docs_fn or format_docs

    from langchain_core.runnables import RunnableLambda

    generation_chain = (
        {
            "context": RunnableLambda(
                lambda x: doc_formatter(x["context"])
            ),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | parser
    )

    return generation_chain


def build_rag_chain(
    retriever_runnable,
    llm,
    prompt_template: Optional[str] = None,
    output_parser=None,
    format_docs_fn: Optional[Callable] = None,
):
    """
    Build a complete RAG LCEL chain.

    Workflow:

        Question
            ↓
        Retriever
            ↓
        Documents
            ↓
        Format documents
            ↓
        Prompt
            ↓
        LLM
            ↓
        Output parser

    This remains the complete RAG chain used by RAGPipeline.
    """

    template = prompt_template or DEFAULT_PROMPT_TEMPLATE

    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", template)
        ]
    )

    parser = output_parser or StrOutputParser()

    doc_formatter = format_docs_fn or format_docs

    from langchain_core.runnables import RunnableLambda

    def _invoke_retriever(inputs):
        try:
            # Dynamic retrievers can receive the complete input dictionary.
            return retriever_runnable.invoke(inputs)

        except AttributeError as e:

            # Fallback for standard retrievers that expect
            # only the question string.
            if (
                "'dict' object" in str(e)
                and isinstance(inputs, dict)
                and "question" in inputs
            ):
                return retriever_runnable.invoke(
                    inputs["question"]
                )

            raise

    rag_chain = (
        {
            "context": RunnableLambda(
                _invoke_retriever
            ) | doc_formatter,

            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | parser
    )

    return rag_chain