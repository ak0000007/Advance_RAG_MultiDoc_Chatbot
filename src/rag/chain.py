"""
RAG generation chain module.

Assembles retriever, prompt template, LLM, and output parser into an executable pipeline.
Follows SOLID:
- Open/Closed: prompt template, doc formatter, and output parser are configurable.
- Dependency Inversion: depends on LangChain Runnables/abstractions, not concrete stores or models.
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
    """Convert retrieved Documents into plain text with source attribution."""
    if not docs:
        return ""

    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source") or doc.metadata.get("document_id") or "unknown"
        formatted.append(f"[{i}] (Source: {source})\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain(
    retriever_runnable,
    llm,
    prompt_template: Optional[str] = None,
    output_parser=None,
    format_docs_fn: Optional[Callable] = None,
):
    """
    Builds a production-ready LCEL RAG chain.

    Args:
        retriever_runnable: LangChain Runnable returning list[Document].
        llm: LangChain BaseLanguageModel or Runnable.
        prompt_template: Optional custom prompt template string.
        output_parser: Optional custom output parser.
        format_docs_fn: Optional formatting function.
    """
    template = prompt_template or DEFAULT_PROMPT_TEMPLATE
    prompt = ChatPromptTemplate.from_messages([("human", template)])
    parser = output_parser or StrOutputParser()
    doc_formatter = format_docs_fn or format_docs

    # LCEL pipeline (RunnableSequence), NOT a dictionary
    rag_chain = (
        {
            "context": retriever_runnable | doc_formatter,
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | parser
    )

    return rag_chain