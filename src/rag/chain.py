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
    """Build a simple RAG chain descriptor.

    Returns a dict with the provided components and sensible defaults.
    """

    if prompt_template is None:
        prompt_template = DEFAULT_PROMPT_TEMPLATE

    if format_docs_fn is None:
        format_docs_fn = format_docs

    return {
        "retriever": retriever_runnable,
        "llm": llm,
        "prompt_template": prompt_template,
        "output_parser": output_parser or StrOutputParser(),
        "format_docs_fn": format_docs_fn,
    }