"""
LangChain tools backed by the project's existing
retrieval infrastructure.
"""

import logging
import re
from typing import Optional
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import PydanticOutputParser
from src.rag.grading import RetrievalGrade
from src.retrieval.citations import format_document_citation

logger = logging.getLogger(__name__)

def build_document_search_tool(retriever, grader_llm, rewriter_chain, max_attempts: int):
    """
    Create a LangChain Tool backed by the project's
    existing hybrid + reranking retriever.
    Includes self-correcting retrieval logic and tenant metadata isolation.
    """

    @tool
    async def search_documents(query: str, config: RunnableConfig) -> str:
        """
        Search the internal knowledge base for company policies, employee guidelines,
        travel insurance terms, FAQs, and general reference documents.

        CRITICAL USAGE RESTRICTION:
        DO NOT use this tool for Salesforce CRM operations or CRM record data (Opportunities,
        Bookings, Travel Packages, Payments, deals). For any CRM record lookup or update,
        you MUST use the dedicated Salesforce tools instead.
        """
        current_query = query

        # Resolve tenant/user metadata filter from execution context
        conf = (
            config.get("configurable", {})
            if isinstance(config, dict)
            else (getattr(config, "configurable", {}) or {})
        )
        tenant_id = conf.get("tenant_id")
        raw_filter = conf.get("metadata_filter")
        metadata_filter = dict(raw_filter) if isinstance(raw_filter, dict) else {}
        if tenant_id and "tenant_id" not in metadata_filter:
            metadata_filter["tenant_id"] = str(tenant_id)

        active_filter = metadata_filter if metadata_filter else None
        
        for attempt in range(1, max_attempts + 1):
            documents = await retriever.ainvoke(
                {
                    "question": current_query,
                    "metadata_filter": active_filter,
                }
            )

            if not documents:
                return (
                    "No relevant documents were found "
                    "for this query."
                )

            results = []
            for index, document in enumerate(documents, start=1):
                xml_entry, _ = format_document_citation(document, index)
                results.append(xml_entry)
            context = "\n\n".join(results)

            try:
                parser = PydanticOutputParser(pydantic_object=RetrievalGrade)
                grade = await grader_llm.ainvoke(
                    {
                        "question": query,
                        "context": context,
                        "format_instructions": parser.get_format_instructions(),
                    }
                )
                relevant = grade.relevant
                reason = grade.reason
            except Exception as e:
                logger.warning(f"Retrieval grading failed: {e}. Defaulting to relevant.")
                relevant = True
                reason = "Grading failed, assumed relevant."

            logger.info(f"Retrieval Attempt {attempt} | Query: {current_query} | Relevant: {relevant} | Reason: {reason}")

            if relevant:
                return context

            if attempt < max_attempts:
                current_query = await rewriter_chain.ainvoke(
                    {
                        "question": query,
                        "reason": reason,
                    }
                )

        return "Note: Relevance of these documents could not be fully confirmed.\n\n" + context

    return search_documents
