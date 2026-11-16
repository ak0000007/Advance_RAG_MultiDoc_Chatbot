"""
LangChain tools backed by the project's existing
retrieval infrastructure.
"""

import asyncio
import logging
import re
from typing import Optional, Any
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import PydanticOutputParser
from src.rag.grading import ChunkGrade, RetrievalGrade
from src.retrieval.citations import format_document_citation

logger = logging.getLogger(__name__)

def build_document_search_tool(
    retriever,
    grader_llm,
    rewriter_chain,
    max_attempts: int,
    irrelevance_threshold: float = 0.8,
    access_resolver: Optional[Any] = None,
    reranker: Optional[Any] = None,
    base_retriever: Optional[Any] = None,
    k: int = 5,
    overfetch_factor: int = 4,
):
    """
    Create a LangChain Tool backed by the project's
    existing hybrid + reranking retriever.
    Includes parallel per-chunk grading, filtering, self-correcting retrieval logic,
    tenant metadata isolation, and Salesforce record-level access filtering.
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

        # Resolve tenant/user metadata filter and SF auth context
        conf = (
            config.get("configurable", {})
            if isinstance(config, dict)
            else (getattr(config, "configurable", {}) or {})
        )
        tenant_id = conf.get("tenant_id")
        sf_auth_active = bool(conf.get("sf_auth_active", False))
        sf_user_id = conf.get("sf_user_id")
        sf_role = conf.get("sf_role")
        sf_org_id = conf.get("sf_org_id", "default")
        trace_id = conf.get("trace_id")

        raw_filter = conf.get("metadata_filter")
        metadata_filter = dict(raw_filter) if isinstance(raw_filter, dict) else {}

        if sf_auth_active:
            metadata_filter["sf_auth_active"] = True
            if tenant_id:
                metadata_filter["tenant_id"] = str(tenant_id)
            if sf_role:
                metadata_filter["sf_role"] = str(sf_role)
            if sf_user_id:
                metadata_filter["sf_user_id"] = str(sf_user_id)
        else:
            if tenant_id and "tenant_id" not in metadata_filter:
                metadata_filter["tenant_id"] = str(tenant_id)

        active_filter = metadata_filter if metadata_filter else None
        parser = PydanticOutputParser(pydantic_object=ChunkGrade)

        active_base = base_retriever or getattr(retriever, "base_retriever", retriever)

        async def _filter_by_sf_access(candidates: list[Any]) -> list[Any]:
            surviving = []
            account_docs = []
            for doc in candidates:
                meta = getattr(doc, "metadata", {}) or {}
                scope = meta.get("sf_scope")
                if scope == "none":
                    continue
                elif scope == "org":
                    surviving.append(doc)
                else:
                    aid = meta.get("sf_account_id")
                    if aid:
                        account_docs.append((doc, aid))

            if account_docs:
                distinct_accounts = {aid for _, aid in account_docs}
                readable = await access_resolver.readable_accounts(
                    user_id=sf_user_id,
                    account_ids=distinct_accounts,
                    org_id=sf_org_id,
                    trace_id=trace_id,
                )
                for doc, aid in account_docs:
                    if aid in readable:
                        surviving.append(doc)

            return surviving

        for attempt in range(1, max_attempts + 1):
            if sf_auth_active and access_resolver:
                # 1. Over-fetch candidates with factor (k * overfetch_factor)
                initial_k = k * overfetch_factor
                raw_candidates = await active_base.ainvoke(
                    {
                        "question": current_query,
                        "metadata_filter": active_filter,
                        "k": initial_k,
                    }
                )
                surviving = await _filter_by_sf_access(raw_candidates)

                # 2. If fewer than k remain, widen once (fetch k * 8) before giving up
                if len(surviving) < k:
                    widen_k = k * overfetch_factor * 2
                    wider_candidates = await active_base.ainvoke(
                        {
                            "question": current_query,
                            "metadata_filter": active_filter,
                            "k": widen_k,
                        }
                    )
                    surviving = await _filter_by_sf_access(wider_candidates)

                # 3. Rerank surviving documents to top k
                if surviving:
                    if reranker and hasattr(reranker, "arerank"):
                        documents = await reranker.arerank(current_query, surviving, top_k=k)
                    elif reranker and hasattr(reranker, "rerank"):
                        documents = await asyncio.to_thread(reranker.rerank, current_query, surviving, top_k=k)
                    else:
                        documents = surviving[:k]
                else:
                    documents = []

                # 4. If zero chunks survive, return neutral message without leaking
                if not documents:
                    return "No accessible documents found for this query."
            else:
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

            # Grade each chunk in parallel using asyncio.gather
            async def _grade_single_chunk(document, idx: int):
                xml_entry, _ = format_document_citation(document, idx)
                chunk_text = getattr(document, "page_content", "") or str(document)
                try:
                    grade = await grader_llm.ainvoke(
                        {
                            "question": query,
                            "chunk": chunk_text,
                            "context": chunk_text,
                            "format_instructions": parser.get_format_instructions(),
                        },
                        config=config,
                    )
                    return document, grade, xml_entry, None
                except Exception as exc:
                    return document, None, xml_entry, exc

            graded_results = await asyncio.gather(
                *[_grade_single_chunk(doc, idx) for idx, doc in enumerate(documents, start=1)]
            )

            # If all grader calls threw exceptions, fail-safe return with loud unverified banner
            failed_exceptions = [exc for _, _, _, exc in graded_results if exc is not None]
            if len(failed_exceptions) == len(graded_results) and failed_exceptions:
                all_context = "\n\n".join([xml for _, _, xml, _ in graded_results])
                first_exc = failed_exceptions[0]
                logger.error(
                    f"CRITICAL: Retrieval grading failed on attempt {attempt}: {first_exc}. "
                    f"Returning retrieval context with unverified flag to prevent silent failure.",
                    exc_info=True,
                )
                return f"[UNVERIFIED RETRIEVAL - Grader Evaluation Failed: {first_exc}]\n\n{all_context}"

            relevant_entries: list[tuple[Any, str]] = []
            irrelevant_entries: list[tuple[Any, Any]] = []
            had_grader_exception = bool(failed_exceptions)

            for doc, grade, xml_entry, exc in graded_results:
                if exc is not None:
                    logger.warning(
                        f"Per-chunk grader threw exception on attempt {attempt}: {exc}. Retaining chunk as unverified.",
                        exc_info=False,
                    )
                    relevant_entries.append((doc, xml_entry))
                    continue

                is_rel = getattr(grade, "is_relevant", getattr(grade, "relevant", True))
                if is_rel:
                    relevant_entries.append((doc, xml_entry))
                else:
                    irrelevant_entries.append((doc, grade))

            total_count = len(documents)
            irrelevant_count = len(irrelevant_entries)
            irrelevance_ratio = (irrelevant_count / total_count) if total_count > 0 else 1.0

            logger.info(
                f"Retrieval Attempt {attempt} | Query: {current_query} | "
                f"Total: {total_count} | Relevant: {len(relevant_entries)} | "
                f"Irrelevant: {irrelevant_count} (ratio: {irrelevance_ratio:.2f}) | "
                f"Threshold: {irrelevance_threshold}"
            )

            # Determine whether to trigger corrective retry
            should_retry = (len(relevant_entries) == 0) or (irrelevance_ratio >= irrelevance_threshold and not had_grader_exception)

            if not should_retry and relevant_entries:
                # Pass ONLY the filtered relevant chunks to generation
                filtered_context = "\n\n".join([xml for _, xml in relevant_entries])
                return filtered_context

            # If retry condition met and attempts remain, rewrite query and search again
            if attempt < max_attempts:
                reasons = [
                    getattr(g, "reason", "")
                    for _, g in irrelevant_entries
                    if getattr(g, "reason", "")
                ]
                reason_summary = "; ".join(reasons[:3]) or "Retrieved chunks did not contain relevant information to answer the question."
                current_query = await rewriter_chain.ainvoke(
                    {
                        "question": query,
                        "reason": reason_summary,
                    }
                )

        # Max attempts reached without satisfying relevance threshold
        if relevant_entries:
            # Return available relevant chunks
            return "\n\n".join([xml for _, xml in relevant_entries])

        # If zero chunks were verified as relevant across all attempts, return last attempt with warning note
        all_last_xml = [format_document_citation(d, i)[0] for i, d in enumerate(documents, start=1)]
        fallback_context = "\n\n".join(all_last_xml)
        return "Note: Relevance of these documents could not be fully confirmed.\n\n" + fallback_context

    return search_documents
