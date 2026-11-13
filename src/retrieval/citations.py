"""
Citation extraction and document formatting for grounded RAG answers.

Provides utilities to:
1. Format retrieved LangChain Documents into secure, metadata-enriched XML tags.
2. Extract citation metadata (chunk ID, title, source, page, snippet) from text/messages.
3. Deduplicate and validate citations for API schemas and frontend display.
"""

import re
import html
from pathlib import Path
from typing import Any


_DOC_TAG_RE = re.compile(
    r'<document\s+([^>]+)>([\s\S]*?)</document>',
    re.IGNORECASE,
)
_ATTR_ID_RE = re.compile(r'id="([^"]+)"')
_ATTR_SOURCE_RE = re.compile(r'source="([^"]+)"')
_ATTR_TITLE_RE = re.compile(r'title="([^"]*)"')
_ATTR_PAGE_RE = re.compile(r'page="([^"]*)"')


def _parse_page_number(raw_page: Any) -> int | None:
    """Safely convert page representation to int if possible."""
    if raw_page is None:
        return None
    if isinstance(raw_page, int):
        return raw_page
    if isinstance(raw_page, str) and raw_page.strip().isdigit():
        return int(raw_page.strip())
    return None


def format_document_citation(document: Any, index: int = 1) -> tuple[str, dict[str, Any]]:
    """
    Format a single document chunk into an XML tag and companion citation dictionary.

    Returns:
        tuple[str, dict[str, Any]]:
            - xml_str: `<document id="..." source="..." [title="..."] [page="..."]>\\ncontent\\n</document>`
            - citation_dict: parsed citation metadata dict
    """
    metadata = getattr(document, "metadata", {}) or {}
    page_content = getattr(document, "page_content", "") or ""

    chunk_id = (
        metadata.get("chunk_id")
        or getattr(document, "id", None)
        or metadata.get("id")
        or str(index)
    )
    source = metadata.get("source") or metadata.get("file_name") or "unknown"
    title = metadata.get("title") or metadata.get("file_name") or None
    page = _parse_page_number(metadata.get("page"))

    # Sanitize content against control characters
    safe_content = re.sub(r"[\x00-\x1f\x7f]", "", page_content).strip()
    snippet = (safe_content[:160] + "...") if len(safe_content) > 160 else safe_content

    attrs = [
        f'id="{html.escape(str(chunk_id), quote=True)}"',
        f'source="{html.escape(str(source), quote=True)}"',
    ]
    if title:
        attrs.append(f'title="{html.escape(str(title), quote=True)}"')
    if page is not None:
        attrs.append(f'page="{page}"')

    xml_tag = f'<document {" ".join(attrs)}>\n{safe_content}\n</document>'

    citation = {
        "chunk_id": str(chunk_id),
        "source": str(source),
        "title": str(title) if title else None,
        "page": page,
        "snippet": snippet,
    }
    return xml_tag, citation


def extract_citations_from_text(text: str) -> list[dict[str, Any]]:
    """
    Parse XML `<document ...>` tags from text and extract citation records.
    Deduplicates citations by chunk_id while preserving document order.
    """
    if not text or "<document" not in text:
        return []

    citations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for match in _DOC_TAG_RE.finditer(text):
        attrs_str = match.group(1)
        body = match.group(2).strip()

        id_match = _ATTR_ID_RE.search(attrs_str)
        source_match = _ATTR_SOURCE_RE.search(attrs_str)
        title_match = _ATTR_TITLE_RE.search(attrs_str)
        page_match = _ATTR_PAGE_RE.search(attrs_str)

        chunk_id = html.unescape(id_match.group(1)) if id_match else f"doc-{len(citations) + 1}"
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        source = html.unescape(source_match.group(1)) if source_match else "unknown"
        title = html.unescape(title_match.group(1)) if title_match else None
        if not title and source != "unknown":
            title = Path(source).name

        page = _parse_page_number(page_match.group(1) if page_match else None)
        safe_snippet = re.sub(r"[\x00-\x1f\x7f]", "", body)
        snippet = (safe_snippet[:160] + "...") if len(safe_snippet) > 160 else safe_snippet

        citations.append({
            "chunk_id": str(chunk_id),
            "source": str(source),
            "title": str(title) if title else None,
            "page": page,
            "snippet": snippet,
        })

    return citations


def extract_citations_from_messages(
    messages: list,
    current_turn_only: bool = True,
) -> list[dict[str, Any]]:
    """
    Inspect a sequence of LangChain / LangGraph messages and extract citations
    from any tool or assistant messages containing `<document>` tags.

    Args:
        messages: Sequence of conversation messages.
        current_turn_only: When True (default), restricts inspection strictly to
            messages within the latest conversation turn (after the most recent
            HumanMessage). This prevents previous document queries from leaking
            citations into subsequent turns or Salesforce CRM operations.
    """
    if not messages:
        return []

    target_messages = messages
    if current_turn_only:
        # Find index of latest human message
        last_human_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            m = messages[idx]
            msg_type = getattr(m, "type", "")
            if msg_type in ("human", "user"):
                last_human_idx = idx
                break
        if last_human_idx != -1:
            target_messages = messages[last_human_idx + 1:]

    # Check if search_documents was actually invoked in target messages
    # If target messages only invoked Salesforce tools, citations must be empty.
    has_doc_search = False
    for msg in target_messages:
        # Check tool message name
        if getattr(msg, "name", "") == "search_documents":
            has_doc_search = True
            break
        # Check tool calls in AI message
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            if tc_name == "search_documents":
                has_doc_search = True
                break
        if has_doc_search:
            break

    # If search_documents tool was not called in this turn, citations must be empty
    # (unless raw messages without tool names are provided, e.g. in standalone unit tests)
    has_any_tool_calls = any(
        getattr(m, "name", "") or getattr(m, "tool_calls", None)
        for m in target_messages
    )
    if has_any_tool_calls and not has_doc_search:
        return []

    all_citations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for msg in target_messages:
        content = getattr(msg, "content", "")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            texts = [
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            ]
            text = "".join(texts)

        if "<document" in text:
            for citation in extract_citations_from_text(text):
                cid = citation.get("chunk_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_citations.append(citation)

    return all_citations


# ==============================================================================
# Inline Per-Claim Citation Verification Engine
# ==============================================================================

_INLINE_CITATION_RE = re.compile(
    r'\[(?:Doc:\s*)?([^,\]\n]+?)(?:,\s*(?:[Pp]age\s*|[Pp]\.?\s*)?(\d+))?\](?!\()',
    re.IGNORECASE,
)


def _normalize_name(val: str | None) -> str:
    if not val:
        return ""
    name = Path(str(val)).name.lower()
    return re.sub(r'[^a-z0-9]', '', name)


def verify_and_filter_inline_citations(
    text: str,
    retrieved_citations: list[dict[str, Any]],
    mode: str = "flag",
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """
    Check that every inline citation written by the generator points to a
    chunk that was actually retrieved in this turn. Drop or flag hallucinated citations.

    Args:
        text: Generated answer text containing inline citations (e.g. `[doc.pdf, page 2]`).
        retrieved_citations: List of citation dictionaries retrieved in the current turn.
        mode: "flag" to replace unverified citation with `[UNVERIFIED CITATION: ...]`,
              "drop" to remove unverified citation from the text completely.

    Returns:
        tuple[str, list[dict[str, Any]], dict[str, Any]]:
            - verified_text: Sanitized text with valid citations preserved and invalid flagged/dropped.
            - verified_citations: Subset of retrieved_citations actually referenced in the text.
            - summary: Detailed verification report.
    """
    if not text:
        return text, [], {
            "total_inline_citations": 0,
            "verified_count": 0,
            "unverified_count": 0,
            "verified_citations": [],
            "unverified_citations": [],
            "mode": mode,
        }

    retrieved_lookup = []
    for rc in (retrieved_citations or []):
        rc_src_norm = _normalize_name(rc.get("source"))
        rc_title_norm = _normalize_name(rc.get("title"))
        rc_id_norm = _normalize_name(str(rc.get("chunk_id", "")))
        rc_page = rc.get("page")
        retrieved_lookup.append({
            "record": rc,
            "src_norm": rc_src_norm,
            "title_norm": rc_title_norm,
            "id_norm": rc_id_norm,
            "page": rc_page,
        })

    verified_citations: list[dict[str, Any]] = []
    seen_verified_ids: set[str] = set()
    unverified_citations: list[dict[str, Any]] = []
    total_found = 0

    def _replace_match(match: re.Match) -> str:
        nonlocal total_found
        raw_match = match.group(0)

        # Skip already flagged citations or non-document tags
        if "UNVERIFIED" in raw_match:
            return raw_match

        source_raw = (match.group(1) or "").strip()
        page_raw = match.group(2)
        cited_page = int(page_raw) if page_raw and page_raw.isdigit() else None
        source_norm = _normalize_name(source_raw)

        if not source_norm or source_norm in ("doc", "page", "source"):
            return raw_match

        total_found += 1

        # Search for a match in retrieved chunks
        matched_rc = None
        for item in retrieved_lookup:
            is_name_match = (
                source_norm in item["src_norm"]
                or item["src_norm"] in source_norm
                or (item["title_norm"] and (source_norm in item["title_norm"] or item["title_norm"] in source_norm))
                or source_norm == item["id_norm"]
            )
            if not is_name_match:
                continue

            if cited_page is not None and item["page"] is not None:
                if cited_page != item["page"]:
                    continue

            matched_rc = item["record"]
            break

        if matched_rc:
            cid = str(matched_rc.get("chunk_id", ""))
            if cid and cid not in seen_verified_ids:
                seen_verified_ids.add(cid)
                verified_citations.append(matched_rc)
            return raw_match

        unverified_record = {
            "raw": raw_match,
            "source": source_raw,
            "page": cited_page,
        }
        unverified_citations.append(unverified_record)

        if mode == "drop":
            return ""
        else:
            page_suffix = f", page {cited_page}" if cited_page is not None else ""
            return f"[UNVERIFIED CITATION: {source_raw}{page_suffix}]"

    verified_text = _INLINE_CITATION_RE.sub(_replace_match, text)

    if mode == "drop":
        verified_text = re.sub(r' +', ' ', verified_text).strip()

    summary = {
        "total_inline_citations": total_found,
        "verified_count": total_found - len(unverified_citations),
        "unverified_count": len(unverified_citations),
        "verified_citations": verified_citations,
        "unverified_citations": unverified_citations,
        "mode": mode,
    }

    return verified_text, verified_citations, summary
