"""
Write safety guards — hard caps enforced in code, independent of LLM behaviour.

These guards run BEFORE any approval flow or DML. If a guard fails, the tool
returns an error string immediately and the graph never reaches human_approval.

SRP : This module owns all write-safety logic. Add new guards here only.
OCP : Each guard is a standalone function. Adding a new check = adding a function.
      Callers (tools) import what they need — nothing else changes.
LSP : All guards return Optional[str]: None = pass, str = error message.
      Callers treat the return type uniformly.

Guards implemented:
    1. single_record_guard   — rejects multiple IDs in one call
    2. bulk_intent_guard     — rejects LLM bulk-phrasing attempts
    3. session_write_cap     — rejects if per-session write limit is hit
"""

import re
from typing import Optional

# ── Salesforce ID patterns ────────────────────────────────────────────────────
# Standard SF ID is 15 or 18 alphanumeric chars, case-sensitive.
_SF_ID_RE = re.compile(r'^[a-zA-Z0-9]{15}([a-zA-Z0-9]{3})?$')
# Salesforce Opportunity IDs always start with the '006' key prefix.
_SF_OPP_PREFIX = '006'

# Tokens that indicate the LLM is trying to act on multiple records at once.
# Keep this list conservative — false positives are worse than false negatives.
_BULK_TOKENS = frozenset([
    "all", "every", "each", "multiple", "several", "batch",
    "all my", "all of", "all the", "all deals", "all opportunities",
    "close all", "update all", "mark all",
])


def single_record_guard(record_id: str, expected_prefix: Optional[str] = _SF_OPP_PREFIX) -> Optional[str]:
    """
    Enforce that exactly ONE valid Salesforce ID was supplied.

    Rejects:
      - Comma-separated lists  ("006aaa, 006bbb")
      - Whitespace-separated   ("006aaa 006bbb")
      - Wildcards / globs      ("*", "%", "all")
      - Non-SF-ID patterns
      - IDs that don't match expected_prefix (defaults to '006' Opportunity prefix)

    Returns None if the ID is valid, an error string if not.
    """
    raw = record_id.strip()

    # Reject obvious multi-record separators
    if ',' in raw or ';' in raw:
        return (
            "Hard cap: this tool updates exactly ONE record per call. "
            "Multiple IDs were supplied. Please call the tool once per record."
        )

    # Reject if more than one whitespace-separated token
    tokens = raw.split()
    if len(tokens) > 1:
        return (
            "Hard cap: this tool updates exactly ONE record per call. "
            f"Received {len(tokens)} tokens. Call the tool once per record."
        )

    single = tokens[0] if tokens else raw

    # Reject wildcards / glob patterns
    if single in ('*', '%', 'all', 'ALL'):
        return (
            "Hard cap: wildcard or 'all' is not allowed. "
            "Provide a single Salesforce ID."
        )

    # Reject if it doesn't look like a SF ID at all
    if not _SF_ID_RE.match(single):
        return (
            f"Validation error: '{single}' does not look like a valid "
            "15- or 18-character Salesforce ID."
        )

    # Reject IDs that don't match expected prefix
    if expected_prefix and not single.startswith(expected_prefix):
        return (
            f"Validation error: '{single}' does not start with expected prefix "
            f"'{expected_prefix}'."
        )

    return None  # guard passed


def bulk_intent_guard(new_status: str, raw_user_message: Optional[str] = None) -> Optional[str]:
    """
    Detect LLM bulk-phrasing in the arguments and optionally in the original
    user message (if surfaced by the agent).

    This is a secondary belt — single_record_guard is the primary.
    Returns None if safe, an error string if bulk intent is detected.
    """
    combined = (new_status + " " + (raw_user_message or "")).lower()

    for token in _BULK_TOKENS:
        if token in combined:
            return (
                f"Hard cap: bulk operations are not allowed (detected '{token}'). "
                "I can update opportunities one at a time. "
                "Please tell me which specific opportunity you'd like to update first."
            )

    return None  # guard passed


def session_write_cap(write_count: int, max_writes: int) -> Optional[str]:
    """
    Enforce a per-session write ceiling.

    Prevents runaway LLM loops from making unlimited writes in one conversation.
    Returns None if under the cap, an error string if at or over it.
    """
    if write_count >= max_writes:
        return (
            f"Hard cap: maximum of {max_writes} write operation(s) per session reached. "
            "Start a new conversation to perform additional updates."
        )

    return None  # guard passed
