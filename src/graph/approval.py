"""
Human-in-the-loop approval node for Salesforce Opportunity updates.

SRP: Single responsibility — interrupt / resume cycle for SF stage updates.
DIP: Receives salesforce_client via factory injection; no direct imports from client module.
OCP: Adding new approval types requires a new factory, not changes here.

Idempotency design:
  - approval_msg.tool_call_id is a stable, unique ID for every LLM tool-call decision.
  - It is passed as the Idempotency-Key HTTP header to the Salesforce client.
  - The key NEVER changes between retries of the same approval (LangGraph re-runs the
    node with the same checkpointed state), so duplicate network calls are deduplicated
    on the server side.
  - If tool_call_id is somehow absent, we fall back to a deterministic composite key
    (thread_id + opportunity_id + new_status) rather than sending no key or a random one.

Write counter:
  - write_count in state is incremented only on a confirmed successful write.
  - This is the single place that writes; the tool's session_write_cap reads it.
"""

import hashlib
import json
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from src.graph.state import RAGState

# Sentinel: no-op for add_messages
_NO_UPDATE: dict = {}


def _build_idempotency_key(
    tool_call_id: str | None,
    thread_id: str | None,
    opportunity_id: str,
    new_status: str,
) -> str:
    """
    Build a stable idempotency key.

    Primary  : tool_call_id  — unique per LLM decision, stable across node re-runs.
    Fallback : SHA-256 of thread_id + opportunity_id + new_status.
               Deterministic and collision-resistant even without a tool_call_id.

    A random UUID fallback (uuid4) would be WRONG: it would generate a different
    key every retry, defeating the entire purpose of idempotency.
    """
    if tool_call_id:
        return tool_call_id

    raw = f"{thread_id or ''}:{opportunity_id}:{new_status}"
    return hashlib.sha256(raw.encode()).hexdigest()


def build_human_approval_node(salesforce_client):
    """
    Factory: returns an async graph node with salesforce_client closed over.
    Call this only when SF is configured; otherwise omit the node entirely.
    """

    async def human_approval_node(state: RAGState, config: RunnableConfig) -> dict:
        """
        1. Scans messages for the pending approval ToolMessage.
        2. Calls interrupt() — graph freezes, returns payload to the frontend.
        3. On resume: executes or aborts the SF update based on human decision.
        4. Replaces the approval-request ToolMessage with the real outcome.
        5. Increments write_count in state on confirmed successful write.
        """
        messages = state.get("messages", [])

        # ── Find the ToolMessage carrying the approval request ──────────────
        approval_msg = None
        approval_data = None
        for msg in reversed(messages):
            if not hasattr(msg, "tool_call_id"):
                break
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict) and data.get("__requires_approval__"):
                    approval_msg = msg
                    approval_data = data
                    break
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue

        if approval_msg is None or approval_data is None:
            return _NO_UPDATE

        # ── Build idempotency key before interrupt() ─────────────────────────
        # Must be before interrupt() so the key is identical on first run
        # and on every re-run after resume — derived from stable state/config.
        thread_id: str | None = config.get("configurable", {}).get("thread_id")
        idempotency_key = _build_idempotency_key(
            tool_call_id=getattr(approval_msg, "tool_call_id", None),
            thread_id=thread_id,
            opportunity_id=approval_data["opportunity_id"],
            new_status=approval_data["new_status"],
        )

        # ── Freeze graph — returns resume value on second execution ─────────
        decision = interrupt({
            "action": "human_approval",
            "message": approval_data.get("message", "Approve this action?"),
            "opportunity_id": approval_data["opportunity_id"],
            "new_status": approval_data["new_status"],
            "idempotency_key": idempotency_key,
            "options": ["Approve", "Reject"],
        })

        # ── Guard: only accept canonical decision values ─────────────────────
        write_success = False
        result: str

        if decision not in ("Approve", "Reject"):
            result = (
                f"Invalid decision '{decision}'. "
                "Must be 'Approve' or 'Reject'. Action aborted."
            )
        elif decision == "Approve":
            try:
                response = await salesforce_client.update_opportunity_status(
                    username=approval_data["sf_username"],
                    opportunity_id=approval_data["opportunity_id"],
                    new_status=approval_data["new_status"],
                    idempotency_key=idempotency_key,
                )
                if response.get("isSuccess"):
                    result = f"Success: {response.get('message')}"
                    write_success = True
                else:
                    result = (
                        f"Failed: {response.get('message')} "
                        f"(HTTP {response.get('statusCode')})"
                    )
            except Exception as e:
                result = f"Failed to execute Salesforce update: {str(e)}"
        else:  # Reject
            result = (
                f"Action aborted: User rejected the update to "
                f"'{approval_data['new_status']}'."
            )

        # ── Replace approval-request msg with real outcome ──────────────────
        # Matching id causes add_messages to overwrite, not append.
        updated_msg = ToolMessage(
            content=result,
            tool_call_id=approval_msg.tool_call_id,
            id=approval_msg.id,
        )

        update: dict = {"messages": [updated_msg]}

        # Increment session write counter only on a confirmed successful write.
        # Single authoritative place — keeps tool's session_write_cap in sync.
        if write_success:
            update["write_count"] = state.get("write_count", 0) + 1

        return update

    return human_approval_node
