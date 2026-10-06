"""
Human-in-the-loop approval node for Salesforce write operations.

Handles four action types dispatched from different update tools:
  - opportunity_update  (legacy, default)
  - booking_update
  - travel_package_update
  - payment_update

SRP: Single responsibility — interrupt / resume cycle for SF updates.
DIP: Receives salesforce_client via factory injection.
OCP: New action types → add a branch in _execute_update only.

Idempotency:
  - tool_call_id is stable per LLM decision, unchanged on re-runs.
  - Falls back to SHA-256(thread_id + record_id + action_type).

Write counter:
  - write_count in state incremented ONLY on confirmed successful write.
"""

import hashlib
import json
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from src.graph.state import RAGState
from src.audit.log import record_proposed, record_decision, record_outcome

_NO_UPDATE: dict = {}


def _build_idempotency_key(
    tool_call_id: str | None,
    thread_id: str | None,
    record_id: str,
    action_suffix: str,
) -> str:
    if tool_call_id:
        return tool_call_id
    raw = f"{thread_id or ''}:{record_id}:{action_suffix}"
    return hashlib.sha256(raw.encode()).hexdigest()


def build_human_approval_node(salesforce_client, pool=None):
    """
    Factory: returns an async graph node with salesforce_client closed over.
    Handles all action types that carry __requires_approval__ in their payload.
    """

    async def _execute_update(action_type: str, approval_data: dict, idempotency_key: str) -> dict:
        """Call the correct client method based on action_type. Returns SF response dict."""
        username = approval_data["sf_username"]
        action_type = action_type or "opportunity_update"

        if action_type == "opportunity_update":
            return await salesforce_client.update_opportunity_status(
                username=username,
                opportunity_id=approval_data["opportunity_id"],
                new_status=approval_data["new_status"],
                idempotency_key=idempotency_key,
            )
        elif action_type == "booking_update":
            return await salesforce_client.update_booking(
                username=username,
                booking_id=approval_data["record_id"],
                update_fields=approval_data.get("update_fields", {}),
                idempotency_key=idempotency_key,
            )
        elif action_type == "travel_package_update":
            return await salesforce_client.update_travel_package(
                username=username,
                package_id=approval_data["record_id"],
                update_fields=approval_data.get("update_fields", {}),
                idempotency_key=idempotency_key,
            )
        elif action_type == "payment_update":
            return await salesforce_client.update_payment(
                username=username,
                payment_id=approval_data["record_id"],
                update_fields=approval_data.get("update_fields", {}),
                idempotency_key=idempotency_key,
            )
        else:
            return {"isSuccess": False, "message": f"Unknown action_type: '{action_type}'."}

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

        action_type = approval_data.get("action_type") or "opportunity_update"

        # Resolve generic record_id / record_name (new types) or legacy opportunity_id fields
        record_id = approval_data.get("record_id") or approval_data.get("opportunity_id", "")
        record_name = (
            approval_data.get("record_name")
            or approval_data.get("opportunity_name")
            or record_id
        )
        # action_suffix used for fallback idempotency key
        action_suffix = approval_data.get("new_status") or action_type

        # ── Build idempotency key before interrupt() ─────────────────────────
        thread_id: str | None = config.get("configurable", {}).get("thread_id") if config else None
        idempotency_key = _build_idempotency_key(
            tool_call_id=getattr(approval_msg, "tool_call_id", None),
            thread_id=thread_id,
            record_id=record_id,
            action_suffix=action_suffix,
        )

        # ── Audit: record proposed write before interrupt ───────────────────
        if pool is not None:
            sf_user = (
                approval_data.get("sf_username")
                or (config.get("configurable", {}).get("sf_username") if config else None)
                or state.get("sf_username")
                or ""
            )
            await record_proposed(
                pool=pool,
                thread_id=thread_id or "",
                sf_username=sf_user,
                action_type=action_type,
                record_id=record_id,
                record_name=record_name,
                idempotency_key=idempotency_key,
                proposed_payload=approval_data,
                status="proposed",
            )

        # ── Freeze graph — returns resume value on second execution ─────────
        decision = interrupt({
            "action": "human_approval",
            "message": approval_data.get("message", "Approve this action?"),
            "record_id": record_id,
            "record_name": record_name,
            # Keep legacy keys for existing frontend/clients
            "opportunity_id": approval_data.get("opportunity_id", record_id),
            "opportunity_name": approval_data.get("opportunity_name", record_name),
            "new_status": approval_data.get("new_status", ""),
            "action_type": action_type,
            "idempotency_key": idempotency_key,
            "options": ["Approve", "Reject"],
        })

        # ── Audit: record decision ──────────────────────────────────────────
        decision_status = "approved" if decision == "Approve" else ("rejected" if decision == "Reject" else "aborted")
        if pool is not None:
            await record_decision(
                pool=pool,
                idempotency_key=idempotency_key,
                decision=str(decision),
                status=decision_status,
            )

        # ── Guard: only accept canonical decision values ─────────────────────
        write_success = False
        result: str

        if decision not in ("Approve", "Reject"):
            result = (
                f"Invalid decision '{decision}'. "
                "Must be 'Approve' or 'Reject'. Action aborted."
            )
            try:
                from src.telemetry_metrics import record_write_attempt
                record_write_attempt(action_type=action_type, status="aborted")
            except Exception:
                pass
        elif decision == "Approve":
            try:
                response = await _execute_update(action_type, approval_data, idempotency_key)
                if response.get("isSuccess"):
                    result = f"Success: {response.get('message')}"
                    write_success = True
                    outcome_status = "executed"
                else:
                    result = (
                        f"Failed: {response.get('message')} "
                        f"(HTTP {response.get('statusCode')})"
                    )
                    outcome_status = "failed"
                if pool is not None:
                    await record_outcome(
                        pool=pool,
                        idempotency_key=idempotency_key,
                        result=response,
                        status=outcome_status,
                        action_type=action_type,
                    )
            except Exception as e:
                result = f"Failed to execute Salesforce update: {str(e)}"
                if pool is not None:
                    await record_outcome(
                        pool=pool,
                        idempotency_key=idempotency_key,
                        result={"error": str(e)},
                        status="failed",
                        action_type=action_type,
                    )
        else:  # Reject
            result = f"Action aborted: User rejected — '{record_name}'."
            try:
                from src.telemetry_metrics import record_write_attempt
                record_write_attempt(action_type=action_type, status="rejected")
            except Exception:
                pass

        # ── Replace approval-request msg with real outcome ──────────────────
        updated_msg = ToolMessage(
            content=result,
            tool_call_id=approval_msg.tool_call_id,
            id=approval_msg.id,
        )

        update: dict = {"messages": [updated_msg]}
        if write_success:
            update["write_count"] = state.get("write_count", 0) + 1

        return update

    return human_approval_node
