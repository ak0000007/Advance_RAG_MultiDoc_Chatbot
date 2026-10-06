"""
Prometheus metrics for Advance_RAG_Chatbot.

Tracks tool usage and Salesforce write attempts without touching src/telemetry.py.
"""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

WRITE_ATTEMPTS_TOTAL = Counter(
    "write_attempts_total",
    "Total number of Salesforce write attempts by action type and status",
    ["action_type", "status"],
)

TOOL_CALLS_TOTAL = Counter(
    "tool_calls_total",
    "Total number of tool calls by tool name",
    ["tool_name"],
)


def record_write_attempt(action_type: str, status: str) -> None:
    """Increment write_attempts_total counter."""
    WRITE_ATTEMPTS_TOTAL.labels(action_type=action_type, status=status).inc()


def record_tool_call(tool_name: str) -> None:
    """Increment tool_calls_total counter."""
    TOOL_CALLS_TOTAL.labels(tool_name=tool_name).inc()
