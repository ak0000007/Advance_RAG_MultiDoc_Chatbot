"""
Prometheus metrics for Advance_RAG_Chatbot.

Tracks tool usage and Salesforce write attempts without touching src/telemetry.py.
"""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

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

GRAPH_NODE_LATENCY_SECONDS = Histogram(
    "graph_node_latency_seconds",
    "Latency of LangGraph node execution in seconds",
    ["node_name"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

TOOL_LATENCY_SECONDS = Histogram(
    "tool_execution_latency_seconds",
    "Latency of tool execution in seconds",
    ["tool_name"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


def record_write_attempt(action_type: str, status: str) -> None:
    """Increment write_attempts_total counter."""
    WRITE_ATTEMPTS_TOTAL.labels(action_type=action_type, status=status).inc()


def record_tool_call(tool_name: str) -> None:
    """Increment tool_calls_total counter."""
    TOOL_CALLS_TOTAL.labels(tool_name=tool_name).inc()


def record_node_latency(node_name: str, duration_seconds: float) -> None:
    """Record duration of a graph node in seconds."""
    GRAPH_NODE_LATENCY_SECONDS.labels(node_name=node_name).observe(duration_seconds)


def record_tool_latency(tool_name: str, duration_seconds: float) -> None:
    """Record duration of a tool call in seconds."""
    TOOL_LATENCY_SECONDS.labels(tool_name=tool_name).observe(duration_seconds)
