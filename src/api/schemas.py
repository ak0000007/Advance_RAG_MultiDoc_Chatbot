"""
Request / response schemas for the RAG API.

ISP: each schema is small and purpose-built.
Extend by adding new models here — no existing code changes.
"""

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message."""

    question: str = Field(
        ...,
        min_length=1,
        description="User question to answer.",
    )

    history: list[Any] = Field(
        default_factory=list,
        description=(
            "Conversation history as list of "
            "LangChain message dicts (legacy stateless mode)."
        ),
    )

    thread_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{8,64}$",
        description=(
            "Unique session ID for database memory. Must match ^[A-Za-z0-9_-]{8,64}$. "
            "If provided, history field is ignored."
        ),
    )

    sf_username: str | None = Field(
        default=None,
        description="DEPRECATED: Salesforce username. Ignored unless DEV_MODE=true.",
    )

    trace_id: str | None = Field(
        default=None,
        description="Optional distributed trace identifier for end-to-end request correlation.",
    )


class Citation(BaseModel):
    """
    Source document citation for grounded RAG answers.
    Identifies the exact chunk, document title, and page.
    """

    chunk_id: str = Field(..., description="Unique chunk identifier or deterministic UUID.")
    source: str = Field(..., description="Source path, document URL, or file name.")
    title: str | None = Field(default=None, description="Human-readable document title or file name.")
    page: int | None = Field(default=None, description="Document page number if applicable.")
    snippet: str | None = Field(default=None, description="Short snippet of retrieved document content.")


class StageUsage(BaseModel):
    """Token usage and cost for a specific graph execution stage/node."""

    node: str = Field(..., description="LangGraph node or execution stage name.")
    model: str = Field(..., description="Underlying model name invoked for this stage.")
    provider: str = Field(default="", description="LLM provider name.")
    input_tokens: int = Field(default=0, description="Input/prompt token count.")
    output_tokens: int = Field(default=0, description="Output/completion token count.")
    cost_usd: float = Field(default=0.0, description="Computed stage cost in USD.")
    estimated: bool = Field(default=False, description="True if tokens or pricing were approximated.")


class Usage(BaseModel):
    """Multi-stage token accounting and cost summary for a conversation turn."""

    input_tokens: int = Field(default=0, description="Total input tokens across all stages.")
    output_tokens: int = Field(default=0, description="Total output tokens across all stages.")
    total_tokens: int = Field(default=0, description="Total tokens consumed.")
    cost_usd: float = Field(default=0.0, description="Total estimated cost in USD.")
    estimated: bool = Field(default=False, description="True if any stage fell back to estimation.")
    by_stage: list[StageUsage] = Field(default_factory=list, description="Per-stage breakdown.")


class ChatResponse(BaseModel):
    """
    Outgoing chat response.

    On normal completion: answer is populated, interrupted=False.
    On human-approval pause: interrupted=True, thread_id and approval_request are set.
    The frontend must display the approval UI and POST to /chat/resume.
    """

    answer: str = Field(default="", description="Agent's final answer (empty when interrupted).")
    interrupted: bool = Field(default=False, description="True when the graph is paused for human approval.")
    thread_id: str | None = Field(default=None, description="Thread to resume (only set when interrupted=True).")
    approval_request: dict | None = Field(
        default=None,
        description=(
            "Approval payload to show the user. Contains: action, message, "
            "opportunity_id, new_status, options=['Approve','Reject']."
        ),
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Source document citations retrieved and used for the response.",
    )
    citation_verification: dict[str, Any] | None = Field(
        default=None,
        description="Automated inline citation verification report (total, verified, unverified counts).",
    )
    trace_id: str | None = Field(
        default=None,
        description="Distributed trace ID for observability correlation across LangSmith / Langfuse / OpenTelemetry.",
    )
    usage: Usage | None = Field(
        default=None,
        description="Multi-stage token accounting and cost report for this turn.",
    )


class ResumeRequest(BaseModel):
    """Resume an interrupted graph with a human decision."""

    thread_id: str = Field(
        ...,
        pattern=r"^[A-Za-z0-9_-]{8,64}$",
        description="The thread_id returned in the interrupted ChatResponse. Must match ^[A-Za-z0-9_-]{8,64}$.",
    )
    decision: Literal["Approve", "Reject"] = Field(
        ...,
        description="Human's approval decision. Must be exactly 'Approve' or 'Reject'.",
    )
    trace_id: str | None = Field(
        default=None,
        description="Optional distributed trace identifier for end-to-end request correlation.",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"


class ThreadSummary(BaseModel):
    """Queryable summary of a conversation thread."""

    thread_id: str = Field(..., description="Unique thread identifier.")
    tenant_id: str = Field(..., description="Tenant owning this thread.")
    owner_user_id: str | None = Field(default=None, description="User owning this thread.")
    title: str = Field(..., description="Thread title generated from first user query.")
    last_activity: datetime = Field(..., description="Timestamp of most recent activity.")
    input_tokens: int = Field(default=0, description="Cumulative input tokens spent.")
    output_tokens: int = Field(default=0, description="Cumulative output tokens spent.")
    cost_usd: float = Field(default=0.0, description="Cumulative cost in USD.")
    message_count: int = Field(default=0, description="Total turns/messages in thread.")
    interrupted: bool = Field(default=False, description="Whether the thread is currently paused waiting for human approval.")


class ThreadMessage(BaseModel):
    """A clean, human-readable user or assistant message turn."""

    role: Literal["user", "assistant"] = Field(..., description="Message role.")
    content: str = Field(..., description="Clean message text content.")
    timestamp: datetime | None = Field(default=None, description="Optional timestamp.")


class ThreadDetail(BaseModel):
    """Full scrubbed history and execution state of a thread."""

    thread_id: str = Field(..., description="Unique thread identifier.")
    tenant_id: str = Field(..., description="Tenant owning this thread.")
    title: str = Field(..., description="Thread title.")
    messages: list[ThreadMessage] = Field(default_factory=list, description="Scrubbed dialogue history.")
    interrupted: bool = Field(default=False, description="True if thread is paused waiting for approval.")
    approval_request: dict | None = Field(default=None, description="Pending approval request payload if interrupted.")
    input_tokens: int = Field(default=0, description="Cumulative input tokens.")
    output_tokens: int = Field(default=0, description="Cumulative output tokens.")
    cost_usd: float = Field(default=0.0, description="Cumulative cost in USD.")
    message_count: int = Field(default=0, description="Total turns in thread.")
    last_activity: datetime = Field(..., description="Timestamp of most recent activity.")
