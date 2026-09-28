"""
Request / response schemas for the RAG API.

ISP: each schema is small and purpose-built.
Extend by adding new models here — no existing code changes.
"""

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
        description=(
            "Unique session ID for database memory. "
            "If provided, history field is ignored."
        ),
    )

    sf_username: str | None = Field(
        default=None,
        description="Salesforce username for executing CRM queries.",
    )


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


class ResumeRequest(BaseModel):
    """Resume an interrupted graph with a human decision."""

    thread_id: str = Field(..., description="The thread_id returned in the interrupted ChatResponse.")
    decision: Literal["Approve", "Reject"] = Field(
        ...,
        description="Human's approval decision. Must be exactly 'Approve' or 'Reject'.",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
