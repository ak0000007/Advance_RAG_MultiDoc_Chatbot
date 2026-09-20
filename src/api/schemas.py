"""
Request / response schemas for the RAG API.

ISP: each schema is small and purpose-built.
Extend by adding new models here — no existing code changes.
"""

from pydantic import BaseModel, Field
from typing import Any


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
            "LangChain message dicts."
        ),
    )


class ChatResponse(BaseModel):
    """Outgoing chat response."""

    answer: str
    answer_supported: bool | None = None
    answer_grade_reason: str | None = None
    retrieval_relevant: bool | None = None
    retrieval_grade_reason: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
