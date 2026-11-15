"""
Unit and integration tests for Rate Limiting & Cost Governance.

Test matrix:
1. test_rate_limiter_allows_under_limit: Requests within RPM window pass.
2. test_rate_limiter_blocks_over_limit_with_headers: 429 raised with RFC 6585 headers.
3. test_rate_limiter_recovers_after_window: Slots freed once window expires.
4. test_rate_limit_disabled_by_default: When disabled, burst requests pass through without error.
5. test_tenant_isolation_in_rate_limiting: Tenant A hitting limit does not throttle Tenant B.
6. test_streaming_token_budget_safeguard: SSE stream halts cleanly upon hitting token limit.
7. test_upstream_model_rate_limit_returns_429: Upstream ModelRateLimitError translates to HTTP 429.
"""

import json
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.rate_limiter import SlidingWindowRateLimiter, configure, get_limiter
from src.api.dependencies import get_active_graph
from src.config import settings
from src.llm.provider import ModelRateLimitError


# ── 1. Unit Tests for SlidingWindowRateLimiter ───────────────────────────────

@pytest.mark.asyncio
async def test_rate_limiter_allows_under_limit():
    """Requests under RPM limit succeed without exception."""
    limiter = SlidingWindowRateLimiter(requests_per_minute=5, enabled=True)
    key = "tenant-1:user-1"

    for _ in range(5):
        await limiter.acquire(key)

    # All 5 acquired successfully
    assert len(limiter._windows[key]) == 5


@pytest.mark.asyncio
async def test_rate_limiter_blocks_over_limit_with_headers():
    """Requests exceeding RPM limit raise HTTPException(429) with RFC 6585 headers."""
    limiter = SlidingWindowRateLimiter(requests_per_minute=2, enabled=True)
    key = "tenant-1:user-1"

    await limiter.acquire(key)
    await limiter.acquire(key)

    with pytest.raises(HTTPException) as exc_info:
        await limiter.acquire(key)

    exc = exc_info.value
    assert exc.status_code == 429
    assert "Rate limit exceeded" in exc.detail
    assert "Retry-After" in exc.headers
    assert exc.headers["X-RateLimit-Limit"] == "2"
    assert exc.headers["X-RateLimit-Remaining"] == "0"
    assert "X-RateLimit-Reset" in exc.headers


@pytest.mark.asyncio
async def test_rate_limiter_recovers_after_window():
    """Sliding window evicts timestamps older than 60s, admitting new requests."""
    limiter = SlidingWindowRateLimiter(requests_per_minute=1, enabled=True)
    key = "tenant-1:user-1"

    base_time = 1000.0

    with patch("time.monotonic", return_value=base_time):
        await limiter.acquire(key)

    # Immediate second request should be blocked
    with patch("time.monotonic", return_value=base_time + 10.0):
        with pytest.raises(HTTPException):
            await limiter.acquire(key)

    # Request after 61s window should succeed
    with patch("time.monotonic", return_value=base_time + 61.0):
        await limiter.acquire(key)


@pytest.mark.asyncio
async def test_rate_limit_disabled_by_default():
    """When enabled=False, large bursts of requests pass through without throttling."""
    limiter = SlidingWindowRateLimiter(requests_per_minute=2, enabled=False)
    key = "tenant-1:user-1"

    # Even 50 requests succeed when disabled
    for _ in range(50):
        await limiter.acquire(key)


@pytest.mark.asyncio
async def test_tenant_isolation_in_rate_limiting():
    """Tenant A hitting rate limit does not block Tenant B."""
    limiter = SlidingWindowRateLimiter(requests_per_minute=2, enabled=True)
    tenant_a = "tenant-alpha:user-1"
    tenant_b = "tenant-beta:user-1"

    # Exhaust tenant A's quota
    await limiter.acquire(tenant_a)
    await limiter.acquire(tenant_a)

    with pytest.raises(HTTPException):
        await limiter.acquire(tenant_a)

    # Tenant B should still have full quota
    await limiter.acquire(tenant_b)
    await limiter.acquire(tenant_b)

    with pytest.raises(HTTPException):
        await limiter.acquire(tenant_b)


# ── 2. Integration Tests: Streaming Token Safeguard ─────────────────────────

def _parse_sse_response(text: str) -> list[dict]:
    """Parse raw text/event-stream into a list of {event: ..., data: ...} dicts."""
    events = []
    blocks = text.strip().split("\n\n")
    for block in blocks:
        if not block.strip():
            continue
        event_type = "message"
        data_str = ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                data_str = line[6:].strip()
        if data_str:
            try:
                events.append({"event": event_type, "data": json.loads(data_str)})
            except Exception:
                events.append({"event": event_type, "data": data_str})
    return events


def test_streaming_token_budget_safeguard():
    """
    When streaming tokens reach max_streaming_tokens, the stream emits a status
    event indicating the token limit safeguard was reached and terminates cleanly.
    """
    class MockLoopingGraph:
        async def astream_events(self, inputs, config=None, version="v2"):
            # Emit 10 chunks from node="agent"
            for i in range(10):
                yield {
                    "event": "on_chat_model_stream",
                    "metadata": {"langgraph_node": "agent"},
                    "data": {
                        "chunk": MagicMock(
                            content=f"token_{i} ",
                            tool_call_chunks=[],
                        )
                    },
                }

        async def aget_state(self, config):
            snapshot = MagicMock()
            snapshot.values = {"answer": "Looping stream truncated"}
            snapshot.tasks = []
            snapshot.next = []
            return snapshot

    mock_graph = MockLoopingGraph()
    app.dependency_overrides[get_active_graph] = lambda: mock_graph

    try:
        # Patch max_streaming_tokens to 3
        with patch.object(settings, "max_streaming_tokens", 3):
            client = TestClient(app)
            response = client.post(
                "/chat/stream",
                json={"question": "Loop forever please"},
            )

            assert response.status_code == 200
            events = _parse_sse_response(response.text)

            token_events = [e for e in events if e.get("event") == "token"]
            status_events = [e for e in events if e.get("event") == "status"]

            # Exactly 3 tokens streamed before hitting cap
            assert len(token_events) == 3

            # Status event indicating token limit reached
            limit_events = [
                e for e in status_events
                if e.get("data", {}).get("stage") == "token_limit"
            ]
            assert len(limit_events) == 1
            assert "safeguard reached" in limit_events[0]["data"]["message"]

    finally:
        app.dependency_overrides.pop(get_active_graph, None)


# ── 3. Integration Tests: Upstream Model Rate Limit Error Translation ────────

def test_upstream_model_rate_limit_returns_429():
    """
    When the LLM provider exhausts quota or hits rate limits (raising ModelRateLimitError),
    the endpoint translates it to a clean HTTP 429 response with Retry-After header.
    """
    class MockFailingGraph:
        async def astream(self, inputs, config=None, stream_mode="values"):
            # Simulate upstream Google ResourceExhausted or ModelRateLimitError
            raise ModelRateLimitError("Resource has been exhausted (e.g. check quota).")
            yield  # Generator syntax

        async def aget_state(self, config):
            return MagicMock(values={}, tasks=[], next=[])

    mock_graph = MockFailingGraph()
    app.dependency_overrides[get_active_graph] = lambda: mock_graph

    try:
        client = TestClient(app)
        response = client.post(
            "/chat",
            json={"question": "Hit upstream rate limit"},
        )

        assert response.status_code == 429
        assert "Retry-After" in response.headers
        assert response.headers["Retry-After"] == "60"
        data = response.json()
        assert "rate limit or quota exceeded" in data["detail"]

    finally:
        app.dependency_overrides.pop(get_active_graph, None)
