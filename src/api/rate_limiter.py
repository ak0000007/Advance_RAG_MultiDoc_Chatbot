"""
Sliding-window request-rate limiter (in-process, asyncio-safe).

Design:
  - One deque per key (tenant:user). Entries are timestamps of recent requests.
  - acquire() pops expired entries, checks length, appends current timestamp.
  - asyncio.Lock per key prevents concurrent coroutines racing the same window.
  - No external dependencies (no Redis). Single-worker deployment assumption.

Configuration (via src/config.settings):
  rate_limit_enabled              — bool, default False (opt-in)
  rate_limit_requests_per_minute  — int,  default 60
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import HTTPException, Request


class SlidingWindowRateLimiter:
    """
    Per-key sliding-window rate limiter.

    Args:
        requests_per_minute: Max requests allowed in any 60-second window.
        enabled: When False, acquire() is a no-op (default: False).
    """

    def __init__(self, requests_per_minute: int = 60, enabled: bool = False) -> None:
        self._rpm = requests_per_minute
        self._enabled = enabled
        self._window = 60.0  # seconds
        # Shared state: key → deque of float timestamps
        self._windows: dict[str, deque[float]] = defaultdict(deque)
        # Per-key lock to prevent concurrent mutation
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def acquire(self, key: str) -> None:
        """
        Attempt to consume one request slot for `key`.

        Raises:
            HTTPException(429) with Retry-After + X-RateLimit-* headers when limit exceeded.
        """
        if not self._enabled:
            return

        async with self._locks[key]:
            now = time.monotonic()
            window_start = now - self._window
            q = self._windows[key]

            # Evict expired timestamps
            while q and q[0] <= window_start:
                q.popleft()

            if len(q) >= self._rpm:
                # Oldest entry tells us when a slot opens
                retry_after = int(q[0] - window_start) + 1
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please slow down.",
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(self._rpm),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(now + retry_after)),
                    },
                )

            q.append(now)
            remaining = self._rpm - len(q)
            # Attach informational headers via request state if available
            # (Callers may also read these after acquire returns)
            return remaining  # type: ignore[return-value]  # optional convenience

    @property
    def enabled(self) -> bool:
        return self._enabled

    def reset(self, key: Optional[str] = None) -> None:
        """Test helper — clear state for a key (or all keys if None)."""
        if key is None:
            self._windows.clear()
            self._locks.clear()
        else:
            self._windows.pop(key, None)
            self._locks.pop(key, None)


# Module-level singleton — reconfigured in app.py lifespan via configure()
_limiter: Optional[SlidingWindowRateLimiter] = None


def configure(requests_per_minute: int, enabled: bool) -> SlidingWindowRateLimiter:
    """Create (or replace) the module-level limiter singleton. Call once at startup."""
    global _limiter
    _limiter = SlidingWindowRateLimiter(requests_per_minute=requests_per_minute, enabled=enabled)
    return _limiter


def get_limiter() -> SlidingWindowRateLimiter:
    """Return the configured singleton (or a disabled no-op if not yet configured)."""
    if _limiter is None:
        return SlidingWindowRateLimiter(enabled=False)
    return _limiter
