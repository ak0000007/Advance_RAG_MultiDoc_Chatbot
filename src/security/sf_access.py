"""
Server-side Salesforce Record-Level Access Resolution Engine.

Queries UserRecordAccess for batch record permissions with:
- Concurrency limiting (asyncio.Semaphore)
- Bounded LRU in-memory caching with TTL
- Injectable clock function for deterministic time testing
- Fail-closed security architecture
"""

import asyncio
import logging
import time
from typing import Any, Callable, Optional, Set

logger = logging.getLogger(__name__)

# Salesforce UserRecordAccess SOQL IN-clause limit is 200 records
_SF_RECORD_ACCESS_BATCH_SIZE = 200


class AccessResolver:
    """
    Resolves readable Salesforce Account IDs for a given user.
    Caches results per (sf_org_id, user_id, account_id) with TTL.
    """

    def __init__(
        self,
        salesforce_client: Any,
        cache_ttl_seconds: int = 60,
        max_cache_size: int = 10000,
        time_fn: Callable[[], float] = time.time,
        batch_size: int = _SF_RECORD_ACCESS_BATCH_SIZE,
    ):
        self.sf_client = salesforce_client
        self.cache_ttl = cache_ttl_seconds
        self.max_cache_size = max_cache_size
        self.time_fn = time_fn
        self.batch_size = batch_size
        self._semaphore = asyncio.Semaphore(5)
        # Cache mapping: (sf_org_id, user_id, account_id) -> (has_read: bool, expires_at: float)
        self._cache: dict[tuple[str, str, str], tuple[bool, float]] = {}

    def _prune_cache(self, now: float) -> None:
        """Evict expired entries or trim cache if max_cache_size exceeded."""
        expired = [k for k, (_, exp) in self._cache.items() if exp <= now]
        for k in expired:
            self._cache.pop(k, None)

        if len(self._cache) > self.max_cache_size:
            # Evict oldest entries
            excess = len(self._cache) - self.max_cache_size
            oldest_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k][1])[:excess]
            for k in oldest_keys:
                self._cache.pop(k, None)

    async def readable_accounts(
        self,
        user_id: str,
        account_ids: Set[str] | list[str],
        org_id: str = "default",
        trace_id: Optional[str] = None,
    ) -> Set[str]:
        """
        Return the subset of account_ids for which user_id has read access.
        Fails closed on any Salesforce client exception or timeout.
        """
        if not user_id or not account_ids:
            return set()

        now = self.time_fn()
        self._prune_cache(now)

        clean_ids = {str(aid).strip() for aid in account_ids if str(aid).strip()}
        if not clean_ids:
            return set()

        readable: Set[str] = set()
        uncached_ids: list[str] = []

        # 1. Check local cache
        for aid in clean_ids:
            cache_key = (org_id, user_id, aid)
            cached = self._cache.get(cache_key)
            if cached is not None:
                has_read, exp = cached
                if exp > now:
                    if has_read:
                        readable.add(aid)
                    continue
                else:
                    self._cache.pop(cache_key, None)
            uncached_ids.append(aid)

        if not uncached_ids:
            return readable

        # 2. Query Salesforce in batches <= batch_size
        chunks = [
            uncached_ids[i : i + self.batch_size]
            for i in range(0, len(uncached_ids), self.batch_size)
        ]

        async def _query_batch(batch: list[str]) -> dict[str, bool]:
            async with self._semaphore:
                try:
                    if hasattr(self.sf_client, "check_user_record_access"):
                        return await self.sf_client.check_user_record_access(
                            user_id=user_id,
                            record_ids=batch,
                        )
                    # If client lacks method, fail closed
                    logger.error(
                        "Salesforce client does not implement check_user_record_access. Failing closed for trace_id=%s",
                        trace_id,
                    )
                    return {rid: False for rid in batch}
                except Exception as exc:
                    # Fail-closed: Never leak access, log at ERROR without PII
                    logger.error(
                        "Salesforce UserRecordAccess query failed for user (trace_id=%s): %s",
                        trace_id,
                        exc,
                        exc_info=False,
                    )
                    return {rid: False for rid in batch}

        batch_results = await asyncio.gather(*[_query_batch(c) for c in chunks])

        expires_at = now + self.cache_ttl
        for res in batch_results:
            for aid, has_read in res.items():
                self._cache[(org_id, user_id, aid)] = (has_read, expires_at)
                if has_read:
                    readable.add(aid)

        return readable
