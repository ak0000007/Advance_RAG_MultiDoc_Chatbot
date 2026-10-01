"""
Async Salesforce Client for JWT Bearer Flow and Apex API calls.
Follows SOLID principles: injected with configuration, independent of global state.
"""

import time
import jwt
import httpx
from pathlib import Path
from typing import Any, Optional

# Default timeout for all Salesforce API calls.
# Prevents hung requests from blocking the graph node indefinitely.
_TIMEOUT = httpx.Timeout(10.0)

# Cache tokens for 4 minutes (JWT exp is 5 min). Leaves 60s safety margin.
_TOKEN_TTL = 240


class SalesforceAsyncClient:
    def __init__(
        self,
        client_id: str,
        login_url: str,
        private_key_path: str,
        domain: str,
    ):
        self.client_id = client_id
        self.login_url = login_url.rstrip("/")
        self.domain = domain.rstrip("/")
        self.private_key_path = Path(private_key_path)

        self._private_key = self._load_private_key()
        # {username: (token, expiry_timestamp)}
        self._token_cache: dict[str, tuple[str, float]] = {}

    def _load_private_key(self) -> str:
        if not self.private_key_path.exists():
            raise FileNotFoundError(f"Salesforce private key not found at {self.private_key_path}")
        with open(self.private_key_path, "r") as f:
            return f.read()

    async def get_access_token(self, username: str) -> str:
        """
        Exchange JWT for a Salesforce Access Token scoped to the user.
        Caches tokens in memory for _TOKEN_TTL seconds to avoid
        redundant network calls during multi-tool sequences.
        """
        now = time.time()
        cached = self._token_cache.get(username)
        if cached and cached[1] > now:
            return cached[0]

        payload = {
            "iss": self.client_id,
            "aud": self.login_url,
            "sub": username,
            "exp": int(now) + 300,
        }

        encoded_jwt = jwt.encode(payload, self._private_key, algorithm="RS256")

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{self.login_url}/services/oauth2/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion": encoded_jwt,
                },
            )
            resp.raise_for_status()
            token = resp.json()["access_token"]

        self._token_cache[username] = (token, now + _TOKEN_TTL)
        return token

    async def get_opportunities(self, username: str, search: Optional[str] = None) -> dict[str, Any]:
        """
        Call the custom Apex REST API: GET /services/apexrest/AgentOpportunities
        """
        token = await self.get_access_token(username)

        url = f"{self.domain}/services/apexrest/AgentOpportunities"
        params = {"search": search} if search else {}

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )
            resp.raise_for_status()
            return resp.json()

    async def update_opportunity_status(
        self, username: str, opportunity_id: str, new_status: str, idempotency_key: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Call the custom Apex REST API: PATCH /services/apexrest/Opportunity/StatusUpdate/
        Returns the structured JSON from OpportunityStatusAPI.cls.
        Raises on HTTP 5xx; passes 4xx back as structured errors for the caller to handle.
        """
        token = await self.get_access_token(username)

        url = f"{self.domain}/services/apexrest/Opportunity/StatusUpdate/"
        payload = {
            "opportunityId": opportunity_id,
            "newStatus": new_status,
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        
        if idempotency_key:
            # Standard header for idempotency (clicked-twice protection)
            # Used by downstream systems to deduplicate retry requests.
            headers["Idempotency-Key"] = idempotency_key

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.patch(
                url,
                headers=headers,
                json=payload,
            )
            # 5xx = server fault → raise so caller sees an exception
            # 4xx = business error → return structured JSON for the agent to report
            if resp.status_code >= 500:
                resp.raise_for_status()

            return resp.json()
