"""
Async Salesforce Client for JWT Bearer Flow and Apex API calls.
Follows SOLID principles: injected with configuration, independent of global state.
"""

import time
import jwt
import httpx
from pathlib import Path
from typing import Any, Optional

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

    def _load_private_key(self) -> str:
        if not self.private_key_path.exists():
            raise FileNotFoundError(f"Salesforce private key not found at {self.private_key_path}")
        with open(self.private_key_path, "r") as f:
            return f.read()

    async def get_access_token(self, username: str) -> str:
        """
        Exchange JWT for a Salesforce Access Token scoped to the user.
        """
        payload = {
            "iss": self.client_id,
            "aud": self.login_url,
            "sub": username,
            "exp": int(time.time()) + 300
        }
        
        encoded_jwt = jwt.encode(payload, self._private_key, algorithm="RS256")
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.login_url}/services/oauth2/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion": encoded_jwt
                }
            )
            resp.raise_for_status()
            return resp.json()["access_token"]

    async def get_opportunities(self, username: str, search: Optional[str] = None) -> dict[str, Any]:
        """
        Call the custom Apex REST API: /services/apexrest/AgentOpportunities
        """
        token = await self.get_access_token(username)
        
        url = f"{self.domain}/services/apexrest/AgentOpportunities"
        params = {"search": search} if search else {}
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                params=params
            )
            resp.raise_for_status()
            return resp.json()
