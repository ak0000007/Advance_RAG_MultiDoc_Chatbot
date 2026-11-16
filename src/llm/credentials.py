from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Optional, Any
from pydantic import SecretStr
from src.config import settings


class MissingCredentialError(RuntimeError):
    """Raised when no usable LLM credential can be resolved for a tenant or request."""
    pass


@dataclass(frozen=True)
class CredentialRef:
    tenant_id: str
    provider: str
    model: str | None
    api_key: SecretStr | None
    source: Literal["tenant", "operator"]
    fallback_provider: str | None = None
    fallback_api_key: SecretStr | None = None

    @property
    def fingerprint(self) -> str:
        raw_key = self.api_key.get_secret_value() if self.api_key else ""
        raw_fb = self.fallback_api_key.get_secret_value() if self.fallback_api_key else ""
        key_material = (
            f"{self.tenant_id}|{self.provider}|{self.model}|{raw_key}|"
            f"{self.fallback_provider}|{raw_fb}|{self.source}"
        )
        return hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:16]

    def __repr__(self) -> str:
        # Never leak secret keys in string representations
        return (
            f"CredentialRef(tenant={self.tenant_id}, provider={self.provider}, "
            f"model={self.model}, source={self.source}, fp={self.fingerprint})"
        )

    def __str__(self) -> str:
        return self.__repr__()


def resolve_credential(tenant_id: str = "default", api_key_id: str | None = None) -> CredentialRef:
    """
    Resolve a tenant or API key ID to a usable LLM credential.
    Enforces strict isolation: tenants never silently fall back to operator credentials.
    """
    mapping = settings.api_key_mapping or {}
    tenant_info: dict[str, Any] | None = None

    # 1. Match by api_key_id (sha256 prefix of mapped key or direct match)
    if api_key_id:
        for mapped_key, info in mapping.items():
            key_hash = hashlib.sha256(mapped_key.encode("utf-8")).hexdigest()[:16]
            if key_hash == api_key_id or mapped_key == api_key_id:
                if isinstance(info, dict):
                    tenant_info = info
                break

    # 2. Match by tenant_id if not resolved by api_key_id
    if not tenant_info and mapping:
        for mapped_key, info in mapping.items():
            if isinstance(info, dict) and info.get("tenant_id") == tenant_id:
                tenant_info = info
                break

    # 3. If tenant provided their own key, build tenant-source credential
    if tenant_info and tenant_info.get("llm_api_key"):
        raw_key = str(tenant_info["llm_api_key"]).strip()
        prov = str(tenant_info.get("llm_provider") or "google").strip()
        mod = tenant_info.get("llm_model")
        fb_prov = tenant_info.get("llm_fallback_provider")
        fb_key_val = tenant_info.get("llm_fallback_api_key")
        fb_key = SecretStr(str(fb_key_val).strip()) if fb_key_val else None

        return CredentialRef(
            tenant_id=tenant_id,
            provider=prov,
            model=mod,
            api_key=SecretStr(raw_key),
            source="tenant",
            fallback_provider=str(fb_prov).strip() if (fb_prov and fb_key) else None,
            fallback_api_key=fb_key,
        )

    # 4. If hosted_byok is active, NEVER allow operator fallback
    if settings.deployment_mode == "hosted_byok":
        raise MissingCredentialError(
            f"Tenant '{tenant_id}' has no 'llm_api_key' configured in API_KEY_MAPPING. "
            "BYOK mode requires every tenant to supply their own key."
        )

    # 5. Check if operator fallback is allowed (solo or hosted_shared mode)
    if settings.operator_fallback_allowed:
        prov = settings.llm_provider or "google"
        mod = settings.llm_model
        op_key: SecretStr | None = None

        if prov == "google" and settings.google_api_key:
            op_key = settings.google_api_key
        elif prov == "openai" and settings.openai_api_key:
            op_key = settings.openai_api_key
        elif prov == "deepseek" and settings.deepseek_api_key:
            op_key = settings.deepseek_api_key
        elif prov == "openrouter" and settings.openrouter_api_key:
            op_key = settings.openrouter_api_key
        elif settings.google_api_key:
            op_key = settings.google_api_key
            prov = "google"
        elif settings.openai_api_key:
            op_key = settings.openai_api_key
            prov = "openai"

        if not op_key or not op_key.get_secret_value():
            raise MissingCredentialError(
                f"No LLM credential configured. In '{settings.deployment_mode}' mode, "
                "please set GOOGLE_API_KEY or OPENAI_API_KEY in .env."
            )

        fb_prov = "openai" if (settings.openai_api_key and prov != "openai") else None
        fb_key = settings.openai_api_key if fb_prov else None

        return CredentialRef(
            tenant_id=tenant_id,
            provider=prov,
            model=mod,
            api_key=op_key,
            source="operator",
            fallback_provider=fb_prov,
            fallback_api_key=fb_key,
        )

    raise MissingCredentialError(
        f"No LLM credential found for tenant '{tenant_id}' and operator fallback is disabled."
    )
