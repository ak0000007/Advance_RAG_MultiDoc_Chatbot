"""
Runtime configuration and startup validator for Advance_RAG_Chatbot.

Validates environment, persistence, authentication, and CRM requirements
before service initialization.
"""

import os
import logging
from typing import Any

from src.graph.memory import resolve_checkpointer_backend

logger = logging.getLogger(__name__)


def validate_runtime_config(settings: Any) -> tuple[list[str], list[str]]:
    """
    Pure, unit-testable validation function.
    Returns (fatal_errors, warnings).
    """
    fatals: list[str] = []
    warnings: list[str] = []

    is_prod = not getattr(settings, "dev_mode", False)
    crm_active = getattr(settings, "is_crm_active", False)
    writes_active = getattr(settings, "writes_enabled", True) and crm_active
    backend, reason = resolve_checkpointer_backend(getattr(settings, "postgres_url", None))

    # 1. Production authentication requirement
    api_key_mapping = getattr(settings, "api_key_mapping", {})
    api_secret_key = getattr(settings, "api_secret_key", None)
    sec_key_val = (
        api_secret_key.get_secret_value()
        if hasattr(api_secret_key, "get_secret_value")
        else api_secret_key
    )
    has_auth = bool(getattr(settings, "sf_auth_enabled", False) or api_key_mapping or sec_key_val)

    if is_prod and not has_auth:
        fatals.append("Production mode requires authentication: configure SF_AUTH_ENABLED, API_KEY_MAPPING, or API_SECRET_KEY.")

    # 2. Checkpointer persistence with write tools
    if writes_active and backend == "memory":
        fatals.append(
            f"HITL write tools are enabled with MemorySaver checkpointer ({reason}). "
            "PostgreSQL persistence (POSTGRES_URL) is required in production to avoid losing pending writes."
        )
    elif not writes_active and backend == "memory":
        warnings.append(
            f"MemorySaver checkpointer is active in read-only / documents-only mode ({reason}). "
            "Conversation memory will not persist across service restarts."
        )

    # 3. POSTGRES_URL configured but driver cannot be built
    if getattr(settings, "postgres_url", None):
        try:
            import psycopg_pool
        except ImportError:
            msg = "POSTGRES_URL is configured but psycopg_pool is not installed."
            if is_prod:
                fatals.append(msg)
            else:
                warnings.append(msg)

    # 4. Audit logging required for write tools
    if writes_active and not getattr(settings, "postgres_url", None):
        msg = "Audit logging is required when write tools are enabled, but POSTGRES_URL is not configured."
        if is_prod:
            fatals.append(msg)
        else:
            warnings.append(msg)

    # 5. Salesforce credentials check
    crm_explicit = getattr(settings, "crm_enabled", None)
    mock_sf = getattr(settings, "mock_salesforce", False)
    has_sf_creds = bool(getattr(settings, "sf_client_id", None) and getattr(settings, "sf_private_key_path", None))

    if crm_explicit is True or (crm_active and not mock_sf):
        if not has_sf_creds and not mock_sf:
            fatals.append(
                "Salesforce credentials (SF_CLIENT_ID and SF_PRIVATE_KEY_PATH) are missing while CRM is enabled. "
                "Configure credentials, enable MOCK_SALESFORCE=true, or set CRM_ENABLED=false for documents-only mode."
            )

    # 6. Shared api_secret_key set without sf_default_username while CRM tools enabled
    if sec_key_val and not api_key_mapping and not getattr(settings, "sf_auth_enabled", False):
        if crm_active and not getattr(settings, "sf_default_username", None):
            fatals.append(
                "Shared API_SECRET_KEY is configured without SF_DEFAULT_USERNAME while CRM tools are enabled. "
                "Specify SF_DEFAULT_USERNAME or use API_KEY_MAPPING."
            )

    # 7. SF_AUTH_ENABLED=true without public key / audience / issuer
    if getattr(settings, "sf_auth_enabled", False):
        pub_key = getattr(settings, "sf_jwt_public_key", None)
        pub_key_val = (
            pub_key.get_secret_value()
            if hasattr(pub_key, "get_secret_value")
            else str(pub_key or "")
        )
        pub_key_path = getattr(settings, "sf_jwt_public_key_path", None)
        has_key = bool(pub_key_val.strip() or (pub_key_path and os.path.exists(pub_key_path)))
        has_aud = bool(getattr(settings, "sf_jwt_audience", None))
        has_iss = bool(getattr(settings, "sf_jwt_issuer", None))
        if not (has_key and has_aud and has_iss):
            fatals.append("SF_AUTH_ENABLED=true requires SF_JWT_PUBLIC_KEY (or path), SF_JWT_AUDIENCE, and SF_JWT_ISSUER.")

    # 8. CORS origins containing wildcard in production
    cors_origins = getattr(settings, "cors_origins", [])
    if is_prod and "*" in cors_origins:
        warnings.append("CORS_ORIGINS contains wildcard '*' in production.")

    # 9. Multi-worker deployment with MemorySaver
    workers_env = os.environ.get("WEB_CONCURRENCY", "1")
    try:
        workers = int(workers_env)
    except ValueError:
        workers = 1
    if workers > 1 and backend == "memory":
        warnings.append(
            f"Multi-worker deployment detected (WEB_CONCURRENCY={workers}) with MemorySaver. "
            "Session states and pending approvals will not be shared across worker processes."
        )

    # 10. Deployment topology and credential isolation (BYOK)
    dep_mode = getattr(settings, "deployment_mode", "solo")
    global_keys_present = getattr(settings, "global_llm_keys_present", False)

    if dep_mode == "hosted_byok":
        if is_prod and global_keys_present:
            fatals.append(
                "Refusing to start: a shared provider key is configured while DEPLOYMENT_MODE=hosted_byok. "
                "Remove GOOGLE_API_KEY/OPENAI_API_KEY so tenants cannot consume operator credit."
            )
        elif global_keys_present:
            warnings.append(
                "DEPLOYMENT_MODE=hosted_byok with global keys present: global keys will be ignored for tenant resolution."
            )

        if not api_key_mapping:
            fatals.append("DEPLOYMENT_MODE=hosted_byok requires API_KEY_MAPPING with per-tenant 'llm_api_key'.")
        else:
            for k, val in api_key_mapping.items():
                if isinstance(val, dict):
                    if not val.get("llm_api_key"):
                        fatals.append(
                            f"Tenant '{val.get('tenant_id', k)}' has no 'llm_api_key' in API_KEY_MAPPING. "
                            "BYOK mode requires every tenant to supply their own key."
                        )
                else:
                    fatals.append(
                        f"Tenant '{k}' has legacy string mapping in API_KEY_MAPPING. "
                        "BYOK mode requires object mapping with 'llm_api_key'."
                    )

    elif dep_mode == "hosted_shared":
        if not global_keys_present:
            fatals.append("DEPLOYMENT_MODE=hosted_shared requires at least one global LLM key (GOOGLE_API_KEY, OPENAI_API_KEY, etc.).")
        quota_daily = getattr(settings, "tenant_quota_daily_tokens", None)
        quota_monthly = getattr(settings, "tenant_quota_monthly_tokens", None)
        if quota_daily is None and quota_monthly is None:
            fatals.append("DEPLOYMENT_MODE=hosted_shared requires a token quota (TENANT_QUOTA_DAILY_TOKENS or TENANT_QUOTA_MONTHLY_TOKENS).")
        if not getattr(settings, "postgres_url", None):
            fatals.append("DEPLOYMENT_MODE=hosted_shared requires POSTGRES_URL for quota persistence.")

    elif dep_mode == "solo":
        if not global_keys_present:
            warnings.append("DEPLOYMENT_MODE=solo has no global LLM key configured. Requests will fail if no tenant key is provided.")

    # 11. Summary log
    auth_mode = (
        "salesforce_jwt"
        if getattr(settings, "sf_auth_enabled", False)
        else ("api_key_mapping" if api_key_mapping else ("api_secret_key" if sec_key_val else "open_unauthenticated"))
    )
    audit_status = "enabled" if (backend == "postgres") else "disabled"
    crm_desc = "active (mock)" if mock_sf else ("active" if crm_active else "disabled (docs_only)")

    logger.info(
        "Startup Configuration Summary: backend=%s, audit=%s, auth=%s, crm=%s, dev_mode=%s, env=%s, deployment_mode=%s",
        backend,
        audit_status,
        auth_mode,
        crm_desc,
        getattr(settings, "dev_mode", False),
        getattr(settings, "environment", "development"),
        dep_mode,
    )

    return fatals, warnings
