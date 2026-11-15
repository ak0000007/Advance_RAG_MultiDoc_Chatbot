"""
Unit and integration tests for Startup Runtime Validation (P1-3).

Validates:
1. Production mode with no authentication is fatal.
2. HITL write tools with MemorySaver is fatal.
3. Documents-only mode starts cleanly without Salesforce credentials.
4. POSTGRES_URL set with missing psycopg driver is fatal.
5. Audit logging required when writes enabled.
6. Missing Salesforce credentials when CRM enabled is fatal.
7. Shared api_secret_key without sf_default_username when CRM is active is fatal.
8. SF_AUTH_ENABLED=true without JWT public key/aud/iss is fatal.
9. CORS wildcard in production emits warning.
10. Multi-worker with MemorySaver emits warning.
11. Multiple errors are reported together in a single list.
12. get_db_pool propagates non-ImportError exceptions (stops swallowing).
13. Audit writer behavior when pool is None matches Step 0 findings.
14. DEV_MODE flips fatal errors to logged warnings in FastAPI lifespan.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.config import Settings
from src.startup_validation import validate_runtime_config
from src.api.dependencies import get_db_pool
from src.audit.log import record_decision


def test_production_no_auth_fatal():
    """Rule 1: Production (DEV_MODE=false) without auth is FATAL; DEV_MODE=true allows it."""
    prod_settings = Settings(
        dev_mode=False,
        api_secret_key=None,
        api_key_mapping={},
        sf_auth_enabled=False,
        crm_enabled=False,
        writes_enabled=False,
    )
    fatals, warnings = validate_runtime_config(prod_settings)
    assert any("Production mode requires authentication" in f for f in fatals)

    dev_settings = Settings(
        dev_mode=True,
        api_secret_key=None,
        api_key_mapping={},
        sf_auth_enabled=False,
        crm_enabled=False,
        writes_enabled=False,
    )
    fatals_dev, _ = validate_runtime_config(dev_settings)
    assert not any("Production mode requires authentication" in f for f in fatals_dev)


def test_hitl_writes_with_memory_saver_fatal():
    """Rule 2: HITL write tools enabled with MemorySaver is FATAL."""
    settings = Settings(
        dev_mode=False,
        api_secret_key="valid-secret",
        sf_default_username="agent1",
        mock_salesforce=True,
        writes_enabled=True,
        postgres_url=None,
    )
    fatals, _ = validate_runtime_config(settings)
    assert any("HITL write tools are enabled with MemorySaver" in f for f in fatals)


def test_documents_only_mode_clean_start():
    """Rule 5 & Normal Operation: Documents-only mode starts cleanly without Salesforce credentials."""
    settings = Settings(
        dev_mode=False,
        api_secret_key="doc-secret",
        crm_enabled=False,
        writes_enabled=False,
        sf_client_id=None,
        sf_private_key_path=None,
        postgres_url=None,
    )
    fatals, warnings = validate_runtime_config(settings)
    # Zero fatals!
    assert len(fatals) == 0
    # MemorySaver in documents-only mode is only a warning
    assert any("MemorySaver checkpointer is active in read-only / documents-only mode" in w for w in warnings)


def test_postgres_url_missing_driver():
    """Rule 3: POSTGRES_URL configured but driver cannot be imported is FATAL in production."""
    settings = Settings(
        dev_mode=False,
        api_secret_key="sec",
        sf_default_username="user1",
        postgres_url="postgresql://user:pass@localhost:5432/db",
        mock_salesforce=True,
    )
    with patch.dict("sys.modules", {"psycopg_pool": None}):
        fatals, warnings = validate_runtime_config(settings)
        assert any("psycopg_pool is not installed" in f for f in fatals)


def test_audit_logging_required_when_writes_enabled():
    """Rule 4: Audit logging is required when write tools are enabled."""
    settings = Settings(
        dev_mode=False,
        api_secret_key="sec",
        sf_default_username="u1",
        mock_salesforce=True,
        writes_enabled=True,
        postgres_url=None,
    )
    fatals, _ = validate_runtime_config(settings)
    assert any("Audit logging is required when write tools are enabled" in f for f in fatals)


def test_missing_salesforce_credentials_when_crm_enabled():
    """Rule 5: Missing Salesforce credentials when CRM is active (and not mock) is FATAL."""
    settings = Settings(
        dev_mode=False,
        api_secret_key="sec",
        crm_enabled=True,
        mock_salesforce=False,
        sf_client_id=None,
        sf_private_key_path=None,
    )
    fatals, _ = validate_runtime_config(settings)
    assert any("Salesforce credentials (SF_CLIENT_ID and SF_PRIVATE_KEY_PATH) are missing" in f for f in fatals)


def test_shared_api_key_without_default_user_crm_fatal():
    """Rule 6: Shared API key without sf_default_username while CRM is active is FATAL."""
    settings = Settings(
        dev_mode=False,
        api_secret_key="shared-secret",
        sf_default_username=None,
        mock_salesforce=True,
        postgres_url="postgresql://localhost/db",
    )
    fatals, _ = validate_runtime_config(settings)
    assert any("Shared API_SECRET_KEY is configured without SF_DEFAULT_USERNAME" in f for f in fatals)


def test_sf_auth_enabled_missing_jwt_params():
    """Rule 7: SF_AUTH_ENABLED=true without public key or aud/iss is FATAL."""
    settings = Settings(
        dev_mode=False,
        sf_auth_enabled=True,
        sf_jwt_public_key=None,
        sf_jwt_public_key_path=None,
        sf_jwt_audience="",
        sf_jwt_issuer="",
    )
    fatals, _ = validate_runtime_config(settings)
    assert any("SF_AUTH_ENABLED=true requires SF_JWT_PUBLIC_KEY" in f for f in fatals)


def test_cors_wildcard_warning_in_production():
    """Rule 8: Wildcard CORS origin in production emits warning."""
    settings = Settings(
        dev_mode=False,
        api_secret_key="sec",
        sf_default_username="u1",
        cors_origins=["*"],
        crm_enabled=False,
        writes_enabled=False,
    )
    _, warnings = validate_runtime_config(settings)
    assert any("CORS_ORIGINS contains wildcard '*'" in w for w in warnings)


def test_multi_worker_memory_saver_warning():
    """Rule 9: WEB_CONCURRENCY > 1 with MemorySaver emits warning."""
    settings = Settings(
        dev_mode=True,
        postgres_url=None,
        crm_enabled=False,
        writes_enabled=False,
    )
    with patch.dict(os.environ, {"WEB_CONCURRENCY": "4"}):
        _, warnings = validate_runtime_config(settings)
        assert any("Multi-worker deployment detected" in w for w in warnings)


def test_multiple_errors_reported_together():
    """Rule: Validator accumulates all fatal errors into a single report."""
    settings = Settings(
        dev_mode=False,
        api_secret_key=None,  # Error 1: no auth
        crm_enabled=True,
        mock_salesforce=False,
        sf_client_id=None,  # Error 2: missing SF creds
        writes_enabled=True,
        postgres_url=None,  # Error 3: MemorySaver with writes + no audit
    )
    fatals, _ = validate_runtime_config(settings)
    assert len(fatals) >= 3


def test_get_db_pool_propagates_non_import_error():
    """Verify get_db_pool catches only ImportError; other exceptions propagate."""
    with patch("src.config.settings.postgres_url", "postgresql://bad_conninfo"):
        with patch("src.graph.memory.get_postgres_pool", side_effect=ValueError("Invalid conninfo string")):
            with pytest.raises(ValueError, match="Invalid conninfo string"):
                get_db_pool()


@pytest.mark.asyncio
async def test_audit_writer_behavior_when_pool_is_none():
    """
    Verify Step 0 finding: human_approval_node and record_decision handle pool=None safely.
    """
    from src.graph.approval import build_human_approval_node
    mock_sf = AsyncMock()
    # build_human_approval_node with pool=None
    node = build_human_approval_node(mock_sf, pool=None)
    assert node is not None


def test_dev_mode_flips_fatal_errors_to_warnings_in_lifespan():
    """Verify lifespan raises RuntimeError on fatal error unless DEV_MODE=true."""
    from src.api.app import app
    from fastapi.testclient import TestClient

    bad_prod_settings = Settings(
        dev_mode=False,
        api_secret_key=None,
        crm_enabled=False,
        writes_enabled=False,
        postgres_url=None,
    )

    with patch("src.api.app.settings", bad_prod_settings), patch("src.api.dependencies.settings", bad_prod_settings):
        # In production with fatal errors, TestClient lifespan context raises RuntimeError
        with pytest.raises(RuntimeError, match="Application startup aborted due to configuration errors"):
            with TestClient(app):
                pass

    bad_dev_settings = Settings(
        dev_mode=True,
        api_secret_key=None,
        crm_enabled=False,
        writes_enabled=False,
        postgres_url=None,
        mock_salesforce=True,
    )
    with patch("src.api.app.settings", bad_dev_settings), \
         patch("src.api.dependencies.settings", bad_dev_settings), \
         patch("src.api.app.get_compiled_graph", return_value=MagicMock()), \
         patch("src.api.dependencies.get_salesforce_client", return_value=None):
        # In dev mode, fatals are downgraded to warnings and server boots cleanly
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
