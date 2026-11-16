import hashlib
import json
import pytest
from unittest.mock import MagicMock, patch
from pydantic import SecretStr

from src.config import Settings
from src.llm.credentials import CredentialRef, MissingCredentialError, resolve_credential
from src.startup_validation import validate_runtime_config
from src.api.dependencies import _build_llm, get_graph_for_tenant


def test_credential_ref_repr_and_fingerprint_masks_secrets():
    """Verify that CredentialRef never leaks secret keys in __repr__ or __str__."""
    cred = CredentialRef(
        tenant_id="acme",
        provider="google",
        model="gemini-2.0-flash",
        api_key=SecretStr("AIzaSyVerySecretKey123"),
        source="tenant",
        fallback_provider="openai",
        fallback_api_key=SecretStr("sk-SecretOpenAIKey456"),
    )
    repr_str = repr(cred)
    str_str = str(cred)

    assert "AIzaSyVerySecretKey123" not in repr_str
    assert "sk-SecretOpenAIKey456" not in repr_str
    assert "AIzaSyVerySecretKey123" not in str_str
    assert "sk-SecretOpenAIKey456" not in str_str
    assert "tenant=acme" in repr_str
    assert "source=tenant" in repr_str
    assert len(cred.fingerprint) == 16


def test_resolve_credential_tenant_source(monkeypatch):
    """Verify that a tenant in api_key_mapping with llm_api_key resolves to source='tenant'."""
    raw_token = "token-acme-secret"
    key_id = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()[:16]

    test_settings = Settings(
        deployment_mode="hosted_byok",
        api_key_mapping={
            raw_token: {
                "username": "acme_user",
                "tenant_id": "acme",
                "llm_provider": "openai",
                "llm_model": "gpt-4o",
                "llm_api_key": "sk-acme-key-999",
                "llm_fallback_provider": "google",
                "llm_fallback_api_key": "AIza-acme-backup",
            }
        },
    )
    monkeypatch.setattr("src.llm.credentials.settings", test_settings)

    cred = resolve_credential(tenant_id="acme", api_key_id=key_id)
    assert cred.source == "tenant"
    assert cred.tenant_id == "acme"
    assert cred.provider == "openai"
    assert cred.model == "gpt-4o"
    assert cred.api_key.get_secret_value() == "sk-acme-key-999"
    assert cred.fallback_provider == "google"
    assert cred.fallback_api_key.get_secret_value() == "AIza-acme-backup"


def test_zero_credit_leakage_fallback_suppression(monkeypatch):
    """
    CRITICAL REGRESSION TEST:
    A tenant without their own fallback key MUST have fallback_provider=None,
    guaranteeing they never silently bill the operator's OpenAI key.
    """
    raw_token = "token-acme-solo"
    key_id = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()[:16]

    test_settings = Settings(
        deployment_mode="hosted_byok",
        openai_api_key=SecretStr("sk-operator-openai-paid-key"),
        api_key_mapping={
            raw_token: {
                "username": "acme_user",
                "tenant_id": "acme",
                "llm_provider": "google",
                "llm_api_key": "AIza-acme-primary-only",
            }
        },
    )
    monkeypatch.setattr("src.llm.credentials.settings", test_settings)

    cred = resolve_credential(tenant_id="acme", api_key_id=key_id)
    assert cred.source == "tenant"
    assert cred.fallback_provider is None
    assert cred.fallback_api_key is None

    # Verify _build_llm builds LLM without fallback
    with patch("src.api.dependencies.create_llm") as mock_create_llm:
        _build_llm(cred)
        mock_create_llm.assert_called_once()
        _, kwargs = mock_create_llm.call_args
        assert kwargs["fallback_provider"] is None
        assert kwargs["fallback_api_key"] is None


def test_solo_mode_preserves_operator_fallback(monkeypatch):
    """
    Verify that in solo mode (developer / operator workflow),
    operator Gemini -> OpenAI fallback is 100% preserved.
    """
    test_settings = Settings(
        deployment_mode="solo",
        allow_operator_fallback=True,
        google_api_key=SecretStr("AIza-operator-gemini"),
        openai_api_key=SecretStr("sk-operator-openai"),
    )
    monkeypatch.setattr("src.llm.credentials.settings", test_settings)

    cred = resolve_credential("default", None)
    assert cred.source == "operator"
    assert cred.provider == "google"
    assert cred.fallback_provider == "openai"
    assert cred.fallback_api_key.get_secret_value() == "sk-operator-openai"


def test_solo_mode_direct_openai_primary(monkeypatch):
    """Verify that setting llm_provider='openai' switches primary without 429 latency."""
    test_settings = Settings(
        deployment_mode="solo",
        llm_provider="openai",
        openai_api_key=SecretStr("sk-operator-openai"),
    )
    monkeypatch.setattr("src.llm.credentials.settings", test_settings)

    cred = resolve_credential("default", None)
    assert cred.source == "operator"
    assert cred.provider == "openai"
    assert cred.fallback_provider is None  # no self-fallback to openai


def test_hosted_byok_rejects_missing_tenant_key(monkeypatch):
    """Verify that in hosted_byok, a tenant without key raises MissingCredentialError immediately."""
    test_settings = Settings(
        deployment_mode="hosted_byok",
        api_key_mapping={},
    )
    monkeypatch.setattr("src.llm.credentials.settings", test_settings)

    with pytest.raises(MissingCredentialError) as exc_info:
        resolve_credential("unknown_tenant", None)
    assert "BYOK mode requires every tenant to supply their own key" in str(exc_info.value)


def test_startup_validation_hosted_byok_fatal_rules():
    """Verify startup validation fatal rules for hosted_byok."""
    # 1. Fatal if global keys present in production
    settings_with_global = Settings(
        environment="production",
        dev_mode=False,
        deployment_mode="hosted_byok",
        google_api_key=SecretStr("AIza-operator-key"),
        api_key_mapping={
            "token1": {"tenant_id": "t1", "llm_api_key": "AIza-tenant-1"}
        },
    )
    fatals, _ = validate_runtime_config(settings_with_global)
    assert any("Refusing to start: a shared provider key is configured" in f for f in fatals)

    # 2. Fatal if tenant in mapping lacks llm_api_key
    settings_missing_key = Settings(
        environment="development",
        dev_mode=True,
        deployment_mode="hosted_byok",
        google_api_key=None,
        openai_api_key=None,
        api_key_mapping={
            "token1": {"tenant_id": "acme", "username": "a@acme.com"}  # Missing llm_api_key
        },
    )
    fatals, _ = validate_runtime_config(settings_missing_key)
    assert any("has no 'llm_api_key' in API_KEY_MAPPING" in f for f in fatals)


def test_startup_validation_hosted_shared_fatal_rules():
    """Verify startup validation fatal rules for hosted_shared."""
    # 1. Fatal if no global key
    settings_no_key = Settings(
        deployment_mode="hosted_shared",
        google_api_key=None,
        openai_api_key=None,
        deepseek_api_key=None,
        openrouter_api_key=None,
    )
    fatals, _ = validate_runtime_config(settings_no_key)
    assert any("requires at least one global LLM key" in f for f in fatals)

    # 2. Fatal if no quota set
    settings_no_quota = Settings(
        deployment_mode="hosted_shared",
        google_api_key=SecretStr("AIza-key"),
        tenant_quota_daily_tokens=None,
        tenant_quota_monthly_tokens=None,
        postgres_url="postgresql://localhost/db",
    )
    fatals, _ = validate_runtime_config(settings_no_quota)
    assert any("requires a token quota" in f for f in fatals)


def test_distinct_graph_instances_per_tenant(monkeypatch):
    """Verify that two distinct tenants receive distinct compiled graph instances in cache."""
    test_settings = Settings(
        deployment_mode="hosted_byok",
        api_key_mapping={
            "token_a": {"tenant_id": "tenant_a", "llm_api_key": "key_a"},
            "token_b": {"tenant_id": "tenant_b", "llm_api_key": "key_b"},
        },
    )
    monkeypatch.setattr("src.api.dependencies.settings", test_settings)
    monkeypatch.setattr("src.llm.credentials.settings", test_settings)

    with patch("src.api.dependencies.build_graph_with_client") as mock_build:
        g1 = MagicMock(name="graph_tenant_a")
        g2 = MagicMock(name="graph_tenant_b")
        mock_build.side_effect = [g1, g2]

        graph_a = get_graph_for_tenant("tenant_a", "google", None, "fp_a", "token_a")
        graph_b = get_graph_for_tenant("tenant_b", "google", None, "fp_b", "token_b")

        assert graph_a is not graph_b
        assert mock_build.call_count == 2
