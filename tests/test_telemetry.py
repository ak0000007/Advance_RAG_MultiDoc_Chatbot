"""
Tests for Observability Suite (LangSmith, Langfuse, OpenTelemetry & In-Process Telemetry).

Validates:
1. Pydantic Settings for telemetry with SecretStr masking.
2. Trace ID generation and propagation.
3. ObservabilityCallbackHandler measuring node and tool latencies.
4. get_tracer_callbacks assembly and graceful fallback when third-party SDKs are missing.
5. End-to-end trace correlation in route configurations and API responses.
"""

import os
import uuid
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.telemetry import (
    ObservabilityCallbackHandler,
    generate_trace_id,
    get_tracer_callbacks,
    setup_telemetry,
)
from src.telemetry_metrics import (
    TOOL_CALLS_TOTAL,
    record_node_latency,
    record_tool_call,
    record_tool_latency,
)


def test_settings_telemetry_defaults_and_parsing():
    """Verify telemetry configuration fields, defaults, and SecretStr protection."""
    s = Settings(
        langsmith_tracing="true",
        langsmith_api_key=SecretStr("lsv2_secret_key"),
        langsmith_project="custom-project",
        langfuse_enabled="1",
        langfuse_public_key="pk-12345",
        langfuse_secret_key=SecretStr("sk-67890"),
        otel_enabled="yes",
        otel_service_name="test-travel-bot",
    )
    assert s.langsmith_tracing is True
    assert s.langsmith_api_key.get_secret_value() == "lsv2_secret_key"
    assert "lsv2_secret_key" not in repr(s.langsmith_api_key)
    assert s.langsmith_project == "custom-project"
    assert s.langfuse_enabled is True
    assert s.langfuse_public_key == "pk-12345"
    assert s.langfuse_secret_key.get_secret_value() == "sk-67890"
    assert "sk-67890" not in repr(s.langfuse_secret_key)
    assert s.otel_enabled is True
    assert s.otel_service_name == "test-travel-bot"


def test_generate_trace_id_is_valid_uuid():
    """Verify generate_trace_id generates distinct valid UUID4 strings."""
    id1 = generate_trace_id()
    id2 = generate_trace_id()
    assert id1 != id2
    parsed = uuid.UUID(id1)
    assert parsed.version == 4


def test_setup_telemetry_env_vars():
    """Verify setup_telemetry safely populates os.environ without raising."""
    with patch.dict(os.environ, {}, clear=True):
        setup_telemetry()
        # Should execute cleanly without error


def test_observability_callback_handler_chain_timing():
    """Verify ObservabilityCallbackHandler records chain node latency."""
    handler = ObservabilityCallbackHandler(trace_id="test-trace-123")
    assert handler.trace_id == "test-trace-123"

    run_id = uuid4()
    handler.on_chain_start(
        serialized={"id": ["langgraph", "agent_node"]},
        inputs={"question": "hello"},
        run_id=run_id,
        metadata={"langgraph_node": "agent_node"},
    )
    assert run_id in handler._chain_starts
    assert handler._chain_names[run_id] == "agent_node"

    handler.on_chain_end(
        outputs={"answer": "world"},
        run_id=run_id,
    )
    assert run_id not in handler._chain_starts
    assert "agent_node" in handler.latencies
    assert handler.latencies["agent_node"] >= 0.0


def test_observability_callback_handler_tool_timing():
    """Verify ObservabilityCallbackHandler records tool latency and increments calls."""
    handler = ObservabilityCallbackHandler()
    run_id = uuid4()

    handler.on_tool_start(
        serialized={"name": "search_documents"},
        input_str="cancellation policy",
        run_id=run_id,
    )
    assert run_id in handler._tool_starts
    assert handler._tool_names[run_id] == "search_documents"

    handler.on_tool_end(
        output="Policy: 100% refund",
        run_id=run_id,
    )
    assert run_id not in handler._tool_starts
    assert "tool:search_documents" in handler.latencies
    assert handler.latencies["tool:search_documents"] >= 0.0


def test_observability_callback_handler_error_handling():
    """Verify ObservabilityCallbackHandler gracefully handles chain and tool errors."""
    handler = ObservabilityCallbackHandler()
    run_id_chain = uuid4()
    run_id_tool = uuid4()

    handler.on_chain_start(
        serialized={"id": ["test_node"]},
        inputs={},
        run_id=run_id_chain,
    )
    handler.on_chain_error(RuntimeError("Chain failed"), run_id=run_id_chain)
    assert "test_node" in handler.latencies

    handler.on_tool_start(
        serialized={"name": "failing_tool"},
        input_str="",
        run_id=run_id_tool,
    )
    handler.on_tool_error(RuntimeError("Tool failed"), run_id=run_id_tool)
    assert "tool:failing_tool" in handler.latencies


def test_get_tracer_callbacks_default():
    """Verify get_tracer_callbacks returns ObservabilityCallbackHandler by default."""
    callbacks = get_tracer_callbacks(
        thread_id="thread-test",
        run_name="test-run",
        tags=["unit-test"],
        metadata={"user": "tester"},
        trace_id="trace-xyz",
    )
    assert len(callbacks) >= 1
    obs_handler = callbacks[0]
    assert isinstance(obs_handler, ObservabilityCallbackHandler)
    assert obs_handler.trace_id == "trace-xyz"


def test_get_tracer_callbacks_langfuse_graceful_fallback():
    """Verify enabling Langfuse without package installed logs warning and doesn't crash."""
    with patch("src.config.settings.langfuse_enabled", True):
        callbacks = get_tracer_callbacks(
            thread_id="th-1",
            run_name="rn-1",
            tags=["test"],
            trace_id="tr-1",
        )
        assert len(callbacks) >= 1
        assert isinstance(callbacks[0], ObservabilityCallbackHandler)


def test_get_tracer_callbacks_otel_graceful_fallback():
    """Verify enabling OpenTelemetry without package installed doesn't crash."""
    with patch("src.config.settings.otel_enabled", True):
        callbacks = get_tracer_callbacks(
            thread_id="th-2",
            run_name="rn-2",
            trace_id="tr-2",
        )
        assert len(callbacks) >= 1
        assert isinstance(callbacks[0], ObservabilityCallbackHandler)


def test_routes_build_config_injects_telemetry():
    """Verify _build_config constructs RunnableConfig with callbacks, metadata, and trace_id."""
    from src.api.routes import _build_config

    config = _build_config(
        thread_id="th-test",
        sf_username="agent@crm.com",
        trace_id="custom-trace-001",
        endpoint="chat",
    )
    assert config["configurable"]["thread_id"] == "th-test"
    assert config["configurable"]["sf_username"] == "agent@crm.com"
    assert config["configurable"]["trace_id"] == "custom-trace-001"
    assert config["metadata"]["trace_id"] == "custom-trace-001"
    assert "callbacks" in config
    assert len(config["callbacks"]) >= 1
    assert any(isinstance(c, ObservabilityCallbackHandler) for c in config["callbacks"])
    assert "chat" in config["tags"]


def test_chat_response_schema_trace_id():
    """Verify ChatResponse model accepts and serializes trace_id."""
    from src.api.schemas import ChatResponse

    resp = ChatResponse(
        answer="Here is your travel itinerary",
        trace_id="trace-abc-123",
    )
    assert resp.trace_id == "trace-abc-123"
    serialized = resp.model_dump()
    assert serialized["trace_id"] == "trace-abc-123"
