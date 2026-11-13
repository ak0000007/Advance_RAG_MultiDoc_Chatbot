"""
Telemetry bootstrap and observability suite for Advance_RAG_Chatbot.

Supports:
1. LangSmith tracing via environment setup and process propagation.
2. Langfuse tracing via CallbackHandler (optional dependency with graceful fallback).
3. OpenTelemetry tracing via OTLP / OpenInference (optional dependency).
4. In-process ObservabilityCallbackHandler measuring LangGraph node & tool latencies
   and recording Prometheus metrics.

Call setup_telemetry() once at each entry point (app.py, ingest.py, scripts)
BEFORE any LangChain import.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from src.config import settings
from src.telemetry_metrics import (
    record_node_latency,
    record_tool_call,
    record_tool_latency,
)

logger = logging.getLogger(__name__)


def generate_trace_id() -> str:
    """Generate a unique trace identifier for distributed request tracing."""
    return str(uuid.uuid4())


def setup_telemetry() -> None:
    """
    Propagate telemetry env vars (LangSmith, Langfuse, OpenTelemetry) into the process environment.
    Uses settings from src.config.settings, falling back to os.environ if needed.
    """
    # 1. LangSmith
    langsmith_tracing = settings.langsmith_tracing
    if not langsmith_tracing:
        raw = os.getenv("LANGSMITH_TRACING", "").lower()
        langsmith_tracing = raw in ("true", "1", "yes", "on")

    if langsmith_tracing:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

        api_key = (
            settings.langsmith_api_key.get_secret_value()
            if settings.langsmith_api_key
            else os.getenv("LANGSMITH_API_KEY")
        )
        if api_key:
            os.environ.setdefault("LANGCHAIN_API_KEY", api_key)

        project = settings.langsmith_project or os.getenv("LANGSMITH_PROJECT")
        if project:
            os.environ.setdefault("LANGCHAIN_PROJECT", project)

        endpoint = settings.langsmith_endpoint or os.getenv("LANGSMITH_ENDPOINT")
        if endpoint:
            os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)

    # 2. Langfuse
    if settings.langfuse_enabled or os.getenv("LANGFUSE_ENABLED", "").lower() in ("true", "1", "yes"):
        public_key = settings.langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        if public_key:
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
        secret_key = (
            settings.langfuse_secret_key.get_secret_value()
            if settings.langfuse_secret_key
            else os.getenv("LANGFUSE_SECRET_KEY")
        )
        if secret_key:
            os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
        host = settings.langfuse_host or os.getenv("LANGFUSE_HOST")
        if host:
            os.environ.setdefault("LANGFUSE_HOST", host)

    # 3. OpenTelemetry
    if settings.otel_enabled or os.getenv("OTEL_ENABLED", "").lower() in ("true", "1", "yes"):
        endpoint = settings.otel_exporter_otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint)
        service_name = settings.otel_service_name or os.getenv("OTEL_SERVICE_NAME")
        if service_name:
            os.environ.setdefault("OTEL_SERVICE_NAME", service_name)


def setup_langsmith() -> None:
    """Backward-compatible alias for setup_telemetry()."""
    setup_telemetry()


class ObservabilityCallbackHandler(BaseCallbackHandler):
    """
    In-process observability callback handler for LangGraph and LangChain.
    Measures node latency and tool latency, emits Prometheus metrics,
    and captures span execution details.
    """

    def __init__(self, trace_id: Optional[str] = None):
        super().__init__()
        self.trace_id = trace_id or generate_trace_id()
        self._chain_starts: dict[Union[UUID, str], float] = {}
        self._chain_names: dict[Union[UUID, str], str] = {}
        self._tool_starts: dict[Union[UUID, str], float] = {}
        self._tool_names: dict[Union[UUID, str], str] = {}
        self.latencies: dict[str, float] = {}

    def on_chain_start(
        self,
        serialized: Optional[dict[str, Any]],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._chain_starts[run_id] = time.perf_counter()
        # Derive human-friendly node name
        node_name = "unknown_chain"
        if metadata and "langgraph_node" in metadata:
            node_name = str(metadata["langgraph_node"])
        elif serialized and "id" in serialized:
            node_name = str(serialized["id"][-1])
        elif tags:
            for t in tags:
                if t not in ("langgraph", "advance-rag"):
                    node_name = str(t)
                    break
        self._chain_names[run_id] = node_name

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        start_time = self._chain_starts.pop(run_id, None)
        node_name = self._chain_names.pop(run_id, "unknown_chain")
        if start_time is not None:
            elapsed = time.perf_counter() - start_time
            self.latencies[node_name] = elapsed
            record_node_latency(node_name, elapsed)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        start_time = self._chain_starts.pop(run_id, None)
        node_name = self._chain_names.pop(run_id, "unknown_chain")
        if start_time is not None:
            elapsed = time.perf_counter() - start_time
            self.latencies[node_name] = elapsed
            record_node_latency(node_name, elapsed)

    def on_tool_start(
        self,
        serialized: Optional[dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        tool_name = "unknown_tool"
        if serialized and "name" in serialized:
            tool_name = str(serialized["name"])
        self._tool_starts[run_id] = time.perf_counter()
        self._tool_names[run_id] = tool_name
        record_tool_call(tool_name)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        start_time = self._tool_starts.pop(run_id, None)
        tool_name = self._tool_names.pop(run_id, "unknown_tool")
        if start_time is not None:
            elapsed = time.perf_counter() - start_time
            self.latencies[f"tool:{tool_name}"] = elapsed
            record_tool_latency(tool_name, elapsed)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        start_time = self._tool_starts.pop(run_id, None)
        tool_name = self._tool_names.pop(run_id, "unknown_tool")
        if start_time is not None:
            elapsed = time.perf_counter() - start_time
            self.latencies[f"tool:{tool_name}"] = elapsed
            record_tool_latency(tool_name, elapsed)


def get_tracer_callbacks(
    thread_id: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> list[BaseCallbackHandler]:
    """
    Assemble and return active tracer callbacks for LangGraph execution.
    Combines:
    1. ObservabilityCallbackHandler (Prometheus metrics & in-memory latency)
    2. Langfuse CallbackHandler (if enabled & installed)
    3. OpenTelemetry Tracer (if enabled & installed)
    """
    callbacks: list[BaseCallbackHandler] = [
        ObservabilityCallbackHandler(trace_id=trace_id)
    ]

    # Langfuse Integration
    if settings.langfuse_enabled or os.getenv("LANGFUSE_ENABLED", "").lower() in ("true", "1", "yes"):
        try:
            from langfuse.callback import CallbackHandler as LangfuseCallbackHandler  # type: ignore

            public_key = settings.langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = (
                settings.langfuse_secret_key.get_secret_value()
                if settings.langfuse_secret_key
                else os.getenv("LANGFUSE_SECRET_KEY")
            )
            host = settings.langfuse_host or os.getenv("LANGFUSE_HOST")
            callbacks.append(
                LangfuseCallbackHandler(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                    session_id=thread_id,
                    trace_name=run_name,
                    tags=tags,
                    metadata=metadata,
                )
            )
        except ImportError:
            logger.warning(
                "Langfuse is enabled in settings, but 'langfuse' package is not installed. "
                "Install with `pip install langfuse` to enable Langfuse tracing."
            )
        except Exception as e:
            logger.warning("Failed to initialize Langfuse callback handler: %s", e)

    # OpenTelemetry Integration
    if settings.otel_enabled or os.getenv("OTEL_ENABLED", "").lower() in ("true", "1", "yes"):
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor  # type: ignore

            LangChainInstrumentor().instrument()
        except ImportError:
            logger.debug(
                "OpenTelemetry is enabled, but 'openinference-instrumentation-langchain' is not installed."
            )
        except Exception as e:
            logger.warning("Failed to initialize OpenTelemetry instrumentation: %s", e)

    return callbacks
