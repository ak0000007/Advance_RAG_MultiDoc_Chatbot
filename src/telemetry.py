"""
Telemetry bootstrap.

Call setup_langsmith() once at each entry point (app.py, ingest.py, scripts)
BEFORE any LangChain import. LangSmith reads os.environ at import time, so
the order is: load_dotenv → setup_langsmith → import langchain.

SRP: this module owns the single responsibility of activating tracing.
OCP: toggle via LANGSMITH_TRACING env var — no code change needed.
"""

import os


def setup_langsmith() -> None:
    """
    Propagate LangSmith env vars into the process environment.

    The SDK recognises both LANGCHAIN_* (v1 legacy) and LANGSMITH_* names.
    We set both to cover older langchain-core versions.

    No-op when LANGSMITH_TRACING is absent or falsy.
    """
    tracing = os.getenv("LANGSMITH_TRACING", "").lower()
    if tracing not in ("true", "1", "yes"):
        return

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    api_key = os.getenv("LANGSMITH_API_KEY")
    if api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)

    project = os.getenv("LANGSMITH_PROJECT")
    if project:
        os.environ.setdefault("LANGCHAIN_PROJECT", project)

    endpoint = os.getenv("LANGSMITH_ENDPOINT")
    if endpoint:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)
