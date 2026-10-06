"""
Pytest configuration and custom hooks for Advance_RAG_Chatbot.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "routing: marks tests that execute the real-LLM routing evaluation suite (run with -m routing)",
    )


def pytest_collection_modifyitems(config, items):
    marker_expr = config.getoption("-m", default="")
    if "routing" in marker_expr:
        # User explicitly requested running routing tests via -m routing
        return

    skip_routing = pytest.mark.skip(
        reason="Real-LLM routing benchmark skipped by default. Run with: pytest -m routing"
    )
    for item in items:
        if "routing" in item.keywords:
            item.add_marker(skip_routing)
