"""
Unit tests for multi-agent intent orchestrator and priority fallback routing.
Uses mock LLMs and mock subgraphs (no real external API calls).
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import RAGState
from src.graph.orchestrator import (
    IntentClassification,
    build_orchestrator_graph,
    resolve_multi_priority,
)


from langchain_core.runnables import Runnable


class MockClassifierLLM(Runnable):
    """Mock LLM returning deterministic structured IntentClassification results."""

    def __init__(self, domain_to_return: str):
        super().__init__()
        self.domain_to_return = domain_to_return

    def with_structured_output(self, schema):
        return self

    def invoke(self, input_val, config=None, **kwargs):
        return IntentClassification(
            domain=self.domain_to_return,
            reason=f"Mock classification for {self.domain_to_return}",
        )

    async def ainvoke(self, input_val, config=None, **kwargs):
        return IntentClassification(
            domain=self.domain_to_return,
            reason=f"Mock classification for {self.domain_to_return}",
        )


def _make_mock_specialist(name: str):
    """Create a minimal compiled StateGraph representing a specialist agent."""
    builder = StateGraph(RAGState)

    def agent_node(state: RAGState):
        return {
            "answer": f"Handled by {name}",
            "messages": [AIMessage(content=f"Handled by {name}")],
        }

    builder.add_node("agent", agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile()


@pytest.fixture
def mock_subgraphs():
    return {
        "crm": _make_mock_specialist("crm_agent"),
        "travel": _make_mock_specialist("travel_agent"),
        "docs": _make_mock_specialist("docs_agent"),
    }


@pytest.mark.asyncio
async def test_crm_only_routes_to_crm(mock_subgraphs):
    """Verify clear CRM message routes exclusively to crm_agent."""
    mock_llm = MockClassifierLLM("crm")
    graph = build_orchestrator_graph(
        crm_graph=mock_subgraphs["crm"],
        travel_graph=mock_subgraphs["travel"],
        docs_graph=mock_subgraphs["docs"],
        llm=mock_llm,
        checkpointer=MemorySaver(),
    )

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Please update opportunity 0065g000001AAAAAA1 stage to Closed Won")],
        },
        config={"configurable": {"thread_id": "test_crm"}},
    )

    assert result["classified_domain"] == "crm"
    assert result["answer"] == "Handled by crm_agent"


@pytest.mark.asyncio
async def test_travel_only_routes_to_travel(mock_subgraphs):
    """Verify clear travel message routes exclusively to travel_agent."""
    mock_llm = MockClassifierLLM("travel")
    graph = build_orchestrator_graph(
        crm_graph=mock_subgraphs["crm"],
        travel_graph=mock_subgraphs["travel"],
        docs_graph=mock_subgraphs["docs"],
        llm=mock_llm,
        checkpointer=MemorySaver(),
    )

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Show all my bookings and transactions for this year")],
        },
        config={"configurable": {"thread_id": "test_travel"}},
    )

    assert result["classified_domain"] == "travel"
    assert result["answer"] == "Handled by travel_agent"


@pytest.mark.asyncio
async def test_docs_only_routes_to_docs(mock_subgraphs):
    """Verify clear knowledge base query routes exclusively to docs_agent."""
    mock_llm = MockClassifierLLM("docs")
    graph = build_orchestrator_graph(
        crm_graph=mock_subgraphs["crm"],
        travel_graph=mock_subgraphs["travel"],
        docs_graph=mock_subgraphs["docs"],
        llm=mock_llm,
        checkpointer=MemorySaver(),
    )

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="What is our corporate policy for international travel insurance coverage?")],
        },
        config={"configurable": {"thread_id": "test_docs"}},
    )

    assert result["classified_domain"] == "docs"
    assert result["answer"] == "Handled by docs_agent"


@pytest.mark.asyncio
async def test_multi_booking_and_policy_routes_to_higher_priority_travel(mock_subgraphs):
    """
    Verify message mentioning both a booking and a policy question classifies as 'multi'
    and falls back to higher-priority travel_agent (priority order: crm > travel > docs).
    """
    mock_llm = MockClassifierLLM("multi")
    graph = build_orchestrator_graph(
        crm_graph=mock_subgraphs["crm"],
        travel_graph=mock_subgraphs["travel"],
        docs_graph=mock_subgraphs["docs"],
        llm=mock_llm,
        checkpointer=MemorySaver(),
    )

    # Contains both booking (travel) and cancellation policy (docs)
    user_msg = "Please update my booking BK-000001 travelers to 3 and what is the cancellation policy?"
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=user_msg)],
        },
        config={"configurable": {"thread_id": "test_multi_travel_docs"}},
    )

    # travel > docs priority order
    assert result["classified_domain"] == "travel"
    assert result["answer"] == "Handled by travel_agent"


@pytest.mark.asyncio
async def test_multi_crm_and_travel_routes_to_crm_priority(mock_subgraphs):
    """
    Verify message mentioning both an opportunity and a booking classifies as 'multi'
    and falls back to higher-priority crm_agent (priority order: crm > travel > docs).
    """
    mock_llm = MockClassifierLLM("multi")
    graph = build_orchestrator_graph(
        crm_graph=mock_subgraphs["crm"],
        travel_graph=mock_subgraphs["travel"],
        docs_graph=mock_subgraphs["docs"],
        llm=mock_llm,
        checkpointer=MemorySaver(),
    )

    user_msg = "Update opportunity Acme to Closed Won and check my booking status."
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=user_msg)],
        },
        config={"configurable": {"thread_id": "test_multi_crm_travel"}},
    )

    # crm > travel priority order
    assert result["classified_domain"] == "crm"
    assert result["answer"] == "Handled by crm_agent"


def test_resolve_multi_priority_rules():
    """Unit tests for multi-domain priority resolution function."""
    # Travel + Docs -> Travel
    assert resolve_multi_priority("booking status and insurance policy") == "travel"
    # CRM + Travel -> CRM
    assert resolve_multi_priority("deal stage Closed Won and booking BK-001") == "crm"
    # CRM + Docs -> CRM
    assert resolve_multi_priority("opportunity Acme and employee policy handbook") == "crm"
    # Docs only -> Docs
    assert resolve_multi_priority("company policy handbook reimbursement") == "docs"
    # Fallback default -> CRM
    assert resolve_multi_priority("hello system override random text") == "crm"
