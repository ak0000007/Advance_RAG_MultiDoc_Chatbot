"""
Tests for multi-stage backend token accounting, pricing calculation,
database persistence, and quota enforcement.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult, Generation, ChatGeneration

from src.llm.pricing import cost_usd, get_model_pricing
from src.llm.usage import (
    StageUsage,
    TurnUsage,
    TokenUsageCallbackHandler,
    collect_from_events,
    record_turn_usage,
    check_tenant_quota,
    _extract_tokens_from_llm_result,
)
from src.api.schemas import Usage, StageUsage as SchemaStageUsage, ChatResponse


def test_pricing_calculation_known_models():
    # gemini-3.6-flash: (0.30, 2.50) per 1M tokens
    cost, estimated = cost_usd("gemini-3.6-flash", 1_000_000, 1_000_000)
    assert not estimated
    assert cost == pytest.approx(2.80, abs=1e-5)

    # Partial / fractional tokens
    # 10,000 in, 2,000 out on gpt-4o-mini: 0.15 * 0.01 + 0.60 * 0.002 = 0.0015 + 0.0012 = 0.0027
    cost, estimated = cost_usd("gpt-4o-mini", 10_000, 2_000)
    assert not estimated
    assert cost == pytest.approx(0.0027, abs=1e-5)


def test_pricing_calculation_unknown_model_fallback():
    cost, estimated = cost_usd("custom-travel-llm-v1", 5_000, 1_000)
    assert estimated is True
    assert cost == 0.0


def test_turn_usage_multi_stage_aggregation():
    turn = TurnUsage()

    # Stage 1: Intent classifier (small prompt)
    turn.add_stage(
        StageUsage(
            node="classify_intent",
            model="gemini-3.6-flash",
            provider="google",
            input_tokens=150,
            output_tokens=10,
            cost_usd=0.00007,
            estimated=False,
        )
    )

    # Stage 2: Chunk Grader 1
    turn.add_stage(
        StageUsage(
            node="grade_chunk_1",
            model="gemini-3.6-flash",
            provider="google",
            input_tokens=300,
            output_tokens=15,
            cost_usd=0.00012,
            estimated=False,
        )
    )

    # Stage 3: Chunk Grader 2
    turn.add_stage(
        StageUsage(
            node="grade_chunk_2",
            model="gemini-3.6-flash",
            provider="google",
            input_tokens=280,
            output_tokens=15,
            cost_usd=0.00012,
            estimated=False,
        )
    )

    # Stage 4: Synthesis
    turn.add_stage(
        StageUsage(
            node="agent",
            model="gemini-3.6-flash",
            provider="google",
            input_tokens=800,
            output_tokens=200,
            cost_usd=0.00074,
            estimated=False,
        )
    )

    assert turn.input_tokens == 150 + 300 + 280 + 800  # 1530
    assert turn.output_tokens == 10 + 15 + 15 + 200  # 240
    assert turn.total_tokens == 1530 + 240
    assert turn.total_tokens > 1000  # Multi-stage sum is strictly > single stage
    assert len(turn.by_stage) == 4
    assert turn.estimated is False

    d = turn.to_dict()
    assert d["total_tokens"] == 1770
    assert len(d["by_stage"]) == 4


def test_extract_tokens_from_usage_metadata():
    msg = AIMessage(
        content="Test reply",
        usage_metadata={"input_tokens": 120, "output_tokens": 45, "total_tokens": 165},
    )
    gen = ChatGeneration(text="Test reply", message=msg)
    llm_result = LLMResult(generations=[[gen]])

    inp, out, est = _extract_tokens_from_llm_result(llm_result)
    assert inp == 120
    assert out == 45
    assert est is False


def test_extract_tokens_fallback_heuristic_when_missing():
    msg = AIMessage(content="A" * 80)  # 80 characters -> ~20 tokens
    gen = ChatGeneration(text="A" * 80, message=msg)
    llm_result = LLMResult(generations=[[gen]])

    inp, out, est = _extract_tokens_from_llm_result(
        llm_result,
        default_input_chars=160,
        default_output_chars=80,
    )
    assert est is True
    assert inp == 40  # 160 // 4
    assert out == 20  # 80 // 4


@pytest.mark.asyncio
async def test_callback_handler_collects_stages():
    handler = TokenUsageCallbackHandler(default_provider="google", default_model="gemini-3.6-flash")

    # Simulate classifier stage
    msg1 = AIMessage(
        content="travel",
        usage_metadata={"input_tokens": 100, "output_tokens": 5, "total_tokens": 105},
    )
    res1 = LLMResult(
        generations=[[ChatGeneration(text="travel", message=msg1)]],
        llm_output={"model_name": "gemini-3.6-flash"},
    )
    await handler.on_chat_model_end(res1, run_id="run-1", metadata={"langgraph_node": "classify_intent"})

    # Simulate synthesis stage
    msg2 = AIMessage(
        content="Booking confirmed.",
        usage_metadata={"input_tokens": 500, "output_tokens": 80, "total_tokens": 580},
    )
    res2 = LLMResult(
        generations=[[ChatGeneration(text="Booking confirmed.", message=msg2)]],
        llm_output={"model_name": "gemini-3.6-flash"},
    )
    await handler.on_chat_model_end(res2, run_id="run-2", metadata={"langgraph_node": "agent"})

    turn_usage = handler.get_turn_usage()
    assert turn_usage.total_tokens == 685
    assert len(turn_usage.by_stage) == 2
    assert turn_usage.by_stage[0].node == "classify_intent"
    assert turn_usage.by_stage[1].node == "agent"


def test_collect_from_events():
    events = [
        {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "classify_intent"},
            "data": {
                "output": AIMessage(
                    content="docs",
                    usage_metadata={"input_tokens": 90, "output_tokens": 8, "total_tokens": 98},
                    response_metadata={"model_name": "gemini-3.6-flash"},
                )
            },
        },
        {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {
                "output": AIMessage(
                    content="Here is the policy.",
                    usage_metadata={"input_tokens": 400, "output_tokens": 50, "total_tokens": 450},
                    response_metadata={"model_name": "gemini-3.6-flash"},
                )
            },
        },
    ]

    usage = collect_from_events(events)
    assert usage.input_tokens == 490
    assert usage.output_tokens == 58
    assert len(usage.by_stage) == 2


@pytest.mark.asyncio
async def test_record_turn_usage_database_resilience():
    # Test that DB exception does NOT raise (resilience)
    failing_pool = MagicMock()
    failing_pool.connection.side_effect = RuntimeError("Postgres connection timeout")

    turn = TurnUsage()
    turn.add_stage(StageUsage(node="agent", model="gemini-3.6-flash", input_tokens=100, output_tokens=20))

    # Should not raise exception
    await record_turn_usage(failing_pool, "th-123", "tenant_a", "user@test.com", turn)


@pytest.mark.asyncio
async def test_record_turn_usage_success():
    mock_conn = AsyncMock()
    mock_pool = MagicMock()
    mock_pool.connection.return_value.__aenter__.return_value = mock_conn

    turn = TurnUsage()
    turn.add_stage(StageUsage(node="agent", model="gemini-3.6-flash", input_tokens=100, output_tokens=20))

    await record_turn_usage(mock_pool, "th-123", "tenant_a", "user@test.com", turn, trace_id="tr-456")
    assert mock_conn.execute.called
    args, kwargs = mock_conn.execute.call_args
    params = kwargs if kwargs else args[1]
    assert params["thread_id"] == "th-123"
    assert params["tenant_id"] == "tenant_a"
    assert params["input_tokens"] == 100
    assert params["output_tokens"] == 20


@pytest.mark.asyncio
async def test_check_tenant_quota_under_and_over():
    mock_conn = AsyncMock()
    mock_pool = MagicMock()
    mock_pool.connection.return_value.__aenter__.return_value = mock_conn

    # 1. Under quota: return 500 consumed tokens against 1000 limit
    mock_cursor = AsyncMock()
    mock_cursor.fetchone.return_value = (500,)
    mock_conn.execute.return_value = mock_cursor

    allowed, reason = await check_tenant_quota(
        mock_pool,
        tenant_id="acme",
        quota_daily=1000,
        quota_monthly=5000,
    )
    assert allowed is True
    assert reason is None

    # 2. Over daily quota: return 1500 consumed tokens against 1000 limit
    mock_cursor.fetchone.return_value = (1500,)
    allowed, reason = await check_tenant_quota(
        mock_pool,
        tenant_id="acme",
        quota_daily=1000,
        quota_monthly=None,
    )
    assert allowed is False
    assert "Daily token quota exceeded" in reason


def test_chat_response_schema_with_usage():
    stage = SchemaStageUsage(
        node="agent",
        model="gemini-3.6-flash",
        provider="google",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.000155,
        estimated=False,
    )
    usage = Usage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cost_usd=0.000155,
        estimated=False,
        by_stage=[stage],
    )
    resp = ChatResponse(answer="Success", usage=usage)
    assert resp.usage is not None
    assert resp.usage.total_tokens == 150
    assert len(resp.usage.by_stage) == 1
    assert resp.usage.by_stage[0].node == "agent"
