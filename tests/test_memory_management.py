"""
Tests for Conversation Memory Management (bounded working context for LLM invocation).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graph.graph import safe_trim_messages, build_rag_graph, SYSTEM_PROMPT
from src.config import settings


def test_safe_trim_under_limit_untouched():
    """When messages are within limit, history is returned untouched with SystemMessage."""
    msgs = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I help you?"),
        HumanMessage(content="What is my booking status?"),
    ]
    trimmed = safe_trim_messages(msgs, max_messages=10, system_prompt="System Prompt")
    
    assert isinstance(trimmed[0], SystemMessage)
    assert trimmed[0].content == "System Prompt"
    # All 3 user/AI messages preserved verbatim
    assert trimmed[1:] == msgs
    assert len(trimmed) == 4


def test_safe_trim_exceeding_limit():
    """When messages exceed limit, older turns are trimmed and window starts on HumanMessage."""
    msgs = []
    for i in range(10):
        msgs.append(HumanMessage(content=f"Question {i}"))
        msgs.append(AIMessage(content=f"Answer {i}"))

    # 20 total messages. Limit to 6.
    trimmed = safe_trim_messages(msgs, max_messages=6, system_prompt="System Prompt")

    assert isinstance(trimmed[0], SystemMessage)
    assert trimmed[0].content == "System Prompt"
    # An archive note is injected when messages are trimmed without an explicit summary
    assert isinstance(trimmed[1], SystemMessage)
    assert "Earlier conversation turns have been archived" in trimmed[1].content

    # The actual conversation messages in working context
    conv_msgs = trimmed[2:]
    assert len(conv_msgs) <= 6
    # Must start on HumanMessage (never orphan AIMessage)
    assert isinstance(conv_msgs[0], HumanMessage)
    # The latest human question and answer must be included
    assert conv_msgs[-1].content == "Answer 9"
    assert conv_msgs[-2].content == "Question 9"


def test_safe_trim_unlimited_when_zero_or_negative():
    """When max_messages <= 0, trimming is disabled and full history is preserved."""
    msgs = [HumanMessage(content=f"Q{i}") for i in range(30)]
    
    trimmed_zero = safe_trim_messages(msgs, max_messages=0)
    assert len(trimmed_zero) == 31  # SystemMessage + 30 questions

    trimmed_neg = safe_trim_messages(msgs, max_messages=-1)
    assert len(trimmed_neg) == 31


def test_active_turn_never_truncated():
    """Active multi-step tool turn is NEVER truncated even if max_messages is smaller than turn."""
    # Historical turn
    msgs = [
        HumanMessage(content="Old question"),
        AIMessage(content="Old answer"),
    ]
    # Active turn with tool call and result
    active_turn = [
        HumanMessage(content="Update booking BK-1"),
        AIMessage(content="", tool_calls=[{"name": "get_booking", "args": {"booking_number": "BK-1"}, "id": "call_1"}]),
        ToolMessage(content='{"status": "Pending"}', tool_call_id="call_1"),
    ]
    all_msgs = msgs + active_turn

    # Set max_messages = 2 (smaller than active turn which is 3 messages)
    trimmed = safe_trim_messages(all_msgs, max_messages=2, system_prompt="System Prompt")

    # System prompt + archive note + active turn preserved in full
    working_msgs = [m for m in trimmed if not isinstance(m, SystemMessage)]
    assert len(working_msgs) == 3
    assert isinstance(working_msgs[0], HumanMessage)
    assert working_msgs[0].content == "Update booking BK-1"
    assert isinstance(working_msgs[1], AIMessage)
    assert isinstance(working_msgs[2], ToolMessage)


def test_summary_injection_when_present():
    """When state contains a summary, it is injected as a SystemMessage after the main prompt."""
    msgs = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi!"),
    ]
    trimmed = safe_trim_messages(
        msgs,
        max_messages=5,
        system_prompt="Base prompt",
        summary="User is asking about vacation packages to Japan.",
    )

    assert isinstance(trimmed[0], SystemMessage)
    assert trimmed[0].content == "Base prompt"
    assert isinstance(trimmed[1], SystemMessage)
    assert "Summary of prior conversation:" in trimmed[1].content
    assert "vacation packages to Japan" in trimmed[1].content
    assert trimmed[2:] == msgs


@pytest.mark.asyncio
async def test_graph_execution_memory_trimming_and_checkpoint_integrity():
    """
    End-to-end graph test:
    - LLM invocation receives trimmed working context.
    - Checkpoint retains complete untrimmed message history across turns.
    """
    mock_llm = MagicMock()
    # Configure mock LLM to record invoke_messages passed to it
    captured_invocations = []

    async def mock_ainvoke(messages, *args, **kwargs):
        captured_invocations.append(list(messages))
        return AIMessage(content="Mock assistant reply")

    mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)

    checkpointer = MemorySaver()
    graph = build_rag_graph(
        llm=mock_llm,
        tools=[],
        checkpointer=checkpointer,
    )

    thread_config = {
        "configurable": {
            "thread_id": "test-memory-thread-1",
            "max_messages": 4,  # Window of 4 messages
        }
    }

    # Turn 1: 1 human msg -> 1 AI msg (total 2)
    await graph.ainvoke({"messages": [HumanMessage(content="Turn 1")]}, config=thread_config)
    # Turn 2: 1 human msg -> 1 AI msg (total 4)
    await graph.ainvoke({"messages": [HumanMessage(content="Turn 2")]}, config=thread_config)
    # Turn 3: 1 human msg -> 1 AI msg (total 6)
    await graph.ainvoke({"messages": [HumanMessage(content="Turn 3")]}, config=thread_config)

    # 1. Check LLM working context on Turn 3
    turn_3_invoke_msgs = captured_invocations[-1]
    working_non_system = [m for m in turn_3_invoke_msgs if not isinstance(m, SystemMessage)]
    # Working context passed to LLM must be trimmed to <= 4 messages
    assert len(working_non_system) <= 4
    # Working context must start on HumanMessage
    assert isinstance(working_non_system[0], HumanMessage)
    # Working context must contain latest prompt
    assert any(m.content == "Turn 3" for m in working_non_system)

    # 2. Check Checkpointer State
    # Checkpoint must preserve ALL 6 messages (3 human + 3 AI), not truncated!
    final_state = await graph.aget_state(thread_config)
    saved_msgs = final_state.values.get("messages", [])
    assert len(saved_msgs) == 6
    assert [m.content for m in saved_msgs] == [
        "Turn 1", "Mock assistant reply",
        "Turn 2", "Mock assistant reply",
        "Turn 3", "Mock assistant reply",
    ]


@pytest.mark.asyncio
async def test_config_override_max_messages():
    """Verify max_messages can be dynamically overridden via RunnableConfig."""
    mock_llm = MagicMock()
    captured = []

    async def mock_ainvoke(messages, *args, **kwargs):
        captured.append(list(messages))
        return AIMessage(content="Reply")

    mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)

    graph = build_rag_graph(llm=mock_llm, tools=[])

    # 8 historical messages
    initial_messages = []
    for i in range(4):
        initial_messages.append(HumanMessage(content=f"Q{i}"))
        initial_messages.append(AIMessage(content=f"A{i}"))

    # Invoke with override max_messages = 2
    await graph.ainvoke(
        {"messages": initial_messages + [HumanMessage(content="New question")]},
        config={"configurable": {"max_messages": 2}},
    )

    working_msgs = [m for m in captured[0] if not isinstance(m, SystemMessage)]
    assert len(working_msgs) <= 2
    assert working_msgs[-1].content == "New question"
