"""
LangGraph workflow construction.

Architecture (ReAct Agent with optional Human Approval):

START
  ↓
Agent Node ──(if tool needed)──> Tool Node
  ↑                                 │
  │                    ┌────────────┘
  │                    ▼
  │           [requires approval?]
  │                Yes │   No
  │                    ▼   └──────────────────┐
  │           Human Approval Node             │
  │           (interrupt / resume)            │
  │                    │                      │
  └────────────────────┴──────────────────────┘
  ↓ (if text response)
 END
"""

import json
from typing import Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.graph.state import RAGState


SYSTEM_PROMPT = """You are an intelligent enterprise assistant.
Your primary role is to assist users by querying the company's internal knowledge base and retrieving Salesforce data.

TOOL USAGE GUIDELINES:
1. Document Search: Use `search_documents` when the user asks general questions about company policies, internal documents, knowledge base articles, or factual information.
2. Salesforce Data: Use the Salesforce tools when the user asks about their opportunities, pipeline, deals, or CRM data.
3. Salesforce Update: Use `update_salesforce_opportunity_status` to change opportunity stages. The system will automatically pause and ask the user for approval before writing to Salesforce.

BEHAVIORAL GUIDELINES:
1. If the user asks for something completely outside of your capabilities (e.g., booking flights, writing arbitrary code, or answering questions unrelated to the business), politely decline.
2. Always base your answers on the context returned by the tools. If the tools return no relevant information, tell the user you don't know. Do not hallucinate facts.
3. When summarizing tool results, be clear and concise.
"""


def _after_tool_router(state: RAGState) -> Literal["human_approval", "agent"]:
    """
    Conditional edge after ToolNode.
    Scans the most recent ToolMessages for an approval-request payload.
    Routes to human_approval if found, otherwise directly to agent.
    Only checks tool messages from the latest round (stops at first non-tool msg).
    """
    for msg in reversed(state.get("messages", [])):
        if not hasattr(msg, "tool_call_id"):
            break  # Past the tool messages from this round
        try:
            data = json.loads(msg.content)
            if isinstance(data, dict) and data.get("__requires_approval__"):
                return "human_approval"
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    return "agent"


def build_rag_graph(
    llm,
    tools: list,
    checkpointer=None,
    human_approval_node=None,
    **kwargs,
):
    """
    Build and compile the ReAct Agent workflow.

    Args:
        llm: The language model (with tools will be bound internally).
        tools: List of LangChain tools to expose to the agent.
        checkpointer: LangGraph checkpointer (required for interrupt to work).
        human_approval_node: Optional pre-built approval node (from approval.py).
                             If None, approval routing is skipped entirely.
    """
    graph_builder = StateGraph(RAGState)

    # ── 1. Bind tools to LLM ────────────────────────────────────────────────
    llm_with_tools = llm.bind_tools(tools)

    # ── 2. Agent Node ────────────────────────────────────────────────────────
    async def agent_node(state: RAGState):
        messages = state.get("messages", [])

        # Safety: cap tool-call loops at 5 per user turn
        tool_call_count = 0
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "human":
                break
            if getattr(msg, "tool_calls", None):
                tool_call_count += 1

        if tool_call_count >= 5:
            fallback_msg = AIMessage(
                content=(
                    "I've reached the maximum number of attempts to find this information. "
                    "Please try rephrasing your request or checking the constraints."
                )
            )
            return {"messages": [fallback_msg], "answer": fallback_msg.content}

        invoke_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = await llm_with_tools.ainvoke(invoke_messages)

        update = {"messages": [response]}
        if not response.tool_calls and response.content:
            update["answer"] = str(response.content)

        return update

    # ── 3. Routing logic ─────────────────────────────────────────────────────
    def should_continue(state: RAGState) -> Literal["tools", "__end__"]:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        if last_message and getattr(last_message, "tool_calls", None):
            return "tools"
        return "__end__"

    # ── 4. Register Nodes ────────────────────────────────────────────────────
    graph_builder.add_node("agent", agent_node)

    # Custom tool_node: wraps ToolNode but injects write_count from state into
    # the RunnableConfig before invoking tools, so the write cap is enforced
    # using authoritative state data — not LLM-supplied arguments.
    #
    # CRITICAL: accept `config` as second param to receive the live RunnableConfig
    # from graph.astream() (carries sf_username, thread_id, etc.), then MERGE
    # write_count into it. Replacing config entirely would wipe sf_username and
    # cause every SF tool call to return "Missing sf_username in configuration".
    _base_tool_node = ToolNode(tools)

    async def tool_node(state: RAGState, config: RunnableConfig):
        write_count = state.get("write_count", 0)
        existing = config.get("configurable", {}) if config else {}
        merged_config = {
            **config,
            "configurable": {**existing, "write_count": write_count},
        }
        return await _base_tool_node.ainvoke(state, config=merged_config)

    graph_builder.add_node("tools", tool_node)

    # ── 5. Connect Edges ─────────────────────────────────────────────────────
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "__end__": END},
    )

    if human_approval_node is not None:
        # Approval path: tools → router → human_approval → agent
        graph_builder.add_node("human_approval", human_approval_node)
        graph_builder.add_conditional_edges(
            "tools",
            _after_tool_router,
            {"human_approval": "human_approval", "agent": "agent"},
        )
        graph_builder.add_edge("human_approval", "agent")
    else:
        # No SF configured: direct tools → agent
        graph_builder.add_edge("tools", "agent")

    return graph_builder.compile(checkpointer=checkpointer)

