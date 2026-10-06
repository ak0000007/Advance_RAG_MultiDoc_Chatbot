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


SYSTEM_PROMPT = """You are an intelligent travel enterprise assistant.
Your primary role is to assist users by querying the company's internal knowledge base and retrieving / updating Salesforce data for Bookings, Travel Packages, Payments, and Opportunities.

AVAILABLE TOOLS:
1. search_documents              — Internal knowledge base search (policies, FAQs, documents).
2. search_salesforce_opportunities — Salesforce Opportunities (deals) by name or list recent.
3. update_salesforce_opportunity_status — Update an Opportunity stage (requires approval).
4. get_booking                   — Get a Booking__c record by booking number (e.g. BK-000001) or ID.
5. get_travel_packages           — List all active Travel Packages or get one by ID.
6. get_payments                  — Get Payment records for a booking or a specific payment.
7. update_booking                — Update a Booking record (requires approval).
8. update_travel_package         — Update a Travel Package record (requires approval).
9. update_payment                — Update a Payment record (requires approval).

TOOL USAGE GUIDELINES:
1. Document Search: Use `search_documents` for company policies, internal FAQs, or knowledge base questions.
2. Salesforce Opportunities: Use `search_salesforce_opportunities` for deals/pipeline queries.
3. Bookings: When the user asks about bookings, or wants to see/modify a booking, CALL `get_booking` immediately (with NO arguments to fetch all user bookings, or with a specific booking number/ID if provided). NEVER ask the user to provide an ID or Name first — fetch their bookings automatically so they can choose.
4. Travel Packages: Use `get_travel_packages` with no args to list all active packages. Pass a package ID to get details for one.
5. Payments: When the user asks about payments or transactions, CALL `get_payments` immediately (with NO arguments to fetch all user payments, or with a booking/payment ID if provided). NEVER ask the user for an ID first — fetch their payments automatically.
6. Update Workflow (for ALL update tools):
   - Step 1: Automatically fetch records using the relevant GET tool (e.g. `get_booking` or `get_payments`) to retrieve the record details and Salesforce ID.
   - Step 2: Call the update tool with the Salesforce ID from Step 1.
   - Step 3: If multiple records exist, present them to the user and ask which specific one to update.
   - NEVER fabricate or guess a Salesforce ID — always look it up first.
   - The system will automatically pause after any update tool call and ask the user for approval before data is written.

BEHAVIORAL GUIDELINES:
1. Decline tasks completely outside your capabilities (flights, arbitrary code, unrelated questions).
2. Base answers on tool results only. If tools return nothing relevant, say so — do not hallucinate.
3. When showing booking/payment/package data, summarize clearly and concisely.
4. For update requests: always confirm WHAT you are changing and for WHICH record before calling the update tool.
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
    async def agent_node(state: RAGState, config: RunnableConfig = None):
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
        username = (config.get("configurable", {}).get("sf_username") if config else None) or state.get("sf_username")
        if username:
            update["sf_username"] = username

        if not response.tool_calls and response.content:
            if isinstance(response.content, list):
                # Extract text from list of blocks (e.g. Gemini/Anthropic format)
                texts = [
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in response.content
                ]
                update["answer"] = "".join(texts)
            else:
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
    # write_count and sf_username (falling back to state) into it.
    _base_tool_node = ToolNode(tools)

    async def tool_node(state: RAGState, config: RunnableConfig = None):
        # Record Prometheus metric for invoked tools
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            tool_calls = getattr(last_message, "tool_calls", None) or []
            for tc in tool_calls:
                tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if tc_name:
                    try:
                        from src.telemetry_metrics import record_tool_call
                        record_tool_call(str(tc_name))
                    except Exception:
                        pass

        write_count = state.get("write_count", 0)
        existing = config.get("configurable", {}) if config else {}
        sf_username = existing.get("sf_username") or state.get("sf_username")
        merged_configurable = {**existing, "write_count": write_count}
        if sf_username:
            merged_configurable["sf_username"] = sf_username
        merged_config = {
            **(config or {}),
            "configurable": merged_configurable,
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

