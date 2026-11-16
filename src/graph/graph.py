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
import logging
from typing import Literal, Optional, Any
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, trim_messages
from langchain_core.runnables import RunnableConfig

from src.graph.state import RAGState
from src.retrieval.citations import extract_citations_from_text

logger = logging.getLogger(__name__)


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
1. Document Search: Use `search_documents` ONLY for company policies, internal FAQs, employee guidelines, or general travel knowledge base questions.
   CRITICAL: NEVER call `search_documents` for Salesforce CRM data queries or CRM record updates (Opportunities, Deals, Bookings, Travel Packages, Payments). For any Salesforce/CRM query or action, ALWAYS and ONLY use the dedicated Salesforce tools (`search_salesforce_opportunities`, `update_salesforce_opportunity_status`, `get_booking`, `update_booking`, `get_travel_packages`, `update_travel_package`, `get_payments`, `update_payment`).
2. Salesforce Opportunities: Use `search_salesforce_opportunities` for deals/pipeline queries.
3. Bookings: When the user asks about bookings or asks to modify a booking by name, CALL `get_booking` immediately (with NO arguments to fetch all user bookings, or with a specific booking number if provided).
4. Travel Packages: Use `get_travel_packages` with no args to list all active packages. Pass a package ID to get details for one.
5. Payments: When the user asks about payments or transactions, CALL `get_payments` immediately (with NO arguments to fetch all user payments, or with a booking reference if provided).
6. Update Workflow (for ALL update tools):
   - Direct Update: If the user's message already contains a valid-looking 15- or 18-character Salesforce ID (e.g. starting with `006`, `a00`, `a01`, `a02`), call the corresponding update tool directly with that ID without a redundant lookup.
   - Lookup First: If only a name, description, or no ID is provided, first call the relevant GET tool (`get_booking`, `get_travel_packages`, `get_payments`, or `search_salesforce_opportunities`) to look up the record.
   - Ambiguity & Multiple Matches: If multiple matching records exist or the target is ambiguous, present the options and ask an explicit clarifying question asking the user which specific record to update before proceeding.
   - NEVER fabricate or guess a Salesforce ID.
   - The system will automatically pause after any update tool call and ask the user for approval before data is written.

BEHAVIORAL GUIDELINES:
1. Decline tasks completely outside your capabilities (flights, arbitrary code, unrelated questions).
2. Base answers on tool results only. If tools return nothing relevant, say so — do not hallucinate.
3. When showing booking/payment/package data, summarize clearly and concisely.
4. When clarifying ambiguity, always ask the user a direct, explicit question specifying which item to act on.
5. Citation Discipline: When answering questions using information from search_documents, cite every factual claim inline immediately using '[source, page]' (e.g. '[cancellation_policy.pdf, page 3]'). NEVER cite documents or page numbers not provided in the retrieved context. Every citation is automatically verified against retrieved documents.
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


def safe_trim_messages(
    messages: list[Any],
    max_messages: int = 50,
    system_prompt: str = SYSTEM_PROMPT,
    summary: Optional[str] = None,
) -> list[Any]:
    """
    Safely trim message history for LLM invocation while preserving conversation integrity:
    1. If max_messages <= 0 or len(messages) <= max_messages, return all messages with system prompt.
    2. Use trim_messages with strategy='last' and start_on='human' to avoid starting on orphaned Tool/AI messages.
    3. Ensure the current active turn (from latest HumanMessage onward) is never truncated.
    4. Prepend system prompt and optional summary.
    """
    if not messages or max_messages <= 0 or len(messages) <= max_messages:
        invoke = [SystemMessage(content=system_prompt)]
        if summary:
            invoke.append(SystemMessage(content=f"Summary of prior conversation:\n{summary}"))
        return invoke + list(messages)

    # Trim to last max_messages with start_on='human'
    trimmed = trim_messages(
        messages,
        max_tokens=max_messages,
        token_counter=len,
        strategy="last",
        start_on="human",
    )

    # Active turn preservation guard: locate the latest human message
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if getattr(msg, "type", "") in ("human", "user") or isinstance(msg, HumanMessage):
            last_human_idx = i
            break

    if last_human_idx != -1:
        current_turn = messages[last_human_idx:]
        # If trimming severed or dropped the active turn, ensure at least the active turn is preserved
        if len(trimmed) < len(current_turn):
            trimmed = current_turn

    invoke = [SystemMessage(content=system_prompt)]
    if summary:
        invoke.append(SystemMessage(content=f"Summary of prior conversation:\n{summary}"))
    elif len(trimmed) < len(messages):
        invoke.append(SystemMessage(content="[Note: Earlier conversation turns have been archived to stay within context limits.]"))

    logger.debug(
        "Bounded memory trim: trimmed working context from %d to %d messages (max_messages=%d)",
        len(messages),
        len(trimmed),
        max_messages,
    )
    return invoke + list(trimmed)


def build_rag_graph(
    llm,
    tools: list,
    checkpointer=None,
    human_approval_node=None,
    system_prompt: str = SYSTEM_PROMPT,
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
        system_prompt: Optional domain-specific system prompt (defaults to SYSTEM_PROMPT).
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

        # Memory Management: Safely trim working context for LLM invocation while checkpoint preserves full history
        from src.config import settings
        cfg_max_msgs = config.get("configurable", {}).get("max_messages") if config else None
        max_msgs = cfg_max_msgs if cfg_max_msgs is not None else getattr(settings, "max_messages", 50)
        summary = state.get("summary")

        invoke_messages = safe_trim_messages(
            messages,
            max_messages=max_msgs,
            system_prompt=system_prompt,
            summary=summary,
        )
        response = await llm_with_tools.ainvoke(invoke_messages, config=config)

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
                raw_answer = "".join(texts)
            else:
                raw_answer = str(response.content)

            # Automated Inline Citation Verification
            current_citations = state.get("citations") or []
            verification_mode = (
                config.get("configurable", {}).get("citation_verification_mode")
                if config
                else None
            )
            if not verification_mode:
                from src.config import settings
                verification_mode = getattr(settings, "citation_verification_mode", "flag")

            if current_citations:
                from src.retrieval.citations import verify_and_filter_inline_citations
                verified_answer, _, verification_summary = verify_and_filter_inline_citations(
                    raw_answer,
                    current_citations,
                    mode=verification_mode,
                )
                update["answer"] = verified_answer
                update["citation_verification"] = verification_summary
            else:
                update["answer"] = raw_answer

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
        tool_result = await _base_tool_node.ainvoke(state, config=merged_config)

        # Capture and merge citation metadata from document search results
        messages = tool_result.get("messages", []) if isinstance(tool_result, dict) else []
        citations = []
        for msg in messages:
            content = getattr(msg, "content", "")
            if isinstance(content, str) and "<document" in content:
                if "Note: Relevance of these documents could not be fully confirmed." not in content:
                    citations.extend(extract_citations_from_text(content))

        if citations and isinstance(tool_result, dict):
            existing = state.get("citations") or []
            seen_ids = {c.get("chunk_id") for c in existing if isinstance(c, dict)}
            new_cites = [c for c in citations if c.get("chunk_id") not in seen_ids]
            tool_result["citations"] = existing + new_cites

        return tool_result

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

