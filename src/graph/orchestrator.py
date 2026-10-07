"""
Orchestrator graph for multi-agent routing.

Classifies incoming user intent into one of three specialist domains:
  - 'crm': Salesforce Opportunities & Deal pipeline
  - 'travel': Bookings, Travel Packages, and Payments
  - 'docs': Internal knowledge base & policy documents
  - 'multi': Requests spanning multiple domains (resolved via priority: crm > travel > docs)
"""

import re
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from src.graph.state import RAGState

_CRM_KEYWORDS_RE = re.compile(
    r"\b(deal|deals|opportunity|opportunities|pipeline|stage|acme|stark)\b",
    re.IGNORECASE,
)
_TRAVEL_KEYWORDS_RE = re.compile(
    r"\b(booking|bookings|book|package|packages|payment|payments|traveler|travelers|vacation|trip|destination|itinerary|itineraries)\b|bk-\w+|pay-\w+",
    re.IGNORECASE,
)
_DOCS_KEYWORDS_RE = re.compile(
    r"\b(policy|policies|handbook|guideline|guidelines|insurance|reimbursement|reimbursements|document|documents|faq|faqs)\b",
    re.IGNORECASE,
)


class IntentClassification(BaseModel):
    """Structured result produced by the intent classification router."""

    domain: Literal["crm", "travel", "docs", "multi"] = Field(
        description=(
            "The classified domain for routing: "
            "'crm' for Salesforce opportunities/deals/sales pipeline; "
            "'travel' for bookings, travel packages, payments, itineraries, or reservations; "
            "'docs' for company policies, employee handbook, reimbursement guidelines, FAQs, or documentation; "
            "'multi' if the message genuinely spans more than one domain (e.g. both a booking and travel policy inquiry)."
        )
    )
    reason: str = Field(
        default="",
        description="Brief justification for the domain decision."
    )


def resolve_multi_priority(user_text: str) -> Literal["crm", "travel", "docs"]:
    """
    Resolve multi-domain intent to the highest-priority domain.

    Documented scope control: True concurrent execution across multiple subgraphs
    is deferred to future work. In this version, conflicts are resolved using the
    priority hierarchy: crm > travel > docs.
    """
    has_crm = bool(_CRM_KEYWORDS_RE.search(user_text))
    has_travel = bool(_TRAVEL_KEYWORDS_RE.search(user_text))
    has_docs = bool(_DOCS_KEYWORDS_RE.search(user_text))

    if has_crm:
        return "crm"
    if has_travel:
        return "travel"
    if has_docs:
        return "docs"
    # Fallback to top priority
    return "crm"


def build_intent_classifier(llm):
    """Build the intent classification chain with structured output."""
    parser = PydanticOutputParser(pydantic_object=IntentClassification)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an intent classification router for an enterprise travel and CRM system.
Analyze the user's message and categorize it into exactly one of these domains:
- 'crm': Salesforce sales pipeline, deals, opportunities, stages (e.g. Acme deal, Closed Won, opportunity updates).
- 'travel': Travel bookings (BK-xxxxxx), travel packages, destinations/prices, payment transactions (PAY-xxxxxx), receipts, reservation updates.
- 'docs': Internal company travel policies, employee handbook, reimbursement guidelines, insurance coverage, FAQs, documentation.
- 'multi': If the user message genuinely spans more than one of the above domains (e.g. both a booking and a travel policy inquiry).

If the message is general chit-chat (greetings, 'what can you do', jokes) or general inquiries, classify as 'travel'.

{format_instructions}""",
        ),
        (
            "human",
            """User message:

{question}

Classify the domain of this message.""",
        ),
    ]).partial(format_instructions=parser.get_format_instructions())

    try:
        structured_llm = llm.with_structured_output(IntentClassification)
    except AttributeError:
        structured_llm = llm

    return prompt | structured_llm


def build_orchestrator_graph(
    crm_graph,
    travel_graph,
    docs_graph,
    llm=None,
    checkpointer=None,
):
    """
    Build and compile the multi-agent orchestrator graph.

    Wires classify_intent as the entry node and routes to whichever compiled
    subgraph was classified, then to END.
    """
    if llm is None:
        from src.llm.provider import create_llm
        from src.config import settings

        fallback = "openai" if settings.openai_api_key else None
        llm = create_llm(
            provider="google",
            fallback_provider=fallback,
            temperature=0.0,
            structured_output=IntentClassification,
        )

    classifier_chain = build_intent_classifier(llm)

    async def classify_intent(state: RAGState, config: RunnableConfig = None) -> dict:
        user_question = state.get("question") or ""
        if not user_question:
            messages = state.get("messages", [])
            for m in reversed(messages):
                if getattr(m, "type", "") in ("human", "user") or isinstance(m, HumanMessage):
                    c = getattr(m, "content", "")
                    if isinstance(c, list):
                        c = " ".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in c])
                    user_question = str(c)
                    break

        try:
            res = await classifier_chain.ainvoke({"question": user_question})
            if isinstance(res, IntentClassification):
                domain = res.domain
            elif isinstance(res, dict):
                domain = res.get("domain", "travel")
            elif hasattr(res, "domain"):
                domain = getattr(res, "domain")
            else:
                domain = "travel"
                for d in ["crm", "travel", "docs", "multi"]:
                    if d in str(res).lower():
                        domain = d
                        break
        except Exception:
            domain = "travel"

        if domain == "multi":
            resolved_domain = resolve_multi_priority(user_question)
        else:
            resolved_domain = domain if domain in ("crm", "travel", "docs") else "travel"

        return {"classified_domain": resolved_domain}

    def route_after_classification(state: RAGState) -> Literal["crm_agent", "travel_agent", "docs_agent"]:
        dom = state.get("classified_domain", "travel")
        if dom == "crm":
            return "crm_agent"
        if dom == "docs":
            return "docs_agent"
        return "travel_agent"

    builder = StateGraph(RAGState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("crm_agent", crm_graph)
    builder.add_node("travel_agent", travel_graph)
    builder.add_node("docs_agent", docs_graph)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "crm_agent": "crm_agent",
            "travel_agent": "travel_agent",
            "docs_agent": "docs_agent",
        },
    )
    builder.add_edge("crm_agent", END)
    builder.add_edge("travel_agent", END)
    builder.add_edge("docs_agent", END)

    cp = checkpointer or getattr(crm_graph, "checkpointer", None)
    return builder.compile(checkpointer=cp)
