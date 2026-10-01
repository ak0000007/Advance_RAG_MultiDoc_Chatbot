"""
LangChain tools backed by the Salesforce Async Client.
SOLID: Receives the client via dependency injection.
"""

import json
import re
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from src.write_guards import single_record_guard, bulk_intent_guard, session_write_cap


def build_salesforce_opportunities_tool(salesforce_client):
    """
    Create a LangChain Tool backed by the injected Salesforce client.
    """

    @tool
    async def search_salesforce_opportunities(search: str, config: RunnableConfig) -> str:
        """
        Search for Salesforce Opportunities (deals).
        Use this tool when the user asks about their Salesforce deals, opportunities, or amounts.

        Args:
            search: A specific company name or deal name to search for (e.g. 'Acme').
                    Leave blank to fetch the most recent deals.
        """
        sf_username = config.get("configurable", {}).get("sf_username")

        if not sf_username:
            return "System Error: Missing sf_username in configuration. Cannot execute action on behalf of user."

        try:
            response = await salesforce_client.get_opportunities(
                username=sf_username,
                search=search if search else None,
            )

            if not response.get("success"):
                return f"Salesforce Error: {response.get('errorMessage')}"

            data = response.get("data", [])
            if not data:
                return f"No opportunities found in Salesforce for '{search}'."

            results = []
            for item in data:
                fields = []
                if item.get("id"):        fields.append(f"ID: {item['id']}")
                # Sanitize name: strip control chars & cap length to prevent prompt injection
                raw_name = item.get("name", "")
                safe_name = re.sub(r'[\x00-\x1f\x7f]', '', raw_name)[:120]
                if safe_name:             fields.append(f"Name: {safe_name}")
                if item.get("stageName"): fields.append(f"Stage: {item['stageName']}")
                if item.get("amount"):    fields.append(f"Amount: ${item['amount']}")
                if item.get("closeDate"): fields.append(f"Close Date: {item['closeDate']}")
                results.append(" | ".join(fields))

            header = f"Found {len(results)} opportunity(ies)."
            if len(results) > 1:
                header += (
                    " Multiple matches found — if the user wants to update one, "
                    "ask them to confirm which specific opportunity before proceeding."
                )
            return header + "\n" + "\n".join(results)

        except Exception as e:
            return f"Failed to execute Salesforce query: {str(e)}"

    return search_salesforce_opportunities


def build_salesforce_update_tool(salesforce_client, max_writes_per_session: int = 5):
    """
    Create a LangChain Tool that initiates a Salesforce Opportunity status update.

    Hard caps enforced (in order, before any approval flow):
      1. single_record_guard  — exactly ONE Salesforce ID, no lists or wildcards.
      2. bulk_intent_guard    — rejects bulk phrasing in arguments.
      3. session_write_cap    — rejects if per-session write limit is already reached.

    Does NOT write to Salesforce directly.
    Returns a structured approval-request payload that the human_approval graph node
    intercepts to call interrupt() — freezing the graph until the user responds.

    SRP: This tool only validates input and signals intent.
    The human_approval node owns the interrupt/resume/execute lifecycle.
    """

    @tool
    async def update_salesforce_opportunity_status(
        opportunity_id: str, new_status: str, config: RunnableConfig,
        opportunity_name: str = "",
    ) -> str:
        """
        Request a Salesforce Opportunity stage update for exactly ONE opportunity.
        The system will pause and ask the user to approve before any data is written.

        IMPORTANT: This tool ONLY accepts a single Salesforce Opportunity ID.
        Bulk operations (e.g. "close all deals") are explicitly refused.

        Args:
            opportunity_id: Single 18-character Salesforce Opportunity ID.
            new_status: New stage name (e.g. 'Closed Won', 'Negotiation/Review').
            opportunity_name: Human-readable name of the opportunity (from search results). Optional but preferred.
        """
        sf_username = config.get("configurable", {}).get("sf_username")
        write_count: int = config.get("configurable", {}).get("write_count", 0)

        if not sf_username:
            return "System Error: Missing sf_username in configuration. Cannot execute on behalf of user."

        # ── Hard cap 1: exactly one record ───────────────────────────────────
        if err := single_record_guard(opportunity_id):
            return err

        # ── Hard cap 2: no bulk intent in arguments ───────────────────────────
        if err := bulk_intent_guard(new_status):
            return err

        # ── Hard cap 3: session write ceiling ────────────────────────────────
        if err := session_write_cap(write_count, max_writes_per_session):
            return err

        # Use human-readable name for display; fall back to ID if not provided.
        display_name = opportunity_name.strip() if opportunity_name.strip() else opportunity_id

        # Return structured payload. The human_approval graph node reads this,
        # calls interrupt(), and executes the real update only on Approve.
        return json.dumps({
            "__requires_approval__": True,
            "sf_username": sf_username,
            "opportunity_id": opportunity_id,
            "opportunity_name": display_name,
            "new_status": new_status,
            "message": f"Change '{display_name}' stage to '{new_status}'?",
        })

    return update_salesforce_opportunity_status
