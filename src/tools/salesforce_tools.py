"""
LangChain tools backed by the Salesforce Async Client.
SOLID: Receives the client via dependency injection.
"""

import json
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

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
        # Read the human's identity from the graph config (injected by the API route)
        sf_username = config.get("configurable", {}).get("sf_username")
        
        if not sf_username:
            return "System Error: Missing sf_username in configuration. Cannot execute action on behalf of user."
            
        try:
            response = await salesforce_client.get_opportunities(
                username=sf_username,
                search=search if search else None
            )
            
            if not response.get("success"):
                return f"Salesforce Error: {response.get('errorMessage')}"
                
            data = response.get("data", [])
            if not data:
                return f"No opportunities found in Salesforce for '{search}'."
                
            # Format nicely for the LLM
            results = []
            for item in data:
                # Omit null values to save tokens
                fields = []
                if item.get("name"): fields.append(f"Name: {item['name']}")
                if item.get("stageName"): fields.append(f"Stage: {item['stageName']}")
                if item.get("amount"): fields.append(f"Amount: ${item['amount']}")
                if item.get("closeDate"): fields.append(f"Close Date: {item['closeDate']}")
                
                results.append(" | ".join(fields))
                
            return "\n".join(results)
            
        except Exception as e:
            return f"Failed to execute Salesforce query: {str(e)}"

    return search_salesforce_opportunities
