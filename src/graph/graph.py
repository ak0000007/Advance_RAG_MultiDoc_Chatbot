"""
LangGraph workflow construction.

New Architecture (Single ReAct Agent):

START
  ↓
Agent Node ──(if tool needed)──> Tool Node
  ↑                               │
  └────────(tool results)─────────┘
  ↓ (if text response)
 END
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, SystemMessage

from src.graph.state import RAGState


SYSTEM_PROMPT = """You are an intelligent enterprise assistant.
Your primary role is to assist users by querying the company's internal knowledge base and retrieving Salesforce data.

TOOL USAGE GUIDELINES:
1. Document Search: Use `search_documents` when the user asks general questions about company policies, internal documents, knowledge base articles, or factual information.
2. Salesforce Data: Use the Salesforce tools when the user asks about their opportunities, pipeline, deals, or CRM data. 

BEHAVIORAL GUIDELINES:
1. If the user asks for something completely outside of your capabilities (e.g., booking flights, writing arbitrary code, or answering questions unrelated to the business), politely decline.
2. Always base your answers on the context returned by the tools. If the tools return no relevant information, tell the user you don't know. Do not hallucinate facts.
3. When summarizing tool results, be clear and concise.
"""

def build_rag_graph(
    llm,
    tools: list,
    checkpointer=None,
    **kwargs
):
    """
    Build a ReAct Agent workflow.
    Replaces the old strict linear RAG pipeline.
    """
    graph_builder = StateGraph(RAGState)
    
    # 1. Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # 2. Define the Agent Node
    async def agent_node(state: RAGState):
        messages = state.get("messages", [])
        
        # Security/Cost: Limit to 5 tool-call iterations per user turn
        tool_call_count = 0
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "human":
                break
            if getattr(msg, "tool_calls", None):
                tool_call_count += 1
                
        if tool_call_count >= 5:
            # Gracefully stop the loop without crashing the API
            from langchain_core.messages import AIMessage, SystemMessage
            fallback_msg = AIMessage(content="I've reached the maximum number of attempts to find this information. Please try rephrasing your request or checking the constraints.")
            return {"messages": [fallback_msg], "answer": fallback_msg.content}
            
        # Invoke LLM
        
        # Prepend the system prompt dynamically so it guides the LLM but isn't saved to history
        invoke_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = await llm_with_tools.ainvoke(invoke_messages)
        
        # Update state natively
        update = {"messages": [response]}
        
        # If it's a final answer (no tool calls), populate 'answer' field for legacy routes compatibility
        if not response.tool_calls and response.content:
            update["answer"] = str(response.content)
            
        return update

    # 3. Define the routing logic
    def should_continue(state: RAGState) -> Literal["tools", "__end__"]:
        messages = state.get("messages", [])
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "__end__"
        
    # 4. Add Nodes
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode(tools))
    
    # 5. Connect Edges
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    graph_builder.add_edge("tools", "agent")
    
    return graph_builder.compile(checkpointer=checkpointer)
