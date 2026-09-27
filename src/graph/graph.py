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
from langchain_core.messages import AIMessage

from src.graph.state import RAGState

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
        
        # Invoke LLM
        response = await llm_with_tools.ainvoke(messages)
        
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
