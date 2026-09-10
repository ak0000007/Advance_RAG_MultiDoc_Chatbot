"""
LangGraph tool-calling workflow.

This module teaches and implements the basic pattern:

    Model
      ↓
    tool_calls?
      ↓
    ToolNode
      ↓
    tool result
      ↓
    Model
      ↓
    final answer

The tool used here is backed by the project's
real document retrieval system.
"""

from langchain_core.messages import HumanMessage

from langgraph.graph import (
    StateGraph,
    START,
    END,
    MessagesState,
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from src.tools.document_search import (
    build_document_search_tool,
)


def build_tool_execution_graph(
    chat_model,
    retriever,
):
    """
    Build a LangGraph tool execution workflow.

    The model can decide whether it needs to call
    search_documents.

    ToolNode executes the tool.

    The result is returned to the model.

    The model can then produce the final answer
    or request another tool call.
    """

    # =====================================================
    # 1. Create real project tool
    # =====================================================

    document_search = (
        build_document_search_tool(
            retriever
        )
    )

    tools = [
        document_search
    ]

    # =====================================================
    # 2. Bind tools to model
    # =====================================================

    model_with_tools = (
        chat_model.bind_tools(
            tools
        )
    )

    # =====================================================
    # 3. Create LangGraph ToolNode
    # =====================================================

    tool_node = ToolNode(
        tools
    )

    # =====================================================
    # 4. Model node
    # =====================================================

    def call_model(
        state: MessagesState,
    ):

        response = (
            model_with_tools.invoke(
                state["messages"]
            )
        )

        return {
            "messages": [
                response
            ]
        }

    # =====================================================
    # 5. Create graph
    # =====================================================

    graph_builder = StateGraph(
        MessagesState
    )

    # -----------------------------------------------------
    # Nodes
    # -----------------------------------------------------

    graph_builder.add_node(
        "agent",
        call_model,
    )

    graph_builder.add_node(
        "tools",
        tool_node,
    )

    # -----------------------------------------------------
    # START → agent
    # -----------------------------------------------------

    graph_builder.add_edge(
        START,
        "agent",
    )

    # -----------------------------------------------------
    # agent → tools OR END
    #
    # tools_condition checks whether the AIMessage
    # contains tool_calls.
    # -----------------------------------------------------

    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    # -----------------------------------------------------
    # tools → agent
    #
    # This creates the tool execution loop.
    # -----------------------------------------------------

    graph_builder.add_edge(
        "tools",
        "agent",
    )

    # =====================================================
    # 6. Compile
    # =====================================================

    return graph_builder.compile()