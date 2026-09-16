"""
LangGraph tool-calling agent with conversation memory.

Architecture:

    User Message
         ↓
    Agent / Model
         ↓
    Tool requested?
       /       \
     NO         YES
     ↓           ↓
    END        ToolNode
                 ↓
               Agent
                 ↓
                ...

Conversation state is persisted by an
InMemorySaver checkpointer using thread_id.
"""

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

from langgraph.checkpoint.memory import (
    InMemorySaver,
)

from src.tools.document_search import (
    build_document_search_tool,
)


def build_tool_execution_graph(
    chat_model,
    retriever,
):
    """
    Build the LangGraph tool-calling agent.

    The graph supports:

    1. LLM tool selection
    2. Tool execution
    3. Tool result → LLM loop
    4. Conversation persistence through
       an InMemorySaver checkpointer
    """

    # =====================================================
    # 1. Create project retrieval tool
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
    # 3. Create ToolNode
    # =====================================================

    tool_node = ToolNode(
        tools
    )

    # =====================================================
    # 4. Model / Agent node
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

    # =====================================================
    # 6. Add nodes
    # =====================================================

    graph_builder.add_node(
        "agent",
        call_model,
    )

    graph_builder.add_node(
        "tools",
        tool_node,
    )

    # =====================================================
    # 7. START → Agent
    # =====================================================

    graph_builder.add_edge(
        START,
        "agent",
    )

    # =====================================================
    # 8. Agent → Tool OR END
    # =====================================================

    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    # =====================================================
    # 9. Tool → Agent
    # =====================================================

    graph_builder.add_edge(
        "tools",
        "agent",
    )

    # =====================================================
    # 10. Create in-memory checkpointer
    # =====================================================

    checkpointer = InMemorySaver()

    # =====================================================
    # 11. Compile graph with checkpointer
    # =====================================================

    return graph_builder.compile(
        checkpointer=checkpointer
    )