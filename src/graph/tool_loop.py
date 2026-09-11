"""
LangGraph tool-calling agent.

Architecture:

    Agent
      ↓
    Tool requested?
      ↓
    ToolNode
      ↓
    Tool result
      ↓
    Agent
      ↓
    ...
      ↓
    Final answer / bounded stop

The agent uses the project's real document
retrieval tool.

The number of agent iterations is explicitly
bounded to prevent uncontrolled tool execution.
"""

from langgraph.graph import (
    StateGraph,
    START,
    END,
    MessagesState,
)

from langgraph.prebuilt import (
    ToolNode,
)

from src.tools.document_search import (
    build_document_search_tool,
)


# =========================================================
# Agent Configuration
# =========================================================

MAX_AGENT_STEPS = 5


# =========================================================
# Agent State
# =========================================================


class AgentState(MessagesState):
    """
    State used by the tool-calling agent.

    MessagesState already provides:

        messages

    We additionally track:

        agent_steps

    This allows the application to enforce a maximum
    number of agent/tool iterations.
    """

    agent_steps: int


# =========================================================
# Agent Graph
# =========================================================


def build_tool_execution_graph(
    chat_model,
    retriever,
):
    """
    Build the bounded LangGraph tool-calling agent.

    Flow:

        START
          ↓
        agent
          ↓
        tool requested?
         /        \
       no          yes
       ↓            ↓
      END      steps < MAX?
                    /   \
                  yes    no
                   ↓      ↓
                 tools  fallback
                   ↓
                 agent
    """

    # =====================================================
    # 1. Create project tool
    # =====================================================

    document_search = build_document_search_tool(
        retriever
    )

    tools = [
        document_search
    ]

    # =====================================================
    # 2. Bind tools to model
    # =====================================================

    model_with_tools = chat_model.bind_tools(
        tools
    )

    # =====================================================
    # 3. ToolNode
    # =====================================================

    tool_node = ToolNode(
        tools
    )

    # =====================================================
    # 4. Agent node
    # =====================================================

    def call_model(state: AgentState):

        response = model_with_tools.invoke(
            state["messages"]
        )

        current_steps = state.get(
            "agent_steps",
            0,
        )

        return {
            "messages": [
                response
            ],
            "agent_steps": current_steps + 1,
        }

    # =====================================================
    # 5. Routing after agent
    # =====================================================

    def route_after_agent(
        state: AgentState,
    ):
        """
        Decide whether the agent should:

        1. Finish
        2. Execute a tool
        3. Stop because the maximum number of
           agent steps has been reached.
        """

        messages = state.get(
            "messages",
            [],
        )

        if not messages:
            return "end"

        last_message = messages[-1]

        # -------------------------------------------------
        # No tool call → final answer
        # -------------------------------------------------

        if not getattr(
            last_message,
            "tool_calls",
            None,
        ):
            return "end"

        # -------------------------------------------------
        # Tool call exists
        # -------------------------------------------------

        steps = state.get(
            "agent_steps",
            0,
        )

        if steps >= MAX_AGENT_STEPS:
            return "max_steps"

        return "tools"

    # =====================================================
    # 6. Maximum-step fallback
    # =====================================================

    def max_steps_node(
        state: AgentState,
    ):
        """
        Safely terminate the agent when it reaches
        the maximum allowed number of model decisions.
        """

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        "I couldn't complete the request "
                        "within the allowed number of "
                        "agent steps."
                    ),
                }
            ]
        }

    # =====================================================
    # 7. Graph
    # =====================================================

    graph_builder = StateGraph(
        AgentState
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

    graph_builder.add_node(
        "max_steps",
        max_steps_node,
    )

    # -----------------------------------------------------
    # START → agent
    # -----------------------------------------------------

    graph_builder.add_edge(
        START,
        "agent",
    )

    # -----------------------------------------------------
    # agent → decision
    # -----------------------------------------------------

    graph_builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "end": END,
            "max_steps": "max_steps",
        },
    )

    # -----------------------------------------------------
    # max_steps → END
    # -----------------------------------------------------

    graph_builder.add_edge(
        "max_steps",
        END,
    )

    # -----------------------------------------------------
    # tools → agent
    # -----------------------------------------------------

    graph_builder.add_edge(
        "tools",
        "agent",
    )

    # =====================================================
    # 8. Compile
    # =====================================================

    return graph_builder.compile()