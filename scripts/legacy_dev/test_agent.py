import asyncio
from src.api.dependencies import get_compiled_graph
from langchain_core.messages import HumanMessage

async def main():
    print("1. Compiling the Agent Graph (Loading tools and LLM)...")
    graph = get_compiled_graph()
    
    # We will test the chatbot with a Salesforce-specific question.
    question = "Can you check my Salesforce opportunities and summarize them?"
    sf_username = "test@example.com" # Dummy email, SF will reject if it's not a real user, but we'll see the tool attempt.
    
    print(f"\n2. Sending question to Agent: '{question}'")
    print(f"   (Running as SF Username: {sf_username})")
    
    inputs = {
        "messages": [HumanMessage(content=question)]
    }
    config = {
        "configurable": {
            "thread_id": "test-123",
            "sf_username": sf_username
        }
    }
    
    print("\n3. Agent Execution Log (Tracing tool calls):")
    
    # Run the graph and stream the steps
    try:
        async for event in graph.astream(inputs, config=config):
            for node, state in event.items():
                print(f"   [Node Executed]: {node}")
                # Print the last message produced by this node
                if "messages" in state and len(state["messages"]) > 0:
                    last_msg = state["messages"][-1]
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        print(f"      -> Tool Called: {last_msg.tool_calls[0]['name']}")
                    elif hasattr(last_msg, "name"):
                        print(f"      -> Tool Result: {last_msg.content[:100]}...")
                    else:
                        print(f"      -> Agent Output: {last_msg.content[:100]}...")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())
