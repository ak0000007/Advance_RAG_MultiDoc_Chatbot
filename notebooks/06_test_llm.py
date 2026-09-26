"""
Colab Test Script for LLM Engine
Run this script in Google Colab with a T4 GPU.

Usage in Colab:
1. !git clone <repo>
2. !pip install -r requirements.txt
3. !python notebooks/06_test_llm.py
"""

import sys
import os

# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm import load_llm
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# Define a dummy tool to test tool binding
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny and 75 degrees."

def main():
    print("Loading LLM... (This downloads the 4B model and takes a few minutes)")
    try:
        llm = load_llm()
        print("LLM loaded successfully in 4-bit quantization.")
    except Exception as e:
        print(f"Failed to load LLM: {e}")
        return

    print("\n--- Test 1: Basic Generation ---")
    messages = [HumanMessage(content="What is the capital of France? Answer in one word.")]
    try:
        response = llm.invoke(messages)
        print(f"Response:\n{response.content}")
    except Exception as e:
        print(f"Generation failed: {e}")

    print("\n--- Test 2: Tool Binding ---")
    try:
        llm_with_tools = llm.bind_tools([get_weather])
        messages = [HumanMessage(content="What is the weather like in Tokyo right now?")]
        response = llm_with_tools.invoke(messages)
        
        if response.tool_calls:
            print(f"Success! Model decided to use a tool:")
            for tool_call in response.tool_calls:
                print(f" - Tool Name: {tool_call['name']}")
                print(f" - Arguments: {tool_call['args']}")
        else:
            print("Model did not call a tool. It responded with:")
            print(response.content)
    except Exception as e:
        print(f"Tool binding test failed: {e}")

if __name__ == "__main__":
    main()
