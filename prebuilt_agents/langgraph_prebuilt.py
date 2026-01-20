# Run: python -m prebuilt_agents.langgraph_prebuilt

import warnings

from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# Define tools using LangChain decorator
@tool
def sum_numbers(a: float, b: float) -> float:
    """Add two numbers together and return the sum"""
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together and return the product"""
    return a * b

# Create agent - that's it! Loop is built-in
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[sum_numbers, multiply_numbers],
    system_prompt="You are a helpful assistant"
)

# Test it
if __name__ == "__main__":
    result = agent.invoke({
        "messages": [("user", "What is 25 + 17? Then add 20 to the result")]
    })

    print("Agent response:")
    for message in result["messages"]:
        print(f"{message.type}: {message.content}")