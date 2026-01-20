import warnings
warnings.filterwarnings("ignore")

from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()

# Create agent with tools
agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant",
)

# Define tools using decorator
@agent.tool_plain
def sum_numbers(a: float, b: float) -> float:
    """Add two numbers together and return the sum"""
    return a + b

@agent.tool_plain
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together and return the product"""
    return a * b


# Test it
if __name__ == "__main__":
    result = agent.run_sync("What is 25 + 17? Then add 20 to the result")

    print("Agent response:")
    print(result.output)
