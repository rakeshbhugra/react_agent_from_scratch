"""
ReAct Agent - A simple agentic loop implementation

The core loop:
    while not done:
        think()
        act()
        observe()
"""

import json
import litellm
from dotenv import load_dotenv
from .tools import AVAILABLE_TOOLS, FUNCTION_MAP

load_dotenv()

MODEL = "openai/gpt-4o-mini"


def think(messages: list) -> dict:
    """Let the model reason about what to do next."""
    response = litellm.completion(
        model=MODEL,
        messages=messages,
        tools=AVAILABLE_TOOLS,
        tool_choice="auto"
    )
    return response.choices[0].message


def act(message) -> list[tuple[str, str, str]] | None:
    """Execute all tool calls if the model decided to use any."""
    if not message.tool_calls:
        return None

    results = []
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        if function_name in FUNCTION_MAP:
            result = FUNCTION_MAP[function_name](**function_args)
            results.append((tool_call.id, function_name, str(result)))
        else:
            results.append((tool_call.id, function_name, f"Error: Unknown function {function_name}"))

    return results


def observe(messages: list, assistant_message, tool_results: list | None) -> bool:
    """Update message history with results. Returns True if done."""
    messages.append({
        "role": "assistant",
        "content": assistant_message.content,
        "tool_calls": assistant_message.tool_calls
    })

    if tool_results is None:
        return True

    # Add all tool results to message history
    for tool_call_id, function_name, result in tool_results:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        })
        print(f"  [Tool] {function_name} -> {result}")

    return False


def run(user_input: str, max_iterations: int = 10) -> str:
    """
    Run the agent loop.

    while not done:
        think()
        act()
        observe()
    """
    messages = [{"role": "user", "content": user_input}]
    print(f"User: {user_input}")

    for _ in range(max_iterations):
        assistant_message = think(messages)
        tool_results = act(assistant_message)
        done = observe(messages, assistant_message, tool_results)

        if done:
            final_response = assistant_message.content
            print(f"Agent: {final_response}")
            return final_response

    return "Max iterations reached"


if __name__ == "__main__":
    print("=== ReAct Agent Demo ===\n")

    test_queries = [
        "What is 25 + 37?",
        "What is 15 times 8?",
        "What's the weather in San Francisco?",
    ]

    for query in test_queries:
        print("-" * 40)
        run(query)
        print()
