"""
Internet Search Agent - Research the web using search and scraping.

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

SYSTEM_PROMPT = """You are an internet research agent. Your job is to help users find and synthesize information from the web.

You have access to these tools:
- search_web: Search the internet for information
- scrape_page: Get detailed content from a specific URL
- create_plan: Create a research plan for complex tasks

Strategy:
1. For simple questions, search and answer directly
2. For complex research, create a plan first
3. Use scrape_page when you need more details from a search result
4. Always cite your sources with URLs

Be concise and factual."""


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
    """Execute all tool calls."""
    if not message.tool_calls:
        return None

    results = []
    for tool_call in message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if name in FUNCTION_MAP:
            result = FUNCTION_MAP[name](**args)
            results.append((tool_call.id, name, str(result)))
        else:
            results.append((tool_call.id, name, f"Unknown tool: {name}"))

    return results


def observe(messages: list, assistant_message, tool_results: list | None) -> bool:
    """Update message history. Returns True if done."""
    messages.append({
        "role": "assistant",
        "content": assistant_message.content,
        "tool_calls": assistant_message.tool_calls
    })

    if tool_results is None:
        return True

    for tool_call_id, name, result in tool_results:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        })
        # Truncate long results for display
        display = result[:200] + "..." if len(result) > 200 else result
        print(f"  [{name}] {display}")

    return False


def run(user_input: str, max_iterations: int = 10) -> str:
    """Run the agent loop."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    print(f"User: {user_input}")

    for _ in range(max_iterations):
        assistant_message = think(messages)
        tool_results = act(assistant_message)
        done = observe(messages, assistant_message, tool_results)

        if done:
            response = assistant_message.content
            print(f"Agent: {response}")
            return response

    return "Max iterations reached"


if __name__ == "__main__":
    print("=== Internet Search Agent ===\n")
    run("What is the latest news about AI agents?")
