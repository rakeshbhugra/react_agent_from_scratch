"""
Pandas Agent - Talk to Excel files using natural language.

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

EXCEL_FILE = "pandas_agent/database.xlsx"

SYSTEM_PROMPT = f"""You are a data analyst. Use Python code to answer questions about the Excel file.

File: {EXCEL_FILE}

Schema:
- Customer ID, Customer Name, Email, Phone
- Amount Due, Due Date, Payment Status
- User Status, Last Contact, Follow Up Required

Example code:
```python
import pandas as pd
df = pd.read_excel('{EXCEL_FILE}')
print(df.head())
```

Rules:
1. Always use pandas to read and analyze the data
2. Always print() your results
3. If you get an error, fix your code and try again
4. Be precise with column names"""


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

        print(f"\n  [Code]\n{args.get('code', '')}\n")

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
        print(f"  [Output] {result}")

    return False


def run(user_input: str, max_iterations: int = 5) -> str:
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
            print(f"\nAgent: {response}")
            return response

    return "Max iterations reached"


if __name__ == "__main__":
    print("=== Pandas Agent ===\n")
    run("Show me the first 3 customers")
