from .tools import AVAILABLE_TOOLS, FUNCTION_MAP
import litellm

model = "openai/gpt-4o-mini"

user_message = "What is 25 + 37?"
messages = [{"role": "user", "content": user_message}]


# Initial completion with tools
response = litellm.completion(
    model=model,
    messages=messages,
    tools=AVAILABLE_TOOLS,
    tool_choice="auto"  # Let model decide when to use tools
)

print(response) 