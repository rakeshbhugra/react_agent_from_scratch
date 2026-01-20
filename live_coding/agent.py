'''
think
act
observe

act
tools

'''
import litellm
import json
from .tools import AVAILABLE_TOOLS, FUNCTION_MAP
import warnings

warnings.filterwarnings("ignore")


model = "openai/gpt-4.1-mini"

def think(messages: list):
    response = litellm.completion(
        model = model,
        messages=messages,
        tools=AVAILABLE_TOOLS,
        tool_choice="auto"
    )

    return response.choices[0].message
    

def act(message):
    if not message.tool_calls:
        return None
    
    results = []
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments

        function_args = json.loads(function_args)

        if function_name in FUNCTION_MAP:
            result = FUNCTION_MAP[function_name](**function_args)
            results.append((tool_call.id, function_name, result))
        else:
            results.append((tool_call.id, function_name, f"Error: Function {function_name} not found."))

    return results

def observe(messages, tool_results, assistant_message):
    messages.append({"role": "assistant", "content": assistant_message.content, "tool_calls": assistant_message.tool_calls})

    if tool_results is None:
        return True

    # Add all tool results to message history
    for tool_call_id, function_name, result in tool_results:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(result)
        })
        print(f"  [Tool] {function_name} -> {result}")

    return False
        


system_prompt = """You are an autonomous agent designed to perform tasks based on user instructions. Follow the think-act-observe loop to achieve the goals set by the user.
"""

messages = [{"role": "system", "content": system_prompt}]

def run_agent(user_input):
    messages.append({"role": "user", "content": user_input})
    max_iterations = 10
    for _ in range(max_iterations):
        assistant_message = think(messages)
        tool_results = act(assistant_message)
        done = observe(messages, tool_results, assistant_message)

        if done:
            final_response = assistant_message.content
            print("Final Response:", final_response)
            return final_response

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        run_agent(user_input)
