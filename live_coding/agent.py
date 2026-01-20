'''
think
act
observe

act
tools

'''
import litellm

model = "openai/gpt-4.1-mini"

def think(messages: list):
    response = litellm.completion(
        model = model,
        messages=messages,
        tools=[],
        tool_choice="auto"
    )

    return response.choices[0].message
    

def act():
    pass

def observe():
    pass


system_prompt = """You are an autonomous agent designed to perform tasks based on user instructions. Follow the think-act-observe loop to achieve the goals set by the user.
"""

messages = [{"role": "system", "content": system_prompt}]

def run_agent():
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
    run_agent()