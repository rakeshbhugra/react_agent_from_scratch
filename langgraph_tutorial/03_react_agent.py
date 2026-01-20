# Run: python -m langgraph_tutorial.03_react_agent

import warnings
warnings.filterwarnings("ignore")

"""
LangGraph - ReAct Agent with LiteLLM

The ReAct pattern as a graph:
    START -> Reasoner -> (tools?) -> Reasoner -> ... -> END
                  ^                      |
                  |______________________|

Key insight: The loop is created by an edge from tools back to reasoner.
Exit condition: Reasoner returns no tool_calls.
"""

import json
from typing import TypedDict, List
from langgraph.graph import START, END, StateGraph
import litellm
from dotenv import load_dotenv

load_dotenv()


# State
class MessagesState(TypedDict):
    messages: List[dict]


# Tool functions
def sum_numbers(a: float, b: float) -> float:
    """Add two numbers together and return the sum"""
    return a + b


def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together and return the product"""
    return a * b


# Tool definitions for LiteLLM
AVAILABLE_TOOLS = [
    {"type": "function", "function": litellm.utils.function_to_dict(sum_numbers)},
    {"type": "function", "function": litellm.utils.function_to_dict(multiply_numbers)},
]

FUNCTION_MAP = {
    "sum_numbers": sum_numbers,
    "multiply_numbers": multiply_numbers,
}


# Node: Reasoner (Think)
def reasoner(state: MessagesState):
    """The LLM decides what to do next."""
    print("  [Reasoner] Thinking...")

    response = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=state["messages"],
        tools=AVAILABLE_TOOLS,
        tool_choice="auto",
    )

    response_message = response.choices[0].message

    # Build message dict
    if response_message.tool_calls:
        ai_message = {
            "role": "assistant",
            "content": response_message.content or "",
            "tool_calls": response_message.tool_calls
        }
        print(f"  [Reasoner] Requesting tools: {[tc.function.name for tc in response_message.tool_calls]}")
    else:
        ai_message = {
            "role": "assistant",
            "content": response_message.content
        }
        print(f"  [Reasoner] Final answer ready")

    return {"messages": state["messages"] + [ai_message]}


# Node: Tools (Act)
def tools_node(state: MessagesState):
    """Execute tool calls."""
    last_message = state["messages"][-1]
    new_messages = []

    if last_message.get("tool_calls"):
        for tool_call in last_message["tool_calls"]:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"  [Tools] Calling {function_name}({function_args})")
            result = FUNCTION_MAP[function_name](**function_args)
            print(f"  [Tools] Result: {result}")

            new_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

    return {"messages": state["messages"] + new_messages}


# Routing: Continue or End?
def should_continue(state: MessagesState) -> str:
    """Check if we should continue to tools or end."""
    last_message = state["messages"][-1]

    # EXIT when: no tool_calls in response
    if last_message.get("tool_calls"):
        return "tools"
    else:
        return "__end__"


# Build the graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("reasoner", reasoner)
builder.add_node("tools", tools_node)

# Add edges
builder.add_edge(START, "reasoner")
builder.add_conditional_edges(
    "reasoner",
    should_continue,
    {"tools": "tools", "__end__": END}
)
builder.add_edge("tools", "reasoner")  # THE LOOP! Tools -> back to Reasoner

# Compile
react_graph = builder.compile()


if __name__ == "__main__":
    print("=== LangGraph ReAct Agent ===\n")

    test_cases = [
        "What is 25 + 17?",
        "Calculate 8 multiplied by 12",
        "Add 15 and 25, then multiply the result by 2",
        "Hello! How are you?",  # No tool needed
    ]

    for test in test_cases:
        print(f"User: {test}")
        print("-" * 40)

        result = react_graph.invoke({
            "messages": [{"role": "user", "content": test}]
        })

        # Get final answer
        final_message = result["messages"][-1]
        print(f"Agent: {final_message.get('content', '')}\n")
        print("=" * 50 + "\n")
