# Run: python -m langgraph_tutorial.01_simple_graph

"""
LangGraph Basics - Simple Graph

Key concepts:
- State: Shared data that flows through the graph
- Nodes: Functions that transform state
- Edges: Connections between nodes
- START/END: Special entry and exit points
"""

from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel


# 1. Define State - the data that flows through your graph
class State(BaseModel):
    messages: list[dict]


# 2. Define Nodes - functions that transform state
def chatbot_node(state: State):
    """Simple echo chatbot"""
    print("  [Node] chatbot processing...")
    user_message = state.messages[-1]["content"]
    bot_response = f"Echo: {user_message}"
    state.messages.append({"role": "assistant", "content": bot_response})
    return state


# 3. Build the Graph
builder = StateGraph(State)

# Add nodes
builder.add_node("chatbot", chatbot_node)

# Add edges (connections)
builder.add_edge(START, "chatbot")  # Start -> chatbot
builder.add_edge("chatbot", END)    # chatbot -> End

# 4. Compile the graph
graph = builder.compile()


# 5. Run it
if __name__ == "__main__":
    print("=== Simple LangGraph Example ===\n")

    initial_state = State(messages=[
        {"role": "user", "content": "Hello, LangGraph!"}
    ])

    result = graph.invoke(initial_state)

    print("\nResult:")
    for msg in result["messages"]:
        print(f"  {msg['role']}: {msg['content']}")
