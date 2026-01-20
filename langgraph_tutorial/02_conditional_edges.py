# Run: python -m langgraph_tutorial.02_conditional_edges

import warnings
warnings.filterwarnings("ignore")

"""
LangGraph - Conditional Edges (Routing)

Key concept: Use conditional edges to route to different nodes
based on the current state. This enables dynamic workflows.
"""

import json
from typing import Literal
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel
from litellm import completion
from dotenv import load_dotenv

load_dotenv()


# Define structured output for sentiment
class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]


# State includes messages and detected sentiment
class State(BaseModel):
    messages: list[dict]
    sentiment: str = "neutral"


# Node: Analyze sentiment using LLM
def analyze_sentiment(state: State):
    print("  [Node] Analyzing sentiment...")

    response = completion(
        model="openai/gpt-4o-mini",
        messages=state.messages + [
            {"role": "system", "content": "Analyze the sentiment of the user's message. Respond with JSON: {\"sentiment\": \"positive\" | \"negative\" | \"neutral\"}"}
        ],
        response_format=SentimentResponse
    )

    json_content = response.choices[0].message.content
    sentiment_data = SentimentResponse(**json.loads(json_content))
    state.sentiment = sentiment_data.sentiment
    print(f"  [Node] Detected sentiment: {state.sentiment}")

    return state


# Response nodes for each sentiment
def positive_response(state: State):
    print("  [Node] Positive response: Thank you for sharing your positive thoughts!")
    return state


def negative_response(state: State):
    print("  [Node] Negative response: I'm sorry to hear that. How can I help?")
    return state


def neutral_response(state: State):
    print("  [Node] Neutral response: I see. Please tell me more.")
    return state


# Routing function - determines which node to go to next
def route_by_sentiment(state: State) -> str:
    """Route to different nodes based on sentiment."""
    if state.sentiment == "positive":
        return "positive_node"
    elif state.sentiment == "negative":
        return "negative_node"
    else:
        return "neutral_node"


# Build the graph
builder = StateGraph(State)

# Add nodes
builder.add_node("analyze_sentiment", analyze_sentiment)
builder.add_node("positive_node", positive_response)
builder.add_node("negative_node", negative_response)
builder.add_node("neutral_node", neutral_response)

# Add edges
builder.add_edge(START, "analyze_sentiment")

# CONDITIONAL EDGES - route based on sentiment
builder.add_conditional_edges(
    "analyze_sentiment",
    route_by_sentiment,
    {
        "positive_node": "positive_node",
        "negative_node": "negative_node",
        "neutral_node": "neutral_node"
    }
)

# All response nodes lead to END
builder.add_edge("positive_node", END)
builder.add_edge("negative_node", END)
builder.add_edge("neutral_node", END)

# Compile
graph = builder.compile()


if __name__ == "__main__":
    print("=== Conditional Edges Example ===\n")

    test_messages = [
        "I just got promoted at work! So excited!",
        "My computer crashed and I lost all my files.",
        "What's the weather like today?"
    ]

    for msg in test_messages:
        print(f"User: {msg}")
        result = graph.invoke(State(messages=[{"role": "user", "content": msg}]))
        print(f"Final sentiment: {result['sentiment']}\n")
        print("-" * 50)
