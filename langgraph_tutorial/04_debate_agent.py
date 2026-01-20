# Run: python -m langgraph_tutorial.04_debate_agent

"""
LangGraph - Debate Pattern (Multi-Agent)

Pattern:
    ┌──────────┐                    ┌──────────┐
    │   Pro    │◄──────────────────►│   Con    │
    │  Agent   │     debate          │  Agent   │
    └────┬─────┘                    └────┬─────┘
         │                               │
         └───────────────┬───────────────┘
                         │
                  ┌──────▼──────┐
                  │    Judge    │
                  │   Agent     │
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │  Synthesis  │
                  └─────────────┘

Exit conditions:
1. Judge calls for synthesis
2. Max debate rounds reached (typically 2-3)

Use case: Critical decisions, exploring multiple viewpoints
"""

from typing import TypedDict, List, Literal
from langgraph.graph import START, END, StateGraph
import litellm
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

MAX_DEBATE_ROUNDS = 2  # Typically 2-3 rounds is sufficient


# State
class DebateState(TypedDict):
    topic: str
    pro_arguments: List[str]
    con_arguments: List[str]
    round: int
    judge_decision: str  # "CONTINUE" or "SYNTHESIZE"
    synthesis: str


# Pro Agent - argues FOR the topic
def pro_agent(state: DebateState) -> DebateState:
    print(f"\n  [Pro Agent] Round {state['round']} - Building argument...")

    # Build context from previous arguments
    context = f"Topic: {state['topic']}\n\n"
    if state["con_arguments"]:
        context += "Previous Con arguments to rebut:\n"
        for arg in state["con_arguments"]:
            context += f"- {arg}\n"

    response = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a debate participant arguing FOR the given topic.
Be persuasive and use facts, logic, and compelling arguments.
If there are opposing arguments, address and rebut them.
Keep your response to 2-3 sentences."""},
            {"role": "user", "content": context}
        ]
    )

    argument = response.choices[0].message.content
    print(f"  [Pro Agent] Argument: {argument}")

    return {
        **state,
        "pro_arguments": state["pro_arguments"] + [argument]
    }


# Con Agent - argues AGAINST the topic
def con_agent(state: DebateState) -> DebateState:
    print(f"\n  [Con Agent] Round {state['round']} - Building counter-argument...")

    # Build context from pro arguments
    context = f"Topic: {state['topic']}\n\n"
    context += "Pro arguments to counter:\n"
    for arg in state["pro_arguments"]:
        context += f"- {arg}\n"

    response = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a debate participant arguing AGAINST the given topic.
Be persuasive and use facts, logic, and compelling counter-arguments.
Directly address and rebut the Pro arguments.
Keep your response to 2-3 sentences."""},
            {"role": "user", "content": context}
        ]
    )

    argument = response.choices[0].message.content
    print(f"  [Con Agent] Argument: {argument}")

    return {
        **state,
        "con_arguments": state["con_arguments"] + [argument],
        "round": state["round"] + 1
    }


# Judge Agent - evaluates and decides whether to continue or synthesize
def judge_agent(state: DebateState) -> DebateState:
    print(f"\n  [Judge] Evaluating debate after round {state['round']}...")

    # Build summary of all arguments
    summary = f"Topic: {state['topic']}\n\n"
    summary += "PRO ARGUMENTS:\n"
    for i, arg in enumerate(state["pro_arguments"], 1):
        summary += f"{i}. {arg}\n"
    summary += "\nCON ARGUMENTS:\n"
    for i, arg in enumerate(state["con_arguments"], 1):
        summary += f"{i}. {arg}\n"

    response = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are an impartial judge evaluating a debate.
Decide if the debate has covered enough ground or needs more rounds.
If arguments are well-developed on both sides, say "SYNTHESIZE".
If more exploration is needed, say "CONTINUE".
Respond with just one word: SYNTHESIZE or CONTINUE"""},
            {"role": "user", "content": summary}
        ]
    )

    decision = response.choices[0].message.content.strip().upper()
    if "SYNTHESIZE" in decision:
        decision = "SYNTHESIZE"
    else:
        decision = "CONTINUE"

    print(f"  [Judge] Decision: {decision}")

    return {**state, "judge_decision": decision}


# Synthesis Agent - creates final balanced conclusion
def synthesis_agent(state: DebateState) -> DebateState:
    print("\n  [Synthesis] Creating balanced conclusion...")

    summary = f"Topic: {state['topic']}\n\n"
    summary += "PRO ARGUMENTS:\n"
    for arg in state["pro_arguments"]:
        summary += f"- {arg}\n"
    summary += "\nCON ARGUMENTS:\n"
    for arg in state["con_arguments"]:
        summary += f"- {arg}\n"

    response = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a synthesis expert.
Create a balanced conclusion that:
1. Acknowledges the strongest points from both sides
2. Identifies areas of agreement
3. Provides a nuanced recommendation
Keep it to 3-4 sentences."""},
            {"role": "user", "content": summary}
        ]
    )

    synthesis = response.choices[0].message.content
    print(f"  [Synthesis] Conclusion: {synthesis}")

    return {**state, "synthesis": synthesis}


# Routing function - should we continue debating or synthesize?
def should_continue_debate(state: DebateState) -> Literal["pro_agent", "synthesis"]:
    """
    Exit conditions:
    1. Judge calls for synthesis
    2. Max debate rounds reached
    """
    if state["judge_decision"] == "SYNTHESIZE":
        return "synthesis"
    if state["round"] >= MAX_DEBATE_ROUNDS:
        print(f"  [Router] Max rounds ({MAX_DEBATE_ROUNDS}) reached, forcing synthesis")
        return "synthesis"
    return "pro_agent"


# Build the graph
builder = StateGraph(DebateState)

# Add nodes
builder.add_node("pro_agent", pro_agent)
builder.add_node("con_agent", con_agent)
builder.add_node("judge", judge_agent)
builder.add_node("synthesis", synthesis_agent)

# Add edges
builder.add_edge(START, "pro_agent")       # Start with Pro
builder.add_edge("pro_agent", "con_agent")  # Pro -> Con
builder.add_edge("con_agent", "judge")      # Con -> Judge

# Conditional: Judge decides to continue or synthesize
builder.add_conditional_edges(
    "judge",
    should_continue_debate,
    {"pro_agent": "pro_agent", "synthesis": "synthesis"}
)

builder.add_edge("synthesis", END)

# Compile
debate_graph = builder.compile()


if __name__ == "__main__":
    print("=" * 60)
    print("       DEBATE PATTERN - Multi-Agent Example")
    print("=" * 60)

    # Test topics
    topics = [
        "Remote work is better than office work",
        # "AI will replace most jobs in the next 10 years",
    ]

    for topic in topics:
        print(f"\n{'='*60}")
        print(f"TOPIC: {topic}")
        print("=" * 60)

        initial_state: DebateState = {
            "topic": topic,
            "pro_arguments": [],
            "con_arguments": [],
            "round": 1,
            "judge_decision": "CONTINUE",
            "synthesis": ""
        }

        result = debate_graph.invoke(initial_state)

        print("\n" + "=" * 60)
        print("FINAL SYNTHESIS:")
        print("=" * 60)
        print(result["synthesis"])
        print(f"\nTotal rounds: {result['round']}")
