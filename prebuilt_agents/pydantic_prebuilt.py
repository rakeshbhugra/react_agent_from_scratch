# Run: python -m prebuilt_agents.pydantic_prebuilt

import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
from crawl4ai import AsyncWebCrawler

load_dotenv()

# Create agent with tools
agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a helpful research assistant. Use search_web to find information and scrape_page to get detailed content from URLs.",
)


@agent.tool_plain
def search_web(query: str) -> str:
    """Search the web for information."""
    search = SerpAPIWrapper()
    results = search.results(query)

    formatted = []
    for r in results.get("organic_results", [])[:5]:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        formatted.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")

    return "\n\n".join(formatted) if formatted else "No results found"


@agent.tool_plain
def scrape_page(url: str) -> str:
    """Scrape content from a webpage."""
    async def _scrape():
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return result.markdown[:5000]

    return asyncio.run(_scrape())


def print_tool_calls(result):
    """Print all tool calls from the conversation."""
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                print(f"  [Tool Call] {part.tool_name}({part.args})")
            elif isinstance(part, ToolReturnPart):
                content = str(part.content)[:200]
                print(f"  [Tool Result] {content}...")


if __name__ == "__main__":
    query = "What are the latest features in Python 3.13?"
    print(f"Query: {query}\n")

    result = agent.run_sync(query)

    print("Tool calls:")
    print_tool_calls(result)

    print(f"\nAgent response:\n{result.output}")
