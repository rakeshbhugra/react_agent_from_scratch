"""
Tools for the Internet Search Agent.
"""

import json
import asyncio
import litellm
from langchain_community.utilities import SerpAPIWrapper
from crawl4ai import AsyncWebCrawler


def search_web(query: str) -> str:
    """Search the web for information.

    Parameters
    ----------
    query : str
        The search query.
    """
    search = SerpAPIWrapper()
    results = search.results(query)

    formatted = []
    for r in results.get("organic_results", [])[:5]:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        formatted.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")

    return "\n\n".join(formatted) if formatted else "No results found"


def scrape_page(url: str) -> str:
    """Scrape content from a webpage.

    Parameters
    ----------
    url : str
        The URL to scrape.
    """
    async def _scrape():
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return result.markdown[:5000]

    return asyncio.run(_scrape())


def create_plan(task: str) -> str:
    """Create a research plan for a task.

    Parameters
    ----------
    task : str
        The research task to plan.
    """
    return f"Plan for: {task}\n1. Search for information\n2. Scrape key sources\n3. Synthesize findings"


# Tool schemas
AVAILABLE_TOOLS = [
    {"type": "function", "function": litellm.utils.function_to_dict(search_web)},
    {"type": "function", "function": litellm.utils.function_to_dict(scrape_page)},
    {"type": "function", "function": litellm.utils.function_to_dict(create_plan)},
]

FUNCTION_MAP = {
    "search_web": search_web,
    "scrape_page": scrape_page,
    "create_plan": create_plan,
}
