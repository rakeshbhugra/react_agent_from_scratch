from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv

load_dotenv()

def search_internet_tool(query: str) -> str:
    """Search the internet and return results with URLs.

    Parameters
    ----------
    query : str
        The search query string.

    Returns
    -------
    str
        Search results with titles, snippets, and URLs.
    """
    search = SerpAPIWrapper()
    results = search.results(query)

    # Format results with URLs
    formatted = []
    for r in results.get("organic_results", []):
        title = r.get("title", "No title")
        snippet = r.get("snippet", "No snippet")
        link = r.get("link", "No URL")
        formatted.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")

    return "\n\n".join(formatted) if formatted else "No results found"

    
if __name__ == "__main__":
    # Example usage
    query = "What is the capital of France?"
    results = search_internet_tool(query)
    print(f"Search results for '{query}':\n{results}")