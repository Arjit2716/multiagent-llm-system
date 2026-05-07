"""
Search tools: Web search via DuckDuckGo (no API key needed) and Wikipedia.
"""
import asyncio
import httpx
from typing import List, Optional

from backend.tools.registry import BaseTool, ToolParameter, ToolResult
from backend.core.logging import get_logger

logger = get_logger(__name__)


class WebSearchTool(BaseTool):
    """
    Web search using DuckDuckGo Instant Answer API.
    No API key required. Returns top search results.
    """

    name = "web_search"
    description = "Search the web for current information. Returns top results with titles, URLs, and snippets."
    category = "search"
    parameters = [
        ToolParameter("query", "str", "The search query", required=True),
        ToolParameter("max_results", "int", "Maximum number of results to return", required=False, default=5),
    ]

    async def run(self, query: str, max_results: int = 5) -> ToolResult:
        """Search DuckDuckGo and return structured results."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # DuckDuckGo Instant Answer API
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": "1",
                        "skip_disambig": "1",
                    },
                )
                response.raise_for_status()
                data = response.json()

            results = []

            # Abstract (main result)
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", "Main Result"),
                    "snippet": data["Abstract"],
                    "url": data.get("AbstractURL", ""),
                    "source": data.get("AbstractSource", ""),
                })

            # Related topics
            for topic in data.get("RelatedTopics", [])[:max_results - len(results)]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Result", "Related")[:100],
                        "snippet": topic["Text"],
                        "url": topic.get("FirstURL", ""),
                        "source": "DuckDuckGo",
                    })

            # Answer box
            if data.get("Answer") and len(results) < max_results:
                results.insert(0, {
                    "title": "Direct Answer",
                    "snippet": data["Answer"],
                    "url": "",
                    "source": "DuckDuckGo",
                })

            if not results:
                # Fallback: try searching with a simple query
                results = [{"title": "No results", "snippet": f"No results found for: {query}", "url": "", "source": ""}]

            return ToolResult(
                tool_name=self.name,
                success=True,
                output=results[:max_results],
                metadata={"query": query, "result_count": len(results)},
            )
        except Exception as e:
            logger.error("web_search_error", query=query, error=str(e))
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=f"Search failed: {e}",
            )


class WikipediaTool(BaseTool):
    """
    Wikipedia article lookup. Returns a summary of a topic.
    """

    name = "wikipedia_search"
    description = "Look up information on Wikipedia. Returns article summary for a given topic."
    category = "search"
    parameters = [
        ToolParameter("topic", "str", "Topic to search for on Wikipedia", required=True),
        ToolParameter("sentences", "int", "Number of sentences to return", required=False, default=5),
    ]

    async def run(self, topic: str, sentences: int = 5) -> ToolResult:
        """Fetch Wikipedia summary for a topic."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_"),
                    headers={"User-Agent": "MultiAgentSystem/1.0"},
                )
                
                if response.status_code == 404:
                    # Try search endpoint
                    search_resp = await client.get(
                        "https://en.wikipedia.org/w/api.php",
                        params={
                            "action": "opensearch",
                            "search": topic,
                            "limit": 1,
                            "format": "json",
                        },
                    )
                    search_data = search_resp.json()
                    if search_data[1]:
                        article_title = search_data[1][0]
                        response = await client.get(
                            f"https://en.wikipedia.org/api/rest_v1/page/summary/{article_title.replace(' ', '_')}",
                            headers={"User-Agent": "MultiAgentSystem/1.0"},
                        )

                if response.status_code != 200:
                    return ToolResult(
                        tool_name=self.name,
                        success=False,
                        output=None,
                        error=f"Wikipedia article not found for: {topic}",
                    )

                data = response.json()
                summary = data.get("extract", "No summary available.")
                
                # Limit to requested sentences
                import re
                sentence_list = re.split(r'(?<=[.!?]) +', summary)
                truncated = " ".join(sentence_list[:sentences])

                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output={
                        "title": data.get("title", topic),
                        "summary": truncated,
                        "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                        "description": data.get("description", ""),
                    },
                    metadata={"topic": topic},
                )
        except Exception as e:
            logger.error("wikipedia_error", topic=topic, error=str(e))
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=f"Wikipedia lookup failed: {e}",
            )
