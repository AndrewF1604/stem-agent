"""DuckDuckGo search wrapper (darmowy, bez API key).

Zwraca top N snippetow z title + url + body.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from duckduckgo_search import DDGS


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def web_search(query: str, max_results: int = 4, retries: int = 2) -> list[SearchResult]:
    last_err = None
    for attempt in range(retries + 1):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                )
                for r in results
            ]
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    return [SearchResult(title="(search failed)", url="", snippet=str(last_err))]


def format_results(results: list[SearchResult]) -> str:
    if not results:
        return "(no results)"
    blocks = []
    for r in results:
        blocks.append(f"- {r.title}\n  {r.url}\n  {r.snippet[:300]}")
    return "\n".join(blocks)
