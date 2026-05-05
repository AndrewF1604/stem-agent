"""Tool registry.

Only use the OpenAI Responses API built-in `web_search` tool — its loop is
executed server-side, so this module is essentially a name->schema mapping
that the phenotype builder consults. Kept as a separate module so that adding
custom function tools later (e.g. fetch_url) is a one-place change.
"""
from __future__ import annotations


WEB_SEARCH_SCHEMA = {"type": "web_search"}


REGISTRY: dict[str, dict] = {
    "web_search": WEB_SEARCH_SCHEMA,
}


def schemas_for(names: list[str]) -> list[dict]:
    return [REGISTRY[n] for n in names if n in REGISTRY]
