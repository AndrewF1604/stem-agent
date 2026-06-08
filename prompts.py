"""Trzy szablony promptow + helper do skladania (pytanie + kontekst) -> finalny prompt.

Szablony odpowiadaja conditions w sweep'ie:
  - ZERO_SHOT:    minimalny, "answer based on context"
  - COT:          chain-of-thought, "think step by step"
  - VERBOSE:      explicit instructions + format hint
"""
from __future__ import annotations

from dataclasses import dataclass


def _format_context(context: list[dict]) -> str:
    blocks = []
    for para in context:
        title = para["title"]
        text = " ".join(para["sentences"])
        blocks.append(f"[{title}]\n{text}")
    return "\n\n".join(blocks)


ZERO_SHOT = """Answer the question based on the following context.

Context:
{context}

Question: {question}

Answer:"""


COT = """Answer the question based on the following context. Think step by step, then give the final answer on a new line prefixed with "Final answer:".

Context:
{context}

Question: {question}

Reasoning:"""


VERBOSE = """You are answering a multi-hop question that requires combining information from multiple passages. Read the context carefully, identify the relevant passages, and answer with a short phrase (a name, year, place, or short noun phrase). Do not add explanations.

Context:
{context}

Question: {question}

Short answer:"""


@dataclass
class PromptTemplate:
    name: str
    template: str

    def render(self, question: str, context: list[dict]) -> str:
        return self.template.format(question=question, context=_format_context(context))


TEMPLATES = {
    "zero_shot": PromptTemplate("zero_shot", ZERO_SHOT),
    "cot":       PromptTemplate("cot", COT),
    "verbose":   PromptTemplate("verbose", VERBOSE),
}


AGENT_SYSTEM_PROMPT = """You are a research agent answering multi-hop factual questions. You have access to a web_search tool.

Steps:
1. Decide what information you need to answer the question.
2. Call web_search with a focused query (3-8 words, not the whole question).
3. Read the results. If you need more facts, call web_search again with a different focused query.
4. After at most 4 searches, give the final answer as a short phrase (a name, year, place, or 'yes'/'no').

Do not include extra explanation in the final answer. Just the short phrase."""


AGENT_INITIAL_USER = """Question: {question}

Find the answer using web_search. Final answer must be a short phrase."""

