"""Genome: the mutable DNA of the agent.

Kept deliberately small. Adding fields here expands the search space — only do it
if a planned mutation needs the new degree of freedom.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Architecture = Literal["single_shot", "react"]
ToolName = Literal["web_search"]


class Genome(BaseModel):
    system_prompt: str
    architecture: Architecture = "react"
    tools: list[ToolName] = Field(default_factory=lambda: ["web_search"])
    max_steps: int = 4
    reflection_enabled: bool = False
    reflection_threshold: float = 1.0
    generation: int = 0
    parent_hash: str | None = None
    notes: str = ""

    def hash(self) -> str:
        payload = self.model_dump(exclude={"generation", "parent_hash", "notes"})
        return hashlib.sha1(
            yaml.safe_dump(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.model_dump(), sort_keys=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Genome":
        return cls(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))


DEFAULT_SINGLE_SHOT_PROMPT = (
    "You answer multi-hop factual questions concisely. "
    "Reply with the final answer in one short phrase."
)

DEFAULT_REACT_PROMPT = (
    "You are a research agent answering multi-hop factual questions. "
    "Use web_search to gather evidence before answering. "
    "Search for entities you don't already know about, then refine queries to confirm specific facts. "
    "Use at most {max_steps} searches; stop searching as soon as you have enough evidence. "
    "Reply with the final answer in one short phrase."
)


def initial_population() -> list[Genome]:
    """Seed genomes for generation 0. Stem agent picks the best one to start from."""
    return [
        Genome(
            system_prompt=DEFAULT_SINGLE_SHOT_PROMPT,
            architecture="single_shot",
            tools=[],
            max_steps=1,
            reflection_enabled=False,
            generation=0,
            notes="seed: minimal single-shot, no tools",
        ),
        Genome(
            system_prompt=DEFAULT_REACT_PROMPT,
            architecture="react",
            tools=["web_search"],
            max_steps=5,
            reflection_enabled=False,
            generation=0,
            notes="seed: ReAct with web_search, generous step budget",
        ),
        Genome(
            system_prompt=DEFAULT_REACT_PROMPT,
            architecture="react",
            tools=["web_search"],
            max_steps=3,
            reflection_enabled=False,
            generation=0,
            notes="seed: ReAct with web_search, tight step budget",
        ),
    ]
