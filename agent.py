"""Phenotype builder: turn a Genome into an agent_fn(question) -> AgentResult.

Two architectures:
  - single_shot: one Responses call, no tools.
  - react: Responses call with built-in web_search; OpenAI executes the search loop
    server-side. We embed `max_steps` in the prompt and (optionally) chain a
    second call for reflection.
"""
from __future__ import annotations

import time
from typing import Callable

from eval import (
    AgentResult,
    WORKER_MODEL,
    _count_tool_calls,
    _output_text,
    _usage,
    call_responses,
)
from genome import Genome
from tools import schemas_for


def _materialize_prompt(genome: Genome) -> str:
    return genome.system_prompt.format(max_steps=genome.max_steps)


def _single_shot(genome: Genome) -> Callable[[str], AgentResult]:
    instructions = _materialize_prompt(genome)

    def fn(question: str) -> AgentResult:
        t0 = time.time()
        response = call_responses(
            instructions=instructions,
            input_data=question,
            max_output_tokens=400,
        )
        text = _output_text(response)
        in_tok, out_tok, cached = _usage(response)
        return AgentResult(
            answer=text,
            trajectory=[{"role": "user", "content": question},
                        {"role": "assistant", "content": text}],
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=cached,
            steps=1,
            latency_s=time.time() - t0,
        )
    return fn


def _react(genome: Genome) -> Callable[[str], AgentResult]:
    instructions = _materialize_prompt(genome)
    tools = schemas_for(genome.tools)

    def fn(question: str) -> AgentResult:
        t0 = time.time()
        trajectory: list[dict] = [{"role": "user", "content": question}]

        response = call_responses(
            instructions=instructions,
            input_data=question,
            tools=tools or None,
            max_output_tokens=1200,
        )
        in_tok, out_tok, cached = _usage(response)
        steps = _count_tool_calls(response)
        answer = _output_text(response)

        for item in getattr(response, "output", []) or []:
            t = getattr(item, "type", "")
            if t.endswith("_call") and t != "function_call":
                trajectory.append({"role": "tool", "name": t, "status": getattr(item, "status", "")})
            elif t == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in ("output_text", "text"):
                        trajectory.append({"role": "assistant", "content": getattr(c, "text", "")})

        if genome.reflection_enabled and answer:
            reflect = call_responses(
                instructions=instructions,
                input_data=(
                    "Briefly reconsider: does the evidence above actually support that answer? "
                    "If yes, repeat it. If no, give the corrected answer. "
                    "Reply with just the final answer in one short phrase."
                ),
                previous_response_id=response.id,
                max_output_tokens=150,
            )
            r_in, r_out, r_cached = _usage(reflect)
            in_tok += r_in
            out_tok += r_out
            cached += r_cached
            reflected = _output_text(reflect)
            trajectory.append({"role": "assistant", "content": reflected, "reflection": True})
            answer = reflected or answer

        return AgentResult(
            answer=answer,
            trajectory=trajectory,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=cached,
            steps=max(steps, 1),
            latency_s=time.time() - t0,
        )
    return fn


def build_agent_fn(genome: Genome) -> Callable[[str], AgentResult]:
    if genome.architecture == "single_shot":
        return _single_shot(genome)
    if genome.architecture == "react":
        return _react(genome)
    raise ValueError(f"unknown architecture: {genome.architecture}")
