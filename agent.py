"""Phenotype builder: turn a Genome into an agent_fn(question) -> AgentResult.

Two architectures:
  - single_shot: one Responses call, no tools.
  - react: Responses call with built-in web_search; OpenAI executes the search loop
    server-side. We embed `max_steps` in the prompt and (optionally) chain a
    second call for reflection.

Reflection can be unconditional (threshold=1.0), conditional on a self-reported
confidence below threshold, or off (reflection_enabled=False).
"""
from __future__ import annotations

import re
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


_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _parse_confidence(text: str) -> tuple[str, float | None]:
    """Extract the trailing 'CONFIDENCE: <float>' if present. Return (stripped_answer, confidence)."""
    m = _CONFIDENCE_RE.search(text)
    if not m:
        return text, None
    try:
        c = float(m.group(1))
        if not 0.0 <= c <= 1.0:
            return text, None
    except ValueError:
        return text, None
    stripped = _CONFIDENCE_RE.sub("", text).strip()
    return stripped, c


def _reflection_mode(genome: Genome) -> str:
    """Returns one of 'off', 'always', 'conditional'."""
    if not genome.reflection_enabled:
        return "off"
    if genome.reflection_threshold >= 1.0 - 1e-9:
        return "always"
    if genome.reflection_threshold <= 0.0 + 1e-9:
        return "off"
    return "conditional"


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
    base_instructions = _materialize_prompt(genome)
    tools = schemas_for(genome.tools)
    mode = _reflection_mode(genome)

    if mode == "conditional":
        instructions = (
            base_instructions
            + "\n\nAfter your final answer, write exactly one line:\n"
              "CONFIDENCE: <a single number between 0.0 and 1.0 indicating "
              "how confident you are in the answer>"
        )
    else:
        instructions = base_instructions

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
        raw = _output_text(response)
        answer, confidence = _parse_confidence(raw) if mode == "conditional" else (raw, None)

        for item in getattr(response, "output", []) or []:
            t = getattr(item, "type", "")
            if t.endswith("_call") and t != "function_call":
                trajectory.append({"role": "tool", "name": t, "status": getattr(item, "status", "")})
            elif t == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in ("output_text", "text"):
                        trajectory.append({"role": "assistant", "content": getattr(c, "text", "")})

        if mode == "always":
            should_reflect = bool(answer)
        elif mode == "conditional":
            c = confidence if confidence is not None else 0.0
            should_reflect = bool(answer) and c < genome.reflection_threshold
        else:
            should_reflect = False

        if should_reflect:
            reflect = call_responses(
                instructions=base_instructions,
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
            trajectory.append({
                "role": "assistant", "content": reflected, "reflection": True,
                "confidence": confidence,
            })
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
