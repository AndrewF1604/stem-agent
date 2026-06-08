"""Inference przez Ollama. Dwa tryby:

  - 'context'   - model dostaje pytanie + 10 paragrafow w prompcie (reading comprehension)
  - 'agent'     - model dostaje tylko pytanie, ma narzedzie web_search, samodzielnie szuka

Tryb 'agent' implementuje ReAct loop: model -> tool_call -> search -> wynik -> model -> ...
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from prompts import TEMPLATES, AGENT_SYSTEM_PROMPT, AGENT_INITIAL_USER
from search import web_search, format_results


OLLAMA_BASE = "http://localhost:11434/v1"
_client = OpenAI(base_url=OLLAMA_BASE, api_key="ollama")


@dataclass
class InferenceConfig:
    name: str
    model: str
    prompt_template: str
    temperature: float
    max_tokens: int = 200
    seed: Optional[int] = None
    mode: str = "context"
    max_agent_steps: int = 4
    iters: int = 1


@dataclass
class InferenceResult:
    answer: str
    raw_output: str
    config_name: str
    prompt_chars: int
    output_tokens: int = 0
    latency_s: float = 0.0
    error: Optional[str] = None
    agent_trajectory: list = field(default_factory=list)


_FINAL_ANSWER_RE = re.compile(r"final\s*answer[:\s\-]+\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE | re.DOTALL)


def _extract_answer(raw: str, template_name: str) -> str:
    raw = raw.strip()
    if template_name == "cot":
        m = _FINAL_ANSWER_RE.search(raw)
        if m:
            return m.group(1).strip().strip('"').strip("'").rstrip(".").strip()
    first = raw.split("\n", 1)[0].strip()
    return first.strip('"').strip("'").rstrip(".").strip()


def _infer_context(question: str, context: list[dict], config: InferenceConfig, seed: Optional[int] = None) -> InferenceResult:
    template = TEMPLATES[config.prompt_template]
    prompt = template.render(question, context)
    t0 = time.time()
    effective_seed = seed if seed is not None else config.seed
    try:
        kwargs = dict(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        if effective_seed is not None:
            kwargs["seed"] = effective_seed
        resp = _client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        out_tok = getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0
        return InferenceResult(
            answer=_extract_answer(raw, config.prompt_template),
            raw_output=raw,
            config_name=config.name,
            prompt_chars=len(prompt),
            output_tokens=out_tok,
            latency_s=time.time() - t0,
        )
    except Exception as e:
        return InferenceResult(
            answer="", raw_output="", config_name=config.name,
            prompt_chars=len(prompt), latency_s=time.time() - t0, error=str(e),
        )


_TOOLS_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for facts. Use focused queries (3-8 words), not full questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, 3-8 words"},
            },
            "required": ["query"],
        },
    },
}]


def _infer_agent(question: str, config: InferenceConfig, seed: Optional[int] = None) -> InferenceResult:
    t0 = time.time()
    effective_seed = seed if seed is not None else config.seed
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": AGENT_INITIAL_USER.format(question=question)},
    ]
    trajectory: list = []
    total_out_tok = 0
    raw_pieces: list[str] = []

    for step in range(config.max_agent_steps):
        try:
            step_kwargs = dict(
                model=config.model,
                messages=messages,
                tools=_TOOLS_SCHEMA,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            if effective_seed is not None:
                step_kwargs["seed"] = effective_seed
            resp = _client.chat.completions.create(**step_kwargs)
        except Exception as e:
            return InferenceResult(
                answer="", raw_output="\n---\n".join(raw_pieces),
                config_name=config.name, prompt_chars=sum(len(str(m)) for m in messages),
                output_tokens=total_out_tok, latency_s=time.time() - t0,
                error=f"step {step}: {e}", agent_trajectory=trajectory,
            )

        msg = resp.choices[0].message
        total_out_tok += getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0
        raw_pieces.append(f"[step {step}] content={msg.content!r} tool_calls={len(msg.tool_calls or [])}")

        if not msg.tool_calls:
            answer_text = (msg.content or "").strip()
            trajectory.append({"step": step, "type": "final_answer", "text": answer_text})
            return InferenceResult(
                answer=_extract_answer(answer_text, "zero_shot"),
                raw_output="\n---\n".join(raw_pieces),
                config_name=config.name,
                prompt_chars=sum(len(str(m)) for m in messages),
                output_tokens=total_out_tok,
                latency_s=time.time() - t0,
                agent_trajectory=trajectory,
            )

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {"query": tc.function.arguments}
            query = args.get("query", "").strip()
            results = web_search(query, max_results=4) if query else []
            results_text = format_results(results)
            trajectory.append({
                "step": step, "type": "search", "query": query,
                "n_results": len(results),
                "results_titles": [r.title for r in results[:4]],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": results_text[:2000],
            })

    messages.append({
        "role": "user",
        "content": "Based on everything above, give your final short answer now. One short phrase only.",
    })
    try:
        forced_kwargs = dict(
            model=config.model, messages=messages,
            max_tokens=100, temperature=config.temperature,
        )
        if effective_seed is not None:
            forced_kwargs["seed"] = effective_seed
        resp = _client.chat.completions.create(**forced_kwargs)
        forced = (resp.choices[0].message.content or "").strip()
        total_out_tok += getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0
        raw_pieces.append(f"[forced final] {forced!r}")
        trajectory.append({"step": config.max_agent_steps, "type": "forced_final", "text": forced})
        return InferenceResult(
            answer=_extract_answer(forced, "zero_shot"),
            raw_output="\n---\n".join(raw_pieces),
            config_name=config.name,
            prompt_chars=sum(len(str(m)) for m in messages),
            output_tokens=total_out_tok,
            latency_s=time.time() - t0,
            agent_trajectory=trajectory,
        )
    except Exception as e:
        return InferenceResult(
            answer="", raw_output="\n---\n".join(raw_pieces),
            config_name=config.name, prompt_chars=0,
            output_tokens=total_out_tok, latency_s=time.time() - t0,
            error=f"forced final: {e}", agent_trajectory=trajectory,
        )


def infer(question: str, context: list[dict], config: InferenceConfig, seed: Optional[int] = None) -> InferenceResult:
    if config.mode == "agent":
        return _infer_agent(question, config, seed=seed)
    return _infer_context(question, context, config, seed=seed)


# iters: ile próbek na pytanie. Tylko temp>0 (i agent, przez zmienność DDG) dają
# realny rozkład — przy temp=0 w trybie context dekodowanie jest greedy, więc
# sweep.py i tak wymusza K=1 dla tych conditions niezależnie od tej wartości.
CONDITIONS = [
    InferenceConfig("A_zero_t0",    "llama3.1:8b", "zero_shot", temperature=0.0, mode="context", iters=1),
    InferenceConfig("B_zero_t07",   "llama3.1:8b", "zero_shot", temperature=0.7, mode="context", iters=10),
    InferenceConfig("C_zero_t10",   "llama3.1:8b", "zero_shot", temperature=1.0, mode="context", iters=10),
    InferenceConfig("D_cot_t0",     "llama3.1:8b", "cot",       temperature=0.0, mode="context", iters=1),
    InferenceConfig("E_cot_t07",    "llama3.1:8b", "cot",       temperature=0.7, mode="context", iters=10),
    InferenceConfig("F_verbose_t0", "llama3.1:8b", "verbose",   temperature=0.0, mode="context", iters=1),
    InferenceConfig("G_agent_search", "llama3.1:8b", "agent",   temperature=0.0, mode="agent", max_agent_steps=4, iters=4),
]
