"""Evaluation harness: load questions, run an agent_fn, judge with an LLM.

This is the foundation. Everything else (agent, mutations, evolution) just
swaps what gets passed as `agent_fn`. The only contract is:

    agent_fn(question: str) -> AgentResult

where AgentResult carries the answer plus telemetry (cost, steps, trajectory).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

WORKER_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
JUDGE_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
DATA_DIR = Path("data")
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

client = OpenAI()


@dataclass
class Question:
    qid: str
    question: str
    gold_answer: str
    supporting_facts: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    answer: str
    trajectory: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    steps: int = 0
    latency_s: float = 0.0
    error: str | None = None


@dataclass
class JudgedResult:
    qid: str
    question: str
    gold: str
    answer: str
    correct: bool
    judge_reasoning: str
    cost_usd: float
    latency_s: float
    steps: int
    trajectory: list[dict]


def load_questions(n: int = 30, seed: int = 42, split: str = "validation") -> list[Question]:
    """Load a deterministic subset of HotpotQA distractor split."""
    cache_path = DATA_DIR / f"hotpot_{split}_{n}_{seed}.json"
    if cache_path.exists():
        return [Question(**q) for q in json.loads(cache_path.read_text(encoding="utf-8"))]

    ds = load_dataset("hotpot_qa", "distractor", split=split)
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)

    questions: list[Question] = []
    for i in indices[: n * 2]:
        row = ds[i]
        if row.get("type") not in ("comparison", "bridge"):
            continue
        if len(row["answer"]) > 100:
            continue
        questions.append(
            Question(
                qid=row["id"],
                question=row["question"],
                gold_answer=row["answer"],
                supporting_facts=row["supporting_facts"]["title"],
            )
        )
        if len(questions) >= n:
            break

    cache_path.write_text(json.dumps([asdict(q) for q in questions], indent=2), encoding="utf-8")
    return questions


def _model_cost_usd(model: str, in_tok: int, out_tok: int, cache_read_tok: int = 0) -> float:
    if "mini" in model:
        return (in_tok * 0.15 + out_tok * 0.60 + cache_read_tok * 0.075) / 1_000_000
    if "gpt-5" in model or "gpt-4" in model:
        return (in_tok * 2.50 + out_tok * 10.0 + cache_read_tok * 1.25) / 1_000_000
    return 0.0


def call_responses(
    instructions: str,
    input_data,
    *,
    tools: list[dict] | None = None,
    previous_response_id: str | None = None,
    max_output_tokens: int | None = None,
    model: str | None = None,
):
    """Thin wrapper around responses.create with a single retry on rate limits."""
    kwargs = dict(model=model or WORKER_MODEL, input=input_data)
    if instructions:
        kwargs["instructions"] = instructions
    if tools:
        kwargs["tools"] = tools
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id
    if max_output_tokens:
        kwargs["max_output_tokens"] = max_output_tokens

    try:
        return client.responses.create(**kwargs)
    except Exception as e:
        if "rate" in str(e).lower() or "429" in str(e):
            time.sleep(8)
            return client.responses.create(**kwargs)
        raise


def _output_text(response) -> str:
    """Robustly extract assistant text from a Responses API response."""
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()
    chunks = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) in ("output_text", "text"):
                    chunks.append(getattr(c, "text", ""))
    return " ".join(chunks).strip()


def _count_tool_calls(response) -> int:
    """Count built-in tool calls (web_search etc.) in a Responses output."""
    n = 0
    for item in getattr(response, "output", []) or []:
        t = getattr(item, "type", "")
        if t.endswith("_call") and t != "function_call":
            n += 1
    return n


def _usage(response) -> tuple[int, int, int]:
    u = getattr(response, "usage", None)
    if not u:
        return 0, 0, 0
    in_tok = getattr(u, "input_tokens", 0) or 0
    out_tok = getattr(u, "output_tokens", 0) or 0
    cached = 0
    details = getattr(u, "input_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return in_tok, out_tok, cached


JUDGE_INSTRUCTIONS = """You are a strict but fair grader for a multi-hop QA benchmark.
Decide whether the candidate answer is semantically equivalent to the gold answer.
Tolerate paraphrase, casing, and extra context, but reject answers that miss the
key entity, get the entity wrong, or are non-committal ("I don't know").

Reply with JSON only: {"correct": true|false, "reasoning": "<one sentence>"}"""

JUDGE_USER_TEMPLATE = """Question: {question}
Gold answer: {gold}
Candidate answer: {answer}"""


def judge(question: str, gold: str, answer: str) -> tuple[bool, str, float]:
    response = call_responses(
        instructions=JUDGE_INSTRUCTIONS,
        input_data=JUDGE_USER_TEMPLATE.format(
            question=question, gold=gold, answer=answer or "(no answer)"
        ),
        max_output_tokens=200,
        model=JUDGE_MODEL,
    )
    text = _output_text(response)
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
        correct = bool(parsed.get("correct", False))
        reasoning = parsed.get("reasoning", "")
    except json.JSONDecodeError:
        correct = False
        reasoning = f"judge parse error: {text[:200]}"
    in_tok, out_tok, cached = _usage(response)
    cost = _model_cost_usd(JUDGE_MODEL, in_tok, out_tok, cached)
    return correct, reasoning, cost


def evaluate(
    agent_fn: Callable[[str], AgentResult],
    questions: list[Question],
    label: str = "eval",
    verbose: bool = True,
) -> dict:
    """Run agent_fn on each question, judge, return aggregate metrics + per-q records."""
    records: list[JudgedResult] = []
    total_cost = 0.0
    t0 = time.time()

    for i, q in enumerate(questions):
        if verbose:
            print(f"[{label}] {i + 1}/{len(questions)}: {q.question[:80]}")

        q_t0 = time.time()
        try:
            result = agent_fn(q.question)
        except Exception as e:
            result = AgentResult(answer="", error=str(e), latency_s=time.time() - q_t0)

        agent_cost = _model_cost_usd(
            WORKER_MODEL, result.input_tokens, result.output_tokens, result.cache_read_tokens
        )
        correct, reasoning, judge_cost = judge(q.question, q.gold_answer, result.answer)
        total_cost += agent_cost + judge_cost

        records.append(JudgedResult(
            qid=q.qid, question=q.question, gold=q.gold_answer, answer=result.answer,
            correct=correct, judge_reasoning=reasoning,
            cost_usd=agent_cost + judge_cost, latency_s=result.latency_s,
            steps=result.steps, trajectory=result.trajectory,
        ))
        if verbose:
            mark = "OK" if correct else "FAIL"
            print(f"  -> {mark}  answer={result.answer[:80]!r}  steps={result.steps}")

    accuracy = sum(r.correct for r in records) / max(len(records), 1)
    metrics = {
        "label": label,
        "accuracy": accuracy,
        "n": len(records),
        "total_cost_usd": total_cost,
        "avg_latency_s": sum(r.latency_s for r in records) / max(len(records), 1),
        "avg_steps": sum(r.steps for r in records) / max(len(records), 1),
        "wall_time_s": time.time() - t0,
    }
    return {"metrics": metrics, "records": [asdict(r) for r in records]}


def single_shot_agent(question: str) -> AgentResult:
    """Sanity baseline: just ask the model. No tools, no retrieval."""
    t0 = time.time()
    response = call_responses(
        instructions=(
            "You answer multi-hop factual questions concisely. "
            "If unsure, give your best guess in one short phrase."
        ),
        input_data=question,
        max_output_tokens=400,
    )
    text = _output_text(response)
    in_tok, out_tok, cached = _usage(response)
    return AgentResult(
        answer=text,
        trajectory=[{"role": "user", "content": question}, {"role": "assistant", "content": text}],
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_read_tokens=cached,
        steps=1,
        latency_s=time.time() - t0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", action="store_true", help="Run single-shot baseline")
    p.add_argument("--final", action="store_true", help="Run final test on held-out set")
    p.add_argument("--genome", type=str, default=None, help="Genome YAML to evaluate")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    questions = load_questions(n=args.n, seed=args.seed)

    if args.baseline:
        out = evaluate(single_shot_agent, questions, label="single_shot_baseline")
    elif args.final and args.genome:
        from agent import build_agent_fn
        from genome import Genome
        g = Genome.load(args.genome)
        out = evaluate(build_agent_fn(g), questions, label="final_test")
    else:
        p.print_help()
        return

    out_path = RUNS_DIR / f"{out['metrics']['label']}_{int(time.time())}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n=== {out['metrics']['label']} ===")
    for k, v in out["metrics"].items():
        print(f"  {k}: {v}")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
