"""Three atomic mutation operators + the failure-analysis-driven proposer.

The proposer asks an LLM to read failed trajectories, classify the dominant
failure mode, and pick a mutation that addresses it. This is the piece that
makes the search *directed* rather than random.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from eval import _output_text, call_responses
from genome import Genome

MUTATION_KINDS = ["REFINE_PROMPT", "TOGGLE_REFLECTION", "ADJUST_MAX_STEPS"]


@dataclass
class Mutation:
    kind: str
    rationale: str
    payload: dict

    def apply(self, parent: Genome) -> Genome:
        child = parent.model_copy(deep=True)
        child.parent_hash = parent.hash()
        child.generation = parent.generation + 1

        if self.kind == "REFINE_PROMPT":
            child.system_prompt = self.payload["new_prompt"]
            child.notes = f"REFINE_PROMPT: {self.rationale[:120]}"
        elif self.kind == "TOGGLE_REFLECTION":
            child.reflection_enabled = not parent.reflection_enabled
            child.notes = f"TOGGLE_REFLECTION -> {child.reflection_enabled}: {self.rationale[:120]}"
        elif self.kind == "ADJUST_MAX_STEPS":
            delta = int(self.payload.get("delta", 0))
            child.max_steps = max(1, min(12, parent.max_steps + delta))
            child.notes = f"ADJUST_MAX_STEPS {parent.max_steps}->{child.max_steps}: {self.rationale[:120]}"
        else:
            raise ValueError(f"unknown mutation kind: {self.kind}")
        return child


PROPOSER_INSTRUCTIONS = """You are the mutation proposer for an evolving research agent.

You will be shown:
  - the current genome (system prompt, architecture, tools, max_steps, reflection flag)
  - a small set of failed trajectories from the validation batch

Your job: identify the single most common failure mode, then propose 3 distinct mutations
that plausibly address it. Mutations are atomic and chosen from this fixed set:

  - REFINE_PROMPT: rewrite the system_prompt. Use this when the failure is about *what* the
    agent does (over-confident answers without searching, premature stop, ignoring snippets,
    failing to compose multi-hop facts). Payload: {"new_prompt": "..."}.
    The new prompt may include the literal placeholder {max_steps} which gets formatted
    at runtime.
  - TOGGLE_REFLECTION: flip the reflection_enabled flag. Use to add a self-check step when
    answers seem hasty, or remove it when reflection is causing flip-flops.
    Payload: {}.
  - ADJUST_MAX_STEPS: change the step budget by +/-1, +/-2. Use when the agent runs out of
    steps mid-research, OR when it wastes steps on dead-end searches.
    Payload: {"delta": <int>}.

Reply with JSON only, schema:
{
  "diagnosis": "<one short sentence: the top failure mode>",
  "mutations": [
    {"kind": "REFINE_PROMPT" | "TOGGLE_REFLECTION" | "ADJUST_MAX_STEPS",
     "rationale": "<one sentence>", "payload": { ... }},
    ...
  ]
}

Rules:
  - Propose exactly 3 mutations.
  - Vary the kinds where it makes sense (don't return 3x REFINE_PROMPT unless prompt is
    clearly the only lever).
  - REFINE_PROMPT new_prompt MUST be a complete drop-in replacement, 2-5 sentences,
    no placeholders other than {max_steps}.
"""


def _format_failures(failed_records: list[dict], max_traj_chars: int = 1500) -> str:
    blocks = []
    for r in failed_records[:5]:
        traj = r.get("trajectory", [])
        traj_text = "\n".join(
            f"  [{t.get('role')}] {str(t.get('content', t))[:240]}"
            for t in traj[-8:]
        )
        if len(traj_text) > max_traj_chars:
            traj_text = traj_text[:max_traj_chars] + "\n  ...(truncated)"
        blocks.append(
            f"Q: {r['question']}\nGOLD: {r['gold']}\nGAVE: {r['answer']!r}\nWHY: {r['judge_reasoning']}\nTRAJ:\n{traj_text}"
        )
    return "\n\n---\n\n".join(blocks) if blocks else "(no failed records)"


def _parse_proposer(text: str) -> tuple[str, list[Mutation]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"proposer JSON parse error: {e}; got: {text[:300]}") from e

    diagnosis = parsed.get("diagnosis", "")
    mutations: list[Mutation] = []
    for m in parsed.get("mutations", []):
        kind = m.get("kind")
        if kind not in MUTATION_KINDS:
            continue
        mutations.append(Mutation(
            kind=kind,
            rationale=m.get("rationale", ""),
            payload=m.get("payload", {}) or {},
        ))
    return diagnosis, mutations


def propose_mutations(parent: Genome, failed_records: list[dict]) -> tuple[str, list[Mutation]]:
    """Ask the proposer LLM for 3 mutations targeting the dominant failure mode."""
    user = (
        f"CURRENT GENOME:\n{parent.model_dump_json(indent=2)}\n\n"
        f"FAILED TRAJECTORIES (n={len(failed_records)}):\n{_format_failures(failed_records)}"
    )
    response = call_responses(
        instructions=PROPOSER_INSTRUCTIONS,
        input_data=user,
        max_output_tokens=1500,
    )
    text = _output_text(response)
    return _parse_proposer(text)
