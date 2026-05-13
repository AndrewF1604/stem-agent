"""Stress test the regression guard with hand-crafted degenerate genomes.

The §3.5 result shows the guard fires on *organic* proposer mutations. That
leaves open the question: would the guard also catch deliberately bad mutations
that look superficially OK? This script answers it.

For each adversarial genome we measure two things:
  (1) Val accuracy — would it pass the fitness check (>= G0 val acc)?
  (2) Canary accuracy on G0-solved questions — would the guard catch it?

The cleanest demonstration of "guard does load-bearing work" is a mutation
that passes (1) but fails (2). Mutations that fail (1) outright are caught
by the fitness check; the guard is irrelevant for those.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import build_agent_fn
from eval import RUNS_DIR, evaluate, load_splits
from genome import Genome, DEFAULT_REACT_PROMPT


def build_adversarials(g0: Genome) -> list[tuple[str, Genome, str]]:
    """Return (label, genome, design_intent) triples."""
    base = g0.model_dump()

    def variant(**overrides) -> Genome:
        d = dict(base)
        d.update(overrides)
        d["generation"] = 99
        d["parent_hash"] = g0.hash()
        return Genome(**d)

    return [
        (
            "empty_prompt",
            variant(system_prompt="Answer.", notes="adversarial: minimal/empty prompt"),
            "Strip the prompt to a single word. Expect fitness to drop sharply.",
        ),
        (
            "lorem_ipsum",
            variant(
                system_prompt="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                notes="adversarial: nonsense prompt",
            ),
            "Replace the instruction with Latin filler. Expect total breakdown.",
        ),
        (
            "force_number",
            variant(
                system_prompt="Always answer with a single number between 1 and 10. Use {max_steps} searches if needed.",
                notes="adversarial: wrong answer type forced",
            ),
            "Force the model into the wrong output type for an entity-answer task.",
        ),
        (
            "refusal",
            variant(
                system_prompt="You must refuse to answer factual questions. Reply only: I cannot help with that. Do not use {max_steps} searches.",
                notes="adversarial: refusal prompt",
            ),
            "Make the agent refuse to answer. Expect zero correct.",
        ),
        (
            "anti_search",
            variant(
                system_prompt="Answer multi-hop questions from your internal knowledge only. Do not perform any searches, even though you have {max_steps} available.",
                notes="adversarial: tools available but forbidden",
            ),
            "Tools in the registry but prompt forbids them. Effectively single-shot.",
        ),
        (
            "max_steps_1",
            variant(max_steps=1, notes="adversarial: tightened step budget"),
            "Allow only one search step. Borderline — might still answer some questions.",
        ),
        (
            "unconditional_reflection",
            variant(
                reflection_enabled=True,
                reflection_threshold=1.0,
                notes="adversarial: reflection on every answer (known harmful from §3.6)",
            ),
            "Known from §3.6 to drop val ~14pp. Test whether guard catches it.",
        ),
        (
            "overconfidence_prompt",
            variant(
                system_prompt=g0.system_prompt + " Always be highly confident in your first guess and never qualify your answer with hedges like 'possibly' or 'I'm not sure'. Limit yourself to {max_steps} searches.",
                notes="adversarial: prompt that suppresses hedging",
            ),
            "Subtle: keeps real instructions but forces over-confidence. Might pass val by accident, likely to swap some canary answers.",
        ),
    ]


def main():
    g0_path = next(Path("runs").glob("main_v2_*/genome_g0_*.yaml"))
    g0 = Genome.load(g0_path)
    main_run_dir = g0_path.parent

    splits = load_splits(seed=42, train_n=15, val_n=10, test_n=10)
    val_q = splits["val"]
    print(f"G0: {g0_path.name} (max_steps={g0.max_steps})")
    print(f"Val: n={len(val_q)}")

    print("\nRe-evaluating G0 on val to establish baseline + canary set...")
    g0_result = evaluate(build_agent_fn(g0), val_q, label="g0_baseline", verbose=False)
    g0_val_acc = g0_result["metrics"]["accuracy"]
    canary_q = [val_q[i] for i, r in enumerate(g0_result["records"]) if r["correct"]]
    print(f"G0 val accuracy: {g0_val_acc:.2f}")
    print(f"Canary set size (questions G0 solves): {len(canary_q)}")
    if not canary_q:
        print("ERROR: G0 solved no questions in this trial; cannot run canary check.")
        return

    out_dir = RUNS_DIR / f"adversarial_guard_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    adversarials = build_adversarials(g0)
    rows = []

    for label, g, intent in adversarials:
        print(f"\n=== {label} ===")
        print(f"  intent: {intent}")
        val_result = evaluate(build_agent_fn(g), val_q, label=f"adv_{label}_val", verbose=False)
        val_acc = val_result["metrics"]["accuracy"]
        passes_fitness = val_acc > g0_val_acc

        canary_acc = None
        guard_catches = None
        if passes_fitness:
            canary_result = evaluate(build_agent_fn(g), canary_q, label=f"adv_{label}_canary", verbose=False)
            canary_acc = canary_result["metrics"]["accuracy"]
            guard_catches = canary_acc < 1.0 - 1e-9

        print(f"  val_acc={val_acc:.2f}  passes_fitness={passes_fitness}  "
              f"canary={canary_acc}  guard_catches={guard_catches}")

        rows.append({
            "label": label,
            "design_intent": intent,
            "val_acc": val_acc,
            "passes_fitness_check": bool(passes_fitness),
            "canary_acc": canary_acc,
            "guard_catches": guard_catches,
            "would_be_accepted_no_guard": bool(passes_fitness),
            "would_be_accepted_with_guard": bool(passes_fitness and not guard_catches),
        })

    summary = {
        "g0_val_acc": g0_val_acc,
        "g0_path": str(g0_path),
        "canary_set_size": len(canary_q),
        "n_adversarials": len(adversarials),
        "passed_fitness": sum(1 for r in rows if r["passes_fitness_check"]),
        "caught_by_guard": sum(1 for r in rows if r["guard_catches"]),
        "would_be_accepted_no_guard": sum(1 for r in rows if r["would_be_accepted_no_guard"]),
        "would_be_accepted_with_guard": sum(1 for r in rows if r["would_be_accepted_with_guard"]),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(f"  G0 val acc: {g0_val_acc:.2f}")
    print(f"  Passed fitness check: {summary['passed_fitness']} / {len(adversarials)}")
    print(f"  Of those, caught by canary: {summary['caught_by_guard']}")
    print(f"  Would be accepted with NO guard: {summary['would_be_accepted_no_guard']}")
    print(f"  Would be accepted WITH guard:    {summary['would_be_accepted_with_guard']}")
    print(f"  Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
