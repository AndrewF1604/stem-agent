"""3-way comparison of reflection strategies on the validation set.

Compares:
  A) reflection off                              (genome.reflection_enabled = False)
  B) reflection on, unconditional                (enabled=True, threshold=1.0)
  C) reflection on, conditional at threshold=0.5 (enabled=True, threshold=0.5)

Same base genome (G0 from main v2 lineage), same val questions, 3 trials each.

This is the §4 follow-up: the writeup observed unconditional reflection harms
fitness. The hypothesis here: a confidence-gated trigger fires only on uncertain
first answers, capturing the upside without the downside.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import build_agent_fn
from eval import RUNS_DIR, evaluate_multi_trial, load_splits
from genome import Genome


def make_variant(base: Genome, reflection_enabled: bool, threshold: float, notes: str) -> Genome:
    g = base.model_copy(deep=True)
    g.reflection_enabled = reflection_enabled
    g.reflection_threshold = threshold
    g.notes = notes
    return g


def main():
    g0_path = next(Path("runs").glob("main_v2_*/genome_g0_*.yaml"))
    base = Genome.load(g0_path)
    print(f"Base genome: {g0_path.name}")
    print(f"  arch={base.architecture}, max_steps={base.max_steps}, prompt_chars={len(base.system_prompt)}")

    splits = load_splits(seed=42, train_n=15, val_n=10, test_n=10)
    val_q = splits["val"]
    print(f"Eval set: val (n={len(val_q)})")

    variants = [
        ("A_off",            make_variant(base, False, 0.0, "reflection off")),
        ("B_unconditional",  make_variant(base, True, 1.0, "unconditional reflection")),
        ("C_conditional_50", make_variant(base, True, 0.5, "conditional at threshold=0.5")),
        ("D_conditional_95", make_variant(base, True, 0.95, "conditional at threshold=0.95")),
    ]

    out_dir = RUNS_DIR / f"conditional_reflection_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for label, g in variants:
        print(f"\n=== {label} ===")
        result = evaluate_multi_trial(
            build_agent_fn(g), val_q,
            label=label, n_trials=3, verbose=False,
        )
        m = result["metrics"]
        summary[label] = {
            "accuracy_mean": m["accuracy_mean"],
            "accuracy_std": m["accuracy_std"],
            "accuracy_per_trial": m["accuracy_per_trial"],
            "total_cost_usd": m["total_cost_usd"],
        }
        print(f"  mean={m['accuracy_mean']:.3f}  std={m['accuracy_std']:.3f}  trials={m['accuracy_per_trial']}")
        (out_dir / f"{label}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== Summary saved to {out_dir / 'summary.json'} ===")
    for label, s in summary.items():
        print(f"  {label}: {s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f}")


if __name__ == "__main__":
    main()
