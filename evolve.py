"""Evolution loop: the stem agent's differentiation process.

Per generation:
  1. Run current phenotype on the val batch → fitness.
  2. Run current phenotype on the train batch → failure trajectories for proposer.
  3. If val accuracy is high enough, stop.
  4. Proposer reads train failures and proposes 3 mutations.
  5. Evaluate each mutation on val → accept/reject by fitness + canary.
  6. Snapshot genome, log, repeat.

Train and val are disjoint by construction (see eval.load_splits). Test is
never touched by this loop — only by eval.py --final.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent import build_agent_fn
from eval import RUNS_DIR, evaluate, evaluate_multi_trial, load_splits, split_class_balance
from genome import Genome, initial_population
from mutations import Mutation, propose_mutations

PLATEAU_PATIENCE = 3
ACCURACY_TARGET = 0.85
EVAL_TRIALS = 1


@dataclass
class GenerationLog:
    gen: int
    parent_hash: str
    chosen_hash: str
    accuracy_before: float
    accuracy_after: float
    train_acc_before: float | None
    diagnosis: str
    mutations_tried: list[dict] = field(default_factory=list)
    accepted: bool = False
    rollback_reason: str | None = None
    cost_usd: float = 0.0
    notes: str = ""
    canary_acc: float | None = None


def _eval_genome(genome: Genome, questions, label: str) -> dict:
    if EVAL_TRIALS > 1:
        agg = evaluate_multi_trial(
            build_agent_fn(genome), questions,
            label=label, n_trials=EVAL_TRIALS, verbose=False,
        )
        first = agg["trials"][0]
        records = first["records"]
        total_cost = sum(t["metrics"]["total_cost_usd"] for t in agg["trials"])
        return {
            "metrics": {
                "label": label,
                "accuracy": agg["metrics"]["accuracy_mean"],
                "accuracy_std": agg["metrics"]["accuracy_std"],
                "accuracy_per_trial": agg["metrics"]["accuracy_per_trial"],
                "n": len(records),
                "total_cost_usd": total_cost,
                "n_trials": EVAL_TRIALS,
            },
            "records": records,
        }
    return evaluate(build_agent_fn(genome), questions, label=label, verbose=False)


def _select_initial(seeds: list[Genome], val_questions, run_dir: Path) -> tuple[Genome, dict]:
    """Phase A — score each seed on val, pick best as G0. Val is used for selection
    so the choice is comparable to subsequent generation fitness."""
    print(f"\n=== Phase A: scoring {len(seeds)} seed genomes on val ===")
    best_genome, best_result = None, None
    for i, g in enumerate(seeds):
        print(f"  seed {i + 1}/{len(seeds)}: {g.notes}")
        result = _eval_genome(g, val_questions, label=f"seed_{i}")
        acc = result["metrics"]["accuracy"]
        print(f"    -> val accuracy={acc:.2f}, cost=${result['metrics']['total_cost_usd']:.3f}")
        (run_dir / f"seed_{i}_g{g.hash()}.json").write_text(
            json.dumps({"genome": g.model_dump(), **result}, indent=2), encoding="utf-8"
        )
        if best_result is None or acc > best_result["metrics"]["accuracy"]:
            best_genome, best_result = g, result
    return best_genome, best_result


def evolve(
    generations: int,
    no_guard: bool,
    tag: str,
    seed: int,
    train_n: int,
    val_n: int,
) -> Path:
    run_dir = RUNS_DIR / f"{tag}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    splits = load_splits(seed=seed, train_n=train_n, val_n=val_n, test_n=10)
    train_q = splits["train"]
    val_q = splits["val"]
    print(f"\n=== Splits (guaranteed disjoint) ===")
    for name, b in split_class_balance(splits).items():
        print(f"  {name}: total={b['total']}  comparison={b['comparison']}  bridge={b['bridge']}")
    (run_dir / "splits_metadata.json").write_text(json.dumps({
        "seed": seed,
        "balance": split_class_balance(splits),
        "train_qids": [q.qid for q in train_q],
        "val_qids": [q.qid for q in val_q],
        "test_qids": [q.qid for q in splits["test"]],
    }, indent=2), encoding="utf-8")

    seeds = initial_population()
    current, current_result = _select_initial(seeds, val_q, run_dir)
    current.save(run_dir / f"genome_g0_{current.hash()}.yaml")

    canary_questions = [
        q for q, rec in zip(val_q, current_result["records"]) if rec["correct"]
    ]
    print(f"\nCanary set size (val questions G0 solves): {len(canary_questions)}")

    history: list[GenerationLog] = []
    plateau = 0

    for gen in range(1, generations + 1):
        print(f"\n=== Generation {gen} ===")
        if current_result["metrics"]["accuracy"] >= ACCURACY_TARGET:
            print(f"Stop: val accuracy {current_result['metrics']['accuracy']:.2f} >= target")
            break

        train_result = _eval_genome(current, train_q, label=f"g{gen}_train_failures")
        train_acc = train_result["metrics"]["accuracy"]
        failed = [r for r in train_result["records"] if not r["correct"]]
        print(f"  train acc of parent: {train_acc:.2f} ({len(failed)} failures for proposer)")

        if not failed:
            print("Stop: no failures on train to learn from")
            break

        try:
            diagnosis, mutations = propose_mutations(current, failed)
        except Exception as e:
            print(f"Proposer error: {e}")
            break
        print(f"Diagnosis: {diagnosis}")

        candidates: list[tuple[Mutation, Genome, dict]] = []
        for m in mutations:
            try:
                child = m.apply(current)
            except Exception as e:
                print(f"  skip {m.kind}: apply error: {e}")
                continue
            print(f"  trying {m.kind}: {m.rationale[:100]}")
            child_result = _eval_genome(child, val_q, label=f"g{gen}_{m.kind}")
            print(f"    -> val accuracy={child_result['metrics']['accuracy']:.2f}")
            candidates.append((m, child, child_result))

        if not candidates:
            print("No valid candidates this generation")
            history.append(GenerationLog(
                gen=gen, parent_hash=current.hash(), chosen_hash=current.hash(),
                accuracy_before=current_result["metrics"]["accuracy"],
                accuracy_after=current_result["metrics"]["accuracy"],
                train_acc_before=train_acc,
                diagnosis=diagnosis, mutations_tried=[], accepted=False,
                rollback_reason="no candidates",
            ))
            plateau += 1
            if plateau >= PLATEAU_PATIENCE:
                print(f"Stop: plateau ({plateau} generations)")
                break
            continue

        candidates.sort(
            key=lambda t: (-t[2]["metrics"]["accuracy"], t[2]["metrics"]["total_cost_usd"])
        )
        best_m, best_child, best_child_result = candidates[0]
        best_acc = best_child_result["metrics"]["accuracy"]
        current_acc = current_result["metrics"]["accuracy"]

        log = GenerationLog(
            gen=gen, parent_hash=current.hash(), chosen_hash=best_child.hash(),
            accuracy_before=current_acc, accuracy_after=best_acc,
            train_acc_before=train_acc,
            diagnosis=diagnosis,
            mutations_tried=[
                {"kind": m.kind, "rationale": m.rationale,
                 "accuracy": r["metrics"]["accuracy"], "hash": c.hash()}
                for m, c, r in candidates
            ],
            cost_usd=sum(r["metrics"]["total_cost_usd"] for _, _, r in candidates)
                     + train_result["metrics"]["total_cost_usd"],
        )

        if best_acc <= current_acc:
            log.accepted = False
            log.rollback_reason = f"no improvement (best={best_acc:.2f} <= current={current_acc:.2f})"
            print(f"  REJECT (no improvement)")
            plateau += 1
        else:
            regression = False
            if canary_questions:
                canary_result = _eval_genome(best_child, canary_questions, label=f"g{gen}_canary")
                canary_acc = canary_result["metrics"]["accuracy"]
                log.canary_acc = canary_acc
                guard_tag = "active" if not no_guard else "telemetry-only"
                print(f"  canary check ({guard_tag}): {canary_acc:.2f} on {len(canary_questions)} previously-solved")
                if not no_guard and canary_acc < 1.0 - 1e-6:
                    regression = True
                    log.rollback_reason = f"canary regression: {canary_acc:.2f} < 1.0"

            if regression:
                log.accepted = False
                print(f"  REJECT ({log.rollback_reason})")
                plateau += 1
            else:
                log.accepted = True
                current = best_child
                current_result = best_child_result
                current.save(run_dir / f"genome_g{gen}_{current.hash()}.yaml")
                print(f"  ACCEPT  val acc {current_acc:.2f} -> {best_acc:.2f}")
                plateau = 0

        history.append(log)
        (run_dir / "history.json").write_text(
            json.dumps([h.__dict__ for h in history], indent=2), encoding="utf-8"
        )

        if plateau >= PLATEAU_PATIENCE:
            print(f"Stop: plateau ({plateau} generations)")
            break

    current.save(run_dir / "best_genome.yaml")
    summary = {
        "generations_run": len(history),
        "final_val_accuracy": current_result["metrics"]["accuracy"],
        "final_genome_hash": current.hash(),
        "no_guard": no_guard,
        "tag": tag,
        "train_n": train_n,
        "val_n": val_n,
        "seed": seed,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== Done ===\n{json.dumps(summary, indent=2)}")
    return run_dir


def main():
    global EVAL_TRIALS
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--train-n", type=int, default=15, help="Training-set size (proposer reads these failures)")
    p.add_argument("--val-n", type=int, default=10, help="Validation-set size (fitness eval)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-guard", action="store_true", help="Ablation: disable regression guard")
    p.add_argument("--trials", type=int, default=1, help="Trials per genome eval (multi-trial averaging in the loop)")
    p.add_argument("--tag", type=str, default="evolve")
    args = p.parse_args()
    EVAL_TRIALS = args.trials
    evolve(args.generations, args.no_guard, args.tag, args.seed, args.train_n, args.val_n)


if __name__ == "__main__":
    main()
