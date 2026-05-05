"""Generate fitness curves and mutation breakdown charts from run logs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from eval import RUNS_DIR


def _load_history(run_dir: Path) -> list[dict]:
    p = run_dir / "history.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def fitness_curve(run_dirs: list[Path], out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = plt.cm.tab10.colors
    for idx, rd in enumerate(run_dirs):
        history = _load_history(rd)
        if not history:
            continue
        gens = [0]
        agent_acc = [history[0]["accuracy_before"]]
        for h in history:
            gens.append(h["gen"])
            agent_acc.append(h["accuracy_after"] if h["accepted"] else agent_acc[-1])

        label = rd.name.split("_")[0] if "_" in rd.name else rd.name
        color = colors[idx % len(colors)]
        ax.plot(gens, agent_acc, marker="o", label=f"{label} agent state",
                color=color, linewidth=2)

        for h in history:
            marker = "o" if h["accepted"] else "x"
            mcolor = color if h["accepted"] else "red"
            mz = 8 if h["accepted"] else 11
            ax.plot(h["gen"], h["accuracy_after"], marker=marker,
                    color=mcolor, markersize=mz, markeredgewidth=2,
                    linestyle="None", alpha=0.9)

        canary_pts = [(h["gen"], h.get("canary_acc")) for h in history if h.get("canary_acc") is not None]
        if canary_pts:
            cgens, cvals = zip(*canary_pts)
            ax.plot(cgens, cvals, marker="s", linestyle="--",
                    label=f"{label} canary acc", color=color, alpha=0.5)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fitness across generations\n(solid line = agent state; red × = rejected candidate; ■ = canary check)")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def mutation_breakdown(run_dirs: list[Path], out_path: Path):
    """Bar chart: per mutation kind, how often was it best vs accepted vs harmful."""
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"trials": 0, "improved": 0, "tied": 0, "harmful": 0, "accepted": 0}
    )
    for rd in run_dirs:
        history = _load_history(rd)
        for h in history:
            parent_acc = h["accuracy_before"]
            chosen_hash = h["chosen_hash"]
            for m in h["mutations_tried"]:
                stats[m["kind"]]["trials"] += 1
                if m["accuracy"] > parent_acc:
                    stats[m["kind"]]["improved"] += 1
                elif m["accuracy"] == parent_acc:
                    stats[m["kind"]]["tied"] += 1
                else:
                    stats[m["kind"]]["harmful"] += 1
                if h["accepted"] and m["hash"] == chosen_hash:
                    stats[m["kind"]]["accepted"] += 1

    if not stats:
        print("no mutation data")
        return

    kinds = list(stats.keys())
    improved = [stats[k]["improved"] for k in kinds]
    tied = [stats[k]["tied"] for k in kinds]
    harmful = [stats[k]["harmful"] for k in kinds]
    accepted = [stats[k]["accepted"] for k in kinds]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(kinds))
    ax.bar(x, improved, label="improved fitness", color="#2ca02c")
    ax.bar(x, tied, bottom=improved, label="tied", color="#bbbbbb")
    ax.bar(x, harmful, bottom=[a + b for a, b in zip(improved, tied)],
           label="hurt fitness", color="#d62728")
    for i, k in enumerate(kinds):
        ax.text(i, stats[k]["trials"] + 0.1,
                f"accepted: {accepted[i]}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(kinds)
    ax.set_ylabel("Number of mutation trials")
    ax.set_title("Mutation outcomes across all generations")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=None,
                   help="Specific run dirs; default: main_* and ablation_*")
    p.add_argument("--out-dir", default="runs")
    args = p.parse_args()

    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
    else:
        run_dirs = sorted(
            d for d in RUNS_DIR.iterdir()
            if d.is_dir() and (d.name.startswith("main_") or d.name.startswith("ablation_"))
        )

    if not run_dirs:
        print("no run dirs found")
        return
    print(f"using runs: {[d.name for d in run_dirs]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fitness_curve(run_dirs, out_dir / "fitness_curve.png")
    mutation_breakdown(run_dirs, out_dir / "mutation_breakdown.png")


if __name__ == "__main__":
    main()
