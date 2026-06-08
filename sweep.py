"""Pełny sweep w designie repeated-measures (paired).

Dla każdej condition i każdego pytania pobiera K próbek (iteracji). Conditions
deterministyczne (temp=0, tryb context) dostają K=1 — przy greedy decoding kolejne
iteracje byłyby identyczne. Stochastyczne (temp>0) oraz agent (zmienność DDG)
dostają K z InferenceConfig.iters.

Output: runs/<condition>/<qid>.json — jeden plik per (condition, pytanie),
zawiera listę `samples` (po jednej na iterację). Każda iteracja używa stałego
seeda (ITER_SEED_BASE + iter), więc cały bieg jest reprodukowalny.

Resume: liczone są udane próbki (error == None). Iteracje które się wywaliły
(np. Ollama padła) są przy wznowieniu powtarzane. Guard na hash zestawu pytań
chroni przed niezauważonym pomieszaniem dwóch różnych zestawów w jednym biegu.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dataset import QuestionSample, question_set_hash
from inference import CONDITIONS, InferenceConfig, infer


SAMPLES_PATH = Path("data/samples_100.json")
RUNS_DIR = Path("runs")
ITER_SEED_BASE = 1000


def load_samples(path: Path) -> list[QuestionSample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [QuestionSample(**r) for r in raw]


def effective_iters(config: InferenceConfig, override: int | None) -> int:
    """K=1 dla deterministycznych (greedy) conditions, inaczej iters z configu/override."""
    if config.temperature == 0.0 and config.mode == "context":
        return 1
    return override if override is not None else config.iters


def _load_done_samples(path: Path) -> list[dict]:
    """Wczytuje udane próbki z istniejącego pliku run (błędne pomija → zostaną powtórzone)."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [s for s in data.get("samples", []) if not s.get("error")]


def run_condition(
    config: InferenceConfig,
    samples: list[QuestionSample],
    set_hash: str,
    iters_override: int | None = None,
    resume: bool = True,
):
    cond_dir = RUNS_DIR / config.name
    cond_dir.mkdir(parents=True, exist_ok=True)
    progress_path = cond_dir / "_progress.json"
    k = effective_iters(config, iters_override)

    if resume and progress_path.exists():
        prev_hash = json.loads(progress_path.read_text(encoding="utf-8")).get("question_set_hash")
        if prev_hash and prev_hash != set_hash:
            sys.exit(
                f"\n[{config.name}] ABORT: question-set hash mismatch.\n"
                f"  progress trzyma: {prev_hash}\n"
                f"  obecny zestaw:   {set_hash}\n"
                f"Odmawiam mieszania dwóch zestawów pytań w jednym biegu. "
                f"Użyj --no-resume żeby zacząć od zera, albo przywróć właściwy plik samples."
            )

    todo = []
    done_iters = 0
    for s in samples:
        existing = _load_done_samples(cond_dir / f"{s.qid}.json") if resume else []
        done_iters += min(len(existing), k)
        if len(existing) < k:
            todo.append((s, existing))

    total_iters = len(samples) * k
    print(f"\n=== {config.name}  (K={k}/pytanie, {done_iters}/{total_iters} iteracji gotowych, {len(todo)} pytań do dokończenia) ===")
    if not todo:
        print("  nothing to do")
        return

    t0 = time.time()
    new_iters = 0
    for qi, (s, existing) in enumerate(todo):
        out_path = cond_dir / f"{s.qid}.json"
        sample_recs = list(existing)
        for it in range(len(sample_recs), k):
            seed = ITER_SEED_BASE + it
            result = infer(s.question, s.context, config, seed=seed)
            sample_recs.append({
                "iter": it,
                "seed": seed,
                "answer": result.answer,
                "raw_output": result.raw_output,
                "latency_s": round(result.latency_s, 2),
                "output_tokens": result.output_tokens,
                "error": result.error,
                "agent_trajectory": result.agent_trajectory,
            })
            rec = {
                "qid": s.qid,
                "config": config.name,
                "model": config.model,
                "temperature": config.temperature,
                "prompt_template": config.prompt_template,
                "mode": config.mode,
                "question": s.question,
                "gold_answer": s.gold_answer,
                "q_type": s.q_type,
                "level": s.level,
                "n_iters_target": k,
                "samples": sample_recs,
            }
            out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
            new_iters += 1

        progress_path.write_text(json.dumps({
            "question_set_hash": set_hash,
            "k_target": k,
            "n_questions": len(samples),
        }), encoding="utf-8")

        if (qi + 1) % 10 == 0 or (qi + 1) == len(todo):
            elapsed = time.time() - t0
            rate = new_iters / elapsed if elapsed > 0 else 0
            remaining = sum(k - len(e) for _, e in todo[qi + 1:])
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{qi + 1}/{len(todo)} pytań, +{new_iters} iteracji] elapsed {elapsed:.0f}s, {rate:.2f} it/s, ETA {eta:.0f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", type=str, default=None, help="Tylko ta condition (po nazwie)")
    p.add_argument("--limit", type=int, default=None, help="Tylko pierwsze N pytań (do testów)")
    p.add_argument("--iters", type=int, default=None, help="Nadpisz K dla conditions stochastycznych")
    p.add_argument("--samples", type=str, default=None, help="Ścieżka do pliku z pytaniami (domyślnie data/samples_100.json)")
    p.add_argument("--no-resume", action="store_true", help="Od zera (ignoruje istniejące próbki)")
    args = p.parse_args()

    samples_path = Path(args.samples) if args.samples else SAMPLES_PATH
    samples = load_samples(samples_path)
    set_hash = question_set_hash(samples)
    print(f"Zestaw: {samples_path}  ({len(samples)} pytań, hash {set_hash})")

    if args.limit:
        samples = samples[: args.limit]
        set_hash = question_set_hash(samples)
        print(f"  --limit {args.limit} → podzbiór, hash {set_hash}")

    if args.condition:
        configs = [c for c in CONDITIONS if c.name == args.condition]
        if not configs:
            print(f"Unknown condition: {args.condition}")
            print(f"Available: {[c.name for c in CONDITIONS]}")
            sys.exit(1)
    else:
        configs = CONDITIONS

    for c in configs:
        run_condition(c, samples, set_hash, iters_override=args.iters, resume=not args.no_resume)


if __name__ == "__main__":
    main()
