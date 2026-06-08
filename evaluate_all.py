"""Ewaluacja sweepu z iteracjami.

Bierze runs/<condition>/<qid>.json (każdy z listą `samples`), liczy EM/F1/Judge
per próbka i agreguje per pytanie. Generuje dwa CSV:

  analysis/raw_long.csv      — 1 wiersz per (qid, config, iter); baza pod wszystko
  analysis/per_question.csv  — 1 wiersz per (qid, config) z metrykami rozkładu
                               (pass@1/@k, majority vote, agreement, entropia,
                                wrong_consistency) — serce analizy self-consistency

Optymalizacja: Judge (wywołanie LLM) jest dedupowany w obrębie (qid, config) po
treści odpowiedzi — przy temp 0.7 model często powtarza tę samą odpowiedź, więc
liczba realnych wywołań sędziego jest dużo niższa niż liczba próbek. EM/F1 są
darmowe i też cache'owane.

Wstecznie zgodne: pliki w starym schemacie (pojedyncze `answer` bez `samples`)
są traktowane jak jedna próbka.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from consistency import majority, pass_at_1, pass_at_k, shannon_entropy, wrong_consistency
from judge import judge
from metrics import evaluate_pair, normalize_answer


RUNS_DIR = Path("runs")
OUT_DIR = Path("analysis")


def load_groups() -> list[dict]:
    """Każdy plik run → grupa {meta..., samples: [...]}. Obsługuje stary i nowy schemat."""
    groups = []
    for cond_dir in sorted(RUNS_DIR.iterdir()):
        if not cond_dir.is_dir():
            continue
        for f in sorted(cond_dir.glob("*.json")):
            if f.name.startswith("_"):
                continue
            data = json.loads(f.read_text(encoding="utf-8"))
            if "samples" in data:
                samples = data["samples"]
            else:
                samples = [{
                    "iter": 0, "seed": data.get("seed"),
                    "answer": data.get("answer", ""), "error": data.get("error"),
                    "latency_s": data.get("latency_s"), "output_tokens": data.get("output_tokens"),
                }]
            groups.append({
                "qid": data["qid"], "config": data["config"],
                "q_type": data["q_type"], "level": data["level"],
                "temperature": data.get("temperature"), "prompt_template": data.get("prompt_template"),
                "gold_answer": data["gold_answer"], "question": data["question"],
                "samples": samples,
            })
    return groups


def score_group(g: dict, judge_calls: list[int]) -> tuple[list[dict], dict]:
    """Liczy per-próbka EM/F1/Judge (z cache) i agregaty per pytanie."""
    gold = g["gold_answer"]
    question = g["question"]
    metric_cache: dict[str, dict] = {}
    judge_cache: dict[str, tuple[int, str]] = {}

    long_rows = []
    norms_ok: list[str] = []
    em_ok: list[int] = []
    judge_ok: list[int] = []

    for smp in g["samples"]:
        ans = (smp.get("answer") or "").strip()
        err = smp.get("error")
        if err or not ans:
            long_rows.append({**_base(g, smp), "answer": ans, "pred_normalized": "",
                              "em": 0, "f1": 0.0, "judge": 0,
                              "judge_reason": "inference error" if err else "empty answer",
                              "error": err or ""})
            continue

        if ans not in metric_cache:
            metric_cache[ans] = evaluate_pair(ans, gold)
        m = metric_cache[ans]
        if ans not in judge_cache:
            jr = judge(question, gold, ans)
            judge_cache[ans] = (int(jr.correct), jr.reasoning[:200])
            judge_calls[0] += 1
        jflag, jreason = judge_cache[ans]

        long_rows.append({**_base(g, smp), "answer": ans, "pred_normalized": m["pred_normalized"],
                          "em": m["em"], "f1": round(m["f1"], 3), "judge": jflag,
                          "judge_reason": jreason, "error": ""})
        norms_ok.append(m["pred_normalized"])
        em_ok.append(m["em"])
        judge_ok.append(jflag)

    agg = _aggregate(g, norms_ok, em_ok, judge_ok, n_total=len(g["samples"]))
    return long_rows, agg


def _base(g: dict, smp: dict) -> dict:
    return {
        "qid": g["qid"], "config": g["config"], "q_type": g["q_type"], "level": g["level"],
        "temperature": g["temperature"], "prompt_template": g["prompt_template"],
        "iter": smp.get("iter"), "seed": smp.get("seed"), "gold_answer": g["gold_answer"],
    }


def _aggregate(g: dict, norms: list[str], em: list[int], judge_flags: list[int], n_total: int) -> dict:
    n = len(norms)
    gold_norm = normalize_answer(g["gold_answer"])
    maj_answer, agreement, n_distinct = majority(norms)

    if n == 0:
        maj_em = maj_judge = 0
    else:
        maj_em = int(maj_answer == gold_norm)
        maj_judge_votes = [jf for nm, jf in zip(norms, judge_flags) if nm == maj_answer]
        maj_judge = int(sum(maj_judge_votes) / len(maj_judge_votes) >= 0.5) if maj_judge_votes else 0

    n_em = sum(em)
    n_judge = sum(judge_flags)
    return {
        "qid": g["qid"], "config": g["config"], "q_type": g["q_type"], "level": g["level"],
        "temperature": g["temperature"], "prompt_template": g["prompt_template"],
        "gold_answer": g["gold_answer"],
        "n_samples": n, "n_error": n_total - n,
        "n_em_correct": n_em, "em_pass1": round(pass_at_1(n_em, n), 3),
        "em_passany": int(n_em > 0),
        "n_judge_correct": n_judge, "judge_pass1": round(pass_at_1(n_judge, n), 3),
        "judge_passany": int(n_judge > 0),
        "majority_answer": maj_answer, "majority_em": maj_em, "majority_judge": maj_judge,
        "agreement": round(agreement, 3), "answer_entropy": round(shannon_entropy(norms), 3),
        "n_distinct": n_distinct,
        "wrong_consistency": _round_opt(wrong_consistency(norms, judge_flags)),
    }


def _round_opt(x):
    return round(x, 3) if x is not None else ""


def main():
    OUT_DIR.mkdir(exist_ok=True)
    groups = load_groups()
    n_samples_total = sum(len(g["samples"]) for g in groups)
    print(f"Found {len(groups)} (qid, config) groups, {n_samples_total} próbek łącznie.")

    long_fields = [
        "qid", "config", "q_type", "level", "temperature", "prompt_template",
        "iter", "seed", "gold_answer", "answer", "pred_normalized",
        "em", "f1", "judge", "judge_reason", "error",
    ]
    pq_fields = [
        "qid", "config", "q_type", "level", "temperature", "prompt_template", "gold_answer",
        "n_samples", "n_error", "n_em_correct", "em_pass1", "em_passany",
        "n_judge_correct", "judge_pass1", "judge_passany",
        "majority_answer", "majority_em", "majority_judge",
        "agreement", "answer_entropy", "n_distinct", "wrong_consistency",
    ]

    long_path = OUT_DIR / "raw_long.csv"
    pq_path = OUT_DIR / "per_question.csv"
    judge_calls = [0]

    with long_path.open("w", encoding="utf-8", newline="") as lf, \
         pq_path.open("w", encoding="utf-8", newline="") as pf:
        lw = csv.DictWriter(lf, fieldnames=long_fields)
        pw = csv.DictWriter(pf, fieldnames=pq_fields)
        lw.writeheader()
        pw.writeheader()

        for i, g in enumerate(groups):
            long_rows, agg = score_group(g, judge_calls)
            lw.writerows(long_rows)
            pw.writerow(agg)
            if (i + 1) % 50 == 0 or (i + 1) == len(groups):
                print(f"  scored {i + 1}/{len(groups)} grup, {judge_calls[0]} wywołań Judge")

    saved = judge_calls[0]
    print(f"\nJudge: {saved} realnych wywołań na {n_samples_total} próbek "
          f"(dedup oszczędził {n_samples_total - saved}).")
    print(f"Saved {long_path}")
    print(f"Saved {pq_path}")


if __name__ == "__main__":
    main()
