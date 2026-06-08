"""Stratified sampling z HotpotQA distractor split.

Generuje zamrożony zestaw pytań (domyślnie data/samples_100.json):
- 25 easy + comparison
- 25 easy + bridge
- 25 hard + comparison
- 25 hard + bridge
= 100 pytań łącznie

Zestaw jest "zamrożony": generowany raz, ze stałym seedem, i używany identycznie
w każdej condition i każdej iteracji (repeated-measures / paired design). Obok
pliku z pytaniami powstaje manifest z hashem zestawu (data/<nazwa>.manifest.json) —
sweep.py używa go jako bezpiecznika, żeby nie pomieszać dwóch różnych zestawów pytań.

Każde pytanie zawiera context (10 paragrafów Wikipedii w trybie distractor):
2 wymagane paragrafy + 8 rozpraszaczy. Model dostaje wszystkie w prompcie.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path

from datasets import load_dataset


@dataclass
class QuestionSample:
    qid: str
    question: str
    gold_answer: str
    q_type: str
    level: str
    context: list[dict] = field(default_factory=list)


def question_set_hash(samples) -> str:
    """Stabilny hash zestawu pytań (po posortowanych qid).

    Niezależny od kolejności w pliku. Przyjmuje listę QuestionSample albo listę qid.
    sweep.py porównuje ten hash przy wznawianiu, żeby wykryć podmianę zestawu.
    """
    qids = [s.qid if isinstance(s, QuestionSample) else str(s) for s in samples]
    joined = "\n".join(sorted(qids))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def format_context_for_prompt(context: list[dict]) -> str:
    """Renderuje listę paragrafów na czytelny tekst do promptu."""
    blocks = []
    for para in context:
        title = para["title"]
        text = " ".join(para["sentences"])
        blocks.append(f"[{title}]\n{text}")
    return "\n\n".join(blocks)


def stratified_sample(
    n_per_bucket: int = 75,
    seed: int = 42,
    levels: tuple[str, ...] = ("easy", "hard"),
    types: tuple[str, ...] = ("comparison", "bridge"),
    split: str = "train",
) -> list[QuestionSample]:
    """Sample n_per_bucket questions for each (level, type) combination.

    Validation split contains only 'hard' questions. Train split has full
    easy/medium/hard distribution. We use train and rely on context-provided
    reading comprehension setup (model gets the passages explicitly, so
    pretraining contamination is bounded — task is reasoning over given text).
    """
    print(f"Loading HotpotQA distractor split ({split})...")
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    print(f"Total questions in split: {len(ds)}")

    by_bucket: dict[tuple[str, str], list[int]] = {(l, t): [] for l in levels for t in types}
    for i, row in enumerate(ds):
        key = (row["level"], row["type"])
        if key in by_bucket:
            by_bucket[key].append(i)

    print("\nBucket sizes in source data:")
    for k, v in by_bucket.items():
        print(f"  {k}: {len(v)}")

    rng = random.Random(seed)
    samples: list[QuestionSample] = []
    for (level, qtype), indices in by_bucket.items():
        rng.shuffle(indices)
        if len(indices) < n_per_bucket:
            raise ValueError(
                f"Not enough questions in bucket {level}/{qtype}: "
                f"have {len(indices)}, need {n_per_bucket}"
            )
        for i in indices[:n_per_bucket]:
            row = ds[i]
            context = [
                {"title": title, "sentences": list(sents)}
                for title, sents in zip(row["context"]["title"], row["context"]["sentences"])
            ]
            samples.append(QuestionSample(
                qid=row["id"],
                question=row["question"],
                gold_answer=row["answer"],
                q_type=row["type"],
                level=row["level"],
                context=context,
            ))
    return samples


def main():
    p = argparse.ArgumentParser(description="Generuj zamrożony stratyfikowany zestaw pytań z HotpotQA")
    p.add_argument("--n-per-bucket", type=int, default=25, help="Pytań na każdy z 4 bucketów (domyślnie 25 → 100 łącznie)")
    p.add_argument("--seed", type=int, default=42, help="Seed losowania (stały dla reprodukowalności zestawu)")
    p.add_argument("--out", type=str, default=None, help="Ścieżka wyjściowa (domyślnie data/samples_<N>.json)")
    p.add_argument("--split", type=str, default="train", help="Split HotpotQA distractor")
    args = p.parse_args()

    total = args.n_per_bucket * 4
    out_path = Path(args.out) if args.out else Path("data") / f"samples_{total}.json"
    out_path.parent.mkdir(exist_ok=True)

    samples = stratified_sample(n_per_bucket=args.n_per_bucket, seed=args.seed, split=args.split)

    out = [asdict(s) for s in samples]
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    set_hash = question_set_hash(samples)
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "question_set_hash": set_hash,
        "n_questions": len(samples),
        "n_per_bucket": args.n_per_bucket,
        "seed": args.seed,
        "split": args.split,
        "qids": [s.qid for s in samples],
    }, indent=2), encoding="utf-8")

    print(f"\nSaved {len(samples)} samples to {out_path}")
    print(f"Question-set hash: {set_hash}  (manifest: {manifest_path})")

    bucket_counts = Counter((s.level, s.q_type) for s in samples)
    print("\nFinal sample distribution:")
    for (level, qtype), count in sorted(bucket_counts.items()):
        print(f"  {level:6s} / {qtype:11s}: {count}")

    print("\nSample question:")
    s = samples[0]
    print(f"  qid: {s.qid}")
    print(f"  level: {s.level}, type: {s.q_type}")
    print(f"  question: {s.question}")
    print(f"  gold: {s.gold_answer}")
    print(f"  context: {len(s.context)} paragrafów, łącznie {sum(len(p['sentences']) for p in s.context)} zdań")
    print(f"  ctx[0] title: {s.context[0]['title']}")


if __name__ == "__main__":
    main()
