"""Exact Match i F1 score zgodne z oryginalna implementacja HotpotQA.

Referencja: Yang et al. 2018, https://github.com/hotpotqa/hotpot
Procedura normalizacji odpowiada eval script z paper-a.
"""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Standardowa normalizacja z HotpotQA:
      - lowercase
      - usun interpunkcje
      - usun rodzajniki a/an/the
      - kolapuj whitespace
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction: str, gold: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    """Word-level F1 na znormalizowanych tokenach."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_pair(prediction: str, gold: str) -> dict:
    return {
        "em": exact_match(prediction, gold),
        "f1": f1_score(prediction, gold),
        "pred_normalized": normalize_answer(prediction),
        "gold_normalized": normalize_answer(gold),
    }
