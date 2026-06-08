"""Metryki rozkładu odpowiedzi przy wielu iteracjach na pytanie.

Wszystkie funkcje operują na liście znormalizowanych odpowiedzi (po metrics.normalize_answer)
i/lub na liście flag poprawności. Czyste, bez I/O — testowalne osobno.

Kluczowe pojęcia:
  - agreement      : udział modalnej odpowiedzi (1.0 = model za każdym razem to samo)
  - answer_entropy : entropia Shannona rozkładu, znormalizowana do [0,1]
  - pass@1         : oczekiwana trafność pojedynczego strzału (n_correct / n)
  - pass@k         : prawdopodobieństwo że k próbek zawiera ≥1 poprawną (nieobciążony estymator)
  - wrong_consistency : zgoda *wśród błędnych* próbek — rozróżnia błąd systematyczny od losowego
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional, Sequence


def majority(answers_norm: Sequence[str]) -> tuple[str, float, int]:
    """Zwraca (modalna_odpowiedź, agreement, liczba_unikalnych)."""
    if not answers_norm:
        return "", 0.0, 0
    counts = Counter(answers_norm)
    top, cnt = counts.most_common(1)[0]
    return top, cnt / len(answers_norm), len(counts)


def shannon_entropy(answers_norm: Sequence[str], normalized: bool = True) -> float:
    """Entropia rozkładu odpowiedzi. normalized=True dzieli przez log(n) → [0,1].

    0.0 = wszystkie próbki identyczne, 1.0 = wszystkie różne (maks. rozproszenie).
    """
    n = len(answers_norm)
    if n <= 1:
        return 0.0
    counts = Counter(answers_norm)
    probs = [c / n for c in counts.values()]
    h = -sum(p * math.log(p) for p in probs)
    if normalized:
        h /= math.log(n)
    return h


def pass_at_1(n_correct: int, n: int) -> float:
    return n_correct / n if n else 0.0


def pass_at_k(n_correct: int, n: int, k: int) -> float:
    """Nieobciążony estymator P(≥1 poprawna w k losowanych z n próbek).

    Wzór z Codex/HumanEval (Chen et al. 2021): 1 - C(n-c, k) / C(n, k).
    Dla k>=n redukuje się do "czy jakakolwiek próbka jest poprawna".
    """
    if n == 0 or n_correct <= 0:
        return 0.0
    if k >= n:
        return 1.0
    if n - n_correct < k:
        return 1.0
    return 1.0 - math.comb(n - n_correct, k) / math.comb(n, k)


def wrong_consistency(answers_norm: Sequence[str], correct_flags: Sequence[int]) -> Optional[float]:
    """Zgoda wśród błędnych próbek (agreement modalnej z podzbioru error).

    Wysoka → model uparcie zwraca tę samą złą odpowiedź (błąd systematyczny,
    możliwy silny prior / kontaminacja). Niska → losowe zgadywanie.
    None gdy nie ma błędnych próbek.
    """
    wrong = [a for a, ok in zip(answers_norm, correct_flags) if not ok]
    if not wrong:
        return None
    _, agreement, _ = majority(wrong)
    return agreement
