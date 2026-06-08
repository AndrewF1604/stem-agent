"""LLM-as-Judge przez Ollama (ta sama lub inna Llama).

Drugie, niezalezne zapytanie do modelu z jasnym promptem oceniajacym
semantyczna rownowaznosc odpowiedzi.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from openai import OpenAI

OLLAMA_BASE = "http://localhost:11434/v1"
_client = OpenAI(base_url=OLLAMA_BASE, api_key="ollama")

JUDGE_MODEL = "llama3.1:8b"


JUDGE_PROMPT = """You are a strict but fair grader for a question-answering task. Decide whether the candidate answer is semantically equivalent to the gold answer.

Rules:
- Tolerate paraphrase (same meaning, different words).
- Tolerate extra context if the core entity matches (e.g. "the city of Newcastle" matches "Newcastle").
- Tolerate translation if it's clearly the same entity (e.g. "Warszawa" matches "Warsaw").
- Reject answers that miss the key entity or get it wrong.
- Reject vague/hedging answers like "it's hard to say" or "I don't know".

Question: {question}
Gold answer: {gold}
Candidate answer: {prediction}

Respond with exactly two lines:
VERDICT: YES or NO
REASON: one short sentence explaining"""


@dataclass
class JudgeResult:
    correct: bool
    reasoning: str
    raw_output: str


_VERDICT_RE = re.compile(r"verdict[:\s]+(yes|no)\b", re.IGNORECASE)
_REASON_RE = re.compile(r"reason[:\s]+(.+?)$", re.IGNORECASE | re.MULTILINE | re.DOTALL)


def judge(question: str, gold: str, prediction: str, model: str = JUDGE_MODEL) -> JudgeResult:
    if not prediction.strip():
        return JudgeResult(correct=False, reasoning="empty prediction", raw_output="")

    prompt = JUDGE_PROMPT.format(question=question, gold=gold, prediction=prediction)
    try:
        resp = _client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        return JudgeResult(correct=False, reasoning=f"judge error: {e}", raw_output="")

    verdict_m = _VERDICT_RE.search(raw)
    reason_m = _REASON_RE.search(raw)
    correct = bool(verdict_m and verdict_m.group(1).lower() == "yes")
    reasoning = reason_m.group(1).strip() if reason_m else raw[:200]
    return JudgeResult(correct=correct, reasoning=reasoning, raw_output=raw)
