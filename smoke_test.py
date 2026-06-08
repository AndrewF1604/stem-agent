"""Sanity check przed Fazą 2:
1. Czy Ollama serwer odpowiada na http://localhost:11434
2. Czy mamy zainstalowany model (lista lokalnych modeli)
3. Czy potrafimy wysłać jedno proste pytanie i dostać odpowiedź
4. Czy potrafimy załadować jedno pytanie z dataset.json i zadać je modelowi z kontekstem
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests
from openai import OpenAI


OLLAMA_BASE = "http://localhost:11434"


def check_server() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception as e:
        print(f"Ollama nie odpowiada na {OLLAMA_BASE}: {e}")
        return False


def list_models() -> list[str]:
    r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
    data = r.json()
    return [m["name"] for m in data.get("models", [])]


def quick_inference(model: str, prompt: str) -> str:
    client = OpenAI(base_url=f"{OLLAMA_BASE}/v1", api_key="ollama")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def test_with_sample(model: str) -> None:
    samples_path = Path("data/samples_300.json")
    if not samples_path.exists():
        print("data/samples_300.json nie istnieje — uruchom najpierw `python dataset.py`")
        return

    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    s = samples[0]

    context_text = "\n\n".join(
        f"[{p['title']}]\n" + " ".join(p["sentences"]) for p in s["context"]
    )
    prompt = (
        f"Answer the question based on the following context.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {s['question']}\n\n"
        f"Answer (short phrase only):"
    )

    print(f"\nReal sample test:")
    print(f"  Question: {s['question']}")
    print(f"  Gold:     {s['gold_answer']}")
    print(f"  Type/level: {s['q_type']} / {s['level']}")
    print(f"  Context: {len(s['context'])} paragrafow, prompt len: {len(prompt)} chars")
    t0 = time.time()
    answer = quick_inference(model, prompt)
    dt = time.time() - t0
    print(f"  Llama:    {answer}")
    print(f"  Latency:  {dt:.1f}s")


def main():
    print("=== Krok 1: Sprawdzanie czy Ollama dziala ===")
    if not check_server():
        print("\nOllama nie chodzi. Sprawdz:")
        print("  1. Czy zainstalowales Ollama z https://ollama.com/download/windows")
        print("  2. Czy uruchomiles aplikacje Ollama (powinna byc w system tray)")
        print("  3. Czy port 11434 nie jest zablokowany")
        sys.exit(1)
    print("  OK")

    print("\n=== Krok 2: Lista zainstalowanych modeli ===")
    models = list_models()
    if not models:
        print("  Brak modeli. Pobierz jakis modelem:")
        print("    ollama pull llama3.1:8b    # lekki, ~5GB, dziala wszedzie")
        print("    ollama pull llama3.3:70b   # mocny, ~40GB, wymaga >=24GB VRAM")
        sys.exit(1)
    for m in models:
        print(f"  - {m}")

    model = models[0]
    print(f"\n=== Krok 3: Test inference z modelem {model} ===")
    t0 = time.time()
    answer = quick_inference(model, "What is the capital of Poland? Answer with one word.")
    dt = time.time() - t0
    print(f"  Pytanie: 'What is the capital of Poland?'")
    print(f"  Odpowiedz: {answer}")
    print(f"  Czas: {dt:.1f}s")

    print(f"\n=== Krok 4: Test na realnym sample z HotpotQA ===")
    test_with_sample(model)

    print("\nWszystko dziala. Mozna isc do Fazy 2.")


if __name__ == "__main__":
    main()
