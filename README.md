# Analiza błędów LLM na multi-hop QA (HotpotQA)

Projekt na kurs NLP. Badam **jakie błędy** popełnia Llama 3.1 8B (lokalnie przez Ollama) odpowiadając na pytania wieloskokowe z HotpotQA — pod różnymi konfiguracjami (temperatura, styl promptu, dostęp do wyszukiwarki) — i czy rodzaj błędów zmienia się wraz z ustawieniami.

Design: **repeated-measures**. Zamrożony zestaw 100 pytań, zadawanych pod 7 konfiguracjami, część wielokrotnie (K iteracji) — żeby z rozkładu odpowiedzi odróżnić błędy **systematyczne** od **stochastycznych**.

## Dokumentacja

- **[OPIS_PROJEKTU.md](OPIS_PROJEKTU.md)** — wysokopoziomowy opis bez szczegółów technicznych: o co chodzi, jak działa, na czym polega
- **[HANDOFF_SETUP.md](HANDOFF_SETUP.md)** — stan projektu, pliki, jak uruchomić
- **[HANDOFF_PLAN.md](HANDOFF_PLAN.md)** — roadmap, pytania badawcze, plan analiz i prezentacji

## Setup

Python 3.12 + Ollama (https://ollama.com).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3.1:8b
```

## Workflow

```powershell
python dataset.py          # losuje raz 100 pytań -> data/samples_100.json + manifest
python sweep.py            # 7 conditions × 100 pytań × K iteracji (resume działa)
python evaluate_all.py     # EM, F1, LLM-Judge per próbka + agregacja per pytanie
```

Wyjście: `analysis/raw_long.csv` (per próbka) i `analysis/per_question.csv` (metryki rozkładu: pass@1/@k, majority vote, agreement, entropia, wrong_consistency).
