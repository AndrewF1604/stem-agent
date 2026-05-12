# Stem Agent — Self-Specializing Deep Research Agent

A minimal "stem" agent that starts generic and evolves into a specialized Deep Research agent for multi-hop factual questions. Inspired by stem cell differentiation: the agent reads signals from its environment (failure modes on a task class) and rebuilds itself, with a regression guard playing the role of cellular safeguards.

**Domain**: Multi-hop factual QA (HotpotQA distractor split).
**Architecture**: Genome (declarative spec) → Phenotype (runnable agent) → Fitness (LLM-judge accuracy) → directed mutations → rollback if regression.
**Stack**: OpenAI Responses API with built-in `web_search`. One provider, one key, no extra search service.

See [`WRITEUP.md`](./WRITEUP.md) for the full write-up of approach, experiments, and findings.

## Setup

Tested on Python 3.12 (Windows). Should work on 3.10+.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `python` isn't found on Windows, use `py` (the Windows Python launcher) or install from https://www.python.org/downloads/ with "Add to PATH" checked.

### API key

Copy `.env.example` to `.env` and set:

```
OPENAI_API_KEY=sk-...
# Optional overrides:
# OPENAI_BASE_URL=https://...    # if a custom gateway is used
# OPENAI_MODEL=gpt-5.4-mini      # default
```

```powershell
copy .env.example .env
notepad .env
```

The project uses **one** API: OpenAI Responses API with built-in `web_search`. No separate search provider needed.

## How to run

Run in this order — each step depends on the previous:

```powershell
# 1. Sanity-check the eval harness with a single-shot baseline on test split (~$0.002, ~1 min)
.\.venv\Scripts\python.exe eval.py --baseline --split test --trials 3

# 2. Run evolution loop with new methodology (train=15 / val=10, ~$0.30, ~5 min)
.\.venv\Scripts\python.exe evolve.py --generations 6 --train-n 15 --val-n 10 --tag main_v2

# 3. Ablation: same loop without regression guard (~$0.60, ~10 min)
.\.venv\Scripts\python.exe evolve.py --generations 6 --train-n 15 --val-n 10 --no-guard --tag ablation_v3

# 4. Final test with 3 trials each (G_final, G0, single-shot already covered by step 1)
.\.venv\Scripts\python.exe eval.py --final --genome runs/main_v2_<timestamp>/best_genome.yaml --split test --trials 3
.\.venv\Scripts\python.exe eval.py --final --genome runs/main_v2_<timestamp>/genome_g0_*.yaml --split test --trials 3
.\.venv\Scripts\python.exe eval.py --final --genome runs/ablation_v3_<timestamp>/best_genome.yaml --split test --trials 3

# 5. Generate plots from the run logs
.\.venv\Scripts\python.exe plots.py
```

Train/val/test are guaranteed-disjoint by construction — single shuffle of HotpotQA, then sliced. Each run saves `splits_metadata.json` with the exact qids for reproducibility.

All artifacts land in `runs/<tag>_<timestamp>/`. Plots land in `runs/fitness_curve.png` and `runs/mutation_breakdown.png`.

Total cost for the full pipeline ≈ **$2.50** on `gpt-5.4-mini`.

## Repository layout

```
eval.py          # Evaluation harness + LLM-judge + baselines (the foundation)
genome.py        # Pydantic genome spec + serialization + hashing
agent.py         # Phenotype builder: turns a genome into a runnable agent
tools.py         # Tool registry — currently just OpenAI built-in web_search
mutations.py     # 3 atomic mutation operators + failure-analysis proposer
evolve.py        # Evolution loop with regression guard
plots.py         # Generate fitness curves and comparison charts
data/            # HotpotQA subset (downloaded on first run)
runs/            # Output: genomes, trajectories, metrics
WRITEUP.md       # The write-up evaluators read first
```

## Reading order for evaluators

1. [`WRITEUP.md`](./WRITEUP.md) — narrative, results, what surprised me
2. [`evolve.py`](./evolve.py) — the core loop
3. [`genome.py`](./genome.py) + [`agent.py`](./agent.py) — the genome→phenotype mechanism
4. [`mutations.py`](./mutations.py) — what kinds of changes the agent can make to itself
