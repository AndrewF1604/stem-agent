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
# 1. Sanity-check the eval harness with a single-shot baseline (~$0.001, ~1 min)
.\.venv\Scripts\python.exe eval.py --baseline --n 20

# 2. Run evolution loop (~$0.25, ~5-10 min wallclock)
.\.venv\Scripts\python.exe evolve.py --generations 6 --val-size 5 --tag main

# 3. Ablation: same loop without regression guard (~$0.30, ~5-10 min)
.\.venv\Scripts\python.exe evolve.py --generations 6 --val-size 5 --no-guard --tag ablation

# 4. Final test on held-out questions (~$0.02, ~1 min) — run ONCE on a different seed
.\.venv\Scripts\python.exe eval.py --final --genome runs/<latest>/best_genome.yaml --n 10 --seed 7

# 5. Optional: also test single-shot baseline + initial G0 on the same held-out set for the comparison table
.\.venv\Scripts\python.exe eval.py --baseline --n 10 --seed 7
.\.venv\Scripts\python.exe eval.py --final --genome runs/<latest>/genome_g0_*.yaml --n 10 --seed 7

# 6. Generate plots from the run logs
.\.venv\Scripts\python.exe plots.py
```

All artifacts (genomes, trajectories, metrics) land in `runs/<tag>_<timestamp>/`. Plots land in `runs/fitness_curve.png` and `runs/mutation_breakdown.png`.

Total cost for the full pipeline (steps 1–5) is ≈ $0.60 on `gpt-5.4-mini`.

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
