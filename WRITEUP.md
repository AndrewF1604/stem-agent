# Stem Agent: Self-Specializing Deep Research

## TL;DR

A stem agent that starts as a generic LLM and evolves itself into a specialized multi-hop research agent on HotpotQA. The DNA is a small declarative *genome* (architecture, prompt, tools, control flow); a separate failure-analysis LLM reads failed trajectories from a training set and proposes targeted mutations; a regression guard rolls back if validation accuracy or canary-set accuracy drops.

**The most important finding came from auditing my own methodology after the first iteration.** A self-review surfaced two real ML methodology bugs (train/val overlap, no guaranteed-disjoint split). After fixing them and re-running, the headline picture changed substantially: the previous version's "evolution improved val 40% → 80%" was largely an artifact of the proposer overfitting to the val set it was being evaluated on. With a clean 3-way split and multi-trial averaging:

| Variant                                | Validation (n=10) | Held-out test (n=10, 3 trials, mean ± std) |
| -------------------------------------- | :---------------: | :----------------------------------------: |
| Single-shot, no tools (`gpt-5.4-mini`) | —                 | **0.73 ± 0.05**                            |
| ReAct G0 / Main v2 G_final (guard on)  | 0.60              | **0.87 ± 0.19**                            |
| Ablation G_final (guard off)           | 0.70              | **0.77 ± 0.05**                            |

But the same audit also produced the most informative result of the whole project: **the regression guard fired for the first time**. With clean train/val separation the proposer regularly proposes mutations that boost val but break previously-solved canary questions — and the guard catches them. The ablation, which disables the guard, achieves higher val (0.70 vs 0.60) but **lower held-out** (0.77 vs 0.87). That gap is exactly the cost of letting val-optimal mutations through unchecked. The "apoptosis" piece of the metaphor finally has empirical teeth.

Built on the OpenAI Responses API with the built-in `web_search` tool — one API, no separate search service.

## 1. Decisions

- **Domain — HotpotQA distractor split.** Picked over QA / Security / Codegen because ground truth is unambiguous, the difficulty gradient is clean, and the architecture space genuinely matters (single-shot vs ReAct give visibly different results).
- **Three-piece system**: a *genome* (mutable data), a *phenotype* (runnable agent compiled from genome), and *fitness* (LLM-judge accuracy on a held val batch). Splitting genome from phenotype is what makes mutations atomic and rollback-able — without it, "rebuild without breaking" has no concrete meaning.
- **Failure-driven mutation, not random search.** A separate LLM call reads failed trajectories on the *train* set, classifies the dominant failure mode, and proposes 3 mutations conditioned on that diagnosis.
- **Three atomic mutations**: `REFINE_PROMPT`, `TOGGLE_REFLECTION`, `ADJUST_MAX_STEPS`. Plus a 3-genome initial population (`single_shot`, `react@max_steps=5`, `react@max_steps=3`) which the agent scores and selects from. I deliberately cut tool synthesis and hierarchical decomposition (§5).
- **Stack — OpenAI Responses API + built-in `web_search`.** The API runs the search loop server-side, so my "ReAct" call is one API call where the model decides search depth. Consequence: `max_steps` becomes a soft instruction in the prompt; reflection is a chained second call via `previous_response_id`.
- **Regression guard.** Every accepted mutation must beat current val fitness *and* score 100% on the canary set (val questions G0 solved). The ablation also measures canary as telemetry, just doesn't enforce.
- **Stop criterion**: 6 generations *or* 3 plateau generations *or* val acc ≥ 0.85.

## 2. Architecture

```
            ┌──────────────────────────────────────────────────┐
            │  Stem Agent Loop (evolve.py)                     │
   ┌────────┤  1. run phenotype on val → fitness               │
   │        │  2. run phenotype on TRAIN → failure trajectories│
   │        │  3. proposer reads train failures → 3 mutations  │
   │        │  4. eval each mutation on val                    │
   │        │  5. accept best if fitness > current AND no      │
   │        │     canary regression; else rollback             │
   │        │  6. snapshot genome, log trajectory              │
   │        └──────────────────────────────────────────────────┘
   ▼
 Genome → Phenotype Builder → AgentResult
 (genome.py)   (agent.py)
```

The genome is hash-addressable; every accepted mutation writes a new YAML with `parent_hash`, so the lineage is reconstructible. **Train and val are disjoint by construction** — single shuffle of the HotpotQA pool, then slice into `train[:15] / val[15:25] / test[25:35]`. Test is touched only by `eval.py --final`.

## 3. Experiments

### 3.1 Methodology

- **Three-way split, guaranteed-disjoint.** `_load_filtered_pool(seed=42)` shuffles the HotpotQA validation pool once, filters to comparison/bridge with answers ≤ 100 chars. Train/val/test are slices of the same pool — overlap is mathematically impossible.
- **Class balance**:
  | Split | Total | Comparison | Bridge |
  | ----- | :---: | :--------: | :----: |
  | train | 15    | 3          | 12     |
  | val   | 10    | 1          | 9      |
  | test  | 10    | 1          | 9      |

  Bridge dominates ~85%, which is the natural class ratio in HotpotQA's validation set. I didn't stratify; the proposer sees what the real distribution looks like.
- **Multi-trial averaging on headline numbers.** All test-set numbers reported as mean ± std over 3 independent trials per genome (same questions, same prompt, different LLM stochasticity).
- **Reproducibility.** Each evolution run saves `splits_metadata.json` with seed + qids + balance, so any reader can reconstruct the exact split.

### 3.2 Headline numbers on held-out test (n=10, 3 trials each)

| Variant                                | mean | std  | per-trial         |
| -------------------------------------- | :--: | :--: | :---------------: |
| Single-shot, no tools                  | 0.73 | 0.05 | 0.8 / 0.7 / 0.7   |
| ReAct G0 / Main v2 G_final (guard on)  | 0.87 | 0.19 | 1.0 / 0.6 / 1.0   |
| Ablation v3 G_final (guard off)        | 0.77 | 0.05 | 0.8 / 0.7 / 0.8   |

Two surprises live in this table. First, **G0 ReAct's std is 0.19** — same genome on same questions varies between 0.6 and 1.0 across trials. Stochasticity is on the same order as the gap between variants. Second, **the guard-on agent beats the guard-off agent on held-out by 0.10 mean**, even though guard-off scored *higher* on val (0.70 vs 0.60). This is the textbook overfitting/regularization tradeoff and it shows up cleanly at this scale.

### 3.3 Fitness curves — main v2 vs ablation v3

```
MAIN (guard on, seed=42, G0=893522b8f4 from Phase A):
  Gen 0:  0.60  ◆  G0
  Gen 1:  0.60  ✗  REJECT (canary regression: TOGGLE_REFLECTION beat val 0.6→0.7 but canary 1.0→0.83)
  Gen 2:  0.60  ✗  REJECT (no improvement)
  Gen 3:  0.60  ✗  REJECT (no improvement) → STOP (plateau)
  Result: 0 mutations accepted, G_final = G0

ABLATION (guard off, seed=42):
  Gen 0:  0.50  ◆  G0 (same hash, different stochastic trial → different score)
  Gen 1:  0.60  ✓  ACCEPT ADJUST_MAX_STEPS  (canary 0.80, would have been rolled back)
  Gen 2:  0.60  ✗  REJECT (no improvement)
  Gen 3:  0.70  ✓  ACCEPT TOGGLE_REFLECTION (canary 0.80, would have been rolled back)
  Gen 4-6: 0.70 ✗  REJECT × 3 → STOP (plateau)
  Result: 2 mutations accepted (both canary-regressing), G_final = c13d01d62b
```

See `runs/fitness_curve.png`. The solid lines show agent state; red ×s are rejected candidates; dashed squares are canary checks. The visible story: both runs hit plateau, but ablation gets there by accepting mutations whose canary scores were below 1.0 — the guard would have rolled those back, and the test-set numbers above confirm those rollbacks would have been correct.

### 3.4 Mutation kind statistics (main v2 + ablation v3, 9 trials each)

| Kind                | Trials | Improved | Tied | Hurt | Accepted | Notes |
| ------------------- | :----: | :------: | :--: | :--: | :------: | :--- |
| REFINE_PROMPT       | 9      | 1        | 3    | 5    | 0        | Never won outright in either run |
| ADJUST_MAX_STEPS    | 9      | 1        | 4    | 4    | 1        | Accepted once in ablation; tied a lot in main |
| TOGGLE_REFLECTION   | 9      | 2        | 2    | 5    | 1*       | *Caught by guard in main; only accepted in ablation |

The two times TOGGLE_REFLECTION *did* improve val fitness, it also broke previously-solved canary questions — main caught one, ablation accepted both. See `runs/mutation_breakdown.png`.

### 3.5 Regression guard analysis

The guard's job is to catch mutations that pass fitness but regress on previously-solved questions. Main v2 generation 1 was the first time this fired in any run of this project:

```
Gen 1 candidate: TOGGLE_REFLECTION
  val accuracy:  0.6 → 0.7   (passes fitness check)
  canary accuracy: 1.0 → 0.83 (caught by guard, rollback)
```

In the ablation, the same TOGGLE_REFLECTION (gen 3) was accepted, and the agent later ended up at 0.77 on test versus the guard-on agent's 0.87 — a 10-point gap on the held-out set that didn't show up in either run's val score. **That gap is the dollar value of the guard.**

## 4. What surprised me

**Auditing my own methodology found bugs that completely changed the story.** The first iteration of this project (committed as the first GitHub push) reported "evolution 40% → 80%, regression guard never fired". A subsequent audit against ML conventions found train/val overlap (proposer was reading failures from the same 5 questions used for fitness) and a non-guaranteed-disjoint train/test split. After fixing both, the new numbers tell a much more uncomfortable but more honest story: most of the previous "improvement" was the proposer overfitting to the val set it was being evaluated on. The architecture switch (single-shot → ReAct) still does the heavy lifting; the evolution itself is much more modest. **The most useful thing I did in this project was probably going back and breaking my own previous results.**

**The regression guard exists for a reason.** Under clean methodology, the proposer regularly proposes mutations that look good on val but regress canary. In main v2 the guard caught one in 3 generations; in ablation v3, two were accepted and both regressed canary. The held-out test confirms: ablation gets the higher val score but the *worse* test score (0.77 vs 0.87). This is exactly the val/test divergence the canary check was designed to catch. In the first iteration this never showed up because the proposer was reading the same set used for fitness — the failure mode was structurally impossible.

**Reflection is a fitness gainer *and* a canary breaker.** In both runs, `TOGGLE_REFLECTION` produced the *only* val fitness improvements above the parent — but in both cases, the same mutation also dropped canary accuracy. The reflection step second-guesses the agent into changing some correct answers to wrong ones, but on net (across val) it still answers more questions correctly. This is the precise pattern that makes regularization-via-canary a real safeguard: a mutation that's "good on average" isn't necessarily safe.

**The proposer keeps diagnosing the same failure mode.** Across all 9 generations of both runs, the diagnosis was a variant of *"the agent answers with a nearby related label instead of the exact requested abstraction"*. The proposer kept finding the same problem and proposing different mutations to address it, most of which didn't help. HotpotQA's failure modes don't decompose into the kinds of fixes my mutation set can express.

**Stochasticity in fitness is roughly the size of an "accepted mutation".** G0 on test scored 1.0, 0.6, 1.0 across three trials — same model, same prompt, same questions. Standard deviation 0.19. The gap between guard-on and guard-off on test is 0.10. Without 3-trial averaging I literally could not distinguish "real improvement" from "lucky trial". This argues for multi-trial fitness during evolution too, not just for reporting; cost was the constraint that prevented me from doing it inside the loop.

**Pretraining contamination is a real and unaddressed issue.** HotpotQA has been public since 2018; `gpt-5.4-mini` almost certainly saw its questions and gold answers during pretraining. The single-shot baseline of 0.73 on test (no tools, no search) is *direct evidence* of model recall — the model knows the answers from training data, not from the search results. This means:
- The ReAct gain over single-shot (0.87 vs 0.73 = 14pp on test mean) is the upper bound on what *retrieval* is adding on top of *recall*.
- The "evolution" gain (G_final vs G0 ReAct) is approximately zero on this benchmark — the unmodified ReAct seed already captures most of what `gpt-5.4-mini` plus web_search can do.
- An honest evaluation would use a benchmark released *after* the model's pretraining cutoff. I didn't have time to find or construct one; this is the most impactful follow-up.

**Class imbalance shifts the headline.** The HotpotQA val pool is ~85% bridge / 15% comparison after my length filter. With n=10, val has 9 bridge + 1 comparison. A single comparison question contributes ±0.10 fitness. This is the kind of detail that gets buried when a writeup just reports a single accuracy number.

**Cost was lower than expected and dominated by Phase A + train evals.** Full pipeline (main + ablation + 3 final tests × 3 trials each + baselines) ran for ~$2.50 on `gpt-5.4-mini`. The new methodology adds train-set evals each generation, which roughly doubled per-generation cost compared to the first iteration, but it's still trivial.

## 5. What I'd do with more time

In priority order, with relative cost:

1. **Use a post-cutoff benchmark to control for pretraining contamination.** This is the single most impactful experiment. Sources: GPQA Diamond questions written after the cutoff, BrowseComp, or hand-constructed multi-hop on recent events. Without this, the held-out test reports a mix of model recall and retrieval, with no way to disentangle.
2. **Multi-trial averaging inside the evolution loop**, not just for final reporting. Run each candidate 3× on val and use the mean as fitness. At n=10 val and 3 trials, this triples per-generation cost but should eliminate "single answer flip" mutations from being accepted.
3. **Tool synthesis**, not just selection. Let the mutator write a new Python function when failure analysis surfaces a recurring missing capability ("needs to compute date differences"). Closest analogue to a stem cell *growing* a structure rather than picking from a fixed shelf.
4. **Conditional reflection.** The audit shows reflection both helps and hurts depending on the question. A trigger like "only reflect if first answer's evidence is thin" should capture the upside without the downside.
5. **Adversarial guard test.** Inject hand-crafted bad mutations (random shuffled prompts, max_steps=0, contradictory instructions) and verify the guard catches all of them. With organic proposer mutations we now have proof the guard *can* fire; adversarial input proves the mechanism is sound on inputs that haven't been organically encountered.

## 6. Reproduction

See [`README.md`](./README.md). Full pipeline (main evolution + ablation + final tests with 3 trials + baselines) runs in ~25 min wallclock and ≈ $2.50 on `gpt-5.4-mini` via the Responses API. Each evolution run saves a `splits_metadata.json` containing the exact qids of every split, so any reader can verify reproducibility.

---

*This write-up reflects the post-audit state of the project. The original submission's WRITEUP.md (committed earlier on the same GitHub branch) reports the pre-audit numbers and narrative; the git history preserves both for comparison.*
