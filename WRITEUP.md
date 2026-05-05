# Stem Agent: Self-Specializing Deep Research

## TL;DR

A stem agent that starts as a generic LLM and evolves itself into a specialized multi-hop research agent on HotpotQA. The DNA is a small declarative *genome* (architecture, prompt, tools, control flow); a separate failure-analysis LLM reads failed trajectories and proposes targeted mutations; a regression guard rolls back if validation accuracy or canary-set accuracy drops.

On a held-out test set of 10 multi-hop questions: **single-shot 40% → ReAct G0 70% → evolved G_final 80%**. G_final scored the same 80% on validation (5 questions, where the evolution actually happened) and on held-out (10 unseen questions) — so the genome generalizes rather than overfits.

What I learned from the experiments was less about the headline number and more about three findings, each of which complicated my pre-experiment intuition. Built on the OpenAI Responses API with the built-in `web_search` tool — one API, no separate search service.

## 1. Decisions

- **Domain — HotpotQA distractor split.** Picked over QA / Security / Codegen because ground truth is unambiguous, the difficulty gradient is clean, and the architecture space genuinely matters (single-shot vs ReAct give visibly different results).
- **Three-piece system**: a *genome* (mutable data), a *phenotype* (runnable agent compiled from genome), and *fitness* (LLM-judge accuracy on a held val batch). Splitting genome from phenotype is what makes mutations atomic and rollback-able — without it, "rebuild without breaking" has no concrete meaning.
- **Failure-driven mutation, not random search.** A separate LLM call reads failed trajectories, classifies the dominant failure mode, and proposes 3 mutations conditioned on that diagnosis.
- **Three atomic mutations**: `REFINE_PROMPT`, `TOGGLE_REFLECTION`, `ADJUST_MAX_STEPS`. Plus a 3-genome initial population (`single_shot`, `react@max_steps=5`, `react@max_steps=3`) which the agent scores and selects from. I deliberately cut tool synthesis and hierarchical decomposition (§5).
- **Stack — OpenAI Responses API + built-in `web_search`.** The API runs the search loop server-side, so my "ReAct" call is one API call where the model decides search depth. Consequence: `max_steps` becomes a soft instruction in the prompt; reflection is a chained second call via `previous_response_id`. Less explicit per-step control, but dramatically simpler infrastructure.
- **Regression guard.** Every accepted mutation must beat current val fitness *and* score 100% on the canary set (questions G0 solved). This is the explicit analogue of cellular safeguards.
- **Stop**: 6 generations *or* 3 plateau generations *or* val acc ≥ 0.85.

## 2. Architecture

```
            ┌──────────────────────────────────────────────────┐
            │  Stem Agent Loop (evolve.py)                     │
   ┌────────┤  1. run phenotype on val batch                   │
   │        │  2. failure-analysis LLM → top failure mode      │
   │        │  3. proposer outputs 3 mutations                 │
   │        │  4. eval each on val batch                       │
   │        │  5. accept best if fitness > current AND no      │
   │        │     canary regression; else rollback             │
   │        │  6. snapshot genome, log trajectory              │
   │        └──────────────────────────────────────────────────┘
   ▼
 Genome → Phenotype Builder → AgentResult
 (genome.py)   (agent.py)
```

The genome is hash-addressable; every accepted mutation writes a new YAML with `parent_hash`, so the lineage is reconstructible. Main run lineage: `G0 (732558b333) → G2 (REFINE_PROMPT, 1db55d8153) → G4 (ADJUST_MAX_STEPS, 5839565d0e)`.

## 3. Experiments

### 3.1 Headline numbers

| Variant                                | Validation | Held-out (n=10) |
| -------------------------------------- | :--------: | :-------------: |
| Single-shot, no tools (`gpt-5.4-mini`) | 15% (n=20) | **40%**         |
| ReAct G0 (best initial seed)           | 40% (n=5)  | **70%**         |
| ReAct G_final (evolved, 6 generations) | 80% (n=5)  | **80%**         |

Most of the gain comes from the architecture switch (single-shot → ReAct with web_search). The evolution itself adds 10 points on held-out and 40 points on validation. The val gap is larger because that batch was harder, by chance of seed; the held-out gain is the *transferable* portion.

### 3.2 Fitness curve and lineage (main run)

```
Gen 0:  0.40   ◆  G0: ReAct, generic prompt, max_steps=5
Gen 1:  0.40   ✗  REJECT (no improvement; all 3 candidates tied)
Gen 2:  0.60   ✓  ACCEPT REFINE_PROMPT  → "verify exact entity, not nearby fact"
Gen 3:  0.60   ✗  REJECT
Gen 4:  0.80   ✓  ACCEPT ADJUST_MAX_STEPS  5 → 6
Gen 5:  0.80   ✗  REJECT
Gen 6:  0.80   ✗  REJECT  →  STOP (plateau)
```

Two accepted mutations across 6 generations. Acceptance rate: **2 / 18 candidates = 11%** — the proposer produces plausible-sounding mutations that mostly *don't* improve fitness, and the loop's job is mostly noise filtering. See `runs/fitness_curve.png`.

### 3.3 Mutation kind statistics (main + ablation, 12 trials each)

| Kind                | Trials | Improved | Tied | Hurt | Accepted |
| ------------------- | :----: | :------: | :--: | :--: | :------: |
| REFINE_PROMPT       | 12     | 2        | 5    | 5    | 2        |
| ADJUST_MAX_STEPS    | 12     | 1        | 4    | 7    | 1        |
| TOGGLE_REFLECTION   | 12     | 1        | 1    | **10** | **0**  |

`TOGGLE_REFLECTION` was proposed in every generation of both runs, hurt fitness in 10 of 12 trials, and was never accepted. See `runs/mutation_breakdown.png`. This is the project's clearest surprise (§4). `REFINE_PROMPT` is the only kind that consistently produced fitness wins, accounting for both accepted mutations in the ablation and one of two in the main run.

### 3.4 Ablation: evolution without regression guard

Same loop, same data, same proposer — `--no-guard` removes the canary check from the rollback decision. I instrumented the ablation to *still measure* canary accuracy on every accepted mutation, just not use it for rollback. This way the ablation answers the question "would the guard have fired if it were on?".

| Variant                  | Phase-A G0 acc | Final val acc | Canary regressions caught | Canary regressions ignored |
| ------------------------ | :------------: | :-----------: | :-----------------------: | :------------------------: |
| Full system (with guard) | 0.40           | **0.80**      | 0                         | n/a                        |
| Without regression guard | 0.60           | **0.80**      | n/a                       | **0**                      |

Both runs converged to the same 0.80 plateau via different lineages. The number that matters for the metaphor's load-bearing-ness is the rightmost column: in the entire 6-generation ablation, **no accepted mutation regressed any G0-solved question**. The proposer simply doesn't generate "passes fitness but breaks canary" mutations — at least not on this task at this scale. The guard is a safety net the proposer never tested.

This is the kind of result I'd have wanted to refute. The honest interpretation: at n=5 validation × 6 generations × the failure modes this task surfaces, the guard is theatrical rather than load-bearing. The natural next experiment (§5) is to inject deliberately-degenerate mutations and confirm the guard catches *those* — testing the mechanism on adversarial input rather than the proposer's organic output.

A side observation: the same G0 genome scored 0.40 (main) vs 0.60 (ablation) on the identical val batch — that's pure LLM stochasticity, and it's the same magnitude as an accepted mutation. See §4.

## 4. What surprised me

**Reflection was actively harmful, not just neutral.** Combining both runs, the proposer suggested toggling on `reflection_enabled` in 12 of 12 generations; in 10 of 12 the result *worsened*. My pre-experiment intuition was that a self-check would help on multi-hop questions where the first guess is plausibly wrong. What actually happened: the agent's first answer is already grounded in citations from web_search, and the reflection prompt — "reconsider whether the evidence supports that answer" — pushes the model to *change its mind* even when the original was right. With more time I'd test conditional reflection — only triggered on low-confidence first answers, not unconditionally.

**The proposer kept diagnosing the same failure mode.** Across all 12 generations of both runs, the diagnosis was a variant of *"the agent answers with a nearby related label instead of the exact requested abstraction"*. This is a real, consistent HotpotQA failure pattern. What surprised me is that the proposer kept finding the same problem and kept proposing different mutations to address it, most of which didn't help. The system has a single bottleneck and my mutation set can't reach it.

**The regression guard never fired in the main run.** Every rejection was via the fitness check, never via canary regression. This is uncomfortable for the central "apoptosis" narrative: in this scale of run, the guard is insurance against a failure mode the proposer never produced. The honest read: at n=5 validation × 6 generations, fitness gates are strict enough that nothing slips past them with a passing fitness score. The guard becomes load-bearing at larger scales where mutations have more room to improve val while breaking unsampled questions.

**Stochasticity in fitness was larger than expected.** The same G0 genome on the same 5 questions produced 0.40 in the main run and 0.60 in the ablation run — same model, same prompt, same data. With n=5 a single answer flip is 0.20 of "fitness" — the same magnitude as an accepted mutation. With more time I'd average over 3 trials per genome to filter LLM-output noise from real signal.

**Some of the held-out "failures" are dataset noise, not agent failures.** Of 2 failures on held-out, one was "dark comedy" against a gold of "comedy-drama" (arguably both right). The other was a question where the gold was "Turkey" against an agent answer of "Fatih district of Istanbul" (both correct depending on how you read the question). The judge was strict by design; at n=10, one disputed question shifts the headline by 10pp.

**Cost was lower than expected.** Full evolution + ablation + final test + baselines: ~$0.60 total on `gpt-5.4-mini`. The Responses API + built-in web_search was the right call; manual ReAct with a custom search tool would have spent 3-5x more.

## 5. What I'd do with more time

1. **Tool synthesis**, not just tool selection. Let the mutator write a new Python tool when failure analysis surfaces a recurring missing capability ("needs to compute date differences"). Closest analogue to a stem cell *growing* a structure rather than picking from a fixed shelf.
2. **Conditional reflection.** Given how badly unconditional reflection performed, the natural next test: let the agent emit a confidence score, only reflect below threshold.
3. **Scale up validation.** n=5 gives 0.20 fitness granularity — single-question flips look like real changes. n=20 with multi-trial averaging would let the proposer see consistent patterns rather than noise.
4. **Hierarchical decomposition.** Add a `DECOMPOSE(node → sub-agents)` mutation where each sub-agent runs its own mini-evolution. Architecture would *emerge* rather than be chosen from {single_shot, react}.
5. **Cross-domain transfer test.** Freeze evolution, run G_final on a different question class (scientific facts, sports trivia). If it generalizes, the stem agent built a *researcher*; if not, it built a *HotpotQA-specific researcher*. Both are interesting findings.

## 6. Reproduction

See [`README.md`](./README.md). Full pipeline runs in ~40 min wallclock; total API cost ≈ $0.60 on `gpt-5.4-mini` via the Responses API.
