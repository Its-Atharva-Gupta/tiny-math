# tiny-math — overnight run status log
<!-- Autonomous agent: update this file at every decision point. -->

## Goal
Find the smallest transformer that can do 4-digit addition. Sizes {1M, 5M, 8M},
from-scratch GPT, digit-level vocab (16 tokens), SFT on generated data, GRPO
polish only if SFT plateaus. Headline number + phase diagram + honest model card.

## Key facts
- Kaggle quota: 30h/week (checked at start: 0h used). Kernels run serially.
- Baselines (local, n=5000, 4-digit l2r): bigram = 0.0%, untrained 1M = 0.0%.
  Floor is zero — any accuracy is real learning.
- Local CPU speed: ~3100 tok/s (1M model). Kaggle T4 expected ~100-200x faster.
- Data: train/eval contamination prevented BY CONSTRUCTION — both eval sets
  built upfront, stream rejects any (a,b) in either. (Bug found + fixed after
  local run crashed on the overlap assert; bakeoff v1 was deleted pre-run.)

## Formats (Phase 0b bakeoff picks winner)
l2r = "47+68=\n115\n" | rtl = "47+68=\n511\n" (answer reversed) |
cot = "47+68=\n7+8=15,4+6+1=11,115\n" (digit steps + answer, comma-sep)
Paper (arXiv 2307.03381) predicts rtl and cot >> l2r for addition.

## Decision rules (pre-committed, no drift overnight)
1. Format winner = highest final acc at 1M/4-digit/10M tokens. Tie → prefer cot
   (most interesting), then rtl.
2. GRPO phase triggers ONLY if best SFT (8m, 50M tokens, 4-digit) < 90%.
   GRPO reuses the 150-line trainer pattern from grpo-gsm8k (shaped reward with
   format floor — zero-variance groups give zero advantage).
3. If bakeoff shows ALL formats ~0% at 1M/10M → raise 1M config to d=192 L=6
   before sweeping (under-parameterized, don't burn sweep quota on a dead cell).
4. Extrapolation kernel runs after sweep; it mounts sweep outputs as a dataset.

## SWEEP RESULTS (n=5000, cot format, all 11 runs complete)
4-digit addition, exact-match on held-out problems:
| model | 71k probs | 357k probs | 1.78M probs |
|---|---|---|---|
| 1M | 0.0% | 0.2% | **100.0%** |
| 5M | 0.0% | 95.7% | 100.0% |
| 8M | 0.0% | 98.8% | 100.0% |
2-digit ceiling: 1M and 8M both 100% at 1.78M probs.
- Headline: **1M params does 4-digit addition perfectly** (0 errors / 5000).
- Phase transition visible in the 1M row: 0.2% -> 100.0% between 357k and 1.78M probs.
- Per rule 2: SFT hit 100% -> NO GRPO needed for 4-digit in-distribution.
- Wall time: whole sweep ≈ 50 min on one T4, ~0.7h quota. Quota check needed
  before extrapol push.

## MISSION UPDATE (user clarification)
Part 1 (from-scratch 1M calculator) = good reference, KEEP AND SHIP IT, but
the REAL target: **the smallest real LLM (pretrained, knows language) that is
good at arithmetic.** Plan:
- Stage A: zero-shot+1-shot quiz of real LLMs {TinyStories-1M/3M/33M,
  pythia-14m/70m, SmolLM2-135M} on the same 4-digit task — kernel READY
  (kernels/llmquiz/script.py, needs internet=true to pull HF weights).
- Stage B: SFT ladder — fine-tune the promising ones on the same arithmetic
  lessons, find smallest to clear >=95%.
- Stage C: verdict number + language-retention probe (does it still speak
  English after arithmetic training?) + final report.
Open question worth measuring: BPE tokenizers split numbers badly, so a
14M LLM might LOSE to our 1M calculator — that contrast IS the story.

- 01:0x extrapol v1 CANCELLED at 59min — MY push timeout (-t 3600) was the
  kill (cancel_acknowledged = Kaggle honoring the timeout). No outputs
  survived (jsonl written only at end). FIXES: (a) -t 7200 now, (b) incremental
  extrapolation.jsonl append after EVERY eval row — a kill can no longer eat
  results, (c) trunc_frac metric added — 5-6 digit cot needs >64 tokens, and
  truncation would otherwise masquerade as model failure, (d) make_eval_set
  attempts-guarded (1-digit space is only 100 pairs; unbounded while-loop
  would hang forever).
- 01:3x extrapol v2 PUSHED + RUNNING, watcher armed (140min window).

## Timeline
- 23:0x Phase 0 done: train.py + baselines.py + unit tests all pass. Local
  2-digit 2M-token run proved learning (loss 2.08→1.34, acc 0→~2% on l2r;
  crashed only at final-eval overlap assert — the bug since fixed).
- 23:1x bakeoff v1+v2 FAILED: Kaggle runs kernels as notebooks; sibling import
  `from train import train` not resolvable (ModuleNotFoundError at ~10s).
  sys.path patch also failed. FIX: single-file merged kernels via make_kernel.py
  (train.py core + driver concatenated, syntax-gated). No more sibling imports.
- 23:5x bakeoff v3 COMPLETE (single-file, ~3 min, 6 configs):
  2-digit: cot 16.6% > l2r 15.0% > rtl 8.7% | 4-digit: cot 0.5%, others 0.
  FORMAT = cot (pre-committed rule 1). 1M@10M tok is below phase transition
  for 4-digit — model learns procedure format (see wrong-examples: valid digit
  step chains with wrong sums) but not yet the arithmetic.
- 00:0x sweep PUSHED + RUNNING (11 configs: 3 sizes × {2M,10M,50M} @ 4d cot
  + 2 ceiling 2d @ 50M). Watcher proc_069900fd4eb2 armed.
- Also fixed: extrapol CHECKPOINTS tags now match sweep tag scheme
  (sw_1m_4d_50000k etc, format not in tag); extrapol metadata kernel_sources
  (was dataset_sources — would have failed at push).
- Next: sweep results → extrapol → decide GRPO (rule 2: only if best SFT <90%).

## Kernel IDs
- tiny-math-bakeoff (DONE v3): format winner = cot
- tiny-math-sweep (RUNNING): {1m,5m,8m} × {2M,10M,50M} tok @ 4d + 2d ceiling
- tiny-math-extrapol (pending, rebuilt single-file): ckpt eval at 1..6 digits
