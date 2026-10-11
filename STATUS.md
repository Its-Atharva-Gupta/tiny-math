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

## STAGE A RESULTS (zero/1-shot, n=2000, 1 worked cot example, complete)
**All 6 real LLMs: strict 0.000, lenient 0.000.** From 3.7M to 134.5M params.
Failure modes by family:
- TinyStories (1M/3M/33M): ignore the task, TELL STORIES ("One day, a 3
  year old boy was walking in the park...")
- pythia-14m: attempts cot format, loops ("3+9=9,9+4=12,9+4=12,...")
- pythia-70m: emits plausible cot chains with wrong digits, then repeats
  the worked example back
- SmolLM2-135M: closest but loops / collapses to "100+100+100..."
Reference: from-scratch 1M (trained) = 1.000. Zero-shot ≠ trained; the LLM
advantage (language) buys nothing on pure arithmetic below 135M.
Saved: results/stage_a_results.jsonl.

## SHIPPED (part 1)
- HF: Atharva1232/tiny-math-1m (model + card + all results + chart), public
- GitHub: Its-Atharva-Gupta/tiny-math (public), pushed clean, token never
  touched git config (one-shot GIT_ASKPASS helper)
- MODEL_CARD.md includes: 100% headline, phase-cliff finding, extrapolation
  0%-everywhere table, bigram/untrained baselines at 0%, contamination-by-
  construction note, positional-pattern failure analysis with examples.

## STAGE B ROUND 1 (500k problems, complete) — NULL RESULT + DIAGNOSIS
All LLMs ~0% after fine-tuning (TinyStories-1M: 0.001 strict = 2/2000 luck;
others 0.000). But curves reveal THREE distinct bugs, not one failure:
- pythia-14m/70m: loss NaN at step 1 (fp16 autocast instability) — models
  were dead on arrival, hence empty generations. NOT a real result.
- TinyStories-1M/3M: loss fell properly (2.83->1.65, 2.30->1.05) — they
  learned the text but eval 0%: (a) BPE splits numbers inconsistently, no
  stable digit mapping; (b) my eval prompt had a worked example models never
  saw in training (format mismatch, my bug).
- 33M/135M: OOM at batch 64 (50k-vocab logits vs 16GB) — never trained.
- Also found: round-1 final eval was same-seed superset of tuning eval ->
  biased. Fixed in round 2 (independent seeds 7781/7782 + overlap assert).
- Side finding: fine-tuning DESTROYS TinyStories language ("Once upon a
  time" -> digit chains). Language-retention probe kept for round 2.

## STAGE B ROUND 2 (RUNNING, watcher proc_9c5edaeb2e6f, -t 18000)
Pre-registered configs:
- TinyStories-1M: 1.78M problems (escalation: smallest LLM candidate gets
  same budget our from-scratch 1M model got)
- pythia-14m/70m: 500k, pure fp32 (NaN fix)
- TinyStories-33M: 500k, batch 16 (OOM fix)
- SmolLM2-135M: 500k, batch 8 (OOM fix)
- All: eval prompt = "a+b=\n" matching training format exactly.
NaN guard aborts a model at first non-finite loss.

## STAGE B ROUND 2 RESULTS (complete, 142min) — BREAKTHROUGH
| LLM | params | problems | strict |
|---|---|---|---|
| **TinyStories-33M** | **68.5M** | 500k | **98.5% PASSES** |
| SmolLM2-135M | 134.5M | 500k | 76.9% |
| TinyStories-1M | 3.7M | 1.78M | 0.2% (loss 1.27->0.74, eval flat 0: BPE wall confirmed at the budget our from-scratch 1M hit 100%) |
| pythia-14m | 14.1M | 500k | NaN @ step 449 even in fp32 — GPT-NeoX instability, not fp16 |
| pythia-70m | 70.4M | 500k | NaN @ step 899 even in fp32 |
- TS-33M wrong answers: gets the sum RIGHT then hallucinates more problems
  ("...15989\\n8984+8984=...") — strict parses first line => 98.5%.
  lenient parses last number => 0. Disclosed, not hidden.
- Language: TS-33M lost English ("Once upon a time" -> digit chains);
  SmolLM2-135M KEPT it ("...a little girl named Lily..."). Capacity helps
  retention. TS-1M also lost English.
- Headline so far: smallest passing real LLM = 68.5M (vs 1.07M from scratch).

## STAGE B ROUND 3 (RUNNING, watcher proc pending, -t 10800)
Two candidates that never got a fair shot:
- TinyStories-3M (8.3M): round-1 0% was under broken eval prompt; now
  round-1-surviving recipe + fixed eval, 500k problems.
- pythia-14m (14.1M): anti-NaN recipe v3 — LR 2e-5, embedding LR x0.1
  (embedding = 46% of its params, spike suspect), no wd on norms/biases,
  10% warmup, NaN auto-retry ladder [5e-6].

## ROUND 3 RESULTS (partial — kernel crashed after TS-3M)
- TinyStories-3M (8.3M) @ 500k: **strict 0.000** — half-learned carry pattern
  but BPE garbage ("cartridge1", "8irs") swallows digits. Below the floor.
- pythia-14m: NaN'd at 2e-5 AND at the 5e-6 retry, then MY retry handler
  crashed (off-by-one: NAN_RETRIES[attempt] with attempt==len). Handler bug
  fixed by design in round 4 (no retry ladder there).
- Verdict math: candidates below 68.5M are now 3.7M (0.2%), 8.3M (0%),
  14.1M (untrainable so far). pythia-14m is the LAST one that could beat
  the 68.5M headline -> round 4.

## ROUND 4 (RUNNING, watcher armed, -t 7200)
Recipe v5 for pythia-14m: freeze embed_in+embed_out (91% of its params,
the NaN suspect — 50277-vocab matrices under full-param AdamW), train only
the ~1.2M transformer blocks at lr 1e-4, grad-norm logged pre-clipping.
If this NaNs too -> documented stability blocker, verdict stands at 68.5M.

## ROUND 4 + LOCAL NaN AUTOPSY (diagnosis in progress)
- Round 4 (frozen "embed*" only): NaN at STEP 2, grad_norm=nan. Log revealed
  my filter froze only embed_in (6.44M) — lm_head (also 6.44M) kept the name
  "lm_head", not caught by "embed" substring.
- Local autopsy (nan_autopsy.py, pythia-14m on CPU, exact Kaggle recipe):
  forward clean (loss 2.42, max|logit| 23), embedding rows used by cot data
  fine (max|w| 0.135), step-1 gradients ALL FINITE (no NaN/inf, total norm
  60.09, top grad 28) -> then ONE AdamW step turns 46/76 tensors NaN.
  Finite grads + finite clip + one optimizer step = NaN weights means an
  optimizer-internals bug (suspect: foreach implementation), not data/LR.
- LR-vs-time pattern across 5 recipes: 1e-4→step2, 6e-5 fp16→step1,
  6e-5 fp32→step449, 2e-5→later, 5e-6→later. Consistent with slow weight
  drift into an overflow threshold, OR two separate mechanisms.
- nan_bisect.py running locally: 4 one-step variants (foreach default /
  foreach=False / wd=0 / SGD) to name the culprit. If foreach=False is
  clean -> that's the Kaggle round-5 fix (one flag).

- nan_bisect RESULT — CASE CLOSED: **param dtype = float16!** transformers
  5.19 loads pythia-14m in its native fp16; Kaggle's transformers 4.x loaded
  fp32. fp16 + AdamW explodes because eps=1e-8 is UNREPRESENTABLE in fp16
  (min normal 6e-5): underflowed second-moment rows divide by ~0 -> inf ->
  NaN weights in ONE step. All 4 bisect variants: AdamW (any wd/foreach)
  = 51 NaN tensors; SGD = clean (no division) — confirms mechanism exactly.
  This explains every instant NaN (rounds 1/3-retry/4 locally). The SLOW
  Kaggle NaNs (step 449 @ 6e-5 fp32) are a separate real instability —
  round 5 recipe addresses both: explicit fp32 + blocks-only + lr 3e-4.
- Skill patched: rlvr-finetuning now carries the fp16/AdamW eps lesson and
  the clip-can't-save-NaN-grads lesson.
- round5_preflight.py running locally: 10-step fp32 blocks-only test; if
  CLEAN -> Kaggle round 5 with same recipe at 500k problems.

## ROUND 5 RESULT (complete, 10min) — NaN SAGA ENDS, REAL LEARNING
pythia-14m blocks-only fp32: **strict 0.642 best / 0.6255 final** @ 500k
problems. Stable the whole run (grad_norm_max 56.7, zero NaN). Same
hallucinate-more-problems failure as TS-33M (gets 8857+7132 right -> 15989
then invents more problems; lenient=0 because last number is hallucination).
- Sub-68.5M ladder now: 3.7M=0.2%, 8.3M=0%, 14.1M=64.2% (blocks-only,
  1.19M trainable), 68.5M=98.5% PASSES, 134.5M=76.9%.
- Per pre-committed escalation rule (10%<=acc<95% -> 1.78M problems):
  ROUND 6 launched — pythia-14m @ 1.78M, same proven recipe.
  If >=95%: headline drops 68.5M -> 14.1M. Else: verdict closes at 68.5M.
- SmolLM2-135M also in the escalation band (76.9%) but ABOVE the 68.5M
  headline — escalating it cannot change the "smallest" answer, skipped
  (quota discipline, reason recorded).

## TOKENIZER ABLATION (user request, RUNNING — kernel tiny-math-tokablation)
One-variable experiment: TinyStories-1M, native BPE scored strict 0.002 @
1.78M problems (control r2_ts1m_1780k). Treatment = SMALLEST tokenizer
change: number runs split PER DIGIT (pre-tok `?\\p{N}+` -> `?\\d`), same
50,257 vocab, SAME embedding matrix, no new tokens, no resize. Everything
else frozen: same checkpoint, same 1.78M problems, same fp16/lr 6e-5/
batch-32 recipe, same eval sets (tune 7781 / final 7782).
- Local validation done: OLD chops `1842 -> ['18','42']`, `9920 -> ['99','20']`
  (inconsistent boundaries — the mechanism we suspected); NEW gives one
  token per digit; 500-string round-trip check passed; alphabet closed,
  no spaces (chunk-interaction assumption holds).
- Compute: digit-split = 1.23x tokens/problem (36.9 -> 45.4), disclosed as
  a consequence of the treatment (same problems, ~1.23x steps).
- PRE-REGISTERED PREDICTIONS: BPE-chunking was THE wall -> big jump
  (>>10%, plausibly 90%+; from-scratch 806K hit 100% on same data);
  tokenization was not it -> <5% (wall = pretraining/capacity).
- Implementation: per-char id dict == patched-pretokenizer for this corpus
  (single-char chunk = BPE identity), asserted in-kernel at startup.

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
