# tiny-math — Final Report

*How small can a model be to do arithmetic — and can one small model also talk?*

---

## TL;DR — the three answers

| Question | Answer |
|---|---|
| **1. Smallest model that does arithmetic (from scratch)?** | **805,632 params → 100.0%** on 4-digit addition (5000/5000), after 1.78M practice problems |
| **2. Smallest real pretrained LLM that does arithmetic?** | **3.75M params (TinyStories-1M) → 99.7%**, but only after a one-rule tokenizer fix; with native BPE, it takes 14.1M (pythia-14m, 99.8%) |
| **3. One model that both talks and does math?** | **Yes: the 7:1 rehearsal mix → 89.8% math with language *perfectly* preserved** (perplexity ×0.98 of base) — with a measured dial showing exactly what math costs you at each mix ratio |

Cost: **15.6 of 30 free GPU-hours**, ~13 GPU kernels + local CPU runs, run autonomously overnight and across the next day.

---

## Part 1 — From scratch: how small can a model be?

A decoder-only GPT with a **16-token vocabulary** (digits, `+`, `=`, newline, comma, pad), trained from random initialization on programmatic chain-of-thought addition. No pretraining, no RL.

**The size × data sweep** (n=5000 held-out, strict exact-match):

| actual params | 71k problems | 357k problems | 1.78M problems |
|---|---|---|---|
| **805,632** | 0.0% | 0.2% | **100.0%** |
| 3.58M | 0.0% | 95.7% | 100.0% |
| 6.34M | 0.0% | 98.8% | 100.0% |

Three findings:

1. **The transition is a cliff, not a ramp.** The 805K model sits at 0.2% with 357K problems and hits 100.0% with 1.78M — reproducing the sharp phase transitions of *Teaching Arithmetic to Small Transformers* (arXiv:2307.03381) at sub-million scale. Bigger models cross the cliff on less data: capacity buys sample efficiency.
2. **The floor is genuinely zero.** A bigram count model (next-char probabilities from 10M training tokens) and an untrained model both score **0.0%**. Every point above zero is learned.
3. **Perfect in-distribution ≠ learning the algorithm.** All three models: **100% at the trained 4-digit length, 0% at every other length** (1-, 2-, 3-, 5-, 6-digit, n≈2000 each, train-seen problems excluded by exact RNG replay). Asked `38+86=`, the model answers `3+8=11,0+6+1=7,…` — carries computed correctly, digits read from 4-digit-trained positions. Learned absolute position embeddings → fixed-slot choreography, not arithmetic. Zero truncation measured, so these are genuine errors.

Format choice was measured, not assumed: a 3-format bake-off at 1M scale picked chain-of-thought (16.6%) over left-to-right (15.0%) over right-to-left (8.7%).

## Part 2 — Real LLMs: the tokenizer is the whole game

**Zero-shot, every real LLM scores 0.0%** (n=2000, one worked example in the prompt): TinyStories-1M (3.7M), TinyStories-3M (8.3M), pythia-14m (14.1M), TinyStories-33M (68.5M), pythia-70m (70.4M), SmolLM2-135M (134.5M). Failure modes: the TinyStories models *tell stories* ("One day, a 3 year old boy was walking in the park…"), pythia loops digit fragments, SmolLM2 collapses to `100+100+…`.

**Fine-tuned on the same 1.78M-problem data** (native BPE tokenizers):

| LLM | params | accuracy |
|---|---|---|
| TinyStories-1M | 3.7M | 0.2% |
| TinyStories-3M | 8.3M | 0.0% |
| **pythia-14m** | **14.1M** | **99.8%** (only its 1.19M transformer blocks trained; vocab frozen) |
| TinyStories-33M | 68.5M | 98.5% |
| SmolLM2-135M | 134.5M | 76.9% |
| pythia-70m | 70.4M | NaN (training instability, not evaluated) |

**The closing experiment — a one-rule tokenizer ablation.** TinyStories-1M, same checkpoint, same data, same recipe; the only change: number runs tokenize **per digit** instead of whole chunks (same 50,257-entry vocab, same embedding matrix, no new tokens):

| tokenizer | accuracy |
|---|---|
| native BPE (`1842 → ['18','42']`, `9920 → ['99','20']` — inconsistent boundaries) | **0.2%** |
| digit-split (`1842 → ['1','8','4','2']`) | **99.7%** |

**A 500× swing from one pre-tokenizer rule.** The BPE floor was never about capacity or pretraining — it was literally the tokenizer chopping numbers at inconsistent boundaries, making a stable digit↔position mapping impossible. pythia-14m's native-BPE success at 14.1M is best read as The Pile's math/code exposure grinding through that inconsistency at larger scale.

**The NaN forensics (pythia-14m saga):** five training recipes died with NaN before a local autopsy found the real cause — transformers 5.x loads checkpoints in their native **float16**, and AdamW's `eps=1e-8` is unrepresentable in fp16 (min normal 6×10⁻⁵), so one optimizer step explodes (`g/(0+eps)` → inf). Proof: finite loss, finite gradients, one step, 46/76 tensors NaN; SGD on the same batch clean (no division). Fix: explicit fp32 + frozen vocab matrices → stable, 99.8%.

## Part 3 — Talk AND arithmetic: the mix-ratio dial

Full fine-tuning on pure math **erases language** (TinyStories-1M at 99.7% math answers "Once upon a time" with digit chains). Seven configurations tested on the same 3.75M model:

| recipe | math (strict) | language (story PPL vs base 12.35) | verdict |
|---|---|---|---|
| 100% math | 99.7% | ❌ dead | can add, can't talk |
| 1:1 math:story mix | 1.4% | ✅ ×0.90 | talks, can't add |
| 3:1 mix | 61.5% | ✅ ×0.92 | dial moving |
| **7:1 mix (A3)** | **89.8%** | ✅ **×0.98** | **delivered model** |
| 7:1 → math-only tail (anneal) | 99.3% | ❌ dead (×586k) | tail undoes anchoring |
| LoRA r=8 on frozen base | 0.0% | ❌ dead (×756k) | too weak for math, strong enough to break talk |
| LoRA r=32, gentler LR | 0.05% | ❌ dead (×397k) | closed: 2/2 LoRA configs fail both |

**Findings:** (1) the math↔language tradeoff is a smooth, tunable dial — the ratio of math to story chunks in training; language survives at every mix below 100% math, with perplexity even *better* than base. (2) **"Frozen base preserves language" (the standard LoRA intuition) is wrong at this scale** — the adapter perturbs outputs enough to destroy language while never learning the task, at two very different ranks and LRs. (3) Annealing to a math-only tail is as destructive as pure-math training — the last gradients dominate the final weights.

**Delivered model: A3 (7:1 rehearsal mix).** 89.8% strict on held-out 4-digit addition, fluent English ("there was a little girl named Lily. She loved to play outside…"), story perplexity slightly *better* than the untouched base. The pre-registered 95% bar was not cleared *with language intact*; the full curve above documents why and what the next lever would be (a staged 15:1 experiment is written and ready, deliberately not run).

## Honest disclosures

- **Metric:** strict exact-match on the FIRST line of the chain-of-thought completion. Both flagship models share a charming failure: they answer **correctly**, then hallucinate additional problems (`…15989\n5885+5690=…`); a "last number in completion" metric scores these 0. Disclosed with examples in the raw results.
- **Contamination prevented by construction:** eval sets built before training streams; streams reject every eval pair; extrapolation control excludes train-seen pairs by exact RNG replay; tuning/final eval sets on independent seeds with overlap asserts.
- **12 bugs found and fixed across the project** — full round-by-round record in `STATUS.md`. Highlights: an eval-set leak caught by assert before any GPU burn; two kernels lost to Kaggle's notebook-execution import rules (→ single-file kernel builder); my own push timeout killing a run at 59 min; the fp16/AdamW bomb; an eval prompt that didn't match training format (round-1 Stage B null results, all re-run); a published param count corrected from 1.07M to the true 805,632.
- **One seed per cell**, no cross-seed error bars; eval n=5000 (Part 1 headline) / n=2000 (everything else).
- **Nothing was dropped for being inconvenient** — every null result, crash, and wrong prediction (the LoRA hypothesis, the anneal hypothesis) is in this report and in `results/`.

## Artifacts

| artifact | link |
|---|---|
| Code, results, session log | github.com/Its-Atharva-Gupta/tiny-math |
| The 805K model + verdict chart + all raw results | huggingface.co/Atharva1232/tiny-math-1m |
| Verdict document | `VERDICT.md` |
| Model card (Part 1) | `MODEL_CARD.md` |
| Full session log with every bug/decision | `STATUS.md` |
| Reproduce: from-scratch arm | `train.py`, `baselines.py` |
| Reproduce: all GPU runs | `kernels/` (single-file, `make_kernel.py` builds them) |
| Reproduce: final tables + chart | `verdict.py` |

**Cost:** 15.62 / 30 weekly GPU-hours on one free Kaggle T4, plus two local CPU training runs.
