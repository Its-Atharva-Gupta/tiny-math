# tiny-math

**How small can a model be and still learn arithmetic?**

Two answers, measured on the same task (4-digit addition, chain-of-thought,
exact-match on held-out problems):

1. **A from-scratch 1M-parameter transformer scores 100%** (5000/5000) after
   ~1.8M practice problems — no pretraining, no RL, next-token prediction
   only. [Model + full results on Hugging Face](https://huggingface.co/Atharva1232/tiny-math-1m)
2. **But it learned a positional pattern, not the algorithm** — perfect at
   the trained digit-length, 0% at every other length (including easier
   2-digit). Evidence and examples in [MODEL_CARD.md](MODEL_CARD.md).

The second finding is the interesting one: at this scale, in-distribution
perfection ≠ capability. It's also the motivation for the second half of the
project: quizzing and fine-tuning *real pretrained LLMs* (3.7M–135M params)
on the same task, to find the smallest LLM that's genuinely good at math.

## Headline results

| model | 71k problems | 357k problems | 1.78M problems |
|---|---|---|---|
| **1M (from scratch)** | 0.0% | 0.2% | **100.0%** |
| 5M | 0.0% | 95.7% | 100.0% |
| 8M | 0.0% | 98.8% | 100.0% |

![phase diagram](phase_diagram.png)

The transition is a **cliff, not a ramp** (0.2% → 100.0% for the 1M model
between 357k and 1.78M problems) — the sharp data-scale phase transitions
reported in [Teaching Arithmetic to Small Transformers](https://arxiv.org/abs/2307.03381),
reproduced at 1M scale.

**Length generalization** (trained on 4-digit only, n≈2000/row):

| tested on | 1M | 5M | 8M |
|---|---|---|---|
| 1-, 2-, 3-digit | 0% | 0% | 0% |
| **4-digit (trained)** | **100%** | **100%** | **100%** |
| 5-, 6-digit | 0% | 0% | 0% |

Asked `38+86=`, the 1M model answers `3+8=11,0+6+1=7,...` — carries computed
correctly, digits read from the wrong positions. Learned absolute positional
embeddings → fixed-slot pattern learning.

**Baselines**: bigram count model 0.0%, untrained model 0.0% — the floor is
zero; everything above it is learned.

## Layout

```
train.py               # tokenizer + data gen (3 formats) + TinyGPT + train/eval loop
baselines.py           # bigram count model + untrained-model baselines
make_kernel.py         # builds single-file Kaggle kernels (no sibling imports)
plot_phase.py          # phase diagram from results.jsonl
MODEL_CARD.md          # full honest report (findings, limitations, usage)
STATUS.md              # session log: bugs found, decisions, pre-committed rules
kernels/               # Kaggle kernels: bakeoff, sweep, extrapol, llmquiz, stageb
results/               # all raw results (jsonl/json/csv) + chart + card
```

## Reproduce

Local CPU (the 1M model trains in minutes):

```bash
pip install torch numpy
python train.py --size 1m --fmt cot --digits 4 --tokens 50000000 --preset sweep
python baselines.py --digits 4 --fmt cot --tokens 10000000
```

Kaggle (full sweep, one free T4): `make_kernel.py` regenerates the single-file
kernels, then `kaggle kernels push -p kernels/sweep -t 14400`.

## Methodology notes (the parts that matter)

- **Contamination prevented by construction**: both eval sets are built
  before the training stream; the stream rejects any problem appearing in
  either. The extrapolation control row additionally excludes train-seen
  problems via exact RNG replay (24/2000 dropped).
- **Exact match only**, never training loss, never shaped rewards.
- **Format chosen by measurement**, not taste: 3-format bake-off
  (cot > l2r > rtl at 1M scale).
- Eval sizes stated per table (n=5000 headline, n≈2000 extrapolation).
- One seed per cell; no cross-seed error bars. Disclosed, not hidden.

## Citation

```bibtex
@misc{atharvagupta123-tiny-math,
  title = {tiny-math: a 1M-parameter transformer that adds 4-digit numbers (and why it doesn't generalize)},
  author = {Atharva Gupta},
  year = {2026},
  howpublished = {\url{https://github.com/Its-Atharva-Gupta/tiny-math}}
}
```
