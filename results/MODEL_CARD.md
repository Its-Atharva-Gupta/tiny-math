# tiny-math-1m: a 1M-parameter transformer that adds 4-digit numbers

**The smallest model in this repo (1,065,088 parameters) solves 4-digit addition
perfectly — 5000/5000 held-out problems, zero errors — given ~1.8M practice
problems. It has a 16-token vocabulary (digits 0-9, `+`, `=`, newline, comma,
pad) and was trained from random initialization. No pretraining, no
demonstrations, no RL. Plain next-token prediction.**

## The finding this repo exists to report

Three model sizes (1M / 5M / 8M) × three data scales (71k / 357k / 1.78M
practice problems), one task (4-digit addition, chain-of-thought digit steps),
exact-match on held-out problems:

| model | 71k probs | 357k probs | 1.78M probs |
|---|---|---|---|
| **1M** | 0.0% | 0.2% | **100.0%** |
| 5M | 0.0% | 95.7% | 100.0% |
| 8M | 0.0% | 98.8% | 100.0% |

Two things stand out:

1. **The transition is a cliff, not a ramp.** The 1M model sits at 0.2% with
   357k problems and hits 100.0% with 1.78M. No gradual climb — consistent
   with the sharp data-scale phase transitions reported in *Teaching Arithmetic
   to Small Transformers* (arXiv:2307.03381).
2. **Bigger models need less practice.** 5M and 8M cross their thresholds by
   357k problems; 1M needs ~5× more data. Capacity buys sample efficiency.

## What it learned (and what it didn't)

The model writes step-by-step carry chains, e.g. for `1842+2957`:
`2+7=9,4+5=9,8+9=17,1+2+1=4,4799` — carries computed correctly, digits
aligned right-to-left. The procedure is real.

**But it did not learn the addition algorithm.** Quizzed on digit lengths it
never trained on (n≈2000 per row, greedy, same format):

| tested on | 1M | 5M | 8M |
|---|---|---|---|
| 1-digit | 0% | 0% | 0% |
| 2-digit | 0% | 0% | 0% |
| 3-digit | 0% | 0% | 0% |
| **4-digit (trained)** | **100%** | **100%** | **100%** |
| 5-digit | 0% | 0% | 0% |
| 6-digit | 0% | 0% | 0% |

Asked `38+86=`, it answers `3+8=11,0+6+1=7,0+6=6,5+8=13,13671` — the carry
logic runs, but digits are read from *4-digit-trained positions*. Zero
truncation was measured at every length, so these are genuine wrong answers,
not context overflows. This is the known failure mode of learned absolute
positional embeddings: the model learned "the choreography at fixed slots",
not the arithmetic. Fine for the headline claim (perfect in-distribution at
1M params), a real limitation for any generalization claim.

## Baselines (the floor is zero)

Both naive baselines score **0.0%** on this task (n=5000):
- **bigram count model** (next-char probabilities from 10M training tokens)
- **untrained 1M model** (random weights, greedy decode)

Any of the numbers above is therefore entirely learned.

## Honest limitations

- **Single task, single format.** Addition only, one chain-of-thought format
  (chosen from a 3-format bake-off: cot 16.6% > left-to-right 15.0% >
  right-to-left 8.7% at 1M/10M tokens, 2-digit). No claims about other tasks.
- **No length generalization** (table above). Not a bug in evaluation: the
  control row at the trained length re-evaluated to 100% with train-seen
  problems excluded (24 of 2000 dropped, exact RNG replay of the training
  stream).
- **Contamination is prevented by construction**, not by dedup-after-the-fact:
  both eval sets are built before the training stream, and the stream rejects
  any (a,b) pair appearing in either. Verified programmatically.
- **Eval n=5000** for headline numbers (±0.6pp at 95%); extrapolation rows
  n≈2000 (±2.2pp at 95%). Where a row shows 0% or 100% the CI is moot.
- **One seed, one run per cell.** No error bars across seeds.

## Format (what the training text looks like)

```
1842+2957=
2+7=9,4+5=9,8+9=17,1+2+1=4,4799
```
Digits of each column added right-to-left, carries folded into the next step,
final answer last (comma-separated), single trailing newline.

## Files

- `model.safetensors` / `best.pt` — trained weights (fp32)
- `curve.csv` — training loss + eval accuracy per checkpoint
- `summary.json` — final metrics, config, wrong examples
- Full sweep / bake-off / extrapolation results: see repo `results/`

## Training details

| | |
|---|---|
| Architecture | decoder-only GPT, d=128, 4 layers, 4 heads, ctx=64, learned absolute positions |
| Vocab | 16 (digits + `+ - = \n ,` + pad); embedding table = 2,048 params (~0.2% of model) |
| Params | 1,065,088 |
| Data | generated on the fly, 1,785,714 problems (~50M tokens), zero eval overlap by construction |
| Optimizer | AdamW, lr 6e-4, β=(0.9,0.95), wd 0.1, 5% warmup → cosine to 10%, grad clip 1.0 |
| Eval | greedy decode, strict exact match, n=5000, best checkpoint by tuning-split accuracy |
| Hardware | 1× Tesla T4 (Kaggle), 2.6 min wall |
| Frameworks | PyTorch 2.x, no transformers/peft/TRL needed |

## Intended use

Educational / research reference point: **a floor measurement for "how small
can a model be and still learn arithmetic"** — and a cautionary tale that
in-distribution perfection at this scale is positional pattern-learning, not
algorithm learning. Compare against real pretrained LLMs before drawing
conclusions about pretraining vs from-scratch at the 1M–135M scale.

```python
import torch
from train import TinyGPT, enc, dec, TOK  # train.py in this repo

ck = torch.load("best.pt", weights_only=False)
m = TinyGPT(**{k: ck["cfg"][k] for k in ("d", "L", "H")}, ctx=64)
m.load_state_dict(ck["model"]); m.eval()

def add(a, b):
    ids = torch.tensor([enc(f"{a}+{b}=\n")])
    for _ in range(16):
        nxt = m(ids)[:, -1].argmax(-1, keepdim=True)
        if nxt.item() == TOK["\n"]: break
        ids = torch.cat([ids, nxt], 1)
    text = dec(ids[0, len(f"{a}+{b}=\n"):].tolist())
    return text.split(",")[-1]  # cot-format answer

print(add(1842, 2957))  # 4799
```
