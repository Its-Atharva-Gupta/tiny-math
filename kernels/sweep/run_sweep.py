#!/usr/bin/env python
"""Phase-1 sweep (Kaggle T4): 3 sizes x 3 data scales on 4-digit addition,
plus 2-digit ceiling runs. FORMAT is set from the bake-off winner.

Crash-safe: every run saves runs/<tag>/{curve.csv,best.pt,summary.json}
immediately and appends to results.jsonl, so a later crash loses nothing.
"""
import json, os, time
import sys
# Kaggle executes kernels as notebooks with cwd=/kaggle/working; sibling
# files live in /kaggle/src but that dir is NOT on sys.path. Fix before imports.
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "/kaggle/src"
for _p in (_here, "/kaggle/src"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from argparse import Namespace
from train import train

FORMAT = "cot"  # <- bake-off winner (2d: 16.6% vs l2r 15.0% vs rtl 8.7%; 4d: 0.5% vs 0/0)

CONFIGS = [
    # 4-digit main grid: size x data
    *[(s, FORMAT, 4, t, f"sw_{s}_4d_{t//1000}k")
      for s in ("1m", "5m", "8m")
      for t in (2_000_000, 10_000_000, 50_000_000)],
    # 2-digit ceiling: data-rich easy task
    ("1m", FORMAT, 2, 50_000_000, "sw_1m_2d_50000k"),
    ("8m", FORMAT, 2, 50_000_000, "sw_8m_2d_50000k"),
]

def main():
    print("device:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)
    print("format:", FORMAT, flush=True)
    results = []
    for size, fmt, digits, tokens, tag in CONFIGS:
        if os.path.exists(f"runs/{tag}/summary.json"):   # resume: skip finished
            print(f"skip {tag} (done)", flush=True)
            continue
        t0 = time.time()
        a = Namespace(size=size, fmt=fmt, digits=digits, tokens=tokens, lr=6e-4,
                      batch=128, eval_n=1000, eval_final_n=5000, eval_points=10,
                      seed=0, preset="sweep", local=False, out=f"runs/{tag}", tag=tag)
        s = train(a)
        s["wall_min_run"] = (time.time() - t0) / 60
        results.append(s)
        with open("results.jsonl", "a") as f:
            f.write(json.dumps(s) + "\n")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\n==== SWEEP RESULTS (final acc on n=5000) ====", flush=True)
    for r in sorted(results, key=lambda r: (r["digits"], r["params"])):
        print(f"{r['size']:>3} {r['digits']}d {r['train_tokens']//1000:>6}k tok  "
              f"final={r['final_bestckpt_acc']:.4f}  ({r['wall_min_run']:.1f} min)", flush=True)

if __name__ == "__main__":
    main()
