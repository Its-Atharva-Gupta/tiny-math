#!/usr/bin/env python
"""Format bake-off (Kaggle T4): 1M model x {l2r, rtl, cot} x {2-digit, 4-digit},
10M tokens each. Picks the format for the Phase-1 sweep.

Crash-safe: each run's summary is appended to results.jsonl immediately, and
train() writes runs/<tag>/{curve.csv,best.pt,summary.json} as it goes.
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

CONFIGS = [
    ("1m", "l2r", 2, 10_000_000, "bo_1m_l2r_2d"),
    ("1m", "rtl", 2, 10_000_000, "bo_1m_rtl_2d"),
    ("1m", "cot", 2, 10_000_000, "bo_1m_cot_2d"),
    ("1m", "l2r", 4, 10_000_000, "bo_1m_l2r_4d"),
    ("1m", "rtl", 4, 10_000_000, "bo_1m_rtl_4d"),
    ("1m", "cot", 4, 10_000_000, "bo_1m_cot_4d"),
]

def main():
    print("device:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)
    results = []
    for size, fmt, digits, tokens, tag in CONFIGS:
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
    print("\n==== BAKEOFF RESULTS (final acc on n=5000) ====", flush=True)
    for r in sorted(results, key=lambda r: -r["final_bestckpt_acc"]):
        print(f"{r['fmt']:>4} {r['digits']}d  final={r['final_bestckpt_acc']:.4f}  "
              f"(tuning={r['best_tuning_acc']:.4f} @ step {r['best_step']}, "
              f"{r['wall_min_run']:.1f} min)", flush=True)

if __name__ == "__main__":
    main()
