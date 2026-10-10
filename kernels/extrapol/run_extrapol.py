#!/usr/bin/env python
"""Length generalization: load a trained checkpoint, evaluate at digit lengths
UNSEEN in training (train was d-digit; test at 1..6). Also in-distribution
control. Reports a per-length accuracy table — the honest extrapolation test.

Run for each sweep winner. Needs runs/<tag>/best.pt from the sweep kernel
(mount that kernel as a source, or copy outputs in).
"""
import json, os, glob
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
from train import TinyGPT, SIZES, evaluate, make_eval_set, sample_pair, format_problem

CHECKPOINTS = [
    # (ckpt tag, size, fmt, trained digits) — tags must match run_sweep.py:
    # f"sw_{size}_{digits}d_{tokens//1000}k" (format not in tag)
    ("sw_1m_4d_50000k", "1m", "cot", 4),
    ("sw_5m_4d_50000k", "5m", "cot", 4),
    ("sw_8m_4d_50000k", "8m", "cot", 4),
]

TRAIN_TOKENS = 50_000_000  # all checkpoints above are the 50M-token sweep runs

def replay_train_pairs(seed, digits, budget_tokens):
    """Exact RNG replay of the sweep's training stream: reproduces every (a,b)
    pair it sampled, so the in-distribution control row can exclude seen
    problems (n=5000 eval vs 81M possible 4-digit pairs would otherwise
    contaminate ~2% of the control)."""
    ev = set(make_eval_set(seed + 999, digits, 1000))
    fi = set(make_eval_set(seed + 12345, digits, 5000, exclude=ev))
    reject = ev | fi
    rng = random.Random(seed)
    out, n = set(), 0
    while n < budget_tokens:
        a, b = sample_pair(rng, digits)
        if (a, b) in reject:
            continue
        n += len(format_problem(a, b, a + b, "cot"))
        out.add((a, b))
    return out

def find_ckpt(tag):
    cands = [f"runs/{tag}/best.pt"]
    cands += glob.glob(f"/kaggle/input/**/runs/{tag}/best.pt", recursive=True)
    for p in cands:
        if os.path.exists(p):
            return p
    return None

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for tag, size, fmt, trained_d in CHECKPOINTS:
        path = find_ckpt(tag)
        if path is None:
            print(f"MISSING {tag}, skipping", flush=True)
            continue
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = ck["cfg"]
        m = TinyGPT(cfg["d"], cfg["L"], cfg["H"], 64).to(device)
        m.load_state_dict(ck["model"])
        for d in range(1, 7):
            pairs = make_eval_set(7777 + d, d, 2000)
            excluded = 0
            if d == trained_d:
                # control row: drop any pair the training stream ever sampled
                seen = replay_train_pairs(0, d, TRAIN_TOKENS)
                before = len(pairs)
                pairs = [p for p in pairs if p not in seen]
                excluded = before - len(pairs)
            acc, nc, _, trunc = evaluate(m, pairs, fmt, device,
                                         digits=max(d, trained_d), return_trunc=True)
            rows.append({"tag": tag, "trained_digits": trained_d, "test_digits": d,
                         "acc": acc, "correct": nc, "n": len(pairs),
                         "trunc_frac": trunc, "excluded_seen": excluded})
            print(f"{tag} (trained {trained_d}d) -> test {d}d: {acc:.4f} ({nc}/{len(pairs)}) "
                  f"trunc={trunc:.3f}"
                  + (f" [excluded {excluded} train-seen]" if excluded else ""), flush=True)
            # incremental save: a kill mid-run loses nothing already measured
            with open("extrapolation.jsonl", "a") as f:
                f.write(json.dumps(rows[-1]) + "\n")
        if device == "cuda":
            torch.cuda.empty_cache()
    with open("extrapolation.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("saved extrapolation.json", flush=True)

if __name__ == "__main__":
    main()
