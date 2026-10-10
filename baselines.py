"""Baselines: (1) bigram count model, (2) untrained TinyGPT.

The bigram is the floor: pure counts P(next|prev) from the same training
stream the neural models see, greedy decode, same exact-match eval. If the
1M model can't beat this, it hasn't learned anything.

Pure stdlib+numpy for the bigram; torch needed only for the untrained model.
"""
import argparse, json, random, time
from collections import defaultdict

import numpy as np

from train import (V, TOK, PAD, enc, dec, format_problem, parse_answer,
                   max_new_for, make_eval_set, make_stream, sample_pair,
                   TinyGPT, SIZES, count_params, evaluate)

import torch

def train_bigram(seed, digits, fmt, budget_tokens):
    counts = np.ones((V, V))  # add-1 smoothing
    for chunk in make_stream(seed, digits, fmt, set(), budget_tokens):
        for a, b in zip(chunk, chunk[1:]):
            counts[a, b] += 1
    probs = counts / counts.sum(1, keepdims=True)
    return probs

def eval_bigram(probs, eval_pairs, fmt, digits=4):
    mx = max_new_for(fmt, digits) + 8
    correct, wrong_ex = 0, []
    for a, b in eval_pairs:
        prompt = f"{a}+{b}=\n"
        cur = enc(prompt)[-1]
        gen = []
        for _ in range(mx):
            nxt = int(np.argmax(probs[cur]))
            if nxt == TOK["\n"]:
                break
            gen.append(nxt)
            cur = nxt
        pred = parse_answer(dec(gen), fmt)
        gold = str(a + b)
        if pred == gold:
            correct += 1
        elif len(wrong_ex) < 5:
            wrong_ex.append((prompt.replace("\n", "\\n"), dec(gen), gold))
    return correct / len(eval_pairs), correct, wrong_ex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits", type=int, default=4)
    ap.add_argument("--fmt", default="l2r", choices=["l2r", "rtl", "cot"])
    ap.add_argument("--tokens", type=int, default=10_000_000)
    ap.add_argument("--eval_final_n", type=int, default=5000)
    ap.add_argument("--size", default="1m", choices=list(SIZES))
    ap.add_argument("--out", default="runs/baselines.json")
    a = ap.parse_args()

    eval_pairs = make_eval_set(999, a.digits, a.eval_final_n)
    results = {}

    # 1. bigram
    t0 = time.time()
    probs = train_bigram(0, a.digits, a.fmt, a.tokens)
    acc, nc, wrong = eval_bigram(probs, eval_pairs, a.fmt, digits=a.digits)
    results["bigram"] = {"acc": acc, "correct": nc, "n": len(eval_pairs),
                         "fmt": a.fmt, "train_tokens": a.tokens,
                         "wall_min": (time.time() - t0) / 60, "wrong_examples": wrong}
    print(f"bigram({a.fmt}, {a.digits}d): {acc:.4f} ({nc}/{len(eval_pairs)})", flush=True)

    # 2. untrained model
    cfg = SIZES[a.size]
    m = TinyGPT(cfg["d"], cfg["L"], cfg["H"], 64)
    acc, nc, wrong = evaluate(m, eval_pairs, a.fmt, "cpu", digits=a.digits)
    results[f"untrained_{a.size}"] = {"acc": acc, "correct": nc, "n": len(eval_pairs),
                                      "params": count_params(m), "fmt": a.fmt,
                                      "wrong_examples": wrong}
    print(f"untrained_{a.size}: {acc:.4f} ({nc}/{len(eval_pairs)})", flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
