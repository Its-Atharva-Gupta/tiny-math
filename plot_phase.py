#!/usr/bin/env python
"""Phase diagram: accuracy vs (train tokens) per model size, 4-digit.
Reads results.jsonl from the sweep kernel output dir; also overlays the
baselines (bigram / untrained = 0%) as the floor line.
"""
import json, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(patterns):
    rows = []
    for pat in patterns:
        for fp in glob.glob(pat):
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows

def main():
    rows = load(sys.argv[1:] or ["results.jsonl"])
    main4 = [r for r in rows if r["digits"] == 4 and r["tag"].startswith("sw_")]
    ceil2 = [r for r in rows if r["digits"] == 2 and r["tag"].startswith("sw_")]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"1m": "#4C72B0", "5m": "#DD8452", "8m": "#55A868"}
    for size in ("1m", "5m", "8m"):
        pts = sorted([r for r in main4 if r["size"] == size],
                     key=lambda r: r["train_tokens"])
        if not pts:
            continue
        xs = [r["train_tokens"] / 1e6 for r in pts]
        ys = [100 * r["final_bestckpt_acc"] for r in pts]
        ax.plot(xs, ys, "o-", color=colors[size],
                label=f"{size} ({pts[0]['params']/1e6:.1f}M params)")
    ax.axhline(0, color="gray", ls=":", lw=1, label="bigram / untrained baseline")
    for r in ceil2:
        ax.annotate(f"2-digit ceiling: {100*r['final_bestckpt_acc']:.0f}%",
                    (r["train_tokens"] / 1e6, 100 * r["final_bestckpt_acc"]),
                    textcoords="offset points", xytext=(8, -12), fontsize=8,
                    color=colors.get(r["size"], "black"))
    ax.set_xscale("log")
    ax.set_xlabel("training tokens (millions, log)")
    ax.set_ylabel("exact-match accuracy on 4-digit addition (%)")
    ax.set_title(f"How small can a model be to add 4-digit numbers?\n"
                 f"(format={main4[0]['fmt'] if main4 else '?'}, eval n=5000, from-scratch GPT)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("phase_diagram.png", dpi=150)
    print("saved phase_diagram.png")
    for r in sorted(rows, key=lambda r: (r["digits"], r["params"], r["train_tokens"])):
        print(f"{r['tag']:>22}  {r['final_bestckpt_acc']:.4f}")

if __name__ == "__main__":
    main()
