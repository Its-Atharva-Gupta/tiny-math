#!/usr/bin/env python
"""Stage C — final verdict: combine ALL results into one report + chart.

Reads: results/*.jsonl (sweep, extrapolation, stage A, stage B round 2)
Writes: verdict.json + verdict_chart.png + prints the final report table.

The headline number: smallest MODEL (from scratch) and smallest REAL LLM
(after fine-tuning) that clears >=95% exact-match on 4-digit addition.
"""
import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = "results"

def load(fp):
    if not os.path.exists(fp):
        return []
    return [json.loads(l) for l in open(fp) if l.strip()]

def main():
    sweep = load(f"{RES}/sweep_results.jsonl")
    extrap = load(f"{RES}/extrapolation.jsonl")
    stage_a = load(f"{RES}/stage_a_results.jsonl")
    stage_b_all = load(f"{RES}/stage_b_all.jsonl")   # all rounds, deduped by tag
    baselines = json.load(open(f"{RES}/baselines.json")) if os.path.exists(f"{RES}/baselines.json") else {}

    print("=" * 74)
    print("FINAL VERDICT — smallest model that can do 4-digit addition")
    print("=" * 74)

    # --- from-scratch ladder
    print("\nFROM SCRATCH (digit vocab, trained):")
    scratch = []
    for r in sweep:
        if r["digits"] == 4:
            scratch.append((r["params"], r["train_tokens"], r["final_bestckpt_acc"]))
    for p, t, a in sorted(scratch):
        flag = " <- PASSES" if a >= 0.95 else ""
        print(f"  {p/1e6:6.2f}M params, {t/1e6:5.1f}M tok: {100*a:6.2f}%{flag}")

    # --- zero-shot LLMs
    print("\nREAL LLMs ZERO-SHOT:")
    for r in stage_a:
        if "error" not in r:
            print(f"  {r['params']/1e6:7.1f}M {r['model'].split('/')[-1]:>16}: "
                  f"strict {100*r['strict']:5.2f}%")

    # --- fine-tuned LLMs (best run per model across all rounds)
    print("\nREAL LLMs FINE-TUNED (best run per model, all rounds):")
    best_per_model = {}
    for r in stage_b_all:
        if "error" in r:
            continue
        m = r["model"]
        if m not in best_per_model or r["best"]["strict"] > best_per_model[m]["best"]["strict"]:
            best_per_model[m] = r
    llm_pass = []
    for r in sorted(best_per_model.values(), key=lambda x: x["params"]):
        acc = r["best"]["strict"]
        flag = " <- PASSES" if acc >= 0.95 else ""
        print(f"  {r['params']/1e6:7.1f}M {r['model'].split('/')[-1]:>16}: "
              f"{100*acc:6.2f}% ({r['n_problems']/1e6:.2f}M problems){flag}")
        if acc >= 0.95:
            llm_pass.append((r["params"], r["model"], acc))

    # --- the verdict
    scratch_pass = [(p, t, a) for p, t, a in scratch if a >= 0.95]
    verdict = {
        "smallest_from_scratch_M": min((p for p, _, _ in scratch_pass), default=None),
        "smallest_llm_M": min((p for p, _, _ in llm_pass), default=None),
        "extrapolation_best": max((r["acc"] for r in extrap if r["test_digits"] != 4), default=None),
        "baselines": {k: v.get("acc") for k, v in baselines.items()},
    }
    print("\n" + "-" * 74)
    if verdict["smallest_from_scratch_M"]:
        print(f"SMALLEST from-scratch model that does 4-digit addition: "
              f"{verdict['smallest_from_scratch_M']/1e6:.2f}M params")
    if verdict["smallest_llm_M"]:
        print(f"SMALLEST real LLM that does it after fine-tuning:     "
              f"{verdict['smallest_llm_M']/1e6:.1f}M params")
    else:
        print("SMALLEST real LLM that does it after fine-tuning:     NONE passed (see round 2)")
    print(f"Best length generalization (unseen digit lengths):      "
          f"{100*(verdict['extrapolation_best'] or 0):.1f}%")
    print("-" * 74)

    with open(f"{RES}/verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    # --- chart: one figure, three panels
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # panel 1: from-scratch phase diagram (accuracy vs data, per size)
    ax = axes[0]
    sizes = {}
    for p, t, a in scratch:
        sizes.setdefault(round(p / 1e6, 2), []).append((t / 1e6, 100 * a))
    for ms, pts in sorted(sizes.items()):
        pts.sort()
        ax.plot([x for x, _ in pts], [y for _, y in pts], "o-", label=f"{ms}M")
    ax.set_xscale("log")
    ax.set_xlabel("practice problems (M, log)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("A. From scratch: capacity buys\nsample efficiency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # panel 2: extrapolation heatmap-ish bars
    ax = axes[1]
    tags = sorted(set(r["tag"] for r in extrap))
    labels = [f"{t.split('_')[1]}" for t in tags]
    for d in (1, 2, 3, 5, 6):
        vals = [100 * next((r["acc"] for r in extrap
                            if r["tag"] == t and r["test_digits"] == d), 0)
                for t in tags]
        ax.bar([f"{l}\nd{d}" for l in labels] if d == 1 else
               [f"d{d}"] * len(tags), vals, alpha=0.4 if d > 1 else 0.9,
               color=plt.cm.viridis(d / 7))
    ax.set_ylim(0, 105)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("B. Length generalization:\n0% at every unseen length")
    ax.grid(alpha=0.3, axis="y")

    # panel 3: LLM comparison (zero-shot vs fine-tuned vs from-scratch best)
    ax = axes[2]
    names, zeros, fts = [], [], []
    for r in stage_a:
        if "error" not in r:
            names.append(r["model"].split("/")[-1].replace("TinyStories-", "TS").replace("SmolLM2-", "Smol"))
            zeros.append(100 * r["strict"])
            ft_vals = [100 * x["best"]["strict"] for x in stage_b_all
                       if x.get("model") == r["model"] and "error" not in x]
            fts.append(max(ft_vals) if ft_vals else 0)
    x = np.arange(len(names))
    ax.bar(x - 0.2, zeros, 0.4, label="zero-shot", color="#4C72B0")
    ax.bar(x + 0.2, fts, 0.4, label="fine-tuned", color="#DD8452")
    ax.axhline(100, color="green", ls="--", lw=1, label="from-scratch 1M (100%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("C. Real LLMs vs our 1M calculator")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("How small can a model be to do arithmetic? — the full story", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{RES}/verdict_chart.png", dpi=150)
    print(f"\nsaved {RES}/verdict.json + {RES}/verdict_chart.png")

if __name__ == "__main__":
    main()
