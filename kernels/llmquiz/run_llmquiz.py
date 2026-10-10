#!/usr/bin/env python
"""Stage A — zero-shot / one-shot arithmetic quiz of REAL pretrained LLMs.

Same 4-digit addition task as part 1, same held-out problems (seed 7781), but
the models are pretrained language models with their native BPE tokenizers.
One worked cot example in the prompt (base models need a pattern to imitate).

Reports BOTH:
  strict: parses answer exactly like part 1 (last comma-field of cot line)
  lenient: last integer anywhere in the completion == gold
Primary metric = strict (comparable to part 1's 100% numbers).
Single file — no sibling imports (Kaggle notebook-execution lesson).
"""
import json, os, re, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

MODELS = [
    "roneneldan/TinyStories-1M",     # ~3.7M total
    "roneneldan/TinyStories-3M",     # ~8.3M
    "EleutherAI/pythia-14m",         # ~14M
    "EleutherAI/pythia-70m",         # ~46M
    "roneneldan/TinyStories-33M",    # ~68M
    "HuggingFaceTB/SmolLM2-135M",    # ~135M (base, not instruct)
]
N_EVAL = 2000
MAX_NEW = 96

def cot(a, b):
    sa, sb = str(a).rjust(4, "0"), str(b).rjust(4, "0")
    steps, carry = [], 0
    for i in range(3, -1, -1):
        d1, d2 = int(sa[i]), int(sb[i])
        s = d1 + d2 + carry
        steps.append(f"{d1}+{d2}" + (f"+{carry}" if carry else "") + f"={s}")
        carry = s // 10
    return f"{a}+{b}=\n{','.join(steps)},{a+b}\n"

def make_eval_set(seed, n=2000):
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out

def strict_parse(completion):
    line = completion.split("\n")[0].strip()
    if not line:
        return None
    cand = line.split(",")[-1].strip()
    return cand if cand.isdigit() else None

def lenient_parse(completion):
    nums = re.findall(r"\d+", completion.replace(",", ""))
    return nums[-1] if nums else None

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    pairs = make_eval_set(7781, N_EVAL)   # same seed as part-1 4-digit control
    worked = cot(7382, 1947)              # pattern example (gold computed in code)
    rows = []
    for mid in MODELS:
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(mid, padding_side="left")
            model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32).to(device)
            model.eval()
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            n_params = sum(p.numel() for p in model.parameters())
            vocab = len(tok)
            strict_ok = lenient_ok = 0
            wrong = []
            B = 64
            for i in range(0, len(pairs), B):
                batch = pairs[i:i + B]
                prompts = [worked + f"{a}+{b}=\n" for a, b in batch]
                enc = tok(prompts, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=MAX_NEW,
                                         do_sample=False, pad_token_id=tok.pad_token_id)
                gen = out[:, enc["input_ids"].shape[1]:]
                for j, (a, b) in enumerate(batch):
                    text = tok.decode(gen[j], skip_special_tokens=True)
                    gold = str(a + b)
                    sp, lp = strict_parse(text), lenient_parse(text)
                    if sp == gold:
                        strict_ok += 1
                    if lp == gold:
                        lenient_ok += 1
                    elif len(wrong) < 4:
                        wrong.append((f"{a}+{b}", text[:90].replace("\n", "\\n"), gold))
            row = {"model": mid, "params": n_params, "vocab": vocab,
                   "strict": strict_ok / len(pairs), "lenient": lenient_ok / len(pairs),
                   "n": len(pairs), "wall_min": (time.time() - t0) / 60,
                   "wrong_examples": wrong}
            print(f"{mid}: params={n_params/1e6:.1f}M vocab={vocab} "
                  f"strict={row['strict']:.4f} lenient={row['lenient']:.4f} "
                  f"({row['wall_min']:.1f} min)", flush=True)
        except Exception as e:
            row = {"model": mid, "error": str(e)}
            print(f"{mid}: FAILED {e}", flush=True)
        rows.append(row)
        with open("stage_a_results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        try:
            del model
        except Exception:
            pass
        if device == "cuda":
            torch.cuda.empty_cache()
    print("\n==== STAGE A SUMMARY (4-digit, 1-shot cot pattern) ====", flush=True)
    for r in rows:
        if "error" in r:
            print(f"{r['model']:>34}  ERROR {r['error'][:60]}", flush=True)
        else:
            print(f"{r['model']:>34}  {r['params']/1e6:7.1f}M  strict={r['strict']:.3f}  lenient={r['lenient']:.3f}", flush=True)
    print("part-1 reference: from-scratch 1M model = 1.000 (after training)", flush=True)

if __name__ == "__main__":
    main()
