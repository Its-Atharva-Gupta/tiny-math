#!/usr/bin/env python
"""Stage B — fine-tune REAL pretrained LLMs on the same cot arithmetic lessons
part 1 used, to find the smallest LLM that gets good at addition.

Round 1: every model gets 500k problems (~25M BPE tokens).
Round 2 (adaptive, decided by caller): models with 10% <= strict < 95% get
escalated to 1.78M problems; models already >=95% are done; models <10% are
flagged unless they're the smallest candidate (escalate for fairness).

Full fine-tuning (no LoRA — models are small), fp16 autocast on T4.
Eval: SAME held-out 4-digit set as Stage A (seed 7781), greedy, both
lenient (last integer) and strict (cot-parse) metrics.
Single file — no sibling imports.
"""
import json, os, re, time, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "roneneldan/TinyStories-1M",
    "roneneldan/TinyStories-3M",
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-70m",
    "roneneldan/TinyStories-33M",
    "HuggingFaceTB/SmolLM2-135M",
]
N_PROBLEMS = int(os.environ.get("NPROB", 500_000))
EVAL_N = 2000
SEQ_LEN = 256
BATCH = 64
LR = 6e-5
MAX_NEW = 96
TAG = os.environ.get("STAG", f"b_{N_PROBLEMS//1000}k")

def cot(a, b):
    sa, sb = str(a).rjust(4, "0"), str(b).rjust(4, "0")
    steps, carry = [], 0
    for i in range(3, -1, -1):
        d1, d2 = int(sa[i]), int(sb[i])
        s = d1 + d2 + carry
        steps.append(f"{d1}+{d2}" + (f"+{carry}" if carry else "") + f"={s}")
        carry = s // 10
    return f"{a}+{b}=\n{','.join(steps)},{a+b}\n"

def make_eval_set(seed, n=EVAL_N):
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out

def make_train_problems(seed, n, reject):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
        if (a, b) in reject:
            continue
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

@torch.no_grad()
def evaluate(model, tok, pairs, device):
    model.eval()
    worked = cot(7382, 1947)
    strict_ok = lenient_ok = 0
    wrong = []
    B = 32
    for i in range(0, len(pairs), B):
        batch = pairs[i:i + B]
        prompts = [worked + f"{a}+{b}=\n" for a, b in batch]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for j, (a, b) in enumerate(batch):
            text = tok.decode(gen[j], skip_special_tokens=True)
            gold = str(a + b)
            if strict_parse(text) == gold:
                strict_ok += 1
            if lenient_parse(text) == gold:
                lenient_ok += 1
            elif len(wrong) < 4:
                wrong.append((f"{a}+{b}", text[:90].replace("\n", "\\n"), gold))
    model.train()
    return strict_ok / len(pairs), lenient_ok / len(pairs), wrong

def pack_stream(problems, tok, budget_problems):
    """Tokenize cot problems, concatenate into SEQ_LEN chunks."""
    ids, out, seen = [], [], 0
    for a, b in problems:
        ids.extend(tok(cot(a, b), add_special_tokens=False)["input_ids"])
        seen += 1
        while len(ids) >= SEQ_LEN:
            out.append(ids[:SEQ_LEN])
            ids = ids[SEQ_LEN:]
        if seen >= budget_problems:
            break
    return out, seen

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| problems:", N_PROBLEMS, "| tag:", TAG, flush=True)
    eval_pairs = make_eval_set(7781)
    reject = set(eval_pairs)
    results = []
    for mid in MODELS:
        out_tag = f"{TAG}__{mid.split('/')[-1]}"
        if os.path.exists(f"runs/{out_tag}/summary.json"):
            print(f"skip {out_tag} (done)", flush=True)
            continue
        t0 = time.time()
        os.makedirs(f"runs/{out_tag}", exist_ok=True)
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(mid).to(device)
            model.train()
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            n_params = sum(p.numel() for p in model.parameters())

            train_probs = make_train_problems(0, N_PROBLEMS, reject)
            chunks, used = pack_stream(train_probs, tok, N_PROBLEMS)
            steps = max(1, len(chunks) // BATCH)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                    weight_decay=0.1)
            warm = max(1, int(0.05 * steps))
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: s / warm if s < warm else
                0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(3.14159 * (s - warm) / max(1, steps - warm))).item()))

            best = {"strict": -1.0, "lenient": -1.0}
            log_f = open(f"runs/{out_tag}/curve.csv", "w")
            log_f.write("step,loss,strict,lenient,elapsed_s\n")
            eval_every = max(20, steps // 8)
            idx = 0
            for step in range(1, steps + 1):
                batch = chunks[idx:idx + BATCH]
                idx = (idx + BATCH) % max(1, len(chunks) - BATCH)
                x = torch.tensor(batch, device=device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    logits = model(input_ids=x, labels=x).loss
                opt.zero_grad()
                logits.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                if step % eval_every == 0 or step == steps:
                    st, le, wrong = evaluate(model, tok, eval_pairs, device)
                    el = time.time() - t0
                    log_f.write(f"{step},{logits.item():.4f},{st:.4f},{le:.4f},{el:.0f}\n")
                    log_f.flush()
                    print(f"[{out_tag}] step {step}/{steps} loss {logits.item():.4f} "
                          f"strict {st:.4f} lenient {le:.4f} ({el/60:.1f} min)", flush=True)
                    if st > best["strict"]:
                        best = {"strict": st, "lenient": le, "step": step}
                        torch.save(model.state_dict(), f"runs/{out_tag}/best.pt")
            st, le, wrong = evaluate(model, tok, eval_pairs, device)
            if st > best["strict"]:
                best = {"strict": st, "lenient": le, "step": steps}
                torch.save(model.state_dict(), f"runs/{out_tag}/best.pt")
            # language retention probe: does it still talk?
            model.eval()
            lang_samples = {}
            for p in ["Once upon a time", "The capital of France is"]:
                enc = tok(p, return_tensors="pt").to(device)
                g = model.generate(**enc, max_new_tokens=60, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
                lang_samples[p] = tok.decode(g[0][enc["input_ids"].shape[1]:],
                                             skip_special_tokens=True)
            summary = {"model": mid, "params": n_params, "n_problems": N_PROBLEMS,
                       "pack_tokens": len(chunks) * SEQ_LEN, "best": best,
                       "final_strict": st, "final_lenient": le,
                       "wrong_examples": wrong, "lang_samples": lang_samples,
                       "wall_min": (time.time() - t0) / 60, "tag": out_tag}
            with open(f"runs/{out_tag}/summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            with open("stage_b_results.jsonl", "a") as f:
                f.write(json.dumps(summary) + "\n")
            print(f"[{out_tag}] DONE best={best} final_strict={st:.4f} "
                  f"({summary['wall_min']:.1f} min)", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            with open("stage_b_results.jsonl", "a") as f:
                f.write(json.dumps({"model": mid, "error": str(e), "tag": out_tag}) + "\n")
            print(f"[{mid}] FAILED: {e}", flush=True)
        try:
            del model, opt
        except Exception:
            pass
        if device == "cuda":
            torch.cuda.empty_cache()
    print("\n==== STAGE B SUMMARY ====", flush=True)
    for r in [json.loads(l) for l in open("stage_b_results.jsonl") if l.strip()]:
        if "error" in r:
            print(f"{r.get('tag','?'):>34}  ERROR", flush=True)
        else:
            print(f"{r['tag']:>34}  {r['params']/1e6:7.1f}M  "
                  f"strict={r['best']['strict']:.3f} lenient={r['best']['lenient']:.3f}", flush=True)

if __name__ == "__main__":
    main()
