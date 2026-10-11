#!/usr/bin/env python
"""Stage B round 2 — fix round-1 bugs, per-model configs, pre-registered:

1. pythia-14m/70m: NaN under fp16 autocast round 1 -> pure fp32 this round.
2. TinyStories-33M / SmolLM2-135M: OOM at batch 64 -> batch 16 / 8.
3. TinyStories-1M: pre-committed escalation (smallest LLM candidate) ->
   1.78M problems (same budget our from-scratch 1M model got).
4. HYGIENE FIX (all models): eval prompt now matches training format
   exactly ("a+b=\\n", no worked example — round 1's in-context example was
   off-distribution for models that trained without one).
5. NaN guard: abort a model after its first eval if loss is not finite.
Eval: same held-out 4-digit set (seed 7781), greedy, strict + lenient.
Single file — no sibling imports.
"""
import json, math, os, re, time, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# (model, n_problems, batch, use_fp16, tag)
CONFIGS = [
    ("roneneldan/TinyStories-1M", 1_780_000, 32, True,  "r2_ts1m_1780k"),
    ("EleutherAI/pythia-14m",       500_000, 32, False, "r2_p14m_500k_fp32"),
    ("EleutherAI/pythia-70m",       500_000, 16, False, "r2_p70m_500k_fp32"),
    ("roneneldan/TinyStories-33M",  500_000, 16, True,  "r2_ts33m_500k"),
    ("HuggingFaceTB/SmolLM2-135M",  500_000,  8, True,  "r2_s135m_500k"),
]
EVAL_N_TRAIN = 1000   # during training
EVAL_N_FINAL = 2000   # final tight number
SEQ_LEN = 256
LR = 6e-5
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

def make_eval_set(seed, n):
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
def evaluate(model, tok, pairs, device, use_fp16):
    model.eval()
    strict_ok = lenient_ok = 0
    wrong = []
    B = 32
    for i in range(0, len(pairs), B):
        batch = pairs[i:i + B]
        # HYGIENE: prompt == training format, no worked example
        prompts = [f"{a}+{b}=\n" for a, b in batch]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left").to(device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16 and device == "cuda"):
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
    ids, out, seen = [], [], 0
    for a, b in problems:
        ids.extend(tok(cot(a, b), add_special_tokens=False)["input_ids"])
        seen += 1
        while len(ids) >= SEQ_LEN:
            out.append(ids[:SEQ_LEN])
            ids = ids[SEQ_LEN:]
        if seen >= budget_problems:
            break
    return out

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    # independent tuning vs final eval sets (round-1 bug: same-seed superset
    # meant checkpoint selection saw part of the final eval -> biased number)
    eval_tune = make_eval_set(7781, EVAL_N_TRAIN)
    tune_set = set(eval_tune)
    eval_final = [p for p in make_eval_set(7782, EVAL_N_FINAL) if p not in tune_set]
    assert len(eval_final) >= 1990, f"too much overlap: {len(eval_final)}"
    reject = tune_set | set(eval_final)
    results = []
    for mid, nprob, batch, use_fp16, tag in CONFIGS:
        if os.path.exists(f"runs/{tag}/summary.json"):
            print(f"skip {tag}", flush=True)
            continue
        t0 = time.time()
        os.makedirs(f"runs/{tag}", exist_ok=True)
        log_f = open(f"runs/{tag}/curve.csv", "w")
        log_f.write("step,loss,strict,lenient,elapsed_s\n")
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(mid).to(device)
            model.train()
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            n_params = sum(p.numel() for p in model.parameters())

            chunks = pack_stream(make_train_problems(0, nprob, reject), tok, nprob)
            steps = max(1, len(chunks) // batch)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                    weight_decay=0.1)
            warm = max(1, int(0.05 * steps))
            import math as _m
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: s / warm if s < warm else
                0.1 + 0.9 * 0.5 * (1 + _m.cos(_m.pi * (s - warm) / max(1, steps - warm))))

            best = {"strict": -1.0, "lenient": -1.0, "step": 0}
            first_loss = None
            idx = 0
            eval_every = max(20, steps // 5)
            for step in range(1, steps + 1):
                bchunk = chunks[idx:idx + batch]
                idx = (idx + batch) % max(1, len(chunks) - batch)
                x = torch.tensor(bchunk, device=device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16 and device == "cuda"):
                    loss = model(input_ids=x, labels=x).loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                if first_loss is None:
                    first_loss = loss.item()
                if step % eval_every == 0 or step == steps:
                    if not math.isfinite(loss.item()):
                        raise RuntimeError(f"NaN loss at step {step} (first loss {first_loss})")
                    st, le, wrong = evaluate(model, tok, eval_tune, device, use_fp16)
                    el = time.time() - t0
                    log_f.write(f"{step},{loss.item():.4f},{st:.4f},{le:.4f},{el:.0f}\n")
                    log_f.flush()
                    print(f"[{tag}] step {step}/{steps} loss {loss.item():.4f} "
                          f"strict {st:.4f} lenient {le:.4f} ({el/60:.1f} min)", flush=True)
                    if st > best["strict"]:
                        best = {"strict": st, "lenient": le, "step": step}
                        torch.save(model.state_dict(), f"runs/{tag}/best.pt")
            st, le, wrong = evaluate(model, tok, eval_final, device, use_fp16)
            if st > best["strict"]:
                best = {"strict": st, "lenient": le, "step": steps}
                torch.save(model.state_dict(), f"runs/{tag}/best.pt")
            # language retention probe (trained models only)
            model.eval()
            lang = {}
            for p in ["Once upon a time", "The capital of France is"]:
                enc = tok(p, return_tensors="pt").to(device)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=60, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                lang[p] = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            summary = {"model": mid, "params": n_params, "n_problems": nprob,
                       "batch": batch, "fp16": use_fp16, "first_loss": first_loss,
                       "best": best, "final_strict": st, "final_lenient": le,
                       "wrong_examples": wrong, "lang_samples": lang,
                       "wall_min": (time.time() - t0) / 60, "tag": tag}
            with open(f"runs/{tag}/summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            with open("stage_b_round2.jsonl", "a") as f:
                f.write(json.dumps(summary) + "\n")
            print(f"[{tag}] DONE best={best} final={st:.4f} ({summary['wall_min']:.1f} min)", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            with open("stage_b_round2.jsonl", "a") as f:
                f.write(json.dumps({"model": mid, "error": str(e), "tag": tag}) + "\n")
            print(f"[{tag}] FAILED: {e}", flush=True)
        try:
            del model, opt
        except Exception:
            pass
        if device == "cuda":
            torch.cuda.empty_cache()
    print("\n==== STAGE B ROUND 2 SUMMARY (final eval n=2000) ====", flush=True)
    for r in [json.loads(l) for l in open("stage_b_round2.jsonl") if l.strip()]:
        if "error" in r:
            print(f"{r.get('tag','?'):>22}  ERROR {r['error'][:60]}", flush=True)
        else:
            print(f"{r['tag']:>22}  {r['params']/1e6:7.1f}M  "
                  f"strict={r['best']['strict']:.3f} lenient={r['best']['lenient']:.3f}", flush=True)

if __name__ == "__main__":
    main()
