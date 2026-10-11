#!/usr/bin/env python
"""Stage B round 3 — the two candidates that never got a fair run:

1. TinyStories-3M (8.3M): round 1 scored 0 but under the broken eval prompt
   (worked-example mismatch). Now: same recipe that survived round 1
   (fp16, 6e-5, batch 32), 500k problems, hygiene-fixed eval.
2. pythia-14m (14.1M): NaN'd at step ~450 in BOTH rounds (fp16 AND fp32).
   Recipe v3: LR 2e-5 (3x lower), embedding LR 10x lower (embedding matrix
   is 46% of its params — AdamW spikes there are the prime suspect), no
   weight decay on norms/biases, longer warmup, NaN auto-retry at 5e-6.

Eval hygiene (from round 2): prompt matches training format, independent
tuning (7781) vs final (7782) eval sets. Single file, no sibling imports.
"""
import json, math, os, re, time, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# (model, n_problems, batch, use_fp16, base_lr, tag)
CONFIGS = [
    ("roneneldan/TinyStories-3M", 500_000, 32, True,  6e-5, "r3_ts3m_500k"),
    ("EleutherAI/pythia-14m",     500_000, 32, False, 2e-5, "r3_p14m_500k_v3"),
]
NAN_RETRIES = [5e-6]      # retry ladder if loss goes NaN
EVAL_N_TRAIN = 1000
EVAL_N_FINAL = 2000
SEQ_LEN = 256
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
    for i in range(0, len(pairs), 32):
        batch = pairs[i:i + 32]
        prompts = [f"{a}+{b}=\n" for a, b in batch]   # == training format
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left").to(device)
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

def make_optimizer(model, base_lr):
    """Param-group recipe: damp embeddings (they dominate small-model params),
    no weight decay on 1-D params (norms/biases)."""
    emb_ids = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            emb_ids.update(id(p) for p in module.parameters())
    decay, no_decay, emb = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in emb_ids:
            emb.append(p)
        elif p.dim() < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW([
        {"params": decay, "lr": base_lr, "weight_decay": 0.1},
        {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
        {"params": emb, "lr": base_lr * 0.1, "weight_decay": 0.1},
    ], betas=(0.9, 0.95))

def train_one(model, tok, tag, nprob, batch, use_fp16, lr, chunks, steps,
              eval_tune, eval_final, device, t_total):
    opt = make_optimizer(model, lr)
    warm = max(1, int(0.1 * steps))          # 10% warmup (longer than r2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    best = {"strict": -1.0, "lenient": -1.0, "step": 0}
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
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"NaN loss at step {step}")
        if step % eval_every == 0 or step == steps:
            st, le, wrong = evaluate(model, tok, eval_tune, device, use_fp16)
            el = time.time() - t_total
            print(f"[{tag}] step {step}/{steps} loss {loss.item():.4f} "
                  f"strict {st:.4f} lenient {le:.4f} ({el/60:.1f} min, lr={lr:.1e})", flush=True)
            if st > best["strict"]:
                best = {"strict": st, "lenient": le, "step": step}
                torch.save(model.state_dict(), f"runs/{tag}/best.pt")
    return best

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    eval_tune = make_eval_set(7781, EVAL_N_TRAIN)
    tune_set = set(eval_tune)
    eval_final = [p for p in make_eval_set(7782, EVAL_N_FINAL) if p not in tune_set]
    reject = tune_set | set(eval_final)
    for mid, nprob, batch, use_fp16, lr0, tag in CONFIGS:
        if os.path.exists(f"runs/{tag}/summary.json"):
            print(f"skip {tag}", flush=True)
            continue
        os.makedirs(f"runs/{tag}", exist_ok=True)
        result = None
        for attempt, lr in enumerate([lr0] + NAN_RETRIES):
            t0 = time.time()
            try:
                torch.manual_seed(0)
                tok = AutoTokenizer.from_pretrained(mid)
                model = AutoModelForCausalLM.from_pretrained(mid).to(device)
                model.train()
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                n_params = sum(p.numel() for p in model.parameters())
                chunks = pack_stream(make_train_problems(0, nprob, reject), tok, nprob)
                steps = max(1, len(chunks) // batch)
                best = train_one(model, tok, tag, nprob, batch, use_fp16, lr,
                                 chunks, steps, eval_tune, eval_final, device, t0)
                st, le, wrong = evaluate(model, tok, eval_final, device, use_fp16)
                if st > best["strict"]:
                    best = {"strict": st, "lenient": le, "step": steps}
                    torch.save(model.state_dict(), f"runs/{tag}/best.pt")
                model.eval()
                lang = {}
                for p in ["Once upon a time", "The capital of France is"]:
                    enc = tok(p, return_tensors="pt").to(device)
                    g = model.generate(**enc, max_new_tokens=60, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                    lang[p] = tok.decode(g[0][enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
                result = {"model": mid, "params": n_params, "n_problems": nprob,
                          "batch": batch, "fp16": use_fp16, "lr": lr,
                          "attempts": attempt + 1, "best": best,
                          "final_strict": st, "final_lenient": le,
                          "wrong_examples": wrong, "lang_samples": lang,
                          "wall_min": (time.time() - t0) / 60, "tag": tag}
                break
            except RuntimeError as e:
                if "NaN" in str(e) and attempt < len([lr0] + NAN_RETRIES):
                    print(f"[{tag}] {e} at lr={lr:.1e} -> retrying at "
                          f"{NAN_RETRIES[attempt]:.1e}", flush=True)
                    try:
                        del model, opt
                    except Exception:
                        pass
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    continue
                raise
        if result is None:
            result = {"model": mid, "error": "NaN at all LR attempts", "tag": tag}
        with open(f"runs/{tag}/summary.json", "w") as f:
            json.dump(result, f, indent=2)
        with open("stage_b_round3.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
        if "error" in result:
            print(f"[{tag}] FAILED: {result['error']}", flush=True)
        else:
            print(f"[{tag}] DONE best={result['best']} final={result['final_strict']:.4f} "
                  f"({result['wall_min']:.1f} min)", flush=True)
        try:
            del model
        except Exception:
            pass
        if device == "cuda":
            torch.cuda.empty_cache()
    print("\n==== ROUND 3 SUMMARY ====", flush=True)
    for r in [json.loads(l) for l in open("stage_b_round3.jsonl") if l.strip()]:
        if "error" in r:
            print(f"{r['tag']:>22}  ERROR {r['error']}", flush=True)
        else:
            print(f"{r['tag']:>22}  {r['params']/1e6:6.1f}M  "
                  f"strict={r['best']['strict']:.3f} lr={r['lr']:.0e}", flush=True)

if __name__ == "__main__":
    main()
