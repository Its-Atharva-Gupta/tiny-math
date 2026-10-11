#!/usr/bin/env python
"""Stage B round 4 — final attempt for pythia-14m (14.1M), the last
candidate below 68.5M that could hold the headline.

Four recipes NaN'd already (fp16/fp32 x 6e-5/2e-5/5e-6). Recipe v5:
FREEZE embeddings + lm_head (they are 91% of its params and the prime NaN
suspect: 50277-vocab matrices at full-param AdamW) and train only the ~1.2M
transformer blocks at lr 1e-4. Gradient-norm logged before clipping so any
spike is visible. Fixed retry handler (round-3 crash was an off-by-one in
my own handler, not the model).

If this NaNs too -> documented training-stability blocker, verdict stands.
Single file, no sibling imports.
"""
import json, math, os, re, time, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "EleutherAI/pythia-14m"
NPROB = 500_000
BATCH = 32
SEQ_LEN = 256
LR = 1e-4
MAX_NEW = 96
EVAL_N_TRAIN, EVAL_N_FINAL = 1000, 2000

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

def strict_parse(c):
    line = c.split("\n")[0].strip()
    if not line:
        return None
    cand = line.split(",")[-1].strip()
    return cand if cand.isdigit() else None

def lenient_parse(c):
    nums = re.findall(r"\d+", c.replace(",", ""))
    return nums[-1] if nums else None

@torch.no_grad()
def evaluate(model, tok, pairs, device):
    model.eval()
    s_ok = l_ok = 0
    wrong = []
    for i in range(0, len(pairs), 32):
        batch = pairs[i:i + 32]
        enc = tok([f"{a}+{b}=\n" for a, b in batch], return_tensors="pt",
                  padding=True, padding_side="left").to(device)
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for j, (a, b) in enumerate(batch):
            text = tok.decode(gen[j], skip_special_tokens=True)
            gold = str(a + b)
            if strict_parse(text) == gold:
                s_ok += 1
            if lenient_parse(text) == gold:
                l_ok += 1
            elif len(wrong) < 4:
                wrong.append((f"{a}+{b}", text[:90].replace("\n", "\\n"), gold))
    model.train()
    return s_ok / len(pairs), l_ok / len(pairs), wrong

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = "r4_p14m_frozenemb"
    print("device:", device, "| recipe: frozen embeddings+lm_head, lr", LR, flush=True)
    os.makedirs(f"runs/{tag}", exist_ok=True)
    eval_tune = make_eval_set(7781, EVAL_N_TRAIN)
    tune_set = set(eval_tune)
    eval_final = [p for p in make_eval_set(7782, EVAL_N_FINAL) if p not in tune_set]
    reject = tune_set | set(eval_final)

    t0 = time.time()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID).to(device)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    n_params = sum(p.numel() for p in model.parameters())

    # freeze embedding + lm_head (GPTNeoX: .embed_in / .embed_out)
    frozen = trained = 0
    for name, p in model.named_parameters():
        if "embed" in name:
            p.requires_grad = False
            frozen += p.numel()
        else:
            trained += p.numel()
    print(f"params: total={n_params/1e6:.1f}M frozen={frozen/1e6:.2f}M "
          f"training={trained/1e6:.2f}M", flush=True)
    model.train()

    ids, chunks, seen = [], [], 0
    for a, b in make_train_problems(0, NPROB, reject):
        ids.extend(tok(cot(a, b), add_special_tokens=False)["input_ids"])
        seen += 1
        while len(ids) >= SEQ_LEN:
            chunks.append(ids[:SEQ_LEN])
            ids = ids[SEQ_LEN:]
        if seen >= NPROB:
            break
    steps = max(1, len(chunks) // BATCH)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    best = {"strict": -1.0, "lenient": -1.0, "step": 0}
    grad_norms, idx = [], 0
    eval_every = max(20, steps // 5)
    for step in range(1, steps + 1):
        x = torch.tensor(chunks[idx:idx + BATCH], device=device)
        idx = (idx + BATCH) % max(1, len(chunks) - BATCH)
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        grad_norms.append(gn)
        opt.step()
        sched.step()
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"NaN loss at step {step}; last grad_norm={gn:.3g}")
        if step % eval_every == 0 or step == steps:
            st, le, wrong = evaluate(model, tok, eval_tune, device)
            print(f"[{tag}] step {step}/{steps} loss {loss.item():.4f} "
                  f"gn_max={max(grad_norms):.2f} strict {st:.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
            if st > best["strict"]:
                best = {"strict": st, "lenient": le, "step": step}
                torch.save(model.state_dict(), f"runs/{tag}/best.pt")
    st, le, wrong = evaluate(model, tok, eval_final, device)
    if st > best["strict"]:
        best = {"strict": st, "lenient": le, "step": steps}
        torch.save(model.state_dict(), f"runs/{tag}/best.pt")
    model.load_state_dict(torch.load(f"runs/{tag}/best.pt", map_location=device))
    model.eval()
    lang = {}
    for p in ["Once upon a time", "The capital of France is"]:
        enc = tok(p, return_tensors="pt").to(device)
        g = model.generate(**enc, max_new_tokens=60, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        lang[p] = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    summary = {"model": MID, "params": n_params, "trained_params": trained,
               "frozen_params": frozen, "n_problems": NPROB, "lr": LR,
               "recipe": "frozen embed_in+embed_out", "best": best,
               "final_strict": st, "final_lenient": le, "wrong_examples": wrong,
               "lang_samples": lang, "grad_norm_max": max(grad_norms),
               "wall_min": (time.time() - t0) / 60, "tag": tag}
    with open(f"runs/{tag}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open("stage_b_round4.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"[{tag}] DONE best={best} final={st:.4f} "
          f"gn_max={max(grad_norms):.2f} ({summary['wall_min']:.1f} min)", flush=True)

if __name__ == "__main__":
    main()
