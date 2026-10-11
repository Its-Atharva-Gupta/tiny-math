#!/usr/bin/env python
"""TOKENIZER ABLATION — the one-variable experiment.

Control: TinyStories-1M fine-tuned on 1.78M cot problems with its native
BPE tokenizer -> strict 0.2% (round 2, tag r2_ts1m_1780k).

Treatment (THIS run): SAME model checkpoint, SAME problems, SAME recipe
(fp16 autocast, lr 6e-5, batch 32, same eval sets), and the SMALLEST
possible tokenizer change: number runs split PER DIGIT instead of whole
chunks. Same 50,257-entry vocab, same embedding matrix, no new tokens,
no resize. Implemented as a per-char id dict — exactly equivalent to the
pre-tokenizer pattern change `?\\p{N}+` -> `?\\d` for this space-free
corpus (single-char chunk = BPE identity; validated on 500 strings).

Pre-registered predictions:
  BPE-chunking was THE wall  -> expect a large jump (>>10%, plausibly 90%+;
    from-scratch 806K hit 100% on this same data)
  tokenization was not it    -> stays <5% (wall is pretraining/capacity)
Compute note: digit-split costs 1.23x tokens/problem (~45.4 vs 36.9),
so same problem count => ~1.23x training steps. That is a consequence of
the treatment, disclosed, not matched away.
Single file — no sibling imports.
"""
import json, math, os, re, time, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "roneneldan/TinyStories-1M"
NPROB, BATCH, SEQ_LEN, LR, MAX_NEW = 1_780_000, 32, 256, 6e-5, 96
EVAL_N_TRAIN, EVAL_N_FINAL = 1000, 2000
ALPHABET = "0123456789+=,\n"

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
def evaluate(model, tok, D, pairs, device, use_fp16):
    model.eval()
    s_ok = l_ok = 0
    wrong = []
    pad = tok.pad_token_id
    for i in range(0, len(pairs), 32):
        batch = pairs[i:i + 32]
        seqs = [[D[c] for c in f"{a}+{b}=\n"] for a, b in batch]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long, device=device)
        for j, s in enumerate(seqs):
            ids[j, L - len(s):] = torch.tensor(s, device=device)   # left pad
        attn = (ids != pad).long()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16 and device == "cuda"):
            out = model.generate(input_ids=ids, attention_mask=attn,
                                 max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=pad)
        gen = out[:, L:]
        for j, (a, b) in enumerate(batch):
            text = tok.decode(gen[j].tolist(), skip_special_tokens=True)
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
    tag = "r7_ts1m_digitsplit_1780k"
    os.makedirs(f"runs/{tag}", exist_ok=True)
    print("device:", device, "| TREATMENT: digit-split tokenizer (same vocab)", flush=True)

    tok = AutoTokenizer.from_pretrained(MID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # build digit-split dict; assert round-trip + the split property
    D = {}
    for ch in ALPHABET:
        ids = tok.encode(ch, add_special_tokens=False)
        assert len(ids) == 1 and tok.decode(ids) == ch, f"char {ch!r} not 1:1"
        D[ch] = ids[0]
    assert len(tok.encode("1842", add_special_tokens=False)) == 2   # control property
    probe = [D[c] for c in "1842"]
    assert len(probe) == 4 and tok.decode(probe) == "1842"          # treatment property
    print(f"tokenizer validated: '1842' old={len(tok.encode('1842'))} tok, "
          f"new={len(probe)} tok; vocab={len(tok.get_vocab())} unchanged", flush=True)

    eval_tune = make_eval_set(7781, EVAL_N_TRAIN)
    tune_set = set(eval_tune)
    eval_final = [p for p in make_eval_set(7782, EVAL_N_FINAL) if p not in tune_set]
    reject = tune_set | set(eval_final)

    t0 = time.time()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(MID).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    use_fp16 = True     # exact round-2 TS-1M recipe (fp16 autocast, fp32 weights)

    ids, chunks, seen = [], [], 0
    for a, b in make_train_problems(0, NPROB, reject):
        ids.extend(D[c] for c in cot(a, b))
        seen += 1
        while len(ids) >= SEQ_LEN:
            chunks.append(ids[:SEQ_LEN])
            ids = ids[SEQ_LEN:]
        if seen >= NPROB:
            break
    steps = max(1, len(chunks) // BATCH)
    print(f"packing: {seen} problems, {len(chunks)} chunks of {SEQ_LEN}, {steps} steps", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    best = {"strict": -1.0, "lenient": -1.0, "step": 0}
    grad_norms, idx = [], 0
    eval_every = max(20, steps // 5)
    for step in range(1, steps + 1):
        x = torch.tensor(chunks[idx:idx + BATCH], device=device)
        idx = (idx + BATCH) % max(1, len(chunks) - BATCH)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16 and device == "cuda"):
            loss = model(input_ids=x, labels=x).loss
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        grad_norms.append(gn)
        opt.step()
        sched.step()
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"NaN loss at step {step}; last gn={gn}")
        if step % eval_every == 0 or step == steps:
            st, le, wrong = evaluate(model, tok, D, eval_tune, device, use_fp16)
            print(f"[{tag}] step {step}/{steps} loss {loss.item():.4f} gn={gn:.2f} "
                  f"strict {st:.4f} lenient {le:.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)
            if st > best["strict"]:
                best = {"strict": st, "lenient": le, "step": step}
                torch.save(model.state_dict(), f"runs/{tag}/best.pt")
    st, le, wrong = evaluate(model, tok, D, eval_final, device, use_fp16)
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
        lang[p] = tok.decode(g[0][enc["input_ids"].shape[1]:].tolist(), skip_special_tokens=True)
    summary = {"model": MID, "params": n_params, "n_problems": NPROB,
               "tokenizer": "digit-split (same 50257 vocab; pre-tok number runs -> per-digit)",
               "control_tag": "r2_ts1m_1780k (native BPE) = 0.002 strict",
               "tokens_per_problem": len(chunks) * SEQ_LEN / seen,
               "fp16": use_fp16, "lr": LR, "best": best,
               "final_strict": st, "final_lenient": le, "wrong_examples": wrong,
               "lang_samples": lang, "grad_norm_max": max(grad_norms),
               "wall_min": (time.time() - t0) / 60, "tag": tag}
    with open(f"runs/{tag}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open("stage_b_round7.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"[{tag}] DONE best={best} final={st:.4f} ({summary['wall_min']:.1f} min)", flush=True)

if __name__ == "__main__":
    main()
