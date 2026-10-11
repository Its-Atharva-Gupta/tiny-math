#!/usr/bin/env python
"""Local NaN autopsy for pythia-14m. Replicates the Kaggle failing recipe
(fp32, full-param AdamW, lr 1e-4, clip 1.0, same cot data) and after every
step scans ALL parameters for NaN grads/weights, naming the culprit module.
Also checks the forward pass alone (logits/loss finite?) before any update.
"""
import math, os, random, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "EleutherAI/pythia-14m"
BATCH, SEQ_LEN, LR, STEPS = 8, 256, 1e-4, 10

def cot(a, b):
    sa, sb = str(a).rjust(4, "0"), str(b).rjust(4, "0")
    steps, carry = [], 0
    for i in range(3, -1, -1):
        d1, d2 = int(sa[i]), int(sb[i])
        s = d1 + d2 + carry
        steps.append(f"{d1}+{d2}" + (f"+{carry}" if carry else "") + f"={s}")
        carry = s // 10
    return f"{a}+{b}=\n{','.join(steps)},{a+b}\n"

def main():
    torch.manual_seed(0)
    print("loading", MID, flush=True)
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.train()

    # same cot training stream as the kernels
    rng = random.Random(0)
    ids, chunks, seen = [], [], 0
    while seen < 4096:
        a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
        ids.extend(tok(cot(a, b), add_special_tokens=False)["input_ids"])
        seen += 1
        while len(ids) >= SEQ_LEN:
            chunks.append(ids[:SEQ_LEN])
            ids = ids[SEQ_LEN:]
    print(f"built {len(chunks)} chunks of {SEQ_LEN} tokens", flush=True)

    # forward-only sanity: is the loss finite before ANY update?
    x = torch.stack([torch.tensor(c) for c in chunks[:BATCH]])
    with torch.no_grad():
        logits = model(input_ids=x).logits
    print(f"forward-only: logits finite={torch.isfinite(logits).all().item()} "
          f"max|logit|={logits.abs().max().item():.3g}", flush=True)
    with torch.no_grad():
        floss = model(input_ids=x, labels=x).loss
    print(f"forward-only loss={floss.item():.4f} finite={math.isfinite(floss.item())}", flush=True)

    # embedding-row histogram for the cot tokens actually used
    emb_mod = next(m for n, m in model.named_modules() if isinstance(m, torch.nn.Embedding))
    used = set()
    for c in chunks[:50]:
        used.update(c)
    emb = emb_mod.weight.detach()
    sub = emb[sorted(used)]
    print(f"embed rows used by cot data: n={len(used)} max|w|={sub.abs().max().item():.3g} "
          f"any_nan={torch.isnan(sub).any().item()}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    for step in range(1, STEPS + 1):
        x = torch.stack([torch.tensor(c) for c in chunks[(step*BATCH) % len(chunks):(step*BATCH) % len(chunks) + BATCH]])
        if x.shape[0] < BATCH:
            x = torch.stack([torch.tensor(c) for c in chunks[:BATCH]])
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad()
        loss.backward()
        nan_grads = [(n, p.grad.abs().max().item())
                     for n, p in model.named_parameters()
                     if p.grad is not None and torch.isnan(p.grad).any()]
        inf_grads = [(n, p.grad.abs().max().item())
                     for n, p in model.named_parameters()
                     if p.grad is not None and not torch.isnan(p.grad).any()
                     and torch.isinf(p.grad).any()]
        big = sorted(((n, g) for n, p in model.named_parameters()
                      if p.grad is not None and not torch.isnan(p.grad).any()
                      and not torch.isinf(p.grad).any()
                      for g in [p.grad.norm().item()]), key=lambda t: -t[1])[:3]
        print(f"step {step}: loss={loss.item():.4f} "
              f"nan_grad_modules={[n for n,_ in nan_grads][:6]} (n={len(nan_grads)}) "
              f"inf_grad_modules={[n for n,_ in inf_grads][:6]} top_grad_norms={[(n, round(g,1)) for n,g in big]}", flush=True)
        if math.isfinite(loss.item()):
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            print(f"         clip total_norm={gn if not math.isfinite(gn) else round(gn,3)}", flush=True)
            opt.step()
            bad_w = [n for n, p in model.named_parameters() if torch.isnan(p).any()]
            if bad_w:
                print(f"         WEIGHTS NaN after step: {bad_w[:6]} (n={len(bad_w)})", flush=True)
                break
        else:
            print("         loss NaN — halting before update", flush=True)
            break

if __name__ == "__main__":
    main()
