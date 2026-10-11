#!/usr/bin/env python
"""Bisect WHY one AdamW step NaNs pythia-14m with finite gradients.
Variants on a fresh model, one step each:
  A: AdamW lr=1e-4 wd=0.1 foreach default   (reproduces the NaN)
  B: AdamW lr=1e-4 wd=0.1 foreach=False
  C: AdamW lr=1e-4 wd=0.0 foreach=False
  D: SGD lr=1e-3 (no state at all)
Prints param dtypes seen, and per-variant first NaN tensor.
"""
import math, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def cot(a, b):
    sa, sb = str(a).rjust(4, "0"), str(b).rjust(4, "0")
    steps, carry = [], 0
    for i in range(3, -1, -1):
        d1, d2 = int(sa[i]), int(sb[i])
        s = d1 + d2 + carry
        steps.append(f"{d1}+{d2}" + (f"+{carry}" if carry else "") + f"={s}")
        carry = s // 10
    return f"{a}+{b}=\n{','.join(steps)},{a+b}\n"

def build_batch():
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    rng = random.Random(0)
    ids, chunks, seen = [], [], 0
    while seen < 512:
        a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
        ids.extend(tok(cot(a, b), add_special_tokens=False)["input_ids"])
        seen += 1
        while len(ids) >= 256:
            chunks.append(ids[:256])
            ids = ids[256:]
    return torch.stack([torch.tensor(c) for c in chunks[:8]])

def run_variant(x, label, make_opt):
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m")
    m.train()
    opt = make_opt(m)
    loss = m(input_ids=x, labels=x).loss
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    bad = [n for n, p in m.named_parameters() if torch.isnan(p).any()]
    print(f"  {label}: loss={loss.item():.4f} -> NaN tensors: {len(bad)}"
          f"{' FIRST=' + bad[0] if bad else '  (clean)'}", flush=True)
    return len(bad)

def main():
    torch.set_num_threads(4)
    x = build_batch()
    torch.manual_seed(0)
    m0 = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m")
    dtypes = {p.dtype for p in m0.parameters()}
    print("param dtypes:", dtypes, flush=True)
    del m0

    print("variant results:", flush=True)
    run_variant(x, "A AdamW wd=.1 foreach=default",
                lambda m: torch.optim.AdamW(m.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1))
    run_variant(x, "B AdamW wd=.1 foreach=False",
                lambda m: torch.optim.AdamW(m.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1, foreach=False))
    run_variant(x, "C AdamW wd=0  foreach=False",
                lambda m: torch.optim.AdamW(m.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, foreach=False))
    run_variant(x, "D SGD lr=1e-3",
                lambda m: torch.optim.SGD(m.parameters(), lr=1e-3))

if __name__ == "__main__":
    main()
