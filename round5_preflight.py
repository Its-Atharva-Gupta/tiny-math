#!/usr/bin/env python
"""Local pre-flight for round 5: explicit fp32 + blocks-only (freeze BOTH
vocab matrices by module type — Embedding AND the output head), AdamW
lr 3e-4 on the ~1.2M blocks. 10 steps: instant-NaN check + loss trend.
If clean -> this recipe goes to Kaggle for the real 500k-problem run.
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

def main():
    torch.manual_seed(0)
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    # THE fix: explicit fp32 regardless of transformers version/dtype defaults
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-14m", torch_dtype=torch.float32)
    print("loaded dtype:", next(model.parameters()).dtype, flush=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # freeze BOTH vocab matrices by module type (names differ across versions)
    frozen = trained = 0
    for name, module in model.named_modules():
        pass
    head_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            for pn, _ in module.named_parameters(recurse=False):
                head_names.add(f"{name}.{pn}")
        if isinstance(module, torch.nn.Linear) and module.out_features == model.config.vocab_size:
            for pn, _ in module.named_parameters(recurse=False):
                head_names.add(f"{name}.{pn}")
    for name, p in model.named_parameters():
        if name in head_names:
            p.requires_grad = False
            frozen += p.numel()
        else:
            trained += p.numel()
    print(f"vocab matrices frozen={frozen/1e6:.2f}M  training={trained/1e6:.2f}M", flush=True)
    model.train()

    rng = random.Random(0)
    ids, chunks, seen = [], [], 0
    while seen < 512:
        a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
        ids.extend(tok(cot(a, b), add_special_tokens=False)["input_ids"])
        seen += 1
        while len(ids) >= 256:
            chunks.append(ids[:256])
            ids = ids[256:]

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    for step in range(1, 11):
        x = torch.stack([torch.tensor(c) for c in chunks[(step*8) % len(chunks):(step*8) % len(chunks) + 8]])
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad()
        loss.backward()
        nan_g = sum(1 for p in params if p.grad is not None and torch.isnan(p.grad).any())
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        bad_w = sum(1 for p in params if torch.isnan(p).any())
        print(f"step {step}: loss={loss.item():.4f} nan_grads={nan_g} "
              f"gn={gn:.2f} nan_weights={bad_w}", flush=True)
        if bad_w:
            break
    print("VERDICT:", "CLEAN — recipe valid for Kaggle round 5"
          if bad_w == 0 else "still NaN — need deeper fix", flush=True)

if __name__ == "__main__":
    main()
