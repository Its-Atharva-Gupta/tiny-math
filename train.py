"""Tiny arithmetic LM — from-scratch GPT, digit-level vocab, SFT on generated data.

Single file, no deps beyond torch+numpy. Runs unchanged on local CPU (dry-run
preset) and Kaggle T4 (sweep preset).

Format experiment (Phase 0b picks the winner):
  l2r : "47+68=\n115\n"        answer left-to-right
  rtl : "47+68=\n511\n"        answer right-to-left (carry-friendly)
  cot : "47+68=\n7+8=15,4+6+1=11,115\n"   digit-wise steps, comma-separated,
        answer = last comma-field. Single trailing newline keeps eval-stopping
        trivial (generate until first '\n').
"""
import argparse, json, math, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- tokenizer
VOCAB = list("0123456789+-=\n,") + ["<pad>"]
TOK = {c: i for i, c in enumerate(VOCAB)}
PAD = TOK["<pad>"]
V = len(VOCAB)  # 16

def enc(s): return [TOK[c] for c in s]
def dec(ids): return "".join(VOCAB[i] for i in ids)

# ---------------------------------------------------------------- data
def sample_pair(rng, digits):
    lo, hi = (0, 9) if digits == 1 else (10 ** (digits - 1), 10 ** digits - 1)
    return rng.randint(lo, hi), rng.randint(lo, hi)

def format_problem(a, b, ans, fmt):
    if fmt == "l2r":
        return f"{a}+{b}=\n{ans}\n"
    if fmt == "rtl":
        return f"{a}+{b}=\n{str(ans)[::-1]}\n"
    if fmt == "cot":
        sa, sb = str(a), str(b)
        n = max(len(sa), len(sb))
        sa, sb = sa.rjust(n, "0"), sb.rjust(n, "0")
        steps, carry = [], 0
        for i in range(n - 1, -1, -1):
            d1, d2 = int(sa[i]), int(sb[i])
            s = d1 + d2 + carry
            steps.append(f"{d1}+{d2}" + (f"+{carry}" if carry else "") + f"={s}")
            carry = s // 10
        return f"{a}+{b}=\n{','.join(steps)},{ans}\n"
    raise ValueError(fmt)

def parse_answer(completion, fmt):
    """completion = generated text after the prompt, before stopping."""
    line = completion.split("\n")[0].strip()
    if not line:
        return None
    if fmt == "rtl":
        return line[::-1]
    if fmt == "cot":
        return line.split(",")[-1]
    return line

def max_new_for(fmt, digits=4):
    # cot completion ≈ 9 chars/step + answer; l2r/rtl ≈ answer + newline
    return {"l2r": digits + 4, "rtl": digits + 4, "cot": 10 * digits + 12}[fmt]

def make_stream(seed, digits, fmt, eval_set, budget_tokens):
    """Generator of token-id chunks; rejects problems in eval_set
    (guarantees zero train/eval contamination by construction).
    Yields one problem's ids per chunk; constant memory at any budget."""
    py_rng = random.Random(seed)
    n = 0
    while n < budget_tokens:
        a, b = sample_pair(py_rng, digits)
        if (a, b) in eval_set:
            continue
        text = format_problem(a, b, a + b, fmt)
        n += len(text)
        yield enc(text)

def make_eval_set(seed, digits, n_problems, exclude=None):
    """Fixed held-out problems. exclude: set of (a,b) pairs to never return
    (used to keep the tuning-eval and final-eval sets disjoint).
    Attempts-guarded: for tiny digit spaces (1-digit = 100 pairs) it returns
    what exists instead of looping forever."""
    rng = random.Random(seed)
    exclude = exclude or set()
    seen, out, attempts = set(), [], 0
    while len(out) < n_problems and attempts < n_problems * 100 + 10_000:
        attempts += 1
        a, b = sample_pair(rng, digits)
        if (a, b) in seen or (a, b) in exclude:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out

# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, H):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.H = H
    def forward(self, x, mask):
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, -1)
        B, T, d = q.shape
        hd = d // self.H
        q = q.view(B, T, self.H, hd).transpose(1, 2)
        k = k.view(B, T, self.H, hd).transpose(1, 2)
        v = v.view(B, T, self.H, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        a = a.transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, d, L, H, ctx=64):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, H) for _ in range(L))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    def forward(self, idx):
        B, T = idx.shape
        p = torch.arange(T, device=idx.device)
        x = self.emb(idx) + self.pos(p)[None]
        for b in self.blocks:
            x = b(x, None)
        return self.head(self.ln(x))

SIZES = {
    "1m":  dict(d=128, L=4, H=4),   # ~1.07M
    "5m":  dict(d=192, L=8, H=6),   # ~4.76M
    "8m":  dict(d=256, L=8, H=8),   # ~8.45M
}

def count_params(m): return sum(p.numel() for p in m.parameters())

# ---------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, eval_pairs, fmt, device, batch=256, digits=4, return_trunc=False):
    """Greedy decode, strict exact-match. Returns (acc, n_correct, examples)
    or (..., trunc_frac) when return_trunc=True — trunc_frac = fraction of
    generations that hit the length cap without emitting a newline (needed
    when evaluating beyond trained sequence lengths, where truncation can
    silently masquerade as model failure)."""
    model.eval()
    mx = max_new_for(fmt, digits)
    prompts = [f"{a}+{b}=\n" for a, b in eval_pairs]
    golds = [str(a + b) for a, b in eval_pairs]
    correct, wrong_ex = 0, []
    truncated = 0
    for i in range(0, len(prompts), batch):
        pb = prompts[i:i + batch]
        gb = golds[i:i + batch]
        ids = torch.tensor([enc(p) for p in pb], device=device)
        done = torch.zeros(len(pb), dtype=torch.bool, device=device)
        for _ in range(mx):
            if ids.shape[1] >= model.pos.num_embeddings - 1:
                break  # hard stop at positional-embedding limit; never crash
            logits = model(ids)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
            done |= (nxt.squeeze(1) == TOK["\n"])
            if done.all():
                break
        truncated += int((~done).sum())
        for j, (p, g) in enumerate(zip(pb, gb)):
            gen = ids[j, len(p):].tolist()
            comp = []
            for t in gen:
                if t == TOK["\n"]:
                    break
                comp.append(t)
            pred = parse_answer(dec(comp), fmt)
            if pred == g:
                correct += 1
            elif len(wrong_ex) < 5:
                wrong_ex.append((p.replace("\n", "\\n"), dec(comp).replace("\n", "\\n"), g))
    model.train()
    if return_trunc:
        return correct / len(prompts), correct, wrong_ex, truncated / len(prompts)
    return correct / len(prompts), correct, wrong_ex

# ---------------------------------------------------------------- train
def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.local else "cpu"
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    cfg = SIZES[args.size]
    ctx = 64
    model = TinyGPT(cfg["d"], cfg["L"], cfg["H"], ctx).to(device)
    n_params = count_params(model)

    eval_pairs = make_eval_set(args.seed + 999, args.digits, args.eval_n)
    eval_keys = set(eval_pairs)
    # final eval set created UPFRONT and excluded from training by construction:
    # the stream must never see any problem in either eval set.
    final_pairs = make_eval_set(args.seed + 12345, args.digits, args.eval_final_n,
                                exclude=eval_keys)
    final_keys = set(final_pairs)
    train_reject = eval_keys | final_keys
    assert not (final_keys & eval_keys), "final eval overlaps tuning eval!"

    B, T = args.batch, ctx
    steps = args.tokens // (B * T)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    stream = make_stream(args.seed, args.digits, args.fmt, train_reject,
                         steps * B * (T + 1) + 4096)
    buf = []

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "curve.csv", "w")
    log_f.write("step,loss,lr,eval_acc,elapsed_s,tok_per_s\n")

    eval_every = max(20, steps // args.eval_points)
    best_acc, best_step = -1.0, -1
    t0 = time.time()
    tok_count = 0

    for step in range(1, steps + 1):
        while len(buf) < B * (T + 1):
            buf.extend(next(stream))
        batch = torch.tensor(buf[: B * (T + 1)], device=device).view(B, T + 1)
        buf = buf[B * (T + 1):]
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tok_count += B * T

        if step % eval_every == 0 or step == steps:
            acc, nc, wrong = evaluate(model, eval_pairs[: args.eval_n], args.fmt, device, digits=args.digits)
            el = time.time() - t0
            tps = tok_count / el
            lr_now = opt.param_groups[0]["lr"]
            log_f.write(f"{step},{loss.item():.4f},{lr_now:.2e},{acc:.4f},{el:.1f},{tps:.0f}\n")
            log_f.flush()
            print(f"[{args.tag}] step {step}/{steps} loss {loss.item():.4f} eval_acc {acc:.4f} "
                  f"({el/60:.1f} min, {tps:.0f} tok/s)", flush=True)
            if wrong:
                print(f"  sample wrong: {wrong[0]}", flush=True)
            if acc > best_acc:
                best_acc, best_step = acc, step
                torch.save({"model": model.state_dict(), "cfg": cfg, "args": vars(args)},
                           out / "best.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg, "args": vars(args)}, out / "final.pt")

    # final eval on the pre-built held-out set (n=eval_final_n)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    acc_best, _, wrong_best = evaluate(model, final_pairs, args.fmt, device, digits=args.digits)
    summary = {
        "tag": args.tag, "size": args.size, "params": n_params, "fmt": args.fmt,
        "digits": args.digits, "train_tokens": args.tokens,
        "best_tuning_acc": best_acc, "best_step": best_step,
        "final_bestckpt_acc": acc_best, "eval_n": args.eval_final_n,
        "wall_min": (time.time() - t0) / 60, "device": device,
        "wrong_examples": wrong_best[:10],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{args.tag}] DONE best_tuning={best_acc:.4f} final(n={args.eval_final_n})={acc_best:.4f}", flush=True)
    return summary

PRESETS = {
    # local dry-run: tiny everything, catches bugs for zero GPU cost
    "local": dict(tokens=50_000, eval_n=30, eval_final_n=50, eval_points=3, batch=32, lr=1e-3),
    # real runs
    "sweep": dict(tokens=10_000_000, eval_n=1000, eval_final_n=5000, eval_points=10, batch=128, lr=6e-4),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(SIZES), required=True)
    ap.add_argument("--fmt", choices=["l2r", "rtl", "cot"], required=True)
    ap.add_argument("--digits", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--eval_n", type=int, default=None)
    ap.add_argument("--eval_final_n", type=int, default=None)
    ap.add_argument("--eval_points", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", choices=list(PRESETS), default="sweep")
    ap.add_argument("--local", action="store_true", help="force CPU even if cuda exists")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    for k, v in PRESETS[a.preset].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    a.tag = a.tag or f"{a.size}_{a.fmt}_{a.digits}d_{a.tokens//1000}k"
    a.out = a.out or f"runs/{a.tag}"
    if torch.cuda.is_available() and not a.local:
        torch.backends.cuda.matmul.allow_tf32 = True
    print(json.dumps({k: v for k, v in vars(a).items()}, indent=2), flush=True)
    train(a)
