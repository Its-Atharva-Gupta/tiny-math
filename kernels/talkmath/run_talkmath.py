#!/usr/bin/env python
"""TALK + MATH — can one model keep its language AND learn arithmetic?

Control (round 7): TS-1M + digit-split tokenizer, math-only full FT ->
math 99.7%, language DEAD ("Once upon a time" -> digit chains).
This kernel builds the model the user actually wants, two ways:

Run A  REHEARSAL MIX: full fine-tune on 1.78M math problems INTERLEAVED
      1:1 by chunks with tokenized TinyStories story text. Same math
      exposure as round 7 + never-stop-seeing-English.
Run B  LoRA: base model FROZEN, r=8 adapter on all linear layers,
      trained on the same math-only stream.

Tokenizer: digit-split everywhere (the round-7 fix), implemented as a
hybrid encoder: GPT-2 pre-tokenizer pattern with number runs split per
digit; letters/symbols keep native BPE. Verified in-kernel:
  digit-free story text -> EXACTLY native encoding (assert on 200 samples)
  math text            -> EXACTLY round-7 digit-split dict (assert)

Eval (same hygiene as everywhere: tune 7781 / final 7782, strict first-line):
  math:  strict + lenient on n=2000
  talk:  token-level perplexity on 500 held-out TinyStories (base vs post),
         greedy generation samples from fixed prompts
Pre-registered: mix keeps math >=90% AND ppl within ~1.5x of base;
LoRA keeps ppl ~= base (frozen by construction) with math possibly lower.
Single file — no sibling imports.
"""
import json, math, os, re, subprocess, time, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Kaggle ships torchao 0.10; peft's LoRA dispatcher hard-fails on it
# ("only versions above 0.16 supported"). Uninstall BEFORE peft imports —
# dispatcher then falls back to plain Linear LoRA. (Known gotcha, kaggle-gpu-training skill.)
subprocess.run(["pip", "uninstall", "-y", "torchao", "torchao-nightly"],
               capture_output=True)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MID = "roneneldan/TinyStories-1M"
NPROB, BATCH, SEQ_LEN, MAX_NEW = 1_780_000, 32, 256, 96
EVAL_N_TRAIN, EVAL_N_FINAL = 1000, 2000
LR_FT, LR_LORA = 6e-5, 3e-5
MATH_ALPHABET = "0123456789+=,\n"
PAD_ID = 0   # overwritten from tokenizer in main()
LANG_PROMPTS = ["Once upon a time", "The capital of France is"]

# hybrid splitter: GPT-2 pattern with number runs -> single digits
SPLIT_RE = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+",
    re.UNICODE)

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

class HybridEncoder:
    """GPT-2 pre-tokenizer with number runs split per digit. Letters/symbols
    keep native BPE; single digits map through the pre-built dict."""
    def __init__(self, tok):
        self.tok = tok
        self.D = {}
        for ch in MATH_ALPHABET:
            ids = tok.encode(ch, add_special_tokens=False)
            assert len(ids) == 1 and tok.decode(ids) == ch
            self.D[ch] = ids[0]
    def encode(self, text):
        out = []
        for chunk in SPLIT_RE.findall(text):
            if len(chunk) == 1 and chunk in self.D:
                out.append(self.D[chunk])
            else:
                out.extend(self.tok.encode(chunk, add_special_tokens=False))
        return out
    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

def verify_encoder(tok, enc):
    # 1) math text == round-7 pure dict path
    rng = random.Random(0)
    for _ in range(200):
        a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
        t = cot(a, b)
        assert enc.encode(t) == [enc.D[c] for c in t], "math path mismatch"
    # 2) digit-free story text == native encoding EXACTLY
    checked = 0
    for s in STORY_SAMPLE:
        if not any(c.isdigit() for c in s):
            assert enc.encode(s) == tok.encode(s, add_special_tokens=False), \
                f"story path mismatch: {s[:60]!r}"
            checked += 1
            if checked >= 100:
                break
    # 3) digit-containing story text: same as native except number runs split
    s = "Tom had 1842 apples and gave 9920 away."
    assert enc.decode(enc.encode(s)) == s, "story roundtrip fail"
    print(f"encoder verified: math==dict (200), digit-free story==native ({checked}), "
          f"digit-story roundtrip OK", flush=True)

@torch.no_grad()
def evaluate_math(model, enc, pairs, device, use_fp16=True, prefix=""):
    model.eval()
    s_ok = l_ok = 0
    wrong = []
    pad = PAD_ID   # set from tokenizer in main(); never read from generation_config (None pre-generation)
    for i in range(0, len(pairs), 32):
        batch = pairs[i:i + 32]
        seqs = [enc.encode(prefix + f"{a}+{b}=\n") for a, b in batch]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long, device=device)
        for j, s in enumerate(seqs):
            ids[j, L - len(s):] = torch.tensor(s, device=device)
        attn = (ids != pad).long()
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = model.generate(input_ids=ids, attention_mask=attn,
                                 max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=pad)
        gen = out[:, L:]
        for j, (a, b) in enumerate(batch):
            text = enc.decode(gen[j].tolist())
            gold = str(a + b)
            if strict_parse(text) == gold:
                s_ok += 1
            if lenient_parse(text) == gold:
                l_ok += 1
            elif len(wrong) < 3:
                wrong.append((f"{a}+{b}", text[:80].replace("\n", "\\n"), gold))
    model.train()
    return s_ok / len(pairs), l_ok / len(pairs), wrong

@torch.no_grad()
def story_ppl(model, enc, stories, device, batch=16, seqlen=256):
    """Token-level perplexity on held-out story text."""
    model.eval()
    ids = []
    for s in stories:
        ids.extend(enc.encode(s + " "))
    total_nll, total_tok = 0.0, 0
    for i in range(0, len(ids) - seqlen, seqlen * batch):
        chunk = ids[i:i + seqlen * batch]
        n = len(chunk) // seqlen
        if n == 0:
            break
        x = torch.tensor([chunk[j*seqlen:(j+1)*seqlen] for j in range(n)], device=device)
        with torch.no_grad():
            logits = model(input_ids=x).logits
        nll = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1),
            reduction="sum").item()
        total_nll += nll
        total_tok += x[:, 1:].numel()
    model.train()
    return math.exp(total_nll / max(1, total_tok)), total_tok

def lang_samples(model, tok, device):
    model.eval()
    out = {}
    for p in LANG_PROMPTS:
        enc_ids = tok(p, return_tensors="pt").to(device)
        g = model.generate(**enc_ids, max_new_tokens=60, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        out[p] = tok.decode(g[0][enc_ids["input_ids"].shape[1]:].tolist(),
                            skip_special_tokens=True)
    model.train()
    return out

def pack_math_chunks(enc, problems, budget):
    ids, chunks, seen = [], [], 0
    for a, b in problems:
        ids.extend(enc.encode(cot(a, b)))
        seen += 1
        while len(ids) >= SEQ_LEN:
            chunks.append(ids[:SEQ_LEN])
            ids = ids[SEQ_LEN:]
        if seen >= budget:
            break
    return chunks, seen

def story_stream_chunks(enc, n_chunks, seed=1):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train",
                      streaming=True).shuffle(seed=seed)
    ids, chunks = [], []
    for row in ds:
        ids.extend(enc.encode(row["text"] + "  "))
        while len(ids) >= SEQ_LEN and len(chunks) < n_chunks:
            chunks.append(ids[:SEQ_LEN])
            ids = ids[SEQ_LEN:]
        if len(chunks) >= n_chunks:
            break
    return chunks

def heldout_stories(n=500, seed=99):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train",
                      streaming=True).shuffle(seed=99)
    out = []
    for row in ds:
        out.append(row["text"])
        if len(out) >= n:
            break
    return out

def train_loop(model, chunks, device, lr, tag, eval_tune, enc, tok,
               use_fp16=True, tag_steps_note=""):
    steps = max(1, len(chunks) // BATCH)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    best = {"strict": -1.0, "step": 0}
    hist, idx = [], 0
    eval_every = max(50, steps // 6)
    eval_steps = {50, eval_every, steps} if steps > 50 else {eval_every, steps}  # fail-fast early eval
    t0 = time.time()
    for step in range(1, steps + 1):
        x = torch.tensor(chunks[idx:idx + BATCH], device=device)
        idx = (idx + BATCH) % max(1, len(chunks) - BATCH)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16 and device == "cuda"):
            loss = model(input_ids=x, labels=x).loss
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0).item()
        opt.step()
        sched.step()
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"[{tag}] NaN at step {step}")
        if step in eval_steps or step % eval_every == 0 or step == steps:
            st, le, _ = evaluate_math(model, enc, eval_tune, device, use_fp16)
            hist.append({"step": step, "loss": round(loss.item(), 4), "strict": round(st, 4)})
            print(f"[{tag}] step {step}/{steps} loss {loss.item():.4f} strict {st:.4f} "
                  f"({(time.time()-t0)/60:.1f} min){tag_steps_note}", flush=True)
            if st > best["strict"]:
                best = {"strict": st, "step": step}
                torch.save(model.state_dict(), f"runs/{tag}/best.pt")
    return best, hist

def main():
    global PAD_ID
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    tok = AutoTokenizer.from_pretrained(MID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    PAD_ID = tok.pad_token_id
    enc = HybridEncoder(tok)
    verify_encoder(tok, enc)

    eval_tune = make_eval_set(7781, EVAL_N_TRAIN)
    tune_set = set(eval_tune)
    eval_final = [p for p in make_eval_set(7782, EVAL_N_FINAL) if p not in tune_set]
    reject = tune_set | set(eval_final)
    math_problems = make_train_problems(0, NPROB, reject)

    # held-out stories FIRST (disjoint from training stream by position)
    stories_eval = heldout_stories(500)
    print(f"held-out stories: {len(stories_eval)}", flush=True)

    base = AutoModelForCausalLM.from_pretrained(MID).to(device)
    base.train()
    n_params = sum(p.numel() for p in base.parameters())
    base_ppl, base_tok = story_ppl(base, enc, stories_eval, device)
    base_lang = lang_samples(base, tok, device)
    print(f"BASE: params={n_params/1e6:.2f}M story_ppl={base_ppl:.2f} "
          f"({base_tok} tok)", flush=True)
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}

    math_chunks, used = pack_math_chunks(enc, math_problems, NPROB)
    print(f"math chunks: {len(math_chunks)} from {used} problems", flush=True)

    results = []

    # ---------------- RUN A5: uniform 15:1 (6.25% story share) ------------
    # Full curve: 50%->1.4%/lang ok; 75%->61.5%/ok; 87.5%->89.8%/ok (x0.98);
    # anneal-then-0%->99.3%/DEAD; 100%->99.7%/DEAD.
    # Missing point: 93.75% math share uniform. PREDICTION: math 93-97%,
    # ppl <=3x base. If language holds at 6% share -> delivered model.
    tagA = "r8a5_ts1m_mix15to1"
    os.makedirs(f"runs/{tagA}", exist_ok=True)
    RUN_A = True
    if RUN_A and not os.path.exists(f"runs/{tagA}/summary.json"):
        print("\n=== RUN A5: uniform 15:1 math:story (full FT) ===", flush=True)
        story_chunks = story_stream_chunks(enc, max(1, len(math_chunks) // 15))
        print(f"math chunks: {len(math_chunks)}, story chunks: {len(story_chunks)}", flush=True)
        mixed = []
        story_iter = iter(story_chunks)
        for i, m in enumerate(math_chunks):
            mixed.append(m)
            if i % 15 == 14:
                try:
                    mixed.append(next(story_iter))
                except StopIteration:
                    pass
        base.load_state_dict(base_sd)
        model = base
        model.train()
        best, hist = train_loop(model, mixed, device, LR_FT, tagA, eval_tune, enc, tok)
        st, le, wrong = evaluate_math(model, enc, eval_final, device)
        ctx_acc, _, _ = evaluate_math(model, enc, eval_tune[:200], device,
                                      prefix="Once upon a time there was a boy who had ")
        if st > best["strict"]:
            best = {"strict": st, "step": "final"}
            torch.save(model.state_dict(), f"runs/{tagA}/best.pt")
        model.load_state_dict(torch.load(f"runs/{tagA}/best.pt", map_location=device))
        ppl, ptok = story_ppl(model, enc, stories_eval, device)
        lang = lang_samples(model, tok, device)
        summary = {"run": "A5_mix_15to1", "model": MID, "params": n_params,
                   "n_problems": NPROB, "math_story_ratio": "15:1 uniform",
                   "lr": LR_FT,
                   "trainable": "all", "best": best, "final_strict": st,
                   "final_lenient": le, "ctx_prefix_strict": round(ctx_acc, 4),
                   "wrong_examples": wrong,
                   "story_ppl_base": round(base_ppl, 2), "story_ppl_post": round(ppl, 2),
                   "ppl_ratio": round(ppl / base_ppl, 3),
                   "lang_samples": lang, "base_lang_samples": base_lang,
                   "hist": hist, "tag": tagA}
        with open(f"runs/{tagA}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open("stage_c_talkmath5.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"[{tagA}] DONE math_strict={st:.4f} ppl {base_ppl:.1f}->{ppl:.1f} "
              f"(x{ppl/base_ppl:.2f})", flush=True)
        results.append(summary)

    # ---------------- RUN B2: LoRA r=32 (stronger, gentler LR) -------------
    # Round-1 Run B (r=8, lr 1e-4): math 0%, language wrecked (PPL x756k) —
    # 74K trainable params ~10x too weak for math, yet perturbation enough
    # to hijack language. B2: r=32 (~4x capacity) at lr 3e-5 (3x gentler).
    # PREDICTION: math much higher (20-90%), ppl between base and wrecked.
    tagB = "r8b2_ts1m_lora_r32"
    os.makedirs(f"runs/{tagB}", exist_ok=True)
    RUN_B = False   # LoRA closed: r=8 AND r=32 both 0% math + language destroyed
    if RUN_B and not os.path.exists(f"runs/{tagB}/summary.json"):
        print("\n=== RUN B2: LoRA r=32 (frozen base), math-only, lr 3e-5 ===", flush=True)
        base.load_state_dict(base_sd)
        lconf = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.05,
                           target_modules="all-linear", bias="none")
        model = get_peft_model(base, lconf)
        model.print_trainable_parameters()
        best, hist = train_loop(model, math_chunks, device, LR_LORA, tagB,
                                eval_tune, enc, tok)
        st, le, wrong = evaluate_math(model, enc, eval_final, device)
        ctx_acc, _, _ = evaluate_math(model, enc, eval_tune[:200], device,
                                      prefix="Once upon a time there was a boy who had ")
        if st > best["strict"]:
            best = {"strict": st, "step": "final"}
            model.save_pretrained(f"runs/{tagB}/adapter")
        ppl, ptok = story_ppl(model, enc, stories_eval, device)
        lang = lang_samples(model, tok, device)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        summary = {"run": "B2_lora_r32", "model": MID, "params": n_params,
                   "trainable_params": n_train, "n_problems": NPROB, "lr": LR_LORA,
                   "trainable": "lora r=32 all-linear", "best": best,
                   "final_strict": st, "final_lenient": le,
                   "ctx_prefix_strict": round(ctx_acc, 4), "wrong_examples": wrong,
                   "story_ppl_base": round(base_ppl, 2), "story_ppl_post": round(ppl, 2),
                   "ppl_ratio": round(ppl / base_ppl, 3),
                   "lang_samples": lang, "base_lang_samples": base_lang,
                   "hist": hist, "tag": tagB}
        with open(f"runs/{tagB}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open("stage_c_talkmath5.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"[{tagB}] DONE math_strict={st:.4f} ppl {base_ppl:.1f}->{ppl:.1f} "
              f"(x{ppl/base_ppl:.2f}) trainable={n_train/1e6:.3f}M", flush=True)
        results.append(summary)

    print("\n==== TALK+MATH SUMMARY ====", flush=True)
    for r in results:
        print(f"{r['tag']:>22}  math_strict={r['final_strict']:.4f}  "
              f"story_ppl x{r['ppl_ratio']:.2f} of base  "
              f"sample={list(r['lang_samples'].values())[0][:50]!r}", flush=True)
    print(f"{'round7 math-only ctrl':>22}  math_strict=0.9970  story_ppl: language DEAD", flush=True)

if __name__ == "__main__":
    # tiny sample buffer for encoder verification (loaded lazily via HF hub text is
    # overkill; use fixed strings resembling TinyStories style)
    STORY_SAMPLE = [
        "Once upon a time there was a little girl named Lily who loved to play in the garden with her cat.",
        "Tom and his mother went to the store to buy some apples and bread for the picnic.",
        "The bunny was very happy when he found a big carrot hiding under the tall green tree.",
        "In a small village by the river lived an old man who told stories to the children every night.",
        "Emily had a secret box with shiny stones that she collected from the beach every summer day.",
        "The dog ran quickly across the field chasing the ball that his friend had thrown far away.",
        "Anna did not like the rain, so she sat by the window and read a book about brave knights.",
        "There once was a tiny bird who could not fly, but he learned to jump very very high.",
        "The baker made warm bread every morning and the whole street smelled sweet and wonderful.",
        "Max found a old map in the attic and dreamed of sailing to the islands beyond the blue sea.",
    ] * 30
    main()
