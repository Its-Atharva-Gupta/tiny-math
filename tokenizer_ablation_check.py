#!/usr/bin/env python
"""Verify the minimal digit-split tokenizer for the ablation (public API only).

The minimal change: pre-tokenizer splits number runs PER DIGIT instead of
whole runs (GPT-2 pattern `?\\p{N}+` -> `?\\d`). Because a single-char chunk
under BPE is identity, this is exactly equivalent to looking each character
up in the existing vocab — same 50,257 entries, same embedding matrix,
zero new tokens.

Checks:
1. CONTROL: original tokenizer absorbs numbers as chunks (the problem).
2. New path: per-char dict round-trips on the full cot alphabet.
3. Equivalence argument, tested where testable: for every single char,
   original.encode(char) == our fast id (single-char chunks behave the
   same under old and new pattern); our corpus has no spaces (asserted),
   so no chunk-boundary interaction can occur.
4. Effect size: tokens-per-problem, old vs new.
"""
import random
from transformers import AutoTokenizer

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
    tok = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")
    vocab_size = len(tok.get_vocab())
    print(f"vocab size: {vocab_size} (UNCHANGED by our patch — no new tokens, no resize)")

    # 1. control: how numbers encode today
    for num in ["47", "15", "115", "1842", "9920"]:
        ids = tok.encode(num, add_special_tokens=False)
        print(f"  ORIGINAL: {num!r} -> {len(ids)} token(s) {ids} decode={[tok.decode([i]) for i in ids]}")

    # 2. build the digit-split dict over the full cot alphabet, round-trip check
    alphabet = "0123456789+=,\n"
    D = {}
    for ch in alphabet:
        ids = tok.encode(ch, add_special_tokens=False)
        assert len(ids) == 1, f"single char {ch!r} mapped to {ids}"
        assert tok.decode(ids) == ch, f"roundtrip fail for {ch!r}"
        D[ch] = ids[0]
    print(f"\nalphabet dict built: {len(D)} chars, all round-trip OK")

    def enc(text):
        return [D[c] for c in text]

    # 3. corpus assumptions
    rng = random.Random(0)
    old_n = new_n = 0
    for _ in range(500):
        a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
        t = cot(a, b)
        assert " " not in t, "cot text contains a space — chunk-interaction assumption broken"
        assert set(t) <= set(alphabet), f"unexpected chars: {set(t)-set(alphabet)}"
        old = tok.encode(t, add_special_tokens=False)
        new = enc(t)
        assert tok.decode(new) == t, "new-path decode mismatch"
        old_n += len(old); new_n += len(new)
    print(f"500 cot strings: decode(new)==original OK; no spaces, alphabet closed")
    print(f"tokens/problem: OLD={old_n/500:.1f}  NEW={new_n/500:.1f}  "
          f"(x{new_n/old_n:.2f} — digit-splitting costs ~this much extra compute)")

    # 4. the property itself, one example side by side
    ex = cot(1842, 2957)
    print(f"\nexample {ex.splitlines()[0]!r}:")
    print(f"  OLD pieces: {[tok.decode([i]) for i in tok.encode(ex, add_special_tokens=False)][:16]}")
    print(f"  NEW pieces: {[tok.decode([i]) for i in enc(ex)][:16]}")
    print("\nVERDICT: digit-split tokenizer validated (same vocab, numbers now per-digit)")

if __name__ == "__main__":
    main()
