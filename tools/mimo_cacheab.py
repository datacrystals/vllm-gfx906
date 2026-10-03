#!/usr/bin/env python3
"""H1 no-reboot A/B: does a prefix-cache HIT return the same tokens as a
forced fresh recompute of the same prompt?

Mechanism: `prompt_logprobs` implies sampling_params.skip_reading_prefix_cache
(sampling_params.py:439) -> the engine skips find_longest_cache_hit and
recomputes the prompt from scratch, while a plain request will hit the cache.

Protocol per prompt P (greedy, temperature=0):
  fresh1 = complete(P, prompt_logprobs=0)   # no cache read
  fresh2 = complete(P, prompt_logprobs=0)   # no cache read (nondeterminism control)
  warm   = complete(P)                      # cache read (hit, P was just written)
  warm2  = complete(P)                      # cache read again
Compare token_ids:
  fresh1 != fresh2            -> kernel nondeterminism; comparison is weak
  fresh* == fresh* and warm != fresh  -> PREFIX CACHE HIT IS WRONG (H1 smoking gun)
  all equal                   -> cache consistent for this prefix

Usage: python3 -u mimo_cacheab.py <port> <label> <outdir>
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

MODEL = "mimo-v2.6-flash"


def complete(port, prompt, max_tokens=220, prompt_logprobs=None):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 7,
        "return_token_ids": True,
    }
    if prompt_logprobs is not None:
        payload["prompt_logprobs"] = prompt_logprobs
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            j = json.load(r)
        ch = j["choices"][0]
        return ch.get("token_ids") or [], ch.get("text") or ""
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:300]}"
    except Exception as e:
        return None, repr(e)


def filler(n_words, seed):
    rng_words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
                 "kilo lima mike november oscar papa quebec romeo sierra tango "
                 "uniform victor whiskey xray yankee zulu").split()
    out = []
    x = seed
    for i in range(n_words):
        x = (1103515245 * x + 12345) % (1 << 31)
        out.append(rng_words[x % len(rng_words)])
    return " ".join(out)


def make_prompts():
    short = ("Explain in three detailed sentences why the seasons change on "
             "Earth, mentioning axial tilt.")
    # ~4k-word filler => ~5-6k tokens, many 256-token blocks, >128 window
    long_needle = (
        "Here is a long operations transcript you must remember.\n\n"
        + filler(4000, 1234) +
        "\n\nCRITICAL FACT: the backup passphrase is 'orchid-lantern-42'.\n\n"
        + filler(4000, 5678) +
        "\n\nQuestion: what is the backup passphrase? Answer with just the "
        "passphrase."
    )
    # multi-turn style: same long prefix, then a second question appended
    # (simulates turn-2 of an agent session reusing the turn-1 prefix)
    long_needle2 = long_needle + (
        "\n\nAlso: what two words in this transcript come right before the "
        "CRITICAL FACT line? Answer briefly."
    )
    logic = (
        "You are tracking a counter. It starts at 10. Then +5, then x2, then "
        "-7. There is also a list: apple, banana, cherry. "
        "Question 1: what is the counter value now? "
        "Question 2: name the three-item list in order."
    )
    return [("short", short, 400), ("long_needle", long_needle, 400),
            ("long_needle2", long_needle2, 400), ("logic", logic, 400)]


def main():
    port, label, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    print(f"# mimo_cacheab label={label} t={time.strftime('%F %T')}")
    results = {}
    for name, prompt, mt in make_prompts():
        print(f"\n=== {name} (prompt {len(prompt)} chars) ===")
        f1_ids, f1_txt = complete(port, prompt, mt, prompt_logprobs=0)
        f2_ids, f2_txt = complete(port, prompt, mt, prompt_logprobs=0)
        w1_ids, w1_txt = complete(port, prompt, mt)
        w2_ids, w2_txt = complete(port, prompt, mt)
        if f1_ids is None:
            print(f"  ERROR fresh1: {f1_txt}")
            continue
        det = f2_ids == f1_ids
        print(f"  fresh1 len={len(f1_ids)} text[:120]={f1_txt[:120]!r}")
        print(f"  fresh2==fresh1 (nondet control): {det}")
        print(f"  warm1 ==fresh1 (cache hit ok?):  {w1_ids == f1_ids}")
        print(f"  warm2 ==fresh1:                  {w2_ids == f1_ids}")
        print(f"  warm2 ==warm1:                   {w2_ids == w1_ids}")
        if w1_ids != f1_ids:
            for i, (a, b) in enumerate(zip(w1_ids, f1_ids)):
                if a != b:
                    print(f"  DIVERGENCE warm vs fresh at tok {i}: "
                          f"warm={w1_ids[max(0,i-3):i+4]} fresh={f1_ids[max(0,i-3):i+4]}")
                    print(f"  warm text : {w1_txt[:200]!r}")
                    print(f"  fresh text: {f1_txt[:200]!r}")
                    break
        results[name] = {
            "fresh1_ids": f1_ids, "fresh2_ids": f2_ids,
            "warm1_ids": w1_ids, "warm2_ids": w2_ids,
            "fresh1_text": f1_txt, "warm1_text": w1_txt,
            "fresh_det_ok": det,
            "warm_eq_fresh": w1_ids == f1_ids,
            "warm2_eq_fresh": w2_ids == f1_ids,
        }
    print("\n=== CACHE A/B SUMMARY ===")
    bad = 0
    for name, r in results.items():
        status = "OK" if (r["warm_eq_fresh"] and r["warm2_eq_fresh"]) else "CACHE-MISMATCH"
        if status != "OK":
            bad += 1
        print(f"  {name}: {status} (nondet_control={r['fresh_det_ok']})")
    print(f"VERDICT: {bad}/{len(results)} prompts show cache-hit != fresh-recompute"
          f" -> {'H1 PREFIX CACHE CORRUPTION' if bad else 'prefix cache consistent'}")
    with open(os.path.join(outdir, f"cacheab_{label}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
