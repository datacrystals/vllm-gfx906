#!/usr/bin/env python3
"""Quantify T=0 (greedy) numerics nondeterminism on the live server.

Runs the SAME request N times (fresh compute: prompt_logprobs forces
skip_reading_prefix_cache) and reports:
  * how many distinct token streams appear
  * where the first token divergence happens (prefill[0] vs decode[k])
  * max |delta logprob| for tokens before the first divergence (numeric noise)
  * same for the chat API (the channel the agent uses)

Usage: python3 -u mimo_nondet_hammer.py <port> <label> <outdir> [n_runs]
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

MODEL = "mimo-v2.6-flash"


def post(path, payload, timeout=300):
    req = urllib.request.Request(
        f"http://127.0.0.1:{sys.argv[1]}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"_http": e.code, "_body": e.read().decode()[:500]}
    except Exception as e:
        return {"_err": repr(e)}


def comp_once(prompt, mt=250):
    j = post("/v1/completions", {
        "model": MODEL, "prompt": prompt, "max_tokens": mt,
        "temperature": 0.0, "seed": 7, "logprobs": 5,
        "prompt_logprobs": 0, "return_token_ids": True,
    })
    ch = (j.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    return (ch.get("token_ids") or [], lp.get("token_logprobs") or [],
            ch.get("text") or "", (lp.get("tokens") or [])[:12])


def chat_once(msgs, mt=250):
    j = post("/v1/chat/completions", {
        "model": MODEL, "messages": msgs, "max_tokens": mt,
        "temperature": 0.0, "seed": 7, "logprobs": True, "top_logprobs": 3,
    })
    ch = (j.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return (msg.get("content") or "", msg.get("reasoning_content") or "")


def analyze(name, runs):
    print(f"\n=== {name}: {len(runs)} runs ===")
    id_seqs = [r[0] for r in runs]
    texts = [r[2] for r in runs]
    uniq = {}
    for i, s in enumerate(id_seqs):
        uniq.setdefault(tuple(s), []).append(i)
    print(f"  distinct token streams: {len(uniq)} of {len(runs)}")
    for tok_seq, idxs in uniq.items():
        print(f"    stream from runs {idxs}: len={len(tok_seq)} first12={list(tok_seq[:12])}")
    # first divergence across runs vs run 0
    first_div = []
    for i in range(1, len(id_seqs)):
        a, b = id_seqs[0], id_seqs[i]
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        first_div.append(k)
    print(f"  first-divergence offsets vs run0: {first_div}")
    # numeric noise in logprob values over the common prefix
    max_dlp = 0.0
    worst = None
    base = runs[0][1]
    for i in range(1, len(runs)):
        other = runs[i][1]
        for k in range(min(len(base), len(other))):
            d = abs((base[k] or 0) - (other[k] or 0))
            if d > max_dlp:
                max_dlp, worst = d, (i, k, base[k], other[k])
    print(f"  max |dlogprob| on common tokens: {max_dlp:.6f} at {worst}")
    return {"name": name, "n": len(runs), "distinct": len(uniq),
            "first_div": first_div, "max_dlogprob": max_dlp}


def main():
    port, label, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    os.makedirs(outdir, exist_ok=True)
    print(f"# nondet_hammer label={label} n={n} t={time.strftime('%F %T')}")
    results = []
    prompts = [
        ("short_explain", "Explain in three detailed sentences why the seasons "
         "change on Earth, mentioning axial tilt."),
        ("medium_story", "Write three sentences about a lighthouse keeper who "
         "finds a message in a bottle. Be specific and concrete."),
    ]
    for name, p in prompts:
        runs = [comp_once(p) for _ in range(n)]
        results.append(analyze(name, runs))
        for i, r in enumerate(runs):
            print(f"  run{i} text[:100]={r[2][:100]!r}")
    # chat channel
    chat_runs = [chat_once([{"role": "user",
                             "content": "Name three fruits. Then say done."}])
                 for _ in range(n)]
    n_distinct = len({r[0] for r in chat_runs})
    print(f"\n=== chat_fruits: {n} runs, distinct contents: {n_distinct} ===")
    for i, r in enumerate(chat_runs):
        print(f"  run{i} content[:100]={r[0][:100]!r}")
    results.append({"name": "chat_fruits", "distinct": n_distinct})
    with open(os.path.join(outdir, f"nondet_{label}.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\n# summary: {[ (r['name'], r.get('distinct')) for r in results ]}")


if __name__ == "__main__":
    main()
