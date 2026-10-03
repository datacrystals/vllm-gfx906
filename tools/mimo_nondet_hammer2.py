#!/usr/bin/env python3
"""Nondeterminism hammer v2: quantify flip MAGNITUDE.

For N identical fresh greedy runs (prompt_logprobs=0 => no cache read):
  * first divergence index per run vs run0
  * at each divergence index: the top-3 candidates + logprobs from BOTH runs
    -> if top1/top2 gap is tiny the flip is a benign near-tie; if the runs
       disagree on a HIGH-margin token the numerics are seriously wrong
  * |dlogprob| strictly BEFORE first divergence (true numeric noise floor)
Saves everything to JSON.

Usage: python3 -u mimo_nondet_hammer2.py <port> <label> <outdir> [n_runs]
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
    except Exception as e:
        return {"_err": repr(e)}


def comp_once(prompt, mt=200):
    j = post("/v1/completions", {
        "model": MODEL, "prompt": prompt, "max_tokens": mt,
        "temperature": 0.0, "seed": 7, "logprobs": 5,
        "prompt_logprobs": 0, "return_token_ids": True,
    })
    ch = (j.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    return {
        "ids": ch.get("token_ids") or [],
        "tokens": lp.get("tokens") or [],
        "logprobs": lp.get("token_logprobs") or [],
        "top": lp.get("top_logprobs") or [],
        "text": ch.get("text") or "",
    }


def main():
    port, label, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    os.makedirs(outdir, exist_ok=True)
    prompt = ("Explain in three detailed sentences why the seasons change on "
              "Earth, mentioning axial tilt.")
    print(f"# nondet_hammer2 label={label} n={n} t={time.strftime('%F %T')}")
    runs = [comp_once(prompt) for _ in range(n)]
    r0 = runs[0]
    max_noise_pre_div = 0.0
    for i, r in enumerate(runs):
        ids = r["ids"]
        k = 0
        while k < min(len(ids), len(r0["ids"])) and ids[k] == r0["ids"][k]:
            k += 1
        # true numeric noise on the shared prefix (same token stream)
        noise = 0.0
        for j in range(k):
            a = r["logprobs"][j] if j < len(r["logprobs"]) else None
            b = r0["logprobs"][j] if j < len(r0["logprobs"]) else None
            if a is not None and b is not None:
                noise = max(noise, abs(a - b))
        max_noise_pre_div = max(max_noise_pre_div, noise)
        print(f"\nrun{i}: first_div_at={k} (of {len(ids)}), "
              f"max|dlogprob| on shared prefix={noise:.6f}")
        if k < len(ids):
            def topk(r, j):
                t = r["top"][j] if j < len(r["top"]) else None
                if isinstance(t, dict):
                    return sorted(((v, kk) for kk, v in t.items()), reverse=True)[:3]
                return t
            print(f"  run0  tok={r0['tokens'][k]!r} lp={r0['logprobs'][k]:.4f} "
                  f"top3={topk(r0, k)}")
            print(f"  run{i} tok={r['tokens'][k]!r} lp={r['logprobs'][k]:.4f} "
                  f"top3={topk(r, k)}")
    print(f"\n# global: max|dlogprob| on shared prefixes = {max_noise_pre_div:.6f}")
    print(f"# distinct streams: {len({tuple(r['ids']) for r in runs})}/{n}")
    with open(os.path.join(outdir, f"nondet2_{label}.json"), "w") as f:
        json.dump({"runs": runs}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
