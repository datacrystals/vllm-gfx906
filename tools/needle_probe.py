#!/usr/bin/env python3
"""Needle-at-depth probe for long-context quality.

Usage:
  python3 needle_probe.py <port> <prompt_tokens> <depth_frac> [max_tokens] [variant]

Needle PASS = answer contains 73912 (prints NEEDLE-PROBE: PASS/FAIL).

IMPORTANT: prompt tokens are prefix-cached block-by-block. Repeating the same
filler reuses cached blocks (timings of repeat prompts are ~7s, NOT real
prefill time) and can skip most of the prefill you meant to exercise. Pass a
different <variant> (0..) to switch the filler wording so every run is cold.
Variant 0 keeps the historical filler text exactly.
"""
import json, sys, time, urllib.request

port = sys.argv[1] if len(sys.argv) > 1 else "9700"
target = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
depth = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
maxtok = int(sys.argv[4]) if len(sys.argv) > 4 else 16
variant = int(sys.argv[5]) if len(sys.argv) > 5 else 0

FILLERS = [
    "The meadow stretches quietly beyond the old stone wall, and the "
    "wind moves through it without hurry. ",
    "Along the ridge the pines stand in patient rows, and the light "
    "settles between them without haste. ",
    "The harbor keeps its slow rhythm while the boats lean against "
    "their ropes and the tide turns. ",
    "Snow gathers on the porch rails overnight, and by morning the "
    "garden path has vanished completely. ",
    "The old mill wheel turns once, pauses, and turns again as the "
    "stream gathers strength after rain. ",
]
filler = FILLERS[variant % len(FILLERS)]
if variant >= len(FILLERS):
    filler = f"[pass {variant}] " + filler

needle = "The special code is 73912. "
q = " What is the special code? Reply with just the number."
approx_tok_per_rep = len(filler) / 3.6
n_reps = int((target - 200) / approx_tok_per_rep)
at = int(n_reps * depth)
parts = [filler * at, needle, filler * (n_reps - at), q]
prompt = "".join(parts)

body = json.dumps({"model": __import__("os").environ.get("NEEDLE_MODEL", "glm-5.3-flash"), "prompt": prompt,
                   "max_tokens": maxtok, "temperature": 0}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions", data=body,
    headers={"Content-Type": "application/json"})
print(f"prompt_chars={len(prompt)} depth={depth} variant={variant} ...")
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=5400) as r:
        d = json.load(r)
    txt = d["choices"][0]["text"]
    pt = d.get("usage", {}).get("prompt_tokens", "?")
    dt = time.time() - t0
    ok = "73912" in txt
    print(f"prompt_tokens={pt} time={dt:.0f}s")
    print(f"ANSWER: {txt!r}")
    print("NEEDLE-PROBE:", "PASS" if ok else "FAIL")
except Exception as e:
    print(f"ERROR after {time.time()-t0:.0f}s: {e}")
