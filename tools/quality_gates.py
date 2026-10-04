#!/usr/bin/env python3
"""Quick quality gates: France probe + needle 24k diagnosis."""
import json
import sys
import urllib.request

MODEL = "mimo-v2.6-flash"


def chat(content, mt=200):
    req = urllib.request.Request(
        "http://127.0.0.1:9700/v1/chat/completions",
        data=json.dumps({"model": MODEL,
                         "messages": [{"role": "user", "content": content}],
                         "max_tokens": mt, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    j = json.load(urllib.request.urlopen(req, timeout=300))
    m = j["choices"][0]["message"]
    print("CONTENT:", repr((m.get("content") or "")[:300]))
    print("REASONING:", repr((m.get("reasoning_content") or "")[:200]))


print("== FRANCE PROBE ==")
chat("What is the capital of France? Answer briefly.")

print("\n== NEEDLE-ISH 24k single-turn recall (via completions) ==")
filler = " ".join(["alpha bravo charlie delta echo foxtrot golf hotel"] * 4000)
needle = (filler[: 90000] +
          "\nCRITICAL FACT: the backup passphrase is orchid-lantern-42.\n" +
          filler[90000:])
prompt = needle + "\nQuestion: what is the backup passphrase? Answer briefly."
req = urllib.request.Request(
    "http://127.0.0.1:9700/v1/completions",
    data=json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": 400,
                     "temperature": 0}).encode(),
    headers={"Content-Type": "application/json"})
try:
    j = json.load(urllib.request.urlopen(req, timeout=600))
    print("NEEDLE TEXT:", repr(j["choices"][0].get("text", "")[:300]))
except Exception as e:
    print("NEEDLE ERROR:", e)
