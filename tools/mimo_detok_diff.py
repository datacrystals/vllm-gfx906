#!/usr/bin/env python3
"""Offline H2 check: are captured token ids themselves punctuation-broken,
or is only the server's incremental decode wrong?

Runs on the vLLM box (needs the model tokenizer + `tokenizers` lib).
Usage:
  python3 -u mimo_detok_diff.py /tmp/qprobe_x/qprobe_x.json

For every completion capture with `token_ids`:
  1. server_text       — text the API returned (incremental detokenizer)
  2. stream_text       — DecodeStream primed with prompt_token_ids (same
                         code path the server's FastIncrementalDetokenizer uses)
  3. full_decode       — tokenizer.decode(token_ids) offline
  4. pieces            — tokenizer.decode([id]) per token
Reports: first divergence server vs full_decode, whether an artifact char
maps to a token whose own decode is a punctuation piece, and the token ids
around any stray '.'.
"""
import json
import re
import sys

from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from transformers import AutoTokenizer

MODEL_DIR = "/data/ModelDownloader/MiMo-V2.6-Flash-INT4"
RE_LOW_DOT_LOW = re.compile(r"(?<=[a-z])\.\s+(?=[a-z])")


def find_artifact_spans(text):
    return [m.start() for m in RE_LOW_DOT_LOW.finditer(text)]


def char_to_token(cum, idx):
    """cum[k] = char length after k tokens. Return token index containing idx."""
    for k, c in enumerate(cum):
        if idx < c:
            return k
    return len(cum) - 1


def analyze(tag, text, token_ids, prompt_ids, hf_tok, raw_tok):
    print(f"\n#### {tag}  ids={len(token_ids)} prompt_ids={len(prompt_ids) if prompt_ids else 0}")
    if not token_ids:
        print("  no token_ids in capture; skip")
        return

    full = hf_tok.decode(token_ids, skip_special_tokens=False)
    # server-style incremental decode: DecodeStream primed with the prompt
    if prompt_ids:
        stream = DecodeStream(ids=list(prompt_ids), skip_special_tokens=True)
    else:
        stream = DecodeStream(skip_special_tokens=True)
    parts = []
    for tid in token_ids:
        try:
            step = stream.step(raw_tok, tid) or ""
        except Exception as e:
            step = f"<ERR {e}>"
        parts.append(step)
    stream_text = "".join(parts)

    pieces = [hf_tok.decode([tid], skip_special_tokens=False) for tid in token_ids]

    print(f"  server_text len={len(text)} stream_text len={len(stream_text)} "
          f"full_decode len={len(full)}")
    if text == full:
        print("  server_text == full_decode  => incremental decode matches offline decode")
    else:
        # find first divergence
        n = min(len(text), len(full))
        i = 0
        while i < n and text[i] == full[i]:
            i += 1
        print(f"  !! MISMATCH server_text vs full_decode at char {i}")
        print(f"     server: {text[max(0,i-40):i+40]!r}")
        print(f"     full  : {full[max(0,i-40):i+40]!r}")
    if text == stream_text:
        print("  server_text == stream_text  => matches DecodeStream replay")
    else:
        n = min(len(text), len(stream_text))
        i = 0
        while i < n and text[i] == stream_text[i]:
            i += 1
        print(f"  !! MISMATCH server_text vs stream_replay at char {i}")
        print(f"     server: {text[max(0,i-40):i+40]!r}")
        print(f"     replay: {stream_text[max(0,i-40):i+40]!r}")

    cum = []
    acc = 0
    for p in parts:
        acc += len(p)
        cum.append(acc)

    for which, ttext in (("server", text), ("full", full)):
        for off in find_artifact_spans(ttext):
            k = char_to_token(cum, off)
            lo, hi = max(0, k - 4), min(len(token_ids), k + 5)
            print(f"  ARTIFACT in {which}_text at char {off}: {ttext[max(0,off-30):off+30]!r}")
            print(f"    token[{k}] id={token_ids[k]} piece={pieces[k]!r} step={parts[k]!r}")
            print(f"    ids[{lo}:{hi}]={token_ids[lo:hi]}")
            print(f"    pieces={pieces[lo:hi]}")

    # any token whose own decode is bare punctuation?
    punct_ids = [(i, tid, pieces[i]) for i, tid in enumerate(token_ids)
                 if re.fullmatch(r"[.,;:!?…]+", pieces[i].strip() or "x")]
    mid = [p for p in punct_ids if 0 < p[0] < len(token_ids) - 2]
    print(f"  punctuation-only tokens: {len(punct_ids)} total, {len(mid)} mid-sequence")
    for i, tid, pc in mid[:12]:
        prev = pieces[max(0, i - 2):i + 3]
        print(f"    tok[{i}] id={tid} piece={pc!r} ctx_pieces={prev}")


def main():
    path = sys.argv[1]
    data = json.load(open(path))
    print(f"# detok_diff on {path}")
    hf_tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    raw_tok = hf_tok._tokenizer
    n_done = 0
    for rec in data.get("dump", []):
        tag = rec.get("tag", "?")
        resp = rec.get("response") or {}
        choices = resp.get("choices") if isinstance(resp, dict) else None
        if not choices:
            continue
        ch = choices[0]
        # completion shape: choices[0].text/token_ids
        text = ch.get("text")
        ids = ch.get("token_ids")
        prompt_ids = ch.get("prompt_token_ids")
        # chat shape: choices[0].message.content/token_ids
        if ids is None and isinstance(ch.get("message"), dict):
            msg = ch["message"]
            text = msg.get("content") or msg.get("reasoning_content") or ""
            ids = msg.get("token_ids")
        if ids is None:
            print(f"\n#### {tag}: no token_ids field (keys={sorted(ch.keys())})")
            continue
        analyze(tag, text or "", ids, prompt_ids, hf_tok, raw_tok)
        n_done += 1
    print(f"\n# analyzed {n_done} captures")


if __name__ == "__main__":
    main()
