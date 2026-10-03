#!/usr/bin/env python3
"""MiMo output-quality probe: stray-'.' hunt + raw token-id capture,
prefix-cache A/B determinism, multi-turn logic probes, chat-template render.

Stdlib-only (runs anywhere). Usage:
  python3 -u mimo_qprobe.py <port> <label> <outdir> [--part all|dot|cache|multi|render]

Every request uses temperature=0 (greedy), max_tokens>=400, seed=7.
Prints BOTH content and reasoning_content for chat calls.
Saves a full JSON dump of every request/response under <outdir>/.
Exit code 0 always (this is a capture tool; verdicts are printed).
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

MODEL = "mimo-v2.6-flash"
SEED = 7

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def post(port, path, payload, timeout=300):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode()
            return r.status, json.loads(body), time.time() - t0
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return e.code, json.loads(body), time.time() - t0
        except Exception:
            return e.code, {"_raw": body[:2000]}, time.time() - t0
    except Exception as e:
        return -1, {"_error": repr(e)}, time.time() - t0


def complete(port, prompt, max_tokens=512, extra=None):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": SEED,
        "logprobs": 5,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
    }
    if extra:
        payload.update(extra)
    return post(port, "/v1/completions", payload)


def chat(port, messages, max_tokens=400, extra=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": SEED,
        "logprobs": True,
        "top_logprobs": 3,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
    }
    if extra:
        payload.update(extra)
    return post(port, "/v1/chat/completions", payload)


# ---------------------------------------------------------------------------
# stray-punctuation scan
# ---------------------------------------------------------------------------
# suspicious: lowercase word, period, space, lowercase word (prose sentence
# boundary should be capital).  Filter known abbreviations before the dot.
ABBR = {
    "e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "st", "no", "approx",
    "incl", "fig", "al", "cf", "ca", "dept", "est", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "jr",
    "sr", "prof", "vol", "pp", "ed", "eds", "min", "max", "avg", "sq", "ft",
    "lb", "oz", "am", "pm", "u.s", "a.m", "p.m",
}
RE_LOW_DOT_LOW = re.compile(r"(?<=[a-z])\.\s+(?=[a-z])")
RE_DOT_GLUED = re.compile(r"(?<=[a-z])\.(?=[a-z])")
RE_SPACE_DOT_SPACE = re.compile(r"\S\s\.\s")


def scan_artifacts(text):
    out = []
    for name, rx in (("low.space.low", RE_LOW_DOT_LOW), ("low.glued.low", RE_DOT_GLUED),
                     ("space.dot.space", RE_SPACE_DOT_SPACE)):
        for m in rx.finditer(text):
            pre = text[max(0, m.start() - 30):m.start()]
            if name == "low.space.low":
                word = re.search(r"([A-Za-z.]+)$", pre)
                if word and word.group(1).lower().rstrip(".") in ABBR:
                    continue
                # single-letter initial like "J. smith" already excluded (?<=[a-z])
            out.append({
                "kind": name,
                "at": m.start(),
                "match": repr(text[m.start():m.end() + 20]),
                "ctx": repr(text[max(0, m.start() - 40):m.end() + 40]),
                "ctx_bytes": text[max(0, m.start() - 20):m.end() + 20].encode("utf-8", "replace").hex(),
            })
    return out


def summarize_completion(status, resp, tag, art_hits, dump):
    if status != 200:
        print(f"[{tag}] HTTP {status}: {str(resp)[:300]}")
        return
    ch = resp["choices"][0]
    text = ch.get("text", "")
    ids = ch.get("token_ids") or []
    lp = ch.get("logprobs") or {}
    toks = lp.get("tokens") or []
    arts = scan_artifacts(text)
    print(f"[{tag}] finish={ch.get('finish_reason')} chars={len(text)} ids={len(ids)} "
          f"pieces={len(toks)} artifacts={len(arts)}")
    print(f"[{tag}] TEXT: {text[:400]!r}")
    if arts:
        for a in arts:
            print(f"[{tag}] ARTIFACT {a['kind']} at {a['at']}: {a['ctx']}")
            print(f"[{tag}] ARTIFACT bytes: {a['ctx_bytes']}")
        art_hits.append({"tag": tag, "artifacts": arts, "text": text, "token_ids": ids})
    dump.append({"tag": tag, "status": status, "response": resp})


# ---------------------------------------------------------------------------
# part: dot  — greedy prose generation, artifact hunt, raw ids
# ---------------------------------------------------------------------------
DOT_PROMPTS = [
    ("lighthouse", "Write a long, detailed story about a lighthouse keeper named "
     "Elias who discovers something strange one stormy night. Write at least "
     "six hundred words, in complete sentences."),
    ("watercycle", "Explain in detail how the water cycle works, covering "
     "evaporation, condensation, precipitation, and runoff. Write at least "
     "five hundred words in full prose sentences."),
    ("history", "Write a detailed history of the telephone from its invention to "
     "the modern smartphone. At least five hundred words, full sentences, no "
     "bullet points."),
    ("quickbrown", "The quick brown fox jumps over the lazy dog. The quick brown "
     "fox jumps over the lazy dog again. Continue this passage with many more "
     "sentences describing what the fox and the dog do next."),
    ("cooking", "Describe, in flowing prose of at least five hundred words, how "
     "to make a traditional French onion soup from start to finish."),
    ("debate", "Write a balanced essay discussing both sides of the debate over "
     "whether homework should be abolished in primary school. At least five "
     "hundred words."),
]


def part_dot(port, outdir, dump, art_hits):
    print("=== PART dot: greedy prose artifact hunt (completions + chat) ===")
    for name, prompt in DOT_PROMPTS:
        st, resp, dt = complete(port, prompt, max_tokens=700)
        summarize_completion(st, resp, f"dot/comp/{name}", art_hits, dump)
        print(f"[dot/comp/{name}] {dt:.1f}s")
        # chat variant: the channel the user's agent actually uses
        st, resp, dt = chat(port, [{"role": "user", "content": prompt}], max_tokens=700)
        _dump_chat(st, resp, f"dot/chat/{name}", art_hits, dump)
        print(f"[dot/chat/{name}] {dt:.1f}s")
    # same-prompt repeat (cache-hit determinism) on one prompt, both channels
    p = DOT_PROMPTS[0][1]
    st1, r1, _ = complete(port, p, max_tokens=300)
    st2, r2, _ = complete(port, p, max_tokens=300)
    ids1 = (r1.get("choices") or [{}])[0].get("token_ids")
    ids2 = (r2.get("choices") or [{}])[0].get("token_ids")
    same = ids1 == ids2
    print(f"[dot/repeat] cache-hit determinism: {'IDENTICAL' if same else 'DIVERGED!!'} "
          f"(ids {None if ids1 is None else len(ids1)} vs {None if ids2 is None else len(ids2)})")
    if not same and ids1 and ids2:
        for i, (a, b) in enumerate(zip(ids1, ids2)):
            if a != b:
                print(f"[dot/repeat] first id divergence at {i}: {a} vs {b}")
                print(f"[dot/repeat] ids1[{max(0,i-3)}:{i+4}]={ids1[max(0,i-3):i+4]}")
                print(f"[dot/repeat] ids2[{max(0,i-3)}:{i+4}]={ids2[max(0,i-3):i+4]}")
                break
    dump.append({"tag": "dot/repeat", "ids1": ids1, "ids2": ids2, "identical": same,
                 "r1": r1, "r2": r2})


# ---------------------------------------------------------------------------
# part: cache — warm vs cold prefix-reuse A/B with FROZEN history
# ---------------------------------------------------------------------------
CACHE_SYS = (
    "You are my notes assistant. You maintain NOTES.md in your head across "
    "this conversation.\nRULES:\n1. Never delete a paragraph without asking "
    "me first.\n2. Always keep every fruit name mentioned in the notes.\n"
    "3. When I ask 'status', answer with exactly the current paragraph list."
)
CACHE_U1 = (
    "NOTES.md currently contains these paragraphs:\n"
    'P1: "Apples are stored in the cellar."\n'
    'P2: "Oranges arrive on Tuesdays."\n'
    'P3: "Bananas must not be refrigerated."\n'
    "Please confirm you have this state."
)
# frozen assistant replies so run1 and run2 send byte-identical histories
CACHE_A1 = ("Confirmed. NOTES.md holds three paragraphs: P1 Apples are stored in "
            "the cellar, P2 Oranges arrive on Tuesdays, P3 Bananas must not be "
            "refrigerated. I will keep all fruit names and will not delete "
            "anything without asking first.")
CACHE_TURNS = [
    ('Add a new paragraph about grapes: "Grapes are stored in the loft."',
     None),  # None => model reply captured live in run1, then FROZEN for run2
    ("Simplify P2 to: \"Oranges arrive Tuesdays.\"",
     None),
    ("status",
     None),
    ("Delete P3 now.",
     None),
    ("status",
     None),
]


def part_cache(port, outdir, dump, art_hits):
    print("=== PART cache: frozen-history warm-vs-cold prefix cache A/B ===")
    frozen = [CACHE_A1]
    run_outputs = {}
    for run in (1, 2):
        msgs = [{"role": "system", "content": CACHE_SYS},
                {"role": "user", "content": CACHE_U1}]
        # turn 1 fixed
        if run == 1:
            st, resp, _ = chat(port, msgs, max_tokens=400)
            _dump_chat(st, resp, f"cache/run{run}/turn0", art_hits, dump)
            out0 = _chat_text(resp)
            frozen[0] = out0
        msgs.append({"role": "assistant", "content": frozen[0]})
        for i, (u, _) in enumerate(CACHE_TURNS):
            msgs.append({"role": "user", "content": u})
            st, resp, _ = chat(port, msgs, max_tokens=400)
            _dump_chat(st, resp, f"cache/run{run}/turn{i+1}", art_hits, dump)
            txt = _chat_text(resp)
            run_outputs.setdefault(i + 1, {})[run] = txt
            if run == 1:
                # freeze this reply so run2 sends identical history
                if len(frozen) <= i + 1:
                    frozen.append(txt)
                msgs.append({"role": "assistant", "content": frozen[i + 1]})
            else:
                msgs.append({"role": "assistant", "content": frozen[i + 1]})
    print("--- warm-vs-cold comparison (same input history, run1 cold vs run2 warm) ---")
    ndiff = 0
    for t in sorted(run_outputs):
        a, b = run_outputs[t][1], run_outputs[t][2]
        ident = a == b
        if not ident:
            ndiff += 1
        print(f"[cache] turn{t}: {'IDENTICAL' if ident else 'DIVERGED!!'} "
              f"({len(a)} vs {len(b)} chars)")
        if not ident:
            print(f"   run1: {a[:300]!r}")
            print(f"   run2: {b[:300]!r}")
    print(f"[cache] VERDICT: {ndiff} diverging turn(s) of {len(run_outputs)} "
          f"-> {'PREFIX-CACHE SUSPECT' if ndiff else 'prefix cache consistent (turns deterministic)'}")
    dump.append({"tag": "cache/ab", "run_outputs": run_outputs})


def _chat_text(resp):
    try:
        m = resp["choices"][0]["message"]
        return (m.get("content") or "")
    except Exception:
        return ""


def _dump_chat(st, resp, tag, art_hits, dump):
    if st != 200:
        print(f"[{tag}] HTTP {st}: {str(resp)[:300]}")
        dump.append({"tag": tag, "status": st, "response": resp})
        return
    m = resp["choices"][0]["message"]
    content = m.get("content") or ""
    reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    ids = m.get("token_ids") or []
    arts = scan_artifacts(content) + scan_artifacts(reasoning)
    print(f"[{tag}] finish={resp['choices'][0].get('finish_reason')} "
          f"content={len(content)} reasoning={len(reasoning)} ids={len(ids)} artifacts={len(arts)}")
    print(f"[{tag}] CONTENT: {content[:400]!r}")
    if reasoning:
        print(f"[{tag}] REASONING: {reasoning[:300]!r}")
    if arts:
        for a in arts:
            print(f"[{tag}] ARTIFACT {a['kind']} at {a['at']}: {a['ctx']}")
        art_hits.append({"tag": tag, "artifacts": arts, "text": content,
                         "reasoning": reasoning, "token_ids": ids})
    dump.append({"tag": tag, "status": st, "response": resp})


# ---------------------------------------------------------------------------
# part: multi — multi-turn logic / state-tracking probes
# ---------------------------------------------------------------------------
def run_scripted_turns(port, dump, art_hits, tag, system, turns, score_fn=None):
    """turns: list of (user_msg, expected_checks) — model replies live."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    for i, (u, checks) in enumerate(turns):
        msgs.append({"role": "user", "content": u})
        st, resp, _ = chat(port, msgs, max_tokens=400)
        _dump_chat(st, resp, f"{tag}/t{i}", art_hits, dump)
        txt = _chat_text(resp)
        rtxt = ""
        if st == 200:
            m = resp["choices"][0]["message"]
            rtxt = m.get("reasoning_content") or ""
        msgs.append({"role": "assistant", "content": txt})
        if checks:
            ok_all = True
            for name, pat, want in checks:
                got = bool(re.search(pat, txt, re.I | re.S))
                ok = got == want
                ok_all = ok_all and ok
                print(f"[{tag}/t{i}] CHECK {name}: {'ok' if ok else 'FAIL'} "
                      f"(want {want}, got {got})")
            print(f"[{tag}/t{i}] TURN-SCORE: {'PASS' if ok_all else 'FAIL'}")
    return msgs


NOTES_SYS = CACHE_SYS
NOTES_TURNS = [
    (CACHE_U1, [("confirm", r"confirm|have this state|noted|P1", True)]),
    ('Add a new paragraph about grapes: "Grapes are stored in the loft."',
     [("grapes", r"grape", True)]),
    ("Simplify P2 to: \"Oranges arrive Tuesdays.\"",
     [("simplified", r"Oranges arrive Tuesdays\.?", True)]),
    ("status",
     [("p1", r"apples are stored in the cellar", True),
      ("p2simplified", r"oranges arrive tuesdays\.?", True),
      ("p3", r"bananas must not be refrigerated", True),
      ("p4", r"grapes are stored in the loft", True),
      ("no_extra_delete", r"celor", False)]),
    ("Delete P3 now.",
     [("asks_first", r"delete|remov|confirm|shall i|are you sure|rule", True),
      ("silent_delete", r"^(done|deleted|i have deleted)", False)]),
    ("status",
     [("p3_still", r"bananas must not be refrigerated", True),
      ("p1_still", r"apples are stored in the cellar", True),
      ("grapes_still", r"grapes are stored in the loft", True)]),
    ("What rules did I give you in my very first message?",
     [("rule1", r"never delete|without asking|without confirming", True),
      ("rule2", r"fruit", True)]),
]


def part_multi(port, outdir, dump, art_hits):
    print("=== PART multi: state-tracking / rule-following probes ===")
    print("--- probe A: notes state machine ---")
    run_scripted_turns(port, dump, art_hits, "multi/notes", NOTES_SYS, NOTES_TURNS)

    print("--- probe B: running counter ---")
    counter_turns = [
        ("Start a counter at 10. Acknowledge.", [("ten", r"\b10\b", True)]),
        ("Add 5 to the counter.", [("fifteen", r"\b15\b", True)]),
        ("Multiply the counter by 2.", [("thirty", r"\b30\b", True)]),
        ("Subtract 7 from the counter. What is the value now?",
         [("twentythree", r"\b23\b", True)]),
        ("What was the value right after the multiply step?",
         [("thirty_again", r"\b30\b", True)]),
        ("Divide the current value by 3 and round down. What is it now?",
         [("seven", r"\b7\b", True)]),
    ]
    run_scripted_turns(port, dump, art_hits, "multi/counter", "", counter_turns)

    print("--- probe C: standing instruction + self-report ---")
    instr_turns = [
        ("From now on, every one of your answers must begin with the word "
         "BANANA. Acknowledge the rule.",
         [("ack", r"banana", True)]),
        ("What is 2+2?",
         [("starts_banana", r"^\s*BANANA", True), ("four", r"\b4\b", True)]),
        ("What color is the sky on a clear day?",
         [("starts_banana", r"^\s*BANANA", True), ("blue", r"blue", True)]),
        ("Which rule did I give you in my first message?",
         [("starts_banana", r"^\s*BANANA", True),
          ("rule_recall", r"banana", True)]),
    ]
    run_scripted_turns(port, dump, art_hits, "multi/instr", "", instr_turns)


# ---------------------------------------------------------------------------
# part: render — dump what the server sends for a 3-turn chat (H4)
# ---------------------------------------------------------------------------
def part_render(port, outdir, dump, art_hits):
    print("=== PART render: chat template / prompt-token dump (H4) ===")
    msgs = [
        {"role": "system", "content": "You are a helpful assistant. Track state carefully."},
        {"role": "user", "content": "I have three apples."},
        {"role": "assistant", "content": "Understood: three apples."},
        {"role": "user", "content": "I eat one. How many are left?"},
    ]
    for path in ("/v1/chat/completions/render", "/v1/completions/render", "/tokenize"):
        st, resp, _ = post(port, path, {"model": MODEL, "messages": msgs, "prompt": "x"})
        print(f"[render] {path} -> HTTP {st}")
        print(f"[render] body: {json.dumps(resp)[:2000]}")
        dump.append({"tag": f"render/{path}", "status": st, "response": resp})
    # also echo the 3-turn prompt through completions to get prompt_token_ids
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant. Track state carefully.<|im_end|>\n"
        "<|im_start|>user\nI have three apples.<|im_end|>\n"
        "<|im_start|>assistant\nUnderstood: three apples.<|im_end|>\n"
        "<|im_start|>user\nI eat one. How many are left?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    st, resp, _ = complete(port, prompt, max_tokens=64, extra={"echo": True, "prompt_logprobs": 0})
    if st == 200:
        ch = resp["choices"][0]
        print(f"[render] echo completions: text={ch.get('text')!r}")
        print(f"[render] prompt_token_ids len={len(ch.get('prompt_token_ids') or [])}")
    else:
        print(f"[render] echo completions HTTP {st}: {str(resp)[:300]}")
    dump.append({"tag": "render/echo", "status": st, "response": resp})


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    port, label, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    part = "all"
    if "--part" in sys.argv:
        part = sys.argv[sys.argv.index("--part") + 1]
    os.makedirs(outdir, exist_ok=True)
    print(f"# mimo_qprobe label={label} part={part} port={port} t={time.strftime('%F %T')}")
    st, resp, _ = post(port, "/v1/models", {})
    print(f"[models] HTTP {st}: {str(resp)[:200]}")

    dump, art_hits = [], []
    t0 = time.time()
    if part in ("all", "dot"):
        part_dot(port, outdir, dump, art_hits)
    if part in ("all", "cache"):
        part_cache(port, outdir, dump, art_hits)
    if part in ("all", "multi"):
        part_multi(port, outdir, dump, art_hits)
    if part in ("all", "render"):
        part_render(port, outdir, dump, art_hits)

    print("=== SUMMARY ===")
    print(f"label={label} elapsed={time.time()-t0:.0f}s artifact_hits={len(art_hits)}")
    for a in art_hits:
        print(f"  ARTIFACT in {a['tag']}: {a['artifacts'][0]['ctx']}")
    with open(os.path.join(outdir, f"qprobe_{label}.json"), "w") as f:
        json.dump({"label": label, "art_hits": art_hits, "dump": dump}, f, indent=1)
    print(f"dump -> {os.path.join(outdir, f'qprobe_{label}.json')}")


if __name__ == "__main__":
    main()
