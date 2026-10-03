#!/usr/bin/env python3
"""Offline unit test for the patched deepseek_r1 parser (no vLLM boot)."""
import sys, types

sys.path.insert(0, "/data/vllm-gfx906-dsv4/vllm_dsv4_env/lib/python3.12/site-packages")

mod = types.ModuleType("vllm.entrypoints.openai.engine.protocol")

class DeltaMessage:
    def __init__(self, reasoning=None, content=None):
        self.reasoning, self.content = reasoning, content
    def __repr__(self):
        return f"Delta(r={self.reasoning!r}, c={self.content!r})"

mod.DeltaMessage = DeltaMessage
sys.modules["vllm.entrypoints.openai.engine.protocol"] = mod

import importlib
p = importlib.import_module("vllm.reasoning.deepseek_r1_reasoning_parser")

# real instance without __init__ (start_token/end_token are pure properties)
f = object.__new__(p.DeepSeekR1ReasoningParser)
D = p.DeepSeekR1ReasoningParser

# 1) non-streaming plain
r, c = f.extract_reasoning( "<think>\nthinking here\n</think>hi", None)
print("1 nonstream:", repr(r), repr(c))
assert r == "\nthinking here\n" and c == "hi", (r, c)

# 2) non-streaming, no boundary yet
r, c = f.extract_reasoning( "<think>\nthinking...", None)
print("2 no-boundary:", repr(r), repr(c))
assert r == "\nthinking..." and c is None, (r, c)

# 3) doubled end marker run
r, c = f.extract_reasoning( "<think>reasoning</think>  </think>reply", None)
print("3 run-end:", repr(r), repr(c))
assert r == "reasoning" and c == "reply", (r, c)

# 4) streaming: start token split across deltas
text = "<think>\nthinking\n</think>answer"
deltas = ["<th", "ink>\nthi", "nking\n</", "think>", "answer"]
assert "".join(deltas) == text
prev, got_r, got_c = "", [], []
for d in deltas:
    cur = prev + d
    dm = f.extract_reasoning_streaming( prev, cur, d, [], [], [])
    if dm:
        if dm.reasoning:
            got_r.append(dm.reasoning)
        if dm.content:
            got_c.append(dm.content)
    prev = cur
R, C = "".join(got_r), "".join(got_c)
print("4 stream-split-start:", repr(R), repr(C))
assert "think>" not in R and not R.startswith("think"), R
assert R == "\nthinking\n", R
assert C == "answer", C

# 5) streaming: whole start token in first delta
prev, got_r, got_c = "", [], []
for d in ["<think>\n", "more\n</", "think>", "out"]:
    cur = prev + d
    dm = f.extract_reasoning_streaming( prev, cur, d, [], [], [])
    if dm:
        if dm.reasoning:
            got_r.append(dm.reasoning)
        if dm.content:
            got_c.append(dm.content)
    prev = cur
R, C = "".join(got_r), "".join(got_c)
print("5 stream-whole-start:", repr(R), repr(C))
assert "think" not in R and R == "\nmore\n", R
assert C == "out", C

# 6) streaming: model emits NO start token (compat path must be untouched)
prev, got_r, got_c = "", [], []
for d in ["plain think", "ing\n</think", ">\nans"]:
    cur = prev + d
    dm = f.extract_reasoning_streaming( prev, cur, d, [], [], [])
    if dm:
        if dm.reasoning:
            got_r.append(dm.reasoning)
        if dm.content:
            got_c.append(dm.content)
    prev = cur
R, C = "".join(got_r), "".join(got_c)
print("6 stream-no-start:", repr(R), repr(C))
assert R == "plain thinking\n", R
assert C == "\nans", C


# 7) streaming: held marker prefix diverges -> chars must be flushed
prev, got_r, got_c = "", [], []
for d in ["<th", "inkX more\n</", "think>", "ok"]:
    cur = prev + d
    dm = f.extract_reasoning_streaming(prev, cur, d, [], [], [])
    if dm:
        if dm.reasoning:
            got_r.append(dm.reasoning)
        if dm.content:
            got_c.append(dm.content)
    prev = cur
R, C = "".join(got_r), "".join(got_c)
print("7 stream-diverge-head:", repr(R), repr(C))
assert R == "<thinkX more\n", R
assert C == "ok", C

# 8) streaming: doubled end-marker run mid-stream
prev, got_r, got_c = "", [], []
for d in ['<th', 'ink>rea</', 'think>', '</', 'think>', "tail"]:
    cur = prev + d
    dm = f.extract_reasoning_streaming(prev, cur, d, [], [], [])
    if dm:
        if dm.reasoning:
            got_r.append(dm.reasoning)
        if dm.content:
            got_c.append(dm.content)
    prev = cur
R, C = "".join(got_r), "".join(got_c)
print("8 stream-run-end:", repr(R), repr(C))
assert R == "rea", R
assert C == "tail", C

print("ALL PARSER TESTS PASS")
