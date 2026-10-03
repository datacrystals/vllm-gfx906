#!/usr/bin/env python3
"""H4: render the chat template for a 3+ turn conversation (CPU-only) and
eyeball role structure, then dump the exact prompt token ids.
Also H2 prep: run the same ids through fast DecodeStream incremental decode
vs full decode and report any divergence (detokenizer sanity, CPU-only).
"""
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from transformers import AutoTokenizer

MODEL_DIR = "/data/ModelDownloader/MiMo-V2.6-Flash-INT4"
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

msgs = [
    {"role": "system", "content": "You are a helpful assistant. Track state carefully."},
    {"role": "user", "content": "I have three apples."},
    {"role": "assistant", "content": "Understood: three apples."},
    {"role": "user", "content": "I eat one. How many are left?"},
]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("=== RENDERED CHAT TEMPLATE (4 msgs / 3 turns) ===")
print(repr(prompt))
print(prompt)
ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
print(f"=== TOKEN IDS len={len(ids)} ===")
print(ids[:80])
print("...")
print(ids[-20:])

# H2 sanity: incremental DecodeStream over a period-heavy synthetic stream
raw = tok._tokenizer
test_ids = tok("The quick brown. fox jumps over. the lazy dog. And more. Text here.",
               add_special_tokens=False)["input_ids"]
print("\n=== DETOK SANITY on synthetic ids ===")
print("ids:", test_ids)
print("pieces:", [tok.decode([i]) for i in test_ids])
print("full :", repr(tok.decode(test_ids)))
st = DecodeStream(ids=[], skip_special_tokens=False)
inc = "".join(st.step(raw, i) or "" for i in test_ids)
print("inc  :", repr(inc))
print("match:", inc == tok.decode(test_ids))

# with a long prompt primed (fast-detok path as used by the server)
st2 = DecodeStream(ids=ids, skip_special_tokens=True)
gen_ids = tok(" This is the answer. It has periods. Inside.", add_special_tokens=False)["input_ids"]
inc2 = "".join(st2.step(raw, i) or "" for i in gen_ids)
full2 = tok.decode(ids + gen_ids, skip_special_tokens=True)
print("\nprimed-stream == full-decode:", inc2 == full2)
if inc2 != full2:
    print("inc2 :", repr(inc2[-60:]))
    print("full2:", repr(full2[-60:]))
