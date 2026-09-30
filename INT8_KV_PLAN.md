# INT8 MLA Latent KV Cache — Design & Wiring Plan (GLM-5.3-Flash / gfx906)

Status: **kernels implemented + CPU self-tested green (21/21)**. GPU triton
variants written but not yet launched (GPU busy: `vllm` PID holds `/dev/kfd`;
re-run `python3 tools/int8_kv_kernel.py --gpu` on `HIP_VISIBLE_DEVICES=7` when
free). Production wiring is designed below but deliberately NOT applied yet —
this doc + `tools/int8_kv_kernel.py` are the green steps being committed.

Goal: halve the dominant MLA-latent term of the 256-token block cache
(measured 2.92 MiB/block/rank, post-B2 ~2.94 MiB/block) -> ~1.7 MiB/block ->
**~1.75x KV pool -> 512k context becomes multi-user capable** (or ~896k
single-stream in the same pool).

Hard boundary (from prior analysis, respected throughout):
> Quantize ONLY the MLA latent rows. The indexer **kpool keys stay fp16** —
> retrieval is discrete-risk: a quantized key can change WHICH blocks get
> retrieved (quality cliff). Values/latents quantize fine. The fp8 torch path
> was abandoned for exactly this boundary ("layout-incompatible with the fp16
> kpool indexer cache"); int8 keeps the fp16 kpool layout untouched.

---

## 1. Tensor layout

### 1.1 Row format (one token, one sparse layer)

The MLA cache row is `d_qk = 576` = `kv_lora_rank(512)` latent +
`qk_rope_head_dim(64)` rope. Write order is `[kv_c | k_pe]` (matches the C++
`concat_and_cache_mla_kernel`, `csrc/cache_kernels.cu:376`).

```
byte    0 .. 575   int8 data   (dims [0:512) latent | [512:576) rope)
byte  576 .. 579   fp32 scale  group 0
byte  580 .. 583   fp32 scale  group 1 (zero-pad when G=1)
=> ROW = align8(576 + 4*G) = 584 bytes for G in {1,2}
```

Quantization: **symmetric int8, zero-point 0**

```
s[g]      = amax(|x[group g]|) / 127        (1.0 for an all-zero group)
q[d]      = clamp(round(x[d] / s[group(d)]), -127, 127)   torch.int8
x_hat[d]  = float(q[d]) * s[group(d)]
```

Scale granularity options (all supported by `tools/int8_kv_kernel.py`):

| G | group_sizes | ROW B/token | % of fp16 (1152 B) | notes |
|---|-------------|-------------|--------------------|-------|
| 1 | (576,)      | 584         | 50.7%              | simplest; one amax covers latent+rope |
| 2 | (512, 64)   | 584         | 50.7%              | **recommended default** — natural latent/rope boundary; rope dynamics don't dilute the latent amax |
| 9 | (64,)*9     | 616         | 53.5%              | quality knob (measured best: 7.8e-3 vs 1.0e-2 rel) |

fp32 scales are deliberate: 4 B of a 584 B row is 0.7%, and fp32 avoids a
second quantization error source in the dequant multiply.

### 1.2 Where the scales live: INLINE in the token row (not a side pool)

The 8 scale bytes sit at offset 576 **inside the same 584-byte row**, i.e.
inside the same page allocation as the data. Rationale:

1. **Block-granular ops stay correct for free.** `swap_blocks`,
   `copy_blocks`, prefix-cache block reuse, block tables, and the KV
   connector treat a block as opaque bytes. With inline scales the scales
   travel with the data automatically. A separate scale pool must be
   threaded through every one of those paths — that is exactly the "passes
   micro-tests, breaks production" failure class (cf. the int32-scatter burn).
2. **In-tree precedent**: `fp8_ds_mla` already uses "448B NoPE + 128B RoPE +
   8B scale = 584B per token" (`vllm/v1/kv_cache_interface.py:333-341`), and
   `TritonAttentionImpl._ensure_scale_caches`
   (`vllm/v1/attention/backends/triton_attn.py:434-484`) carves strided fp32
   scale views out of inline head padding. `views_from_packed()` in the tools
   file is the same trick (careful: `set_()` storage_offset is in *float32*
   elements, so the 576-byte offset must be divided by 4 — footgun noted in
   code).
3. Gather/index ops can pull the whole 584 B row and split it in registers —
   dequant-on-gather needs data+scale of the same row anyway.

View helper (the only sanctioned way to split a cache tensor):

```python
data, scale = views_from_packed(raw)   # raw [..., 584] int8 contiguous
# data  -> [..., 576] int8   (strided row slice)
# scale -> [..., G]   float32 (strided storage view at byte 576)
```

### 1.3 Cache spec / allocator semantics

- Storage dtype: `torch.int8`; storage width per token = **584 elements**
  (bytes). Semantic `head_size` stays 576 (the attention dims).
- `MLAAttentionSpec.real_page_size_bytes`
  (`vllm/v1/kv_cache_interface.py:327-341`) gains a third branch keyed on
  `cache_dtype_str == "int8_mla"`:
  `return self.storage_block_size * 584` — exactly the shape of the existing
  `fp8_ds_mla` branch (which hardcodes 584/656 per token). Semantic head_size
  vs storage width divergence is already the established pattern there
  ("head_size stays semantic (512); bytes are determined here").
- `get_kv_cache_shape` (`rocm_aiter_mla_sparse.py:123-130`) returns
  `(num_blocks, block_size, 584)` in int8 mode; the raw page then views
  contiguously (no `page_size_padded` striding needed since page bytes are
  exactly `block_size * 584`).
- Memory estimator, startup memory check (B2 fixed-cost accounting) and
  `get_num_blocks` all flow from `page_size_bytes`, so they pick up the new
  size automatically once the spec branch exists.

### 1.4 Memory math (per 256-token block per rank, 11 sparse layers)

| | bytes/token/layer | 256-token block, 11 layers | + 0.24 MiB kpool tail |
|---|---|---|---|
| fp16 today | 576 x 2 = 1152 | 3,244,032 (3.09 MiB) | ~2.94 MiB measured block |
| int8 G=2 | 576 + 8 = 584 | 1,644,544 (1.57 MiB) | **~1.7 MiB/block** |
| int8 G=9 | 576 + 36 = 616 | 1,736,704 (1.66 MiB) | ~1.78 MiB/block |

=> **~1.73-1.75x more blocks** at G<=2. 512k single-user becomes ~1.75x
concurrent capacity at 512k, or one ~896k stream, in the same pool. The
kpool tail (~0.24 MiB) and the KDA state (own pool post-B2) are unchanged.

---

## 2. Consumers — read-first findings and int8 treatment

File refs are `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`
("backends") and `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` ("ops")
unless another path is given. (Note: both files share a basename; line
numbers below disambiguate.)

### 2.1 PREFILL — `reference_mla_sparse_prefill` (backends:678-728)

torch gather + 2 matmuls per chunk (`VLLM_ROCM_MLA_SPARSE_CHUNK_SIZE=512`):

- gather: `kv.index_select(dim=0, index=idx_chunk.flatten())` (backends:707)
  on the flattened `[rows, 576]` cache view
- scores `q @ K^T` (:710-712), masked softmax, `P @ V[..., :512]` (:721-723)

Int8: **dequant-on-gather** — gather the int8 rows + their scales, multiply
(`gathered_i8.to(kv.dtype) * scale`), then the matmuls are byte-identical to
today. Implemented as `int8_mla_sparse_prefill()` (tools). The env-gated
union variant `union_gather_prefill` (backends:624-676, gather at :664)
gets the same treatment at its single `index_select`.

Perf follow-up (not needed for correctness): fold the row scale into the
score output (`S_ij *= s_j`) and into `P` (`p_ij *= s_j`) so the PV term uses
raw int8 rows — that lets a future int8 GEMM skip materializing the fp16
gather entirely. The reference path materializes fp16 gather (1.5x transient
peak vs today for one chunk: int8 gather + fp16 dequant copy) — acceptable
because prefill is chunked at 512 rows.

### 2.2 DECODE

Two families, and the boundary matters:

**(a) The sparse-attention decode over the LATENT cache — this is the int8
consumer.** Sparse impls run decode-style MQA for every token through
`forward_mqa` -> `_forward_kv` (backends:789-847): the fp16 path
(`VLLM_ROCM_MLA_SPARSE_FP16=1`) routes single-token decode through the same
`reference_mla_sparse_prefill` (dispatch `mla_sparse`, backends:731-756,
`s_q == 1` at :746-748) and the triton path through
`triton_mla_sparse_vec` / `_mla_sparse_vec_kernel` (backends:405-535) when
`VLLM_ROCM_MLA_SPARSE_FP16_TRITON=1`.

- `_mla_sparse_vec_kernel` loads the cache twice per tile: K at
  backends:456-459 (`kv_pos[None,:]*stride + d_offs[:,None]`) and V at
  backends:478-481 — both fp16-hardwired via `tl.load` on the fp16 pointer.
- Int8: **dequant-load variant** `tl.load(int8) -> .to(tl.float32) * scale`
  then the same `tl.dot`. Implemented as `_int8_sparse_vec_kernel` +
  `int8_sparse_vec_decode()` (tools). The V side reads only dims [0:512),
  which is group 0 for both G=1 and the (512,64) split, so the value-side
  scale index is compile-time constant. The K side picks its group per
  constexpr `d_start` (unrolled loop), so no runtime branching.

**(b) The paged-MQA logits family scores the INDEXER kpool — STAYS FP16.**
`deepgemm_fp16_paged_mqa_logits_stage1` (ops:225-438), `fp16_mqa_logits_*`
(ops:~950-1060) and their kpool callers
(`rocm_aiter_mla_sparse_kpool.py:280,411`) read the **indexer key pool**
(retrieval scoring -> top-k block selection). Per the hard boundary these
keys are never quantized. The int8 dequant-load variant of the paged-logits
kernel (`_int8_paged_mqa_logits_kernel`, tools) is provided anyway as the
ready pattern for the day a value-side or non-retrieval cache wants int8 —
it is self-tested but **not wired** to the indexer.

### 2.3 CACHE WRITE — `do_kv_cache_update` -> `concat_and_cache_mla`

- Call site: `vllm/model_executor/layers/attention/mla_attention.py:555-563`
  passes `kv_c_normed [T,512]`, `k_pe [T,64]` every forward step.
- `SparseMLAAttentionImpl.do_kv_cache_update`
  (`vllm/v1/attention/backend.py:990-1010`; also the base variant at :910)
  calls `ops.concat_and_cache_mla`
  (`vllm/_custom_ops.py:2754-2763`) ->
  `csrc/cache_kernels.cu:376` `concat_and_cache_mla_kernel`.
- C++ semantics to preserve exactly: row content `[kv_c | k_pe]` in that
  order; `slot_mapping < 0` is a **no-op** (padding), including CUDA-graph
  padded tokens (`slot_mapping.size(0) <= kv.size(0)`).
- Int8: `int8_concat_and_cache_mla()` (tools) — quantize the concatenated
  576-row per token (per group), pack int8+scale into the 584 B row, scatter
  to `flat[slot]` with **int64** indices (the int32 scatter burn: force the
  cast, never trust the incoming dtype).
- Fused variant `concat_and_cache_mla_rope_fused`
  (`csrc/cache_kernels_fused.cu:26`): if that path is ever enabled, quantize
  **after** rope is applied — rope before quantization changes the amax
  structure and would break the scale layout invariant.

---

## 3. Env-gated wiring plan (minimal shared-file edits)

New env knobs (read via `os.environ` in the touched sites to avoid a broad
`vllm/envs.py` refactor; a later cleanup can register them properly):

```
VLLM_GLM53_INT8_KV=1          # master switch (default off = today's fp16)
VLLM_GLM53_INT8_KV_GROUPS=2   # 1 | 2 | 9   (default 2 = (512,64) split)
```

Every edit below: `.bak-int8kv` backup first, `py_compile` after, and the
edit is inside `if int8_kv_enabled():` so the default path is bit-identical
to today.

1. **`vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`** (the one
   shared file this project already owns changes in):
   - `ROCMAiterMLASparseImpl.do_kv_cache_update` — **override** the inherited
     method (new method on the impl class, ~30 lines) so `backend.py` stays
     untouched: if int8 enabled, call `int8_concat_and_cache_mla(kv_c_normed,
     k_pe, kv_cache, slot_mapping, GROUPS)`; else `super()`.
   - `get_kv_cache_shape` (:123-130): return `(num_blocks, block_size, 584)`
     when int8.
   - `_forward_kv` (:789-847): split the cache once per call via
     `views_from_packed`, then route the fp16 branch to
     `int8_mla_sparse_prefill` / `int8_sparse_vec_decode` (triton if
     `VLLM_ROCM_MLA_SPARSE_FP16_TRITON`).
   - `union_gather_prefill` (:624-676): int8 branch dequanting at its
     `index_select` (:664).
2. **`vllm/v1/kv_cache_interface.py`** (one branch, mirrors fp8_ds_mla):
   `MLAAttentionSpec.real_page_size_bytes` += `cache_dtype_str == "int8_mla"`
   -> `storage_block_size * 584`.
3. **`tools/int8_kv_kernel.py`** (already written; move/copy into the tree as
   `vllm/v1/attention/ops/int8_kv_kernel.py` when wiring) — no vllm imports
   today, so it stays importable standalone.
4. NOT touched: `backend.py` (override avoids it), `csrc/*` (torch/triton
   write kernel instead of a CUDA rebuild), the whole indexer/kpool family
   (`indexer.py`, `rocm_aiter_mla_sparse_kpool.py`, ops MQA-logits), the
   KDA pool, and the DSV4 C128A path.

Wiring acceptance gates before flipping default on:
- `python3 tools/int8_kv_kernel.py` green (done: 21/21 CPU).
- `python3 tools/int8_kv_kernel.py --gpu` green on a free GPU (pending).
- Server boot with `VLLM_GLM53_INT8_KV=1` on 8xMI50: block count in the
  startup banner rises by ~1.7x; needle suite + logprob A/B per section 4.
- A/B with the knob off must reproduce fp16 outputs bit-for-bit (the env
  gate makes this trivially checkable by running the same dump twice).

---

## 4. Quality validation protocol

1. **Micro gate (implemented)**: int8-vs-fp16 relative error on the attention
   output < 2e-2 (typical int8-KV level). Measured on CPU:
   prefill 1.00e-2 (G=1) / 9.79e-3 (G=2) / 7.75e-3 (G=9); decode 9.94e-3 /
   9.90e-3; quant round-trip 7.6e-3; paged-logits 7.0e-3. The harness
   includes a **layout-sensitivity negative test** (rolled scales must blow
   the gate: measured 3.14e-1) so a wrong scale layout cannot pass silently.
2. **Needle suite**: `tools/needle_probe.py` / `tools/concurrent_needle.py`
   at 128k and 256k, fp16 vs int8: exact-match rate and retrieval-position
   accuracy must be >= fp16 baseline (no degradation beyond run-to-run
   variance; n>=5 seeds). Multi-user concurrency needles specifically
   validate the flagship claim.
3. **Logprob A/B**: `VLLM_GLM53_KC_DUMP` capture
   (`tools/engine_contract_replay.py`) of real prefill+decode traffic, then
   replay fp16 vs int8 on identical inputs: report mean/percentile
   |delta logprob| and top-1 token agreement (target: >99% top-1 agreement
   on teacher-forced traffic, mean |dlogprob| < 0.05).
4. **Long-form generation sanity**: teacher-forced perplexity on long docs
   (< 1% relative degradation) + spot-check long-context chats for
   repetition/derailment regressions.
5. If G=2 fails any gate, retry G=9 (616 B/row, still 1.66x) before
   abandoning.

---

## 5. Self-test results (CPU, 2026-09-30)

```
torch=2.9.0+rocm6.3 triton=yes cuda=False     (CUDA_VISIBLE_DEVICES="" ;
                                              /dev/kfd busy with vllm PID 36285)
[PASS] pack/unpack bit-exact gs=(576,)        ROW=584 G=1
[PASS] views_from_packed gs=(576,)
[PASS] pack/unpack bit-exact gs=(512, 64)     ROW=584 G=2
[PASS] views_from_packed gs=(512, 64)
[PASS] pack/unpack bit-exact gs=(64,)*9       ROW=616 G=9
[PASS] views_from_packed gs=(64,)*9
[PASS] quant/dequant rel gs=(576,)            rel=7.558e-03
[PASS] quant/dequant rel gs=(512, 64)         rel=7.277e-03
[PASS] zero-row scale=1.0 q=0
[PASS] write/read exact gs=(576,)             T=32 rows=128
[PASS] slot=-1 no-write gs=(576,)
[PASS] write/read exact gs=(512, 64)
[PASS] slot=-1 no-write gs=(512, 64)
[PASS] prefill int8-vs-fp16 rel gs=(576,)     rel=1.000e-02
[PASS] prefill int8-vs-fp16 rel gs=(512, 64)  rel=9.790e-03
[PASS] prefill int8-vs-fp16 rel gs=(64,)*9    rel=7.753e-03
[PASS] decode int8-vs-fp16 rel gs=(576,)      rel=9.944e-03
[PASS] decode int8-vs-fp16 rel gs=(512, 64)   rel=9.895e-03
[PASS] paged-logits int8-vs-fp32 rel          rel=7.028e-03
[PASS] paged-logits causal -inf mask
[PASS] layout-sensitivity (rolled scales)     rel=3.141e-01 (must fail loudly)
== 21 passed, 0 failed ==
```

Dtype discipline exercised in the tests: write path fed **int32** slot_mapping
on purpose (must be forced to int64 internally); all scatter/gather indices
asserted int64; scales asserted float32; data asserted torch.int8; the
`views_from_packed` f32 storage offset is divided by 4 (byte->float units).

---

## 6. Risks / follow-ups

- **Triton variants untested on silicon** (GPUs busy). `_int8_sparse_vec_kernel`
  and `_int8_paged_mqa_logits_kernel` compile-shape follows the fp16 originals
  (`num_stages=1`, `num_warps=4` MI50 stability settings) but the `--gpu`
  self-test must run before wiring the triton decode path. The torch paths are
  the correctness reference and are green.
- **Transient prefill peak**: the reference dequant-on-gather briefly holds
  int8 gather + fp16 copy (1.5x one fp16 chunk). Fine at chunk=512; the
  fold-scale / int8-GEMM optimization in section 2.1 removes it if prefill
  memory pressure shows up in the 256k soak.
- **spec-decode / CUDA-graph padding**: relies on `slot_mapping < 0` no-op
  (kept, and tested). If `concat_and_cache_mla_rope_fused` is ever enabled,
  re-verify quantize-after-rope.
- **`supported_kv_cache_dtypes`** (backends:92-96) currently lists
  auto/float16/bfloat16 — the int8 mode rides the env knob, not
  `--kv-cache-dtype`; exposing `int8_mla` as a first-class cache dtype is the
  cleanup once the knob earns its keep.
- DeepseekV4 `deepseek_v4_attention.py:968,1164` also calls
  `reference_mla_sparse_prefill`; out of scope here (GLM-5.3 first), but the
  same dequant-on-gather applies when DSV4 wants int8.
