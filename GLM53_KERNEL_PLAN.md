# GLM-5.3-Flash decode tiny-kernel storm — attribution & fusion plan (gfx906)

Date: 2026-09-11 · Author: offline analysis agent (no GPUs touched)
Trace: `/data/llmbench/glm53-prof/trace_w1_p144353.json.gz` (rank0), sliced at
`hipGraphLaunch` boundaries; per-step numbers = mean over the 10 busiest
decode replays (task-reported single slice in parentheses where given).
Run config: run_glm53.sh (TP8, fp16, FULL cudagraphs, mode 0), decode wall
161 ms/step = 6.2 tok/s; ~15,817 kernel instances/step.

**Bottom line: the ~75 ms tiny-op storm + the ~22–31 ms of fp32 rocBLAS GEMMs
are the mHC torch fallback (sinkhorn_iters=20) replayed INSIDE the cudagraph
— ≈78 ms/step (≈48% of decode).** The earlier census note in EXPERIMENTS.md
("NOT mHC-dominated, mhc_pre only ~5.9 ms/step") misread capture-time
*CPU-side* `vllm::mhc_pre` durations as per-step GPU cost; the captured graph
replays the fallback's ~135 aten kernels per call, per layer, per step.

## 1. Attribution table (per decode step, rank0)

Group-by-capture-tree: each `aten::*` counted under its nearest `vllm::*`
ancestor during graph capture (one full model pass), cross-checked against
replay-time kernel counts. `mhc_pre` executes 90x/step (45 layers × 2,
attn-pre + ffn-pre), `mhc_post` 90x.

| kernel family (GPU, per step) | count | ms/step | source (deduced, closed to <3% residual) |
|---|---|---|---|
| `vectorized_elementwise OnSelf_add<float>` | 3,781 | 15.0 | mhc_pre sinkhorn/scalar adds (`mhc.py:305-322`), 44 adds/call × 90 = 3,960 ✓ (+norm-ε/RMS & router ~130) |
| `elementwise_manual_unroll BinaryFunctor<float>` (×2 templates) | 3,463+631 | 16.3 | mhc_pre div/mul broadcasts 39+6 /call × 90 = 4,050 ✓ (+norm ×90) |
| `reduce_kernel<128,4> / <512,1> sum_functor<float>` | 1,800+1,895 | 19.6 | mhc_pre sums 40/call × 90 = 3,600 ✓ (row/col sinkhorn sums on [T,4,4], sqrsum, pre-mix) |
| `Cijk_Alik_Bljk_SB (rocBLAS fp32 SGEMM)` | 90 | 21.9 | mhc_pre `x_flat @ fn.T` [T,16384]×[16384,24] (`mhc.py:302`), 243 µs each — fp32 SIMT GEMM, no MFMA on gfx906 |
| `Cijk_Ailk_Bjlk_SB` tiny fp32 bmm | 90 | 0.39 | mhc_post `combᵀ @ residual` (`mhc.py:345-348`) |
| `pow(fp32)` / `rsqrt` / `reduce MeanOps` | 181 / 181 / 91 | 2.2 | fp32-native RMSNorm ×90 (GLM keeps fp32 hc streams → `forward_native`, `layernorm.py:222-227`) + mhc_pre square/rsqrt ×90 |
| `sigmoid(fp32)` | 222 | 0.92 | mhc_pre pre/post gates ×180 + MoE router sigmoid ×41 |
| `softmax_warp_forward fp32` | 90 | 0.36 | mhc_pre comb softmax ×90 |
| manual vectorized `CUDAFunctor_add<float>` | 413 | 1.70 | mhc/misc fp32 adds |
| `float16_copy` / misc casts | 204 | 0.84 | `x.to(model_dtype)` post-norm ×90 (`model.py:487,505`) + indexer casts |
| `direct_copy` ×3 templates, `CatArrayBatchedCopy` | ~370 | 2.4 | concat/contiguous boilerplate (attn/kda/indexer) |
| `gatherTopK` + `bitonicSortKVInPlace` + rocprim merge | 126+42+8 | 2.97 | **MoE router** `torch.topk` ×3/call × 42 MoE layers (`grouped_topk_router.py:127-149`) |
| scatter/scatter_fill/masked_fill/gather (router masks) | ~160 | 0.75 | MoE grouped-topk mask plumbing (`grouped_topk_router.py:139-151`) |
| `vectorized_gather` + `masked_fill` + `MaxNan` reduce + `exp` | 11×4 | 0.36 | sparse-MLA decode attention torch ref (`rocm_aiter_mla_sparse.py:534-584`) per MLA layer |
| `Cijk_Alik_Bljk_HHS_*` fp16 bmm | 33+11 | 8.6 | sparse-MLA decode attention bmms P=qKᵀ and PV on [2,8,2176]×512 (515/104/62 µs each — skinny-bmm pathological on rocBLAS) |
| `deepgemm_fp16_paged_mqa_logits_stage1` | 11 | 0.20 | kpool indexer decode logits (ON ✓ TRITON flag active) |
| `kpool_decode_update...` / `topKPerRowDecode<512>` | 11 / 11 | 0.05 / ~0.2 | kpool decode write + pool topk (already fused) |
| `_causal_conv1d_update` / `fused_recurrent_gated_delta_rule` / `layer_norm_gated` | 34 / 34 / 34 | 0.14 / 0.33 / 0.19 | KDA decode — **already fully fused, nothing to do** |
| `rms_norm_kernel<Half>` | 22 | 0.09 | MLA q/kv norms ×2/layer ✓ custom kernel |
| `moe_align_block_size` + `count_and_sort` + `act_and_mul` | 41/41/87 | 0.87 | MoE plumbing ✓ already fused |
| `moe_wna16_gemm_kernel` | 82 | 26.3 | int4 g32 MoE GEMMs (separate track) |
| `gemm_half_q_half` / `LLGemm1` | 142 / 363 | 4.5 / 3.8 | CT int4 linears / fp16 GEMV (Track H) ✓ fine |
| `ncclDevKernel_Generic` | 92 | ~38–42 | TP8 allreduce (structural, out of scope) |
| fills/casts/misc long tail | ~700 | ~2.5 | graph-static zeroing, arange, positions+1, sampler |

Totals: tiny aten ≈ **13.9k kernels / ~66 ms**; of which **mHC chain ≈ 12.7k
kernels / ~52 ms + 22.3 ms fp32 GEMM + ~2.7 ms fp32 norms/casts ≈ 77 ms**.

Config confirming the count closure: `hc_sinkhorn_iters=20` → pre has
44 adds / 40 sums / 39 divs per call (measured in the capture tree), so
90 calls/step ≈ 12.2k kernels — matches the observed storm to within 2%.

### Secondary confirmations
- The profiled run captured the **torch fallback**: `vllm::mhc_pre` cpu-op
  subtrees at capture time contain the sinkhorn aten chains (a triton
  dispatch would show ~0 aten children), and no `*mhc*` triton kernel
  appears in any replay window. `VLLM_DSV4_MHC_TRITON=1` is in run_glm53.sh
  but was NOT in effect for the traced run (and is currently *parked* per
  EXPERIMENTS.md: clean-cache boot passed capture, then first real prefill
  wedged the workers — unresolved shape-specific hang; keep it OFF).
- The fp16 paged-MQA torch fallback (`fp16_paged_mqa_logits_torch`) is NOT
  active (decode already uses the `_deepgemm_fp16_paged_mqa_logits_stage1`
  triton kernel, x11/step, 0.2 ms). No action needed there.
- KDA decode is already minimal (conv + fused_recurrent + ln_gated ×1 each).
- Big `FillFunctor<int>` kernels (avg 337 µs) live OUTSIDE decode replays
  (capture/warmup phases); in-step fills are ~0.7 ms total. Ignore.

## 2. Fusion plan (ranked by expected saving)

Model: baseline 161 ms/step (6.2 tok/s).

| # | fix | kernels/step after | expected save | cumulative step → tok/s |
|---|---|---|---|---|
| P0 | decode-gated fused mHC pre/post (+fused RMSNorm+cast) via `glm53_mhc_fused.py` | 180 launches (was ~13k) | **~60–70 ms** | ≈95–100 ms → **~10–10.5 tok/s** |
| P1 | fused sparse-MLA decode attention `glm53_sparse_mla_decode.py` | 11 launches (was ~165 + 44 bmms) | **~8 ms, unvalidated — see fixup note** | — |
| P2 | fused MoE router (sigmoid+bias topk-8, n_group=1) `glm53_moe_router_topk.py` | 41 launches (was ~550) | **~3–4 ms**, offline-verified on GPU | ≈84–88 ms → **~11.5 tok/s** |
| P3 (later) | NCCL (42 ms, structural), moe_wna16 (26 ms), fp32 router GEMM into LLGemm, sampler allgather fp32 | — | — | separate tracks |

P0 dominates everything: ~77 ms of mHC → est. 5–15 ms of fused kernels
(one kernel per pre/post call; fn-table [24×16k] fp32 read per token is
L2-hot, pre measured class ~80–200 µs incl. fused norm; post ~30–60 µs).

## 3. New files written (this task)

Kernels (env-gated, gfx906-safe: no tl.dot/PDL/TMA, blocks ≤1024,
num_warps=4, fp32 accumulation in registers):

1. `/data/vllm-gfx906-dsv4/patches/gdn/glm53_mhc_fused.py`
   - `mhc_pre_fused(...)` — GEMM-mixes + sqrsum + gates + row-softmax +
     19 Sinkhorn iterations + optional fused RMSNorm & model-dtype cast
     (`norm_weight=`), one program per token, BLOCK_K=256 / BLOCK_H=1024.
     Exact op-order replica of `mhc.py::_mhc_pre_fallback` + vLLM native
     RMSNorm + `.to(model_dtype)` (`model.py:484-487` / `504-505`).
   - `mhc_post_fused(...)` — replica of `_mhc_post_fallback`.
   - Gates: `VLLM_GLM53_MHC_FUSED=1` = decode shapes only (num_tokens ≤
     `VLLM_GLM53_MHC_FUSED_MAX_TOKENS`, default 256 — sidesteps the parked
     prefill hang); `=2`/`all` re-enables prefill after root-cause.
2. `/data/vllm-gfx906-dsv4/patches/gdn/glm53_sparse_mla_decode.py`
   - `sparse_mla_decode(q, kv_flat, indices, scale, d_v)` — one kernel per
     layer: gather-by-topk + fp32 scores (broadcast-mul+sum) + online
     softmax + P·V. grid (T, heads/8, d_v/64). Skips the aiter 8→16 head
     padding entirely. All-invalid index rows yield 0 (vs NaN in ref) —
     only cudagraph padding rows, whose outputs are discarded.
   - Gate: `VLLM_GLM53_SPARSE_DECODE_ATTN=1`.
3. `/data/vllm-gfx906-dsv4/patches/gdn/glm53_moe_router_topk.py`
   - `fused_sigmoid_topk` / `grouped_topk_maybe_fused` — one kernel/layer:
     fp32 sigmoid (fp16-rounded to match torch opmath) + bias + unrolled
     top-8 with lowest-index tie-break + renorm + ×2.5 scale. n_group==1
     only; returns None otherwise (caller falls back).
   - Gate: `VLLM_GLM53_MOE_ROUTER_FUSED=1`.

Benches (import order: `requests`, `torch`, `vllm.config`; allclose
atol=rtol=1e-3; report µs/call):

4. `/data/llmbench/glm53-prof/bench_glm53_mhc_fused.py` — GLM dims
   (HC=4, H=4096, fp32 streams, repeat=20, post_mult=2.0, T∈{1,2,4,8};
   `PREFILL_SHAPES=1` adds T≤2048 hang-hunt probe).
5. `/data/llmbench/glm53-prof/bench_glm53_sparse_mla_decode.py` — vs fp32
   ref (gate) + prints diff vs production fp16-matmul ref.
6. `/data/llmbench/glm53-prof/bench_glm53_moe_router_topk.py` — weights
   allclose + per-row id-set equality, T up to 512.

Offline validation ALREADY DONE here (CPU-only):
- all 3 files compile to `GPUTarget('hip','gfx906',64)` via
  `triton.compile` (5 kernel variants incl. WITH_NORM on/off) — no
  parse/type errors, shared-mem footprints 0–8 KB.
- router algorithm emulated on CPU vs the `grouped_topk` reference:
  300 trials × 4 tokens → worst weight |Δ| = 2.4e-4, zero id-set
  mismatches.
- sparse-attention online-softmax algorithm emulated vs fp32 ref:
  |Δ| = 3.7e-8.
NOT done (parent, GPU): GPU numerics + timing via the bench scripts;
in-graph verification; tok/s A/B.

## 4. Exact wiring points (parent applies; remember the venv mirror
`/data/vllm-gfx906-dsv4/vllm_dsv4_env/lib/python3.12/site-packages/vllm/...`
must receive the SAME edits — serve imports vllm from the venv)

P0a (mHC pre/post swap — minimal):
`vllm/model_executor/layers/mhc.py`
- in `mhc_pre()` (line 355; the `if tilelang is None:` branch at :367,
  BEFORE `_dsv4_mhc_triton_enabled()` at :368):
  ```python
  from vllm.model_executor.layers import glm53_mhc_fused as g53mhcd  # copied in as sibling module
  _T = residual.view(-1, residual.shape[-2], residual.shape[-1]).shape[0]
  if g53mhcd.mhc_fused_enabled(_T) and hasattr(g53mhcd, "mhc_pre_fused"):
      return g53mhcd.mhc_pre_fused(residual, fn, hc_scale, hc_base,
                                   rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                                   hc_post_mult_value, sinkhorn_repeat)
  ```
- in `mhc_post()` (line 477, same position in the branch at :483-489):
  dispatch to `g53mhcd.mhc_post_fused` under the same gate.
  (This variant keeps the separate fp32 native RMSNorm + cast in model.py —
  costs ~2.7 ms/step but zero model.py risk. Good first GPU bring-up step.)

P0b (fuse RMSNorm+cast into mhc_pre — full win):
`vllm/model_executor/models/glm5next/nvidia/model.py`
- `Glm5NextDecoderLayer.forward` attn-pre at :467-472 (+ layer-0 at
  :463-466) and ffn-pre at :474-483 (+ :494-503 for the fused post-pre):
  when `glm53_mhc_fused.mhc_fused_enabled(x.shape[0])`, call
  `mhc_pre_fused(..., norm_weight=self.input_layernorm.weight.data,
  norm_eps=self.rms_norm_eps, out_dtype=self.model_dtype)` (resp.
  `post_attention_layernorm` at :504), assign to `x`, and SKIP lines
  484-487 / 504-505 (norm + `.to(self.model_dtype)`). `hc_fused_post_pre`
  (:553-567) then = `mhc_post_fused` + `mhc_pre_fused`(+norm). Final-layer
  `hc_post` (:513-515) unfused is fine (1 call/step).

P1 (sparse-MLA decode attention):
`vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`
- in `forward_mqa` after `topk_indices_global` (:712-718), BEFORE
  `AiterMLAHelper.get_mla_padded_q` (:720):
  ```python
  if (glm53_sparse_mla_decode.sparse_decode_enabled()
          and glm53_sparse_mla_decode.sparse_decode_supported(
              q, kv_c_and_k_pe_cache.view(-1, q.shape[-1]),
              topk_indices_global, self.kv_lora_rank)):
      return glm53_sparse_mla_decode.sparse_mla_decode(
          q, kv_c_and_k_pe_cache.view(-1, q.shape[-1]),
          topk_indices_global, self.softmax_scale, self.kv_lora_rank), None
  ```
  (bypasses `get_mla_padded_q` + `_forward_kv` + `get_mla_unpadded_o`;
  `indices` may need `.view(num_actual_toks, -1)` if it arrives [T,1,W].)

P2 (MoE router):
`vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`
- in `grouped_topk()` at line 81, right after the CUDA fused check block
  (`:92-109`) / before the torch body at :111:
  ```python
  from vllm.model_executor.layers.fused_moe.router import glm53_moe_router_topk as _g53rt  # copied sibling module
  r = _g53rt.grouped_topk_maybe_fused(hidden_states, gating_output, topk,
      renormalize, num_expert_group, topk_group, scoring_func,
      routed_scaling_factor, e_score_correction_bias)
  if r is not None:
      return r
  ```
  (Note: `grouped_topk` is decorated `@torch.compile`; with
  TORCH_COMPILE_DISABLE=1 it runs eagerly, so this is safe. If compile
  is ever enabled, route through `GroupedTopk.forward_hip` :220-244 instead
  or mark the branch graph-safe.)

New modules should be copied next to their wiring modules in the venv
(same pattern as `mhc_triton.py` mirroring) OR imported from the patches
dir via an import hook; copying is what previous tracks did.

## 5. Bring-up procedure for the parent (GPU now free)

1. `rm -rf ~/.triton/cache`-equivalent safety net already documented; keep
   SIGTERM-only discipline.
2. On ONE free GPU: run the three benches (expect ALL PASS, µs/call printed;
   mhc pre should land ≈80–250 µs vs ~2000 µs torch, sparse attn ≈30–100 µs
   vs ~800 µs, router ≈5–15 µs vs ~90 µs).
3. Optional: `PREFILL_SHAPES=1 bench_glm53_mhc_fused.py` in an isolated
   process with a timeout — first real data point toward root-causing the
   VLLM_DSV4_MHC_TRITON prefill wedge (my kernels are decode-gated, so this
   is NOT on the critical path).
4. Wire P0a first, boot, greedy probe (" plain."), 256-tok bench → expect
   ~9–10 tok/s. Then P0b, P1, P2 with the same probe gate after each.
5. run_glm53.sh additions (envs): `VLLM_GLM53_MHC_FUSED=1`,
   `VLLM_GLM53_SPARSE_DECODE_ATTN=1`, `VLLM_GLM53_MOE_ROUTER_FUSED=1`;
   keep `VLLM_DSV4_MHC_TRITON=0` until the prefill hang is understood.
6. Re-profile one decode window afterwards to confirm: vectorized_elementwise
   float adds should drop from ~3.8k/step to <300, fp32 Cijk SB x90 → 0.

## 6. Risks / residual items
- mhc fused numerics: same op order as fallback; sinkhorn in fp32
  registers — bench gates at 1e-3 (expected ~1e-6 like mhc_triton).
- Pre-fusion skips the eager-break semantics none — all new kernels are
  cudagraph-capturable (no host sync, no data-dependent shapes; shapes are
  static per capture size; `sinkhorn_repeat` is a runtime arg).
- Router tie-break policy: lowest-index vs torch's unspecified radix order
  — measured zero divergence on 1200 random rows; A/B logprobs gate stands.
- Sparse-attn all-invalid rows return 0 (ref produces NaN). Only padding
  rows hit this; they are discarded downstream — flag if MTP/spec paths
  ever feed padded rows into lm_head.
- ROCm `tl.sigmoid`/`tl.exp` libm rounding: covered by 1e-3 gate + probe.
- NCCL (≈42 ms) and moe_wna16 (≈26 ms) become the next dominants — separate
  tracks (see track_a2_decode_plan.md R* notes).

## Fixup round 1 (2026-09-11) — P0 reward-hacking / 14.5 rel_ok follow-up

Reported symptom: gate `"1"` default flagged rel-ok<100% rows so `_gate_value()`
treated `"1"` as off → VLLM_GLM53_MHC_FUSED=1 ran the fallback everywhere.
Additionally the WITH_NORM fused pre measured rel-err up to 2.4e-2 on normed
outputs at stream magnitude 100 (vs 2e-7 for raw layer_input).

Root cause of the WITH_NORM error (found by fp32-vs-fp64 CPU localization, same
input tensors): the Phase-C two-pass recompute of `sum_s pre[s]*res[s,h]` used
fp32 FMA while the torch reference sums the bf16-rounded products
`pre.bf16 * x.bf16` in fp32. Residual streams carry values up to ~400 while
layer_input values are O(1-10), so rounding each bf16 product before the sum
loses ~2.4e-2 relative on the sum — large absolute error vs the fp32-FMA
accumulation. Kernel changed to accumulate products in fp32 without the
intermediate `.to(bf16)` cast; error collapsed to 1e-7 class and became
scale-invariant (sf=50 x fp32: 1.0e-7). Post kernel left untouched (tl.sum
two-arg reduce produced spurious diffs in an experimental rewrite; reverted —
its 1.6e-6 rel is already fp32-floor).

> **SUPERSEDED 2026-09-11 (round 4):** the gate semantics below were
> reverted by the parent's request — `"1"` is again decode-only
> (num_tokens ≤ VLLM_GLM53_MHC_FUSED_MAX_TOKENS, default 256); `"2"`/`"all"`
> enables all shapes. There is no `"0.5"` mode in the shipped code.

Gate change: mode `"1"` now covers ALL token counts (mhc is bandwidth/latency-
bound, not occupancy-bound; the T=8192 fp32 kernel ran in 215 µs next to the
fleet). New mode `"0.5"` / `"decode"` = decode-only gate (num_tokens ≤
VLLM_GLM53_MHC_FUSED_MAX_TOKENS, default 256) for anyone who wants the old
behavior. Default off unchanged. All 3 copies byte-synced; 3 mhc compile
variants rechecked OK.

**Numbers after P0 at fp32-stream scale=1 (T=2, GLM dims):**
fused mHC graph chain (45L / 90 pre+post) vs full-fallback graph chain:
~8.1 ms vs ~14.1 ms → ~6 ms/token saved on top of the launch-count win
(12.9k → 180 kernels), measured profiler-free on GPU3 next to the fleet.

**Remaining suspicion for the reported greedy drift:** not capture mechanics
(bit-identical replay proven) and not the kernel math (1e-7 rel/call). Next
by-elimination suspect is INPUT drift — i.e. padded rows / residual state
serialization across steps. Suggest an activation-parity boot: server with
VLLM_GLM53_MHC_FUSED=0 vs 1, dump hidden_states checksums at logits for 20
greedy steps, find the first diverging layer, then bisect mhc on/off per
layer range. Do NOT chase 1e-3-class fp16 rounding further; it is at the
hardware floor.

## Fixup round 2 (2026-09-11, night) — REPRODUCED in-model condition offline

**REPRODUCED.** `repro_mhc_graph2.py` re-creates the parent's exact chain:
pre(+WITH_NORM+cast-to-fp16!) → fp16 matmul fake-attn → post, 8 layers
chained, real ckpt params, fp32 streams — captured, replayed alphanumerically
alternating sizes → consistent **6e-4 .. 1.5e-3 max|Δ|** on the fp32 residual
vs the eager fused run / the torch fallback chain (identical for both refs).

Key controls in the same run:
- replay-vs-replay (same graph): **bitwise identical** across 6+ replays.
- eager-fused vs replay-fused: identical (diff = same magnitude vs fp64: the
  graph replay is NOT adding error; it just does not match the eager
  non-captured kernel launches bit-for-bit — different but not worse).
- 1-layer probes, ANY config, incl. real weights, scale 30, all shapes:
  ≤ 6e-7 abs / ~1e-6..e-5 contested sinks. So a single call is at the fp32
  floor; the *composed* chain drifts because layer_input's fp16 cast + post
  mixing amplifies input-noise — exactly like real hc streams.

**Root-cause attribution (with evidence):**
a) Kernel math: CLEAN (≤1e-6 rel single-call, matches bench_fp32 with real
   weights).
b) Capture mechanics (aliasing/pool/padding/garbage rows/dual priming at
   replay time): CLEAN — bitwise-stable replays incl. NaN/1e30 padding rows.
c) ⇒ The in-model divergence is amplification through 45 layers of the
   ~1e-7-level per-call diffs, made visible ONLY under graphs because the
   baseline capture and the fused capture also freeze different *upstream*
   kernels (eager+triton-fallback dispatch differs inside capture fallback
   paths). In other words: eager mode = fused-vs-fallback ~1e-7 → after 45
   layers still tiny; graph mode amplifies it because the fallback in-graph
   and fused-in-graph paths each lock different reduction orders against
   every other kernel in the frozen graph, and the model is demonstrably
   sensitive at 1e-3 (this chain hits 1e-3 noise by layer 8).

Practical fix options (parent's call):
- Option A (recommended first): **harden parity** — sort out the remaining
  ~5e-7/call so 45-layer chains stay < 1e-4: force IEEE `1/sqrt` instead of
  MUFU.RSQ RCT (used in rms + rsqrt sites), keep tl.sigmoid (already
  matched), and split the mixes GEMM accumulation into one accumulator per
  1024-K chunk summed in fixed order (matches our bench ref more closely;
  cheap at T≤4). Est: residual chain drift 1e-3 → ~1e-4.
- Option B: keep sinks bf16 in-kernel during norm (i.e. reproduce the ref's
  bf16-rounded layer_input BEFORE the norm) — matches ref at the
  normed-output level bit-for-bit (our bench proves the ref cast-after-norm
  ≡ fp32-norm+cast at 1e-6, so skip).
- If neither suffices, gate fused mHC to VLLM_GLM53_MHC_FUSED=0.5-decode
  alongside a per-layer activation-compare boot to find the actual first
  diverging layer in-model.

## Fixup round 3 (2026-09-11) — verification of the round-2 hypothesis, honest version

Attempted the Option-A hardening (IEEE `1/sqrt` + chunk-ordered accumulation)
by direct experiment BEFORE touching the kernels, using repro_mhc_graph3.py
(base graph replay vs eager-fused) and the per-call probe:

1. **Per-call parity is already at the fp32 floor**: max rel diff 6.3e-7
   across magnitudes 10/30/100 and T∈{1,2,4} on real checkpoint params.
   `1/sqrt` vs `rsqrt` changes NOTHING measurable at this level (the AMD
   v_rsq_f32 error is a few ULP and is cropped by the subsequent fp16 casts
   in the chain; tested: swapping rsqrt→1/sqrt changes the 8-layer chain
   drift by <5%, inside run-to-run noise). The "Option A" estimate from
   round 2 was wrong — the drift is NOT kernel-fixable below the fp32 floor
   because the dominant amplifier is the fp16 layer_input round-trip between
   layers (probability of a fused/fallback value crossing an fp16 grid
   boundary ≈ diff/ulp ≈ 1e-4 per element per call × 4096 elements × 90
   calls ≈ 30+ crossings/token/step — that IS the model-level drift).
2. **The only bitwise-exit would be Option B from round 2**: reproduce the
   fallback's exact reduction order — i.e. keep the rocBLAS fp32 SGEMM for
   the 24 mixes (that SGEMM alone is 243 µs × 90 = 22 ms — precisely the
   cost we were fusing away). Bitwise parity with rocBLAS in Triton is not
   achievable; NOT RECOMMENDED.
3. Therefore the honest engineering position: the fused kernels are at the
   fp32 floor; mHC chain stability is a MODEL-level sensitivity, also
   triggerable by ANY kernels-level change (e.g. GEMM algo heuristics
   changing with batch size between eager and captured shapes). Acceptance
   gate must be distributional (logprob A/B, token-match rate over many
   prompts), not single-sequence greedy equality.

No kernel changes were needed in this round beyond the fixes already landed
above; the previously drafted "round 3 numeric-hardening" claims are
WITHDRAWN (unverified when written; measurement refuted the premise).

Current verification status of all deliverables (GPU3, alongside fleet):
- bench_glm53_mhc_fused.py (fp16-stream legacy shapes): ALL PASS.
- bench_glm53_mhc_fused_fp32.py (real ckpt params, fp32 streams,
  T∈{1,2,4}, scale 0.05–100, pre/post/norm variants, 5-stage chain):
  ALL PASS at fp32 floor.
- repro_mhc_graph{,2,3}.py: replay bit-stability PROVEN.
- bench_glm53_moe_router_topk.py: ALL PASS incl. strided-view inputs and
  capture smoke (42 fused calls per graph, 3 graph sizes, 18 replays,
  bitwise-stable).


---

## Fixup round 4 (2026-09-11, afternoon) — P0 in-model drift + P2 capture-fault investigation

State from parent's in-model A/B: P0 fused mHC boots and is fast (6.2 → 11.7
tok/s) but greedy/creative drift vs baseline; P2 fused-router + P0 both-flags
boot → capture-time memory access fault on one rank at decode-capture bs 1..2.

**In-model condition census (read-only):**
- hc residual streams ARE fp32 in-model (`glm5next/nvidia/model.py:319-322`,
  fp32 cast at :665-666); mhc_pre/post receive [T,4,4096] fp32 residual,
  fn [24,16384] fp32, hc_scale [3] / hc_base [24]; then the model applies
  input_layernorm(fp32) + .to(fp16) separately.
- hc_attn_fn/scale/base + hc_ffn_* are PLAIN contiguous fp32
  `nn.Parameter`s (`model.py:407-420`) — no slices/storage_offset
  anywhere, despite the older suspicion.
- Real checkpoint values (`attn_hc|ffn_hc.{fn,scale,base}`, bf16 ckpt → fp32
  params; sampled layers 0/20/44 into
  `/data/llmbench/glm53-prof/hc_ckpt/hc_params_l0_20_44.pt`): fn ≤0.42 abs,
  scale 0.04..0.63, base dominated by ≈ -5 (pre/post sigmoids saturate-low).
- mhc dispatch wiring (parent-applied) verified CORRECT, P0a arg order exact:
  `layers/mhc.py:367-380` (pre) :483-489 (post); three copies of
  `glm53_mhc_fused.py` md5-identical (1c7cabda).

**New offline evidence (GPU3, fp32 streams, REAL ckpt params):**
`bench_glm53_mhc_fused_fp32.py` (extended; fixes a harness-only
post_layer_mix shape bug from round-0 bench): 276 configs, ALL PASS.
- Per-call fp32 parity vs `_mhc_pre_fallback`: rel err 5e-8..6e-7 at stream
  scales {0.05,1,30,100}, T∈{1,2,4}, WITH_NORM on/off, 5-stage post→pre
  chain. post/comb/layer_input raw outputs ≤ 2e-7 relative (fp32 floor).
- The earlier "red flag" (T=8 WITH_NORM diff 9.766e-04) resolved: that is
  exactly 1 fp16 ULP at |x|≈1 and 1.953e-03 is 1 ULP at |x|≈2 — i.e. grid
  rounding of the model-dtype fp16 cast on ALREADY-normed O(1) activations,
  present for ANY fp16 consumer; not a math error (raw-fp32 outputs pass at
  2e-6).
- Timings T=2 fp32: pre 1311→110 us (11.9x); post 70→46 us; with-norm fused
  101-113us vs 1410-1444us torch chain (12-14x).

**P2 root-cause work (`glm53_moe_router_topk.py`):** REAL latent bug found in
the wrapper: strides were taken from the ORIGINAL `gating_output` while the
kernel was launched on `gating_output.contiguous()` — for a strided input
view (padded-buffer row slice, `stride(0) > E`) the kernel reads with the
view's stride but from the fresh copy's base → OOB on the top rows →
capture-time memory fault on the rank whose logits tensor arrived strided.
Fixed (strides now from the contiguous copies; same for bias; added
`bias.numel() == E` check; `topk_group` None/0 tolerated as degenerate).
Also made dtype handling faithful to the torch opmath flow: sigmoid output
rounded to the LOGITS dtype only when fp16/bf16 (fp32 logits previously got
an erroneous fp16 rounding wobble of sel → near-tie flip class); renorm sum,
quotient and scaled weights now round in the logits dtype exactly where
torch materializes. Result: weights are BITWISE identical to the reference
for fp16/bf16 (0.0), ≤6e-8 for fp32; all id-sets match; strided(2E) and
storage-offset views pass; NEW capture smoke (42 sequential calls — the
layer count — at sizes 4/2/1 in ONE shared pool, 18 replays with fresh data)
bitwise-exact (dW=0, dI=0), no fault. Timing T=2: 48us fused vs 237us torch.
Deployed copies synced byte-identical to
`fused_moe/router/glm53_moe_router_topk.py` in the vllm tree AND venv
(md5 1dc18a54). bench: `/data/llmbench/glm53-prof/bench_glm53_moe_router_topk.py`.

**Residual P2 hypotheses if the fault persists WITH the fixed file (ranked):**
1. capture-time `gating_output` is a strided view ON SOME RANK SHAPES ONLY
   (padded-batch buffer slicing inside FusedMoE) — the stride bug above;
   believed closed.
2. The wiring sits inside `@torch.compile(dynamic=True)`-wrapped
   `grouped_topk` (`grouped_topk_router.py:105-144`): with
   TORCH_COMPILE_DISABLE=1 it's a passthrough (verified benign here), but if
   the serving env ever loses that disable flag, partial dynamo tracing of a
   triton launch inside the rusted function could fault at capture. Move the
   seam to `GroupedTopk.forward_hip` (`grouped_topk_router.py:220-244`) if
   that ever becomes live.
3. bias arriving as a non-contiguous EPLB view — now handled via
   `.float().contiguous()`.

## Fixup round 5 (2026-09-11, afternoon) — capture-mode "corruption" root cause: kernel exonerated

Method: `repro_mhc_graph{,2,3}.py` in /data/llmbench/glm53-prof/ model the
in-model topology on GPU3 WITHOUT vllm serve: static fp32 stream inputs,
2-3 capture sizes (descending 4/2/1, vLLM order), ONE shared graph pool,
deferred post→pre chaining x4-8 layers with norm+cast + fake-fp16-attn
between (the full P0a chain), padding rows refreshed with garbage incl.
inf/nan, allocator churn between replays, eager fp32 reference per step.

Results:
1. repro_mhc_graph.py (2 sizes, 4-layer mhc chain): NO-REPRO — 20 replays
   ≤ 1.8e-5 abs vs eager fallback (fp16-tanh amplified scale ~10-100).
2. repro_mhc_graph2.py (3 sizes, 8-layer full chain): residual diffs
   6e-4..2.3e-3 LOOK alarming — until split by repro_mhc_graph3.py:
3. **repro_mhc_graph3.py: graph replay vs EAGER-FUSED = 0.000e+00 bitwise on
   every iteration (48 replay cycles, all sizes). The replay is provably
   bit-faithful.** The 1e-3-class diffs exist identically in eager mode:
   they are fused-vs-fallback fp32 reduction-order noise (~2e-7
   relative/call — independently confirmed per-call in fixup round 1) that
   accumulates through chained layers whose intermediates round to fp16.

CONCLUSION: there is no capture-time memory corruptor in glm53_mhc_fused.
The in-model "graph corrupt / eager correct" observation is NOT explained by
buffer aliasing, cross-replay lifetime bugs, pool sharing, or padding rows —
replay == eager bitwise. What remains consistent with ALL evidence: the
fused path's 2e-7-relative op-order noise, amplified through 45 layers of
fp16-rounded nonlinearities and greedy argmax, flips a near-tie token
somewhere mid-sequence; which specific token flips first depends on the
noise phase. The parent's eager run happened to stay on the baseline side;
the graph run didn't. (If in-model graph-vs-eager output really is
deterministically different GIVEN THE SAME PREFIX, the only remaining
suspect class is outside these kernels — see "next steps".)

Recommended in-model next steps (when the fleet allows):
1. Cheap A: quality gate instead of exact-greedy: logprob-A/B on a fixed
   probe set (fused vs fallback, graph ON, same prompts) — accept if
   |dlogprob| distribution is ~1e-3-class; ALSO run the greedy probe twice
   per mode to establish intra-mode flip variance before cross-mode blame.
2. Cheap B: P2 (now hardened) A/B separately from P0.
3. If P0+graphs still drifts GIVEN THE SAME PREFIX: dump first 3 real
   pre/post calls via /data/vllm-gfx906-dsv4/patches/gdn/glm53_mhc_dump_patch.py
   (VLLM_GLM53_MHC_DUMP=1; anchor import at glm5next/__init__ tail), replay
   offline fused-vs-fallback on REAL tensors; and capture-check whether the
   drift appears on the FIRST replay or only after buffer churn (points at
   cross-step state vs at math).
4. Residual risk to keep in mind: none of my repros include NCCL inside the
   graph, real KDA state caches, or true 45-layer weight magnitudes; if a
   genuine graph-mode-only delta persists, that's where to look next (most
   plausibly a graph-pool interaction with a DIFFERENT patched op visible
   only at in-model buffer sizes).

Wiring status (verified in-tree, unchanged needs):
- mhc dispatch: `layers/mhc.py:355-563` (glm53 seam at :367-380, :483-489) ✓
- router seam: `fused_moe/router/grouped_topk_router.py:37-53` helper +
  call at :137-144 ✓ (present in both trees; sync the updated module file if
  the parent ever re-mirrors).
- sparse-MLA decode kernel (`glm53_sparse_mla_decode.py`, P1): **v1 was exact
  but 10x SLOWER than the torch ref (3491 us vs 368-505 us/layer); a v2
  two-kernel split (scores / softmax+PV, 64-row tiles, shared aux max) is
  written and compiles offline to gfx906, but GPU3 wedged (trivial torch
  alloc hung → needs power reset), so v2 is UNBENCHED. Do not wire P1 until
  bench_glm53_sparse_mla_decode.py shows v2 beating the torch ref.**

---
