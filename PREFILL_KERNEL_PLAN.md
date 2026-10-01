# PREFILL_KERNEL_PLAN — kill the sparse-gather wall

Date: 2026-09-30. Target: sparse-MLA prefill, 150-355 tok/s today -> 600-1200 tok/s.

## Measured wall (from py-spy + arithmetic)

- Prefill path runs `reference_mla_sparse_prefill` (torch) because
  `VLLM_ROCM_MLA_SPARSE_FP16=1` forces it ("triton slower + unstable" — old verdict).
- Per 64-token chunk it materializes gathered_kv [cs=64, topk=2048, d_qk=576] fp16
  = 151 MB written then read twice (P matmul + V matmul).
- Per 2048-token step: 32 chunks x ~450 MB traffic x 11 sparse layers ~ 158 GB of
  RANDOM-row gathers per step. Achieved ~30 GB/s vs 900 GB/s HBM peak (3%).
- Conclusion: wall is random gather ACCESS, not GEMM FLOPs. py-spy confirms CPU
  parks at sync points; GPU 100% busy on gather+small GEMMs.

## Existing asset

`_mla_sparse_vec_kernel` in rocm_aiter_mla_sparse.py is already a fused
gather-attention kernel (online softmax over gathered KV, no materialize) but
structured ONE PROGRAM PER TOKEN with tiny tl.dots (grid: tokens x heads x dv-chunks)
= 1.5 TFLOPS. Correct idea, wrong granularity. The "HSA_STATUS_ERROR_OUT_OF_RESOURCES"
note also suggests per-program memory pressure at large mnb.

## Exploit: overlapping retrieval in contiguous prefill

Adjacent query tokens' topk index sets overlap heavily (they read neighboring
pools of the same request). Today each token gathers its 2048 rows independently
= 64x redundant reads of the same rows within a tile.

## Plan (ranked)

### Step 1 — topk 2048 -> 1024 (fast win)
Halve gather volume. Env knob `VLLM_GLM53_INDEX_TOPK` if configurable, else edit
indexer config. Validate: needle suite (42k/60k/100k) + logprob A/B vs baseline.
Expected: ~1.6-1.8x prefill. Risk: retrieval recall at depth — needles are the test.

### Step 2 — union-gather kernel (the big one)
New triton kernel per query tile of 64 tokens:
  a) collect the tile's indices [64, 2048] -> sort/unique -> unique rows U (~2048-8000)
  b) gather each unique row ONCE into a tile-local buffer (or process in U-tiles)
  c) map each token's topk positions to the unique-buffer slots (small int map)
  d) fused online-softmax attention against the unique buffer
Traffic: ~|U| rows instead of 64*2048 rows = 2-10x less random access.
Extra wins: unique buffer fits cache better; V reuse same as K.
Complications: sort/unique on GPU per tile (cub/rocprim radix sort, cheap at 128k ids);
tile-local buffer sizing (|U| x 576 x 2B ~ 9 MB worst case -> chunk U into K-blocks).

### Step 3 — re-tile the fused kernel
While in there: fix granularity of _mla_sparse_vec_kernel (BLOCK_M over tokens
sharing the unique buffer, not per-token programs). Gets GEMM shapes to
[64,576]x[576,TILE] class = 8+ TFLOPS territory.

## Benchmark harness
tools/prefill_bench.py: standalone single-GPU (like kda_fp16_micro2.py) — build
q/kv/indices with REALISTIC overlapping indices (contiguous-pool model), compare:
  A) reference_mla_sparse_prefill (current)
  B) _mla_sparse_vec_kernel (existing triton)
  C) new union kernel
Report tok/s + GB/s + numeric diff vs A. Then in-server A/B via bench_write prefill
numbers + needle suite.

## Non-goals / guardrails
- No change to decode path (s_q==1 route) — already fast.
- Keep VLLM_ROCM_MLA_SPARSE_FP16=1 fallback until C passes needles.
- Edit BOTH trees (venv + /data/vllm-gfx906-dsv4/vllm), py_compile, .bak-prefill.

## MEASURED: real prefill index overlap (2026-09-30) — union kernel VALIDATED
Dumped real topk_indices from a live 60k-token prefill (1068-token step, 2176
indices/row incl. kpool padding):
- WHOLE STEP: 89.9x redundancy (25,848 unique rows of 2.32M entries)
- PER-64-TILE: mean 7.2x (min 5.2x, max 7.6x)
Conclusion: adjacent-token retrieval overlap is REAL (between synthetic 27x
and worst-case 1x). Union-GEMM+scatter at 64-token tiles wins:
- v3 test at real shapes (T=1068, TOPK=2176, S_KV=200k): 1.45x vs reference
  core, rel 0.0003. Intermediate size ~39MB/tile = fine.
- End-to-end expectation: sparse core ~20% of prefill (Amdahl from 318 tok/s
  baseline) -> union alone ~+8%, union+topk1024 combined core ~2.5x -> ~+20%.
- The bigger prefill wins live in KDA chunk_gla tuning (34/45 layers).

## UNION VERDICT (2026-09-30): REJECTED in-engine (quality gate)
- Standalone replay at real shapes: 1.45x core, rel 1e-4. Micro-bench PASS.
- In-engine with VLLM_GLM53_PREFILL_UNION=1: needle FAILS (model echoes
  filler = broken retrieval at depth), twice, including after the
  int32->int64 scatter_add_ fix. Union OFF: needle PASS.
- Mystery for the capture/replay harness session: likely call-contract
  subtlety (strided q view, or topk_indices not being identity-mapped into
  the flattened kv view). NOTE: needle_probe prompts prefix-cache-hit ->
  timings of repeat runs are ~7s, NOT real prefill time. Use fresh prompts
  (vary the filler) for any timing A/B!
- Code stays in-tree (env-gated, default off). Production unaffected.

## CAPTURE/REPLAY HARNESS (2026-09-30) — permanent deliverable

Kernel-dev loop without booting vLLM per iteration:

1. CAPTURE (in-engine, once): instrument `ROCMAiterMLASparseImpl._forward_kv`
   is already in both trees (`_glm53_kc_dump`, marker GLM53-KC-DUMP). Launch
   with a clean reference config (VLLM_GLM53_PREFILL_UNION unset) plus
   `export VLLM_GLM53_KC_DUMP=/data/tmp/kc_dump.npz`. The first call with
   num_tokens > 256 lands in that file, the first call with num_tokens <= 4
   (decode) in `<path>.decode.npz`. Dumps are once-per-file (O_EXCL),
   TP-rank-safe, and skipped during CUDA-graph capture. The npz holds the
   REAL q (logical layout preserved), the full kv cache, global int32
   topk_indices, scale, engine reference output (`out_ref`) plus its unpadded
   form, and metadata (num_heads_impl, q_stride, block_size, topk_tokens,
   req_id_per_token, block_table, dtypes).
2. REPLAY (offline, CPU is fine for correctness):
   `python3 tools/engine_contract_replay.py /data/tmp/kc_dump.npz [--bench N]
   [--device cuda] [--gate 1e-3] [--candidates union,union_flat]`
   Runs a verbatim port of reference_mla_sparse_prefill (checked against
   out_ref) and each candidate; prints time + rel error; PASS gate 1e-3.
   Captured dtypes are kept as-is (int32 indices!) so deploy-blocking bugs
   surface in replay, not in production.

Example session (this run): kc_dump.npz q(2048,16,512) fp16 stride contiguous,
kv 412160x512, idx(2048,2176) int32 (52.5% = -1 padding), scale 0.0625.
REPLAY: reference vs engine out_ref rel 5.7e-4 (fp16 GEMM noise; decode
contract matched bitwise rel=0.0), union rel 8.6e-4 PASS, union_flat same.

NOTE: the plan's earlier "d_qk=576" is DeepSeek-dim folklore — real GLM-5.3
contract is d_qk=d_v=512 (qk_rope_head_dim=0), scale=256^-0.5=0.0625.

## ROOT CAUSE (2026-09-30): union kernel was correct — the _forward_kv wrapper
## broke the head-pad contract

THE BUG (code): `patch_prefill_union.py` routed the union path with an
early `return union_gather_prefill(...)` from `_forward_kv`, SKIPPING the
common tail `return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, output)`
that the reference path goes through.

WHY IT MATTERS (call contract): `forward_mqa` calls
`AiterMLAHelper.get_mla_padded_q(self.num_heads, q)` before `_forward_kv`.
When num_heads < 16 (_AITER_MIN_MLA_HEADS), q is REPEAT-INTERLEAVED to 16
heads. GLM-5.3 has num_attention_heads=64, TP=8 -> 8 heads/rank -> padded
factor 2 (padded head j == real head j//2). Every _forward_kv return path must
therefore un-pad with `output[:, ::2, :]`.

WHAT THE EARLY RETURN DID: _forward_kv handed [s_q, 16, d_v] to `_v_up_proj`,
which does `x.view(-1, self.num_heads=8, d_v)` — silently reinterpreting the
tensor as [2*s_q, 8, d_v] (token t head h folds into token 2t + h//8, head
h%8) — then `torch.bmm(x, W_UV, out=out)` auto-resized the output buffer
(union_e2e2.log: UserWarning "An output ... was resized ... shape [8, 3, 256]
... required [8, 6, 256]", and [8,12,256] -> [8,24,256] — exactly the 2x
fold). The sparse-MLA layer outputs are scrambled/stale, no crash.

WHY THE SYMPTOM WAS "fluent but can't retrieve": the 11 sparse-MLA layers'
prefill outputs (and the KV written downstream of them) are garbage, while
the residual stream + 34 KDA layers still carry language structure — so
generation continues in the prompt's style (echoing filler) and needle
retrieval at depth dies. Decode via CUDA graphs was unaffected (graph capture
bakes the reference path because the union branch is disabled while
capturing), which is why generation ran at all.

WHY STANDALONE TESTS PASSED: micro-tests never padded heads (h_q ==
num_heads in the test builders), so the unpad was a no-op there and the
wrapper bug was invisible. Kernel MATH was never wrong — replay on the real
captured contract passes at rel 8.6e-4.

SUSPECTS CLEARED by the capture: (a) q is contiguous at this call site
(q_stride=(8192,512,1) from torch.cat of (ql_nope, q_pe)); (b) topk_indices
ARE identity-mapped into kv.view(-1,1,d) row space
(triton_convert_req_index_to_global_index: block_table[req, tok//BS]*BS +
tok%BS with kv shape (blocks, block_size, head_size)); (c) no token-count
mismatch (num_tokens == q.shape[0] == indices rows); (d) call sites all go
through forward_mqa's single _forward_kv call.

EVIDENCE CHAIN:
- code: early-return vs the reference path's fall-through (see .bak-union)
- union_e2e2.log: torch.bmm out-resize warnings, 2x shape fold, all 8 workers
- capture: q_shape=[2048,16,512] with num_heads_impl=8 (pad factor 2 proven);
  out_ref (2048,16,512) vs out_ref_unpadded (2048,8,512)
- engine_contract_replay.py prints WRAPPER-WARNING on this contract

FIX (applied, marker GLM53-UNION-UNPAD-FIX): the union call assigns
`output = union_gather_prefill(...)` and falls through to get_mla_unpadded_o
exactly like the reference (env-gated, default off; patch_prefill_union.py
updated to install the fixed routing). Verified in-engine:
- reference baseline (union OFF): needle 20000 0.5 -> NEEDLE-PROBE: PASS
- union ON + fix, cold 85k-char needle (fresh filler variant 1, 61682 prompt
  tokens, depth 0.5): ANSWER ' 73912' -> NEEDLE-PROBE: PASS (212s)

## KDA chunk_gla tuning (2026-09-30)

### Call path (seq_len > 1, one KDA layer)
Glm5NextLinearAttention.forward -> _forward (num_prefills > 0,
glm5next/nvidia/kda.py:633) -> chunk_kda_with_fused_gate
(glm5next/amd/ops/third_party/kda/kernels.py):
1. l2norm_fwd(q), l2norm_fwd(k)   fla/ops/l2norm.py, l2norm_fwd_kernel2 (MBLOCK=32)
2. fused_kda_gate_chunk_cumsum    kda_gate_cumsum_fwd_kernel (autotune BD 32/64 x w 2/4/8)
3. _chunk_kda_fwd_with_cumulative_g:
   a. chunk_kda_scaled_dot_kkt_fwd  intra_sub_inter (autotune BK 32/64 x w 1/2/4/8
      x s 2/3/4) + intra_sub_intra (autotune w 1/2/4/8); BC=min(16,BT) host-fixed
   b. solve_tril                    merge_16x16_to_{32,64}_inverse (asserts BT in 16/32/64)
   c. recompute_w_u_fwd             recompute_w_u_fwd_kernel (BK=BV=64 host-fixed;
      autotune w 2/4/8 x s 2/3/4)
   d. chunk_gated_delta_rule_fwd_h  chunk_gated_delta_rule_fwd_kernel_h_blockdim64
      (K-blocks 64 fixed in body; autotune BV 32/64 x w 2/4 x s 2/3/4)
   e. chunk_gla_fwd_o_gk            chunk_gla_fwd_kernel_o (autotune BK 32/64 x
      BV 64/128 x w 2/4/8 x s 2/3/4, key=["BT"])
Chunk size FLA_CHUNK_SIZE=64 (fla/ops/utils.py:31) threads through all stages as
chunk_size=. Env knobs in this stack: FLA_COMPILER_MODE, FLA_CI_ENV,
GDN_RECOMPUTE_SUPPRESS_LEVEL, FLA_USE_CUDA_GRAPH, FLA_USE_TMA, FLA_USE_FAST_OPS,
FLA_TRIL_PRECISION, USE_DEFAULT_FLA_NORM. No chunk-size knob before this work.

### Harness
tools/kda_prefill_bench.py - standalone single-GPU bench of the exact chain at
engine shapes (hidden 4096 -> 64 heads / TP8 = H=8 per rank, head_dim K=V=128 from
recurrent state [8,128,128], prefill step T=2048 = --max-num-batched-tokens on the
live server, q/k/v bf16, beta [1,T,H] pre-sigmoided fp32, recurrent state [N,H,V,K]
fp32, safe-gate on). Ground truth = default-config chain output, gate
rel = max|out-ref|/max|ref| < 1e-2. Modes: --profile, --sweep K1,K2, --sweep-chunks,
--preset-defaults, --bt. Logs: tools/kda_prefill_bench_T2048.log,
tools/kda_bench_r2{a..e}.log.

### Stage profile (T=2048, H=8, default configs)
BT=64 (stock): delta_h 3.33ms (48%), kkt 0.82 (12%), recompute_wu 0.74 (11%),
gla_o 0.77 (11%), solve_tril 0.32, gate_cumsum 0.29, l2norm 0.25
-> chain 5.83 ms/layer (351k tok/s/layer), 34 layers = 0.198 s/step.
BT=16: delta_h 1.59ms (41%), gla_o 0.55 (14%), kkt 0.49 (12%), recompute_wu 0.30
-> chain 2.79 ms/layer. delta_h grid is (V/BV, N*H) = (4, 8) programs at BT=64 -
grossly occupancy-starved on a 60-CU MI50 (each program loops NT chunks serially).
Same disease as the "1.5 vs 8 TFLOPS" note in rocm_aiter_mla_sparse.py: smaller
tiles / more parallel programs win big on gfx906.

### Sweeps (full-chain ms/layer; all rel gates PASS at nseq=1)
Per-kernel autotune configs at BT=64 - BLOCK 32/64/128 x warps 2/4/8 x stages 1/2/3
over gla_o, delta_h, kkt_inter, recompute_wu, kkt_intra, gate_cumsum, tril64
(159 configs): ALL 0.98-1.01x vs default. Stock triton autotune already picks
near-optimal configs at BT=64; the win is NOT in those knobs.

Chunk size (the real win):
  T     nseq  BT=64     BT=16     speedup  rel     BT=32    speedup
  2048  1     5.83 ms   2.90 ms   2.01x    5.9e-3  3.24 ms  1.80x
  4096  1     11.08 ms  5.22 ms   2.12x    2.5e-3  6.08 ms  1.82x
  512   1     2.66 ms   1.29 ms   2.07x    3.3e-3  1.35 ms  1.98x
  2048  4     5.56 ms   2.32 ms   2.40x    2.9e-3  2.85 ms  1.95x
BT=8 blocked: solve_tril asserts A.shape[-1] in {16,32,64}.
Config sweep at BT=16 (extended grids incl. BV=16, warps 1/16): delta_h
BV=16 w=2 s=1 (+16%), gla_o BK=32 BV=128 w=2 s=2 (+9%); combined pins ->
2.09 ms/layer = 2.79x vs stock chain, rel 5.9e-3 (same fp-reassociation delta as
the pure BT change; recurrent-state rel ~4e-7).

### Wiring (env-gated, default bit-identical)
kernels.py both trees (venv site-packages + /data/vllm-gfx906-dsv4/vllm):
chunk_size = _apply_kda_prefill_tuning() at the two prefill call sites
(chunk_kda_fwd, chunk_kda_with_fused_gate_fwd):
  VLLM_KDA_PREFILL_TUNING=1   -> chunk BT=16 + pinned configs (2.79x measured)
  VLLM_KDA_PREFILL_TUNING=32  -> chunk BT=32, stock configs (~1.8-1.95x)
  unset / 0                   -> stock FLA_CHUNK_SIZE=64 + stock autotune spaces
Backups *.bak-kdatune; py_compile OK both trees. Side win: mode 1 pins each
autotuner to 1 config, killing the ~100-variant compile/bench storm on first prefill.

### Reality check against the 600-1000 tok/s target
Measured: chunk chain = 5.8 ms/layer x 34 = 0.198 s per 2048-token step = ~10.4k
tok/s-equivalent at STOCK settings. At the 318 tok/s system baseline a 2048-token
step takes ~6.4 s, so the chunk_gla kernels are only ~3% of it. Even the full 2.79x
chain win moves end-to-end prefill by only ~2% (~+127 ms/step). The premise
"the KDA chunk path is where the big wins" is NOT supported by these measurements:
the missing 5-6 s/step live elsewhere. Most likely candidates (not covered by this
harness): the KDA projection GEMMs (in_proj_qkvbfg_a / f_b_proj / g_b_proj / o_proj
plus the 3xhidden conv), the sparse layers beyond their core matmul, or engine-level
CPU/scheduling overheads. NEXT STEP for the prefill target: profile one real 60k
prefill step (per-op CUDA timing or torch profiler) before spending more effort on
kernel configs; the projection GEMMs are the natural suspect (skinny GEMMs at
T=2048 on gfx906).

### Validation status
- Micro gate PASS: rel <= 5.9e-3 vs default output at all tested shapes (512/2048/
  4096 tokens, nseq 1 and 4), recurrent-state rel <= 3.6e-6, chain bitwise
  deterministic across repeated calls.
- In-server boot + needle A/B: NOT RUN - a full 8xGPU vLLM instance was live at
  wrap-up (ops rules: do not disturb). Wiring ships OFF by default.
  Coordinator checklist: with VLLM_KDA_PREFILL_TUNING=1 vs unset, run the needle
  suite + fresh-prompt prefill timing (vary the filler: needle_probe prefix-caches
  repeat runs, see union note above), and confirm state-cache behavior over a long
  60k prefill. BT=16 changes fp summation order (outputs rel ~6e-3, state ~4e-7 at
  T=2048) - needle quality is the real gate before prod enablement.
- Open anomaly: in the nseq=4 run the BT=64 baseline differed from its own
  reference by rel 3.6e-2 (expected 0). An independent 3-run probe later proved the
  chain bitwise repeatable at nseq=4, so this is an unexplained harness/measurement
  artifact, not an isolated kernel race. Re-check with:
  ./vllm_dsv4_env/bin/python tools/kda_prefill_bench.py --preset-defaults
  --T 2048 --nseq 4 --sweep-chunks  (GPUs were busy at wrap-up).
- historical failing test re-run verbatim (`needle_probe.py 9700 85000 0.5`,
  original filler, cold on the fresh boot, 60462 prompt tokens): ANSWER
  '  73912' -> NEEDLE-PROBE: PASS (211s). UNION VERDICT OVERTURNED: the
  union prefill path is quality-safe once the unpad contract is honored;
  it stays env-gated (VLLM_GLM53_PREFILL_UNION=1, default off).

REPRO COMMANDS:
  capture boot:  bash /data/tmp/run_kcdump.sh 9700   # KC_DUMP set, union off
  replay:        python3 tools/engine_contract_replay.py /data/tmp/kc_dump.npz
  fixed boot:    bash /data/tmp/run_unionfix.sh 9700  # + VLLM_GLM53_PREFILL_UNION=1
  quality gate:  python3 tools/needle_probe.py 9700 85000 0.5 [maxtok] [variant]
                 (pass a fresh variant to defeat block-level prefix caching)

## INCIDENT 2026-09-30 night: "the !!!! corruption"
- Symptom: all large-prompt (60k+) completions returned degenerate
  "!!!!!!!!!!!!!!!!" while short prompts stayed fluent. Hit EVERY config
  including no-flag baseline. Three isolation rounds (union-only, control,
  .bak restores) failed to clear it.
- INTERIM THEORY (SUPERSEDED -- see the postmortem section below): the
  morning-code restore PASS was read as implicating the perf patches. The
  exact-prompt repro later exonerated all code; root cause was GPU/driver
  state from the 19:25 hung-fence era.
- RULE ADDED: every perf patch gets a needle gate on the OFF path before
  its ON path is ever tested. "Inert by default" is a claim, not a fact.
- STATE (at the time, superseded): production = morning code. All perf
  work (union, KDA tuning, topk knob) quarantined in git history for
  careful one-at-a-time revival.

## Corruption incident postmortem (2026-09-30 evening) — MACHINE STATE, NOT CODE

Symptom: long prompts (68-85k tokens) returned degenerate `!!!!!!!!!!!!!!!!`
(16 bangs = max_tokens wall); short prompts stayed fluent. First seen during
Union+KDA validation ~20:37.

Timeline / evidence:
- 19:25 — dmesg: 4x `dma_fence_wait_timeout` hung-task stacks (GPU hung),
  410 `amdgpu queue evicted` events in 19:00-22:00. GPU trouble PREDATES the
  first corrupted output.
- 20:37-21:42 — all needles FAIL on several boots (union ON, union OFF,
  restored-backup trees). Cold prefills, real token counts (68-71k).
- 23:39 — git-restored morning code: needle PASS @ 61k.
- 23:45+ — morning+union-patch: PASS @ 62k, 58k, 55k, 50k.
- 00:2x — EXACT repro of the failing prompts on healthy code: variant 6
  (prompt_tokens=70672 — identical to the FAIL) PASS; variant 7
  (prompt_tokens=68171 — identical to the FAIL) PASS. Same triton cache,
  same prompts, same sizes.

Code audit (all suspects env-inert at default, verified byte-level):
- patch_prefill_union.py OFF path == stock reference call + unpad fall-through.
- ee7cde4d1c (KC-DUMP + unpad fix): capture is env-gated (VLLM_GLM53_KC_DUMP).
- 435bc54596 (KDA tuning): _apply_kda_prefill_tuning() returns stock
  FLA_CHUNK_SIZE when VLLM_KDA_PREFILL_TUNING unset.
- agent-5 never edited fla/ops/chunk_delta_h.py or solve_tril.py (the
  .bak-kdatune copies of those two were precautionary backups; its patch
  script touched only kda/kernels.py).
Cache audit:
- Triton cache 521MB/5841 files; the 113 "partial" entries are benign
  (launcher .so / autotune.json only). 716 .hsaco with normal size spread.
  The PASS runs used the SAME cache as the FAILs -> cache not the cause.
- venv *.pyc wiped after the incident (this box has prior corrupt-pyc history).

Conclusion: outputs were corrupted by GPU/driver state left from the 19:25
hung-fence era (queue teardown churn). Cleared by process teardown/reboot.
Not reproducible. All perf patches exonerated.

Lessons:
1. Before bisecting code on a needle failure, check dmesg for hung-task /
   queue-eviction in the failure window. Machine-state corruption masquerades
   as a code regression and "restores" can look like fixes.
2. Needle-fail evidence must be re-proed on a fresh boot before blaming code;
   identical prompt+token-count repro is the gold standard.
3. Boot scripts must gate on fleet_free.sh with `&&` (a `;` boot can race a
   dying server) and carry the segfault retry loop (1-2-3-4-5).

### Addendum: the wedge outlived the incident (00:30-00:55)
The degraded GPU state did NOT clear with process teardown: it served the
green repro runs (23:39-00:30), then killed two fresh boots (segfault, then
workers dying after NCCL init) with `amdgpu ... Trying to push to a killed
entity` in dmesg and rocm-smi/ps HANGING. Symptom triage for this box:
  - boots dying after "distributed_init"/NCCL + killed-entity dmesg = GPU
    wedge, not code -- do NOT bisect code, ipmitool chassis power reset.
  - after ANY power reset: wipe venv *.pyc (mandatory), then boot with the
    segfault retry loop.

## KDA prefill tuning: QUALITY VERDICT (2026-10-01)
VLLM_KDA_PREFILL_TUNING=1 (BT=16 + pinned configs) is BROKEN in-engine:
needle probes return degenerate "!!!!!!!!!!!!!!!!" at ALL sizes (17k, 35k,
70k -- deterministic, 4/4 FAIL). The offline bench (T=512..4096, rel<=5.9e-3)
did not catch this. Do NOT enable mode 1. Suspect: the pinned Config list
replaces the autotuner candidates and leaves other constexprs at defaults
that are invalid for the engine call shapes (or BV=16 on
chunk_gated_delta_rule_fwd_kernel_h_blockdim64 violates a blockdim64
assumption). Mode 32 (BT=32, stock configs) isolation test pending.
OFF path remains verified-safe (Gate 1 PASS).
Mode 32 (BT=32, stock configs) ALSO FAILS (2/2, 68k + 19.6k) -> the chunk
size change itself is unsafe in-engine (likely fp16 recurrent-state /
fused-gate path semantics vs the standalone bench). Only stock BT=64 is
safe. All VLLM_KDA_PREFILL_TUNING != 0 modes remain disabled.
