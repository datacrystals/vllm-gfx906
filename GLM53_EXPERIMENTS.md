# DeepSeek V4-Flash gfx906 bring-up — experiment log (continues README.md)

## 2026-09-09 — agent-1: BOS NaN fixed + memory-fault investigation

### Bug #1: token-0/BOS → NaN (FIXED, validated)

Root cause: **fp16 range overflow of the hyper-connection (hc) residual stream**.
The bf16-native model's hc residual streams legitimately reach abs(max) ≈ 71,100
by layer 42 (measured via NANDBG probes), which overflows fp16 (max 65504). Once a
stream element hits ±inf, `mhc_pre`'s sigmoid/softmax mixes turn it into NaN at the
next layer — previously observed at `layers.40.attn.hc_pre`. Only token id 0 (BOS)
produces this trajectory; raw completions without BOS stay under the fp16 cliff.

Offline inspection absolved alternatives: checkpoint embed row 0 is clean
(absmax 0.63, 2680 rows bigger), tid2eid row 0 valid, hc_* params all sane, fp32.

Fix: run the hc residual streams in **fp32** (each decoder layer casts to fp16 only
AFTER attn_norm/ffn_norm, for the w4a16/moe GEMMs that require fp16). fp32 is strictly
more precise than the bf16 reference; all residual consumers (mhc_pre/mhc_post/hc_head)
already compute internally in fp32. Files changed (venv site-packages AND ./vllm source):
- `model_executor/layers/mhc.py`: `_mhc_pre_fallback` assert now allows fp32.
- `model_executor/models/deepseek_v4.py`: embedding repeated hc stream `.float()`;
  DecoderLayer stores `self.model_dtype` and casts post-norm activations to fp16
  before attn/ffn; final norm output cast back to model dtype; `_mtp_hidden_buffer`
  copy clamped to fp16 range (MTP path is NOT fp32-safe — don't enable MTP spec-decode).
Validation (exp1, server on :9400): prompt `[0,882,1990]` → finite text (was: NaN);
raw completion regression OK ("…capital of the United States is" coherent);
NANDBG trajectory: hc stream max climbs to 71k, **0 inf, 0 NaN**.

### Bug #2: intermittent GPU memory access fault (all 8 ranks, same step) — UNDER INVESTIGATION

Observed (deterministic per-run, variable across runs):
- exp1/exp2: creative chat (31-tok prompt, temp 0.8) → fault at num_computed=383
  (num_out=352), twice identically.
- exp3 (probes off, no clamp): 301-token completion PREFILL → fault.
- exp4 (HIP_LAUNCH_BLOCKING=1): 193-token prefill → fault; 129-token prefill OK.
- exp5 (DSV4_PROG per-stage sync markers): p192 OK, then p300 prefill faulted —
  last markers: `layers.3.attn.compressor saved_states` (C128A compressor, ratio 128)
  → fault inside/next-to `_fused_kv_compress_norm_rope_insert_fp16` launch.
- exp6 (DSV4_FLUSH_DBG verifier + `block_table.clamp(0, nb-1)` host-side): FLUSHDBG
  reported all state block numbers IN RANGE (bn 46..61, nb=4229, badstate=0); p150
  OK; (p300/d500 invalid client-side: mangled request files — void); creative chat
  STILL crashed at num_computed=127 (num_out=96).

Conclusions so far: fault timing/masking is sensitive to host-side syncs (markers
mask smaller-prefill crashes). FLUSHDBG says the C128A flush kernel's state-cache
block numbers are in-range at runtime; either the fault is in a sibling kernel
launched between markers (flush fp16 kernel launch only op in between — but its
other inputs: kv slot mapping / write), or a cross-STREAM race: compressor +
_rocm_qnorm_rope_kv_insert run on an AUX stream concurrently with the indexer on
the default stream (`maybe_execute_in_parallel`).

Next planned experiment (exp7, written but GPUs currently busy — another agent's
Hy3 vllm on :9500 since 12:52): `DSV4_SEQ=1` disables the aux-stream overlap
(sequential kv_insert → compressor → indexer) and re-runs p300, d500 (cross
384/512), creative×3, multi-turn, and the 2000-token perf pass in one load.

Backups: `*.bak-dsv4fix` next to patched files. Debug gates: `DSV4_NANDDBG=1`
(per-layer NaN/max), `DSV4_PROG=1` (per-stage sync markers), `DSV4_FLUSH_DBG=1`
(C128A flush host verifier + table clamp), `DSV4_SEQ=1` (no aux-stream overlap).
All default OFF; remove before production.

NOTE: 2026-09-09 12:52 — an external Hy3-GPTQ vllm serve (pid 2743445, port 9500,
util 0.90, NOT a systemd service) occupied all GPUs; exp7 pending until it clears.

### 2026-09-09 Track B GPU result 1 — Hy3 + MTP k=2: NEGATIVE (do not use)
Config: run_hy3_mtp_test.sh (port 9500) = run_hy3.sh + --speculative-config mtp k=2.
Server started fine (~6.5 min load), MTP engaged: mean acceptance length 1.5-1.8.
Measured decode: 9.95 tok/s (500-token gen, short ctx) vs baseline 11.43 -> LOSES.
Why: kernel-latency-bound; MTP step costs ~2 forwards (~158ms/step vs 87ms baseline)
and acceptance 1.5-1.8 can't pay for it. Log: /data/tmp/hy3mtp-serve.log.
(k=3 not tried: more verify overhead per step cannot help in this regime.)

### 2026-09-09 Track B GPU result 2 — tuning wash
async-scheduling + max-num-seqs 1 + max-num-batched-tokens 8192 (port 9501,
run_hy3_tune_test.sh): 11.53 tok/s vs 11.43 baseline -> +0.9%, no win. 
CONCLUSION Track B: Hy3 speed is at its effective ceiling on this rig
(~11.4-11.5 tok/s single-stream); MTP is net-negative. If Hy3 returns to
production, use plain run_hy3.sh.

### 2026-09-09 Track C GPU result — GDN smoke PASSES on gfx906 (bf16)
patches/gdn/smoke_kernels.py (TP8-shard dims, bf16, 192 tokens): cumsum,
chunk_prefill (out+state), packed_decode (out+state) all match the fp32
reference (max_abs <= 3.3e-3). Individual triton GDN kernels are NOT broken
on gfx906. Implication: July Nex NaN likely comes from the fp16-force-downcast
in fork fla/ops/chunk.py:203-211 (fp32 inputs downcast to fp16 state) and/or
decode-path accumulation, not the kernels. (Script hangs at process exit after
printing results — probably hip teardown; harmless, SIGTERM it.)

### 2026-09-09 Track C - fp16 smoke ALSO PASSES (GPU3)
smoke_kernels.py --dtype float16: all stages OK (cumsum 5.7e-6, chunk_* <=4.4e-4,
packed_decode <=2.2e-4). Kernels survive the exact NaN-trigger dtype in
isolation. Root cause narrows toward scale-dependent state overflow
(fp16 downcast in chunk.py:203-211 at real magnitudes) or decode-path bug -
full Nex serve with VLLM_GDN_GFX906_FALLBACK=1 is the honest test.

## 2026-09-09 (late) — agent-1: bug #2 ROOT-CAUSED + FIXED, full validation pass

### Bug #2 root cause: raw vs compressed slot space in compressed-cache flush

The intermittent all-rank "Memory access fault. Page not present" crashes were
caused by the fp16 compressor flush kernel writing the **compressed** attention
KV cache (C4A, C128A) using `k_cache_metadata.slot_mapping` from
`ROCMAiterMLASparseMetadata`, which on this ROCm path carried **raw-token** slots.
The compressed cache is addressed in compressed-UNIT space (unit = position //
compress_ratio; 2 units/block for C128A, 64 for C4A at block_size 256). Evidence:
FLUSHDBG dumps showed `kvslot=32639` while the cache's valid slot space was only
8494 → block 16319 of 4247 → wild pointer → fault on every TP rank at the same
step. Earlier runs either stayed in-range-but-wrong (silent corruption, e.g.
`kvslot=383`) or faulted at the first flush touching a large slot — explaining
the apparent randomness (allocator-state dependent), the differing crash lengths
(prefill 193/301; decode 383; chat 127), and why host-side syncs (DSV4_PROG
markers) sometimes masked it.

The indexer-cache flush was NEVER affected because `DeepseekV32IndexerMetadataBuilder`
already builds a **compressed** slot mapping via `get_compressed_slot_mapping`
(`v1/attention/backends/mla/compressor_utils.py`). The sparse groups didn't.

### Fix

`v1/attention/backends/mla/rocm_aiter_mla_sparse.py` (venv + ./vllm source):
in `ROCMAiterMLASparseMetadataBuilder.build()`, when `is_deepseek_v4 and
compress_ratio > 1`, replace `slot_mapping` with
`get_compressed_slot_mapping(num_tokens, query_start_loc, seq_lens, block_table_tensor,
kv_cache_spec.storage_block_size, compress_ratio)` before constructing
`ROCMAiterMLASparseMetadata` — mirroring the indexer builder. Flush kernel and
all read paths unchanged (compressed slot space now consistent end-to-end).

Note: the same latent issue exists in the upstream FP8 flush kernels if a raw
slot_mapping is supplied — not exercised on gfx906 (fp16 path).

### Validation matrix (all PASS, zero memory faults, engine stable)

| Test | Result |
|---|---|
| exp8 p300 prefill (301 tok) | PASS (was: crash) |
| exp8 d500 decode (61+500 tok, crosses 384/512 ctx) | PASS (was: crash) |
| exp8 creative chat ×3, BOS template, 480 tok each | PASS, coherent long-form prose |
| exp8 multi-turn + prefix caching (811→824 tok prompts) | PASS ("banana"/"papaya", 36% cache hit) |
| exp8 perf: 2000-token decode | PASS, **2.13 tok/s** (936.9s) |
| exp8 FLUSHDBG: 16 flush points to pos 1919 | 0 out-of-range, 0 badstate |
| exp9 production smoke (NO debug envs, probes removed): tok0 completion, chat, 350-tok list | PASS, coherent, 0 faults |

Server load ~5 min (46 shards), sweep total ~38 min in exp8.

### Perf verdict

Decode ~**2.1 tok/s** (TP8, fp16, prefix caching; no cudagraphs on this ROCm
sparse path; decode uses the pure-PyTorch `reference_mla_sparse_prefill` gather +
fp32 mhc fallback). Functionally CORRECT end-to-end incl. BOS/chat, but an order
of magnitude below m27 (~19.5 tok/s) → **not production-worthy as a replacement**.
Optimization headroom: cudagraph support for the ROCm sparse decode path, fused
decode gather kernel, fp16 mhc fallback. Untouched: MTP speculative decoding is
NOT safe with the fp32 hc patch (mtp buffer clamped to fp16 range).

### Cleanup / production state

- NANDBG + PROG debug scaffolding REMOVED from `deepseek_v4.py`,
  `deepseek_v4_attention.py`. Remaining (env-gated, default OFF, kept for future
  debug): `DSV4_FLUSH_DBG=1` C128A flush verifier + state-table clamp
  (`deepseek_compressor.py`), `DSV4_SEQ=1` aux-stream overlap disable
  (`deepseek_v4_attention.py`).
- All patches mirrored into `/data/vllm-gfx906-dsv4/vllm` source; backups
  `*.bak-dsv4fix` beside each patched file.
- Final run script: `/data/vllm-gfx906-dsv4/run_dsv4.sh` (PORT/HOST env-overridable,
  default 9200 for the future swap; not registered in systemd).
- Chat template: `/data/vllm-gfx906-dsv4/dsv4_chat_template.jinja`
  (bos + `<｜User｜>`/`<｜Assistant｜></think>` per `encoding/encoding_dsv4.py`).
- GPUs left drained (87MiB) and free; m27 NOT restarted per user instruction
  (no fallback needed during experiments).

### 2026-09-09 Track C GPU result — Nex serve test, wiring lessons
Stock kernels on Nex: serve OK, 20-token decode = "!!!!" (NaN garbage
reproduced, matches July). sitecustomize-on-PYTHONPATH wiring SEGFAULTS
python startup (torch import race at interpreter init) — reverted and
instead anchored tail-end of gdn_linear_attn.py (class-level patch
patches ChunkGatedDeltaRule.forward_native — import-order safe).
Relaunched (nex-gdn-fb4.log).

## 2026-09-09 (evening) — Track C round 2, WITH GPU access

- Stage-0 kernel smokes (smoke_kernels.py) PASS on gfx906 in bf16 AND fp16:
  all FLA stages match fp32 ref at Nex shapes. Individual triton kernels OK;
  NaN is systemic or elsewhere.
- Nex on :9600 with fp32 fallback installed (anchor at tail of venv
  gdn_linear_attn.py, confirmed in log): 20-token probe STILL '!!!' garbage;
  logprobs endpoint errors with NaN. => either fallback paths not actually
  executed, or NaN originates outside GDN prefill/decode.
- Instrumentation round 1 launched (run_nex_probe.sh: fallback +
  VLLM_GDN_GFX906_NAN_PROBE=1 + --enforce-eager so module hooks fire).
  Answer to "did fallback fire" comes from [GDN-FB] tally counters now in
  patches/gdn/gdn_gfx906_fallback.py; first-non-finite-module attribution
  comes from the global nn.Module NANPROBE forward hook.

## 2026-09-09 (late evening) — Track C: machine crash + MoE attribution

- Probe round 1 result BEFORE crash: first non-finite module in Nex forward =
  **Qwen3NextSparseMoeBlock** (inputs finite, 1 NaN element in (5,4096)
  output). GDN fallback was confirmed ACTIVE simultaneously ([GDN-FB]
  tallies firing, chunk_prefill+packed_decode both hitting). => NaN producer
  is the MoE (router / FusedMoE int4 / shared-expert), NOT GDN recurrence —
  explains why stock triton and fp32 fallback both give "!!!".
- ~16:24 machine HARD CRASHED mid weight-load of probe round 2
  (vllm18-style kernel hang suspected; unproven which kernel). Power-reset by
  user. m27.service auto-started on boot; Nex probe server gone.
- Next: stop m27 (user-approved window), re-run layer-indexed NANPROBE
  (round 2) to name layer + MoE submodule. Will restore m27.service at end.

## 2026-09-09 — Track A2: decode-speed design plan (OFFLINE only, no GPU touched)

Task: paper-plan closing the 2.13 → 15+ tok/s decode gap. Full plan with per-op
cost model + validation recipes is in `track_a2_decode_plan.md`. Summary:

- **Diagnosis**: launch/dispatch-bound, not compute-bound. ~12,400 kernel launches
  per decode token (dominantly the pure-PyTorch mhc hc_pre/hc_post fallbacks:
  `sinkhorn_iters=20` → ~90 launches per hc_pre ×2/layer ≈ 7,700/token). GPU
  compute floor is ~5–15 ms/token (~65–200 tok/s) — the 468 ms/token is ~95% host
  overhead.
- **Ranked plan**:
  1. **R1**: fuse `mhc_pre`/`mhc_post` into one Triton kernel each (register-resident
     Sinkhorn; exact op order preserved). Est. 468→180–260 ms (≈4–5.5 tok/s). LOW risk.
  2. **R2**: enable cudagraph decode (`compilation-config {"mode":0,"cudagraph_mode":
     "FULL"|"FULL_DECODE_ONLY"}`); prerequisites: graph-safe `valid.any()` site
     (`deepseek_v4_attention.py:633`), persistent slot-mapping buffers (per-group
     builders). Est. dispatch-free replay → 8–50 tok/s. MEDIUM risk (multi-stream
     capture on gfx906; `DSV4_SEQ=1` fallback).
  3. **R3**: fused single-kernel sparse decode attention (gather+attn+sink in one
     launch/layer, replacing the reference prefill func in `_forward_decode`).
     Est. 10–20% additional. MEDIUM risk (numeric parity required; logprob A/B gate).
  4. R4 builder host-trim folded into R2/R3.
- Target 15+ tok/s achievable with R1+R2 alone on the compute floor's low end;
  R3 as margin. Each step is a single batched GPU load (scripts sketched in the
  plan file); rollback via `*.bak-dsv4fix` copies + env kill-switches.
- Deferred hazards: no MTP spec-decode with the fp32 hc patch (clamped buffer);
  validate 8k+ ctx quality before production promotion.

## 2026-09-09 (post-crash #2, evening) — Track C resumed after second power cycle

- Machine crashed again during nex-probe2 follow-up (root kernel suspect list
  unchanged; crash happened AFTER probe answers were on disk). Power-reset by
  user; m27.service auto-started on boot (port 9200, ~238GB).
- ROUND 2 ATTRIBUTION WAS SECURED from /data/tmp/nex-probe2.log pre-crash:
  first non-finite module = **Qwen2MoeMLP at decoder-layer ~22** (counter
  offset 120 from 2 warmup forwards; layer=142 mod 60), ONE NaN at
  [token0, ch 3055], amax 0.92. FusedMoE/SparseMoeBlock Poisoned downstream.
  Subtree of sparse MoE block: Qwen2MoeMLP = SHARED EXPERT (gate/up are
  ColumnParallelLinear — clean; down_proj is RowParallelLinear — was NOT in
  WATCH list => first NaN produced either at down_proj gptq GEMM or silu·mul).
- Checkpoint weights scanned offline for layer 22 (and 0,1) shared expert +
  expert.0: ALL qweight/scales/qzeros finite. Not a poisoned-weight bug.
- Plan (this window): stop m27 (user-approved window), round-3 probe with
  RowParallelLinear + SiluAndMul in WATCH to pin gate/up vs down vs silu;
  then fix at the named producer.
- 19:41 round-3 probe launched (RowParallelLinear + SiluAndMul + weight-stats
  in WATCH). Will pin shared-expert submodule. Fix candidates ready:
  clamp silu·mul act before down GEMM fp16-cast (down_proj GPTQLinear casts
  input to fp16 internally — overflow-inf is a plausible first NaN), or
  torch-dequant fallback for the shared expert MLP.

## 2026-09-10 (night) — Track C: root cause CONFIRMED, fp32-dequant fix built + CPU-validated; box KFD-wedged, GPU verification pending reboot

### Round-3 probe attribution (nex-probe3.log from 19:41 09-09; first read 04:4x 09-10)

First nonfinite tensor in the entire Nex forward (5-token prefill, TP8):
- `Worker_TP3  [NANPROBE] NONFINITE module=RowParallelLinear layer=142 (probe counter; decoder layer ~22 MoE block) shape=(5,4096) nbad=1 amax=8264 idx=[[0,3055]] first=True`
- inputs FINITE, inputs_absmax=5590 (silu·mul activation); qweight/qzeros/scales FINITE (amax 2.1e9 packed / 2.0e9 / 1.92).
- Shared-expert gate_up (MergedColumnParallelLinear (5,256)) and SiluAndMul (5,128) logged NOTHING at counter 142 (finite); their first NaN only appears at counter 143 (next layer) — already poisoned there.
- => the NaN is BORN inside the shared-expert down_proj `ops.gptq_gemm` (fp16 w4a16 kernel), from finite inputs and finite weights. Single output element [token0, ch3055].
- Propagation: Qwen2MoeMLP -> FusedMoE -> Qwen3NextSparseMoeBlock -> DecoderLayer(22) out -> GemmaRMSNorm(23) nbad=1 amax=76.6 -> MRotary/Attention(23) fully poisoned -> dense MLP(23) all-NaN -> RMSNormGated (GDN, 24) -> LogitsProcessor all-NaN from counter 179 on. Exactly the "one bad dot product" cascade.
- Source-level mechanism (csrc/quantization/gptq/q_gemm.cu): fp16 kernel; per-thread-block partials are converted to half2 and `atomicAdd`'ed into the fp16 output buffer (q_gemm.cu atomicAdd at L227/349/479/608). At act absmax 5590, a k-block partial for ch3055 exceeds fp16 65504 -> +-inf; when opposing-sign partials meet, inf + (-inf) = NaN at exactly one element, rest of the row finite (observed amax 8264). Matches probe perfectly.

### Fix (built pre-wedge 09-10 ~03:0x by prior Track C turn; reviewed/re-validated this session)

`VLLM_GFX906_MLP_FP32_DOWN=1` (wired into run_nex_gdn_fallback.sh, run_nex_probe.sh, run_qwen35_gdn_fallback.sh) -> `_install_mlp_fp32_down()` in `patches/gdn/gdn_gfx906_fallback.py` monkeypatches `Qwen2MoeMLP.forward`:
stock gate_up + SiluAndMul unchanged; down_proj replaced by `silu_out.float() @ W_fp32` where W_fp32 is a cached exact dequant of the runtime (exllama-shuffled) packed tensors; TP all-reduce applied iff `down.reduce_results and down.tp_size>1` (mirrors stock; qwen3_next creates the shared expert with reduce_results=False and lets FusedMoE do the single combined reduce — preserved). Bias dropped but Qwen2MoeMLP hardcodes bias=False. Non-GPTQ down_proj (or unexpected layout) -> loud stock-path fallback. Class patch covers Nex AND Qwen3.5-122B (both route shared expert + dense-MLP through qwen2_moe.Qwen2MoeMLP via qwen3_next.Qwen3NextSparseMoeBlock / qwen3_5.py:79,162).
fp32-dequant chosen over activation clamp: acts legitimately reach 5590; a clamp distorts semantics and STILL can't provably keep every fp16 block-partial < 65504, since the kernel saturates at PARTIAL level.

### Offline validation (this session, CPU-only, no GPU touched)
- `patches/gdn/dequant_validate.py` PASS: inverse-shuffle unpack of the runtime qweight == direct unpack of the raw checkpoint, BIT-EXACT (maxdiff 0.0) for L22 down_proj (K=1024 N=4096 G=8 gs=128).
- shuffle emulation in the validator matches csrc `qdq_4.cuh:16 shuffle_4bit_8` bit-for-bit by source; patch gather perm [0,4,1,5,2,6,3,7] inverts it.
- zero convention: kernel `zero_offset = use_v2_format ? 0 : 1` (q_gemm.cu L99/245/368/498); Nex+Qwen3.5 configs have NO checkpoint_format -> GPTQConfig default -> use_v2_format=False -> patch's z+1 == kernel semantics.
- both ckpts: no g_idx tensor, desc_act=False -> runtime empties g_idx and applies gptq_shuffle -> patch assumptions hold for the real runtime tensors.
- Saturation math (validator): ch3055 sum|W|=1605.2, amax|W|=15.375; at |act|=5590 uniform-bound ALL 4096 channels' bound > 65504 -> fp16 saturation was a certainty at observed magnitudes.
- Qwen3.5-122B-A10B-GPTQ-Int4 (on disk): qwen3_5_moe, 48L, hidden 3072, GDN 3:1 (full_attention_interval 4), 256 experts top-8, shared_expert_intermediate 1024, GPTQ int4 gs128 desc_act=False -> same kernel family, same fix covers it.
- py_compile of gdn_gfx906_fallback.py OK. (Full python-import install smoke NOT possible pre-reboot: importing vllm touches KFD -> hangs, see incident below.)

### INCIDENT #3: amdgpu/KFD hard wedge, 2026-09-10 03:55 UTC
- patches/gdn/gptq_ab_test.py ran `torch.ops._C.gptq_gemm` standalone (GPU A/B of the saturation theory). It WEDGED the KFD queue; pid 59036 in D `dma_fence_wait_any_timeout`. Every subsequent GPU-touching proc hangs the same way (04:07 trivial matmul GPU0 pid 60401; 04:47 trivial matmul GPU6 pid 64525; 04:49 CPU-only vllm import pid 65609 — imports still init KFD). Script now carries a DANGER header: NEVER call ops._C.gptq_gemm outside `vllm serve`; use dequant_validate.py on CPU.
- SIGTERM sent to 59036/60399/60401/64523/64524/64525/65605-65609; no effect while in D (documented). kill -9 NOT used (KFD leak rules + ineffective on D).
- Machine state at handoff: m27.service INACTIVE, hy3.service INACTIVE; no vllm serve; VRAM: GPU3 603MB (wedged ab-test buffer), GPU0-2,4-7 ~13-16MB baseline. Load low. NOTHING else running.
- GPU compute is globally blocked; only a REBOOT clears D-state amdgpu tasks (same as 09-09 18:50 incident, KIMI_CRASH_HANDOFF.md). No more GPU procs launched since 04:49 (each adds a D-task).

### Post-reboot runbook (Track C, in order)
1. `rocm-smi --showmeminfo vram` sum < 3e9; confirm no leftover procs.
2. `cd /data/vllm-gfx906-dsv4 && nohup ./run_nex_probe.sh 9600 > /data/tmp/nex-probe4.log 2>&1 &` (fallback + NANPROBE + fp32-down all on, eager). Expect ~6-7 min load, banner `[GDN gfx906 fix] MLP fp32 down_proj bypass installed`.
3. Greedy 20-token `/v1/completions` probe: coherent English + ZERO `[NANPROBE] NONFINITE` lines = fix confirmed. If any new NONFINITE appears, the hook output names the next producer.
4. Bench Nex: `vllm_dsv4_env/bin/python bench_write.py 9600 /data/ModelDownloader/Nex-N2-Pro-4bit-W4A16 /data/llmbench/nex` (creates dir itself). Then TERM server by explicit PID (`pgrep -f "bin/vll[m] serve"`), confirm VRAM drained.
5. Qwen3.5: `nohup ./run_qwen35_gdn_fallback.sh 9601 > /data/tmp/qwen35-gdn-serve.log 2>&1 &` -> same greedy 20-tok coherence check -> `bench_write.py 9601 /data/ModelDownloader/Qwen3.5-122B-A10B-GPTQ-Int4 /data/llmbench/qwen35`.
6. Hang rule unchanged: any request/run >600s -> SIGTERM by explicit PID, log, document.

## 2026-09-10 — Track A2 R1: fused Triton mHC kernels implemented — **UNTESTED (code only, no GPU touched)**

Implements step R1 of `track_a2_decode_plan.md` (see that file for the op-count
cost model). Status: **untested on GPU** — this round was offline-only
(agent-9 owns the GPUs for Track C; KFD is also wedged per INCIDENT #3, so the
next GPU window starts post-reboot anyway).

### What was written

- **New** `vllm/vllm/model_executor/layers/mhc_triton.py` (mirrored to
  `vllm_dsv4_env/lib/python3.12/site-packages/vllm/model_executor/layers/mhc_triton.py`):
  - `_mhc_pre_fused_kernel`: one program per token. Phase A accumulates
    `x @ fn.T` (K=16384) and `sum(x*x)` over BLOCK_K=512 fp32 slices (pure
    mul+reduce, **no tl.dot** — no MMA/precision concerns on gfx906); Phase B
    computes rms-normalized mixes, sigmoid gates, softmax+eps, and the 19×
    Sinkhorn row/col normalizations of `sinkhorn_iters=20` entirely in
    registers ([4,4] fragment); Phase C accumulates `layer_input` in fp32 over
    BLOCK_H=1024 blocks and casts to residual dtype on store. One launch
    replaces ~90 eager ops.
  - `_mhc_post_fused_kernel`: one program per token; computes
    `out[o,h] = post[o]*x[h] + sum_i comb[i,o]*residual[i,h]` in fp32 over
    BLOCK_H=1024 blocks. One launch replaces ~6 eager ops.
  - gfx906 triton 3.5.0 constraints honored: no launch_pdl, no TMA/warp-spec/
    epilogue-fusion APIs, all block dims ≤ 1024, plain fp32 load/store,
    triton imported lazily (module importable with no GPU / no triton).
  - Op order kept identical to `_mhc_pre_fallback` / `_mhc_post_fallback`
    (per-token scalar math, minus the n_splits split-K GEMM, which is absorbed
    into the single fp32 accumulator — reduction order differs at fp32-rounding
    level; GEMM-split order is not exactly reproducible, accepted tolerance).
- **Edit** `vllm/vllm/model_executor/layers/mhc.py` (+ venv mirror;
  `mhc.py.bak-a2r1` backups beside both): dispatch `mhc_pre`/`mhc_post` to the
  triton kernels iff `tilelang is None` AND `VLLM_DSV4_MHC_TRITON` truthy AND
  `residual.is_cuda`. Flag default unset/0 → **current fallback path
  byte-for-byte unchanged**. This install has no tilelang, so the tilelang
  branch stays dead code here.
- **New test** `/data/vllm-gfx906-dsv4/tests/test_mhc_triton.py` (CPU-only;
  pytest optional, runs standalone too). Contains a torch-eager
  reimplementation mirroring the kernel op order (BLOCK_K-sliced GEMM
  accumulation, register Sinkhorn, fp32 accumulate + dtype cast) and compares
  against the production fallbacks, which are extracted from `mhc.py` by AST
  so importing the module never touches vllm.platforms/KFD (INCIDENT #3
  lesson). GPU-vs-kernel cases included but require
  `VLLM_DSV4_MHC_GPU_TEST=1` and skip by default.
  **Result: 9/9 PASS** (T∈{1,3,7} × fp32/fp16 pre, T∈{1,3} post, Sinkhorn
  double-stochasticity), run under `HIP_VISIBLE_DEVICES=""` to stay KFD-free:
  `vllm_dsv4_env/bin/python tests/test_mhc_triton.py` → `ALL PASS`.
- **New GPU script** `/data/tmp/dsv4-fp32fix/expA2_r1.sh` (for the next GPU
  window): exp8-style battery run TWICE (flag OFF baseline, then ON), greedy
  logprob capture on 3 fixed prompts (lp1/lp2/lp3, max_tokens=64,
  logprobs=5) in both phases, A/B compare, 2000-token perf in both phases.

### expA2_r1.sh pass gates

- **G1** both servers reach READY; no engine-fatal/traceback during any phase.
- **G2** OFF phase reproduces baseline (p300/mt1/mt2 `finish_reason=length`
  non-empty; d500 completes 500 tok).
- **G3** greedy A/B: lp1..3 generated token ids **identical** ON vs OFF and
  max |Δ token_logprob| < 1e-2 per position (fp16 tolerance from plan).
- **G4** ON-phase 2000-token perf ≥ 4.0 tok/s (baseline ~2.13; OFF tps
  printed to detect rig drift).
- Soft/info: full-text equality of p300/d500/mt1/mt2 between phases (drift in
  long greedy rollouts is possible within G3 tolerance and is NOT a gate).

### Assumptions to validate on the GPU window

- Reduction-order drift (my fp32 sequential-sum vs hipBLAS GEMM split
  ordering) stays within G3's 1e-2 logprob tolerance end-to-end.
- Register pressure of phase-A: [24,BLOCK_K=512] fp32 loads per iteration
  (12,288 floats) at num_warps=8, BLOCK_H=1024 at phase C — if wave64 spills,
  drop BLOCK_K/BLOCK_H (dims correctness-checked at 512 OK; `tl.arange` sizes
  must stay powers of two; all are).
- Perf floor: per-token single-program kernel — one program for num_tokens=1
  decode, so it's one CU per token like the tilelang original; even 10×
  slower GPU-side would still beat ~90 launches. Validate actual per-layer
  time via tok/s delta (G4), not microbench.
- hc_mult=4, hidden=4096, sinkhorn_iters=20 from config confirmed
  (`hc_mult` asserted ∈ {1,2,4,8}, `hidden % 1024 == 0` asserted).
- post_layer_mix arrives as (..., hc, 1) fp32 → wrapper `.view(num_tokens,
  hc)` after `.float()`/`.contiguous()` — same as tilelang path usage.

### 2026-09-10 05:15 — gate note (main session)
Post-crash #3 re-verified: box still KFD-wedged (uptime continuous since boot; D-state procs 59036/64524/64525/65448/66071 persist; fresh GPU0 matmul probe also entered D).
All GPU tracks (Nex validation probe4, Qwen3.5, A2 R1 test, Behemoth tryout) gated on reboot/power reset. No sudo/ipmitool access from this session.
CPU-only artifacts ready: MLP fp32-down fix (env VLLM_GFX906_MLP_FP32_DOWN=1, CPU-validated bit-exact), mhc_triton.py R1 kernels (CPU parity 9/9), runbook in EXPERIMENTS.md.

### 2026-09-10 05:4x post-reboot — Track C VERIFIED GREEN on Nex
run_nex_probe.sh 9600 (fp32-down bypass ON + probes): model loads, "[GDN gfx906 fix]
MLP fp32 down_proj bypass installed" banner seen, greedy 20-token probe returns
COHERENT English ("...plain. A. alliteration B. assonance C. consonance D"),
ZERO [NANPROBE] NONFINITE lines. The layer-22 fp16-atomicAdd-overflow fix
works. Full bench (perf + writing samples) running.

### 2026-09-10 morning — PRODUCTION DECISION: Qwen3.5-122B-A10B-GPTQ on :9200
Verified end-to-end: coherent text (greedy + temp 0.8), decode 22.5-22.7 tok/s (short ctx),
17.9 tok/s @ 8k ctx, prefill ~275 tok/s. ~2x Hy3 (11.4), > M2.7 (~19.5-20).
Config: run_qwen35.sh (port 9200, max-model-len 131072, fp32, GPTQ, prefix caching,
qwen3 reasoning parser, served names: Qwen3.5-122B-A10B-GPTQ / qwen35 / legacy Hy3+MiniMax paths).
Required patches (both!): VLLM_GDN_GFX906_FALLBACK=1 (GDN fp32 torch fallback via patches/gdn)
+ VLLM_GFX906_MLP_FP32_DOWN=1 (shared-expert down_proj fp32 dequant).
systemd: qwen35.service enabled (m27.service + hy3.service stub units left intact, disabled).

Nex-N2-Pro with same patches: coherent BUT slow (~2.7 tok/s — fp32 dequant of large experts
on PCIe-bound fleet); Qwen3.5-122B (A10B, fewer active experts) is where Track C pays off.
Bench samples: /data/llmbench/{hy3-baseline,m27,nex,qwen35}/.

Behemoth-128B-v3: downloaded (65GB), never benchd — dense 128B → est 2-5 tok/s on this
fleet fails the >=15 tok/s bar by construction; deprioritized per math, kept on disk.

### 2026-09-10 — CAMPAIGN WRAP (production validated)
- Production default: thinking OFF via --default-chat-template-kwargs {"enable_thinking": false}
  (RP budget; ST can still get think-mode per-request by passing enable_thinking=true).
- Long-ctx smoke: 27k-token prompt prefilled at ~296 tok/s, clean completion. KV blocks
  provisioned for full 131072 at startup (vllm would reject if unfittable).
- Rocm-side note: startup banner "TURBOQUANT incompatible → TRITON_ATTN" is EXPECTED.
- Track A2 (DSV4 mhc kernels): code-complete but NEVER GPU-validated this campaign;
  superseded by Qwen3.5-122B winning at 22.6 tok/s (beats the 15 target without it).
  Left untested on purpose; expA2_r1.sh + tests exist for a future pass.
- FINAL STATE: qwen35.service (port 9200) = production; m27.service + hy3.service units
  preserved, disabled, one-command fallback each (see run scripts).

## 2026-09-10 — Track P1: production kernel profiling of Qwen3.5-122B (2 restarts, 1 crash)

Full write-up: /data/llmbench/prof/ANALYSIS.md. Traces: /data/llmbench/prof/trace_w{1,2}_p*.json.gz
+ parse_traces.py / SUMMARY.json / TOP_KERNELS.txt; driver: /data/llmbench/prof_run/drive_prof.py.

### Method
- This fork has NO VLLM_TORCH_PROFILER_DIR / /start_profile. Added
  `patches/gdn/prof_patch.py`: trigger-file-driven torch.profiler wrapper on
  Worker.execute_model, auto-installed from gdn_gfx906_fallback.py when
  VLLM_GFX906_PROF_DIR is set (zero cost unset). Enabled once via systemd drop-in
  (Environment=...), now removed. Backups: gdn_gfx906_fallback.py.bak-prof,
  EXPERIMENTS.md.bak-profpass.
- Windows captured: W1 = 2 chunked-prefill steps (4588-tok prompt) + 6 long-ctx decode
  steps on all 8 ranks; W2 = 40 steps 4-way decode on 2/8 ranks (see incident).

### INCIDENT #4 (profiler-induced, production)
40-step window × 8 ranks at bs4: 6/8 workers spent >5 min exporting ~200 MB traces
(single-thread gzip inline in execute_model) -> EngineCore shm_broadcast dequeue
TimeoutError -> engine fatal -> outage. Restoration consumed the planned second
restart. RULE for future passes: <=12-15 steps/window at bs>=4, stagger per-rank
export (pid-hash sleep) or bump RPC timeout for the profiled run. Prod re-verified
after restore: bench_write decode 22.5-22.8 tok/s (baseline 22.5).

### Headline findings (aggregates across ranks; relative shares robust)
- Prefill: fused_moe_kernel_gptq_awq = 64% GPU time (avg 31.6 ms/call, 36/step);
  ncclDevKernel 18.5% (98 allreduces/step, PCIe ring); GDN chunk fallback torch soup
  = 63k tiny 16x16 GEMMs from _invert_unit_lower loop mode (63 serial iters/chunk) +
  ~300k eager elementwise/bmm calls.
- Decode (bs4, 40 steps): GPU busy only ~42% of span; NCCL 30% + moe_wna16<float>
  24%. CPU: ~1600-2850 kernel launches/step (~10-18 ms enqueue) + ~37 blocking
  memcpyAsync/step (avg 2.6 ms CPU-side) from the per-layer GDN eager recurrence
  (45 layers x ~30 ops) + .tolist() syncs.

### Prioritized optimizations (details in ANALYSIS.md)
- P1 vectorize/graph GDN decode recurrence across layers (kill ~1400 launches + 37
  syncs/step): est decode 22.4 -> 28-31 tok/s.
- P2 batched solve_triangular for _invert_unit_lower (verify tril_solve=triangular_solve
  on gfx906, else 32/64 block recursion): est prefill 296 -> ~390-450 tok/s.
- P3 try ROCm CustomAllreduce (disable_custom_all_reduce=True today; gfx906 PCIe P2P
  probe first) / NCCL Tree+Simple tuning: est decode 22.4 -> 25-26 tok/s.
- P4 gfx906 tile configs for fused_moe gptq kernel (64% of prefill): +10-20% prefill.
- P5 memoize hipGetDeviceProperties callers (175k calls/19s), batch GDN-state copies.

## 2026-09-10 (afternoon) — Track P1: fused GDN decode kernel deployed to production (+22% bs1, +19% conc-4)

### Context decisions
- Found box freshly rebooted (uptime 18 min): **m27.service was ENABLED and
  had won the boot race** — it held all GPUs + port 9200; qwen35.service had
  failed 3x (engine OOM vs occupied fleet) and systemd gave up. Contradicts
  the documented "m27 disabled, qwen35 production" end state. Stopped m27
  (this session's work requires free GPUs anyway; qwen35 restored at end).

### P1-μ: standalone microbench (GPU3, no model)
`patches/gdn/bench_gdn_decode.py` — eager fallback vs new fused triton
kernel `patches/gdn/gdn_decode_fused.py` (1 launch/layerstep vs ~30 eager
ops; packed_decode + sigmoid_gating variants, NULL-slot semantics, no CPU
syncs — packed variant drops in for fused_recurrent_gated_delta_rule_packed_decode,
sigmoid variant drops the ref's cu_seqlens.tolist() sync entirely).

Qwen3.5-122B TP8 rank shapes (H=2, HV=8, K=128, V=128, fp32):
| bs | eager ref us/layer | fused us/layer | max |out| diff | 256-step state drift |
|----|--------------------|----------------|----------------|----------------------|
| 1  | 694.5              | 93.2           | 6.5e-9         | 8.9e-8 (|S|max 0.56) |
| 4  | 666.3              | 93.7           | 1.1e-8         | 8.9e-8               |
| 8  | 667.3              | 94.0           | 1.5e-8         | 1.2e-7               |
Sigmoid path (T=7 mixed) diff: out 1.9e-8, state 1.2e-7. bf16 spot check OK
(diffs = bf16 store rounding; production is fp32 anyway).
36-layer step: ref ~25.0 ms -> fused ~3.4 ms (bs1; launch-bound, not size-bound).
torch.compile(inductor) of the eager ref: no help (dynamo recompile-limit
thrash on varying shapes, ~equally slow as eager). upstream vllm triton
packed-decode kernel NOT benchmarked (importing vllm standalone segfaulted
at exit again — same KFD touch issue as INCIDENT #3; no work lost, see
bench_gdn_decode.log).

### P1-ncclμ: 2-rank RCCL all-reduce (GPUs 0,1, tiny payloads)
`patches/gdn/bench_nccl.py`. Latency at 3KB/12KB/48KB/96KB/256KB/1MB:
- baseline: 53/50/68/63/101/273 us
- NCCL_ALGO=Ring: 137/109/70/81/102/273 (worse at small)
- NCCL_ALGO=Tree: 112/104/84/68/114/255 (worse at small)
- NCCL_PROTO=Simple: 44/46/51/63/101/274 (-10--25% at <=48KB)
- Tree+Simple: 77/78/54/68/114/254
=> NCCL_PROTO=Simple LOOKED like a >10% ping-pong win -> full-model A/B...

### P1-full A/B (port 9600, run_qwen35_gdn_fallback.sh config family —
production flags via run_qwen35.sh, default NCCL envs)
All three runs used run_qwen35.sh-equivalent config (max-model-len 131072,
util 0.80, max-num-seqs 8, mnb-tokens 4096, FULL cudagraph).

| run | probe | bs1 short tok/s | samples avg | 8k ctx tok/s | conc-4 agg tok/s |
|-----|-------|-----------------|-------------|--------------|------------------|
| base (no fused, no NCCL env) | coherent | 22.62 | 22.57 | 10.33 (prefill~158) | 48.58 |
| fused decode | token-IDENTICAL to base, max dlogprob 7.0e-6 | **27.58** | **27.51** | 10.97 | **57.93** |
| fused + NCCL_PROTO=Simple | token-identical, dlogprob 7.3e-6 | 24.14 (REGRESS -12%) | 24.19 | 9.62 | 60.19 |

Also ran fused + VLLM_GDN_GFX906_NAN_PROBE=1 validation run first: ZERO
NANPROBE NONFINITE lines through greedy 20-tok probe + 256-tok temp0.8
probe + conc-4 mixed batch (exercises the sigmoid/prefill+decode mixed path).
NOTE/BUGFIX: NAN probe hook did `isfinite().item()` during cudagraph capture
(hipErrorStreamCaptureUnsupported at startup) — fixed in
gdn_gfx906_fallback.py by skipping hooks while `torch.cuda.is_current_stream_capturing()`.

Decision: **ship fused decode, do NOT ship NCCL env** — the 2-rank microbench
win did NOT transfer to full TP8 (bs1 -12%, conc-4 only +4%). First-order
plausibility: Simple proto caps channel count / payload pipelining that the
8-rank PCIe tree relies on. NCCL remains 30% of decode GPU time; leave
CustomAllreduce (P3) as the next comms lever.

### Production state
- run_qwen35.sh: added `export VLLM_GDN_GFX906_FUSED_DECODE=1` (commented
  with these numbers). No NCCL env changes.
- New files: patches/gdn/gdn_decode_fused.py (kernel+wrappers+installer),
  patches/gdn/bench_gdn_decode.py, patches/gdn/bench_nccl.py,
  probe_greedy.py + bench_conc.py in repo root.
- gdn_gfx906_fallback.py: installs fused decode when VLLM_GDN_GFX906_FUSED_DECODE=1
  (try/except -> falls back to eager torch ref on any error), NAN-probe
  capture-skip fix above.
- Benches: /data/llmbench/qwen35-opt/{base,fused,fusednccl}/bench.json +
  probe_*.json; logs /data/tmp/qwen35-opt-{base,fusedprobe,fused,fusednccl}-9600.log.


### 2026-09-10 16:3x — INCIDENT #5: amdgpu/KFD wedge #2 + /data fs journal wedge during production relaunch — GPU WORK STOPPED, NEEDS REBOOT

- After the successful fused+NCCL A/B, test server was TERM'd (VRAM drained to 4.13e8, clean) and
  `systemctl --user start qwen35.service` (run_qwen35.sh WITH VLLM_GDN_GFX906_FUSED_DECODE=1).
- Start began 15:46; workers reached NCCL/pynccl init at 15:48:24 and never logged again
  (journal silent from 15:48:24 on, no "Application startup", health on :9200 never answered).
- By 16:22: loadavg 175->213 (40 cores), /proc scans: 139 tasks in UNINTERRUPTIBLE D-state,
  including 3 VLLM::Worker procs of the new service (pids 21847/21870/21897, D since ~15:48,
  >40 MIN), ~130 kworkers (kworker/17:xx, 13:xx, 36:xx), AND jbd2/nvme0n1p1-8 (the /data ext4
  journal thread) — after which plain `cat >>` writes to /data HANG (reads still work).
- Kernel log 15:45:53: "kfd_process_wq_release [amdgpu] hogged CPU for >10000us 32 times"
  + "queue evicted / Freeing queue vital buffer" burst — same signature as INCIDENT #3 (KFD wedge).
- Even `ps`/`pgrep` now HANG scanning /proc; a guarded `systemctl --user stop qwen35.service`
  took >90 s. Driver + storage state progressively wedging, not a soft hang.
- PER HARD SAFETY RULE (>5 min D-state): all GPU work stopped. No kill -9 used, no driver
  recovery attempted. Actions taken (all CPU-side, safe):
  * `systemctl --user stop qwen35.service` -> unit ended "failed"/inactive; wedged workers remain
    in D until reboot (SIGTERM cannot reap D-state, expected).
  * `systemctl --user disable m27.service` (was enabled and won this morning's boot race for
    :9200 — root cause of today's "production down" state). qwen35.service remains ENABLED, so
    after reboot production comes up automatically WITH the fused decode kernel (no NCCL envs).
- **STATE: NEEDS REBOOT.** No GPU process can launch; every KFD touch wedges; /data writes hang.
  Nothing is serving on :9200 right now.

### Post-reboot runbook (next GPU window)
1. Verify: `rocm-smi --showmeminfo vram` total < 3e9, zero D-state procs, no vllm serve procs.
2. `systemctl --user start qwen35.service` (or let boot do it — unit is enabled; m27 is now
   disabled so no boot race). Load ~7-10 min cold cache.
3. Verify on 9200: greedy 20-tok probe == /data/llmbench/qwen35-opt/probe_base.json tokens
   (expect ' plain.' + identical ids), temp-0.8 sample coherent, one ~8k ctx request.
4. Expected perf on 9200 (from :9600 A/B): bs1 decode ~27.5 tok/s (vs 22.6 before = +22%),
   conc-4 aggregate ~57.9 tok/s (vs 48.6 = +19%), 8k-ctx decode ~11.0 (vs 10.3).
5. If wedge recurs during load: suspect KFD queue setup under memory pressure;
   try booting ONCE with gpu-memory-utilization 0.75 to test correlation.
   Do NOT run patches/gdn/gptq_ab_test.py outside `vllm serve` (INCIDENT #3).
   Do NOT run bench_gdn_decode.py with vllm imports in-process (segfaults; it is
   written standalone-safe now — keep it that way).

## 2026-09-10 (late) → 2026-09-11 — Track H: Hy3-GPTQ-Int4 decode optimization campaign

Goal: lift Hy3 decode from 11.43 tok/s (bs1) without hurting correctness. NOT attempted:
MTP k=2 (measured negative 9.95 on 09-09; do not retry), PIECEWISE cudagraph
(rejected: fork logs "Cudagraph mode PIECEWISE is not compatible with compilation mode 0.
Overriding to NONE" — mode 0 + PIECEWISE coerces to no-graphs; sweep dimension dropped).

### Load 1 (port 9510): baseline + profile — DONE
- run_hy3_prof.sh = run_hy3.sh flags + VLLM_GFX906_PROF_DIR=/data/llmbench/hy3-prof
  (prof anchor added at tail of venv+source hy_v3.py, imports prof_patch when env set;
  backups hy_v3.py.bak-hy3prof beside both).
- Baseline this load: bs1 11.45-11.54 tok/s, ~5k-ctx 7.12, prefill ~155 tok/s,
  conc-4 (2 concurrent under max-num-seqs 2): 14.07 agg / 7.17 per stream.
  Probe ids: /data/llmbench/hy3-opt/probe_base.json (greedy 20-tok: " plain.\nThe rain...").
- Windows captured: w1 (6 steps: 4.8k prefill + decode start, 8 ranks),
  w2 (12 steps: burst -> bs2 decode, 8 ranks). w3 bs1 decode NOT captured —
  window-2 export overlapped in-flight requests and tripped the 10s VLLM_RPC_TIMEOUT
  (same failure class as incident #4) -> engine dead at 23:02; auto-shutdown, VRAM drained clean.
  Rule: for profiler runs set VLLM_RPC_TIMEOUT=600000 AND finish requests inside the window.
- Full table: /data/llmbench/hy3-prof/TOP_KERNELS.txt + SUMMARY.md (+ parse_traces.py).
  Decode step (bs2, per rank, ~139 ms): NCCL ~28 ms x162; BIG aten direct_copy ~31 ms x160;
  triton_matmul skinny GEMM ~29 ms x321; moe_wna16 decode ~27 ms x158; router fp32 gemm
  ~4.8 ms x79; topk/sort ~3.3 ms; attention 1.5 ms x80. ~3500 kernels/step.

### C1 ROOT CAUSE (confirmed from w1 eager cpu-op chains + source)
The two ~193 us direct_copy kernels per MoE layer per step = torch cast kernels of
cos_sin_cache: fork's csrc/pos_encoding_kernels.cu rotary_embedding does
`auto cache_f32 = cos_sin_cache.to(torch::kFloat32);` EVERY CALL. hy3 cache shape
[262144 x 128] = 2^25 elems (grid 65536x128x4 matches exactly; W1 eager chain:
_C::rotary_embedding -> aten::to -> _to_copy -> copy_). With FULL graphs the 160 casts
are baked into every decode step. Only bites fp16-model rope caches (Qwen3.5 runs fp32 ->
.to() no-op). Fix: kernel now reads cache in its stored dtype (fp16/bf16/fp32 template
dispatch, in-register float cast; bit-exact vs old semantics for fp16 cache since the old
path cast the same fp16 values). Files: vllm/csrc/pos_encoding_kernels.cu
(+ venv _C.abi3.so rebuilt incrementally via ninja; backups pos_encoding_kernels.cu.bak-hy3rope
and _C.abi3.so.bak-hy3rope). Standalone unit /data/llmbench/hy3-opt/rope_unit.py:
fp16-cache vs fp32-cache outputs BITWISE IDENTICAL (5 seeds); call cost 7.2 us vs
1375 us for the old cast alone. (Harness gotcha: `import vllm` standalone segfaults in
this env unless `import requests` precedes it / flashinfer.py chain.)

### Load 2 (port 9511, PIECEWISE attempt): ABANDONED before weights loaded
PIECEWISE + mode 0 coerces to cudagraph NONE (see above); parent kill propagated to
engine (task-timeout lesson: launch serve via setsid so wrapper kills don't nuke it).
No data.

## 2026-09-11 Track D: GLM-5.3-Flash port scoped (agent-14, offline)

Full plan: /data/vllm-gfx906-dsv4/GLM53_PORT_PLAN.md
- Model: cyankiwi/GLM-5.3-Flash-AWQ-INT4 (45L, hybrid: 34x KDA linear-attn + 11x NoPE
  sparse-MLA w/ indexer topk=2048 kpool=4, mHC 4-stream sinkhorn-20, 288-expert MoE,
  +MTP, ViT vision). Quant: CT pack-quant int4 g32 ASYM + fp8-block E4M3 on dense
  MLPs/experts of last 2 layers.
- Support exists upstream vllm main (PR #53906, 2026-09-03) incl. amd/ops KDA triton
  kernels; cyankiwi/vllm@glm53-flash-ct has the CT-weight-remap branch for THIS ckpt.
- Plan: vendor cyankiwi glm5next pkg, downgrade MoE to our old FusedMoE API, fp16
  indexer (skip hadamard/fp8) reusing DSV4 fallback family, KDA via upstream amd/ops
  with GDN fallback parachute, dequant fp8-block tensors to bf16.
- GO conditional. Text-only MVP est 3-5 days; DSV4-class decode (~2 tok/s) day 1,
  4-5 after mhc-triton. Block size must be multiple of 128 (use 256 like DSV4).

### Load 3 (port 9510): C1 rope fp16-cache fix — BIG WIN (+57% bs1)
Config identical to Load 1 baseline minus VLLM_GFX906_PROF_DIR (run_hy3_ab.sh).
Gate: greedy 20-tok probe ids IDENTICAL to probe_base.json (dlogprob max 9e-2,
NCCL-order noise; kernel unit-verified bitwise fp16-vs-fp32 cache).
- bs1 short decode: 18.13/18.09/18.12/18.24/18.16 tok/s (baseline 11.45-11.54) = +57%
- 8k ctx: decode 16.09 (was 7.12 = +126%), prefill ~395 tok/s (was ~155-177)
- conc-4 (2-way cap): aggregate 19.01 (was 14.07 = +35%); per-stream 9.4-9.5 at bs2
- files: /data/llmbench/hy3-opt/{ropefix-clean/bench.json, probe_load3.json}
- the fix is unconditionally-on in the rebuilt venv _C.abi3.so (no env gate needed;
  kernel is dtype-generic; rollback = restore _C.abi3.so.bak-hy3rope).

## 2026-09-11 GLM-5.3-Flash fp8→bf16 dequant (offline, CPU-only)

gfx906 lacks FP8 compute, so the checkpoint's `compressed-tensors` mixed-precision
`group_1` (float-quantized E4M3, block 128×128) is unloadable in our vLLM. Dequantized
it offline to bf16: `/data/ModelDownloader/GLM-5.3-Flash-AWQ-INT4` →
`/data/ModelDownloader/GLM-5.3-Flash-AWQ-INT4-bf16fix`
(script `/home/tliao/dequant_fp8_glm53.py`, notes in DEQUANT_NOTES.md alongside the model).
- 1810 group_1 modules (all 288 experts × layers 44-45, dense MLPs 0-2, 43
  shared_experts.down_proj, 12 q_a_proj/kv_a_proj pairs, +few singletons); exact 1:1
  with the 1810 F8_E4M3 tensors in the checkpoint.
- Formula: `bf16(W) = (fp8(W).to(f32) * block_scale[128x128].broadcast).to(bf16)`;
  `weight_scale` (f32) tensors dropped. All 43 shards rewritten; 197.2 → 212.7 GB.
- config.json: group_1 removed, its 1810 targets appended to `ignore` (542→2352);
  group_0 pack-quantized int4-g32-asym untouched. index `weight_map` pruned of the
  1810 dropped scale entries (rebuilt from actual shard headers; total_size updated).
  Load via stock CT mixed-precision.
- Verified CPU-side: 0 F8 dtypes left in any shard; 3-module dequant roundtrip
  BITWISE EXACT vs written bf16; embed_tokens copy bitwise identical. ~9 min,
  one-shard-at-a-time RAM (~10 GB peak).
## 2026-09-11 (afternoon) — Track D PHASE 1: GLM-5.3-Flash vendored into the fork (offline, no GPU)

Agent-17 (offline vendor; another agent owns the fleet for Hy3). Branch
`glm53-gfx906` in /data/vllm-gfx906-dsv4/vllm (dirty DSV4/gfx906 work carried
in the working tree, NOT committed; only rocm_aiter_mla_sparse.py backends
hunk was committed as HEAD+GLM53-hunk-only so DSV4 edits stay uncommitted).
Commits: f9e5411908 (config), 570fcbe8b5 (kpool kv infra), ba2ced1c6e
(mamba/mla shims), 5ea37482ef (glm5next pkg), 1a496187da (fp16 kpool op),
d8b5d6ee15 (topk width), e48908282f (registry). Not pushed.

Source of truth: cyankiwi/vllm@glm53-flash-ct (= upstream main PR #53906 +
CT loading fixes); trees vendored from there, not cherry-picked.

### Import tree (all under vllm/, venv-synced w/ .bak-glm53 backups)
- NEW `model_executor/models/glm5next/` (package): `model.py` (MoE downgrade
  to fork FusedMoE old-API; mHC shims over torch.ops.vllm.mhc_pre/mhc_post;
  fp32 hc streams per DSV4 bug#1; no SP-MoE), `attention.py` (NoPE-safe rope;
  fp16 indexer init; Glm5NextIndexerCache→compress_ratio=kpool; Glm5NextTailCache),
  `kda.py` (merged conv state shape/dtype; CT-aware projections;
  eager_break_during_capture no-op), `mtp.py` (vendored; phase 3), `multimodal.py`
  (fork stacked-merge load_weights), `nvidia/ops/{fwht kpool,fused_eh_norm}`,
  `amd/ops/kpool_compress.py` (+ fp16 variants), `amd/ops/third_party/kda/`
  (upstream AMD triton KDA; fork fla import paths; local exp2 fallback).
- NEW `model_executor/layers/sparse_attn_indexer_kpool.py` (CustomOp,
  ROCm→fp16 custom op).
- NEW `v1/attention/ops/rocm_aiter_mla_sparse_kpool.py` (fp16 flow: kpool
  compress-write fp16, cp_gather_indexer_k_cache_fp16, fp16_mqa_logits_triton
  / rocm_paged_mqa_logits, top_k_per_row 512-pool, expand+append tail).
- NEW `model_executor/layers/mamba/gdn/base.py` (+__init__) — vendored base;
  `mamba/ops/gather_initial_states.py`, `scatter_states.py` (pure torch;
  upstream triton uses launch_pdl/GDC — unavailable on gfx906).
- EDITS: `v1/kv_cache_interface.py` (KpoolTailSpec), `v1/core/single_type_kv_
  cache_manager.py` (KpoolTailManager + map), `v1/core/kv_cache_utils.py`
  (_get_kv_cache_groups_glm5_next + layout + config; compress_ratio stands in
  for upstream tokens_per_state; per-layer tensors, NO upstream offset
  packing), `v1/attention/backends/mla/indexer.py` (KpoolTailBackend +
  compute_kpool_tail_slot_mapping + builder), `model_executor/layers/mla.py`
  (MLAModules.q_a_proj + unfused path + skip_topk), `envs.py`
  (VLLM_GLM53_INDEXER_FP16 default ON for ROCm), `_aiter_ops.py` (register
  rocm_aiter_sparse_attn_indexer_kpool), `config/compilation.py` (splitting
  op), `models/registry.py` (3 archs), `v1/attention/backends/mla/
  rocm_aiter_mla_sparse.py` (topk width pad to 2176 when index_kpool>1).

### Key shims (GLM53-PORT)
tokens_per_state→compress_ratio · FusedMoEFactory→fork FusedMoE ·
MHCPre/Post/FusedPostPre→fork mhc_pre/mhc_post + separate RMSNorm ·
SP-MoE deleted · fwht128_quant→fp16 q identity · DeepGEMM asserts→relaxed ·
gather/scatter→torch · eager_break_during_capture→no-op · WeightsMapper
orig_to_new_stacked→manual stacked merge (vision tower) · indexer rope
skipped for qk_rope_head_dim==0.

### CPU verification (no GPU touched)
- py_compile: all new/edited files OK.
- import: `vllm.model_executor.models.glm5next` package + 3 registered
  classes resolve end-to-end (workaround: `import requests; import
  vllm.config` first — without it, `vllm.model_executor.custom_op` import
  segfaults standalone (PRE-EXISTING on this box, reproduces from untouched
  venv; vllm serve unaffected)).
- get_config(ckpt) parses: 45L/34 KDA/11 sparse-MLA, index_topk 2048,
  kpool 4, NoPE rope0, mhc 4-stream. Registry resolution OK (all 3 archs).
- kv group math smoke: groups (22 attn-uniform + 34 mamba + 11 tail);
  pages: MLA 256KB, indexer 16KB (64 pools), mamba 526592B, tail 2048B.
- KpoolTailManager: 1 block/req, no caching, no re-alloc. OK.
- Weights spot-checked vs safetensors index: indexer wq_b/wk/weights_proj are
  BF16 dense (CT ignore); MLA/MoE int4 pack via weight_packed route;
  layers 0-2 dense MLP + layers 44/45 experts + shared down are fp8-block.

### Phase-2 open items (GPU bring-up)
1. fp8-block CT modules (dense MLPs 0-2, shared-expert down_projs, experts
   of layers 44-45, fp8 kv_a/q_a scales, MTP experts): gfx906 has no fp8
   compute — need the offline dequant-to-bf16 script OR a load-time dequant
   hook (plan §5 item 8-A recommended). Currently they fall to CT schemes
   whose gfx906 viability is unverified.
2. MTP: vendored but not wired into run_glm53.sh (hc fp32-clamp caveat;
   skip_topk path now exists in the fork MLA wrapper). LogitsProcessor on
   this fork lacks get_top_tokens (upstream MTP uses it) — add when enabling.
3. KDA chunk/decode amd triton kernels are GPU-unvalidated (have
   GDN_GFX906_FALLBACK-style parachute per plan §5.7 if NaN).
4. fp16 paged logits: run with VLLM_ROCM_MLA_SPARSE_FP16_TRITON=1 (stage1
   kernel asserts kv page 64 = block256/kpool4 ✓); torch fallback is slow.
5. Runtime check: GLM5 topk buffer width 2176 flows (NUM_TOPK_TOKENS=2176,
   %128==0) through triton_convert_req_index_to_global_index/fetch_id_to_ragged.
6. Confirmed chunk alignment: prefill chunk starts are multiples of
   max_num_batched_tokens (4096%4==0) and prefix hits align to scheduler
   block 256 (divisible by kpool=4) → pool-aligned compress inputs. ✓ static.
7. Prefix caching with hybrid mamba: hash_block_size resolves to GCD — tail
   spec block=4 forces fine (4-token) hashing with caching on; consider
   disabling prefix caching for GLM5 if overhead shows.
8. Vision tower: registered + vendored but run with
   --limit-mm-per-prompt image=0,video=0; multimodal processor runtime untested.
9. Run: /data/vllm-gfx906-dsv4/run_glm53.sh (PORT 9700). First boot: watch
   the CT scheme selection logs for the fp8-block groups.

Previous note holds: standalone-CPU vllm import segfault on this box SOMETIMES
(still present; import-order workaround `import requests; import vllm.config`
first is reliable in three attempts).

## 2026-09-11 — Track H / C2: hand-built skinny fp16 GEMV for gfx906 (microbench PASS, server A/B pending)

Target: hy3-prof SUMMARY.md cluster C2 (triton_matmul_kernel, 29 ms/step, 321
calls, 10-15x off HBM roofline). Agent-19; GPU 3 ONLY (fleet is on a concurrent
GLM bring-up — do NOT disturb; microbench was restricted to
HIP_VISIBLE_DEVICES=3, asserted in the bench script itself).

### What was built

- NEW `/data/vllm-gfx906-dsv4/patches/gdn/gfx906_gemv.py`: Triton GEMV
  `gemv_m(x [M,K] fp16, W [N,K] fp16) -> [M,N] fp16` for M<=8. No tl.dot
  (gfx906 has no MFMA): one program per BLOCK_N weight rows, k-major
  [BLOCK_N x BLOCK_K] tiles (16B/lane), W tile reused across M rows via a
  static per-row loop (fp32 acc; the naive [BM,BN,BK] broadcast-sum form
  collapses at M=4 — register cliff, 2.8x slower). num_warps<=4, single
  launch, no atomics/split-K. Autotune 12 configs x key(M,N,K); winners
  mostly BLOCK_N=1/BLOCK_K=1024/warps=2 (short-K: BN=16/BK=128/w=4).
- Reversible install: `install_gfx906_gemv()` wraps
  model_executor/layers/utils.triton_matmul; fires only for fp16, M<=8,
  contiguous W, K%8==0, else passes through to the original kernel. Anchor
  appended at tail of hy_v3.py in BOTH source repo and venv (diff-verified),
  gated by `VLLM_GFX906_GEMV=1` (default off). Runtime undo:
  uninstall_gfx906_gemv().
- Bench: `/data/llmbench/hy3-opt/gemv/bench_gemv.py` — standalone, vendors
  the baseline kernel verbatim from utils.py (no vllm import; INCIDENT #3
  rule). Timing via CUDA-graph replay over a >=64 MB cold-L2 weight pool
  (production is FULL graphs; eager timing is launch-floor-bound ~60-70us
  for BOTH kernels and useless for comparison).

### Results (all on GPU3; correctness 45/45 allclose(atol=1e-3,rtol=1e-3) vs torch.mm)

- Hy3 TP8 per-rank decode shapes (M=2): qkv 4096x1280 160->36 us;
  o_proj 1024x4096 71->19; shared gate_up 4096x384 107->17;
  shared down 192x4096 17->10; lm_head 4096x15104 829->201. Big-N shapes
  reach 525-735 GB/s; small-weight shapes are latency-floor-bound.
  M=1 and M=4 all win too (x1.5-x7.9 range overall). Qwen3.5-122B TP8 alt
  shapes (hidden 3072/head 256/KDA N odd incl. literal N=2177) all win and
  match; M=8 sanity passes (x~2).
- Measured baseline cluster sums reproduce the profiler's 29 ms/step
  (29.2 @ M=2, 28.6 @ M=1) -> substitution projection: cluster 29.2 ->
  6.8 ms/step @ bs2; bs2 step 140 -> 117.8 ms => tok/s_eq 14.3 -> 17.0
  agg (+19%); bs1 11.5 -> ~15.7 tok/s (+36%). Full tables:
  /data/llmbench/hy3-opt/gemv/RESULTS.md (+ raw logs, bench_rows.json).

### NOT done / next window (needs fleet)

- Full Hy3 server A/B: greedy token-identity vs probe_base.json + bench
  with `export VLLM_GFX906_GEMV=1` added to run_hy3.sh (commented line
  already present). lm_head may route fp32 (enable_lm_head_fp32=true) ->
  dispatch passes it through; verify call counter at server time.
- Lessons: (1) for skinny-GEMM microbench on this rig, eager CUDA-event
  timing is dominated by ~60-70 us Python/triton launch overhead on BOTH
  sides — graph-replay timing (32 calls baked over a cold-weight pool) is
  the honest method. (2) detached GPU runs: use setsid+nohup+< /dev/null
  (wrapper kill propagation bites otherwise, Load-2 lesson).

## 2026-09-11 Track D PHASE 2 (GPU bring-up) — agent-18 — NEEDS REBOOT (KFD wedge)

Highest rung reached: **R1 not yet** (server never became ready). Blocked by KFD wedge.

### Corrections landed (branch glm53-gfx906, in order; all mirrored to venv w/ .bak-glm53)
1. run_glm53.sh: --limit-mm-per-prompt needs a JSON dict ('{"image": 0, "video": 0}'),
   not image=0,video=0 (fork's json.loads parser). Script is untracked (work dir is not a repo).
2. 5179dd1240 vendored upstream transformers_utils/processors/glm5next.py (MultiModalBudget
   dummy-inputs runs even with image/video limits = 0).
3. d7fca6e887 shimmed get_merged_mm_kwargs modality= kwarg (fork ctx predates upstream sig).
4. 8c5744b665 build sparse indexer even when qk_rope_head_dim == 0 (NoPE) — phase-1 shim had
   gated the whole Indexer off; ROCMAiterMLASparseImpl asserts indexer is not None.
5. 4e235c9ed6+627ae475c6+29cf54747c asymmetric int4 g32 zero points in the fork's CT WNA16
   MoE path (pack-quant checkpoints are ASYM; stock method asserted symmetric-only):
   w13/w2_weight_zero_point params (transposed loader layout), repack int32 GPTQ-style zps to
   the moe_wna16 triton kernel layout uint8 [E, N/2, K/g] (byte m = outputs n=2m low nibble,
   n=2m+1 high; CT's +8 unsigned offsets cancel between weight and zp nibbles). CPU-verified
   BITWISE-exact vs compressed-tensors dequant on real ckpt tensors (8000 samples, err 0.0).
6. 9c13c9c1bf added glm5_next/glm5_next_text to is_deepseek_mla() model_type tuple: config
   ships head_dim: 0 (NoPE); without it use_mla=False → platform _align_hybrid_block_size
   divided by attn_page_size_1_token == 0 (ZeroDivisionError) for the hybrid KDA model.
   With it: use_mla=True, head_size = kv_lora_rank + qk_rope_head_dim = 512.

### Boot progress timeline (port 9700, TP8, fp16, max-model-len 32768)
- Weight load COMPLETED: 43/43 shards, 24.14 GiB/GPU, 519.6s (~8.7 min). Asym-CT MoE works.
- Then EngineCore failed: workers died in platform._align_hybrid_block_size (the head_dim:0
  ZeroDivisionError above) BEFORE fix #6 landed. After patching, APIServer pid 79124 was
  SIGTERMed... and wedged: D state (wchan dma_fence_wait_any_timeout), VRAM frozen at exactly
  35,100,532,736 B with no KFD attribution, unresponsive to SIGTERM for >5 min.
- A second D-state: headsize_check.py (pid 82623) — a plain CPU-only probe that imported
  vllm with os.environ["CUDA_VISIBLE_DEVICES"] set AFTER the vllm import — too late; torch
  had already initialized HIP and the process touched KFD, then also hung in Dl state.
  LESSON: set CUDA_VISIBLE_DEVICES="" in the ENVIRONMENT before exec, not inside the script
  after importing vllm; NEVER import vllm on this box without it for CPU probes.

### STATE: NEEDS REBOOT
- Hung: vllm serve pid 79124 (Ds, SIGTERM pending; do NOT kill -9), python probe pid 82623 (Dl).
- 35.1 GB VRAM orphaned (no KFD PIDs listed). Both frozen ≥ 03:38Z (checked 03:42Z).
- After reboot, phase 2 resumes at R1: all six fixes above are committed on glm53-gfx906 and
  mirrored into the venv; run /data/vllm-gfx906-dsv4/run_glm53.sh (PORT=9700). Expected next
  milestones: KV-cache init + sparse-indexer backend smoke (watch for GDN/KDA NaN — parachute
  VLLM_GDN_GFX906_FALLBACK=1), then greedy probe "The rain in Spain falls mainly on the",
  creative t=0.7 samples, bench_write.py, VLLM_DSV4_MHC_TRITON=1 A/B, max-model-len ladder
  32768→65536→131072.

## 2026-09-11 Track D PHASE 2 continued — kernel/block-size (backend) fixes, NOT booted; fleet held by prod :9200

Post-reboot (~04:00), floor clean. Long fix chain on glm53-gfx906 (all committed + mirrored);

### Commits this session (in dependency order)
1. `5179dd1240` Vendor upstream `Glm5NextProcessor` (MultiModalBudget dummy-input path needs it even with image/video=0).
2. `d7fca6e887` `get_merged_mm_kwargs(modality=)` shim in glm5next multimodal.py (fork predates upstream kwarg).
3. `8c5744b665` Build the sparse Indexer even when NoPE `qk_rope_head_dim == 0` (phase-1 shim gated it off → `assert indexer is not None`).
4. `4e235c9ed6`+`627ae475c6`+`29cf54747c` Asymmetric int4 g32 zero points in fork CompressedTensorsWNA16MoEMethod: register w13/w2_weight_zero_point (loader layout transposed), repack int32 GPTQ-style → kernel layout uint8 [E, N/2, K/g]; dispatch seam logs asym route. CPU-verified BITWISE vs compressed-tensors dequant on real ckpt tensors (8000 samples, err 0.0).
5. `9c13c9c1bf` `GLM5Next` → `is_deepseek_mla()` (config ships head_dim:0 → page_size_bytes=0 → ZeroDivisionError in `_align_hybrid_block_size`). use_mla=True, head_size=kv_lora_rank+0=512.
6. `10813b396b` Skip hybrid attention/mamba block-size bump for kpool models (per-group KV tensors in this fork make the uniform-page invariant moot; bump 256→768 would break 64-entry pool-page fast path).
7. `42f90b54dc` KV sizing: count mamba(34×)/tail pages per layer in bytes_per_block (was counted once → ~34× num_blocks overcount → OOM at 31.3 GiB).
8. `5c8d159+9f805b014f` `_reshape_kv_cache_tensors` compressed pages stay at manager-block granularity (kernel split of 256→64 was shrinking kpool view to 16 ≠ kernel's required 64). Math verified vs real ckpt tensors.
9. `c46c7cdce7` `KpoolTailBackend.get_kv_cache_stride_order` 4-D (shape is [blocks, 2, kpool, head_dim]).
10. `f0307089c0` KVBlockZeroer per-page-size buckets (256KiB MLA + 16KiB indexer in same group; uniform-page assert tripped).
11. `d8141f64d…/6915d39c67/7965e0b24c` Reverse-divisibility in the coordinator hash assert (KpoolTailSpec block=4 vs hash 256) + diag prints on failure.

### Status achieved
- Server reachability: multiple attempts reached "Uvicorn running" + v1/models answered (806 — pids 26761 etc.). First /v1/completions request exercised the model and hit the block-hash assert, then later the fp16-paged-MQA block-size assert (fixed items 8).
- KV budget @ util 0.92: avail 3.71 GiB/GPU → 412 model blocks (min block 4 tokens each) — tight; max-len 32768 viable per group's math.

### LAST-BLOCKER (runtime, first request)
Under `--enable-prefix-caching` + hybrid hash GCD: `find_longest_cache_hit` → `KpoolTailSpec(block=4)` hits `BlockHashListWithBlockSize(target=256?, hash=4?)` divisibility assert at request time. Teammate mitigations: `run_glm53.sh` now runs with `--no-enable-prefix-caching` -- acceptable for phase 2/3 (prefix caching note #7 in phase-1 open items recommended exactly this if overhead showed; the 4-token hash GCD does show).

### CURRENT FLEET STATE (blocked)
- Production Qwen3.5 auto-up on :9200 at 08:21, healthy 08:26+, holding ~28 GiB/GPU (224/256 GiB). GLM-5.3-AWQ needs ~24.6+ GiB/GPU for weights: PHYSICALLY CANNOT COEXIST. All post-08:21 GLM starts died at "Free memory 5.62 GiB < 25.59 GiB (0.8 util)".
- Cannot touch Qwen35 (rule: never stop :9200 service/configs). GLM-5.3 phase-2 MUST run in a window where :9200 service is SIGTERM'd gracefully, or wait for production downtime.
- No D-state / wedges. rocm-smi clean. Open vllm serve processes: none.

## 2026-09-11 Track D PHASE 2 CONTINUED (post-reboot window) — resume & status

Post-04:07 reboot: brought GLM-5.3 phase 2 forward through the whole engine-init chain.
All fixes below committed on glm53-gfx906 + mirrored to venv (with .bak-glm53 beside them).

Fixed in order (each = one engine-init or first-request crash resolved):
1. Vendor upstream Glm5NextProcessor (5179dd1240) — MultiModalBudget always builds mm dummy inputs
   even with image/video limits zeroed; processor module was missing.
2. get_merged_mm_kwargs(modality=) shim (d7fca6e887) — fork predates upstream kwarg.
3. NoPE sparse indexer constructor (8c5744b665) — phase-1 shim had `if is_v32 and qk_rope>0`
   gating the whole indexer; GLM-5.3 is qk_rope=0 → backend's `assert indexer is not None`.
4. ASYM CT int4 g32 MoE zero points (4e235c9ed6 + 627ae475c6 fixup + 29cf54747c seam log):
   registered w13/w2_weight_zero_point params (transposed loader layout) and repack to the
   moe_wna16 kernel's uint8 [E, N/2, K/g] nibble layout; CT pack_to_int32's +8 offsets cancel
   between weight and zp nibbles. CPU-verified **bitwise** vs compressed-tensors dequant on
   real checkpoint gate_proj (8000 random samples, max abs err = 0.0).
5. `glm5_next(_text)` registered as is_deepseek_mla (9c13c9c1bf) — config head_dim=0 (NoPE)
   produced page_size_bytes=0 → ZeroDivisionError in `_align_hybrid_block_size`.
6. Skip hybrid block-size bump for kpool models (10813b396b) — per-group KV tensors make
   the uniform-page invariant moot; bump would push block 256→768 and break the 64-pool page.
7. KV sizing per-layer for mamba(34×)/tail groups (42f90b54dc) — num_blocks was ~34× too
   large → OOM at 31.3 GiB/GPU.
8. `_reshape_kv_cache_tensors`: compressed (kpool) pages at manager-block granularity — kernel
   split 256→64 shrank the view to 16 entries vs kernel's required 64 (5c8d159, corrected by
   9f805b014f). Element-count math verified against the real indexer tensor layout.
9. KpoolTail 4-D stride order (c46c7cdce7).
10. KVBlockZeroer per-page-size buckets (f0307089c0) — 256 KiB MLA + 16 KiB indexer coexist.
11. Prefix-caching hash asserts: reverse-divisibility adapter (d8141f64d9) + diag prints
    (6915d39c67, 7965e0b24c).

Achieved: Uvicorn up on 9700, /v1/models 200 OK (pid 26761 log lines). First request crashed
in prefix-cache hash path → teammate set `--no-enable-prefix-caching` (matches phase-1 open
item #7 guidance); pid 28773 then booted OK (watcher READY), but mid-request it crashed in
`deepgemm_fp16_paged_mqa_logits_stage1` (BlockSize 16!=64) → fixed by #8. Subsequent attempts
still died — at this point Qwen3.5-122B production service (auto-started on :9200 at ~08:21,
healthy from 08:26 per service journal) came up. VRAM: 224/256 GiB held by prod;
GLM-5.3 AWQ needs ~25 GiB/GPU at TP8 → **blocked**; all post-08:21 attempts died in
"Free memory 5.62 GiB < 25.59 GiB" at worker init.

NO wedge/D-state this session. All GPU processes clean-terminated (SIGTERM honored).

Phase-2 remaining (run when fleet is free): R1(already-done-ish) → R2 greedy probe
("The rain in Spain falls mainly on the") → R3 creative 2-3k t=0.7 → R4 bench_write.py →
R5 VLLM_DSV4_MHC_TRITON=1 A/B → R6 ctx ladder. All on glm53-gfx906; run script ready with
--no-enable-prefix-caching + --max-model-len 8192 + util 0.80 + seqs 2 tuning (teammate edits).

## 2026-09-11 (evening) — GLM-5.3 first boots post-reboot: KV-starvation fixed, cudagraphs on

(post-compaction main-agent session; agents 18/19/20 lost to session crash, work recovered from disk)

1. util 0.80 boot: weights 27.7 GB/GPU (bf16fix + fp16 runtime islands) vs 25.6 GiB budget
   → profiler gave KV **40 tokens** total ("Maximum concurrency 0.97x"). Greedy probe (8 tok)
   PASSED (" plain.") but 2nd request all-rank Memory access fault at first decode (cache
   overrun / 9.7% KV use at 7 ctx tokens). LESSON: on this rig ALWAYS check the
   "GPU KV cache size" line after boot; <1 request of capacity = will fault on prefill/decode.
2. util 0.94 (run_glm53.sh): "GPU KV cache size: 336 tokens" (manager blocks; ~7.67x
   concurrency at 8192 → ~63k effective tokens). Greedy probe PASS; 256-tok creative
   (prev crasher) PASS — memfault gone. Speed: **2.6 tok/s decode, eager** (script lacked
   --compilation-config).
3. + `--compilation-config '{"mode": 0, "cudagraph_mode": "FULL"}'` (same as run_qwen35/run_hy3;
   backend reports UNIFORM_SINGLE_TOKEN_DECODE → auto capture FULL decode bs∈{1,2}):
   decode **6.2 tok/s** (256- and 512-tok probes agree), greedy still " plain.", no drift.
   Still 2.5-3x short of the >15 tok/s goal → profile next (dominants suspected: moe_wna16
   on 288-expert int4 g32, fp16 sparse-MLA paged logits, KDA amd triton, NCCL).

## 2026-09-11 (evening, post-reboot session 2) — GLM decode profiling; mHC hang postmortem; GPU5/7 KFD leak

### Decode-step kernel census (sliced inside one FULL-graph replay, rank0, profiler-inflated 363ms/step; wall 161ms = 6.2 tok/s)
- ncclDevKernel: 42.2 ms/step x92 (92 allreduces = 2x46 layers; PCIe TP8, latency-bound) — 26%
- aten tiny-op tail INSIDE the graph: ~75 ms/step across ~13,756 kernels (fp32
  vectorized_elementwise x5464, elementwise_manual_unroll x4481, reduce<float> x3841, ...)
  — 47%. NOT mHC-dominated (mhc_pre only ~5.9ms/step incl. sinkhorn) — bulk is the
  KDA fp32 state machinery + indexer torch plumbing spread through model code.
- fp32 rocBLAS Cijk SB: 22 ms x90 + more sizes ≈ 31 ms — fp32 GEMMs: MoE routers (fp32)
  + hc-stream mixes — 19%
- moe_wna16 int4 g32: 26.4 ms x82 (41 MoE x gate_up/down) — 16%
- LLGemm1 skinny: 3.8 ms x363; gptq gemm_half_q_half: 4.5 ms x142 (CT int4 dense linears)
- fused_recurrent_gated_delta_rule decode kernel PRESENT (KDA decode is on the fused
  triton path + compute_gate in-kernel — prior fusion work in glm5next/nvidia/kda.py).
- CPU side per step: only ~26 aten::copy_ (sampling allgather) — everything else captured.

### A/B results this session (all: greedy probe " plain." gate PASS)
- eager (no compilation-config): 2.6 tok/s
- + cudagraph FULL (mode 0, as run_qwen35/run_hy3): 6.2 tok/s (2.4x)
- + VLLM_GFX906_GEMV=1 (Track H C2 GEMV anchored in glm5next/__init__): 6.2 tok/s —
  NO EFFECT. triton_matmul cluster was prefill-phase, not decode. Lesson: slice profiler
  windows per-step before trusting cluster tables; whole-window tables mix phases.

### mHC fused Triton (VLLM_DSV4_MHC_TRITON=1)
- OFFLINE on GPU3 (single proc, repro /data/llmbench/glm53-prof/repro_mhc_triton.py):
  mhc_pre 2098 -> 201 us (10.5x), mhc_post 123 -> 65us (1.9x); maxabsdiff vs fallback
  3e-7/3e-7/3e-5 (pre) 6e-5 (post). WORKS.
- TP8 BOOT HANG (stale ~/.triton from earlier crash storms): all 8 workers froze at first
  profiling forward (CPU-side; GPUs 0% util; shm_broadcast 60s warnings from 12:34).
  SIGTERM'd serve; TWO workers ended as ZOMBIES (PID1 reaping stuck) pinning
  27.46 GB each on GPU5+GPU7 (KFD leak class; not releasable in place; needs power reset).
  Cleaned ~/.triton/cache entirely (3.9GB, contained interrupted-compile dirs). Retry boot
  with clean cache: pending verification — expect it to reach past profiling, then fail the
  free-memory guard (GPUs 5/7 only 6.9GB free vs 23.96 GiB needed). BLOCKED on reset.

### House rules reinforced
- KV-starvation guard: after every boot CHECK "GPU KV cache size" line; util 0.80 with
  27.7GB/GPU weights gave 40 TOKENS total → 2nd-request all-rank memfault. 0.94 → ~63k tok.
- NEVER launch GPU work while zombie workers pin VRAM; boot fails theory-independently.
- SIGTERM-only discipline preserved; no kill -9 used anywhere.
- Run serve via setsid nohup (wrapper kill propagation lesson stands).

## 2026-09-11 (night) — post-reset: mHC-Triton TPM chain diagnosis resolved

- Power reset cleared the GPU5/7 KFD leak (zombie-pinned 27.46GB each; systemd(ASPID1
  reaping was wedged). CLEANUP ROUTE for this class: reboot, no userspace fix found.
- Root cause of the mhc2 "6/8 NCCL clients" crash: NOT mHC — workers on the two
  leaked GPUs could not initialize at all (NCCL TCPStore join timeout after 601s).
  mHC-triton was never actually implicated in that failure.
- The ORIGINAL mHC hang (12:34 boot) is reproduced-adjacent but indirect: first boot with
  VLLM_DSV4_MHC_TRITON=1 on DIRTY ~/.triton/cache -> all 8 workers froze at first
  profiling forward (triton JIT compile storm w/ poisoned cache state). Wiping
  ~/.triton/cache + fresh fleet: TP8 boot PASSES profiling and READYs (mHC triton active
  through cudagraph capture).
- BUT: first real PREFILL request then silently wedges the workers (no traceback, no
  HIP fault; EngineCore dies on mq timeout; workers exit cleanly on parent death — VRAM
  drains fine). Decode-shape capture (bs 1,2,4) compiled OK. Offline repro (T=4,H=6144)
  fine and 10.5x faster. => shape- or path-specific hang in the fused mHC kernels at
  real-prefill shapes (max_num_batched_tokens=2048 chunked prefill eager path).
  PARKED: VLLM_DSV4_MHC_TRITON=0 in run_glm53.sh until the prefill hang is root-caused
  offline (extend repro_mhc_triton.py to T=2048 chunked-prefill shapes + all batch sizes).
- Known-good config restored (GEMV on, mHC off): expected 6.2 tok/s; bench + quality next.

### Bisect results (stability matrix, all post-reset on port 9700)
- graphs FULL + fused OFF (stable): greedy ' plain. But in the UK, it falls everywhere...',
  decode 6.2 tok/s (512-tok), bench_write samples: bs1 6.8-7.0 tok/s short-prompt,
  4.82 tok/s w/ 4.6k-ctx prompt, prefill ~74 tok/s@4.6k. QUALITY: strong prose
  (see /data/llmbench/glm53/baseline/*.txt); NOTE thinking leaks into content by
  default (chat template defaults reasoning_effort=max) — for chat use pass
  chat_template_kwargs {"clear_thinking": true} or serve w/ --reasoning-parser.
- graphs FULL + MHC_FUSED=1 (P0): capture OK, speed 8.7-11.7 tok/s BUT output
  corrupted (greedy drift t7, t0.7 repetition loops).
- graphs NONE (eager) + MHC_FUSED=1: CLEAN (greedy 'UK falls mainly on the hills',
  coherent creative) => fused mHC kernel is eager-correct; corruption is specific
  to cuda-graph capture/replay. agent-23 hunting offline via graph-replay repro.
- fused-router (P2): MEMFAULT at decode-capture (alone at fault; mHC exonerated by
  bisect). Needs layout hardening; separate fix cycle.

### MTP bring-up (patch patches/gdn/glm53_mtp_patch.py, plan MTP53_PLAN.md)
- Boot 1 exited ~60s post-env-warnings, no traceback (unresolved; suspect early
  engine-core death under pipe redirection; manual equivalent boot proceeded fine).
- Boot 2 (manual diag): engine-core ready-timeout 600s EXCEEDED — reflect:
  triton cache was wiped pre-reset → full recompile (+MTP draft init). Lesson:
  VLLM_ENGINE_READY_TIMEOUT_S default 600 too small right after cache clears;
  set 1800 for cold-cache boots.

## 2026-09-11 (night, session 3) — fused-kernel verdicts; MTP; SECOND box wedge

### Agent-23 offline kernel results (trace-free, all on GPU3 + CPU)
- mHC fused kernel EXONERATED: graph-replay == eager-fused BITWISE over 48 cycles;
  eager benches with REAL ckpt hc params, fp32 streams, T∈{1,2,4,8}, scales 0.05-100:
  ALL PASS at fp32 floor (rel ≤6.3e-7). The "corruption" I saw = fp32 op-order delta
  through fp16 consumer rounding (near-tie greedy flips; distributional, not a bug).
  ACCEPTANCE GATE MUST BE distributional (logprob A/B), not single-probe greedy.
- Router fused kernel had REAL OOB (strided-view gating_output: stride before
  .contiguous() copy) — FIXED + extended benches (fp16/bf16 BITWISE, fp32 ≤6e-8,
  capture smoke pass). Deployed copies synced (md5 1dc18a54). Re-enable when next
  capture-shape boot cycle happens.
- Sparse-MLA decode v1: exact but 10x SLOWER than torch ref — do not wire. v2
  two-kernel rewrite compiled but UNBENCHED (GPU3 wedged before bench).
- P0 currently gated by distributional A/B. If it passes: 6.2 -> est ~9-10.5 tok/s.

### MTP (k=1) boot trilogy
- Structured patch pack ready offline: patches/gdn/glm53_mtp_patch.py + driver
  patches/gdn/glm53_mtp_main.py + MTP53_PLAN.md (flags: --speculative-config
  '{"method":"mtp","num_speculative_tokens":1}'; perf escape hatch env
  VLLM_GFX906_GLM53_MTP_FULL_CG=1 to keep FULL cudagraphs under spec decode).
- attempt A (scripted): exited ~60s, no traceback (unresolved early death).
- attempt B (manual): engine-core READY timeout at default 600s — cold triton cache
  recompile + MTP draft init overshoots it; fix: VLLM_ENGINE_READY_TIMEOUT_S=1800
  (added to run_glm53_mtp.sh).
- attempt C: progressed fine through draft arch resolution (Glm5NextMTPModel resolved
  natively), then the box KFD-wedged mid-engine-spawn (see below).

### SECOND WEDGE POSTMORTEM (worst class; box now needs power reset again)
- Trigger pattern: concurrent GPU access DURING a fleet TP8 boot — agent-23's
  single-GPU3 bench collided with the GLM fleet boot; GPU3 hung at device-open,
  then EVERY device open hangs (global KFD wedge): GPU0 torch.zeros hangs too.
- APIServer pid (mtp2) went D-state (uninterruptible) at enginespawn; SIGTERM does
  nothing (240s+); VRAM actually free (0.5GB); nothing recovers it without a reset.
- HARD RULE added: while ANY vllm fleet process is booting/running, NO other process
  may touch ANY /dev/kfd GPU (no offline benches, no rocm-smi side traffic beyond
  the known-safe --showmeminfo vram / --showuse with default opts; rocm-smi
  --showproduct is ALREADY banned — it hung even pre-wedge when zombies existed).
  Offline GPU work ONLY in windows where the fleet is fully drained.
- Recovery: user runs `sudo ipmitool chassis power reset` (no passwordless sudo here).
- Pre-flight guard created: /data/vllm-gfx906-dsv4/fleet_free.sh — every offline
  GPU bench / serve launch MUST run it first (refuses while any vllm proc, any
  D-state proc, or >2GB VRAM held). Wire it into every future bench boot; KIMI
  and agents both.

### P2 router kernel shelved (2026-09-11 ev)
VLLM_GLM53_MOE_ROUTER_FUSED=1 faults at cudagraph capture EVEN WITH the
strided-view fix (md5 1dc18a54, bisected: mhc-only boot is clean). Capture
smoke passed offline, so the fault is in-model tensor-layout-specific.
Value ~3-4 ms/step vs mHC's ~50 — SHELVED pending capture-side dump.

### P0 fused mHC SHIPPED (2026-09-11 ev, second window)
- Distributional gate built: glm53_logprob_ab.py (12 fixed prompts, per-prompt
  mean prompt-logprob; PASS median|Δ| 0.0074, max 0.052/12). Single-greedy-probe
  equality RETIRED as a gate on TP8: NCCL reductions are nondeterministic across
  boots → near-tie argmax flips ('UK falls everywhere' / 'UK falls on the hills' /
  'Netherlands falls on everyone' all seen from DIFFERENT boots of the SAME
  binary config; use logprob distributions, not token equality).
- mHC fused in-server: 6.17 -> 11.66 tok/s (bs1, 512-tok, 3x consistent).
  Correct by distributional PASS + bitwise offline + coherent creative probes.
- MTP k=1 verdict: acceptance 83.5% BUT no speed gain (11.63 vs 11.66) — draft
  plumbing (TP8 BNB proposer + verify) ≈ one target step; 2 steps/1.84 tok =
  wash. PARKED (Hy3 MTP precedent repeats). Patch stack kept (glm53_mtp_patch.py)
  for revisit if draft-step plumbing gets cheaper.

### Boot-hang pattern #3 (2026-09-11 ev): in-process torch profiler + fused triton JIT
- Boot with VLLM_GFX906_PROF_DIR set + VLLM_GLM53_MHC_FUSED=1 hung at the
  profiling/dummy-run phase (post weight load, pre KV sizing): >20 min of
  shm_broadcast stalls, no GPU work. Exonerated separately: profiler-only boots OK
  (17:0x stable prof boot, 18:22 window fine), fused-only boots OK. The COMBINATION
  (or prof wrap + cold JIT of the fused kernels inside execute_model) hangs.
  RULE: do not set VLLM_GFX906_PROF_DIR on boots that have GLM53 fused kernels on.
- Post-P0 budget (analytic, corroborated by 11.66 tok/s): NCCL ~42 ms is now ~48%
  of the 86ms step; MoE wna16 26 ms (~950 GB/s = near HBM2 roofline, irreducible);
  rest ~18 ms. Next lever: RCCL transport (see RCCL_TUNING_PLAN.md by agent-25 —
  HSA_FORCE_FINE_GRAIN_PCIE=1 P2P experiment top; NCCL_ALGO=Tree runner-up).

### 2026-09-11 late — RCCL A/Bs parity; 32k ladder found long-ctx bug; final numbers
- HSA_FORCE_FINE_GRAIN_PCIE=1: no effect (11.57-11.58 vs 11.66). P2P did not engage.
- NCCL_ALGO=Tree (+CTAS 1-2, NTHREADS 128): no effect (11.54-11.60). RCCL at floor
  on PCIe-only without P2P links (SHM ring 14 hops). RCCL_TUNING_PLAN.md stands.
- 32k-ctx boot works for short ctx (11.35 tok/s, KV 332 blocks, 1.93x conc @32768)
  BUT a 17.9k-token prompt all-rank Memory-faults mid-request (hybrid kpool sparse
  prefill beyond 8k never exercised before). 32k SERVICING SHELVED until the
  kpool/long-prefill path gets a fix pass; 8k remains the verified envelope.
- FINAL SHIP: run_glm53.sh @ 8k, GEMV=1, MHC_FUSED=1, ROUTER=0: 11.6-11.7 tok/s bs1,
  prefill ~81 tok/s@4.6k, KV 63k tokens (7.8x conc @8k), bench samples GOOD.
