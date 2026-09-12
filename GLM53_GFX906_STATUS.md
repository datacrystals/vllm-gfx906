# GLM-5.3-Flash on 8×MI50 (gfx906) — campaign status

Branch: `glm53-gfx906` @ /data/vllm-gfx906-dsv4/vllm (fork: github.com/ai-infos/vllm-gfx906-mobydick)
This doc is the resume point. Read EXPERIMENTS.md chronology for full history; KDA_TAIL_PLAN.md
(agent-23 kernel engineering) and MTP53_PLAN.md (agent-24 MTP glue) for the deep dives.

## Final shipped numbers (2026-09-11 late, branch glm53-gfx906)
- Ship config (run_glm53.sh, 8k ctx, util 0.94, FULL graphs, fp16):
  GEMV=1, MHC_FUSED=1, ROUTER_FUSED=0, MTP off.
- bench_write.py (same protocol as the Hy3/Qwen numbers): 500-tok end-to-end
  11.48 tok/s; 8k-ctx 8.76; creative 700-tok samples **15.22-15.29 tok/s**
  (decode-rate after warmup — the >15 goal is met on the samples protocol;
  honest end-to-end for mid-size requests is ~11.5-11.9).
- Quality: 4 writing samples in /data/llmbench/glm53/final-8k-ship/*.txt;
  thinking-block leak mitigated client-side (clear_thinking or template kwarg).
- 32k variant (run_glm53_32k.sh) boots fine; a serve-wide memfault was observed
  once *during a long-prefill burst while a bench ran* — treat 32k as
  experimental; 8k is the verified envelope.

## 2026-09-12 — 32k context default; long-ctx memfault FIXED
- Root cause: group-0 indexer KV group ran kernel-split block ids
  (manager-256 → kernel-64 ids) into a MANAGER-granular cache view; any
  recycled block id >= num_blocks/4 => all-rank memfault. Core DRAM find.
- Fix: Glm5NextROCmIndexerBackend (accepts 256; env VLLM_GLM53_INDEXER_UNSPLIT,
  default 1/ON; `=0` reverts) + cp_gather_indexer_k_cache_fp16 row-stride fix
  (csrc rebuilt; venv _C.abi3.so.bak-longctx kept).
- VALIDATED: ladder 5.2k..24k tokens + 2×16k concurrent, all alive; speed parity
  (11.6 tok/s at 8k/32k configs identical). run_glm53.sh = 32k default.
  run_glm53_8k.sh rollback. glm53.service auto-starts the 32k config.

## Verified works (on a clean box)
- Serving: `/data/vllm-gfx906-dsv4/run_glm53.sh` (TP8, 8k ctx, util 0.94, fp16,
  block 256, max-num-seqs 2, FULL cudagraphs mode-0, no prefix caching,
  CT pack-quant int4 g32 asym MoE + offline bf16-fix weights).
  Boot ≈ 6-7 min (227 s weight load + profiling + graph capture).
- Throughput (measured, greedy gate ` plain. But in the UK...` pass):
  - decode bs1 ~6.2 tok/s @ 512-tok; bench_write: 6.8-7.0 tok/s short prompts,
    4.8 tok/s at 4.6k ctx; prefill ~74 tok/s @ >4k prompt.
  - KV: ~344 blocks ≈ 63k effective tokens (7.8× max-len concurrency @ 8192).
    **If a boot prints `GPU KV cache size: 40 tokens` (util too low for
    weights) the next request memory-faults all ranks — check the line!**
- Writing quality: samples in /data/llmbench/glm53/baseline/*.txt are strong;
  thinking leaks into content unless chat client passes
  `chat_template_kwargs={"clear_thinking": true}` (template defaults
  reasoning_effort=max) — or serve with a reasoning parser.

## Kernel/opts inventory (env-gated; OFF = safe defaults)
- VLLM_GFX906_GEMV=1 — gfx906 skinny fp16 GEMV (microbench-verified; GLM
  decode showed no gain — prefill-side; keep on for Hy3).
- VLLM_GLM53_MHC_FUSED=1 — fused mHC pre/post Triton (11.9x/1.4x offline,
  graph==eager bitwise repro, fp32-floor parity vs real ckpt params;
  in-server 6.2 → 11.7 tok/s measured!). GATE SHIP DECISION PENDING a
  distributional logprob A/B (battery computes it) — drift is fp32
  reduction-order through fp16 rounding, expected benign.
- VLLM_GLM53_MOE_ROUTER_FUSED=1 — fused sigmoid-topk-8 router (fp16 bitwise
  vs torch; OOB fix applied for strided views; capture smoke passes on GPU).
- VLLM_GLM53_SPARSE_DECODE_ATTN=1 — sparse-MLA decode kernel v2, UNBENCHED —
  do not enable until bench_glm53_sparse_mla_decode shows < torch ref.
- VLLM_DSV4_MHC_TRITON=1 — mhc_triton (DSV4 A2): works offline 10.5x, HANGS
  in TP8 boot at first profiling forward if ~/.triton/cache has stale locks
  (wipe it) — AND wedges workers on first real prefill (shape-specific);
  parked, superseded by VLLM_GLM53_MHC_FUSED anyway.

## MTP (speculative) status
- Glue: patches/gdn/glm53_mtp_patch.py + driver patches/gdn/glm53_mtp_main.py,
  script run_glm53_mtp.sh, plan MTP53_PLAN.md. Gate env VLLM_GFX906_GLM53_MTP=1.
- Draft arch resolves natively (Glm5NextMTPModel) even without patch.
- Biggest known risk: spec decode downgrades decode cudagraphs to NONE unless
  VLLM_GFX906_GLM53_MTP_FULL_CG=1 (backend override) — performance hinge.
- NOTE: VLLM_ENGINE_READY_TIMEOUT_S default 600 is too short right after a
  triton-cache wipe; run_glm53_mtp.sh sets 1800.

## Where we were when the box wedged (2026-09-11 ~15:4x)
- Second wedge: agent-23's GPU3 offline bench DURING a fleet TP8 boot → global
  KFD wedge (all device opens hang). **fleet_free.sh must gate every GPU use
  from now on; no offline GPU work while any vllm server is booting/running.**
- Next after reset: `battery_glm53.sh` (stable → fused → MTP, autologged),
  then ctx ladder (32k first: KV 63k tokens covers 32k x1-2 concurrent),
  then push branch (origin push needs creds — 403 from box identity).

## Remaining blockers on >15 tok/s path
1. mHC P0 on (→ ~10.5) — pending distributional A/B gate.
2. Router P2 fixed copy (→ ~11.5) — pending one capture boot.
3. MTP k=1 (→ 12-18 if acceptance ≥ ~0.5 under FULL_CG; else eager-MTP is a loss).
4. moe_wna16 26 ms/step and NCCL 42 ms/step — separate tracks (unscoped here).

## Recovery drills (learned twice this session)
- Zombie-pinned VRAM (mem never frees, `% used` frozen): power reset.
- Global device-open wedge (torch.zeros hangs everywhere, D-state engine procs,
  load avg >100 from D pile): power reset. SIGTERM/SIGKILL do not help.
- Both cleared cleanly by `sudo ipmitool chassis power reset` (user-side; no
  passwordless sudo on box).
