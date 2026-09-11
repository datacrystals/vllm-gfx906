# RCCL/NCCL small-message all-reduce tuning plan — 8× MI50 (gfx906), PCIe-only, ROCm 6.4.x

Date: 2026-09-11 · Status: plan only — **no GPU was touched while writing this.**
Companion bench: `patches/gdn/bench_rccl_smallmsg.py` (run later, drained-fleet window).

## 0. Problem statement and verified ground truth

Decode step today (GLM-5.3 profile, `glm53-prof/TOP_KERNELS.txt` + per-step slice):
**92 all-reduces of ~8 KB fp16 ([B,4096], B≈1–2) per step, avg 456 µs/call ⇒ ~42 ms of an
86 ms step is `ncclDevKernel_Generic_4(ncclDevComm*, channelMasks, ncclWork*)`.**
Same story in the Qwen3.5 profile (NCCL ≈ 30% of decode). Everything below was verified
offline on 2026-09-11:

| Fact | Evidence |
|---|---|
| No XGMI, no GPU↔GPU P2P links at all | `/sys/class/kfd/kfd/topology/nodes/*/io_links`: GPUs (nodes 2–9) have **only type=2 (PCIe) links to the 2 CPU nodes**; zero GPU↔GPU links |
| 2 CPU sockets; GPUs 0–3 (BDF 1c/1f/3f/42) on NUMA0, 4–7 (8a/8d/b3/b6) on NUMA1, each group behind an AMD-switch downstream port | `/sys/bus/pci/devices/*` |
| Large BAR **is** enabled: each GPU's BAR0 maps the full 32 GiB | sysfs `resource0` size 0x800000000, kernel cmdline `iommu=pt` |
| Production all-reduce path = vLLM `PyNcclCommunicator` (ctypes `CDLL("librccl.so.1")`) | `vllm/distributed/device_communicators/cuda_communicator.py:all_reduce()` → `pynccl_comm.all_reduce()` |
| **`librccl.so.1` SONAME resolves to the torch-bundled library for BOTH torch PG and pynccl**: torch bundles `torch/lib/librccl.so` (845 MB, RCCL "60302" = NCCL **2.21.5**-based, ROCm 6.3.2-era, has gfx906 code objects). Its SONAME is `librccl.so.1`, so pynccl's `CDLL("librccl.so.1")` matches the already-loaded bundled copy. | `readelf -d` on both libs; `ldd libtorch_hip.so` → `$ORIGIN/librccl.so` |
| A **newer** system RCCL exists: `/opt/rocm-6.4.3/lib/librccl.so.1.0.60403` (NCCL **2.22.3**-based, gfx906 built-in, on the default `ld.so` path) but it is NOT what runs today. NB: `LD_PRELOAD`/`LD_LIBRARY_PATH` cannot swap the torch PG path (DT_NEEDED is `librccl.so`, RPATH=`$ORIGIN` wins; SONAME mismatch). Only `VLLM_NCCL_SO_PATH=<abs path>` swaps the pynccl path (both RCCL copies then coexist in-process; supported pattern, rccl symbols are RTLD_LOCAL). | readelf + `/etc/ld.so.conf.d/*rocm*` |
| vLLM CustomAllreduce **can never run here**: gate is `world>2 and !is_fully_connected(physical_ids)` → disabled; `is_fully_connected` on ROCm = **XGMI-1-hop only** (amdsmi link type) — PCIe does not count. Code throws "not supported on more than two PCIe-only GPUs". | `custom_all_reduce.py:151`, `platforms/rocm.py:696` |
| QuickAllReduce: **gfx94/gfx95 (MI300) only** (`quick_all_reduce.py:229`). Symm-mem / FlashInfer AR: CUDA-only. So all four in-tree fast-AR alternates are out. | inspected files above |
| RCCL tuning tables ship **PCI `hwLat` = Tree-LL 2.2 µs / Ring-LL 2.2 µs / Simple 5.7 µs** (all models) → tuner prefers **LL** for tiny payloads; LL128 only where `ll128Enabled` | `src/graph/tuning.cc` @ rocm-6.4.3 |
| `ll128Enabled` defaults **false** and is only set by recognized static "Rome models" (MI100/MI300) or `RCCL_LL128_FORCE_ENABLE=1`. Changelog 6.4.2: *"Added support for the LL128 protocol on gfx942"* — i.e. **LL128 on gfx906 is not a supported config**. NCCL docs: forcing LL128 on unsupported platforms "can lead to data corruption" → bench with correctness gate only. | `src/init.cc:1167,1358`; RCCL CHANGELOG; NCCL env docs |
| MSCCL++ tunables exist in the binary but RCCL 2.21.5 changelog scopes MSCCL++ AllReduce/AllGather to **gfx942**; `RCCL_MSCCLPP_ENABLE` defaults 0 → on gfx906 expect a no-op (or failure) — don't invest. | `src/init.cc:121`, RCCL CHANGELOG 6.3.0 |
| Prior campaign data: 2-rank micro-bench (3 KB) baseline 53 µs → Simple 44 µs, but full TP8 A/B with `NCCL_PROTO=Simple` **regressed bs1 −12%** (24.14 vs 27.58 tok/s) — micro-bench wins do **not** automatically transfer; every candidate must pass the full-model gate (distributional logprob A/B, not single greedy probe). | `EXPERIMENTS.md` Track P1 |

Working model of the 456 µs: with no P2P/XGMI, RCCL uses the **SHM transport** (GPU kernels
write/read pinned-host FIFOs over PCIe; CPU does or assists the inter-buffer copy). An 8-rank
ring = 2·(8−1)=14 hops ⇒ ≈32 µs/hop, matching the 2-rank ~45–53 µs measurement. A tree =
2·⌈log₂8⌉ = 6 hops ⇒ ≈190–230 µs at the same per-hop cost. That hop count is the #1 lever.
`NCCL_ALGO`/`NCCL_PROTO` forcing at **TP8 was never measured** (only 2-rank was).

## 1. Ranked, testable knobs

Each row = one A/B. Expected effect is on the 456 µs figure (device-side, in-graph), from the
hop model + RCCL tuning tables — all must be validated with `bench_rccl_smallmsg.py`
(world=4 both socket-local and cross-socket, then world=8) and a full-model A/B.

| # | Env (values) | Mechanism | Expected vs 456 µs | Risk |
|---|---|---|---|---|
| 1 | `NCCL_ALGO=Tree` | 14→6 transport hops over SHM; tree depth ⌈log2 8⌉; PCIe `hwLat` table treats Tree-LL == Ring-LL (2.2 µs), so this is a pure hop-count play | **−35…−50 %** (→ ~230–290 µs) ⇒ step 42→~21–27 ms | Low-moderate: 2-rank control showed Tree slightly worse (echo of #4 below); MUST be validated at 8 ranks; per-hop imbalance (cross-UPI hops) decides |
| 2 | `NCCL_MIN_CTAS=1 NCCL_MAX_CTAS=1` (legacy alias: `NCCL_MIN/MAX_NCHANNELS=1`) | 8 KB needs exactly one channel; any extra CTAs add launch/tail + FIFO overhead per hop inside the Generic_4 kernel. NCCL docs: for very small messages NCCL may use fewer than min, but capping the top prevents tuner over-allocation | −0…−20 % | Low: can only throttle large-message bandwidth (prefill/logits); keep per-size data from the bench |
| 3 | `NCCL_NTHREADS=128` (also try 64, default-on-gfx906 likely 256/512) | Fewer threads per comm block = fewer partial-line flag round-trips per hop; classic small-message knob | −0…−15 % | Low |
| 4 | `HSA_FORCE_FINE_GRAIN_PCIE=1` (RCCL docs "Enabling peer-to-peer transport" — needs PCIe-P2P-capable GPUs + large BAR ⇒ **this box has large BAR** + `iommu=pt`) | Enables KFD fine-grained PCIe mappings → GPU-GPU P2P transport may appear → RCCL switches hops from SHM (CPU staged) to P2P (GPU direct): per-hop ~30 µs → ~5–10 µs | **−60…−85 %** (→ 70–180 µs) — the biggest single lever *if* it engages | **High**: unknown on gfx906 behind these AMD switch ports; fine-grain PCIe atomics can be very slow; verify hipDeviceCanAccessPeer + NCCL `via P2P`/`via SHM` lines in the diag run; correctness-gate in bench |
| 5 | combo of winners, e.g. `NCCL_ALGO=Tree NCCL_MIN/MAX_CTAS=2 NCCL_NTHREADS=128` | Multiplicative on different terms | −40…−60 % total vs baseline | Low if each component validated |
| 6 | `VLLM_NCCL_SO_PATH=/opt/rocm-6.4.3/lib/librccl.so.1` (serve-time env) | Swaps the *pynccl* all-reduce from bundled RCCL 2.21.5 → 2.22.3 (proxy-thread hang fix, `RCCL_OUTPUT_TREES`/`NCCL_RUNTIME_CONNECT` diag, misc tuning deltas). torch-PG collectives stay on 2.21.5 (RPATH, see facts). | −0…−10 %; mainly a diagnostics/freshness play | Medium: two RCCL builds coexist in-process (supported but verify at TP8); startup change only |
| 7 | `NCCL_PROTO=LL` | Pin the protocol the PCI tuning table already implies (LL 2.2 vs Simple 5.7 µs) — insurance that LL is what runs; harmless to keep | −0…−10 % | Very low. **Never ship `=Simple`** (proven −12 % bs1 at TP8) |
| 8 | `RCCL_LL128_FORCE_ENABLE=1` + `NCCL_PROTO=LL,LL128` | LL128 moves 120 B/flag-line vs LL's 8 B ⇒ fewer flag round-trips per byte — *if* functional | potentially −30…−50 % within-protocol, but… | **High**: LL128 supported only on gfx942 (RCCL 6.4.2 changelog); gfx906 force = unsupported, NCCL docs warn of *data corruption*. Only with the bench's CORRUPT gate + logprob A/B. Expect rejection. |
| 9 | `NCCL_SHM_USE_CUDA_MEMCPY=1` | Uses GPU copy engine for the SHM inter-buffer hop instead of CPU memcpy — cheaper or more overlappable per hop | −0…−20 % | Medium-low; present in both libs; correctness-gate |
| 10 | `HSA_NO_SCRATCH_RECLAIM=1` (+ optionally `HIP_FORCE_DEV_KERNARG=1`) | ROCm-runtime per-launch overhead cuts. Documented for MI200 on ROCm 7.13 ("restores expected small-message latency"); pre-7 behavior on gfx906 unknown → measure. Helps every tiny kernel, not only RCCL — note glm53's eager tails. | −0… few % of step (not only NCCL) | Low (slightly higher scratch memory footprint) |
| 11 | `RCCL_MSCCL_ENABLE=0` | Stability only: ROCm 6.4.1 known issue — MSCCL can segfault on `ncclCommSplit`; kills an experimental path that gfx906 can't benefit from | ~0 (perf); removes a hang class | None expected |
| 12 | diagnostics: `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,GRAPH` + (with 2.22.3) `RCCL_OUTPUT_TREES=1`, or `NCCL_TOPO_DUMP_FILE=`/`NCCL_GRAPH_DUMP_FILE=` | Ground truth: confirms SHM-vs-P2P per link, ring order, tree shape, proto/algo/nChannels chosen per size — settles every "probably" in this table. | n/a | Noise in logs; run once per config family |

**Deliberately skipped / not applicable** (verified against the actual binaries with `strings`):
`NCCL_IB_*`, `NCCL_NET_*`, `NCCL_PXN*`, `NCCL_SOCKET_*` (no net transport except bootstrap),
`NCCL_P2P_*` (no P2P transport today; revisit only if #4 works), `NVLS*`/`CollNet*`/`PAT`
(hardware absent), `RCCL_CLIQUE_*` (clique kernels removed in ROCm 5.2.3), `RCCL_MSCCLPP_*`
(gfx942-only; `defaultEnableMscclpp=0`; MSCCL removed in RCCL 2.30), `NCCL_SINGLE_RING_THRESHOLD`
(not in this build), `NCCL_BUFFSIZE`/`NCCL_LL_BUFFSIZE` (FIFO depth; 8 KB fits trivially),
`NCCL_GDRCOPY_*` (no GDR path), `NCCL_CGA_CLUSTER_SIZE` (sm90 only), `NCCL_IGNORE_CPU_AFFINITY`
(only if a launcher badly mis-pins ranks; defaults are sane here), `NCCL_NCHANNELS_PER_NET_PEER`
(net-only).

## 2. vLLM-side alternates for tiny-payload all-reduce — verdict

- **CustomAllreduce: permanently unavailable.** Two independent gates fail:
  `is_fully_connected()` requires XGMI (`platforms/rocm.py:696`), and the two-stage kernel
  itself needs real P2P IPC buffers. Even if #4 (`HSA_FORCE_FINE_GRAIN_PCIE=1`) lights up P2P,
  the XGMI gate stays; enabling it would require patching that check AND validating the
  custom kernel under PCIe-P2P (latency ~ P2P writes + flag spins; plausible win but a code
  change, not an env).
- **QuickAllReduce**: compiled for gfx94x/gfx95 only (explicit arch check) — dead on gfx906.
- **torch symmetric-memory AR / FlashInfer AR / zero-CU CE collectives**: CUDA-only or
  ROCm ≥ 7.12 + P2P preconditions — unavailable here.
- **CPU/SHM all-reduce fallback (e.g., gloo over `shm_broadcast.MessageQueue`)**: exists for CPU
  tensors only. Feasibility math: D2H(8 KB)+H2D ≈ 6–10 µs + 8-rank SHM tree ~40–80 µs → maybe
  ~100 µs/call, *but it cannot be captured in a cudagraph* (host-side op), while FULL graphs on
  this rig are worth 2.4× end-to-end (glm53: 2.6→6.2 tok/s). **Net loss — rejected** for the TP
  decode path. (Only reconsider if decode ever moves out of graphs.)
- **Realistic vLLM-side levers that remain**: `VLLM_NCCL_SO_PATH` (#6), keeping
  `--disable-custom-all-reduce` to skip futile init (hygiene), all the `NCCL_*`/`RCCL_*` envs
  in §1 (propagated to worker procs by the run scripts), and — orthogonal but larger — reducing
  the *count* of all-reduces or their exposure (TP degree, overlap) as separate projects.

## 3. Bench harness — `patches/gdn/bench_rccl_smallmsg.py`

Runbook (drained-fleet window only; box rules from EXPERIMENTS.md apply: `fleet_free.sh` guard
runs automatically, SIGTERM-only, never bench while a fleet server is up):

```bash
cd /data/vllm-gfx906-dsv4/patches/gdn
PY=/data/vllm-gfx906-dsv4/vllm_dsv4_env/bin/python

# 1-GPU launch/copy floor ("what if comms were free") — any single device:
HIP_VISIBLE_DEVICES=0 $PY bench_rccl_smallmsg.py --single

# Order of operations (each ~3-5 min; aborts on hang >420 s):
HIP_VISIBLE_DEVICES=0,1,2,3   $PY bench_rccl_smallmsg.py --diag      # see algo/proto/transport
HIP_VISIBLE_DEVICES=0,1,2,3   $PY bench_rccl_smallmsg.py             # full combo matrix, numa0
HIP_VISIBLE_DEVICES=4,5,6,7   $PY bench_rccl_smallmsg.py --combos baseline,tree,ll,ctas1
HIP_VISIBLE_DEVICES=0,2,4,6   $PY bench_rccl_smallmsg.py --combos baseline,tree,p2p_pcie
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY bench_rccl_smallmsg.py       # production TP8 shape
# ad-hoc p2p retry + any custom combo:
HIP_VISIBLE_DEVICES=0,1,2,3 $PY bench_rccl_smallmsg.py --env 'HSA_FORCE_FINE_GRAIN_PCIE=1,NCCL_ALGO=Tree'
```

What it measures (worker code, per payload N∈{1,2,4}×4096 fp16 ⇒ 8/16/32 KB):
correctness gate (`OK|CORRUPT` — mandatory for the `ll128` and `p2p_*` rows), 100 eager-warm
calls, then **torch cudagraph capture of 32 all-reduces replayed 50×** with CUDA events
(matches production FULL-graph replay; eager numbers printed alongside for reference), and a
separate **92-call burst graph** = the production per-step comms figure (42 ms today).
Orchestrator parses `RESULT`/`STEP` lines, prints a sorted table per payload size, and saves
raw rows to `/data/tmp/bench_rccl_smallmsg_<gpus>.json`. Env combos are table-driven in
`COMBOS` and overridable with `--combos` / `--env`.

Acceptance gates before any production A/B: no `CORRUPT`, no hangs, TP8 gain ≥10 % vs
baseline at 8–32 KB, and the 92-call burst improves more than eager (i.e., the win is in the
transport, not launch noise). Full-model gate per house rules: distributional logprob A/B
(`glm53_logprob_ab.py`) + bench_write suite — token-equality probes reject valid configs.

## 4. Expected outcome (exec summary)

The 456 µs/call is dominated by 14 SHM ring hops. Best ranked line:
if `HSA_FORCE_FINE_GRAIN_PCIE=1` engages PCIe P2P (large BAR + `iommu=pt` say it might),
expect 456→~70–180 µs ⇒ **save ~26–35 ms/step**. If P2P stays off, `NCCL_ALGO=Tree` +
CTA/thread trimming is the realistic champion at 456→~230–290 µs ⇒ **save ~15–20 ms/step**.
That maps to ≈ +18–40 % on the 86 ms/step decode rate (e.g., 11.7 → ~14–16 tok/s class).
Conservative expectation if the tuner was already optimal: ~0–5 ms (then NCCL is a hardware
floor and effort should move to hop-count-independent work — fewer/larger all-reduces).
Non-goals already ruled out upstream of the bench: vLLM custom/quick/symm AR, MSCCL++,
LL128 default-on, SHM/CPU fallbacks (not graph-capturable).

## 5. Risks & rollback

- Every knob is an env var set **only** in the bench children / a test run-script copy —
  rollback = unset. Keep one config change per A/B; identical env on all ranks of a run
  (asymmetric RCCL envs historically hang).
- `ll128` and `p2p_*` rows can corrupt or hang: bench gate exists; if the full-model logprob
  A/B shows anything beyond NCCL-order noise, drop the knob.
- `VLLM_NCCL_SO_PATH` loads a second RCCL into each worker (≈ +memory, longer init);
  abort boot on any new warning class. torch-PG path stays 2.21.5 regardless.
- Bump `VLLM_ENGINE_READY_TIMEOUT_S` for cold-cache/boot experiments (established lesson).

## Sources

- [NCCL 2.31 env variables (NVIDIA docs)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) — semantics for PROTO/ALGO/NTHREADS/CTAS/SHM/BUFFSIZE + LL128-corruption warning
- [RCCL CHANGELOG (rocm-6.4.3 tag)](https://raw.githubusercontent.com/ROCm/rccl/rocm-6.4.3/CHANGELOG.md) — LL128 gfx942-only, rail trees, MSCCL comm-split issue + `RCCL_MSCCL_ENABLE=0` workaround
- [RCCL usage tips (latest ROCm docs)](https://rocm.docs.amd.com/projects/rccl/en/latest/how-to/rccl-usage-tips.html) — `HSA_FORCE_FINE_GRAIN_PCIE=1` PCIe-P2P recipe, `HSA_NO_SCRATCH_RECLAIM`, `NCCL_IGNORE_CPU_AFFINITY`, MSCCL removal
- RCCL 6.4.3 source (fetched, tagged `rocm-6.4.3`): `src/init.cc` (env params, `ll128Enabled=false` default, MSCCLPP default-off), `src/graph/tuning.cc` (PCI `hwLat` tables = Tree/Ring-LL 2.2 µs vs Simple 5.7 µs; per-coll LL ranges only for MI300 models), `src/graph/rome_models.cc` (static models — no gfx906 PCIe entry)
- This box: kfd topology + PCI sysfs reads, `strings`/`readelf` on both librccl builds (env-var lists, SONAME, gfx906 targets), vLLM fork source (`cuda_communicator.py`, `custom_all_reduce.py`, `quick_all_reduce.py`, `pynccl_wrapper.py`, `platforms/rocm.py`), `EXPERIMENTS.md` Track P1 numbers.
