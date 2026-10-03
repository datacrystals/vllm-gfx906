# Quality hypothesis: router-topk nondeterminism (2026-10-03)

Observations so far (mimo_cacheab + mimo_qprobe baseline):
- fresh-vs-fresh disagrees at T=0 on short prompts (nondeterministic GREEDY).
- warm-vs-fresh diverges at the SAME token position (~tok 5) on 2/4 prompts;
  warm side degrades into degenerate question-loop text (the user's
  "doesn't know what it is doing").
- 8/8 single-turn prose probes: 0 stray-dot artifacts. The '.' bug needs
  cached/multi-turn context (or is the visible tail of near-tie flips).

HYPOTHESIS: the MoE router top-k (256 experts, sigmoid grouped topk) is fed
by the router-gate GEMM. Decode was recently made fast by routing that GEMM
through the fp16 skinny path (VLLM_MIMO_GATE_FP16_GEMV=1, 41ms->1ms).
Split-K/small-M GEMMs on gfx906 commonly use atomics -> nondeterministic
sum order -> near-tie logits flip between runs -> different experts routed
-> token-level wobble. With prefix-cached KV computed under one routing
decision and continuation under another, the result is exactly the observed
incoherence (warm runs degenerate, fresh runs coherent).

ALSO CONSIDER: gatherTopK in bf16 (ties), moe_wna16 gather/scatter atomics,
nccl all-reduce ordering under different arrival patterns.

DECISIVE TESTS (agent-20's H3 path):
1. A/B VLLM_MIMO_GATE_FP16_GEMV=1 vs =0: if =0 is deterministic and
   artifact-free, the gate fix traded quality for speed -> need a
   deterministic fp16 path (or fp32 accumulate in the skinny GEMM).
2. Run same prompt 10x at T=0 and count unique outputs; compare variance
   with prefix cache off.
3. If both noisy: instrument top-k logits pre/post for tie flips.

If =0 is clean but slow, options: fp32-accumulate fp16 GEMM (keep speed,
kill reordering), or deterministic split-K reduction for the gate only
(1 GEMM/layer-step is cheap to make deterministic).

## MEASURED CONFIRMATION (mimo_nondet_hammer, 2026-10-03 21:18 UTC)
n=10 identical greedy (T=0) runs of the same short prompt:
  distinct token streams: 10 of 10
  first-divergence offsets vs run0: [25, 17, 11, 17, 25, 46, 18, 11, 11]
  max |dlogprob| on common tokens: 1.88 nats (~6.5x probability swing)
  runs even disagree on what the prompt asked (diagram vs Python program)
VERDICT: forward-pass numerical NONDETERMINISM confirmed. Not near-tie
sampling noise -- identical prefixes carry 1.88-nat logprob drift, so the
compute itself is non-reproducible run to run. This single root cause
explains all user symptoms: stray '.' tokens (punctuation flips),
"doesn't know what it is doing" (flips interpretation of the same prompt),
and cache warm/cold divergence (cached KV computed under one numerical
realization, continuation under another).
FIX TARGET: locate the nondeterministic kernel in the hot path. Ranked:
(1) MoE gather/scatter atomics (256-expert grouped GEMM),
(2) fp16 gate GEMM split-K (VLLM_MIMO_GATE_FP16_GEMV path),
(3) attention reduction order (unified_attention), (4) nccl all-reduce
ordering. Verification bar: hammer must go 10/10 -> 1/1 distinct streams
AND cacheab warm-vs-cold must go byte-identical, with multi-turn + behavior
probes staying PASS.

## DIAGNOSIS CLOSED (2026-10-03 21:5x, nondet_hammer2 on cache-OFF boot)
Boot with enable_prefix_caching=False (verified in engine args):
  # distinct streams: 8 of 8 greedy runs
CONCLUSION: forward-pass compute nondeterminism is the ROOT CAUSE. The
prefix cache is NOT the source -- it is an AMPLIFIER (frozen KV from one
numerical realization vs continuation under another -> warm/cold splits
and degenerate loops in multi-turn). Disabling cache = symptom relief
only; the cure is deterministic kernels.
FIX SCOPE (final): kernel-level. Ranked suspects unchanged:
  1. MoE gather/scatter atomics (256-expert grouped GEMM, index_add family)
  2. fp16 gate GEMM split-K (VLLM_MIMO_GATE_FP16_GEMV path)
  3. attention reduction order (triton_unified_attention)
  4. nccl all-reduce arrival-order effects
VERIFICATION BAR (unchanged): hammer 8/8 -> 1/1 distinct on the same boot
config, warm==cold byte-identical in cacheab, multi-turn + behavior probes
green, needle 24k + France probe + thinkstrip pass.

## TWO BUGS, NOT ONE (2026-10-03 noprefix suite verdict)
Frozen-history probe (notes-state, 2 runs x 5 turns):
  cache ON  baseline: 4 of 5 turns DIVERGED (warm != cold)
  cache OFF now:      0 of 5 diverged -- ALL turns byte-IDENTICAL
  VERDICT line: "prefix cache consistent (turns deterministic)"
MEANING: on deterministic trajectories the cache is the SOLE divergence
source -- a genuine KV-correctness bug (SWA block accounting serving
subtly-wrong cached KV), not merely noise amplification.
Simultaneously the hammer shows 8/8 distinct streams cache-OFF on
near-tie-heavy prompts (short_explain) -> separate compute nondeterminism.
FINAL DECOMPOSITION:
  Bug A: prefix-cache KV mismatch -> multi-turn context rot (the user's
    "doesn't know what it is doing"). Fix in v1/core SWA block accounting.
    Verified measurable: 4/5 -> 0/5 diverging turns by disabling cache.
  Bug B: kernel compute nondeterminism -> stray dots + wobble on
    near-tie-heavy generations (hammer 8/8 cache-off). Kernel-level
    (atomics/split-K/reduction); harder fix.
FIX ORDER: Bug A first (clear target, big multi-turn win), then Bug B.
If Bug B resists, documented mitigation: temperature slightly >0 makes
near-tie flips intentional sampling rather than silent corruption.

## H3 VERDICT (gate0 run): gate GEMM EXONERATED, Bug B is deeper
Boot: cache OFF + VLLM_MIMO_GATE_FP16_GEMV=0 (both suspects disabled):
  # distinct streams: 8 of 8
  global max|dlogprob| on shared prefixes = 0.239
The fp16 skinny gate GEMM is NOT the nondeterminism source -- the 28.07
tok/s decode fix is SAFE. Bug B is in the MoE scatter/gather atomics
(256-expert grouped GEMM), attention reductions, or NCCL ordering.
KEY DETAIL: run6 top-3 shows an EXACT float tie (-0.7732484340667725 for
both ' is' and ' cannot'). INT4 quantization creates logit plateaus where
argmax between tied tokens is decided by reduction order -> the stray '.'
mechanism. run5 diverged at token 0 with zero shared-prefix drift (the
first forward pass differs run to run).
FIX ROUTING (final): Bug A = SWA prefix-cache block accounting (clear
target). Bug B = MoE kernel determinism (rank-1 suspect: scatter_add
atomics) or, if unfixable at acceptable cost, the documented mitigation
(small temperature makes tied-token flips intentional sampling).
