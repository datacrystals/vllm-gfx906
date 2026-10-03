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
