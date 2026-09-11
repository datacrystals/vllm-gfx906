# GLM-5.3-Flash MTP (speculative decoding) enablement plan

Date: 2026-09-11. Static analysis only — no GPU runs performed. Patch written to
`patches/gdn/glm53_mtp_patch.py`; no vLLM tree files were modified.

## Feasibility verdict

Feasible with glue. The vendored `Glm5NextMTP` draft model is complete and registered
(`Glm5NextMTPModel`, `vllm/model_executor/models/registry.py:604`), the checkpoint ships
full layer-45 MTP weights (889 bf16 tensors, all CT-ignored), `LogitsProcessor.get_top_tokens`
already exists in the fork (vllm/model_executor/layers/logits_processor.py:106 — the
EXPERIMENTS.md "fork lacks get_top_tokens" note is STALE), and the KDA layer is already
spec-aware (kda.py:442-697 + `GDNAttentionMetadata` spec fields). Four small pieces of glue
are missing: the `glm5_next → glm5_next_mtp` config rewrite, an MTPModelTypes entry, the
`model_returns_tuple` dispatch, and a multimodal `image_token_index` crash workaround —
plus one structural gap (draft layers span TWO kv-cache groups: attn + kpool-tail) that
needs a GLM5-gated multi-group proposer patch. All are delivered in
`patches/gdn/glm53_mtp_patch.py` (env-gated, no tree edits).

## Serve flags (run_glm53.sh delta)

```bash
export VLLM_GFX906_GLM53_MTP=1
export PYTHONPATH=/data/vllm-gfx906-dsv4/patches/gdn:$PYTHONPATH
# wrap the launch so the patch installs in the driver before EngineArgs parse:
exec python -c "import sys, glm53_mtp_patch; sys.argv=['vllm','serve']+sys.argv[1:]; \
from vllm.entrypoints.cli.main import main; main()" \
    "$MODEL_PATH" ...existing flags... \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

(or copy `glm53_mtp_patch.py` to `sitecustomize`/`sitecustomize.py` import inside the
venv; the file is idempotent and env-gated.)

Recommended flag set for first bring-up:

```
--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

- `num_speculative_tokens: 1` first (n_predict=1 in ckpt text_config; k>1 re-uses the
  same MTP layer — works, but expect acceptance drop per vLLM warning; also k must be
  ≤ 3 for the kpool-4 tail-ring during verify: verify writes k+1 ≤ 4 tokens/step).
- Keep `--dtype float16`, `--compilation-config '{"mode":0,"cudagraph_mode":"FULL"}'`
  and let the resolver auto-downgrade to NONE with the spec-decode warning, or set
  `"cudagraph_mode": "NONE"` explicitly for a clean log.
- Optional once baseline works: `"use_local_argmax_reduction": true` in the
  speculative-config JSON (uses the fork's `LogitsProcessor.get_top_tokens`,
  avoids full-vocab all-gather on TP8 draft steps).

## Gap list (file:line)

| # | Location | Issue | Fix in patch |
|---|----------|-------|----------------|
| A (patch P1) | vllm/config/speculative.py:302-330 | `hf_config_override` has branches for deepseek_v3/v32/v4, glm_moe_dsa, glm4_moe_*, qwen3_next, etc. but NOT glm5_next | staticmethod wrapper adds the glm5_next→glm5_next_mtp/Glm5NextMTPModel rewrite (upstream parity with cyankiwi/glm53-flash-ct:1030-1036) |
| B (P2) | vllm/config/speculative.py:35-52 | `"glm5_next_mtp"` not in MTPModelTypes literal → NotImplementedError in method detection (speculative.py:654-679) | module-level Literal rebound with the extra member (all uses are runtime get_args at speculative.py:512,654) |
| C (P3) | vllm/v1/spec_decode/llm_base_proposer.py:836-837 | `model_returns_tuple()` returns False for all "mtp" drafts, but Glm5NextMTP returns `(hidden, hidden)` tuple (mtp.py:110) → crash in propose() (llm_base_proposer.py:482-487, 642-647) | override returns True only when draft arch is Glm5NextMTPModel (matches upstream cyankiwi llm_base_proposer.py:1021-1033); fork deepseek drafts keep False |
| D (P4) | llm_base_proposer.py:1356-1383 | multimodal else-branch reads `target_model.config.image_token_index`; Glm5NextConfig defines only `image_token_id` (configs/glm5_next.py:345,374) → AttributeError when loading the draft under the multimodal target arch | `get_model_name` alias routes Glm5NextForConditionalGeneration into the existing `image_token_id` branch (get_model_name is only used in that branch) |
| E (P5+P6) | llm_base_proposer.py:1653-1736 | `validate_same_kv_cache_group` asserts all draft layers share ONE kv group, and `initialize_attn_backend` builds every draft layer against that single group's spec; GLM5's draft spans the uniform attn group (MLA + kpool indexer specs) AND the KpoolTail group (draft tail spec `model.layers.45.self_attn.indexer.tail_cache`) → assert fail → KeyError. NOTE: upstream has the same limitation; GLM5-MTP appears untested end-to-end there too. | GLM5-gated rewrite: relaxed validation (one primary group + ≤1 tail group), AttentionGroups keyed per (backend, gid), block tables for the tail builder swapped from `runner.input_batch.block_table[tail_gid]` during draft metadata builds (runner ref stashed in __init__) |
| F (P7) | vllm/v1/worker/gpu_model_runner.py:6357-6391 + rocm_aiter_mla_sparse.py:171-173 | ROCM_AITER_MLA_SPARSE builder is `UNIFORM_SINGLE_TOKEN_DECODE`; with spec decode, `resolve_cudagraph_mode_and_sizes` (compilation.py:1373-1391) downgrades decode cudagraphs to NONE for the whole engine | opt-in env flag `VLLM_GFX906_GLM53_MTP_FULL_CG=1` flips to UNIFORM_BATCH; default stays safe/eager |

## What needs NO change

- `LogitsProcessor.get_top_tokens` — EXPERIMENTS.md note is stale; exists at
  layers/logits_processor.py:106 (and in the venv mirror).
- `Glm5NextMTP` registration (registry.py:604), weight loader incl. CT/fp8 index
  compat, fused eh_norm Triton op, `set_skip_topk`/`compact_topk_indices` scaffolding,
  `skip_topk` gate in layers/mla.py:194.
- KDA spec-verify support: kda.py:442-697 reads `spec_sequence_masks`,
  `spec_state_indices_tensor`, `spec_token_indx`, `num_spec_decodes` from
  GDNAttentionMetadata; gdn_attn.py:40-372 builds them (spec decode exercised on Hy3).
- The kpool ROCm op is spec-aware: decode path uses `decode_lens`/`requires_padding`
  (rocm_aiter_mla_sparse_kpool.py:197-209, 313-453) incl. "Truncation guard for
  spec-decode cg padding"; `DeepseekV32IndexerMetadataBuilder` bumps
  reorder_batch_threshold by num_spec (indexer.py:432-433) and has native 2-D
  (B, next_n) seq_lens for MTP (indexer.py:609-628).
- TP=8 head divisibility: 64 attn heads ✓; indexer wq_b/wk are ReplicatedLinear
  (replicated) ✓; embed/vocab 154880 splits 8-way ✓.

## Checkpoint facts (/data/ModelDownloader/GLM-5.3-Flash-AWQ-INT4-bf16fix)

- text_config: 45 hidden layers (34 KDA / 11 sparse MLA), 1 MTP layer at index 45,
  `index_topk=2048`, `index_kpool=4`, `index_share_for_mtp_iteration: true`,
  vocab 154880, hidden 4096, NoPE indexer (qk_rope_head_dim=0).
- All 889 layer-45 MTP weights present in the safetensors index; all 877 layer-45
  quant entries are in the CT ignore list (bf16, no packed int4 in the MTP layer).
- config.json carries `index_share_for_mtp_iteration: true` — the upstream
  `_share_mtp_indices` optimization (skip_topk/compact_topk_indices toggling in
  the draft loop) is NOT ported; steps always recompute top-k (correct, slower).
  Port later if draft overhead shows.

## Patch install / serve incantation

```bash
# one-time (or in run_glm53_mtp.sh):
export PYTHONPATH=/data/vllm-gfx906-dsv4/patches/gdn:$PYTHONPATH
export VLLM_GFX906_GLM53_MTP=1
# add to run_glm53.sh launch line:
--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
# optional later: "use_local_argmax_reduction": true inside the same JSON
```

Validation protocol (when GPUs are free):
1. Boot with k=1. Expect log line "GLM-5.3 MTP glue installed" and the
   cudagraph downgrade warning (spec-decode + UNIFORM_SINGLE_TOKEN_DECODE).
2. Boot must pass `MTP speculative decoding layer 45 weights missing` check in
   mtp.py load_weights (889 ckpt tensors resolve).
3. Short greedy prompt; confirm acceptance stats (`spec_decode` metrics) and
   that outputs match non-spec greedy on a fixed probe (CPU diff of token ids).
4. Check acceptance rate; if healthy try k=2. Watch draft tail-block writes.

## Risks (ordered)

1. **Draft tail-table plumbing (P5/P6)** — most intricate patch; wrong tail
   block table = corrupted draft tail ring (acceptance collapse, still
   verify-safe). Bring up with k=1 and a fixed greedy probe first.
2. Perf may be NET-NEGATIVE on gfx906 (Hy3 MTP precedent: kernel-latency-bound,
   ~2× step cost, acceptance 1.5-1.8; EXPERIMENTS.md 2026-09-09). MTP also
   forces cudagraphs OFF (risk: decode slower than baseline even at good
   acceptance). Measure bs1 tok/s vs the 18 tok/s Hy3 reference and vs the
   current GLM5 baseline before adopting. P7 env flag is the escape hatch to
   re-enable FULL cudagraphs if the sparse backend proves graph-safe.
3. hc/fp32: no hard conflict for GLM5 (mtp_block skips mhc; handoff tensor is
   post-norm fp16), but re-check numerics on the first A/B; the DSV4 overflow
   history means any NaN in draft logits should first blame the fp32 hc path.
4. k>1 shares ONE MTP layer (warning-level acceptance loss, by design).
5. `index_share_for_mtp_iteration=true` from the ckpt is left unwired in the
   fork (proposer never calls set_skip_topk); it's a perf-only optimization —
   port the upstream `_share_mtp_indices` block later if acceptance is good
   but draft overhead is high.

## Files written

- `/data/vllm-gfx906-dsv4/patches/gdn/glm53_mtp_patch.py` — env-gated
  monkeypatch module (P1-P6 above + P7 opt-in cudagraph bump).
- `/data/vllm-gfx906-dsv4/MTP53_PLAN.md` — this plan.

Follow-ups left to the operator: GPU bring-up run (acceptance + tok/s A/B vs
the non-spec baseline), then decide on `_share_mtp_indices`/skip_topk port and
the UNIFORM_BATCH cudagraph experiment (P7).
