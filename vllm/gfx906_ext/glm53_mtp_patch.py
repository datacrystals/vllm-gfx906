# GLM53-PORT / phase 3: GLM-5.3-Flash MTP (speculative decoding) enablement glue.
# STATUS: PARKED/optional — MTP is parked (see GLM53_GFX906_STATUS.md); kept
# for a future bring-up attempt.
#
# Env-gated monkeypatch module. Import as early as possible in EVERY process
# of the serving stack: the driver parses --speculative-config and builds the
# draft ModelConfig inside SpeculativeConfig.__post_init__; engine-core and
# TP workers build the draft model and its attention metadata. Import is a
# no-op unless the gate is set:
#
#   VLLM_GFX906_GLM53_MTP=1        master gate
#   VLLM_GFX906_GLM53_MTP_FULL_CG=1  opt-in extra (P7), default OFF
#
# Wiring (vendored in-tree as vllm/gfx906_ext/; supersedes the old
# PYTHONPATH=/data/vllm-gfx906-dsv4/patches/gdn wiring):
#   VLLM_GFX906_GLM53_MTP=1 python -m vllm.gfx906_ext.glm53_mtp_main \
#       serve ... --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
# Or put `from vllm.gfx906_ext import glm53_mtp_patch  # noqa` into a
# sitecustomize.py so forked/spawned engine-core and worker processes
# install it too.
#
# Patch list (file:line refs are fork HEAD ea8406d369):
#   P1  SpeculativeConfig.hf_config_override: add the glm5_next ->
#       glm5_next_mtp branch (architectures=["Glm5NextMTPModel"], n_predict),
#       mirroring cyankiwi/glm53-flash-ct speculative.py ~line 1030. Without
#       it, method="mtp" on a glm5_next checkpoint keeps the target arch and
#       fails with "Unsupported speculative method: 'mtp'".
#   P2  MTPModelTypes literal += "glm5_next_mtp" (all uses are runtime
#       get_args() lookups inside speculative.py:512,654, so rebinding the
#       module attribute is sufficient).
#   P3  SpecDecodeBaseProposer.model_returns_tuple: fork returns False for
#       every "mtp" draft (llm_base_proposer.py:836), but Glm5NextMTP.forward
#       returns a (last_hidden, hidden) tuple (glm5next/nvidia/mtp.py:106-110)
#       -> propose() would index a tuple and crash. Match upstream's
#       arch-aware variant for Glm5NextMTPModel only.
#   P4  Multimodal target: load_model (llm_base_proposer.py:1356-1383) is not
#       Glm5Next-aware; the else-branch reads target_model.config
#       .image_token_index, which Glm5NextConfig does not define (only
#       image_token_id). Alias the class name to the image_token_id branch.
#   P5  GLM5 MTP draft spans TWO kv-cache groups: its MLA attention + kpool
#       indexer land in the uniform attention group, its KpoolTailSpec lands
#       in the tail group (kv_cache_utils.py:1171 _get_kv_cache_groups_glm5_
#       next). The base proposer assumes exactly one group for all draft
#       layers (validate_same_kv_cache_group llm_base_proposer.py:1653) and
#       builds metadata from that single group's spec (KeyError on the tail
#       layer). We relax validation (one primary group + <=1 tail group) and
#       build per-group AttentionGroups so every draft layer gets the spec,
#       gid and kernel block size of its own group.
#   P6  The draft tail builder needs the TAIL group's block table, not the
#       primary group's table carried by common_attn_metadata. We keep a
#       runner reference (fork drops the `runner` ctor arg, l.61-66) and swap
#       in block_table[tail_gid] when building tail metadata for the draft.
#   P7  (opt-in, VLLM_GFX906_GLM53_MTP_FULL_CG=1) raise ROCM_AITER_MLA_SPARSE
#       builder _cudagraph_support UNIFORM_SINGLE_TOKEN_DECODE -> UNIFORM_BATCH
#       so FULL decode cudagraphs survive spec decode (compilation.py:1373-1391
#       downgrades to NONE otherwise). Default OFF: eager is the safe default
#       until the uniform (1+k)-token decode kernels are graph-validated on GPU.
#
# Known non-gaps (checked 2026-09-11): get_top_tokens exists in this fork
# (layers/logits_processor.py:106) and in the venv mirror -- the EXPERIMENTS.md
# note claiming otherwise is stale. Registry entry, MTP weight loader, fused
# eh_norm triton op, lm_head/embed sharing and the shared_head.head fixup
# (llm_base_proposer.py:1528-1544) are all already in place.

import copy as _copy
import os
import typing

GATE_ENV = "VLLM_GFX906_GLM53_MTP"
CG_ENV = "VLLM_GFX906_GLM53_MTP_FULL_CG"

_INSTALLED = False
_logger = None


def _log(msg, *args):
    global _logger
    if _logger is None:
        try:
            from vllm.logger import init_logger

            _logger = init_logger("glm53_mtp_patch")
        except Exception:
            import logging

            _logger = logging.getLogger("glm53_mtp_patch")
    _logger.info("[glm53-mtp] " + (msg % args if args else msg))


def _env_on(name):
    return os.environ.get(name, "0") in ("1", "true", "True")


def _draft_is_glm5_mtp(proposer) -> bool:
    try:
        archs = getattr(proposer.draft_model_config.hf_config, "architectures", None)
        return "Glm5NextMTPModel" in (archs or [])
    except Exception:
        return False


def _patch_speculative_config():
    import vllm.config.speculative as spec_mod

    orig_override = spec_mod.SpeculativeConfig.hf_config_override

    def hf_config_override(hf_config):
        # Mirror cyankiwi/glm53-flash-ct speculative.py ~line 1030.
        if getattr(hf_config, "model_type", None) == "glm5_next":
            hf_config.model_type = "glm5_next_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None) or 1
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Glm5NextMTPModel"]}
            )
            return hf_config
        return orig_override(hf_config)

    spec_mod.SpeculativeConfig.hf_config_override = staticmethod(hf_config_override)

    mtp_args = tuple(typing.get_args(spec_mod.MTPModelTypes))
    if "glm5_next_mtp" not in mtp_args:
        spec_mod.MTPModelTypes = typing.Literal.__getitem__(
            mtp_args + ("glm5_next_mtp",)
        )
        assert "glm5_next_mtp" in typing.get_args(spec_mod.MTPModelTypes)
    _log(
        "SpeculativeConfig patched: glm5_next->glm5_next_mtp draft override; "
        "MTPModelTypes += glm5_next_mtp"
    )


def _patch_proposer():
    import vllm.v1.spec_decode.llm_base_proposer as LBP
    from vllm.v1.kv_cache_interface import KpoolTailSpec

    proposer_cls = LBP.SpecDecodeBaseProposer

    # --- keep a runner reference (fork accepts but drops the `runner` arg) ---
    _orig_init = proposer_cls.__init__

    def __init__(self, vllm_config, device, pass_hidden_states_to_model, runner=None):
        _orig_init(self, vllm_config, device, pass_hidden_states_to_model, runner=runner)
        self.runner = runner

    proposer_cls.__init__ = __init__

    # --- P3: Glm5NextMTP returns a (last_hidden, hidden) tuple ---
    _orig_returns_tuple = proposer_cls.model_returns_tuple

    def model_returns_tuple(self):
        if self.method == "mtp" and _draft_is_glm5_mtp(self):
            return True
        return _orig_returns_tuple(self)

    proposer_cls.model_returns_tuple = model_returns_tuple

    # --- P4: multimodal target -> use the image_token_id branch ---
    _orig_get_model_name = proposer_cls.get_model_name

    def get_model_name(self, model):
        name = _orig_get_model_name(self, model)
        if name == "Glm5NextForConditionalGeneration":
            # load_model only special-cases listed mm wrappers; Glm5NextConfig
            # has image_token_id (not image_token_index), same as GlmOcr.
            return "GlmOcrForConditionalGeneration"
        return name

    proposer_cls.get_model_name = get_model_name

    # --- P5/P6: multi-kv-group draft attention init + tail block table ------
    def _locate_draft_layers(self, kv_cache_config):
        """(layer_name -> gid) for draft layers, plus the tail-layer subset."""
        all_layers = LBP.get_layers_from_vllm_config(
            self.vllm_config, LBP.AttentionLayerBase
        )
        name_to_gid = {}
        tail_names = set()
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            per_layer = (
                spec.kv_cache_specs
                if isinstance(spec, LBP.UniformTypeKVCacheSpecs)
                else {}
            )
            for layer_name in group.layer_names:
                if layer_name not in self._draft_attn_layer_names:
                    continue
                name_to_gid[layer_name] = gid
                layer_spec = per_layer.get(layer_name, spec)
                if isinstance(layer_spec, KpoolTailSpec):
                    tail_names.add(layer_name)
        missing = self._draft_attn_layer_names - set(name_to_gid)
        assert not missing, f"GLM5 MTP: draft layers missing from kv config: {sorted(missing)}"
        return all_layers, name_to_gid, kv_cache_config.kv_cache_groups, tail_names

    def locate(self, kv_cache_config):
        cache = getattr(self, "_glm5_draft_layout", None)
        if cache is None or cache[0] is not kv_cache_config:
            cache = (kv_cache_config, *_locate_draft_layers(self, kv_cache_config))
            self._glm5_draft_layout = cache
        return cache[1:]

    _orig_validate = proposer_cls.validate_same_kv_cache_group

    def validate_same_kv_cache_group(self, kv_cache_config):
        if not _draft_is_glm5_mtp(self):
            return _orig_validate(self, kv_cache_config)
        _, name_to_gid, _, tail_names = locate(self, kv_cache_config)
        primary = {gid for n, gid in name_to_gid.items() if n not in tail_names}
        tail_gids = {gid for n, gid in name_to_gid.items() if n in tail_names}
        assert len(primary) == 1, (
            "GLM5 MTP: non-tail draft layers must share one kv cache group, "
            f"got {sorted(primary)}"
        )
        assert len(tail_gids) <= 1, (
            "GLM5 MTP: draft tail layers in multiple groups: "
            f"{sorted(tail_gids)}"
        )

    proposer_cls.validate_same_kv_cache_group = validate_same_kv_cache_group

    def _layer_spec(group, layer_name):
        spec = group.kv_cache_spec
        if isinstance(spec, LBP.UniformTypeKVCacheSpecs):
            return spec.kv_cache_specs[layer_name]
        return spec

    _orig_init_attn = proposer_cls.initialize_attn_backend

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None):
        if not _draft_is_glm5_mtp(self):
            return _orig_init_attn(self, kv_cache_config, kernel_block_sizes)
        all_layers, name_to_gid, groups, tail_names = locate(self, kv_cache_config)
        primary_gid = min(gid for n, gid in name_to_gid.items() if n not in tail_names)

        attention_groups = {}
        for layer_name in sorted(self._draft_attn_layer_names):
            gid = name_to_gid[layer_name]
            attn_backend = all_layers[layer_name].get_attn_backend()
            backend_key = (attn_backend.full_cls_name(), gid)
            if backend_key not in attention_groups:
                kernel_block_size = (
                    kernel_block_sizes[gid]
                    if kernel_block_sizes is not None and gid < len(kernel_block_sizes)
                    else None
                )
                attn_group = LBP.AttentionGroup(
                    backend=attn_backend,
                    layer_names=[layer_name],
                    kv_cache_spec=_layer_spec(groups[gid], layer_name),
                    kv_cache_group_id=gid,
                )
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_size=kernel_block_size,
                )
                attention_groups[backend_key] = attn_group
            else:
                attention_groups[backend_key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        self.kv_cache_gid = primary_gid
        # Slot-mapping kernel block size must come from the MLA attention
        # group (token-space blocks), never from the kpool tail ring.
        self.block_size = None
        for attn_group in self.draft_attn_groups:
            if attn_group.kv_cache_group_id != primary_gid:
                continue
            if isinstance(attn_group.kv_cache_spec, KpoolTailSpec):
                continue
            if kernel_block_sizes is not None and primary_gid < len(kernel_block_sizes):
                self.block_size = kernel_block_sizes[primary_gid]
            else:
                self.block_size = attn_group.get_metadata_builder().kv_cache_spec.block_size
            break
        assert self.block_size, "GLM5 MTP: primary attention group not found"
        _log(
            "draft attn backends across %d kv group(s); primary gid=%d, "
            "block_size=%d, tail layers=%d",
            len({g.kv_cache_group_id for g in self.draft_attn_groups}),
            primary_gid,
            self.block_size,
            len(tail_names),
        )

    proposer_cls.initialize_attn_backend = initialize_attn_backend

    _orig_build_meta = proposer_cls.build_per_group_and_layer_attn_metadata

    def _group_is_tail(attn_group):
        spec = attn_group.kv_cache_spec
        inner = (
            spec.kv_cache_specs.values()
            if isinstance(spec, LBP.UniformTypeKVCacheSpecs)
            else (spec,)
        )
        return any(isinstance(s, KpoolTailSpec) for s in inner)

    def build_per_group_and_layer_attn_metadata(self, common_attn_metadata,
                                                draft_index=0):
        runner = getattr(self, "runner", None)
        if not _draft_is_glm5_mtp(self) or runner is None:
            return _orig_build_meta(self, common_attn_metadata, draft_index)
        per_group_attn_metadata = []
        per_layer_attn_metadata = {}
        for attn_group in self.draft_attn_groups:
            cad = common_attn_metadata
            if _group_is_tail(attn_group):
                gid = attn_group.kv_cache_group_id
                try:
                    cad = _copy.copy(common_attn_metadata)
                    cad.block_table_tensor = runner.input_batch.block_table[
                        gid
                    ].get_device_tensor(common_attn_metadata.num_reqs)
                except Exception as exc:
                    _log(
                        "tail block table lookup failed (gid=%d): %r; using "
                        "primary-group block table", gid, exc,
                    )
                    cad = common_attn_metadata
            md = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cad, draft_index=draft_index
            )
            per_group_attn_metadata.append(md)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = md
        return per_group_attn_metadata, per_layer_attn_metadata

    proposer_cls.build_per_group_and_layer_attn_metadata = (
        build_per_group_and_layer_attn_metadata
    )

    _log(
        "SpecDecodeBaseProposer patched: tuple-return + mm alias for GLM5 MTP, "
        "multi-kv-group draft backend init, tail block-table substitution"
    )


def _patch_cudagraph_support():
    """P7 (opt-in): allow FULL decode cudagraphs under spec decode."""
    if not _env_on(CG_ENV):
        return
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse

    rocm_aiter_mla_sparse.ROCMAiterMLASparseMetadataBuilder._cudagraph_support = (
        AttentionCGSupport.UNIFORM_BATCH
    )
    _log(
        "EXPERIMENTAL: ROCM_AITER_MLA_SPARSE _cudagraph_support -> UNIFORM_BATCH. "
        "If graph capture misbehaves, unset %s.",
        CG_ENV,
    )


def install() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    if not _env_on(GATE_ENV):
        return False

    _patch_speculative_config()
    _patch_proposer()
    _patch_cudagraph_support()

    _INSTALLED = True
    _log("GLM-5.3 MTP glue installed; pass --speculative-config "
         '\'{"method": "mtp", "num_speculative_tokens": 1}\'')
    return True


install()
