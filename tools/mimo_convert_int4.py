#!/usr/bin/env python3
"""Convert the XiaomiMiMo/MiMo-V2.6-Flash-RL checkpoint to a single consistent
compressed-tensors INT4 group-32 ASYMMETRIC ("pack-quantized", GPTQ-style)
text-only checkpoint that this vLLM fork loads with `--quantization
compressed-tensors` (scheme CompressedTensorsWNA16), reusing the fork's GLM
AWQ-INT4 machinery.

Source tensor formats (verified against the checkpoint and HF transformers):
  * MoE experts `model.layers.N.mlp.experts.E.{gate,up,down}_proj`:
      weight       U8  [out, in/2]   two e2m1 (mxfp4) values per byte,
                                     LOW nibble = first element of the pair
      weight_scale U8  [out, in/32]  e8m0 scale, value = 2^(scale - 127)
      dequant:  w = e2m1(nibble) * 2^(s - 127)      (per 32-group along `in`)
    Ground truth for the decode: transformers/integrations/mxfp4.py
    `_convert_moe_packed_tensors` (copied from the GPT-OSS repo and vLLM):
    idx_lo = (blk & 0x0F) -> even slots, idx_hi = (blk >> 4) -> odd slots,
    torch.ldexp(values, scale - 127).
  * Dense linears (qkv_proj on all 48 layers, mlp.{gate,up,down}_proj on the
    dense layer 0, MTP qkv/mlp) `weight` F8_E4M3 [out, in] +
    `weight_scale_inv` F32 [ceil(out/128), ceil(in/128)] (128x128 blocks).
    dequant:  w = fp8_code * weight_scale_inv[row//128, col//128]
    (same method as /home/tliao/dequant_fp8_glm53.py).  Full-attention qkv
    tensors carry 2 extra scale rows (grid computed on 13824 = 72*192 rows,
    v padded); extra rows are cropped.
  * Copy-through BF16/F32: norms, embed_tokens, lm_head, self_attn.o_proj,
    routers (mlp.gate.{weight,e_score_correction_bias}), attention_sink_bias,
    MTP norms/eh_proj/o_proj.
  * Dropped entirely: visual.*, audio_encoder.*, speech_embeddings.*.

Output per converted module (bit-exact the layout of
tools/requant_islands_int4.py / GLM-5.3-Flash-AWQ-INT4-64k group_0):
    <mod>.weight_packed      I32 [R, C/8]     8x int4 per int32 along dim 1,
                                              nibble j at bits [4j, 4j+4),
                                              stored (q + 8) unsigned
    <mod>.weight_scale       BF16 [R, C/32]   per (row, 32-col group) scale
    <mod>.weight_shape       I64 [2]          original logical shape [R, C]
    <mod>.weight_zero_point  I32 [R/8, C/32]  packed int4 zero points,
                                              pack_to_int32(dim=0), (zp + 8)
quantize:   q = clamp(round(x / s) + zp, -8, 7)     (CT group strategy)
dequantize: x = (q - zp) * s
RTN scales: minmax per (row, group-of-32) with exhaustive zp grid in [-8..7].

Subcommands:
    inspect      tensor inventory of the source (dtype/shape counts, bytes per
                 category, expected output size) + the empirical mxfp4
                 nibble-order / scale-semantics determination on one expert
                 tensor and the fp8 block-scale grid probe.
    convert      streaming, tensor-by-tensor, shard-by-shard conversion.
                 Resumable (per-shard done markers in <dst>/_done/) and
                 parallel-safe (one process per output shard via
                 --only-shard; finalize writes config/index after all shards).
    verify       random-sample N converted tensors, dequant both source and
                 converted to fp16, report max/mean rel error + cosine, and
                 compare against the source format's own quant grid step.
    drop-modules write a text-only config.json (vision/audio/processor configs
                 removed, architectures + model_type unchanged) with the
                 compressed-tensors quantization_config substituted.

Only CPU + safetensors are used; source tensors are read one at a time via
safe_open (or pread byte-copies for pass-through tensors); output shards are
written to *.partial and atomically renamed.
"""

import argparse
import gc
import json
import os
import re
import shutil
import struct
import sys
import time

import torch

SRC_DIR = "/data/ModelDownloader/MiMo-V2.6-Flash-RL"
DST_DIR = "/data/ModelDownloader/MiMo-V2.6-Flash-INT4"
CT_CONFIG_REF = "/data/ModelDownloader/GLM-5.3-Flash-AWQ-INT4-64k/config.json"
DONE_DIR = "_done"
MANIFEST_NAME = "_mimo_int4_manifest.json"

GROUP_SIZE = 32
NUM_BITS = 4
Q_MIN, Q_MAX = -8, 7
COPY_CHUNK = 32 * 1024 * 1024
FP8_BLOCK = (128, 128)
E4M3_MAX = 448.0

# e2m1 (mxfp4) codes 0..15 -- transformers/integrations/mxfp4.py FP4_VALUES
FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

ST_DSIZE = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "U8": 1,
            "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1}
ST_DTYPE = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
            "I32": torch.int32, "I64": torch.int64, "U8": torch.uint8,
            "I8": torch.int8, "F8_E4M3": torch.float8_e4m3fn}

DROP_PREFIXES = ("visual.", "audio_encoder.", "speech_embeddings.")
EXPERT_RE = re.compile(
    r"^(?P<mod>model\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate|up|down)_proj)"
    r"\.(?P<leaf>weight|weight_scale)$")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# safetensors low-level helpers (header json + raw byte regions; no full load)
# --------------------------------------------------------------------------

def read_shard_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def read_tensor_region(path, entry, buf_base):
    """pread exactly one tensor as a torch tensor (low RAM)."""
    b, e = entry["data_offsets"]
    n = e - b
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    with open(path, "rb") as f:
        fd = f.fileno()
        while got < n:
            r = os.preadv(fd, [mv[got:]], buf_base + b + got)
            if r == 0:
                raise IOError(f"short read on {path}")
            got += r
    dt = ST_DTYPE[entry["dtype"]]
    if dt is torch.float8_e4m3fn:
        t = torch.frombuffer(buf, dtype=torch.uint8).view(torch.float8_e4m3fn)
    else:
        t = torch.frombuffer(buf, dtype=dt)
    return t.reshape(entry["shape"])


def copy_region(src_fd, dst_fd, src_off, n):
    done = 0
    while done < n:
        c = min(COPY_CHUNK, n - done)
        buf = os.pread(src_fd, c, src_off + done)
        if not buf:
            raise IOError("short read during copy")
        view = memoryview(buf)
        w = 0
        while w < len(buf):
            w += os.write(dst_fd, view[w:])
        done += c


def write_bytes(fd, buf):
    view = memoryview(buf)
    w = 0
    while w < len(buf):
        w += os.write(fd, view[w:])


def tensor_payload(t):
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).contiguous().numpy().tobytes()
    return t.contiguous().numpy().tobytes()


def write_shard_stream(path_tmp, entries, data_writer):
    """entries: list of (name, dtype_str, shape, nbytes) in write order.
    data_writer(dst_fd) writes the tensor bytes in the same order."""
    header = {}
    off = 0
    for name, dtype, shape, nbytes in entries:
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + nbytes]}
        off += nbytes
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((-len(hj)) % 8)
    with open(path_tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        data_writer(f.fileno())
        f.flush()
        os.fsync(f.fileno())


# --------------------------------------------------------------------------
# quantization core -- KEEP IN SYNC with tools/requant_islands_int4.py
# --------------------------------------------------------------------------

def rtn_scale_zp(xg):
    """xg: [R, G, group] float32 -> (scale [R, G] f32, zp [R, G] f32)."""
    mn = xg.amin(dim=-1)
    mx = xg.amax(dim=-1)
    zps = torch.arange(Q_MIN, Q_MAX + 1, dtype=torch.float32)
    d_hi = (Q_MAX - zps)
    d_lo = (Q_MIN - zps)
    mx_b = mx.unsqueeze(-1)
    mn_b = mn.unsqueeze(-1)
    inf = torch.tensor(float("inf"))
    s_hi = torch.where(d_hi > 0, mx_b / d_hi.clamp(min=1e-12),
                       torch.where(mx_b <= 0, torch.zeros_like(mx_b),
                                   inf.expand_as(mx_b)))
    s_lo = torch.where(d_lo < 0, mn_b / d_lo.clamp(max=-1e-12),
                       torch.where(mn_b >= 0, torch.zeros_like(mn_b),
                                   inf.expand_as(mn_b)))
    s_cand = torch.maximum(s_hi, s_lo)
    best, idx = s_cand.min(dim=-1)
    zp = zps[idx]
    eps = torch.finfo(torch.bfloat16).tiny
    scale = best.clamp(min=eps)
    return scale, zp


def pack_to_int32_manual(q_int8, num_bits, packed_dim):
    """Inverse of requant_islands_int4.unpack_i32_manual (bitwise == CT
    pack_to_int32): store (q + 8) unsigned, nibble j at bits [4j, 4j+4)."""
    assert num_bits == 4
    uq = (q_int8.to(torch.int64) + (1 << (num_bits - 1))) & 0xF
    pf = 32 // num_bits
    sh = torch.arange(pf, dtype=torch.int64) * num_bits
    if packed_dim == 1:
        R, C = q_int8.shape
        uq = uq.reshape(R, C // pf, pf)
        packed = (uq << sh).sum(-1)
    else:
        R, C = q_int8.shape
        uq = uq.reshape(R // pf, pf, C)
        packed = (uq << sh.view(pf, 1)).sum(1)
    packed = packed & 0xFFFFFFFF
    packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed)
    return packed.to(torch.int32).contiguous()


_PACK_CHECKED = {"ok": False}


def pack_to_int32(q_int8, num_bits, packed_dim):
    """CT pack_to_int32 when available (bitwise-checked once), manual else."""
    if not _PACK_CHECKED["ok"]:
        try:
            from compressed_tensors.compressors.pack_quantized.helpers import (
                pack_to_int32 as ct_pack,
            )
            probe = torch.randint(Q_MIN, Q_MAX + 1, (8, 64), dtype=torch.int8)
            a = ct_pack(probe, num_bits, packed_dim=1)
            b = pack_to_int32_manual(probe, num_bits, 1)
            assert torch.equal(a, b), "pack(dim=1) mismatch vs compressed_tensors"
            probe0 = torch.randint(Q_MIN, Q_MAX + 1, (16, 8), dtype=torch.int8)
            a0 = ct_pack(probe0, num_bits, packed_dim=0)
            b0 = pack_to_int32_manual(probe0, num_bits, 0)
            assert torch.equal(a0, b0), "pack(dim=0) mismatch vs compressed_tensors"
            _PACK_CHECKED["ok"] = True
            log("pack_to_int32: bitwise-verified against compressed_tensors")
        except ImportError:
            _PACK_CHECKED["ok"] = True
            log("pack_to_int32: compressed_tensors not importable; using manual")
    return pack_to_int32_manual(q_int8, num_bits, packed_dim)


def quantize_weight(w_fp16):
    """fp16 [R, C] -> dict of the 4 packed CT tensors. Same math as
    tools/requant_islands_int4.quantize_weight."""
    R, C = w_fp16.shape
    assert C % GROUP_SIZE == 0 and R % 8 == 0, f"bad shape {tuple(w_fp16.shape)}"
    xf = w_fp16.to(torch.float32)
    xg = xf.view(R, C // GROUP_SIZE, GROUP_SIZE)
    scale, zp = rtn_scale_zp(xg)
    q = torch.round(xg / scale.unsqueeze(-1) + zp.unsqueeze(-1))
    q = q.clamp_(Q_MIN, Q_MAX).view(R, C).to(torch.int8)
    zp_i8 = zp.view(R, C // GROUP_SIZE).to(torch.int8)
    packed = pack_to_int32(q, NUM_BITS, packed_dim=1)
    packed_zp = pack_to_int32(zp_i8, NUM_BITS, packed_dim=0)
    return {
        "weight_packed": (packed, "I32"),
        "weight_scale": (scale.to(torch.bfloat16), "BF16"),
        "weight_shape": (torch.tensor([R, C], dtype=torch.int64), "I64"),
        "weight_zero_point": (packed_zp, "I32"),
    }


def unpack_i32_manual(packed, num_bits, shape, packed_dim):
    assert packed.dtype == torch.int32
    u = packed.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    pf = 32 // num_bits
    mask = (1 << num_bits) - 1
    sh = (torch.arange(pf, dtype=torch.int64) * num_bits)
    if packed_dim == 1:
        R, Cp = u.shape
        out = torch.zeros((R, Cp, pf), dtype=torch.int64)
        out.copy_((u.unsqueeze(-1) >> sh) & mask)
        out = out.view(R, Cp * pf)[:, : shape[1]]
    else:
        Rp, C = u.shape
        out = torch.zeros((Rp, pf, C), dtype=torch.int64)
        out.copy_((u.unsqueeze(1) >> sh.view(1, pf, 1)) & mask)
        out = out.view(Rp * pf, C)[: shape[0], :]
    return (out - (1 << (num_bits - 1))).to(torch.int8)


def dequant_ct(packed, zp_packed, scale, shape, group=GROUP_SIZE):
    """Manual unpack + dequant of the CT int4 layout -> fp16 [R, C]."""
    R, C = shape
    q = unpack_i32_manual(packed, NUM_BITS, (R, C), packed_dim=1).to(torch.float32)
    zp = unpack_i32_manual(zp_packed, NUM_BITS, (R, C // group), packed_dim=0
                           ).to(torch.float32)
    s = scale.to(torch.float32)
    qg = q.view(R, C // group, group)
    out = ((qg - zp.unsqueeze(-1)) * s.unsqueeze(-1)).view(R, C)
    return out.to(torch.float16)


# --------------------------------------------------------------------------
# source dequant: mxfp4 experts and fp8 block-scale dense
# --------------------------------------------------------------------------

def dequant_mxfp4(w_u8, scale_u8, out_shape, rows_chunk=2048):
    """U8 [R, C/2] + U8 [R, C/32] -> fp16 [R, C].  Low nibble first,
    e2m1 LUT, e8m0 scale 2^(s-127)."""
    R, Ch = w_u8.shape
    assert out_shape == (R, 2 * Ch), f"{out_shape} vs {tuple(w_u8.shape)}"
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32)
    out = torch.empty(R, 2 * Ch, dtype=torch.float16)
    for r0 in range(0, R, rows_chunk):
        r1 = min(r0 + rows_chunk, R)
        blk = w_u8[r0:r1]
        vals = torch.empty(r1 - r0, 2 * Ch, dtype=torch.float32)
        vals[:, 0::2] = lut[(blk & 0x0F).long()]
        vals[:, 1::2] = lut[(blk >> 4).long()]
        exp = (scale_u8[r0:r1].to(torch.int32) - 127).repeat_interleave(32, 1)
        torch.ldexp(vals, exp, out=vals)
        out[r0:r1] = vals.to(torch.float16)
    return out


def dequant_fp8_block(w_f8, scale_inv):
    """F8_E4M3 [R, C] + F32 [ceil(R/128), ceil(C/128)] -> fp16 [R, C].
    Method of /home/tliao/dequant_fp8_glm53.py (crop the scale grid)."""
    rows, cols = w_f8.shape
    s_full = (scale_inv.repeat_interleave(FP8_BLOCK[0], 0)
              .repeat_interleave(FP8_BLOCK[1], 1))[:rows, :cols]
    return (w_f8.to(torch.float32) * s_full).to(torch.float16)


# --------------------------------------------------------------------------
# classification / planning
# --------------------------------------------------------------------------


# GLM53-QKV-LAYOUT-FIX (2026-10-03): fused qkv_proj is stored TP4-chunked
# (4 chunks of [q-chunk | k-chunk | v-chunk]) with a PER-CHUNK-PADDED 128x128
# scale grid (full: 27 blocks/chunk with v padded 128->192; swa: 29 exact).
# Reading rows as flat [q|k|v] scrambles Q/K/V into word salad (NVIDIA DGX
# MiMo-V2.6 field guide, Fix #1). Dequant with per-chunk scale rows and
# regroup to canonical [q|k|v].
_orig_dequant_fp8_block = dequant_fp8_block


def dequant_fp8_block(w, s, **kw):  # noqa: F811
    if w.shape[0] not in (13568, 14848):
        return _orig_dequant_fp8_block(w, s, **kw)
    R = w.shape[0]
    full = R == 13568
    QC = 3072
    KC = 192 if full else 384
    VC = 128 if full else 256
    cs = QC + KC + VC
    blocks = 27 if full else 29
    w32 = w.to(torch.float32)
    out = torch.empty(R, w.shape[1], dtype=torch.float32)
    for c in range(4):
        for j in range(0, cs, 128):
            n = min(128, cs - j)
            rows = list(range(c * cs + j, c * cs + j + n))
            b = c * blocks + (j // 128)
            sc = s[b, :].to(torch.float32)
            out[rows, :] = w32[rows, :] * torch.repeat_interleave(sc, 128)[None, :]
    q = torch.cat([out[c * cs:c * cs + QC] for c in range(4)], 0)
    k = torch.cat([out[c * cs + QC:c * cs + QC + KC] for c in range(4)], 0)
    v = torch.cat([out[c * cs + QC + KC:(c + 1) * cs] for c in range(4)], 0)
    return torch.cat([q, k, v], 0)

def is_dropped(name):
    if os.environ.get("CONVERT_OMNI") == "1":
        return False  # keep visual./audio_encoder./speech_embeddings. towers
    return name.startswith(DROP_PREFIXES)


def classify(name, all_names):
    """-> 'drop' | 'expert_w' | 'expert_s' | 'fp8_w' | 'copy'"""
    if is_dropped(name):
        return "drop"
    m = EXPERT_RE.match(name)
    if m:
        return "expert_w" if m.group("leaf") == "weight" else "expert_s"
    if name.endswith(".weight_scale_inv"):
        return "fp8_s"
    if name.endswith(".weight"):
        mod = name[:-len(".weight")]
        if mod + ".weight_scale_inv" in all_names:
            return "fp8_w"
    return "copy"


def logical_shape(kind, entry):
    """True [R, C] of a weight being converted.  Expert mxfp4 weights are
    stored U8 [R, C/2] (2 values per byte); fp8 weights are stored [R, C]."""
    R, C = entry["shape"]
    if kind == "expert_w":
        return R, 2 * C
    return R, C


def plan_shard(header):
    names = [k for k in header if k != "__metadata__"]
    nset = set(names)
    plan = []  # (name, entry, kind, mod_or_None)
    for name in names:
        kind = classify(name, nset)
        mod = None
        if kind == "expert_w":
            mod = EXPERT_RE.match(name).group("mod")
        elif kind == "fp8_w":
            mod = name[:-len(".weight")]
        elif kind == "fp8_s":
            mod = name[:-len(".weight_scale_inv")]
        plan.append((name, header[name], kind, mod))
    return plan


def out_entries_for(plan):
    """List of (name, dtype, shape, nbytes) that convert writes for this shard."""
    entries = []
    for name, entry, kind, mod in plan:
        if kind in ("expert_s", "fp8_s"):
            continue  # absorbed into the quartet / dropped
        if kind == "drop":
            continue
        if kind in ("expert_w", "fp8_w"):
            R, C = logical_shape(kind, entry)
            g = C // GROUP_SIZE
            entries.extend([
                (f"{mod}.weight_packed", "I32", [R, C // 8], R * (C // 8) * 4),
                (f"{mod}.weight_scale", "BF16", [R, g], R * g * 2),
                (f"{mod}.weight_shape", "I64", [2], 16),
                (f"{mod}.weight_zero_point", "I32", [R // 8, g],
                 (R // 8) * g * 4),
            ])
        else:
            b, e = entry["data_offsets"]
            entries.append((name, entry["dtype"], entry["shape"], e - b))
    return entries


# --------------------------------------------------------------------------
# config.json construction
# --------------------------------------------------------------------------

def build_quantization_config(ignore_list):
    if os.path.exists(CT_CONFIG_REF):
        qc = json.load(open(CT_CONFIG_REF))["quantization_config"]
    else:  # structure of GLM-5.3-Flash-AWQ-INT4-64k config.json group_0
        qc = {
            "config_groups": {
                "group_0": {
                    "format": "pack-quantized",
                    "input_activations": None,
                    "output_activations": None,
                    "targets": ["Linear"],
                    "weights": {
                        "actorder": None, "block_structure": None,
                        "dynamic": False, "group_size": 32, "num_bits": 4,
                        "observer": "mse", "observer_kwargs": {},
                        "scale_dtype": None, "strategy": "group",
                        "symmetric": False, "type": "int",
                        "zp_dtype": "torch.int8",
                    },
                }
            },
            "format": "mixed-precision",
            "global_compression_ratio": None,
            "kv_cache_scheme": None,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "sparsity_config": {},
            "transform_config": {},
        }
    qc = dict(qc)
    qc["ignore"] = sorted(set(ignore_list))
    return qc


def build_config(src_dir, ignore_list):
    cfg = json.load(open(os.path.join(src_dir, "config.json")))
    if os.environ.get("CONVERT_OMNI") == "1":
        cfg["architectures"] = ["MiMoV2OmniForCausalLM"]
    else:
        for k in ("vision_config", "audio_config", "processor_config"):
            cfg.pop(k, None)
    cfg["quantization_config"] = build_quantization_config(ignore_list)
    return cfg


def compute_ignore_list(src_dir):
    """Module paths of every tensor that is NOT converted to int4."""
    ignore = set()
    for shard in shard_list(src_dir):
        header, _ = read_shard_header(os.path.join(src_dir, shard))
        plan = plan_shard(header)
        nset = {name for name, _e, _k, _m in plan}
        for name, _e, kind, mod in plan:
            if kind in ("drop", "expert_w", "expert_s", "fp8_w", "fp8_s"):
                continue
            ignore.add(name.rsplit(".", 1)[0])
    return sorted(ignore)


def shard_list(src_dir):
    idx_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        idx = json.load(open(idx_path))
        shards = sorted(set(idx["weight_map"].values()))
    else:
        shards = sorted(f for f in os.listdir(src_dir)
                        if f.endswith(".safetensors"))
    return shards


# --------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------

def cmd_inspect(args):
    src = args.src
    shards = shard_list(src)
    log(f"inspect: {len(shards)} shards under {src}")

    agg = {}
    def bump(cat, nbytes=0, n=1, shape=None, dtype=None):
        d = agg.setdefault(cat, {"tensors": 0, "bytes": 0, "dtype": {},
                                 "shape_examples": {}})
        d["tensors"] += n
        d["bytes"] += nbytes
        if dtype is not None:
            d["dtype"][dtype] = d["dtype"].get(dtype, 0) + 1
        if shape is not None and len(d["shape_examples"]) < 6:
            d["shape_examples"][str(tuple(shape))] = \
                d["shape_examples"].get(str(tuple(shape)), 0) + 1

    out_bytes = 0
    dropped_bytes = 0
    src_bytes = 0
    quant_modules = {"expert": 0, "fp8": 0}
    expert_shapes = {}
    fp8_anomalies = []
    for shard in shards:
        spath = os.path.join(src, shard)
        if not os.path.exists(spath):
            log(f"  MISSING {shard} (download in progress?) -- skipped")
            continue
        header, _base = read_shard_header(spath)
        plan = plan_shard(header)
        for name, entry, kind, mod in plan:
            b, e = entry["data_offsets"]
            nb = e - b
            src_bytes += nb
            if kind == "drop":
                dropped_bytes += nb
                bump("dropped", nb, dtype=entry["dtype"], shape=entry["shape"])
            elif kind in ("expert_w", "expert_s"):
                bump("expert_mxfp4", nb, dtype=entry["dtype"],
                     shape=entry["shape"])
                if kind == "expert_w":
                    quant_modules["expert"] += 1
                    lsh = logical_shape(kind, entry)
                    expert_shapes[lsh] = expert_shapes.get(lsh, 0) + 1
            elif kind in ("fp8_w", "fp8_s"):
                bump("fp8_block", nb, dtype=entry["dtype"],
                     shape=entry["shape"])
                if kind == "fp8_w":
                    quant_modules["fp8"] += 1
                else:
                    wname = mod + ".weight"
                    ws = tuple(header[wname]["shape"])
                    if entry["shape"][0] * FP8_BLOCK[0] != ws[0] or \
                       entry["shape"][1] * FP8_BLOCK[1] != ws[1]:
                        fp8_anomalies.append(
                            (name, tuple(entry["shape"]), ws))
            else:
                bump("copy_through", nb, dtype=entry["dtype"],
                     shape=entry["shape"])
            if kind in ("expert_w", "fp8_w"):
                R, C = logical_shape(kind, entry)
                g = C // GROUP_SIZE
                out_bytes += R * (C // 8) * 4 + R * g * 2 + 16 + (R // 8) * g * 4
            elif kind == "copy":
                out_bytes += nb
        gc.collect()

    log("---- source inventory (bytes are raw tensor payload) ----")
    total = 0
    for cat in ("expert_mxfp4", "fp8_block", "copy_through", "dropped"):
        d = agg.get(cat, {"tensors": 0, "bytes": 0, "dtype": {}})
        total += d["bytes"]
        log(f"  {cat:14s}: {d['tensors']:6d} tensors  "
            f"{d['bytes'] / 2**30:8.2f} GiB  dtypes={d['dtype']}")
    log(f"  {'TOTAL':14s}: {sum(d['tensors'] for d in agg.values()):6d} tensors  "
        f"{total / 2**30:8.2f} GiB")
    log(f"  quantized modules: {quant_modules['expert']} expert (mxfp4->int4), "
        f"{quant_modules['fp8']} dense (fp8->int4)")
    log(f"  expert weight shapes: { {str(k): v for k, v in expert_shapes.items()} }")
    log(f"  dropped-module byte savings: {dropped_bytes / 2**30:.2f} GiB "
        f"({100.0 * dropped_bytes / max(total, 1):.1f}% of source)")
    log(f"  expected INT4 output size: {out_bytes / 2**30:.2f} GiB "
        f"(source {total / 2**30:.2f} GiB; experts grow slightly because CT "
        f"stores fp16 scale + packed zp per 32-group vs mxfp4's 1-byte e8m0)")
    if fp8_anomalies:
        log(f"  fp8 scale-grid anomalies ({len(fp8_anomalies)}): "
            f"scale rows*128 != weight rows")
        for name, ss, ws in fp8_anomalies[:6]:
            log(f"    {name}: scale {ss} vs weight {ws} "
                f"(extra rows cropped; full-attn qkv grid computed on "
                f"13824 = 72 heads * 192 rows)")
    print(json.dumps({
        "shards": len(shards),
        "categories": {k: {"tensors": v["tensors"], "bytes": v["bytes"],
                           "dtype": v["dtype"]} for k, v in agg.items()},
        "quantized_modules": quant_modules,
        "dropped_bytes": dropped_bytes,
        "expected_output_bytes": out_bytes,
        "src_bytes": src_bytes,
        "fp8_scale_grid_anomalies": [
            {"name": n, "scale_shape": list(s), "weight_shape": list(w)}
            for n, s, w in fp8_anomalies],
    }, indent=2))

    probe_mxfp4(args)
    probe_fp8(args)


def probe_mxfp4(args):
    """Empirical determination of the mxfp4 nibble order + scale semantics."""
    from safetensors import safe_open
    name = args.probe_mxfp4
    shard = None
    idx_path = os.path.join(args.src, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        wm = json.load(open(idx_path))["weight_map"]
        shard = wm.get(name + ".weight")
    if shard is None:
        for s in shard_list(args.src):
            h, _ = read_shard_header(os.path.join(args.src, s))
            if name + ".weight" in h:
                shard = s
                break
    log(f"---- mxfp4 format probe on {name} ({shard}) ----")
    with safe_open(os.path.join(args.src, shard), framework="pt") as f:
        w = f.get_tensor(name + ".weight")        # U8 [R, C/2]
        s = f.get_tensor(name + ".weight_scale")  # U8 [R, C/32]
    assert w.dtype == torch.uint8 and s.dtype == torch.uint8
    R, Ch = w.shape
    C = 2 * Ch
    log(f"  packed U8 {tuple(w.shape)} -> logical [{R}, {C}]; "
        f"scale U8 {tuple(s.shape)} (1 byte per 32-in group)")

    sb = s.reshape(-1)
    log(f"  scale byte stats: min={int(sb.min())} max={int(sb.max())} "
        f"mean={sb.float().mean():.2f} "
        f"percentiles(1,50,99)={[int(x) for x in torch.quantile(sb.float(), torch.tensor([0.01, 0.5, 0.99])).tolist()]}")
    exp_lo, exp_hi = 2.0 ** (int(sb.min()) - 127), 2.0 ** (int(sb.max()) - 127)
    log(f"  e8m0 reading 2^(s-127): scale range [{exp_lo:.3e}, {exp_hi:.3e}]"
        f"  -> max |w| per group up to 6*scale = [{6 * exp_lo:.3e}, {6 * exp_hi:.3e}]")
    log(f"  alternative readings: raw byte as float scale would be "
        f"[{int(sb.min())}, {int(sb.max())}] (implausible); "
        f"2^(s-63) would give scales ~[{2.0 ** (int(sb.min()) - 63):.3e}, ...] "
        f"(implausible for LLM weights).")

    # decode both nibble orders and compare group statistics
    lut = torch.tensor(FP4_VALUES)
    lo = (w & 0x0F).long()
    hi = (w >> 4).long()
    stats = {}
    for order in ("low-first", "high-first"):
        a, b = (lo, hi) if order == "low-first" else (hi, lo)
        vals = torch.empty(R, C, dtype=torch.float32)
        vals[:, 0::2] = lut[a]
        vals[:, 1::2] = lut[b]
        exp = (s.to(torch.int64) - 127).repeat_interleave(32, 1).float()
        vals = torch.ldexp(vals, exp)
        # e2m1 code histogram
        codes = torch.zeros(16, dtype=torch.long)
        codes.scatter_add_(0, lo.reshape(-1), torch.ones_like(lo.reshape(-1)))
        codes.scatter_add_(0, hi.reshape(-1), torch.ones_like(hi.reshape(-1)))
        sign_frac = float((hi >= 8).float().mean() * 0.5 +
                          (lo >= 8).float().mean() * 0.5)
        # per-group max-fit: max|x| == 6 * 2^(s-127)?
        gmax = vals.view(R, C // 32, 32).abs().amax(-1)
        ref = 6.0 * torch.pow(2.0, (s.float() - 127.0))
        fit = (torch.abs(gmax - ref) <= 0.51 * ref / 6 * 1.0).float().mean()
        fit2 = (gmax / ref.clamp(min=1e-30)).median()
        stats[order] = {
            "std": float(vals.std()), "absmax": float(vals.abs().max()),
            "code_hist": codes.tolist(), "sign_frac": sign_frac,
            "group_max_eq_6s_frac": float(fit), "group_max_ratio_median": float(fit2),
        }
        log(f"  decode {order}: std={stats[order]['std']:.5f} "
            f"absmax={stats[order]['absmax']:.4f} sign_frac={sign_frac:.3f} "
            f"groups with max|w|==6*2^(s-127): {float(fit):.3f} "
            f"(median max/(6s) = {float(fit2):.3f})")
    log(f"  e2m1 code histogram (shared by both orders): "
        f"{stats['low-first']['code_hist']}")
    log("  NOTE: both nibble orders yield the SAME value multiset per "
        "32-group (they only swap the two elements of each byte pair), so "
        "value statistics cannot separate them; the order is fixed by the "
        "reference decode.")
    log("  nibble-order ground truth: transformers/integrations/mxfp4.py "
        "_convert_moe_packed_tensors (copied from GPT-OSS repo and vLLM): "
        "idx_lo=(blk & 0x0F) -> sub[:, 0::2], idx_hi=(blk >> 4) -> sub[:, 1::2], "
        "torch.ldexp(sub, scale-127).")
    # torch float4_e2m1fn_x2 pack-direction cross-check (best effort)
    try:
        pair = torch.tensor([0.5, 1.0], dtype=torch.float32)
        pk = pair.to(torch.float4_e2m1fn_x2).view(torch.uint8)
        log(f"  torch.float4_e2m1fn_x2 cross-check: [0.5, 1.0] packs to byte(s) "
            f"{[int(x) for x in pk.tolist()]}  (0x21 => low nibble holds 0.5 "
            f"first => low-nibble-first)")
    except Exception as e:
        log(f"  torch.float4_e2m1fn_x2 cross-check unavailable on this build "
            f"({type(e).__name__}: {e})")
    log("  VERDICT: e2m1 LUT [0,0.5,1,1.5,2,3,4,6,-0,-0.5,...,-6] x "
        "2^(scale-127); LOW nibble = first element of the byte pair. "
        "Implemented accordingly in dequant_mxfp4().")


def probe_fp8(args):
    """Probe the fp8 128x128 block-scale grid semantics on one dense tensor."""
    from safetensors import safe_open
    name = args.probe_fp8
    shard = None
    idx_path = os.path.join(args.src, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        wm = json.load(open(idx_path))["weight_map"]
        shard = wm.get(name + ".weight")
    if shard is None:
        return log(f"  (fp8 probe: {name}.weight not found)")
    log(f"---- fp8 block-scale probe on {name} ({shard}) ----")
    with safe_open(os.path.join(args.src, shard), framework="pt") as f:
        w = f.get_tensor(name + ".weight")
        s = f.get_tensor(name + ".weight_scale_inv")
    R, C = w.shape
    log(f"  weight F8_E4M3 {tuple(w.shape)}, scale_inv F32 {tuple(s.shape)}, "
        f"R/128={R / 128}, C/128={C / 128}")
    wf = w.to(torch.float32)
    bs = 128
    nbr, nbc = (R + bs - 1) // bs, (C + bs - 1) // bs
    use = torch.zeros(nbr, nbc)
    for b in range(min(nbr, s.shape[0])):
        for j in range(nbc):
            use[b, j] = wf[b * bs:(b + 1) * bs,
                           j * bs:(j + 1) * bs].abs().max() / E4M3_MAX
    sat = float((use >= 0.999).float().mean())
    log(f"  code-usage amax(code)/448 under global 128x128 tiles: "
        f"frac==1.0 is {sat:.3f} (1.000 => scale = amax/448 per tile)")
    real = dequant_fp8_block(w, s)
    log(f"  dequant (code * scale_inv[row//128, col//128], extra rows cropped): "
        f"std={float(real.float().std()):.5f} absmax={float(real.float().abs().max()):.4f}")
    if s.shape[0] * bs != R or s.shape[1] * bs != C:
        log(f"  FORMAT SURPRISE: scale grid {tuple(s.shape)} does not equal "
            f"[R/128, C/128] = [{R // 128}, {C // 128}]; extra rows/cols are "
            f"cropped (full-attn qkv scale grid was computed on 13824 = "
            f"72 heads x 192 rows while the weight stores v at v_head_dim=128)")


# --------------------------------------------------------------------------
# convert
# --------------------------------------------------------------------------

def marker_path(dst, shard):
    return os.path.join(dst, DONE_DIR, shard + ".json")


def shard_done(dst, shard):
    mp = marker_path(dst, shard)
    fp = os.path.join(dst, shard)
    if not (os.path.exists(mp) and os.path.exists(fp)):
        return False
    try:
        info = json.load(open(mp))
    except Exception:
        return False
    return info.get("size") == os.path.getsize(fp)


def convert_shard(src, dst, shard, t0):
    spath = os.path.join(src, shard)
    dtmp = os.path.join(dst, shard + ".partial")
    dfinal = os.path.join(dst, shard)
    header, base = read_shard_header(spath)
    plan = plan_shard(header)
    entries = out_entries_for(plan)
    src_fd = os.open(spath, os.O_RDONLY)
    stats = {"quantized_modules": 0, "copied_tensors": 0,
             "dropped_tensors": 0, "src_bytes": 0, "dst_bytes": 0}
    ei = 0  # running index into entries

    def writer(fd):
        nonlocal ei
        for name, entry, kind, mod in plan:
            b, e = entry["data_offsets"]
            stats["src_bytes"] += e - b
            if kind == "drop":
                stats["dropped_tensors"] += 1
                continue
            if kind in ("expert_s", "fp8_s"):
                continue
            if kind == "copy":
                copy_region(src_fd, fd, base + b, e - b)
                stats["dst_bytes"] += e - b
                stats["copied_tensors"] += 1
                ei += 1
                continue
            # expert_w / fp8_w: dequant to fp16, requant to CT int4 g32
            w = read_tensor_region(spath, entry, base)
            if kind == "expert_w":
                sname = mod + ".weight_scale"
                sc = read_tensor_region(spath, header[sname], base)
                fp16 = dequant_mxfp4(w, sc, logical_shape(kind, entry))
                del sc
            else:
                sname = mod + ".weight_scale_inv"
                sc = read_tensor_region(spath, header[sname], base)
                fp16 = dequant_fp8_block(w, sc)
                del sc
            qt = quantize_weight(fp16)
            for suf in ("weight_packed", "weight_scale", "weight_shape",
                        "weight_zero_point"):
                t, _dt = qt[suf]
                buf = tensor_payload(t)
                write_bytes(fd, buf)
                stats["dst_bytes"] += len(buf)
                ei += 1
            stats["quantized_modules"] += 1
            del w, fp16, qt
            gc.collect()
            if stats["quantized_modules"] % 50 == 0:
                log(f"  {shard}: {stats['quantized_modules']} modules "
                    f"({(time.time() - t0) / 60:.1f} min)")
    assert ei == 0
    write_shard_stream(dtmp, entries, writer)
    assert ei == len(entries), \
        f"writer emitted {ei} tensor payloads, header lists {len(entries)}"
    os.close(src_fd)
    os.replace(dtmp, dfinal)
    stats["size"] = os.path.getsize(dfinal)
    stats["entries"] = len(entries)
    return stats


def cmd_convert(args):
    src, dst = args.src, args.dst
    os.makedirs(dst, exist_ok=True)
    os.makedirs(os.path.join(dst, DONE_DIR), exist_ok=True)
    shards = shard_list(src)
    if args.only_shard:
        shards = [s for s in shards if args.only_shard in s]
        if not shards:
            sys.exit(f"no shard matching {args.only_shard!r}")
    t0 = time.time()
    todo = []
    for shard in shards:
        if not os.path.exists(os.path.join(src, shard)):
            log(f"skip missing source shard {shard}")
            continue
        if shard_done(dst, shard) and not args.redo:
            log(f"skip {shard} (done)")
            continue
        todo.append(shard)
    log(f"convert: {len(todo)} shard(s) to do -> {dst}")
    for i, shard in enumerate(todo):
        log(f"[{i + 1}/{len(todo)}] {shard}")
        stats = convert_shard(src, dst, shard, t0)
        mp = marker_path(dst, shard)
        tmp = mp + ".tmp"
        with open(tmp, "w") as f:
            json.dump(stats, f, indent=1)
        os.replace(tmp, mp)
        log(f"  done: {stats['quantized_modules']} modules quantized, "
            f"{stats['copied_tensors']} copied, {stats['dropped_tensors']} "
            f"dropped, {stats['size'] / 2**30:.2f} GiB out")
    if args.only_shard and not args.finalize_only:
        log("single-shard mode: skipping config/index finalize "
            "(run `convert --finalize-only` after all shards are done)")
        return
    finalize(dst, src, t0)


def finalize(dst, src, t0):
    shards = shard_list(src)
    done = [s for s in shards if shard_done(dst, s)]
    missing = [s for s in shards if s not in done]
    if missing:
        log(f"finalize: {len(missing)} shard(s) still missing "
            f"(e.g. {missing[:3]}); writing config.json but NOT the index")
    # config.json: text-only + CT quantization_config
    ignore = compute_ignore_list(src)
    cfg = build_config(src, ignore)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    log(f"config.json written ({len(ignore)} ignore entries, "
        f"model_type={cfg.get('model_type')}, "
        f"architectures={cfg.get('architectures')})")
    if missing:
        return
    # rebuild index from actual dst shard headers
    weight_map = {}
    total = 0
    for shard in done:
        header, _b = read_shard_header(os.path.join(dst, shard))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            weight_map[name] = shard
            total += entry["data_offsets"][1] - entry["data_offsets"][0]
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total},
                   "weight_map": weight_map}, f, indent=2)
        f.write("\n")
    log(f"index rebuilt: {len(weight_map)} tensors, total {total / 2**30:.2f} GiB")
    # hardlink/copy all non-model files
    skip = {"config.json", "model.safetensors.index.json", MANIFEST_NAME}
    for fn in sorted(os.listdir(src)):
        if fn in skip or fn.endswith(".safetensors"):
            continue
        sfn = os.path.join(src, fn)
        dfn = os.path.join(dst, fn)
        if not os.path.isfile(sfn) or os.path.islink(sfn):
            continue
        if os.path.exists(dfn) and os.path.getsize(dfn) == os.path.getsize(sfn):
            continue
        if os.path.exists(dfn):
            os.remove(dfn)
        try:
            os.link(sfn, dfn)
        except OSError:
            shutil.copy2(sfn, dfn)
    # aggregate manifest
    man = {"tool": "mimo_convert_int4", "src": src, "dst": dst,
           "finished": True, "shards": {}}
    for s in done:
        man["shards"][s] = json.load(open(marker_path(dst, s)))
    with open(os.path.join(dst, MANIFEST_NAME + ".tmp"), "w") as f:
        json.dump(man, f, indent=1)
    os.replace(os.path.join(dst, MANIFEST_NAME + ".tmp"),
               os.path.join(dst, MANIFEST_NAME))
    log(f"convert DONE in {(time.time() - t0) / 60:.1f} min")


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------

def quality_metrics(orig_fp16, dq_fp16):
    o = orig_fp16.to(torch.float32)
    d = dq_fp16.to(torch.float32)
    adiff = (d - o).abs()
    eps = 1e-3 * o.abs().max().clamp(min=1e-12)
    rel_el = adiff / (o.abs() + eps)
    l2rel = (adiff.square().sum().sqrt()
             / o.square().sum().sqrt().clamp(min=1e-30))
    cos = torch.nn.functional.cosine_similarity(
        o.flatten().double(), d.flatten().double(), dim=0)
    amax = o.abs().max()
    sig = o.abs() >= 0.05 * amax
    return {
        "cosine": cos.item(),
        "l2_rel_err": l2rel.item(),
        "max_el_rel_err": rel_el.max().item(),
        "mean_el_rel_err": rel_el.mean().item(),
        "mean_el_rel_err_sig": (rel_el[sig].mean().item()
                                if int(sig.sum()) else 0.0),
        "frac_significant": sig.float().mean().item(),
        "mean_abs_err": adiff.mean().item(),
        "rms_err": adiff.square().mean().sqrt().item(),
        "rms_rel_to_absmax": (adiff.square().mean().sqrt()
                              / amax.clamp(min=1e-30)).item(),
        "w_abs_max": amax.item(),
    }


def src_grid_halfstep(fp16_vals, kind, scale=None):
    """Half-step of the SOURCE quant grid -- a proxy for the source format's
    own quantization error.  Returns (mean absolute half-step,
    mean relative half-step) over nonzero-magnitude codes."""
    v = fp16_vals.to(torch.float32).abs().reshape(-1)
    if kind == "expert_w":
        # e2m1 codes: |c| in {0,.5,1,1.5,2,3,4,6}; spacing to next-higher
        # code is {.5,.5,.5,.5,1,1,2,2} * 2^(s-127)
        sc = scale.reshape(-1).abs()
        cmag = (v / sc.clamp(min=1e-30)).clamp(max=6.0)
        half_abs = torch.zeros_like(v)
        covered = torch.zeros_like(v, dtype=torch.bool)
        for c, sp in ((0.5, 0.5), (1.0, 0.5), (1.5, 0.5), (2.0, 1.0),
                      (3.0, 1.0), (4.0, 2.0), (6.0, 2.0)):
            m = torch.isclose(cmag, torch.tensor(c), rtol=1e-3, atol=1e-6)
            half_abs = torch.where(m, sp / 2 * sc, half_abs)
            covered |= m
        half_abs = half_abs[covered]
        v = v[covered]
    else:
        # e4m3: 3 mantissa bits -> half-ULP rel 2^-4 on normals
        half_abs = v * (2.0 ** -4)
    return float(half_abs.mean()), float((half_abs / v.clamp(min=1e-30)).mean())


def cmd_verify(args):
    import random
    from safetensors import safe_open
    src, dst = args.src, args.dst
    idx_path = os.path.join(src, "model.safetensors.index.json")
    wm = json.load(open(idx_path))["weight_map"]
    if args.only_shard:
        wm = {n: s for n, s in wm.items() if args.only_shard in s}
    rng = random.Random(args.seed)

    by_cat = {"expert_w": [], "fp8_w": [], "copy": []}
    for name in wm:
        if is_dropped(name):
            continue
        kind = classify(name, set(wm))
        if kind == "expert_w":
            by_cat["expert_w"].append(name)
        elif kind == "fp8_w":
            by_cat["fp8_w"].append(name)
        elif kind == "copy" and name.endswith((".weight", ".bias")):
            by_cat["copy"].append(name)
    per = max(1, args.n // len(by_cat))
    sample = []
    for cat in sorted(by_cat):
        sample += [(cat, n) for n in rng.sample(by_cat[cat], min(per, len(by_cat[cat])))]
    log(f"verify: sampling {len(sample)} tensors "
        f"({per} per category) seed={args.seed}")

    reports = []
    for cat, name in sample:
        shard = wm[name]
        if not shard_done(dst, shard):
            log(f"  skip {name}: output shard {shard} not converted yet")
            continue
        with safe_open(os.path.join(src, shard), framework="pt") as fs, \
             safe_open(os.path.join(dst, shard), framework="pt") as fd:
            if cat == "copy":
                a = tensor_payload(fs.get_tensor(name))
                b = tensor_payload(fd.get_tensor(name))
                ok = a == b
                log(f"  [copy] {name}: bitwise_identical={ok}")
                reports.append({"module": name, "cat": cat,
                                "bitwise_identical": ok})
                assert ok, f"copy-through mismatch on {name}"
                continue
            mod = EXPERT_RE.match(name).group("mod") if cat == "expert_w" \
                else name[:-len(".weight")]
            w = fs.get_tensor(name)
            if cat == "expert_w":
                lsh = (w.shape[0], 2 * w.shape[1])
                s_u8 = fs.get_tensor(mod + ".weight_scale")
                src_fp16 = dequant_mxfp4(w, s_u8, lsh)
                scale_f = torch.ldexp(
                    torch.ones_like(s_u8, dtype=torch.float32),
                    s_u8.to(torch.int32) - 127).repeat_interleave(32, 1)
                grid_abs, grid_rel = src_grid_halfstep(
                    src_fp16, cat, scale=scale_f)
            else:
                lsh = tuple(w.shape)
                src_fp16 = dequant_fp8_block(
                    w, fs.get_tensor(mod + ".weight_scale_inv"))
                grid_abs, grid_rel = src_grid_halfstep(src_fp16, cat)
            packed = fd.get_tensor(mod + ".weight_packed")
            zp = fd.get_tensor(mod + ".weight_zero_point")
            sc = fd.get_tensor(mod + ".weight_scale")
            shp = tuple(fd.get_tensor(mod + ".weight_shape").tolist())
            assert shp == lsh, f"{name}: shape {shp} vs {lsh}"
            dq = dequant_ct(packed, zp, sc, shp)
            m = quality_metrics(src_fp16, dq)
            m.update({"module": mod, "cat": cat,
                      "src_grid_halfstep_rel": grid_rel,
                      "src_grid_halfstep_abs": grid_abs})
            reports.append(m)
            log(f"  [{cat}] {mod}: cos={m['cosine']:.6f} "
                f"l2rel={m['l2_rel_err']:.5f} "
                f"max_rel={m['max_el_rel_err']:.4f} "
                f"mean_rel={m['mean_el_rel_err']:.5f} "
                f"mean_rel_sig={m['mean_el_rel_err_sig']:.5f} "
                f"rms/absmax={m['rms_rel_to_absmax']:.5f} "
                f"mean_abs_err={m['mean_abs_err']:.2e} "
                f"(src grid half-step abs {grid_abs:.2e} rel {grid_rel:.4f})")
        del w
        gc.collect()

    quant = [r for r in reports if r.get("cat") in ("expert_w", "fp8_w")]
    if quant:
        n = len(quant)
        agg = {
            "n": n,
            "cosine_min": min(r["cosine"] for r in quant),
            "l2_rel_err_max": max(r["l2_rel_err"] for r in quant),
            "mean_rel_err_mean": sum(r["mean_el_rel_err"] for r in quant) / n,
            "mean_rel_err_sig_mean":
                sum(r["mean_el_rel_err_sig"] for r in quant) / n,
            "max_rel_err_max": max(r["max_el_rel_err"] for r in quant),
            "rms_rel_to_absmax_mean":
                sum(r["rms_rel_to_absmax"] for r in quant) / n,
            "mean_abs_err_mean": sum(r["mean_abs_err"] for r in quant) / n,
            "src_grid_halfstep_rel_mean":
                sum(r["src_grid_halfstep_rel"] for r in quant) / n,
            "src_grid_halfstep_abs_mean":
                sum(r["src_grid_halfstep_abs"] for r in quant) / n,
        }
        log("aggregate (quantized tensors): " + json.dumps(agg, indent=2))
        ok = agg["mean_rel_err_sig_mean"] < args.target_rel
        ok_rms = agg["rms_rel_to_absmax_mean"] < args.target_rel
        log(f"target rel err < {args.target_rel}: mean_el_rel_sig "
            f"{'PASS' if ok else 'FAIL'} ({agg['mean_rel_err_sig_mean']:.5f}); "
            f"rms/absmax {'PASS' if ok_rms else 'FAIL'} "
            f"({agg['rms_rel_to_absmax_mean']:.5f}); "
            f"mean_el_rel over all els incl. near-zero: "
            f"{agg['mean_rel_err_mean']:.5f}")
        ratio = (agg["mean_abs_err_mean"]
                 / max(agg["src_grid_halfstep_abs_mean"], 1e-30))
        log(f"int4 added error vs source grid step (absolute): "
            f"{agg['mean_abs_err_mean']:.3e} vs "
            f"{agg['src_grid_halfstep_abs_mean']:.3e}  ratio={ratio:.3f} "
            f"-> {'comparable or better' if ratio <= 1.0 else 'larger than source grid step'}")
        with open(args.report, "w") as f:
            json.dump({"aggregate": agg, "per_tensor": reports}, f, indent=2)
        log(f"wrote {args.report}")
        if not ok and not ok_rms:
            sys.exit(2)
    else:
        log("no quantized tensors sampled (shards not converted?)")


# --------------------------------------------------------------------------
# drop-modules
# --------------------------------------------------------------------------

def cmd_drop_modules(args):
    src = args.src
    ignore = compute_ignore_list(src)
    cfg = build_config(src, ignore)
    out = args.out or os.path.join(args.dst, "config.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    log(f"wrote {out}: architectures={cfg['architectures']} "
        f"model_type={cfg['model_type']} "
        f"removed=[vision_config, audio_config, processor_config] "
        f"quantization_config=compressed-tensors group_0 "
        f"({len(ignore)} ignore entries)")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "convert", "verify", "drop-modules"):
        sp = sub.add_parser(name)
        sp.add_argument("--src", default=SRC_DIR)
        sp.add_argument("--dst", default=DST_DIR)
    sub.choices["inspect"].add_argument(
        "--probe-mxfp4", default="model.layers.1.mlp.experts.0.gate_proj")
    sub.choices["inspect"].add_argument(
        "--probe-fp8", default="model.layers.0.self_attn.qkv_proj")
    sub.choices["convert"].add_argument("--only-shard", default=None,
                                        help="substring of one shard filename")
    sub.choices["convert"].add_argument("--finalize-only", action="store_true")
    sub.choices["convert"].add_argument("--redo", action="store_true")
    sub.choices["verify"].add_argument("-n", type=int, default=12)
    sub.choices["verify"].add_argument("--seed", type=int, default=20261002)
    sub.choices["verify"].add_argument("--only-shard", default=None,
                                       help="restrict sampling to one shard")
    sub.choices["verify"].add_argument("--target-rel", type=float, default=3e-2)
    sub.choices["verify"].add_argument(
        "--report", default="/data/tmp/mimo_int4_verify_report.json")
    sub.choices["drop-modules"].add_argument("--out", default=None)
    args = p.parse_args()
    torch.set_num_threads(min(32, os.cpu_count() or 8))
    if args.cmd == "convert" and args.finalize_only:
        finalize(args.dst, args.src, time.time())
        return
    {"inspect": cmd_inspect, "convert": cmd_convert,
     "verify": cmd_verify, "drop-modules": cmd_drop_modules}[args.cmd](args)


if __name__ == "__main__":
    main()
