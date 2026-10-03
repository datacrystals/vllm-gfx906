"""Env-gated hidden-state capture for MiMo-V2.6-Flash differential debug.

Usage: boot the server with VLLM_MIMO_STATE_DUMP=/data/tmp/mimo_vllm_states.pt.
On the first model forward with <= 8 tokens (VLLM_MIMO_STATE_DUMP_MAX_TOKENS),
and NOT during CUDA-graph capture / torch.compile tracing, the residual-stream
output of the embedding plus each decoder layer is recorded and saved to that
path; the forward then continues normally.  Capture happens once.

Recorded quantity per layer: hidden_states + residual (the residual stream after
the layer's MLP), which is the value comparable to HF's decoder-layer output
(HF layers return the full post-residual hidden state).  Tag "embed" is the
embedding output (pre-layer-0), tags "0".."47" are post-layer streams, tag
"final" is the final-norm output (= lm_head input).

Caveats:
  * TP: only rank 0 writes.  For this model the residual stream is replicated
    across TP ranks (o_proj uses reduce_results=True all-reduce and hidden_size
    is never sharded), so rank-0 tensors are the full [T, hidden] stream; the
    saved "meta" records tp_rank/tp_world and per-tag shapes to re-verify.
  * With cudagraph_mode=FULL only eager prefill executes this python; decode
    replays captured graphs and would not re-dump (we dump once, then done).
"""

import os

import torch


class _MimoStateDump:
    def __init__(self) -> None:
        self._done = False
        self._buf: list | None = None
        self._meta: dict | None = None

    def begin(self, num_tokens: int, positions=None) -> bool:
        if self._done or not os.environ.get("VLLM_MIMO_STATE_DUMP"):
            return False
        max_t = int(os.environ.get("VLLM_MIMO_STATE_DUMP_MAX_TOKENS", "8"))
        if num_tokens > max_t:
            return False
        try:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return False
            if torch.compiler.is_compiling():
                return False
        except Exception:
            pass
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            tp_rank = get_tensor_model_parallel_rank()
        except Exception:
            tp_rank = 0
        if tp_rank != 0:
            self._done = True
            return False
        try:
            from vllm.distributed import get_tensor_model_parallel_world_size
            tp_world = get_tensor_model_parallel_world_size()
        except Exception:
            tp_world = 1
        self._buf = []
        self._meta = {
            "num_tokens": num_tokens,
            "tp_rank": tp_rank,
            "tp_world": tp_world,
            "positions": None if positions is None else positions.detach().cpu(),
        }
        return True

    def record(self, tag, hidden_states, residual=None) -> None:
        if self._buf is None:
            return
        stream = hidden_states if residual is None else hidden_states + residual
        self._buf.append((str(tag), stream.detach().to("cpu", torch.float32)))

    def end(self) -> None:
        if self._buf is None:
            return
        path = os.environ["VLLM_MIMO_STATE_DUMP"]
        self._meta["tags"] = [t for t, _ in self._buf]
        self._meta["shapes"] = [tuple(s.shape) for _, s in self._buf]
        torch.save({"states": self._buf, "meta": self._meta}, path)
        self._done = True
        self._buf = None
        print(f"[mimo_state_dump] saved {len(self._meta['tags'])} states to {path}", flush=True)


mimo_state_dump = _MimoStateDump()
