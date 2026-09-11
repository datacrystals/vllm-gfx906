# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

# GLM53-PORT: vendored from upstream vLLM main. The upstream implementation is
# a Triton kernel using launch_pdl (unavailable in this fork's gfx906 triton);
# replaced with an equivalent pure-torch scatter here.


def scatter_states(
    state: torch.Tensor,
    src: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    """Scatter ``src`` rows into ``state`` at ``indices`` (in place).

    Equivalent to ``state[indices] = src`` but non-atomic and bandwidth-bound,
    since mamba cache slots are unique per sequence. ``gather_initial_states``
    is the read-side counterpart.
    """
    assert state.ndim >= 2
    assert indices.ndim == 1
    assert indices.device == state.device
    assert src.shape[1:] == state.shape[1:]
    assert src.shape[0] == indices.shape[0]
    assert indices.dtype in (torch.int32, torch.int64)

    state[indices.to(torch.int64)] = src.to(state.dtype)
