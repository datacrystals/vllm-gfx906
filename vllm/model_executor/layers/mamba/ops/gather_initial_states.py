# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

# GLM53-PORT: vendored from upstream vLLM main. The upstream implementation is
# a Triton kernel using launch_pdl (unavailable in this fork's gfx906 triton);
# replaced with an equivalent pure-torch gather here -- this runs once per
# prefill request batch and is not latency-critical.


def gather_initial_states(
    state: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Gather dense state rows, replacing uninitialized rows with zeros."""
    assert state.ndim >= 2
    assert indices.ndim == 1 and has_initial_state.ndim == 1
    assert indices.shape == has_initial_state.shape
    assert indices.device == state.device
    assert has_initial_state.device == state.device
    assert indices.dtype in (torch.int32, torch.int64)
    assert has_initial_state.dtype == torch.bool

    safe_indices = torch.where(has_initial_state, indices, 0)
    output = state.index_select(0, safe_indices.to(torch.int64))
    output = torch.where(
        has_initial_state.view(-1, *([1] * (state.ndim - 1))),
        output,
        torch.zeros((), dtype=state.dtype, device=state.device),
    )
    return output
