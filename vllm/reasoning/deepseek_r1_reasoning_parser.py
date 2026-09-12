# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


class DeepSeekR1ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for DeepSeek R1 model.

    The DeepSeek R1 model uses <think>...</think> tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        # GLM53-PORT: pure text-level split on the FIRST "</think>".
        #
        # The token-ID driven base implementation leaks/mangles text whenever
        # the model emits "</think>" as composed text tokens (e.g. "</thi" +
        # "nk>"), when multi-token deltas straddle the boundary, or when the
        # boundary delta's text lookup misses (find() == -1 silently drops a
        # character from reasoning and slices content at +len(end)). Splitting
        # on the accumulated text is robust to every tokenization of the end
        # marker: content can never contain a literal "</think>", and no
        # reasoning characters are ever lost.
        end = self.end_token

        if end in previous_text:
            # Reasoning ended before this delta: everything new is content.
            return DeltaMessage(content=delta_text or None)

        idx = current_text.find(end)
        if idx >= 0:
            # The boundary is crossed inside current_text. Any chars of the
            # end token that already sit in previous_text (a partial suffix
            # like "</thi") were held back below and must not be re-emitted
            # as reasoning.
            overlap = 0
            for k in range(min(len(end) - 1, len(previous_text)), 0, -1):
                if previous_text.endswith(end[:k]):
                    overlap = k
                    break
            reasoning_so_far = len(previous_text) - overlap
            new_reasoning = current_text[reasoning_so_far:idx]
            new_content = current_text[idx + len(end) :]
            return DeltaMessage(
                reasoning=new_reasoning or None,
                content=new_content or None,
            )

        # Still inside reasoning. Hold back any trailing partial of the end
        # token so its fragments are never emitted as reasoning text.
        emit = delta_text
        tail = previous_text + delta_text
        for k in range(min(len(end) - 1, len(tail)), 0, -1):
            if tail.endswith(end[:k]):
                emit = emit[: len(emit) - k] if k <= len(emit) else ""
                break
        return DeltaMessage(reasoning=emit or None)
