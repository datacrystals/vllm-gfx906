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

    def _split_reasoning(self, text: str, final: bool = False) -> tuple[str, str | None, bool]:
        """GLM53-PORT: split on the first RUN of consecutive end tokens.

        GLM-5.3-Flash sometimes emits the end marker twice (or more) in a
        row, e.g. "reasoning</think></think>reply" (observed in production
        captures). Splitting on only the first marker leaks the extra ones
        into content verbatim. Treat consecutive end tokens separated only
        by whitespace as a single boundary.

        Returns (reasoning, content); content is None when no boundary yet.
        Content that is entirely whitespace so far is reported as None: the
        whitespace may precede another end token still streaming in, so it
        is held back rather than emitted.
        """
        end = self.end_token
        first = text.find(end)
        if first < 0:
            return text, None, False
        j = first
        while True:
            m = j + len(end)
            while m < len(text) and text[m] in " \t\n\r":
                m += 1
            if text.startswith(end, m):
                j = m
            else:
                break
        content = text[j + len(end):]
        if not final:
            # Hold back a trailing partial end token: the run may continue
            # in the next delta ("...</think></thi" + "nk>").
            for k in range(min(len(end) - 1, len(content)), 0, -1):
                if content.endswith(end[:k]):
                    content = content[:-k]
                    break
        if content.strip() == "":
            content = None
        return text[:first], content, True

    def extract_reasoning(self, model_output, request):
        # Non-streaming path: same run-aware split instead of partition().
        reasoning, content, has_boundary = self._split_reasoning(
            model_output, final=True)
        if not has_boundary:
            return model_output, None
        return reasoning, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        # GLM53-PORT: pure text-level split on the first RUN of "</think>".
        #
        # The token-ID driven base implementation leaks/mangles text whenever
        # the model emits "</think>" as composed text tokens (e.g. "</thi" +
        # "nk>"), when multi-token deltas straddle the boundary, or when the
        # boundary delta's text lookup misses. Splitting on the accumulated
        # text is robust to every tokenization of the end marker: content
        # can never contain a literal "</think>", and no reasoning
        # characters are ever lost. A run of consecutive end markers (the
        # model occasionally emits "</think></think>") is consumed as a
        # single boundary so extras never leak into content.
        end = self.end_token

        prev_reasoning, prev_content, prev_boundary = self._split_reasoning(
            previous_text)
        cur_reasoning, cur_content, cur_boundary = self._split_reasoning(
            current_text)

        if cur_boundary:
            if prev_boundary:
                # Boundary crossed earlier: emit only the new content chars.
                new_content = (cur_content or "")[len(prev_content or ""):]
                return DeltaMessage(content=new_content or None)
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
            new_reasoning = cur_reasoning[reasoning_so_far:]
            return DeltaMessage(
                reasoning=new_reasoning or None,
                content=cur_content or None,
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
