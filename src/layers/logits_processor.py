import torch

_NEG_INF = float("-inf")


class LogitsProcessor:
    """transformers' logits rules for one SamplingParams, applied to a whole batch."""

    def __init__(
        self, sampling_params, *, vocab_size: int, max_length: int, begin_index: int
    ):
        self.sampling_params = sampling_params
        self.max_length = max_length
        self.begin_index = begin_index
        sp = sampling_params
        # Only these two rules read the token history; the rest need lengths alone.
        self.needs_history = sp.repetition_penalty != 1.0 or sp.no_repeat_ngram_size > 0
        self.eos = _in_vocab(sorted(sp.eos_ids()), vocab_size)
        self.suppress = _in_vocab(sp.suppress_tokens, vocab_size)
        self.begin_suppress = _in_vocab(sp.begin_suppress_tokens, vocab_size)
        self.forced_bos = _in_vocab(sp.forced_bos_token_id, vocab_size)
        self.forced_eos = _in_vocab(sp.forced_eos_token_id, vocab_size)

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor, lengths=None
    ) -> torch.Tensor:
        """Apply every enabled rule to ``scores`` in place, in transformers' order.

        ``input_ids`` is [batch, width], left-padded with -1 (may be None when
        ``needs_history`` is False and ``lengths`` is given); ``lengths`` holds each
        row's true length (default: every row is ``width`` long).
        """
        sp = self.sampling_params
        if lengths is None:
            lengths = torch.full((input_ids.shape[0],), input_ids.shape[1])

        if sp.repetition_penalty != 1.0:
            _repetition_penalty(scores, input_ids, sp.repetition_penalty)
        if sp.no_repeat_ngram_size > 0:
            _no_repeat_ngram(scores, input_ids, sp.no_repeat_ngram_size)
        if sp.min_length > 0 and self.eos:
            _set(scores, lengths < sp.min_length, self.eos, _NEG_INF)
        if self.forced_bos:
            _force(scores, lengths == 1, self.forced_bos)
        if self.forced_eos:
            _force(scores, lengths == self.max_length - 1, self.forced_eos)
        if self.suppress:
            scores[:, self.suppress] = _NEG_INF
        if self.begin_suppress:
            _set(scores, lengths == self.begin_index, self.begin_suppress, _NEG_INF)
        return scores


def _in_vocab(tokens: int | list[int] | None, vocab_size: int) -> list[int]:
    """Token id(s) as a list, dropping ids outside the vocabulary."""
    if tokens is None:
        return []
    if isinstance(tokens, int):
        tokens = [tokens]
    return [t for t in tokens if 0 <= t < vocab_size]


def _repetition_penalty(scores: torch.Tensor, input_ids: torch.Tensor, penalty: float):
    """Make every token already in a row's history less likely."""
    rows, cols = torch.where(input_ids >= 0)  # skip the -1 padding
    tokens = input_ids[rows, cols]
    seen = scores[rows, tokens]
    scores[rows, tokens] = torch.where(seen < 0, seen * penalty, seen / penalty)


def _no_repeat_ngram(scores: torch.Tensor, input_ids: torch.Tensor, n: int):
    """Ban every token that would complete an n-gram already in the row's history."""
    width = input_ids.shape[1]
    if width < n:
        return
    ngrams = input_ids.unfold(1, n, 1)  # [batch, num_ngrams, n]
    last_tokens = input_ids[:, width - n + 1 :].unsqueeze(1)  # [batch, 1, n - 1]
    prefix_matches = (ngrams[..., :-1] == last_tokens).all(-1)
    rows, idx = torch.where(prefix_matches & (ngrams[..., -1] >= 0))
    scores[rows, ngrams[rows, idx, -1]] = _NEG_INF


def _set(scores: torch.Tensor, rows: torch.Tensor, tokens: list[int], value: float):
    """Set ``scores[r, t] = value`` for each row where ``rows`` is True and each t in ``tokens``."""
    scores[:, tokens] = scores[:, tokens].masked_fill(rows.unsqueeze(1), value)


def _force(scores: torch.Tensor, rows: torch.Tensor, tokens: list[int]):
    """Make ``tokens`` the only allowed choice in each row where ``rows`` is True."""
    scores[rows.nonzero().squeeze(1)] = _NEG_INF  # ~10x faster than a bool-mask index
    _set(scores, rows, tokens, 0.0)
