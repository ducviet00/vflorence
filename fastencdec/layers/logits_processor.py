import torch

_NEG_INF = float("-inf")


def _valid(tokens, vocab_size: int) -> list[int]:
    if tokens is None:
        return []
    ids = [tokens] if isinstance(tokens, int) else tokens
    return [t for t in ids if 0 <= t < vocab_size]


def _repetition_penalty(scores: torch.Tensor, input_ids: torch.Tensor, penalty: float):
    ids = input_ids.clamp_min(0)
    tok = scores.gather(1, ids)
    tok = torch.where(tok < 0, tok * penalty, tok / penalty)
    if (input_ids < 0).any():
        b, t = torch.where(input_ids >= 0)
        scores[b, ids[b, t]] = tok[b, t]
    else:
        scores.scatter_(1, ids, tok)


def _no_repeat_ngram(scores: torch.Tensor, input_ids: torch.Tensor, n: int):
    cur_len = input_ids.shape[1]
    if cur_len < n:
        return
    grams = input_ids.unfold(1, n, 1)
    match = (grams[..., :-1] == input_ids[:, cur_len - n + 1 :].unsqueeze(1)).all(-1)
    b, g = torch.where(match & (grams[..., -1] >= 0))
    if b.numel():
        scores[b, grams[b, g, -1]] = _NEG_INF


class LogitsProcessor:
    def __init__(self, params, *, vocab_size: int, max_length: int, begin_index: int):
        self.params = params
        self.max_length = max_length
        self.begin_index = begin_index
        self.eos = _valid(sorted(params.eos_ids()), vocab_size)
        self.suppress = _valid(params.suppress_tokens, vocab_size)
        self.begin_suppress = _valid(params.begin_suppress_tokens, vocab_size)
        self.forced_bos = _valid(params.forced_bos_token_id, vocab_size)
        self.forced_eos = _valid(params.forced_eos_token_id, vocab_size)

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor, lengths=None
    ) -> torch.Tensor:
        """Apply every enabled rule in place, in transformers' order.

        ``input_ids`` is [batch, width], left-padded with -1; ``lengths`` holds each
        row's true length (default: every row is ``width`` long).
        """
        p = self.params
        if lengths is None:
            lengths = torch.full((input_ids.shape[0],), input_ids.shape[1])
        if p.repetition_penalty != 1.0:
            _repetition_penalty(scores, input_ids, p.repetition_penalty)
        if p.no_repeat_ngram_size > 0:
            _no_repeat_ngram(scores, input_ids, p.no_repeat_ngram_size)

        def rows(mask: torch.Tensor) -> torch.Tensor:
            return mask.nonzero()  # [k, 1]: indexes rows, broadcasts over columns

        if p.min_length > 0 and self.eos:
            scores[rows(lengths < p.min_length), self.eos] = _NEG_INF
        if self.forced_bos:
            _force(scores, rows(lengths == 1), self.forced_bos)
        if self.forced_eos:
            _force(scores, rows(lengths == self.max_length - 1), self.forced_eos)
        if self.suppress:
            scores[:, self.suppress] = _NEG_INF
        if self.begin_suppress:
            scores[rows(lengths == self.begin_index), self.begin_suppress] = _NEG_INF
        return scores


def _force(scores: torch.Tensor, rows: torch.Tensor, tokens: list[int]) -> None:
    """Make ``tokens`` the only allowed choice in ``rows``."""
    scores[rows[:, 0]] = _NEG_INF
    scores[rows, tokens] = 0.0
