import copy
from dataclasses import dataclass, field

import torch
import xxhash

from ..sampling_params import SamplingParams


@dataclass
class Request:
    input_ids: list[int]
    pixel_values: torch.Tensor | None = None


@dataclass
class RequestOut:
    input_ids: list[int]
    completion_token_ids: list[int]
    pixel_values: torch.Tensor | None = None


@dataclass
class Sequence:
    """State and paged KV-cache bookkeeping for a single decoder hypothesis."""

    request_id: int
    decoder_token_ids: list[int]
    encoder_token_ids: list[int]
    sampling_params: SamplingParams
    pixel_values: torch.Tensor | None = None
    mm_hash: str | None = (
        None  # image cache key; hashed from pixel_values if unset (forks reuse it)
    )
    token_ids: list[int] = field(init=False)
    block_table: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    cum_logprob: float = 0.0

    def __post_init__(self) -> None:
        self.decoder_token_ids = list(self.decoder_token_ids)
        self.encoder_token_ids = list(self.encoder_token_ids)
        self.token_ids = list(self.decoder_token_ids)
        if self.pixel_values is not None and self.mm_hash is None:
            px = self.pixel_values.contiguous()
            h = xxhash.xxh3_128(f"{tuple(px.shape)}{px.dtype}".encode())
            h.update(px.view(torch.uint8).numpy())
            self.mm_hash = h.hexdigest()

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def fork(self, token_id: int, cum_logprob: float = 0.0) -> "Sequence":
        child = copy.copy(self)
        child.token_ids = [*self.token_ids, token_id]
        child.block_table = []
        child.cum_logprob = cum_logprob
        return child

    @property
    def num_generated(self) -> int:
        return len(self.token_ids) - len(self.decoder_token_ids)

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[len(self.decoder_token_ids) :]

    @property
    def is_finished(self) -> bool:
        return (
            self.token_ids[-1] in self.sampling_params.eos_ids()
            or self.num_generated >= self.sampling_params.max_tokens
        )

    @property
    def score(self) -> float:
        return self.cum_logprob / (
            max(self.num_generated, 1) ** self.sampling_params.length_penalty
        )

    def to_output(self) -> RequestOut:
        return RequestOut(
            input_ids=self.encoder_token_ids,
            completion_token_ids=self.completion_token_ids,
            pixel_values=self.pixel_values,
        )
