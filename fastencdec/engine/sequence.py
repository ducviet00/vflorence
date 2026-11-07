import hashlib
from dataclasses import dataclass, field

import torch

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
    sampling: SamplingParams
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
            h = hashlib.sha256(f"{tuple(px.shape)}{px.dtype}".encode())
            h.update(px.view(torch.uint8).numpy())
            self.mm_hash = h.hexdigest()

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    @property
    def num_generated(self) -> int:
        return len(self.token_ids) - len(self.decoder_token_ids)

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[len(self.decoder_token_ids) :]
