from dataclasses import dataclass, field

import torch

from ..sampling_params import SamplingParams


@dataclass
class Sequence:
    """State and paged KV-cache bookkeeping for a single decoder hypothesis."""

    request_id: int
    decoder_prompt_ids: list[int]
    encoder_token_ids: list[int]
    sampling: SamplingParams
    pixel_values: torch.Tensor | None = None
    token_ids: list[int] = field(init=False)
    block_table: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    cum_logprob: float = 0.0

    def __post_init__(self) -> None:
        self.decoder_prompt_ids = list(self.decoder_prompt_ids)
        self.encoder_token_ids = list(self.encoder_token_ids)
        self.token_ids = list(self.decoder_prompt_ids)

    @property
    def num_generated(self) -> int:
        return len(self.token_ids) - len(self.decoder_prompt_ids)

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[len(self.decoder_prompt_ids) :]
