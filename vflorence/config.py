import json
import math
import platform
from dataclasses import dataclass, field

import torch
from transformers import AutoConfig, GenerationConfig, PretrainedConfig
from transformers.utils import cached_file

from .sampling_params import SamplingParams


@dataclass
class Config:
    """Engine configuration shared by the scheduler, block manager, and runner."""

    model: str
    max_num_seqs: int = 32
    dtype: torch.dtype = torch.float32
    compile: bool = True
    mm_cache_size: int = 256  # image embeddings kept across calls; 0 disables
    # Decoder positions per sequence (its K/V entries): fits max_tokens <= max_decoder_len,
    # since the last generated token is never fed. None: the generation_config's max_tokens,
    # rounded up to whole blocks.
    max_decoder_len: int | None = None
    # Longest encoder prompt. None: image tokens + 64 text tokens for Florence-2 (641),
    # else max_position_embeddings; rounded up to whole blocks.
    max_encoder_len: int | None = None
    num_blocks: int | None = None  # None: max_num_seqs sequences of max_decoder_len
    block_size: int = field(init=False)
    num_cross_blocks: int = field(init=False)
    # From preprocessor_config.json; None for text-only checkpoints.
    image_seq_length: int | None = field(init=False)
    image_size: tuple[int, int] | None = field(init=False)
    hf_config: PretrainedConfig = field(init=False)
    text_config: PretrainedConfig = field(init=False)
    generation_config: GenerationConfig = field(init=False)

    def __post_init__(self) -> None:
        self.block_size = (
            16 if platform.machine().lower() in ("arm64", "aarch64") else 32
        )
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.mm_cache_size < 0:
            raise ValueError("mm_cache_size must be >= 0")
        if self.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("dtype must be torch.float32 or torch.bfloat16")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.text_config = self.hf_config.get_text_config()
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model)
        except OSError:
            self.generation_config = GenerationConfig()
        max_positions = self.text_config.max_position_embeddings
        # Defaults fill their last block: those slots are allocated anyway.
        bs = self.block_size
        whole_blocks = lambda n: min(
            -(-min(n, max_positions) // bs) * bs, max_positions
        )
        if self.max_decoder_len is None:
            self.max_decoder_len = whole_blocks(
                SamplingParams.from_generation_config(self.generation_config).max_tokens
            )
        preprocessor = _preprocessor_config(self.model)
        self.image_seq_length = preprocessor.get("image_seq_length")
        size = preprocessor.get("crop_size") or preprocessor.get("size")
        has_size = isinstance(size, dict) and {"height", "width"} <= size.keys()
        self.image_size = (size["height"], size["width"]) if has_size else None
        if self.max_encoder_len is None:
            # Visual tokens + text budget (Florence-2: 577 + 64); else no limit.
            self.max_encoder_len = whole_blocks(
                math.inf
                if self.image_seq_length is None
                else self.image_seq_length + _TEXT_PROMPT_BUDGET
            )
        for name in ("max_decoder_len", "max_encoder_len"):
            if not 0 < getattr(self, name) <= max_positions:
                raise ValueError(
                    f"{name} must be in [1, max_position_embeddings={max_positions}]"
                )
        # Fixed pools, allocated once: self-attn per sequence, encoder K/V per request.
        if self.num_blocks is None:
            self.num_blocks = self.max_num_seqs * math.ceil(
                self.max_decoder_len / self.block_size
            )
        self.num_cross_blocks = self.max_num_seqs * math.ceil(
            self.max_encoder_len / self.block_size
        )
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")


# Text tokens allowed next to the image in the default max_encoder_len.
# shortcut: fixed budget for task prompt + free text; pass max_encoder_len for longer text.
_TEXT_PROMPT_BUDGET = 64


def _preprocessor_config(model: str) -> dict:
    """The checkpoint's preprocessor_config.json; {} if it has none (text-only)."""
    path = cached_file(
        model, "preprocessor_config.json", _raise_exceptions_for_missing_entries=False
    )
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)
