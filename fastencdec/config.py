import math
import platform
from dataclasses import dataclass, field

import torch
from transformers import AutoConfig, GenerationConfig, PretrainedConfig


@dataclass
class Config:
    """Engine configuration shared by the scheduler, block manager, and runner."""

    model: str
    max_num_seqs: int = 32
    dtype: torch.dtype = torch.float32
    compile: bool = True
    mm_cache_size: int = 256  # image embeddings kept across calls; 0 disables
    num_blocks: int | None = None  # None: enough for max_num_seqs full-length sequences
    block_size: int = field(init=False)
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
        if self.num_blocks is None:
            self.num_blocks = self.max_num_seqs * math.ceil(
                self.text_config.max_position_embeddings / self.block_size
            )
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
