import platform
from dataclasses import dataclass

import torch


def default_block_size() -> int:
    """Return the default KV page size (16 on ARM NEON, 32 on x86)."""
    return 16 if platform.machine().lower() in ("arm64", "aarch64") else 32


@dataclass
class Config:
    """Engine configuration shared by the scheduler, block manager, and runner."""

    num_blocks: int = 512
    block_size: int | None = None
    max_num_seqs: int = 32
    dtype: torch.dtype = torch.float32
    compile: bool = True
    compile_mm_encoder: bool = True
    mm_cache_size: int = 256  # image embeddings kept across calls; 0 disables

    def __post_init__(self) -> None:
        if self.block_size is None:
            self.block_size = default_block_size()
        if self.num_blocks <= 0 or self.block_size <= 0 or self.max_num_seqs <= 0:
            raise ValueError("num_blocks, block_size and max_num_seqs must be positive")
        if self.block_size % 16 != 0:
            raise ValueError("block_size must be a multiple of 16")
        if self.mm_cache_size < 0:
            raise ValueError("mm_cache_size must be >= 0")
        if self.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("dtype must be torch.float32 or torch.bfloat16")
