from collections import deque

from ..config import Config
from .block_manager import BlockManager
from .sequence import Sequence


class Scheduler:
    """Continuous-batching scheduler that admits requests up to max_num_seqs."""

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.block_manager = BlockManager(config.num_blocks, config.block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        if seq.sampling.num_beams > self.max_num_seqs:
            raise ValueError(
                f"num_beams ({seq.sampling.num_beams}) exceeds "
                f"max_num_seqs ({self.max_num_seqs})"
            )
        self.waiting.append(seq)

    def remove(self, seq: Sequence) -> None:
        self.running.remove(seq)
        self.block_manager.deallocate(seq)

    def schedule(self) -> tuple[list[Sequence], list[tuple[int, int]]]:
        """Admit waiting sequences up to max_num_seqs, allocate KV blocks, and return (running, copies)."""
        used = sum({s.request_id: s.sampling.num_beams for s in self.running}.values())
        while (
            self.waiting
            and used + self.waiting[0].sampling.num_beams <= self.max_num_seqs
        ):
            seq = self.waiting.popleft()
            self.running.append(seq)
            used += seq.sampling.num_beams

        seqs = list(self.running)
        if (
            self.block_manager.required_new_blocks(seqs)
            > self.block_manager.num_free_blocks
        ):
            raise RuntimeError("KV cache is full; increase num_blocks")
        copies = []
        for seq in seqs:
            if (copy := self.block_manager.ensure_capacity(seq)) is not None:
                copies.append(copy)
        return seqs, copies
