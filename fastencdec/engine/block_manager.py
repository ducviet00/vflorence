from collections import deque
from dataclasses import dataclass


@dataclass
class Block:
    ref_count: int = 0


class BlockManager:
    """Paged KV-cache block allocator with reference counting and copy-on-write."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [Block() for _ in range(num_blocks)]
        self.reset()

    def reset(self) -> None:
        """Release every block after an aborted batch."""
        for block in self.blocks:
            block.ref_count = 0
        self.free_block_ids = deque(range(self.num_blocks))

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def _allocate_block(self) -> int:
        if not self.free_block_ids:
            raise RuntimeError("KV cache is full; increase num_blocks")
        block_id = self.free_block_ids.popleft()
        self.blocks[block_id].ref_count = 1
        return block_id

    def required_new_blocks(self, seqs) -> int:
        """Count new blocks and copy-on-write blocks needed to append 1 token per sequence."""
        bs = self.block_size
        required, remaining = 0, {}
        for seq in seqs:
            idx = seq.num_cached_tokens // bs
            if idx >= len(seq.block_table):
                required += 1
            else:
                block_id = seq.block_table[idx]
                refs = remaining.get(block_id, self.blocks[block_id].ref_count)
                if refs > 1:
                    required += 1
                    remaining[block_id] = refs - 1
        return required

    def ensure_capacity(self, seq, copy_block) -> None:
        """Allocate a new block or copy-on-write a shared block for seq's next token."""
        idx = seq.num_cached_tokens // self.block_size
        if idx >= len(seq.block_table):
            seq.block_table.append(self._allocate_block())
            return
        block_id = seq.block_table[idx]
        if self.blocks[block_id].ref_count > 1:
            new_block_id = self._allocate_block()
            try:
                copy_block(block_id, new_block_id)
            except BaseException:
                self.blocks[new_block_id].ref_count = 0
                self.free_block_ids.append(new_block_id)
                raise
            self.blocks[block_id].ref_count -= 1
            seq.block_table[idx] = new_block_id

    def fork(self, parent, child) -> None:
        """Share parent's block table with child by incrementing block refcounts."""
        child.block_table = list(parent.block_table)
        for block_id in child.block_table:
            self.blocks[block_id].ref_count += 1

    def deallocate(self, seq) -> None:
        """Decrement refcounts for all blocks in seq and free any that reach zero."""
        for block_id in reversed(seq.block_table):
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                self.free_block_ids.append(block_id)
        seq.block_table.clear()
