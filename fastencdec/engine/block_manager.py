from collections import deque


class BlockManager:
    """Paged KV-cache block allocator with reference counting and copy-on-write."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.reset()

    def reset(self) -> None:
        """Release every block after an aborted batch."""
        self.ref_counts = [0] * self.num_blocks
        self.free_block_ids = deque(range(self.num_blocks))

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def _allocate_block(self) -> int:
        if not self.free_block_ids:
            raise RuntimeError("KV cache is full; increase num_blocks")
        block_id = self.free_block_ids.popleft()
        self.ref_counts[block_id] = 1
        return block_id

    def required_new_blocks(self, seqs) -> int:
        """Count new blocks and copy-on-write blocks needed to append 1 token per sequence."""
        # The last holder of a shared block writes in place; every other holder copies.
        required, remaining = 0, {}
        for seq in seqs:
            idx = seq.num_cached_tokens // self.block_size
            if idx >= len(seq.block_table):
                required += 1
            else:
                block_id = seq.block_table[idx]
                refs = remaining.get(block_id, self.ref_counts[block_id])
                if refs > 1:
                    required += 1
                    remaining[block_id] = refs - 1
        return required

    def ensure_capacity(self, seq) -> tuple[int, int] | None:
        """Make room for seq's next token: a new block, or a private copy of a shared one.

        Returns the (src, dst) block pair the caller must copy, or None.
        """
        idx = seq.num_cached_tokens // self.block_size
        if idx >= len(seq.block_table):
            seq.block_table.append(self._allocate_block())
            return None
        block_id = seq.block_table[idx]
        if self.ref_counts[block_id] == 1:
            return None
        new_block_id = self._allocate_block()
        self.ref_counts[block_id] -= 1
        seq.block_table[idx] = new_block_id
        return block_id, new_block_id

    def fork(self, parent, child) -> None:
        """Share parent's block table with child by incrementing block refcounts."""
        child.block_table = list(parent.block_table)
        for block_id in child.block_table:
            self.ref_counts[block_id] += 1

    def deallocate(self, seq) -> None:
        """Decrement refcounts for all blocks in seq and free any that reach zero."""
        for block_id in reversed(seq.block_table):
            self.ref_counts[block_id] -= 1
            if self.ref_counts[block_id] == 0:
                self.free_block_ids.append(block_id)
        seq.block_table.clear()
