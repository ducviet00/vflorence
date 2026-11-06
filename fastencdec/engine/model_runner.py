import platform
from collections import defaultdict
from contextlib import nullcontext

import torch
import vllm_cpu_attn

from ..layers import attention
from ..utils.context import set_context

_COMPILED_RUNNERS = 0

# On aarch64 with torch >= 2.15, oneDNN runs a plain (unpacked) weight fastest for
# one row, so the batch-1 decode graph is compiled without freezing (weight packing).
# Batch 1 is its own graph anyway (Dynamo's 0/1 specialization).
_UNFROZEN_SINGLE_ROW = (
    platform.machine().lower() in ("arm64", "aarch64")
    and torch.torch_version.TorchVersion(torch.__version__) >= "2.15"
)


class ModelRunner:
    """Manages KV caches, encoder caching, step preparation, and model execution."""

    def __init__(
        self,
        model,
        block_manager,
        dtype=torch.float32,
        compile=True,
        compile_mm_encoder=True,
    ):
        self.model = model.eval()
        self.block_manager = block_manager
        self.block_size = block_manager.block_size
        self.dtype = dtype
        self.decoder_layers = list(model.decoder_layers)
        self.num_heads = self.decoder_layers[0].self_attn.num_heads
        self.head_dim = self.decoder_layers[0].self_attn.head_dim
        if self.head_dim not in attention.HEAD_DIMS:
            raise ValueError(
                f"vllm-cpu-attn does not support head_dim={self.head_dim}; "
                f"must be one of {sorted(attention.HEAD_DIMS)}"
            )
        self.isa = vllm_cpu_attn.get_isa(dtype, self.block_size, self.head_dim)
        # Eager by default; _compile swaps in compiled versions.
        self.compiled = compile
        self.decode_fn = self.model
        self.encode_fn = self.model.encode
        self.image_fn = getattr(self.model, "get_image_features", None)
        if compile:
            self._compile(compile_mm_encoder)
        # Cross-attention K/V staged for the current batch, keyed by its request ids.
        self._cross_key = None
        self._cross_staged = None

        # Separate tensors per layer: views of one shared tensor break Inductor codegen.
        # zeros, not empty: the kernel reads whole blocks, and 0 * NaN from garbage
        # in a block's unused tail would poison the output.
        shape = (
            block_manager.num_blocks,
            self.num_heads,
            self.block_size,
            self.head_dim,
        )
        for layer in self.decoder_layers:
            layer.self_attn.k_cache = torch.zeros(shape, dtype=dtype)
            layer.self_attn.v_cache = torch.zeros(shape, dtype=dtype)

    def _compile(self, compile_mm_encoder: bool) -> None:
        global _COMPILED_RUNNERS
        _COMPILED_RUNNERS += 1
        # Each engine adds ~4 Dynamo cache entries; past the limit Dynamo silently
        # falls back to eager.
        torch._dynamo.config.recompile_limit = max(
            torch._dynamo.config.recompile_limit, 2 * _COMPILED_RUNNERS + 4
        )
        # Parallel C++ compile workers use GBs of RAM on small machines.
        torch._inductor.config.compile_threads = min(
            4, torch._inductor.config.compile_threads
        )
        # One dynamic graph covers every batch size / encoder length.
        self.decode_fn = torch.compile(self.model, dynamic=True)
        self.encode_fn = torch.compile(self.model.encode, dynamic=True)
        if self.image_fn is not None and compile_mm_encoder:
            self.image_fn = torch.compile(self.image_fn, dynamic=False)

    def _copy_blocks(self, copies: list[tuple[int, int]]) -> None:
        """Copy-on-write: copy KV blocks src -> dst in every decoder layer, one op per cache."""
        src, dst = (torch.tensor(ids) for ids in zip(*copies))
        for layer in self.decoder_layers:
            for cache in (layer.self_attn.k_cache, layer.self_attn.v_cache):
                cache[dst] = cache[src]

    @torch.no_grad()
    def _encode(self, seqs) -> None:
        """Batch fresh sequences by (encoder_len, image_shape) and cache cross-attn K/V."""
        groups = defaultdict(list)
        for seq in seqs:
            px_shape = (
                None if seq.pixel_values is None else tuple(seq.pixel_values.shape)
            )
            groups[(len(seq.encoder_token_ids), px_shape)].append(seq)
        for group in groups.values():
            ids = torch.tensor([s.encoder_token_ids for s in group], dtype=torch.long)
            px = (
                None
                if group[0].pixel_values is None
                else torch.stack([s.pixel_values for s in group])
            )
            with self._inductor_ctx():
                if px is None:
                    hidden = self.encode_fn(ids)
                else:
                    hidden = self.encode_fn(ids, image_embeds=self.image_fn(px))
            for layer in self.decoder_layers:
                attn = layer.encoder_attn
                shape = (len(group), ids.shape[1], attn.num_heads, attn.head_dim)
                key, value = attn.kv_proj(hidden).chunk(2, dim=-1)
                key, value = (
                    key.reshape(shape).contiguous(),
                    value.reshape(shape).contiguous(),
                )
                for j, seq in enumerate(group):
                    attn.encoder_kv_cache[seq.request_id] = (key[j], value[j])

    def clear_request(self, request_id: int) -> None:
        """Release cached encoder K/V and staged cross-attention state for a finished request."""
        for layer in self.decoder_layers:
            layer.encoder_attn.encoder_kv_cache.pop(request_id, None)
        self._cross_key = self._cross_staged = None

    def _prepare(self, seqs):
        """Allocate KV blocks and build input_ids, positions, and Context tensors for the step."""
        bs = self.block_size
        for seq in seqs:
            num_new = len(seq.token_ids) - seq.num_cached_tokens
            if num_new != 1:
                raise ValueError(
                    f"each step must feed 1 token per sequence, got {num_new}"
                )
        if (
            self.block_manager.required_new_blocks(seqs)
            > self.block_manager.num_free_blocks
        ):
            raise RuntimeError("KV cache is full; increase num_blocks")

        input_ids, positions, new_slots = [], [], []
        context_lens, seq_block_tables, copies = [], [], []
        for seq in seqs:
            if (copy := self.block_manager.ensure_capacity(seq)) is not None:
                copies.append(copy)
            pos = seq.num_cached_tokens
            input_ids.append(seq.token_ids[pos])
            positions.append(pos)
            new_slots.append(seq.block_table[pos // bs] * bs + pos % bs)
            seq.num_cached_tokens = pos + 1
            context_lens.append(pos + 1)
            seq_block_tables.append(seq.block_table)
        # Safe to copy after the loop: a dst is a fresh block, never another pair's src.
        if copies:
            self._copy_blocks(copies)

        max_blocks = max(len(t) for t in seq_block_tables)
        padded = [t + [0] * (max_blocks - len(t)) for t in seq_block_tables]
        context_lens_t = torch.tensor(context_lens, dtype=torch.int32)
        query_start_loc = torch.arange(len(seqs) + 1, dtype=torch.int32)
        self_metadata = vllm_cpu_attn.get_scheduler_metadata(
            num_reqs=len(seqs),
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim=self.head_dim,
            seq_lens=context_lens_t,
            dtype=self.dtype,
            query_start_loc=query_start_loc,
            causal=False,
            isa=self.isa,
            block_size=bs,
        )
        request_ids = tuple(seq.request_id for seq in seqs)
        if request_ids != self._cross_key:
            self._cross_staged = attention.stage_cross(
                self.decoder_layers, request_ids, bs, self.isa
            )
            self._cross_key = request_ids
        set_context(
            context_lens=context_lens_t,
            block_tables=torch.tensor(padded, dtype=torch.int32),
            slot_mapping=torch.tensor(new_slots, dtype=torch.int64),
            query_start_loc=query_start_loc,
            self_metadata=self_metadata,
            isa=self.isa,
            cross_staged=self._cross_staged,
        )
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(positions, dtype=torch.long),
        )

    @torch.no_grad()
    def run(self, seqs):
        """Encode any fresh sequences, run one decoder step, and return next-token logits."""
        fresh = [seq for seq in seqs if seq.num_cached_tokens == 0]
        if fresh:
            self._encode(fresh)

        input_ids, positions = self._prepare(seqs)
        with self._inductor_ctx(len(seqs) > 1 or not _UNFROZEN_SINGLE_ROW):
            return self.decode_fn(input_ids, positions)

    def _inductor_ctx(self, freeze=True):
        """Freezing lets Inductor pack weights into oneDNN layouts; it only applies when compiled."""
        if not self.compiled:
            return nullcontext()
        return torch._inductor.config.patch(freezing=freeze)
