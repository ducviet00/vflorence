import platform
from collections import defaultdict
from contextlib import nullcontext

import torch
import vllm_cpu_attn

from ..layers import attention
from ..utils.context import set_context

_COMPILED_RUNNERS = 0

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
        self.decode_step = None
        self.encode_fn = self.model.encode
        self.mm_step = None
        if compile:
            global _COMPILED_RUNNERS
            _COMPILED_RUNNERS += 1
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, 2 * _COMPILED_RUNNERS + 4
            )
            torch._inductor.config.compile_threads = min(
                4, torch._inductor.config.compile_threads
            )
            self.decode_step = torch.compile(self.model, dynamic=True)
            self.encode_fn = torch.compile(self.model.encode, dynamic=True)
            if hasattr(self.model, "get_image_features") and compile_mm_encoder:
                self.mm_step = torch.compile(
                    self.model.get_image_features, dynamic=False
                )

        shape = (
            block_manager.num_blocks,
            self.num_heads,
            self.block_size,
            self.head_dim,
        )
        for layer in self.decoder_layers:
            layer.self_attn.k_cache = torch.zeros(shape, dtype=dtype)
            layer.self_attn.v_cache = torch.zeros(shape, dtype=dtype)

    def _copy_block(self, src: int, dst: int) -> None:
        """Copy KV cache contents from block src to dst across all decoder layers."""
        for layer in self.decoder_layers:
            layer.self_attn.k_cache[dst].copy_(layer.self_attn.k_cache[src])
            layer.self_attn.v_cache[dst].copy_(layer.self_attn.v_cache[src])

    @torch.no_grad()
    def _encode(self, seqs) -> None:
        """Batch fresh sequences by (encoder_len, image_shape) and cache cross-attn K/V."""
        groups = defaultdict(list)
        for seq in seqs:
            px_shape = (
                None if seq.pixel_values is None else tuple(seq.pixel_values.shape)
            )
            groups[(len(seq.encoder_token_ids), px_shape)].append(seq)
        encode = self.encode_fn
        for group in groups.values():
            ids = torch.tensor([s.encoder_token_ids for s in group], dtype=torch.long)
            px = (
                None
                if group[0].pixel_values is None
                else torch.stack([s.pixel_values for s in group])
            )
            with self._inductor_ctx():
                embeds = (
                    None
                    if px is None
                    else (self.mm_step or self.model.get_image_features)(px)
                )
                hidden = (
                    encode(ids) if embeds is None else encode(ids, image_embeds=embeds)
                )
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
            attn = layer.encoder_attn
            attn.encoder_kv_cache.pop(request_id, None)
            attn.cross_staged = None
            attn.cross_staged_key = None

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
        context_lens, seq_block_tables = [], []
        for seq in seqs:
            self.block_manager.ensure_capacity(seq, self._copy_block)
            pos = seq.num_cached_tokens
            input_ids.append(seq.token_ids[pos])
            positions.append(pos)
            new_slots.append(seq.block_table[pos // bs] * bs + pos % bs)
            seq.num_cached_tokens = pos + 1
            context_lens.append(pos + 1)
            seq_block_tables.append(seq.block_table)

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
        set_context(
            context_lens=context_lens_t,
            block_tables=torch.tensor(padded, dtype=torch.int32),
            slot_mapping=torch.tensor(new_slots, dtype=torch.int64),
            query_start_loc=query_start_loc,
            self_metadata=self_metadata,
            isa=self.isa,
            cross_staged=attention.stage_cross(
                self.decoder_layers, [seq.request_id for seq in seqs], bs, self.isa
            ),
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
        if self.decode_step is None:
            return self.model(input_ids, positions)
        with self._inductor_ctx(len(seqs) > 1 or not _UNFROZEN_SINGLE_ROW):
            return self.decode_step(input_ids, positions)

    def _inductor_ctx(self, freeze=True):
        return (
            torch._inductor.config.patch(freezing=freeze)
            if self.decode_step is not None
            else nullcontext()
        )
