import logging
import platform
from collections import OrderedDict, defaultdict
from contextlib import nullcontext

import torch
import vllm_cpu_attn

from ..config import Config
from ..layers import attention
from ..layers.logits_processor import LogitsProcessor
from ..sampling_params import SamplingParams
from ..utils.context import set_context
from ..utils.loader import load_hf_weights

logger = logging.getLogger(__name__)
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

    def __init__(self, config: Config):
        self.config = config
        self.model = load_hf_weights(config)
        self.block_size = config.block_size
        self.dtype = config.dtype
        self.decoder_layers = list(self.model.decoder_layers)
        self.num_heads = self.decoder_layers[0].self_attn.num_heads
        self.head_dim = self.decoder_layers[0].self_attn.head_dim
        if self.head_dim not in attention.HEAD_DIMS:
            raise ValueError(
                f"vllm-cpu-attn does not support head_dim={self.head_dim}; "
                f"must be one of {sorted(attention.HEAD_DIMS)}"
            )
        self.isa = vllm_cpu_attn.get_isa(self.dtype, self.block_size, self.head_dim)
        # Eager by default; _compile swaps in compiled versions.
        self.compiled = config.compile
        self.decode_fn = self.model
        self.encode_fn = self.model.encode
        self.image_fn = getattr(self.model, "get_image_features", None)
        if config.compile:
            self._compile()
        # Multimodal encoder cache: LRU of mm_hash -> image embeddings, kept across calls.
        self.mm_cache = OrderedDict()
        self.mm_cache_size = config.mm_cache_size
        self.mm_cache_hits = self.mm_cache_misses = 0
        # Cross-attention K/V staged for the current batch, keyed by its request ids.
        self._cross_key = None
        self._cross_staged = None
        self._logits_processor: LogitsProcessor | None = None
        self._sampling_params: SamplingParams | None = None
        self.allocate_kv_cache()

    def allocate_kv_cache(self) -> None:
        # Separate tensors per layer: views of one shared tensor break Inductor codegen.
        # zeros, not empty: the kernel reads whole blocks, and 0 * NaN from garbage
        # in a block's unused tail would poison the output.
        shape = (
            self.config.num_blocks,
            self.num_heads,
            self.block_size,
            self.head_dim,
        )
        for layer in self.decoder_layers:
            layer.self_attn.k_cache = torch.zeros(shape, dtype=self.dtype)
            layer.self_attn.v_cache = torch.zeros(shape, dtype=self.dtype)

        weight_bytes = sum(
            t.numel() * t.element_size()
            for t in (*self.model.parameters(), *self.model.buffers())
        )
        kv_bytes = sum(
            c.numel() * c.element_size()
            for layer in self.decoder_layers
            for c in (layer.self_attn.k_cache, layer.self_attn.v_cache)
        )
        logger.info(
            "Allocated %.1f MB (weights %.1f MB + KV cache %.1f MB, %d blocks of %d tokens)",
            (weight_bytes + kv_bytes) / (1024**2),
            weight_bytes / (1024**2),
            kv_bytes / (1024**2),
            self.config.num_blocks,
            self.block_size,
        )

    def _compile(self) -> None:
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
        if self.image_fn is not None:
            self.image_fn = torch.compile(self.image_fn, dynamic=False)

    def _copy_blocks(self, copies: list[tuple[int, int]]) -> None:
        """Copy-on-write: copy KV blocks src -> dst in every decoder layer, one op per cache."""
        src, dst = (torch.tensor(ids) for ids in zip(*copies))
        for layer in self.decoder_layers:
            for cache in (layer.self_attn.k_cache, layer.self_attn.v_cache):
                cache[dst] = cache[src]

    def _image_embeds(self, seqs) -> dict[str, torch.Tensor]:
        """Return {mm_hash: image embeddings}; run the vision encoder only on cache misses."""
        embeds, misses = {}, {}
        for seq in seqs:
            key = seq.mm_hash
            if key in embeds or key in misses:
                self.mm_cache_hits += 1  # same image twice in one step
            elif key in self.mm_cache:
                embeds[key] = self.mm_cache[key]
                self.mm_cache_hits += 1
            else:
                misses[key] = seq.pixel_values
                self.mm_cache_misses += 1
        by_shape = defaultdict(list)
        for key, px in misses.items():
            by_shape[px.shape].append(key)
        for keys in by_shape.values():
            with self._inductor_ctx():
                out = self.image_fn(torch.stack([misses[k] for k in keys]))
            for key, emb in zip(keys, out):
                embeds[key] = emb.clone()  # own storage, not a view of the batch
        if self.mm_cache_size:
            # Update recency in request order, so hits and misses rank by last use.
            for seq in seqs:
                self.mm_cache[seq.mm_hash] = embeds[seq.mm_hash]
                self.mm_cache.move_to_end(seq.mm_hash)
                if len(self.mm_cache) > self.mm_cache_size:
                    self.mm_cache.popitem(last=False)
        return embeds

    @torch.no_grad()
    def _encode(self, seqs) -> None:
        """Batch fresh sequences by (encoder_len, image_shape) and cache cross-attn K/V."""
        image_seqs = [seq for seq in seqs if seq.pixel_values is not None]
        embeds = self._image_embeds(image_seqs) if image_seqs else {}
        groups = defaultdict(list)
        for seq in seqs:
            px_shape = (
                None if seq.pixel_values is None else tuple(seq.pixel_values.shape)
            )
            groups[(len(seq.encoder_token_ids), px_shape)].append(seq)
        for group in groups.values():
            input_ids = torch.tensor(
                [seq.encoder_token_ids for seq in group], dtype=torch.long
            )
            with self._inductor_ctx():
                if group[0].pixel_values is None:
                    hidden = self.encode_fn(input_ids)
                else:
                    image_embeds = torch.stack([embeds[seq.mm_hash] for seq in group])
                    hidden = self.encode_fn(input_ids, image_embeds=image_embeds)
            for layer in self.decoder_layers:
                attn = layer.encoder_attn
                shape = (len(group), input_ids.shape[1], attn.num_heads, attn.head_dim)
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

    def _prepare(self, seqs, copies=()):
        """Build input_ids, positions, and Context tensors for the step."""
        block_size = self.block_size
        for seq in seqs:
            num_new = len(seq.token_ids) - seq.num_cached_tokens
            if num_new != 1:
                raise ValueError(
                    f"each step must feed 1 token per sequence, got {num_new}"
                )

        if copies:
            self._copy_blocks(copies)

        input_ids, positions, slot_mapping = [], [], []
        context_lens, block_tables = [], []
        for seq in seqs:
            pos = seq.num_cached_tokens
            input_ids.append(seq.token_ids[pos])
            positions.append(pos)
            slot_mapping.append(
                seq.block_table[pos // block_size] * block_size + pos % block_size
            )
            context_lens.append(pos + 1)
            block_tables.append(seq.block_table)

        max_blocks = max(len(t) for t in block_tables)
        padded_tables = [t + [0] * (max_blocks - len(t)) for t in block_tables]
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
            block_size=block_size,
        )
        request_ids = tuple(seq.request_id for seq in seqs)
        if request_ids != self._cross_key:
            self._cross_staged = attention.stage_cross(
                self.decoder_layers, request_ids, block_size, self.isa
            )
            self._cross_key = request_ids
        set_context(
            context_lens=context_lens_t,
            block_tables=torch.tensor(padded_tables, dtype=torch.int32),
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int64),
            query_start_loc=query_start_loc,
            self_metadata=self_metadata,
            isa=self.isa,
            cross_staged=self._cross_staged,
        )
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(positions, dtype=torch.long),
        )

    def _get_logits_processor(self, sampling_params: SamplingParams) -> LogitsProcessor:
        if sampling_params != self._sampling_params or self._logits_processor is None:
            # Lengths count the single decoder start token, as transformers does.
            self._logits_processor = LogitsProcessor(
                sampling_params,
                vocab_size=self.model.lm_head.out_features,
                max_length=1 + sampling_params.max_tokens,
                begin_index=1 + int(sampling_params.forced_bos_token_id is not None),
            )
            self._sampling_params = sampling_params
        return self._logits_processor

    def _sample(self, seqs, logits):
        """Apply LogitsProcessor and return greedy token IDs or beam top-k candidates."""
        sampling_params = seqs[0].sampling_params
        logits_processor = self._get_logits_processor(sampling_params)
        if sampling_params.num_beams == 1:
            lengths = [len(seq.token_ids) for seq in seqs]
            max_len = max(lengths)
            # Histories differ in length: left-pad with -1, which every logits rule ignores.
            histories = [
                [-1] * (max_len - len(seq.token_ids)) + seq.token_ids for seq in seqs
            ]
            scores = logits_processor(
                torch.tensor(histories, dtype=torch.long),
                logits.float(),
                torch.tensor(lengths, dtype=torch.long),
            )
            return scores.argmax(dim=-1).tolist()

        num_beams = sampling_params.num_beams
        vocab_size = logits.shape[-1]
        by_request: dict[int, list[int]] = defaultdict(list)
        for index, seq in enumerate(seqs):
            by_request[seq.request_id].append(index)
        groups: dict[tuple[int, int], list] = defaultdict(list)
        for request_id, indices in by_request.items():
            groups[(len(indices), len(seqs[indices[0]].token_ids))].append(
                (request_id, indices)
            )
        topk_candidates = {}
        for (beam_count, _), group in groups.items():
            rows = [i for _, seq_indices in group for i in seq_indices]
            input_ids = torch.tensor(
                [seqs[i].token_ids for i in rows], dtype=torch.long
            )
            log_probs = logits_processor(
                input_ids, torch.log_softmax(logits[rows].float(), dim=-1)
            )
            running_beam_scores = torch.tensor(
                [
                    [seqs[i].cum_logprob for i in seq_indices]
                    for _, seq_indices in group
                ],
                dtype=torch.float32,
            ).unsqueeze(-1)
            log_probs = (
                log_probs.reshape(len(group), beam_count, vocab_size)
                + running_beam_scores
            ).reshape(len(group), -1)
            # Keep top K = max(2, 1 + n_eos_tokens) * num_beams continuations.
            n_eos_tokens = len(sampling_params.eos_ids())
            beams_to_keep = min(
                max(2, 1 + n_eos_tokens) * num_beams, log_probs.shape[-1]
            )
            topk_log_probs, topk_indices = torch.topk(log_probs, beams_to_keep, dim=-1)
            for (request_id, seq_indices), log_probs_row, indices_row in zip(
                group, topk_log_probs.tolist(), topk_indices.tolist()
            ):
                topk_candidates[request_id] = [
                    (
                        seqs[seq_indices[idx // vocab_size]],
                        idx % vocab_size,
                        cum_logprob,
                    )
                    for idx, cum_logprob in zip(indices_row, log_probs_row)
                ]
        return topk_candidates

    @torch.no_grad()
    def run(self, seqs, copies=()):
        """Encode any fresh sequences, run one decoder step, and return sampled candidates."""
        if not seqs:
            return []
        fresh = [seq for seq in seqs if seq.num_cached_tokens == 0]
        if fresh:
            self._encode(fresh)

        try:
            input_ids, positions = self._prepare(seqs, copies)
            with self._inductor_ctx(len(seqs) > 1 or not _UNFROZEN_SINGLE_ROW):
                logits = self.decode_fn(input_ids, positions)
        finally:
            set_context()
        return self._sample(seqs, logits)

    def _inductor_ctx(self, freeze=True):
        """Freezing lets Inductor pack weights into oneDNN layouts; it only applies when compiled."""
        if not self.compiled:
            return nullcontext()
        return torch._inductor.config.patch(freezing=freeze)
