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

# On aarch64, oneDNN runs a plain (unpacked) weight fastest for one row, so the
# batch-1 decode graph is compiled without freezing (weight packing).
# Batch 1 is its own graph anyway (Dynamo's 0/1 specialization).
_UNFROZEN_SINGLE_ROW = platform.machine().lower() in ("arm64", "aarch64")


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
        # Cross-attention pool: request_id -> (pages, enc_len), and the unused pages.
        self._cross_pages: dict[int, tuple[torch.Tensor, int]] = {}
        self._cross_free = list(range(config.num_cross_blocks))
        # Cross-attention metadata staged for the current batch, keyed by its request ids.
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
            cross = (self.config.num_cross_blocks, *shape[1:])
            layer.encoder_attn.k_cache = torch.zeros(cross, dtype=self.dtype)
            layer.encoder_attn.v_cache = torch.zeros(cross, dtype=self.dtype)

        weight_bytes = sum(
            t.nbytes for t in (*self.model.parameters(), *self.model.buffers())
        )
        self_kv, cross_kv = (
            sum(attn.k_cache.nbytes + attn.v_cache.nbytes for attn in attns)
            for attns in (
                [layer.self_attn for layer in self.decoder_layers],
                [layer.encoder_attn for layer in self.decoder_layers],
            )
        )
        mb = 1024**2
        logger.info(
            "Allocated %.1f MB: weights %.1f MB, self KV %.1f MB (%d blocks), "
            "cross KV %.1f MB (%d blocks), %d tokens per block",
            (weight_bytes + self_kv + cross_kv) / mb,
            weight_bytes / mb,
            self_kv / mb,
            self.config.num_blocks,
            cross_kv / mb,
            self.config.num_cross_blocks,
            self.block_size,
        )

    def _compile(self) -> None:
        global _COMPILED_RUNNERS
        _COMPILED_RUNNERS += 1
        # encode and the vision tower share one Dynamo frame: per engine up to 4 encode
        # graphs (batch 1 / >1, with / without image) + one tower graph per image shape.
        # Past the limit Dynamo silently falls back to eager.
        torch._dynamo.config.recompile_limit = max(
            torch._dynamo.config.recompile_limit, 8 * _COMPILED_RUNNERS
        )
        # Parallel C++ compile workers use GBs of RAM on small machines.
        torch._inductor.config.compile_threads = min(
            4, torch._inductor.config.compile_threads
        )
        # x86 bf16 decode: autotune each GEMM between oneDNN and Inductor's C++ template
        # (12-24% faster per step on a Xeon 8581C, ~2x the compile time). Measured no
        # gain for the encoder, the vision tower, fp32 on x86, or bf16 on aarch64.
        autotune = self.dtype == torch.bfloat16 and platform.machine().lower() in (
            "x86_64",
            "amd64",
        )
        # One dynamic graph covers every batch size / encoder length.
        self.decode_fn = torch.compile(
            self.model, dynamic=True, mode="max-autotune" if autotune else None
        )
        self.encode_fn = torch.compile(self.model.encode, dynamic=True)
        if self.image_fn is not None:
            self.image_fn = torch.compile(self.image_fn, dynamic=False)

    def _copy_blocks(self, copies: list[tuple[int, int]]) -> None:
        """Copy-on-write: copy KV blocks src -> dst in every decoder layer, one op per cache."""
        src, dst = (torch.tensor(ids) for ids in zip(*copies))
        for layer in self.decoder_layers:
            for cache in (layer.self_attn.k_cache, layer.self_attn.v_cache):
                cache.index_copy_(0, dst, cache.index_select(0, src))

    def _allocate_cross_pages(self, num_pages: int) -> list[int]:
        """Take pages from the fixed encoder K/V pool."""
        if num_pages > len(self._cross_free):
            raise RuntimeError("encoder K/V pool is full; increase max_encoder_len")
        # A slice, not pops: keeps a request's pages in ascending memory order.
        pages = self._cross_free[-num_pages:]
        del self._cross_free[-num_pages:]
        return pages

    def _image_embeds(self, seqs) -> dict[str, torch.Tensor]:
        """Return {mm_hash: image embeddings}; run the vision encoder only on cache misses."""
        # Split into cache hits and misses (the same image twice in a step is one miss).
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
        # One image per call: the tower graph is static, so every batch size would be
        # its own ~20 s compile (and batching measured slower per image).
        for key, px in misses.items():
            with self._inductor_ctx():
                embeds[key] = self.image_fn(px.unsqueeze(0))[0]
        # Store everything and evict the least recently used entries beyond the cache size.
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
        """Batch fresh sequences by (encoder_len, image_shape) and page their cross-attn K/V."""
        # Vision tower first (Florence-2 only): its output is spliced into the encoder input.
        image_seqs = [seq for seq in seqs if seq.pixel_values is not None]
        embeds = self._image_embeds(image_seqs) if image_seqs else {}
        # Sequences with the same encoder length and image shape share one batched forward,
        # so no padding is needed.
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
            # Encoder forward: the hidden states are what every decoder layer cross-attends to.
            with self._inductor_ctx():
                if group[0].pixel_values is None:
                    hidden = self.encode_fn(input_ids)
                else:
                    image_embeds = torch.stack([embeds[seq.mm_hash] for seq in group])
                    hidden = self.encode_fn(input_ids, image_embeds=image_embeds)
            # Each request owns ceil(enc_len / block_size) pages; slots are its tokens' places.
            enc_len, block_size = input_ids.shape[1], self.block_size
            pages = torch.tensor(
                self._allocate_cross_pages(len(group) * -(-enc_len // block_size)),
                dtype=torch.int32,
            ).view(len(group), -1)
            token = torch.arange(enc_len)
            slots = pages[:, token // block_size].long() * block_size
            slots = (slots + token % block_size).flatten()
            # Remember the pages so decode steps can find them and clear_request can free them.
            for seq, seq_pages in zip(group, pages):
                self._cross_pages[seq.request_id] = (seq_pages, enc_len)
            # Cross-attn K/V depend only on the encoder output, so compute them once here
            # and every decode step just reads the pages.
            for layer in self.decoder_layers:
                attn = layer.encoder_attn
                shape = (len(group) * enc_len, attn.num_heads, attn.head_dim)
                key, value = attn.kv_proj(hidden).chunk(2, dim=-1)
                vllm_cpu_attn.reshape_and_cache(
                    key.reshape(shape).contiguous(),
                    value.reshape(shape).contiguous(),
                    attn.k_cache,
                    attn.v_cache,
                    slots,
                    isa=self.isa,
                )

    def clear_request(self, request_id: int) -> None:
        """Return a finished request's encoder K/V pages and drop the staged cross-attention state."""
        pages, _ = self._cross_pages.pop(request_id, ((), 0))
        self._cross_free.extend(int(page) for page in pages)
        self._cross_key = self._cross_staged = None

    def reset(self) -> None:
        """Return every encoder K/V page to the pool after an aborted batch."""
        self._cross_pages.clear()
        self._cross_free = list(range(self.config.num_cross_blocks))
        self._cross_key = self._cross_staged = None

    def _prepare(self, seqs, copies=()):
        """Build input_ids, positions, and Context tensors for the step."""
        block_size = self.block_size
        # Invariant: every step feeds exactly one uncached token per sequence.
        for seq in seqs:
            num_new = len(seq.token_ids) - seq.num_cached_tokens
            if num_new != 1:
                raise ValueError(
                    f"each step must feed 1 token per sequence, got {num_new}"
                )

        # Copy-on-write must finish before this step writes into the (now private) blocks.
        if copies:
            self._copy_blocks(copies)

        # Per sequence: the token to feed, its position, the cache slot its K/V goes to,
        # and how much context it attends to (including itself).
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

        # Pad block tables to a rectangle; the kernel only reads up to each context_len.
        max_blocks = max(len(t) for t in block_tables)
        padded_tables = [t + [0] * (max_blocks - len(t)) for t in block_tables]
        context_lens_t = torch.tensor(context_lens, dtype=torch.int32)
        query_start_loc = torch.arange(len(seqs) + 1, dtype=torch.int32)
        # Kernel scheduling metadata for self-attention: one query token per sequence.
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
        # Rebuild the cross-attn block tables only when the batch composition changed.
        request_ids = tuple(seq.request_id for seq in seqs)
        if request_ids != self._cross_key:
            pages, enc_lens = zip(*(self._cross_pages[rid] for rid in request_ids))
            self._cross_staged = attention.stage_cross(
                self.decoder_layers, pages, enc_lens, block_size, self.isa
            )
            self._cross_key = request_ids
        # Publish the step state; the attention custom ops read it via get_context().
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
        """Apply LogitsProcessor; return a token id per seq (greedy) or {request_id: candidates} (beam)."""
        sampling_params = seqs[0].sampling_params
        logits_processor = self._get_logits_processor(sampling_params)
        # Some rules (repetition, min length, ...) need each sequence's token history.
        lengths = [len(seq.token_ids) for seq in seqs]
        histories = None
        if logits_processor.needs_history:
            max_len = max(lengths)
            # Histories differ in length: left-pad with -1, which every logits rule ignores.
            histories = torch.tensor(
                [[-1] * (max_len - len(seq.token_ids)) + seq.token_ids for seq in seqs],
                dtype=torch.long,
            )
        # One float buffer for the whole batch; .float() is a no-op for fp32 logits.
        scores = logits.float()
        if sampling_params.num_beams > 1:
            scores = torch.log_softmax(scores, dim=-1)
        scores = logits_processor(  # in place
            histories,
            scores,
            torch.tensor(lengths, dtype=torch.long),
        )
        # Greedy takes the best token per row; beam search keeps top-k candidates for the scheduler.
        if sampling_params.num_beams == 1:
            return scores.argmax(dim=-1).tolist()
        return self._beam_search(seqs, scores, sampling_params)

    def _beam_search(self, seqs, scores, sampling_params) -> dict[int, list]:
        """Return each request's top-k (parent, token_id, cum_logprob) candidates.
        scores holds processed log-probs, one row per seq.
        """
        # Keep top K = max(2, 1 + n_eos_tokens) * num_beams continuations.
        n_eos_tokens = len(sampling_params.eos_ids())
        beams_to_keep = max(2, 1 + n_eos_tokens) * sampling_params.num_beams
        # topk each row over the vocab, then add cum_logprob to the survivors only.
        top_scores, top_ids = torch.topk(
            scores, min(beams_to_keep, scores.shape[-1]), dim=-1
        )
        top_scores += torch.tensor(
            [seq.cum_logprob for seq in seqs], dtype=torch.float32
        ).unsqueeze(-1)

        # Group the rows by request: a request's beams compete with each other.
        topk_candidates = defaultdict(list)
        for seq, row_scores, row_ids in zip(
            seqs, top_scores.tolist(), top_ids.tolist()
        ):
            topk_candidates[seq.request_id].extend(
                zip([seq] * len(row_ids), row_ids, row_scores)
            )
        # Keep each request's best candidates overall (not per beam).
        for candidates in topk_candidates.values():
            candidates.sort(key=lambda candidate: candidate[2], reverse=True)
            del candidates[beams_to_keep:]
        return dict(topk_candidates)

    @torch.no_grad()
    def run(self, seqs, copies=()):
        """Encode any fresh sequences, run one decoder step, and return `_sample`'s result ([] if no seqs)."""
        if not seqs:
            return []
        # Nothing cached yet means a new request: run its encoder and page its cross-attn K/V first.
        fresh = [seq for seq in seqs if seq.num_cached_tokens == 0]
        if fresh:
            self._encode(fresh)

        # The context is global, so always reset it, even if the forward raises.
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
