import torch
import vllm_cpu_attn

from ..utils.context import get_context

HEAD_DIMS = {32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512}


def stage_cross(layers, request_ids, block_size: int, isa: str):
    """Pack each decoder layer's dense encoder K/V into a paged cache for this step.

    ``request_ids`` has one entry per running sequence; beams of one request repeat
    its id and share one copy of its encoder K/V.
    """
    cache0 = layers[0].encoder_attn.encoder_kv_cache
    unique_ids = list(dict.fromkeys(request_ids))
    row_of = {rid: i for i, rid in enumerate(unique_ids)}
    enc_lens = [cache0[rid][0].shape[0] for rid in unique_ids]
    pages = [(length + block_size - 1) // block_size for length in enc_lens]
    _, num_heads, head_dim = cache0[unique_ids[0]][0].shape
    dtype = cache0[unique_ids[0]][0].dtype

    # Page layout: each request gets consecutive pages; slots are where its tokens go.
    table = torch.zeros(len(unique_ids), max(pages), dtype=torch.int32)
    slots = []
    next_page = 0
    for i, (length, num_pages) in enumerate(zip(enc_lens, pages)):
        table[i, :num_pages] = torch.arange(next_page, next_page + num_pages)
        start = next_page * block_size
        slots.append(torch.arange(start, start + length, dtype=torch.long))
        next_page += num_pages
    slots = torch.cat(slots)

    # Per-sequence views: beams point at their request's row.
    rows = torch.tensor([row_of[rid] for rid in request_ids], dtype=torch.long)
    block_table = table[rows]
    seq_lens = torch.tensor([enc_lens[i] for i in rows], dtype=torch.int32)
    metadata = vllm_cpu_attn.get_scheduler_metadata(
        num_reqs=len(request_ids),
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        seq_lens=seq_lens,
        dtype=dtype,
        query_start_loc=torch.arange(len(request_ids) + 1, dtype=torch.int32),
        causal=False,
        isa=isa,
        block_size=block_size,
    )

    shape = (next_page, num_heads, block_size, head_dim)
    staged = []
    for layer in layers:
        cache = layer.encoder_attn.encoder_kv_cache
        k = torch.cat([cache[rid][0] for rid in unique_ids])
        v = torch.cat([cache[rid][1] for rid in unique_ids])
        # zeros, not empty: the kernel reads whole pages, including the unused tail.
        k_cache = torch.zeros(shape, dtype=dtype)
        v_cache = torch.zeros(shape, dtype=dtype)
        vllm_cpu_attn.reshape_and_cache(k, v, k_cache, v_cache, slots, isa=isa)
        staged.append((k_cache, v_cache, block_table, seq_lens, metadata))
    return staged


@torch.library.custom_op(
    "fastencdec::self_attention", mutates_args={"k_cache", "v_cache"}
)
def decoder_self_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Write new K/V into the paged cache and run self-attention."""
    ctx = get_context()
    vllm_cpu_attn.reshape_and_cache(
        key, value, k_cache, v_cache, ctx.slot_mapping, isa=ctx.isa
    )
    out = torch.empty_like(query)
    # causal=False is still causal: each sequence queries only its newest token,
    # and the cache holds nothing after it.
    vllm_cpu_attn.attention_with_kv_cache(
        query,
        k_cache,
        v_cache,
        out,
        ctx.query_start_loc,
        ctx.context_lens,
        scale=scale,
        causal=False,
        block_table=ctx.block_tables,
        scheduler_metadata=ctx.self_metadata,
    )
    return out


@decoder_self_attention.register_fake
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return torch.empty_like(query)


@torch.library.custom_op("fastencdec::cross_attention", mutates_args={})
def decoder_cross_attention(
    query: torch.Tensor,
    layer_idx: int,
    scale: float,
) -> torch.Tensor:
    """Run cross-attention over the staged encoder K/V for decoder layer ``layer_idx``."""
    ctx = get_context()
    k_cache, v_cache, block_table, seq_lens, metadata = ctx.cross_staged[layer_idx]
    out = torch.empty_like(query)
    vllm_cpu_attn.attention_with_kv_cache(
        query,
        k_cache,
        v_cache,
        out,
        ctx.query_start_loc,
        seq_lens,
        scale=scale,
        causal=False,
        block_table=block_table,
        scheduler_metadata=metadata,
    )
    return out


@decoder_cross_attention.register_fake
def _(query: torch.Tensor, layer_idx: int, scale: float) -> torch.Tensor:
    return torch.empty_like(query)
