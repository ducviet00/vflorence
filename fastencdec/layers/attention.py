import torch
import vllm_cpu_attn

from ..utils.context import get_context

HEAD_DIMS = {32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512}


def stage_cross(layers, request_ids, block_size: int, isa: str):
    """Stage every decoder layer's encoder K/V into a paged cache."""
    key = tuple(request_ids)
    if all(
        l.encoder_attn.cross_staged is not None
        and l.encoder_attn.cross_staged_key == key
        for l in layers
    ):
        return [l.encoder_attn.cross_staged for l in layers]

    cache0 = layers[0].encoder_attn.encoder_kv_cache
    rows = list(dict.fromkeys(request_ids))
    seen = {rid: i for i, rid in enumerate(rows)}
    enc_lens = [cache0[rid][0].shape[0] for rid in rows]
    pages = [(length + block_size - 1) // block_size for length in enc_lens]
    max_pages, total = max(pages), sum(pages)
    key0 = cache0[rows[0]][0]
    dtype = key0.dtype
    _, num_heads, head_dim = key0.shape

    table = torch.zeros(len(rows), max_pages, dtype=torch.int32)
    pid = 0
    for i, page_num in enumerate(pages):
        table[i, :page_num] = torch.arange(pid, pid + page_num, dtype=torch.int32)
        pid += page_num

    slots_per_req = [
        table[i, torch.arange(length) // block_size].long() * block_size
        + (torch.arange(length) % block_size)
        for i, length in enumerate(enc_lens)
    ]
    block_table = table[
        torch.tensor([seen[rid] for rid in request_ids], dtype=torch.long)
    ]
    seq_lens = torch.tensor(
        [enc_lens[seen[rid]] for rid in request_ids], dtype=torch.int32
    )
    query_start_loc = torch.arange(len(request_ids) + 1, dtype=torch.int32)
    metadata = vllm_cpu_attn.get_scheduler_metadata(
        num_reqs=len(request_ids),
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        seq_lens=seq_lens,
        dtype=dtype,
        query_start_loc=query_start_loc,
        causal=False,
        isa=isa,
        block_size=block_size,
    )

    shape = (total, num_heads, block_size, head_dim)
    staged = []
    for layer in layers:
        attn = layer.encoder_attn
        k_cache = torch.zeros(shape, dtype=dtype)
        v_cache = torch.zeros(shape, dtype=dtype)
        for rid, slots in zip(rows, slots_per_req):
            k, v = attn.encoder_kv_cache[rid]
            vllm_cpu_attn.reshape_and_cache(k, v, k_cache, v_cache, slots, isa=isa)
        attn.cross_staged = (k_cache, v_cache, block_table, seq_lens, metadata)
        attn.cross_staged_key = key
        staged.append(attn.cross_staged)
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
