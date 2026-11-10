import torch
import vllm_cpu_attn

from ..utils.context import get_context

HEAD_DIMS = {32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512}


def stage_cross(layers, pages, enc_lens, block_size: int, isa: str):
    """Point this step's sequences at their encoder K/V pages in each layer's pool.

    ``pages[i]`` (int32 tensor) and ``enc_lens[i]`` belong to running sequence i;
    beams of one request repeat its pages. Nothing is copied.
    """
    attn = layers[0].encoder_attn
    block_table = torch.nn.utils.rnn.pad_sequence(pages, batch_first=True)
    seq_lens = torch.tensor(enc_lens, dtype=torch.int32)
    metadata = vllm_cpu_attn.get_scheduler_metadata(
        num_reqs=len(pages),
        num_heads=attn.num_heads,
        num_kv_heads=attn.num_heads,
        head_dim=attn.head_dim,
        seq_lens=seq_lens,
        dtype=attn.k_cache.dtype,
        query_start_loc=torch.arange(len(pages) + 1, dtype=torch.int32),
        causal=False,
        isa=isa,
        block_size=block_size,
    )
    return [
        (
            layer.encoder_attn.k_cache,
            layer.encoder_attn.v_cache,
            block_table,
            seq_lens,
            metadata,
        )
        for layer in layers
    ]


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
