import math

import torch
import torch.nn.functional as F
from torch import nn

from ..config import AARCH64
from .bart import BartModel

# Alignment (elements) for the bias-augmented K dimension in channel attention
# (64 B in bf16, 128 B in fp32).
_AUG_K_ALIGN = 32


def _dwconv_residual(
    conv: nn.Conv2d, hidden_states: torch.Tensor, center: torch.Tensor
) -> torch.Tensor:
    """Fold the identity residual into a 3x3 depthwise conv's center tap.

    oneDNN depthwise conv does not fuse a binary add post-op and falls back to
    an eager aten::add_ per call; adding a constant center tap folds into the
    frozen conv weight at Inductor compile time.
    """
    assert (
        conv.kernel_size == (3, 3)
        and conv.stride == (1, 1)
        and conv.padding == (1, 1)
        and conv.dilation == (1, 1)
        and conv.groups == conv.in_channels == conv.out_channels
    )
    return F.conv2d(
        hidden_states,
        conv.weight + center,
        conv.bias,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
    )


class Florence2VisionMLP(nn.Module):
    def __init__(self, config, stage_idx: int):
        super().__init__()
        dim = config.embed_dim[stage_idx]
        hidden_dim = int(dim * config.mlp_ratio)
        self.activation_fn = F.gelu
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the two-layer GELU MLP on vision features."""
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class Florence2VisionConvEmbed(nn.Module):
    def __init__(self, config, stage_idx: int):
        super().__init__()
        self.stage_idx = stage_idx
        self.patch_size = config.patch_size[stage_idx]
        self.in_channels = (
            config.in_channels if stage_idx == 0 else config.embed_dim[stage_idx - 1]
        )
        self.embed_dim = config.embed_dim[stage_idx]
        self.stride = config.patch_stride[stage_idx]
        self.padding = config.patch_padding[stage_idx]
        self.pre_norm = config.patch_prenorm[stage_idx]

        self.conv = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=self.padding,
        )
        self.norm = nn.LayerNorm(self.in_channels if self.pre_norm else self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply strided patch convolution with pre- or post-LayerNorm."""
        if self.pre_norm:
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = self.conv(hidden_states)
        if not self.pre_norm:
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)
        return hidden_states


class Florence2VisionChannelAttention(nn.Module):
    def __init__(self, config, stage_idx: int):
        super().__init__()
        self.dim = config.embed_dim[stage_idx]
        self.groups = config.num_groups[stage_idx]
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim)
        if config.qkv_bias:
            # Pad K = dim + pad to a multiple of _AUG_K_ALIGN so W_aug = [W_qk, b_qk, 0..0]
            # stays 64B-aligned in bf16. A buffer (rather than F.pad on X) keeps
            # norm1's fused cat kernel unmasked SIMD.
            pad = _AUG_K_ALIGN - (self.dim % _AUG_K_ALIGN)
            assert (self.dim + pad) % _AUG_K_ALIGN == 0
            x_tail = torch.zeros(1, pad)
            x_tail[0, 0] = 1.0
            self.register_buffer("_x_tail", x_tail, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run grouped channel self-attention over the head_dim axis."""
        batch_size, num_tokens, hidden_size = hidden_states.shape
        head_dim = hidden_size // self.groups
        if batch_size == 1 and num_tokens * 2 >= hidden_size * 3:
            # Empirical cutoff (N >= 1.5*C on Xeon AMX and Axion Neoverse-V2;
            # 1.0*C loses slightly due to unpacked GEMMs, 1.5*C+ wins): compute
            # only Q/K (2C) in [..., N] order, then re-associate V + proj into
            # one (N, C) @ (C, C) GEMM with W_comb = (W_proj @ P) @ W_v (packed
            # mkldnn on constant W_v). Freezing constant-folds w_aug/w_proj;
            # eager mode (compile=False) rebuilds them per call.
            w_qk = self.qkv.weight[: 2 * hidden_size]
            b_qk = (
                self.qkv.bias[: 2 * hidden_size] if self.qkv.bias is not None else None
            )
            w_v = self.qkv.weight[2 * hidden_size :]
            if b_qk is not None:
                pad = self._x_tail.shape[1]
                w_aug = F.pad(torch.cat([w_qk, b_qk.unsqueeze(1)], dim=1), (0, pad - 1))
                x_aug = torch.cat(
                    [hidden_states[0], self._x_tail.expand(num_tokens, -1)], dim=1
                )
                qk = torch.mm(w_aug, x_aug.transpose(0, 1))
                # Reuse x_aug's prefix so norm1 writes one buffer instead of two.
                x_use = x_aug[:, :hidden_size]
            else:
                qk = torch.mm(w_qk, hidden_states[0].transpose(0, 1))
                x_use = hidden_states[0]
            query, key = qk.view(2, self.groups, head_dim, num_tokens).unbind(0)
            scores = torch.bmm(query, key.transpose(1, 2)).float()
            probs = (scores * (num_tokens**-0.5)).softmax(-1).to(hidden_states.dtype)
            w_proj = (
                self.proj.weight.view(hidden_size, self.groups, head_dim)
                .transpose(0, 1)
                .contiguous()
            )
            m_cc = (
                torch.bmm(w_proj, probs)
                .transpose(0, 1)
                .contiguous()
                .view(hidden_size, hidden_size)
            )
            w_comb = F.linear(m_cc, w_v.transpose(0, 1))
            out = torch.mm(x_use, w_comb.transpose(0, 1)).unsqueeze(0)
            b_comb = (
                torch.mv(m_cc, self.qkv.bias[2 * hidden_size :])
                if self.qkv.bias is not None
                else None
            )
            if self.proj.bias is not None:
                b_comb = self.proj.bias if b_comb is None else b_comb + self.proj.bias
            return out + b_comb if b_comb is not None else out

        # N < 1.5*C or batched: keep packed mkldnn weights on both qkv and proj.
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, num_tokens, 3, self.groups, head_dim)
            .permute(2, 0, 3, 4, 1)
            .contiguous()
        )
        query, key, value = qkv.unbind(0)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, scale=num_tokens**-0.5
        )
        # Explicit contiguous() lets Inductor fuse proj bias and residual add
        # into mkldnn._linear_pointwise.binary (reshape on permute blocked it).
        hidden_states = (
            hidden_states.view(batch_size, hidden_size, num_tokens)
            .transpose(1, 2)
            .contiguous()
        )
        return self.proj(hidden_states)


class Florence2VisionWindowAttention(nn.Module):
    def __init__(self, config, stage_idx: int):
        super().__init__()
        self.dim = config.embed_dim[stage_idx]
        self.window_size = config.window_size
        self.num_heads = config.num_heads[stage_idx]
        self.scale = (self.dim // self.num_heads) ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Partition spatial features into local windows, run self-attention, and merge back."""
        batch_size, height, width, embed_dim = hidden_states.shape
        ws = self.window_size

        pad_right = (ws - width % ws) % ws
        pad_bottom = (ws - height % ws) % ws
        if pad_right or pad_bottom:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_right, 0, pad_bottom))
        padded_height, padded_width = height + pad_bottom, width + pad_right

        windows = (
            hidden_states.view(
                batch_size,
                padded_height // ws,
                ws,
                padded_width // ws,
                ws,
                embed_dim,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, ws * ws, embed_dim)
        )
        num_windows = windows.shape[0]
        head_dim = embed_dim // self.num_heads

        if not AARCH64 and embed_dim >= 1024:
            # On x86, splitting large QKV projections (3C >= 3072) into separate
            # packed Q/K/V linears cuts CPU Flash SDPA's row stride 3x (C vs 3C),
            # saving ~2.5 ms/image on Xeon; skipped on aarch64 where ACL prefers fused 3C.
            wq, wk, wv = self.qkv.weight.chunk(3)
            bq, bk, bv = (
                self.qkv.bias.chunk(3)
                if self.qkv.bias is not None
                else (None, None, None)
            )
            query = (
                F.linear(windows, wq, bq)
                .view(num_windows, ws * ws, self.num_heads, head_dim)
                .transpose(1, 2)
            )
            key = (
                F.linear(windows, wk, bk)
                .view(num_windows, ws * ws, self.num_heads, head_dim)
                .transpose(1, 2)
            )
            value = (
                F.linear(windows, wv, bv)
                .view(num_windows, ws * ws, self.num_heads, head_dim)
                .transpose(1, 2)
            )
        else:
            qkv = (
                self.qkv(windows)
                .reshape(num_windows, ws * ws, 3, self.num_heads, head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            query, key, value = qkv.unbind(0)
        windows = F.scaled_dot_product_attention(query, key, value, scale=self.scale)
        windows = self.proj(
            windows.transpose(1, 2).reshape(num_windows, ws * ws, embed_dim)
        )

        hidden_states = (
            windows.view(
                -1,
                padded_height // ws,
                padded_width // ws,
                ws,
                ws,
                embed_dim,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch_size, padded_height, padded_width, embed_dim)
        )
        if pad_right or pad_bottom:
            hidden_states = hidden_states[:, :height, :width]
        return hidden_states.reshape(batch_size, height * width, embed_dim)


class Florence2VisionSpatialBlock(nn.Module):
    def __init__(self, config, stage_idx: int):
        super().__init__()
        dim = config.embed_dim[stage_idx]
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.window_attn = Florence2VisionWindowAttention(config, stage_idx)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = Florence2VisionMLP(config, stage_idx)
        center = torch.zeros(1, 1, 3, 3)
        center[0, 0, 1, 1] = 1.0
        self.register_buffer("_dw_center", center, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run depthwise conv + local window attention + FFN on spatial features."""
        batch_size, embed_dim, height, width = hidden_states.shape

        hidden_states = _dwconv_residual(self.conv1, hidden_states, self._dw_center)
        residual = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.window_attn(
            self.norm1(residual).view(batch_size, height, width, embed_dim)
        )
        hidden_states = (
            (residual + hidden_states)
            .transpose(1, 2)
            .view(batch_size, embed_dim, height, width)
        )

        hidden_states = _dwconv_residual(self.conv2, hidden_states, self._dw_center)
        residual = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.ffn(self.norm2(residual))
        return (
            (residual + hidden_states)
            .transpose(1, 2)
            .view(batch_size, embed_dim, height, width)
        )


class Florence2VisionChannelBlock(nn.Module):
    def __init__(self, config, stage_idx: int):
        super().__init__()
        dim = config.embed_dim[stage_idx]
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.channel_attn = Florence2VisionChannelAttention(config, stage_idx)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = Florence2VisionMLP(config, stage_idx)
        center = torch.zeros(1, 1, 3, 3)
        center[0, 0, 1, 1] = 1.0
        self.register_buffer("_dw_center", center, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run depthwise conv + grouped channel attention + FFN on spatial features."""
        batch_size, embed_dim, height, width = hidden_states.shape

        hidden_states = _dwconv_residual(self.conv1, hidden_states, self._dw_center)
        residual = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.channel_attn(self.norm1(residual))
        hidden_states = (
            (residual + hidden_states)
            .transpose(1, 2)
            .view(batch_size, embed_dim, height, width)
        )

        hidden_states = _dwconv_residual(self.conv2, hidden_states, self._dw_center)
        residual = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.ffn(self.norm2(residual))
        return (
            (residual + hidden_states)
            .transpose(1, 2)
            .view(batch_size, embed_dim, height, width)
        )


class Florence2VisionBlock(nn.Module):
    def __init__(self, config, stage_idx: int):
        super().__init__()
        self.spatial_block = Florence2VisionSpatialBlock(config, stage_idx)
        self.channel_block = Florence2VisionChannelBlock(config, stage_idx)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run one DaViT block (spatial window block followed by channel block)."""
        return self.channel_block(self.spatial_block(hidden_states))


class Florence2VisionBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.num_groups = config.num_groups
        self.num_stages = len(self.embed_dim)

        self.convs = nn.ModuleList(
            [Florence2VisionConvEmbed(config, i) for i in range(self.num_stages)]
        )
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    Florence2VisionBlock(config, i) for _ in range(config.depths[i])
                )
                for i in range(self.num_stages)
            ]
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images [B, 3, H, W] through the four DaViT stages into [B, C, H', W']."""
        hidden_states = pixel_values.to(dtype=self.convs[0].conv.weight.dtype)
        for conv, block in zip(self.convs, self.blocks):
            hidden_states = conv(hidden_states)
            for layer in block:
                hidden_states = layer(hidden_states)
        return hidden_states


class Florence2VisionLearnedAbsolutePositionEmbedding2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_pos = config.vision_config.max_position_embeddings
        embedding_dim = config.vision_config.embed_dim[-1]
        self.row_embeddings = nn.Embedding(num_pos, embedding_dim // 2)
        self.column_embeddings = nn.Embedding(
            num_pos, embedding_dim - (embedding_dim // 2)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Build 2D row/column positional embeddings [1, C, H, W] for a spatial feature map."""
        height, width = pixel_values.shape[-2:]
        x_emb = self.column_embeddings(torch.arange(width, device=pixel_values.device))
        y_emb = self.row_embeddings(torch.arange(height, device=pixel_values.device))
        pos = torch.cat(
            [
                x_emb.unsqueeze(0).expand(height, -1, -1),
                y_emb.unsqueeze(1).expand(-1, width, -1),
            ],
            dim=-1,
        )
        return pos.permute(2, 0, 1).unsqueeze(0)


class Florence2VisionPositionalEmbeddingCosine1D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.vision_config.embed_dim[-1]
        self.max_seq_len = config.vision_config.max_temporal_embeddings
        pos_idx_to_embed = torch.empty((self.max_seq_len, self.embed_dim))
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(self.max_seq_len, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        pos_idx_to_embed[:, 0::2] = torch.sin(emb)
        pos_idx_to_embed[:, 1::2] = torch.cos(emb)
        self.register_buffer("pos_idx_to_embed", pos_idx_to_embed)

    def forward(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        """Return 1D cosine temporal positional embeddings for seq_embeds."""
        return self.pos_idx_to_embed[: seq_embeds.size(1)]


class Florence2MultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_projection = nn.Linear(
            config.vision_config.embed_dim[-1],
            config.vision_config.projection_dim,
            bias=False,
        )
        self.image_proj_norm = nn.LayerNorm(config.vision_config.projection_dim)
        self.image_position_embed = Florence2VisionLearnedAbsolutePositionEmbedding2D(
            config
        )
        self.visual_temporal_embed = Florence2VisionPositionalEmbeddingCosine1D(config)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Add 2D/temporal embeddings, pool global token, and project to language model dim."""
        position_features = (
            (image_features + self.image_position_embed(image_features))
            .flatten(2)
            .transpose(1, 2)
        )
        temporal_features = self.visual_temporal_embed(
            position_features[:, :1, :]
        ).unsqueeze(1)
        visual_token_features = (position_features + temporal_features).unsqueeze(1)
        image_features = torch.cat(
            [visual_token_features.mean(dim=2), visual_token_features.mean(dim=1)],
            dim=1,
        )
        return self.image_proj_norm(self.image_projection(image_features))


class Florence2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_tower = Florence2VisionBackbone(config.vision_config)
        self.multi_modal_projector = Florence2MultiModalProjector(config)
        self.language_model = BartModel(config.text_config)


class Florence2ForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Florence2Model(config)
        self.lm_head = nn.Linear(
            config.text_config.d_model, config.text_config.vocab_size, bias=False
        )
        if config.text_config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.shared.weight

    @property
    def decoder_layers(self):
        """Return the language model's decoder layer list."""
        return self.model.language_model.decoder.layers

    @torch.no_grad()
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the vision tower and multimodal projector to produce visual token embeddings."""
        return self.model.multi_modal_projector(self.model.vision_tower(pixel_values))

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Run one decoder step and return logits [B, vocab] (the compiled decode graph)."""
        return self.lm_head(self.model.language_model.decoder(input_ids, positions))

    @torch.no_grad()
    def encode(
        self,
        encoder_token_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scatter projected visual tokens into image_token_id slots and run the BART encoder."""
        language_model = self.model.language_model
        positions = (
            torch.arange(encoder_token_ids.shape[1], device=encoder_token_ids.device)
            .unsqueeze(0)
            .expand_as(encoder_token_ids)
        )
        inputs_embeds = language_model.shared(encoder_token_ids)
        if image_embeds is None and pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values)
        if image_embeds is not None:
            image_embeds = image_embeds.to(inputs_embeds.dtype)
            image_mask = encoder_token_ids == self.config.image_token_id
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1), image_embeds
            )
        return language_model.encoder(positions=positions, inputs_embeds=inputs_embeds)
