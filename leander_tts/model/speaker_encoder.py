"""Speaker encoder for extracting global speaker embeddings from reference audio.

BiCodec-inspired design: encode reference SNAC tokens into a single fixed-size
speaker embedding vector that captures speaker identity independently of content.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from leander_tts.model.attention import CrossAttentionPooling


class SpeakerEncoder(nn.Module):
    """Extract a global speaker embedding from SNAC-encoded reference audio.

    Architecture:
    1. Embed SNAC tokens from all 3 levels
    2. Process with Transformer encoder (self-attention)
    3. Pool to single vector via learned cross-attention query
    4. Project to LLM d_model dimension
    """

    def __init__(
        self,
        codebook_size: int = 4096,
        n_levels: int = 3,
        d_model: int = 768,
        output_dim: int = 1536,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_ref_frames: int = 256,  # ~21 seconds at 12 Hz
    ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.n_levels = n_levels

        # Separate embedding per level
        self.level_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, d_model) for _ in range(n_levels)
        ])

        # Level indicator embedding (which level is this token from)
        self.level_indicator = nn.Embedding(n_levels, d_model)

        # Positional embedding (frame-level, not token-level)
        self.pos_embedding = nn.Embedding(max_ref_frames, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        # Cross-attention pooling → single vector
        self.pool = CrossAttentionPooling(d_model, n_heads=n_heads, dropout=dropout)

        # Project to LLM dimension
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        snac_codes: list[torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode reference audio SNAC tokens to global speaker embedding.

        Args:
            snac_codes: List of 3 tensors:
                - L1: [B, T1] codebook indices
                - L2: [B, T2] where T2 = 2*T1
                - L3: [B, T3] where T3 = 4*T1
            mask: Optional [B, T_total] mask for padded sequences

        Returns:
            [B, output_dim] global speaker embedding
        """
        B = snac_codes[0].shape[0]
        T1 = snac_codes[0].shape[1]

        embedded_tokens = []

        # Embed each level's tokens with level indicator + positional encoding
        for level_idx, codes in enumerate(snac_codes):
            # codes: [B, T_level]
            T_level = codes.shape[1]
            tokens_per_frame = 2 ** level_idx  # 1, 2, 4

            emb = self.level_embeddings[level_idx](codes)  # [B, T_level, D]
            level_emb = self.level_indicator(
                torch.full((B, T_level), level_idx, device=codes.device, dtype=torch.long)
            )  # [B, T_level, D]

            # Frame-level positional encoding (shared across tokens within a frame)
            frame_indices = torch.arange(T_level, device=codes.device) // tokens_per_frame
            frame_indices = frame_indices.clamp(max=self.pos_embedding.num_embeddings - 1)
            pos_emb = self.pos_embedding(frame_indices).unsqueeze(0).expand(B, -1, -1)

            embedded_tokens.append(emb + level_emb + pos_emb)

        # Concatenate all levels: [B, T1 + T2 + T3, D]
        x = torch.cat(embedded_tokens, dim=1)

        # Transformer encoding
        if mask is not None:
            # Convert padding mask to attention mask format
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Pool to single vector
        pooled = self.pool(x, mask=mask)  # [B, D]

        # Project to output dimension
        return self.output_proj(pooled)  # [B, output_dim]

    def encode_audio(
        self,
        codec: "SNACCodec",
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience: encode raw audio directly to speaker embedding.

        Args:
            codec: SNACCodec instance
            audio: [B, 1, T] waveform at 24kHz

        Returns:
            [B, output_dim] speaker embedding
        """
        codes = codec.encode(audio)
        return self.forward(codes)
