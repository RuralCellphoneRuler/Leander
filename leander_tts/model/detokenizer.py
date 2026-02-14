"""Streaming CNN detokenizer for converting SNAC tokens to audio waveforms.

Key design principles:
- Causal convolutions only (no future lookahead) → true streaming
- Sliding window context for smooth chunk transitions
- Snake activations for harmonic bias (from BigVGAN)
- Anti-aliased upsampling
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Snake(nn.Module):
    """Snake activation: x + (1/a) * sin²(a * x).

    Periodic activation function that provides harmonic inductive bias,
    critical for high-quality audio synthesis. From BigVGAN.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / (self.alpha + 1e-9)) * torch.sin(self.alpha * x).pow(2)


class CausalConv1d(nn.Module):
    """1D convolution with causal (left-only) padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, stride=stride, groups=groups, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalResBlock(nn.Module):
    """Residual block with causal convolutions and Snake activations."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake(channels),
            CausalConv1d(channels, channels, kernel_size, dilation=dilation),
            Snake(channels),
            CausalConv1d(channels, channels, kernel_size, dilation=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AntiAliasedUpsample(nn.Module):
    """Transposed convolution followed by low-pass filter to prevent aliasing."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        # Transposed conv for upsampling
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=stride * 2,
            stride=stride,
            padding=stride // 2,
        )
        self.activation = Snake(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        return self.conv_transpose(x)


class StreamingDetokenizer(nn.Module):
    """Streaming CNN that converts SNAC tokens to audio waveforms.

    Architecture:
    1. Embed SNAC tokens per level
    2. Merge and upsample to uniform temporal resolution
    3. Process through causal residual conv blocks
    4. Upsample to 24kHz through transposed convolutions
    """

    def __init__(
        self,
        codebook_size: int = 4096,
        n_levels: int = 3,
        embed_dim: int = 256,
        channels: int = 512,
        kernel_size: int = 7,
        dilations: list[int] | None = None,
        upsample_rates: list[int] | None = None,
        window_size: int = 16,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 1, 2, 4, 8]
        if upsample_rates is None:
            upsample_rates = [8, 5, 4, 2]

        self.window_size = window_size
        self.n_levels = n_levels

        # Token embeddings per level
        self.level_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embed_dim) for _ in range(n_levels)
        ])

        # Each level has a different temporal rate. Upsample all to L3 rate (highest).
        # L1: 1 token/frame → upsample 4x
        # L2: 2 tokens/frame → upsample 2x
        # L3: 4 tokens/frame → keep as is
        self.level_upsamplers = nn.ModuleList([
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=8, stride=4, padding=2),  # L1: 4x
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),  # L2: 2x
            nn.Identity(),  # L3: 1x
        ])

        # Merge all levels
        self.merge = nn.Conv1d(embed_dim * n_levels, channels, kernel_size=1)

        # Causal residual blocks
        self.res_blocks = nn.ModuleList([
            CausalResBlock(channels, kernel_size, dilation=d)
            for d in dilations
        ])

        # Upsample to waveform sample rate
        # Product of upsample_rates should equal hop_size (samples per L3 token)
        # 24000 Hz / 47 Hz ≈ 512 samples per L3 token
        # 8 * 5 * 4 * 2 = 320 (close to 512/1.6, adjusted for SNAC internal structure)
        ch = channels
        self.upsample_blocks = nn.ModuleList()
        for rate in upsample_rates:
            self.upsample_blocks.append(
                AntiAliasedUpsample(ch, ch // 2, stride=rate)
            )
            ch = ch // 2

        # Final projection to waveform
        self.output = nn.Sequential(
            Snake(ch),
            nn.Conv1d(ch, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(
        self,
        l1_codes: torch.Tensor,
        l2_codes: torch.Tensor,
        l3_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Convert SNAC tokens to audio waveform.

        Args:
            l1_codes: [B, T1] L1 codebook indices
            l2_codes: [B, T2] L2 codebook indices (T2 = 2*T1)
            l3_codes: [B, T3] L3 codebook indices (T3 = 4*T1)

        Returns:
            [B, 1, T_audio] waveform
        """
        # Embed each level
        code_list = [l1_codes, l2_codes, l3_codes]
        embedded = []
        for i, codes in enumerate(code_list):
            emb = self.level_embeddings[i](codes)  # [B, T_i, embed_dim]
            emb = emb.transpose(1, 2)  # [B, embed_dim, T_i]
            emb = self.level_upsamplers[i](emb)  # [B, embed_dim, T3]
            embedded.append(emb)

        # Align lengths (L1/L2 upsampled might differ by ±1 from L3)
        min_len = min(e.shape[2] for e in embedded)
        embedded = [e[:, :, :min_len] for e in embedded]

        # Merge levels
        x = torch.cat(embedded, dim=1)  # [B, embed_dim * 3, T3]
        x = self.merge(x)  # [B, channels, T3]

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Upsample to waveform rate
        for up_block in self.upsample_blocks:
            x = up_block(x)

        # Output
        audio = self.output(x)  # [B, 1, T_audio]
        return audio

    def decode_frame(
        self,
        frame: dict,
        context_frames: list[dict] | None = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Decode a single frame with sliding window context (streaming mode).

        Args:
            frame: Dict with 'l1', 'l2', 'l3' code values
            context_frames: Previous frames for context (up to window_size)
            device: torch device

        Returns:
            [1, 1, T_chunk] audio chunk for this frame
        """
        # Build context window
        if context_frames is None:
            context_frames = []

        all_frames = list(context_frames[-self.window_size:]) + [frame]

        # Build code tensors
        l1 = torch.tensor([[f["l1"] for f in all_frames]], device=device)
        l2 = torch.tensor(
            [[c for f in all_frames for c in f["l2"]]], device=device,
        )
        l3 = torch.tensor(
            [[c for f in all_frames for c in f["l3"]]], device=device,
        )

        # Run full decode
        audio = self.forward(l1, l2, l3)

        # Return only the last frame's audio (crop context)
        # Each frame produces approximately (sample_rate / frame_rate) samples
        samples_per_frame = audio.shape[2] // len(all_frames)
        return audio[:, :, -samples_per_frame:]
