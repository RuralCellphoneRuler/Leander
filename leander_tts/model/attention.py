"""Attention modules for Leander-TTS.

Contains:
- CrossAttentionPooling: Learnable query that pools a variable-length sequence into a single vector
- SpeakerCrossAttention: Injects speaker embedding into LLM hidden states
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionPooling(nn.Module):
    """Pool a variable-length sequence into a fixed-size vector using a learnable query.

    Uses a single learnable query token that attends to all positions in the input
    sequence via cross-attention, producing a single output vector.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool input sequence to single vector.

        Args:
            x: [B, T, D] input sequence
            mask: [B, T] boolean mask, True for valid positions

        Returns:
            [B, D] pooled vector
        """
        B, T, D = x.shape
        query = self.query.expand(B, -1, -1)  # [B, 1, D]

        q = self.q_proj(query)  # [B, 1, D]
        k = self.k_proj(x)  # [B, T, D]
        v = self.v_proj(x)  # [B, T, D]

        # Reshape for multi-head attention
        q = q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, Dh]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, Dh]
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, Dh]

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, 1, T]

        if mask is not None:
            # mask: [B, T] → [B, 1, 1, T]
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, 1, Dh]
        out = out.transpose(1, 2).reshape(B, 1, D)  # [B, 1, D]
        out = self.out_proj(out)  # [B, 1, D]
        out = self.norm(out.squeeze(1))  # [B, D]

        return out


class SpeakerCrossAttention(nn.Module):
    """Cross-attention layer that injects speaker embedding into LLM hidden states.

    Inserted at regular intervals in the LLM backbone to condition generation
    on the target speaker identity.

    hidden_state = hidden_state + alpha * CrossAttn(Q=hidden, KV=speaker_emb)

    The alpha is a learnable gate initialized to 0, allowing gradual integration
    during training (similar to ControlNet / IP-Adapter approach).
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.layer_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Learnable gate initialized to 0 → no effect at start of training
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Inject speaker conditioning into hidden states.

        Args:
            hidden_states: [B, T, D] from the LLM layer
            speaker_embedding: [B, D] global speaker embedding

        Returns:
            [B, T, D] conditioned hidden states (residual addition)
        """
        B, T, D = hidden_states.shape

        residual = hidden_states
        x = self.layer_norm(hidden_states)

        # Query from hidden states, Key/Value from speaker embedding
        q = self.q_proj(x)  # [B, T, D]

        # Expand speaker embedding to be a single KV token
        spk = speaker_embedding.unsqueeze(1)  # [B, 1, D]
        k = self.k_proj(spk)  # [B, 1, D]
        v = self.v_proj(spk)  # [B, 1, D]

        # Multi-head reshape
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, Dh]
        k = k.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, Dh]
        v = v.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, Dh]

        # Attention (no mask needed — single KV token)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, 1]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, T, Dh]
        out = out.transpose(1, 2).reshape(B, T, D)  # [B, T, D]
        out = self.out_proj(out)

        # Gated residual connection
        return residual + self.gate.tanh() * out
