"""Tests for the speaker encoder module."""

import pytest
import torch

from leander_tts.model.speaker_encoder import SpeakerEncoder
from leander_tts.model.attention import CrossAttentionPooling, SpeakerCrossAttention


class TestCrossAttentionPooling:
    def test_output_shape(self):
        pool = CrossAttentionPooling(d_model=64, n_heads=4)
        x = torch.randn(2, 10, 64)  # B=2, T=10, D=64
        out = pool(x)
        assert out.shape == (2, 64)

    def test_with_mask(self):
        pool = CrossAttentionPooling(d_model=64, n_heads=4)
        x = torch.randn(2, 10, 64)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[0, 5:] = False  # Mask out second half for first batch
        out = pool(x, mask=mask)
        assert out.shape == (2, 64)

    def test_gradient_flow(self):
        pool = CrossAttentionPooling(d_model=64, n_heads=4)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = pool(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestSpeakerCrossAttention:
    def test_output_shape(self):
        xattn = SpeakerCrossAttention(d_model=64, n_heads=4)
        hidden = torch.randn(2, 20, 64)  # B=2, T=20, D=64
        speaker = torch.randn(2, 64)  # B=2, D=64
        out = xattn(hidden, speaker)
        assert out.shape == (2, 20, 64)

    def test_gate_starts_at_zero(self):
        xattn = SpeakerCrossAttention(d_model=64, n_heads=4)
        assert xattn.gate.item() == 0.0

    def test_residual_connection(self):
        """With gate=0, output should equal input (residual only)."""
        xattn = SpeakerCrossAttention(d_model=64, n_heads=4)
        hidden = torch.randn(2, 20, 64)
        speaker = torch.randn(2, 64)
        out = xattn(hidden, speaker)
        # Gate is 0, so tanh(0) = 0, output = residual
        torch.testing.assert_close(out, hidden)

    def test_gradient_flow(self):
        xattn = SpeakerCrossAttention(d_model=64, n_heads=4)
        hidden = torch.randn(2, 20, 64, requires_grad=True)
        speaker = torch.randn(2, 64, requires_grad=True)
        out = xattn(hidden, speaker)
        loss = out.sum()
        loss.backward()
        assert hidden.grad is not None
        assert speaker.grad is not None


class TestSpeakerEncoder:
    def test_output_shape(self):
        encoder = SpeakerEncoder(
            codebook_size=128,
            n_levels=3,
            d_model=64,
            output_dim=128,
            n_layers=2,
            n_heads=4,
            max_ref_frames=64,
        )
        # Simulate SNAC codes
        l1 = torch.randint(0, 128, (2, 10))  # B=2, T1=10
        l2 = torch.randint(0, 128, (2, 20))  # T2=2*T1
        l3 = torch.randint(0, 128, (2, 40))  # T3=4*T1
        codes = [l1, l2, l3]

        out = encoder(codes)
        assert out.shape == (2, 128)  # B=2, output_dim=128

    def test_gradient_flow(self):
        encoder = SpeakerEncoder(
            codebook_size=128,
            n_levels=3,
            d_model=64,
            output_dim=128,
            n_layers=2,
            n_heads=4,
        )
        l1 = torch.randint(0, 128, (2, 10))
        l2 = torch.randint(0, 128, (2, 20))
        l3 = torch.randint(0, 128, (2, 40))

        out = encoder([l1, l2, l3])
        loss = out.sum()
        loss.backward()

        # Check gradients exist on encoder parameters
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_different_batch_sizes(self):
        encoder = SpeakerEncoder(
            codebook_size=128, n_levels=3, d_model=64,
            output_dim=128, n_layers=2, n_heads=4,
        )
        for B in [1, 4, 8]:
            l1 = torch.randint(0, 128, (B, 10))
            l2 = torch.randint(0, 128, (B, 20))
            l3 = torch.randint(0, 128, (B, 40))
            out = encoder([l1, l2, l3])
            assert out.shape == (B, 128)
