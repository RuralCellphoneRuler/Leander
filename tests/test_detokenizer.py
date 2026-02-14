"""Tests for the streaming CNN detokenizer."""

import pytest
import torch

from leander_tts.model.detokenizer import (
    Snake,
    CausalConv1d,
    CausalResBlock,
    StreamingDetokenizer,
)


class TestSnake:
    def test_output_shape(self):
        snake = Snake(channels=32)
        x = torch.randn(2, 32, 100)
        out = snake(x)
        assert out.shape == x.shape

    def test_not_relu(self):
        """Snake should not zero out negative values like ReLU."""
        snake = Snake(channels=1)
        x = torch.tensor([[[-1.0, -0.5, 0.0, 0.5, 1.0]]])
        out = snake(x)
        # Snake adds sin^2 term, so negative inputs should not be zeroed
        assert out[0, 0, 0].item() != 0.0


class TestCausalConv1d:
    def test_output_length(self):
        """Causal conv should not change temporal dimension."""
        conv = CausalConv1d(32, 64, kernel_size=7)
        x = torch.randn(2, 32, 100)
        out = conv(x)
        assert out.shape == (2, 64, 100)

    def test_causal_no_future(self):
        """Causal conv output at time t should not depend on future inputs."""
        conv = CausalConv1d(1, 1, kernel_size=3)
        x = torch.randn(1, 1, 10)

        # Forward with full input
        out_full = conv(x)

        # Forward with truncated input (first 5 timesteps only)
        out_partial = conv(x[:, :, :5])

        # First 5 outputs should be identical (causal = no future dependency)
        torch.testing.assert_close(
            out_full[:, :, :5], out_partial, atol=1e-5, rtol=1e-5,
        )


class TestCausalResBlock:
    def test_residual_connection(self):
        block = CausalResBlock(channels=32, kernel_size=3, dilation=1)
        x = torch.randn(2, 32, 50)
        out = block(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        block = CausalResBlock(channels=32, kernel_size=3, dilation=2)
        x = torch.randn(2, 32, 50, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestStreamingDetokenizer:
    @pytest.fixture
    def detokenizer(self):
        return StreamingDetokenizer(
            codebook_size=128,
            n_levels=3,
            embed_dim=32,
            channels=64,
            kernel_size=3,
            dilations=[1, 2, 1, 2],
            upsample_rates=[4, 4, 2, 2],
            window_size=8,
        )

    def test_forward_output_shape(self, detokenizer):
        """Test that forward produces audio output."""
        T1 = 10
        l1 = torch.randint(0, 128, (2, T1))
        l2 = torch.randint(0, 128, (2, T1 * 2))
        l3 = torch.randint(0, 128, (2, T1 * 4))
        out = detokenizer(l1, l2, l3)
        assert out.shape[0] == 2  # batch
        assert out.shape[1] == 1  # mono
        assert out.shape[2] > 0  # some audio samples

    def test_output_is_bounded(self, detokenizer):
        """Tanh output should be in [-1, 1]."""
        l1 = torch.randint(0, 128, (1, 5))
        l2 = torch.randint(0, 128, (1, 10))
        l3 = torch.randint(0, 128, (1, 20))
        out = detokenizer(l1, l2, l3)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_gradient_flow(self, detokenizer):
        """Test that gradients flow through the detokenizer."""
        l1 = torch.randint(0, 128, (1, 5))
        l2 = torch.randint(0, 128, (1, 10))
        l3 = torch.randint(0, 128, (1, 20))
        out = detokenizer(l1, l2, l3)
        loss = out.sum()
        loss.backward()
        for name, p in detokenizer.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_streaming_decode_frame(self, detokenizer):
        """Test single-frame streaming decoding."""
        frame = {"l1": 0, "l2": [0, 1], "l3": [0, 1, 2, 3]}
        chunk = detokenizer.decode_frame(frame, context_frames=[])
        assert chunk.shape[0] == 1
        assert chunk.shape[1] == 1
        assert chunk.shape[2] > 0

    def test_streaming_with_context(self, detokenizer):
        """Test streaming with accumulated context frames."""
        frames = [
            {"l1": i, "l2": [i, i + 1], "l3": [i, i + 1, i + 2, i + 3]}
            for i in range(5)
        ]

        context = []
        for frame in frames:
            chunk = detokenizer.decode_frame(frame, context_frames=context)
            assert chunk.shape[2] > 0
            context.append(frame)

    def test_param_count(self, detokenizer):
        """Detokenizer should be relatively lightweight."""
        n_params = sum(p.numel() for p in detokenizer.parameters())
        # Should be under 50M parameters
        assert n_params < 50_000_000, f"Too many params: {n_params:,}"
