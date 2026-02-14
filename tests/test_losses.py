"""Tests for loss functions."""

import pytest
import torch

from leander_tts.training.losses import (
    MultiScaleSTFTLoss,
    SpeakerContrastiveLoss,
    SpeakerConsistencyLoss,
    MelSpectrogramLoss,
)


class TestMultiScaleSTFTLoss:
    def test_zero_loss_identical(self):
        loss_fn = MultiScaleSTFTLoss(fft_sizes=[256], hop_sizes=[64], win_sizes=[256])
        audio = torch.randn(1, 1, 8000)
        loss = loss_fn(audio, audio)
        assert loss.item() < 0.1  # Should be very small

    def test_nonzero_loss_different(self):
        loss_fn = MultiScaleSTFTLoss(fft_sizes=[256], hop_sizes=[64], win_sizes=[256])
        a = torch.randn(1, 1, 8000)
        b = torch.randn(1, 1, 8000)
        loss = loss_fn(a, b)
        assert loss.item() > 0.0

    def test_gradient_flow(self):
        loss_fn = MultiScaleSTFTLoss(fft_sizes=[256], hop_sizes=[64], win_sizes=[256])
        pred = torch.randn(1, 1, 4000, requires_grad=True)
        target = torch.randn(1, 1, 4000)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestSpeakerContrastiveLoss:
    def test_identical_embeddings(self):
        loss_fn = SpeakerContrastiveLoss(temperature=0.07)
        emb = torch.randn(4, 64)
        # When a and b are identical, loss should be low
        loss = loss_fn(emb, emb)
        assert loss.item() < 2.0  # Less than random

    def test_gradient_flow(self):
        loss_fn = SpeakerContrastiveLoss()
        a = torch.randn(4, 64, requires_grad=True)
        b = torch.randn(4, 64, requires_grad=True)
        loss = loss_fn(a, b)
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_batch_size_invariance(self):
        loss_fn = SpeakerContrastiveLoss()
        # Should work with different batch sizes
        for B in [2, 4, 8]:
            a = torch.randn(B, 64)
            b = torch.randn(B, 64)
            loss = loss_fn(a, b)
            assert loss.item() > 0.0


class TestSpeakerConsistencyLoss:
    def test_identical_zero_loss(self):
        loss_fn = SpeakerConsistencyLoss(target_similarity=0.9)
        emb = torch.randn(4, 64)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        # Same embedding → similarity = 1.0 > 0.9 → loss = 0
        loss = loss_fn(emb, emb)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_nonzero_loss(self):
        loss_fn = SpeakerConsistencyLoss(target_similarity=0.9)
        # Orthogonal embeddings → similarity ≈ 0
        a = torch.zeros(1, 4)
        a[0, 0] = 1.0
        b = torch.zeros(1, 4)
        b[0, 1] = 1.0
        loss = loss_fn(a, b)
        assert loss.item() > 0.0
