"""Tests for the SNAC codec wrapper."""

import pytest
import torch

from leander_tts.model.codec import SNACCodec


class TestSNACCodecInterface:
    """Test codec interface without loading actual model."""

    def test_properties(self):
        codec = SNACCodec()
        assert codec.sample_rate == 24000
        assert codec.n_levels == 3
        assert codec.codebook_size == 4096
        assert codec.tokens_per_frame == 7

    def test_flatten_codes(self):
        codec = SNACCodec()

        # Simulate codec output
        T1 = 5  # 5 frames
        l1 = torch.randint(0, 4096, (1, T1))
        l2 = torch.randint(0, 4096, (1, T1 * 2))
        l3 = torch.randint(0, 4096, (1, T1 * 4))

        frames = codec.flatten_codes([l1, l2, l3])
        assert len(frames) == 1  # batch size 1
        assert len(frames[0]) == T1  # 5 frames

        # Check frame structure
        for frame in frames[0]:
            assert "l1" in frame
            assert "l2" in frame
            assert "l3" in frame
            assert isinstance(frame["l1"], int)
            assert len(frame["l2"]) == 2
            assert len(frame["l3"]) == 4

    def test_unflatten_codes(self):
        codec = SNACCodec()

        # Create frames
        frames = [
            {"l1": i, "l2": [i * 2, i * 2 + 1], "l3": [i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3]}
            for i in range(5)
        ]

        codes = codec.unflatten_codes(frames)
        assert len(codes) == 3  # 3 levels
        assert codes[0].shape == (1, 5)
        assert codes[1].shape == (1, 10)
        assert codes[2].shape == (1, 20)

    def test_roundtrip_flatten_unflatten(self):
        codec = SNACCodec()

        T1 = 8
        l1 = torch.randint(0, 4096, (1, T1))
        l2 = torch.randint(0, 4096, (1, T1 * 2))
        l3 = torch.randint(0, 4096, (1, T1 * 4))

        # Flatten
        frames = codec.flatten_codes([l1, l2, l3])[0]

        # Unflatten
        codes = codec.unflatten_codes(frames)

        # Should match original
        torch.testing.assert_close(codes[0], l1)
        torch.testing.assert_close(codes[1], l2)
        torch.testing.assert_close(codes[2], l3)

    def test_batch_flatten(self):
        codec = SNACCodec()

        T1 = 4
        B = 3
        l1 = torch.randint(0, 4096, (B, T1))
        l2 = torch.randint(0, 4096, (B, T1 * 2))
        l3 = torch.randint(0, 4096, (B, T1 * 4))

        frames = codec.flatten_codes([l1, l2, l3])
        assert len(frames) == B
        for batch_frames in frames:
            assert len(batch_frames) == T1
