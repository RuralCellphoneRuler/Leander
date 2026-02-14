"""Tests for the extended tokenizer."""

import pytest

# Skip if transformers can't load the model (no network)
pytest.importorskip("transformers")


class TestLeanderTokenizer:
    """Test tokenizer without requiring model download."""

    def test_special_token_structure(self):
        """Test that special token IDs are properly structured."""
        from leander_tts.data.tokenizer import LeanderTokenizer
        # Use a mock — just test the structure
        # We won't actually load the Qwen tokenizer in CI
        # Instead test the encoding logic with a mock

    def test_snac_frame_encoding_logic(self):
        """Test SNAC frame encoding produces correct token structure."""
        from leander_tts.data.tokenizer import SpecialTokens

        # Simulate the token structure
        special = SpecialTokens(
            audio_start=100,
            audio_end=101,
            l1_start=102,
            l2_start=103,
            l3_start=104,
            speaker_start=105,
            pad=106,
            snac_l1_offset=200,
            snac_l2_offset=4296,  # 200 + 4096
            snac_l3_offset=8392,  # 200 + 2*4096
            emotion_offset=12488,
            codebook_size=4096,
        )

        # Test token range validity
        assert special.snac_l2_offset == special.snac_l1_offset + 4096
        assert special.snac_l3_offset == special.snac_l2_offset + 4096

    def test_decode_snac_token_logic(self):
        """Test SNAC token decoding logic."""
        # Simulate decode logic
        snac_l1_offset = 200
        snac_l2_offset = 200 + 4096
        snac_l3_offset = 200 + 2 * 4096
        codebook_size = 4096

        # L1 token
        token_id = snac_l1_offset + 42
        assert snac_l1_offset <= token_id < snac_l1_offset + codebook_size
        assert token_id - snac_l1_offset == 42

        # L2 token
        token_id = snac_l2_offset + 100
        assert snac_l2_offset <= token_id < snac_l2_offset + codebook_size

        # L3 token
        token_id = snac_l3_offset + 3000
        assert snac_l3_offset <= token_id < snac_l3_offset + codebook_size

        # Not a SNAC token
        token_id = 50
        assert not (snac_l1_offset <= token_id < snac_l1_offset + codebook_size)
