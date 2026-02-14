"""SNAC codec wrapper for encoding/decoding audio to/from discrete tokens."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SNACCodec(nn.Module):
    """Wrapper around the SNAC neural audio codec.

    Provides a clean interface for:
    - Encoding audio waveforms to multi-scale discrete tokens
    - Decoding tokens back to audio waveforms
    - Flattening/unflattening the multi-scale token structure
    """

    def __init__(
        self,
        model_name: str = "hubertsiuzdak/snac_24khz",
        device: str = "cpu",
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self._model: Optional[nn.Module] = None

    def load(self) -> None:
        """Load the pre-trained SNAC model."""
        from snac import SNAC  # type: ignore
        self._model = SNAC.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self.load()
        return self._model  # type: ignore

    @property
    def sample_rate(self) -> int:
        return 24000

    @property
    def n_levels(self) -> int:
        return 3

    @property
    def codebook_size(self) -> int:
        return 4096

    @property
    def tokens_per_frame(self) -> int:
        """Total tokens per time frame: 1 (L1) + 2 (L2) + 4 (L3) = 7."""
        return 7

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor) -> list[torch.Tensor]:
        """Encode audio waveform to multi-scale SNAC tokens.

        Args:
            audio: [B, 1, T] waveform at 24kHz

        Returns:
            List of 3 tensors, one per level:
            - L1: [B, T1] where T1 ≈ T/2000 (12 Hz)
            - L2: [B, T2] where T2 ≈ T/1043 (23 Hz)
            - L3: [B, T3] where T3 ≈ T/512 (47 Hz)
        """
        codes = self.model.encode(audio)
        return codes

    @torch.inference_mode()
    def decode(self, codes: list[torch.Tensor]) -> torch.Tensor:
        """Decode multi-scale SNAC tokens back to audio.

        Args:
            codes: List of 3 tensors [L1, L2, L3] (same format as encode output)

        Returns:
            audio: [B, 1, T] reconstructed waveform at 24kHz
        """
        audio = self.model.decode(codes)
        return audio

    def flatten_codes(self, codes: list[torch.Tensor]) -> list[list[dict]]:
        """Flatten multi-scale codes into frame-aligned token sequences.

        The SNAC codec produces tokens at different rates:
        - L1: 12 Hz (1 token per frame)
        - L2: 23 Hz (2 tokens per frame)
        - L3: 47 Hz (4 tokens per frame)

        This interleaves them into frames of 7 tokens each.

        Args:
            codes: List of [B, T_level] tensors from encode()

        Returns:
            List (batch) of lists (frames) of dicts with keys:
            {'l1': int, 'l2': [int, int], 'l3': [int, int, int, int]}
        """
        batch_size = codes[0].shape[0]
        l1, l2, l3 = codes[0], codes[1], codes[2]

        results = []
        for b in range(batch_size):
            frames = []
            n_frames = l1.shape[1]
            for i in range(n_frames):
                frame = {
                    "l1": l1[b, i].item(),
                    "l2": [l2[b, 2 * i].item(), l2[b, 2 * i + 1].item()],
                    "l3": [
                        l3[b, 4 * i].item(),
                        l3[b, 4 * i + 1].item(),
                        l3[b, 4 * i + 2].item(),
                        l3[b, 4 * i + 3].item(),
                    ],
                }
                frames.append(frame)
            results.append(frames)
        return results

    def unflatten_codes(
        self, frames: list[dict], device: str = "cpu",
    ) -> list[torch.Tensor]:
        """Convert frame-aligned tokens back to multi-scale code tensors.

        Args:
            frames: List of frame dicts from flatten_codes (single example)

        Returns:
            List of 3 tensors [L1, L2, L3] each with batch dim 1.
        """
        n_frames = len(frames)
        l1 = torch.zeros(1, n_frames, dtype=torch.long, device=device)
        l2 = torch.zeros(1, 2 * n_frames, dtype=torch.long, device=device)
        l3 = torch.zeros(1, 4 * n_frames, dtype=torch.long, device=device)

        for i, frame in enumerate(frames):
            l1[0, i] = frame["l1"]
            l2[0, 2 * i] = frame["l2"][0]
            l2[0, 2 * i + 1] = frame["l2"][1]
            l3[0, 4 * i] = frame["l3"][0]
            l3[0, 4 * i + 1] = frame["l3"][1]
            l3[0, 4 * i + 2] = frame["l3"][2]
            l3[0, 4 * i + 3] = frame["l3"][3]

        return [l1, l2, l3]

    @torch.inference_mode()
    def encode_to_flat(self, audio: torch.Tensor) -> list[list[dict]]:
        """Convenience: encode audio directly to flattened frame sequences."""
        codes = self.encode(audio)
        return self.flatten_codes(codes)

    @torch.inference_mode()
    def decode_from_flat(self, frames: list[dict]) -> torch.Tensor:
        """Convenience: decode flattened frame sequence directly to audio."""
        codes = self.unflatten_codes(frames, device=self.device)
        return self.decode(codes)
