"""Streaming inference pipeline for Leander-TTS."""

from __future__ import annotations

import time
from typing import Iterator, Optional

import torch
from leander_tts.config import DictConfig, load_config

from leander_tts.model.leander import LeanderTTS
from leander_tts.data.text_normalizer import normalize_german_text


class StreamingTTSPipeline:
    """High-level streaming TTS pipeline.

    Handles text normalization, speaker encoding, token generation,
    and streaming audio decoding in a single interface.
    """

    def __init__(
        self,
        model: LeanderTTS,
        device: str = "cuda",
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        self._cached_speaker_embedding: Optional[torch.Tensor] = None
        self._cached_speaker_ref_hash: Optional[int] = None

    def set_speaker(self, reference_audio: torch.Tensor) -> None:
        """Pre-compute and cache speaker embedding from reference audio.

        Args:
            reference_audio: [1, 1, T] waveform at 24kHz (3-10 seconds recommended)
        """
        ref_hash = hash(reference_audio.data_ptr())
        if ref_hash == self._cached_speaker_ref_hash:
            return  # Already cached

        with torch.inference_mode():
            self._cached_speaker_embedding = self.model.encode_speaker(
                reference_audio.to(self.device)
            )
        self._cached_speaker_ref_hash = ref_hash

    def synthesize_streaming(
        self,
        text: str,
        reference_audio: Optional[torch.Tensor] = None,
        max_duration: float = 30.0,
        normalize_text: bool = True,
    ) -> Iterator[torch.Tensor]:
        """Generate speech from text, yielding audio chunks for streaming playback.

        Args:
            text: Input text (may contain emotion tags like <lacht>, <flüstert>)
            reference_audio: [1, 1, T] reference audio for voice cloning.
                If None, uses previously cached speaker embedding.
            max_duration: Maximum output duration in seconds
            normalize_text: Whether to apply German text normalization

        Yields:
            [1, 1, T_chunk] audio chunks (~83ms each at 24kHz)
        """
        if reference_audio is not None:
            self.set_speaker(reference_audio)

        if self._cached_speaker_embedding is None:
            raise ValueError("No speaker reference set. Call set_speaker() first.")

        if normalize_text:
            text = normalize_german_text(text)

        yield from self.model.generate(
            text=text,
            reference_audio=torch.zeros(1, 1, 24000, device=self.device),  # dummy, embedding cached
            max_duration=max_duration,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            streaming=True,
        )

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        reference_audio: Optional[torch.Tensor] = None,
        max_duration: float = 30.0,
        normalize_text: bool = True,
    ) -> torch.Tensor:
        """Generate complete audio (non-streaming).

        Args:
            text: Input text
            reference_audio: [1, 1, T] reference audio
            max_duration: Maximum output duration in seconds
            normalize_text: Whether to apply German text normalization

        Returns:
            [1, 1, T] complete audio waveform at 24kHz
        """
        chunks = list(self.synthesize_streaming(
            text=text,
            reference_audio=reference_audio,
            max_duration=max_duration,
            normalize_text=normalize_text,
        ))

        if not chunks:
            return torch.zeros(1, 1, 0, device=self.device)

        return torch.cat(chunks, dim=2)

    @classmethod
    def from_pretrained(
        cls,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        **kwargs,
    ) -> "StreamingTTSPipeline":
        """Load a pre-trained pipeline from config and checkpoint."""
        model = LeanderTTS.from_config(config_path)
        model.load_pretrained(lm_path=checkpoint_path)
        model = model.to(device)
        model.eval()
        return cls(model=model, device=device, **kwargs)
