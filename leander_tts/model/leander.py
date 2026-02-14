"""Full Leander-TTS model composition.

Wires together all components:
- SNAC codec (encode/decode)
- Speaker encoder (reference audio → embedding)
- LM (text + speaker → SNAC tokens)
- Streaming detokenizer (SNAC tokens → audio)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from leander_tts.config import DictConfig, load_config

from leander_tts.model.codec import SNACCodec
from leander_tts.model.speaker_encoder import SpeakerEncoder
from leander_tts.model.lm import LeanderLM
from leander_tts.model.detokenizer import StreamingDetokenizer
from leander_tts.data.tokenizer import LeanderTokenizer


class LeanderTTS(nn.Module):
    """Full Leander-TTS model."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        mc = config.model

        # Tokenizer (not nn.Module, handles vocab mapping)
        self.tokenizer = LeanderTokenizer(
            backbone=mc.lm.backbone,
            codebook_size=mc.codec.codebook_size,
        )

        # SNAC codec (frozen, pre-trained)
        self.codec = SNACCodec(
            model_name=mc.codec.model,
        )

        # Speaker encoder
        self.speaker_encoder = SpeakerEncoder(
            codebook_size=mc.codec.codebook_size,
            n_levels=mc.codec.n_levels,
            d_model=mc.speaker_encoder.d_model,
            output_dim=mc.speaker_encoder.output_dim,
            n_layers=mc.speaker_encoder.n_layers,
            n_heads=mc.speaker_encoder.n_heads,
        )

        # Language model
        self.lm = LeanderLM(
            backbone=mc.lm.backbone,
            n_new_tokens=self.tokenizer.n_new_tokens,
            d_model=mc.lm.d_model,
            n_layers=mc.lm.n_layers,
            inject_every_n_layers=mc.speaker_injection.inject_every_n_layers,
        )

        # Streaming detokenizer
        self.detokenizer = StreamingDetokenizer(
            codebook_size=mc.codec.codebook_size,
            n_levels=mc.codec.n_levels,
            channels=mc.detokenizer.channels,
            kernel_size=mc.detokenizer.kernel_size,
            dilations=list(mc.detokenizer.dilations),
            upsample_rates=list(mc.detokenizer.upsample_rates),
            window_size=mc.detokenizer.window_size,
        )

    def load_pretrained(
        self,
        codec_path: Optional[str] = None,
        speaker_encoder_path: Optional[str] = None,
        lm_path: Optional[str] = None,
        detokenizer_path: Optional[str] = None,
    ) -> None:
        """Load pre-trained weights for each component."""
        if codec_path:
            self.codec.load()

        if speaker_encoder_path:
            state = torch.load(speaker_encoder_path, map_location="cpu", weights_only=True)
            self.speaker_encoder.load_state_dict(state)

        if lm_path:
            state = torch.load(lm_path, map_location="cpu", weights_only=True)
            self.lm.load_state_dict(state)

        if detokenizer_path:
            state = torch.load(detokenizer_path, map_location="cpu", weights_only=True)
            self.detokenizer.load_state_dict(state)

    def encode_speaker(self, reference_audio: torch.Tensor) -> torch.Tensor:
        """Encode reference audio to speaker embedding.

        Args:
            reference_audio: [B, 1, T] waveform at 24kHz

        Returns:
            [B, D] speaker embedding
        """
        snac_codes = self.codec.encode(reference_audio)
        return self.speaker_encoder(snac_codes)

    def forward_train(
        self,
        text_tokens: torch.Tensor,
        snac_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_embedding: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            text_tokens: [B, T_text] text token IDs
            snac_tokens: [B, T_audio] SNAC token IDs (flattened with level markers)
            attention_mask: [B, T_total] attention mask
            speaker_embedding: [B, D] pre-computed speaker embedding
            labels: [B, T_total] target IDs (-100 for text positions)

        Returns:
            Dict with 'loss' and 'logits'
        """
        # Concatenate text and audio tokens
        input_ids = torch.cat([text_tokens, snac_tokens], dim=1)

        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            speaker_embedding=speaker_embedding,
            labels=labels,
        )

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        reference_audio: torch.Tensor,
        max_duration: float = 30.0,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        streaming: bool = True,
    ):
        """Generate speech from text with voice cloning.

        Args:
            text: Input text (may contain emotion tags)
            reference_audio: [1, 1, T] reference waveform at 24kHz
            max_duration: Maximum output duration in seconds
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            streaming: If True, yield audio chunks; if False, return full audio

        Yields (streaming=True) or Returns (streaming=False):
            torch.Tensor: Audio chunks [1, 1, T_chunk]
        """
        device = next(self.lm.parameters()).device

        # 1. Encode speaker
        speaker_emb = self.encode_speaker(reference_audio.to(device))

        # 2. Tokenize text
        text_tokens = self.tokenizer.encode_text(text)
        text_tokens.append(self.tokenizer.special.audio_start)
        input_ids = torch.tensor([text_tokens], device=device)

        # 3. Calculate max tokens from duration
        max_frames = int(max_duration * 12)  # 12 Hz L1 frame rate
        max_tokens = max_frames * 10  # 10 tokens per frame (with level markers)

        # 4. Generate
        context_frames: list[dict] = []
        current_frame_tokens: list[int] = []
        frame_position = 0  # track position in 10-token frame cycle

        audio_chunks = []

        for token_id in self.lm.generate_streaming(
            input_ids=input_ids,
            speaker_embedding=speaker_emb,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            audio_end_token_id=self.tokenizer.special.audio_end,
        ):
            # Skip level markers
            if self.tokenizer.is_level_marker(token_id):
                continue

            if self.tokenizer.is_audio_end(token_id):
                break

            # Decode SNAC token
            decoded = self.tokenizer.decode_snac_token(token_id)
            if decoded is None:
                continue

            current_frame_tokens.append(decoded)

            # Check if we have a complete frame (1 L1 + 2 L2 + 4 L3 = 7 tokens)
            if len(current_frame_tokens) == 7:
                frame = {
                    "l1": current_frame_tokens[0][1],
                    "l2": [current_frame_tokens[1][1], current_frame_tokens[2][1]],
                    "l3": [
                        current_frame_tokens[3][1],
                        current_frame_tokens[4][1],
                        current_frame_tokens[5][1],
                        current_frame_tokens[6][1],
                    ],
                }

                if streaming:
                    audio_chunk = self.detokenizer.decode_frame(
                        frame, context_frames=context_frames, device=device,
                    )
                    yield audio_chunk

                context_frames.append(frame)
                current_frame_tokens = []

        if not streaming:
            # Decode all frames at once
            if context_frames:
                codes = self.codec.unflatten_codes(context_frames, device=device)
                audio = self.detokenizer(codes[0], codes[1], codes[2])
                return audio

    @classmethod
    def from_config(cls, config_path: str | Path) -> "LeanderTTS":
        """Create model from a YAML config file."""
        config = load_config(config_path)
        return cls(config)

    def get_param_summary(self) -> dict[str, int]:
        """Get parameter count for each component."""
        return {
            "speaker_encoder": sum(p.numel() for p in self.speaker_encoder.parameters()),
            "lm": self.lm.get_total_params() if self.lm.backbone else 0,
            "detokenizer": sum(p.numel() for p in self.detokenizer.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
