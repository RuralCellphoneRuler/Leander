"""Evaluation metrics for Leander-TTS."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def compute_wer_whisper(
    audio: torch.Tensor,
    reference_text: str,
    language: str = "de",
    model_size: str = "large-v3",
) -> float:
    """Compute WER using Whisper ASR.

    Args:
        audio: [1, T] or [T] waveform at 16kHz (Whisper's native rate)
        reference_text: ground truth text
        language: language code
        model_size: Whisper model size

    Returns:
        Word Error Rate (0.0 = perfect)
    """
    import whisper

    model = whisper.load_model(model_size)

    # Ensure correct shape
    if audio.dim() == 2:
        audio = audio.squeeze(0)
    audio_np = audio.cpu().numpy()

    result = model.transcribe(audio_np, language=language)
    hypothesis = result["text"].strip().lower()
    reference = reference_text.strip().lower()

    # Simple WER computation
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Edit distance
    d = _edit_distance(ref_words, hyp_words)
    wer = d / max(len(ref_words), 1)
    return wer


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Compute Levenshtein edit distance between two word lists."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m]


def compute_speaker_similarity(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
) -> float:
    """Compute cosine similarity between two speaker embeddings.

    Args:
        embedding_a: [D] or [1, D] speaker embedding
        embedding_b: [D] or [1, D] speaker embedding

    Returns:
        Cosine similarity (-1 to 1, higher = more similar)
    """
    a = embedding_a.flatten()
    b = embedding_b.flatten()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def compute_rtf(
    generation_time: float,
    audio_duration: float,
) -> float:
    """Compute Real-Time Factor.

    RTF < 1.0 means faster than real-time.

    Args:
        generation_time: time to generate audio (seconds)
        audio_duration: duration of generated audio (seconds)

    Returns:
        Real-Time Factor
    """
    return generation_time / max(audio_duration, 1e-6)
