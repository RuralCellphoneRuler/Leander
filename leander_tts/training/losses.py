"""Loss functions for Leander-TTS training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale STFT loss for audio reconstruction quality.

    Computes spectral convergence + log magnitude loss at multiple STFT resolutions.
    Used in Phase 1 (codec) and Phase 4 (detokenizer) training.
    """

    def __init__(
        self,
        fft_sizes: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        win_sizes: list[int] | None = None,
    ):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [512, 1024, 2048]
        if hop_sizes is None:
            hop_sizes = [128, 256, 512]
        if win_sizes is None:
            win_sizes = [512, 1024, 2048]

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def _stft_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_stft = torch.stft(
            x.squeeze(1), fft_size, hop_size, win_size,
            window=torch.hann_window(win_size, device=x.device),
            return_complex=True,
        )
        y_stft = torch.stft(
            y.squeeze(1), fft_size, hop_size, win_size,
            window=torch.hann_window(win_size, device=y.device),
            return_complex=True,
        )

        x_mag = x_stft.abs()
        y_mag = y_stft.abs()

        # Spectral convergence
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-8)

        # Log magnitude
        log_loss = F.l1_loss(
            torch.log(x_mag + 1e-8),
            torch.log(y_mag + 1e-8),
        )

        return sc_loss, log_loss

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale STFT loss.

        Args:
            predicted: [B, 1, T] predicted audio
            target: [B, 1, T] target audio

        Returns:
            Scalar loss
        """
        # Align lengths
        min_len = min(predicted.shape[2], target.shape[2])
        predicted = predicted[:, :, :min_len]
        target = target[:, :, :min_len]

        total_loss = torch.tensor(0.0, device=predicted.device)
        for fft_size, hop_size, win_size in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes,
        ):
            if min_len < fft_size:
                continue
            sc, log_mag = self._stft_loss(predicted, target, fft_size, hop_size, win_size)
            total_loss = total_loss + sc + log_mag

        return total_loss


class SpeakerContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for speaker encoder training.

    Pulls same-speaker embeddings together, pushes different-speaker embeddings apart.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            embeddings_a: [B, D] embeddings from audio segment A
            embeddings_b: [B, D] embeddings from audio segment B
            Pairs (a_i, b_i) are from the same speaker.

        Returns:
            Scalar loss
        """
        # Normalize
        a = F.normalize(embeddings_a, dim=-1)
        b = F.normalize(embeddings_b, dim=-1)

        # Similarity matrix [B, B]
        sim = torch.matmul(a, b.T) / self.temperature

        # Labels: diagonal (same speaker pairs)
        labels = torch.arange(sim.shape[0], device=sim.device)

        # Cross-entropy both ways
        loss_ab = F.cross_entropy(sim, labels)
        loss_ba = F.cross_entropy(sim.T, labels)

        return (loss_ab + loss_ba) / 2


class MelSpectrogramLoss(nn.Module):
    """Mel spectrogram reconstruction loss.

    L1 loss on mel spectrograms. Used for detokenizer training and E2E fine-tuning.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
    ):
        super().__init__()
        self.mel_transform = None
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def _get_mel_transform(self, device: torch.device):
        if self.mel_transform is None or self.mel_transform.fb.device != device:
            import torchaudio
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            ).to(device)
        return self.mel_transform

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram L1 loss.

        Args:
            predicted: [B, 1, T] predicted audio
            target: [B, 1, T] target audio

        Returns:
            Scalar loss
        """
        min_len = min(predicted.shape[2], target.shape[2])
        predicted = predicted[:, :, :min_len]
        target = target[:, :, :min_len]

        mel_fn = self._get_mel_transform(predicted.device)

        pred_mel = torch.log(mel_fn(predicted.squeeze(1)) + 1e-5)
        target_mel = torch.log(mel_fn(target.squeeze(1)) + 1e-5)

        return F.l1_loss(pred_mel, target_mel)


class SpeakerConsistencyLoss(nn.Module):
    """Ensure generated audio maintains speaker identity.

    Computes cosine similarity between speaker embeddings extracted from
    generated audio and the target speaker embedding.
    """

    def __init__(self, target_similarity: float = 0.9):
        super().__init__()
        self.target_similarity = target_similarity

    def forward(
        self,
        generated_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute speaker consistency loss.

        Args:
            generated_embedding: [B, D] speaker embedding from generated audio
            target_embedding: [B, D] target speaker embedding

        Returns:
            Scalar loss (lower = more similar speakers)
        """
        sim = F.cosine_similarity(generated_embedding, target_embedding, dim=-1)
        # Loss is how far we are from target similarity
        return F.relu(self.target_similarity - sim).mean()
