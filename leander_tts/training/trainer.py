"""Training loop for Leander-TTS."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from leander_tts.config import DictConfig

logger = logging.getLogger(__name__)


class Trainer:
    """Generic training loop with logging, checkpointing, and evaluation."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[DictConfig] = None,
        device: str = "cuda",
        bf16: bool = True,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.bf16 = bf16

        self.global_step = 0
        self.best_val_loss = float("inf")

        # GradScaler for mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if device == "cuda" and not bf16 else None

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step. Override in subclasses for custom logic."""
        self.model.train()

        # Move batch to device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            outputs = self.model(**batch)
            loss = outputs["loss"]

        # Backward
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {"loss": loss.item()}

    def optimizer_step(self, max_grad_norm: float = 1.0) -> None:
        """Optimizer step with gradient clipping."""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

        if self.scheduler:
            self.scheduler.step()

        self.global_step += 1

    @torch.inference_mode()
    def evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        """Run evaluation on validation set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in val_loader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
                outputs = self.model(**batch)
                total_loss += outputs["loss"].item()
            n_batches += 1

        return {"val_loss": total_loss / max(n_batches, 1)}

    def save_checkpoint(self, path: str | Path, extra: dict | None = None) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        if extra:
            state.update(extra)

        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path} at step {self.global_step}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_step = state.get("global_step", 0)
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        if self.scheduler and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        logger.info(f"Loaded checkpoint from {path} at step {self.global_step}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        max_steps: int = 100000,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_every: int = 50,
        eval_every: int = 1000,
        save_every: int = 5000,
        save_dir: str = "checkpoints",
    ) -> None:
        """Main training loop."""
        logger.info(f"Starting training for {max_steps} steps")
        logger.info(f"Gradient accumulation: {gradient_accumulation_steps}")

        self.optimizer.zero_grad()
        accumulation_loss = 0.0

        while self.global_step < max_steps:
            for batch in train_loader:
                if self.global_step >= max_steps:
                    break

                metrics = self.train_step(batch)
                accumulation_loss += metrics["loss"]

                # Accumulate gradients
                if (self.global_step + 1) % gradient_accumulation_steps == 0:
                    self.optimizer_step(max_grad_norm=max_grad_norm)

                    avg_loss = accumulation_loss / gradient_accumulation_steps
                    accumulation_loss = 0.0

                    # Logging
                    if self.global_step % log_every == 0:
                        lr = self.optimizer.param_groups[0]["lr"]
                        logger.info(
                            f"Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                        )

                    # Evaluation
                    if val_loader and self.global_step % eval_every == 0:
                        val_metrics = self.evaluate(val_loader)
                        logger.info(
                            f"Step {self.global_step} | Val Loss: {val_metrics['val_loss']:.4f}"
                        )

                        if val_metrics["val_loss"] < self.best_val_loss:
                            self.best_val_loss = val_metrics["val_loss"]
                            self.save_checkpoint(f"{save_dir}/best.pt")

                    # Save checkpoint
                    if self.global_step % save_every == 0:
                        self.save_checkpoint(
                            f"{save_dir}/step_{self.global_step}.pt"
                        )
                else:
                    self.global_step += 1

        logger.info(f"Training complete at step {self.global_step}")
        self.save_checkpoint(f"{save_dir}/final.pt")
