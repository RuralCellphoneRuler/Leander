#!/usr/bin/env python3
"""Phase 3: LM fine-tuning on text → SNAC token prediction.

This is the main training script — fine-tunes Qwen2.5 to predict SNAC tokens
from text input, conditioned on speaker embeddings.

Usage:
    python scripts/train_lm.py --config configs/training/phase3_lm.yaml
    python scripts/train_lm.py --config configs/training/phase3_lm.yaml --resume checkpoints/phase3/step_5000.pt
"""

from __future__ import annotations

import argparse
import logging

import torch
from leander_tts.config import load_config
from torch.utils.data import DataLoader

from leander_tts.data.dataset import TTSDataset, collate_tts
from leander_tts.data.tokenizer import LeanderTokenizer
from leander_tts.model.lm import LeanderLM
from leander_tts.model.speaker_encoder import SpeakerEncoder
from leander_tts.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_optimizer(model: LeanderLM, config) -> torch.optim.Optimizer:
    opt_cfg = config.training.optimizer
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.stage_b.lr,
        betas=tuple(opt_cfg.betas),
        eps=opt_cfg.eps,
        weight_decay=config.training.stage_b.get("weight_decay", 0.01),
    )


def build_scheduler(optimizer, config, stage):
    total_steps = stage.steps
    warmup_steps = stage.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        min_ratio = config.training.scheduler.get("min_lr_ratio", 0.1)
        import math
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class LMTrainer(Trainer):
    """Specialized trainer for the LM with speaker embedding support."""

    def __init__(
        self,
        model: LeanderLM,
        speaker_encoder: SpeakerEncoder,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ):
        super().__init__(model=model, optimizer=optimizer, **kwargs)
        self.speaker_encoder = speaker_encoder.to(self.device)
        self.speaker_encoder.eval()

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Get speaker embedding if available
        speaker_embedding = batch.get("speaker_embedding")
        if speaker_embedding is not None:
            speaker_embedding = speaker_embedding.to(self.device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                speaker_embedding=speaker_embedding,
            )
            loss = outputs["loss"]

        loss.backward()
        return {"loss": loss.item()}


def main():
    parser = argparse.ArgumentParser(description="Phase 3: LM fine-tuning")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = load_config(config.training.model_config)
    mc = model_config.model

    logger.info("Building tokenizer...")
    tokenizer = LeanderTokenizer(backbone=mc.lm.backbone, codebook_size=mc.codec.codebook_size)

    logger.info("Building LM...")
    lm = LeanderLM(
        backbone=mc.lm.backbone,
        n_new_tokens=tokenizer.n_new_tokens,
        d_model=mc.lm.d_model,
        n_layers=mc.lm.n_layers,
        inject_every_n_layers=mc.speaker_injection.inject_every_n_layers,
    )
    lm.load_backbone(pretrained=True)

    logger.info("Building speaker encoder...")
    speaker_encoder = SpeakerEncoder(
        codebook_size=mc.codec.codebook_size,
        n_levels=mc.codec.n_levels,
        d_model=mc.speaker_encoder.d_model,
        output_dim=mc.speaker_encoder.output_dim,
        n_layers=mc.speaker_encoder.n_layers,
        n_heads=mc.speaker_encoder.n_heads,
    )

    tc = config.training

    # Stage 3a: Train new components only
    if tc.stage_a.enabled:
        logger.info("=== Stage 3a: Training new components (backbone frozen) ===")
        lm.freeze_backbone()
        logger.info(f"Trainable params: {lm.get_trainable_params():,}")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, lm.parameters()),
            lr=tc.stage_a.lr,
            betas=tuple(tc.optimizer.betas),
        )
        scheduler = build_scheduler(optimizer, config, tc.stage_a)

        train_dataset = TTSDataset(
            manifest_path=tc.data.train_manifest,
            tokenizer=tokenizer,
            max_seq_len=tc.data.max_seq_len,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=tc.batch_size,
            shuffle=True,
            num_workers=tc.data.num_workers,
            collate_fn=collate_tts,
            pin_memory=tc.data.pin_memory,
        )

        trainer = LMTrainer(
            model=lm,
            speaker_encoder=speaker_encoder,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            bf16=tc.bf16,
        )

        trainer.train(
            train_loader=train_loader,
            max_steps=tc.stage_a.steps,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            max_grad_norm=tc.max_grad_norm,
            save_dir=tc.checkpoint.save_dir + "/stage_a",
        )

    # Stage 3b: Full fine-tuning
    if tc.stage_b.enabled:
        logger.info("=== Stage 3b: Full fine-tuning ===")
        lm.unfreeze_backbone()
        logger.info(f"Trainable params: {lm.get_trainable_params():,}")

        optimizer = build_optimizer(lm, config)
        scheduler = build_scheduler(optimizer, config, tc.stage_b)

        train_dataset = TTSDataset(
            manifest_path=tc.data.train_manifest,
            tokenizer=tokenizer,
            max_seq_len=tc.data.max_seq_len,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=tc.batch_size,
            shuffle=True,
            num_workers=tc.data.num_workers,
            collate_fn=collate_tts,
            pin_memory=tc.data.pin_memory,
        )

        trainer = LMTrainer(
            model=lm,
            speaker_encoder=speaker_encoder,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            bf16=tc.bf16,
        )

        if args.resume:
            trainer.load_checkpoint(args.resume)

        trainer.train(
            train_loader=train_loader,
            max_steps=tc.stage_b.steps,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            max_grad_norm=tc.max_grad_norm,
            save_dir=tc.checkpoint.save_dir + "/stage_b",
        )

    # Stage 3c: Expressive fine-tuning
    if tc.stage_c.enabled:
        logger.info("=== Stage 3c: Expressive fine-tuning ===")

        optimizer = torch.optim.AdamW(
            lm.parameters(), lr=tc.stage_c.lr, betas=tuple(tc.optimizer.betas),
        )
        scheduler = build_scheduler(optimizer, config, tc.stage_c)

        trainer = LMTrainer(
            model=lm,
            speaker_encoder=speaker_encoder,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            bf16=tc.bf16,
        )

        # Would use expressive-only data here
        trainer.train(
            train_loader=train_loader,
            max_steps=tc.stage_c.steps,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            max_grad_norm=tc.max_grad_norm,
            save_dir=tc.checkpoint.save_dir + "/stage_c",
        )

    logger.info("Phase 3 training complete!")


if __name__ == "__main__":
    main()
