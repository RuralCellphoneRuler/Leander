#!/usr/bin/env python3
"""Interactive demo for Leander-TTS.

Usage:
    python scripts/demo.py \
        --config configs/model/leander_m.yaml \
        --checkpoint checkpoints/phase5/best.pt \
        --reference audio/reference.wav \
        --output output.wav
"""

from __future__ import annotations

import argparse
import time

import torch
import torchaudio

from leander_tts.inference.streaming import StreamingTTSPipeline
from leander_tts.model.leander import LeanderTTS


def main():
    parser = argparse.ArgumentParser(description="Leander-TTS Demo")
    parser.add_argument("--config", type=str, required=True, help="Model config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--reference", type=str, required=True, help="Reference audio for cloning")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-duration", type=float, default=30.0)
    args = parser.parse_args()

    print("Loading model...")
    model = LeanderTTS.from_config(args.config)
    model.load_pretrained(lm_path=args.checkpoint)
    model = model.to(args.device)
    model.eval()

    pipeline = StreamingTTSPipeline(
        model=model,
        device=args.device,
        temperature=args.temperature,
    )

    # Load reference audio
    print(f"Loading reference: {args.reference}")
    ref_audio, sr = torchaudio.load(args.reference)
    if ref_audio.shape[0] > 1:
        ref_audio = ref_audio.mean(dim=0, keepdim=True)
    if sr != 24000:
        ref_audio = torchaudio.transforms.Resample(sr, 24000)(ref_audio)
    ref_audio = ref_audio.unsqueeze(0)  # [1, 1, T]

    pipeline.set_speaker(ref_audio.to(args.device))
    print("Speaker embedding cached.")

    # Interactive mode or single generation
    if args.text:
        texts = [args.text]
    else:
        print("\nEnter text to synthesize (Ctrl+C to quit):")
        print("Supports emotion tags: <lacht>, <seufzt>, <flüstert>, <betont>...</betont>")
        print("-" * 60)
        texts = []
        try:
            while True:
                text = input("\n> ").strip()
                if text:
                    texts.append(text)
        except (KeyboardInterrupt, EOFError):
            pass

    for i, text in enumerate(texts):
        print(f"\nSynthesizing: '{text}'")
        start_time = time.time()

        audio = pipeline.synthesize(
            text=text,
            max_duration=args.max_duration,
        )

        elapsed = time.time() - start_time
        duration = audio.shape[2] / 24000
        rtf = elapsed / max(duration, 0.001)

        out_path = args.output if len(texts) == 1 else f"output_{i}.wav"
        torchaudio.save(out_path, audio.squeeze(0).cpu(), 24000)

        print(f"  Duration: {duration:.2f}s")
        print(f"  Generation time: {elapsed:.2f}s")
        print(f"  RTF: {rtf:.3f}")
        print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
