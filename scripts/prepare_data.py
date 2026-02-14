#!/usr/bin/env python3
"""Data preparation script: preprocess audio and generate SNAC tokens.

Usage:
    python scripts/prepare_data.py \
        --input-dir data/raw/mls_german \
        --output-dir data/processed \
        --manifest data/processed/train_manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torchaudio

from leander_tts.data.preprocessing import preprocess_audio
from leander_tts.model.codec import SNACCodec

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def find_audio_files(input_dir: Path) -> list[Path]:
    """Recursively find all audio files in a directory."""
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".opus"}
    files = []
    for ext in extensions:
        files.extend(input_dir.rglob(f"*{ext}"))
    return sorted(files)


def find_transcript(audio_path: Path) -> str | None:
    """Try to find transcript for an audio file.

    Supports common formats:
    - .txt file with same name
    - transcripts.txt in same directory (MLS format)
    """
    # Same name .txt
    txt_path = audio_path.with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text().strip()

    # MLS format: transcripts.txt in same directory
    transcript_file = audio_path.parent / "transcripts.txt"
    if transcript_file.exists():
        stem = audio_path.stem
        with open(transcript_file) as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2 and parts[0] == stem:
                    return parts[1]

    return None


def extract_speaker_id(audio_path: Path) -> str:
    """Extract speaker ID from path (assumes speaker/chapter/file structure)."""
    # MLS structure: speaker_id/book_id/speaker_id_book_id_segment.flac
    parts = audio_path.relative_to(audio_path.parent.parent.parent).parts
    if len(parts) >= 1:
        return parts[0]
    return "unknown"


def process_file(
    audio_path: Path,
    output_dir: Path,
    codec: SNACCodec,
    device: str = "cpu",
) -> dict | None:
    """Process a single audio file: preprocess, encode to SNAC tokens, save."""
    # Preprocess audio
    audio = preprocess_audio(audio_path)
    if audio is None:
        return None

    # Find transcript
    transcript = find_transcript(audio_path)
    if transcript is None:
        return None

    # Encode with SNAC
    audio_input = audio.unsqueeze(0).to(device)  # [1, 1, T]
    codes = codec.encode(audio_input)
    frames = codec.flatten_codes(codes)[0]  # First (only) batch item

    # Save SNAC tokens
    snac_filename = audio_path.stem + ".snac.pt"
    snac_path = output_dir / "snac" / snac_filename
    snac_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"frames": frames}, snac_path)

    # Build manifest entry
    duration = audio.shape[1] / 24000
    speaker_id = extract_speaker_id(audio_path)

    return {
        "audio_path": str(audio_path),
        "snac_path": str(snac_path),
        "text": transcript,
        "speaker_id": speaker_id,
        "duration": round(duration, 2),
        "n_frames": len(frames),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--codec", type=str, default="hubertsiuzdak/snac_24khz")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SNAC codec
    logger.info("Loading SNAC codec...")
    codec = SNACCodec(model_name=args.codec, device=args.device)
    codec.load()

    # Find audio files
    audio_files = find_audio_files(input_dir)
    logger.info(f"Found {len(audio_files)} audio files")

    # Process
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    n_processed = 0
    n_skipped = 0

    with open(manifest_path, "w") as manifest_f:
        for i, audio_path in enumerate(audio_files):
            if i % 1000 == 0:
                logger.info(f"Processing {i}/{len(audio_files)}...")

            try:
                entry = process_file(audio_path, output_dir, codec, args.device)
                if entry:
                    manifest_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    n_processed += 1
                else:
                    n_skipped += 1
            except Exception as e:
                logger.warning(f"Error processing {audio_path}: {e}")
                n_skipped += 1

    logger.info(f"Done! Processed: {n_processed}, Skipped: {n_skipped}")
    logger.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
