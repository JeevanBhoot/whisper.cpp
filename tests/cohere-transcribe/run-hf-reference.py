#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official HF Cohere Transcribe model on a local audio file.")
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Path to the local HF snapshot directory",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to the input audio file",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code to force during transcription (default: en)",
    )
    parser.add_argument(
        "--mode",
        choices=("native", "remote", "both"),
        default="native",
        help="Inference path to run. 'native' is recommended for transformers>=5.4.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of decode steps for generate()",
    )
    return parser.parse_args()


def run_native(snapshot: Path, audio_path: Path, language: str, max_new_tokens: int) -> str:
    processor = AutoProcessor.from_pretrained(snapshot)
    model = CohereAsrForConditionalGeneration.from_pretrained(snapshot).to("cpu")
    model.eval()

    audio = load_audio(str(audio_path), sampling_rate=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language=language)
    inputs.to(model.device, dtype=model.dtype)

    start = time.time()
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    elapsed = time.time() - start

    text = processor.decode(outputs, skip_special_tokens=True)
    value = text[0] if isinstance(text, list) else str(text)

    print("mode=native")
    print(f"elapsed_s={elapsed:.3f}")
    print(f"text={value}")
    return value


def run_remote(snapshot: Path, audio_path: Path, language: str) -> str:
    processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(snapshot, trust_remote_code=True, dtype=torch.float32).to("cpu")
    model.eval()

    start = time.time()
    texts = model.transcribe(processor=processor, audio_files=[str(audio_path)], language=language)
    elapsed = time.time() - start

    value = texts[0] if isinstance(texts, list) else str(texts)

    print("mode=remote")
    print(f"elapsed_s={elapsed:.3f}")
    print(f"text={value}")
    return value


def main() -> int:
    args = parse_args()

    if not args.snapshot.exists():
        raise SystemExit(f"snapshot does not exist: {args.snapshot}")
    if not args.audio.exists():
        raise SystemExit(f"audio file does not exist: {args.audio}")

    print(f"python={sys.version.split()[0]}")
    try:
        import transformers

        print(f"transformers={transformers.__version__}")
    except Exception:
        pass
    print(f"torch={torch.__version__}")

    if args.mode in ("native", "both"):
        run_native(args.snapshot, args.audio, args.language, args.max_new_tokens)

    if args.mode in ("remote", "both"):
        run_remote(args.snapshot, args.audio, args.language)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
