#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRISPEECH_TESTS_DIR = REPO_ROOT / "tests" / "librispeech"

jiwer = None
sf = None
torch = None
AutoProcessor = None
CohereAsrForConditionalGeneration = None
EnglishTextNormalizer = None


@dataclass
class Clip:
    code: str
    audio_path: Path
    reference: str
    duration_s: float


@dataclass
class ClipResult:
    code: str
    read_s: float | None
    transcribe_s: float | None
    warm_e2e_s: float | None
    reference: str
    hypothesis: str
    reference_normalized: str
    hypothesis_normalized: str
    exact_match_normalized: bool


@dataclass
class ModelSummary:
    name: str
    clip_count: int
    audio_total_s: float
    wer: float
    exact_match_count: int
    exact_match_rate: float
    load_s: float | None
    read_total_s: float
    transcribe_total_s: float
    warm_e2e_total_s: float
    dataset_total_s: float
    avg_transcribe_latency_s: float
    median_transcribe_latency_s: float
    avg_warm_e2e_latency_s: float
    median_warm_e2e_latency_s: float
    rtf_transcribe: float
    rtf_warm_e2e: float
    clips_per_s_transcribe: float
    clips_per_s_warm_e2e: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HF Cohere Transcribe vs local GGUF Cohere on a LibriSpeech-style tree.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Path to the local HF snapshot directory",
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        required=True,
        help="Path to the Cohere GGUF model (recommended: f16 runtime export)",
    )
    parser.add_argument(
        "--librispeech-root",
        type=Path,
        required=True,
        help="Path to the local LibriSpeech root containing */*/*/*.flac and *.trans.txt",
    )
    parser.add_argument(
        "--whisper-cli",
        type=Path,
        default=REPO_ROOT / "build" / "bin" / "whisper-cli",
        help="Path to whisper-cli (used only for optional cold/manual checks)",
    )
    parser.add_argument(
        "--cohere-bench",
        type=Path,
        default=REPO_ROOT / "build" / "bin" / "cohere-bench",
        help="Path to the local cohere-bench helper used for fair GGUF timings",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Forced language code for both models (default: en)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit evaluation to the first N clips after sorting (default: all)",
    )
    parser.add_argument(
        "--pattern",
        default="*/*/*/*.flac",
        help="Glob pattern under --librispeech-root used to discover audio files",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum decode steps for HF generate()",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, (os_cpu_count() // 2) if os_cpu_count() else 1),
        help="Default CPU thread count for both HF and GGUF unless overridden",
    )
    parser.add_argument(
        "--hf-threads",
        type=int,
        default=0,
        help="Override CPU thread count for HF only",
    )
    parser.add_argument(
        "--gguf-threads",
        type=int,
        default=0,
        help="Override CPU thread count for GGUF only",
    )
    parser.add_argument(
        "--show-errors",
        type=int,
        default=5,
        help="Show up to N normalized mismatches per model (default: 5)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write a JSON report",
    )
    return parser.parse_args()


def os_cpu_count() -> int | None:
    try:
        import os

        return os.cpu_count()
    except Exception:
        return None


def load_runtime_dependencies() -> None:
    global jiwer
    global sf
    global torch
    global AutoProcessor
    global CohereAsrForConditionalGeneration
    global EnglishTextNormalizer

    if jiwer is not None:
        return

    try:
        import jiwer as _jiwer
        import soundfile as _sf
        import torch as _torch
        from transformers import AutoProcessor as _AutoProcessor
        from transformers import CohereAsrForConditionalGeneration as _CohereAsrForConditionalGeneration
    except ImportError as exc:
        raise SystemExit(
            "missing dependency while importing comparison script requirements: "
            f"{exc}. Install torch, transformers, soundfile, jiwer, regex, and more-itertools."
        )

    if str(LIBRISPEECH_TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(LIBRISPEECH_TESTS_DIR))

    try:
        from normalizers import EnglishTextNormalizer as _EnglishTextNormalizer
    except ImportError as exc:
        raise SystemExit(f"failed to import LibriSpeech normalizer from {LIBRISPEECH_TESTS_DIR}: {exc}")

    jiwer = _jiwer
    sf = _sf
    torch = _torch
    AutoProcessor = _AutoProcessor
    CohereAsrForConditionalGeneration = _CohereAsrForConditionalGeneration
    EnglishTextNormalizer = _EnglishTextNormalizer


def load_reference_map(librispeech_root: Path) -> dict[str, str]:
    references: dict[str, str] = {}
    for path in sorted(librispeech_root.glob("*/*/*/*.trans.txt")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                code, text = stripped.split(" ", maxsplit=1)
                references[code] = text
    if not references:
        raise SystemExit(f"no LibriSpeech reference transcripts found under {librispeech_root}")
    return references


def gather_clips(librispeech_root: Path, pattern: str, limit: int) -> list[Clip]:
    references = load_reference_map(librispeech_root)
    clips: list[Clip] = []

    for audio_path in sorted(librispeech_root.glob(pattern)):
        code = audio_path.stem
        if code not in references:
            raise SystemExit(f"missing reference text for clip {code} ({audio_path})")
        info = sf.info(str(audio_path))
        clips.append(
            Clip(
                code=code,
                audio_path=audio_path,
                reference=references[code],
                duration_s=float(info.duration),
            )
        )

    if not clips:
        raise SystemExit(f"no audio files matching pattern '{pattern}' under {librispeech_root}")

    if limit > 0:
        clips = clips[:limit]

    return clips


def normalize_text(text: str, normalizer: Any) -> str:
    return str(normalizer(text))


def summarize_results(
    name: str,
    clips: list[Clip],
    clip_results: list[ClipResult],
    *,
    load_s: float | None,
    read_total_s: float,
    transcribe_total_s: float,
) -> ModelSummary:
    refs = [result.reference_normalized for result in clip_results]
    hyps = [result.hypothesis_normalized for result in clip_results]
    exact_matches = sum(1 for result in clip_results if result.exact_match_normalized)
    audio_total_s = float(sum(clip.duration_s for clip in clips))
    warm_e2e_total_s = float(read_total_s + transcribe_total_s)

    transcribe_latencies = [result.transcribe_s for result in clip_results if result.transcribe_s is not None]
    warm_e2e_latencies = [result.warm_e2e_s for result in clip_results if result.warm_e2e_s is not None]
    avg_transcribe_latency = (
        float(sum(transcribe_latencies) / len(transcribe_latencies))
        if transcribe_latencies else float(transcribe_total_s / max(len(clips), 1))
    )
    median_transcribe_latency = (
        float(statistics.median(transcribe_latencies))
        if transcribe_latencies else avg_transcribe_latency
    )
    avg_warm_e2e_latency = (
        float(sum(warm_e2e_latencies) / len(warm_e2e_latencies))
        if warm_e2e_latencies else float(warm_e2e_total_s / max(len(clips), 1))
    )
    median_warm_e2e_latency = (
        float(statistics.median(warm_e2e_latencies))
        if warm_e2e_latencies else avg_warm_e2e_latency
    )
    dataset_total_s = float((load_s or 0.0) + warm_e2e_total_s)

    return ModelSummary(
        name=name,
        clip_count=len(clips),
        audio_total_s=audio_total_s,
        wer=float(jiwer.wer(refs, hyps)),
        exact_match_count=exact_matches,
        exact_match_rate=float(exact_matches / max(len(clips), 1)),
        load_s=load_s,
        read_total_s=float(read_total_s),
        transcribe_total_s=float(transcribe_total_s),
        warm_e2e_total_s=warm_e2e_total_s,
        dataset_total_s=dataset_total_s,
        avg_transcribe_latency_s=avg_transcribe_latency,
        median_transcribe_latency_s=median_transcribe_latency,
        avg_warm_e2e_latency_s=avg_warm_e2e_latency,
        median_warm_e2e_latency_s=median_warm_e2e_latency,
        rtf_transcribe=float(transcribe_total_s / max(audio_total_s, 1e-12)),
        rtf_warm_e2e=float(warm_e2e_total_s / max(audio_total_s, 1e-12)),
        clips_per_s_transcribe=float(len(clips) / max(transcribe_total_s, 1e-12)),
        clips_per_s_warm_e2e=float(len(clips) / max(warm_e2e_total_s, 1e-12)),
    )


def read_mono_audio(audio_path: Path) -> tuple[Any, int]:
    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if int(sample_rate) != 16000:
        raise SystemExit(f"expected 16 kHz audio for {audio_path}, got {sample_rate} Hz")
    return audio, int(sample_rate)


def run_hf_native(clips: list[Clip], args: argparse.Namespace, normalizer: Any) -> tuple[ModelSummary, list[ClipResult]]:
    hf_threads = args.hf_threads or args.threads
    torch.set_num_threads(hf_threads)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.snapshot)
    model = CohereAsrForConditionalGeneration.from_pretrained(args.snapshot, dtype=torch.float32).to("cpu")
    model.eval()
    load_s = time.perf_counter() - load_start

    clip_results: list[ClipResult] = []
    read_total_s = 0.0
    transcribe_total_s = 0.0

    with torch.inference_mode():
        for index, clip in enumerate(clips, start=1):
            read_start = time.perf_counter()
            audio, sample_rate = read_mono_audio(clip.audio_path)
            read_elapsed = time.perf_counter() - read_start

            start = time.perf_counter()
            inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", language=args.language)
            inputs = inputs.to(model.device)
            output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
            transcribe_elapsed = time.perf_counter() - start

            hypothesis = str(decoded[0] if decoded else "").strip()
            reference_normalized = normalize_text(clip.reference, normalizer)
            hypothesis_normalized = normalize_text(hypothesis, normalizer)
            read_total_s += read_elapsed
            transcribe_total_s += transcribe_elapsed
            clip_results.append(
                ClipResult(
                    code=clip.code,
                    read_s=float(read_elapsed),
                    transcribe_s=float(transcribe_elapsed),
                    warm_e2e_s=float(read_elapsed + transcribe_elapsed),
                    reference=clip.reference,
                    hypothesis=hypothesis,
                    reference_normalized=reference_normalized,
                    hypothesis_normalized=hypothesis_normalized,
                    exact_match_normalized=reference_normalized == hypothesis_normalized,
                )
            )

            print(
                f"[hf] {index}/{len(clips)} {clip.code} "
                f"read={read_elapsed:.3f}s transcribe={transcribe_elapsed:.3f}s text={hypothesis}",
                file=sys.stderr,
            )

    summary = summarize_results(
        "hf-native",
        clips,
        clip_results,
        load_s=float(load_s),
        read_total_s=float(read_total_s),
        transcribe_total_s=float(transcribe_total_s),
    )
    return summary, clip_results


def run_gguf_warm(clips: list[Clip], args: argparse.Namespace, normalizer: Any) -> tuple[ModelSummary, list[ClipResult], dict[str, Any]]:
    gguf_threads = args.gguf_threads or args.threads
    cmd = [
        str(args.cohere_bench),
        "--model",
        str(args.gguf),
        "--language",
        args.language,
        "--threads",
        str(gguf_threads),
    ]
    for clip in clips:
        cmd.extend(["--file", str(clip.audio_path)])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise SystemExit(
            "GGUF bench invocation failed.\n"
            f"command: {' '.join(cmd)}\n"
            f"exit_code: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            "failed to parse JSON from cohere-bench.\n"
            f"error: {exc}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    by_code = {str(item["code"]): item for item in payload.get("clips", [])}
    clip_results: list[ClipResult] = []
    for clip in clips:
        item = by_code.get(clip.code)
        if item is None:
            raise SystemExit(f"cohere-bench output is missing clip {clip.code}")
        hypothesis = str(item["text"]).strip()
        reference_normalized = normalize_text(clip.reference, normalizer)
        hypothesis_normalized = normalize_text(hypothesis, normalizer)
        clip_results.append(
            ClipResult(
                code=clip.code,
                read_s=float(item["read_s"]),
                transcribe_s=float(item["transcribe_s"]),
                warm_e2e_s=float(item["warm_e2e_s"]),
                reference=clip.reference,
                hypothesis=hypothesis,
                reference_normalized=reference_normalized,
                hypothesis_normalized=hypothesis_normalized,
                exact_match_normalized=reference_normalized == hypothesis_normalized,
            )
        )

    summary = summarize_results(
        "gguf-f16-warm",
        clips,
        clip_results,
        load_s=float(payload.get("load_s", 0.0)),
        read_total_s=float(payload.get("read_total_s", 0.0)),
        transcribe_total_s=float(payload.get("transcribe_total_s", 0.0)),
    )
    return summary, clip_results, payload


def print_summary(summary: ModelSummary) -> None:
    print(f"{summary.name}:")
    if summary.load_s is not None:
        print(f"  load_s: {summary.load_s:.3f}")
    print(f"  read_total_s: {summary.read_total_s:.3f}")
    print(f"  transcribe_total_s: {summary.transcribe_total_s:.3f}")
    print(f"  warm_e2e_total_s: {summary.warm_e2e_total_s:.3f}")
    print(f"  dataset_total_s: {summary.dataset_total_s:.3f}")
    print(f"  avg_transcribe_latency_s: {summary.avg_transcribe_latency_s:.3f}")
    print(f"  median_transcribe_latency_s: {summary.median_transcribe_latency_s:.3f}")
    print(f"  avg_warm_e2e_latency_s: {summary.avg_warm_e2e_latency_s:.3f}")
    print(f"  median_warm_e2e_latency_s: {summary.median_warm_e2e_latency_s:.3f}")
    print(f"  rtf_transcribe: {summary.rtf_transcribe:.4f}")
    print(f"  rtf_warm_e2e: {summary.rtf_warm_e2e:.4f}")
    print(f"  clips_per_s_transcribe: {summary.clips_per_s_transcribe:.3f}")
    print(f"  clips_per_s_warm_e2e: {summary.clips_per_s_warm_e2e:.3f}")
    print(f"  wer: {summary.wer * 100:.2f}%")
    print(f"  exact_match_normalized: {summary.exact_match_count}/{summary.clip_count} ({summary.exact_match_rate * 100:.1f}%)")


def print_mismatches(name: str, clip_results: list[ClipResult], limit: int) -> None:
    if limit <= 0:
        return

    mismatches = [result for result in clip_results if not result.exact_match_normalized]
    if not mismatches:
        print(f"{name}: no normalized mismatches")
        return

    print(f"{name}: showing {min(limit, len(mismatches))} normalized mismatches")
    for result in mismatches[:limit]:
        print(f"  [{result.code}]")
        print(f"    ref: {result.reference}")
        print(f"    hyp: {result.hypothesis}")


def print_gguf_stage_breakdown(payload: dict[str, Any]) -> None:
    aggregate_profile = payload.get("aggregate_profile")
    if not isinstance(aggregate_profile, dict):
        return

    timings = aggregate_profile.get("timings_s")
    shares = aggregate_profile.get("timing_share_of_transcribe")
    if not isinstance(timings, dict) or not isinstance(shares, dict):
        return

    ranked: list[tuple[str, float, float]] = []
    for name, value in timings.items():
        try:
            seconds = float(value)
            share = float(shares.get(name, 0.0))
        except Exception:
            continue
        ranked.append((str(name), seconds, share))

    ranked.sort(key=lambda item: item[1], reverse=True)
    if not ranked:
        return

    print()
    print("gguf aggregate stage breakdown:")
    for name, seconds, share in ranked[:8]:
        print(f"  {name}: {seconds:.3f}s ({share * 100:.1f}% of transcribe)")


def maybe_write_json(
    path: Path | None,
    args: argparse.Namespace,
    clips: list[Clip],
    hf_summary: ModelSummary,
    hf_results: list[ClipResult],
    gguf_summary: ModelSummary,
    gguf_results: list[ClipResult],
    gguf_payload: dict[str, Any],
) -> None:
    if path is None:
        return

    report = {
        "snapshot": str(args.snapshot),
        "gguf": str(args.gguf),
        "whisper_cli": str(args.whisper_cli),
        "cohere_bench": str(args.cohere_bench),
        "librispeech_root": str(args.librispeech_root),
        "language": args.language,
        "benchmark_contract": {
            "device": "cpu",
            "batch_size": 1,
            "model_load_scope": "loaded_once_per_run",
            "primary_metric": "transcribe_total_s",
            "secondary_metric": "warm_e2e_total_s",
            "hf_batching_in_primary_comparison": False,
        },
        "measurement_scope": {
            "load_s": "model load once per dataset run",
            "read_total_s": "audio file decoding/reading only",
            "transcribe_total_s": "loaded-model transcription time excluding file I/O",
            "warm_e2e_total_s": "read_total_s + transcribe_total_s",
            "dataset_total_s": "load_s + warm_e2e_total_s",
        },
        "clip_count": len(clips),
        "audio_total_s": float(sum(clip.duration_s for clip in clips)),
        "hf_native": {
            "summary": asdict(hf_summary),
            "clips": [asdict(result) for result in hf_results],
        },
        "gguf_f16": {
            "summary": asdict(gguf_summary),
            "clips": [asdict(result) for result in gguf_results],
            "aggregate_profile": gguf_payload.get("aggregate_profile"),
            "benchmark_contract": gguf_payload.get("benchmark_contract"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote report to {path}")


def main() -> int:
    args = parse_args()

    if not args.snapshot.exists():
        raise SystemExit(f"snapshot does not exist: {args.snapshot}")
    if not args.gguf.exists():
        raise SystemExit(f"GGUF model does not exist: {args.gguf}")
    if not args.cohere_bench.exists():
        raise SystemExit(f"cohere-bench does not exist: {args.cohere_bench}")
    if not args.librispeech_root.exists():
        raise SystemExit(f"LibriSpeech root does not exist: {args.librispeech_root}")

    load_runtime_dependencies()

    clips = gather_clips(args.librispeech_root, args.pattern, args.limit)
    normalizer = EnglishTextNormalizer()
    audio_total_s = sum(clip.duration_s for clip in clips)

    print(f"selected {len(clips)} clips from {args.librispeech_root}")
    print(f"audio_total_s={audio_total_s:.3f}")
    print(f"hf_threads={args.hf_threads or args.threads}")
    print(f"gguf_threads={args.gguf_threads or args.threads}")

    hf_summary, hf_results = run_hf_native(clips, args, normalizer)
    gguf_summary, gguf_results, gguf_payload = run_gguf_warm(clips, args, normalizer)

    print()
    print_summary(hf_summary)
    print()
    print_summary(gguf_summary)
    print_gguf_stage_breakdown(gguf_payload)

    if gguf_summary.transcribe_total_s > 0.0 and gguf_summary.warm_e2e_total_s > 0.0:
        print()
        print("ratios:")
        print(f"  gguf_vs_hf_transcribe: {hf_summary.transcribe_total_s / gguf_summary.transcribe_total_s:.3f}x")
        print(f"  hf_vs_gguf_transcribe: {gguf_summary.transcribe_total_s / hf_summary.transcribe_total_s:.3f}x")
        print(f"  gguf_vs_hf_warm_e2e: {hf_summary.warm_e2e_total_s / gguf_summary.warm_e2e_total_s:.3f}x")
        print(f"  hf_vs_gguf_warm_e2e: {gguf_summary.warm_e2e_total_s / hf_summary.warm_e2e_total_s:.3f}x")
        if hf_summary.load_s is not None and gguf_summary.load_s is not None and gguf_summary.load_s > 0.0:
            print(f"  gguf_vs_hf_load: {hf_summary.load_s / gguf_summary.load_s:.3f}x")
            print(f"  hf_vs_gguf_load: {gguf_summary.load_s / hf_summary.load_s:.3f}x")
        if gguf_summary.dataset_total_s > 0.0:
            print(f"  gguf_vs_hf_dataset_total: {hf_summary.dataset_total_s / gguf_summary.dataset_total_s:.3f}x")
            print(f"  hf_vs_gguf_dataset_total: {gguf_summary.dataset_total_s / hf_summary.dataset_total_s:.3f}x")

    print()
    print_mismatches("hf-native", hf_results, args.show_errors)
    print()
    print_mismatches("gguf-f16-warm", gguf_results, args.show_errors)

    maybe_write_json(args.json_out, args, clips, hf_summary, hf_results, gguf_summary, gguf_results, gguf_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
