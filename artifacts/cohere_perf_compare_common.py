#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    mode: str
    run_index: int
    elapsed_s: float
    returncode: int
    stdout: str
    stderr: str
    command: list[str]


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_cli(repo_root: Path) -> Path:
    return repo_root / "build" / "bin" / "whisper-cli"


def _total_audio_seconds(paths: list[Path]) -> float | None:
    total = 0.0
    for path in paths:
        if path.suffix.lower() != ".wav":
            return None
        try:
            with wave.open(str(path), "rb") as wav_file:
                frame_rate = wav_file.getframerate()
                if frame_rate <= 0:
                    return None
                total += wav_file.getnframes() / frame_rate
        except wave.Error:
            return None
    return total


def _build_command(
    cli_path: Path,
    model_path: Path,
    audio_paths: list[Path],
    language: str,
    threads: int,
    extra_flags: list[str],
) -> list[str]:
    command = [
        str(cli_path),
        "-m",
        str(model_path),
        "-l",
        language,
        "-t",
        str(threads),
        "-np",
    ]
    command.extend(extra_flags)
    for audio_path in audio_paths:
        command.extend(["-f", str(audio_path)])
    return command


def _run_once(mode: str, run_index: int, command: list[str]) -> RunResult:
    t0 = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed_s = time.perf_counter() - t0
    return RunResult(
        mode=mode,
        run_index=run_index,
        elapsed_s=elapsed_s,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        command=command,
    )


def _run_mode(
    mode: str,
    command: list[str],
    warmup_runs: int,
    repeats: int,
    verbose: bool,
) -> tuple[list[RunResult], list[RunResult]]:
    warmups: list[RunResult] = []
    measured: list[RunResult] = []

    for warmup_index in range(warmup_runs):
        result = _run_once(mode, -(warmup_index + 1), command)
        if result.returncode != 0:
            raise RuntimeError(
                f"{mode} warmup failed with exit code {result.returncode}\n"
                f"command: {' '.join(result.command)}\n"
                f"stderr:\n{result.stderr}"
            )
        warmups.append(result)
        if verbose:
            print(f"[warmup] {mode}: {result.elapsed_s:.3f}s", file=sys.stderr)

    for run_index in range(1, repeats + 1):
        result = _run_once(mode, run_index, command)
        if result.returncode != 0:
            raise RuntimeError(
                f"{mode} run {run_index} failed with exit code {result.returncode}\n"
                f"command: {' '.join(result.command)}\n"
                f"stderr:\n{result.stderr}"
            )
        measured.append(result)
        if verbose:
            print(f"[run {run_index}] {mode}: {result.elapsed_s:.3f}s", file=sys.stderr)

    return warmups, measured


def _summarize(mode: str, results: list[RunResult], total_audio_s: float | None) -> dict[str, Any]:
    elapsed_values = [result.elapsed_s for result in results]
    avg_s = statistics.mean(elapsed_values)
    min_s = min(elapsed_values)
    max_s = max(elapsed_values)
    stddev_s = statistics.pstdev(elapsed_values) if len(elapsed_values) > 1 else 0.0
    rtf = avg_s / total_audio_s if total_audio_s and total_audio_s > 0 else None
    xrt = total_audio_s / avg_s if total_audio_s and avg_s > 0 else None

    return {
        "mode": mode,
        "runs": len(results),
        "avg_s": avg_s,
        "min_s": min_s,
        "max_s": max_s,
        "stddev_s": stddev_s,
        "rtf": rtf,
        "xrt": xrt,
        "stdout_lines": [line for line in results[0].stdout.splitlines() if line.strip()],
    }


def _print_summary(
    cpu_summary: dict[str, Any],
    gpu_summary: dict[str, Any],
    total_audio_s: float | None,
    gpu_label: str,
) -> None:
    def fmt(value: float | None, digits: int = 3) -> str:
        if value is None or not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"

    cpu_avg = cpu_summary["avg_s"]
    gpu_avg = gpu_summary["avg_s"]
    speedup = cpu_avg / gpu_avg if gpu_avg > 0 else None
    transcript_match = cpu_summary["stdout_lines"] == gpu_summary["stdout_lines"]

    print("")
    print(f"Audio seconds: {fmt(total_audio_s)}")
    print(f"CPU avg:       {fmt(cpu_summary['avg_s'])} s  | xRT {fmt(cpu_summary['xrt'])} | RTF {fmt(cpu_summary['rtf'])}")
    print(f"{gpu_label} avg:     {fmt(gpu_summary['avg_s'])} s  | xRT {fmt(gpu_summary['xrt'])} | RTF {fmt(gpu_summary['rtf'])}")
    print(f"Speedup:       {fmt(speedup)}x ({gpu_label} vs CPU)")
    print(f"Transcript match: {'yes' if transcript_match else 'no'}")

    if not transcript_match:
        print("")
        print("CPU transcript lines:")
        for line in cpu_summary["stdout_lines"]:
            print(f"  {line}")
        print("")
        print(f"{gpu_label} transcript lines:")
        for line in gpu_summary["stdout_lines"]:
            print(f"  {line}")


def main(gpu_label: str, default_output_name: str) -> int:
    repo_root = _default_repo_root()

    parser = argparse.ArgumentParser(
        description=f"Quick local Cohere perf comparison: CPU-only vs {gpu_label}.",
    )
    parser.add_argument(
        "--bin",
        dest="cli_path",
        type=Path,
        default=_default_cli(repo_root),
        help="Path to whisper-cli (default: ./build/bin/whisper-cli).",
    )
    parser.add_argument(
        "--model",
        dest="model_path",
        type=Path,
        required=True,
        help="Path to a Cohere GGUF model.",
    )
    parser.add_argument(
        "--file",
        dest="audio_paths",
        type=Path,
        action="append",
        default=[],
        help="Audio file to transcribe. Pass multiple times. Defaults to ./samples/jfk.wav if omitted.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code passed to whisper-cli.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Thread count passed to whisper-cli for both runs.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured runs per mode.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup runs per mode before measuring.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help=f"{gpu_label} device index passed via --device.",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable flash attention for the GPU run.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=repo_root / "artifacts" / default_output_name,
        help="Where to write the raw JSON results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-run timings while executing.",
    )

    args = parser.parse_args()

    cli_path = args.cli_path.resolve()
    model_path = args.model_path.resolve()
    audio_paths = [path.resolve() for path in (args.audio_paths or [repo_root / "samples" / "jfk.wav"])]

    if not cli_path.is_file():
        raise SystemExit(f"whisper-cli not found: {cli_path}")
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")
    for audio_path in audio_paths:
        if not audio_path.is_file():
            raise SystemExit(f"audio file not found: {audio_path}")

    cpu_command = _build_command(
        cli_path=cli_path,
        model_path=model_path,
        audio_paths=audio_paths,
        language=args.language,
        threads=args.threads,
        extra_flags=["-ng"],
    )

    gpu_flags = ["-dev", str(args.device)]
    gpu_flags.append("-nfa" if args.no_flash_attn else "-fa")
    gpu_command = _build_command(
        cli_path=cli_path,
        model_path=model_path,
        audio_paths=audio_paths,
        language=args.language,
        threads=args.threads,
        extra_flags=gpu_flags,
    )

    print("CPU command:")
    print("  " + " ".join(cpu_command))
    print(f"{gpu_label} command:")
    print("  " + " ".join(gpu_command))

    total_audio_s = _total_audio_seconds(audio_paths)

    cpu_warmups, cpu_results = _run_mode(
        mode="cpu",
        command=cpu_command,
        warmup_runs=args.warmup_runs,
        repeats=args.repeats,
        verbose=args.verbose,
    )
    gpu_warmups, gpu_results = _run_mode(
        mode=gpu_label,
        command=gpu_command,
        warmup_runs=args.warmup_runs,
        repeats=args.repeats,
        verbose=args.verbose,
    )

    cpu_summary = _summarize("cpu", cpu_results, total_audio_s)
    gpu_summary = _summarize(gpu_label, gpu_results, total_audio_s)
    _print_summary(cpu_summary, gpu_summary, total_audio_s, gpu_label)

    output_payload = {
        "gpu_label": gpu_label,
        "cli_path": str(cli_path),
        "model_path": str(model_path),
        "audio_paths": [str(path) for path in audio_paths],
        "language": args.language,
        "threads": args.threads,
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "device": args.device,
        "flash_attn": not args.no_flash_attn,
        "total_audio_s": total_audio_s,
        "cpu": {
            "summary": cpu_summary,
            "warmups": [result.__dict__ for result in cpu_warmups],
            "runs": [result.__dict__ for result in cpu_results],
        },
        gpu_label: {
            "summary": gpu_summary,
            "warmups": [result.__dict__ for result in gpu_warmups],
            "runs": [result.__dict__ for result in gpu_results],
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    print("")
    print(f"Wrote raw results to {args.output_json}")

    return 0

