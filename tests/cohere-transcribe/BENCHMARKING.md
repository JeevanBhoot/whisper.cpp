# Cohere Benchmarking

Run from the repo root.

## 1. Install prerequisites

- `git`
- `cmake`
- C/C++ build tools
- `python3`

On macOS:

```bash
xcode-select --install
```

## 2. Clone and enter the repo

```bash
git clone <your-whisper-cpp-fork>
cd whisper.cpp
```

## 3. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 4. Install Python dependencies

```bash
pip install torch transformers soundfile safetensors sentencepiece gguf huggingface_hub librosa
pip install -r tests/librispeech/requirements.txt
```

## 5. Build the tools

```bash
cmake -B build
cmake --build build --target whisper-cli cohere-bench -j8
```

## 6. Download the gated HF snapshot

Login:

```bash
hf auth login
```

Download:

```bash
hf download CohereLabs/cohere-transcribe-03-2026 \
  --local-dir ./artifacts/hf/cohere-transcribe-snapshot
```

## 7. Convert to GGUF

F16 runtime export:

```bash
python models/convert-cohere-transcribe-to-gguf.py \
  ./artifacts/hf/cohere-transcribe-snapshot \
  ./artifacts/gguf/cohere-transcribe-f16.gguf \
  --dtype f16
```

Optional F32 debug export:

```bash
python models/convert-cohere-transcribe-to-gguf.py \
  ./artifacts/hf/cohere-transcribe-snapshot \
  ./artifacts/gguf/cohere-transcribe-f32.gguf \
  --dtype f32
```

## 8. Download LibriSpeech test-clean

```bash
make -C tests/librispeech get-audio
```

## 9. Run the benchmark

Benchmark contract:

- CPU only
- batch size 1
- model loaded once per dataset run
- primary metric: `transcribe_total_s`
- secondary metric: `warm_e2e_total_s`
- no HF batching in the headline comparison

macOS note:

- For fair CPU benchmarking on macOS, set `VECLIB_MAXIMUM_THREADS` yourself before running both the HF and GGUF paths so Accelerate thread limits are explicit and reproducible.
- A reasonable starting point for `--threads 8` is `VECLIB_MAXIMUM_THREADS=4`, but treat that as a benchmark setting to report, not a hidden runtime default.

Quick single-file sanity check:

```bash
export VECLIB_MAXIMUM_THREADS=4

./build/bin/cohere-bench \
  --model ./artifacts/gguf/cohere-transcribe-f16.gguf \
  --file ./samples/jfk.wav \
  --language en \
  --threads 8
```

10-clip smoke subset:

```bash
export VECLIB_MAXIMUM_THREADS=4

python tests/cohere-transcribe/compare-hf-vs-gguf.py \
  --snapshot ./artifacts/hf/cohere-transcribe-snapshot \
  --gguf ./artifacts/gguf/cohere-transcribe-f16.gguf \
  --librispeech-root ./tests/librispeech/LibriSpeech \
  --threads 8 \
  --limit 10 \
  --json-out ./cohere-vs-gguf-smoke.json
```

Full `test-clean`:

```bash
export VECLIB_MAXIMUM_THREADS=4

python tests/cohere-transcribe/compare-hf-vs-gguf.py \
  --snapshot ./artifacts/hf/cohere-transcribe-snapshot \
  --gguf ./artifacts/gguf/cohere-transcribe-f16.gguf \
  --librispeech-root ./tests/librispeech/LibriSpeech \
  --threads 8 \
  --json-out ./cohere-vs-gguf.json
```

100 clips:

```bash
export VECLIB_MAXIMUM_THREADS=4

python tests/cohere-transcribe/compare-hf-vs-gguf.py \
  --snapshot ./artifacts/hf/cohere-transcribe-snapshot \
  --gguf ./artifacts/gguf/cohere-transcribe-f16.gguf \
  --librispeech-root ./tests/librispeech/LibriSpeech \
  --threads 8 \
  --limit 100 \
  --json-out ./cohere-vs-gguf_100.json
```

## 10. What to look at

- `wer`
- `transcribe_total_s`
- `warm_e2e_total_s`
- `gguf_vs_hf_transcribe`
- `hf_vs_gguf_transcribe`
- `aggregate_profile.timings_s`
- `aggregate_profile.timing_share_of_transcribe`
- `aggregate_profile.counters`

For the cleanest accuracy check, rerun with the F32 GGUF too.

## 11. Profiling follow-up

After the internal timings are in place, use a system profiler on the same warm benchmark contract:

- macOS: Instruments Time Profiler or `sample`
- Linux / Raspberry Pi: `perf record` + `perf report`

Use `cohere-bench` JSON as the GGUF source of truth for stage and overhead accounting.
