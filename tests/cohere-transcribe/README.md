# Cohere Transcribe Parity Workflow

This directory holds the manual parity tooling for the first-pass Cohere runtime.

The native `transformers` Cohere model path is the canonical reference in this repo as of April 1, 2026. Older locally exported fixtures should be treated as stale unless they were regenerated with the current exporter.

What is committed:
- the HF fixture exporter
- the C++ parity checker
- docs only

What is not committed by default:
- gated model weights
- exported local fixture tensors
- copied audio clips

The local fixture directory is gitignored at `tests/cohere-transcribe/local-fixtures/`.

## Reference baseline

- Default fixture export mode is native HF `CohereAsrForConditionalGeneration`.
- `--reference-mode remote` exists only as a debug fallback.
- The exporter disables frontend dither by default for deterministic numeric parity. Pass `--keep-dither` if you explicitly want the native noisy frontend instead.
- The exporter writes provenance into `manifest.json`, including:
  - `reference_mode`
  - `transformers_version`
  - `torch_version`
  - `snapshot_id`
  - `matrix_storage`
  - `source_feature_extractor_dither`
  - `effective_feature_extractor_dither`

## Recommended flow

1. Install local Python deps:
```bash
python3 -m pip install torch transformers safetensors sentencepiece numpy soundfile librosa gguf
```

2. Convert the gated HF snapshot to an F32 GGUF for parity:
```bash
python3 models/convert-cohere-transcribe-to-gguf.py /path/to/cohere-transcribe-snapshot /path/to/cohere-transcribe-f32.gguf --dtype f32
```

3. Export reference fixtures for a few short 16 kHz mono clips:
```bash
python3 tests/cohere-transcribe/export-parity-fixtures.py \
  --snapshot /path/to/cohere-transcribe-snapshot \
  --out-dir tests/cohere-transcribe/local-fixtures \
  --language en \
  --reference-mode native \
  --clips ./samples/jfk.wav
```

4. Build the C++ checker:
```bash
cmake --build build --target cohere-parity
```

5. Compare stages:
```bash
./build/bin/cohere-parity \
  --model /path/to/cohere-transcribe-f32.gguf \
  --fixtures tests/cohere-transcribe/local-fixtures \
  --clip jfk \
  --stage all
```

`cohere-parity` uses `1e-4` / `1e-4` defaults for most stages, and relaxes only `mel` and `first_step_logits` to a `5e-4` absolute tolerance by default. That matches the current recovery status: encoder parity, prompt IDs, greedy IDs, and final text are exact, while the custom C++ frontend still shows a very small residual numeric drift versus the native `torch.stft` reference.

## Supported stages

- `mel`
- `subsampling_out`
- `block0_after_ff1`
- `block0_after_attn`
- `block0_after_conv`
- `block0_out`
- `encoder_out`
- `encoder_projected`
- `prompt_ids`
- `first_step_logits`
- `greedy_ids`
- `text`
- `all`

`encoder_block0` is still accepted by the C++ checker as an alias for `block0_out` so older notes do not immediately break.
