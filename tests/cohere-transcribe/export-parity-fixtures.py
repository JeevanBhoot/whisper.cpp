#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer
    from transformers.modeling_outputs import BaseModelOutput
except ImportError as exc:
    raise SystemExit(
        "missing dependency while importing parity exporter requirements: "
        f"{exc}. Install torch, transformers, soundfile, sentencepiece, and numpy."
    )

try:
    from transformers import CohereAsrForConditionalGeneration
except ImportError:
    CohereAsrForConditionalGeneration = None


REQUIRED_PROMPT_TOKENS = (
    "<|startofcontext|>",
    "<|startoftranscript|>",
    "<|emo:undefined|>",
    "{lang}",
    "{lang}",
    "<|pnc|>",
    "<|noitn|>",
    "<|notimestamp|>",
    "<|nodiarize|>",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export local Cohere parity fixtures from the official HF model.")
    parser.add_argument("--snapshot", required=True, type=Path, help="Path to the local HF snapshot")
    parser.add_argument(
        "--clips",
        required=True,
        nargs="+",
        type=Path,
        help="One or more short audio clips to export",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Fixture output directory (recommended: tests/cohere-transcribe/local-fixtures)",
    )
    parser.add_argument("--language", default="en", help="Language code for the fixed decoder prompt")
    parser.add_argument(
        "--reference-mode",
        choices=("native", "remote"),
        default="native",
        help="HF reference path to use. 'native' is the default for current transformers versions.",
    )
    parser.add_argument(
        "--keep-dither",
        action="store_true",
        help="Keep the feature extractor's configured dither instead of forcing 0.0 for deterministic stage parity.",
    )
    return parser.parse_args()


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    elif torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(dtype=torch.float32)
        arr = tensor.numpy()
    else:
        arr = np.asarray(value)
    return np.ascontiguousarray(arr)


def load_model_bundle(snapshot_dir: Path, reference_mode: str) -> tuple[Any, Any, Any, str]:
    if reference_mode == "native":
        if CohereAsrForConditionalGeneration is None:
            raise SystemExit(
                "transformers does not expose CohereAsrForConditionalGeneration. "
                "Install a newer transformers version or rerun with --reference-mode remote."
            )

        processor = AutoProcessor.from_pretrained(snapshot_dir)
        model = CohereAsrForConditionalGeneration.from_pretrained(snapshot_dir, dtype=torch.float32)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, use_fast=False)
        model.eval()
        return model, processor, tokenizer, "native"

    processor = AutoProcessor.from_pretrained(snapshot_dir, trust_remote_code=True)

    model = None
    errors: list[str] = []
    for loader in (AutoModelForSpeechSeq2Seq, AutoModelForSeq2SeqLM, AutoModel):
        try:
            model = loader.from_pretrained(
                snapshot_dir,
                trust_remote_code=True,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            break
        except Exception as exc:  # pragma: no cover - best effort loader fallback
            errors.append(f"{loader.__name__}: {exc}")
    if model is None:
        raise SystemExit("failed to load HF model with trust_remote_code=True:\n" + "\n".join(errors))

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, trust_remote_code=True, use_fast=False)

    model.eval()
    return model, processor, tokenizer, "remote"


def configure_feature_extractor_for_parity(processor: Any, keep_dither: bool) -> tuple[float | None, float | None]:
    feature_extractor = getattr(processor, "feature_extractor", None)
    if feature_extractor is None or not hasattr(feature_extractor, "dither"):
        return None, None

    source_dither = float(feature_extractor.dither)
    if not keep_dither:
        feature_extractor.dither = 0.0
    return source_dither, float(feature_extractor.dither)


def infer_time_steps(input_features: torch.Tensor) -> int:
    if input_features.ndim != 3:
        raise SystemExit(f"expected rank-3 input_features, got shape={tuple(input_features.shape)}")

    if input_features.shape[-1] == 128:
        return int(input_features.shape[1])
    if input_features.shape[1] == 128:
        return int(input_features.shape[2])
    raise SystemExit(f"could not infer time dimension from input_features shape={tuple(input_features.shape)}")


def build_sequence_mask(length: torch.Tensor, max_length: int, device: torch.device | None = None) -> torch.Tensor:
    if length.ndim == 0:
        length = length.unsqueeze(0)
    length = length.to(dtype=torch.long)
    positions = torch.arange(max_length, device=device or length.device).unsqueeze(0)
    return positions < length.unsqueeze(1)


def build_attention_mask_from_length(input_features: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    time_steps = infer_time_steps(input_features)
    return build_sequence_mask(length, time_steps, input_features.device)


def prepare_features(
    processor: Any,
    tokenizer: Any,
    audio: np.ndarray,
    sample_rate: int,
    language: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if callable(processor):
        encoded = processor(audio, sampling_rate=sample_rate, return_tensors="pt", language=language)
    elif hasattr(processor, "feature_extractor"):
        encoded = processor.feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
    else:
        raise SystemExit("processor does not expose a callable audio preprocessing path")

    if "input_features" not in encoded:
        raise SystemExit("processor output is missing 'input_features'")

    input_features = encoded["input_features"].to(dtype=torch.float32)
    if "attention_mask" in encoded:
        attention_mask = encoded["attention_mask"].to(dtype=torch.bool)
    elif "length" in encoded:
        attention_mask = build_attention_mask_from_length(input_features, encoded["length"])
    else:
        attention_mask = torch.ones((input_features.shape[0], infer_time_steps(input_features)), dtype=torch.bool)

    if "decoder_input_ids" in encoded:
        prompt_ids = [int(token_id) for token_id in encoded["decoder_input_ids"][0].tolist()]
    else:
        prompt_ids = build_prompt_ids(tokenizer, language)

    return input_features, attention_mask, prompt_ids


def is_native_reference_model(model: Any) -> bool:
    return (
        hasattr(model, "model")
        and hasattr(model.model, "encoder")
        and hasattr(model.model.encoder, "subsampling")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "proj")
    )


def mel_stage_tensor(input_features: torch.Tensor) -> torch.Tensor:
    stage = input_features.squeeze(0)
    if stage.ndim != 2:
        raise SystemExit(f"expected rank-2 mel tensor after squeeze, got shape={tuple(stage.shape)}")
    if stage.shape[0] == 128:
        return stage
    if stage.shape[1] == 128:
        return stage.transpose(0, 1)
    raise SystemExit(f"could not infer mel axis from input_features shape={tuple(stage.shape)}")


def run_native_encoder_trace(
    model: Any,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    encoder = model.model.encoder

    subsampling_out = encoder.subsampling(input_features, attention_mask)
    hidden_states = subsampling_out * encoder.input_scale
    position_embeddings = encoder.encode_positions(hidden_states)

    output_mask = encoder._get_output_attention_mask(attention_mask, target_length=hidden_states.shape[1])
    att_mask = output_mask.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
    att_mask = att_mask & att_mask.transpose(1, 2)
    att_mask = att_mask.unsqueeze(1)

    block0 = encoder.layers[0]
    ff1_out = block0.feed_forward1(block0.norm_feed_forward1(hidden_states))
    block0_after_ff1 = hidden_states + 0.5 * ff1_out

    attn_out, _ = block0.self_attn(
        hidden_states=block0.norm_self_att(block0_after_ff1),
        attention_mask=att_mask,
        position_embeddings=position_embeddings,
    )
    block0_after_attn = block0_after_ff1 + attn_out

    conv_out = block0.conv(block0.norm_conv(block0_after_attn), attention_mask=att_mask)
    block0_after_conv = block0_after_attn + conv_out

    ff2_out = block0.feed_forward2(block0.norm_feed_forward2(block0_after_conv))
    block0_out = block0.norm_out(block0_after_conv + 0.5 * ff2_out)

    encoder_out = block0_out
    for layer in list(encoder.layers)[1:]:
        encoder_out = layer(
            encoder_out,
            attention_mask=att_mask,
            position_embeddings=position_embeddings,
        )

    encoder_projected = model.model.decoder.proj(encoder_out)
    stages = {
        "subsampling_out": subsampling_out,
        "block0_after_ff1": block0_after_ff1,
        "block0_after_attn": block0_after_attn,
        "block0_after_conv": block0_after_conv,
        "block0_out": block0_out,
        "encoder_out": encoder_out,
        "encoder_projected": encoder_projected,
    }
    return stages, output_mask.to(dtype=torch.long).sum(-1)


def build_prompt_ids(tokenizer: Any, language: str) -> list[int]:
    ids: list[int] = []
    space_id = tokenizer.convert_tokens_to_ids("▁")
    if space_id is not None:
        space_id = int(space_id)
        if space_id >= 0:
            ids.append(space_id)

    tokens = []
    for token in REQUIRED_PROMPT_TOKENS:
        if token == "{lang}":
            tokens.append(f"<|{language}|>")
        else:
            tokens.append(token)
    ids.extend(int(tokenizer.convert_tokens_to_ids(token)) for token in tokens)
    if any(idx < 0 for idx in ids):
        raise SystemExit("failed to resolve one or more prompt token ids")
    return ids


def detokenize(tokenizer: Any, ids: list[int], special_ids: set[int]) -> str:
    pieces = tokenizer.convert_ids_to_tokens(ids)
    output = bytearray()

    for token_id, piece in zip(ids, pieces):
        if token_id in special_ids:
            continue
        if piece in ("<unk>", "<s>", "</s>"):
            continue
        if piece.startswith("▁"):
            if output:
                output.extend(b" ")
            output.extend(piece[1:].encode("utf-8"))
            continue
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            output.append(int(piece[3:5], 16))
            continue
        output.extend(piece.encode("utf-8"))

    return output.decode("utf-8", errors="replace")


def write_f32(path: Path, array: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(array.astype(np.float32, copy=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)
    meta = {
        "path": path.name,
        "dtype": "f32",
        "shape": [int(dim) for dim in array.shape],
    }
    if array.ndim >= 2:
        meta["storage"] = "row_major"
    return meta


def snapshot_id(snapshot_dir: Path) -> str:
    config_path = snapshot_dir / "config.json"
    digest = hashlib.sha1(config_path.read_bytes()).hexdigest()[:12]
    return f"{snapshot_dir.name}:{digest}"


def to_stage_matrix(value: Any) -> np.ndarray:
    arr = as_numpy(value)
    if arr.ndim != 2:
        raise SystemExit(f"expected a rank-2 stage matrix, got shape={arr.shape}")
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def stage_meta(path: Path, value: Any) -> dict[str, Any]:
    return write_f32(path, to_stage_matrix(value))


def export_clip(
    model: Any,
    processor: Any,
    tokenizer: Any,
    clip_path: Path,
    clip_dir: Path,
    language: str,
) -> dict[str, Any]:
    audio, sample_rate = sf.read(str(clip_path))
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    input_features, attention_mask, prompt_ids = prepare_features(processor, tokenizer, audio, int(sample_rate), language)

    with torch.no_grad():
        if is_native_reference_model(model):
            stage_tensors, encoder_len = run_native_encoder_trace(model, input_features, attention_mask)
            subsampling_out = stage_tensors["subsampling_out"]
            block0_after_ff1 = stage_tensors["block0_after_ff1"]
            block0_after_attn = stage_tensors["block0_after_attn"]
            block0_after_conv = stage_tensors["block0_after_conv"]
            block0_out = stage_tensors["block0_out"]
            encoder_out = stage_tensors["encoder_out"]
            encoder_projected = stage_tensors["encoder_projected"]
            encoder_outputs = SimpleNamespace(
                last_hidden_state=encoder_out,
                attention_mask=build_sequence_mask(
                    encoder_len,
                    encoder_out.shape[1],
                    encoder_out.device,
                ).to(dtype=torch.long),
                hidden_states=None,
                attentions=None,
            )
        else:
            length = attention_mask.sum(-1).to(dtype=torch.long)
            subsampling_out, subsampling_len = model.encoder.pre_encode(input_features, length)
            x, pos_emb = model.encoder.pos_enc(subsampling_out)
            pad_mask, att_mask = model.encoder._create_masks(
                padding_length=subsampling_len.to(torch.int64),
                max_audio_length=x.size(1),
                device=x.device,
            )

            block0 = model.encoder.layers[0]
            ff1_in = block0.norm_feed_forward1(x)
            ff1_out = block0.feed_forward1(ff1_in)
            block0_after_ff1 = x + 0.5 * ff1_out

            att_in = block0.norm_self_att(block0_after_ff1)
            att_out = block0.self_attn(att_in, pos_emb, att_mask)
            block0_after_attn = block0_after_ff1 + att_out

            conv_in = block0.norm_conv(block0_after_attn)
            conv_out = block0.conv(conv_in, pad_mask=pad_mask)
            block0_after_conv = block0_after_attn + conv_out

            ff2_in = block0.norm_feed_forward2(block0_after_conv)
            ff2_out = block0.feed_forward2(ff2_in)
            block0_out = block0.norm_out(block0_after_conv + 0.5 * ff2_out)

            encoder_out = block0_out
            for layer in list(model.encoder.layers)[1:]:
                encoder_out = layer(encoder_out, pos_emb, mask=att_mask, pad_mask=pad_mask)
            encoder_len = subsampling_len

            if getattr(model, "encoder_decoder_proj", None) is not None:
                encoder_projected = model.encoder_decoder_proj(encoder_out)
            else:
                encoder_projected = encoder_out
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_out)

        decoder_ids = list(prompt_ids)
        generated_ids: list[int] = []
        eos_id = int(tokenizer.convert_tokens_to_ids("<|endoftext|>"))
        first_step_logits = None

        if is_native_reference_model(model):
            decoder_max_len = int(model.model.decoder.config.max_position_embeddings)
        else:
            decoder_max_len = int(model.config.transf_decoder["config_dict"]["max_sequence_length"])
        max_new_tokens = min(256, decoder_max_len - len(prompt_ids))
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([decoder_ids], dtype=torch.long)
            if is_native_reference_model(model):
                outputs = model(
                    decoder_input_ids=input_ids,
                    encoder_outputs=encoder_outputs,
                    return_dict=True,
                )
            else:
                positions = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
                outputs = model(
                    input_ids=input_ids,
                    positions=positions,
                    encoder_outputs=encoder_outputs,
                    return_dict=True,
                )
            logits = outputs.logits[0, -1].detach().cpu().to(torch.float32)
            if first_step_logits is None:
                first_step_logits = logits

            next_id = int(torch.argmax(logits).item())
            if next_id == eos_id:
                break
            generated_ids.append(next_id)
            decoder_ids.append(next_id)

    if first_step_logits is None:
        raise SystemExit("failed to collect first-step logits")

    special_tokens = list(getattr(tokenizer, "all_special_tokens", []))
    special_tokens.extend(
        [
            "<|endoftext|>",
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            "<|pnc|>",
            "<|nopnc|>",
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
            f"<|{language}|>",
        ]
    )
    special_ids = {
        int(tokenizer.convert_tokens_to_ids(token))
        for token in special_tokens
        if int(tokenizer.convert_tokens_to_ids(token)) >= 0
    }
    text = detokenize(tokenizer, generated_ids, special_ids)

    stages = {
        "mel": stage_meta(clip_dir / "mel.f32", mel_stage_tensor(input_features)),
        "subsampling_out": stage_meta(clip_dir / "subsampling_out.f32", subsampling_out.squeeze(0).T),
        "block0_after_ff1": stage_meta(clip_dir / "block0_after_ff1.f32", block0_after_ff1.squeeze(0).T),
        "block0_after_attn": stage_meta(clip_dir / "block0_after_attn.f32", block0_after_attn.squeeze(0).T),
        "block0_after_conv": stage_meta(clip_dir / "block0_after_conv.f32", block0_after_conv.squeeze(0).T),
        "block0_out": stage_meta(clip_dir / "block0_out.f32", block0_out.squeeze(0).T),
        "encoder_out": stage_meta(clip_dir / "encoder_out.f32", encoder_out.squeeze(0).T),
        "encoder_projected": stage_meta(clip_dir / "encoder_projected.f32", encoder_projected.squeeze(0).T),
        "first_step_logits": write_f32(clip_dir / "first_step_logits.f32", as_numpy(first_step_logits)),
    }

    shutil.copy2(clip_path, clip_dir / "audio.wav")

    return {
        "id": clip_dir.name,
        "audio": f"{clip_dir.name}/audio.wav",
        "language": language,
        "mel_len": int(attention_mask.shape[-1]),
        "subsampling_len": int(encoder_len[0].item()),
        "block0_after_ff1_len": int(encoder_len[0].item()),
        "block0_after_attn_len": int(encoder_len[0].item()),
        "block0_after_conv_len": int(encoder_len[0].item()),
        "block0_out_len": int(encoder_len[0].item()),
        "encoder_out_len": int(encoder_len[0].item()),
        "encoder_projected_len": int(encoder_len[0].item()),
        "prompt_ids": prompt_ids,
        "greedy_ids": generated_ids,
        "text": text,
        "stages": {
            name: {
                **stage_meta_,
                "path": f"{clip_dir.name}/{stage_meta_['path']}",
            }
            for name, stage_meta_ in stages.items()
        },
    }


def main() -> None:
    args = parse_args()
    snapshot_dir = args.snapshot.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor, tokenizer, reference_mode = load_model_bundle(snapshot_dir, args.reference_mode)
    source_dither, effective_dither = configure_feature_extractor_for_parity(processor, args.keep_dither)

    clips = []
    for clip_path in args.clips:
        clip_path = clip_path.resolve()
        clip_id = clip_path.stem
        clip_dir = out_dir / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        clips.append(export_clip(model, processor, tokenizer, clip_path, clip_dir, args.language))

    manifest = {
        "version": 2,
        "language": args.language,
        "matrix_storage": "row_major",
        "reference_mode": reference_mode,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "snapshot_id": snapshot_id(snapshot_dir),
        "source_feature_extractor_dither": source_dither,
        "effective_feature_extractor_dither": effective_dither,
        "clips": clips,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"wrote fixtures to {out_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
