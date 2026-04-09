#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import librosa
import gguf
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, CohereAsrForConditionalGeneration


GGUF_ARCH = "cohere-transcribe"
GGUF_PREFIX = "cohere_transcribe."
REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "configuration_cohere_asr.py",
    "modeling_cohere_asr.py",
    "preprocessor_config.json",
    "processing_cohere_asr.py",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
)
REQUIRED_SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|startofcontext|>",
    "<|startoftranscript|>",
    "<|emo:undefined|>",
    "<|pnc|>",
    "<|nopnc|>",
    "<|noitn|>",
    "<|notimestamp|>",
    "<|nodiarize|>",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a local Cohere Transcribe HF snapshot to a single GGUF file.",
    )
    parser.add_argument("snapshot", type=Path, help="Path to the local HF snapshot directory")
    parser.add_argument("outfile", type=Path, help="Output GGUF path")
    parser.add_argument(
        "--dtype",
        choices=("f16", "f32"),
        default="f16",
        help="Tensor storage type for large weights (default: f16; use f32 for parity work)",
    )
    return parser.parse_args()


def load_preprocessor_json(snapshot_dir: Path) -> dict[str, Any]:
    with (snapshot_dir / "preprocessor_config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_snapshot(snapshot_dir: Path) -> None:
    missing = [name for name in REQUIRED_SNAPSHOT_FILES if not (snapshot_dir / name).exists()]
    if missing:
        raise SystemExit(
            "snapshot is missing required files: " + ", ".join(missing)
        )


def as_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    elif torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(dtype=torch.float32)
        arr = tensor.numpy()
    else:
        arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


def maybe_get_attr(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def pick_frontend_value(
    feature_extractor: Any,
    preprocessor_json: dict[str, Any],
    attr_names: Iterable[str],
    json_names: Iterable[str],
    default: Any,
) -> Any:
    value = maybe_get_attr(feature_extractor, *attr_names)
    if value is not None:
        return value
    for name in json_names:
        if name in preprocessor_json:
            return preprocessor_json[name]
    return default


def find_frontend_buffer(feature_extractor: Any, candidate_paths: Iterable[str]) -> np.ndarray | None:
    for path in candidate_paths:
        current = feature_extractor
        ok = True
        for part in path.split("."):
            if not hasattr(current, part):
                ok = False
                break
            current = getattr(current, part)
        if ok and current is not None:
            try:
                arr = as_numpy(current, np.float32)
            except (TypeError, ValueError):
                continue
            if arr.size > 0:
                return arr
    return None


def load_reference_feature_extractor(snapshot_dir: Path) -> Any | None:
    try:
        processor = AutoProcessor.from_pretrained(snapshot_dir)
    except Exception:
        return None
    return getattr(processor, "feature_extractor", processor)


def is_native_model(model: Any) -> bool:
    return (
        hasattr(model, "model")
        and hasattr(model.model, "encoder")
        and hasattr(model.model.encoder, "subsampling")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "proj")
        and hasattr(model, "proj_out")
    )


def load_frontend_buffer_from_safetensors(snapshot_dir: Path, key_candidates: Iterable[str]) -> np.ndarray | None:
    safetensors_path = snapshot_dir / "model.safetensors"
    with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in key_candidates:
            if key in keys:
                return as_numpy(handle.get_tensor(key), np.float32)
    return None


def create_librosa_mel(frontend: dict[str, Any]) -> np.ndarray:
    mel = librosa.filters.mel(
        sr=int(frontend["sample_rate"]),
        n_fft=int(frontend["n_fft"]),
        n_mels=int(frontend["n_mels"]),
        fmin=float(frontend["fmin"]),
        fmax=float(frontend["fmax"]),
        htk=False,
        norm="slaney",
    )
    return as_numpy(mel, np.float32)


def extract_frontend(
    snapshot_dir: Path,
    feature_extractor: Any,
    preprocessor_json: dict[str, Any],
) -> dict[str, Any]:
    sample_rate = int(pick_frontend_value(feature_extractor, preprocessor_json, ("sample_rate", "sampling_rate"), ("sample_rate",), 16000))
    n_mels = int(pick_frontend_value(feature_extractor, preprocessor_json, ("n_mels", "num_mel_bins", "feature_size"), ("n_mels", "num_mel_bins", "feature_size"), 128))
    n_fft = int(pick_frontend_value(feature_extractor, preprocessor_json, ("n_fft", "fft_length"), ("n_fft", "fft_length"), 512))
    win_length = int(pick_frontend_value(feature_extractor, preprocessor_json, ("win_length", "window_size"), ("win_length", "window_size"), 400))
    hop_length = int(pick_frontend_value(feature_extractor, preprocessor_json, ("hop_length", "window_stride"), ("hop_length", "window_stride"), 160))
    fmin = float(pick_frontend_value(feature_extractor, preprocessor_json, ("fmin", "f_min"), ("fmin", "f_min"), 0.0))
    fmax = float(pick_frontend_value(feature_extractor, preprocessor_json, ("fmax", "f_max"), ("fmax", "f_max"), sample_rate / 2.0))
    preemph = float(pick_frontend_value(feature_extractor, preprocessor_json, ("preemph", "preemphasis", "preemphasis_coeff"), ("preemph", "preemphasis", "preemphasis_coeff"), 0.97))
    dither = float(pick_frontend_value(feature_extractor, preprocessor_json, ("dither",), ("dither",), 0.0))
    log_zero_guard = float(pick_frontend_value(feature_extractor, preprocessor_json, ("log_zero_guard", "log_zero_guard_value"), ("log_zero_guard", "log_zero_guard_value"), 2.0 ** -24))
    normalize_per_feature = bool(pick_frontend_value(feature_extractor, preprocessor_json, ("normalize_per_feature",), ("normalize_per_feature",), True))

    mel = find_frontend_buffer(
        feature_extractor,
        (
            "filterbank.fb",
            "filterbank.filterbank",
            "filterbank.filters",
            "fb",
            "mel_filters",
        ),
    )
    if mel is None:
        mel = load_frontend_buffer_from_safetensors(
            snapshot_dir,
            (
                "preprocessor.featurizer.fb",
                "model.preprocessor.featurizer.fb",
            ),
        )
    if mel is None:
        mel = create_librosa_mel(
            {
                "sample_rate": sample_rate,
                "n_fft": n_fft,
                "n_mels": n_mels,
                "fmin": fmin,
                "fmax": fmax,
            }
        )

    mel = as_numpy(mel, np.float32)
    if mel.ndim == 3 and mel.shape[0] == 1:
        mel = mel[0]
    if mel.ndim == 3 and mel.shape[-1] == 1:
        mel = mel[..., 0]
    expected_freq_bins = n_fft // 2 + 1
    if mel.shape == (expected_freq_bins, n_mels):
        mel = mel.T
    if mel.shape != (n_mels, expected_freq_bins):
        raise SystemExit(f"unexpected mel filterbank shape: {tuple(mel.shape)}")

    window = find_frontend_buffer(
        feature_extractor,
        (
            "window",
            "filterbank.window",
            "featurizer.window",
        ),
    )
    if window is None:
        window_name = maybe_get_attr(feature_extractor, "window")
        if isinstance(window_name, str) and window_name.lower() == "hann":
            window = as_numpy(torch.hann_window(win_length, periodic=False), np.float32)
    if window is None:
        window = load_frontend_buffer_from_safetensors(
            snapshot_dir,
            (
                "preprocessor.featurizer.window",
                "model.preprocessor.featurizer.window",
            ),
        )
    if window is None:
        window = as_numpy(torch.hann_window(win_length, periodic=False), np.float32)
    window = as_numpy(window, np.float32).reshape(-1)
    if window.shape != (win_length,):
        raise SystemExit(f"unexpected window shape: {tuple(window.shape)}")

    return {
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "win_length": win_length,
        "hop_length": hop_length,
        "fmin": fmin,
        "fmax": fmax,
        "preemph": preemph,
        "dither": dither,
        "log_zero_guard": log_zero_guard,
        "normalize_per_feature": normalize_per_feature,
        "window": window,
        "mel_filters": mel,
    }


def load_model_and_tokenizer(snapshot_dir: Path) -> tuple[Any, Any, Any, str]:
    if CohereAsrForConditionalGeneration is not None:
        try:
            processor = AutoProcessor.from_pretrained(snapshot_dir)
            model = CohereAsrForConditionalGeneration.from_pretrained(snapshot_dir, dtype=torch.float32)
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, use_fast=False)
            model.eval()
            return model, processor, tokenizer, "native"
        except Exception:
            pass

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


def weight_dtype(name: str, export_dtype: str) -> np.dtype:
    if export_dtype == "f32":
        return np.float32
    if name.endswith(".bias") or ".norm_" in name or ".layer_norm." in name or name.endswith(".weight") and "norm" in name:
        return np.float32
    if "pos_bias_" in name:
        return np.float32
    return np.float16


def add_tensor(writer: Any, name: str, array: np.ndarray, export_dtype: str) -> None:
    dtype = weight_dtype(name, export_dtype)
    writer.add_tensor(name, as_numpy(array, dtype))


def linear_weight(module: Any) -> np.ndarray:
    return as_numpy(module.weight, np.float32)


def linear_bias(module: Any) -> np.ndarray:
    return as_numpy(module.bias, np.float32).reshape(-1)


def conv1d_weight(module: Any) -> np.ndarray:
    return as_numpy(module.weight, np.float32)


def conv2d_weight(module: Any) -> np.ndarray:
    return as_numpy(module.weight, np.float32)


def embedding_weight(module: Any) -> np.ndarray:
    return as_numpy(module.weight, np.float32)


def fuse_depthwise_conv_and_bn(conv_module: Any, bn_module: Any) -> tuple[np.ndarray, np.ndarray]:
    conv_w = as_numpy(conv_module.weight, np.float32)
    conv_b = np.zeros(conv_w.shape[0], dtype=np.float32)
    if conv_module.bias is not None:
        conv_b = as_numpy(conv_module.bias, np.float32).reshape(-1)

    gamma = as_numpy(bn_module.weight, np.float32).reshape(-1)
    beta = as_numpy(bn_module.bias, np.float32).reshape(-1)
    running_mean = as_numpy(bn_module.running_mean, np.float32).reshape(-1)
    running_var = as_numpy(bn_module.running_var, np.float32).reshape(-1)
    scale = gamma / np.sqrt(running_var + float(bn_module.eps))

    fused_w = conv_w * scale[:, None, None]
    fused_b = beta + (conv_b - running_mean) * scale

    return fused_w, fused_b


def extract_tokenizer_metadata(tokenizer: Any, config: Any) -> dict[str, Any]:
    vocab_size = len(tokenizer)
    pieces: list[str] = []
    for idx in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens(idx)
        if token is None and hasattr(tokenizer, "_convert_id_to_token"):
            token = tokenizer._convert_id_to_token(idx)
        if token is None:
            raise SystemExit(f"failed to recover tokenizer piece for id {idx}")
        pieces.append(str(token))

    special_map: dict[str, int] = {}
    all_special_tokens = list(getattr(tokenizer, "all_special_tokens", []))
    for token in all_special_tokens + list(REQUIRED_SPECIAL_TOKENS):
        token = str(token)
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        if token_id < 0 or token_id >= vocab_size:
            raise SystemExit(f"missing required special token '{token}' in tokenizer")
        special_map[token] = token_id

    supported_languages = list(getattr(config, "supported_languages", []))
    if not supported_languages:
        raise SystemExit("config.supported_languages is empty; refusing to export incomplete tokenizer metadata")

    language_token_ids: dict[str, int] = {}
    for language in supported_languages:
        token = f"<|{language}|>"
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        if token_id < 0 or token_id >= vocab_size:
            raise SystemExit(f"missing language token '{token}' in tokenizer")
        language_token_ids[language] = token_id

    return {
        "pieces": pieces,
        "supported_languages": supported_languages,
        "special_token_names": sorted(special_map),
        "special_token_ids": [special_map[name] for name in sorted(special_map)],
        "language_codes": supported_languages,
        "language_token_ids": [language_token_ids[language] for language in supported_languages],
    }


def add_metadata(writer: Any, key: str, value: Any) -> None:
    if isinstance(value, bool):
        writer.add_bool(key, value)
    elif isinstance(value, int):
        if hasattr(writer, "add_int32"):
            writer.add_int32(key, value)
        else:
            writer.add_uint32(key, value)
    elif isinstance(value, float):
        writer.add_float32(key, value)
    elif isinstance(value, str):
        writer.add_string(key, value)
    else:
        writer.add_array(key, value)


def export_model(snapshot_dir: Path, outfile: Path, export_dtype: str) -> None:
    model, processor, tokenizer, model_mode = load_model_and_tokenizer(snapshot_dir)
    native_mode = model_mode == "native" and is_native_model(model)
    if not native_mode:
        for attr in ("encoder", "transf_decoder", "log_softmax"):
            if not hasattr(model, attr):
                raise SystemExit(f"loaded model is missing expected Cohere module '{attr}'")
    preprocessor_json = load_preprocessor_json(snapshot_dir)
    feature_extractor = load_reference_feature_extractor(snapshot_dir)
    if feature_extractor is None:
        feature_extractor = getattr(processor, "feature_extractor", processor)
    frontend = extract_frontend(snapshot_dir, feature_extractor, preprocessor_json)
    tokenizer_meta = extract_tokenizer_metadata(tokenizer, model.config)

    if native_mode:
        encoder_cfg = model.config.encoder_config.to_dict()
        encoder_cfg = {
            "d_model": int(encoder_cfg["hidden_size"]),
            "feat_in": int(encoder_cfg.get("num_mel_bins", frontend["n_mels"])),
            "feat_out": int(encoder_cfg["hidden_size"]),
            "n_layers": int(encoder_cfg["num_hidden_layers"]),
            "n_heads": int(encoder_cfg["num_attention_heads"]),
            "ff_expansion_factor": int(encoder_cfg["intermediate_size"] // encoder_cfg["hidden_size"]),
            "conv_kernel_size": int(encoder_cfg["conv_kernel_size"]),
            "subsampling_factor": int(encoder_cfg["subsampling_factor"]),
            "subsampling_conv_channels": int(encoder_cfg["subsampling_conv_channels"]),
            "pos_emb_max_len": int(encoder_cfg["max_position_embeddings"]),
        }
        dec_cfg = model.model.decoder.config.to_dict()
        decoder_cfg = {
            "hidden_size": int(dec_cfg["hidden_size"]),
            "inner_size": int(dec_cfg["intermediate_size"]),
            "num_attention_heads": int(dec_cfg["num_attention_heads"]),
            "num_layers": int(dec_cfg["num_hidden_layers"]),
            "max_sequence_length": int(dec_cfg["max_position_embeddings"]),
            "hidden_act": str(dec_cfg.get("hidden_act", "relu")).lower().replace("swish", "silu"),
        }
        head_cfg = {
            "hidden_size": int(model.config.hidden_size),
            "num_classes": int(model.config.vocab_size),
            "log_softmax": False,
        }
    else:
        encoder_cfg = dict(model.config.encoder)
        encoder_cfg["feat_out"] = int(encoder_cfg.get("feat_out", 0) or encoder_cfg["d_model"])
        decoder_cfg = dict(model.config.transf_decoder["config_dict"])
        head_cfg = dict(model.config.head)
    max_audio_clip_s = float(getattr(model.config, "max_audio_clip_s", 0.0))

    writer = gguf.GGUFWriter(str(outfile), GGUF_ARCH)
    if hasattr(writer, "add_name"):
        writer.add_name("Cohere Transcribe")
    if hasattr(writer, "add_file_type"):
        qtype = gguf.GGMLQuantizationType.F16 if export_dtype == "f16" else gguf.GGMLQuantizationType.F32
        writer.add_file_type(qtype)

    add_metadata(writer, GGUF_PREFIX + "max_audio_clip_s", max_audio_clip_s)

    for key, value in frontend.items():
        if isinstance(value, np.ndarray):
            add_metadata(writer, GGUF_PREFIX + "frontend." + key, value.reshape(-1).tolist())
        else:
            add_metadata(writer, GGUF_PREFIX + "frontend." + key, value)

    add_metadata(writer, GGUF_PREFIX + "encoder.d_model", int(encoder_cfg["d_model"]))
    add_metadata(writer, GGUF_PREFIX + "encoder.feat_in", int(encoder_cfg["feat_in"]))
    add_metadata(writer, GGUF_PREFIX + "encoder.feat_out", int(encoder_cfg["feat_out"]))
    add_metadata(writer, GGUF_PREFIX + "encoder.n_layers", int(encoder_cfg["n_layers"]))
    add_metadata(writer, GGUF_PREFIX + "encoder.n_heads", int(encoder_cfg["n_heads"]))
    add_metadata(writer, GGUF_PREFIX + "encoder.ff_expansion_factor", int(encoder_cfg["ff_expansion_factor"]))
    add_metadata(writer, GGUF_PREFIX + "encoder.conv_kernel_size", int(encoder_cfg["conv_kernel_size"]))
    add_metadata(writer, GGUF_PREFIX + "encoder.subsampling_factor", int(encoder_cfg["subsampling_factor"]))
    add_metadata(writer, GGUF_PREFIX + "encoder.subsampling_conv_channels", int(encoder_cfg["subsampling_conv_channels"]))
    add_metadata(writer, GGUF_PREFIX + "encoder.pos_emb_max_len", int(encoder_cfg["pos_emb_max_len"]))

    add_metadata(writer, GGUF_PREFIX + "decoder.hidden_size", int(decoder_cfg["hidden_size"]))
    add_metadata(writer, GGUF_PREFIX + "decoder.inner_size", int(decoder_cfg["inner_size"]))
    add_metadata(writer, GGUF_PREFIX + "decoder.num_attention_heads", int(decoder_cfg["num_attention_heads"]))
    add_metadata(writer, GGUF_PREFIX + "decoder.num_layers", int(decoder_cfg["num_layers"]))
    add_metadata(writer, GGUF_PREFIX + "decoder.max_sequence_length", int(decoder_cfg["max_sequence_length"]))
    add_metadata(writer, GGUF_PREFIX + "decoder.hidden_act", str(decoder_cfg.get("hidden_act", "relu")).lower().replace("swish", "silu"))

    add_metadata(writer, GGUF_PREFIX + "head.hidden_size", int(head_cfg["hidden_size"]))
    add_metadata(writer, GGUF_PREFIX + "head.num_classes", int(head_cfg["num_classes"]))
    add_metadata(writer, GGUF_PREFIX + "head.log_softmax", bool(head_cfg.get("log_softmax", False)))

    if native_mode:
        has_encoder_decoder_proj = getattr(model.model.decoder, "proj", None) is not None
    else:
        has_encoder_decoder_proj = model.encoder_decoder_proj is not None
    add_metadata(writer, GGUF_PREFIX + "encoder_decoder_proj", has_encoder_decoder_proj)
    add_metadata(writer, GGUF_PREFIX + "tokenizer.tokens", tokenizer_meta["pieces"])
    add_metadata(writer, GGUF_PREFIX + "tokenizer.supported_languages", tokenizer_meta["supported_languages"])
    add_metadata(writer, GGUF_PREFIX + "tokenizer.special_token_names", tokenizer_meta["special_token_names"])
    add_metadata(writer, GGUF_PREFIX + "tokenizer.special_token_ids", tokenizer_meta["special_token_ids"])
    add_metadata(writer, GGUF_PREFIX + "tokenizer.language_codes", tokenizer_meta["language_codes"])
    add_metadata(writer, GGUF_PREFIX + "tokenizer.language_token_ids", tokenizer_meta["language_token_ids"])

    if native_mode:
        enc = model.model.encoder
        add_tensor(writer, "encoder.pre_encode.conv0.weight", conv2d_weight(enc.subsampling.layers[0]), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv0.bias", as_numpy(enc.subsampling.layers[0].bias, np.float32), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv1_dw.weight", conv2d_weight(enc.subsampling.layers[2]), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv1_dw.bias", as_numpy(enc.subsampling.layers[2].bias, np.float32), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv1_pw.weight", conv2d_weight(enc.subsampling.layers[3]), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv1_pw.bias", as_numpy(enc.subsampling.layers[3].bias, np.float32), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv2_dw.weight", conv2d_weight(enc.subsampling.layers[5]), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv2_dw.bias", as_numpy(enc.subsampling.layers[5].bias, np.float32), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv2_pw.weight", conv2d_weight(enc.subsampling.layers[6]), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv2_pw.bias", as_numpy(enc.subsampling.layers[6].bias, np.float32), export_dtype)
        add_tensor(writer, "encoder.pre_encode.out.weight", linear_weight(enc.subsampling.linear), export_dtype)
        add_tensor(writer, "encoder.pre_encode.out.bias", linear_bias(enc.subsampling.linear), export_dtype)

        for il, layer in enumerate(enc.layers):
            prefix = f"encoder.layers.{il}."
            add_tensor(writer, prefix + "norm_feed_forward1.weight", as_numpy(layer.norm_feed_forward1.weight, np.float32), export_dtype)
            add_tensor(writer, prefix + "norm_feed_forward1.bias", as_numpy(layer.norm_feed_forward1.bias, np.float32), export_dtype)
            add_tensor(writer, prefix + "feed_forward1.linear1.weight", linear_weight(layer.feed_forward1.linear1), export_dtype)
            add_tensor(writer, prefix + "feed_forward1.linear1.bias", linear_bias(layer.feed_forward1.linear1), export_dtype)
            add_tensor(writer, prefix + "feed_forward1.linear2.weight", linear_weight(layer.feed_forward1.linear2), export_dtype)
            add_tensor(writer, prefix + "feed_forward1.linear2.bias", linear_bias(layer.feed_forward1.linear2), export_dtype)
            add_tensor(writer, prefix + "norm_self_att.weight", as_numpy(layer.norm_self_att.weight, np.float32), export_dtype)
            add_tensor(writer, prefix + "norm_self_att.bias", as_numpy(layer.norm_self_att.bias, np.float32), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_q.weight", linear_weight(layer.self_attn.q_proj), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_q.bias", linear_bias(layer.self_attn.q_proj), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_k.weight", linear_weight(layer.self_attn.k_proj), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_k.bias", linear_bias(layer.self_attn.k_proj), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_v.weight", linear_weight(layer.self_attn.v_proj), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_v.bias", linear_bias(layer.self_attn.v_proj), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_pos.weight", linear_weight(layer.self_attn.relative_k_proj), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_out.weight", linear_weight(layer.self_attn.o_proj), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_out.bias", linear_bias(layer.self_attn.o_proj), export_dtype)
            add_tensor(writer, prefix + "self_attn.pos_bias_u", as_numpy(layer.self_attn.bias_u, np.float32), export_dtype)
            add_tensor(writer, prefix + "self_attn.pos_bias_v", as_numpy(layer.self_attn.bias_v, np.float32), export_dtype)
            add_tensor(writer, prefix + "norm_conv.weight", as_numpy(layer.norm_conv.weight, np.float32), export_dtype)
            add_tensor(writer, prefix + "norm_conv.bias", as_numpy(layer.norm_conv.bias, np.float32), export_dtype)
            add_tensor(writer, prefix + "conv.pointwise_conv1.weight", conv1d_weight(layer.conv.pointwise_conv1), export_dtype)
            add_tensor(writer, prefix + "conv.pointwise_conv1.bias", as_numpy(layer.conv.pointwise_conv1.bias, np.float32), export_dtype)
            fused_dw_weight, fused_dw_bias = fuse_depthwise_conv_and_bn(layer.conv.depthwise_conv, layer.conv.norm)
            add_tensor(writer, prefix + "conv.depthwise_conv.weight", fused_dw_weight, export_dtype)
            add_tensor(writer, prefix + "conv.depthwise_conv.bias", fused_dw_bias, export_dtype)
            add_tensor(writer, prefix + "conv.pointwise_conv2.weight", conv1d_weight(layer.conv.pointwise_conv2), export_dtype)
            add_tensor(writer, prefix + "conv.pointwise_conv2.bias", as_numpy(layer.conv.pointwise_conv2.bias, np.float32), export_dtype)
            add_tensor(writer, prefix + "norm_feed_forward2.weight", as_numpy(layer.norm_feed_forward2.weight, np.float32), export_dtype)
            add_tensor(writer, prefix + "norm_feed_forward2.bias", as_numpy(layer.norm_feed_forward2.bias, np.float32), export_dtype)
            add_tensor(writer, prefix + "feed_forward2.linear1.weight", linear_weight(layer.feed_forward2.linear1), export_dtype)
            add_tensor(writer, prefix + "feed_forward2.linear1.bias", linear_bias(layer.feed_forward2.linear1), export_dtype)
            add_tensor(writer, prefix + "feed_forward2.linear2.weight", linear_weight(layer.feed_forward2.linear2), export_dtype)
            add_tensor(writer, prefix + "feed_forward2.linear2.bias", linear_bias(layer.feed_forward2.linear2), export_dtype)
            add_tensor(writer, prefix + "norm_out.weight", as_numpy(layer.norm_out.weight, np.float32), export_dtype)
            add_tensor(writer, prefix + "norm_out.bias", as_numpy(layer.norm_out.bias, np.float32), export_dtype)

        dec = model.model.decoder
        add_tensor(writer, "decoder.embedding.token_embedding.weight", embedding_weight(dec.embed_tokens), export_dtype)
        add_tensor(writer, "decoder.embedding.layer_norm.weight", as_numpy(dec.embedding_layernorm.weight, np.float32), export_dtype)
        add_tensor(writer, "decoder.embedding.layer_norm.bias", as_numpy(dec.embedding_layernorm.bias, np.float32), export_dtype)

        for il, layer in enumerate(dec.layers):
            prefix = f"decoder.layers.{il}."
            add_tensor(writer, prefix + "layer_norm_1.weight", as_numpy(layer.input_layernorm.weight, np.float32), export_dtype)
            add_tensor(writer, prefix + "layer_norm_1.bias", as_numpy(layer.input_layernorm.bias, np.float32), export_dtype)
            add_tensor(writer, prefix + "self_attn.query.weight", linear_weight(layer.self_attn.q_proj), export_dtype)
            add_tensor(writer, prefix + "self_attn.query.bias", linear_bias(layer.self_attn.q_proj), export_dtype)
            add_tensor(writer, prefix + "self_attn.key.weight", linear_weight(layer.self_attn.k_proj), export_dtype)
            add_tensor(writer, prefix + "self_attn.key.bias", linear_bias(layer.self_attn.k_proj), export_dtype)
            add_tensor(writer, prefix + "self_attn.value.weight", linear_weight(layer.self_attn.v_proj), export_dtype)
            add_tensor(writer, prefix + "self_attn.value.bias", linear_bias(layer.self_attn.v_proj), export_dtype)
            add_tensor(writer, prefix + "self_attn.out.weight", linear_weight(layer.self_attn.o_proj), export_dtype)
            add_tensor(writer, prefix + "self_attn.out.bias", linear_bias(layer.self_attn.o_proj), export_dtype)
            add_tensor(writer, prefix + "layer_norm_2.weight", as_numpy(layer.post_attention_layernorm.weight, np.float32), export_dtype)
            add_tensor(writer, prefix + "layer_norm_2.bias", as_numpy(layer.post_attention_layernorm.bias, np.float32), export_dtype)
            add_tensor(writer, prefix + "cross_attn.query.weight", linear_weight(layer.encoder_attn.q_proj), export_dtype)
            add_tensor(writer, prefix + "cross_attn.query.bias", linear_bias(layer.encoder_attn.q_proj), export_dtype)
            add_tensor(writer, prefix + "cross_attn.key.weight", linear_weight(layer.encoder_attn.k_proj), export_dtype)
            add_tensor(writer, prefix + "cross_attn.key.bias", linear_bias(layer.encoder_attn.k_proj), export_dtype)
            add_tensor(writer, prefix + "cross_attn.value.weight", linear_weight(layer.encoder_attn.v_proj), export_dtype)
            add_tensor(writer, prefix + "cross_attn.value.bias", linear_bias(layer.encoder_attn.v_proj), export_dtype)
            add_tensor(writer, prefix + "cross_attn.out.weight", linear_weight(layer.encoder_attn.o_proj), export_dtype)
            add_tensor(writer, prefix + "cross_attn.out.bias", linear_bias(layer.encoder_attn.o_proj), export_dtype)
            add_tensor(writer, prefix + "layer_norm_3.weight", as_numpy(layer.final_layernorm.weight, np.float32), export_dtype)
            add_tensor(writer, prefix + "layer_norm_3.bias", as_numpy(layer.final_layernorm.bias, np.float32), export_dtype)
            add_tensor(writer, prefix + "feed_forward.dense_in.weight", linear_weight(layer.mlp.fc1), export_dtype)
            add_tensor(writer, prefix + "feed_forward.dense_in.bias", linear_bias(layer.mlp.fc1), export_dtype)
            add_tensor(writer, prefix + "feed_forward.dense_out.weight", linear_weight(layer.mlp.fc2), export_dtype)
            add_tensor(writer, prefix + "feed_forward.dense_out.bias", linear_bias(layer.mlp.fc2), export_dtype)

        add_tensor(writer, "decoder.final_layer_norm.weight", as_numpy(dec.norm.weight, np.float32), export_dtype)
        add_tensor(writer, "decoder.final_layer_norm.bias", as_numpy(dec.norm.bias, np.float32), export_dtype)
        add_tensor(writer, "encoder_decoder_proj.weight", linear_weight(dec.proj), export_dtype)
        add_tensor(writer, "encoder_decoder_proj.bias", linear_bias(dec.proj), export_dtype)

        lm_head_weight = as_numpy(model.proj_out.weight, np.float32)
        emb_weight = as_numpy(dec.embed_tokens.weight, np.float32)
    else:
        enc = model.encoder
        add_tensor(writer, "encoder.pre_encode.conv0.weight", conv2d_weight(enc.pre_encode.conv[0]), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv0.bias", as_numpy(enc.pre_encode.conv[0].bias, np.float32), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv1_dw.weight", conv2d_weight(enc.pre_encode.conv[2]), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv1_dw.bias", as_numpy(enc.pre_encode.conv[2].bias, np.float32), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv1_pw.weight", conv2d_weight(enc.pre_encode.conv[3]), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv1_pw.bias", as_numpy(enc.pre_encode.conv[3].bias, np.float32), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv2_dw.weight", conv2d_weight(enc.pre_encode.conv[5]), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv2_dw.bias", as_numpy(enc.pre_encode.conv[5].bias, np.float32), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv2_pw.weight", conv2d_weight(enc.pre_encode.conv[6]), export_dtype)
        add_tensor(writer, "encoder.pre_encode.conv2_pw.bias", as_numpy(enc.pre_encode.conv[6].bias, np.float32), export_dtype)
        add_tensor(writer, "encoder.pre_encode.out.weight", linear_weight(enc.pre_encode.out), export_dtype)
        add_tensor(writer, "encoder.pre_encode.out.bias", linear_bias(enc.pre_encode.out), export_dtype)

        for il, layer in enumerate(enc.layers):
            prefix = f"encoder.layers.{il}."
            add_tensor(writer, prefix + "norm_feed_forward1.weight", as_numpy(layer.norm_feed_forward1.weight, np.float32), export_dtype)
            add_tensor(writer, prefix + "norm_feed_forward1.bias", as_numpy(layer.norm_feed_forward1.bias, np.float32), export_dtype)
            add_tensor(writer, prefix + "feed_forward1.linear1.weight", linear_weight(layer.feed_forward1.linear1), export_dtype)
            add_tensor(writer, prefix + "feed_forward1.linear1.bias", linear_bias(layer.feed_forward1.linear1), export_dtype)
            add_tensor(writer, prefix + "feed_forward1.linear2.weight", linear_weight(layer.feed_forward1.linear2), export_dtype)
            add_tensor(writer, prefix + "feed_forward1.linear2.bias", linear_bias(layer.feed_forward1.linear2), export_dtype)
            add_tensor(writer, prefix + "norm_self_att.weight", as_numpy(layer.norm_self_att.weight, np.float32), export_dtype)
            add_tensor(writer, prefix + "norm_self_att.bias", as_numpy(layer.norm_self_att.bias, np.float32), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_q.weight", linear_weight(layer.self_attn.linear_q), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_q.bias", linear_bias(layer.self_attn.linear_q), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_k.weight", linear_weight(layer.self_attn.linear_k), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_k.bias", linear_bias(layer.self_attn.linear_k), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_v.weight", linear_weight(layer.self_attn.linear_v), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_v.bias", linear_bias(layer.self_attn.linear_v), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_pos.weight", linear_weight(layer.self_attn.linear_pos), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_out.weight", linear_weight(layer.self_attn.linear_out), export_dtype)
            add_tensor(writer, prefix + "self_attn.linear_out.bias", linear_bias(layer.self_attn.linear_out), export_dtype)
            add_tensor(writer, prefix + "self_attn.pos_bias_u", as_numpy(layer.self_attn.pos_bias_u, np.float32), export_dtype)
            add_tensor(writer, prefix + "self_attn.pos_bias_v", as_numpy(layer.self_attn.pos_bias_v, np.float32), export_dtype)
            add_tensor(writer, prefix + "norm_conv.weight", as_numpy(layer.norm_conv.weight, np.float32), export_dtype)
            add_tensor(writer, prefix + "norm_conv.bias", as_numpy(layer.norm_conv.bias, np.float32), export_dtype)
            add_tensor(writer, prefix + "conv.pointwise_conv1.weight", conv1d_weight(layer.conv.pointwise_conv1), export_dtype)
            add_tensor(writer, prefix + "conv.pointwise_conv1.bias", as_numpy(layer.conv.pointwise_conv1.bias, np.float32), export_dtype)
            fused_dw_weight, fused_dw_bias = fuse_depthwise_conv_and_bn(layer.conv.depthwise_conv, layer.conv.batch_norm)
            add_tensor(writer, prefix + "conv.depthwise_conv.weight", fused_dw_weight, export_dtype)
            add_tensor(writer, prefix + "conv.depthwise_conv.bias", fused_dw_bias, export_dtype)
            add_tensor(writer, prefix + "conv.pointwise_conv2.weight", conv1d_weight(layer.conv.pointwise_conv2), export_dtype)
            add_tensor(writer, prefix + "conv.pointwise_conv2.bias", as_numpy(layer.conv.pointwise_conv2.bias, np.float32), export_dtype)
            add_tensor(writer, prefix + "norm_feed_forward2.weight", as_numpy(layer.norm_feed_forward2.weight, np.float32), export_dtype)
            add_tensor(writer, prefix + "norm_feed_forward2.bias", as_numpy(layer.norm_feed_forward2.bias, np.float32), export_dtype)
            add_tensor(writer, prefix + "feed_forward2.linear1.weight", linear_weight(layer.feed_forward2.linear1), export_dtype)
            add_tensor(writer, prefix + "feed_forward2.linear1.bias", linear_bias(layer.feed_forward2.linear1), export_dtype)
            add_tensor(writer, prefix + "feed_forward2.linear2.weight", linear_weight(layer.feed_forward2.linear2), export_dtype)
            add_tensor(writer, prefix + "feed_forward2.linear2.bias", linear_bias(layer.feed_forward2.linear2), export_dtype)
            add_tensor(writer, prefix + "norm_out.weight", as_numpy(layer.norm_out.weight, np.float32), export_dtype)
            add_tensor(writer, prefix + "norm_out.bias", as_numpy(layer.norm_out.bias, np.float32), export_dtype)

        dec_embed = model.transf_decoder._embedding
        dec_core = model.transf_decoder._decoder
        add_tensor(writer, "decoder.embedding.token_embedding.weight", embedding_weight(dec_embed.token_embedding), export_dtype)
        add_tensor(writer, "decoder.embedding.layer_norm.weight", as_numpy(dec_embed.layer_norm.weight, np.float32), export_dtype)
        add_tensor(writer, "decoder.embedding.layer_norm.bias", as_numpy(dec_embed.layer_norm.bias, np.float32), export_dtype)

        for il, layer in enumerate(dec_core.layers):
            prefix = f"decoder.layers.{il}."
            add_tensor(writer, prefix + "layer_norm_1.weight", as_numpy(layer.layer_norm_1.weight, np.float32), export_dtype)
            add_tensor(writer, prefix + "layer_norm_1.bias", as_numpy(layer.layer_norm_1.bias, np.float32), export_dtype)
            add_tensor(writer, prefix + "self_attn.query.weight", linear_weight(layer.first_sub_layer.query_net), export_dtype)
            add_tensor(writer, prefix + "self_attn.query.bias", linear_bias(layer.first_sub_layer.query_net), export_dtype)
            add_tensor(writer, prefix + "self_attn.key.weight", linear_weight(layer.first_sub_layer.key_net), export_dtype)
            add_tensor(writer, prefix + "self_attn.key.bias", linear_bias(layer.first_sub_layer.key_net), export_dtype)
            add_tensor(writer, prefix + "self_attn.value.weight", linear_weight(layer.first_sub_layer.value_net), export_dtype)
            add_tensor(writer, prefix + "self_attn.value.bias", linear_bias(layer.first_sub_layer.value_net), export_dtype)
            add_tensor(writer, prefix + "self_attn.out.weight", linear_weight(layer.first_sub_layer.out_projection), export_dtype)
            add_tensor(writer, prefix + "self_attn.out.bias", linear_bias(layer.first_sub_layer.out_projection), export_dtype)
            add_tensor(writer, prefix + "layer_norm_2.weight", as_numpy(layer.layer_norm_2.weight, np.float32), export_dtype)
            add_tensor(writer, prefix + "layer_norm_2.bias", as_numpy(layer.layer_norm_2.bias, np.float32), export_dtype)
            add_tensor(writer, prefix + "cross_attn.query.weight", linear_weight(layer.second_sub_layer.query_net), export_dtype)
            add_tensor(writer, prefix + "cross_attn.query.bias", linear_bias(layer.second_sub_layer.query_net), export_dtype)
            add_tensor(writer, prefix + "cross_attn.key.weight", linear_weight(layer.second_sub_layer.key_net), export_dtype)
            add_tensor(writer, prefix + "cross_attn.key.bias", linear_bias(layer.second_sub_layer.key_net), export_dtype)
            add_tensor(writer, prefix + "cross_attn.value.weight", linear_weight(layer.second_sub_layer.value_net), export_dtype)
            add_tensor(writer, prefix + "cross_attn.value.bias", linear_bias(layer.second_sub_layer.value_net), export_dtype)
            add_tensor(writer, prefix + "cross_attn.out.weight", linear_weight(layer.second_sub_layer.out_projection), export_dtype)
            add_tensor(writer, prefix + "cross_attn.out.bias", linear_bias(layer.second_sub_layer.out_projection), export_dtype)
            add_tensor(writer, prefix + "layer_norm_3.weight", as_numpy(layer.layer_norm_3.weight, np.float32), export_dtype)
            add_tensor(writer, prefix + "layer_norm_3.bias", as_numpy(layer.layer_norm_3.bias, np.float32), export_dtype)
            add_tensor(writer, prefix + "feed_forward.dense_in.weight", linear_weight(layer.third_sub_layer.dense_in), export_dtype)
            add_tensor(writer, prefix + "feed_forward.dense_in.bias", linear_bias(layer.third_sub_layer.dense_in), export_dtype)
            add_tensor(writer, prefix + "feed_forward.dense_out.weight", linear_weight(layer.third_sub_layer.dense_out), export_dtype)
            add_tensor(writer, prefix + "feed_forward.dense_out.bias", linear_bias(layer.third_sub_layer.dense_out), export_dtype)

        add_tensor(writer, "decoder.final_layer_norm.weight", as_numpy(dec_core.final_layer_norm.weight, np.float32), export_dtype)
        add_tensor(writer, "decoder.final_layer_norm.bias", as_numpy(dec_core.final_layer_norm.bias, np.float32), export_dtype)

        if model.encoder_decoder_proj is not None:
            add_tensor(writer, "encoder_decoder_proj.weight", linear_weight(model.encoder_decoder_proj), export_dtype)
            add_tensor(writer, "encoder_decoder_proj.bias", linear_bias(model.encoder_decoder_proj), export_dtype)

        lm_head_weight = as_numpy(model.log_softmax.mlp.layer0.weight, np.float32)
        emb_weight = as_numpy(model.transf_decoder._embedding.token_embedding.weight, np.float32)

    if not np.allclose(lm_head_weight, emb_weight, atol=0.0, rtol=0.0):
        print("warning: lm_head.weight is not tied to decoder token_embedding.weight; exporting separate lm_head.weight", file=sys.stderr)
        add_tensor(writer, "lm_head.weight", lm_head_weight, export_dtype)
    if native_mode:
        add_tensor(writer, "lm_head.bias", as_numpy(model.proj_out.bias, np.float32), export_dtype)
    else:
        add_tensor(writer, "lm_head.bias", as_numpy(model.log_softmax.mlp.layer0.bias, np.float32), export_dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    args = parse_args()
    snapshot_dir = args.snapshot.resolve()
    outfile = args.outfile.resolve()
    require_snapshot(snapshot_dir)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    export_model(snapshot_dir, outfile, args.dtype)
    print(f"wrote {outfile}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
