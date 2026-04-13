#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration


GGUF_ARCH = "cohere-transcribe"
GGUF_PREFIX = "cohere_transcribe."
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
        help="Tensor storage type for weights (default: f16)",
    )
    return parser.parse_args()


def as_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(dtype=torch.float32)
        array = tensor.numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        array = np.asarray(value)

    if dtype is not None:
        array = array.astype(dtype, copy=False)

    return np.ascontiguousarray(array)


def load_snapshot(snapshot_dir: Path) -> tuple[Any, Any, Any]:
    processor = AutoProcessor.from_pretrained(snapshot_dir)
    model = CohereAsrForConditionalGeneration.from_pretrained(snapshot_dir, dtype=torch.float32)
    model.eval()
    return processor, processor.feature_extractor, model


def normalize_hidden_act(value: str) -> str:
    return str(value).lower().replace("swish", "silu")


def extract_frontend(feature_extractor: Any) -> dict[str, Any]:
    sample_rate = int(feature_extractor.sampling_rate)
    n_mels = int(feature_extractor.feature_size)
    n_fft = int(feature_extractor.n_fft)
    win_length = int(feature_extractor.win_length)
    hop_length = int(feature_extractor.hop_length)
    mel_filters = as_numpy(feature_extractor.mel_filters, np.float32)

    expected_shape = (n_mels, n_fft // 2 + 1)
    if mel_filters.shape != expected_shape:
        raise SystemExit(f"unexpected mel filterbank shape: {tuple(mel_filters.shape)}")

    if str(feature_extractor.window).lower() != "hann":
        raise SystemExit(f"unsupported window type: {feature_extractor.window!r}")

    return {
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "win_length": win_length,
        "hop_length": hop_length,
        "fmin": 0.0,
        "fmax": sample_rate / 2.0,
        "preemph": float(feature_extractor.preemphasis),
        "dither": float(feature_extractor.dither),
        "log_zero_guard": float(2.0 ** -24),
        "normalize_per_feature": feature_extractor.normalize == "per_feature",
        "window": as_numpy(torch.hann_window(win_length, periodic=False), np.float32),
        "mel_filters": mel_filters,
    }


def extract_tokenizer_metadata(tokenizer: Any, supported_languages: list[str]) -> dict[str, Any]:
    special_token_names = sorted(set(tokenizer.all_special_tokens) | set(REQUIRED_SPECIAL_TOKENS))

    return {
        "tokens": [str(tokenizer.convert_ids_to_tokens(i)) for i in range(len(tokenizer))],
        "supported_languages": supported_languages,
        "special_token_names": special_token_names,
        "special_token_ids": [int(tokenizer.convert_tokens_to_ids(token)) for token in special_token_names],
        "language_codes": supported_languages,
        "language_token_ids": [int(tokenizer.convert_tokens_to_ids(f"<|{language}|>")) for language in supported_languages],
    }


def weight_dtype(name: str, export_dtype: str) -> np.dtype:
    if export_dtype == "f32":
        return np.float32
    if name.endswith(".bias") or "norm" in name or "pos_bias_" in name:
        return np.float32
    return np.float16


def add_tensor(writer: Any, name: str, array: Any, export_dtype: str) -> None:
    writer.add_tensor(name, as_numpy(array, weight_dtype(name, export_dtype)))


def add_value_metadata(writer: Any, key: str, value: Any) -> None:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()

    if isinstance(value, bool):
        writer.add_bool(key, value)
    elif isinstance(value, int):
        writer.add_int32(key, value)
    elif isinstance(value, float):
        writer.add_float32(key, value)
    elif isinstance(value, str):
        writer.add_string(key, value)
    else:
        writer.add_array(key, value)


def add_metadata_group(writer: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        add_value_metadata(writer, GGUF_PREFIX + key, value)


def add_module(writer: Any, prefix: str, module: Any, export_dtype: str) -> None:
    add_tensor(writer, prefix + ".weight", module.weight, export_dtype)
    add_tensor(writer, prefix + ".bias", module.bias, export_dtype)


def add_weight_only(writer: Any, name: str, value: Any, export_dtype: str) -> None:
    add_tensor(writer, name, value, export_dtype)


def fuse_depthwise_conv_and_bn(conv_module: Any, bn_module: Any) -> tuple[np.ndarray, np.ndarray]:
    conv_weight = as_numpy(conv_module.weight, np.float32)
    conv_bias = as_numpy(conv_module.bias, np.float32).reshape(-1)
    gamma = as_numpy(bn_module.weight, np.float32).reshape(-1)
    beta = as_numpy(bn_module.bias, np.float32).reshape(-1)
    running_mean = as_numpy(bn_module.running_mean, np.float32).reshape(-1)
    running_var = as_numpy(bn_module.running_var, np.float32).reshape(-1)
    scale = gamma / np.sqrt(running_var + float(bn_module.eps))

    fused_weight = conv_weight * scale[:, None, None]
    fused_bias = beta + (conv_bias - running_mean) * scale

    return fused_weight, fused_bias


def export_pre_encode(writer: Any, encoder: Any, export_dtype: str) -> None:
    layers = encoder.subsampling.layers

    for name, module in (
        ("encoder.pre_encode.conv0", layers[0]),
        ("encoder.pre_encode.conv1_dw", layers[2]),
        ("encoder.pre_encode.conv1_pw", layers[3]),
        ("encoder.pre_encode.conv2_dw", layers[5]),
        ("encoder.pre_encode.conv2_pw", layers[6]),
        ("encoder.pre_encode.out", encoder.subsampling.linear),
    ):
        add_module(writer, name, module, export_dtype)


def export_encoder(writer: Any, encoder: Any, export_dtype: str) -> None:
    export_pre_encode(writer, encoder, export_dtype)

    for layer_index, layer in enumerate(encoder.layers):
        prefix = f"encoder.layers.{layer_index}."

        for name, module in (
            ("norm_feed_forward1", layer.norm_feed_forward1),
            ("feed_forward1.linear1", layer.feed_forward1.linear1),
            ("feed_forward1.linear2", layer.feed_forward1.linear2),
            ("norm_self_att", layer.norm_self_att),
            ("self_attn.linear_q", layer.self_attn.q_proj),
            ("self_attn.linear_k", layer.self_attn.k_proj),
            ("self_attn.linear_v", layer.self_attn.v_proj),
            ("self_attn.linear_out", layer.self_attn.o_proj),
            ("norm_conv", layer.norm_conv),
            ("conv.pointwise_conv1", layer.conv.pointwise_conv1),
            ("conv.pointwise_conv2", layer.conv.pointwise_conv2),
            ("norm_feed_forward2", layer.norm_feed_forward2),
            ("feed_forward2.linear1", layer.feed_forward2.linear1),
            ("feed_forward2.linear2", layer.feed_forward2.linear2),
            ("norm_out", layer.norm_out),
        ):
            add_module(writer, prefix + name, module, export_dtype)

        add_weight_only(writer, prefix + "self_attn.linear_pos.weight", layer.self_attn.relative_k_proj.weight, export_dtype)
        add_weight_only(writer, prefix + "self_attn.pos_bias_u", layer.self_attn.bias_u, export_dtype)
        add_weight_only(writer, prefix + "self_attn.pos_bias_v", layer.self_attn.bias_v, export_dtype)

        depthwise_weight, depthwise_bias = fuse_depthwise_conv_and_bn(layer.conv.depthwise_conv, layer.conv.norm)
        add_weight_only(writer, prefix + "conv.depthwise_conv.weight", depthwise_weight, export_dtype)
        add_weight_only(writer, prefix + "conv.depthwise_conv.bias", depthwise_bias, export_dtype)


def export_decoder(writer: Any, decoder: Any, export_dtype: str) -> None:
    add_weight_only(writer, "decoder.embedding.token_embedding.weight", decoder.embed_tokens.weight, export_dtype)
    add_module(writer, "decoder.embedding.layer_norm", decoder.embedding_layernorm, export_dtype)

    for layer_index, layer in enumerate(decoder.layers):
        prefix = f"decoder.layers.{layer_index}."

        for name, module in (
            ("layer_norm_1", layer.input_layernorm),
            ("self_attn.query", layer.self_attn.q_proj),
            ("self_attn.key", layer.self_attn.k_proj),
            ("self_attn.value", layer.self_attn.v_proj),
            ("self_attn.out", layer.self_attn.o_proj),
            ("layer_norm_2", layer.post_attention_layernorm),
            ("cross_attn.query", layer.encoder_attn.q_proj),
            ("cross_attn.key", layer.encoder_attn.k_proj),
            ("cross_attn.value", layer.encoder_attn.v_proj),
            ("cross_attn.out", layer.encoder_attn.o_proj),
            ("layer_norm_3", layer.final_layernorm),
            ("feed_forward.dense_in", layer.mlp.fc1),
            ("feed_forward.dense_out", layer.mlp.fc2),
        ):
            add_module(writer, prefix + name, module, export_dtype)

    add_module(writer, "decoder.final_layer_norm", decoder.norm, export_dtype)
    add_module(writer, "encoder_decoder_proj", decoder.proj, export_dtype)


def export_model(snapshot_dir: Path, outfile: Path, export_dtype: str) -> None:
    processor, feature_extractor, model = load_snapshot(snapshot_dir)
    frontend = extract_frontend(feature_extractor)
    supported_languages = list(model.config.supported_languages)
    tokenizer_meta = extract_tokenizer_metadata(processor.tokenizer, supported_languages)

    encoder_config = model.config.encoder_config.to_dict()
    encoder_metadata = {
        "d_model": int(encoder_config["hidden_size"]),
        "feat_in": int(encoder_config.get("num_mel_bins", frontend["n_mels"])),
        "feat_out": int(encoder_config["hidden_size"]),
        "n_layers": int(encoder_config["num_hidden_layers"]),
        "n_heads": int(encoder_config["num_attention_heads"]),
        "ff_expansion_factor": int(encoder_config["intermediate_size"] // encoder_config["hidden_size"]),
        "conv_kernel_size": int(encoder_config["conv_kernel_size"]),
        "subsampling_factor": int(encoder_config["subsampling_factor"]),
        "subsampling_conv_channels": int(encoder_config["subsampling_conv_channels"]),
        "pos_emb_max_len": int(encoder_config["max_position_embeddings"]),
    }

    decoder_config = model.model.decoder.config.to_dict()
    decoder_metadata = {
        "hidden_size": int(decoder_config["hidden_size"]),
        "inner_size": int(decoder_config["intermediate_size"]),
        "num_attention_heads": int(decoder_config["num_attention_heads"]),
        "num_layers": int(decoder_config["num_hidden_layers"]),
        "max_sequence_length": int(decoder_config["max_position_embeddings"]),
        "hidden_act": normalize_hidden_act(decoder_config["hidden_act"]),
    }

    head_metadata = {
        "hidden_size": int(model.config.hidden_size),
        "num_classes": int(model.config.vocab_size),
        "log_softmax": False,
    }

    writer = gguf.GGUFWriter(str(outfile), GGUF_ARCH)
    writer.add_name("Cohere Transcribe")
    writer.add_file_type(gguf.GGMLQuantizationType.F16 if export_dtype == "f16" else gguf.GGMLQuantizationType.F32)

    add_metadata_group(
        writer,
        {
            "max_audio_clip_s": float(model.config.max_audio_clip_s),
            **{f"frontend.{key}": value for key, value in frontend.items()},
            **{f"encoder.{key}": value for key, value in encoder_metadata.items()},
            **{f"decoder.{key}": value for key, value in decoder_metadata.items()},
            **{f"head.{key}": value for key, value in head_metadata.items()},
            "encoder_decoder_proj": True,
            "tokenizer.tokens": tokenizer_meta["tokens"],
            "tokenizer.supported_languages": tokenizer_meta["supported_languages"],
            "tokenizer.special_token_names": tokenizer_meta["special_token_names"],
            "tokenizer.special_token_ids": tokenizer_meta["special_token_ids"],
            "tokenizer.language_codes": tokenizer_meta["language_codes"],
            "tokenizer.language_token_ids": tokenizer_meta["language_token_ids"],
        },
    )

    export_encoder(writer, model.model.encoder, export_dtype)
    export_decoder(writer, model.model.decoder, export_dtype)
    add_tensor(writer, "lm_head.bias", model.proj_out.bias, export_dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    args = parse_args()
    snapshot_dir = args.snapshot.resolve()
    outfile = args.outfile.resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    export_model(snapshot_dir, outfile, args.dtype)
    print(f"wrote {outfile}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
