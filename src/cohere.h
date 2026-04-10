#pragma once

#include "ggml.h"
#include "gguf.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace cohere {

struct frontend_config {
    int32_t sample_rate = 16000;
    int32_t n_mels = 128;
    int32_t n_fft = 512;
    int32_t win_length = 400;
    int32_t hop_length = 160;
    float fmin = 0.0f;
    float fmax = 8000.0f;
    float preemph = 0.97f;
    float dither = 0.0f;
    float log_zero_guard = 0.0f;
    bool normalize_per_feature = true;
    std::vector<float> window;
    std::vector<float> mel_filters;
};

struct encoder_config {
    int32_t d_model = 0;
    int32_t feat_in = 0;
    int32_t feat_out = 0;
    int32_t n_layers = 0;
    int32_t n_heads = 0;
    int32_t ff_expansion_factor = 0;
    int32_t conv_kernel_size = 0;
    int32_t subsampling_factor = 0;
    int32_t subsampling_conv_channels = 0;
    int32_t pos_emb_max_len = 0;
};

struct decoder_config {
    int32_t hidden_size = 0;
    int32_t inner_size = 0;
    int32_t num_attention_heads = 0;
    int32_t num_layers = 0;
    int32_t max_sequence_length = 0;
    std::string hidden_act;
};

struct head_config {
    int32_t hidden_size = 0;
    int32_t num_classes = 0;
    bool log_softmax = false;
};

struct tokenizer {
    std::vector<std::string> id_to_piece;
    std::vector<std::string> supported_languages;
    std::set<int32_t> special_ids;
    std::map<std::string, int32_t> special_token_ids;
    std::map<std::string, int32_t> language_token_ids;

    bool is_language_supported(const std::string & language) const;
    int32_t special_token_id(const std::string & name) const;
    std::vector<int32_t> build_prompt(const std::string & language, bool punctuation = true) const;
    std::string detokenize(const std::vector<int32_t> & ids) const;
};

struct model {
    struct gguf_context * gguf = nullptr;
    struct ggml_context * weights = nullptr;

    frontend_config frontend;
    encoder_config encoder;
    decoder_config decoder;
    head_config head;
    tokenizer vocab;

    float max_audio_clip_s = 0.0f;

    bool has_encoder_decoder_proj = false;

    std::map<std::string, struct ggml_tensor *> tensors;
};

struct transcribe_params {
    std::string language = "en";
    int32_t n_threads = 1;
    int32_t max_new_tokens = 256;
    bool punctuation = true;
};

struct matrix_output {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<float> data;
};

struct debug_outputs {
    matrix_output mel;
    int32_t mel_len = 0;

    matrix_output subsampling_out;
    int32_t subsampling_len = 0;

    matrix_output block0_after_ff1;
    int32_t block0_after_ff1_len = 0;

    matrix_output block0_after_attn;
    int32_t block0_after_attn_len = 0;

    matrix_output block0_after_conv;
    int32_t block0_after_conv_len = 0;

    matrix_output block0_out;
    int32_t block0_out_len = 0;

    matrix_output encoder_out;
    int32_t encoder_out_len = 0;

    matrix_output encoder_projected;
    int32_t encoder_projected_len = 0;

    std::vector<int32_t> prompt_ids;
    std::vector<float> first_step_logits;
    std::vector<int32_t> greedy_ids;
    std::string text;
};

struct timing_breakdown_us {
    int64_t frontend = 0;
    int64_t subsampling = 0;
    int64_t encoder_total = 0;
    int64_t encoder_ffn = 0;
    int64_t encoder_self_attention = 0;
    int64_t encoder_conv = 0;
    int64_t encoder_projection = 0;
    int64_t cross_kv_build = 0;
    int64_t decoder_total = 0;
    int64_t decoder_self_attention = 0;
    int64_t decoder_cross_attention = 0;
    int64_t decoder_ffn = 0;
    int64_t lm_head = 0;
};

struct overhead_counters {
    int64_t eval_linear_calls = 0;
    int64_t eval_linear_rank3_calls = 0;
    int64_t ggml_context_creations = 0;
    int64_t ggml_context_resets = 0;
    int64_t ggml_graph_launches = 0;
    int64_t input_tensor_copies = 0;
    int64_t output_tensor_copies = 0;
    int64_t rel_pos_cache_hits = 0;
    int64_t rel_pos_cache_misses = 0;
};

struct transcribe_profile {
    timing_breakdown_us timings_us;
    overhead_counters counters;
    int32_t prompt_token_count = 0;
    int32_t greedy_token_count = 0;
    int32_t decoder_step_count = 0;
    int32_t encoder_frame_count = 0;
};

bool probe_model_architecture(const std::string & path_model, std::string & architecture, std::string * error = nullptr);

bool load_model(const std::string & path_model, model & out, std::string & error);
void free_model(model & model);

bool transcribe(
        model & model,
        const std::vector<float> & pcmf32,
        const transcribe_params & params,
        std::string & text,
        std::string & error);

bool transcribe_with_profile(
        model & model,
        const std::vector<float> & pcmf32,
        const transcribe_params & params,
        transcribe_profile & profile,
        std::string & text,
        std::string & error);

bool collect_debug_outputs(
        model & model,
        const std::vector<float> & pcmf32,
        const transcribe_params & params,
        debug_outputs & outputs,
        std::string & error);

int32_t conv_subsampling_output_length(int32_t n_frames);

std::vector<float> rel_shift_reference(
        const std::vector<float> & input,
        int32_t n_head,
        int32_t q_len,
        int32_t pos_len);

} // namespace cohere
