#include "cohere.h"
#include "common-whisper.h"
#include "json.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>

#if defined(__APPLE__)
#include <unistd.h>
#endif

using json = nlohmann::json;

struct bench_params {
    std::string model;
    std::string language = "en";
    int32_t threads = 1;
    std::vector<std::string> files;
};

#if defined(__APPLE__)
static int32_t cohere_choose_veclib_threads(int32_t n_threads) {
    if (n_threads <= 1) {
        return 1;
    }

    return std::max(1, std::min(4, n_threads / 2));
}

static void cohere_configure_veclib_threads(int32_t n_threads) {
    if (std::getenv("VECLIB_MAXIMUM_THREADS") != nullptr) {
        return;
    }

    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d", cohere_choose_veclib_threads(n_threads));
    setenv("VECLIB_MAXIMUM_THREADS", buf, 1);
}

static void cohere_reexec_with_veclib_threads(int argc, char ** argv, int32_t n_threads) {
    (void) argc;

    if (std::getenv("VECLIB_MAXIMUM_THREADS") != nullptr || std::getenv("COHERE_VECLIB_REEXEC") != nullptr) {
        return;
    }

    cohere_configure_veclib_threads(n_threads);
    setenv("COHERE_VECLIB_REEXEC", "1", 1);
    execv(argv[0], argv);

    std::fprintf(stderr, "warning: failed to re-exec with VECLIB_MAXIMUM_THREADS: %s\n", std::strerror(errno));
}
#endif

static json profile_to_json(const cohere::transcribe_profile & profile, double transcribe_s) {
    const auto & t = profile.timings_us;
    const auto & c = profile.counters;
    const double denom = transcribe_s > 0.0 ? transcribe_s : 1e-12;

    json timings = {
        {"frontend", t.frontend / 1e6},
        {"subsampling", t.subsampling / 1e6},
        {"encoder_total", t.encoder_total / 1e6},
        {"encoder_ffn", t.encoder_ffn / 1e6},
        {"encoder_self_attention", t.encoder_self_attention / 1e6},
        {"encoder_conv", t.encoder_conv / 1e6},
        {"encoder_projection", t.encoder_projection / 1e6},
        {"cross_kv_build", t.cross_kv_build / 1e6},
        {"decoder_total", t.decoder_total / 1e6},
        {"decoder_self_attention", t.decoder_self_attention / 1e6},
        {"decoder_cross_attention", t.decoder_cross_attention / 1e6},
        {"decoder_ffn", t.decoder_ffn / 1e6},
        {"lm_head", t.lm_head / 1e6},
    };

    json shares = {
        {"frontend", timings["frontend"].get<double>() / denom},
        {"subsampling", timings["subsampling"].get<double>() / denom},
        {"encoder_total", timings["encoder_total"].get<double>() / denom},
        {"encoder_ffn", timings["encoder_ffn"].get<double>() / denom},
        {"encoder_self_attention", timings["encoder_self_attention"].get<double>() / denom},
        {"encoder_conv", timings["encoder_conv"].get<double>() / denom},
        {"encoder_projection", timings["encoder_projection"].get<double>() / denom},
        {"cross_kv_build", timings["cross_kv_build"].get<double>() / denom},
        {"decoder_total", timings["decoder_total"].get<double>() / denom},
        {"decoder_self_attention", timings["decoder_self_attention"].get<double>() / denom},
        {"decoder_cross_attention", timings["decoder_cross_attention"].get<double>() / denom},
        {"decoder_ffn", timings["decoder_ffn"].get<double>() / denom},
        {"lm_head", timings["lm_head"].get<double>() / denom},
    };

    json counters = {
        {"eval_linear_calls", c.eval_linear_calls},
        {"eval_linear_rank3_calls", c.eval_linear_rank3_calls},
        {"ggml_context_creations", c.ggml_context_creations},
        {"ggml_context_resets", c.ggml_context_resets},
        {"ggml_graph_launches", c.ggml_graph_launches},
        {"input_tensor_copies", c.input_tensor_copies},
        {"output_tensor_copies", c.output_tensor_copies},
        {"rel_pos_cache_hits", c.rel_pos_cache_hits},
        {"rel_pos_cache_misses", c.rel_pos_cache_misses},
    };

    json tokens = {
        {"prompt", profile.prompt_token_count},
        {"greedy", profile.greedy_token_count},
        {"decoder_steps", profile.decoder_step_count},
        {"encoder_frames", profile.encoder_frame_count},
    };

    return {
        {"timings_s", timings},
        {"timing_share_of_transcribe", shares},
        {"timing_notes", json::array({
            "encoder_total overlaps encoder_ffn, encoder_self_attention, and encoder_conv",
            "decoder_total overlaps decoder_self_attention, decoder_cross_attention, decoder_ffn, and lm_head",
        })},
        {"counters", counters},
        {"token_counts", tokens},
    };
}

static void accumulate_profile(cohere::transcribe_profile & total, const cohere::transcribe_profile & add) {
    total.timings_us.frontend += add.timings_us.frontend;
    total.timings_us.subsampling += add.timings_us.subsampling;
    total.timings_us.encoder_total += add.timings_us.encoder_total;
    total.timings_us.encoder_ffn += add.timings_us.encoder_ffn;
    total.timings_us.encoder_self_attention += add.timings_us.encoder_self_attention;
    total.timings_us.encoder_conv += add.timings_us.encoder_conv;
    total.timings_us.encoder_projection += add.timings_us.encoder_projection;
    total.timings_us.cross_kv_build += add.timings_us.cross_kv_build;
    total.timings_us.decoder_total += add.timings_us.decoder_total;
    total.timings_us.decoder_self_attention += add.timings_us.decoder_self_attention;
    total.timings_us.decoder_cross_attention += add.timings_us.decoder_cross_attention;
    total.timings_us.decoder_ffn += add.timings_us.decoder_ffn;
    total.timings_us.lm_head += add.timings_us.lm_head;

    total.counters.eval_linear_calls += add.counters.eval_linear_calls;
    total.counters.eval_linear_rank3_calls += add.counters.eval_linear_rank3_calls;
    total.counters.ggml_context_creations += add.counters.ggml_context_creations;
    total.counters.ggml_context_resets += add.counters.ggml_context_resets;
    total.counters.ggml_graph_launches += add.counters.ggml_graph_launches;
    total.counters.input_tensor_copies += add.counters.input_tensor_copies;
    total.counters.output_tensor_copies += add.counters.output_tensor_copies;
    total.counters.rel_pos_cache_hits += add.counters.rel_pos_cache_hits;
    total.counters.rel_pos_cache_misses += add.counters.rel_pos_cache_misses;

    total.prompt_token_count += add.prompt_token_count;
    total.greedy_token_count += add.greedy_token_count;
    total.decoder_step_count += add.decoder_step_count;
    total.encoder_frame_count += add.encoder_frame_count;
}

static void print_usage(const char * argv0) {
    fprintf(
            stderr,
            "usage: %s --model MODEL.gguf --file AUDIO [--file AUDIO ...] [--language LANG] [--threads N]\n",
            argv0);
}

static bool parse_args(int argc, char ** argv, bench_params & params) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", flag);
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--model") {
            params.model = require_value("--model");
        } else if (arg == "--file") {
            params.files.push_back(require_value("--file"));
        } else if (arg == "--language") {
            params.language = require_value("--language");
        } else if (arg == "--threads") {
            params.threads = std::stoi(require_value("--threads"));
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return false;
        }
    }

    if (params.model.empty() || params.files.empty()) {
        return false;
    }

    return true;
}

static std::string basename_stem(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    const size_t begin = slash == std::string::npos ? 0 : slash + 1;
    const size_t dot = path.find_last_of('.');
    const size_t end = (dot == std::string::npos || dot < begin) ? path.size() : dot;
    return path.substr(begin, end - begin);
}

int main(int argc, char ** argv) {
    bench_params params;
    if (!parse_args(argc, argv, params)) {
        print_usage(argv[0]);
        return 1;
    }

    try {
#if defined(__APPLE__)
        cohere_reexec_with_veclib_threads(argc, argv, params.threads);
#endif
        const double load_start_ms = ggml_time_ms();
        cohere::model model;
        std::string error;
        if (!cohere::load_model(params.model, model, error)) {
            throw std::runtime_error(error);
        }
        const double load_end_ms = ggml_time_ms();

        cohere::transcribe_params tparams;
        tparams.language = params.language;
        tparams.n_threads = params.threads;
        tparams.max_new_tokens = 256;
        tparams.punctuation = true;

        json result;
        result["model"] = params.model;
        result["language"] = params.language;
        result["threads"] = params.threads;
        result["load_s"] = (load_end_ms - load_start_ms) / 1000.0;
        result["benchmark_contract"] = {
            {"device", "cpu"},
            {"batch_size", 1},
            {"model_load_scope", "loaded_once_per_run"},
            {"primary_metric", "transcribe_total_s"},
            {"secondary_metric", "warm_e2e_total_s"},
        };

        double read_total_ms = 0.0;
        double transcribe_total_ms = 0.0;
        const double run_start_ms = ggml_time_ms();
        cohere::transcribe_profile aggregate_profile;

        json clips = json::array();
        for (size_t i = 0; i < params.files.size(); ++i) {
            const std::string & path = params.files[i];

            std::vector<float> pcmf32;
            std::vector<std::vector<float>> pcmf32s;

            const double read_start_ms = ggml_time_ms();
            if (!read_audio_data(path, pcmf32, pcmf32s, false)) {
                cohere::free_model(model);
                throw std::runtime_error("failed to read audio file: " + path);
            }
            const double read_end_ms = ggml_time_ms();

            std::string text;
            cohere::transcribe_profile profile;
            const double transcribe_start_ms = ggml_time_ms();
            if (!cohere::transcribe_with_profile(model, pcmf32, tparams, profile, text, error)) {
                cohere::free_model(model);
                throw std::runtime_error("failed to transcribe '" + path + "': " + error);
            }
            const double transcribe_end_ms = ggml_time_ms();

            const double read_ms = read_end_ms - read_start_ms;
            const double transcribe_ms = transcribe_end_ms - transcribe_start_ms;
            read_total_ms += read_ms;
            transcribe_total_ms += transcribe_ms;
            accumulate_profile(aggregate_profile, profile);

            json clip;
            clip["index"] = int(i);
            clip["code"] = basename_stem(path);
            clip["file"] = path;
            clip["samples"] = int64_t(pcmf32.size());
            clip["audio_s"] = double(pcmf32.size()) / 16000.0;
            clip["read_s"] = read_ms / 1000.0;
            clip["transcribe_s"] = transcribe_ms / 1000.0;
            clip["warm_e2e_s"] = (read_ms + transcribe_ms) / 1000.0;
            clip["text"] = text;
            clip["profile"] = profile_to_json(profile, transcribe_ms / 1000.0);
            clips.push_back(clip);
        }

        const double run_end_ms = ggml_time_ms();

        result["read_total_s"] = read_total_ms / 1000.0;
        result["transcribe_total_s"] = transcribe_total_ms / 1000.0;
        result["warm_e2e_total_s"] = (read_total_ms + transcribe_total_ms) / 1000.0;
        result["run_total_s"] = (run_end_ms - run_start_ms) / 1000.0;
        result["end_to_end_s"] = result["load_s"].get<double>() + result["run_total_s"].get<double>();
        result["aggregate_profile"] = profile_to_json(aggregate_profile, result["transcribe_total_s"].get<double>());
        result["clips"] = clips;

        cohere::free_model(model);
        printf("%s\n", result.dump(2).c_str());
        return 0;
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
