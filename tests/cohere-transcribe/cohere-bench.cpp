#include "cohere.h"
#include "common-whisper.h"
#include "json.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

struct bench_params {
    std::string model;
    std::string language = "en";
    int32_t threads = 1;
    std::vector<std::string> files;
};

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

        double read_total_ms = 0.0;
        double transcribe_total_ms = 0.0;
        const double run_start_ms = ggml_time_ms();

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
            const double transcribe_start_ms = ggml_time_ms();
            if (!cohere::transcribe(model, pcmf32, tparams, text, error)) {
                cohere::free_model(model);
                throw std::runtime_error("failed to transcribe '" + path + "': " + error);
            }
            const double transcribe_end_ms = ggml_time_ms();

            const double read_ms = read_end_ms - read_start_ms;
            const double transcribe_ms = transcribe_end_ms - transcribe_start_ms;
            read_total_ms += read_ms;
            transcribe_total_ms += transcribe_ms;

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
            clips.push_back(clip);
        }

        const double run_end_ms = ggml_time_ms();

        result["read_total_s"] = read_total_ms / 1000.0;
        result["transcribe_total_s"] = transcribe_total_ms / 1000.0;
        result["warm_e2e_total_s"] = (read_total_ms + transcribe_total_ms) / 1000.0;
        result["run_total_s"] = (run_end_ms - run_start_ms) / 1000.0;
        result["end_to_end_s"] = result["load_s"].get<double>() + result["run_total_s"].get<double>();
        result["clips"] = clips;

        cohere::free_model(model);
        printf("%s\n", result.dump(2).c_str());
        return 0;
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
