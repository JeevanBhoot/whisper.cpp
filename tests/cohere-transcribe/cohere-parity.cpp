#include "cohere.h"
#include "common-whisper.h"
#include "json.hpp"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

struct parity_params {
    std::string model;
    std::string fixtures;
    std::string clip;
    std::string stage = "all";
    float atol = 1e-4f;
    float rtol = 1e-4f;
};

struct stage_tolerance {
    float atol;
    float rtol;
};

static void print_usage(const char * argv0) {
    fprintf(
            stderr,
            "usage: %s --model MODEL.gguf --fixtures DIR --clip CLIP_ID [--stage STAGE] [--atol X] [--rtol Y]\n",
            argv0);
}

static bool parse_args(int argc, char ** argv, parity_params & params) {
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
        } else if (arg == "--fixtures") {
            params.fixtures = require_value("--fixtures");
        } else if (arg == "--clip") {
            params.clip = require_value("--clip");
        } else if (arg == "--stage") {
            params.stage = require_value("--stage");
        } else if (arg == "--atol") {
            params.atol = std::stof(require_value("--atol"));
        } else if (arg == "--rtol") {
            params.rtol = std::stof(require_value("--rtol"));
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return false;
        }
    }

    if (params.model.empty() || params.fixtures.empty() || params.clip.empty()) {
        return false;
    }

    return true;
}

static json load_manifest(const std::string & fixtures_dir) {
    std::ifstream in(fixtures_dir + "/manifest.json");
    if (!in.is_open()) {
        throw std::runtime_error("failed to open manifest.json");
    }
    json manifest;
    in >> manifest;
    return manifest;
}

static const json & find_clip(const json & manifest, const std::string & clip_id) {
    for (size_t i = 0; i < manifest.at("clips").size(); ++i) {
        const json & clip = manifest.at("clips").at(i);
        if (clip.at("id").get<std::string>() == clip_id) {
            return clip;
        }
    }
    throw std::runtime_error("clip id not found in manifest: " + clip_id);
}

static std::vector<float> read_f32_file(const std::string & path, size_t count) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open tensor file: " + path);
    }

    std::vector<float> out(count);
    in.read(reinterpret_cast<char *>(out.data()), std::streamsize(count * sizeof(float)));
    if (!in) {
        throw std::runtime_error("failed to read tensor file: " + path);
    }
    return out;
}

static float row_major_at(const std::vector<float> & data, int rows, int cols, int row, int col) {
    GGML_UNUSED(rows);
    return data[size_t(row) * size_t(cols) + size_t(col)];
}

static std::string canonical_stage_name(const std::string & stage_name) {
    return stage_name == "encoder_block0" ? "block0_out" : stage_name;
}

static std::string resolve_fixture_stage_name(const json & clip, const std::string & stage_name) {
    if (clip.at("stages").contains(stage_name)) {
        return stage_name;
    }
    if (stage_name == "block0_out" && clip.at("stages").contains("encoder_block0")) {
        return "encoder_block0";
    }
    throw std::runtime_error("stage not found in fixture manifest: " + stage_name);
}

static std::string length_key_for_stage(const json & clip, const std::string & stage_name) {
    if (stage_name == "mel") {
        return "mel_len";
    }
    if (stage_name == "subsampling_out") {
        return "subsampling_len";
    }

    const std::string direct = stage_name + "_len";
    if (clip.contains(direct)) {
        return direct;
    }
    if (stage_name == "block0_out" && clip.contains("encoder_block0_len")) {
        return "encoder_block0_len";
    }
    return "";
}

static stage_tolerance tolerance_for_stage(const std::string & stage_name, const parity_params & params) {
    stage_tolerance out{params.atol, params.rtol};

    // The custom C++ frontend tracks the native HF path closely enough for exact greedy decoding,
    // but it still differs slightly from torch.stft-based reference features.
    if (stage_name == "mel" || stage_name == "first_step_logits") {
        out.atol = std::max(out.atol, 5e-4f);
    }

    return out;
}

static bool compare_numeric(
        const std::string & name,
        const std::vector<float> & actual,
        const std::vector<float> & reference,
        float atol,
        float rtol) {
    if (actual.size() != reference.size()) {
        fprintf(stderr, "mismatch in %s element count: actual=%zu reference=%zu\n", name.c_str(), actual.size(), reference.size());
        return false;
    }

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t worst_index = 0;
    bool ok = true;

    for (size_t i = 0; i < actual.size(); ++i) {
        const float abs_err = std::fabs(actual[i] - reference[i]);
        const float tol = atol + rtol * std::fabs(reference[i]);
        const float rel_err = std::fabs(reference[i]) > 0.0f ? abs_err / std::fabs(reference[i]) : abs_err;
        if (abs_err > max_abs) {
            max_abs = abs_err;
            max_rel = rel_err;
            worst_index = i;
        }
        if (abs_err > tol) {
            ok = false;
        }
    }

    if (!ok) {
        fprintf(
                stderr,
                "FAILED %s: max_abs=%g max_rel=%g worst_index=%zu actual=%g reference=%g\n",
                name.c_str(),
                max_abs,
                max_rel,
                worst_index,
                actual[worst_index],
                reference[worst_index]);
        return false;
    }

    fprintf(stderr, "OK %s: max_abs=%g max_rel=%g\n", name.c_str(), max_abs, max_rel);
    return true;
}

static bool compare_matrix_stage(
        const std::string & fixtures_dir,
        const json & clip,
        const std::string & stage_name,
        const cohere::matrix_output & actual,
        int actual_len,
        float atol,
        float rtol) {
    const std::string fixture_stage_name = resolve_fixture_stage_name(clip, stage_name);
    const json & stage = clip.at("stages").at(fixture_stage_name);
    const std::vector<int> shape = stage.at("shape").get<std::vector<int>>();
    if (shape.size() != 2) {
        fprintf(stderr, "invalid shape rank for %s\n", stage_name.c_str());
        return false;
    }

    if (stage.contains("storage") && stage.at("storage").get<std::string>() != "row_major") {
        fprintf(stderr, "unsupported storage order for %s\n", stage_name.c_str());
        return false;
    }

    const int rows = shape[0];
    const int cols = shape[1];
    if (actual.rows != rows || actual.cols != cols) {
        fprintf(
                stderr,
                "FAILED %s shape: actual=[%d,%d] reference=[%d,%d]\n",
                stage_name.c_str(),
                actual.rows,
                actual.cols,
                rows,
                cols);
        return false;
    }

    const std::string len_key = length_key_for_stage(clip, stage_name);
    if (!len_key.empty() && actual_len != clip.at(len_key).get<int>()) {
        fprintf(
                stderr,
                "FAILED %s length: actual=%d reference=%d\n",
                stage_name.c_str(),
                actual_len,
                clip.at(len_key).get<int>());
        return false;
    }

    const std::vector<float> reference = read_f32_file(fixtures_dir + "/" + stage.at("path").get<std::string>(), size_t(rows) * size_t(cols));
    if (actual.data.size() != reference.size()) {
        fprintf(stderr, "FAILED %s element count: actual=%zu reference=%zu\n", stage_name.c_str(), actual.data.size(), reference.size());
        return false;
    }

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int worst_row = 0;
    int worst_col = 0;
    bool ok = true;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const float actual_value = row_major_at(actual.data, rows, cols, row, col);
            const float reference_value = row_major_at(reference, rows, cols, row, col);
            const float abs_err = std::fabs(actual_value - reference_value);
            const float tol = atol + rtol * std::fabs(reference_value);
            const float rel_err = std::fabs(reference_value) > 0.0f ? abs_err / std::fabs(reference_value) : abs_err;
            if (abs_err > max_abs) {
                max_abs = abs_err;
                max_rel = rel_err;
                worst_row = row;
                worst_col = col;
            }
            if (abs_err > tol) {
                ok = false;
            }
        }
    }

    if (!ok) {
        fprintf(
                stderr,
                "FAILED %s: max_abs=%g max_rel=%g worst_rc=(%d,%d) actual=%g reference=%g\n",
                stage_name.c_str(),
                max_abs,
                max_rel,
                worst_row,
                worst_col,
                row_major_at(actual.data, rows, cols, worst_row, worst_col),
                row_major_at(reference, rows, cols, worst_row, worst_col));
        return false;
    }

    fprintf(stderr, "OK %s: max_abs=%g max_rel=%g\n", stage_name.c_str(), max_abs, max_rel);
    return true;
}

static bool compare_vector_stage(
        const std::string & fixtures_dir,
        const json & clip,
        const std::string & stage_name,
        const std::vector<float> & actual,
        float atol,
        float rtol) {
    const json & stage = clip.at("stages").at(stage_name);
    const std::vector<int> shape = stage.at("shape").get<std::vector<int>>();
    const size_t count = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    const std::vector<float> reference = read_f32_file(fixtures_dir + "/" + stage.at("path").get<std::string>(), count);
    return compare_numeric(stage_name, actual, reference, atol, rtol);
}

static bool compare_i32_stage(const std::string & stage_name, const std::vector<int32_t> & actual, const std::vector<int32_t> & reference) {
    if (actual != reference) {
        fprintf(stderr, "FAILED %s exact match\n", stage_name.c_str());
        fprintf(stderr, "actual   :");
        for (size_t i = 0; i < actual.size(); ++i) {
            fprintf(stderr, " %d", actual[i]);
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "reference:");
        for (size_t i = 0; i < reference.size(); ++i) {
            fprintf(stderr, " %d", reference[i]);
        }
        fprintf(stderr, "\n");
        return false;
    }
    fprintf(stderr, "OK %s exact match\n", stage_name.c_str());
    return true;
}

static bool compare_text_stage(const std::string & actual, const std::string & reference) {
    if (actual != reference) {
        fprintf(stderr, "FAILED text exact match\n");
        fprintf(stderr, "actual   : %s\n", actual.c_str());
        fprintf(stderr, "reference: %s\n", reference.c_str());
        return false;
    }
    fprintf(stderr, "OK text exact match\n");
    return true;
}

int main(int argc, char ** argv) {
    parity_params params;
    if (!parse_args(argc, argv, params)) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        const json manifest = load_manifest(params.fixtures);
        const json & clip = find_clip(manifest, params.clip);

        std::vector<float> pcmf32;
        std::vector<std::vector<float>> pcmf32s;
        const std::string audio_path = params.fixtures + "/" + clip.at("audio").get<std::string>();
        if (!read_audio_data(audio_path, pcmf32, pcmf32s, false)) {
            throw std::runtime_error("failed to read fixture audio");
        }

        cohere::model model;
        std::string error;
        if (!cohere::load_model(params.model, model, error)) {
            throw std::runtime_error(error);
        }

        cohere::transcribe_params tparams;
        tparams.language = clip.value("language", std::string("en"));
        tparams.n_threads = 1;
        tparams.max_new_tokens = 256;
        tparams.punctuation = true;

        cohere::debug_outputs outputs;
        if (!cohere::collect_debug_outputs(model, pcmf32, tparams, outputs, error)) {
            cohere::free_model(model);
            throw std::runtime_error(error);
        }

        bool ok = true;
        bool ran_stage = false;
        const std::string stage = canonical_stage_name(params.stage);
        if (stage == "mel" || stage == "all") {
            ran_stage = true;
            const stage_tolerance tol = tolerance_for_stage("mel", params);
            ok = compare_matrix_stage(params.fixtures, clip, "mel", outputs.mel, outputs.mel_len, tol.atol, tol.rtol) && ok;
        }
        if (stage == "subsampling_out" || stage == "all") {
            ran_stage = true;
            const stage_tolerance tol = tolerance_for_stage("subsampling_out", params);
            ok = compare_matrix_stage(params.fixtures, clip, "subsampling_out", outputs.subsampling_out, outputs.subsampling_len, tol.atol, tol.rtol) && ok;
        }
        if (stage == "block0_after_ff1" || stage == "all") {
            ran_stage = true;
            const stage_tolerance tol = tolerance_for_stage("block0_after_ff1", params);
            ok = compare_matrix_stage(
                    params.fixtures,
                    clip,
                    "block0_after_ff1",
                    outputs.block0_after_ff1,
                    outputs.block0_after_ff1_len,
                    tol.atol,
                    tol.rtol) && ok;
        }
        if (stage == "block0_after_attn" || stage == "all") {
            ran_stage = true;
            const stage_tolerance tol = tolerance_for_stage("block0_after_attn", params);
            ok = compare_matrix_stage(
                    params.fixtures,
                    clip,
                    "block0_after_attn",
                    outputs.block0_after_attn,
                    outputs.block0_after_attn_len,
                    tol.atol,
                    tol.rtol) && ok;
        }
        if (stage == "block0_after_conv" || stage == "all") {
            ran_stage = true;
            const stage_tolerance tol = tolerance_for_stage("block0_after_conv", params);
            ok = compare_matrix_stage(
                    params.fixtures,
                    clip,
                    "block0_after_conv",
                    outputs.block0_after_conv,
                    outputs.block0_after_conv_len,
                    tol.atol,
                    tol.rtol) && ok;
        }
        if (stage == "block0_out" || stage == "all") {
            ran_stage = true;
            const stage_tolerance tol = tolerance_for_stage("block0_out", params);
            ok = compare_matrix_stage(
                    params.fixtures,
                    clip,
                    "block0_out",
                    outputs.block0_out,
                    outputs.block0_out_len,
                    tol.atol,
                    tol.rtol) && ok;
        }
        if (stage == "encoder_out" || stage == "all") {
            ran_stage = true;
            const stage_tolerance tol = tolerance_for_stage("encoder_out", params);
            ok = compare_matrix_stage(params.fixtures, clip, "encoder_out", outputs.encoder_out, outputs.encoder_out_len, tol.atol, tol.rtol) && ok;
        }
        if (stage == "encoder_projected" || stage == "all") {
            ran_stage = true;
            const stage_tolerance tol = tolerance_for_stage("encoder_projected", params);
            ok = compare_matrix_stage(
                    params.fixtures,
                    clip,
                    "encoder_projected",
                    outputs.encoder_projected,
                    outputs.encoder_projected_len,
                    tol.atol,
                    tol.rtol) && ok;
        }
        if (stage == "prompt_ids" || stage == "all") {
            ran_stage = true;
            ok = compare_i32_stage("prompt_ids", outputs.prompt_ids, clip.at("prompt_ids").get<std::vector<int32_t>>()) && ok;
        }
        if (stage == "first_step_logits" || stage == "all") {
            ran_stage = true;
            const stage_tolerance tol = tolerance_for_stage("first_step_logits", params);
            ok = compare_vector_stage(params.fixtures, clip, "first_step_logits", outputs.first_step_logits, tol.atol, tol.rtol) && ok;
        }
        if (stage == "greedy_ids" || stage == "all") {
            ran_stage = true;
            ok = compare_i32_stage("greedy_ids", outputs.greedy_ids, clip.at("greedy_ids").get<std::vector<int32_t>>()) && ok;
        }
        if (stage == "text" || stage == "all") {
            ran_stage = true;
            ok = compare_text_stage(outputs.text, clip.at("text").get<std::string>()) && ok;
        }

        if (!ran_stage) {
            cohere::free_model(model);
            throw std::runtime_error("unknown stage: " + params.stage);
        }

        cohere::free_model(model);
        return ok ? 0 : 1;
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
