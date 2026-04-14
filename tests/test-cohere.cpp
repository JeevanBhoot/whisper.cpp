#include "cohere.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

static cohere::tokenizer make_tokenizer() {
    cohere::tokenizer tokenizer;
    tokenizer.id_to_piece = {
        "<|endoftext|>",
        "<|startofcontext|>",
        "<|startoftranscript|>",
        "<|emo:undefined|>",
        "<|pnc|>",
        "<|nopnc|>",
        "<|noitn|>",
        "<|notimestamp|>",
        "<|nodiarize|>",
        "<|en|>",
        "▁hello",
        "▁world",
        "<0xC3>",
        "<0xA9>",
        "<unk>",
    };

    const std::vector<std::string> specials = {
        "<|endoftext|>",
        "<|startofcontext|>",
        "<|startoftranscript|>",
        "<|emo:undefined|>",
        "<|pnc|>",
        "<|nopnc|>",
        "<|noitn|>",
        "<|notimestamp|>",
        "<|nodiarize|>",
    };

    for (size_t i = 0; i < specials.size(); ++i) {
        tokenizer.special_token_ids[specials[i]] = (int32_t) i;
        tokenizer.special_ids.insert((int32_t) i);
    }

    tokenizer.language_token_ids["en"] = 9;
    tokenizer.special_ids.insert(9);
    tokenizer.supported_languages.push_back("en");
    return tokenizer;
}

static void test_prompt_building() {
    const cohere::tokenizer tokenizer = make_tokenizer();
    const std::vector<int32_t> prompt = tokenizer.build_prompt("en", true);
    const std::vector<int32_t> expected = {1, 2, 3, 9, 9, 4, 6, 7, 8};
    assert(prompt == expected);

    const std::vector<int32_t> prompt_no_punc = tokenizer.build_prompt("en", false);
    assert(prompt_no_punc[5] == 5);
}

static void test_rel_shift_reference() {
    const std::vector<float> input = {
        1, 2, 3,
        4, 5, 6,
    };
    const std::vector<float> shifted = cohere::rel_shift_reference(input, 1, 2, 3);
    const std::vector<float> expected = {
        2, 3, 0,
        4, 5, 6,
    };
    assert(shifted == expected);
}

static void test_subsampling_length() {
    assert(cohere::conv_subsampling_output_length(1) == 1);
    assert(cohere::conv_subsampling_output_length(2) == 1);
    assert(cohere::conv_subsampling_output_length(3) == 1);
    assert(cohere::conv_subsampling_output_length(4) == 1);
    assert(cohere::conv_subsampling_output_length(8) == 1);
    assert(cohere::conv_subsampling_output_length(16) == 2);
    assert(cohere::conv_subsampling_output_length(21) == 3);
}

static void test_detokenizer() {
    const cohere::tokenizer tokenizer = make_tokenizer();

    const std::string basic = tokenizer.detokenize({1, 10, 11, 0});
    assert(basic == "hello world");

    const std::string bytes = tokenizer.detokenize({10, 12, 13});
    assert(bytes == u8"helloé");

    const std::string invalid = tokenizer.detokenize({12});
    assert(invalid == u8"\uFFFD");
}

static void test_long_form_transcribe() {
    cohere::model model;
    std::string error;
    assert(cohere::load_model(COHERE_TEST_MODEL_PATH, model, error));
    assert(std::fabs(model.overlap_chunk_second - 5.0f) < 1e-6f);
    assert(model.min_energy_window_samples == 1600);

    const int32_t sample_rate = model.frontend.sample_rate;
    float duration_s = model.max_audio_clip_s + 5.0f;
    if (duration_s < 40.0f) {
        duration_s = 40.0f;
    }

    const int32_t n_samples = (int32_t) (duration_s * sample_rate);
    const float kPi = 3.14159265358979323846f;
    std::vector<float> pcmf32(size_t(n_samples), 0.0f);
    for (int32_t i = 0; i < n_samples; ++i) {
        const float t = float(i) / float(sample_rate);
        const float amp = (t >= 31.0f && t <= 33.0f) ? 0.0f : 0.1f;
        pcmf32[size_t(i)] = amp * std::sin(2.0f * kPi * 220.0f * t);
    }

    cohere::transcribe_params params;
    params.language = "en";
    params.n_threads = 1;
    params.max_new_tokens = 16;
    params.punctuation = true;

    std::string text;
    assert(cohere::transcribe(model, pcmf32, params, text, error));
    assert(error.empty());

    cohere::free_model(model);
}

int main() {
    test_prompt_building();
    test_rel_shift_reference();
    test_subsampling_length();
    test_detokenizer();
    test_long_form_transcribe();
    return 0;
}
