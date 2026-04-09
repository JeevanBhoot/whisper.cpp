#include "cohere.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
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

int main() {
    test_prompt_building();
    test_rel_shift_reference();
    test_subsampling_length();
    test_detokenizer();
    return 0;
}
