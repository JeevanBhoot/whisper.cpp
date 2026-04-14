#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "common-ggml.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static const char * GGUF_ARCH_KEY   = "general.architecture";
static const char * GGUF_ARCH_VALUE = "cohere-transcribe";

struct options {
    std::string in_path;
    std::string out_path;
    ggml_ftype  ftype = GGML_FTYPE_MOSTLY_Q4_K;
    int         n_threads = 0; // reserved (single-threaded quant in this initial version)
    bool        dry_run = false;
    std::vector<std::regex> include;
    std::vector<std::regex> exclude;
};

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s <in.gguf> <out.gguf> <type> [--threads N] [--dry-run]\\n"
        "       [--include REGEX]... [--exclude REGEX]...\\n\n",
        argv0);
    fprintf(stderr, "supported types (or numeric):\n");
    ggml_print_ftypes(stderr);
}

static bool parse_args(int argc, char ** argv, options & opt) {
    if (argc < 4) {
        print_usage(argv[0]);
        return false;
    }

    opt.in_path  = argv[1];
    opt.out_path = argv[2];
    opt.ftype    = ggml_parse_ftype(argv[3]);
    if (opt.ftype == GGML_FTYPE_UNKNOWN) {
        fprintf(stderr, "%s: unknown quantization type %s\n", __func__, argv[3]);
        return false;
    }

    // defaults (may be overridden by flags)
    const char * default_includes[] = {
        "^encoder\\.pre_encode\\.out\\.weight$",
        "^encoder\\.layers\\.\\d+\\.(feed_forward[12]\\.linear[12]|self_attn\\.linear_(q|k|v|out)|self_attn\\.linear_pos)\\.weight$",
        "^decoder\\.layers\\.\\d+\\.(self_attn\\.(query|key|value|out)|cross_attn\\.(query|key|value|out)|feed_forward\\.(dense_in|dense_out))\\.weight$",
        "^encoder_decoder_proj\\.weight$",
    };
    const char * default_excludes[] = {
        "\\.(bias)$",
        ".*norm.*\\.(weight|bias)$",
        "^encoder\\.layers\\.\\d+\\.self_attn\\.pos_bias_(u|v)$",
        "^encoder\\.pre_encode\\.conv.*\\.weight$",
        "^decoder\\.embedding\\.token_embedding\\.weight$",
    };
    for (const char * s : default_includes) opt.include.emplace_back(s);
    for (const char * s : default_excludes) opt.exclude.emplace_back(s);

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            opt.n_threads = std::max(0, atoi(argv[++i]));
        } else if (arg == "--dry-run") {
            opt.dry_run = true;
        } else if (arg == "--include" && i + 1 < argc) {
            opt.include.emplace_back(argv[++i]);
        } else if (arg == "--exclude" && i + 1 < argc) {
            opt.exclude.emplace_back(argv[++i]);
        } else {
            fprintf(stderr, "%s: unknown or incomplete option %s\n", __func__, arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

static ggml_type target_qtype_from_ftype(ggml_ftype ftype) {
    switch (ftype) {
        case GGML_FTYPE_MOSTLY_Q4_K: return GGML_TYPE_Q4_K;
        case GGML_FTYPE_MOSTLY_Q5_K: return GGML_TYPE_Q5_K;
        case GGML_FTYPE_MOSTLY_Q6_K: return GGML_TYPE_Q6_K;
        case GGML_FTYPE_MOSTLY_Q8_0: return GGML_TYPE_Q8_0;
        default: return GGML_TYPE_COUNT;
    }
}

static bool name_matches_any(const std::string & name, const std::vector<std::regex> & patterns) {
    for (const auto & re : patterns) {
        if (std::regex_match(name, re)) return true;
    }
    return false;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    options opt;
    if (!parse_args(argc, argv, opt)) {
        return 2;
    }

    const ggml_type qtype = target_qtype_from_ftype(opt.ftype);
    if (!ggml_is_quantized(qtype)) {
        fprintf(stderr, "error: unsupported ftype -> qtype mapping for this tool\n");
        return 2;
    }

    struct ggml_context * weights = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &weights,
    };

    struct gguf_context * ctx_in = gguf_init_from_file(opt.in_path.c_str(), params);
    if (!ctx_in || !weights) {
        fprintf(stderr, "error: failed to load GGUF %s\n", opt.in_path.c_str());
        return 1;
    }

    const int64_t k_arch = gguf_find_key(ctx_in, GGUF_ARCH_KEY);
    if (k_arch < 0) {
        fprintf(stderr, "error: missing GGUF key %s\n", GGUF_ARCH_KEY);
        return 1;
    }
    const char * arch = gguf_get_val_str(ctx_in, k_arch);
    if (std::string(arch) != GGUF_ARCH_VALUE) {
        fprintf(stderr, "error: unsupported architecture %s (expected %s)\n", arch, GGUF_ARCH_VALUE);
        return 1;
    }

    // Prepare output GGUF
    std::unique_ptr<gguf_context, void(*)(gguf_context*)> ctx_out(gguf_init_empty(), gguf_free);
    if (!ctx_out) {
        fprintf(stderr, "error: failed to create output GGUF context\n");
        return 1;
    }
    gguf_set_kv(ctx_out.get(), ctx_in);
    gguf_set_val_u32(ctx_out.get(), "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out.get(), "general.file_type", (uint32_t)opt.ftype);

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    // keep quantized blobs alive until write
    std::map<std::string, std::vector<uint8_t>> qdata;

    const int64_t n_tensors = gguf_get_n_tensors(ctx_in);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * tname = gguf_get_tensor_name(ctx_in, i);
        ggml_tensor * tin   = ggml_get_tensor(weights, tname);
        if (!tin) {
            fprintf(stderr, "error: missing tensor body for %s\n", tname);
            return 1;
        }

        // Register tensor metadata in output
        gguf_add_tensor(ctx_out.get(), tin);

        const std::string name(tname);
        const int n_dims = ggml_n_dims(tin);
        const size_t org_bytes = ggml_nbytes(tin);
        total_size_org += org_bytes;

        bool allow_quant = (n_dims == 2) && name_matches_any(name, opt.include) && !name_matches_any(name, opt.exclude);
        bool is_floating_src = (tin->type == GGML_TYPE_F16 || tin->type == GGML_TYPE_F32);

        if (allow_quant && is_floating_src && !opt.dry_run) {
            const int64_t n0 = tin->ne[0];
            const int64_t n1 = tin->ne[1];

            std::vector<float> f32;
            f32.resize(n0 * n1);

            if (tin->type == GGML_TYPE_F32 && ggml_is_contiguous(tin)) {
                std::memcpy(f32.data(), tin->data, f32.size() * sizeof(float));
            } else if (tin->type == GGML_TYPE_F16 && ggml_is_contiguous(tin)) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *)tin->data, f32.data(), n0 * n1);
            } else {
                // fallback per element
                for (int64_t j = 0; j < n1; ++j) {
                    for (int64_t i0 = 0; i0 < n0; ++i0) {
                        f32[size_t(i0) + size_t(n0) * size_t(j)] = ggml_get_f32_nd(tin, i0, j, 0, 0);
                    }
                }
            }

            const size_t row_size_q = ggml_row_size(qtype, n0);
            std::vector<uint8_t> & dst = qdata[name];
            dst.resize(row_size_q * n1);

            // For now, single-call chunk quantization (can be parallelized later if needed)
            const size_t qbytes = ggml_quantize_chunk(qtype, f32.data(), dst.data(), /*start_row=*/0, /*nrows=*/n1, /*n_per_row=*/n0, /*imatrix=*/nullptr);

            gguf_set_tensor_type(ctx_out.get(), tname, qtype);
            gguf_set_tensor_data(ctx_out.get(), tname, dst.data());

            total_size_new += qbytes;
            printf("%-64s - [%5lld, %5lld]  %6s  ->  %6s  | size = %8.2f MB -> %8.2f MB\n",
                   tname,
                   (long long)tin->ne[0], (long long)tin->ne[1],
                   ggml_type_name(tin->type), ggml_type_name(qtype),
                   org_bytes/1024.0/1024.0, qbytes/1024.0/1024.0);
        } else {
            // keep original
            gguf_set_tensor_type(ctx_out.get(), tname, tin->type);
            if (!opt.dry_run) {
                gguf_set_tensor_data(ctx_out.get(), tname, tin->data);
            }
            total_size_new += org_bytes;
            printf("%-64s - %6s (kept) | size = %8.2f MB\n",
                   tname,
                   ggml_type_name(tin->type),
                   org_bytes/1024.0/1024.0);
        }
    }

    printf("\nmodel size  = %8.2f MB\n", total_size_org/1024.0/1024.0);
    printf("quant size  = %8.2f MB | ftype = %d (%s)\n\n",
           total_size_new/1024.0/1024.0, (int)opt.ftype, ggml_type_name(qtype));

    if (!opt.dry_run) {
        if (!gguf_write_to_file(ctx_out.get(), opt.out_path.c_str(), /*only_meta=*/false)) {
            fprintf(stderr, "error: failed to write output GGUF %s\n", opt.out_path.c_str());
            return 1;
        }
        printf("wrote %s\n", opt.out_path.c_str());
    } else {
        printf("dry-run: no file written\n");
    }

    return 0;
}
