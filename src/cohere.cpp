#include "cohere.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>

#if defined(__has_include)
#if __has_include(<Accelerate/Accelerate.h>)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define COHERE_HAVE_CBLAS 1
#elif __has_include(<cblas.h>)
#include <cblas.h>
#define COHERE_HAVE_CBLAS 1
#endif
#endif

#ifndef COHERE_HAVE_CBLAS
#define COHERE_HAVE_CBLAS 0
#endif

namespace cohere {
namespace {

static const char * GGUF_ARCH_KEY = "general.architecture";
static const char * GGUF_ARCH_VALUE = "cohere-transcribe";
static const char * GGUF_PREFIX = "cohere_transcribe.";
static const float  LAYER_NORM_EPS = 1e-5f;
static const size_t SCRATCH_SIZE = 512ull * 1024ull * 1024ull;
static const size_t COHERE_MAX_NODES = 16384;

struct gguf_context_deleter {
    void operator()(struct gguf_context * ctx) const {
        if (ctx) {
            gguf_free(ctx);
        }
    }
};

struct ggml_context_deleter {
    void operator()(struct ggml_context * ctx) const {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

typedef std::unique_ptr<struct gguf_context, gguf_context_deleter> gguf_ptr;
typedef std::unique_ptr<struct ggml_context, ggml_context_deleter> ggml_ptr;

struct tensor2d {
    int32_t n0 = 0;
    int32_t n1 = 0;
    std::vector<float> data;

    tensor2d() {
    }

    tensor2d(int32_t n0_, int32_t n1_) : n0(n0_), n1(n1_), data(size_t(n0_) * size_t(n1_)) {
    }

    inline float & at(int32_t i0, int32_t i1) {
        return data[size_t(i0) + size_t(n0) * size_t(i1)];
    }

    inline const float & at(int32_t i0, int32_t i1) const {
        return data[size_t(i0) + size_t(n0) * size_t(i1)];
    }
};

struct tensor4d {
    int32_t n0 = 0;
    int32_t n1 = 0;
    int32_t n2 = 0;
    int32_t n3 = 0;
    std::vector<float> data;

    tensor4d() {
    }

    tensor4d(int32_t n0_, int32_t n1_, int32_t n2_, int32_t n3_) :
            n0(n0_), n1(n1_), n2(n2_), n3(n3_), data(size_t(n0_) * size_t(n1_) * size_t(n2_) * size_t(n3_)) {
    }

    inline float & at(int32_t i0, int32_t i1, int32_t i2, int32_t i3) {
        return data[size_t(i0) +
                size_t(n0) * (size_t(i1) +
                size_t(n1) * (size_t(i2) +
                size_t(n2) * size_t(i3)))];
    }

    inline const float & at(int32_t i0, int32_t i1, int32_t i2, int32_t i3) const {
        return data[size_t(i0) +
                size_t(n0) * (size_t(i1) +
                size_t(n1) * (size_t(i2) +
                size_t(n2) * size_t(i3)))];
    }
};

struct cross_kv_cache {
    tensor2d key;
    tensor2d value;
};

struct self_kv_cache {
    tensor2d key;
    tensor2d value;
    int32_t length = 0;
};

struct linear_workspace {
    ggml_ptr ctx;
};

struct backend_model_data;
struct backend_state_data;

struct inference_runtime {
    linear_workspace linear_ws;
    std::map<int64_t, tensor2d> rel_pos_cache;
    std::vector<uint8_t> vec_dot_input_buffer;
    std::vector<ggml_backend_buffer_t> temp_input_buffers;
    const backend_model_data * backend_model = nullptr;
    backend_state_data * backend_state = nullptr;
};

struct cohere_sched {
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;
};

struct backend_kv_cache {
    std::vector<uint8_t> ctx_buf;
    ggml_backend_buffer_t buffer = nullptr;
    struct ggml_tensor * k = nullptr;
    struct ggml_tensor * v = nullptr;
    int32_t size = 0;
    int32_t n = 0;
};

struct backend_model_data {
    context_params params;
    std::string path_model;
    bool use_gpu_active = false;
    std::string backend_name;
    ggml_backend_dev_t device = nullptr;
    int32_t max_valid_mel_frames = 0;
    int32_t max_mel_frames = 0;
    int32_t max_encoder_valid_length = 0;
    int32_t max_encoder_length = 0;
    int32_t cross_hidden_size = 0;
    std::vector<struct ggml_context *> ctxs;
    std::map<std::string, struct ggml_tensor *> tensors;
    std::vector<ggml_backend_buffer_t> buffers;
};

struct backend_state_data {
    context_params params;
    bool legacy_cpu = false;
    std::string backend_name;

    std::vector<ggml_backend_t> backends;
    cohere_sched sched_conv;
    cohere_sched sched_encode;
    cohere_sched sched_cross;
    cohere_sched sched_decode;

    backend_kv_cache kv_self;
    backend_kv_cache kv_cross;

    std::vector<uint8_t> tensor_ctx_buf;
    ggml_backend_buffer_t tensor_buffer = nullptr;
    struct ggml_tensor * conv_out = nullptr;
    struct ggml_tensor * encoder_out = nullptr;
    struct ggml_tensor * encoder_rel_pos = nullptr;
    struct ggml_tensor * decoder_pos = nullptr;

    int32_t encoder_valid_length = 0;
    int32_t encoder_length = 0;
    int32_t self_kv_length = 0;

    std::vector<float> mel_input;
    std::vector<float> conv_mask0;
    std::vector<float> conv_mask1;
    std::vector<float> conv_mask2;
    std::vector<float> encoder_mask;
    std::vector<float> decode_mask;
    std::vector<int32_t> rel_shift_indices;
    std::vector<float> logits;
};

static bool runtime_uses_backend(const inference_runtime * runtime);
static struct ggml_tensor * maybe_backend_tensor(inference_runtime * runtime, const struct ggml_tensor * tensor);
static bool ggml_graph_compute_helper(
        ggml_backend_sched_t sched,
        struct ggml_cgraph * graph,
        int32_t n_threads,
        bool sched_reset = true);

struct fft_trig_cache {
    int32_t n = 0;
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;
};

static std::string format(const char * fmt, ...) {
    char buf[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    return std::string(buf);
}

static std::string gguf_key(const std::string & suffix) {
    return std::string(GGUF_PREFIX) + suffix;
}

static ggml_ptr make_compute_ctx(size_t mem_size = SCRATCH_SIZE, bool no_alloc = false) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ no_alloc,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to allocate ggml compute context");
    }

    return ggml_ptr(ctx);
}

static int64_t rel_pos_cache_key(int32_t d_model, int32_t length) {
    return (int64_t(d_model) << 32) | uint32_t(length);
}

static int32_t choose_blas_thread_limit(int32_t n_threads) {
    if (n_threads <= 1) {
        return 1;
    }

    return std::max(1, std::min(4, n_threads / 2));
}

static void configure_blas_threads_for_inference(int32_t n_threads) {
#if defined(__APPLE__) && COHERE_HAVE_CBLAS
    struct blas_thread_state {
        bool configured_by_runtime = false;
        int32_t last_limit = -1;
    };

    static blas_thread_state state;
    const int32_t limit = choose_blas_thread_limit(n_threads);
    const char * env = std::getenv("VECLIB_MAXIMUM_THREADS");

    if (env != nullptr && !state.configured_by_runtime) {
        return;
    }

    if (state.configured_by_runtime && state.last_limit == limit) {
        return;
    }

    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d", limit);
    setenv("VECLIB_MAXIMUM_THREADS", buf, 1);
    state.configured_by_runtime = true;
    state.last_limit = limit;
#else
    (void) n_threads;
#endif
}

static const fft_trig_cache & get_fft_trig_cache(int32_t n) {
    static std::map<int32_t, fft_trig_cache> caches;
    const std::map<int32_t, fft_trig_cache>::iterator it = caches.find(n);
    if (it != caches.end()) {
        return it->second;
    }

    fft_trig_cache cache;
    cache.n = n;
    cache.sin_vals.resize(size_t(n));
    cache.cos_vals.resize(size_t(n));
    for (int32_t i = 0; i < n; ++i) {
        const double theta = (2.0 * M_PI * double(i)) / double(n);
        cache.sin_vals[size_t(i)] = std::sin(theta);
        cache.cos_vals[size_t(i)] = std::cos(theta);
    }

    return caches.emplace(n, std::move(cache)).first->second;
}

static linear_workspace make_linear_workspace(inference_runtime * runtime) {
    linear_workspace ws;
    ws.ctx = make_compute_ctx(SCRATCH_SIZE, runtime_uses_backend(runtime));
    return ws;
}

static struct ggml_context * begin_linear_workspace(inference_runtime * runtime) {
    if (runtime == nullptr) {
        return nullptr;
    }
    if (!runtime->linear_ws.ctx) {
        runtime->linear_ws = make_linear_workspace(runtime);
    } else {
        ggml_reset(runtime->linear_ws.ctx.get());
    }
    return runtime->linear_ws.ctx.get();
}

static cohere_sched * get_runtime_sched(inference_runtime * runtime) {
    if (!runtime_uses_backend(runtime)) {
        return nullptr;
    }

    return &runtime->backend_state->sched_encode;
}

static void ensure_runtime_sched(inference_runtime * runtime, cohere_sched & sched) {
    if (sched.sched != nullptr) {
        return;
    }

    sched.sched = ggml_backend_sched_new(
            runtime->backend_state->backends.data(),
            nullptr,
            runtime->backend_state->backends.size(),
            COHERE_MAX_NODES,
            false,
            true);
    if (sched.sched == nullptr) {
        throw std::runtime_error("failed to create Cohere backend scheduler");
    }

    sched.meta.resize(ggml_tensor_overhead() * COHERE_MAX_NODES + ggml_graph_overhead());
}

static void run_graph(struct ggml_context * ctx, struct ggml_tensor * output, int32_t n_threads, inference_runtime * runtime = nullptr) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);

    auto cleanup_inputs = [&](void) {
        if (runtime == nullptr) {
            return;
        }
        for (ggml_backend_buffer_t buffer : runtime->temp_input_buffers) {
            if (buffer != nullptr) {
                ggml_backend_buffer_free(buffer);
            }
        }
        runtime->temp_input_buffers.clear();
    };

    cohere_sched * sched = get_runtime_sched(runtime);
    if (sched == nullptr) {
        const bool ok = ggml_graph_compute_with_ctx(ctx, gf, n_threads) == GGML_STATUS_SUCCESS;
        cleanup_inputs();
        if (!ok) {
            throw std::runtime_error("ggml graph execution failed");
        }
        return;
    }

    ensure_runtime_sched(runtime, *sched);
    ggml_backend_sched_reset(sched->sched);
    if (!ggml_backend_sched_alloc_graph(sched->sched, gf)) {
        cleanup_inputs();
        throw std::runtime_error("failed to allocate Cohere backend graph");
    }
    const bool ok = ggml_graph_compute_helper(sched->sched, gf, n_threads, false);
    cleanup_inputs();
    if (!ok) {
        throw std::runtime_error("ggml backend graph execution failed");
    }
}

static void run_graph_outputs(
        struct ggml_context * ctx,
        std::initializer_list<struct ggml_tensor *> outputs,
        int32_t n_threads,
        inference_runtime * runtime = nullptr) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    for (struct ggml_tensor * output : outputs) {
        ggml_build_forward_expand(gf, output);
    }

    auto cleanup_inputs = [&](void) {
        if (runtime == nullptr) {
            return;
        }
        for (ggml_backend_buffer_t buffer : runtime->temp_input_buffers) {
            if (buffer != nullptr) {
                ggml_backend_buffer_free(buffer);
            }
        }
        runtime->temp_input_buffers.clear();
    };

    cohere_sched * sched = get_runtime_sched(runtime);
    if (sched == nullptr) {
        const bool ok = ggml_graph_compute_with_ctx(ctx, gf, n_threads) == GGML_STATUS_SUCCESS;
        cleanup_inputs();
        if (!ok) {
            throw std::runtime_error("ggml graph execution failed");
        }
        return;
    }

    ensure_runtime_sched(runtime, *sched);
    ggml_backend_sched_reset(sched->sched);
    if (!ggml_backend_sched_alloc_graph(sched->sched, gf)) {
        cleanup_inputs();
        throw std::runtime_error("failed to allocate Cohere backend graph");
    }
    const bool ok = ggml_graph_compute_helper(sched->sched, gf, n_threads, false);
    cleanup_inputs();
    if (!ok) {
        throw std::runtime_error("ggml backend graph execution failed");
    }
}

static bool ggml_graph_compute_helper(
        ggml_backend_sched_t sched,
        struct ggml_cgraph * graph,
        int32_t n_threads,
        bool sched_reset) {
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;

        ggml_backend_set_n_threads_t fn_set_n_threads = reg
                ? (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads")
                : nullptr;
        if (fn_set_n_threads != nullptr) {
            fn_set_n_threads(backend, n_threads);
        }
    }

    const bool ok = ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
    if (!ok || sched_reset) {
        ggml_backend_sched_reset(sched);
    }

    return ok;
}

static size_t cohere_sched_size(const cohere_sched & sched) {
    size_t size = sched.meta.size();
    if (sched.sched == nullptr) {
        return size;
    }

    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched.sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched.sched, i);
        size += ggml_backend_sched_get_buffer_size(sched.sched, backend);
    }

    return size;
}

static bool cohere_sched_graph_init(
        cohere_sched & allocr,
        const std::vector<ggml_backend_t> & backends,
        std::function<struct ggml_cgraph *()> && get_graph) {
    allocr.sched = ggml_backend_sched_new(
            const_cast<ggml_backend_t *>(backends.data()),
            nullptr,
            int(backends.size()),
            COHERE_MAX_NODES,
            false,
            true);
    allocr.meta.resize(ggml_tensor_overhead() * COHERE_MAX_NODES + ggml_graph_overhead());

    if (!ggml_backend_sched_alloc_graph(allocr.sched, get_graph())) {
        return false;
    }

    ggml_backend_sched_reset(allocr.sched);
    return true;
}

static int32_t gguf_get_key_id_or_throw(const struct gguf_context * gguf, const std::string & key) {
    const int64_t kid = gguf_find_key(gguf, key.c_str());
    if (kid < 0) {
        throw std::runtime_error(format("missing GGUF key '%s'", key.c_str()));
    }
    return int32_t(kid);
}

static int32_t gguf_get_i32_or_throw(const struct gguf_context * gguf, const std::string & key) {
    return gguf_get_val_i32(gguf, gguf_get_key_id_or_throw(gguf, key));
}

static float gguf_get_f32_or_throw(const struct gguf_context * gguf, const std::string & key) {
    return gguf_get_val_f32(gguf, gguf_get_key_id_or_throw(gguf, key));
}

static bool gguf_get_bool_or_throw(const struct gguf_context * gguf, const std::string & key) {
    return gguf_get_val_bool(gguf, gguf_get_key_id_or_throw(gguf, key));
}

static std::string gguf_get_str_or_throw(const struct gguf_context * gguf, const std::string & key) {
    return gguf_get_val_str(gguf, gguf_get_key_id_or_throw(gguf, key));
}

static bool gguf_has_key(const struct gguf_context * gguf, const std::string & key) {
    return gguf_find_key(gguf, key.c_str()) >= 0;
}

template <typename T>
static std::vector<T> gguf_get_scalar_array_or_throw(const struct gguf_context * gguf, const std::string & key, enum gguf_type type) {
    const int32_t kid = gguf_get_key_id_or_throw(gguf, key);
    if (gguf_get_kv_type(gguf, kid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gguf, kid) != type) {
        throw std::runtime_error(format("GGUF key '%s' has unexpected type", key.c_str()));
    }

    const size_t n = gguf_get_arr_n(gguf, kid);
    const T * data = reinterpret_cast<const T *>(gguf_get_arr_data(gguf, kid));
    return std::vector<T>(data, data + n);
}

static std::vector<std::string> gguf_get_string_array_or_throw(const struct gguf_context * gguf, const std::string & key) {
    const int32_t kid = gguf_get_key_id_or_throw(gguf, key);
    if (gguf_get_kv_type(gguf, kid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gguf, kid) != GGUF_TYPE_STRING) {
        throw std::runtime_error(format("GGUF key '%s' has unexpected type", key.c_str()));
    }

    const size_t n = gguf_get_arr_n(gguf, kid);
    std::vector<std::string> result;
    result.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        result.push_back(gguf_get_arr_str(gguf, kid, i));
    }

    return result;
}

static struct ggml_tensor * require_tensor(
        const model & model,
        const std::string & name,
        enum ggml_type expected_type = GGML_TYPE_COUNT) {
    const std::map<std::string, struct ggml_tensor *>::const_iterator it = model.tensors.find(name);
    if (it == model.tensors.end() || it->second == nullptr) {
        throw std::runtime_error(format("missing tensor '%s'", name.c_str()));
    }

    if (expected_type != GGML_TYPE_COUNT && it->second->type != expected_type) {
        throw std::runtime_error(format(
                "tensor '%s' has unexpected type %s (expected %s)",
                name.c_str(),
                ggml_type_name(it->second->type),
                ggml_type_name(expected_type)));
    }

    return it->second;
}

static struct ggml_tensor * find_tensor(const model & model, const std::string & name) {
    const std::map<std::string, struct ggml_tensor *>::const_iterator it = model.tensors.find(name);
    return it == model.tensors.end() ? nullptr : it->second;
}

static void expect_tensor_shape(
        const model & model,
        const std::string & name,
        int64_t ne0,
        int64_t ne1 = -1,
        int64_t ne2 = -1,
        int64_t ne3 = -1) {
    const struct ggml_tensor * tensor = require_tensor(model, name);
    if (ne0 >= 0 && tensor->ne[0] != ne0) {
        throw std::runtime_error(format("tensor '%s' has ne[0]=%lld, expected %lld", name.c_str(), (long long) tensor->ne[0], (long long) ne0));
    }
    if (ne1 >= 0 && tensor->ne[1] != ne1) {
        throw std::runtime_error(format("tensor '%s' has ne[1]=%lld, expected %lld", name.c_str(), (long long) tensor->ne[1], (long long) ne1));
    }
    if (ne2 >= 0 && tensor->ne[2] != ne2) {
        throw std::runtime_error(format("tensor '%s' has ne[2]=%lld, expected %lld", name.c_str(), (long long) tensor->ne[2], (long long) ne2));
    }
    if (ne3 >= 0 && tensor->ne[3] != ne3) {
        throw std::runtime_error(format("tensor '%s' has ne[3]=%lld, expected %lld", name.c_str(), (long long) tensor->ne[3], (long long) ne3));
    }
}

static const float * tensor_data_f32(const struct ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
        return nullptr;
    }
    return reinterpret_cast<const float *>(tensor->data);
}

static float tensor_get_f32(const struct ggml_tensor * tensor, int32_t i0, int32_t i1 = 0, int32_t i2 = 0, int32_t i3 = 0);

static void copy_f32_from_tensor(const struct ggml_tensor * tensor, std::vector<float> & out) {
    const bool has_backend_buffer = tensor->buffer != nullptr && !ggml_backend_buffer_is_host(tensor->buffer);
    std::vector<uint8_t> raw;

    if (has_backend_buffer) {
        raw.resize(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    }

    if (tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor)) {
        const void * data = has_backend_buffer ? raw.data() : tensor->data;
        std::memcpy(out.data(), data, out.size() * sizeof(float));
        return;
    }

    if (tensor->type == GGML_TYPE_F16 && ggml_is_contiguous(tensor)) {
        const ggml_fp16_t * data = has_backend_buffer
                ? reinterpret_cast<const ggml_fp16_t *>(raw.data())
                : reinterpret_cast<const ggml_fp16_t *>(tensor->data);
        ggml_fp16_to_fp32_row(data, out.data(), (int64_t) out.size());
        return;
    }

    if (has_backend_buffer) {
        throw std::runtime_error("unsupported non-contiguous backend tensor copy");
    }

    if (ggml_n_dims(tensor) == 2) {
        const int32_t n0 = (int32_t) tensor->ne[0];
        const int32_t n1 = (int32_t) tensor->ne[1];
        for (int32_t j = 0; j < n1; ++j) {
            for (int32_t i = 0; i < n0; ++i) {
                out[size_t(i) + size_t(n0) * size_t(j)] = tensor_get_f32(tensor, i, j, 0, 0);
            }
        }
        return;
    }

    if (ggml_n_dims(tensor) == 4) {
        const int32_t n0 = (int32_t) tensor->ne[0];
        const int32_t n1 = (int32_t) tensor->ne[1];
        const int32_t n2 = (int32_t) tensor->ne[2];
        const int32_t n3 = (int32_t) tensor->ne[3];
        size_t index = 0;
        for (int32_t l = 0; l < n3; ++l) {
            for (int32_t k = 0; k < n2; ++k) {
                for (int32_t j = 0; j < n1; ++j) {
                    for (int32_t i = 0; i < n0; ++i) {
                        out[index++] = tensor_get_f32(tensor, i, j, k, l);
                    }
                }
            }
        }
        return;
    }

    throw std::runtime_error("unsupported tensor rank for float copy");
}

static float tensor_get_f32(const struct ggml_tensor * tensor, int32_t i0, int32_t i1, int32_t i2, int32_t i3) {
    return ggml_get_f32_nd(tensor, i0, i1, i2, i3);
}

static tensor2d tensor_from_ggml_2d(const struct ggml_tensor * tensor) {
    tensor2d out((int32_t) tensor->ne[0], (int32_t) tensor->ne[1]);
    copy_f32_from_tensor(tensor, out.data);
    return out;
}

static tensor4d tensor_from_ggml_4d(const struct ggml_tensor * tensor) {
    tensor4d out((int32_t) tensor->ne[0], (int32_t) tensor->ne[1], (int32_t) tensor->ne[2], (int32_t) tensor->ne[3]);
    copy_f32_from_tensor(tensor, out.data);
    return out;
}

static enum ggml_type matmul_input_type_from_weight(const struct ggml_tensor * weight) {
    if (weight == nullptr) {
        return GGML_TYPE_F32;
    }

    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(weight->type);
    if (traits == nullptr) {
        return GGML_TYPE_F32;
    }

    const enum ggml_type type = traits->vec_dot_type;
    if (type == GGML_TYPE_F16 || type == GGML_TYPE_F32) {
        return type;
    }

    return GGML_TYPE_F32;
}

static struct ggml_tensor * create_input_tensor_2d(
        struct ggml_context * ctx,
        const tensor2d & input,
        enum ggml_type type,
        inference_runtime * runtime = nullptr) {
    struct ggml_tensor * tensor = ggml_new_tensor_2d(ctx, type, input.n0, input.n1);
    if (tensor == nullptr) {
        throw std::runtime_error("failed to allocate ggml input tensor");
    }

    if (runtime_uses_backend(runtime)) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(tensor));
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate host buffer for Cohere input tensor");
        }

        tensor->data = ggml_backend_buffer_get_base(buffer);
        tensor->buffer = buffer;
        if (ggml_backend_buffer_init_tensor(buffer, tensor) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            tensor->buffer = nullptr;
            throw std::runtime_error("failed to initialize Cohere input tensor backend buffer");
        }
        runtime->temp_input_buffers.push_back(buffer);
    }

    if (type == GGML_TYPE_F32) {
        std::memcpy(tensor->data, input.data.data(), input.data.size() * sizeof(float));
    } else if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(
                input.data.data(),
                reinterpret_cast<ggml_fp16_t *>(tensor->data),
                (int64_t) input.data.size());
    } else {
        throw std::runtime_error("unsupported ggml input tensor type");
    }

    ggml_set_input(tensor);
    GGML_UNUSED(runtime);
    return tensor;
}

static struct ggml_tensor * create_matmul_input_tensor_2d(
        struct ggml_context * ctx,
        const struct ggml_tensor * weight,
        const tensor2d & input,
        inference_runtime * runtime = nullptr) {
    return create_input_tensor_2d(ctx, input, matmul_input_type_from_weight(weight), runtime);
}

static struct ggml_tensor * create_input_tensor_4d(struct ggml_context * ctx, const tensor4d & input, inference_runtime * runtime = nullptr) {
    struct ggml_tensor * tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, input.n0, input.n1, input.n2, input.n3);
    if (tensor == nullptr) {
        throw std::runtime_error("failed to allocate ggml input tensor");
    }
    if (runtime_uses_backend(runtime)) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(tensor));
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate host buffer for Cohere input tensor");
        }

        tensor->data = ggml_backend_buffer_get_base(buffer);
        tensor->buffer = buffer;
        if (ggml_backend_buffer_init_tensor(buffer, tensor) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            tensor->buffer = nullptr;
            throw std::runtime_error("failed to initialize Cohere input tensor backend buffer");
        }
        runtime->temp_input_buffers.push_back(buffer);
    }
    std::memcpy(tensor->data, input.data.data(), input.data.size() * sizeof(float));
    ggml_set_input(tensor);
    GGML_UNUSED(runtime);
    return tensor;
}

static uint8_t * get_vec_dot_input_buffer(inference_runtime * runtime, size_t bytes) {
    if (runtime != nullptr) {
        if (runtime->vec_dot_input_buffer.size() < bytes) {
            runtime->vec_dot_input_buffer.resize(bytes);
        }
        return runtime->vec_dot_input_buffer.data();
    }

    static thread_local std::vector<uint8_t> fallback;
    if (fallback.size() < bytes) {
        fallback.resize(bytes);
    }
    return fallback.data();
}

static bool can_use_vec_dot_linear(const struct ggml_tensor * weight, const tensor2d & input) {
    if (input.n1 != 1 || weight == nullptr || ggml_n_dims(weight) != 2 || !ggml_is_contiguous(weight)) {
        return false;
    }

    if (weight->type != GGML_TYPE_F16 && weight->type != GGML_TYPE_F32) {
        return false;
    }

    return weight->ne[1] <= 8192;
}

static tensor2d eval_linear_vec_dot(
        const struct ggml_tensor * weight,
        const struct ggml_tensor * bias,
        const tensor2d & input,
        inference_runtime * runtime = nullptr) {
    const int32_t in_dim = input.n0;
    const int32_t out_dim = (int32_t) weight->ne[1];

    const struct ggml_type_traits_cpu * weight_traits = ggml_get_type_traits_cpu(weight->type);
    if (weight_traits == nullptr || weight_traits->vec_dot == nullptr) {
        throw std::runtime_error("missing ggml vec_dot for linear weight type");
    }

    const enum ggml_type vec_type = weight_traits->vec_dot_type;
    const struct ggml_type_traits_cpu * vec_traits = ggml_get_type_traits_cpu(vec_type);
    if (vec_traits == nullptr || vec_traits->from_float == nullptr) {
        throw std::runtime_error("missing ggml from_float for vec_dot input type");
    }

    const size_t input_bytes = ggml_row_size(vec_type, in_dim);
    uint8_t * input_buf = get_vec_dot_input_buffer(runtime, input_bytes);
    vec_traits->from_float(input.data.data(), input_buf, in_dim);

    const float * bias_data = tensor_data_f32(bias);
    const char * weight_data = reinterpret_cast<const char *>(weight->data);
    const size_t row_size = size_t(weight->nb[1]);

    tensor2d out(out_dim, 1);
    for (int32_t oc = 0; oc < out_dim; ++oc) {
        float value = 0.0f;
        weight_traits->vec_dot(
                in_dim,
                &value,
                0,
                weight_data + row_size * size_t(oc),
                0,
                input_buf,
                0,
                1);
        const float b = bias_data ? bias_data[oc] : (bias ? tensor_get_f32(bias, oc) : 0.0f);
        out.at(oc, 0) = value + b;
    }

    return out;
}

static tensor2d eval_linear(
        struct ggml_tensor * weight,
        const struct ggml_tensor * bias,
        const tensor2d & input,
        int32_t n_threads,
        inference_runtime * runtime = nullptr) {
    struct ggml_tensor * weight_eval = maybe_backend_tensor(runtime, weight);
    const struct ggml_tensor * bias_eval = maybe_backend_tensor(runtime, bias);

    if (weight_eval == weight && bias_eval == bias && can_use_vec_dot_linear(weight, input)) {
        return eval_linear_vec_dot(weight, bias, input, runtime);
    }

    ggml_ptr ctx_holder;
    struct ggml_context * ctx = begin_linear_workspace(runtime);
    if (ctx == nullptr) {
        ctx_holder = make_compute_ctx();
        ctx = ctx_holder.get();
    }

    struct ggml_tensor * src = create_matmul_input_tensor_2d(ctx, weight_eval, input, runtime);
    struct ggml_tensor * cur = ggml_mul_mat(ctx, weight_eval, src);
    if (bias_eval != nullptr) {
        struct ggml_tensor * bias2 = ggml_reshape_2d(ctx, const_cast<struct ggml_tensor *>(bias_eval), bias_eval->ne[0], 1);
        cur = ggml_add(ctx, cur, ggml_repeat(ctx, bias2, cur));
    }
    run_graph(ctx, cur, n_threads, runtime);

    tensor2d out = tensor_from_ggml_2d(cur);

    return out;
}

static tensor2d eval_linear_from_rank3_weight(
        struct ggml_tensor * weight,
        const struct ggml_tensor * bias,
        const tensor2d & input,
        int32_t n_threads,
        inference_runtime * runtime = nullptr) {
    struct ggml_tensor * weight_eval = maybe_backend_tensor(runtime, weight);
    const struct ggml_tensor * bias_eval = maybe_backend_tensor(runtime, bias);

    ggml_ptr ctx_holder;
    struct ggml_context * ctx = begin_linear_workspace(runtime);
    if (ctx == nullptr) {
        ctx_holder = make_compute_ctx();
        ctx = ctx_holder.get();
    }

    struct ggml_tensor * src = create_matmul_input_tensor_2d(ctx, weight_eval, input, runtime);
    struct ggml_tensor * w2 = ggml_reshape_2d(ctx, weight_eval, weight_eval->ne[0] * weight_eval->ne[1], weight_eval->ne[2]);
    struct ggml_tensor * cur = ggml_mul_mat(ctx, w2, src);
    if (bias_eval != nullptr) {
        struct ggml_tensor * bias2 = ggml_reshape_2d(ctx, const_cast<struct ggml_tensor *>(bias_eval), bias_eval->ne[0], 1);
        cur = ggml_add(ctx, cur, ggml_repeat(ctx, bias2, cur));
    }
    run_graph(ctx, cur, n_threads, runtime);

    tensor2d out = tensor_from_ggml_2d(cur);

    return out;
}

static tensor2d eval_linear_from_rank4_weight(
        struct ggml_tensor * weight,
        const struct ggml_tensor * bias,
        const tensor2d & input,
        int32_t n_threads,
        inference_runtime * runtime = nullptr) {
    struct ggml_tensor * weight_eval = maybe_backend_tensor(runtime, weight);
    const struct ggml_tensor * bias_eval = maybe_backend_tensor(runtime, bias);

    ggml_ptr ctx_holder;
    struct ggml_context * ctx = begin_linear_workspace(runtime);
    if (ctx == nullptr) {
        ctx_holder = make_compute_ctx();
        ctx = ctx_holder.get();
    }

    struct ggml_tensor * src = create_matmul_input_tensor_2d(ctx, weight_eval, input, runtime);
    struct ggml_tensor * w2 = ggml_reshape_2d(
            ctx,
            weight_eval,
            weight_eval->ne[0] * weight_eval->ne[1] * weight_eval->ne[2],
            weight_eval->ne[3]);
    struct ggml_tensor * cur = ggml_mul_mat(ctx, w2, src);
    if (bias_eval != nullptr) {
        struct ggml_tensor * bias2 = ggml_reshape_2d(ctx, const_cast<struct ggml_tensor *>(bias_eval), bias_eval->ne[0], 1);
        cur = ggml_add(ctx, cur, ggml_repeat(ctx, bias2, cur));
    }
    run_graph(ctx, cur, n_threads, runtime);

    tensor2d out = tensor_from_ggml_2d(cur);

    return out;
}

static struct ggml_tensor * apply_activation_graph(
        struct ggml_context * ctx,
        struct ggml_tensor * input,
        const std::string & act) {
    if (act == "relu") {
        return ggml_relu(ctx, input);
    }

    if (act == "silu" || act == "swish") {
        return ggml_silu(ctx, input);
    }

    throw std::runtime_error(format("unsupported activation '%s'", act.c_str()));
}

static void apply_activation(tensor2d & x, const std::string & act);

static struct ggml_tensor * build_linear_graph(
        struct ggml_context * ctx,
        struct ggml_tensor * weight,
        const struct ggml_tensor * bias,
        struct ggml_tensor * src) {
    struct ggml_tensor * cur = ggml_mul_mat(ctx, weight, src);
    if (bias != nullptr) {
        struct ggml_tensor * bias2 = ggml_reshape_2d(ctx, const_cast<struct ggml_tensor *>(bias), bias->ne[0], 1);
        cur = ggml_add(ctx, cur, ggml_repeat(ctx, bias2, cur));
    }
    return cur;
}

static tensor2d eval_linear_activation_linear(
        struct ggml_tensor * w1,
        const struct ggml_tensor * b1,
        struct ggml_tensor * w2,
        const struct ggml_tensor * b2,
        const tensor2d & input,
        const std::string & act,
        int32_t n_threads,
        inference_runtime * runtime = nullptr) {
    if (maybe_backend_tensor(runtime, w1) != w1 || maybe_backend_tensor(runtime, w2) != w2) {
        tensor2d hidden = eval_linear(w1, b1, input, n_threads, runtime);
        apply_activation(hidden, act);
        return eval_linear(w2, b2, hidden, n_threads, runtime);
    }

    if (input.n1 == 1 &&
            can_use_vec_dot_linear(w1, input) &&
            ggml_is_contiguous(w2) &&
            (w2->type == GGML_TYPE_F16 || w2->type == GGML_TYPE_F32) &&
            w2->ne[0] == w1->ne[1] &&
            w2->ne[1] <= 8192) {
        tensor2d hidden = eval_linear_vec_dot(w1, b1, input, runtime);
        apply_activation(hidden, act);
        return eval_linear_vec_dot(w2, b2, hidden, runtime);
    }

    ggml_ptr ctx_holder;
    struct ggml_context * ctx = begin_linear_workspace(runtime);
    if (ctx == nullptr) {
        ctx_holder = make_compute_ctx();
        ctx = ctx_holder.get();
    }

    struct ggml_tensor * src = create_matmul_input_tensor_2d(ctx, w1, input, runtime);
    struct ggml_tensor * cur = ggml_mul_mat(ctx, w1, src);
    if (b1 != nullptr) {
        struct ggml_tensor * bias1 = ggml_reshape_2d(ctx, const_cast<struct ggml_tensor *>(b1), b1->ne[0], 1);
        cur = ggml_add(ctx, cur, ggml_repeat(ctx, bias1, cur));
    }

    cur = apply_activation_graph(ctx, cur, act);

    cur = ggml_mul_mat(ctx, w2, cur);
    if (b2 != nullptr) {
        struct ggml_tensor * bias2 = ggml_reshape_2d(ctx, const_cast<struct ggml_tensor *>(b2), b2->ne[0], 1);
        cur = ggml_add(ctx, cur, ggml_repeat(ctx, bias2, cur));
    }

    run_graph(ctx, cur, n_threads, runtime);

    tensor2d out = tensor_from_ggml_2d(cur);

    return out;
}

static tensor4d eval_conv2d(
        struct ggml_tensor * weight,
        const struct ggml_tensor * bias,
        const tensor4d & input,
        int32_t stride_x,
        int32_t stride_y,
        int32_t pad_x,
        int32_t pad_y,
        int32_t n_threads,
        inference_runtime * runtime = nullptr) {
    struct ggml_tensor * weight_eval = maybe_backend_tensor(runtime, weight);
    const struct ggml_tensor * bias_eval = maybe_backend_tensor(runtime, bias);
    const bool use_backend_conv = false;

    if (use_backend_conv && (weight_eval != weight || bias_eval != bias)) {
        ggml_ptr ctx_holder;
        struct ggml_context * ctx = begin_linear_workspace(runtime);
        if (ctx == nullptr) {
            ctx_holder = make_compute_ctx();
            ctx = ctx_holder.get();
        }

        struct ggml_tensor * src = create_input_tensor_4d(ctx, input, runtime);
        struct ggml_tensor * cur = ggml_conv_2d(ctx, weight_eval, src, stride_x, stride_y, pad_x, pad_y, 1, 1);
        if (bias_eval != nullptr) {
            struct ggml_tensor * bias4 = ggml_reshape_4d(ctx, const_cast<struct ggml_tensor *>(bias_eval), 1, 1, bias_eval->ne[0], 1);
            cur = ggml_add(ctx, cur, ggml_repeat(ctx, bias4, cur));
        }
        run_graph(ctx, cur, n_threads, runtime);
        return tensor_from_ggml_4d(cur);
    }

    const tensor4d kernel = tensor_from_ggml_4d(weight);
    if (kernel.n2 != input.n2) {
        throw std::runtime_error(format(
                "conv2d input channel mismatch: kernel=%d input=%d",
                kernel.n2,
                input.n2));
    }

    const int32_t out_w = (input.n0 + 2 * pad_x - (kernel.n0 - 1) - 1) / stride_x + 1;
    const int32_t out_h = (input.n1 + 2 * pad_y - (kernel.n1 - 1) - 1) / stride_y + 1;
    if (out_w <= 0 || out_h <= 0) {
        throw std::runtime_error("conv2d produced non-positive output shape");
    }

    tensor4d out(out_w, out_h, kernel.n3, input.n3);
    for (int32_t n = 0; n < input.n3; ++n) {
        for (int32_t oc = 0; oc < kernel.n3; ++oc) {
            const float b = bias != nullptr ? tensor_get_f32(bias, oc) : 0.0f;
            for (int32_t oy = 0; oy < out_h; ++oy) {
                for (int32_t ox = 0; ox < out_w; ++ox) {
                    float acc = b;
                    for (int32_t ic = 0; ic < kernel.n2; ++ic) {
                        for (int32_t ky = 0; ky < kernel.n1; ++ky) {
                            const int32_t iy = oy * stride_y + ky - pad_y;
                            if (iy < 0 || iy >= input.n1) {
                                continue;
                            }
                            for (int32_t kx = 0; kx < kernel.n0; ++kx) {
                                const int32_t ix = ox * stride_x + kx - pad_x;
                                if (ix < 0 || ix >= input.n0) {
                                    continue;
                                }
                                acc += input.at(ix, iy, ic, n) * kernel.at(kx, ky, ic, oc);
                            }
                        }
                    }
                    out.at(ox, oy, oc, n) = acc;
                }
            }
        }
    }

    return out;
}

static tensor4d eval_conv2d_pointwise(
        struct ggml_tensor * weight,
        const struct ggml_tensor * bias,
        const tensor4d & input,
        int32_t n_threads,
        inference_runtime * runtime = nullptr) {
    if (weight->ne[0] != 1 || weight->ne[1] != 1) {
        throw std::runtime_error("pointwise conv helper expects a 1x1 kernel");
    }

    const int32_t out_channels = (int32_t) weight->ne[3];
    const int32_t in_channels = input.n2;
    const int32_t positions = input.n0 * input.n1 * input.n3;

    tensor2d flat(in_channels, positions);
    int32_t pos = 0;
    for (int32_t n = 0; n < input.n3; ++n) {
        for (int32_t y = 0; y < input.n1; ++y) {
            for (int32_t x = 0; x < input.n0; ++x) {
                for (int32_t c = 0; c < in_channels; ++c) {
                    flat.at(c, pos) = input.at(x, y, c, n);
                }
                ++pos;
            }
        }
    }

    tensor2d projected = eval_linear_from_rank4_weight(weight, bias, flat, n_threads, runtime);
    tensor4d out(input.n0, input.n1, out_channels, input.n3);

    pos = 0;
    for (int32_t n = 0; n < input.n3; ++n) {
        for (int32_t y = 0; y < input.n1; ++y) {
            for (int32_t x = 0; x < input.n0; ++x) {
                for (int32_t c = 0; c < out_channels; ++c) {
                    out.at(x, y, c, n) = projected.at(c, pos);
                }
                ++pos;
            }
        }
    }

    return out;
}

static tensor4d eval_conv2d_dw(
        struct ggml_tensor * weight,
        const struct ggml_tensor * bias,
        const tensor4d & input,
        int32_t stride_x,
        int32_t stride_y,
        int32_t pad_x,
        int32_t pad_y,
        int32_t n_threads,
        inference_runtime * runtime = nullptr) {
    struct ggml_tensor * weight_eval = maybe_backend_tensor(runtime, weight);
    const struct ggml_tensor * bias_eval = maybe_backend_tensor(runtime, bias);
    const bool use_backend_conv = false;

    if (use_backend_conv && (weight_eval != weight || bias_eval != bias)) {
        ggml_ptr ctx_holder;
        struct ggml_context * ctx = begin_linear_workspace(runtime);
        if (ctx == nullptr) {
            ctx_holder = make_compute_ctx();
            ctx = ctx_holder.get();
        }

        struct ggml_tensor * src = create_input_tensor_4d(ctx, input, runtime);
        struct ggml_tensor * cur = ggml_conv_2d_dw(ctx, weight_eval, src, stride_x, stride_y, pad_x, pad_y, 1, 1);
        if (bias_eval != nullptr) {
            struct ggml_tensor * bias4 = ggml_reshape_4d(ctx, const_cast<struct ggml_tensor *>(bias_eval), 1, 1, bias_eval->ne[0], 1);
            cur = ggml_add(ctx, cur, ggml_repeat(ctx, bias4, cur));
        }
        run_graph(ctx, cur, n_threads, runtime);
        return tensor_from_ggml_4d(cur);
    }

    const tensor4d kernel = tensor_from_ggml_4d(weight);
    if (kernel.n2 != 1) {
        throw std::runtime_error("depthwise conv expects kernel input channel dimension to be 1");
    }
    if (kernel.n3 != input.n2) {
        throw std::runtime_error(format(
                "depthwise conv channel mismatch: kernel=%d input=%d",
                kernel.n3,
                input.n2));
    }

    const int32_t out_w = (input.n0 + 2 * pad_x - (kernel.n0 - 1) - 1) / stride_x + 1;
    const int32_t out_h = (input.n1 + 2 * pad_y - (kernel.n1 - 1) - 1) / stride_y + 1;
    if (out_w <= 0 || out_h <= 0) {
        throw std::runtime_error("depthwise conv produced non-positive output shape");
    }

    tensor4d out(out_w, out_h, input.n2, input.n3);
    for (int32_t n = 0; n < input.n3; ++n) {
        for (int32_t c = 0; c < input.n2; ++c) {
            const float b = bias != nullptr ? tensor_get_f32(bias, c) : 0.0f;
            for (int32_t oy = 0; oy < out_h; ++oy) {
                for (int32_t ox = 0; ox < out_w; ++ox) {
                    float acc = b;
                    for (int32_t ky = 0; ky < kernel.n1; ++ky) {
                        const int32_t iy = oy * stride_y + ky - pad_y;
                        if (iy < 0 || iy >= input.n1) {
                            continue;
                        }
                        for (int32_t kx = 0; kx < kernel.n0; ++kx) {
                            const int32_t ix = ox * stride_x + kx - pad_x;
                            if (ix < 0 || ix >= input.n0) {
                                continue;
                            }
                            acc += input.at(ix, iy, c, n) * kernel.at(kx, ky, 0, c);
                        }
                    }
                    out.at(ox, oy, c, n) = acc;
                }
            }
        }
    }

    return out;
}

static void apply_relu(tensor4d & x) {
    for (size_t i = 0; i < x.data.size(); ++i) {
        if (x.data[i] < 0.0f) {
            x.data[i] = 0.0f;
        }
    }
}

static void apply_relu(tensor2d & x) {
    for (size_t i = 0; i < x.data.size(); ++i) {
        if (x.data[i] < 0.0f) {
            x.data[i] = 0.0f;
        }
    }
}

static float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static void apply_silu(tensor2d & x) {
    for (size_t i = 0; i < x.data.size(); ++i) {
        x.data[i] = silu(x.data[i]);
    }
}

static void apply_activation(tensor2d & x, const std::string & act) {
    if (act == "relu") {
        apply_relu(x);
        return;
    }

    if (act == "silu" || act == "swish") {
        apply_silu(x);
        return;
    }

    throw std::runtime_error(format("unsupported activation '%s'", act.c_str()));
}

static tensor2d add_scaled(const tensor2d & a, const tensor2d & b, float scale) {
    if (a.n0 != b.n0 || a.n1 != b.n1) {
        throw std::runtime_error("shape mismatch in add_scaled");
    }

    tensor2d out(a.n0, a.n1);
    for (size_t i = 0; i < out.data.size(); ++i) {
        out.data[i] = a.data[i] + scale * b.data[i];
    }

    return out;
}

static tensor2d layer_norm(const tensor2d & x, const struct ggml_tensor * weight, const struct ggml_tensor * bias) {
    tensor2d out(x.n0, x.n1);
    const float * weight_data = tensor_data_f32(weight);
    const float * bias_data = tensor_data_f32(bias);

    for (int32_t col = 0; col < x.n1; ++col) {
        double mean = 0.0;
        for (int32_t row = 0; row < x.n0; ++row) {
            mean += x.at(row, col);
        }
        mean /= double(x.n0);

        double var = 0.0;
        for (int32_t row = 0; row < x.n0; ++row) {
            const double diff = double(x.at(row, col)) - mean;
            var += diff * diff;
        }
        var /= double(x.n0);
        const float inv_std = 1.0f / std::sqrt(float(var) + LAYER_NORM_EPS);

        for (int32_t row = 0; row < x.n0; ++row) {
            const float norm = (x.at(row, col) - float(mean)) * inv_std;
            const float w = weight_data ? weight_data[row] : tensor_get_f32(weight, row);
            const float b = bias_data ? bias_data[row] : (bias ? tensor_get_f32(bias, row) : 0.0f);
            out.at(row, col) = norm * w + b;
        }
    }

    return out;
}

static tensor2d feed_forward(
        const tensor2d & x,
        struct ggml_tensor * w1,
        const struct ggml_tensor * b1,
        struct ggml_tensor * w2,
        const struct ggml_tensor * b2,
        int32_t n_threads,
        inference_runtime * runtime = nullptr) {
    return eval_linear_activation_linear(w1, b1, w2, b2, x, "silu", n_threads, runtime);
}

static tensor2d decoder_feed_forward(
        const tensor2d & x,
        struct ggml_tensor * w1,
        const struct ggml_tensor * b1,
        struct ggml_tensor * w2,
        const struct ggml_tensor * b2,
        const std::string & act,
        int32_t n_threads,
        inference_runtime * runtime = nullptr) {
    return eval_linear_activation_linear(w1, b1, w2, b2, x, act, n_threads, runtime);
}

static tensor2d glu(const tensor2d & x) {
    if (x.n0 % 2 != 0) {
        throw std::runtime_error("GLU input channel count must be even");
    }

    const int32_t half = x.n0 / 2;
    tensor2d out(half, x.n1);

    for (int32_t col = 0; col < x.n1; ++col) {
        for (int32_t row = 0; row < half; ++row) {
            const float gate = x.at(row + half, col);
            out.at(row, col) = x.at(row, col) / (1.0f + std::exp(-gate));
        }
    }

    return out;
}

static tensor2d depthwise_conv1d(
        const tensor2d & x,
        const struct ggml_tensor * weight,
        const struct ggml_tensor * bias,
        int32_t n_threads,
        inference_runtime * runtime = nullptr) {
    const struct ggml_tensor * weight_eval = maybe_backend_tensor(runtime, weight);
    const struct ggml_tensor * bias_eval = maybe_backend_tensor(runtime, bias);
    const bool use_backend_conv = false;

    if (use_backend_conv && (weight_eval != weight || bias_eval != bias)) {
        ggml_ptr ctx_holder;
        struct ggml_context * ctx = begin_linear_workspace(runtime);
        if (ctx == nullptr) {
            ctx_holder = make_compute_ctx();
            ctx = ctx_holder.get();
        }

        tensor2d transposed(x.n1, x.n0);
        for (int32_t t = 0; t < x.n1; ++t) {
            for (int32_t c = 0; c < x.n0; ++c) {
                transposed.at(t, c) = x.at(c, t);
            }
        }

        struct ggml_tensor * src = create_input_tensor_2d(ctx, transposed, GGML_TYPE_F32, runtime);
        struct ggml_tensor * cur = ggml_conv_1d_dw_ph(
                ctx,
                const_cast<struct ggml_tensor *>(weight_eval),
                src,
                1,
                1);
        if (bias_eval != nullptr) {
            struct ggml_tensor * bias3 = ggml_reshape_3d(ctx, const_cast<struct ggml_tensor *>(bias_eval), 1, bias_eval->ne[0], 1);
            cur = ggml_add(ctx, cur, ggml_repeat(ctx, bias3, cur));
        }
        cur = ggml_reshape_2d(ctx, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
        run_graph(ctx, cur, n_threads, runtime);

        tensor2d out_tc = tensor_from_ggml_2d(cur);
        tensor2d out(x.n0, x.n1);
        for (int32_t t = 0; t < x.n1; ++t) {
            for (int32_t c = 0; c < x.n0; ++c) {
                out.at(c, t) = out_tc.at(t, c);
            }
        }

        return out;
    }

    const int32_t kernel = (int32_t) weight->ne[0];
    const int32_t channels = x.n0;
    const int32_t pad = kernel / 2;
    const float * bias_data = tensor_data_f32(bias);
    const float * weight_f32 = weight->type == GGML_TYPE_F32 && ggml_is_contiguous(weight)
            ? reinterpret_cast<const float *>(weight->data)
            : nullptr;
    const ggml_fp16_t * weight_f16 = weight->type == GGML_TYPE_F16 && ggml_is_contiguous(weight)
            ? reinterpret_cast<const ggml_fp16_t *>(weight->data)
            : nullptr;
    std::vector<float> weight_cache;

    if (weight_f32 == nullptr && weight_f16 != nullptr) {
        weight_cache.resize(size_t(kernel) * size_t(channels));
        ggml_fp16_to_fp32_row(weight_f16, weight_cache.data(), (int64_t) weight_cache.size());
        weight_f32 = weight_cache.data();
        weight_f16 = nullptr;
    }

    tensor2d out(x.n0, x.n1);
    for (int32_t c = 0; c < channels; ++c) {
        const float b = bias_data ? bias_data[c] : (bias ? tensor_get_f32(bias, c) : 0.0f);
        for (int32_t t = 0; t < x.n1; ++t) {
            float sum = b;
            for (int32_t k = 0; k < kernel; ++k) {
                const int32_t src_t = t + k - pad;
                if (src_t < 0 || src_t >= x.n1) {
                    continue;
                }
                const size_t index = size_t(k) + size_t(kernel) * size_t(c);
                const float w = weight_f32 ? weight_f32[index]
                        : (weight_f16 ? ggml_fp16_to_fp32(weight_f16[index]) : tensor_get_f32(weight, k, 0, c));
                sum += w * x.at(c, src_t);
            }
            out.at(c, t) = sum;
        }
    }

    return out;
}

static void zero_masked_positions(tensor2d & x, int32_t valid_length) {
    for (int32_t t = valid_length; t < x.n1; ++t) {
        for (int32_t c = 0; c < x.n0; ++c) {
            x.at(c, t) = 0.0f;
        }
    }
}

static void zero_masked_time_positions(tensor4d & x, int32_t valid_length) {
    for (int32_t n = 0; n < x.n3; ++n) {
        for (int32_t c = 0; c < x.n2; ++c) {
            for (int32_t t = valid_length; t < x.n1; ++t) {
                for (int32_t f = 0; f < x.n0; ++f) {
                    x.at(f, t, c, n) = 0.0f;
                }
            }
        }
    }
}

static int32_t conv_output_length_1d(int32_t input_length, int32_t kernel, int32_t stride, int32_t pad_total) {
    return (input_length + pad_total - kernel) / stride + 1;
}

static tensor2d build_relative_positional_encoding(int32_t d_model, int32_t length) {
    tensor2d out(d_model, 2 * length - 1);
    for (int32_t pos_idx = 0; pos_idx < out.n1; ++pos_idx) {
        const float position = float((length - 1) - pos_idx);
        for (int32_t i = 0; i < d_model; i += 2) {
            const float div = std::exp(-(std::log(10000.0f) / float(d_model)) * float(i));
            out.at(i + 0, pos_idx) = std::sin(position * div);
            if (i + 1 < d_model) {
                out.at(i + 1, pos_idx) = std::cos(position * div);
            }
        }
    }
    return out;
}

static const tensor2d & get_relative_positional_encoding(inference_runtime * runtime, int32_t d_model, int32_t length) {
    if (runtime == nullptr) {
        static thread_local tensor2d fallback;
        fallback = build_relative_positional_encoding(d_model, length);
        return fallback;
    }

    const int64_t key = rel_pos_cache_key(d_model, length);
    const std::map<int64_t, tensor2d>::iterator it = runtime->rel_pos_cache.find(key);
    if (it != runtime->rel_pos_cache.end()) {
        return it->second;
    }

    return runtime->rel_pos_cache.emplace(key, build_relative_positional_encoding(d_model, length)).first->second;
}

static tensor2d encoder_self_attention(
        const tensor2d & x,
        struct ggml_tensor * q_weight,
        const struct ggml_tensor * q_bias,
        struct ggml_tensor * k_weight,
        const struct ggml_tensor * k_bias,
        struct ggml_tensor * v_weight,
        const struct ggml_tensor * v_bias,
        struct ggml_tensor * pos_weight,
        struct ggml_tensor * out_weight,
        const struct ggml_tensor * out_bias,
        const struct ggml_tensor * pos_bias_u,
        const struct ggml_tensor * pos_bias_v,
        int32_t n_heads,
        int32_t n_threads,
        inference_runtime * runtime = nullptr) {
    const int32_t hidden = x.n0;
    const int32_t length = x.n1;
    const int32_t head_dim = hidden / n_heads;
    const float scale = 1.0f / std::sqrt(float(head_dim));

    const tensor2d & rel_pos = get_relative_positional_encoding(runtime, hidden, length);
    tensor2d q = eval_linear(q_weight, q_bias, x, n_threads, runtime);
    tensor2d k = eval_linear(k_weight, k_bias, x, n_threads, runtime);
    tensor2d v = eval_linear(v_weight, v_bias, x, n_threads, runtime);
    tensor2d p = eval_linear(pos_weight, nullptr, rel_pos, n_threads, runtime);

    tensor2d attn(hidden, length);
    const float * pos_u = tensor_data_f32(pos_bias_u);
    const float * pos_v = tensor_data_f32(pos_bias_v);

#if COHERE_HAVE_CBLAS
    const float * q_data = q.data.data();
    const float * k_data = k.data.data();
    const float * v_data = v.data.data();
    const float * p_data = p.data.data();
    float * attn_data = attn.data.data();

    std::vector<float> bias_u_vals{std::vector<float>(size_t(head_dim))};
    std::vector<float> bias_v_vals{std::vector<float>(size_t(head_dim))};
    std::vector<float> q_u(size_t(head_dim) * size_t(length));
    std::vector<float> q_v(size_t(head_dim) * size_t(length));
    std::vector<float> ac(size_t(length) * size_t(length));
    std::vector<float> bd(size_t(length) * size_t(2 * length - 1));
    std::vector<float> probs(size_t(length) * size_t(length));
    std::vector<float> score_row{std::vector<float>(size_t(length))};

    for (int32_t h = 0; h < n_heads; ++h) {
        const int32_t base = h * head_dim;
        const size_t bias_offset = size_t(head_dim) * size_t(h);

        for (int32_t d = 0; d < head_dim; ++d) {
            bias_u_vals[size_t(d)] = pos_u ? pos_u[bias_offset + size_t(d)] : tensor_get_f32(pos_bias_u, d, h);
            bias_v_vals[size_t(d)] = pos_v ? pos_v[bias_offset + size_t(d)] : tensor_get_f32(pos_bias_v, d, h);
        }

        for (int32_t pos = 0; pos < length; ++pos) {
            const float * q_col = q_data + size_t(base) + size_t(hidden) * size_t(pos);
            float * q_u_col = q_u.data() + size_t(head_dim) * size_t(pos);
            float * q_v_col = q_v.data() + size_t(head_dim) * size_t(pos);
            for (int32_t d = 0; d < head_dim; ++d) {
                const float value = q_col[d];
                q_u_col[d] = value + bias_u_vals[size_t(d)];
                q_v_col[d] = value + bias_v_vals[size_t(d)];
            }
        }

        cblas_sgemm(
                CblasColMajor,
                CblasTrans,
                CblasNoTrans,
                length,
                length,
                head_dim,
                1.0f,
                q_u.data(),
                head_dim,
                k_data + base,
                hidden,
                0.0f,
                ac.data(),
                length);

        cblas_sgemm(
                CblasColMajor,
                CblasTrans,
                CblasNoTrans,
                length,
                2 * length - 1,
                head_dim,
                1.0f,
                q_v.data(),
                head_dim,
                p_data + base,
                hidden,
                0.0f,
                bd.data(),
                length);

        for (int32_t q_pos = 0; q_pos < length; ++q_pos) {
            float max_score = -std::numeric_limits<float>::infinity();
            for (int32_t k_pos = 0; k_pos < length; ++k_pos) {
                const int32_t pos_index = (length - 1) - q_pos + k_pos;
                const float score = (ac[size_t(q_pos) + size_t(length) * size_t(k_pos)] +
                        bd[size_t(q_pos) + size_t(length) * size_t(pos_index)]) * scale;
                score_row[size_t(k_pos)] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            float denom = 0.0f;
            for (int32_t k_pos = 0; k_pos < length; ++k_pos) {
                const float value = std::exp(score_row[size_t(k_pos)] - max_score);
                probs[size_t(q_pos) + size_t(length) * size_t(k_pos)] = value;
                denom += value;
            }
            denom = denom > 0.0f ? denom : 1.0f;

            for (int32_t k_pos = 0; k_pos < length; ++k_pos) {
                probs[size_t(q_pos) + size_t(length) * size_t(k_pos)] /= denom;
            }
        }

        cblas_sgemm(
                CblasColMajor,
                CblasNoTrans,
                CblasTrans,
                head_dim,
                length,
                length,
                1.0f,
                v_data + base,
                hidden,
                probs.data(),
                length,
                0.0f,
                attn_data + base,
                hidden);
    }
#else
    std::vector<float> scores(length);
    std::vector<float> probs(length);
    for (int32_t h = 0; h < n_heads; ++h) {
        const int32_t base = h * head_dim;
        const size_t bias_offset = size_t(head_dim) * size_t(h);

        for (int32_t q_pos = 0; q_pos < length; ++q_pos) {
            float max_score = -std::numeric_limits<float>::infinity();

            for (int32_t k_pos = 0; k_pos < length; ++k_pos) {
                float ac = 0.0f;
                float bd = 0.0f;
                const int32_t pos_index = (length - 1) - q_pos + k_pos;

                for (int32_t d = 0; d < head_dim; ++d) {
                    const int32_t idx = base + d;
                    const float bias_u = pos_u ? pos_u[bias_offset + size_t(d)] : tensor_get_f32(pos_bias_u, d, h);
                    const float bias_v = pos_v ? pos_v[bias_offset + size_t(d)] : tensor_get_f32(pos_bias_v, d, h);
                    ac += (q.at(idx, q_pos) + bias_u) * k.at(idx, k_pos);
                    bd += (q.at(idx, q_pos) + bias_v) * p.at(idx, pos_index);
                }

                scores[k_pos] = (ac + bd) * scale;
                if (scores[k_pos] > max_score) {
                    max_score = scores[k_pos];
                }
            }

            float denom = 0.0f;
            for (int32_t k_pos = 0; k_pos < length; ++k_pos) {
                probs[k_pos] = std::exp(scores[k_pos] - max_score);
                denom += probs[k_pos];
            }
            denom = denom > 0.0f ? denom : 1.0f;

            for (int32_t d = 0; d < head_dim; ++d) {
                const int32_t idx = base + d;
                float sum = 0.0f;
                for (int32_t k_pos = 0; k_pos < length; ++k_pos) {
                    sum += (probs[k_pos] / denom) * v.at(idx, k_pos);
                }
                attn.at(idx, q_pos) = sum;
            }
        }
    }
#endif

    return eval_linear(out_weight, out_bias, attn, n_threads, runtime);
}

static tensor2d conformer_convolution(
        const tensor2d & x,
        struct ggml_tensor * pointwise1_w,
        const struct ggml_tensor * pointwise1_b,
        const struct ggml_tensor * depthwise_w,
        const struct ggml_tensor * depthwise_b,
        struct ggml_tensor * pointwise2_w,
        const struct ggml_tensor * pointwise2_b,
        int32_t valid_length,
        int32_t n_threads,
        inference_runtime * runtime = nullptr) {
    tensor2d cur = eval_linear_from_rank3_weight(pointwise1_w, pointwise1_b, x, n_threads, runtime);
    cur = glu(cur);
    zero_masked_positions(cur, valid_length);
    cur = depthwise_conv1d(cur, depthwise_w, depthwise_b, n_threads, runtime);
    apply_silu(cur);
    return eval_linear_from_rank3_weight(pointwise2_w, pointwise2_b, cur, n_threads, runtime);
}

static tensor2d run_conformer_layer(
        const model & model,
        int32_t layer_idx,
        const tensor2d & input,
        int32_t valid_length,
        int32_t n_threads,
        inference_runtime * runtime = nullptr) {
    const std::string prefix = format("encoder.layers.%d.", layer_idx);

    tensor2d cur = input;

    tensor2d ff_in = layer_norm(
            cur,
            require_tensor(model, prefix + "norm_feed_forward1.weight", GGML_TYPE_F32),
            require_tensor(model, prefix + "norm_feed_forward1.bias", GGML_TYPE_F32));

    tensor2d ff_out = feed_forward(
            ff_in,
            require_tensor(model, prefix + "feed_forward1.linear1.weight"),
            require_tensor(model, prefix + "feed_forward1.linear1.bias", GGML_TYPE_F32),
            require_tensor(model, prefix + "feed_forward1.linear2.weight"),
            require_tensor(model, prefix + "feed_forward1.linear2.bias", GGML_TYPE_F32),
            n_threads,
            runtime);
    cur = add_scaled(cur, ff_out, 0.5f);

    tensor2d att_in = layer_norm(
            cur,
            require_tensor(model, prefix + "norm_self_att.weight", GGML_TYPE_F32),
            require_tensor(model, prefix + "norm_self_att.bias", GGML_TYPE_F32));

    tensor2d att_out = encoder_self_attention(
            att_in,
            require_tensor(model, prefix + "self_attn.linear_q.weight"),
            require_tensor(model, prefix + "self_attn.linear_q.bias", GGML_TYPE_F32),
            require_tensor(model, prefix + "self_attn.linear_k.weight"),
            require_tensor(model, prefix + "self_attn.linear_k.bias", GGML_TYPE_F32),
            require_tensor(model, prefix + "self_attn.linear_v.weight"),
            require_tensor(model, prefix + "self_attn.linear_v.bias", GGML_TYPE_F32),
            require_tensor(model, prefix + "self_attn.linear_pos.weight"),
            require_tensor(model, prefix + "self_attn.linear_out.weight"),
            require_tensor(model, prefix + "self_attn.linear_out.bias", GGML_TYPE_F32),
            require_tensor(model, prefix + "self_attn.pos_bias_u", GGML_TYPE_F32),
            require_tensor(model, prefix + "self_attn.pos_bias_v", GGML_TYPE_F32),
            model.encoder.n_heads,
            n_threads,
            runtime);
    cur = add_scaled(cur, att_out, 1.0f);

    tensor2d conv_in = layer_norm(
            cur,
            require_tensor(model, prefix + "norm_conv.weight", GGML_TYPE_F32),
            require_tensor(model, prefix + "norm_conv.bias", GGML_TYPE_F32));

    tensor2d conv_out = conformer_convolution(
            conv_in,
            require_tensor(model, prefix + "conv.pointwise_conv1.weight"),
            require_tensor(model, prefix + "conv.pointwise_conv1.bias", GGML_TYPE_F32),
            require_tensor(model, prefix + "conv.depthwise_conv.weight"),
            require_tensor(model, prefix + "conv.depthwise_conv.bias", GGML_TYPE_F32),
            require_tensor(model, prefix + "conv.pointwise_conv2.weight"),
            require_tensor(model, prefix + "conv.pointwise_conv2.bias", GGML_TYPE_F32),
            valid_length,
            n_threads,
            runtime);
    cur = add_scaled(cur, conv_out, 1.0f);

    ff_in = layer_norm(
            cur,
            require_tensor(model, prefix + "norm_feed_forward2.weight", GGML_TYPE_F32),
            require_tensor(model, prefix + "norm_feed_forward2.bias", GGML_TYPE_F32));

    ff_out = feed_forward(
            ff_in,
            require_tensor(model, prefix + "feed_forward2.linear1.weight"),
            require_tensor(model, prefix + "feed_forward2.linear1.bias", GGML_TYPE_F32),
            require_tensor(model, prefix + "feed_forward2.linear2.weight"),
            require_tensor(model, prefix + "feed_forward2.linear2.bias", GGML_TYPE_F32),
            n_threads,
            runtime);
    cur = add_scaled(cur, ff_out, 0.5f);

    return layer_norm(
            cur,
            require_tensor(model, prefix + "norm_out.weight", GGML_TYPE_F32),
            require_tensor(model, prefix + "norm_out.bias", GGML_TYPE_F32));
}

static tensor2d run_conv_subsampling(
        const model & model,
        const tensor2d & mel,
        int32_t valid_frames,
        int32_t n_threads,
        inference_runtime * runtime = nullptr) {
    tensor4d x(model.encoder.feat_in, mel.n1, 1, 1);
    for (int32_t t = 0; t < mel.n1; ++t) {
        for (int32_t f = 0; f < mel.n0; ++f) {
            x.at(f, t, 0, 0) = mel.at(f, t);
        }
    }

    int32_t current_length = valid_frames;

    x = eval_conv2d(
            require_tensor(model, "encoder.pre_encode.conv0.weight"),
            require_tensor(model, "encoder.pre_encode.conv0.bias", GGML_TYPE_F32),
            x,
            2, 2, 1, 1, n_threads, runtime);
    current_length = conv_output_length_1d(current_length, 3, 2, 2);
    zero_masked_time_positions(x, current_length);
    apply_relu(x);

    x = eval_conv2d_dw(
            require_tensor(model, "encoder.pre_encode.conv1_dw.weight"),
            require_tensor(model, "encoder.pre_encode.conv1_dw.bias", GGML_TYPE_F32),
            x,
            2, 2, 1, 1, n_threads, runtime);
    current_length = conv_output_length_1d(current_length, 3, 2, 2);
    zero_masked_time_positions(x, current_length);
    x = eval_conv2d_pointwise(
            require_tensor(model, "encoder.pre_encode.conv1_pw.weight"),
            require_tensor(model, "encoder.pre_encode.conv1_pw.bias", GGML_TYPE_F32),
            x,
            n_threads,
            runtime);
    current_length = conv_output_length_1d(current_length, 1, 1, 0);
    zero_masked_time_positions(x, current_length);
    apply_relu(x);

    x = eval_conv2d_dw(
            require_tensor(model, "encoder.pre_encode.conv2_dw.weight"),
            require_tensor(model, "encoder.pre_encode.conv2_dw.bias", GGML_TYPE_F32),
            x,
            2, 2, 1, 1, n_threads, runtime);
    current_length = conv_output_length_1d(current_length, 3, 2, 2);
    zero_masked_time_positions(x, current_length);
    x = eval_conv2d_pointwise(
            require_tensor(model, "encoder.pre_encode.conv2_pw.weight"),
            require_tensor(model, "encoder.pre_encode.conv2_pw.bias", GGML_TYPE_F32),
            x,
            n_threads,
            runtime);
    current_length = conv_output_length_1d(current_length, 1, 1, 0);
    zero_masked_time_positions(x, current_length);
    apply_relu(x);

    const int32_t feat = x.n0;
    const int32_t time = x.n1;
    const int32_t channels = x.n2;

    tensor2d flat(feat * channels, time);
    for (int32_t t = 0; t < time; ++t) {
        for (int32_t c = 0; c < channels; ++c) {
            for (int32_t f = 0; f < feat; ++f) {
                flat.at(c * feat + f, t) = x.at(f, t, c, 0);
            }
        }
    }

    return eval_linear(
            require_tensor(model, "encoder.pre_encode.out.weight"),
            require_tensor(model, "encoder.pre_encode.out.bias", GGML_TYPE_F32),
            flat,
            n_threads,
            runtime);
}

static std::vector<float> build_hann_window(int32_t length) {
    std::vector<float> out(length);
    for (int32_t i = 0; i < length; ++i) {
        out[i] = 0.5f * (1.0f - std::cos((2.0f * float(M_PI) * float(i)) / float(length - 1)));
    }
    return out;
}

static void dft_cached(const float * in, int32_t n, float * out, const fft_trig_cache & cache) {
    const int32_t step = cache.n / n;

    for (int32_t k = 0; k < n; ++k) {
        float re = 0.0f;
        float im = 0.0f;
        for (int32_t t = 0; t < n; ++t) {
            const int32_t idx = (k * t * step) % cache.n;
            re += in[t] * cache.cos_vals[size_t(idx)];
            im -= in[t] * cache.sin_vals[size_t(idx)];
        }
        out[size_t(2 * k + 0)] = re;
        out[size_t(2 * k + 1)] = im;
    }
}

static void fft_cached(float * in, int32_t n, float * out, const fft_trig_cache & cache) {
    if (n == 1) {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }

    const int32_t half_n = n / 2;
    if (n - half_n * 2 == 1) {
        dft_cached(in, n, out, cache);
        return;
    }

    float * even = in + n;
    for (int32_t i = 0; i < half_n; ++i) {
        even[i] = in[2 * i];
    }
    float * even_fft = out + 2 * n;
    fft_cached(even, half_n, even_fft, cache);

    float * odd = even;
    for (int32_t i = 0; i < half_n; ++i) {
        odd[i] = in[2 * i + 1];
    }
    float * odd_fft = even_fft + n;
    fft_cached(odd, half_n, odd_fft, cache);

    const int32_t step = cache.n / n;
    for (int32_t k = 0; k < half_n; ++k) {
        const int32_t idx = k * step;
        const float re = cache.cos_vals[size_t(idx)];
        const float im = -cache.sin_vals[size_t(idx)];

        const float re_odd = odd_fft[2 * k + 0];
        const float im_odd = odd_fft[2 * k + 1];

        out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

        out[2 * (k + half_n) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
        out[2 * (k + half_n) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

static tensor2d compute_mel_features(const std::vector<float> & audio, const frontend_config & cfg, int32_t n_threads) {
    if (audio.empty()) {
        throw std::runtime_error("audio is empty");
    }

    std::vector<float> waveform = audio;

    std::vector<float> preemphasized(audio.size());
    preemphasized[0] = waveform[0];
    for (size_t i = 1; i < audio.size(); ++i) {
        preemphasized[i] = waveform[i] - cfg.preemph * waveform[i - 1];
    }

    const std::vector<float> window = cfg.window.size() == size_t(cfg.win_length)
            ? cfg.window
            : build_hann_window(cfg.win_length);
    const int32_t pad = cfg.n_fft / 2;
    const int32_t valid_frames = std::max<int32_t>(1, int32_t(audio.size() / size_t(cfg.hop_length)));
    const int32_t offset = (cfg.n_fft - cfg.win_length) / 2;

    std::vector<float> padded(audio.size() + size_t(2 * pad), 0.0f);
    std::copy(preemphasized.begin(), preemphasized.end(), padded.begin() + pad);

    const int32_t n_frames = 1 + int32_t((padded.size() - size_t(cfg.n_fft)) / size_t(cfg.hop_length));
    const int32_t n_freqs = cfg.n_fft / 2 + 1;
    const fft_trig_cache & fft_cache = get_fft_trig_cache(cfg.n_fft);

    tensor2d mel(cfg.n_mels, n_frames);
    const int32_t worker_count = std::max(1, n_threads);

    auto worker = [&](int32_t ith) {
        std::vector<float> fft_in(size_t(cfg.n_fft) * 2u, 0.0f);
        std::vector<float> fft_out(size_t(cfg.n_fft) * 2u * 2u * 2u, 0.0f);
        std::vector<float> power(size_t(n_freqs), 0.0f);

        for (int32_t frame_idx = ith; frame_idx < n_frames; frame_idx += worker_count) {
            if (frame_idx >= valid_frames) {
                for (int32_t m = 0; m < cfg.n_mels; ++m) {
                    mel.at(m, frame_idx) = 0.0f;
                }
                continue;
            }

            std::fill(fft_in.begin(), fft_in.begin() + cfg.n_fft, 0.0f);

            const int32_t start = frame_idx * cfg.hop_length;
            for (int32_t i = 0; i < cfg.win_length; ++i) {
                const int32_t src = start + offset + i;
                if (src >= 0 && src < (int32_t) padded.size()) {
                    fft_in[size_t(offset + i)] = padded[size_t(src)] * window[size_t(i)];
                }
            }

            fft_cached(fft_in.data(), cfg.n_fft, fft_out.data(), fft_cache);

            for (int32_t k = 0; k < n_freqs; ++k) {
                const float re = fft_out[size_t(2 * k + 0)];
                const float im = fft_out[size_t(2 * k + 1)];
                power[size_t(k)] = re * re + im * im;
            }

            for (int32_t m = 0; m < cfg.n_mels; ++m) {
                double sum = 0.0;
                const float * filter = cfg.mel_filters.data() + size_t(m) * size_t(n_freqs);
                int32_t k = 0;
                for (; k < n_freqs - 3; k += 4) {
                    sum +=
                            double(power[size_t(k + 0)]) * double(filter[k + 0]) +
                            double(power[size_t(k + 1)]) * double(filter[k + 1]) +
                            double(power[size_t(k + 2)]) * double(filter[k + 2]) +
                            double(power[size_t(k + 3)]) * double(filter[k + 3]);
                }
                for (; k < n_freqs; ++k) {
                    sum += double(power[size_t(k)]) * double(filter[k]);
                }
                mel.at(m, frame_idx) = std::log(float(sum) + cfg.log_zero_guard);
            }
        }
    };

    if (worker_count > 1) {
        std::vector<std::thread> workers(size_t(worker_count - 1));
        for (int32_t ith = 1; ith < worker_count; ++ith) {
            workers[size_t(ith - 1)] = std::thread(worker, ith);
        }
        worker(0);
        for (std::thread & t : workers) {
            t.join();
        }
    } else {
        worker(0);
    }

    if (cfg.normalize_per_feature) {
        for (int32_t m = 0; m < cfg.n_mels; ++m) {
            double mean = 0.0;
            for (int32_t t = 0; t < valid_frames; ++t) {
                mean += mel.at(m, t);
            }
            mean /= double(valid_frames);

            double var = 0.0;
            const double denom = std::max(1, valid_frames - 1);
            for (int32_t t = 0; t < valid_frames; ++t) {
                const double diff = double(mel.at(m, t)) - mean;
                var += diff * diff;
            }
            var /= denom;
            const float inv_std = 1.0f / (std::sqrt(float(var)) + 1e-5f);

            for (int32_t t = 0; t < valid_frames; ++t) {
                mel.at(m, t) = (mel.at(m, t) - float(mean)) * inv_std;
            }
        }
    }

    return mel;
}

static tensor2d project_encoder_for_decoder(model & model, const tensor2d & encoder_out, int32_t n_threads, inference_runtime * runtime = nullptr);

static tensor2d project_encoder_for_decoder(model & model, const tensor2d & encoder_out, int32_t n_threads, inference_runtime * runtime) {
    if (!model.has_encoder_decoder_proj) {
        return encoder_out;
    }

    return eval_linear(
            require_tensor(model, "encoder_decoder_proj.weight"),
            require_tensor(model, "encoder_decoder_proj.bias", GGML_TYPE_F32),
            encoder_out,
            n_threads,
            runtime);
}

static tensor2d run_encoder(model & model, const std::vector<float> & pcmf32, int32_t n_threads, int32_t & encoder_length, inference_runtime * runtime = nullptr) {
    const int32_t valid_mel_frames = std::max<int32_t>(1, int32_t(pcmf32.size() / size_t(model.frontend.hop_length)));
    encoder_length = conv_subsampling_output_length(valid_mel_frames);

    tensor2d mel = compute_mel_features(pcmf32, model.frontend, n_threads);
    tensor2d cur = run_conv_subsampling(model, mel, valid_mel_frames, n_threads, runtime);
    for (int32_t il = 0; il < model.encoder.n_layers; ++il) {
        cur = run_conformer_layer(model, il, cur, encoder_length, n_threads, runtime);
    }

    return project_encoder_for_decoder(model, cur, n_threads, runtime);
}

static tensor2d get_decoder_embedding(const model & model, int32_t token_id, int32_t position) {
    const struct ggml_tensor * token_embedding = require_tensor(model, "decoder.embedding.token_embedding.weight");
    const int32_t hidden = model.decoder.hidden_size;
    tensor2d out(hidden, 1);
    float * out_data = out.data.data();

    if (token_embedding->type == GGML_TYPE_F32 && ggml_is_contiguous(token_embedding)) {
        const float * token_col = reinterpret_cast<const float *>(
                reinterpret_cast<const char *>(token_embedding->data) + size_t(token_id) * size_t(token_embedding->nb[1]));
        std::memcpy(out_data, token_col, size_t(hidden) * sizeof(float));
    } else if (token_embedding->type == GGML_TYPE_F16 && ggml_is_contiguous(token_embedding)) {
        const ggml_fp16_t * token_col = reinterpret_cast<const ggml_fp16_t *>(
                reinterpret_cast<const char *>(token_embedding->data) + size_t(token_id) * size_t(token_embedding->nb[1]));
        ggml_fp16_to_fp32_row(token_col, out_data, hidden);
    } else {
        for (int32_t i = 0; i < hidden; ++i) {
            out_data[i] = tensor_get_f32(token_embedding, i, token_id);
        }
    }

    const float pos_scale = 1.0f / std::sqrt(float(hidden));
    for (int32_t i = 0; i < hidden; ++i) {
        float value = out_data[i];
        if ((i % 2) == 0) {
            const float div = std::exp(-(std::log(10000.0f) / float(hidden)) * float(i));
            value += std::sin(float(position) * div) * pos_scale;
        } else {
            const float div = std::exp(-(std::log(10000.0f) / float(hidden)) * float(i - 1));
            value += std::cos(float(position) * div) * pos_scale;
        }
        out_data[i] = value;
    }

    return layer_norm(
            out,
            require_tensor(model, "decoder.embedding.layer_norm.weight", GGML_TYPE_F32),
            require_tensor(model, "decoder.embedding.layer_norm.bias", GGML_TYPE_F32));
}

static void append_cache_column(self_kv_cache & cache, const tensor2d & column) {
    if (column.n1 != 1 || column.n0 != cache.key.n0 || column.n0 != cache.value.n0) {
        throw std::runtime_error("invalid cache append");
    }
    if (cache.length >= cache.key.n1) {
        throw std::runtime_error("decoder KV cache exhausted");
    }
    for (int32_t i = 0; i < column.n0; ++i) {
        cache.key.at(i, cache.length) = column.at(i, 0);
    }
}

static tensor2d decoder_self_attention_single(
        const tensor2d & query,
        self_kv_cache & cache,
        const tensor2d & key_column,
        const tensor2d & value_column,
        int32_t n_heads) {
    if (query.n1 != 1 || key_column.n1 != 1 || value_column.n1 != 1) {
        throw std::runtime_error("decoder self attention expects single-token tensors");
    }

    if (cache.length >= cache.key.n1) {
        throw std::runtime_error("decoder KV cache exhausted");
    }

    for (int32_t i = 0; i < key_column.n0; ++i) {
        cache.key.at(i, cache.length) = key_column.at(i, 0);
        cache.value.at(i, cache.length) = value_column.at(i, 0);
    }
    cache.length += 1;

    const int32_t hidden = query.n0;
    const int32_t head_dim = hidden / n_heads;
    const float scale = 1.0f / std::sqrt(float(head_dim));

    tensor2d out(hidden, 1);
#if COHERE_HAVE_CBLAS
    const float * query_data = query.data.data();
    const float * key_data = cache.key.data.data();
    const float * value_data = cache.value.data.data();
    float * out_data = out.data.data();
    std::vector<float> scores{std::vector<float>(size_t(cache.length))};
    std::vector<float> probs{std::vector<float>(size_t(cache.length))};

    for (int32_t h = 0; h < n_heads; ++h) {
        const int32_t base = h * head_dim;

        cblas_sgemv(
                CblasColMajor,
                CblasTrans,
                head_dim,
                cache.length,
                scale,
                key_data + base,
                hidden,
                query_data + base,
                1,
                0.0f,
                scores.data(),
                1);

        float max_score = -std::numeric_limits<float>::infinity();
        for (int32_t pos = 0; pos < cache.length; ++pos) {
            if (scores[size_t(pos)] > max_score) {
                max_score = scores[size_t(pos)];
            }
        }

        float denom = 0.0f;
        for (int32_t pos = 0; pos < cache.length; ++pos) {
            const float value = std::exp(scores[size_t(pos)] - max_score);
            probs[size_t(pos)] = value;
            denom += value;
        }
        denom = denom > 0.0f ? denom : 1.0f;

        for (int32_t pos = 0; pos < cache.length; ++pos) {
            probs[size_t(pos)] /= denom;
        }

        cblas_sgemv(
                CblasColMajor,
                CblasNoTrans,
                head_dim,
                cache.length,
                1.0f,
                value_data + base,
                hidden,
                probs.data(),
                1,
                0.0f,
                out_data + base,
                1);
    }
#else
    std::vector<float> scores(cache.length);
    std::vector<float> probs(cache.length);

    for (int32_t h = 0; h < n_heads; ++h) {
        const int32_t base = h * head_dim;
        float max_score = -std::numeric_limits<float>::infinity();

        for (int32_t pos = 0; pos < cache.length; ++pos) {
            float dot = 0.0f;
            for (int32_t d = 0; d < head_dim; ++d) {
                dot += query.at(base + d, 0) * cache.key.at(base + d, pos);
            }
            scores[pos] = dot * scale;
            if (scores[pos] > max_score) {
                max_score = scores[pos];
            }
        }

        float denom = 0.0f;
        for (int32_t pos = 0; pos < cache.length; ++pos) {
            probs[pos] = std::exp(scores[pos] - max_score);
            denom += probs[pos];
        }
        denom = denom > 0.0f ? denom : 1.0f;

        for (int32_t d = 0; d < head_dim; ++d) {
            float sum = 0.0f;
            for (int32_t pos = 0; pos < cache.length; ++pos) {
                sum += (probs[pos] / denom) * cache.value.at(base + d, pos);
            }
            out.at(base + d, 0) = sum;
        }
    }
#endif

    return out;
}

static tensor2d decoder_cross_attention_single(
        const tensor2d & query,
        const cross_kv_cache & cross,
        int32_t n_heads) {
    const int32_t hidden = query.n0;
    const int32_t head_dim = hidden / n_heads;
    const int32_t length = cross.key.n1;
    const float scale = 1.0f / std::sqrt(float(head_dim));

    tensor2d out(hidden, 1);
#if COHERE_HAVE_CBLAS
    const float * query_data = query.data.data();
    const float * key_data = cross.key.data.data();
    const float * value_data = cross.value.data.data();
    float * out_data = out.data.data();
    std::vector<float> scores{std::vector<float>(size_t(length))};
    std::vector<float> probs{std::vector<float>(size_t(length))};

    for (int32_t h = 0; h < n_heads; ++h) {
        const int32_t base = h * head_dim;

        cblas_sgemv(
                CblasColMajor,
                CblasTrans,
                head_dim,
                length,
                scale,
                key_data + base,
                hidden,
                query_data + base,
                1,
                0.0f,
                scores.data(),
                1);

        float max_score = -std::numeric_limits<float>::infinity();
        for (int32_t pos = 0; pos < length; ++pos) {
            if (scores[size_t(pos)] > max_score) {
                max_score = scores[size_t(pos)];
            }
        }

        float denom = 0.0f;
        for (int32_t pos = 0; pos < length; ++pos) {
            const float value = std::exp(scores[size_t(pos)] - max_score);
            probs[size_t(pos)] = value;
            denom += value;
        }
        denom = denom > 0.0f ? denom : 1.0f;

        for (int32_t pos = 0; pos < length; ++pos) {
            probs[size_t(pos)] /= denom;
        }

        cblas_sgemv(
                CblasColMajor,
                CblasNoTrans,
                head_dim,
                length,
                1.0f,
                value_data + base,
                hidden,
                probs.data(),
                1,
                0.0f,
                out_data + base,
                1);
    }
#else
    std::vector<float> scores(length);
    std::vector<float> probs(length);

    for (int32_t h = 0; h < n_heads; ++h) {
        const int32_t base = h * head_dim;
        float max_score = -std::numeric_limits<float>::infinity();

        for (int32_t pos = 0; pos < length; ++pos) {
            float dot = 0.0f;
            for (int32_t d = 0; d < head_dim; ++d) {
                dot += query.at(base + d, 0) * cross.key.at(base + d, pos);
            }
            scores[pos] = dot * scale;
            if (scores[pos] > max_score) {
                max_score = scores[pos];
            }
        }

        float denom = 0.0f;
        for (int32_t pos = 0; pos < length; ++pos) {
            probs[pos] = std::exp(scores[pos] - max_score);
            denom += probs[pos];
        }
        denom = denom > 0.0f ? denom : 1.0f;

        for (int32_t d = 0; d < head_dim; ++d) {
            float sum = 0.0f;
            for (int32_t pos = 0; pos < length; ++pos) {
                sum += (probs[pos] / denom) * cross.value.at(base + d, pos);
            }
            out.at(base + d, 0) = sum;
        }
    }
#endif

    return out;
}

static std::vector<cross_kv_cache> build_cross_kv(
        model & model,
        const tensor2d & encoder_states,
        int32_t n_threads,
        inference_runtime * runtime = nullptr) {
    std::vector<cross_kv_cache> caches(size_t(model.decoder.num_layers));
    for (int32_t il = 0; il < model.decoder.num_layers; ++il) {
        const std::string prefix = format("decoder.layers.%d.cross_attn.", il);
        caches[size_t(il)].key = eval_linear(
                require_tensor(model, prefix + "key.weight"),
                require_tensor(model, prefix + "key.bias", GGML_TYPE_F32),
                encoder_states,
                n_threads,
                runtime);
        caches[size_t(il)].value = eval_linear(
                require_tensor(model, prefix + "value.weight"),
                require_tensor(model, prefix + "value.bias", GGML_TYPE_F32),
                encoder_states,
                n_threads,
                runtime);
    }
    return caches;
}

static std::vector<self_kv_cache> build_self_kv(model & model) {
    std::vector<self_kv_cache> caches(size_t(model.decoder.num_layers));
    for (size_t i = 0; i < caches.size(); ++i) {
        caches[i].key = tensor2d(model.decoder.hidden_size, model.decoder.max_sequence_length);
        caches[i].value = tensor2d(model.decoder.hidden_size, model.decoder.max_sequence_length);
    }
    return caches;
}

static std::vector<float> decoder_step(
        model & model,
        int32_t token_id,
        int32_t position,
        const std::vector<cross_kv_cache> & cross_kv,
        std::vector<self_kv_cache> & self_kv,
        int32_t n_threads,
        bool need_logits,
        inference_runtime * runtime = nullptr) {
    tensor2d hidden = get_decoder_embedding(model, token_id, position);

    for (int32_t il = 0; il < model.decoder.num_layers; ++il) {
        const std::string prefix = format("decoder.layers.%d.", il);

        tensor2d norm = layer_norm(
                hidden,
                require_tensor(model, prefix + "layer_norm_1.weight", GGML_TYPE_F32),
                require_tensor(model, prefix + "layer_norm_1.bias", GGML_TYPE_F32));

        tensor2d q = eval_linear(
                require_tensor(model, prefix + "self_attn.query.weight"),
                require_tensor(model, prefix + "self_attn.query.bias", GGML_TYPE_F32),
                norm,
                n_threads,
                runtime);
        tensor2d k = eval_linear(
                require_tensor(model, prefix + "self_attn.key.weight"),
                require_tensor(model, prefix + "self_attn.key.bias", GGML_TYPE_F32),
                norm,
                n_threads,
                runtime);
        tensor2d v = eval_linear(
                require_tensor(model, prefix + "self_attn.value.weight"),
                require_tensor(model, prefix + "self_attn.value.bias", GGML_TYPE_F32),
                norm,
                n_threads,
                runtime);

        tensor2d attn = decoder_self_attention_single(q, self_kv[size_t(il)], k, v, model.decoder.num_attention_heads);
        tensor2d proj = eval_linear(
                require_tensor(model, prefix + "self_attn.out.weight"),
                require_tensor(model, prefix + "self_attn.out.bias", GGML_TYPE_F32),
                attn,
                n_threads,
                runtime);
        hidden = add_scaled(hidden, proj, 1.0f);

        norm = layer_norm(
                hidden,
                require_tensor(model, prefix + "layer_norm_2.weight", GGML_TYPE_F32),
                require_tensor(model, prefix + "layer_norm_2.bias", GGML_TYPE_F32));

        q = eval_linear(
                require_tensor(model, prefix + "cross_attn.query.weight"),
                require_tensor(model, prefix + "cross_attn.query.bias", GGML_TYPE_F32),
                norm,
                n_threads,
                runtime);
        attn = decoder_cross_attention_single(q, cross_kv[size_t(il)], model.decoder.num_attention_heads);
        proj = eval_linear(
                require_tensor(model, prefix + "cross_attn.out.weight"),
                require_tensor(model, prefix + "cross_attn.out.bias", GGML_TYPE_F32),
                attn,
                n_threads,
                runtime);
        hidden = add_scaled(hidden, proj, 1.0f);

        norm = layer_norm(
                hidden,
                require_tensor(model, prefix + "layer_norm_3.weight", GGML_TYPE_F32),
                require_tensor(model, prefix + "layer_norm_3.bias", GGML_TYPE_F32));
        tensor2d ff = decoder_feed_forward(
                norm,
                require_tensor(model, prefix + "feed_forward.dense_in.weight"),
                require_tensor(model, prefix + "feed_forward.dense_in.bias", GGML_TYPE_F32),
                require_tensor(model, prefix + "feed_forward.dense_out.weight"),
                require_tensor(model, prefix + "feed_forward.dense_out.bias", GGML_TYPE_F32),
                model.decoder.hidden_act,
                n_threads,
                runtime);
        hidden = add_scaled(hidden, ff, 1.0f);
    }

    if (!need_logits) {
        return {};
    }

    hidden = layer_norm(
            hidden,
            require_tensor(model, "decoder.final_layer_norm.weight", GGML_TYPE_F32),
            require_tensor(model, "decoder.final_layer_norm.bias", GGML_TYPE_F32));

    struct ggml_tensor * lm_head_weight = find_tensor(model, "lm_head.weight");
    if (lm_head_weight == nullptr) {
        lm_head_weight = require_tensor(model, "decoder.embedding.token_embedding.weight");
    }

    tensor2d logits = eval_linear(
            lm_head_weight,
            require_tensor(model, "lm_head.bias", GGML_TYPE_F32),
            hidden,
            n_threads,
            runtime);

    std::vector<float> out(size_t(logits.n0));
    for (int32_t i = 0; i < logits.n0; ++i) {
        out[size_t(i)] = logits.at(i, 0);
    }

    if (model.head.log_softmax) {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < out.size(); ++i) {
            if (out[i] > max_logit) {
                max_logit = out[i];
            }
        }
        double denom = 0.0;
        for (size_t i = 0; i < out.size(); ++i) {
            denom += std::exp(double(out[i] - max_logit));
        }
        const float log_denom = max_logit + float(std::log(denom));
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] -= log_denom;
        }
    }

    return out;
}

static int32_t argmax(const std::vector<float> & values) {
    int32_t best = 0;
    float best_value = values.empty() ? -std::numeric_limits<float>::infinity() : values[0];
    for (int32_t i = 1; i < (int32_t) values.size(); ++i) {
        if (values[size_t(i)] > best_value) {
            best_value = values[size_t(i)];
            best = i;
        }
    }
    return best;
}

static void append_utf8_replacement(std::string & out) {
    out.append("\xEF\xBF\xBD");
}

static std::string sanitize_utf8(const std::string & input) {
    std::string out;
    out.reserve(input.size());

    for (size_t i = 0; i < input.size();) {
        const uint8_t b0 = static_cast<uint8_t>(input[i]);
        size_t len = 0;
        uint32_t min_cp = 0;
        uint32_t cp = 0;

        if ((b0 & 0x80u) == 0) {
            len = 1;
            cp = b0;
            min_cp = 0;
        } else if ((b0 & 0xE0u) == 0xC0u) {
            len = 2;
            cp = b0 & 0x1Fu;
            min_cp = 0x80u;
        } else if ((b0 & 0xF0u) == 0xE0u) {
            len = 3;
            cp = b0 & 0x0Fu;
            min_cp = 0x800u;
        } else if ((b0 & 0xF8u) == 0xF0u) {
            len = 4;
            cp = b0 & 0x07u;
            min_cp = 0x10000u;
        } else {
            append_utf8_replacement(out);
            ++i;
            continue;
        }

        if (i + len > input.size()) {
            append_utf8_replacement(out);
            break;
        }

        bool valid = true;
        for (size_t j = 1; j < len; ++j) {
            const uint8_t bx = static_cast<uint8_t>(input[i + j]);
            if ((bx & 0xC0u) != 0x80u) {
                valid = false;
                break;
            }
            cp = (cp << 6) | (bx & 0x3Fu);
        }

        if (!valid || cp < min_cp || cp > 0x10FFFFu || (cp >= 0xD800u && cp <= 0xDFFFu)) {
            append_utf8_replacement(out);
            ++i;
            continue;
        }

        out.append(input, i, len);
        i += len;
    }

    return out;
}

static void validate_model(model & model) {
    if (model.encoder.d_model % model.encoder.n_heads != 0) {
        throw std::runtime_error("encoder hidden size is not divisible by number of heads");
    }
    if (model.decoder.hidden_size % model.decoder.num_attention_heads != 0) {
        throw std::runtime_error("decoder hidden size is not divisible by number of heads");
    }
    if ((int32_t) model.vocab.id_to_piece.size() != model.head.num_classes) {
        throw std::runtime_error("tokenizer size does not match lm head size");
    }
    if ((int32_t) model.frontend.mel_filters.size() != model.frontend.n_mels * (model.frontend.n_fft / 2 + 1)) {
        throw std::runtime_error("mel filterbank shape does not match frontend metadata");
    }
    if (!model.frontend.window.empty() && (int32_t) model.frontend.window.size() != model.frontend.win_length) {
        throw std::runtime_error("frontend window shape does not match frontend metadata");
    }

    require_tensor(model, "encoder.pre_encode.conv0.weight");
    require_tensor(model, "encoder.pre_encode.conv0.bias", GGML_TYPE_F32);
    require_tensor(model, "encoder.pre_encode.conv1_dw.weight");
    require_tensor(model, "encoder.pre_encode.conv1_dw.bias", GGML_TYPE_F32);
    require_tensor(model, "encoder.pre_encode.conv1_pw.weight");
    require_tensor(model, "encoder.pre_encode.conv1_pw.bias", GGML_TYPE_F32);
    require_tensor(model, "encoder.pre_encode.conv2_dw.weight");
    require_tensor(model, "encoder.pre_encode.conv2_dw.bias", GGML_TYPE_F32);
    require_tensor(model, "encoder.pre_encode.conv2_pw.weight");
    require_tensor(model, "encoder.pre_encode.conv2_pw.bias", GGML_TYPE_F32);
    require_tensor(model, "encoder.pre_encode.out.weight");
    require_tensor(model, "encoder.pre_encode.out.bias", GGML_TYPE_F32);
    require_tensor(model, "decoder.embedding.token_embedding.weight");
    require_tensor(model, "decoder.embedding.layer_norm.weight", GGML_TYPE_F32);
    require_tensor(model, "decoder.embedding.layer_norm.bias", GGML_TYPE_F32);
    require_tensor(model, "decoder.final_layer_norm.weight", GGML_TYPE_F32);
    require_tensor(model, "decoder.final_layer_norm.bias", GGML_TYPE_F32);
    require_tensor(model, "lm_head.bias", GGML_TYPE_F32);

    expect_tensor_shape(model, "encoder.pre_encode.conv0.weight", 3, 3, 1, model.encoder.subsampling_conv_channels);
    expect_tensor_shape(model, "encoder.pre_encode.conv0.bias", model.encoder.subsampling_conv_channels);
    expect_tensor_shape(model, "encoder.pre_encode.conv1_dw.weight", 3, 3, 1, model.encoder.subsampling_conv_channels);
    expect_tensor_shape(model, "encoder.pre_encode.conv1_dw.bias", model.encoder.subsampling_conv_channels);
    expect_tensor_shape(model, "encoder.pre_encode.conv1_pw.weight", 1, 1, model.encoder.subsampling_conv_channels, model.encoder.subsampling_conv_channels);
    expect_tensor_shape(model, "encoder.pre_encode.conv1_pw.bias", model.encoder.subsampling_conv_channels);
    expect_tensor_shape(model, "encoder.pre_encode.conv2_dw.weight", 3, 3, 1, model.encoder.subsampling_conv_channels);
    expect_tensor_shape(model, "encoder.pre_encode.conv2_dw.bias", model.encoder.subsampling_conv_channels);
    expect_tensor_shape(model, "encoder.pre_encode.conv2_pw.weight", 1, 1, model.encoder.subsampling_conv_channels, model.encoder.subsampling_conv_channels);
    expect_tensor_shape(model, "encoder.pre_encode.conv2_pw.bias", model.encoder.subsampling_conv_channels);
    expect_tensor_shape(
            model,
            "encoder.pre_encode.out.weight",
            model.encoder.subsampling_conv_channels * (model.encoder.feat_in / model.encoder.subsampling_factor),
            model.encoder.feat_out);
    expect_tensor_shape(model, "encoder.pre_encode.out.bias", model.encoder.feat_out);
    expect_tensor_shape(model, "decoder.embedding.token_embedding.weight", model.decoder.hidden_size, model.head.num_classes);
    expect_tensor_shape(model, "decoder.embedding.layer_norm.weight", model.decoder.hidden_size);
    expect_tensor_shape(model, "decoder.embedding.layer_norm.bias", model.decoder.hidden_size);
    expect_tensor_shape(model, "decoder.final_layer_norm.weight", model.decoder.hidden_size);
    expect_tensor_shape(model, "decoder.final_layer_norm.bias", model.decoder.hidden_size);
    expect_tensor_shape(model, "lm_head.bias", model.head.num_classes);
    if (find_tensor(model, "lm_head.weight") != nullptr) {
        expect_tensor_shape(model, "lm_head.weight", model.decoder.hidden_size, model.head.num_classes);
    }

    for (int32_t il = 0; il < model.encoder.n_layers; ++il) {
        const std::string prefix = format("encoder.layers.%d.", il);
        require_tensor(model, prefix + "norm_feed_forward1.weight", GGML_TYPE_F32);
        require_tensor(model, prefix + "norm_feed_forward1.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "feed_forward1.linear1.weight");
        require_tensor(model, prefix + "feed_forward1.linear1.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "feed_forward1.linear2.weight");
        require_tensor(model, prefix + "feed_forward1.linear2.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "norm_self_att.weight", GGML_TYPE_F32);
        require_tensor(model, prefix + "norm_self_att.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "self_attn.linear_q.weight");
        require_tensor(model, prefix + "self_attn.linear_q.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "self_attn.linear_k.weight");
        require_tensor(model, prefix + "self_attn.linear_k.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "self_attn.linear_v.weight");
        require_tensor(model, prefix + "self_attn.linear_v.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "self_attn.linear_pos.weight");
        require_tensor(model, prefix + "self_attn.linear_out.weight");
        require_tensor(model, prefix + "self_attn.linear_out.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "self_attn.pos_bias_u", GGML_TYPE_F32);
        require_tensor(model, prefix + "self_attn.pos_bias_v", GGML_TYPE_F32);
        require_tensor(model, prefix + "norm_conv.weight", GGML_TYPE_F32);
        require_tensor(model, prefix + "norm_conv.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "conv.pointwise_conv1.weight");
        require_tensor(model, prefix + "conv.pointwise_conv1.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "conv.depthwise_conv.weight");
        require_tensor(model, prefix + "conv.depthwise_conv.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "conv.pointwise_conv2.weight");
        require_tensor(model, prefix + "conv.pointwise_conv2.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "norm_feed_forward2.weight", GGML_TYPE_F32);
        require_tensor(model, prefix + "norm_feed_forward2.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "feed_forward2.linear1.weight");
        require_tensor(model, prefix + "feed_forward2.linear1.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "feed_forward2.linear2.weight");
        require_tensor(model, prefix + "feed_forward2.linear2.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "norm_out.weight", GGML_TYPE_F32);
        require_tensor(model, prefix + "norm_out.bias", GGML_TYPE_F32);

        const int32_t head_dim = model.encoder.d_model / model.encoder.n_heads;
        const int32_t d_ff = model.encoder.d_model * model.encoder.ff_expansion_factor;

        expect_tensor_shape(model, prefix + "norm_feed_forward1.weight", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "norm_feed_forward1.bias", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "feed_forward1.linear1.weight", model.encoder.d_model, d_ff);
        expect_tensor_shape(model, prefix + "feed_forward1.linear1.bias", d_ff);
        expect_tensor_shape(model, prefix + "feed_forward1.linear2.weight", d_ff, model.encoder.d_model);
        expect_tensor_shape(model, prefix + "feed_forward1.linear2.bias", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "norm_self_att.weight", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "norm_self_att.bias", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "self_attn.linear_q.weight", model.encoder.d_model, model.encoder.d_model);
        expect_tensor_shape(model, prefix + "self_attn.linear_q.bias", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "self_attn.linear_k.weight", model.encoder.d_model, model.encoder.d_model);
        expect_tensor_shape(model, prefix + "self_attn.linear_k.bias", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "self_attn.linear_v.weight", model.encoder.d_model, model.encoder.d_model);
        expect_tensor_shape(model, prefix + "self_attn.linear_v.bias", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "self_attn.linear_pos.weight", model.encoder.d_model, model.encoder.d_model);
        expect_tensor_shape(model, prefix + "self_attn.linear_out.weight", model.encoder.d_model, model.encoder.d_model);
        expect_tensor_shape(model, prefix + "self_attn.linear_out.bias", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "self_attn.pos_bias_u", head_dim, model.encoder.n_heads);
        expect_tensor_shape(model, prefix + "self_attn.pos_bias_v", head_dim, model.encoder.n_heads);
        expect_tensor_shape(model, prefix + "norm_conv.weight", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "norm_conv.bias", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "conv.pointwise_conv1.weight", 1, model.encoder.d_model, model.encoder.d_model * 2);
        expect_tensor_shape(model, prefix + "conv.pointwise_conv1.bias", model.encoder.d_model * 2);
        expect_tensor_shape(model, prefix + "conv.depthwise_conv.weight", model.encoder.conv_kernel_size, 1, model.encoder.d_model);
        expect_tensor_shape(model, prefix + "conv.depthwise_conv.bias", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "conv.pointwise_conv2.weight", 1, model.encoder.d_model, model.encoder.d_model);
        expect_tensor_shape(model, prefix + "conv.pointwise_conv2.bias", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "norm_feed_forward2.weight", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "norm_feed_forward2.bias", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "feed_forward2.linear1.weight", model.encoder.d_model, d_ff);
        expect_tensor_shape(model, prefix + "feed_forward2.linear1.bias", d_ff);
        expect_tensor_shape(model, prefix + "feed_forward2.linear2.weight", d_ff, model.encoder.d_model);
        expect_tensor_shape(model, prefix + "feed_forward2.linear2.bias", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "norm_out.weight", model.encoder.d_model);
        expect_tensor_shape(model, prefix + "norm_out.bias", model.encoder.d_model);
    }

    for (int32_t il = 0; il < model.decoder.num_layers; ++il) {
        const std::string prefix = format("decoder.layers.%d.", il);
        require_tensor(model, prefix + "layer_norm_1.weight", GGML_TYPE_F32);
        require_tensor(model, prefix + "layer_norm_1.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "self_attn.query.weight");
        require_tensor(model, prefix + "self_attn.query.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "self_attn.key.weight");
        require_tensor(model, prefix + "self_attn.key.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "self_attn.value.weight");
        require_tensor(model, prefix + "self_attn.value.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "self_attn.out.weight");
        require_tensor(model, prefix + "self_attn.out.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "layer_norm_2.weight", GGML_TYPE_F32);
        require_tensor(model, prefix + "layer_norm_2.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "cross_attn.query.weight");
        require_tensor(model, prefix + "cross_attn.query.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "cross_attn.key.weight");
        require_tensor(model, prefix + "cross_attn.key.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "cross_attn.value.weight");
        require_tensor(model, prefix + "cross_attn.value.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "cross_attn.out.weight");
        require_tensor(model, prefix + "cross_attn.out.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "layer_norm_3.weight", GGML_TYPE_F32);
        require_tensor(model, prefix + "layer_norm_3.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "feed_forward.dense_in.weight");
        require_tensor(model, prefix + "feed_forward.dense_in.bias", GGML_TYPE_F32);
        require_tensor(model, prefix + "feed_forward.dense_out.weight");
        require_tensor(model, prefix + "feed_forward.dense_out.bias", GGML_TYPE_F32);

        expect_tensor_shape(model, prefix + "layer_norm_1.weight", model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "layer_norm_1.bias", model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "self_attn.query.weight", model.decoder.hidden_size, model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "self_attn.query.bias", model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "self_attn.key.weight", model.decoder.hidden_size, model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "self_attn.key.bias", model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "self_attn.value.weight", model.decoder.hidden_size, model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "self_attn.value.bias", model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "self_attn.out.weight", model.decoder.hidden_size, model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "self_attn.out.bias", model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "layer_norm_2.weight", model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "layer_norm_2.bias", model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "cross_attn.query.weight", model.decoder.hidden_size, model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "cross_attn.query.bias", model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "cross_attn.key.weight", model.decoder.hidden_size, model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "cross_attn.key.bias", model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "cross_attn.value.weight", model.decoder.hidden_size, model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "cross_attn.value.bias", model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "cross_attn.out.weight", model.decoder.hidden_size, model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "cross_attn.out.bias", model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "layer_norm_3.weight", model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "layer_norm_3.bias", model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "feed_forward.dense_in.weight", model.decoder.hidden_size, model.decoder.inner_size);
        expect_tensor_shape(model, prefix + "feed_forward.dense_in.bias", model.decoder.inner_size);
        expect_tensor_shape(model, prefix + "feed_forward.dense_out.weight", model.decoder.inner_size, model.decoder.hidden_size);
        expect_tensor_shape(model, prefix + "feed_forward.dense_out.bias", model.decoder.hidden_size);
    }

    if (model.has_encoder_decoder_proj) {
        require_tensor(model, "encoder_decoder_proj.weight");
        require_tensor(model, "encoder_decoder_proj.bias", GGML_TYPE_F32);
        expect_tensor_shape(model, "encoder_decoder_proj.weight", model.encoder.d_model, model.decoder.hidden_size);
        expect_tensor_shape(model, "encoder_decoder_proj.bias", model.decoder.hidden_size);
    }
}

static void load_tokenizer_metadata(model & out) {
    const std::vector<std::string> tokens = gguf_get_string_array_or_throw(out.gguf, gguf_key("tokenizer.tokens"));
    const std::vector<std::string> supported_languages = gguf_get_string_array_or_throw(out.gguf, gguf_key("tokenizer.supported_languages"));
    const std::vector<std::string> special_names = gguf_get_string_array_or_throw(out.gguf, gguf_key("tokenizer.special_token_names"));
    const std::vector<int32_t> special_ids = gguf_get_scalar_array_or_throw<int32_t>(out.gguf, gguf_key("tokenizer.special_token_ids"), GGUF_TYPE_INT32);
    const std::vector<std::string> language_codes = gguf_get_string_array_or_throw(out.gguf, gguf_key("tokenizer.language_codes"));
    const std::vector<int32_t> language_ids = gguf_get_scalar_array_or_throw<int32_t>(out.gguf, gguf_key("tokenizer.language_token_ids"), GGUF_TYPE_INT32);

    if (special_names.size() != special_ids.size()) {
        throw std::runtime_error("special token metadata is inconsistent");
    }
    if (language_codes.size() != language_ids.size()) {
        throw std::runtime_error("language token metadata is inconsistent");
    }

    out.vocab.id_to_piece = tokens;
    out.vocab.supported_languages = supported_languages;

    for (size_t i = 0; i < special_names.size(); ++i) {
        out.vocab.special_token_ids[special_names[i]] = special_ids[i];
        out.vocab.special_ids.insert(special_ids[i]);
    }

    for (size_t i = 0; i < language_codes.size(); ++i) {
        out.vocab.language_token_ids[language_codes[i]] = language_ids[i];
        out.vocab.special_ids.insert(language_ids[i]);
    }
}

static void load_model_common_metadata(model & out) {
    const std::string architecture = gguf_get_str_or_throw(out.gguf, GGUF_ARCH_KEY);
    if (architecture != GGUF_ARCH_VALUE) {
        throw std::runtime_error(format(
                "unsupported architecture '%s' (expected '%s')",
                architecture.c_str(),
                GGUF_ARCH_VALUE));
    }

    out.max_audio_clip_s = gguf_get_f32_or_throw(out.gguf, gguf_key("max_audio_clip_s"));

    out.frontend.sample_rate = gguf_get_i32_or_throw(out.gguf, gguf_key("frontend.sample_rate"));
    out.frontend.n_mels = gguf_get_i32_or_throw(out.gguf, gguf_key("frontend.n_mels"));
    out.frontend.n_fft = gguf_get_i32_or_throw(out.gguf, gguf_key("frontend.n_fft"));
    out.frontend.win_length = gguf_get_i32_or_throw(out.gguf, gguf_key("frontend.win_length"));
    out.frontend.hop_length = gguf_get_i32_or_throw(out.gguf, gguf_key("frontend.hop_length"));
    out.frontend.fmin = gguf_get_f32_or_throw(out.gguf, gguf_key("frontend.fmin"));
    out.frontend.fmax = gguf_get_f32_or_throw(out.gguf, gguf_key("frontend.fmax"));
    out.frontend.preemph = gguf_get_f32_or_throw(out.gguf, gguf_key("frontend.preemph"));
    out.frontend.dither = gguf_get_f32_or_throw(out.gguf, gguf_key("frontend.dither"));
    out.frontend.log_zero_guard = gguf_get_f32_or_throw(out.gguf, gguf_key("frontend.log_zero_guard"));
    out.frontend.normalize_per_feature = gguf_get_bool_or_throw(out.gguf, gguf_key("frontend.normalize_per_feature"));
    if (gguf_has_key(out.gguf, gguf_key("frontend.window"))) {
        out.frontend.window = gguf_get_scalar_array_or_throw<float>(out.gguf, gguf_key("frontend.window"), GGUF_TYPE_FLOAT32);
    }
    out.frontend.mel_filters = gguf_get_scalar_array_or_throw<float>(out.gguf, gguf_key("frontend.mel_filters"), GGUF_TYPE_FLOAT32);

    out.encoder.d_model = gguf_get_i32_or_throw(out.gguf, gguf_key("encoder.d_model"));
    out.encoder.feat_in = gguf_get_i32_or_throw(out.gguf, gguf_key("encoder.feat_in"));
    out.encoder.feat_out = gguf_get_i32_or_throw(out.gguf, gguf_key("encoder.feat_out"));
    out.encoder.n_layers = gguf_get_i32_or_throw(out.gguf, gguf_key("encoder.n_layers"));
    out.encoder.n_heads = gguf_get_i32_or_throw(out.gguf, gguf_key("encoder.n_heads"));
    out.encoder.ff_expansion_factor = gguf_get_i32_or_throw(out.gguf, gguf_key("encoder.ff_expansion_factor"));
    out.encoder.conv_kernel_size = gguf_get_i32_or_throw(out.gguf, gguf_key("encoder.conv_kernel_size"));
    out.encoder.subsampling_factor = gguf_get_i32_or_throw(out.gguf, gguf_key("encoder.subsampling_factor"));
    out.encoder.subsampling_conv_channels = gguf_get_i32_or_throw(out.gguf, gguf_key("encoder.subsampling_conv_channels"));
    out.encoder.pos_emb_max_len = gguf_get_i32_or_throw(out.gguf, gguf_key("encoder.pos_emb_max_len"));

    out.decoder.hidden_size = gguf_get_i32_or_throw(out.gguf, gguf_key("decoder.hidden_size"));
    out.decoder.inner_size = gguf_get_i32_or_throw(out.gguf, gguf_key("decoder.inner_size"));
    out.decoder.num_attention_heads = gguf_get_i32_or_throw(out.gguf, gguf_key("decoder.num_attention_heads"));
    out.decoder.num_layers = gguf_get_i32_or_throw(out.gguf, gguf_key("decoder.num_layers"));
    out.decoder.max_sequence_length = gguf_get_i32_or_throw(out.gguf, gguf_key("decoder.max_sequence_length"));
    out.decoder.hidden_act = gguf_get_str_or_throw(out.gguf, gguf_key("decoder.hidden_act"));

    out.head.hidden_size = gguf_get_i32_or_throw(out.gguf, gguf_key("head.hidden_size"));
    out.head.num_classes = gguf_get_i32_or_throw(out.gguf, gguf_key("head.num_classes"));
    out.head.log_softmax = gguf_get_bool_or_throw(out.gguf, gguf_key("head.log_softmax"));

    out.has_encoder_decoder_proj = gguf_get_bool_or_throw(out.gguf, gguf_key("encoder_decoder_proj"));

    load_tokenizer_metadata(out);
}

static void populate_model_tensor_map(model & out) {
    out.tensors.clear();
    const int64_t n_tensors = gguf_get_n_tensors(out.gguf);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(out.gguf, i);
        out.tensors[name] = ggml_get_tensor(out.weights, name);
    }
}

static backend_model_data * get_backend_model(model & model) {
    return reinterpret_cast<backend_model_data *>(model.impl);
}

static const backend_model_data * get_backend_model(const model & model) {
    return reinterpret_cast<const backend_model_data *>(model.impl);
}

static backend_state_data * get_backend_state(state & state) {
    return reinterpret_cast<backend_state_data *>(state.impl);
}

static const backend_state_data * get_backend_state(const state & state) {
    return reinterpret_cast<const backend_state_data *>(state.impl);
}

static bool runtime_uses_backend(const inference_runtime * runtime) {
    return runtime != nullptr &&
            runtime->backend_model != nullptr &&
            runtime->backend_state != nullptr &&
            !runtime->backend_state->legacy_cpu &&
            runtime->backend_model->use_gpu_active &&
            !runtime->backend_state->backends.empty();
}

static struct ggml_tensor * maybe_backend_tensor(
        inference_runtime * runtime,
        const struct ggml_tensor * tensor) {
    if (tensor == nullptr || !runtime_uses_backend(runtime)) {
        return const_cast<struct ggml_tensor *>(tensor);
    }

    const std::map<std::string, struct ggml_tensor *>::const_iterator it =
            runtime->backend_model->tensors.find(tensor->name);
    if (it == runtime->backend_model->tensors.end()) {
        return const_cast<struct ggml_tensor *>(tensor);
    }

    return it->second;
}

static void free_backend_model(model & model);
static void free_backend_state(state & state);
static bool load_model_backend(const std::string & path_model, model & out, const context_params & params, std::string & error);
static bool init_state_backend(model & model, state & state, std::string & error);
static bool transcribe_backend(
        model & model,
        state & state,
        const std::vector<float> & pcmf32,
        const transcribe_params & params,
        std::string & text,
        std::string & error);

} // namespace

bool tokenizer::is_language_supported(const std::string & language) const {
    return language_token_ids.find(language) != language_token_ids.end();
}

int32_t tokenizer::special_token_id(const std::string & name) const {
    const std::map<std::string, int32_t>::const_iterator it = special_token_ids.find(name);
    if (it == special_token_ids.end()) {
        throw std::runtime_error(format("missing special token '%s'", name.c_str()));
    }
    return it->second;
}

std::vector<int32_t> tokenizer::build_prompt(const std::string & language, bool punctuation) const {
    const std::map<std::string, int32_t>::const_iterator lang_it = language_token_ids.find(language);
    if (lang_it == language_token_ids.end()) {
        throw std::runtime_error(format("unsupported language '%s'", language.c_str()));
    }

    std::vector<int32_t> prompt;
    const std::vector<std::string>::const_iterator space_it = std::find(id_to_piece.begin(), id_to_piece.end(), "▁");
    if (space_it != id_to_piece.end()) {
        prompt.push_back(int32_t(std::distance(id_to_piece.begin(), space_it)));
    }

    const std::vector<int32_t> tail{
        special_token_id("<|startofcontext|>"),
        special_token_id("<|startoftranscript|>"),
        special_token_id("<|emo:undefined|>"),
        lang_it->second,
        lang_it->second,
        special_token_id(punctuation ? "<|pnc|>" : "<|nopnc|>"),
        special_token_id("<|noitn|>"),
        special_token_id("<|notimestamp|>"),
        special_token_id("<|nodiarize|>"),
    };
    prompt.insert(prompt.end(), tail.begin(), tail.end());
    return prompt;
}

std::string tokenizer::detokenize(const std::vector<int32_t> & ids) const {
    std::string bytes;

    for (size_t i = 0; i < ids.size(); ++i) {
        const int32_t id = ids[i];
        if (special_ids.find(id) != special_ids.end()) {
            continue;
        }
        if (id < 0 || id >= (int32_t) id_to_piece.size()) {
            continue;
        }

        const std::string & piece = id_to_piece[size_t(id)];

        if (piece == "<unk>" || piece == "<s>" || piece == "</s>") {
            continue;
        }

        if (piece.compare(0, 3, "\xE2\x96\x81") == 0) {
            if (!bytes.empty()) {
                bytes.push_back(' ');
            }
            bytes.append(piece.substr(3));
            continue;
        }

        if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece[5] == '>') {
            const char hi = piece[3];
            const char lo = piece[4];
            const std::string hex = piece.substr(3, 2);
            const char byte = (char) strtol(hex.c_str(), nullptr, 16);
            GGML_UNUSED(hi);
            GGML_UNUSED(lo);
            bytes.push_back(byte);
            continue;
        }

        bytes.append(piece);
    }

    return sanitize_utf8(bytes);
}

namespace {

static void clear_backend_model_data(backend_model_data & backend_model) {
    for (ggml_backend_buffer_t buffer : backend_model.buffers) {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }
    backend_model.buffers.clear();
    backend_model.tensors.clear();

    for (struct ggml_context * ctx : backend_model.ctxs) {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
    backend_model.ctxs.clear();

    backend_model.device = nullptr;
    backend_model.use_gpu_active = false;
    backend_model.backend_name = "cpu";
}

static ggml_backend_dev_t find_requested_gpu_device(int32_t gpu_device) {
    int32_t cur_gpu = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        if (cur_gpu == gpu_device) {
            return dev;
        }

        ++cur_gpu;
    }

    return nullptr;
}

static ggml_backend_t cohere_backend_init_gpu(const context_params & params) {
    ggml_backend_dev_t dev = find_requested_gpu_device(params.gpu_device);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        return nullptr;
    }

    return backend;
}

static std::vector<ggml_backend_t> cohere_backend_init(const context_params & params) {
    std::vector<ggml_backend_t> backends;

    if (params.use_gpu) {
        if (ggml_backend_t backend_gpu = cohere_backend_init_gpu(params)) {
            backends.push_back(backend_gpu);
        }
    }

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            if (ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr)) {
                backends.push_back(backend);
            }
        }
    }

    if (ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr)) {
        backends.push_back(backend_cpu);
    }

    return backends;
}

static bool duplicate_model_tensors_to_backend(
        model & out,
        backend_model_data & backend_model,
        ggml_backend_dev_t device,
        std::string & error) {
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(device);
    if (buft == nullptr) {
        error = format("backend '%s' does not expose a default buffer type", ggml_backend_dev_name(device));
        return false;
    }

    const size_t n_tensors = std::max<size_t>(1, out.tensors.size());
    struct ggml_init_params init_params = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(init_params);
    if (ctx == nullptr) {
        error = "failed to allocate Cohere backend tensor context";
        return false;
    }

    backend_model.ctxs.push_back(ctx);

    for (std::map<std::string, struct ggml_tensor *>::const_iterator it = out.tensors.begin(); it != out.tensors.end(); ++it) {
        struct ggml_tensor * dup = ggml_dup_tensor(ctx, it->second);
        if (dup == nullptr) {
            error = format("failed to duplicate tensor '%s' for backend loading", it->first.c_str());
            return false;
        }
        ggml_set_name(dup, it->first.c_str());
        backend_model.tensors[it->first] = dup;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (buffer == nullptr) {
        error = format("failed to allocate backend weights buffer for '%s'", ggml_backend_dev_name(device));
        return false;
    }
    backend_model.buffers.push_back(buffer);

    for (std::map<std::string, struct ggml_tensor *>::const_iterator it = out.tensors.begin(); it != out.tensors.end(); ++it) {
        struct ggml_tensor * src = it->second;
        struct ggml_tensor * dst = backend_model.tensors[it->first];
        if (src == nullptr || src->data == nullptr || dst == nullptr) {
            error = format("failed to resolve tensor '%s' during backend copy", it->first.c_str());
            return false;
        }
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    }

    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    backend_model.device = device;
    backend_model.backend_name = ggml_backend_dev_name(device);
    backend_model.use_gpu_active = true;
    backend_model.max_valid_mel_frames = std::max(
            1,
            int32_t((out.max_audio_clip_s * out.frontend.sample_rate) / std::max(1, out.frontend.hop_length)));
    backend_model.max_mel_frames = backend_model.max_valid_mel_frames;
    backend_model.max_encoder_valid_length = std::min(
            conv_subsampling_output_length(backend_model.max_valid_mel_frames),
            out.encoder.pos_emb_max_len);
    backend_model.max_encoder_length = backend_model.max_encoder_valid_length;
    backend_model.cross_hidden_size = out.has_encoder_decoder_proj ? out.decoder.hidden_size : out.encoder.d_model;

    return true;
}

static void free_backend_model(model & model) {
    backend_model_data * backend_model = get_backend_model(model);
    if (backend_model == nullptr) {
        return;
    }

    clear_backend_model_data(*backend_model);
    delete backend_model;
    model.impl = nullptr;
}

static bool load_model_backend(const std::string & path_model, model & out, const context_params & params, std::string & error) {
    if (!cohere::load_model(path_model, out, error)) {
        return false;
    }

    std::unique_ptr<backend_model_data> backend_model(new backend_model_data());
    backend_model->params = params;
    backend_model->path_model = path_model;
    backend_model->backend_name = "cpu";

    if (params.use_gpu) {
        ggml_backend_load_all();
        if (ggml_backend_dev_t device = find_requested_gpu_device(params.gpu_device)) {
            std::string backend_error;
            if (!duplicate_model_tensors_to_backend(out, *backend_model, device, backend_error)) {
                clear_backend_model_data(*backend_model);
            }
        }
    }

    out.impl = backend_model.release();
    error.clear();
    return true;
}

static void free_backend_state(state & state) {
    backend_state_data * backend_state = get_backend_state(state);
    if (backend_state == nullptr) {
        return;
    }

    if (backend_state->sched_conv.sched != nullptr) {
        ggml_backend_sched_free(backend_state->sched_conv.sched);
    }
    if (backend_state->sched_encode.sched != nullptr) {
        ggml_backend_sched_free(backend_state->sched_encode.sched);
    }
    if (backend_state->sched_cross.sched != nullptr) {
        ggml_backend_sched_free(backend_state->sched_cross.sched);
    }
    if (backend_state->sched_decode.sched != nullptr) {
        ggml_backend_sched_free(backend_state->sched_decode.sched);
    }

    if (backend_state->kv_self.buffer != nullptr) {
        ggml_backend_buffer_free(backend_state->kv_self.buffer);
    }
    if (backend_state->kv_cross.buffer != nullptr) {
        ggml_backend_buffer_free(backend_state->kv_cross.buffer);
    }
    if (backend_state->tensor_buffer != nullptr) {
        ggml_backend_buffer_free(backend_state->tensor_buffer);
    }

    for (ggml_backend_t backend : backend_state->backends) {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }

    delete backend_state;
    state.impl = nullptr;
}

static bool init_state_backend(model & model, state & state, std::string & error) {
    free_backend_state(state);

    std::unique_ptr<backend_state_data> backend_state(new backend_state_data());
    const backend_model_data * backend_model = get_backend_model(model);

    if (backend_model == nullptr || !backend_model->use_gpu_active || !backend_model->params.use_gpu) {
        backend_state->legacy_cpu = true;
        backend_state->backend_name = "cpu";
        state.impl = backend_state.release();
        error.clear();
        return true;
    }

    ggml_backend_load_all();
    backend_state->params = backend_model->params;
    backend_state->backends = cohere_backend_init(backend_model->params);

    bool have_gpu_backend = false;
    for (ggml_backend_t backend : backend_state->backends) {
        if (backend == nullptr) {
            continue;
        }
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(ggml_backend_get_device(backend));
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            have_gpu_backend = true;
            backend_state->backend_name = ggml_backend_name(backend);
            break;
        }
    }

    if (!have_gpu_backend) {
        for (ggml_backend_t backend : backend_state->backends) {
            if (backend != nullptr) {
                ggml_backend_free(backend);
            }
        }
        backend_state->backends.clear();
        backend_state->legacy_cpu = true;
        backend_state->backend_name = "cpu";
    } else {
        backend_state->legacy_cpu = false;
    }

    state.impl = backend_state.release();
    error.clear();
    return true;
}

static bool transcribe_backend(
        model & model,
        state & state,
        const std::vector<float> & pcmf32,
        const transcribe_params & params,
        std::string & text,
        std::string & error);

} // namespace

bool load_model(const std::string & path_model, model & out, std::string & error) {
    free_model(out);

    try {
        struct ggml_context * weights = nullptr;
        struct gguf_init_params params = {
            /*.no_alloc =*/ false,
            /*.ctx      =*/ &weights,
        };

        out.gguf = gguf_init_from_file(path_model.c_str(), params);
        out.weights = weights;

        if (out.gguf == nullptr || out.weights == nullptr) {
            throw std::runtime_error(format("failed to load GGUF model '%s'", path_model.c_str()));
        }

        load_model_common_metadata(out);
        populate_model_tensor_map(out);

        validate_model(out);
        return true;
    } catch (const std::exception & ex) {
        error = ex.what();
        free_model(out);
        return false;
    }
}

context_params context_default_params() {
    context_params params;
    params.use_gpu = true;
    params.flash_attn = true;
    params.gpu_device = 0;
    return params;
}

bool load_model(const std::string & path_model, model & out, const context_params & params, std::string & error) {
    return load_model_backend(path_model, out, params, error);
}

void free_model(model & model) {
    free_backend_model(model);
    model.tensors.clear();
    if (model.weights != nullptr) {
        ggml_free(model.weights);
        model.weights = nullptr;
    }
    if (model.gguf != nullptr) {
        gguf_free(model.gguf);
        model.gguf = nullptr;
    }
    model.frontend = frontend_config();
    model.encoder = encoder_config();
    model.decoder = decoder_config();
    model.head = head_config();
    model.vocab = tokenizer();
    model.max_audio_clip_s = 0.0f;
    model.has_encoder_decoder_proj = false;
    model.impl = nullptr;
}

int32_t conv_subsampling_output_length(int32_t n_frames) {
    int32_t length = n_frames;
    for (int i = 0; i < 3; ++i) {
        length = (length + 2 - 3) / 2 + 1;
    }
    return std::max(length, 1);
}

std::vector<float> rel_shift_reference(
        const std::vector<float> & input,
        int32_t n_head,
        int32_t q_len,
        int32_t pos_len) {
    std::vector<float> out(input.size(), 0.0f);
    for (int32_t h = 0; h < n_head; ++h) {
        for (int32_t q = 0; q < q_len; ++q) {
            for (int32_t p = 0; p < pos_len; ++p) {
                const int32_t src = p + q_len - 1 - q;
                if (src >= 0 && src < pos_len) {
                    const size_t dst_idx = size_t(h) * size_t(q_len) * size_t(pos_len) + size_t(q) * size_t(pos_len) + size_t(p);
                    const size_t src_idx = size_t(h) * size_t(q_len) * size_t(pos_len) + size_t(q) * size_t(pos_len) + size_t(src);
                    out[dst_idx] = input[src_idx];
                }
            }
        }
    }
    return out;
}

static bool run_inference(
        model & model,
        const std::vector<float> & pcmf32,
        const transcribe_params & params,
        std::string & text,
        std::string & error,
        backend_state_data * backend_state = nullptr) {
    text.clear();
    error.clear();

    try {
        if (!model.vocab.is_language_supported(params.language)) {
            throw std::runtime_error(format("unsupported language '%s'", params.language.c_str()));
        }

        if (pcmf32.empty()) {
            throw std::runtime_error("audio is empty");
        }

        const float duration_s = float(pcmf32.size()) / float(model.frontend.sample_rate);
        if (model.max_audio_clip_s > 0.0f && duration_s > model.max_audio_clip_s) {
            throw std::runtime_error(format(
                    "audio is too long for v0 (%0.2fs > max %0.2fs); long-form chunking is not implemented",
                    duration_s,
                    model.max_audio_clip_s));
        }

        const int32_t n_threads = std::max(1, params.n_threads);
        configure_blas_threads_for_inference(n_threads);
        inference_runtime runtime;
        runtime.backend_model = get_backend_model(model);
        runtime.backend_state = backend_state;
        if (!runtime_uses_backend(&runtime)) {
            runtime.backend_model = nullptr;
            runtime.backend_state = nullptr;
        }
        int32_t encoder_length = 0;
        tensor2d decoder_memory = run_encoder(model, pcmf32, n_threads, encoder_length, &runtime);
        std::vector<cross_kv_cache> cross_kv;
        cross_kv = build_cross_kv(model, decoder_memory, n_threads, &runtime);
        std::vector<self_kv_cache> self_kv = build_self_kv(model);

        const std::vector<int32_t> prompt = model.vocab.build_prompt(params.language, params.punctuation);
        const int32_t eos_token_id = model.vocab.special_token_id("<|endoftext|>");

        if ((int32_t) prompt.size() > model.decoder.max_sequence_length) {
            throw std::runtime_error("decoder prompt exceeds decoder max sequence length");
        }

        std::vector<int32_t> generated;
        generated.reserve(size_t(params.max_new_tokens));

        int32_t current_position = 0;
        std::vector<float> logits;
        for (size_t i = 0; i < prompt.size(); ++i) {
            const bool need_logits = (i + 1 == prompt.size());
            logits = decoder_step(model, prompt[i], current_position++, cross_kv, self_kv, n_threads, need_logits, &runtime);
        }

        const int32_t remaining = std::max(0, model.decoder.max_sequence_length - current_position);
        const int32_t max_new_tokens = std::min(params.max_new_tokens, remaining);
        for (int32_t i = 0; i < max_new_tokens; ++i) {
            const int32_t next = argmax(logits);
            if (next == eos_token_id) {
                break;
            }
            generated.push_back(next);
            logits = decoder_step(model, next, current_position++, cross_kv, self_kv, n_threads, true, &runtime);
        }

        text = model.vocab.detokenize(generated);
        GGML_UNUSED(encoder_length);
        return true;
    } catch (const std::exception & ex) {
        error = ex.what();
        return false;
    }
}

bool transcribe(
        model & model,
        const std::vector<float> & pcmf32,
        const transcribe_params & params,
        std::string & text,
        std::string & error) {
    return run_inference(model, pcmf32, params, text, error);
}

bool init_state(model & model, state & state, std::string & error) {
    return init_state_backend(model, state, error);
}

void free_state(state & state) {
    free_backend_state(state);
}

namespace {

static bool transcribe_backend(
        model & model,
        state & state,
        const std::vector<float> & pcmf32,
        const transcribe_params & params,
        std::string & text,
        std::string & error) {
    backend_state_data * backend_state = get_backend_state(state);
    if (backend_state == nullptr || backend_state->legacy_cpu) {
        return run_inference(model, pcmf32, params, text, error);
    }

    return run_inference(model, pcmf32, params, text, error, backend_state);
}

} // namespace

bool transcribe_with_state(
        model & model,
        state & state,
        const std::vector<float> & pcmf32,
        const transcribe_params & params,
        std::string & text,
        std::string & error) {
    if (state.impl == nullptr) {
        if (!init_state(model, state, error)) {
            return false;
        }
    }

    return transcribe_backend(model, state, pcmf32, params, text, error);
}

} // namespace cohere
