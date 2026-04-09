#include "cohere.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace cohere {
namespace {

static const char * GGUF_ARCH_KEY = "general.architecture";
static const char * GGUF_ARCH_VALUE = "cohere-transcribe";
static const char * GGUF_PREFIX = "cohere_transcribe.";
static const float  LAYER_NORM_EPS = 1e-5f;
static const size_t SCRATCH_SIZE = 512ull * 1024ull * 1024ull;

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

static ggml_ptr make_compute_ctx(size_t mem_size = SCRATCH_SIZE) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to allocate ggml compute context");
    }

    return ggml_ptr(ctx);
}

static void run_graph(struct ggml_context * ctx, struct ggml_tensor * output, int32_t n_threads) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);

    if (ggml_graph_compute_with_ctx(ctx, gf, n_threads) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml graph execution failed");
    }
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

static float tensor_get_f32(const struct ggml_tensor * tensor, int32_t i0, int32_t i1 = 0, int32_t i2 = 0, int32_t i3 = 0) {
    return ggml_get_f32_nd(tensor, i0, i1, i2, i3);
}

static tensor2d tensor_from_ggml_2d(const struct ggml_tensor * tensor) {
    tensor2d out((int32_t) tensor->ne[0], (int32_t) tensor->ne[1]);
    for (int32_t j = 0; j < out.n1; ++j) {
        for (int32_t i = 0; i < out.n0; ++i) {
            out.at(i, j) = tensor_get_f32(tensor, i, j, 0, 0);
        }
    }
    return out;
}

static tensor4d tensor_from_ggml_4d(const struct ggml_tensor * tensor) {
    tensor4d out((int32_t) tensor->ne[0], (int32_t) tensor->ne[1], (int32_t) tensor->ne[2], (int32_t) tensor->ne[3]);
    for (int32_t n = 0; n < out.n3; ++n) {
        for (int32_t c = 0; c < out.n2; ++c) {
            for (int32_t y = 0; y < out.n1; ++y) {
                for (int32_t x = 0; x < out.n0; ++x) {
                    out.at(x, y, c, n) = tensor_get_f32(tensor, x, y, c, n);
                }
            }
        }
    }
    return out;
}

static matrix_output matrix_output_from_tensor2d(const tensor2d & tensor) {
    matrix_output out;
    out.rows = tensor.n0;
    out.cols = tensor.n1;
    out.data.resize(size_t(out.rows) * size_t(out.cols));
    for (int32_t row = 0; row < out.rows; ++row) {
        for (int32_t col = 0; col < out.cols; ++col) {
            out.data[size_t(row) * size_t(out.cols) + size_t(col)] = tensor.at(row, col);
        }
    }
    return out;
}

static struct ggml_tensor * create_input_tensor_2d(struct ggml_context * ctx, const tensor2d & input) {
    struct ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input.n0, input.n1);
    if (tensor == nullptr) {
        throw std::runtime_error("failed to allocate ggml input tensor");
    }
    std::memcpy(tensor->data, input.data.data(), input.data.size() * sizeof(float));
    return tensor;
}

static struct ggml_tensor * create_input_tensor_4d(struct ggml_context * ctx, const tensor4d & input) {
    struct ggml_tensor * tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, input.n0, input.n1, input.n2, input.n3);
    if (tensor == nullptr) {
        throw std::runtime_error("failed to allocate ggml input tensor");
    }
    std::memcpy(tensor->data, input.data.data(), input.data.size() * sizeof(float));
    return tensor;
}

static tensor2d eval_linear(struct ggml_tensor * weight, const struct ggml_tensor * bias, const tensor2d & input, int32_t n_threads) {
    ggml_ptr ctx = make_compute_ctx();
    struct ggml_tensor * src = create_input_tensor_2d(ctx.get(), input);
    struct ggml_tensor * cur = ggml_mul_mat(ctx.get(), weight, src);
    run_graph(ctx.get(), cur, n_threads);

    tensor2d out = tensor_from_ggml_2d(cur);

    if (bias != nullptr) {
        for (int32_t col = 0; col < out.n1; ++col) {
            for (int32_t row = 0; row < out.n0; ++row) {
                out.at(row, col) += tensor_get_f32(bias, row);
            }
        }
    }

    return out;
}

static tensor2d eval_linear_from_rank3_weight(
        struct ggml_tensor * weight,
        const struct ggml_tensor * bias,
        const tensor2d & input,
        int32_t n_threads) {
    ggml_ptr ctx = make_compute_ctx();
    struct ggml_tensor * src = create_input_tensor_2d(ctx.get(), input);
    struct ggml_tensor * w2 = ggml_reshape_2d(ctx.get(), weight, weight->ne[0] * weight->ne[1], weight->ne[2]);
    struct ggml_tensor * cur = ggml_mul_mat(ctx.get(), w2, src);
    run_graph(ctx.get(), cur, n_threads);

    tensor2d out = tensor_from_ggml_2d(cur);

    if (bias != nullptr) {
        for (int32_t col = 0; col < out.n1; ++col) {
            for (int32_t row = 0; row < out.n0; ++row) {
                out.at(row, col) += tensor_get_f32(bias, row);
            }
        }
    }

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
        int32_t n_threads) {
    (void) n_threads;

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

static tensor4d eval_conv2d_dw(
        struct ggml_tensor * weight,
        const struct ggml_tensor * bias,
        const tensor4d & input,
        int32_t stride_x,
        int32_t stride_y,
        int32_t pad_x,
        int32_t pad_y,
        int32_t n_threads) {
    (void) n_threads;

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
            const float w = tensor_get_f32(weight, row);
            const float b = bias ? tensor_get_f32(bias, row) : 0.0f;
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
        int32_t n_threads) {
    tensor2d cur = eval_linear(w1, b1, x, n_threads);
    apply_silu(cur);
    return eval_linear(w2, b2, cur, n_threads);
}

static tensor2d decoder_feed_forward(
        const tensor2d & x,
        struct ggml_tensor * w1,
        const struct ggml_tensor * b1,
        struct ggml_tensor * w2,
        const struct ggml_tensor * b2,
        const std::string & act,
        int32_t n_threads) {
    tensor2d cur = eval_linear(w1, b1, x, n_threads);
    apply_activation(cur, act);
    return eval_linear(w2, b2, cur, n_threads);
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
        const struct ggml_tensor * bias) {
    const int32_t kernel = (int32_t) weight->ne[0];
    const int32_t channels = x.n0;
    const int32_t pad = kernel / 2;

    tensor2d out(x.n0, x.n1);

    for (int32_t c = 0; c < channels; ++c) {
        const float b = bias ? tensor_get_f32(bias, c) : 0.0f;
        for (int32_t t = 0; t < x.n1; ++t) {
            float sum = b;
            for (int32_t k = 0; k < kernel; ++k) {
                const int32_t src_t = t + k - pad;
                if (src_t < 0 || src_t >= x.n1) {
                    continue;
                }
                sum += tensor_get_f32(weight, k, 0, c) * x.at(c, src_t);
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
        int32_t n_threads) {
    const int32_t hidden = x.n0;
    const int32_t length = x.n1;
    const int32_t head_dim = hidden / n_heads;
    const float scale = 1.0f / std::sqrt(float(head_dim));

    tensor2d q = eval_linear(q_weight, q_bias, x, n_threads);
    tensor2d k = eval_linear(k_weight, k_bias, x, n_threads);
    tensor2d v = eval_linear(v_weight, v_bias, x, n_threads);
    tensor2d p = eval_linear(pos_weight, nullptr, build_relative_positional_encoding(hidden, length), n_threads);

    tensor2d attn(hidden, length);

    std::vector<float> scores(length);
    std::vector<float> probs(length);

    for (int32_t h = 0; h < n_heads; ++h) {
        const int32_t base = h * head_dim;

        for (int32_t q_pos = 0; q_pos < length; ++q_pos) {
            float max_score = -std::numeric_limits<float>::infinity();

            for (int32_t k_pos = 0; k_pos < length; ++k_pos) {
                float ac = 0.0f;
                float bd = 0.0f;
                const int32_t pos_index = (length - 1) - q_pos + k_pos;

                for (int32_t d = 0; d < head_dim; ++d) {
                    const int32_t idx = base + d;
                    ac += (q.at(idx, q_pos) + tensor_get_f32(pos_bias_u, d, h)) * k.at(idx, k_pos);
                    bd += (q.at(idx, q_pos) + tensor_get_f32(pos_bias_v, d, h)) * p.at(idx, pos_index);
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

    return eval_linear(out_weight, out_bias, attn, n_threads);
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
        int32_t n_threads) {
    tensor2d cur = eval_linear_from_rank3_weight(pointwise1_w, pointwise1_b, x, n_threads);
    cur = glu(cur);
    zero_masked_positions(cur, valid_length);
    cur = depthwise_conv1d(cur, depthwise_w, depthwise_b);
    apply_silu(cur);
    return eval_linear_from_rank3_weight(pointwise2_w, pointwise2_b, cur, n_threads);
}

struct conformer_layer_trace {
    tensor2d after_ff1;
    tensor2d after_attn;
    tensor2d after_conv;
    tensor2d out;
};

static tensor2d run_conformer_layer(
        const model & model,
        int32_t layer_idx,
        const tensor2d & input,
        int32_t valid_length,
        int32_t n_threads,
        conformer_layer_trace * trace = nullptr) {
    const std::string prefix = format("encoder.layers.%d.", layer_idx);

    tensor2d cur = input;

    {
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
                n_threads);

        cur = add_scaled(cur, ff_out, 0.5f);
        if (trace != nullptr) {
            trace->after_ff1 = cur;
        }
    }

    {
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
                n_threads);

        cur = add_scaled(cur, att_out, 1.0f);
        if (trace != nullptr) {
            trace->after_attn = cur;
        }
    }

    {
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
                n_threads);

        cur = add_scaled(cur, conv_out, 1.0f);
        if (trace != nullptr) {
            trace->after_conv = cur;
        }
    }

    {
        tensor2d ff_in = layer_norm(
                cur,
                require_tensor(model, prefix + "norm_feed_forward2.weight", GGML_TYPE_F32),
                require_tensor(model, prefix + "norm_feed_forward2.bias", GGML_TYPE_F32));

        tensor2d ff_out = feed_forward(
                ff_in,
                require_tensor(model, prefix + "feed_forward2.linear1.weight"),
                require_tensor(model, prefix + "feed_forward2.linear1.bias", GGML_TYPE_F32),
                require_tensor(model, prefix + "feed_forward2.linear2.weight"),
                require_tensor(model, prefix + "feed_forward2.linear2.bias", GGML_TYPE_F32),
                n_threads);

        cur = add_scaled(cur, ff_out, 0.5f);
    }

    tensor2d out = layer_norm(
            cur,
            require_tensor(model, prefix + "norm_out.weight", GGML_TYPE_F32),
            require_tensor(model, prefix + "norm_out.bias", GGML_TYPE_F32));
    if (trace != nullptr) {
        trace->out = out;
    }
    return out;
}

static tensor2d run_conv_subsampling(
        const model & model,
        const tensor2d & mel,
        int32_t valid_frames,
        int32_t n_threads) {
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
            2, 2, 1, 1, n_threads);
    current_length = conv_output_length_1d(current_length, 3, 2, 2);
    zero_masked_time_positions(x, current_length);
    apply_relu(x);

    x = eval_conv2d_dw(
            require_tensor(model, "encoder.pre_encode.conv1_dw.weight"),
            require_tensor(model, "encoder.pre_encode.conv1_dw.bias", GGML_TYPE_F32),
            x,
            2, 2, 1, 1, n_threads);
    current_length = conv_output_length_1d(current_length, 3, 2, 2);
    zero_masked_time_positions(x, current_length);
    x = eval_conv2d(
            require_tensor(model, "encoder.pre_encode.conv1_pw.weight"),
            require_tensor(model, "encoder.pre_encode.conv1_pw.bias", GGML_TYPE_F32),
            x,
            1, 1, 0, 0, n_threads);
    current_length = conv_output_length_1d(current_length, 1, 1, 0);
    zero_masked_time_positions(x, current_length);
    apply_relu(x);

    x = eval_conv2d_dw(
            require_tensor(model, "encoder.pre_encode.conv2_dw.weight"),
            require_tensor(model, "encoder.pre_encode.conv2_dw.bias", GGML_TYPE_F32),
            x,
            2, 2, 1, 1, n_threads);
    current_length = conv_output_length_1d(current_length, 3, 2, 2);
    zero_masked_time_positions(x, current_length);
    x = eval_conv2d(
            require_tensor(model, "encoder.pre_encode.conv2_pw.weight"),
            require_tensor(model, "encoder.pre_encode.conv2_pw.bias", GGML_TYPE_F32),
            x,
            1, 1, 0, 0, n_threads);
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
            n_threads);
}

static std::vector<float> build_hann_window(int32_t length) {
    std::vector<float> out(length);
    for (int32_t i = 0; i < length; ++i) {
        out[i] = 0.5f * (1.0f - std::cos((2.0f * float(M_PI) * float(i)) / float(length - 1)));
    }
    return out;
}

static void dft(const std::vector<float> & in, int32_t n, std::vector<float> & out) {
    out.assign(size_t(n) * 2u, 0.0f);
    for (int32_t k = 0; k < n; ++k) {
        double re = 0.0;
        double im = 0.0;
        for (int32_t t = 0; t < n; ++t) {
            const double phase = (2.0 * M_PI * double(k) * double(t)) / double(n);
            re += double(in[t]) * std::cos(phase);
            im -= double(in[t]) * std::sin(phase);
        }
        out[size_t(2 * k + 0)] = float(re);
        out[size_t(2 * k + 1)] = float(im);
    }
}

static tensor2d compute_mel_features(const std::vector<float> & audio, const frontend_config & cfg) {
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

    tensor2d mel(cfg.n_mels, n_frames);
    std::vector<float> frame(size_t(cfg.n_fft), 0.0f);
    std::vector<float> fft_out;

    for (int32_t frame_idx = 0; frame_idx < n_frames; ++frame_idx) {
        std::fill(frame.begin(), frame.end(), 0.0f);

        const int32_t start = frame_idx * cfg.hop_length;

        for (int32_t i = 0; i < cfg.win_length; ++i) {
            const int32_t src = start + offset + i;
            if (src >= 0 && src < (int32_t) padded.size()) {
                frame[size_t(offset + i)] = padded[size_t(src)] * window[size_t(i)];
            }
        }

        dft(frame, cfg.n_fft, fft_out);

        std::vector<float> power(size_t(n_freqs), 0.0f);
        for (int32_t k = 0; k < n_freqs; ++k) {
            const float re = fft_out[size_t(2 * k + 0)];
            const float im = fft_out[size_t(2 * k + 1)];
            power[size_t(k)] = re * re + im * im;
        }

        for (int32_t m = 0; m < cfg.n_mels; ++m) {
            double sum = 0.0;
            const size_t off = size_t(m) * size_t(n_freqs);
            for (int32_t k = 0; k < n_freqs; ++k) {
                sum += double(cfg.mel_filters[off + size_t(k)]) * double(power[size_t(k)]);
            }
            mel.at(m, frame_idx) = std::log(float(sum) + cfg.log_zero_guard);
        }
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

    for (int32_t t = valid_frames; t < n_frames; ++t) {
        for (int32_t m = 0; m < cfg.n_mels; ++m) {
            mel.at(m, t) = 0.0f;
        }
    }

    return mel;
}

struct encoder_trace {
    tensor2d mel;
    tensor2d subsampling_out;
    tensor2d block0_after_ff1;
    tensor2d block0_after_attn;
    tensor2d block0_after_conv;
    tensor2d block0_out;
    tensor2d encoder_out;
    tensor2d encoder_projected;
    int32_t encoder_length = 0;
};

static tensor2d project_encoder_for_decoder(model & model, const tensor2d & encoder_out, int32_t n_threads);

static encoder_trace run_encoder_trace(model & model, const std::vector<float> & pcmf32, int32_t n_threads) {
    encoder_trace trace;
    trace.mel = compute_mel_features(pcmf32, model.frontend);
    const int32_t valid_mel_frames = std::max<int32_t>(1, int32_t(pcmf32.size() / size_t(model.frontend.hop_length)));
    trace.encoder_length = conv_subsampling_output_length(valid_mel_frames);

    trace.subsampling_out = run_conv_subsampling(model, trace.mel, valid_mel_frames, n_threads);
    tensor2d cur = trace.subsampling_out;
    for (int32_t il = 0; il < model.encoder.n_layers; ++il) {
        conformer_layer_trace layer_trace;
        cur = run_conformer_layer(
                model,
                il,
                cur,
                trace.encoder_length,
                n_threads,
                il == 0 ? &layer_trace : nullptr);
        if (il == 0) {
            trace.block0_after_ff1 = layer_trace.after_ff1;
            trace.block0_after_attn = layer_trace.after_attn;
            trace.block0_after_conv = layer_trace.after_conv;
            trace.block0_out = layer_trace.out;
        }
    }

    trace.encoder_out = cur;
    trace.encoder_projected = project_encoder_for_decoder(model, trace.encoder_out, n_threads);
    return trace;
}

static tensor2d project_encoder_for_decoder(model & model, const tensor2d & encoder_out, int32_t n_threads) {
    if (!model.has_encoder_decoder_proj) {
        return encoder_out;
    }

    return eval_linear(
            require_tensor(model, "encoder_decoder_proj.weight"),
            require_tensor(model, "encoder_decoder_proj.bias", GGML_TYPE_F32),
            encoder_out,
            n_threads);
}

static tensor2d run_encoder(model & model, const std::vector<float> & pcmf32, int32_t n_threads, int32_t & encoder_length) {
    encoder_trace trace = run_encoder_trace(model, pcmf32, n_threads);
    encoder_length = trace.encoder_length;
    return trace.encoder_projected;
}

static tensor2d get_decoder_embedding(const model & model, int32_t token_id, int32_t position) {
    const struct ggml_tensor * token_embedding = require_tensor(model, "decoder.embedding.token_embedding.weight");
    tensor2d out(model.decoder.hidden_size, 1);

    const float pos_scale = 1.0f / std::sqrt(float(model.decoder.hidden_size));
    for (int32_t i = 0; i < model.decoder.hidden_size; ++i) {
        float value = tensor_get_f32(token_embedding, i, token_id);
        if ((i % 2) == 0) {
            const float div = std::exp(-(std::log(10000.0f) / float(model.decoder.hidden_size)) * float(i));
            value += std::sin(float(position) * div) * pos_scale;
        } else {
            const float div = std::exp(-(std::log(10000.0f) / float(model.decoder.hidden_size)) * float(i - 1));
            value += std::cos(float(position) * div) * pos_scale;
        }
        out.at(i, 0) = value;
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

    return out;
}

static std::vector<cross_kv_cache> build_cross_kv(
        model & model,
        const tensor2d & encoder_states,
        int32_t n_threads) {
    std::vector<cross_kv_cache> caches(size_t(model.decoder.num_layers));
    for (int32_t il = 0; il < model.decoder.num_layers; ++il) {
        const std::string prefix = format("decoder.layers.%d.cross_attn.", il);
        caches[size_t(il)].key = eval_linear(
                require_tensor(model, prefix + "key.weight"),
                require_tensor(model, prefix + "key.bias", GGML_TYPE_F32),
                encoder_states,
                n_threads);
        caches[size_t(il)].value = eval_linear(
                require_tensor(model, prefix + "value.weight"),
                require_tensor(model, prefix + "value.bias", GGML_TYPE_F32),
                encoder_states,
                n_threads);
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
        int32_t n_threads) {
    tensor2d hidden = get_decoder_embedding(model, token_id, position);

    for (int32_t il = 0; il < model.decoder.num_layers; ++il) {
        const std::string prefix = format("decoder.layers.%d.", il);

        {
            tensor2d norm = layer_norm(
                    hidden,
                    require_tensor(model, prefix + "layer_norm_1.weight", GGML_TYPE_F32),
                    require_tensor(model, prefix + "layer_norm_1.bias", GGML_TYPE_F32));

            tensor2d q = eval_linear(
                    require_tensor(model, prefix + "self_attn.query.weight"),
                    require_tensor(model, prefix + "self_attn.query.bias", GGML_TYPE_F32),
                    norm,
                    n_threads);
            tensor2d k = eval_linear(
                    require_tensor(model, prefix + "self_attn.key.weight"),
                    require_tensor(model, prefix + "self_attn.key.bias", GGML_TYPE_F32),
                    norm,
                    n_threads);
            tensor2d v = eval_linear(
                    require_tensor(model, prefix + "self_attn.value.weight"),
                    require_tensor(model, prefix + "self_attn.value.bias", GGML_TYPE_F32),
                    norm,
                    n_threads);

            tensor2d attn = decoder_self_attention_single(q, self_kv[size_t(il)], k, v, model.decoder.num_attention_heads);
            tensor2d proj = eval_linear(
                    require_tensor(model, prefix + "self_attn.out.weight"),
                    require_tensor(model, prefix + "self_attn.out.bias", GGML_TYPE_F32),
                    attn,
                    n_threads);
            hidden = add_scaled(hidden, proj, 1.0f);
        }

        {
            tensor2d norm = layer_norm(
                    hidden,
                    require_tensor(model, prefix + "layer_norm_2.weight", GGML_TYPE_F32),
                    require_tensor(model, prefix + "layer_norm_2.bias", GGML_TYPE_F32));

            tensor2d q = eval_linear(
                    require_tensor(model, prefix + "cross_attn.query.weight"),
                    require_tensor(model, prefix + "cross_attn.query.bias", GGML_TYPE_F32),
                    norm,
                    n_threads);
            tensor2d attn = decoder_cross_attention_single(q, cross_kv[size_t(il)], model.decoder.num_attention_heads);
            tensor2d proj = eval_linear(
                    require_tensor(model, prefix + "cross_attn.out.weight"),
                    require_tensor(model, prefix + "cross_attn.out.bias", GGML_TYPE_F32),
                    attn,
                    n_threads);
            hidden = add_scaled(hidden, proj, 1.0f);
        }

        {
            tensor2d norm = layer_norm(
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
                    n_threads);
            hidden = add_scaled(hidden, ff, 1.0f);
        }
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
            n_threads);

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

bool probe_model_architecture(const std::string & path_model, std::string & architecture, std::string * error) {
    architecture.clear();

    try {
        struct gguf_init_params params = {
            /*.no_alloc =*/ true,
            /*.ctx      =*/ nullptr,
        };

        gguf_ptr gguf(gguf_init_from_file(path_model.c_str(), params));
        if (!gguf) {
            throw std::runtime_error(format("failed to open model '%s'", path_model.c_str()));
        }

        const int64_t kid = gguf_find_key(gguf.get(), GGUF_ARCH_KEY);
        if (kid >= 0 && gguf_get_kv_type(gguf.get(), kid) == GGUF_TYPE_STRING) {
            architecture = gguf_get_val_str(gguf.get(), kid);
        }
        return true;
    } catch (const std::exception & ex) {
        if (error) {
            *error = ex.what();
        }
        return false;
    }
}

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

        out.tensors.clear();
        const int64_t n_tensors = gguf_get_n_tensors(out.gguf);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(out.gguf, i);
            out.tensors[name] = ggml_get_tensor(out.weights, name);
        }

        validate_model(out);
        return true;
    } catch (const std::exception & ex) {
        error = ex.what();
        free_model(out);
        return false;
    }
}

void free_model(model & model) {
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
        debug_outputs * outputs,
        std::string & text,
        std::string & error) {
    text.clear();
    error.clear();
    if (outputs != nullptr) {
        *outputs = debug_outputs();
    }

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
        encoder_trace trace = run_encoder_trace(model, pcmf32, n_threads);
        tensor2d decoder_memory = trace.encoder_projected;
        std::vector<cross_kv_cache> cross_kv = build_cross_kv(model, decoder_memory, n_threads);
        std::vector<self_kv_cache> self_kv = build_self_kv(model);

        const std::vector<int32_t> prompt = model.vocab.build_prompt(params.language, params.punctuation);
        const int32_t eos_token_id = model.vocab.special_token_id("<|endoftext|>");

        if ((int32_t) prompt.size() > model.decoder.max_sequence_length) {
            throw std::runtime_error("decoder prompt exceeds decoder max sequence length");
        }

        if (outputs != nullptr) {
            outputs->mel = matrix_output_from_tensor2d(trace.mel);
            outputs->mel_len = trace.mel.n1;
            outputs->subsampling_out = matrix_output_from_tensor2d(trace.subsampling_out);
            outputs->subsampling_len = trace.subsampling_out.n1;
            outputs->block0_after_ff1 = matrix_output_from_tensor2d(trace.block0_after_ff1);
            outputs->block0_after_ff1_len = trace.block0_after_ff1.n1;
            outputs->block0_after_attn = matrix_output_from_tensor2d(trace.block0_after_attn);
            outputs->block0_after_attn_len = trace.block0_after_attn.n1;
            outputs->block0_after_conv = matrix_output_from_tensor2d(trace.block0_after_conv);
            outputs->block0_after_conv_len = trace.block0_after_conv.n1;
            outputs->block0_out = matrix_output_from_tensor2d(trace.block0_out);
            outputs->block0_out_len = trace.block0_out.n1;
            outputs->encoder_out = matrix_output_from_tensor2d(trace.encoder_out);
            outputs->encoder_out_len = trace.encoder_out.n1;
            outputs->encoder_projected = matrix_output_from_tensor2d(trace.encoder_projected);
            outputs->encoder_projected_len = trace.encoder_projected.n1;
            outputs->prompt_ids = prompt;
        }

        std::vector<int32_t> generated;
        generated.reserve(size_t(params.max_new_tokens));

        int32_t current_position = 0;
        std::vector<float> logits;
        for (size_t i = 0; i < prompt.size(); ++i) {
            logits = decoder_step(model, prompt[i], current_position++, cross_kv, self_kv, n_threads);
        }

        if (outputs != nullptr) {
            outputs->first_step_logits = logits;
        }

        const int32_t remaining = std::max(0, model.decoder.max_sequence_length - current_position);
        const int32_t max_new_tokens = std::min(params.max_new_tokens, remaining);
        for (int32_t i = 0; i < max_new_tokens; ++i) {
            const int32_t next = argmax(logits);
            if (next == eos_token_id) {
                break;
            }
            generated.push_back(next);
            logits = decoder_step(model, next, current_position++, cross_kv, self_kv, n_threads);
        }

        text = model.vocab.detokenize(generated);
        if (outputs != nullptr) {
            outputs->greedy_ids = generated;
            outputs->text = text;
        }
        return true;
    } catch (const std::exception & ex) {
        error = ex.what();
        if (outputs != nullptr) {
            *outputs = debug_outputs();
        }
        return false;
    }
}

bool transcribe(
        model & model,
        const std::vector<float> & pcmf32,
        const transcribe_params & params,
        std::string & text,
        std::string & error) {
    return run_inference(model, pcmf32, params, nullptr, text, error);
}

bool collect_debug_outputs(
        model & model,
        const std::vector<float> & pcmf32,
        const transcribe_params & params,
        debug_outputs & outputs,
        std::string & error) {
    std::string text;
    return run_inference(model, pcmf32, params, &outputs, text, error);
}

} // namespace cohere
