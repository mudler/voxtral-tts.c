/*
 * voxtral_tts_kernels.c - Math kernels for Voxtral TTS inference
 * Adapted from antirez/voxtral.c with TTS-specific additions.
 */

#include "voxtral_tts_kernels.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#elif defined(USE_NVPL_BLAS)
#include <nvpl_blas_cblas.h>
#else
#include <cblas.h>
#endif
#endif

/* ========================================================================
 * Basic Element-wise Operations
 * ======================================================================== */

void tts_add_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

void tts_mul_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] *= b[i];
}

void tts_axpy(float *a, float scale, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] += scale * b[i];
}

void tts_scale(float *x, float s, int n) {
    for (int i = 0; i < n; i++) x[i] *= s;
}

void tts_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

/* ========================================================================
 * BF16 Utilities
 * ======================================================================== */

void tts_bf16_to_f32_buf(float *dst, const uint16_t *src, size_t n) {
    uint32_t *d = (uint32_t *)(void *)dst;
    for (size_t i = 0; i < n; i++)
        d[i] = ((uint32_t)src[i]) << 16;
}

void tts_embed_token_bf16(float *out, const uint16_t *embeddings_bf16,
                          int token_id, int dim) {
    const uint16_t *src = embeddings_bf16 + (size_t)token_id * dim;
    tts_bf16_to_f32_buf(out, src, dim);
}

/* Reusable scratch buffer for bf16->f32 conversion */
static float *bf16_scratch = NULL;
static size_t bf16_scratch_cap = 0;

static float *bf16_get_scratch(size_t n) {
    if (n > bf16_scratch_cap) {
        free(bf16_scratch);
        bf16_scratch = (float *)malloc(n * sizeof(float));
        bf16_scratch_cap = bf16_scratch ? n : 0;
    }
    return bf16_scratch;
}

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

void tts_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
#endif
}

void tts_matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
#endif
}

void tts_linear(float *y, const float *x, const float *W, const float *b,
                int seq_len, int in_dim, int out_dim) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, out_dim, in_dim,
                1.0f, x, in_dim, W, in_dim,
                0.0f, y, out_dim);
    if (b != NULL) {
        for (int s = 0; s < seq_len; s++) {
            for (int o = 0; o < out_dim; o++) {
                y[s * out_dim + o] += b[o];
            }
        }
    }
#else
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * in_dim;
        float *y_row = y + s * out_dim;
        for (int o = 0; o < out_dim; o++) {
            const float *w_row = W + o * in_dim;
            float sum = (b != NULL) ? b[o] : 0.0f;
            for (int i = 0; i < in_dim; i++) {
                sum += x_row[i] * w_row[i];
            }
            y_row[o] = sum;
        }
    }
#endif
}

void tts_linear_nobias(float *y, const float *x, const float *W,
                       int seq_len, int in_dim, int out_dim) {
    tts_linear(y, x, W, NULL, seq_len, in_dim, out_dim);
}

/* Fused BF16 matvec for single-token decode */
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static void bf16_matvec_fused(float *y, const float *x, const uint16_t *W_bf16,
                               const float *bias, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = bias ? bias[o] : 0.0f;
        int k = 0;

#ifdef __ARM_NEON
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);

        for (; k + 8 <= in_dim; k += 8) {
            uint16x8_t bf = vld1q_u16(w_row + k);
            uint32x4_t lo = vshll_n_u16(vget_low_u16(bf), 16);
            uint32x4_t hi = vshll_n_u16(vget_high_u16(bf), 16);
            float32x4_t w0 = vreinterpretq_f32_u32(lo);
            float32x4_t w1 = vreinterpretq_f32_u32(hi);
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            acc0 = vfmaq_f32(acc0, w0, x0);
            acc1 = vfmaq_f32(acc1, w1, x1);
        }
        sum += vaddvq_f32(vaddq_f32(acc0, acc1));
#endif

        for (; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }
        y[o] = sum;
    }
}

void tts_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                            int seq_len, int in_dim, int out_dim) {
    if (seq_len == 1) {
        bf16_matvec_fused(y, x, W_bf16, NULL, in_dim, out_dim);
        return;
    }
    size_t n = (size_t)out_dim * in_dim;
    float *W_f32 = bf16_get_scratch(n);
    if (!W_f32) return;
    tts_bf16_to_f32_buf(W_f32, W_bf16, n);
    tts_linear_nobias(y, x, W_f32, seq_len, in_dim, out_dim);
}

void tts_linear_bf16(float *y, const float *x, const uint16_t *W_bf16,
                     const float *b, int seq_len, int in_dim, int out_dim) {
    if (seq_len == 1) {
        bf16_matvec_fused(y, x, W_bf16, b, in_dim, out_dim);
        return;
    }
    size_t n = (size_t)out_dim * in_dim;
    float *W_f32 = bf16_get_scratch(n);
    if (!W_f32) return;
    tts_bf16_to_f32_buf(W_f32, W_bf16, n);
    tts_linear(y, x, W_f32, b, seq_len, in_dim, out_dim);
}

/* ========================================================================
 * INT8 Quantized Linear Operations
 *
 * Weight layout: W_int8[out_dim, in_dim] (int8)
 * Scale layout:  scale[out_dim] (float32, per output channel)
 * Dequantization: W_float[o,i] = W_int8[o,i] * scale[o]
 * ======================================================================== */

/* Fused INT8 matvec for single-token decode (hot path) */
static void int8_matvec_fused(float *y, const float *x, const int8_t *W_int8,
                               const float *scale, const float *bias,
                               int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        const int8_t *w_row = W_int8 + (size_t)o * in_dim;
        float sum = 0.0f;
        int k = 0;

#ifdef __SSE2__
        /* SSE2 vectorized path: process 16 int8 values at a time */
        /* Accumulate as int32, then convert to float at end */
#endif

        for (; k < in_dim; k++) {
            sum += x[k] * (float)w_row[k];
        }

        /* Apply per-channel scale */
        sum *= scale[o];

        /* Add bias if present */
        if (bias) sum += bias[o];

        y[o] = sum;
    }
}

void tts_linear_nobias_int8(float *y, const float *x, const int8_t *W_int8,
                            const float *scale, int seq_len, int in_dim, int out_dim) {
    if (seq_len == 1) {
        int8_matvec_fused(y, x, W_int8, scale, NULL, in_dim, out_dim);
        return;
    }
    /* Multi-token path: dequantize to scratch then use BLAS */
    size_t n = (size_t)out_dim * in_dim;
    float *W_f32 = bf16_get_scratch(n);
    if (!W_f32) return;
    for (int o = 0; o < out_dim; o++) {
        float s = scale[o];
        const int8_t *row = W_int8 + (size_t)o * in_dim;
        float *dst = W_f32 + (size_t)o * in_dim;
        for (int i = 0; i < in_dim; i++) {
            dst[i] = (float)row[i] * s;
        }
    }
    tts_linear_nobias(y, x, W_f32, seq_len, in_dim, out_dim);
}

void tts_linear_int8(float *y, const float *x, const int8_t *W_int8,
                     const float *scale, const float *b,
                     int seq_len, int in_dim, int out_dim) {
    if (seq_len == 1) {
        int8_matvec_fused(y, x, W_int8, scale, b, in_dim, out_dim);
        return;
    }
    size_t n = (size_t)out_dim * in_dim;
    float *W_f32 = bf16_get_scratch(n);
    if (!W_f32) return;
    for (int o = 0; o < out_dim; o++) {
        float s = scale[o];
        const int8_t *row = W_int8 + (size_t)o * in_dim;
        float *dst = W_f32 + (size_t)o * in_dim;
        for (int i = 0; i < in_dim; i++) {
            dst[i] = (float)row[i] * s;
        }
    }
    tts_linear(y, x, W_f32, b, seq_len, in_dim, out_dim);
}

/* ========================================================================
 * 1D Convolution
 * ======================================================================== */

void tts_causal_conv1d(float *out, const float *in, const float *weight,
                       const float *bias, int ch_in, int ch_out, int length,
                       int kernel_size, int stride) {
    /*
     * Causal padding: left_pad = kernel - stride (for stride > 1)
     * or left_pad = kernel - 1 (for stride = 1)
     * Matches PyTorch VoxtralTTS CausalConv1d:
     *   effective_kernel = (kernel-1)*dilation + 1
     *   padding_total = effective_kernel - stride
     *   left_pad = padding_total
     *   extra_right_pad = target_length - length (ensures correct output size)
     */
    int effective_kernel = kernel_size; /* dilation=1 */
    int padding_total = effective_kernel - stride;

    float n_frames = ((float)(length - effective_kernel + padding_total)) / (float)stride + 1.0f;
    int out_length = (int)ceilf(n_frames);
    if (out_length <= 0) return;

    int left_pad = padding_total;
    /* Extra right padding to reach target length */
    int target_length = (out_length - 1) * stride + effective_kernel - padding_total;
    int extra_right = target_length - length;
    if (extra_right < 0) extra_right = 0;

    /* Padded input length */
    (void)(left_pad + length + extra_right); /* padded_length for reference */

    for (int oc = 0; oc < ch_out; oc++) {
        float b = (bias != NULL) ? bias[oc] : 0.0f;
        for (int ol = 0; ol < out_length; ol++) {
            float sum = b;
            for (int ic = 0; ic < ch_in; ic++) {
                for (int k = 0; k < kernel_size; k++) {
                    int padded_pos = ol * stride + k;
                    int il = padded_pos - left_pad;
                    float val = 0.0f;
                    if (il >= 0 && il < length) {
                        val = in[(size_t)ic * length + il];
                    } else if (il < 0) {
                        /* Reflect padding for left side */
                        int reflect_idx = -il;
                        if (reflect_idx < length) val = in[(size_t)ic * length + reflect_idx];
                    } else if (il >= length) {
                        /* Reflect padding for right side */
                        int reflect_idx = 2 * length - 2 - il;
                        if (reflect_idx >= 0 && reflect_idx < length)
                            val = in[(size_t)ic * length + reflect_idx];
                    }
                    int w_idx = (size_t)oc * ch_in * kernel_size + ic * kernel_size + k;
                    sum += val * weight[w_idx];
                }
            }
            out[(size_t)oc * out_length + ol] = sum;
        }
    }
}

void tts_causal_conv_transpose_1d(float *out, const float *in,
                                   const float *weight, const float *bias,
                                   int ch_in, int ch_out, int length,
                                   int kernel_size, int stride,
                                   int *out_length_ptr) {
    /*
     * Transposed conv1d (upsample). PyTorch ConvTranspose1d layout:
     * weight: [ch_in, ch_out, kernel]  (note: in/out swapped vs conv1d!)
     * Output before trim: (length - 1) * stride + kernel
     *
     * Causal trim: total_padding = kernel - stride
     *   right_trim = ceil(total_padding * 1.0)  // trim_ratio=1.0
     *   left_trim = total_padding - right_trim
     */
    int raw_out_len = (length - 1) * stride + kernel_size;
    int total_padding = kernel_size - stride;
    int right_trim = (int)ceilf((float)total_padding * 1.0f);
    int left_trim = total_padding - right_trim;
    int final_len = raw_out_len - left_trim - right_trim;
    if (final_len <= 0) {
        *out_length_ptr = 0;
        return;
    }

    /* Compute raw transposed conv output */
    float *raw_out = (float *)calloc((size_t)ch_out * raw_out_len, sizeof(float));

    for (int ic = 0; ic < ch_in; ic++) {
        for (int il = 0; il < length; il++) {
            float x_val = in[(size_t)ic * length + il];
            int out_start = il * stride;
            for (int oc = 0; oc < ch_out; oc++) {
                for (int k = 0; k < kernel_size; k++) {
                    int w_idx = (size_t)ic * ch_out * kernel_size + oc * kernel_size + k;
                    raw_out[(size_t)oc * raw_out_len + out_start + k] +=
                        x_val * weight[w_idx];
                }
            }
        }
    }

    /* Add bias and trim */
    for (int oc = 0; oc < ch_out; oc++) {
        float b = (bias != NULL) ? bias[oc] : 0.0f;
        for (int ol = 0; ol < final_len; ol++) {
            out[(size_t)oc * final_len + ol] =
                raw_out[(size_t)oc * raw_out_len + left_trim + ol] + b;
        }
    }

    free(raw_out);
    *out_length_ptr = final_len;
}

/* ========================================================================
 * Normalization
 * ======================================================================== */

void tts_rms_norm(float *out, const float *x, const float *weight,
                  int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum_sq += x_row[i] * x_row[i];
        }
        float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);

        for (int i = 0; i < hidden; i++) {
            out_row[i] = x_row[i] * rms_inv * weight[i];
        }
    }
}

void tts_qk_norm(float *out, const float *x, const float *weight,
                 int seq_len, int dim, float eps) {
    /* RMS norm applied to Q/K for codec attention */
    tts_rms_norm(out, x, weight, seq_len, dim, eps);
}

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

void tts_silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        x[i] = val / (1.0f + expf(-val));
    }
}

void tts_gelu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        float x3 = val * val * val;
        float inner = 0.7978845608028654f * (val + 0.044715f * x3);
        x[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

void tts_softmax(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;

        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }

        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }

        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row[c] *= inv_sum;
        }
    }
}

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

void tts_causal_attention(float *out, const float *Q, const float *K,
                          const float *V, int seq_q, int seq_k,
                          int n_heads, int n_kv_heads, int head_dim,
                          float scale, int window_size, int q_offset) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;

        for (int i = 0; i < seq_q; i++) {
            const float *q_row = Q + i * q_hidden + h * head_dim;
            float *o_row = out + i * q_hidden + h * head_dim;

            int global_pos = q_offset + i;
            int k_start = 0;
            if (window_size > 0 && global_pos - window_size + 1 > 0) {
                k_start = global_pos - window_size + 1;
            }
            int k_end = global_pos + 1;
            if (k_end > seq_k) k_end = seq_k;

            /* Online softmax */
            float max_score = -1e30f;
            float sum_exp = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            for (int j = k_start; j < k_end; j++) {
                const float *k_row = K + j * kv_hidden + kv_h * head_dim;
                const float *v_row = V + j * kv_hidden + kv_h * head_dim;

                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q_row[d] * k_row[d];
                }
                score *= scale;

                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    for (int d = 0; d < head_dim; d++) {
                        o_row[d] = o_row[d] * correction + v_row[d];
                    }
                    max_score = score;
                } else {
                    float w = expf(score - max_score);
                    sum_exp += w;
                    for (int d = 0; d < head_dim; d++) {
                        o_row[d] += w * v_row[d];
                    }
                }
            }

            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                for (int d = 0; d < head_dim; d++) {
                    o_row[d] *= inv_sum;
                }
            }
        }
    }
}

void tts_bidirectional_attention(float *out, const float *Q, const float *K,
                                  const float *V, int seq_len,
                                  int n_heads, int n_kv_heads, int head_dim,
                                  float scale) {
    /*
     * Full bidirectional attention (no causal mask, no positional encoding).
     * Used by the acoustic transformer over 3 tokens.
     */
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;

        for (int i = 0; i < seq_len; i++) {
            const float *q_row = Q + i * q_hidden + h * head_dim;
            float *o_row = out + i * q_hidden + h * head_dim;

            /* Attend to all positions */
            float max_score = -1e30f;
            float sum_exp = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            for (int j = 0; j < seq_len; j++) {
                const float *k_row = K + j * kv_hidden + kv_h * head_dim;
                const float *v_row = V + j * kv_hidden + kv_h * head_dim;

                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q_row[d] * k_row[d];
                }
                score *= scale;

                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    for (int d = 0; d < head_dim; d++) {
                        o_row[d] = o_row[d] * correction + v_row[d];
                    }
                    max_score = score;
                } else {
                    float w = expf(score - max_score);
                    sum_exp += w;
                    for (int d = 0; d < head_dim; d++) {
                        o_row[d] += w * v_row[d];
                    }
                }
            }

            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                for (int d = 0; d < head_dim; d++) {
                    o_row[d] *= inv_sum;
                }
            }
        }
    }
}

void tts_alibi_attention(float *out, const float *Q, const float *K,
                         const float *V, int seq_len,
                         int n_heads, int n_kv_heads, int head_dim,
                         float scale, int window_size,
                         const float *alibi_slopes) {
    /*
     * Causal attention with ALiBi bias and sliding window.
     * score[h,i,j] = Q[i] . K[j] * scale + slopes[h] * (j - i)
     * Causal: j <= i only
     * Sliding window: j >= i - window_left
     */
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;
        float slope = alibi_slopes[h];

        for (int i = 0; i < seq_len; i++) {
            const float *q_row = Q + i * q_hidden + h * head_dim;
            float *o_row = out + i * q_hidden + h * head_dim;

            int k_start = 0;
            if (window_size > 0 && i - window_size + 1 > 0) {
                k_start = i - window_size + 1;
            }
            int k_end = i + 1; /* causal */

            float max_score = -1e30f;
            float sum_exp = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            for (int j = k_start; j < k_end; j++) {
                const float *k_row = K + j * kv_hidden + kv_h * head_dim;
                const float *v_row = V + j * kv_hidden + kv_h * head_dim;

                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q_row[d] * k_row[d];
                }
                score *= scale;
                /* ALiBi bias: slope * (j - i), always <= 0 for causal */
                score += slope * (float)(j - i);

                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    for (int d = 0; d < head_dim; d++) {
                        o_row[d] = o_row[d] * correction + v_row[d];
                    }
                    max_score = score;
                } else {
                    float w = expf(score - max_score);
                    sum_exp += w;
                    for (int d = 0; d < head_dim; d++) {
                        o_row[d] += w * v_row[d];
                    }
                }
            }

            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                for (int d = 0; d < head_dim; d++) {
                    o_row[d] *= inv_sum;
                }
            }
        }
    }
}

/* ========================================================================
 * Rotary Position Embeddings
 * ======================================================================== */

void tts_compute_rope_freqs(float *freqs, const int *pos, int seq,
                            int dim, float theta) {
    int half_dim = dim / 2;
    for (int s = 0; s < seq; s++) {
        float p = (float)pos[s];
        for (int d = 0; d < half_dim; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)dim);
            float angle = p * freq;
            freqs[s * half_dim * 2 + d * 2] = cosf(angle);
            freqs[s * half_dim * 2 + d * 2 + 1] = sinf(angle);
        }
    }
}

void tts_apply_rope(float *x, const float *freqs, int seq,
                    int heads, int head_dim) {
    int half_dim = head_dim / 2;
    int hidden = heads * head_dim;

    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < heads; h++) {
            float *vec = x + s * hidden + h * head_dim;
            for (int d = 0; d < half_dim; d++) {
                float cos_val = freqs[s * half_dim * 2 + d * 2];
                float sin_val = freqs[s * half_dim * 2 + d * 2 + 1];
                float x0 = vec[d * 2];
                float x1 = vec[d * 2 + 1];
                vec[d * 2]     = x0 * cos_val - x1 * sin_val;
                vec[d * 2 + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

/* ========================================================================
 * Random Number Generation (xorshift64 + Box-Muller)
 * ======================================================================== */

void tts_rng_seed(uint64_t *state, uint64_t seed) {
    *state = seed ? seed : 0x12345678ABCDEF01ULL;
}

static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static float uniform01(uint64_t *state) {
    return (float)(xorshift64(state) >> 11) * (1.0f / 9007199254740992.0f);
}

float tts_randn(uint64_t *state) {
    /* Box-Muller transform */
    float u1, u2;
    do { u1 = uniform01(state); } while (u1 < 1e-30f);
    u2 = uniform01(state);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853071795864f * u2);
}

void tts_randn_fill(uint64_t *state, float *buf, int n) {
    for (int i = 0; i < n; i++) {
        buf[i] = tts_randn(state);
    }
}
