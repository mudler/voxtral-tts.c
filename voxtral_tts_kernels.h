/*
 * voxtral_tts_kernels.h - Math kernels for Voxtral TTS inference
 *
 * Low-level math operations. All operate on float32 tensors in row-major order.
 * Adapted from antirez/voxtral.c with additions for TTS-specific operations.
 */

#ifndef VOXTRAL_TTS_KERNELS_H
#define VOXTRAL_TTS_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

void tts_add_inplace(float *a, const float *b, int n);
void tts_mul_inplace(float *a, const float *b, int n);
void tts_axpy(float *a, float scale, const float *b, int n);
void tts_scale(float *x, float s, int n);
void tts_copy(float *dst, const float *src, int n);

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

/* C = A @ B: A[M,K], B[K,N], C[M,N] */
void tts_matmul(float *C, const float *A, const float *B, int M, int K, int N);

/* C = A @ B^T: A[M,K], B[N,K], C[M,N] */
void tts_matmul_t(float *C, const float *A, const float *B, int M, int K, int N);

/* y = x @ W^T + b: x[seq,in_dim], W[out_dim,in_dim], b[out_dim] */
void tts_linear(float *y, const float *x, const float *W, const float *b,
                int seq_len, int in_dim, int out_dim);

void tts_linear_nobias(float *y, const float *x, const float *W,
                       int seq_len, int in_dim, int out_dim);

/* Linear with bf16 weights */
void tts_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                            int seq_len, int in_dim, int out_dim);

void tts_linear_bf16(float *y, const float *x, const uint16_t *W_bf16,
                     const float *b, int seq_len, int in_dim, int out_dim);

/* ========================================================================
 * 1D Convolution
 * ======================================================================== */

/*
 * Causal conv1d: left_pad = kernel - stride, output_len = ceil(input_frames)
 * in: [ch_in, length], weight: [ch_out, ch_in, kernel], bias: [ch_out]
 * out: [ch_out, out_length]
 */
void tts_causal_conv1d(float *out, const float *in, const float *weight,
                       const float *bias, int ch_in, int ch_out, int length,
                       int kernel_size, int stride);

/*
 * Causal transposed conv1d (upsample):
 * in: [ch_in, length], weight: [ch_in, ch_out, kernel], bias: [ch_out]
 * out: [ch_out, out_length] where out_length depends on kernel/stride
 *
 * Padding trim: total_padding = kernel - stride
 *   right_trim = ceil(total_padding * trim_ratio)
 *   left_trim = total_padding - right_trim
 * trim_ratio defaults to 1.0 (all trimmed from right for causal)
 */
void tts_causal_conv_transpose_1d(float *out, const float *in,
                                   const float *weight, const float *bias,
                                   int ch_in, int ch_out, int length,
                                   int kernel_size, int stride,
                                   int *out_length);

/* ========================================================================
 * Normalization
 * ======================================================================== */

/* RMS Normalization: out = x / rms(x) * weight */
void tts_rms_norm(float *out, const float *x, const float *weight,
                  int seq_len, int hidden, float eps);

/* QK Normalization for codec attention (L2 normalize then scale) */
void tts_qk_norm(float *out, const float *x, const float *weight,
                 int seq_len, int dim, float eps);

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

void tts_silu(float *x, int n);
void tts_gelu(float *x, int n);
void tts_softmax(float *x, int rows, int cols);

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

/*
 * Causal attention with GQA and optional sliding window.
 * For LLM decoder (with RoPE, applied externally before calling).
 */
void tts_causal_attention(float *out, const float *Q, const float *K,
                          const float *V, int seq_q, int seq_k,
                          int n_heads, int n_kv_heads, int head_dim,
                          float scale, int window_size, int q_offset);

/*
 * Bidirectional attention (no causal mask, no positional encoding).
 * For acoustic transformer flow matching. Supports GQA.
 * Q: [seq, n_heads*head_dim], K/V: [seq, n_kv_heads*head_dim]
 */
void tts_bidirectional_attention(float *out, const float *Q, const float *K,
                                  const float *V, int seq_len,
                                  int n_heads, int n_kv_heads, int head_dim,
                                  float scale);

/*
 * ALiBi attention with sliding window (for codec decoder).
 * Causal attention with ALiBi bias and sliding window.
 * alibi_slopes: [n_heads] precomputed slopes.
 */
void tts_alibi_attention(float *out, const float *Q, const float *K,
                         const float *V, int seq_len,
                         int n_heads, int n_kv_heads, int head_dim,
                         float scale, int window_size,
                         const float *alibi_slopes);

/* ========================================================================
 * Rotary Position Embeddings
 * ======================================================================== */

void tts_compute_rope_freqs(float *freqs, const int *pos, int seq,
                            int dim, float theta);

void tts_apply_rope(float *x, const float *freqs, int seq,
                    int heads, int head_dim);

/* ========================================================================
 * Random Number Generation
 * ======================================================================== */

/* Seed the RNG */
void tts_rng_seed(uint64_t *state, uint64_t seed);

/* Generate standard normal random float (Box-Muller) */
float tts_randn(uint64_t *state);

/* Fill buffer with standard normal random values */
void tts_randn_fill(uint64_t *state, float *buf, int n);

/* ========================================================================
 * Utility
 * ======================================================================== */

/* BF16 to F32 conversion (single value) */
static inline float tts_bf16_to_f32(uint16_t bf16) {
    uint32_t f32 = ((uint32_t)bf16) << 16;
    float result;
    __builtin_memcpy(&result, &f32, sizeof(float));
    return result;
}

/* BF16 buffer to F32 buffer conversion */
void tts_bf16_to_f32_buf(float *dst, const uint16_t *src, size_t n);

/* Token embedding lookup from bf16 table */
void tts_embed_token_bf16(float *out, const uint16_t *embeddings_bf16,
                          int token_id, int dim);

/* Global verbose flag */
extern int tts_verbose;

/* CUDA support (included here for dispatch convenience) */
#ifdef USE_CUDA
#include "voxtral_tts_cuda.h"
#endif

#endif /* VOXTRAL_TTS_KERNELS_H */
