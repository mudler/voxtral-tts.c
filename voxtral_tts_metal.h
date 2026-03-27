/*
 * voxtral_tts_metal.h - Metal GPU acceleration for Voxtral TTS (Apple Silicon)
 *
 * Uses Metal Performance Shaders (MPS) for GEMM and custom Metal compute
 * kernels for attention, normalization, and activation functions.
 *
 * Following the patterns established in antirez/voxtral.c's Metal backend.
 */

#ifndef VOXTRAL_TTS_METAL_H
#define VOXTRAL_TTS_METAL_H

#include <stdint.h>
#include <stddef.h>

#ifdef USE_METAL

/* ========================================================================
 * Initialization
 * ======================================================================== */

/* Initialize Metal: detect GPU, compile shaders, create command queue.
 * Returns 1 on success, 0 if Metal not available. */
int tts_metal_init(void);

/* Check if Metal is available and initialized */
int tts_metal_available(void);

/* Shutdown Metal and free all GPU resources */
void tts_metal_shutdown(void);

/* ========================================================================
 * Shared Memory (Zero-Copy GPU-CPU Buffers)
 * ======================================================================== */

/* Allocate GPU-CPU shared buffer. Returns CPU pointer.
 * Falls back to calloc if Metal unavailable. Caller uses this pointer
 * for CPU access; Metal finds the GPU buffer via pointer lookup. */
void *tts_metal_shared_alloc(size_t size);

/* Free shared buffer */
void tts_metal_shared_free(void *ptr);

/* ========================================================================
 * Matrix Multiplication (MPS)
 * ======================================================================== */

/* y = x @ W^T (bf16 weights converted to f16 on GPU, f32 activations)
 * x: [M, K] f32, W_bf16: [N, K] bf16 (mmap'd), y: [M, N] f32 */
void tts_metal_sgemm_bf16(int M, int N, int K,
                           const float *A, const uint16_t *B_bf16, float *C);

/* y = x @ W^T (f32 weights)
 * x: [M, K] f32, W: [N, K] f32, y: [M, N] f32 */
void tts_metal_sgemm(int M, int N, int K,
                     const float *A, const float *B, float *C);

/* ========================================================================
 * Custom Compute Kernels
 * ======================================================================== */

/* RMS normalization */
void tts_metal_rms_norm(float *out, const float *x, const float *weight,
                        int seq_len, int hidden, float eps);

/* Apply RoPE (single position, in-place) */
void tts_metal_apply_rope_pos(float *data, int n_heads, int head_dim,
                               int pos, float theta);

/* Apply RoPE with precomputed frequencies */
void tts_metal_apply_rope(float *data, const float *freqs,
                          int seq, int heads, int head_dim);

/* Fused SiLU(gate) * up */
void tts_metal_silu_mul(float *gate, const float *up, int n);

/* Residual add: a += b */
void tts_metal_add_inplace(float *a, const float *b, int n);

/* KV cache copy (single token) */
void tts_metal_kv_cache_copy(float *cache, const float *data,
                              int float_offset, int kv_dim);

/* ========================================================================
 * Attention
 * ======================================================================== */

/* Single-token causal attention with GQA (for autoregressive decode).
 * Uses shared-memory KV cache for zero-copy GPU access.
 * Q: [n_heads * head_dim], K/V cache: [seq_k, n_kv_heads * head_dim] */
void tts_metal_decoder_attention(float *out, const float *Q,
                                  const float *K_cache, const float *V_cache,
                                  int seq_k, int n_heads, int n_kv_heads,
                                  int head_dim, float scale,
                                  int window_size, int q_pos);

/* Bidirectional attention (for acoustic transformer, seq_len=3) */
void tts_metal_bidirectional_attention(float *out, const float *Q,
                                        const float *K, const float *V,
                                        int seq_len, int n_heads,
                                        int n_kv_heads, int head_dim,
                                        float scale);

/* ========================================================================
 * Fused Layer Operations
 * ======================================================================== */

/* Fused RMSNorm + QKV projections (1 command buffer, 4 ops) */
void tts_metal_fused_norm_qkv_bf16(
    const float *x, float *x_norm, float *q, float *k, float *v,
    const float *norm_weight, float eps, int dim,
    const uint16_t *wq_bf16, int q_dim,
    const uint16_t *wk_bf16, int kv_dim,
    const uint16_t *wv_bf16);

/* Fused SwiGLU FFN: w1+w3 (parallel) -> silu*mul -> w2 */
void tts_metal_fused_ffn_bf16(
    int M, int dim, int hidden,
    const float *input,
    const uint16_t *w1_bf16, const uint16_t *w3_bf16,
    const uint16_t *w2_bf16,
    float *output);

/* ========================================================================
 * Monolithic Decoder Step (all 26 layers in one command buffer)
 * ======================================================================== */

/* Full single-token decode: all 26 layers on GPU.
 * Persistent x buffer stays on GPU across layers.
 * Only input embedding upload + hidden state download cross the bus.
 *
 * ctx: tts_ctx_t* (opaque, for accessing decoder weights and KV cache)
 * input_embed: [dim] f32 (CPU)
 * out_hidden: [dim] f32 (CPU, output)
 * pos: KV cache position for this token
 */
void tts_metal_llm_forward(void *ctx, const float *input_embed,
                            float *out_hidden, int pos);

/* Full prefill: all 26 layers for seq_len tokens.
 * embeds: [seq_len, dim] f32 (CPU)
 * Populates shared-memory KV cache. */
void tts_metal_llm_prefill(void *ctx, const float *embeds,
                            int seq_len, int start_pos);

/* ========================================================================
 * Acoustic Transformer (Flow Matching)
 * ======================================================================== */

/* Predict velocity for one flow matching step (on GPU).
 * x_t: [36] f32, llm_hidden: [dim] f32, t_val: scalar
 * out_velocity: [36] f32 */
void tts_metal_predict_velocity(void *ctx,
                                 float *out_velocity,
                                 const float *x_t,
                                 const float *llm_hidden,
                                 float t_val);

/* ========================================================================
 * Weight Cache Management
 * ======================================================================== */

/* Pre-warm bf16->f16 weight cache (converts and uploads to GPU) */
void tts_metal_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements);

/* Report GPU memory usage */
size_t tts_metal_memory_used(void);

#else /* !USE_METAL */

/* Stubs when Metal is not available */
static inline int tts_metal_available(void) { return 0; }
static inline int tts_metal_init(void) { return 0; }
static inline void tts_metal_shutdown(void) {}
static inline void *tts_metal_shared_alloc(size_t size) { return calloc(1, size); }
static inline void tts_metal_shared_free(void *ptr) { free(ptr); }

#endif /* USE_METAL */

#endif /* VOXTRAL_TTS_METAL_H */
