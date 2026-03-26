/*
 * voxtral_tts_cuda.h - CUDA GPU acceleration for Voxtral TTS
 *
 * Provides cuBLAS GEMM + custom CUDA kernels for transformer inference.
 * All functions are no-ops when compiled without -DUSE_CUDA.
 */

#ifndef VOXTRAL_TTS_CUDA_H
#define VOXTRAL_TTS_CUDA_H

#include <stdint.h>
#include <stddef.h>

#ifdef USE_CUDA

/* ========================================================================
 * GPU Context
 * ======================================================================== */

/* Per-layer weight pointers on GPU (bf16) */
typedef struct {
    void *wq, *wk, *wv, *wo;       /* attention weights (bf16) */
    void *w1, *w2, *w3;            /* FFN weights (bf16) */
    float *attention_norm;          /* f32 */
    float *ffn_norm;                /* f32 */
} tts_cuda_layer_weights_t;

typedef struct {
    /* cuBLAS handle (opaque) */
    void *cublas_handle;

    /* GPU compute capability */
    int sm_major, sm_minor;
    int has_bf16;                   /* sm >= 80 */

    /* LLM decoder weights on GPU */
    void *tok_embeddings_gpu;       /* bf16 [vocab, dim] */
    float *dec_norm_gpu;            /* f32 [dim] */
    tts_cuda_layer_weights_t dec_layers[26];

    /* Acoustic transformer weights on GPU */
    void *ac_input_proj_gpu;
    void *ac_time_proj_gpu;
    void *ac_llm_proj_gpu;
    void *ac_semantic_out_gpu;
    void *ac_acoustic_out_gpu;
    float *ac_norm_gpu;
    float *ac_time_inv_freq_gpu;
    tts_cuda_layer_weights_t ac_layers[3];

    /* KV cache on GPU */
    float *kv_cache_k_gpu;          /* [layers, max_seq, kv_dim] */
    float *kv_cache_v_gpu;
    int kv_cache_max;

    /* Persistent activation buffers on GPU */
    float *d_x;                     /* [max_seq, dim] */
    float *d_x_norm;
    float *d_q, *d_k, *d_v;
    float *d_attn_out, *d_proj_out;
    float *d_gate, *d_up, *d_ffn_out;
    float *d_rope_freqs;            /* [max_seq, head_dim] */
    float *d_hidden;                /* [dim] output hidden state */

    /* Acoustic transformer buffers */
    float *d_ac_tokens;             /* [3, dim] */
    float *d_ac_tokens_norm;
    float *d_ac_q, *d_ac_k, *d_ac_v;
    float *d_ac_attn_out, *d_ac_proj_out;
    float *d_ac_gate, *d_ac_up, *d_ac_ffn_out;

    /* Embedding lookup buffer */
    float *d_embed;                 /* [dim] */

    /* Max allocated sequence length */
    int max_alloc_seq;

    int initialized;
} tts_cuda_ctx_t;

/* Global CUDA context */
extern tts_cuda_ctx_t g_cuda;

/* ========================================================================
 * Initialization / Cleanup
 * ======================================================================== */

/* Initialize CUDA: detect GPU, create cuBLAS handle, allocate buffers */
int tts_cuda_init(int kv_cache_max);

/* Upload model weights from CPU (bf16 mmap'd) to GPU VRAM */
int tts_cuda_upload_llm_weights(void *decoder_ptr);
int tts_cuda_upload_acoustic_weights(void *acoustic_ptr);

/* Free all GPU resources */
void tts_cuda_free(void);

/* Check if CUDA is available and initialized */
int tts_cuda_available(void);

/* ========================================================================
 * cuBLAS Linear Operations
 * ======================================================================== */

/* y = x @ W^T (bf16 weights, f32 activations)
 * x: host or device [seq, in_dim], W_bf16: device [out_dim, in_dim]
 * y: host or device [seq, out_dim]
 * If W_gpu is NULL, uploads W_bf16 from host on-the-fly (slower). */
void tts_cuda_linear_bf16(float *y, const float *x, const uint16_t *W_bf16,
                          int seq_len, int in_dim, int out_dim,
                          const void *W_gpu);

/* ========================================================================
 * Custom CUDA Kernels
 * ======================================================================== */

/* RMS normalization on GPU */
void tts_cuda_rms_norm(float *d_out, const float *d_x, const float *d_weight,
                       int seq_len, int dim, float eps);

/* Apply RoPE on GPU */
void tts_cuda_apply_rope(float *d_x, const float *d_freqs,
                         int seq, int heads, int head_dim);

/* Compute RoPE frequencies on GPU */
void tts_cuda_compute_rope_freqs(float *d_freqs, int start_pos, int seq,
                                  int dim, float theta);

/* Fused SiLU(gate) * up on GPU */
void tts_cuda_silu_mul(float *d_gate, const float *d_up, int n);

/* Residual add on GPU: a += b */
void tts_cuda_add_inplace(float *d_a, const float *d_b, int n);

/* Causal attention with GQA on GPU (single-token decode) */
void tts_cuda_causal_attention_decode(
    float *d_out, const float *d_Q, const float *d_K_cache, const float *d_V_cache,
    int seq_k, int n_heads, int n_kv_heads, int head_dim, float scale,
    int kv_cache_stride);

/* Causal attention for prefill (multi-token) */
void tts_cuda_causal_attention_prefill(
    float *d_out, const float *d_Q, const float *d_K, const float *d_V,
    int seq_len, int start_pos,
    int n_heads, int n_kv_heads, int head_dim, float scale);

/* Bidirectional attention on GPU (for acoustic transformer, 3 tokens) */
void tts_cuda_bidirectional_attention(
    float *d_out, const float *d_Q, const float *d_K, const float *d_V,
    int seq_len, int n_heads, int n_kv_heads, int head_dim, float scale);

/* Token embedding lookup from bf16 table on GPU */
void tts_cuda_embed_token(float *d_out, const void *d_embeddings_bf16,
                          int token_id, int dim);

/* ========================================================================
 * GPU LLM Forward Pass
 * ======================================================================== */

/* Single-token forward (autoregressive decode) — fully on GPU */
void tts_cuda_llm_forward(float *out_hidden, const float *input_embed,
                           int pos);

/* Multi-token prefill — fully on GPU */
void tts_cuda_llm_prefill(const float *embeds, int seq_len, int start_pos);

/* ========================================================================
 * GPU Acoustic Transformer
 * ======================================================================== */

/* Predict velocity for flow matching (on GPU) */
void tts_cuda_predict_velocity(float *out_velocity,
                                const float *x_t, const float *llm_hidden,
                                float t_val);

/* ========================================================================
 * Host<->Device Transfers
 * ======================================================================== */

void tts_cuda_to_device(void *d_dst, const void *h_src, size_t bytes);
void tts_cuda_to_host(void *h_dst, const void *d_src, size_t bytes);
void tts_cuda_memset(void *d_ptr, int value, size_t bytes);
void *tts_cuda_alloc(size_t bytes);
void tts_cuda_free_ptr(void *d_ptr);
void tts_cuda_sync(void);

#else /* !USE_CUDA */

/* Stubs when CUDA is not available */
static inline int tts_cuda_available(void) { return 0; }
static inline int tts_cuda_init(int kv_cache_max) { (void)kv_cache_max; return -1; }
static inline void tts_cuda_free(void) {}

#endif /* USE_CUDA */

#endif /* VOXTRAL_TTS_CUDA_H */
