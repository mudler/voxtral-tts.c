/*
 * voxtral_tts_llm.c - LLM decoder (26-layer Mistral)
 *
 * Architecture (per layer):
 *   RMSNorm -> GQA Attention (32 heads, 8 KV heads, RoPE)
 *   RMSNorm -> SwiGLU FFN (dim=3072, hidden=9216)
 *
 * Note: Unlike ASR, TTS model has NO ada_rms_norm_t_cond.
 */

#include "voxtral_tts.h"
#include "voxtral_tts_kernels.h"
#include "voxtral_tts_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_CUDA
#include "voxtral_tts_cuda.h"
#endif

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

static float *load_f32(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "llm: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static uint16_t *load_bf16_direct(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "llm: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

static int8_t *load_int8_direct(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "llm: int8 weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_int8_direct(sf, t);
}

static float *load_scale(safetensors_file_t *sf, const char *weight_name) {
    char scale_name[512];
    snprintf(scale_name, sizeof(scale_name), "%s_scale", weight_name);
    const safetensor_t *t = safetensors_find(sf, scale_name);
    if (!t) {
        fprintf(stderr, "llm: scale not found: %s\n", scale_name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

/* Check if the model uses INT8 quantized weights */
static int detect_int8(safetensors_file_t *sf) {
    const safetensor_t *t = safetensors_find(sf, "layers.0.attention.wq.weight");
    if (t && safetensor_is_int8(t)) return 1;
    return 0;
}

int tts_llm_load(tts_decoder_t *dec, void *sf_ptr) {
    safetensors_file_t *sf = (safetensors_file_t *)sf_ptr;
    char name[512];

    /* Auto-detect INT8 vs BF16 */
    int use_int8 = detect_int8(sf);
    dec->is_int8 = use_int8;

    if (tts_verbose)
        fprintf(stderr, "  LLM weights: %s\n", use_int8 ? "INT8 quantized" : "BF16");

    /* Token embeddings (always bf16, not quantized) */
    dec->tok_embeddings_bf16 = load_bf16_direct(sf,
        "mm_audio_embeddings.tok_embeddings.weight");
    if (!dec->tok_embeddings_bf16) return -1;

    /* Transformer layers */
    for (int i = 0; i < TTS_DEC_LAYERS; i++) {
        tts_dec_layer_t *l = &dec->layers[i];

        if (use_int8) {
            /* Load INT8 weights + scales */
            snprintf(name, sizeof(name), "layers.%d.attention.wq.weight", i);
            l->wq_int8 = load_int8_direct(sf, name);
            l->wq_scale = load_scale(sf, name);
            snprintf(name, sizeof(name), "layers.%d.attention.wk.weight", i);
            l->wk_int8 = load_int8_direct(sf, name);
            l->wk_scale = load_scale(sf, name);
            snprintf(name, sizeof(name), "layers.%d.attention.wv.weight", i);
            l->wv_int8 = load_int8_direct(sf, name);
            l->wv_scale = load_scale(sf, name);
            snprintf(name, sizeof(name), "layers.%d.attention.wo.weight", i);
            l->wo_int8 = load_int8_direct(sf, name);
            l->wo_scale = load_scale(sf, name);

            snprintf(name, sizeof(name), "layers.%d.feed_forward.w1.weight", i);
            l->w1_int8 = load_int8_direct(sf, name);
            l->w1_scale = load_scale(sf, name);
            snprintf(name, sizeof(name), "layers.%d.feed_forward.w2.weight", i);
            l->w2_int8 = load_int8_direct(sf, name);
            l->w2_scale = load_scale(sf, name);
            snprintf(name, sizeof(name), "layers.%d.feed_forward.w3.weight", i);
            l->w3_int8 = load_int8_direct(sf, name);
            l->w3_scale = load_scale(sf, name);

            if (!l->wq_int8 || !l->wq_scale || !l->wk_int8 || !l->wk_scale ||
                !l->wv_int8 || !l->wv_scale || !l->wo_int8 || !l->wo_scale ||
                !l->w1_int8 || !l->w1_scale || !l->w2_int8 || !l->w2_scale ||
                !l->w3_int8 || !l->w3_scale) {
                fprintf(stderr, "llm: failed to load INT8 layer %d\n", i);
                return -1;
            }
        } else {
            /* Load BF16 weights (original path) */
            snprintf(name, sizeof(name), "layers.%d.attention.wq.weight", i);
            l->wq_bf16 = load_bf16_direct(sf, name);
            snprintf(name, sizeof(name), "layers.%d.attention.wk.weight", i);
            l->wk_bf16 = load_bf16_direct(sf, name);
            snprintf(name, sizeof(name), "layers.%d.attention.wv.weight", i);
            l->wv_bf16 = load_bf16_direct(sf, name);
            snprintf(name, sizeof(name), "layers.%d.attention.wo.weight", i);
            l->wo_bf16 = load_bf16_direct(sf, name);

            snprintf(name, sizeof(name), "layers.%d.feed_forward.w1.weight", i);
            l->w1_bf16 = load_bf16_direct(sf, name);
            snprintf(name, sizeof(name), "layers.%d.feed_forward.w2.weight", i);
            l->w2_bf16 = load_bf16_direct(sf, name);
            snprintf(name, sizeof(name), "layers.%d.feed_forward.w3.weight", i);
            l->w3_bf16 = load_bf16_direct(sf, name);

            if (!l->wq_bf16 || !l->wk_bf16 || !l->wv_bf16 || !l->wo_bf16 ||
                !l->w1_bf16 || !l->w2_bf16 || !l->w3_bf16) {
                fprintf(stderr, "llm: failed to load BF16 layer %d\n", i);
                return -1;
            }
        }

        /* Norms (always f32) */
        snprintf(name, sizeof(name), "layers.%d.attention_norm.weight", i);
        l->attention_norm = load_f32(sf, name);
        snprintf(name, sizeof(name), "layers.%d.ffn_norm.weight", i);
        l->ffn_norm = load_f32(sf, name);

        if (!l->attention_norm || !l->ffn_norm) {
            fprintf(stderr, "llm: failed to load norms for layer %d\n", i);
            return -1;
        }

        if (tts_verbose >= 2)
            fprintf(stderr, "  LLM layer %d/%d loaded\n", i + 1, TTS_DEC_LAYERS);
    }

    /* Final norm */
    dec->norm = load_f32(sf, "norm.weight");
    if (!dec->norm) return -1;

    return 0;
}

/* ========================================================================
 * KV Cache Management
 * ======================================================================== */

static float *kv_cache_k_at(tts_ctx_t *ctx, int layer, int pos) {
    int kv_dim = TTS_DEC_KV_HEADS * TTS_DEC_HEAD_DIM;
    return ctx->kv_cache_k + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

static float *kv_cache_v_at(tts_ctx_t *ctx, int layer, int pos) {
    int kv_dim = TTS_DEC_KV_HEADS * TTS_DEC_HEAD_DIM;
    return ctx->kv_cache_v + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

int tts_llm_kv_cache_alloc(tts_ctx_t *ctx, int max_seq) {
    int kv_dim = TTS_DEC_KV_HEADS * TTS_DEC_HEAD_DIM;
    size_t elems = (size_t)TTS_DEC_LAYERS * max_seq * kv_dim;
    size_t bytes = elems * sizeof(float);

    ctx->kv_cache_k = (float *)calloc(1, bytes);
    ctx->kv_cache_v = (float *)calloc(1, bytes);

    if (!ctx->kv_cache_k || !ctx->kv_cache_v) {
        fprintf(stderr, "llm: failed to allocate KV cache (%.1f MB)\n",
                (float)(2 * bytes) / (1024 * 1024));
        return -1;
    }

    ctx->kv_cache_len = 0;
    ctx->kv_cache_max = max_seq;

    if (tts_verbose)
        fprintf(stderr, "  KV cache: %d positions, %.1f MB\n",
                max_seq, (float)(2 * bytes) / (1024 * 1024));

    return 0;
}

/* ========================================================================
 * Decoder Buffers
 * ======================================================================== */

static int alloc_decoder_buffers(tts_ctx_t *ctx) {
    int q_dim = TTS_DEC_HEADS * TTS_DEC_HEAD_DIM;
    int kv_dim = TTS_DEC_KV_HEADS * TTS_DEC_HEAD_DIM;

    ctx->dec_x       = (float *)calloc(TTS_DEC_DIM, sizeof(float));
    ctx->dec_x_norm  = (float *)calloc(TTS_DEC_DIM, sizeof(float));
    ctx->dec_q       = (float *)calloc(q_dim, sizeof(float));
    ctx->dec_k       = (float *)calloc(kv_dim, sizeof(float));
    ctx->dec_v       = (float *)calloc(kv_dim, sizeof(float));
    ctx->dec_attn_out = (float *)calloc(q_dim, sizeof(float));
    ctx->dec_proj_out = (float *)calloc(TTS_DEC_DIM, sizeof(float));
    ctx->dec_gate    = (float *)calloc(TTS_DEC_HIDDEN, sizeof(float));
    ctx->dec_up      = (float *)calloc(TTS_DEC_HIDDEN, sizeof(float));
    ctx->dec_ffn_out = (float *)calloc(TTS_DEC_DIM, sizeof(float));

    if (!ctx->dec_x || !ctx->dec_x_norm || !ctx->dec_q || !ctx->dec_k ||
        !ctx->dec_v || !ctx->dec_attn_out || !ctx->dec_proj_out ||
        !ctx->dec_gate || !ctx->dec_up || !ctx->dec_ffn_out) {
        fprintf(stderr, "llm: failed to allocate decoder buffers\n");
        return -1;
    }
    return 0;
}

/* ========================================================================
 * Single-Token Forward Pass (autoregressive generation)
 * ======================================================================== */

void tts_llm_forward(tts_ctx_t *ctx, const float *input_embed, float *out_hidden) {
    /*
     * Process a single token through all 26 layers.
     * input_embed: [3072] (token embedding or audio code embedding)
     * out_hidden: [3072] (hidden state at last position, before output projection)
     */
#ifdef USE_CUDA
    if (tts_cuda_available()) {
        tts_cuda_llm_forward(out_hidden, input_embed, ctx->kv_cache_len);
        ctx->kv_cache_len++;
        return;
    }
#endif
    tts_decoder_t *dec = &ctx->decoder;
    int dim = TTS_DEC_DIM;
    int q_dim = TTS_DEC_HEADS * TTS_DEC_HEAD_DIM;
    int kv_dim = TTS_DEC_KV_HEADS * TTS_DEC_HEAD_DIM;
    int pos = ctx->kv_cache_len;
    float scale = 1.0f / sqrtf((float)TTS_DEC_HEAD_DIM);

    /* Allocate buffers on first call */
    if (!ctx->dec_x) {
        if (alloc_decoder_buffers(ctx) != 0) return;
    }

    /* Compute RoPE frequencies for this position */
    float rope_freqs[TTS_DEC_HEAD_DIM]; /* [head_dim/2, 2] */
    int pos_arr[1] = { pos };
    tts_compute_rope_freqs(rope_freqs, pos_arr, 1, TTS_DEC_HEAD_DIM, TTS_ROPE_THETA);

    /* Start with input embedding */
    tts_copy(ctx->dec_x, input_embed, dim);

    int is_int8 = dec->is_int8;

    /* Process each layer */
    for (int layer = 0; layer < TTS_DEC_LAYERS; layer++) {
        tts_dec_layer_t *l = &dec->layers[layer];

        /* === Attention === */
        /* RMSNorm */
        tts_rms_norm(ctx->dec_x_norm, ctx->dec_x, l->attention_norm,
                     1, dim, TTS_DEC_NORM_EPS);

        /* Q, K, V projections */
        if (is_int8) {
            tts_linear_nobias_int8(ctx->dec_q, ctx->dec_x_norm, l->wq_int8,
                                   l->wq_scale, 1, dim, q_dim);
            tts_linear_nobias_int8(ctx->dec_k, ctx->dec_x_norm, l->wk_int8,
                                   l->wk_scale, 1, dim, kv_dim);
            tts_linear_nobias_int8(ctx->dec_v, ctx->dec_x_norm, l->wv_int8,
                                   l->wv_scale, 1, dim, kv_dim);
        } else {
            tts_linear_nobias_bf16(ctx->dec_q, ctx->dec_x_norm, l->wq_bf16,
                                   1, dim, q_dim);
            tts_linear_nobias_bf16(ctx->dec_k, ctx->dec_x_norm, l->wk_bf16,
                                   1, dim, kv_dim);
            tts_linear_nobias_bf16(ctx->dec_v, ctx->dec_x_norm, l->wv_bf16,
                                   1, dim, kv_dim);
        }

        /* Apply RoPE to Q and K */
        tts_apply_rope(ctx->dec_q, rope_freqs, 1, TTS_DEC_HEADS, TTS_DEC_HEAD_DIM);
        tts_apply_rope(ctx->dec_k, rope_freqs, 1, TTS_DEC_KV_HEADS, TTS_DEC_HEAD_DIM);

        /* Store K, V in cache */
        tts_copy(kv_cache_k_at(ctx, layer, pos), ctx->dec_k, kv_dim);
        tts_copy(kv_cache_v_at(ctx, layer, pos), ctx->dec_v, kv_dim);

        /* Attention over all cached K, V */
        tts_causal_attention(ctx->dec_attn_out,
                             ctx->dec_q,
                             kv_cache_k_at(ctx, layer, 0),
                             kv_cache_v_at(ctx, layer, 0),
                             1, pos + 1,
                             TTS_DEC_HEADS, TTS_DEC_KV_HEADS,
                             TTS_DEC_HEAD_DIM, scale,
                             0, /* no sliding window */
                             pos);

        /* Output projection */
        if (is_int8) {
            tts_linear_nobias_int8(ctx->dec_proj_out, ctx->dec_attn_out, l->wo_int8,
                                   l->wo_scale, 1, q_dim, dim);
        } else {
            tts_linear_nobias_bf16(ctx->dec_proj_out, ctx->dec_attn_out, l->wo_bf16,
                                   1, q_dim, dim);
        }

        /* Residual */
        tts_add_inplace(ctx->dec_x, ctx->dec_proj_out, dim);

        /* === FFN === */
        /* RMSNorm */
        tts_rms_norm(ctx->dec_x_norm, ctx->dec_x, l->ffn_norm,
                     1, dim, TTS_DEC_NORM_EPS);

        /* SwiGLU: gate = silu(w1(x)) * w3(x), out = w2(gate) */
        if (is_int8) {
            tts_linear_nobias_int8(ctx->dec_gate, ctx->dec_x_norm, l->w1_int8,
                                   l->w1_scale, 1, dim, TTS_DEC_HIDDEN);
            tts_linear_nobias_int8(ctx->dec_up, ctx->dec_x_norm, l->w3_int8,
                                   l->w3_scale, 1, dim, TTS_DEC_HIDDEN);
        } else {
            tts_linear_nobias_bf16(ctx->dec_gate, ctx->dec_x_norm, l->w1_bf16,
                                   1, dim, TTS_DEC_HIDDEN);
            tts_linear_nobias_bf16(ctx->dec_up, ctx->dec_x_norm, l->w3_bf16,
                                   1, dim, TTS_DEC_HIDDEN);
        }
        tts_silu(ctx->dec_gate, TTS_DEC_HIDDEN);
        tts_mul_inplace(ctx->dec_gate, ctx->dec_up, TTS_DEC_HIDDEN);
        if (is_int8) {
            tts_linear_nobias_int8(ctx->dec_ffn_out, ctx->dec_gate, l->w2_int8,
                                   l->w2_scale, 1, TTS_DEC_HIDDEN, dim);
        } else {
            tts_linear_nobias_bf16(ctx->dec_ffn_out, ctx->dec_gate, l->w2_bf16,
                                   1, TTS_DEC_HIDDEN, dim);
        }

        /* Residual */
        tts_add_inplace(ctx->dec_x, ctx->dec_ffn_out, dim);
    }

    /* Final norm */
    tts_rms_norm(out_hidden, ctx->dec_x, dec->norm, 1, dim, TTS_DEC_NORM_EPS);

    /* Advance cache position */
    ctx->kv_cache_len++;
}

/* ========================================================================
 * Prefill (multiple tokens at once)
 * ======================================================================== */

void tts_llm_prefill(tts_ctx_t *ctx, const float *embeds, int seq_len) {
    /*
     * Process multiple tokens through the LLM (e.g., voice prompt + text).
     * embeds: [seq_len, 3072]
     * Updates KV cache. Does not return hidden states (only need last one).
     */
    /* GPU prefill: use CPU prefill (correctness proven), then upload KV cache to GPU
     * for subsequent GPU-accelerated decode. */
    tts_decoder_t *dec = &ctx->decoder;
    int dim = TTS_DEC_DIM;
    int q_dim = TTS_DEC_HEADS * TTS_DEC_HEAD_DIM;
    int kv_dim = TTS_DEC_KV_HEADS * TTS_DEC_HEAD_DIM;
    float scale = 1.0f / sqrtf((float)TTS_DEC_HEAD_DIM);

    /* Allocate temporary buffers for prefill */
    float *x = (float *)malloc((size_t)seq_len * dim * sizeof(float));
    float *x_norm = (float *)malloc((size_t)seq_len * dim * sizeof(float));
    float *q = (float *)malloc((size_t)seq_len * q_dim * sizeof(float));
    float *k = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float *v = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)seq_len * q_dim * sizeof(float));
    float *proj_out = (float *)malloc((size_t)seq_len * dim * sizeof(float));
    float *gate = (float *)malloc((size_t)seq_len * TTS_DEC_HIDDEN * sizeof(float));
    float *up = (float *)malloc((size_t)seq_len * TTS_DEC_HIDDEN * sizeof(float));
    float *ffn_out = (float *)malloc((size_t)seq_len * dim * sizeof(float));
    int *positions = (int *)malloc(seq_len * sizeof(int));
    float *rope_freqs = (float *)malloc((size_t)seq_len * TTS_DEC_HEAD_DIM * sizeof(float));

    if (!x || !x_norm || !q || !k || !v || !attn_out || !proj_out ||
        !gate || !up || !ffn_out || !positions || !rope_freqs) {
        fprintf(stderr, "llm: prefill allocation failed\n");
        goto cleanup;
    }

    int start_pos = ctx->kv_cache_len;
    for (int i = 0; i < seq_len; i++) positions[i] = start_pos + i;
    tts_compute_rope_freqs(rope_freqs, positions, seq_len, TTS_DEC_HEAD_DIM, TTS_ROPE_THETA);

    /* Copy input embeddings */
    memcpy(x, embeds, (size_t)seq_len * dim * sizeof(float));

    int is_int8 = dec->is_int8;

    /* Process each layer */
    for (int layer = 0; layer < TTS_DEC_LAYERS; layer++) {
        tts_dec_layer_t *l = &dec->layers[layer];

        /* RMSNorm */
        tts_rms_norm(x_norm, x, l->attention_norm, seq_len, dim, TTS_DEC_NORM_EPS);

        /* Q, K, V */
        if (is_int8) {
            tts_linear_nobias_int8(q, x_norm, l->wq_int8, l->wq_scale, seq_len, dim, q_dim);
            tts_linear_nobias_int8(k, x_norm, l->wk_int8, l->wk_scale, seq_len, dim, kv_dim);
            tts_linear_nobias_int8(v, x_norm, l->wv_int8, l->wv_scale, seq_len, dim, kv_dim);
        } else {
            tts_linear_nobias_bf16(q, x_norm, l->wq_bf16, seq_len, dim, q_dim);
            tts_linear_nobias_bf16(k, x_norm, l->wk_bf16, seq_len, dim, kv_dim);
            tts_linear_nobias_bf16(v, x_norm, l->wv_bf16, seq_len, dim, kv_dim);
        }

        /* Apply RoPE */
        tts_apply_rope(q, rope_freqs, seq_len, TTS_DEC_HEADS, TTS_DEC_HEAD_DIM);
        tts_apply_rope(k, rope_freqs, seq_len, TTS_DEC_KV_HEADS, TTS_DEC_HEAD_DIM);

        /* Store K, V in cache */
        for (int i = 0; i < seq_len; i++) {
            tts_copy(kv_cache_k_at(ctx, layer, start_pos + i),
                     k + i * kv_dim, kv_dim);
            tts_copy(kv_cache_v_at(ctx, layer, start_pos + i),
                     v + i * kv_dim, kv_dim);
        }

        /* Self-attention over all cached K, V (including newly added) */
        tts_causal_attention(attn_out, q,
                             kv_cache_k_at(ctx, layer, 0),
                             kv_cache_v_at(ctx, layer, 0),
                             seq_len, start_pos + seq_len,
                             TTS_DEC_HEADS, TTS_DEC_KV_HEADS,
                             TTS_DEC_HEAD_DIM, scale,
                             0, start_pos);

        /* Output projection + residual */
        if (is_int8) {
            tts_linear_nobias_int8(proj_out, attn_out, l->wo_int8, l->wo_scale,
                                   seq_len, q_dim, dim);
        } else {
            tts_linear_nobias_bf16(proj_out, attn_out, l->wo_bf16,
                                   seq_len, q_dim, dim);
        }
        tts_add_inplace(x, proj_out, seq_len * dim);

        /* FFN */
        tts_rms_norm(x_norm, x, l->ffn_norm, seq_len, dim, TTS_DEC_NORM_EPS);
        if (is_int8) {
            tts_linear_nobias_int8(gate, x_norm, l->w1_int8, l->w1_scale,
                                   seq_len, dim, TTS_DEC_HIDDEN);
            tts_linear_nobias_int8(up, x_norm, l->w3_int8, l->w3_scale,
                                   seq_len, dim, TTS_DEC_HIDDEN);
        } else {
            tts_linear_nobias_bf16(gate, x_norm, l->w1_bf16,
                                   seq_len, dim, TTS_DEC_HIDDEN);
            tts_linear_nobias_bf16(up, x_norm, l->w3_bf16,
                                   seq_len, dim, TTS_DEC_HIDDEN);
        }
        tts_silu(gate, seq_len * TTS_DEC_HIDDEN);
        tts_mul_inplace(gate, up, seq_len * TTS_DEC_HIDDEN);
        if (is_int8) {
            tts_linear_nobias_int8(ffn_out, gate, l->w2_int8, l->w2_scale,
                                   seq_len, TTS_DEC_HIDDEN, dim);
        } else {
            tts_linear_nobias_bf16(ffn_out, gate, l->w2_bf16,
                                   seq_len, TTS_DEC_HIDDEN, dim);
        }
        tts_add_inplace(x, ffn_out, seq_len * dim);

        if (tts_verbose >= 2)
            fprintf(stderr, "  Prefill layer %d/%d done\n", layer + 1, TTS_DEC_LAYERS);
    }

    /* Update cache position */
    ctx->kv_cache_len += seq_len;

#ifdef USE_CUDA
    /* Upload CPU KV cache to GPU for subsequent GPU-accelerated decode */
    if (tts_cuda_available()) {
        int kv_dim = TTS_DEC_KV_HEADS * TTS_DEC_HEAD_DIM;
        size_t kv_bytes = (size_t)TTS_DEC_LAYERS * ctx->kv_cache_max * kv_dim * sizeof(float);
        tts_cuda_to_device(g_cuda.kv_cache_k_gpu, ctx->kv_cache_k, kv_bytes);
        tts_cuda_to_device(g_cuda.kv_cache_v_gpu, ctx->kv_cache_v, kv_bytes);
        if (tts_verbose)
            fprintf(stderr, "  Uploaded KV cache to GPU (%d positions)\n", ctx->kv_cache_len);
    }
#endif

cleanup:
    free(x); free(x_norm); free(q); free(k); free(v);
    free(attn_out); free(proj_out); free(gate); free(up); free(ffn_out);
    if (positions) free(positions);
    if (rope_freqs) free(rope_freqs);
}
