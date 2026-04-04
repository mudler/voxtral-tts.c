/*
 * voxtral_tts_acoustic.c - Flow-matching acoustic transformer
 *
 * 3-layer bidirectional transformer with flow matching.
 * Converts LLM hidden states to audio codes (1 semantic + 36 acoustic per frame).
 *
 * Flow matching: 8 Euler ODE steps with classifier-free guidance (alpha=1.2).
 */

#include "voxtral_tts.h"
#include "voxtral_tts_kernels.h"
#include "voxtral_tts_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

static float *load_f32(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "acoustic: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static uint16_t *load_bf16_direct(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "acoustic: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

static int8_t *load_int8_direct(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "acoustic: int8 weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_int8_direct(sf, t);
}

static float *load_scale(safetensors_file_t *sf, const char *weight_name) {
    char scale_name[512];
    snprintf(scale_name, sizeof(scale_name), "%s_scale", weight_name);
    const safetensor_t *t = safetensors_find(sf, scale_name);
    if (!t) {
        fprintf(stderr, "acoustic: scale not found: %s\n", scale_name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

/* Check if acoustic model uses INT8 quantized weights */
static int detect_int8_acoustic(safetensors_file_t *sf) {
    const safetensor_t *t = safetensors_find(sf,
        "acoustic_transformer.layers.0.attention.wq.weight");
    if (t && safetensor_is_int8(t)) return 1;
    return 0;
}

int tts_acoustic_load(tts_acoustic_t *ac, void *sf_ptr) {
    safetensors_file_t *sf = (safetensors_file_t *)sf_ptr;
    char name[512];

    /* Auto-detect INT8 vs BF16 */
    int use_int8 = detect_int8_acoustic(sf);
    ac->is_int8 = use_int8;

    if (tts_verbose)
        fprintf(stderr, "  Acoustic weights: %s\n", use_int8 ? "INT8 quantized" : "BF16");

    /* Time embedding inverse frequencies
     * May not be in checkpoint (it's a precomputed buffer).
     * Formula: inv_freq[i] = exp(-log(theta) * i / (dim/2)) for i in 0..dim/2-1
     * where theta=10000.0 and dim=TTS_AC_DIM=3072 */
    ac->time_inv_freq = load_f32(sf, "acoustic_transformer.time_embedding.inv_freq");
    if (!ac->time_inv_freq) {
        /* Compute it ourselves */
        int half_dim = TTS_AC_DIM / 2;
        ac->time_inv_freq = (float *)malloc(half_dim * sizeof(float));
        if (!ac->time_inv_freq) return -1;
        float theta = 10000.0f;
        for (int i = 0; i < half_dim; i++) {
            ac->time_inv_freq[i] = expf(-logf(theta) * (float)i / (float)half_dim);
        }
        if (tts_verbose)
            fprintf(stderr, "  Computed time_embedding.inv_freq (%d values)\n", half_dim);
    }

    /* Input projections */
    if (use_int8) {
        ac->input_proj_int8 = load_int8_direct(sf, "acoustic_transformer.input_projection.weight");
        ac->input_proj_scale = load_scale(sf, "acoustic_transformer.input_projection.weight");
        ac->time_proj_int8 = load_int8_direct(sf, "acoustic_transformer.time_projection.weight");
        ac->time_proj_scale = load_scale(sf, "acoustic_transformer.time_projection.weight");
        ac->llm_proj_int8 = load_int8_direct(sf, "acoustic_transformer.llm_projection.weight");
        ac->llm_proj_scale = load_scale(sf, "acoustic_transformer.llm_projection.weight");
        if (!ac->input_proj_int8 || !ac->input_proj_scale ||
            !ac->time_proj_int8 || !ac->time_proj_scale ||
            !ac->llm_proj_int8 || !ac->llm_proj_scale)
            return -1;
    } else {
        ac->input_proj_bf16 = load_bf16_direct(sf, "acoustic_transformer.input_projection.weight");
        ac->time_proj_bf16 = load_bf16_direct(sf, "acoustic_transformer.time_projection.weight");
        ac->llm_proj_bf16 = load_bf16_direct(sf, "acoustic_transformer.llm_projection.weight");
        if (!ac->input_proj_bf16 || !ac->time_proj_bf16 || !ac->llm_proj_bf16)
            return -1;
    }

    /* Output heads — try INT8 first, fall back to BF16 (these may not be quantized) */
    if (use_int8) {
        ac->semantic_out_int8 = load_int8_direct(sf,
            "acoustic_transformer.semantic_codebook_output.weight");
        if (ac->semantic_out_int8) {
            ac->semantic_out_scale = load_scale(sf,
                "acoustic_transformer.semantic_codebook_output.weight");
            if (!ac->semantic_out_scale) return -1;
        }
    }
    if (!ac->semantic_out_int8) {
        ac->semantic_out_bf16 = load_bf16_direct(sf,
            "acoustic_transformer.semantic_codebook_output.weight");
        if (!ac->semantic_out_bf16) return -1;
    }

    if (use_int8) {
        ac->acoustic_out_int8 = load_int8_direct(sf,
            "acoustic_transformer.acoustic_codebook_output.weight");
        if (ac->acoustic_out_int8) {
            ac->acoustic_out_scale = load_scale(sf,
                "acoustic_transformer.acoustic_codebook_output.weight");
            if (!ac->acoustic_out_scale) return -1;
        }
    }
    if (!ac->acoustic_out_int8) {
        ac->acoustic_out_bf16 = load_bf16_direct(sf,
            "acoustic_transformer.acoustic_codebook_output.weight");
        if (!ac->acoustic_out_bf16) return -1;
    }

    /* Check for bias (always f32) */
    const safetensor_t *bias_t = safetensors_find(sf,
        "acoustic_transformer.semantic_codebook_output.bias");
    ac->semantic_out_bias = bias_t ? safetensors_get_f32(sf, bias_t) : NULL;

    /* Final norm */
    ac->norm = load_f32(sf, "acoustic_transformer.norm.weight");
    if (!ac->norm) return -1;

    /* Transformer layers */
    for (int i = 0; i < TTS_AC_LAYERS; i++) {
        tts_ac_layer_t *l = &ac->layers[i];

        if (use_int8) {
            snprintf(name, sizeof(name), "acoustic_transformer.layers.%d.attention.wq.weight", i);
            l->wq_int8 = load_int8_direct(sf, name);
            l->wq_scale = load_scale(sf, name);
            snprintf(name, sizeof(name), "acoustic_transformer.layers.%d.attention.wk.weight", i);
            l->wk_int8 = load_int8_direct(sf, name);
            l->wk_scale = load_scale(sf, name);
            snprintf(name, sizeof(name), "acoustic_transformer.layers.%d.attention.wv.weight", i);
            l->wv_int8 = load_int8_direct(sf, name);
            l->wv_scale = load_scale(sf, name);
            snprintf(name, sizeof(name), "acoustic_transformer.layers.%d.attention.wo.weight", i);
            l->wo_int8 = load_int8_direct(sf, name);
            l->wo_scale = load_scale(sf, name);

            snprintf(name, sizeof(name), "acoustic_transformer.layers.%d.feed_forward.w1.weight", i);
            l->w1_int8 = load_int8_direct(sf, name);
            l->w1_scale = load_scale(sf, name);
            snprintf(name, sizeof(name), "acoustic_transformer.layers.%d.feed_forward.w2.weight", i);
            l->w2_int8 = load_int8_direct(sf, name);
            l->w2_scale = load_scale(sf, name);
            snprintf(name, sizeof(name), "acoustic_transformer.layers.%d.feed_forward.w3.weight", i);
            l->w3_int8 = load_int8_direct(sf, name);
            l->w3_scale = load_scale(sf, name);

            if (!l->wq_int8 || !l->wq_scale || !l->wk_int8 || !l->wk_scale ||
                !l->wv_int8 || !l->wv_scale || !l->wo_int8 || !l->wo_scale ||
                !l->w1_int8 || !l->w1_scale || !l->w2_int8 || !l->w2_scale ||
                !l->w3_int8 || !l->w3_scale) {
                fprintf(stderr, "acoustic: failed to load INT8 layer %d\n", i);
                return -1;
            }
        } else {
            snprintf(name, sizeof(name), "acoustic_transformer.layers.%d.attention.wq.weight", i);
            l->wq_bf16 = load_bf16_direct(sf, name);
            snprintf(name, sizeof(name), "acoustic_transformer.layers.%d.attention.wk.weight", i);
            l->wk_bf16 = load_bf16_direct(sf, name);
            snprintf(name, sizeof(name), "acoustic_transformer.layers.%d.attention.wv.weight", i);
            l->wv_bf16 = load_bf16_direct(sf, name);
            snprintf(name, sizeof(name), "acoustic_transformer.layers.%d.attention.wo.weight", i);
            l->wo_bf16 = load_bf16_direct(sf, name);

            snprintf(name, sizeof(name), "acoustic_transformer.layers.%d.feed_forward.w1.weight", i);
            l->w1_bf16 = load_bf16_direct(sf, name);
            snprintf(name, sizeof(name), "acoustic_transformer.layers.%d.feed_forward.w2.weight", i);
            l->w2_bf16 = load_bf16_direct(sf, name);
            snprintf(name, sizeof(name), "acoustic_transformer.layers.%d.feed_forward.w3.weight", i);
            l->w3_bf16 = load_bf16_direct(sf, name);

            if (!l->wq_bf16 || !l->wk_bf16 || !l->wv_bf16 || !l->wo_bf16 ||
                !l->w1_bf16 || !l->w2_bf16 || !l->w3_bf16) {
                fprintf(stderr, "acoustic: failed to load BF16 layer %d\n", i);
                return -1;
            }
        }

        /* Norms (always f32) */
        snprintf(name, sizeof(name), "acoustic_transformer.layers.%d.attention_norm.weight", i);
        l->attention_norm = load_f32(sf, name);
        snprintf(name, sizeof(name), "acoustic_transformer.layers.%d.ffn_norm.weight", i);
        l->ffn_norm = load_f32(sf, name);

        if (!l->attention_norm || !l->ffn_norm) {
            fprintf(stderr, "acoustic: failed to load norms for layer %d\n", i);
            return -1;
        }
    }

    if (tts_verbose)
        fprintf(stderr, "  Acoustic transformer loaded (%d layers, %s)\n",
                TTS_AC_LAYERS, use_int8 ? "INT8" : "BF16");

    return 0;
}

/* ========================================================================
 * Acoustic Transformer Buffers
 * ======================================================================== */

static int alloc_acoustic_buffers(tts_ctx_t *ctx) {
    int seq = 3; /* always 3 tokens: [noise, time, llm] */
    int dim = TTS_AC_DIM;
    int q_dim = TTS_AC_HEADS * TTS_AC_HEAD_DIM;
    int kv_dim = TTS_AC_KV_HEADS * TTS_AC_HEAD_DIM;

    ctx->ac_tokens     = (float *)calloc(seq * dim, sizeof(float));
    ctx->ac_tokens_norm = (float *)calloc(seq * dim, sizeof(float));
    ctx->ac_q          = (float *)calloc(seq * q_dim, sizeof(float));
    ctx->ac_k          = (float *)calloc(seq * kv_dim, sizeof(float));
    ctx->ac_v          = (float *)calloc(seq * kv_dim, sizeof(float));
    ctx->ac_attn_out   = (float *)calloc(seq * q_dim, sizeof(float));
    ctx->ac_proj_out   = (float *)calloc(seq * dim, sizeof(float));
    ctx->ac_gate       = (float *)calloc(seq * TTS_AC_HIDDEN, sizeof(float));
    ctx->ac_up         = (float *)calloc(seq * TTS_AC_HIDDEN, sizeof(float));
    ctx->ac_ffn_out    = (float *)calloc(seq * dim, sizeof(float));
    ctx->ac_time_emb   = (float *)calloc(dim, sizeof(float));
    ctx->ac_velocity   = (float *)calloc(TTS_ACOUSTIC_DIM, sizeof(float));
    ctx->ac_noise      = (float *)calloc(TTS_ACOUSTIC_DIM, sizeof(float));

    if (!ctx->ac_tokens || !ctx->ac_tokens_norm || !ctx->ac_q || !ctx->ac_k ||
        !ctx->ac_v || !ctx->ac_attn_out || !ctx->ac_proj_out ||
        !ctx->ac_gate || !ctx->ac_up || !ctx->ac_ffn_out ||
        !ctx->ac_time_emb || !ctx->ac_velocity || !ctx->ac_noise) {
        fprintf(stderr, "acoustic: buffer allocation failed\n");
        return -1;
    }
    return 0;
}

/* ========================================================================
 * Time Embedding (sinusoidal)
 * ======================================================================== */

static void compute_time_embedding(const tts_acoustic_t *ac, float t_val,
                                    float *out, int dim) {
    /* out[i] = cos(t * inv_freq[i]) for i < dim/2
     * out[i] = sin(t * inv_freq[i-dim/2]) for i >= dim/2 */
    int half_dim = dim / 2;
    for (int i = 0; i < half_dim; i++) {
        float angle = t_val * ac->time_inv_freq[i];
        out[i] = cosf(angle);
        out[half_dim + i] = sinf(angle);
    }
}

/* ========================================================================
 * Predict Velocity (single forward pass through 3-layer transformer)
 * ======================================================================== */

static void predict_velocity(tts_ctx_t *ctx, const float *x_t,
                              const float *llm_hidden, float t_val,
                              float *out_velocity) {
#ifdef USE_CUDA
    if (tts_cuda_available()) {
        tts_cuda_predict_velocity(out_velocity, x_t, llm_hidden, t_val);
        return;
    }
#endif
    tts_acoustic_t *ac = &ctx->acoustic;
    int dim = TTS_AC_DIM;
    int seq = 3;
    int q_dim = TTS_AC_HEADS * TTS_AC_HEAD_DIM;
    int kv_dim = TTS_AC_KV_HEADS * TTS_AC_HEAD_DIM;
    float scale = 1.0f / sqrtf((float)TTS_AC_HEAD_DIM);

    int is_int8 = ac->is_int8;

    /* Build 3 tokens: [input_proj(x_t), time_proj(time_emb(t)), llm_proj(h)] */
    float *tokens = ctx->ac_tokens;

    /* Token 0: input_projection(x_t) — [36] -> [3072] */
    if (is_int8) {
        tts_linear_nobias_int8(tokens + 0 * dim, x_t, ac->input_proj_int8,
                               ac->input_proj_scale, 1, TTS_ACOUSTIC_DIM, dim);
    } else {
        tts_linear_nobias_bf16(tokens + 0 * dim, x_t, ac->input_proj_bf16,
                               1, TTS_ACOUSTIC_DIM, dim);
    }

    /* Token 1: time_projection(time_embedding(t)) */
    compute_time_embedding(ac, t_val, ctx->ac_time_emb, dim);
    if (is_int8) {
        tts_linear_nobias_int8(tokens + 1 * dim, ctx->ac_time_emb, ac->time_proj_int8,
                               ac->time_proj_scale, 1, dim, dim);
    } else {
        tts_linear_nobias_bf16(tokens + 1 * dim, ctx->ac_time_emb, ac->time_proj_bf16,
                               1, dim, dim);
    }

    /* Token 2: llm_projection(llm_hidden) */
    if (is_int8) {
        tts_linear_nobias_int8(tokens + 2 * dim, llm_hidden, ac->llm_proj_int8,
                               ac->llm_proj_scale, 1, dim, dim);
    } else {
        tts_linear_nobias_bf16(tokens + 2 * dim, llm_hidden, ac->llm_proj_bf16,
                               1, dim, dim);
    }

    /* Forward through 3 bidirectional transformer layers */
    for (int layer = 0; layer < TTS_AC_LAYERS; layer++) {
        tts_ac_layer_t *l = &ac->layers[layer];

        /* RMSNorm */
        tts_rms_norm(ctx->ac_tokens_norm, tokens, l->attention_norm,
                     seq, dim, TTS_AC_NORM_EPS);

        /* Q, K, V */
        if (is_int8) {
            tts_linear_nobias_int8(ctx->ac_q, ctx->ac_tokens_norm, l->wq_int8,
                                   l->wq_scale, seq, dim, q_dim);
            tts_linear_nobias_int8(ctx->ac_k, ctx->ac_tokens_norm, l->wk_int8,
                                   l->wk_scale, seq, dim, kv_dim);
            tts_linear_nobias_int8(ctx->ac_v, ctx->ac_tokens_norm, l->wv_int8,
                                   l->wv_scale, seq, dim, kv_dim);
        } else {
            tts_linear_nobias_bf16(ctx->ac_q, ctx->ac_tokens_norm, l->wq_bf16,
                                   seq, dim, q_dim);
            tts_linear_nobias_bf16(ctx->ac_k, ctx->ac_tokens_norm, l->wk_bf16,
                                   seq, dim, kv_dim);
            tts_linear_nobias_bf16(ctx->ac_v, ctx->ac_tokens_norm, l->wv_bf16,
                                   seq, dim, kv_dim);
        }

        /* Bidirectional attention (no causal mask, no positional encoding) */
        tts_bidirectional_attention(ctx->ac_attn_out, ctx->ac_q, ctx->ac_k,
                                    ctx->ac_v, seq,
                                    TTS_AC_HEADS, TTS_AC_KV_HEADS,
                                    TTS_AC_HEAD_DIM, scale);

        /* Output projection + residual */
        if (is_int8) {
            tts_linear_nobias_int8(ctx->ac_proj_out, ctx->ac_attn_out, l->wo_int8,
                                   l->wo_scale, seq, q_dim, dim);
        } else {
            tts_linear_nobias_bf16(ctx->ac_proj_out, ctx->ac_attn_out, l->wo_bf16,
                                   seq, q_dim, dim);
        }
        tts_add_inplace(tokens, ctx->ac_proj_out, seq * dim);

        /* FFN: RMSNorm -> SwiGLU */
        tts_rms_norm(ctx->ac_tokens_norm, tokens, l->ffn_norm,
                     seq, dim, TTS_AC_NORM_EPS);
        if (is_int8) {
            tts_linear_nobias_int8(ctx->ac_gate, ctx->ac_tokens_norm, l->w1_int8,
                                   l->w1_scale, seq, dim, TTS_AC_HIDDEN);
            tts_linear_nobias_int8(ctx->ac_up, ctx->ac_tokens_norm, l->w3_int8,
                                   l->w3_scale, seq, dim, TTS_AC_HIDDEN);
        } else {
            tts_linear_nobias_bf16(ctx->ac_gate, ctx->ac_tokens_norm, l->w1_bf16,
                                   seq, dim, TTS_AC_HIDDEN);
            tts_linear_nobias_bf16(ctx->ac_up, ctx->ac_tokens_norm, l->w3_bf16,
                                   seq, dim, TTS_AC_HIDDEN);
        }
        tts_silu(ctx->ac_gate, seq * TTS_AC_HIDDEN);
        tts_mul_inplace(ctx->ac_gate, ctx->ac_up, seq * TTS_AC_HIDDEN);
        if (is_int8) {
            tts_linear_nobias_int8(ctx->ac_ffn_out, ctx->ac_gate, l->w2_int8,
                                   l->w2_scale, seq, TTS_AC_HIDDEN, dim);
        } else {
            tts_linear_nobias_bf16(ctx->ac_ffn_out, ctx->ac_gate, l->w2_bf16,
                                   seq, TTS_AC_HIDDEN, dim);
        }
        tts_add_inplace(tokens, ctx->ac_ffn_out, seq * dim);
    }

    /* Final norm on first token (the noise/x_t position) */
    float normed[TTS_AC_DIM]; /* stack alloc ok, only 3072 floats */
    tts_rms_norm(normed, tokens, ac->norm, 1, dim, TTS_AC_NORM_EPS);

    /* Predict velocity: acoustic_codebook_output(normed) -> [36] */
    if (ac->acoustic_out_int8) {
        tts_linear_nobias_int8(out_velocity, normed, ac->acoustic_out_int8,
                               ac->acoustic_out_scale, 1, dim, TTS_ACOUSTIC_DIM);
    } else {
        tts_linear_nobias_bf16(out_velocity, normed, ac->acoustic_out_bf16,
                               1, dim, TTS_ACOUSTIC_DIM);
    }
}

/* ========================================================================
 * Main Forward: Generate Audio Codes for One Frame
 * ======================================================================== */

void tts_acoustic_forward(tts_ctx_t *ctx, const float *llm_hidden,
                           int *out_codes) {
    /*
     * Given LLM hidden state [3072], produce 37 audio codes:
     *   out_codes[0] = semantic code (0..8191 or END_AUDIO special)
     *   out_codes[1..36] = acoustic codes (0..20 FSQ) + offset
     */
    tts_acoustic_t *ac = &ctx->acoustic;
    int dim = TTS_AC_DIM;

    /* Allocate buffers on first call */
    if (!ctx->ac_tokens) {
        if (alloc_acoustic_buffers(ctx) != 0) return;
    }

    /* === Step 1: Semantic code prediction === */
    float *semantic_logits = (float *)malloc(TTS_SEMANTIC_CB_PADDED * sizeof(float));
    if (!semantic_logits) return;

    if (ac->semantic_out_int8) {
        tts_linear_int8(semantic_logits, llm_hidden, ac->semantic_out_int8,
                        ac->semantic_out_scale, ac->semantic_out_bias,
                        1, dim, TTS_SEMANTIC_CB_PADDED);
    } else {
        tts_linear_bf16(semantic_logits, llm_hidden, ac->semantic_out_bf16,
                        ac->semantic_out_bias, 1, dim, TTS_SEMANTIC_CB_PADDED);
    }

    /* Mask invalid tokens */
    semantic_logits[TTS_AUDIO_SPECIAL_EMPTY] = -1e30f; /* empty not allowed */
    /* Mask padding beyond valid semantic range */
    for (int i = TTS_AUDIO_SPECIAL_COUNT + TTS_SEMANTIC_CB_SIZE;
         i < TTS_SEMANTIC_CB_PADDED; i++) {
        semantic_logits[i] = -1e30f;
    }

    /* Greedy argmax */
    int semantic_code = 0;
    float best = semantic_logits[0];
    for (int i = 1; i < TTS_SEMANTIC_CB_PADDED; i++) {
        if (semantic_logits[i] > best) {
            best = semantic_logits[i];
            semantic_code = i;
        }
    }
    free(semantic_logits);

    out_codes[0] = semantic_code;

    /* === Step 2: Check for END_AUDIO === */
    int should_decode = (semantic_code != TTS_AUDIO_SPECIAL_END);

    if (!should_decode) {
        /* Fill acoustic codes with EMPTY */
        for (int i = 1; i < TTS_CODES_PER_FRAME; i++) {
            out_codes[i] = TTS_AUDIO_SPECIAL_EMPTY + TTS_AUDIO_SPECIAL_COUNT;
        }
        return;
    }

    /* === Step 3: Flow matching — 8 Euler steps with CFG === */
    float x[TTS_ACOUSTIC_DIM]; /* current sample */
    float v_cond[TTS_ACOUSTIC_DIM];
    float v_uncond[TTS_ACOUSTIC_DIM];
    float llm_zero[TTS_AC_DIM]; /* zero hidden for unconditional */

    /* Initialize from Gaussian noise */
    tts_randn_fill(&ctx->rng_state, x, TTS_ACOUSTIC_DIM);
    tts_scale(x, TTS_NOISE_SCALE, TTS_ACOUSTIC_DIM);

    memset(llm_zero, 0, dim * sizeof(float));

    /* Timesteps: linspace(0, 1, n_steps) = [0, 1/7, 2/7, ..., 6/7, 1]
     * We iterate over n_steps-1 intervals with variable dt = 1/(n_steps-1) */
    float timesteps[TTS_FLOW_STEPS];
    for (int i = 0; i < TTS_FLOW_STEPS; i++)
        timesteps[i] = (float)i / (float)(TTS_FLOW_STEPS - 1);

    for (int step = 0; step < TTS_FLOW_STEPS - 1; step++) {
        float t = timesteps[step];
        float dt = timesteps[step + 1] - timesteps[step];

        /* Predict conditional velocity */
        predict_velocity(ctx, x, llm_hidden, t, v_cond);

        /* Predict unconditional velocity (zero LLM hidden) */
        predict_velocity(ctx, x, llm_zero, t, v_uncond);

        /* CFG combination: v = alpha * v_cond + (1 - alpha) * v_uncond */
        for (int i = 0; i < TTS_ACOUSTIC_DIM; i++) {
            float v = TTS_CFG_ALPHA * v_cond[i] + (1.0f - TTS_CFG_ALPHA) * v_uncond[i];
            x[i] += v * dt;
        }
    }

    /* === Step 4: Quantize to FSQ codes === */
    for (int i = 0; i < TTS_ACOUSTIC_DIM; i++) {
        /* Clamp to [-1, 1] */
        float val = x[i];
        if (val > 1.0f) val = 1.0f;
        if (val < -1.0f) val = -1.0f;

        /* Scale to [0, FSQ_LEVELS-1] and round */
        float scaled = ((val + 1.0f) / 2.0f) * (float)(TTS_FSQ_LEVELS - 1);
        int code = (int)(scaled + 0.5f);
        if (code < 0) code = 0;
        if (code >= TTS_FSQ_LEVELS) code = TTS_FSQ_LEVELS - 1;

        /* Offset by special token count */
        out_codes[i + 1] = code + TTS_AUDIO_SPECIAL_COUNT;
    }
}
