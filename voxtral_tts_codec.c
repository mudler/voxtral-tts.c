/*
 * voxtral_tts_codec.c - Audio codec decoder
 *
 * Converts audio codes (37 per frame) to raw waveform at 24kHz.
 *
 * Decoder pipeline:
 *   codes -> quantizer.decode -> [292, T]
 *   -> CausalConv1d(292->1024, k=3, s=1)         (decoder_blocks.0)
 *   -> Transformer(2 layers, window=2)             (decoder_blocks.1)
 *   -> CausalConvTranspose1d(1024->1024, k=4, s=2) (decoder_blocks.2)
 *   -> Transformer(2 layers, window=4)             (decoder_blocks.3)
 *   -> CausalConvTranspose1d(1024->1024, k=4, s=2) (decoder_blocks.4)
 *   -> Transformer(2 layers, window=8)             (decoder_blocks.5)
 *   -> CausalConvTranspose1d(1024->1024, k=4, s=2) (decoder_blocks.6)
 *   -> Transformer(2 layers, window=16)            (decoder_blocks.7)
 *   -> CausalConv1d(1024->240, k=7)               (output_proj)
 *   -> reshape [240, T'] -> [T' * 240] waveform
 *
 * Transformer layers use ALiBi attention with sliding windows,
 * QK norm, layer_scale, and weight_norm on convolutions.
 */

#include "voxtral_tts.h"
#include "voxtral_tts_kernels.h"
#include "voxtral_tts_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Weight Norm Reconstruction
 *
 * weight_norm stores weight_v and weight_g separately.
 * Effective weight = weight_v * (weight_g / ||weight_v||_per_output_channel)
 * ======================================================================== */

static float *reconstruct_weight_norm(const float *weight_v, const float *weight_g,
                                       int out_ch, int fan_in) {
    /* weight_v: [out_ch, fan_in], weight_g: [out_ch, 1, 1] or [out_ch] */
    float *weight = (float *)malloc((size_t)out_ch * fan_in * sizeof(float));
    if (!weight) return NULL;

    for (int o = 0; o < out_ch; o++) {
        const float *v_row = weight_v + (size_t)o * fan_in;
        float norm_sq = 0.0f;
        for (int i = 0; i < fan_in; i++)
            norm_sq += v_row[i] * v_row[i];
        float norm = sqrtf(norm_sq + 1e-12f);
        float scale = weight_g[o] / norm;

        float *w_row = weight + (size_t)o * fan_in;
        for (int i = 0; i < fan_in; i++)
            w_row[i] = v_row[i] * scale;
    }
    return weight;
}

/* ========================================================================
 * Weight Loading Helpers
 * ======================================================================== */

static float *load_f32(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) return NULL;
    return safetensors_get_f32(sf, t);
}

/* Load a weight-normed conv: looks for .parametrizations.weight.original0 (weight_g)
 * and .parametrizations.weight.original1 (weight_v), reconstructs effective weight */
static int load_wn_conv(safetensors_file_t *sf, const char *prefix,
                        tts_conv_t *conv, int in_ch, int out_ch,
                        int kernel, int stride, int is_transpose) {
    char name[512];

    /* Weight norm parameters */
    snprintf(name, sizeof(name), "%s.conv.parametrizations.weight.original0", prefix);
    float *weight_g = load_f32(sf, name);
    snprintf(name, sizeof(name), "%s.conv.parametrizations.weight.original1", prefix);
    float *weight_v = load_f32(sf, name);

    if (!weight_g || !weight_v) {
        /* Try non-parametrized weight (for models without weight_norm) */
        snprintf(name, sizeof(name), "%s.conv.weight", prefix);
        float *w = load_f32(sf, name);
        if (!w) {
            fprintf(stderr, "codec: conv weight not found: %s\n", prefix);
            free(weight_g); free(weight_v);
            return -1;
        }
        conv->weight = w;
    } else {
        int fan_in;
        if (is_transpose) {
            /* ConvTranspose1d weight shape: [in_ch, out_ch, kernel] */
            fan_in = out_ch * kernel;
        } else {
            /* Conv1d weight shape: [out_ch, in_ch, kernel] */
            fan_in = in_ch * kernel;
        }
        conv->weight = reconstruct_weight_norm(weight_v, weight_g,
                                                is_transpose ? in_ch : out_ch, fan_in);
        free(weight_g);
        free(weight_v);
        if (!conv->weight) return -1;
    }

    /* Bias (optional) */
    snprintf(name, sizeof(name), "%s.conv.bias", prefix);
    conv->bias = load_f32(sf, name); /* NULL if not found */

    conv->in_ch = in_ch;
    conv->out_ch = out_ch;
    conv->kernel = kernel;
    conv->stride = stride;
    conv->is_transpose = is_transpose;

    return 0;
}

static int load_codec_transformer_layer(safetensors_file_t *sf, const char *prefix,
                                         tts_codec_layer_t *layer, int window_size) {
    char name[768];

    /* Attention weights */
    snprintf(name, sizeof(name), "%s.attention.wq.weight", prefix);
    layer->wq = load_f32(sf, name);
    snprintf(name, sizeof(name), "%s.attention.wk.weight", prefix);
    layer->wk = load_f32(sf, name);
    snprintf(name, sizeof(name), "%s.attention.wv.weight", prefix);
    layer->wv = load_f32(sf, name);
    snprintf(name, sizeof(name), "%s.attention.wo.weight", prefix);
    layer->wo = load_f32(sf, name);

    /* QK norm */
    snprintf(name, sizeof(name), "%s.attention.q_norm.weight", prefix);
    layer->q_norm = load_f32(sf, name);
    snprintf(name, sizeof(name), "%s.attention.k_norm.weight", prefix);
    layer->k_norm = load_f32(sf, name);

    /* Norms */
    snprintf(name, sizeof(name), "%s.attention_norm.weight", prefix);
    layer->attention_norm = load_f32(sf, name);
    snprintf(name, sizeof(name), "%s.ffn_norm.weight", prefix);
    layer->ffn_norm = load_f32(sf, name);

    /* FFN */
    snprintf(name, sizeof(name), "%s.feed_forward.w1.weight", prefix);
    layer->w1 = load_f32(sf, name);
    snprintf(name, sizeof(name), "%s.feed_forward.w2.weight", prefix);
    layer->w2 = load_f32(sf, name);
    snprintf(name, sizeof(name), "%s.feed_forward.w3.weight", prefix);
    layer->w3 = load_f32(sf, name);

    /* Layer scale */
    snprintf(name, sizeof(name), "%s.attention_scale", prefix);
    layer->attn_scale = load_f32(sf, name);
    snprintf(name, sizeof(name), "%s.ffn_scale", prefix);
    layer->ffn_scale = load_f32(sf, name);

    layer->window_size = window_size;

    if (!layer->wq || !layer->wk || !layer->wv || !layer->wo ||
        !layer->attention_norm || !layer->ffn_norm ||
        !layer->w1 || !layer->w2 || !layer->w3) {
        fprintf(stderr, "codec: missing transformer weights at %s\n", prefix);
        return -1;
    }

    return 0;
}

/* ========================================================================
 * Codec Loading
 * ======================================================================== */

int tts_codec_load(tts_codec_t *codec, void *sf_ptr) {
    safetensors_file_t *sf = (safetensors_file_t *)sf_ptr;
    char prefix[512];

    /* === Semantic codebook (precompute embedding) === */
    const safetensor_t *emb_sum_t = safetensors_find(sf,
        "audio_tokenizer.quantizer.semantic_codebook.embedding_sum");
    const safetensor_t *cluster_t = safetensors_find(sf,
        "audio_tokenizer.quantizer.semantic_codebook.cluster_usage");

    if (!emb_sum_t || !cluster_t) {
        fprintf(stderr, "codec: semantic codebook not found\n");
        return -1;
    }

    float *emb_sum = safetensors_get_f32(sf, emb_sum_t);
    float *cluster_usage = safetensors_get_f32(sf, cluster_t);
    if (!emb_sum || !cluster_usage) return -1;

    codec->semantic_embedding = (float *)malloc(
        (size_t)TTS_SEMANTIC_CB_SIZE * TTS_SEMANTIC_CB_DIM * sizeof(float));
    for (int i = 0; i < TTS_SEMANTIC_CB_SIZE; i++) {
        float usage = cluster_usage[i];
        if (usage < 1e-5f) usage = 1e-5f;
        for (int j = 0; j < TTS_SEMANTIC_CB_DIM; j++)
            codec->semantic_embedding[i * TTS_SEMANTIC_CB_DIM + j] =
                emb_sum[i * TTS_SEMANTIC_CB_DIM + j] / usage;
    }
    free(emb_sum);
    free(cluster_usage);

    /* Precompute ALiBi slopes for 8 heads */
    for (int h = 0; h < TTS_CODEC_HEADS; h++)
        codec->alibi_slopes[h] = powf(2.0f, -8.0f / (float)TTS_CODEC_HEADS * (float)(h + 1));

    /* === decoder_blocks.0: CausalConv1d(292->1024, k=3, s=1) === */
    if (load_wn_conv(sf, "audio_tokenizer.decoder_blocks.0",
                     &codec->input_conv, TTS_EMBED_DIM, TTS_CODEC_DIM, 3, 1, 0) != 0)
        return -1;

    /* === Decoder blocks: alternating Transformer + ConvTranspose1d ===
     * decoder_blocks layout (from params.json decoder_*_str):
     *   [0] CausalConv1d(292->1024, k=3, s=1)
     *   [1] Transformer(2 layers)
     *   [2] CausalConvTranspose1d(1024->1024, k=4, s=2)
     *   [3] Transformer(2 layers)
     *   [4] CausalConvTranspose1d(1024->1024, k=4, s=2)
     *   [5] Transformer(2 layers)
     *   [6] CausalConvTranspose1d(1024->1024, k=4, s=2)
     *   [7] Transformer(2 layers)
     */
    int windows[] = {2, 4, 8, 16};
    int block_idx = 1; /* decoder_blocks.1 is first Transformer */
    int layer_idx = 0;

    for (int stage = 0; stage < TTS_CODEC_STAGES; stage++) {
        /* Transformer (2 layers) */
        for (int l = 0; l < TTS_CODEC_LAYERS_PER_STAGE; l++) {
            snprintf(prefix, sizeof(prefix),
                     "audio_tokenizer.decoder_blocks.%d.layers.%d", block_idx, l);
            if (load_codec_transformer_layer(sf, prefix,
                    &codec->transformers[layer_idx], windows[stage]) != 0) {
                fprintf(stderr, "codec: failed to load transformer stage %d layer %d\n",
                        stage, l);
                return -1;
            }
            layer_idx++;
        }
        block_idx++;

        /* ConvTranspose1d upsample (stages 0-2 only, last stage has no upsample) */
        if (stage < TTS_CODEC_STAGES - 1) {
            snprintf(prefix, sizeof(prefix),
                     "audio_tokenizer.decoder_blocks.%d", block_idx);
            if (load_wn_conv(sf, prefix, &codec->stage_convs[stage],
                             TTS_CODEC_DIM, TTS_CODEC_DIM, 4, 2, 1) != 0) {
                fprintf(stderr, "codec: failed to load upsample conv stage %d\n", stage);
                return -1;
            }
            block_idx++;
        }
    }

    /* === Output projection: CausalConv1d(1024->240, k=7) === */
    if (load_wn_conv(sf, "audio_tokenizer.output_proj",
                     &codec->output_proj, TTS_CODEC_DIM, TTS_PATCH_SIZE, 7, 1, 0) != 0)
        return -1;

    if (tts_verbose)
        fprintf(stderr, "  Codec decoder loaded (%d transformer layers, %d stages)\n",
                TTS_CODEC_TOTAL_LAYERS, TTS_CODEC_STAGES);

    return 0;
}

/* ========================================================================
 * Code Embedding
 * ======================================================================== */

static void embed_codes(const tts_codec_t *codec, const int *codes,
                        int n_frames, float *out) {
    /* codes: [n_frames * 37], out: [292, n_frames] (channel-first) */
    for (int t = 0; t < n_frames; t++) {
        const int *fc = codes + t * TTS_CODES_PER_FRAME;

        /* Semantic code -> 256-dim embedding lookup */
        int sem = fc[0] - TTS_AUDIO_SPECIAL_COUNT;
        if (sem < 0) sem = 0;
        if (sem >= TTS_SEMANTIC_CB_SIZE) sem = TTS_SEMANTIC_CB_SIZE - 1;

        const float *sem_emb = codec->semantic_embedding + sem * TTS_SEMANTIC_CB_DIM;
        for (int d = 0; d < TTS_SEMANTIC_CB_DIM; d++)
            out[d * n_frames + t] = sem_emb[d];

        /* Acoustic codes -> FSQ decode: (code * 2 / (levels-1)) - 1 */
        for (int i = 0; i < TTS_ACOUSTIC_DIM; i++) {
            int ac = fc[i + 1] - TTS_AUDIO_SPECIAL_COUNT;
            if (ac < 0) ac = 0;
            if (ac >= TTS_FSQ_LEVELS) ac = TTS_FSQ_LEVELS - 1;
            float val = (float)ac * 2.0f / (float)(TTS_FSQ_LEVELS - 1) - 1.0f;
            out[(TTS_SEMANTIC_CB_DIM + i) * n_frames + t] = val;
        }
    }
}

/* ========================================================================
 * Codec Transformer Layer Forward
 * ======================================================================== */

static void codec_transformer_forward(const tts_codec_layer_t *layer,
                                       const float *alibi_slopes,
                                       float *x, int seq_len) {
    /*
     * x: [seq_len, 1024] (row-major, in-place)
     * ALiBi causal attention with sliding window + QK norm + layer_scale
     */
    int dim = TTS_CODEC_DIM;
    int q_dim = TTS_CODEC_HEADS * TTS_CODEC_HEAD_DIM;
    int kv_dim = TTS_CODEC_KV_HEADS * TTS_CODEC_HEAD_DIM;
    float scale = 1.0f / sqrtf((float)TTS_CODEC_HEAD_DIM);

    float *x_norm = (float *)malloc((size_t)seq_len * dim * sizeof(float));
    float *q = (float *)malloc((size_t)seq_len * q_dim * sizeof(float));
    float *k = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float *v = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)seq_len * q_dim * sizeof(float));
    float *proj_out = (float *)malloc((size_t)seq_len * dim * sizeof(float));
    float *gate = (float *)malloc((size_t)seq_len * TTS_CODEC_HIDDEN * sizeof(float));
    float *up = (float *)malloc((size_t)seq_len * TTS_CODEC_HIDDEN * sizeof(float));
    float *ffn_out = (float *)malloc((size_t)seq_len * dim * sizeof(float));

    if (!x_norm || !q || !k || !v || !attn_out || !proj_out || !gate || !up || !ffn_out)
        goto done;

    /* === Attention === */
    tts_rms_norm(x_norm, x, layer->attention_norm, seq_len, dim, TTS_CODEC_NORM_EPS);

    tts_linear_nobias(q, x_norm, layer->wq, seq_len, dim, q_dim);
    tts_linear_nobias(k, x_norm, layer->wk, seq_len, dim, kv_dim);
    tts_linear_nobias(v, x_norm, layer->wv, seq_len, dim, kv_dim);

    /* QK norm (if weights available) */
    if (layer->q_norm)
        tts_qk_norm(q, q, layer->q_norm, seq_len, q_dim, 1e-6f);
    if (layer->k_norm)
        tts_qk_norm(k, k, layer->k_norm, seq_len, kv_dim, 1e-6f);

    /* ALiBi causal attention with sliding window */
    tts_alibi_attention(attn_out, q, k, v, seq_len,
                        TTS_CODEC_HEADS, TTS_CODEC_KV_HEADS,
                        TTS_CODEC_HEAD_DIM, scale,
                        layer->window_size, alibi_slopes);

    /* Output projection */
    tts_linear_nobias(proj_out, attn_out, layer->wo, seq_len, q_dim, dim);

    /* Layer scale + residual */
    if (layer->attn_scale) {
        for (int s = 0; s < seq_len; s++)
            for (int d = 0; d < dim; d++)
                x[s * dim + d] += proj_out[s * dim + d] * layer->attn_scale[d];
    } else {
        tts_add_inplace(x, proj_out, seq_len * dim);
    }

    /* === FFN === */
    tts_rms_norm(x_norm, x, layer->ffn_norm, seq_len, dim, TTS_CODEC_NORM_EPS);

    tts_linear_nobias(gate, x_norm, layer->w1, seq_len, dim, TTS_CODEC_HIDDEN);
    tts_linear_nobias(up, x_norm, layer->w3, seq_len, dim, TTS_CODEC_HIDDEN);
    tts_silu(gate, seq_len * TTS_CODEC_HIDDEN);
    tts_mul_inplace(gate, up, seq_len * TTS_CODEC_HIDDEN);
    tts_linear_nobias(ffn_out, gate, layer->w2, seq_len, TTS_CODEC_HIDDEN, dim);

    /* Layer scale + residual */
    if (layer->ffn_scale) {
        for (int s = 0; s < seq_len; s++)
            for (int d = 0; d < dim; d++)
                x[s * dim + d] += ffn_out[s * dim + d] * layer->ffn_scale[d];
    } else {
        tts_add_inplace(x, ffn_out, seq_len * dim);
    }

done:
    free(x_norm); free(q); free(k); free(v);
    free(attn_out); free(proj_out); free(gate); free(up); free(ffn_out);
}

/* ========================================================================
 * Codec Decoder Forward
 * ======================================================================== */

void tts_codec_decode(tts_ctx_t *ctx, const int *codes, int n_frames,
                       float **out_samples, int *out_n_samples) {
    tts_codec_t *codec = &ctx->codec;
    int dim = TTS_CODEC_DIM;

    /* Step 1: Embed codes -> [292, n_frames] */
    float *conv_out = NULL;
    float *x = NULL;

    float *emb = (float *)calloc((size_t)TTS_EMBED_DIM * n_frames, sizeof(float));
    if (!emb) goto fail;
    embed_codes(codec, codes, n_frames, emb);

    /* Step 2: Input conv: [292, T] -> [1024, T'] */
    int cur_len = n_frames;
    conv_out = (float *)calloc((size_t)dim * (cur_len + 16), sizeof(float));
    if (!conv_out) goto fail;

    tts_causal_conv1d(conv_out, emb, codec->input_conv.weight,
                      codec->input_conv.bias,
                      TTS_EMBED_DIM, dim, cur_len,
                      codec->input_conv.kernel, codec->input_conv.stride);
    free(emb); emb = NULL;
    /* cur_len stays the same for stride=1 */

    /* Transpose to [T, 1024] for transformer processing */
    x = (float *)malloc((size_t)cur_len * dim * sizeof(float));
    if (!x) goto fail;
    for (int t = 0; t < cur_len; t++)
        for (int d = 0; d < dim; d++)
            x[t * dim + d] = conv_out[d * cur_len + t];
    free(conv_out); conv_out = NULL;

    /* Step 3: Decoder stages */
    int layer_idx = 0;
    for (int stage = 0; stage < TTS_CODEC_STAGES; stage++) {
        /* Transformer layers */
        for (int l = 0; l < TTS_CODEC_LAYERS_PER_STAGE; l++) {
            codec_transformer_forward(&codec->transformers[layer_idx],
                                       codec->alibi_slopes,
                                       x, cur_len);
            layer_idx++;
        }

        /* Upsample conv (stages 0-2) */
        if (stage < TTS_CODEC_STAGES - 1) {
            tts_conv_t *uconv = &codec->stage_convs[stage];

            /* Transpose [T, 1024] -> [1024, T] for conv */
            float *ch_first = (float *)malloc((size_t)dim * cur_len * sizeof(float));
            if (!ch_first) goto fail;
            for (int t = 0; t < cur_len; t++)
                for (int d = 0; d < dim; d++)
                    ch_first[d * cur_len + t] = x[t * dim + d];

            int new_len = 0;
            float *up_out = (float *)calloc((size_t)dim * (cur_len * uconv->stride + uconv->kernel + 8),
                                            sizeof(float));
            if (!up_out) { free(ch_first); goto fail; }

            tts_causal_conv_transpose_1d(up_out, ch_first, uconv->weight, uconv->bias,
                                          uconv->in_ch, uconv->out_ch, cur_len,
                                          uconv->kernel, uconv->stride, &new_len);
            free(ch_first);

            /* Transpose back [1024, new_len] -> [new_len, 1024] */
            free(x);
            x = (float *)malloc((size_t)new_len * dim * sizeof(float));
            if (!x) { free(up_out); goto fail; }
            for (int t = 0; t < new_len; t++)
                for (int d = 0; d < dim; d++)
                    x[t * dim + d] = up_out[d * new_len + t];
            free(up_out);
            cur_len = new_len;
        }

        if (tts_verbose >= 2)
            fprintf(stderr, "  Codec stage %d done, seq_len=%d\n", stage, cur_len);
    }

    /* Step 4: Transpose [T, 1024] -> [1024, T] for output conv */
    float *final_ch_first = (float *)malloc((size_t)dim * cur_len * sizeof(float));
    if (!final_ch_first) goto fail;
    for (int t = 0; t < cur_len; t++)
        for (int d = 0; d < dim; d++)
            final_ch_first[d * cur_len + t] = x[t * dim + d];
    free(x); x = NULL;

    /* Output projection: [1024, T] -> [240, T] */
    float *out_proj = (float *)calloc((size_t)TTS_PATCH_SIZE * (cur_len + 16), sizeof(float));
    if (!out_proj) { free(final_ch_first); goto fail; }

    tts_causal_conv1d(out_proj, final_ch_first, codec->output_proj.weight,
                      codec->output_proj.bias,
                      dim, TTS_PATCH_SIZE, cur_len,
                      codec->output_proj.kernel, codec->output_proj.stride);
    free(final_ch_first);

    /* Step 5: Reshape [240, T] -> [T * 240] waveform */
    int n_samples = cur_len * TTS_PATCH_SIZE;
    float *samples = (float *)malloc(n_samples * sizeof(float));
    if (!samples) { free(out_proj); goto fail; }

    for (int t = 0; t < cur_len; t++)
        for (int h = 0; h < TTS_PATCH_SIZE; h++)
            samples[t * TTS_PATCH_SIZE + h] = out_proj[h * cur_len + t];
    free(out_proj);

    *out_samples = samples;
    *out_n_samples = n_samples;

    if (tts_verbose)
        fprintf(stderr, "  Codec decoded: %d frames -> %d samples (%.2fs)\n",
                n_frames, n_samples, (float)n_samples / TTS_SAMPLE_RATE);
    return;

fail:
    free(emb); free(conv_out); free(x);
    *out_samples = NULL;
    *out_n_samples = 0;
    fprintf(stderr, "codec: decode failed (allocation error)\n");
}
