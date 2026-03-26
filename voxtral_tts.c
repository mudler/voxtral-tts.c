/*
 * voxtral_tts.c - Main orchestrator for Voxtral TTS inference
 *
 * Loads the model, constructs prompts, runs the inference pipeline.
 */

#include "voxtral_tts.h"
#include "voxtral_tts_kernels.h"
#include "voxtral_tts_safetensors.h"
#include "voxtral_tts_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int tts_verbose = 0;

/* ========================================================================
 * Model Loading
 * ======================================================================== */

tts_ctx_t *tts_load(const char *model_dir) {
    tts_ctx_t *ctx = (tts_ctx_t *)calloc(1, sizeof(tts_ctx_t));
    if (!ctx) return NULL;

    snprintf(ctx->model_dir, sizeof(ctx->model_dir), "%s", model_dir);
    ctx->verbose = tts_verbose;

    /* Seed RNG */
    tts_rng_seed(&ctx->rng_state, (uint64_t)time(NULL));

    /* Open safetensors file */
    char path[1024];
    snprintf(path, sizeof(path), "%s/consolidated.safetensors", model_dir);
    fprintf(stderr, "Loading model from %s\n", path);

    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) {
        fprintf(stderr, "Failed to open %s\n", path);
        tts_free(ctx);
        return NULL;
    }
    ctx->safetensors = sf;

    fprintf(stderr, "  %d tensors found\n", sf->num_tensors);

    /* Load tokenizer */
    snprintf(path, sizeof(path), "%s/tekken.json", model_dir);
    if (tts_tokenizer_load(path) != 0) {
        fprintf(stderr, "Failed to load tokenizer\n");
        tts_free(ctx);
        return NULL;
    }

    /* Load LLM decoder */
    fprintf(stderr, "Loading LLM decoder...\n");
    if (tts_llm_load(&ctx->decoder, sf) != 0) {
        fprintf(stderr, "Failed to load LLM decoder\n");
        tts_free(ctx);
        return NULL;
    }

    /* Allocate KV cache (8192 positions should be enough for most TTS) */
    if (tts_llm_kv_cache_alloc(ctx, 8192) != 0) {
        tts_free(ctx);
        return NULL;
    }

    /* Load acoustic transformer */
    fprintf(stderr, "Loading acoustic transformer...\n");
    if (tts_acoustic_load(&ctx->acoustic, sf) != 0) {
        fprintf(stderr, "Failed to load acoustic transformer\n");
        tts_free(ctx);
        return NULL;
    }

    /* Load audio codec decoder */
    fprintf(stderr, "Loading audio codec decoder...\n");
    if (tts_codec_load(&ctx->codec, sf) != 0) {
        fprintf(stderr, "Failed to load codec decoder\n");
        tts_free(ctx);
        return NULL;
    }

    /* Load audio codebook embeddings */
    fprintf(stderr, "Loading audio embeddings...\n");
    if (tts_audio_embed_load(&ctx->audio_embed, sf) != 0) {
        fprintf(stderr, "Failed to load audio embeddings\n");
        tts_free(ctx);
        return NULL;
    }

    /* Initialize GPU backend */
#ifdef USE_METAL
    fprintf(stderr, "Initializing Metal...\n");
    if (tts_metal_init()) {
        fprintf(stderr, "  Metal GPU acceleration enabled\n");
    } else {
        fprintf(stderr, "  Metal not available, using CPU\n");
    }
#endif
#ifdef USE_CUDA
    fprintf(stderr, "Initializing CUDA...\n");
    if (tts_cuda_init(ctx->kv_cache_max) == 0) {
        fprintf(stderr, "Uploading weights to GPU...\n");
        tts_cuda_upload_llm_weights(&ctx->decoder);
        tts_cuda_upload_acoustic_weights(&ctx->acoustic);
    } else {
        fprintf(stderr, "CUDA not available, using CPU\n");
    }
#endif

    fprintf(stderr, "Model loaded successfully.\n");
    return ctx;
}

void tts_free(tts_ctx_t *ctx) {
    if (!ctx) return;

#ifdef USE_CUDA
    tts_cuda_free();
#endif
#ifdef USE_METAL
    tts_metal_shutdown();
#endif

    /* Close safetensors (unmaps memory, invalidates bf16 pointers) */
    if (ctx->safetensors) safetensors_close((safetensors_file_t *)ctx->safetensors);

    /* Free allocated f32 weights */
    for (int i = 0; i < TTS_DEC_LAYERS; i++) {
        free(ctx->decoder.layers[i].attention_norm);
        free(ctx->decoder.layers[i].ffn_norm);
    }
    free(ctx->decoder.norm);

    for (int i = 0; i < TTS_AC_LAYERS; i++) {
        free(ctx->acoustic.layers[i].attention_norm);
        free(ctx->acoustic.layers[i].ffn_norm);
    }
    free(ctx->acoustic.time_inv_freq);
    free(ctx->acoustic.norm);
    free(ctx->acoustic.semantic_out_bias);

    free(ctx->codec.semantic_embedding);

    /* Free KV cache */
    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);

    /* Free working buffers */
    free(ctx->dec_x); free(ctx->dec_x_norm);
    free(ctx->dec_q); free(ctx->dec_k); free(ctx->dec_v);
    free(ctx->dec_attn_out); free(ctx->dec_proj_out);
    free(ctx->dec_gate); free(ctx->dec_up); free(ctx->dec_ffn_out);
    free(ctx->dec_rope_freqs);

    free(ctx->ac_tokens); free(ctx->ac_tokens_norm);
    free(ctx->ac_q); free(ctx->ac_k); free(ctx->ac_v);
    free(ctx->ac_attn_out); free(ctx->ac_proj_out);
    free(ctx->ac_gate); free(ctx->ac_up); free(ctx->ac_ffn_out);
    free(ctx->ac_time_emb); free(ctx->ac_velocity); free(ctx->ac_noise);

    /* Free voice */
    if (ctx->voice) tts_voice_free(ctx->voice);

    /* Free tokenizer */
    tts_tokenizer_free();

    free(ctx);
}

void tts_set_seed(tts_ctx_t *ctx, uint64_t seed) {
    tts_rng_seed(&ctx->rng_state, seed);
}

/* ========================================================================
 * TTS Generation Pipeline
 * ======================================================================== */

int tts_generate(tts_ctx_t *ctx, const char *text, const char *voice_name,
                 float **out_samples, int *out_n_samples) {
    /*
     * Full TTS pipeline:
     *
     * Prompt format (from mistral_common encode_speech_request):
     *   [BOS=1] [BEGIN_AUDIO=25] [AUDIO=24 x N] [/INST=36] text_tokens [INST=35] [BEGIN_AUDIO=25]
     *
     * Where the N AUDIO token positions are replaced with voice embeddings
     * (pre-computed [N, 3072] BF16 tensors from .pt files).
     *
     * After prefill, the model autoregressively generates audio codes:
     *   LLM hidden -> acoustic transformer -> 37 codes/frame
     *   codes -> audio_embed -> next LLM input
     * Until END_AUDIO is generated.
     *
     * Finally, codec decoder converts all codes to 24kHz waveform.
     */

    *out_samples = NULL;
    *out_n_samples = 0;

    /* Reset KV cache */
    ctx->kv_cache_len = 0;

    /* Step 1: Tokenize text */
    int text_tokens[4096];
    int n_text_tokens = tts_tokenizer_encode(text, text_tokens, 4096);
    if (n_text_tokens <= 0) {
        fprintf(stderr, "generate: tokenization failed\n");
        return -1;
    }

    if (tts_verbose)
        fprintf(stderr, "Text tokenized: %d tokens\n", n_text_tokens);

    /* Step 2: Load voice embedding */
    tts_voice_t *voice = NULL;
    float *voice_embeds = NULL;
    int voice_frames = 0;

    if (voice_name) {
        char voice_path[1024];
        snprintf(voice_path, sizeof(voice_path), "%s/voice_embedding/%s.pt",
                 ctx->model_dir, voice_name);
        voice = tts_voice_load(voice_path);
        if (voice) {
            tts_voice_embed(ctx, voice, &voice_embeds, &voice_frames);
            if (tts_verbose)
                fprintf(stderr, "Voice: %s (%d frames)\n", voice_name, voice_frames);
        }
    }

    /* Step 3: Build prompt embeddings
     * Format: [BOS] [BEGIN_AUDIO] [voice_embeds x N] [/INST] text_tokens [INST] [BEGIN_AUDIO]
     * Special tokens: BOS=1, BEGIN_AUDIO=25, /INST=36, INST=35 */
    int n_voice = voice_frames > 0 ? voice_frames : 0;
    int prompt_len = 1 + 1 + n_voice + 1 + n_text_tokens + 1 + 1;
    int dim = TTS_DEC_DIM;

    float *prompt_embeds = (float *)calloc((size_t)prompt_len * dim, sizeof(float));
    if (!prompt_embeds) goto cleanup;

    int pos = 0;

    /* [BOS] */
    tts_embed_token_bf16(prompt_embeds + pos * dim,
                         ctx->decoder.tok_embeddings_bf16, TTS_TOK_BOS, dim);
    pos++;

    /* [BEGIN_AUDIO] */
    tts_embed_token_bf16(prompt_embeds + pos * dim,
                         ctx->decoder.tok_embeddings_bf16, TTS_TOK_BEGIN_AUDIO, dim);
    pos++;

    /* Voice embeddings (replace AUDIO token positions) */
    if (voice_embeds && n_voice > 0) {
        memcpy(prompt_embeds + pos * dim, voice_embeds,
               (size_t)n_voice * dim * sizeof(float));
        pos += n_voice;
    }

    /* [/INST] = token 36 */
    tts_embed_token_bf16(prompt_embeds + pos * dim,
                         ctx->decoder.tok_embeddings_bf16, 36, dim);
    pos++;

    /* Text tokens */
    for (int i = 0; i < n_text_tokens; i++) {
        tts_embed_token_bf16(prompt_embeds + pos * dim,
                             ctx->decoder.tok_embeddings_bf16, text_tokens[i], dim);
        pos++;
    }

    /* [INST] = token 35 */
    tts_embed_token_bf16(prompt_embeds + pos * dim,
                         ctx->decoder.tok_embeddings_bf16, 35, dim);
    pos++;

    /* [BEGIN_AUDIO] */
    tts_embed_token_bf16(prompt_embeds + pos * dim,
                         ctx->decoder.tok_embeddings_bf16, TTS_TOK_BEGIN_AUDIO, dim);
    pos++;

    if (tts_verbose)
        fprintf(stderr, "Prompt: %d tokens (%d voice + %d text), prefilling...\n",
                prompt_len, n_voice, n_text_tokens);

    /* Step 4: Prefill */
    tts_llm_prefill(ctx, prompt_embeds, prompt_len);
    free(prompt_embeds);
    prompt_embeds = NULL;

    if (tts_verbose)
        fprintf(stderr, "Prefill done, KV cache: %d positions\n", ctx->kv_cache_len);

    /* Step 5: Autoregressive audio generation */
    int max_frames = 2000; /* ~160 seconds at 12.5 Hz */
    int *all_codes = (int *)malloc((size_t)max_frames * TTS_CODES_PER_FRAME * sizeof(int));
    if (!all_codes) goto cleanup;

    float hidden[TTS_DEC_DIM];
    float next_embed[TTS_DEC_DIM];

    /* First decode step: feed AUDIO token to get first hidden state */
    tts_embed_token_bf16(next_embed, ctx->decoder.tok_embeddings_bf16,
                         TTS_TOK_AUDIO, dim);

    int n_frames = 0;
    fprintf(stderr, "Generating audio");

    for (int frame = 0; frame < max_frames; frame++) {
        tts_llm_forward(ctx, next_embed, hidden);

        int codes[TTS_CODES_PER_FRAME];
        tts_acoustic_forward(ctx, hidden, codes);

        if (codes[0] == TTS_AUDIO_SPECIAL_END) {
            if (tts_verbose)
                fprintf(stderr, "\nEND_AUDIO at frame %d\n", frame);
            break;
        }

        memcpy(all_codes + n_frames * TTS_CODES_PER_FRAME, codes,
               TTS_CODES_PER_FRAME * sizeof(int));
        n_frames++;

        /* Embed audio codes for next LLM input */
        tts_audio_embed_forward(ctx, codes, next_embed);

        if (frame % 25 == 0) fprintf(stderr, ".");
    }
    fprintf(stderr, "\n");

    if (n_frames == 0) {
        fprintf(stderr, "generate: no audio frames produced\n");
        free(all_codes);
        goto cleanup;
    }

    fprintf(stderr, "Generated %d audio frames (%.2f seconds)\n",
            n_frames, (float)n_frames / TTS_FRAME_RATE);

    /* Step 6: Codec decode */
    tts_codec_decode(ctx, all_codes, n_frames, out_samples, out_n_samples);
    free(all_codes);

    if (voice) tts_voice_free(voice);
    free(voice_embeds);
    return 0;

cleanup:
    free(prompt_embeds);
    free(voice_embeds);
    if (voice) tts_voice_free(voice);
    return -1;
}
