/*
 * voxtral_tts.h - Voxtral TTS Pure C Inference Engine
 *
 * Main API header for the Voxtral text-to-speech model.
 * Architecture: LLM backbone -> Flow-matching acoustic transformer -> Audio codec decoder
 */

#ifndef VOXTRAL_TTS_H
#define VOXTRAL_TTS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* ========================================================================
 * LLM Decoder Constants (Ministral 3B / 26-layer Mistral)
 * ======================================================================== */

#define TTS_DEC_DIM          3072
#define TTS_DEC_LAYERS       26
#define TTS_DEC_HEADS        32
#define TTS_DEC_KV_HEADS     8
#define TTS_DEC_HEAD_DIM     128
#define TTS_DEC_HIDDEN       9216
#define TTS_DEC_NORM_EPS     1e-5f
#define TTS_VOCAB_SIZE       131072
#define TTS_ROPE_THETA       1000000.0f

/* ========================================================================
 * Acoustic Transformer Constants (Flow Matching)
 * ======================================================================== */

#define TTS_AC_DIM           3072
#define TTS_AC_LAYERS        3
#define TTS_AC_HEADS         32
#define TTS_AC_KV_HEADS      8
#define TTS_AC_HEAD_DIM      128
#define TTS_AC_HIDDEN        9216
#define TTS_AC_NORM_EPS      1e-5f

/* Flow matching parameters */
#define TTS_FLOW_STEPS       8
#define TTS_CFG_ALPHA        1.2f
#define TTS_NOISE_SCALE      1.0f

/* ========================================================================
 * Audio Codec Constants
 * ======================================================================== */

#define TTS_CODEC_DIM        1024
#define TTS_CODEC_HIDDEN     4096
#define TTS_CODEC_HEADS      8
#define TTS_CODEC_KV_HEADS   8
#define TTS_CODEC_HEAD_DIM   128
#define TTS_CODEC_NORM_EPS   1e-2f

#define TTS_PATCH_SIZE       240     /* pretransform_patch_size */
#define TTS_SAMPLE_RATE      24000
#define TTS_FRAME_RATE       12.5f

/* ========================================================================
 * Codebook Constants
 * ======================================================================== */

#define TTS_SEMANTIC_CB_SIZE 8192
#define TTS_SEMANTIC_CB_DIM  256
#define TTS_ACOUSTIC_DIM     36      /* n_acoustic_codebook */
#define TTS_FSQ_LEVELS       21      /* acoustic_codebook_size */
#define TTS_EMBED_DIM        292     /* 256 + 36 */
#define TTS_CODES_PER_FRAME  37      /* 1 semantic + 36 acoustic */
#define TTS_NUM_CODEBOOKS    37

/* Padded codebook sizes for embedding table (pad to 128) */
#define TTS_SEMANTIC_CB_PADDED 8320  /* ceil((8192+2)/128)*128 = 8320 */

/* Special audio tokens (offset audio codes by 2 to avoid conflicts) */
#define TTS_AUDIO_SPECIAL_EMPTY 0    /* [EMPTY_AUDIO] */
#define TTS_AUDIO_SPECIAL_END   1    /* [END_AUDIO] */
#define TTS_AUDIO_SPECIAL_COUNT 2

/* ========================================================================
 * Codec Decoder Architecture
 * Decoder blocks: 4 stages, each with [transformer layers + conv upsample]
 * ======================================================================== */

#define TTS_CODEC_STAGES        4
#define TTS_CODEC_LAYERS_PER_STAGE 2
#define TTS_CODEC_TOTAL_LAYERS  8    /* 4 stages x 2 layers */

/* Decoder conv/upsample config: kernels and strides per stage */
/* Stage 0: k=3,s=1  Stage 1: k=4,s=2  Stage 2: k=4,s=2  Stage 3: k=4,s=2 */
/* Total upsample = 1 * 2 * 2 * 2 = 8x */

/* Sliding window sizes per stage (halved upon downsampling in encoder,
 * doubled in decoder): 2, 4, 8, 16 */

/* ========================================================================
 * LLM Decoder Layer
 * ======================================================================== */

typedef struct {
    /* Attention weights (bf16 mmap direct) */
    uint16_t *wq_bf16;        /* [n_heads*head_dim, dim] = [4096, 3072] */
    uint16_t *wk_bf16;        /* [n_kv_heads*head_dim, dim] = [1024, 3072] */
    uint16_t *wv_bf16;        /* [n_kv_heads*head_dim, dim] = [1024, 3072] */
    uint16_t *wo_bf16;        /* [dim, n_heads*head_dim] = [3072, 4096] */
    float *attention_norm;     /* [3072] */

    /* SwiGLU FFN weights (bf16 mmap direct) */
    uint16_t *w1_bf16;        /* [hidden, dim] = [9216, 3072] gate */
    uint16_t *w2_bf16;        /* [dim, hidden] = [3072, 9216] down */
    uint16_t *w3_bf16;        /* [hidden, dim] = [9216, 3072] up */
    float *ffn_norm;           /* [3072] */
} tts_dec_layer_t;

typedef struct {
    /* Token embeddings (bf16 mmap, tied with output projection) */
    uint16_t *tok_embeddings_bf16;  /* [131072, 3072] */

    /* Transformer layers */
    tts_dec_layer_t layers[TTS_DEC_LAYERS];

    /* Final norm */
    float *norm;               /* [3072] */
} tts_decoder_t;

/* ========================================================================
 * Acoustic Transformer (Flow Matching)
 * ======================================================================== */

typedef struct {
    /* Bidirectional attention (no RoPE, no causal mask) */
    uint16_t *wq_bf16;        /* [4096, 3072] */
    uint16_t *wk_bf16;        /* [1024, 3072] */
    uint16_t *wv_bf16;        /* [1024, 3072] */
    uint16_t *wo_bf16;        /* [3072, 4096] */
    float *attention_norm;     /* [3072] */

    /* SwiGLU FFN */
    uint16_t *w1_bf16;        /* [9216, 3072] */
    uint16_t *w2_bf16;        /* [3072, 9216] */
    uint16_t *w3_bf16;        /* [9216, 3072] */
    float *ffn_norm;           /* [3072] */
} tts_ac_layer_t;

typedef struct {
    /* Time embedding (sinusoidal) */
    float *time_inv_freq;              /* [dim/2 = 1536] */

    /* Input projections */
    uint16_t *input_proj_bf16;         /* [3072, 36] noise -> dim */
    uint16_t *time_proj_bf16;          /* [3072, 3072] time -> dim */
    uint16_t *llm_proj_bf16;           /* [3072, 3072] llm hidden -> dim */

    /* Output heads */
    uint16_t *semantic_out_bf16;       /* [8320, 3072] -> semantic logits */
    float *semantic_out_bias;          /* [8320] or NULL if no bias */
    uint16_t *acoustic_out_bf16;       /* [36, 3072] -> velocity */

    /* Final norm */
    float *norm;                        /* [3072] */

    /* Transformer layers */
    tts_ac_layer_t layers[TTS_AC_LAYERS];
} tts_acoustic_t;

/* ========================================================================
 * Audio Codec Decoder
 * ======================================================================== */

/* Weight-normed convolution (effective weight reconstructed at load time) */
typedef struct {
    float *weight;             /* [out_ch, in_ch, kernel] (effective) */
    float *bias;               /* [out_ch] or NULL */
    int in_ch;
    int out_ch;
    int kernel;
    int stride;
    int is_transpose;          /* 0 = conv1d, 1 = conv_transpose_1d */
} tts_conv_t;

/* Codec transformer layer (ALiBi attention + layer_scale) */
typedef struct {
    /* Attention */
    float *wq;                 /* [1024, 1024] */
    float *wk;                 /* [1024, 1024] */
    float *wv;                 /* [1024, 1024] */
    float *wo;                 /* [1024, 1024] */
    float *q_norm;             /* [1024] QK norm */
    float *k_norm;             /* [1024] QK norm */
    float *attention_norm;     /* [1024] RMSNorm */

    /* SwiGLU FFN */
    float *w1;                 /* [4096, 1024] */
    float *w2;                 /* [1024, 4096] */
    float *w3;                 /* [4096, 1024] */
    float *ffn_norm;           /* [1024] RMSNorm */

    /* Layer scale parameters */
    float *attn_scale;         /* [1024] */
    float *ffn_scale;          /* [1024] */

    /* ALiBi sliding window size */
    int window_size;
} tts_codec_layer_t;

typedef struct {
    /* Semantic codebook embedding (precomputed at load time) */
    float *semantic_embedding; /* [8192, 256] = embedding_sum / cluster_usage */

    /* Entry convolution: latent_dim -> codec_dim */
    tts_conv_t input_conv;     /* CausalConv1d(292->1024, k=3, s=1) */

    /* 4 decoder stages: transformer blocks + upsample convolutions */
    tts_codec_layer_t transformers[TTS_CODEC_TOTAL_LAYERS]; /* 8 layers total */

    /* Upsample convolutions between stages (3 of them, first stage has s=1) */
    tts_conv_t stage_convs[TTS_CODEC_STAGES]; /* includes initial + 3 upsamples */

    /* Output projection */
    tts_conv_t output_proj;    /* CausalConv1d(1024->240, k=7) */

    /* Precomputed ALiBi slopes for 8 heads */
    float alibi_slopes[TTS_CODEC_HEADS];
} tts_codec_t;

/* ========================================================================
 * Audio Codebook Embeddings (for LLM input)
 * ======================================================================== */

typedef struct {
    uint16_t *embeddings_bf16; /* Combined embedding table (padded) */
    int offsets[TTS_NUM_CODEBOOKS + 1]; /* Cumulative codebook offsets */
    int total_size;            /* Total embedding table rows */
    int embed_dim;             /* Embedding dimension (3072) */
} tts_audio_embed_t;

/* ========================================================================
 * Voice Embedding
 * ======================================================================== */

typedef struct {
    int *codes;                /* Raw BF16 embedding data (cast from uint16_t*) */
    int n_frames;              /* Number of embedding frames [n_frames, 3072] */
} tts_voice_t;

/* ========================================================================
 * Main Context
 * ======================================================================== */

typedef struct {
    tts_decoder_t decoder;
    tts_acoustic_t acoustic;
    tts_codec_t codec;
    tts_audio_embed_t audio_embed;

    /* Model file (kept open for mmap) */
    void *safetensors;         /* safetensors_file_t* */
    char model_dir[512];

    /* KV cache for LLM decoder */
    float *kv_cache_k;         /* [layers, max_seq, kv_heads * head_dim] */
    float *kv_cache_v;         /* [layers, max_seq, kv_heads * head_dim] */
    int kv_cache_len;          /* Current physical cache length */
    int kv_cache_max;          /* Maximum cache size */

    /* Precomputed RoPE frequencies for decoder */
    float *dec_rope_freqs;     /* [max_seq, head_dim/2, 2] */

    /* Persistent LLM decoder working buffers */
    float *dec_x;              /* [1, 3072] current hidden state */
    float *dec_x_norm;         /* [1, 3072] normalized hidden state */
    float *dec_q;              /* [1, 4096] queries */
    float *dec_k;              /* [1, 1024] keys */
    float *dec_v;              /* [1, 1024] values */
    float *dec_attn_out;       /* [1, 4096] attention output */
    float *dec_proj_out;       /* [1, 3072] projection output */
    float *dec_gate;           /* [1, 9216] gate projection */
    float *dec_up;             /* [1, 9216] up projection */
    float *dec_ffn_out;        /* [1, 3072] FFN output */

    /* Persistent acoustic transformer working buffers */
    float *ac_tokens;          /* [3, 3072] concatenated tokens */
    float *ac_tokens_norm;     /* [3, 3072] normalized */
    float *ac_q;               /* [3, 4096] */
    float *ac_k;               /* [3, 1024] */
    float *ac_v;               /* [3, 1024] */
    float *ac_attn_out;        /* [3, 4096] */
    float *ac_proj_out;        /* [3, 3072] */
    float *ac_gate;            /* [3, 9216] */
    float *ac_up;              /* [3, 9216] */
    float *ac_ffn_out;         /* [3, 3072] */
    float *ac_time_emb;        /* [3072] time embedding */
    float *ac_velocity;        /* [36] predicted velocity */
    float *ac_noise;           /* [36] current noise/sample */

    /* Random number generator state */
    uint64_t rng_state;

    /* Voice embedding (loaded separately) */
    tts_voice_t *voice;

    /* Verbose output flag */
    int verbose;
} tts_ctx_t;

/* ========================================================================
 * Special Token IDs (from Tekken tokenizer)
 * ======================================================================== */

#define TTS_TOK_BOS          1
#define TTS_TOK_EOS          2
#define TTS_TOK_INST_START   3     /* [INST] */
#define TTS_TOK_INST_END     4     /* [/INST] */
#define TTS_TOK_AUDIO        24    /* audio token placeholder */
#define TTS_TOK_BEGIN_AUDIO  25    /* [begin_audio] */

/* ========================================================================
 * API Functions
 * ======================================================================== */

/* Load model from directory containing consolidated.safetensors + tekken.json */
tts_ctx_t *tts_load(const char *model_dir);

/* Free all resources */
void tts_free(tts_ctx_t *ctx);

/* Load a voice embedding from .pt file or raw binary */
tts_voice_t *tts_voice_load(const char *path);
void tts_voice_free(tts_voice_t *voice);

/* Set random seed for reproducibility */
void tts_set_seed(tts_ctx_t *ctx, uint64_t seed);

/* ========================================================================
 * Text-to-Speech Generation
 * ======================================================================== */

/* Generate speech from text, returns raw audio samples.
 * voice_name: name of voice (e.g. "neutral_female") or NULL for default
 * out_samples: receives pointer to allocated float array (caller must free)
 * out_n_samples: receives number of samples
 * Returns 0 on success, -1 on error. */
int tts_generate(tts_ctx_t *ctx, const char *text, const char *voice_name,
                 float **out_samples, int *out_n_samples);

/* ========================================================================
 * WAV File Output
 * ======================================================================== */

/* Write audio samples as WAV file (24kHz, 16-bit, mono) */
int tts_write_wav(const char *path, const float *samples, int n_samples,
                  int sample_rate);

/* ========================================================================
 * Internal Functions
 * ======================================================================== */

/* LLM decoder */
int tts_llm_load(tts_decoder_t *dec, void *sf);
int tts_llm_kv_cache_alloc(tts_ctx_t *ctx, int max_seq);
void tts_llm_prefill(tts_ctx_t *ctx, const float *embeds, int seq_len);
void tts_llm_forward(tts_ctx_t *ctx, const float *input_embed, float *out_hidden);

/* Acoustic transformer */
int tts_acoustic_load(tts_acoustic_t *ac, void *sf);
void tts_acoustic_forward(tts_ctx_t *ctx, const float *llm_hidden,
                           int *out_codes);

/* Audio codec decoder */
int tts_codec_load(tts_codec_t *codec, void *sf);
void tts_codec_decode(tts_ctx_t *ctx, const int *codes, int n_frames,
                       float **out_samples, int *out_n_samples);

/* Audio codebook embeddings */
int tts_audio_embed_load(tts_audio_embed_t *ae, void *sf);
void tts_audio_embed_forward(tts_ctx_t *ctx, const int *codes, float *out_embed);

/* Tokenizer */
int tts_tokenizer_load(const char *path);
void tts_tokenizer_free(void);
int tts_tokenizer_encode(const char *text, int *out_tokens, int max_tokens);
const char *tts_tokenizer_decode(int token_id);

/* Voice embedding loading */
int tts_voice_embed(tts_ctx_t *ctx, const tts_voice_t *voice,
                     float **out_embeds, int *out_seq_len);

/* Global verbose flag */
extern int tts_verbose;

#endif /* VOXTRAL_TTS_H */
