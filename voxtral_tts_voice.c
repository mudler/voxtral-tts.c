/*
 * voxtral_tts_voice.c - Voice embedding loader and audio codebook embeddings
 *
 * Voice embeddings are stored as .pt files (PyTorch format).
 * They contain pre-computed LLM input embeddings of shape [N, 3072] in BF16.
 * These replace the audio token (id=24) positions in the prompt.
 *
 * The .pt file is a ZIP archive containing:
 *   - archive/data.pkl (pickle metadata)
 *   - archive/data/0 (raw tensor data)
 */

#include "voxtral_tts.h"
#include "voxtral_tts_kernels.h"
#include "voxtral_tts_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Minimal ZIP reader for .pt files
 *
 * PyTorch .pt files are ZIP archives. We look for a file named
 * "archive/data/0" which contains the raw tensor data.
 * The tensor shape is inferred from file size and known dim (3072).
 * ======================================================================== */

/* ZIP local file header signature */
#define ZIP_LOCAL_MAGIC 0x04034b50

typedef struct {
    uint32_t signature;
    uint16_t version;
    uint16_t flags;
    uint16_t compression;
    uint16_t mod_time;
    uint16_t mod_date;
    uint32_t crc32;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
    uint16_t filename_len;
    uint16_t extra_len;
} __attribute__((packed)) zip_local_header_t;

/*
 * Find an entry in a ZIP file by name. Handles data descriptor (flag bit 3)
 * where sizes in the local header are 0 and actual sizes follow the data.
 * Fallback: scan the central directory at the end of the file.
 */
static int find_zip_entry(const unsigned char *data, size_t file_size,
                          const char *target_name,
                          const unsigned char **out_data, size_t *out_size) {
    /* Strategy: scan from the end for the End of Central Directory record,
     * then parse the central directory to find our target entry with
     * correct sizes and offsets. */
    size_t target_len = strlen(target_name);

    /* Find End of Central Directory (signature 0x06054b50) */
    size_t eocd_pos = 0;
    int found_eocd = 0;
    for (size_t i = file_size < 22 ? 0 : file_size - 22; i > 0 && i > file_size - 65536; i--) {
        uint32_t sig;
        memcpy(&sig, data + i, 4);
        if (sig == 0x06054b50) {
            eocd_pos = i;
            found_eocd = 1;
            break;
        }
    }

    if (!found_eocd) {
        fprintf(stderr, "voice: ZIP end-of-central-dir not found\n");
        return -1;
    }

    /* Read central directory offset and size from EOCD */
    uint32_t cd_size, cd_offset;
    memcpy(&cd_size, data + eocd_pos + 12, 4);
    memcpy(&cd_offset, data + eocd_pos + 16, 4);

    /* Scan central directory entries (signature 0x02014b50) */
    size_t pos = cd_offset;
    while (pos + 46 < file_size) {
        uint32_t sig;
        memcpy(&sig, data + pos, 4);
        if (sig != 0x02014b50) break;

        uint16_t fname_len, extra_len, comment_len;
        uint32_t comp_size, uncomp_size, local_offset;
        uint16_t compression;

        memcpy(&compression, data + pos + 10, 2);
        memcpy(&comp_size, data + pos + 20, 4);
        memcpy(&uncomp_size, data + pos + 24, 4);
        memcpy(&fname_len, data + pos + 28, 2);
        memcpy(&extra_len, data + pos + 30, 2);
        memcpy(&comment_len, data + pos + 32, 2);
        memcpy(&local_offset, data + pos + 42, 4);

        const char *fname = (const char *)(data + pos + 46);

        if (fname_len == (uint16_t)target_len &&
            memcmp(fname, target_name, fname_len) == 0) {
            if (compression != 0) {
                fprintf(stderr, "voice: compressed entry not supported\n");
                return -1;
            }
            /* Parse local header to find data start */
            uint16_t local_fname_len, local_extra_len;
            memcpy(&local_fname_len, data + local_offset + 26, 2);
            memcpy(&local_extra_len, data + local_offset + 28, 2);
            size_t data_start = local_offset + 30 + local_fname_len + local_extra_len;

            size_t entry_size = uncomp_size > 0 ? uncomp_size : comp_size;
            if (data_start + entry_size > file_size) return -1;

            *out_data = data + data_start;
            *out_size = entry_size;
            return 0;
        }

        pos += 46 + fname_len + extra_len + comment_len;
    }

    return -1; /* not found */
}

/* ========================================================================
 * Voice Loading
 * ======================================================================== */

tts_voice_t *tts_voice_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "voice: cannot open %s\n", path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_size <= 0) { fclose(f); return NULL; }

    unsigned char *file_data = (unsigned char *)malloc(file_size);
    if (!file_data || fread(file_data, 1, file_size, f) != (size_t)file_size) {
        fclose(f);
        free(file_data);
        return NULL;
    }
    fclose(f);

    /* Find the raw tensor data in the ZIP archive */
    const unsigned char *tensor_data = NULL;
    size_t tensor_size = 0;

    /* Try common PyTorch archive paths */
    const char *paths[] = {
        "voice_embed/data/0",
        "archive/data/0",
        "data/0",
        NULL
    };

    int found = 0;
    for (int i = 0; paths[i]; i++) {
        if (find_zip_entry(file_data, file_size, paths[i],
                           &tensor_data, &tensor_size) == 0) {
            found = 1;
            break;
        }
    }

    if (!found) {
        fprintf(stderr, "voice: could not find tensor data in %s\n", path);
        free(file_data);
        return NULL;
    }

    /* Voice embeddings are BF16, shape [N, 3072]
     * N = tensor_size / (3072 * 2 bytes per bf16) */
    int dim = TTS_DEC_DIM;
    int n_frames = (int)(tensor_size / (dim * sizeof(uint16_t)));

    if (n_frames <= 0 || (size_t)n_frames * dim * 2 != tensor_size) {
        fprintf(stderr, "voice: unexpected tensor size %zu (expected multiple of %d)\n",
                tensor_size, dim * 2);
        free(file_data);
        return NULL;
    }

    tts_voice_t *voice = (tts_voice_t *)calloc(1, sizeof(tts_voice_t));
    if (!voice) { free(file_data); return NULL; }

    /* Store raw BF16 data - we'll convert to F32 during embedding */
    voice->n_frames = n_frames;
    /* Repurpose codes field to store the raw BF16 embedding data pointer
     * We allocate a separate buffer since file_data will be freed */
    size_t emb_bytes = (size_t)n_frames * dim * sizeof(uint16_t);
    voice->codes = (int *)malloc(emb_bytes);
    if (!voice->codes) {
        free(voice);
        free(file_data);
        return NULL;
    }
    memcpy(voice->codes, tensor_data, emb_bytes);

    free(file_data);

    if (tts_verbose)
        fprintf(stderr, "  Voice loaded: %d frames from %s\n", n_frames, path);

    return voice;
}

void tts_voice_free(tts_voice_t *voice) {
    if (!voice) return;
    free(voice->codes);
    free(voice);
}

/* ========================================================================
 * Voice Embedding into LLM Input
 *
 * The voice .pt files contain pre-computed BF16 embeddings [N, 3072]
 * that directly replace audio token positions in the prompt.
 * ======================================================================== */

int tts_voice_embed(tts_ctx_t *ctx, const tts_voice_t *voice,
                     float **out_embeds, int *out_seq_len) {
    if (!voice || voice->n_frames <= 0) {
        *out_embeds = NULL;
        *out_seq_len = 0;
        return -1;
    }

    int n = voice->n_frames;
    int dim = TTS_DEC_DIM;
    float *embeds = (float *)malloc((size_t)n * dim * sizeof(float));
    if (!embeds) return -1;

    /* Convert BF16 voice embeddings to F32 */
    const uint16_t *bf16_data = (const uint16_t *)voice->codes;
    tts_bf16_to_f32_buf(embeds, bf16_data, (size_t)n * dim);

    *out_embeds = embeds;
    *out_seq_len = n;
    return 0;
}

/* ========================================================================
 * Audio Codebook Embeddings (for feeding generated audio codes back to LLM)
 * ======================================================================== */

int tts_audio_embed_load(tts_audio_embed_t *ae, void *sf_ptr) {
    safetensors_file_t *sf = (safetensors_file_t *)sf_ptr;

    const safetensor_t *t = safetensors_find(sf,
        "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight");
    if (!t) {
        fprintf(stderr, "audio_embed: weight not found\n");
        return -1;
    }

    ae->embeddings_bf16 = safetensors_get_bf16_direct(sf, t);
    if (!ae->embeddings_bf16) return -1;

    ae->embed_dim = TTS_DEC_DIM;

    /* Compute codebook offsets (cumulative sum of UNpadded sizes).
     * Python MultiVocabEmbeddings uses pad_to_multiple=None:
     *   codebook_sizes = [8194, 23, 23, ..., 23]  (37 entries)
     *   offsets = cumsum([0] + codebook_sizes[:-1])
     * Codebook 0 (semantic): 8192 + 2 special = 8194
     * Codebooks 1-36 (acoustic): 21 + 2 special = 23 each */
    int semantic_size = TTS_SEMANTIC_CB_SIZE + TTS_AUDIO_SPECIAL_COUNT;  /* 8194 */
    int acoustic_size = TTS_FSQ_LEVELS + TTS_AUDIO_SPECIAL_COUNT;       /* 23 */

    ae->offsets[0] = 0;
    ae->offsets[1] = semantic_size;
    for (int i = 2; i <= TTS_NUM_CODEBOOKS; i++) {
        ae->offsets[i] = ae->offsets[i - 1] + acoustic_size;
    }
    ae->total_size = ae->offsets[TTS_NUM_CODEBOOKS];

    if (tts_verbose)
        fprintf(stderr, "  Audio embeddings: %d total entries, dim=%d\n",
                ae->total_size, ae->embed_dim);

    return 0;
}

void tts_audio_embed_forward(tts_ctx_t *ctx, const int *codes, float *out_embed) {
    /*
     * Embed 37 audio codes into a single [3072] vector for LLM input.
     * Each codebook has its own offset in the combined embedding table.
     * Sum across all 37 codebooks.
     *
     * codes: [TTS_CODES_PER_FRAME] = [37]
     * out_embed: [3072]
     */
    tts_audio_embed_t *ae = &ctx->audio_embed;
    int dim = ae->embed_dim;

    memset(out_embed, 0, dim * sizeof(float));

    float embed_buf[TTS_DEC_DIM];

    for (int cb = 0; cb < TTS_NUM_CODEBOOKS; cb++) {
        int code = codes[cb];
        int global_idx = ae->offsets[cb] + code;

        tts_embed_token_bf16(embed_buf, ae->embeddings_bf16, global_idx, dim);
        tts_add_inplace(out_embed, embed_buf, dim);
    }
}
