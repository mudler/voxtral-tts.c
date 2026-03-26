/*
 * voxtral_tts_wav.c - WAV file writer
 */

#include "voxtral_tts.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

int tts_write_wav(const char *path, const float *samples, int n_samples,
                  int sample_rate) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "tts_write_wav: cannot open %s\n", path);
        return -1;
    }

    int channels = 1;
    int bits_per_sample = 16;
    int byte_rate = sample_rate * channels * bits_per_sample / 8;
    int block_align = channels * bits_per_sample / 8;
    int data_size = n_samples * block_align;
    int chunk_size = 36 + data_size;

    /* RIFF header */
    fwrite("RIFF", 1, 4, f);
    uint32_t val32 = (uint32_t)chunk_size;
    fwrite(&val32, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    /* fmt chunk */
    fwrite("fmt ", 1, 4, f);
    val32 = 16; /* PCM */
    fwrite(&val32, 4, 1, f);
    uint16_t val16 = 1; /* PCM format */
    fwrite(&val16, 2, 1, f);
    val16 = (uint16_t)channels;
    fwrite(&val16, 2, 1, f);
    val32 = (uint32_t)sample_rate;
    fwrite(&val32, 4, 1, f);
    val32 = (uint32_t)byte_rate;
    fwrite(&val32, 4, 1, f);
    val16 = (uint16_t)block_align;
    fwrite(&val16, 2, 1, f);
    val16 = (uint16_t)bits_per_sample;
    fwrite(&val16, 2, 1, f);

    /* data chunk */
    fwrite("data", 1, 4, f);
    val32 = (uint32_t)data_size;
    fwrite(&val32, 4, 1, f);

    /* Convert float32 [-1, 1] to int16 */
    for (int i = 0; i < n_samples; i++) {
        float s = samples[i];
        /* Clamp to [-1, 1] */
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        int16_t s16 = (int16_t)(s * 32767.0f);
        fwrite(&s16, 2, 1, f);
    }

    fclose(f);

    if (tts_verbose) {
        float duration = (float)n_samples / (float)sample_rate;
        fprintf(stderr, "Wrote %s: %.2fs, %d samples, %d Hz\n",
                path, duration, n_samples, sample_rate);
    }

    return 0;
}
