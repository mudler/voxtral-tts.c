/*
 * main.c - CLI entry point for Voxtral TTS
 *
 * Usage: ./voxtral_tts -d <model_dir> [-v voice] [-o output.wav] [-s seed] "text"
 */

#include "voxtral_tts.h"
#include "voxtral_tts_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr, "Voxtral TTS - Pure C Text-to-Speech Inference\n\n");
    fprintf(stderr, "Usage: %s [options] \"text to speak\"\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -d <dir>        Model directory (required)\n");
    fprintf(stderr, "  -v <voice>      Voice name (default: neutral_female)\n");
    fprintf(stderr, "  -o <file>       Output WAV file (default: output.wav)\n");
    fprintf(stderr, "  -s <seed>       Random seed for reproducibility\n");
    fprintf(stderr, "  --verbose       Enable verbose output\n");
    fprintf(stderr, "  --list-voices   List available voices\n");
    fprintf(stderr, "  --inspect       Print model tensor info and exit\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Available voices:\n");
    fprintf(stderr, "  English: casual_female, casual_male, cheerful_female, neutral_female, neutral_male\n");
    fprintf(stderr, "  French: fr_female, fr_male    German: de_female, de_male\n");
    fprintf(stderr, "  Spanish: es_female, es_male    Italian: it_female, it_male\n");
    fprintf(stderr, "  Portuguese: pt_female, pt_male  Dutch: nl_female, nl_male\n");
    fprintf(stderr, "  Arabic: ar_male    Hindi: hi_female, hi_male\n");
}

int main(int argc, char *argv[]) {
    const char *model_dir = NULL;
    const char *voice = "neutral_female";
    const char *output = "output.wav";
    const char *text = NULL;
    uint64_t seed = 0;
    int verbose = 0;
    int inspect = 0;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
            voice = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            seed = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-V") == 0) {
            verbose++;
        } else if (strcmp(argv[i], "--inspect") == 0) {
            inspect = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (argv[i][0] != '-') {
            text = argv[i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_dir) {
        fprintf(stderr, "Error: model directory required (-d)\n\n");
        usage(argv[0]);
        return 1;
    }

    /* Set verbosity */
    tts_verbose = verbose;

    /* Inspect mode */
    if (inspect) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/consolidated.safetensors", model_dir);
        safetensors_file_t *sf = safetensors_open(path);
        if (!sf) {
            fprintf(stderr, "Failed to open %s\n", path);
            return 1;
        }
        safetensors_print_all(sf);
        safetensors_close(sf);
        return 0;
    }

    if (!text) {
        fprintf(stderr, "Error: no text to speak\n\n");
        usage(argv[0]);
        return 1;
    }

    /* Load model */
    tts_ctx_t *ctx = tts_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    /* Set seed if specified */
    if (seed > 0) tts_set_seed(ctx, seed);

    /* Generate speech */
    float *samples = NULL;
    int n_samples = 0;

    fprintf(stderr, "Generating speech for: \"%s\"\n", text);
    fprintf(stderr, "Voice: %s\n", voice);

    int ret = tts_generate(ctx, text, voice, &samples, &n_samples);
    if (ret != 0 || !samples || n_samples == 0) {
        fprintf(stderr, "Speech generation failed\n");
        tts_free(ctx);
        return 1;
    }

    /* Write WAV */
    if (tts_write_wav(output, samples, n_samples, TTS_SAMPLE_RATE) != 0) {
        fprintf(stderr, "Failed to write %s\n", output);
        free(samples);
        tts_free(ctx);
        return 1;
    }

    float duration = (float)n_samples / (float)TTS_SAMPLE_RATE;
    fprintf(stderr, "Output: %s (%.2fs, %d samples)\n", output, duration, n_samples);

    free(samples);
    tts_free(ctx);
    return 0;
}
