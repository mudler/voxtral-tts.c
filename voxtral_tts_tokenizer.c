/*
 * voxtral_tts_tokenizer.c - Tekken BPE tokenizer with encoding support
 *
 * Tekken tokenizer format (tekken.json):
 *   - vocab: array of {rank, token_bytes (base64), token_str}
 *   - special_tokens: array of {rank, token_str, is_control}
 *   - config: {default_vocab_size: 131072, default_num_special_tokens: 1000}
 *
 * Token ID mapping:
 *   - IDs 0..999: special tokens
 *   - IDs 1000..131071: regular vocabulary tokens (token_id = 1000 + rank)
 *
 * BPE encoding: greedy longest-match from vocabulary.
 * The rank order defines merge priority for BPE.
 *
 * Adapted from antirez/voxtral.c with encoding added.
 */

#include "voxtral_tts.h"
#include "voxtral_tts_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_VOCAB     130072
#define MAX_SPECIAL   1000
#define MAX_TOKEN_LEN 256
#define TEKKEN_NUM_SPECIAL 1000

/* ========================================================================
 * Tokenizer State (module-global singleton)
 * ======================================================================== */

typedef struct {
    /* Decode tables */
    char **vocab_str;        /* [MAX_VOCAB] decoded byte strings */
    int *vocab_len;          /* [MAX_VOCAB] lengths of vocab_str entries */
    char **special_str;      /* [MAX_SPECIAL] special token strings */
    int n_vocab;
    int n_special;

    /* Encode: trie for fast prefix matching */
    /* Simple approach: sorted vocab by length (longest first) for greedy match */
    int *sorted_by_len;      /* indices into vocab sorted by descending length */
    int n_sorted;

    int loaded;
} tokenizer_state_t;

static tokenizer_state_t g_tok = {0};

/* ========================================================================
 * Base64 Decoder
 * ======================================================================== */

static const int b64_table[256] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
    52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
    15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
    41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
};

static int b64_decode(const char *in, char *out, int max_out) {
    int len = 0;
    int val = 0, bits = 0;
    for (; *in; in++) {
        int c = b64_table[(unsigned char)*in];
        if (c == -1) {
            if (*in == '=') break;
            continue;
        }
        val = (val << 6) | c;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            if (len < max_out - 1) {
                out[len++] = (char)((val >> bits) & 0xFF);
            }
        }
    }
    out[len] = '\0';
    return len;
}

/* ========================================================================
 * JSON Parser (minimal, for tekken.json)
 * ======================================================================== */

static void skip_ws(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++;
}

static int parse_str(const char **p, char *out, int max_len) {
    skip_ws(p);
    if (**p != '"') return -1;
    (*p)++;
    int i = 0;
    while (**p && **p != '"' && i < max_len - 1) {
        if (**p == '\\') {
            (*p)++;
            if (**p == 'n') out[i++] = '\n';
            else if (**p == 't') out[i++] = '\t';
            else if (**p == 'r') out[i++] = '\r';
            else if (**p == '"') out[i++] = '"';
            else if (**p == '\\') out[i++] = '\\';
            else if (**p == 'u') {
                (*p)++;
                unsigned int cp = 0;
                for (int j = 0; j < 4 && **p; j++, (*p)++) {
                    cp <<= 4;
                    if (**p >= '0' && **p <= '9') cp |= **p - '0';
                    else if (**p >= 'a' && **p <= 'f') cp |= **p - 'a' + 10;
                    else if (**p >= 'A' && **p <= 'F') cp |= **p - 'A' + 10;
                }
                if (cp < 0x80 && i < max_len - 1) {
                    out[i++] = cp;
                } else if (cp < 0x800 && i < max_len - 2) {
                    out[i++] = 0xC0 | (cp >> 6);
                    out[i++] = 0x80 | (cp & 0x3F);
                } else if (i < max_len - 3) {
                    out[i++] = 0xE0 | (cp >> 12);
                    out[i++] = 0x80 | ((cp >> 6) & 0x3F);
                    out[i++] = 0x80 | (cp & 0x3F);
                }
                continue;
            } else {
                out[i++] = **p;
            }
        } else {
            out[i++] = **p;
        }
        (*p)++;
    }
    out[i] = '\0';
    if (**p == '"') (*p)++;
    return 0;
}

static long parse_long(const char **p) {
    skip_ws(p);
    long val = 0;
    int neg = 0;
    if (**p == '-') { neg = 1; (*p)++; }
    while (**p >= '0' && **p <= '9') {
        val = val * 10 + (**p - '0');
        (*p)++;
    }
    return neg ? -val : val;
}

static void skip_value(const char **p) {
    skip_ws(p);
    if (**p == '"') {
        (*p)++;
        while (**p && **p != '"') { if (**p == '\\') (*p)++; if (**p) (*p)++; }
        if (**p == '"') (*p)++;
    } else if (**p == '{') {
        int d = 1; (*p)++;
        while (**p && d > 0) {
            if (**p == '"') { (*p)++; while (**p && **p != '"') { if (**p == '\\') (*p)++; (*p)++; } if (**p) (*p)++; }
            else if (**p == '{') { d++; (*p)++; }
            else if (**p == '}') { d--; (*p)++; }
            else (*p)++;
        }
    } else if (**p == '[') {
        int d = 1; (*p)++;
        while (**p && d > 0) {
            if (**p == '"') { (*p)++; while (**p && **p != '"') { if (**p == '\\') (*p)++; (*p)++; } if (**p) (*p)++; }
            else if (**p == '[') { d++; (*p)++; }
            else if (**p == ']') { d--; (*p)++; }
            else (*p)++;
        }
    } else {
        while (**p && **p != ',' && **p != '}' && **p != ']') (*p)++;
    }
}

/* ========================================================================
 * Tokenizer Loading
 * ======================================================================== */

/* Compare function for sorting vocab by descending token length */
static int cmp_by_len_desc(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    int la = g_tok.vocab_len[ia];
    int lb = g_tok.vocab_len[ib];
    if (la != lb) return lb - la; /* descending by length */
    return ia - ib; /* ascending by rank for ties */
}

int tts_tokenizer_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "tokenizer: cannot open %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    if (size <= 0) { fclose(f); return -1; }
    fseek(f, 0, SEEK_SET);

    char *json = (char *)malloc(size + 1);
    if (!json || fread(json, 1, size, f) != (size_t)size) {
        fclose(f);
        free(json);
        return -1;
    }
    fclose(f);
    json[size] = '\0';

    g_tok.vocab_str = (char **)calloc(MAX_VOCAB, sizeof(char *));
    g_tok.vocab_len = (int *)calloc(MAX_VOCAB, sizeof(int));
    g_tok.special_str = (char **)calloc(MAX_SPECIAL, sizeof(char *));
    g_tok.n_vocab = 0;
    g_tok.n_special = 0;

    const char *p = json;
    skip_ws(&p);
    if (*p != '{') goto fail;
    p++;

    while (*p && *p != '}') {
        skip_ws(&p);
        if (*p == ',') { p++; continue; }

        char key[64];
        if (parse_str(&p, key, sizeof(key)) != 0) break;
        skip_ws(&p);
        if (*p != ':') break;
        p++;
        skip_ws(&p);

        if (strcmp(key, "vocab") == 0) {
            if (*p != '[') break;
            p++;
            while (*p && *p != ']') {
                skip_ws(&p);
                if (*p == ',') { p++; continue; }
                if (*p != '{') break;
                p++;

                int rank = -1;
                char token_bytes[512] = {0};

                while (*p && *p != '}') {
                    skip_ws(&p);
                    if (*p == ',') { p++; continue; }
                    char k[32];
                    if (parse_str(&p, k, sizeof(k)) != 0) break;
                    skip_ws(&p);
                    if (*p != ':') break;
                    p++;
                    skip_ws(&p);

                    if (strcmp(k, "rank") == 0) {
                        rank = (int)parse_long(&p);
                    } else if (strcmp(k, "token_bytes") == 0) {
                        parse_str(&p, token_bytes, sizeof(token_bytes));
                    } else {
                        skip_value(&p);
                    }
                }
                if (*p == '}') p++;

                if (rank >= 0 && rank < MAX_VOCAB && token_bytes[0]) {
                    char decoded[MAX_TOKEN_LEN];
                    int len = b64_decode(token_bytes, decoded, sizeof(decoded));
                    g_tok.vocab_str[rank] = (char *)malloc(len + 1);
                    memcpy(g_tok.vocab_str[rank], decoded, len);
                    g_tok.vocab_str[rank][len] = '\0';
                    g_tok.vocab_len[rank] = len;
                    if (rank >= g_tok.n_vocab) g_tok.n_vocab = rank + 1;
                }
            }
            if (*p == ']') p++;

        } else if (strcmp(key, "special_tokens") == 0) {
            if (*p != '[') break;
            p++;
            while (*p && *p != ']') {
                skip_ws(&p);
                if (*p == ',') { p++; continue; }
                if (*p != '{') break;
                p++;

                int rank = -1;
                char token_str[256] = {0};

                while (*p && *p != '}') {
                    skip_ws(&p);
                    if (*p == ',') { p++; continue; }
                    char k[32];
                    if (parse_str(&p, k, sizeof(k)) != 0) break;
                    skip_ws(&p);
                    if (*p != ':') break;
                    p++;
                    skip_ws(&p);

                    if (strcmp(k, "rank") == 0) {
                        rank = (int)parse_long(&p);
                    } else if (strcmp(k, "token_str") == 0) {
                        parse_str(&p, token_str, sizeof(token_str));
                    } else {
                        skip_value(&p);
                    }
                }
                if (*p == '}') p++;

                if (rank >= 0 && rank < MAX_SPECIAL && token_str[0]) {
                    g_tok.special_str[rank] = strdup(token_str);
                    if (rank >= g_tok.n_special) g_tok.n_special = rank + 1;
                }
            }
            if (*p == ']') p++;
        } else {
            skip_value(&p);
        }
    }

    free(json);

    /* Build sorted index for encoding (greedy longest match) */
    g_tok.sorted_by_len = (int *)malloc(g_tok.n_vocab * sizeof(int));
    g_tok.n_sorted = 0;
    for (int i = 0; i < g_tok.n_vocab; i++) {
        if (g_tok.vocab_str[i] && g_tok.vocab_len[i] > 0) {
            g_tok.sorted_by_len[g_tok.n_sorted++] = i;
        }
    }
    qsort(g_tok.sorted_by_len, g_tok.n_sorted, sizeof(int), cmp_by_len_desc);

    g_tok.loaded = 1;

    if (tts_verbose)
        fprintf(stderr, "  Tokenizer: %d vocab + %d special tokens\n",
                g_tok.n_vocab, g_tok.n_special);

    return 0;

fail:
    free(json);
    tts_tokenizer_free();
    return -1;
}

void tts_tokenizer_free(void) {
    if (g_tok.vocab_str) {
        for (int i = 0; i < MAX_VOCAB; i++) free(g_tok.vocab_str[i]);
        free(g_tok.vocab_str);
    }
    free(g_tok.vocab_len);
    if (g_tok.special_str) {
        for (int i = 0; i < MAX_SPECIAL; i++) free(g_tok.special_str[i]);
        free(g_tok.special_str);
    }
    free(g_tok.sorted_by_len);
    memset(&g_tok, 0, sizeof(g_tok));
}

/* ========================================================================
 * Decoding (token ID -> string)
 * ======================================================================== */

const char *tts_tokenizer_decode(int token_id) {
    if (token_id >= TEKKEN_NUM_SPECIAL && token_id < TEKKEN_NUM_SPECIAL + g_tok.n_vocab) {
        return g_tok.vocab_str[token_id - TEKKEN_NUM_SPECIAL];
    }
    if (token_id >= 0 && token_id < g_tok.n_special) {
        return g_tok.special_str[token_id];
    }
    return NULL;
}

/* ========================================================================
 * Encoding (text -> token IDs)
 *
 * Uses greedy longest-match: at each position, find the longest vocab entry
 * that matches the remaining text, emit its token ID, advance.
 *
 * This is a simplification of true BPE merge ordering but works well in
 * practice for Tekken's large vocabulary (130K tokens with long entries).
 *
 * Fallback: unknown bytes are encoded as individual byte tokens (rank 0-255
 * typically map to single bytes in Tekken).
 * ======================================================================== */

int tts_tokenizer_encode(const char *text, int *out_tokens, int max_tokens) {
    if (!g_tok.loaded || !text) return 0;

    int text_len = strlen(text);
    int n_tokens = 0;
    int pos = 0;

    while (pos < text_len && n_tokens < max_tokens) {
        int best_rank = -1;
        int best_len = 0;

        /* Try longest match first (sorted_by_len is descending by length) */
        for (int i = 0; i < g_tok.n_sorted; i++) {
            int rank = g_tok.sorted_by_len[i];
            int tlen = g_tok.vocab_len[rank];

            /* Early exit: if token is shorter than best match, stop */
            if (tlen <= best_len) break;

            /* Check if remaining text is long enough */
            if (pos + tlen > text_len) continue;

            /* Compare bytes */
            if (memcmp(text + pos, g_tok.vocab_str[rank], tlen) == 0) {
                best_rank = rank;
                best_len = tlen;
                break; /* sorted by length desc, first match is longest */
            }
        }

        if (best_rank >= 0) {
            out_tokens[n_tokens++] = TEKKEN_NUM_SPECIAL + best_rank;
            pos += best_len;
        } else {
            /* Fallback: try to find a single-byte token */
            unsigned char byte = (unsigned char)text[pos];
            int found = 0;
            for (int rank = 0; rank < g_tok.n_vocab && rank < 512; rank++) {
                if (g_tok.vocab_str[rank] && g_tok.vocab_len[rank] == 1 &&
                    (unsigned char)g_tok.vocab_str[rank][0] == byte) {
                    out_tokens[n_tokens++] = TEKKEN_NUM_SPECIAL + rank;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                /* Last resort: skip byte */
                fprintf(stderr, "tokenizer: no token for byte 0x%02x at pos %d\n",
                        byte, pos);
            }
            pos++;
        }
    }

    return n_tokens;
}

int tts_tokenizer_vocab_size(void) {
    return TEKKEN_NUM_SPECIAL + MAX_VOCAB;
}
