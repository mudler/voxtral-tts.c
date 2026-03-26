/*
 * voxtral_tts_tokenizer.h - Tekken tokenizer for Voxtral TTS
 *
 * Supports both encoding (text -> tokens) and decoding (tokens -> text).
 * Uses byte-pair encoding (BPE) with the Tekken vocabulary.
 */

#ifndef VOXTRAL_TTS_TOKENIZER_H
#define VOXTRAL_TTS_TOKENIZER_H

#include <stdint.h>

/* Load tokenizer from tekken.json file */
int tts_tokenizer_load(const char *path);

/* Free tokenizer resources */
void tts_tokenizer_free(void);

/* Encode text to token IDs. Returns number of tokens written.
 * out_tokens must have space for max_tokens entries. */
int tts_tokenizer_encode(const char *text, int *out_tokens, int max_tokens);

/* Decode a single token ID to string (returns internal buffer, do not free) */
const char *tts_tokenizer_decode(int token_id);

/* Get the vocabulary size */
int tts_tokenizer_vocab_size(void);

#endif /* VOXTRAL_TTS_TOKENIZER_H */
