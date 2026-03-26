# voxtral-tts.c

Pure C inference engine for [Mistral's Voxtral-4B-TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) text-to-speech model. Zero external dependencies beyond the C standard library and math. Reads weights directly from safetensors via memory-mapped I/O.

## Features

- Single-file model loading from `consolidated.safetensors`
- BF16 weights accessed directly from mmap (no full conversion needed)
- 20 preset voices across 9 languages
- WAV output at 24kHz
- Optional BLAS acceleration (OpenBLAS, Apple Accelerate)
- NEON-optimized BF16 matvec on ARM

## Architecture

Voxtral TTS is a three-stage pipeline:

```
Text --> LLM Backbone (3.4B, 26-layer Mistral)
     --> Flow-Matching Acoustic Transformer (390M, 3 layers, 8 Euler steps)
     --> Audio Codec Decoder (300M, 4-stage conv + ALiBi transformer)
     --> 24kHz waveform
```

The LLM autoregressively generates hidden states conditioned on text and a voice prompt. The acoustic transformer converts each hidden state to 37 audio codes (1 semantic + 36 acoustic) via flow matching with classifier-free guidance. The codec decoder converts all collected codes into a raw waveform.

## Quick Start

### Build

```bash
# Linux with OpenBLAS (recommended)
make blas

# macOS with Accelerate
make apple

# Portable (no BLAS, slower)
make noblas
```

### Download Model

```bash
# Requires huggingface-cli or wget
./download_model.sh voxtral-tts-model
```

This downloads the model weights (~8GB), tokenizer, and voice embeddings from HuggingFace.

### Run

```bash
./voxtral_tts -d voxtral-tts-model -v neutral_female -o output.wav "Hello world"
```

### Options

```
Usage: ./voxtral_tts [options] "text to speak"

  -d <dir>        Model directory (required)
  -v <voice>      Voice name (default: neutral_female)
  -o <file>       Output WAV file (default: output.wav)
  -s <seed>       Random seed for reproducibility
  --verbose       Enable verbose output
  --inspect       Print model tensor info and exit
```

### Available Voices

| Language   | Voices                                     |
|------------|--------------------------------------------|
| English    | casual_female, casual_male, cheerful_female, neutral_female, neutral_male |
| French     | fr_female, fr_male                         |
| German     | de_female, de_male                         |
| Spanish    | es_female, es_male                         |
| Italian    | it_female, it_male                         |
| Portuguese | pt_female, pt_male                         |
| Dutch      | nl_female, nl_male                         |
| Arabic     | ar_male                                    |
| Hindi      | hi_female, hi_male                         |

## Project Structure

```
voxtral_tts.h                 Main header (constants, structs, API)
voxtral_tts.c                 Model loading and inference orchestrator
voxtral_tts_llm.c             26-layer Mistral decoder with KV cache
voxtral_tts_acoustic.c        Flow-matching acoustic transformer
voxtral_tts_codec.c           Audio codec decoder (ALiBi + weight_norm)
voxtral_tts_kernels.{c,h}     Math kernels (matmul, attention, conv, RoPE, ...)
voxtral_tts_tokenizer.{c,h}   Tekken BPE tokenizer (encode + decode)
voxtral_tts_voice.c           Voice embedding loader (.pt) + audio codebook embeddings
voxtral_tts_wav.c             WAV file writer
voxtral_tts_safetensors.{c,h} Safetensors mmap reader
main.c                        CLI entry point
```

## Utilities

- `inspect_weights` -- dump tensor names/shapes from safetensors (`make inspect`)
- `convert_voice.py` -- convert .pt voice embeddings to raw binary
- `download_model.sh` -- download model from HuggingFace

## How It Works

The prompt format follows `mistral_common`'s `encode_speech_request`:

```
[BOS] [BEGIN_AUDIO] [voice_embedding x N] [/INST] text_tokens [INST] [BEGIN_AUDIO]
```

Voice embeddings are pre-computed BF16 tensors of shape `[N, 3072]` that replace audio token placeholder positions. After prefill, the model enters an autoregressive loop:

1. LLM produces a hidden state
2. Acoustic transformer predicts a semantic code (greedy argmax) and 36 acoustic codes (flow matching with 8 Euler ODE steps and CFG alpha=1.2)
3. The 37 codes are embedded back into LLM input space via multi-vocabulary embeddings (sum across codebooks)
4. Repeat until `[END_AUDIO]` is generated
5. All collected codes are decoded by the audio codec into a 24kHz waveform

## Requirements

- C11 compiler (gcc, clang)
- ~10GB RAM (8GB mmap'd weights + working memory)
- Optional: OpenBLAS or Apple Accelerate for faster matrix operations

## License

MIT License. See [LICENSE](LICENSE).

**Note**: The Voxtral-4B-TTS model weights are released by Mistral AI under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This inference engine is MIT-licensed but the model weights have their own license terms.

## Author

**Ettore Di Giacinto** ([@mudler](https://github.com/mudler))

## Acknowledgements

This project builds on the work of several open-source projects:

- **[voxtral.c](https://github.com/antirez/voxtral.c)** by Salvatore Sanfilippo (antirez) -- Pure C inference engine for Voxtral Realtime (ASR). The safetensors reader, math kernels, Mistral decoder implementation, and overall architecture of this project are directly adapted from voxtral.c. The project demonstrated that a full transformer inference engine can be written in clean, dependency-free C.

- **[vLLM](https://github.com/vllm-project/vllm)** and **[vLLM-Omni](https://github.com/vllm-project/vllm-omni)** -- The reference Python implementation for Voxtral TTS inference. The flow-matching acoustic transformer, audio codec decoder, and the overall TTS pipeline were implemented based on the vLLM-Omni model code. The prompt format, voice embedding handling, and audio code generation logic follow vLLM-Omni's implementation.

- **[Mistral AI](https://mistral.ai/)** -- For developing and open-sourcing the Voxtral TTS model and the [mistral_common](https://github.com/mistralai/mistral-common) tokenizer library.
