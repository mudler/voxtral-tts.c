# Voxtral TTS - Pure C Text-to-Speech Inference
#
# Targets:
#   make          - Build with OpenBLAS (default)
#   make blas     - Build with OpenBLAS
#   make apple    - Build with Apple Accelerate (macOS)
#   make noblas   - Build without BLAS (slow, portable)
#   make clean    - Remove build artifacts
#   make inspect  - Build weight inspector tool

CC ?= gcc
CFLAGS = -O3 -Wall -Wextra -Wno-unused-parameter -std=c11 -D_GNU_SOURCE
LDFLAGS = -lm

# Source files
SRCS = voxtral_tts.c \
       voxtral_tts_safetensors.c \
       voxtral_tts_kernels.c \
       voxtral_tts_llm.c \
       voxtral_tts_acoustic.c \
       voxtral_tts_codec.c \
       voxtral_tts_voice.c \
       voxtral_tts_wav.c \
       voxtral_tts_tokenizer.c \
       main.c

OBJS = $(SRCS:.c=.o)
TARGET = voxtral_tts

# Default: OpenBLAS
all: blas

blas: CFLAGS += -DUSE_BLAS
blas: LDFLAGS += -lopenblas
blas: $(TARGET)

# macOS with Accelerate
apple: CFLAGS += -DUSE_BLAS
apple: LDFLAGS += -framework Accelerate
apple: $(TARGET)

# No BLAS (portable but slower)
noblas: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Weight inspector (useful for debugging model files)
inspect: voxtral_tts_safetensors.o inspect_weights.o
	$(CC) $(CFLAGS) -o inspect_weights $^ $(LDFLAGS)

inspect_weights.o: inspect_weights.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET) inspect_weights inspect_weights.o

# Dependencies
voxtral_tts.o: voxtral_tts.c voxtral_tts.h voxtral_tts_safetensors.h voxtral_tts_kernels.h
voxtral_tts_safetensors.o: voxtral_tts_safetensors.c voxtral_tts_safetensors.h
voxtral_tts_kernels.o: voxtral_tts_kernels.c voxtral_tts_kernels.h
voxtral_tts_llm.o: voxtral_tts_llm.c voxtral_tts.h voxtral_tts_kernels.h voxtral_tts_safetensors.h
voxtral_tts_acoustic.o: voxtral_tts_acoustic.c voxtral_tts.h voxtral_tts_kernels.h voxtral_tts_safetensors.h
voxtral_tts_codec.o: voxtral_tts_codec.c voxtral_tts.h voxtral_tts_kernels.h voxtral_tts_safetensors.h
voxtral_tts_voice.o: voxtral_tts_voice.c voxtral_tts.h voxtral_tts_kernels.h
voxtral_tts_wav.o: voxtral_tts_wav.c voxtral_tts.h
voxtral_tts_tokenizer.o: voxtral_tts_tokenizer.c voxtral_tts.h
main.o: main.c voxtral_tts.h

.PHONY: all blas apple noblas clean inspect
