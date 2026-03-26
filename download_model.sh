#!/bin/bash
# Download Voxtral-4B-TTS model from HuggingFace
#
# Usage: ./download_model.sh [target-dir]
#
# Requires: huggingface-cli (pip install huggingface_hub)

set -e

TARGET_DIR="${1:-voxtral-tts-model}"
REPO_ID="mistralai/Voxtral-4B-TTS-2603"

echo "Downloading Voxtral TTS model to: $TARGET_DIR"
echo "Repository: $REPO_ID"
echo ""

mkdir -p "$TARGET_DIR"

# Check if huggingface-cli is available
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli download "$REPO_ID" \
        --local-dir "$TARGET_DIR" \
        --include "consolidated.safetensors" "params.json" "tekken.json" "voice_embedding/*"
elif command -v wget &> /dev/null; then
    BASE_URL="https://huggingface.co/$REPO_ID/resolve/main"

    echo "Downloading consolidated.safetensors (~8GB)..."
    wget -c "$BASE_URL/consolidated.safetensors" -O "$TARGET_DIR/consolidated.safetensors"

    echo "Downloading params.json..."
    wget -c "$BASE_URL/params.json" -O "$TARGET_DIR/params.json"

    echo "Downloading tekken.json..."
    wget -c "$BASE_URL/tekken.json" -O "$TARGET_DIR/tekken.json"

    echo "Downloading voice embeddings..."
    mkdir -p "$TARGET_DIR/voice_embedding"
    for voice in casual_female casual_male cheerful_female neutral_female neutral_male \
                 ar_male de_female de_male es_female es_male fr_female fr_male \
                 hi_female hi_male it_female it_male nl_female nl_male pt_female pt_male; do
        echo "  $voice.pt"
        wget -c "$BASE_URL/voice_embedding/$voice.pt" -O "$TARGET_DIR/voice_embedding/$voice.pt"
    done
else
    echo "Error: Need huggingface-cli or wget"
    echo "Install: pip install huggingface_hub"
    exit 1
fi

echo ""
echo "Download complete. Model directory: $TARGET_DIR"
echo "Run: ./voxtral_tts -d $TARGET_DIR -v neutral_female -o output.wav \"Hello world\""
