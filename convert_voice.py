#!/usr/bin/env python3
"""Convert voice embedding .pt files to raw binary format.

Usage:
    python3 convert_voice.py voice_embedding/neutral_female.pt neutral_female.bin

The .bin format is simply the raw BF16 tensor data, shape [N, 3072].
voxtral_tts can read both .pt and .bin files.
"""

import sys
import torch

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.pt> <output.bin>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    t = torch.load(input_path, map_location="cpu", weights_only=True)
    print(f"Shape: {t.shape}, dtype: {t.dtype}")

    # Ensure BF16
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
        print(f"Converted to {t.dtype}")

    # Save as raw bytes
    with open(output_path, "wb") as f:
        f.write(t.numpy().view("uint16").tobytes())

    print(f"Saved {output_path} ({t.numel() * 2} bytes)")

if __name__ == "__main__":
    main()
