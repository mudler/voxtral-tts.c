#!/usr/bin/env python3
"""Quantize Voxtral TTS BF16 safetensors to INT8 with per-channel scales.

Usage:
    python quantize_model.py <model_dir> <output_dir>

Example:
    python quantize_model.py ~/.cache/voxtral/model ~/.cache/voxtral/model_int8

The output directory will contain:
    consolidated.safetensors  — INT8 weights + FP32 norms/embeddings
    (scales are stored as companion tensors with "_scale" suffix)
"""

import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def quantize_per_channel_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight tensor to INT8 with per-output-channel scales.

    Returns (int8_weights, scales) where:
        int8_weights: shape [out, in], dtype int8
        scales: shape [out], dtype float32
        original ≈ int8_weights * scales.unsqueeze(1)
    """
    assert tensor.ndim == 2, f"Expected 2D tensor, got {tensor.ndim}D"
    fp32 = tensor.float()

    # Per-channel (per-row) scale: max absolute value / 127
    amax = fp32.abs().amax(dim=1)  # [out_dim]
    scale = amax / 127.0
    scale = scale.clamp(min=1e-10)  # avoid division by zero

    # Quantize
    int8_weights = (fp32 / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

    return int8_weights, scale


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output_dir>")
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    src = model_dir / "consolidated.safetensors"
    if not src.exists():
        print(f"Error: {src} not found")
        sys.exit(1)

    print(f"Loading {src}...")
    tensors = load_file(str(src))
    print(f"  {len(tensors)} tensors loaded")

    output_tensors = {}
    quantized_count = 0
    kept_count = 0

    for name, tensor in tensors.items():
        # Quantize 2D weight tensors (linear layers) that are large enough
        is_weight = tensor.ndim == 2 and tensor.numel() > 1024
        # Skip embeddings (vocab embeddings, codebook) — keep as BF16
        is_embedding = "embedding" in name or "codebook" in name
        # Skip norms and biases — keep as FP32
        is_norm_or_bias = "norm" in name or "bias" in name
        # Skip codec decoder weights (conv layers, not standard linear)
        is_codec = "audio_tokenizer.decoder" in name or "audio_tokenizer.quantizer" in name

        if is_weight and not is_embedding and not is_norm_or_bias and not is_codec:
            int8_w, scale = quantize_per_channel_int8(tensor)
            output_tensors[name] = int8_w
            output_tensors[name + "_scale"] = scale
            quantized_count += 1

            # Print size savings
            orig_mb = tensor.numel() * 2 / 1024 / 1024  # BF16 = 2 bytes
            new_mb = (int8_w.numel() + scale.numel() * 4) / 1024 / 1024
            print(f"  INT8: {name} [{list(tensor.shape)}] "
                  f"{orig_mb:.1f}MB -> {new_mb:.1f}MB")
        else:
            # Keep as-is (convert BF16 to FP32 for norms, keep BF16 for embeddings)
            if tensor.dtype == torch.bfloat16 and is_norm_or_bias:
                output_tensors[name] = tensor.float()
            else:
                output_tensors[name] = tensor
            kept_count += 1

    dst = output_dir / "consolidated.safetensors"
    print(f"\nSaving {dst}...")
    print(f"  {quantized_count} tensors quantized to INT8")
    print(f"  {kept_count} tensors kept as-is")
    save_file(output_tensors, str(dst))

    # Copy other model files
    for fname in ["params.json", "tekken.json"]:
        src_f = model_dir / fname
        dst_f = output_dir / fname
        if src_f.exists() and not dst_f.exists():
            import shutil
            shutil.copy2(str(src_f), str(dst_f))
            print(f"  Copied {fname}")

    # Copy voice embeddings
    voice_src = model_dir / "voice_embedding"
    voice_dst = output_dir / "voice_embedding"
    if voice_src.exists() and not voice_dst.exists():
        import shutil
        shutil.copytree(str(voice_src), str(voice_dst))
        print(f"  Copied voice_embedding/")

    # Report sizes
    orig_size = (model_dir / "consolidated.safetensors").stat().st_size
    new_size = dst.stat().st_size
    print(f"\nOriginal: {orig_size / 1024**3:.2f} GB")
    print(f"Quantized: {new_size / 1024**3:.2f} GB")
    print(f"Ratio: {new_size / orig_size:.1%}")


if __name__ == "__main__":
    main()
