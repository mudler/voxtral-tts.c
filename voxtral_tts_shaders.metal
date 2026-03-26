/*
 * voxtral_tts_shaders.metal - Metal Shading Language compute kernels
 *
 * Kernels for Voxtral TTS GPU-accelerated inference on Apple Silicon.
 * Embedded into the binary at compile time via xxd -i.
 */

#include <metal_stdlib>
using namespace metal;

/* ========================================================================
 * RMS Normalization
 *
 * out[i] = x[i] / rms(x) * weight[i]
 * rms = sqrt(mean(x^2) + eps)
 *
 * Grid: (seq_len, 1, 1)  Block: (256, 1, 1)
 * ======================================================================== */

kernel void rms_norm(
    device const float *x       [[buffer(0)]],
    device const float *weight   [[buffer(1)]],
    device float *out            [[buffer(2)]],
    constant int &hidden         [[buffer(3)]],
    constant float &eps          [[buffer(4)]],
    uint row                     [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint threads                 [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];

    device const float *x_row = x + row * hidden;
    device float *out_row = out + row * hidden;

    /* Sum of squares (cooperative) */
    float local_sum = 0.0f;
    for (int i = (int)tid; i < hidden; i += (int)threads) {
        float val = x_row[i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Binary tree reduction */
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_inv = rsqrt(shared_sum[0] / float(hidden) + eps);

    /* Normalize and scale */
    for (int i = (int)tid; i < hidden; i += (int)threads) {
        out_row[i] = x_row[i] * rms_inv * weight[i];
    }
}

/* ========================================================================
 * SiLU Activation (in-place)
 *
 * x[i] = x[i] / (1 + exp(-x[i]))
 * ======================================================================== */

kernel void silu(
    device float *x              [[buffer(0)]],
    constant int &n              [[buffer(1)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if ((int)gid < n) {
        float val = x[gid];
        x[gid] = val / (1.0f + exp(-val));
    }
}

/* ========================================================================
 * Fused SiLU * Multiply (SwiGLU)
 *
 * gate[i] = silu(gate[i]) * up[i]
 * ======================================================================== */

kernel void silu_mul(
    device float *gate           [[buffer(0)]],
    device const float *up       [[buffer(1)]],
    constant int &n              [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if ((int)gid < n) {
        float g = gate[gid];
        gate[gid] = (g / (1.0f + exp(-g))) * up[gid];
    }
}

/* ========================================================================
 * Residual Add (in-place)
 *
 * a[i] += b[i]
 * ======================================================================== */

kernel void add_inplace(
    device float *a              [[buffer(0)]],
    device const float *b        [[buffer(1)]],
    constant int &n              [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if ((int)gid < n) {
        a[gid] += b[gid];
    }
}

/* ========================================================================
 * RoPE (Rotary Position Embedding)
 *
 * Applied in-place to Q or K vectors.
 * data layout: [n_heads, head_dim] (contiguous)
 * freqs: [head_dim/2, 2] pairs of (cos, sin)
 *
 * Grid: (n_heads * head_dim/2, 1, 1)
 * ======================================================================== */

kernel void rope_apply(
    device float *data           [[buffer(0)]],
    device const float *freqs    [[buffer(1)]],
    constant int &n_heads        [[buffer(2)]],
    constant int &head_dim       [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    int half_dim = head_dim / 2;
    int total = n_heads * half_dim;
    if ((int)gid >= total) return;

    int head = (int)gid / half_dim;
    int i = (int)gid % half_dim;

    float cos_val = freqs[i * 2];
    float sin_val = freqs[i * 2 + 1];

    int base = head * head_dim;
    float x0 = data[base + i * 2];
    float x1 = data[base + i * 2 + 1];

    data[base + i * 2]     = x0 * cos_val - x1 * sin_val;
    data[base + i * 2 + 1] = x0 * sin_val + x1 * cos_val;
}

/* ========================================================================
 * RoPE with position parameter (for single-token decode)
 *
 * Computes frequencies on-the-fly from position and theta.
 * ======================================================================== */

kernel void rope_apply_pos(
    device float *data           [[buffer(0)]],
    constant int &n_heads        [[buffer(1)]],
    constant int &head_dim       [[buffer(2)]],
    constant int &pos            [[buffer(3)]],
    constant float &theta        [[buffer(4)]],
    uint gid                     [[thread_position_in_grid]]
) {
    int half_dim = head_dim / 2;
    int total = n_heads * half_dim;
    if ((int)gid >= total) return;

    int head = (int)gid / half_dim;
    int i = (int)gid % half_dim;

    float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
    float angle = float(pos) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    int base = head * head_dim;
    float x0 = data[base + i * 2];
    float x1 = data[base + i * 2 + 1];

    data[base + i * 2]     = x0 * cos_val - x1 * sin_val;
    data[base + i * 2 + 1] = x0 * sin_val + x1 * cos_val;
}

/* ========================================================================
 * KV Cache Copy (single token)
 *
 * cache[offset + i] = data[i]  for i in [0, kv_dim)
 * ======================================================================== */

kernel void kv_cache_copy(
    device float *cache          [[buffer(0)]],
    device const float *data     [[buffer(1)]],
    constant int &float_offset   [[buffer(2)]],
    constant int &kv_dim         [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if ((int)gid < kv_dim) {
        cache[float_offset + (int)gid] = data[gid];
    }
}

/* ========================================================================
 * Batched KV Cache Copy (prefill, multiple tokens)
 * ======================================================================== */

kernel void batched_kv_cache_copy(
    device float *cache          [[buffer(0)]],
    device const float *data     [[buffer(1)]],
    constant int &cache_offset   [[buffer(2)]],
    constant int &total          [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if ((int)gid < total) {
        cache[cache_offset + (int)gid] = data[gid];
    }
}

/* ========================================================================
 * Single-Token Decoder Attention (Causal, GQA, Online Softmax)
 *
 * One threadgroup per query head (128 threads = 4 SIMD groups).
 * Uses online softmax for numerically stable single-pass attention.
 *
 * Q: [n_heads * head_dim] (single query)
 * K_cache, V_cache: [seq_k, n_kv_heads * head_dim]
 * out: [n_heads * head_dim]
 *
 * Grid: (n_heads, 1, 1)  Block: (128, 1, 1)
 * ======================================================================== */

kernel void decoder_attention(
    device const float *Q        [[buffer(0)]],
    device const float *K_cache  [[buffer(1)]],
    device const float *V_cache  [[buffer(2)]],
    device float *out            [[buffer(3)]],
    constant int &n_heads        [[buffer(4)]],
    constant int &n_kv_heads     [[buffer(5)]],
    constant int &head_dim       [[buffer(6)]],
    constant int &kv_dim         [[buffer(7)]],
    constant int &seq_k          [[buffer(8)]],
    constant float &scale        [[buffer(9)]],
    constant int &window_size    [[buffer(10)]],
    constant int &q_pos          [[buffer(11)]],
    uint head_idx                [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint simd_gid                [[simdgroup_index_in_threadgroup]],
    uint simd_lid                [[thread_index_in_simdgroup]]
) {
    int gqa_ratio = n_heads / n_kv_heads;
    int kv_head = (int)head_idx / gqa_ratio;

    /* Load query value for this thread's dimension */
    float q_val = ((int)tid < head_dim) ?
        Q[(int)head_idx * head_dim + (int)tid] : 0.0f;

    /* Determine valid key range (causal + sliding window) */
    int valid_end = min(q_pos, seq_k - 1);
    int valid_start = (window_size > 0) ? max(0, q_pos - window_size + 1) : 0;

    /* Online softmax state */
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc = 0.0f;

    threadgroup float shared_simd[4]; /* for cross-SIMD reduction */

    for (int j = valid_start; j <= valid_end; j++) {
        device const float *k_j = K_cache + j * kv_dim + kv_head * head_dim;

        /* Cooperative dot product: Q . K[j] via SIMD reduction */
        float partial = ((int)tid < head_dim) ? q_val * k_j[tid] : 0.0f;
        float simd_partial = simd_sum(partial);

        /* Cross-SIMD reduction (4 SIMD groups -> 1 scalar) */
        if (simd_lid == 0) shared_simd[simd_gid] = simd_partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float score = 0.0f;
        if (tid == 0) {
            for (int s = 0; s < 4; s++) score += shared_simd[s];
            score *= scale;
            shared_simd[0] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        score = shared_simd[0];

        /* Online softmax update */
        float old_max = running_max;
        running_max = max(running_max, score);
        float correction = exp(old_max - running_max);
        running_sum = running_sum * correction + exp(score - running_max);

        /* Weighted value accumulation */
        if ((int)tid < head_dim) {
            device const float *v_j = V_cache + j * kv_dim + kv_head * head_dim;
            acc = acc * correction + exp(score - running_max) * v_j[tid];
        }
    }

    /* Normalize and write output */
    if ((int)tid < head_dim) {
        out[(int)head_idx * head_dim + (int)tid] =
            acc / (running_sum + 1e-10f);
    }
}

/* ========================================================================
 * Causal Softmax (for prefill attention scores)
 *
 * Applies causal mask + sliding window + softmax to score matrix.
 * Grid: (n_heads * seq_q, 1, 1)  Block: (256, 1, 1)
 * ======================================================================== */

kernel void causal_softmax(
    device float *scores         [[buffer(0)]],
    constant int &seq_q          [[buffer(1)]],
    constant int &seq_k          [[buffer(2)]],
    constant int &window_size    [[buffer(3)]],
    constant int &q_offset       [[buffer(4)]],
    uint group_id                [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint tg_size                 [[threads_per_threadgroup]]
) {
    int qi = (int)group_id % seq_q;
    int head = (int)group_id / seq_q;
    device float *row = scores + ((long)head * seq_q + qi) * seq_k;

    int q_pos = q_offset + qi;
    int valid_end = min(q_pos, seq_k - 1);
    int valid_start = (window_size > 0) ? max(0, q_pos - window_size + 1) : 0;

    threadgroup float shared[256];

    /* Phase 1: Apply mask + find max */
    float local_max = -INFINITY;
    for (int j = (int)tid; j < seq_k; j += (int)tg_size) {
        float val = (j >= valid_start && j <= valid_end) ? row[j] : -INFINITY;
        row[j] = val;
        local_max = max(local_max, val);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];

    /* Phase 2: Exp + sum */
    float local_sum = 0.0f;
    for (int j = (int)tid; j < seq_k; j += (int)tg_size) {
        float val = exp(row[j] - row_max);
        row[j] = val;
        local_sum += val;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / (shared[0] + 1e-10f);

    /* Phase 3: Normalize */
    for (int j = (int)tid; j < seq_k; j += (int)tg_size) {
        row[j] *= inv_sum;
    }
}

/* ========================================================================
 * BF16 to F32 Conversion
 *
 * Used when MPS can't handle bf16 directly.
 * ======================================================================== */

kernel void bf16_to_f32(
    device const ushort *src     [[buffer(0)]],
    device float *dst            [[buffer(1)]],
    constant int &n              [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if ((int)gid < n) {
        uint bits = uint(src[gid]) << 16;
        dst[gid] = as_type<float>(bits);
    }
}

/* ========================================================================
 * Time Embedding (for acoustic transformer flow matching)
 *
 * out[i] = cos(t * inv_freq[i])       for i < half_dim
 * out[half_dim + i] = sin(t * inv_freq[i])  for i < half_dim
 * ======================================================================== */

kernel void time_embedding(
    device float *out            [[buffer(0)]],
    device const float *inv_freq [[buffer(1)]],
    constant float &t_val        [[buffer(2)]],
    constant int &half_dim       [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if ((int)gid < half_dim) {
        float angle = t_val * inv_freq[gid];
        out[gid] = cos(angle);
        out[half_dim + (int)gid] = sin(angle);
    }
}

/* ========================================================================
 * Argmax (for token prediction)
 *
 * Finds index of maximum value in a float array.
 * Grid: (1, 1, 1)  Block: (256, 1, 1)
 * ======================================================================== */

kernel void argmax(
    device const float *data     [[buffer(0)]],
    device int *result           [[buffer(1)]],
    constant int &n              [[buffer(2)]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint threads                 [[threads_per_threadgroup]]
) {
    threadgroup float shared_vals[256];
    threadgroup int shared_idxs[256];

    float best_val = -INFINITY;
    int best_idx = 0;

    for (int i = (int)tid; i < n; i += (int)threads) {
        if (data[i] > best_val) {
            best_val = data[i];
            best_idx = i;
        }
    }
    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = threads / 2; s > 0; s >>= 1) {
        if (tid < s && shared_vals[tid + s] > shared_vals[tid]) {
            shared_vals[tid] = shared_vals[tid + s];
            shared_idxs[tid] = shared_idxs[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) result[0] = shared_idxs[0];
}
