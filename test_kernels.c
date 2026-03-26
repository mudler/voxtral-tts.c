/*
 * test_kernels.c - GPU kernel correctness test harness
 *
 * Compares CPU reference kernels against GPU (CUDA/Metal) implementations.
 * Each test generates deterministic random data, runs both versions,
 * and reports max/mean relative error.
 *
 * Build:
 *   make test-cuda   (requires CUDA toolkit)
 *   make test-cpu    (CPU-only sanity check)
 *
 * Usage:
 *   ./test_kernels
 */

#include "voxtral_tts.h"
#include "voxtral_tts_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_CUDA
#include "voxtral_tts_cuda.h"
#endif

/* ========================================================================
 * Test Infrastructure
 * ======================================================================== */

static int total_tests = 0;
static int passed_tests = 0;
static int failed_tests = 0;

/* Deterministic RNG (same as voxtral_tts_kernels.c) */
static uint64_t test_rng_state;

static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return x;
}

static float rand_uniform(uint64_t *s) {
    return (float)(xorshift64(s) >> 11) * (1.0f / 9007199254740992.0f);
}

static float rand_normal(uint64_t *s) {
    float u1, u2;
    do { u1 = rand_uniform(s); } while (u1 < 1e-30f);
    u2 = rand_uniform(s);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static void fill_random(float *buf, int n, uint64_t seed) {
    test_rng_state = seed;
    for (int i = 0; i < n; i++)
        buf[i] = rand_normal(&test_rng_state) * 0.1f;
}

static void fill_random_positive(float *buf, int n, uint64_t seed) {
    test_rng_state = seed;
    for (int i = 0; i < n; i++)
        buf[i] = fabsf(rand_normal(&test_rng_state)) * 0.1f + 0.01f;
}

static void fill_random_bf16(uint16_t *buf, int n, uint64_t seed) {
    test_rng_state = seed;
    for (int i = 0; i < n; i++) {
        float val = rand_normal(&test_rng_state) * 0.02f;
        uint32_t bits;
        memcpy(&bits, &val, 4);
        buf[i] = (uint16_t)(bits >> 16);
    }
}

/* Compare two float arrays, return 0 if within tolerance */
static int compare_tensors(const char *name, const float *ref, const float *test,
                           int n, float rtol, float atol) {
    float max_rel = 0.0f, sum_rel = 0.0f;
    float max_abs = 0.0f;
    int max_idx = 0;
    int n_bad = 0;

    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - test[i]);
        float denom = fabsf(ref[i]);
        if (denom < 1e-8f) denom = 1e-8f;
        float rel = diff / denom;

        if (diff > max_abs) max_abs = diff;
        if (rel > max_rel) { max_rel = rel; max_idx = i; }
        sum_rel += rel;

        if (rel > rtol && diff > atol) n_bad++;
    }

    float mean_rel = sum_rel / (float)n;
    int pass = (max_rel <= rtol || max_abs <= atol) && n_bad == 0;

    total_tests++;
    if (pass) {
        passed_tests++;
        printf("[PASS] %-35s max_rel=%.2e mean_rel=%.2e max_abs=%.2e\n",
               name, max_rel, mean_rel, max_abs);
    } else {
        failed_tests++;
        printf("[FAIL] %-35s max_rel=%.2e mean_rel=%.2e max_abs=%.2e "
               "bad=%d/%d (idx=%d: ref=%.6f test=%.6f)\n",
               name, max_rel, mean_rel, max_abs,
               n_bad, n, max_idx, ref[max_idx], test[max_idx]);
    }

    return pass ? 0 : -1;
}

/* ========================================================================
 * Test: RMS Norm
 * ======================================================================== */

static void test_rms_norm(void) {
    int dim = TTS_DEC_DIM; /* 3072 */
    float *x = malloc(dim * sizeof(float));
    float *weight = malloc(dim * sizeof(float));
    float *cpu_out = malloc(dim * sizeof(float));
    float *gpu_out = malloc(dim * sizeof(float));

    fill_random(x, dim, 42);
    fill_random_positive(weight, dim, 123);

    /* CPU reference */
    tts_rms_norm(cpu_out, x, weight, 1, dim, TTS_DEC_NORM_EPS);

#ifdef USE_CUDA
    if (tts_cuda_available()) {
        float *d_x, *d_w, *d_out;
        d_x = tts_cuda_alloc(dim * sizeof(float));
        d_w = tts_cuda_alloc(dim * sizeof(float));
        d_out = tts_cuda_alloc(dim * sizeof(float));
        tts_cuda_to_device(d_x, x, dim * sizeof(float));
        tts_cuda_to_device(d_w, weight, dim * sizeof(float));

        tts_cuda_rms_norm(d_out, d_x, d_w, 1, dim, TTS_DEC_NORM_EPS);
        tts_cuda_sync();

        tts_cuda_to_host(gpu_out, d_out, dim * sizeof(float));
        compare_tensors("rms_norm(dim=3072)", cpu_out, gpu_out, dim, 1e-4f, 1e-6f);

        tts_cuda_free_ptr(d_x); tts_cuda_free_ptr(d_w); tts_cuda_free_ptr(d_out);
    } else
#endif
    {
        /* CPU-only: compare against self */
        float *cpu_out2 = malloc(dim * sizeof(float));
        tts_rms_norm(cpu_out2, x, weight, 1, dim, TTS_DEC_NORM_EPS);
        compare_tensors("rms_norm(cpu-self)", cpu_out, cpu_out2, dim, 0.0f, 0.0f);
        free(cpu_out2);
    }

    free(x); free(weight); free(cpu_out); free(gpu_out);
}

/* ========================================================================
 * Test: Causal Attention (single-token decode)
 * ======================================================================== */

static void test_causal_attention(int seq_k) {
    int n_heads = TTS_DEC_HEADS;       /* 32 */
    int n_kv_heads = TTS_DEC_KV_HEADS; /* 8 */
    int head_dim = TTS_DEC_HEAD_DIM;   /* 128 */
    int q_dim = n_heads * head_dim;    /* 4096 */
    int kv_dim = n_kv_heads * head_dim; /* 1024 */
    float scale = 1.0f / sqrtf((float)head_dim);

    float *Q = malloc(q_dim * sizeof(float));
    float *K = malloc((size_t)seq_k * kv_dim * sizeof(float));
    float *V = malloc((size_t)seq_k * kv_dim * sizeof(float));
    float *cpu_out = calloc(q_dim, sizeof(float));
    float *gpu_out = calloc(q_dim, sizeof(float));

    fill_random(Q, q_dim, 100 + seq_k);
    fill_random(K, seq_k * kv_dim, 200 + seq_k);
    fill_random(V, seq_k * kv_dim, 300 + seq_k);

    int q_offset = seq_k - 1; /* last position */

    /* CPU reference */
    tts_causal_attention(cpu_out, Q, K, V,
                         1, seq_k, n_heads, n_kv_heads, head_dim,
                         scale, 0, q_offset);

#ifdef USE_CUDA
    if (tts_cuda_available()) {
        float *d_Q, *d_K, *d_V, *d_out;
        d_Q = tts_cuda_alloc(q_dim * sizeof(float));
        d_K = tts_cuda_alloc((size_t)seq_k * kv_dim * sizeof(float));
        d_V = tts_cuda_alloc((size_t)seq_k * kv_dim * sizeof(float));
        d_out = tts_cuda_alloc(q_dim * sizeof(float));
        tts_cuda_memset(d_out, 0, q_dim * sizeof(float));

        tts_cuda_to_device(d_Q, Q, q_dim * sizeof(float));
        tts_cuda_to_device(d_K, K, (size_t)seq_k * kv_dim * sizeof(float));
        tts_cuda_to_device(d_V, V, (size_t)seq_k * kv_dim * sizeof(float));

        tts_cuda_causal_attention_decode(d_out, d_Q, d_K, d_V,
                                          seq_k, n_heads, n_kv_heads, head_dim,
                                          scale, kv_dim);
        tts_cuda_sync();

        tts_cuda_to_host(gpu_out, d_out, q_dim * sizeof(float));

        char name[64];
        snprintf(name, sizeof(name), "causal_attn(seq_k=%d)", seq_k);
        compare_tensors(name, cpu_out, gpu_out, q_dim, 1e-3f, 1e-5f);

        tts_cuda_free_ptr(d_Q); tts_cuda_free_ptr(d_K); tts_cuda_free_ptr(d_V); tts_cuda_free_ptr(d_out);
    } else
#endif
    {
        char name[64];
        snprintf(name, sizeof(name), "causal_attn(seq_k=%d,cpu-self)", seq_k);
        float *cpu_out2 = calloc(q_dim, sizeof(float));
        tts_causal_attention(cpu_out2, Q, K, V,
                             1, seq_k, n_heads, n_kv_heads, head_dim,
                             scale, 0, q_offset);
        compare_tensors(name, cpu_out, cpu_out2, q_dim, 0.0f, 0.0f);
        free(cpu_out2);
    }

    free(Q); free(K); free(V); free(cpu_out); free(gpu_out);
}

/* ========================================================================
 * Test: Bidirectional Attention (3 tokens, acoustic transformer)
 * ======================================================================== */

static void test_bidirectional_attention(void) {
    int seq = 3;
    int n_heads = TTS_AC_HEADS;
    int n_kv_heads = TTS_AC_KV_HEADS;
    int head_dim = TTS_AC_HEAD_DIM;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    float *Q = malloc((size_t)seq * q_dim * sizeof(float));
    float *K = malloc((size_t)seq * kv_dim * sizeof(float));
    float *V = malloc((size_t)seq * kv_dim * sizeof(float));
    float *cpu_out = calloc((size_t)seq * q_dim, sizeof(float));
    float *gpu_out = calloc((size_t)seq * q_dim, sizeof(float));

    fill_random(Q, seq * q_dim, 500);
    fill_random(K, seq * kv_dim, 600);
    fill_random(V, seq * kv_dim, 700);

    tts_bidirectional_attention(cpu_out, Q, K, V, seq,
                                n_heads, n_kv_heads, head_dim, scale);

#ifdef USE_CUDA
    if (tts_cuda_available()) {
        float *d_Q, *d_K, *d_V, *d_out;
        d_Q = tts_cuda_alloc(seq * q_dim * sizeof(float));
        d_K = tts_cuda_alloc(seq * kv_dim * sizeof(float));
        d_V = tts_cuda_alloc(seq * kv_dim * sizeof(float));
        d_out = tts_cuda_alloc(seq * q_dim * sizeof(float));
        tts_cuda_memset(d_out, 0, seq * q_dim * sizeof(float));

        tts_cuda_to_device(d_Q, Q, seq * q_dim * sizeof(float));
        tts_cuda_to_device(d_K, K, seq * kv_dim * sizeof(float));
        tts_cuda_to_device(d_V, V, seq * kv_dim * sizeof(float));

        tts_cuda_bidirectional_attention(d_out, d_Q, d_K, d_V,
                                          seq, n_heads, n_kv_heads, head_dim, scale);
        tts_cuda_sync();

        tts_cuda_to_host(gpu_out, d_out, seq * q_dim * sizeof(float));
        compare_tensors("bidirectional_attn(seq=3)", cpu_out, gpu_out,
                        seq * q_dim, 1e-3f, 1e-5f);

        tts_cuda_free_ptr(d_Q); tts_cuda_free_ptr(d_K); tts_cuda_free_ptr(d_V); tts_cuda_free_ptr(d_out);
    } else
#endif
    {
        float *cpu_out2 = calloc((size_t)seq * q_dim, sizeof(float));
        tts_bidirectional_attention(cpu_out2, Q, K, V, seq,
                                    n_heads, n_kv_heads, head_dim, scale);
        compare_tensors("bidirectional_attn(cpu-self)", cpu_out, cpu_out2,
                        seq * q_dim, 0.0f, 0.0f);
        free(cpu_out2);
    }

    free(Q); free(K); free(V); free(cpu_out); free(gpu_out);
}

/* ========================================================================
 * Test: SiLU * mul (SwiGLU fused)
 * ======================================================================== */

static void test_silu_mul(void) {
    int n = TTS_DEC_HIDDEN; /* 9216 */
    float *gate = malloc(n * sizeof(float));
    float *up = malloc(n * sizeof(float));
    float *gate_ref = malloc(n * sizeof(float));
    float *gpu_out = malloc(n * sizeof(float));

    fill_random(gate, n, 800);
    fill_random(up, n, 900);

    /* CPU reference: silu(gate) then mul by up */
    memcpy(gate_ref, gate, n * sizeof(float));
    tts_silu(gate_ref, n);
    tts_mul_inplace(gate_ref, up, n);

#ifdef USE_CUDA
    if (tts_cuda_available()) {
        float *d_gate, *d_up;
        d_gate = tts_cuda_alloc(n * sizeof(float));
        d_up = tts_cuda_alloc(n * sizeof(float));
        tts_cuda_to_device(d_gate, gate, n * sizeof(float));
        tts_cuda_to_device(d_up, up, n * sizeof(float));

        tts_cuda_silu_mul(d_gate, d_up, n);
        tts_cuda_sync();

        tts_cuda_to_host(gpu_out, d_gate, n * sizeof(float));
        compare_tensors("silu_mul(n=9216)", gate_ref, gpu_out, n, 1e-5f, 1e-7f);

        tts_cuda_free_ptr(d_gate); tts_cuda_free_ptr(d_up);
    } else
#endif
    {
        float *gate2 = malloc(n * sizeof(float));
        memcpy(gate2, gate, n * sizeof(float));
        tts_silu(gate2, n);
        tts_mul_inplace(gate2, up, n);
        compare_tensors("silu_mul(cpu-self)", gate_ref, gate2, n, 0.0f, 0.0f);
        free(gate2);
    }

    free(gate); free(up); free(gate_ref); free(gpu_out);
}

/* ========================================================================
 * Test: Residual Add
 * ======================================================================== */

static void test_add_inplace(void) {
    int n = TTS_DEC_DIM;
    float *a = malloc(n * sizeof(float));
    float *b = malloc(n * sizeof(float));
    float *a_ref = malloc(n * sizeof(float));
    float *gpu_out = malloc(n * sizeof(float));

    fill_random(a, n, 1000);
    fill_random(b, n, 1001);
    memcpy(a_ref, a, n * sizeof(float));
    tts_add_inplace(a_ref, b, n);

#ifdef USE_CUDA
    if (tts_cuda_available()) {
        float *d_a, *d_b;
        d_a = tts_cuda_alloc(n * sizeof(float));
        d_b = tts_cuda_alloc(n * sizeof(float));
        tts_cuda_to_device(d_a, a, n * sizeof(float));
        tts_cuda_to_device(d_b, b, n * sizeof(float));

        tts_cuda_add_inplace(d_a, d_b, n);
        tts_cuda_sync();

        tts_cuda_to_host(gpu_out, d_a, n * sizeof(float));
        compare_tensors("add_inplace(n=3072)", a_ref, gpu_out, n, 0.0f, 0.0f);

        tts_cuda_free_ptr(d_a); tts_cuda_free_ptr(d_b);
    } else
#endif
    {
        compare_tensors("add_inplace(cpu-self)", a_ref, a_ref, n, 0.0f, 0.0f);
    }

    free(a); free(b); free(a_ref); free(gpu_out);
}

/* ========================================================================
 * Test: cuBLAS GEMM (bf16 weights)
 * ======================================================================== */

static void test_gemm_bf16(void) {
    int M = 1, K = 3072, N = 4096;
    float *x = malloc(M * K * sizeof(float));
    uint16_t *W_bf16 = malloc((size_t)N * K * sizeof(uint16_t));
    float *cpu_out = malloc(M * N * sizeof(float));
    float *gpu_out = malloc(M * N * sizeof(float));

    fill_random(x, M * K, 1100);
    fill_random_bf16(W_bf16, N * K, 1200);

    /* CPU reference */
    tts_linear_nobias_bf16(cpu_out, x, W_bf16, M, K, N);

#ifdef USE_CUDA
    if (tts_cuda_available()) {
        /* Upload to GPU */
        void *d_W;
        float *d_x, *d_y;
        d_W = tts_cuda_alloc((size_t)N * K * sizeof(uint16_t));
        d_x = tts_cuda_alloc(M * K * sizeof(float));
        d_y = tts_cuda_alloc(M * N * sizeof(float));
        tts_cuda_to_device(d_W, W_bf16, (size_t)N * K * sizeof(uint16_t));
        tts_cuda_to_device(d_x, x, M * K * sizeof(float));

        tts_cuda_linear_bf16(d_y, d_x, NULL, M, K, N, d_W);
        tts_cuda_sync();

        tts_cuda_to_host(gpu_out, d_y, M * N * sizeof(float));
        compare_tensors("gemm_bf16(1x3072 @ 4096x3072)", cpu_out, gpu_out,
                        M * N, 1e-3f, 1e-4f);

        tts_cuda_free_ptr(d_W); tts_cuda_free_ptr(d_x); tts_cuda_free_ptr(d_y);
    } else
#endif
    {
        float *cpu_out2 = malloc(M * N * sizeof(float));
        tts_linear_nobias_bf16(cpu_out2, x, W_bf16, M, K, N);
        compare_tensors("gemm_bf16(cpu-self)", cpu_out, cpu_out2, M * N, 0.0f, 0.0f);
        free(cpu_out2);
    }

    free(x); free(W_bf16); free(cpu_out); free(gpu_out);
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;

    printf("=== Voxtral TTS Kernel Test Harness ===\n");

#ifdef USE_CUDA
    printf("CUDA: enabled\n");
    if (tts_cuda_init(512) == 0) {
        printf("GPU initialized successfully\n");
    } else {
        printf("GPU init failed, running CPU-only tests\n");
    }
#else
    printf("CUDA: disabled (CPU-only tests)\n");
#endif

    printf("\n--- Element-wise Operations ---\n");
    test_add_inplace();
    test_silu_mul();

    printf("\n--- Normalization ---\n");
    test_rms_norm();

    printf("\n--- GEMM ---\n");
    test_gemm_bf16();

    printf("\n--- Attention ---\n");
    test_causal_attention(1);
    test_causal_attention(3);
    test_causal_attention(10);
    test_causal_attention(50);
    test_causal_attention(100);
    test_causal_attention(225);
    test_bidirectional_attention();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           passed_tests, total_tests, failed_tests);

#ifdef USE_CUDA
    tts_cuda_free();
#endif

    return failed_tests > 0 ? 1 : 0;
}
