/*
 * voxtral_tts_cuda.cu - CUDA kernels and cuBLAS integration for Voxtral TTS
 *
 * Custom CUDA kernels for: RMS norm, RoPE, SiLU*mul, residual add, attention.
 * cuBLAS used for GEMM (bf16 weights x f32 activations -> f32 output).
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "voxtral_tts_cuda.h"
#include "voxtral_tts.h"
}

/* ========================================================================
 * Error Checking
 * ======================================================================== */

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

#define CUDA_CHECK_VOID(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
                __FILE__, __LINE__, status); \
    } \
} while(0)

/* ========================================================================
 * Global CUDA Context
 * ======================================================================== */

tts_cuda_ctx_t g_cuda = {0};

/* ========================================================================
 * Host<->Device Transfers
 * ======================================================================== */

extern "C" void tts_cuda_to_device(void *d_dst, const void *h_src, size_t bytes) {
    cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice);
}

extern "C" void tts_cuda_to_host(void *h_dst, const void *d_src, size_t bytes) {
    cudaMemcpy(h_dst, d_src, bytes, cudaMemcpyDeviceToHost);
}

extern "C" void tts_cuda_memset(void *d_ptr, int value, size_t bytes) {
    cudaMemset(d_ptr, value, bytes);
}

extern "C" void *tts_cuda_alloc(size_t bytes) {
    void *ptr = NULL;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

extern "C" void tts_cuda_free_ptr(void *d_ptr) {
    if (d_ptr) cudaFree(d_ptr);
}

extern "C" void tts_cuda_sync(void) {
    cudaDeviceSynchronize();
}

/* ========================================================================
 * Initialization
 * ======================================================================== */

extern "C" int tts_cuda_init(int kv_cache_max) {
    if (g_cuda.initialized) return 0;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "cuda: no GPU found\n");
        return -1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    g_cuda.sm_major = prop.major;
    g_cuda.sm_minor = prop.minor;
    g_cuda.has_bf16 = (prop.major >= 8); /* Ampere+ for native bf16 */

    fprintf(stderr, "  CUDA: %s (sm_%d%d), %.1f GB VRAM, bf16=%s\n",
            prop.name, prop.major, prop.minor,
            (float)prop.totalGlobalMem / (1024*1024*1024),
            g_cuda.has_bf16 ? "yes" : "no (will use fp16)");

    /* Create cuBLAS handle */
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    g_cuda.cublas_handle = (void *)handle;

    /* Set math mode for tensor cores */
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    /* Allocate KV cache on GPU */
    int kv_dim = TTS_DEC_KV_HEADS * TTS_DEC_HEAD_DIM; /* 1024 */
    size_t kv_bytes = (size_t)TTS_DEC_LAYERS * kv_cache_max * kv_dim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&g_cuda.kv_cache_k_gpu, kv_bytes));
    CUDA_CHECK(cudaMalloc(&g_cuda.kv_cache_v_gpu, kv_bytes));
    CUDA_CHECK(cudaMemset(g_cuda.kv_cache_k_gpu, 0, kv_bytes));
    CUDA_CHECK(cudaMemset(g_cuda.kv_cache_v_gpu, 0, kv_bytes));
    g_cuda.kv_cache_max = kv_cache_max;

    fprintf(stderr, "  CUDA KV cache: %d positions, %.1f MB\n",
            kv_cache_max, (float)(2 * kv_bytes) / (1024*1024));

    /* Allocate activation buffers (for max single-token decode) */
    int dim = TTS_DEC_DIM;
    int q_dim = TTS_DEC_HEADS * TTS_DEC_HEAD_DIM;
    int kv_dim_alloc = TTS_DEC_KV_HEADS * TTS_DEC_HEAD_DIM;

    CUDA_CHECK(cudaMalloc(&g_cuda.d_x, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_x_norm, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_q, q_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_k, kv_dim_alloc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_v, kv_dim_alloc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_attn_out, q_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_proj_out, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_gate, TTS_DEC_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_up, TTS_DEC_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_ffn_out, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_hidden, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_embed, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_rope_freqs, TTS_DEC_HEAD_DIM * sizeof(float)));

    /* Acoustic transformer buffers (3 tokens x dim) */
    int ac_seq = 3;
    int ac_q_dim = TTS_AC_HEADS * TTS_AC_HEAD_DIM;
    int ac_kv_dim = TTS_AC_KV_HEADS * TTS_AC_HEAD_DIM;
    CUDA_CHECK(cudaMalloc(&g_cuda.d_ac_tokens, ac_seq * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_ac_tokens_norm, ac_seq * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_ac_q, ac_seq * ac_q_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_ac_k, ac_seq * ac_kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_ac_v, ac_seq * ac_kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_ac_attn_out, ac_seq * ac_q_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_ac_proj_out, ac_seq * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_ac_gate, ac_seq * TTS_AC_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_ac_up, ac_seq * TTS_AC_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_cuda.d_ac_ffn_out, ac_seq * dim * sizeof(float)));

    g_cuda.initialized = 1;
    return 0;
}

extern "C" void tts_cuda_free(void) {
    if (!g_cuda.initialized) return;

    if (g_cuda.cublas_handle) cublasDestroy((cublasHandle_t)g_cuda.cublas_handle);

    /* Free weight buffers */
    cudaFree(g_cuda.tok_embeddings_gpu);
    cudaFree(g_cuda.dec_norm_gpu);
    for (int i = 0; i < TTS_DEC_LAYERS; i++) {
        tts_cuda_layer_weights_t *l = &g_cuda.dec_layers[i];
        cudaFree(l->wq); cudaFree(l->wk); cudaFree(l->wv); cudaFree(l->wo);
        cudaFree(l->w1); cudaFree(l->w2); cudaFree(l->w3);
        cudaFree(l->attention_norm); cudaFree(l->ffn_norm);
    }
    for (int i = 0; i < TTS_AC_LAYERS; i++) {
        tts_cuda_layer_weights_t *l = &g_cuda.ac_layers[i];
        cudaFree(l->wq); cudaFree(l->wk); cudaFree(l->wv); cudaFree(l->wo);
        cudaFree(l->w1); cudaFree(l->w2); cudaFree(l->w3);
        cudaFree(l->attention_norm); cudaFree(l->ffn_norm);
    }
    cudaFree(g_cuda.ac_input_proj_gpu); cudaFree(g_cuda.ac_time_proj_gpu);
    cudaFree(g_cuda.ac_llm_proj_gpu); cudaFree(g_cuda.ac_semantic_out_gpu);
    cudaFree(g_cuda.ac_acoustic_out_gpu); cudaFree(g_cuda.ac_norm_gpu);
    cudaFree(g_cuda.ac_time_inv_freq_gpu);

    /* Free KV cache */
    cudaFree(g_cuda.kv_cache_k_gpu); cudaFree(g_cuda.kv_cache_v_gpu);

    /* Free activation buffers */
    cudaFree(g_cuda.d_x); cudaFree(g_cuda.d_x_norm);
    cudaFree(g_cuda.d_q); cudaFree(g_cuda.d_k); cudaFree(g_cuda.d_v);
    cudaFree(g_cuda.d_attn_out); cudaFree(g_cuda.d_proj_out);
    cudaFree(g_cuda.d_gate); cudaFree(g_cuda.d_up); cudaFree(g_cuda.d_ffn_out);
    cudaFree(g_cuda.d_hidden); cudaFree(g_cuda.d_embed);
    cudaFree(g_cuda.d_rope_freqs);

    cudaFree(g_cuda.d_ac_tokens); cudaFree(g_cuda.d_ac_tokens_norm);
    cudaFree(g_cuda.d_ac_q); cudaFree(g_cuda.d_ac_k); cudaFree(g_cuda.d_ac_v);
    cudaFree(g_cuda.d_ac_attn_out); cudaFree(g_cuda.d_ac_proj_out);
    cudaFree(g_cuda.d_ac_gate); cudaFree(g_cuda.d_ac_up); cudaFree(g_cuda.d_ac_ffn_out);

    memset(&g_cuda, 0, sizeof(g_cuda));
}

extern "C" int tts_cuda_available(void) {
    return g_cuda.initialized;
}

/* ========================================================================
 * Weight Upload (CPU mmap bf16 -> GPU VRAM bf16)
 * ======================================================================== */

static void *upload_bf16(const uint16_t *host_bf16, size_t n_elements) {
    void *d_ptr = NULL;
    size_t bytes = n_elements * sizeof(uint16_t);
    cudaMalloc(&d_ptr, bytes);
    if (d_ptr) cudaMemcpy(d_ptr, host_bf16, bytes, cudaMemcpyHostToDevice);
    return d_ptr;
}

static float *upload_f32(const float *host_f32, int n_elements) {
    float *d_ptr = NULL;
    size_t bytes = n_elements * sizeof(float);
    cudaMalloc(&d_ptr, bytes);
    if (d_ptr) cudaMemcpy(d_ptr, host_f32, bytes, cudaMemcpyHostToDevice);
    return d_ptr;
}

extern "C" int tts_cuda_upload_llm_weights(void *decoder_ptr) {
    tts_decoder_t *dec = (tts_decoder_t *)decoder_ptr;

    /* Token embeddings */
    g_cuda.tok_embeddings_gpu = upload_bf16(dec->tok_embeddings_bf16,
                                            (size_t)TTS_VOCAB_SIZE * TTS_DEC_DIM);
    if (!g_cuda.tok_embeddings_gpu) return -1;

    /* Final norm */
    g_cuda.dec_norm_gpu = upload_f32(dec->norm, TTS_DEC_DIM);

    /* Per-layer weights */
    for (int i = 0; i < TTS_DEC_LAYERS; i++) {
        tts_dec_layer_t *src = &dec->layers[i];
        tts_cuda_layer_weights_t *dst = &g_cuda.dec_layers[i];

        int q_dim = TTS_DEC_HEADS * TTS_DEC_HEAD_DIM;
        int kv_dim = TTS_DEC_KV_HEADS * TTS_DEC_HEAD_DIM;

        dst->wq = upload_bf16(src->wq_bf16, (size_t)q_dim * TTS_DEC_DIM);
        dst->wk = upload_bf16(src->wk_bf16, (size_t)kv_dim * TTS_DEC_DIM);
        dst->wv = upload_bf16(src->wv_bf16, (size_t)kv_dim * TTS_DEC_DIM);
        dst->wo = upload_bf16(src->wo_bf16, (size_t)TTS_DEC_DIM * q_dim);
        dst->w1 = upload_bf16(src->w1_bf16, (size_t)TTS_DEC_HIDDEN * TTS_DEC_DIM);
        dst->w2 = upload_bf16(src->w2_bf16, (size_t)TTS_DEC_DIM * TTS_DEC_HIDDEN);
        dst->w3 = upload_bf16(src->w3_bf16, (size_t)TTS_DEC_HIDDEN * TTS_DEC_DIM);
        dst->attention_norm = upload_f32(src->attention_norm, TTS_DEC_DIM);
        dst->ffn_norm = upload_f32(src->ffn_norm, TTS_DEC_DIM);

        if (!dst->wq || !dst->wk || !dst->wv || !dst->wo ||
            !dst->w1 || !dst->w2 || !dst->w3) {
            fprintf(stderr, "cuda: failed to upload LLM layer %d\n", i);
            return -1;
        }
    }

    fprintf(stderr, "  CUDA: LLM weights uploaded to GPU\n");
    return 0;
}

extern "C" int tts_cuda_upload_acoustic_weights(void *acoustic_ptr) {
    tts_acoustic_t *ac = (tts_acoustic_t *)acoustic_ptr;
    int dim = TTS_AC_DIM;

    g_cuda.ac_input_proj_gpu = upload_bf16(ac->input_proj_bf16, (size_t)dim * TTS_ACOUSTIC_DIM);
    g_cuda.ac_time_proj_gpu = upload_bf16(ac->time_proj_bf16, (size_t)dim * dim);
    g_cuda.ac_llm_proj_gpu = upload_bf16(ac->llm_proj_bf16, (size_t)dim * dim);
    g_cuda.ac_semantic_out_gpu = upload_bf16(ac->semantic_out_bf16, (size_t)TTS_SEMANTIC_CB_PADDED * dim);
    g_cuda.ac_acoustic_out_gpu = upload_bf16(ac->acoustic_out_bf16, (size_t)TTS_ACOUSTIC_DIM * dim);
    g_cuda.ac_norm_gpu = upload_f32(ac->norm, dim);
    g_cuda.ac_time_inv_freq_gpu = upload_f32(ac->time_inv_freq, dim / 2);

    for (int i = 0; i < TTS_AC_LAYERS; i++) {
        tts_ac_layer_t *src = &ac->layers[i];
        tts_cuda_layer_weights_t *dst = &g_cuda.ac_layers[i];
        int q_dim = TTS_AC_HEADS * TTS_AC_HEAD_DIM;
        int kv_dim = TTS_AC_KV_HEADS * TTS_AC_HEAD_DIM;

        dst->wq = upload_bf16(src->wq_bf16, (size_t)q_dim * dim);
        dst->wk = upload_bf16(src->wk_bf16, (size_t)kv_dim * dim);
        dst->wv = upload_bf16(src->wv_bf16, (size_t)kv_dim * dim);
        dst->wo = upload_bf16(src->wo_bf16, (size_t)dim * q_dim);
        dst->w1 = upload_bf16(src->w1_bf16, (size_t)TTS_AC_HIDDEN * dim);
        dst->w2 = upload_bf16(src->w2_bf16, (size_t)dim * TTS_AC_HIDDEN);
        dst->w3 = upload_bf16(src->w3_bf16, (size_t)TTS_AC_HIDDEN * dim);
        dst->attention_norm = upload_f32(src->attention_norm, dim);
        dst->ffn_norm = upload_f32(src->ffn_norm, dim);
    }

    fprintf(stderr, "  CUDA: Acoustic weights uploaded to GPU\n");
    return 0;
}

/* ========================================================================
 * cuBLAS Linear (bf16 weights x f32 activations -> f32)
 * ======================================================================== */

extern "C" void tts_cuda_linear_bf16(float *y, const float *x,
                                      const uint16_t *W_bf16_host,
                                      int seq_len, int in_dim, int out_dim,
                                      const void *W_gpu) {
    cublasHandle_t handle = (cublasHandle_t)g_cuda.cublas_handle;

    /* Upload x to GPU if needed (check if it's a device pointer) */
    float *d_x_local = NULL;
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, x);
    if (err != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
        cudaMalloc(&d_x_local, (size_t)seq_len * in_dim * sizeof(float));
        cudaMemcpy(d_x_local, x, (size_t)seq_len * in_dim * sizeof(float),
                   cudaMemcpyHostToDevice);
    } else {
        d_x_local = (float *)x;
    }

    /* Upload W if no GPU version provided */
    void *d_W = (void *)W_gpu;
    int free_W = 0;
    if (!d_W) {
        d_W = upload_bf16(W_bf16_host, (size_t)out_dim * in_dim);
        free_W = 1;
    }

    /* Allocate output on GPU */
    float *d_y = NULL;
    cudaPointerGetAttributes(&attr, y);
    int y_on_device = (attr.type == cudaMemoryTypeDevice);
    if (!y_on_device) {
        cudaMalloc(&d_y, (size_t)seq_len * out_dim * sizeof(float));
    } else {
        d_y = y;
    }

    /* cuBLAS GEMM: y = x @ W^T
     * cuBLAS is column-major, so we compute: y^T = W @ x^T
     * which in row-major terms is: y[M,N] = x[M,K] @ W[N,K]^T
     * cuBLAS call: C = alpha * op(A) * op(B) + beta * C
     *   op(A) = W^T (transposed), op(B) = x (not transposed)
     *   M_cublas = out_dim, N_cublas = seq_len, K_cublas = in_dim
     */
    float alpha = 1.0f, beta = 0.0f;

    if (g_cuda.has_bf16) {
        /* Native bf16 GEMM on Ampere+ */
        CUBLAS_CHECK(cublasGemmEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            out_dim, seq_len, in_dim,
            &alpha,
            d_W, CUDA_R_16BF, in_dim,
            d_x_local, CUDA_R_32F, in_dim,
            &beta,
            d_y, CUDA_R_32F, out_dim,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        /* Fallback: convert bf16->fp16 then GEMM */
        /* For simplicity, use f32 compute with bf16 upcast */
        CUBLAS_CHECK(cublasGemmEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            out_dim, seq_len, in_dim,
            &alpha,
            d_W, CUDA_R_16BF, in_dim,
            d_x_local, CUDA_R_32F, in_dim,
            &beta,
            d_y, CUDA_R_32F, out_dim,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT));
    }

    /* Copy output back if needed */
    if (!y_on_device) {
        cudaMemcpy(y, d_y, (size_t)seq_len * out_dim * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_y);
    }

    if (d_x_local != x && d_x_local != NULL) cudaFree(d_x_local);
    if (free_W) cudaFree(d_W);
}

/* ========================================================================
 * RMS Norm Kernel
 * ======================================================================== */

__global__ void rms_norm_kernel(float *out, const float *x, const float *weight,
                                 int dim, float eps) {
    int row = blockIdx.x;
    const float *x_row = x + row * dim;
    float *out_row = out + row * dim;

    /* Shared memory reduction for sum of squares */
    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = x_row[i];
        local_sum += val * val;
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();

    /* Block reduction */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(shared[0] / (float)dim + eps);

    /* Normalize and scale */
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] = x_row[i] * rms_inv * weight[i];
    }
}

extern "C" void tts_cuda_rms_norm(float *d_out, const float *d_x, const float *d_weight,
                                   int seq_len, int dim, float eps) {
    int threads = min(dim, 1024);
    /* Round up to power of 2 for reduction */
    int block = 1;
    while (block < threads) block <<= 1;
    if (block > 1024) block = 1024;

    rms_norm_kernel<<<seq_len, block, block * sizeof(float)>>>(
        d_out, d_x, d_weight, dim, eps);
}

/* ========================================================================
 * RoPE Kernel
 * ======================================================================== */

__global__ void rope_kernel(float *x, int heads, int head_dim, int hidden,
                            int pos, float theta) {
    /* One thread per (head, dim_pair) */
    int h = blockIdx.x;
    int d = threadIdx.x; /* pair index, d < head_dim/2 */

    if (h >= heads || d >= head_dim / 2) return;

    float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    float *vec = x + h * head_dim;
    float x0 = vec[d * 2];
    float x1 = vec[d * 2 + 1];
    vec[d * 2]     = x0 * cos_val - x1 * sin_val;
    vec[d * 2 + 1] = x0 * sin_val + x1 * cos_val;
}

extern "C" void tts_cuda_apply_rope(float *d_x, const float *d_freqs,
                                     int seq, int heads, int head_dim) {
    /* For single-token decode (seq=1): launch heads x (head_dim/2) threads */
    rope_kernel<<<heads, head_dim / 2>>>(
        d_x, heads, head_dim, heads * head_dim, 0, TTS_ROPE_THETA);
}

extern "C" void tts_cuda_compute_rope_freqs(float *d_freqs, int start_pos, int seq,
                                              int dim, float theta) {
    /* TODO: kernel for multi-position RoPE (prefill) */
    (void)d_freqs; (void)start_pos; (void)seq; (void)dim; (void)theta;
}

/* ========================================================================
 * Fused SiLU * mul Kernel (SwiGLU)
 * ======================================================================== */

__global__ void silu_mul_kernel(float *gate, const float *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

extern "C" void tts_cuda_silu_mul(float *d_gate, const float *d_up, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads>>>(d_gate, d_up, n);
}

/* ========================================================================
 * Residual Add Kernel
 * ======================================================================== */

__global__ void add_inplace_kernel(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

extern "C" void tts_cuda_add_inplace(float *d_a, const float *d_b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_inplace_kernel<<<blocks, threads>>>(d_a, d_b, n);
}

/* ========================================================================
 * Causal Attention Kernel (single-token decode)
 *
 * Each thread block handles one attention head.
 * Q: [1, n_heads * head_dim] (single query)
 * K_cache: [layers * max_seq * kv_dim] (we pass the layer's slice)
 * V_cache: same layout
 * ======================================================================== */

__global__ void causal_attention_decode_kernel(
    float *out, const float *Q, const float *K, const float *V,
    int seq_k, int n_kv_heads, int head_dim, float scale,
    int heads_per_kv, int kv_stride) {

    int h = blockIdx.x; /* query head index */
    int kv_h = h / heads_per_kv;
    int tid = threadIdx.x;

    const float *q = Q + h * head_dim;
    float *o = out + h * head_dim;

    extern __shared__ float smem[]; /* [seq_k] for scores */

    /* Compute attention scores: q . k[j] for all j in [0, seq_k) */
    for (int j = tid; j < seq_k; j += blockDim.x) {
        const float *k_j = K + j * (n_kv_heads * head_dim) + kv_h * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q[d] * k_j[d];
        }
        smem[j] = score * scale;
    }
    __syncthreads();

    /* Softmax: find max */
    float max_val = -1e30f;
    for (int j = tid; j < seq_k; j += blockDim.x) {
        if (smem[j] > max_val) max_val = smem[j];
    }
    /* Block-wide max reduction via shared memory */
    __shared__ float smax[32]; /* warp-level */
    float warp_max = max_val;
    for (int offset = 16; offset > 0; offset >>= 1)
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, offset));
    if (tid % 32 == 0) smax[tid / 32] = warp_max;
    __syncthreads();
    if (tid < 32) {
        warp_max = (tid < (blockDim.x + 31) / 32) ? smax[tid] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, offset));
        if (tid == 0) smax[0] = warp_max;
    }
    __syncthreads();
    max_val = smax[0];

    /* Softmax: exp and sum */
    float sum_exp = 0.0f;
    for (int j = tid; j < seq_k; j += blockDim.x) {
        smem[j] = expf(smem[j] - max_val);
        sum_exp += smem[j];
    }
    /* Reduce sum */
    float warp_sum = sum_exp;
    for (int offset = 16; offset > 0; offset >>= 1)
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
    __shared__ float ssum[32];
    if (tid % 32 == 0) ssum[tid / 32] = warp_sum;
    __syncthreads();
    if (tid < 32) {
        warp_sum = (tid < (blockDim.x + 31) / 32) ? ssum[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
        if (tid == 0) ssum[0] = warp_sum;
    }
    __syncthreads();
    float inv_sum = 1.0f / (ssum[0] + 1e-10f);

    /* Normalize */
    for (int j = tid; j < seq_k; j += blockDim.x) {
        smem[j] *= inv_sum;
    }
    __syncthreads();

    /* Weighted sum of values */
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < seq_k; j++) {
            const float *v_j = V + j * (n_kv_heads * head_dim) + kv_h * head_dim;
            acc += smem[j] * v_j[d];
        }
        o[d] = acc;
    }
}

extern "C" void tts_cuda_causal_attention_decode(
    float *d_out, const float *d_Q, const float *d_K_cache, const float *d_V_cache,
    int seq_k, int n_heads, int n_kv_heads, int head_dim, float scale,
    int kv_cache_stride) {

    int heads_per_kv = n_heads / n_kv_heads;
    int threads = min(seq_k, 256);
    if (threads < 32) threads = 32;
    size_t smem = seq_k * sizeof(float);

    causal_attention_decode_kernel<<<n_heads, threads, smem>>>(
        d_out, d_Q, d_K_cache, d_V_cache,
        seq_k, n_kv_heads, head_dim, scale, heads_per_kv, kv_cache_stride);
}

/* ========================================================================
 * Bidirectional Attention Kernel (acoustic transformer, 3 tokens)
 * ======================================================================== */

extern "C" void tts_cuda_bidirectional_attention(
    float *d_out, const float *d_Q, const float *d_K, const float *d_V,
    int seq_len, int n_heads, int n_kv_heads, int head_dim, float scale) {
    /* For 3 tokens, the attention matrix is tiny (3x3).
     * Just use the causal kernel without masking — all positions attend to all. */
    int heads_per_kv = n_heads / n_kv_heads;
    int threads = 32; /* 3 tokens, minimal parallelism */
    size_t smem = seq_len * sizeof(float);

    causal_attention_decode_kernel<<<n_heads, threads, smem>>>(
        d_out, d_Q, d_K, d_V,
        seq_len, n_kv_heads, head_dim, scale, heads_per_kv, 0);
}

/* ========================================================================
 * Token Embedding Lookup (bf16 -> f32)
 * ======================================================================== */

__global__ void embed_token_kernel(float *out, const __nv_bfloat16 *embeddings,
                                    int token_id, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        out[i] = __bfloat162float(embeddings[(size_t)token_id * dim + i]);
    }
}

extern "C" void tts_cuda_embed_token(float *d_out, const void *d_embeddings_bf16,
                                      int token_id, int dim) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    embed_token_kernel<<<blocks, threads>>>(
        d_out, (const __nv_bfloat16 *)d_embeddings_bf16, token_id, dim);
}

/* ========================================================================
 * GPU LLM Forward (single token, all 26 layers on GPU)
 * ======================================================================== */

extern "C" void tts_cuda_llm_forward(float *out_hidden, const float *input_embed,
                                      int pos) {
    int dim = TTS_DEC_DIM;
    int q_dim = TTS_DEC_HEADS * TTS_DEC_HEAD_DIM;
    int kv_dim = TTS_DEC_KV_HEADS * TTS_DEC_HEAD_DIM;
    float scale = 1.0f / sqrtf((float)TTS_DEC_HEAD_DIM);

    /* Upload input embedding to GPU */
    cudaMemcpy(g_cuda.d_x, input_embed, dim * sizeof(float), cudaMemcpyHostToDevice);

    /* Process 26 layers on GPU */
    for (int layer = 0; layer < TTS_DEC_LAYERS; layer++) {
        tts_cuda_layer_weights_t *l = &g_cuda.dec_layers[layer];

        /* === Attention === */
        tts_cuda_rms_norm(g_cuda.d_x_norm, g_cuda.d_x, l->attention_norm,
                          1, dim, TTS_DEC_NORM_EPS);

        /* Q, K, V projections (cuBLAS, all data on GPU) */
        float alpha = 1.0f, beta = 0.0f;
        cublasHandle_t handle = (cublasHandle_t)g_cuda.cublas_handle;

        /* Q = x_norm @ Wq^T */
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            q_dim, 1, dim, &alpha,
            l->wq, CUDA_R_16BF, dim,
            g_cuda.d_x_norm, CUDA_R_32F, dim,
            &beta, g_cuda.d_q, CUDA_R_32F, q_dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        /* K = x_norm @ Wk^T */
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            kv_dim, 1, dim, &alpha,
            l->wk, CUDA_R_16BF, dim,
            g_cuda.d_x_norm, CUDA_R_32F, dim,
            &beta, g_cuda.d_k, CUDA_R_32F, kv_dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        /* V = x_norm @ Wv^T */
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            kv_dim, 1, dim, &alpha,
            l->wv, CUDA_R_16BF, dim,
            g_cuda.d_x_norm, CUDA_R_32F, dim,
            &beta, g_cuda.d_v, CUDA_R_32F, kv_dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        /* Apply RoPE to Q and K */
        rope_kernel<<<TTS_DEC_HEADS, TTS_DEC_HEAD_DIM / 2>>>(
            g_cuda.d_q, TTS_DEC_HEADS, TTS_DEC_HEAD_DIM,
            TTS_DEC_HEADS * TTS_DEC_HEAD_DIM, pos, TTS_ROPE_THETA);
        rope_kernel<<<TTS_DEC_KV_HEADS, TTS_DEC_HEAD_DIM / 2>>>(
            g_cuda.d_k, TTS_DEC_KV_HEADS, TTS_DEC_HEAD_DIM,
            TTS_DEC_KV_HEADS * TTS_DEC_HEAD_DIM, pos, TTS_ROPE_THETA);

        /* Store K, V in GPU KV cache */
        size_t kv_offset = ((size_t)layer * g_cuda.kv_cache_max + pos) * kv_dim;
        cudaMemcpy(g_cuda.kv_cache_k_gpu + kv_offset, g_cuda.d_k,
                   kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(g_cuda.kv_cache_v_gpu + kv_offset, g_cuda.d_v,
                   kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);

        /* Attention over KV cache */
        float *layer_k = g_cuda.kv_cache_k_gpu + (size_t)layer * g_cuda.kv_cache_max * kv_dim;
        float *layer_v = g_cuda.kv_cache_v_gpu + (size_t)layer * g_cuda.kv_cache_max * kv_dim;

        tts_cuda_causal_attention_decode(
            g_cuda.d_attn_out, g_cuda.d_q, layer_k, layer_v,
            pos + 1, TTS_DEC_HEADS, TTS_DEC_KV_HEADS, TTS_DEC_HEAD_DIM,
            scale, kv_dim);

        /* WO projection */
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            dim, 1, q_dim, &alpha,
            l->wo, CUDA_R_16BF, q_dim,
            g_cuda.d_attn_out, CUDA_R_32F, q_dim,
            &beta, g_cuda.d_proj_out, CUDA_R_32F, dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        /* Residual: x += proj_out */
        tts_cuda_add_inplace(g_cuda.d_x, g_cuda.d_proj_out, dim);

        /* === FFN === */
        tts_cuda_rms_norm(g_cuda.d_x_norm, g_cuda.d_x, l->ffn_norm,
                          1, dim, TTS_DEC_NORM_EPS);

        /* w1 (gate) and w3 (up) */
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            TTS_DEC_HIDDEN, 1, dim, &alpha,
            l->w1, CUDA_R_16BF, dim,
            g_cuda.d_x_norm, CUDA_R_32F, dim,
            &beta, g_cuda.d_gate, CUDA_R_32F, TTS_DEC_HIDDEN,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            TTS_DEC_HIDDEN, 1, dim, &alpha,
            l->w3, CUDA_R_16BF, dim,
            g_cuda.d_x_norm, CUDA_R_32F, dim,
            &beta, g_cuda.d_up, CUDA_R_32F, TTS_DEC_HIDDEN,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        /* Fused SiLU(gate) * up */
        tts_cuda_silu_mul(g_cuda.d_gate, g_cuda.d_up, TTS_DEC_HIDDEN);

        /* w2 (down) */
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            dim, 1, TTS_DEC_HIDDEN, &alpha,
            l->w2, CUDA_R_16BF, TTS_DEC_HIDDEN,
            g_cuda.d_gate, CUDA_R_32F, TTS_DEC_HIDDEN,
            &beta, g_cuda.d_ffn_out, CUDA_R_32F, dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        /* Residual: x += ffn_out */
        tts_cuda_add_inplace(g_cuda.d_x, g_cuda.d_ffn_out, dim);
    }

    /* Final norm */
    tts_cuda_rms_norm(g_cuda.d_hidden, g_cuda.d_x, g_cuda.dec_norm_gpu,
                      1, dim, TTS_DEC_NORM_EPS);

    /* Download hidden state back to CPU */
    cudaMemcpy(out_hidden, g_cuda.d_hidden, dim * sizeof(float),
               cudaMemcpyDeviceToHost);
}

/* ========================================================================
 * GPU LLM Prefill (multi-token)
 * ======================================================================== */

extern "C" void tts_cuda_llm_prefill(const float *embeds, int seq_len, int start_pos) {
    /* TODO: Implement GPU prefill with batched GEMM.
     * For now, fall back to CPU prefill (still fast for ~225 tokens).
     * The main bottleneck is per-token decode, not prefill. */
    (void)embeds; (void)seq_len; (void)start_pos;
    fprintf(stderr, "  cuda: prefill not yet GPU-accelerated, using CPU\n");
}

/* ========================================================================
 * GPU Acoustic Transformer Velocity Prediction
 * ======================================================================== */

extern "C" void tts_cuda_predict_velocity(float *out_velocity,
                                            const float *x_t,
                                            const float *llm_hidden,
                                            float t_val) {
    /* TODO: Implement GPU acoustic transformer forward.
     * The acoustic transformer is small (3 tokens, 3 layers) so
     * the GPU overhead per-call may not be worth it for the
     * current 14 calls per frame. But for batch inference it helps. */
    (void)out_velocity; (void)x_t; (void)llm_hidden; (void)t_val;
}
