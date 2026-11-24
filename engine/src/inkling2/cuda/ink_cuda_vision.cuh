// engine/src/inkling2/cuda/ink_cuda_vision.cuh - the Inkling-2-Mini picture reader (23-layer ViT) on an NVIDIA card.
//
// The CPU reference is ink_vision.h CpuVisionEncoder; HIP, Metal and SYCL had GPU readers, CUDA did not, so on NVIDIA every
// picture was read on the CPU inside the engine's single request slot (kingdonb fork, chudmax 2026-11-23: one 2,000x1,088
// screenshot kept the slot for more than 468 s with the card idle). This file is a port of the HIP reader
// (hip/ink_hip_backend.hip.cpp encode_image): the same patchify, position and RoPE tables on the host, the same 23 layers,
// the same 2x2 merge projector, and the same output layout as CpuVisionEncoder::encode.
//
// It must never take the engine down. The card it was written for runs an elastic KV cache that grows to fill VRAM, so:
//   * the weights live in host RAM (bf16 for the large matrices, f32 for the rest, read once) and are uploaded for each
//     picture, then freed with everything else; nothing stays on the card between pictures;
//   * free card memory is checked before anything is allocated, and every CUDA and cuBLAS call is checked: any failure frees
//     what this picture allocated, clears the error and returns false, and the caller reads the picture on the CPU. It
//     never calls INK_CK, whose failure ends the process.
//
// Linear layers: cuBLAS GEMM, bf16 weights x bf16 activations, f32 accumulate (as the HIP reader). Attention: per head,
// scores = Q K^T over all S tokens in query chunks sized to a memory budget, a row softmax, then P V (f32 GEMMs).
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "ink_manifest.h"   // ink_vision.h uses its tensor formats (F_BF16, ...)
#include "ink_vision.h"

namespace strata::ink {
namespace vis {

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// f32 -> bf16 bits, round to nearest even (the same rounding the host loader uses for f32 / f16 weights)
__device__ __forceinline__ uint16_t to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return static_cast<uint16_t>((u >> 16) | ((u & 0xffffu) ? 0x40u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

__global__ void k_f32_to_bf16(const float* __restrict__ x, uint16_t* __restrict__ y, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = to_bf16(x[i]);
}

__global__ void k_add_bias_rows(float* __restrict__ y, const float* __restrict__ b, int M, int N) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(M) * N) y[i] += b[i % N];
}

__global__ void k_vec_add(float* __restrict__ y, const float* __restrict__ a, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a[i];
}

__global__ void k_gelu_exact(float* __restrict__ x, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
    }
}

// cur[t, d] += gamma[d] * proj[t, d]
__global__ void k_layerscale_res(float* __restrict__ cur, const float* __restrict__ proj, const float* __restrict__ gamma, int S, int D) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(S) * D) cur[i] += gamma[i % D] * proj[i];
}

// one block (256 threads) per row: out = (x - mean) / sqrt(var + eps) * w + b
__global__ void __launch_bounds__(256) k_layernorm(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                                   float* __restrict__ out, int N, int C, float eps) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float* xr = x + static_cast<size_t>(row) * C;
    float* yr = out + static_cast<size_t>(row) * C;
    __shared__ float s_w[8];
    __shared__ float s_mean, s_inv;
    float s = 0.0f;
    for (int c = tid; c < C; c += 256) s += xr[c];
    s = warp_sum(s);
    if (lane == 0) s_w[warp] = s;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t += s_w[i];
        s_mean = t / static_cast<float>(C);
    }
    __syncthreads();
    const float mean = s_mean;
    float v = 0.0f;
    for (int c = tid; c < C; c += 256) {
        const float d = xr[c] - mean;
        v = fmaf(d, d, v);
    }
    v = warp_sum(v);
    if (lane == 0) s_w[warp] = v;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t += s_w[i];
        s_inv = rsqrtf(t / static_cast<float>(C) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (int c = tid; c < C; c += 256) yr[c] = fmaf((xr[c] - mean) * inv, w[c], b ? b[c] : 0.0f);
}

// pairwise axial 2D RoPE on Q and K, [S, H, hd]; grid (H, S), block hd/2
__global__ void k_rope_2d(float* __restrict__ Q, float* __restrict__ K, const float* __restrict__ rc, const float* __restrict__ rs,
                          int S, int H, int hd) {
    const int pair = threadIdx.x, h = blockIdx.x, t = blockIdx.y;
    if (pair * 2 + 1 >= hd || h >= H || t >= S) return;
    const int j = pair * 2;
    const float c0 = rc[static_cast<size_t>(t) * hd + j], c1 = rc[static_cast<size_t>(t) * hd + j + 1];
    const float s0 = rs[static_cast<size_t>(t) * hd + j], s1 = rs[static_cast<size_t>(t) * hd + j + 1];
    float* qh = Q + (static_cast<size_t>(t) * H + h) * hd;
    float* kh = K + (static_cast<size_t>(t) * H + h) * hd;
    const float q0 = qh[j], q1 = qh[j + 1];
    qh[j] = q0 * c0 - q1 * s0;
    qh[j + 1] = q1 * c1 + q0 * s1;
    const float k0 = kh[j], k1 = kh[j + 1];
    kh[j] = k0 * c0 - k1 * s0;
    kh[j + 1] = k1 * c1 + k0 * s1;
}

// softmax in place over each row of length `cols` (one block of 256 per row)
__global__ void __launch_bounds__(256) k_softmax_rows(float* __restrict__ x, int cols) {
    float* r = x + static_cast<size_t>(blockIdx.x) * cols;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    __shared__ float s_w[8];
    __shared__ float s_v;
    float m = -INFINITY;
    for (int c = tid; c < cols; c += 256) m = fmaxf(m, r[c]);
    m = warp_max(m);
    if (lane == 0) s_w[warp] = m;
    __syncthreads();
    if (tid == 0) {
        float t = s_w[0];
        for (int i = 1; i < 8; ++i) t = fmaxf(t, s_w[i]);
        s_v = t;
    }
    __syncthreads();
    const float mx = s_v;
    float s = 0.0f;
    for (int c = tid; c < cols; c += 256) {
        const float e = __expf(r[c] - mx);
        r[c] = e;
        s += e;
    }
    s = warp_sum(s);
    __syncthreads();
    if (lane == 0) s_w[warp] = s;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t += s_w[i];
        s_v = 1.0f / fmaxf(t, 1e-30f);
    }
    __syncthreads();
    const float inv = s_v;
    for (int c = tid; c < cols; c += 256) r[c] *= inv;
}

inline unsigned blocks_for(size_t n, unsigned threads = 256) { return static_cast<unsigned>((n + threads - 1) / threads); }

// host f32 -> bf16 bits, round to nearest even
inline uint16_t host_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) return static_cast<uint16_t>((u >> 16) | ((u & 0xffffu) ? 0x40u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

}  // namespace vis

class InkCudaVision {
public:
    // Attention scores of one query chunk: H x chunk x S floats. The chunk is sized to stay within this many bytes.
    size_t attn_budget_bytes = 128ull << 20;
    // Room left free on the card beyond what one picture needs (the KV cache and the next prompt need it more).
    size_t vram_margin_bytes = 512ull << 20;

    // Reads `img` on the card. false (with `err`) when it cannot, for any reason; the caller then reads it on the CPU.
    bool encode(const std::string& vision_path, const ink2::ImageRec& img, cublasHandle_t cb, cudaStream_t st,
                std::vector<float>& out, std::string& err) {
        if (!cb) { err = "no cuBLAS handle"; return false; }
        if (!host_loaded_ || path_ != vision_path) {
            if (!load_host(vision_path, err)) return false;
        }
        Run r(*this, cb, st);
        const bool ok = r.go(img, out, err);
        r.release();
        if (!ok) {
            (void)cudaGetLastError();   // a failed allocation or launch is not sticky: leave no error for the engine's next check
            out.clear();
        }
        return ok;
    }

    // Bytes the weights take on the card while a picture is read.
    size_t weight_bytes() const { return weight_bytes_; }

private:
    struct HostTensor {
        bool bf16 = false;
        std::vector<uint16_t> b;   // bf16 bits
        std::vector<float> f;
        size_t off = 0;            // byte offset in the device weight block
        size_t bytes() const { return bf16 ? b.size() * 2 : f.size() * 4; }
    };

    bool load_host(const std::string& path, std::string& err) {
        host_loaded_ = false;
        T_.clear();
        h_cls_.clear(); h_cls_pos_.clear(); h_pos_emb_.clear();
        if (!ink2::vision_probe(path, info_, err)) return false;
        std::vector<uint8_t> s1;
        if (!info_.is_safetensors) {
            strata::artifact::StrataFileReader rd;
            if (!rd.open(path, err)) return false;
            if (!rd.read_section(1, s1, err)) return false;
        }
        size_t off = 0;
        for (const auto& kv : info_.T) {
            const auto& vt = kv.second;
            size_t n = 1;
            for (int64_t d : vt.shape) n *= static_cast<size_t>(d);
            std::vector<uint8_t> raw;
            const uint8_t* src = nullptr;
            if (info_.is_safetensors) {
                raw.resize(static_cast<size_t>(vt.bytes));
                if (!ink2::detail::read_slice_from_file(vt.shard_path, vt.off, raw.data(), static_cast<size_t>(vt.bytes), err)) return false;
                src = raw.data();
            } else {
                src = s1.data() + vt.off;
            }
            const std::string& name = kv.first;
            const bool host_only = name == "model.visual.backbone.embeddings.class_embedding" ||
                                   name == "model.visual.backbone.embeddings.cls_position_embedding" ||
                                   name == "model.visual.backbone.embeddings.position_embedding";
            // the HIP reader's rule: the patch embedding and every large 2-D matrix as bf16, the rest as f32
            const bool big = name == "model.visual.backbone.embeddings.patch_embedding.weight" ||
                             (vt.shape.size() >= 2 && vt.shape[0] >= 64 && (n / static_cast<size_t>(vt.shape[0])) >= 64);
            HostTensor t;
            if (big && !host_only) {
                t.bf16 = true;
                t.b.resize(n);
                if (vt.fmt == ink2::F_BF16) {
                    std::memcpy(t.b.data(), src, n * 2);          // already bf16: exactly the values the CPU reader decodes
                } else if (vt.fmt == ink2::F_F16) {
                    const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
                    for (size_t i = 0; i < n; ++i) t.b[i] = vis::host_to_bf16(ink2::fp16_to_f32(p[i]));
                } else {
                    const float* p = reinterpret_cast<const float*>(src);
                    for (size_t i = 0; i < n; ++i) t.b[i] = vis::host_to_bf16(p[i]);
                }
            } else {
                t.f.resize(n);
                if (vt.fmt == ink2::F_BF16) {
                    const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
                    for (size_t i = 0; i < n; ++i) t.f[i] = ink2::bf16_to_f32(p[i]);
                } else if (vt.fmt == ink2::F_F16) {
                    const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
                    for (size_t i = 0; i < n; ++i) t.f[i] = ink2::fp16_to_f32(p[i]);
                } else {
                    std::memcpy(t.f.data(), src, n * 4);
                }
            }
            if (name == "model.visual.backbone.embeddings.class_embedding") { h_cls_ = t.f; continue; }
            if (name == "model.visual.backbone.embeddings.cls_position_embedding") { h_cls_pos_ = t.f; continue; }
            if (name == "model.visual.backbone.embeddings.position_embedding") { h_pos_emb_ = t.f; continue; }
            t.off = off;
            off += (t.bytes() + 255) & ~static_cast<size_t>(255);
            T_[name] = std::move(t);
        }
        weight_bytes_ = off;
        if (h_cls_.empty() || h_cls_pos_.empty() || h_pos_emb_.empty()) { err = "vision shard lacks the embedding tables"; return false; }
        path_ = path;
        host_loaded_ = true;
        return true;
    }

    // One picture's device state; release() frees all of it.
    struct Run {
        InkCudaVision& V;
        cublasHandle_t cb;
        cudaStream_t st;
        std::vector<void*> allocs;
        uint8_t* w = nullptr;   // the weight block

        Run(InkCudaVision& v, cublasHandle_t c, cudaStream_t s) : V(v), cb(c), st(s) {}

        void release() {
            if (st) (void)cudaStreamSynchronize(st);
            for (void* p : allocs) (void)cudaFree(p);
            allocs.clear();
        }

        bool cuda_ok(cudaError_t e, const char* what, std::string& err) {
            if (e == cudaSuccess) return true;
            err = std::string("CUDA picture reader: ") + what + ": " + cudaGetErrorString(e);
            return false;
        }
        bool blas_ok(cublasStatus_t s, const char* what, std::string& err) {
            if (s == CUBLAS_STATUS_SUCCESS) return true;
            err = std::string("CUDA picture reader: ") + what + ": cuBLAS status " + std::to_string(static_cast<int>(s));
            return false;
        }
        template <typename P>
        bool alloc(P*& p, size_t bytes, const char* what, std::string& err) {
            void* raw = nullptr;
            if (!cuda_ok(cudaMalloc(&raw, std::max<size_t>(bytes, 256)), what, err)) return false;
            allocs.push_back(raw);
            p = reinterpret_cast<P*>(raw);
            return true;
        }
        bool launched(const char* what, std::string& err) { return cuda_ok(cudaGetLastError(), what, err); }

        const uint16_t* wb(const std::string& k) const {
            auto it = V.T_.find(k);
            return (it == V.T_.end() || !it->second.bf16) ? nullptr : reinterpret_cast<const uint16_t*>(w + it->second.off);
        }
        const float* wf(const std::string& k) const {
            auto it = V.T_.find(k);
            return (it == V.T_.end() || it->second.bf16) ? nullptr : reinterpret_cast<const float*>(w + it->second.off);
        }

        uint16_t* xb = nullptr;   // bf16 copy of a linear layer's input

        // Y[M, N] = X[M, K] W[N, K]^T (+ b)
        bool linear(const uint16_t* W, const float* X, const float* b, float* Y, int M, int N, int K, const char* what, std::string& err) {
            if (!W) { err = std::string("CUDA picture reader: missing weight for ") + what; return false; }
            const size_t nx = static_cast<size_t>(M) * K;
            vis::k_f32_to_bf16<<<vis::blocks_for(nx), 256, 0, st>>>(X, xb, nx);
            if (!launched(what, err)) return false;
            const float one = 1.0f, zero = 0.0f;
            if (!blas_ok(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, W, CUDA_R_16BF, K, xb, CUDA_R_16BF, K,
                                      &zero, Y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), what, err)) return false;
            if (b) {
                vis::k_add_bias_rows<<<vis::blocks_for(static_cast<size_t>(M) * N), 256, 0, st>>>(Y, b, M, N);
                if (!launched(what, err)) return false;
            }
            return true;
        }

        bool go(const ink2::ImageRec& img, std::vector<float>& out, std::string& err) {
            const ink2::VitCfg& vc = V.info_.c;
            const int merge = vc.merge, patch = vc.patch, cell = patch * merge;
            const int n_tok = img.num_lm_tokens();
            const int ph = img.grid_h * merge, pw = img.grid_w * merge;
            const int n_patches = ph * pw, S = n_patches + 1;
            const int D = vc.hidden, H = vc.heads, hd = vc.head_dim, I = vc.inter;
            const int patch_in = 3 * patch * patch;
            const int merged_dim = merge * merge * D;
            if (n_tok <= 0 || img.pixel_values.size() < static_cast<size_t>(n_tok) * cell * cell * 3) { err = "CUDA picture reader: empty picture"; return false; }

            // the query chunk of the attention, and what one picture needs on the card
            const size_t row_bytes = static_cast<size_t>(S) * H * 4;
            const int chunk = static_cast<int>(std::max<size_t>(16, std::min<size_t>(static_cast<size_t>(S), V.attn_budget_bytes / std::max<size_t>(row_bytes, 1))));
            const size_t SD = static_cast<size_t>(S) * D;
            const size_t xb_elems = std::max({static_cast<size_t>(S) * std::max({D, I, patch_in}), static_cast<size_t>(n_tok) * std::max(merged_dim, vc.proj_hidden)});
            const size_t need = V.weight_bytes_ + 7 * SD * 4 + static_cast<size_t>(S) * I * 4 + xb_elems * 2 +
                                static_cast<size_t>(chunk) * row_bytes + static_cast<size_t>(n_patches) * patch_in * 4 +
                                2 * static_cast<size_t>(S) * hd * 4 + SD * 4 +
                                static_cast<size_t>(n_tok) * (vc.proj_hidden + vc.out_hidden) * 4;
            size_t vf = 0, vt = 0;
            if (!cuda_ok(cudaMemGetInfo(&vf, &vt), "cudaMemGetInfo", err)) return false;
            if (vf < need + V.vram_margin_bytes) {
                err = "CUDA picture reader: not enough free card memory (" + std::to_string(vf >> 20) + " MiB free, " +
                      std::to_string((need + V.vram_margin_bytes) >> 20) + " MiB needed)";
                return false;
            }

            // 1. patchify, position embedding and RoPE tables on the host (as CpuVisionEncoder::encode)
            std::vector<float> patch_buf(static_cast<size_t>(n_patches) * patch_in);
            std::vector<int> pos_h(S, 0), pos_w(S, 0);
            int pi = 0;
            for (int gr = 0; gr < img.grid_h; ++gr) {
                for (int gc = 0; gc < img.grid_w; ++gc) {
                    const size_t tok = static_cast<size_t>(gr) * img.grid_w + gc;
                    const float* cell_px = img.pixel_values.data() + tok * cell * cell * 3;
                    for (int mr = 0; mr < merge; ++mr) {
                        for (int mc = 0; mc < merge; ++mc) {
                            float* dst = patch_buf.data() + static_cast<size_t>(pi) * patch_in;
                            int idx = 0;
                            for (int py = 0; py < patch; ++py) {
                                const int ly = mr * patch + py;
                                for (int px = 0; px < patch; ++px) {
                                    const int lx = mc * patch + px;
                                    const float* rgb = cell_px + (static_cast<size_t>(ly) * cell + lx) * 3;
                                    dst[idx++] = rgb[0]; dst[idx++] = rgb[1]; dst[idx++] = rgb[2];
                                }
                            }
                            pos_h[pi + 1] = gr * merge + mr + 1;
                            pos_w[pi + 1] = gc * merge + mc + 1;
                            ++pi;
                        }
                    }
                }
            }
            std::vector<float> h_pos(SD, 0.0f);
            for (int d = 0; d < D; ++d) h_pos[d] = V.h_cls_[d] + V.h_cls_pos_[d];
            const int G = vc.posemb_grid;
            for (int p = 0; p < n_patches; ++p) {
                const float y = (static_cast<float>(pos_h[p + 1] - 1) + 0.5f) * (static_cast<float>(G) / ph) - 0.5f;
                const float x = (static_cast<float>(pos_w[p + 1] - 1) + 0.5f) * (static_cast<float>(G) / pw) - 0.5f;
                const int y0 = std::max(0, std::min(G - 1, static_cast<int>(std::floor(y))));
                const int y1 = std::min(G - 1, y0 + 1);
                const int x0 = std::max(0, std::min(G - 1, static_cast<int>(std::floor(x))));
                const int x1 = std::min(G - 1, x0 + 1);
                const float wy = std::max(0.0f, std::min(1.0f, y - y0));
                const float wx = std::max(0.0f, std::min(1.0f, x - x0));
                const float w00 = (1.0f - wy) * (1.0f - wx), w01 = (1.0f - wy) * wx, w10 = wy * (1.0f - wx), w11 = wy * wx;
                const float* p00 = V.h_pos_emb_.data() + (static_cast<size_t>(y0) * G + x0) * D;
                const float* p01 = V.h_pos_emb_.data() + (static_cast<size_t>(y0) * G + x1) * D;
                const float* p10 = V.h_pos_emb_.data() + (static_cast<size_t>(y1) * G + x0) * D;
                const float* p11 = V.h_pos_emb_.data() + (static_cast<size_t>(y1) * G + x1) * D;
                float* dst = h_pos.data() + static_cast<size_t>(p + 1) * D;
                for (int d = 0; d < D; ++d) dst[d] = w00 * p00[d] + w01 * p01[d] + w10 * p10[d] + w11 * p11[d];
            }
            const int sp_dim = hd / 2, n_freq = sp_dim / 2;
            std::vector<float> inv_freq(n_freq);
            for (int i = 0; i < n_freq; ++i) inv_freq[i] = 1.0f / std::pow(vc.rope_theta, static_cast<float>(2 * i) / sp_dim);
            std::vector<float> h_rc(static_cast<size_t>(S) * hd), h_rs(static_cast<size_t>(S) * hd);
            for (int t = 0; t < S; ++t) {
                for (int i = 0; i < n_freq; ++i) {
                    const float aw = static_cast<float>(pos_w[t]) * inv_freq[i];
                    const float ah = static_cast<float>(pos_h[t]) * inv_freq[i];
                    h_rc[t * hd + 2 * i] = h_rc[t * hd + 2 * i + 1] = std::cos(aw);
                    h_rs[t * hd + 2 * i] = h_rs[t * hd + 2 * i + 1] = std::sin(aw);
                    h_rc[t * hd + sp_dim + 2 * i] = h_rc[t * hd + sp_dim + 2 * i + 1] = std::cos(ah);
                    h_rs[t * hd + sp_dim + 2 * i] = h_rs[t * hd + sp_dim + 2 * i + 1] = std::sin(ah);
                }
            }

            // 2. the card: weights, activations, scratch
            if (!alloc(w, V.weight_bytes_, "weights", err)) return false;
            for (const auto& kv : V.T_) {
                const HostTensor& t = kv.second;
                const void* src = t.bf16 ? static_cast<const void*>(t.b.data()) : static_cast<const void*>(t.f.data());
                if (!cuda_ok(cudaMemcpyAsync(w + t.off, src, t.bytes(), cudaMemcpyHostToDevice, st), "weight upload", err)) return false;
            }
            float *d_patch, *d_pos, *d_rc, *d_rs, *cur, *nrm, *q, *k, *v, *ao, *proj, *mlp, *sc, *mid, *d_out;
            if (!alloc(d_patch, patch_buf.size() * 4, "patches", err) || !alloc(d_pos, SD * 4, "positions", err) ||
                !alloc(d_rc, h_rc.size() * 4, "rope cos", err) || !alloc(d_rs, h_rs.size() * 4, "rope sin", err) ||
                !alloc(cur, SD * 4, "hidden", err) || !alloc(nrm, SD * 4, "norm", err) || !alloc(q, SD * 4, "q", err) ||
                !alloc(k, SD * 4, "k", err) || !alloc(v, SD * 4, "v", err) || !alloc(ao, SD * 4, "attn out", err) ||
                !alloc(proj, SD * 4, "proj", err) || !alloc(mlp, static_cast<size_t>(S) * I * 4, "mlp", err) ||
                !alloc(sc, static_cast<size_t>(chunk) * row_bytes, "scores", err) || !alloc(xb, xb_elems * 2, "bf16 input", err) ||
                !alloc(mid, static_cast<size_t>(n_tok) * vc.proj_hidden * 4, "merge mid", err) ||
                !alloc(d_out, static_cast<size_t>(n_tok) * vc.out_hidden * 4, "embeddings", err)) return false;
            if (!cuda_ok(cudaMemcpyAsync(d_patch, patch_buf.data(), patch_buf.size() * 4, cudaMemcpyHostToDevice, st), "patch upload", err) ||
                !cuda_ok(cudaMemcpyAsync(d_pos, h_pos.data(), SD * 4, cudaMemcpyHostToDevice, st), "position upload", err) ||
                !cuda_ok(cudaMemcpyAsync(d_rc, h_rc.data(), h_rc.size() * 4, cudaMemcpyHostToDevice, st), "rope upload", err) ||
                !cuda_ok(cudaMemcpyAsync(d_rs, h_rs.data(), h_rs.size() * 4, cudaMemcpyHostToDevice, st), "rope upload", err)) return false;
            if (!blas_ok(cublasSetStream(cb, st), "cublasSetStream", err)) return false;

            // 3. embeddings: CLS row 0 = cls + cls_pos (from d_pos), patches = patch_embedding(patches) + positions, then ln_pre
            if (!cuda_ok(cudaMemsetAsync(cur, 0, static_cast<size_t>(D) * 4, st), "memset", err)) return false;
            if (!linear(wb("model.visual.backbone.embeddings.patch_embedding.weight"), d_patch, nullptr, cur + D, n_patches, D, patch_in, "patch embedding", err)) return false;
            vis::k_vec_add<<<vis::blocks_for(SD), 256, 0, st>>>(cur, d_pos, SD);
            if (!launched("positions", err)) return false;
            vis::k_layernorm<<<S, 256, 0, st>>>(cur, wf("model.visual.backbone.ln_pre.weight"), wf("model.visual.backbone.ln_pre.bias"), cur, S, D, vc.ln_eps);
            if (!launched("ln_pre", err)) return false;

            // 4. the 23 layers
            const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
            const float one = 1.0f, zero = 0.0f;
            const int ld = H * hd;
            for (int il = 0; il < vc.layers; ++il) {
                const std::string p = "model.visual.backbone.layers." + std::to_string(il);
                vis::k_layernorm<<<S, 256, 0, st>>>(cur, wf(p + ".ln_1.weight"), wf(p + ".ln_1.bias"), nrm, S, D, vc.ln_eps);
                if (!launched("ln_1", err)) return false;
                if (!linear(wb(p + ".self_attn.q_proj.weight"), nrm, wf(p + ".self_attn.q_proj.bias"), q, S, D, D, "q_proj", err) ||
                    !linear(wb(p + ".self_attn.k_proj.weight"), nrm, wf(p + ".self_attn.k_proj.bias"), k, S, D, D, "k_proj", err) ||
                    !linear(wb(p + ".self_attn.v_proj.weight"), nrm, wf(p + ".self_attn.v_proj.bias"), v, S, D, D, "v_proj", err)) return false;
                vis::k_rope_2d<<<dim3(H, S), hd / 2, 0, st>>>(q, k, d_rc, d_rs, S, H, hd);
                if (!launched("rope", err)) return false;
                // attention, query chunk by chunk. Per head h (batch), column-major views with leading dimension H*hd:
                //   scores(key, c) = scale * sum_j K(key, j) Q(c, j)      [S x chunk, ld S]
                //   O(j, c)        = sum_key V(key, j) P(key, c)          [hd x chunk, ld H*hd]
                for (int q0 = 0; q0 < S; q0 += chunk) {
                    const int C = std::min(chunk, S - q0);
                    if (!blas_ok(cublasSgemmStridedBatched(cb, CUBLAS_OP_T, CUBLAS_OP_N, S, C, hd, &scale,
                                                           k, ld, hd, q + static_cast<size_t>(q0) * ld, ld, hd,
                                                           &zero, sc, S, static_cast<long long>(C) * S, H), "attention scores", err)) return false;
                    vis::k_softmax_rows<<<H * C, 256, 0, st>>>(sc, S);
                    if (!launched("softmax", err)) return false;
                    if (!blas_ok(cublasSgemmStridedBatched(cb, CUBLAS_OP_N, CUBLAS_OP_N, hd, C, S, &one,
                                                           v, ld, hd, sc, S, static_cast<long long>(C) * S,
                                                           &zero, ao + static_cast<size_t>(q0) * ld, ld, hd, H), "attention values", err)) return false;
                }
                if (!linear(wb(p + ".self_attn.out_proj.weight"), ao, wf(p + ".self_attn.out_proj.bias"), proj, S, D, D, "out_proj", err)) return false;
                vis::k_layerscale_res<<<vis::blocks_for(SD), 256, 0, st>>>(cur, proj, wf(p + ".ls_1.gamma"), S, D);
                if (!launched("ls_1", err)) return false;
                vis::k_layernorm<<<S, 256, 0, st>>>(cur, wf(p + ".ln_2.weight"), wf(p + ".ln_2.bias"), nrm, S, D, vc.ln_eps);
                if (!launched("ln_2", err)) return false;
                if (!linear(wb(p + ".mlp.c_fc.weight"), nrm, wf(p + ".mlp.c_fc.bias"), mlp, S, I, D, "c_fc", err)) return false;
                vis::k_gelu_exact<<<vis::blocks_for(static_cast<size_t>(S) * I), 256, 0, st>>>(mlp, static_cast<size_t>(S) * I);
                if (!launched("gelu", err)) return false;
                if (!linear(wb(p + ".mlp.c_proj.weight"), mlp, wf(p + ".mlp.c_proj.bias"), proj, S, D, I, "c_proj", err)) return false;
                vis::k_layerscale_res<<<vis::blocks_for(SD), 256, 0, st>>>(cur, proj, wf(p + ".ls_2.gamma"), S, D);
                if (!launched("ls_2", err)) return false;
            }

            // 5. drop CLS, 2x2 merge (rows are already [n_tok, 4 x D] in raster order) -> proj_in -> GELU -> proj_out
            if (!linear(wb("model.visual.proj_in.weight"), cur + D, nullptr, mid, n_tok, vc.proj_hidden, merged_dim, "proj_in", err)) return false;
            vis::k_gelu_exact<<<vis::blocks_for(static_cast<size_t>(n_tok) * vc.proj_hidden), 256, 0, st>>>(mid, static_cast<size_t>(n_tok) * vc.proj_hidden);
            if (!launched("gelu", err)) return false;
            if (!linear(wb("model.visual.proj_out.weight"), mid, nullptr, d_out, n_tok, vc.out_hidden, vc.proj_hidden, "proj_out", err)) return false;

            out.resize(static_cast<size_t>(n_tok) * vc.out_hidden);
            return cuda_ok(cudaMemcpyAsync(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost, st), "download", err) &&
                   cuda_ok(cudaStreamSynchronize(st), "sync", err);
        }
    };

    ink2::VisionShardInfo info_;
    std::unordered_map<std::string, HostTensor> T_;
    std::vector<float> h_cls_, h_cls_pos_, h_pos_emb_;
    std::string path_;
    bool host_loaded_ = false;
    size_t weight_bytes_ = 0;
};

}  // namespace strata::ink
