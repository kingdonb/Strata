// engine/src/inkling2/cuda/ink_vision_check.cu - strata-inkling-vision-check: one picture through the CPU reference reader
// (ink_vision.h CpuVisionEncoder) and the CUDA reader (ink_cuda_vision.cuh); prints both times and how far apart the
// embeddings are, as one JSON line. Needs no model, only the picture reader file, and about 1-2 GB of the card.
//
//   strata-inkling-vision-check --vision Inkling-2-Mini-Vision.strata --image shot.png [--max-tokens N] [--no-cpu] [--no-gpu]
//
// --max-tokens N scales the picture down to at most N merged tokens first (the serve loop's --vision-cpu-max-tokens rule).
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ink_manifest.h"   // ink_vision.h uses its tensor formats (F_BF16, ...)
#include "ink_vision.h"
#include "ink_cuda_vision.cuh"

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool prepare_capped(const std::string& path, const ink2::VitCfg& base, int cap, ink2::ImageRec& rec, std::string& err) {
    if (!ink2::prepare_picture(path, base, rec, err)) return false;
    if (cap <= 0 || rec.num_lm_tokens() <= cap) return true;
    const int64_t cell = static_cast<int64_t>(base.patch) * base.merge;
    int64_t px = static_cast<int64_t>(cap) * cell * cell;
    for (int tries = 0; tries < 8; ++tries) {
        ink2::VitCfg vc = base;
        vc.max_pixels = std::max<int64_t>(cell * cell, px);
        ink2::ImageRec small;
        if (!ink2::prepare_picture(path, vc, small, err)) return false;
        if (small.num_lm_tokens() <= cap || tries == 7) { rec = std::move(small); return true; }
        px = px * 85 / 100;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string vision, image;
    int cap = 0;
    bool run_cpu = true, run_gpu = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (a == "--vision") vision = next();
        else if (a == "--image") image = next();
        else if (a == "--max-tokens") cap = std::atoi(next().c_str());
        else if (a == "--no-cpu") run_cpu = false;
        else if (a == "--no-gpu") run_gpu = false;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (vision.empty() || image.empty()) {
        std::fprintf(stderr, "usage: strata-inkling-vision-check --vision FILE --image FILE [--max-tokens N] [--no-cpu] [--no-gpu]\n");
        return 2;
    }
    std::string err;
    ink2::VisionShardInfo vi;
    if (!ink2::vision_probe(vision, vi, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    ink2::ImageRec rec;
    if (!prepare_capped(image, vi.c, cap, rec, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }

    std::vector<float> cpu_out, gpu_out;
    double t_cpu = -1, t_gpu = -1, t_gpu_first = -1;
    std::string gpu_err;
    if (run_gpu) {
        cudaStream_t st = nullptr;
        cublasHandle_t cb = nullptr;
        if (cudaStreamCreate(&st) != cudaSuccess || cublasCreate(&cb) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "no CUDA device or cuBLAS\n");
            return 1;
        }
        strata::ink::InkCudaVision enc;
        double t0 = now_s();
        bool ok = enc.encode(vision, rec, cb, st, gpu_out, gpu_err);   // includes reading the weights into RAM
        t_gpu_first = now_s() - t0;
        if (ok) {
            t0 = now_s();
            ok = enc.encode(vision, rec, cb, st, gpu_out, gpu_err);     // a second picture: what a request pays
            t_gpu = now_s() - t0;
        }
        if (!ok) std::fprintf(stderr, "CUDA reader: %s\n", gpu_err.c_str());
        else std::fprintf(stderr, "card: %d tokens, first picture %.2f s (weights read from disk), next %.2f s\n",
                          rec.num_lm_tokens(), t_gpu_first, t_gpu);
        std::fflush(stderr);
        cublasDestroy(cb);
        cudaStreamDestroy(st);
    }
    if (run_cpu) {
        ink2::CpuVisionEncoder cpu;
        if (!cpu.load(vision, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        const double t0 = now_s();
        if (!cpu.encode(rec, cpu_out, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        t_cpu = now_s() - t0;
    }
    double rel_l2 = -1, max_abs = -1, cos = -1;
    if (!cpu_out.empty() && cpu_out.size() == gpu_out.size()) {
        double num = 0, den = 0, dot = 0, ng = 0;
        max_abs = 0;
        for (size_t i = 0; i < cpu_out.size(); ++i) {
            const double c = cpu_out[i], g = gpu_out[i], d = g - c;
            num += d * d; den += c * c; dot += c * g; ng += g * g;
            max_abs = std::max(max_abs, std::fabs(d));
        }
        rel_l2 = std::sqrt(num / (den + 1e-30));
        cos = dot / (std::sqrt(den * ng) + 1e-30);
    }
    std::printf("{\"image\": \"%s\", \"size\": [%d, %d], \"tokens\": %d, \"patches\": %d, \"cpu_s\": %.2f, \"gpu_first_s\": %.2f, \"gpu_s\": %.2f, "
                "\"rel_l2\": %.6f, \"max_abs\": %.6f, \"cosine\": %.8f, \"gpu_error\": \"%s\"}\n",
                image.c_str(), rec.orig_w, rec.orig_h, rec.num_lm_tokens(), rec.num_lm_tokens() * vi.c.merge * vi.c.merge,
                t_cpu, t_gpu_first, t_gpu, rel_l2, max_abs, cos, gpu_err.c_str());
    return (run_gpu && gpu_out.empty()) ? 1 : 0;
}
