// metal/kernels/q36.metal - Universal Metal compute kernels for Qwen3.6-35B-A3B (qwen35moe).
// Compiled after quant.metal (fmt<F>, load_tab<F>, deq<F>).

static inline float block_sum(float v, threadgroup float* sh, ushort tid, ushort lane, ushort sgid) {
    v = simd_sum(v);
    if (lane == 0) sh[sgid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float tot = (lane < 4) ? sh[lane] : 0.0f;
        tot = simd_sum(tot);
        if (lane == 0) sh[0] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sh[0];
}

static inline float silu(float v) { return v / (1.0f + exp(-v)); }
static inline float sigm(float v) { return 1.0f / (1.0f + exp(-v)); }
static inline float softplus(float v) { return v > 20.0f ? v : log(1.0f + exp(v)); }

struct GemvParam {
    ulong row_bytes;
    ulong expert_bytes;
    int n_in;
    int n_out;
    ulong x_stride;
};

template <int F>
kernel void k_gemv(device const uchar* W [[buffer(0)]],
                   device const float* x [[buffer(1)]],
                   device float* y [[buffer(2)]],
                   device const int* ids [[buffer(3)]],
                   constant GemvParam& p [[buffer(4)]],
                   uint3 tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]],
                   ushort sgid [[simdgroup_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT;
    constexpr int QK = fmt<F>::QK;
    constexpr int BB = fmt<F>::BB;
    constexpr int UPB = QK / UNIT;
    constexpr int TAB = fmt<F>::TAB;

    threadgroup uint tab[TAB];
    load_tab<F>(tab, tid, 128);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int o = int(tg.x * 8 + sgid);
    const int sel = tg.y;
    if (o >= p.n_out) return;

    const int e = ids[sel];
    device const uchar* row = W + (ulong)e * p.expert_bytes + (ulong)o * p.row_bytes;
    device const float* xs = x + (ulong)sel * p.x_stride;

    const int total_units = p.n_in / UNIT;
    float acc = 0.0f;

    for (int u = lane; u < total_units; u += 32) {
        const int blk = u / UPB;
        const int ui = u % UPB;
        device const uchar* b = row + (ulong)blk * BB;
        device const float* xu = xs + u * UNIT;
        acc += deq_dot<F>(b, ui, xu, tab);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[(ulong)sel * p.n_out + o] = acc;
}


#define INST_GEMV(T) template [[host_name("k_gemv_" #T)]] kernel decltype(k_gemv<T>) k_gemv<T>;
INST_GEMV(T_F32)
INST_GEMV(T_BF16)
INST_GEMV(T_Q8_0)
INST_GEMV(T_Q3_K)
INST_GEMV(T_Q4_K)
INST_GEMV(T_Q5_K)
INST_GEMV(T_Q6_K)
INST_GEMV(T_IQ2_XS)
INST_GEMV(T_IQ3_XXS)
INST_GEMV(T_IQ4_XS)

template <int F>
kernel void k_gemv_lpr8(device const uchar* W [[buffer(0)]],
                        device const float* x [[buffer(1)]],
                        device float* y [[buffer(2)]],
                        device const int* ids [[buffer(3)]],
                        constant GemvParam& p [[buffer(4)]],
                        uint3 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]],
                        ushort sgid [[simdgroup_index_in_threadgroup]]) {
    constexpr int QK = fmt<F>::QK;
    constexpr int BB = fmt<F>::BB;

    threadgroup float sh_x[2048];
    const int sel = tg.y;
    device const float4* xs4 = (device const float4*)(x + (ulong)sel * p.x_stride);
    threadgroup float4* sh_x4 = (threadgroup float4*)sh_x;
    for (int i = tid; i < 512; i += 128) {
        sh_x4[i] = xs4[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int r_in_sg = lane / 8;
    const int li = lane % 8;

    const int o = int(tg.x * 16 + sgid * 4 + r_in_sg);
    if (o >= p.n_out) return;

    const int e = ids[sel];
    device const uchar* row = W + (ulong)e * p.expert_bytes + (ulong)o * p.row_bytes;

    device const uchar* b = row + (ulong)li * BB;
    threadgroup const float* x_blk = sh_x + li * QK;
    float acc = deq_dot_block<F>(b, x_blk, nullptr);

    acc += simd_shuffle_xor(acc, 4);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);

    if (li == 0) {
        y[(ulong)sel * p.n_out + o] = acc;
    }
}

#define INST_GEMV_LPR8(T) template [[host_name("k_gemv_lpr8_" #T)]] kernel decltype(k_gemv_lpr8<T>) k_gemv_lpr8<T>;
INST_GEMV_LPR8(T_Q4_K)
INST_GEMV_LPR8(T_Q5_K)
INST_GEMV_LPR8(T_Q6_K)
INST_GEMV_LPR8(T_IQ2_XS)


template <int F>
kernel void k_row(device const uchar* W [[buffer(0)]],
                  device float* out [[buffer(1)]],
                  constant ulong& row_bytes [[buffer(2)]],
                  constant int& row [[buffer(3)]],
                  constant int& n [[buffer(4)]],
                  uint tg [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT;
    constexpr int QK = fmt<F>::QK;
    constexpr int BB = fmt<F>::BB;
    constexpr int UPB = QK / UNIT;
    constexpr int TAB = fmt<F>::TAB;

    threadgroup uint tab[TAB];
    load_tab<F>(tab, tid, 128);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const uchar* r = W + (ulong)row * row_bytes;
    const int total_units = n / UNIT;
    thread float q[UNIT];

    for (int u = tg * 128 + tid; u < total_units; u += 16 * 128) {
        const int blk = u / UPB;
        const int ui = u % UPB;
        device const uchar* b = r + (ulong)blk * BB;
        const float scale = deq<F>(b, ui, q, tab);
        device float* dst = out + u * UNIT;
        for (int l = 0; l < UNIT; ++l) {
            dst[l] = scale * q[l];
        }
    }
}

#define INST_ROW(T) template [[host_name("k_row_" #T)]] kernel decltype(k_row<T>) k_row<T>;
INST_ROW(T_F32)
INST_ROW(T_BF16)
INST_ROW(T_Q8_0)
INST_ROW(T_Q3_K)
INST_ROW(T_Q4_K)
INST_ROW(T_Q5_K)
INST_ROW(T_Q6_K)
INST_ROW(T_IQ2_XS)
INST_ROW(T_IQ3_XXS)
INST_ROW(T_IQ4_XS)

kernel void k_rms(device const float* x [[buffer(0)]],
                  device const float* w [[buffer(1)]],
                  device float* y [[buffer(2)]],
                  constant int& n [[buffer(3)]],
                  constant float& eps [[buffer(4)]],
                  uint gid [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  ushort lane [[thread_index_in_simdgroup]],
                  ushort sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sh[4];
    device const float* xg = x + (ulong)gid * n;
    device float* yg = y + (ulong)gid * n;
    float s = 0.0f;
    for (int i = tid; i < n; i += 128) {
        float val = xg[i];
        s += val * val;
    }
    s = block_sum(s, sh, tid, lane, sgid);
    const float r = rsqrt(s / float(n) + eps);
    for (int i = tid; i < n; i += 128) {
        yg[i] = xg[i] * r * w[i];
    }
}

kernel void k_conv(device const float* qkv [[buffer(0)]],
                   device float* state [[buffer(1)]],
                   device const float* w [[buffer(2)]],
                   device float* out [[buffer(3)]],
                   constant int& C [[buffer(4)]],
                   constant int& K [[buffer(5)]],
                   uint c [[thread_position_in_grid]]) {
    if (c >= (uint)C) return;
    float acc = 0.0f;
    for (int k = 0; k < K - 1; ++k) acc += state[(ulong)k * C + c] * w[(ulong)c * K + k];
    acc += qkv[c] * w[(ulong)c * K + K - 1];
    for (int k = 0; k < K - 2; ++k) state[(ulong)k * C + c] = state[(ulong)(k + 1) * C + c];
    state[(ulong)(K - 2) * C + c] = qkv[c];
    out[c] = silu(acc);
}

kernel void k_gdn(device const float* conv [[buffer(0)]],
                  device const float* alpha [[buffer(1)]],
                  device const float* dt_bias [[buffer(2)]],
                  device const float* ssm_a [[buffer(3)]],
                  device const float* beta_raw [[buffer(4)]],
                  device float* S [[buffer(5)]],
                  device float* out [[buffer(6)]],
                  constant int& HK [[buffer(7)]],
                  constant int& D [[buffer(8)]],
                  constant float& eps [[buffer(9)]],
                  uint h [[threadgroup_position_in_grid]],
                  uint i [[thread_index_in_threadgroup]],
                  ushort lane [[thread_index_in_simdgroup]],
                  ushort sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sh_q[128];
    threadgroup float sh_k[128];
    threadgroup float sh_red[4];

    const int kh = h % HK;
    const float qi = conv[kh * D + i];
    const float ki = conv[HK * D + kh * D + i];
    const float vi = conv[2 * HK * D + h * D + i];

    float qn = block_sum(qi * qi, sh_red, i, lane, sgid);
    float kn = block_sum(ki * ki, sh_red, i, lane, sgid);

    sh_q[i] = qi * rsqrt(qn + eps) * (1.0f / sqrt(float(D)));
    sh_k[i] = ki * rsqrt(kn + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float g = exp(softplus(alpha[h] + dt_bias[h]) * ssm_a[h]);
    const float b = sigm(beta_raw[h]);

    device float* Si = S + ((ulong)h * D + i) * D;
    float dot_k = 0.0f;
    for (int j = 0; j < D; ++j) {
        Si[j] *= g;
        dot_k += Si[j] * sh_k[j];
    }
    const float d = (vi - dot_k) * b;
    float o = 0.0f;
    for (int j = 0; j < D; ++j) {
        Si[j] += d * sh_k[j];
        o += Si[j] * sh_q[j];
    }
    out[(ulong)h * D + i] = o;
}

kernel void k_gdn_gb(device float* alpha [[buffer(0)]],
                     device const float* dt_bias [[buffer(1)]],
                     device const float* ssm_a [[buffer(2)]],
                     device float* beta [[buffer(3)]],
                     constant int& HV [[buffer(4)]],
                     uint h [[thread_position_in_grid]]) {
    if (h >= (uint)HV) return;
    alpha[h] = softplus(alpha[h] + dt_bias[h]) * ssm_a[h];
    beta[h] = sigm(beta[h]);
}

kernel void k_gated_norm(device float* o [[buffer(0)]],
                         device const float* w [[buffer(1)]],
                         device const float* z [[buffer(2)]],
                         constant int& D [[buffer(3)]],
                         constant float& eps [[buffer(4)]],
                         uint h [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]],
                         ushort sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sh[4];
    device float* oh = o + (ulong)h * D;
    float v = oh[tid];
    float s = block_sum(v * v, sh, tid, lane, sgid);
    oh[tid] = v * rsqrt(s / float(D) + eps) * w[tid] * silu(z[(ulong)h * D + tid]);
}

kernel void k_attn_prep(device const float* qg [[buffer(0)]],
                        device const float* kx [[buffer(1)]],
                        device const float* vx [[buffer(2)]],
                        device const float* qw [[buffer(3)]],
                        device const float* kw [[buffer(4)]],
                        device float* q [[buffer(5)]],
                        device float* gate [[buffer(6)]],
                        device float* kc [[buffer(7)]],
                        device float* vc [[buffer(8)]],
                        constant int& H [[buffer(9)]],
                        constant int& D [[buffer(10)]],
                        constant int& n_rot [[buffer(11)]],
                        constant int& pos [[buffer(12)]],
                        constant int& ctx [[buffer(13)]],
                        constant float& base [[buffer(14)]],
                        constant float& eps [[buffer(15)]],
                        uint head [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]],
                        ushort sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sh[4];
    threadgroup float buf[256];
    const bool isq = head < (uint)H;
    device const float* src = isq ? qg + (ulong)head * 2 * D : kx + (ulong)(head - H) * D;

    float s = 0.0f;
    for (int i = tid; i < D; i += 128) {
        float val = src[i];
        s += val * val;
    }
    s = block_sum(s, sh, tid, lane, sgid);
    const float r = rsqrt(s / float(D) + eps);

    for (int i = tid; i < D; i += 128) {
        buf[i] = src[i] * r * (isq ? qw[i] : kw[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* dst = isq ? q + (ulong)head * D : kc + ((ulong)(head - H) * ctx + pos) * D;
    const int half_rot = n_rot / 2;
    for (int i = tid; i < D; i += 128) {
        float v = buf[i];
        if (i < n_rot) {
            int p = (i < half_rot) ? i : i - half_rot;
            float theta = float(pos) * pow(base, -2.0f * float(p) / float(n_rot));
            float cs = cos(theta), sn = sin(theta);
            v = (i < half_rot) ? buf[i] * cs - buf[i + half_rot] * sn : buf[i - half_rot] * sn + buf[i] * cs;
        }
        dst[i] = v;
    }
    if (isq) {
        for (int i = tid; i < D; i += 128) {
            gate[(ulong)head * D + i] = qg[(ulong)head * 2 * D + D + i];
        }
    } else {
        for (int i = tid; i < D; i += 128) {
            vc[((ulong)(head - H) * ctx + pos) * D + i] = vx[(ulong)(head - H) * D + i];
        }
    }
}

kernel void k_attn(device const float* q [[buffer(0)]],
                   device const float* kc [[buffer(1)]],
                   device const float* vc [[buffer(2)]],
                   device const float* gate [[buffer(3)]],
                   device float* out [[buffer(4)]],
                   constant int& H [[buffer(5)]],
                   constant int& HKV [[buffer(6)]],
                   constant int& D [[buffer(7)]],
                   constant int& n_pos [[buffer(8)]],
                   constant int& ctx [[buffer(9)]],
                   constant float& scale [[buffer(10)]],
                   threadgroup float* sc [[threadgroup(0)]],
                   uint h [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]],
                   ushort sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sh[4];
    const int kvh = h / (H / HKV);
    device const float* qh = q + (ulong)h * D;

    for (int p = tid; p < n_pos; p += 128) {
        device const float* kp = kc + ((ulong)kvh * ctx + p) * D;
        float s = 0.0f;
        for (int i = 0; i < D; ++i) s += qh[i] * kp[i];
        sc[p] = s * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mx = -INFINITY;
    for (int p = tid; p < n_pos; p += 128) mx = max(mx, sc[p]);
    mx = simd_max(mx);
    if (lane == 0) sh[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float tot = (lane < 4) ? sh[lane] : -INFINITY;
        tot = simd_max(tot);
        if (lane == 0) sh[0] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = sh[0];

    float sum = 0.0f;
    for (int p = tid; p < n_pos; p += 128) {
        float e = exp(sc[p] - mx);
        sc[p] = e;
        sum += e;
    }
    sum = block_sum(sum, sh, tid, lane, sgid);
    const float inv_sum = 1.0f / sum;

    for (int i = tid; i < D; i += 128) {
        float acc = 0.0f;
        for (int p = 0; p < n_pos; ++p) acc += sc[p] * vc[((ulong)kvh * ctx + p) * D + i];
        out[(ulong)h * D + i] = (acc * inv_sum) * sigm(gate[(ulong)h * D + i]);
    }
}

kernel void k_swiglu(device const float* g [[buffer(0)]],
                     device const float* u [[buffer(1)]],
                     device float* h [[buffer(2)]],
                     constant int& n [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < (uint)n) h[i] = silu(g[i]) * u[i];
}

kernel void k_add(device float* x [[buffer(0)]],
                  device const float* y [[buffer(1)]],
                  constant int& n [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    if (i < (uint)n) x[i] += y[i];
}

kernel void k_moe_sum(device float* x [[buffer(0)]],
                      device const float* y [[buffer(1)]],
                      device const float* w [[buffer(2)]],
                      device const float* ys [[buffer(3)]],
                      device const float* gs [[buffer(4)]],
                      constant int& k [[buffer(5)]],
                      constant int& n [[buffer(6)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= (uint)n) return;
    float acc = 0.0f;
    for (int s = 0; s < k; ++s) acc += w[s] * y[(ulong)s * n + i];
    x[i] += acc + sigm(gs[0]) * ys[i];
}

kernel void k_topk_router(device const float* router [[buffer(0)]],
                          device int* ids [[buffer(1)]],
                          device float* wts [[buffer(2)]],
                          constant int& n_expert [[buffer(3)]],
                          constant int& K [[buffer(4)]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float p[256];
    if (tid != 0) return;

    float mx = router[0];
    for (int i = 1; i < n_expert; ++i) {
        if (router[i] > mx) mx = router[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n_expert; ++i) {
        float val = exp(router[i] - mx);
        p[i] = val;
        sum += val;
    }
    float inv_sum = 1.0f / (sum > 0.0f ? sum : 1.0f);
    for (int i = 0; i < n_expert; ++i) {
        p[i] *= inv_sum;
    }

    int top_idx[16];
    float top_val[16];
    for (int k = 0; k < K; ++k) {
        top_val[k] = -1.0f;
        top_idx[k] = -1;
    }

    for (int i = 0; i < n_expert; ++i) {
        float v = p[i];
        if (v > top_val[K - 1] || (v == top_val[K - 1] && (top_idx[K - 1] < 0 || i < top_idx[K - 1]))) {
            int pos = K - 1;
            while (pos > 0 && (v > top_val[pos - 1] || (v == top_val[pos - 1] && i < top_idx[pos - 1]))) {
                top_val[pos] = top_val[pos - 1];
                top_idx[pos] = top_idx[pos - 1];
                --pos;
            }
            top_val[pos] = v;
            top_idx[pos] = i;
        }
    }

    float wsum = 0.0f;
    for (int k = 0; k < K; ++k) {
        wsum += top_val[k];
    }
    float inv_wsum = 1.0f / (wsum > 0.0f ? wsum : 1.0f);
    for (int k = 0; k < K; ++k) {
        ids[k] = top_idx[k];
        wts[k] = top_val[k] * inv_wsum;
    }
}

// ---------------------------------------------------------------------------
// Split-K attention (q36_engine.cu's k_attn_part / k_attn_merge, on Metal): the attention scores no longer
// live in threadgroup memory, so k_attn's 32 KB / 8,192-token threadgroup ceiling is gone and the context
// is limited by the KV cache only.  One partial per (query row, chunk of SK_CELLS cells): the chunk's
// scores, running max m and sum l, and the chunk's P.V accumulator; k_attn_sk_merge folds the chunks (the
// softmax rescale) and applies the sigmoid gate.  The math and its order match the CUDA engine's kernels,
// so long-context attention agrees with q36_engine.cu; at n_pos <= 8192 the original k_attn stays the
// default (the decode path is unchanged there).  Query rows are (t * H + h): the decode step passes one
// token (t = 0), the prompt chunks pass a row-block.  HD is attention.key_length - 256 on qwen35moe, and
// the host refuses this path with anything else.
enum { SK_HD = 256 };
enum { SK_CELLS = 256 };

// 256-thread reductions (8 simdgroups; block_sum above assumes 128 threads).  Both end with a full
// threadgroup barrier, so writes made before the call (k_attn_sk_part's sc[]) are visible to every thread
// after it returns - the same guarantee block_sum gives k_attn.
static inline float sk_max(float v, threadgroup float* sh, ushort lane, ushort sgid) {
    v = simd_max(v);
    if (lane == 0) sh[sgid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float tot = (lane < 8) ? sh[lane] : -INFINITY;
        tot = simd_max(tot);
        if (lane == 0) sh[0] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sh[0];
}

static inline float sk_sum(float v, threadgroup float* sh, ushort lane, ushort sgid) {
    v = simd_sum(v);
    if (lane == 0) sh[sgid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float tot = (lane < 8) ? sh[lane] : 0.0f;
        tot = simd_sum(tot);
        if (lane == 0) sh[0] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sh[0];
}

kernel void k_attn_sk_part(device const float* q [[buffer(0)]],
                           device const float* kc [[buffer(1)]],
                           device const float* vc [[buffer(2)]],
                           constant int& H [[buffer(3)]],
                           constant int& HKV [[buffer(4)]],
                           constant int& ctx [[buffer(5)]],
                           constant float& scale [[buffer(6)]],
                           constant int& n_chunk [[buffer(7)]],
                           constant int& pos0 [[buffer(8)]],
                           constant int& row0 [[buffer(9)]],
                           device float* pacc [[buffer(10)]],
                           device float* pm [[buffer(11)]],
                           device float* pl [[buffer(12)]],
                           uint3 g [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           ushort sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sh[8];
    threadgroup float qs[SK_HD];
    threadgroup float sc[SK_CELLS];
    const int c = (int) g.x, row = row0 + (int) g.y;   // g.y is the row-block-local row; scratch is indexed by it
    const int t = row / H, h = row - t * H, kvh = h / (H / HKV);
    const int npos = pos0 + t + 1;   // this query row sees the cells [0, pos0 + t]
    const int p0 = c * SK_CELLS;
    // whole threadgroup exits together: p0 and npos are threadgroup-uniform, no barrier is skipped midway
    if (p0 >= npos) return;

    const int j = (int) tid;
    qs[j] = q[(ulong) row * SK_HD + j];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int p = p0 + j;
    float s = -INFINITY;
    if (p < npos) {
        device const float* kp = kc + ((ulong) kvh * ctx + p) * SK_HD;
        float a = 0.0f;
        for (int i = 0; i < SK_HD; ++i) a += qs[i] * kp[i];
        s = a * scale;
    }
    const float m = sk_max(s, sh, lane, sgid);
    const float e = (p < npos) ? exp(s - m) : 0.0f;
    sc[j] = e;
    const float l = sk_sum(e, sh, lane, sgid);   // its trailing barrier publishes every thread's sc[j]

    const int n = min(SK_CELLS, npos - p0);
    float acc = 0.0f;
    device const float* vp = vc + ((ulong) kvh * ctx + p0) * SK_HD + j;
    for (int k = 0; k < n; ++k) acc += sc[k] * vp[(ulong) k * SK_HD];

    const ulong o = (ulong) (row - row0) * (ulong) n_chunk + (ulong) c;
    pacc[o * SK_HD + j] = acc;
    if (tid == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}

kernel void k_attn_sk_merge(device const float* pacc [[buffer(0)]],
                            device const float* pm [[buffer(1)]],
                            device const float* pl [[buffer(2)]],
                            device const float* gate [[buffer(3)]],
                            device float* out [[buffer(4)]],
                            constant int& H [[buffer(5)]],
                            constant int& n_chunk [[buffer(6)]],
                            constant int& pos0 [[buffer(7)]],
                            constant int& row0 [[buffer(8)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    const int grow = row0 + (int) row;
    const int t = grow / H;
    const int npos = pos0 + t + 1;
    const int nc = (npos + SK_CELLS - 1) / SK_CELLS;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = max(M, pm[(ulong) row * (ulong) n_chunk + c]);
    float num = 0.0f, den = 0.0f;
    for (int c = 0; c < nc; ++c) {
        const ulong o = (ulong) row * (ulong) n_chunk + (ulong) c;
        const float w = exp(pm[o] - M);
        num += w * pacc[o * SK_HD + tid];
        den += w * pl[o];
    }
    const ulong gi = (ulong) grow * SK_HD + tid;
    out[gi] = num / den * sigm(gate[gi]);
}

// ---------------------------------------------------------------------------
// Prompt-chunk kernels: the chunk of T prompt tokens runs as T-wide dispatches instead of T separate
// single-token graphs, with every weight read once per chunk.  Each (token, output) is accumulated in
// exactly the decode kernels' order (units lane-strided, deq_dot per unit, simd_sum; the shift-register
// conv and the in-place delta rule walk the tokens sequentially), so the chunk's states and the last
// token's logits equal the token-by-token run's bit for bit.  The prompt's last token is NOT chunked: it
// runs as a decode window (that is where the logits come from), like mac-kern's prompt chain.

// Embedding rows: grid (T), one threadgroup per token; the same deq as k_row (identical values).
template <int F>
kernel void k_emb_chunk(device const uchar* W [[buffer(0)]],
                        device const int* toks [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant ulong& row_bytes [[buffer(3)]],
                        constant int& n [[buffer(4)]],
                        constant int& n_embd [[buffer(5)]],
                        uint t [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT;
    constexpr int QK = fmt<F>::QK;
    constexpr int BB = fmt<F>::BB;
    constexpr int UPB = QK / UNIT;
    constexpr int TAB = fmt<F>::TAB;

    threadgroup uint tab[TAB];
    load_tab<F>(tab, tid, 128);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const uchar* r = W + (ulong) toks[t] * row_bytes;
    device float* dst = out + (ulong) t * n_embd;
    const int total_units = n / UNIT;
    thread float q[UNIT];
    for (int u = tid; u < total_units; u += 128) {
        const int blk = u / UPB;
        const int ui = u % UPB;
        device const uchar* b = r + (ulong) blk * BB;
        const float scale = deq<F>(b, ui, q, tab);
        device float* d = dst + u * UNIT;
        for (int l = 0; l < UNIT; ++l) d[l] = scale * q[l];
    }
}

#define INST_EMB_CHUNK(T) template [[host_name("k_emb_chunk_" #T)]] kernel decltype(k_emb_chunk<T>) k_emb_chunk<T>;
INST_EMB_CHUNK(T_F32)
INST_EMB_CHUNK(T_BF16)
INST_EMB_CHUNK(T_Q8_0)
INST_EMB_CHUNK(T_Q3_K)
INST_EMB_CHUNK(T_Q4_K)
INST_EMB_CHUNK(T_Q5_K)
INST_EMB_CHUNK(T_Q6_K)
INST_EMB_CHUNK(T_IQ2_XS)
INST_EMB_CHUNK(T_IQ3_XXS)
INST_EMB_CHUNK(T_IQ4_XS)

// Dense projections for the chunk: k_gemv with the tokens walked INSIDE the threadgroup, so each weight
// row is read from memory once no matter how many tokens follow.  grid (n_out/8, ceil(T/tpb)); tpb =
// tokens per threadgroup-row (1 = the decode kernel's shape, one token per row; large = one threadgroup
// per output window serving every token).  x rows are the chunk's activations at x_stride; y rows are
// chunk-local (row (t - t0)).
template <int F>
kernel void k_gemv_c(device const uchar* W [[buffer(0)]],
                     device const float* x [[buffer(1)]],
                     device float* y [[buffer(2)]],
                     constant GemvParam& p [[buffer(3)]],
                     constant int& tpb [[buffer(4)]],
                     constant int& t0 [[buffer(5)]],
                     constant int& nt [[buffer(6)]],
                     uint3 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]],
                     ushort sgid [[simdgroup_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT;
    constexpr int QK = fmt<F>::QK;
    constexpr int BB = fmt<F>::BB;
    constexpr int UPB = QK / UNIT;
    constexpr int TAB = fmt<F>::TAB;

    threadgroup uint tab[TAB];
    load_tab<F>(tab, tid, 128);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int o = int(tg.x * 8 + sgid);
    if (o >= p.n_out) return;
    const int total_units = p.n_in / UNIT;
    device const uchar* row = W + (ulong) o * p.row_bytes;
    const int tb = t0 + (int) tg.y * tpb;
    const int te = min(tb + tpb, t0 + nt);
    for (int t = tb; t < te; ++t) {
        device const float* xs = x + (ulong) (t - t0) * p.x_stride;
        float acc = 0.0f;
        for (int u = lane; u < total_units; u += 32) {
            const int blk = u / UPB;
            const int ui = u % UPB;
            device const uchar* b = row + (ulong) blk * BB;
            acc += deq_dot<F>(b, ui, xs + u * UNIT, tab);
        }
        acc = simd_sum(acc);
        if (lane == 0) y[(ulong) (t - t0) * p.n_out + o] = acc;
    }
}

#define INST_GEMV_C(T) template [[host_name("k_gemv_c_" #T)]] kernel decltype(k_gemv_c<T>) k_gemv_c<T>;
INST_GEMV_C(T_F32)
INST_GEMV_C(T_BF16)
INST_GEMV_C(T_Q8_0)
INST_GEMV_C(T_Q3_K)
INST_GEMV_C(T_Q4_K)
INST_GEMV_C(T_Q5_K)
INST_GEMV_C(T_Q6_K)
INST_GEMV_C(T_IQ2_XS)
INST_GEMV_C(T_IQ3_XXS)
INST_GEMV_C(T_IQ4_XS)

// The n_in = 2048 products of the chunk do NOT get a token-loop kernel: the serial deq_dot path that the
// threadgroup-cached x forces compiles to different fma contraction inside a token loop than in the decode
// kernel's straight-line shape (the parity test: ~1 ulp on most outputs), and greedy decoding turns a
// 1-ulp logit difference into different tokens on near-ties.  The engine dispatches the DECODE kernel
// itself with grid[1] = nt instead (push_c_gemv): bit-identical by construction, and the re-read weights
// stay in the L2 across the chunk's tokens of one layer.

// Routed experts for the chunk, grouped per expert so every expert's weights are read once: the chunk's
// T x K entries are placed by k_place into per-expert runs; this kernel walks an expert's entries with
// each entry's (token, output) summed in k_gemv's exact order.  grid (n_out/8, n_expert); y rows are
// entry-major (row = the entry's index).
template <int F>
kernel void k_expc(device const uchar* W [[buffer(0)]],
                   device const float* x [[buffer(1)]],
                   device const int* srt_in [[buffer(2)]],   // entry -> the input row it reads
                   device const int* off [[buffer(3)]],      // expert -> first entry
                   device const int* cnt [[buffer(4)]],      // expert -> entry count
                   device float* y [[buffer(5)]],
                   constant GemvParam& p [[buffer(6)]],
                   uint3 tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]],
                   ushort sgid [[simdgroup_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT;
    constexpr int QK = fmt<F>::QK;
    constexpr int BB = fmt<F>::BB;
    constexpr int UPB = QK / UNIT;
    constexpr int TAB = fmt<F>::TAB;

    threadgroup uint tab[TAB];
    load_tab<F>(tab, tid, 128);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int o = int(tg.x * 8 + sgid);
    if (o >= p.n_out) return;
    const int e = (int) tg.y;
    const int b0 = off[e], n_e = cnt[e];
    device const uchar* row = W + (ulong) e * p.expert_bytes + (ulong) o * p.row_bytes;
    const int total_units = p.n_in / UNIT;
    for (int r = 0; r < n_e; ++r) {
        const int entry = b0 + r;
        device const float* xs = x + (ulong) srt_in[entry] * p.x_stride;
        float acc = 0.0f;
        for (int u = lane; u < total_units; u += 32) {
            const int blk = u / UPB;
            const int ui = u % UPB;
            device const uchar* b = row + (ulong) blk * BB;
            acc += deq_dot<F>(b, ui, xs + u * UNIT, tab);
        }
        acc = simd_sum(acc);
        if (lane == 0) y[(ulong) entry * p.n_out + o] = acc;
    }
}

#define INST_EXPC(T) template [[host_name("k_expc_" #T)]] kernel decltype(k_expc<T>) k_expc<T>;
INST_EXPC(T_F32)
INST_EXPC(T_BF16)
INST_EXPC(T_Q8_0)
INST_EXPC(T_Q3_K)
INST_EXPC(T_Q4_K)
INST_EXPC(T_Q5_K)
INST_EXPC(T_Q6_K)
INST_EXPC(T_IQ2_XS)
INST_EXPC(T_IQ3_XXS)
INST_EXPC(T_IQ4_XS)

// Top-K routing for all T tokens in one launch: each token's work is k_topk_router's, done by its
// threadgroup's first thread in the same order (parallel would reorder the softmax's sum).
kernel void k_route_chunk(device const float* router [[buffer(0)]],
                          device int* ids [[buffer(1)]],
                          device float* wts [[buffer(2)]],
                          constant int& n_expert [[buffer(3)]],
                          constant int& K [[buffer(4)]],
                          uint t [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float p[256];
    if (tid != 0) return;
    if (n_expert > 256) return;

    device const float* r = router + (ulong) t * n_expert;
    float mx = r[0];
    for (int i = 1; i < n_expert; ++i) {
        if (r[i] > mx) mx = r[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n_expert; ++i) {
        float val = exp(r[i] - mx);
        p[i] = val;
        sum += val;
    }
    float inv_sum = 1.0f / (sum > 0.0f ? sum : 1.0f);
    for (int i = 0; i < n_expert; ++i) {
        p[i] *= inv_sum;
    }

    int top_idx[16];
    float top_val[16];
    for (int k = 0; k < K; ++k) {
        top_val[k] = -1.0f;
        top_idx[k] = -1;
    }
    for (int i = 0; i < n_expert; ++i) {
        float v = p[i];
        if (v > top_val[K - 1] || (v == top_val[K - 1] && (top_idx[K - 1] < 0 || i < top_idx[K - 1]))) {
            int pos = K - 1;
            while (pos > 0 && (v > top_val[pos - 1] || (v == top_val[pos - 1] && i < top_idx[pos - 1]))) {
                top_val[pos] = top_val[pos - 1];
                top_idx[pos] = top_idx[pos - 1];
                --pos;
            }
            top_val[pos] = v;
            top_idx[pos] = i;
        }
    }

    float wsum = 0.0f;
    for (int k = 0; k < K; ++k) wsum += top_val[k];
    float inv_wsum = 1.0f / (wsum > 0.0f ? wsum : 1.0f);
    for (int k = 0; k < K; ++k) {
        ids[(ulong) t * K + k] = top_idx[k];
        wts[(ulong) t * K + k] = top_val[k] * inv_wsum;
    }
}

kernel void k_zero(device int* buf [[buffer(0)]],
                   constant int& n [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < (uint) n) buf[i] = 0;
}

kernel void k_hist(device const int* ids [[buffer(0)]],
                   device uint* cnt [[buffer(1)]],
                   constant int& n_ent [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < (uint) n_ent) atomic_fetch_add_explicit((device atomic_uint*)&cnt[ids[i]], 1u, memory_order_relaxed);
}

kernel void k_exp_off(device const int* cnt [[buffer(0)]],
                      device int* off [[buffer(1)]],
                      constant int& n_expert [[buffer(2)]],
                      uint tid [[thread_index_in_threadgroup]]) {
    if (tid != 0) return;
    int acc = 0;
    for (int e = 0; e < n_expert; ++e) {
        off[e] = acc;
        acc += cnt[e];
    }
}

// Entry placement: entry i = (t, k) lands at row = off[e] + its rank among the entries of the same expert
// (device atomics; the rank order varies, the numbers do not - each token's K outputs are combined in its
// own fixed slot order by k_moe_sum_chunk).  Kept for the parity test; the engine's chunk graph uses
// k_tables instead - the chain of four dispatches with relaxed atomics and memory barriers between them
// did not order on the M2 (the grouped experts read stale tables inside one barrier-ordered graph).
kernel void k_place(device const int* ids [[buffer(0)]],
                    device uint* run [[buffer(1)]],
                    device const int* off [[buffer(2)]],
                    device int* srt_in_gu [[buffer(3)]],
                    device int* srt_in_d [[buffer(4)]],
                    device int* pos [[buffer(5)]],
                    constant int& K [[buffer(6)]],
                    constant int& n_ent [[buffer(7)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= (uint) n_ent) return;
    const int t = (int) i / K;
    const int e = ids[i];
    const uint row = off[e] + atomic_fetch_add_explicit((device atomic_uint*)&run[e], 1u, memory_order_relaxed);
    srt_in_gu[row] = t;
    srt_in_d[row] = (int) i;
    pos[i] = (int) row;
}

kernel void k_moe_sum_chunk(device float* x [[buffer(0)]],
                            device const float* y [[buffer(1)]],
                            device const float* w [[buffer(2)]],
                            device const float* ys [[buffer(3)]],
                            device const float* gs [[buffer(4)]],
                            device const int* pos [[buffer(5)]],
                            constant int& k [[buffer(6)]],
                            constant int& n_out [[buffer(7)]],
                            constant int& n [[buffer(8)]],
                            uint3 g [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    const int t = (int) g.x;
    const int i = (int) g.y * 128 + (int) tid;
    if (i >= n) return;
    float acc = 0.0f;
    for (int s = 0; s < k; ++s) acc += w[(ulong) t * k + s] * y[(ulong) pos[(ulong) t * k + s] * n_out + i];
    x[(ulong) t * n + i] += acc + sigm(gs[t]) * ys[(ulong) t * n + i];
}

// The chunk's entry tables in ONE dispatch: one threadgroup of 256 threads does zero, count (threadgroup
// atomics - the moe_mm pattern), the exclusive offsets, and the placement; the tables leave as plain
// stores for the grouped experts' plain loads in the NEXT dispatch (across its memory barrier - the
// dependency pattern the whole decode path is built on).  The engine's earlier four-dispatch version
// (device atomics + inter-dispatch memory barriers) raced: k_expc read stale tables inside one
// barrier-ordered graph, and the expert rows came back NaN-garbage.
kernel void k_tables(device const int* ids [[buffer(0)]],
                     device int* cnt_off [[buffer(1)]],       // [0..256) counts, [256..512) offsets
                     device int* srt_in_gu [[buffer(2)]],
                     device int* srt_in_d [[buffer(3)]],
                     device int* pos [[buffer(4)]],
                     constant int& n_expert [[buffer(5)]],
                     constant int& K [[buffer(6)]],
                     constant int& n_ent [[buffer(7)]],
                     uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint sh_cnt[256];
    threadgroup atomic_uint sh_run[256];
    threadgroup int sh_off[256];
    if (n_expert > 256) return;

    atomic_store_explicit(&sh_cnt[tid], 0u, memory_order_relaxed);
    atomic_store_explicit(&sh_run[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < (uint) n_ent; i += 256)
        atomic_fetch_add_explicit(&sh_cnt[ids[i]], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int acc = 0;
        for (int e = 0; e < n_expert; ++e) {
            sh_off[e] = acc;
            acc += (int) atomic_load_explicit(&sh_cnt[e], memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < (uint) n_ent; i += 256) {
        const int t = (int) i / K;
        const int e = ids[i];
        const uint row = (uint) sh_off[e] + atomic_fetch_add_explicit(&sh_run[e], 1u, memory_order_relaxed);
        srt_in_gu[row] = t;
        srt_in_d[row] = (int) i;
        pos[i] = (int) row;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < (uint) n_expert) {
        cnt_off[tid] = (int) atomic_load_explicit(&sh_cnt[tid], memory_order_relaxed);
        cnt_off[256 + tid] = sh_off[tid];
    }
}

// The chunk's routed-expert input rows in (token, slot) order: entry i = t*K + k is a copy of its token's
// row (t = i / K).  The gate/up products then run the DECODE k_gemv_lpr8 per entry (grid[1] = n_ent,
// ids[entry] picking the expert) - its deq_dot_block + 8-lane butterfly is the token-by-token path's exact
// arithmetic, and k_expc's plain-gemv order (32-lane strided units + simd_sum) differs by ~1 ulp at
// n_in = 2048 (round 9's G8 cut: a 5e-7 relative diff on the eh sum - greedy turns that into different
// tokens).  Pure row copies, no arithmetic.
kernel void k_gather_t(device const float* x [[buffer(0)]],
                       device float* y [[buffer(1)]],
                       constant int& width [[buffer(2)]],
                       constant int& K [[buffer(3)]],
                       constant int& count [[buffer(4)]],   // elements = n_ent * width; bounds the grid's tail
                       uint i [[thread_position_in_grid]]) {
    // Round 11 (M2): the dispatches passed the ELEMENT count as the threadgroup count, so 128x too many
    // threads ran; with no bound here the extra ones read past x (fresh arena pages read as zeros) and
    // wrote 128x past y, smashing live buffers above it (stage 16's x2/x3; the engine's state above
    // hg_c on every chunk - the ladder's 0/10).  Grids now count threadgroups and this bound guards
    // the tail threadgroup.
    if (i >= (uint) count) return;
    const uint row = i / (uint) width;   // the entry index (t*K + k); its token is row / K
    y[i] = x[(ulong) (row / (uint) K) * width + i % (uint) width];
}

// Conv for T tokens, tiles of 16 side by side (q36_engine.cu's k_conv_seq tile mode, k_conv's math):
// per channel a K-1-deep shift register; tile 0's registers start at the held history, tile t > 0's at
// the tile's own preceding inputs; tile 0, as the only history writer, stores the last K - 1 inputs.
kernel void k_conv_chunk(device const float* qkv [[buffer(0)]],
                         device float* state [[buffer(1)]],
                         device const float* w [[buffer(2)]],
                         device float* out [[buffer(3)]],
                         constant int& C [[buffer(4)]],
                         constant int& K [[buffer(5)]],
                         constant int& T [[buffer(6)]],
                         uint3 g [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    const int c = (int) g.x * 128 + (int) tid;
    if (c >= C) return;
    const int tile = (int) g.y;
    const int t_first = tile * 16;
    if (t_first >= T) return;
    if (K - 1 > 16) return;   // threadgroup-uniform; the host refuses d_conv > 17 before dispatching
    const int t_end = min(T, t_first + 16);

    thread float v[16];
    thread float oldv[16];
    if (tile == 0) {
        for (int k = 0; k < K - 1; ++k) {
            v[k] = state[(ulong) k * C + c];
            oldv[k] = v[k];
        }
        // tile 0 is the only history writer: the chunk's last K - 1 inputs (exactly the values the
        // sequential loop would leave in the shift register), old history filling a short chunk
        for (int k = 0; k < K - 1; ++k) {
            const int idx = T - (K - 1) + k;
            state[(ulong) k * C + c] = idx >= 0 ? qkv[(ulong) idx * C + c] : oldv[(K - 1) + idx];
        }
    } else {
        for (int k = 0; k < K - 1; ++k) v[k] = qkv[(ulong) (t_first - (K - 1) + k) * C + c];
    }

    for (int t = t_first; t < t_end; ++t) {
        const float x = qkv[(ulong) t * C + c];
        float acc = 0.0f;
        for (int k = 0; k < K - 1; ++k) acc += v[k] * w[(ulong) c * K + k];
        acc += x * w[(ulong) c * K + K - 1];
        for (int k = 0; k < K - 2; ++k) v[k] = v[k + 1];
        v[K - 2] = x;
        out[(ulong) t * C + c] = silu(acc);
    }
}

// The gated delta rule for T tokens in one launch: k_gdn with the token loop inside.  The state rows stay
// at the same device addresses across the tokens (L1-hot) and every token's arithmetic - the q/k RMS
// norms' block reductions included - is exactly k_gdn's, so the states match the token-by-token run.
kernel void k_gdn_chunk(device const float* conv [[buffer(0)]],
                        device const float* alpha [[buffer(1)]],
                        device const float* dt_bias [[buffer(2)]],
                        device const float* ssm_a [[buffer(3)]],
                        device const float* beta_raw [[buffer(4)]],
                        device float* S [[buffer(5)]],
                        device float* out [[buffer(6)]],
                        constant int& HK [[buffer(7)]],
                        constant int& D [[buffer(8)]],
                        constant int& HV [[buffer(9)]],
                        constant int& C [[buffer(10)]],
                        constant float& eps [[buffer(11)]],
                        constant int& T [[buffer(12)]],
                        uint h [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]],
                        ushort sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sh_q[128];
    threadgroup float sh_k[128];
    threadgroup float sh_red[4];

    const int kh = h % HK;
    for (int t = 0; t < T; ++t) {
        device const float* conv_t = conv + (ulong) t * C;
        const float qi = conv_t[kh * D + tid];
        const float ki = conv_t[HK * D + kh * D + tid];
        const float vi = conv_t[2 * HK * D + h * D + tid];

        float qn = block_sum(qi * qi, sh_red, tid, lane, sgid);
        float kn = block_sum(ki * ki, sh_red, tid, lane, sgid);

        sh_q[tid] = qi * rsqrt(qn + eps) * (1.0f / sqrt(float(D)));
        sh_k[tid] = ki * rsqrt(kn + eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float g = exp(softplus(alpha[(ulong) t * HV + h] + dt_bias[h]) * ssm_a[h]);
        const float b = sigm(beta_raw[(ulong) t * HV + h]);

        device float* Si = S + ((ulong) h * D + tid) * D;
        float dot_k = 0.0f;
        for (int j = 0; j < D; ++j) {
            Si[j] *= g;
            dot_k += Si[j] * sh_k[j];
        }
        const float d = (vi - dot_k) * b;
        float o = 0.0f;
        for (int j = 0; j < D; ++j) {
            Si[j] += d * sh_k[j];
            o += Si[j] * sh_q[j];
        }
        out[(ulong) t * HV * D + (ulong) h * D + tid] = o;
        // the next token rewrites sh_q / sh_k: every thread's reads of this token's must finish first
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// GDN output gating for T tokens: k_gated_norm over grid (T, HV).
kernel void k_gated_norm_chunk(device float* o [[buffer(0)]],
                               device const float* w [[buffer(1)]],
                               device const float* z [[buffer(2)]],
                               constant int& D [[buffer(3)]],
                               constant int& ZV [[buffer(4)]],
                               constant float& eps [[buffer(5)]],
                               uint3 g [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]],
                               ushort sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sh[4];
    const int t = (int) g.x, h = (int) g.y;
    device float* oh = o + (ulong) t * ZV + (ulong) h * D;
    float v = oh[tid];
    float s = block_sum(v * v, sh, tid, lane, sgid);
    oh[tid] = v * rsqrt(s / float(D) + eps) * w[tid] * silu(z[(ulong) t * ZV + (ulong) h * D + tid]);
}

// Attention prep for T tokens: k_attn_prep over grid (T, H + HKV); q / gate rows land at (t * H + head),
// the split-K kernels' row order; k / v land in the shared KV cache at position pos0 + t.  Buffer indices
// are the args' positions (the runtime binds them positionally).
kernel void k_attn_prep_chunk(device const float* qg [[buffer(0)]],
                              device const float* kx [[buffer(1)]],
                              device const float* vx [[buffer(2)]],
                              device const float* qw [[buffer(3)]],
                              device const float* kw [[buffer(4)]],
                              device float* q [[buffer(5)]],
                              device float* gate [[buffer(6)]],
                              device float* kc [[buffer(7)]],
                              device float* vc [[buffer(8)]],
                              constant int& H [[buffer(9)]],
                              constant int& HKV [[buffer(10)]],
                              constant int& D [[buffer(11)]],
                              constant int& n_rot [[buffer(12)]],
                              constant int& pos0 [[buffer(13)]],
                              constant int& ctx [[buffer(14)]],
                              constant float& base [[buffer(15)]],
                              constant float& eps [[buffer(16)]],
                              uint3 g [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]],
                              ushort sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sh[4];
    threadgroup float buf[256];
    const int t = (int) g.x;
    const uint head = g.y;
    const bool isq = head < (uint) H;
    const int pos = pos0 + t;
    device const float* src = isq ? qg + (ulong) t * H * 2 * D + (ulong) head * 2 * D
                                  : kx + (ulong) t * HKV * D + (ulong)(head - H) * D;

    float s = 0.0f;
    for (int i = tid; i < D; i += 128) {
        float val = src[i];
        s += val * val;
    }
    s = block_sum(s, sh, tid, lane, sgid);
    const float r = rsqrt(s / float(D) + eps);

    for (int i = tid; i < D; i += 128) {
        buf[i] = src[i] * r * (isq ? qw[i] : kw[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* dst = isq ? q + (ulong) (t * H + (int) head) * D
                            : kc + ((ulong)(head - H) * ctx + pos) * D;
    const int half_rot = n_rot / 2;
    for (int i = tid; i < D; i += 128) {
        float v = buf[i];
        if (i < n_rot) {
            int p = (i < half_rot) ? i : i - half_rot;
            float theta = float(pos) * pow(base, -2.0f * float(p) / float(n_rot));
            float cs = cos(theta), sn = sin(theta);
            v = (i < half_rot) ? buf[i] * cs - buf[i + half_rot] * sn : buf[i - half_rot] * sn + buf[i] * cs;
        }
        dst[i] = v;
    }
    if (isq) {
        for (int i = tid; i < D; i += 128) {
            gate[(ulong) (t * H + (int) head) * D + i] = qg[(ulong) t * H * 2 * D + (ulong) head * 2 * D + D + i];
        }
    } else {
        for (int i = tid; i < D; i += 128) {
            vc[((ulong)(head - H) * ctx + pos) * D + i] = vx[(ulong) t * HKV * D + (ulong)(head - H) * D + i];
        }
    }
}

