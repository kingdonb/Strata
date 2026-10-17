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

