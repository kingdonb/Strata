# Implementation Tasks: Qwen3.6 Metal Long Context, Chunked Prefill and MTP

## STATUS (round 9, where we are)

Goal: build on the merged Qwen3.6 Metal port (12.5-14.5 tok/s, 8K ctx cap, token-by-token prefill) the four
follow-ups from the hand-off: split-K attention (32K+ ctx), chunked prefill (fast TTFT), MTP speculative
decoding, FP16 KV cache. Branch `feat/qwen36-metal`, everything verified on the user's M2 (sandbox = Linux).

| Phase | Scope | State |
|---|---|---|
| A split-K attention | ctx past 8K, 32K runs | **DONE, Mac-verified** (10/10 token-identical, 32K OK) |
| B chunked prefill | 512-token chunks, bit-identical, then default flip | **round 10: fix implemented, Mac verification pending.** Round 9 cuts decoded: G1-G7 bit-exact; G8 (routed experts) diverges ~5e-7 - the chunk's k_expc (plain-gemv order) vs the decode's k_gemv_lpr8 (deq_dot_block + butterfly) at n_in=2048, never stage-compared; G9/G10 downstream; G9's huge size also the down expc reading the placed-ordered eh through the entry-index table (a second bug). Fixed: k_gather_t + the per-entry decode-lpr8 for the gate/up (exact by construction, (t,k)-ordered eh, which also makes srt_in_d correct) + the identity map for non-lpr8 types + the FF==2048 down edge + stage 16 (the full MoE at n_in=2048, chunk vs decode). G0 (embedding) still diverges unexplained (-2.0214 vs -0.8025 while G1-G7 match) - r0/rl diagnostics added to settle it |
| C MTP (--mtp 2) | blk.40 draft + verify, 1.2x+ decode | not started (gated on B) |
| D FP16 KV / sampling | opt-in kv-f16, sampling profile, docs | not started |
| wrap-up | docs/QWEN36_METAL.md numbers, full gates, PR | not started |

Mac rounds spent on B: ~9. Each eliminated a hypothesis: serial-deq kernels (dead), the 4-dispatch atomic
table chain race (FOUND + FIXED with k_tables), all kernels proven (34/34), divergence pinned to the MoE
experts (round 9 cuts). Next round = the MoE fix + embtype/raw-G0 printout.

All work on branch `feat/qwen36-metal`, built and tested on the Mac (`/opt/homebrew/bin/cmake --build
build-metal`). No CUDA/HIP/SYCL files are touched.

Reviewer answers (approved design): M2 is the only test Mac (user runs intermediary tests on pushed
intermediates); do what upstream did on the prefill default (flip after measured gates — default flip lands
only after the Mac run passes); solo-path MTP with fixed `--mtp 2` is fine; include the opt-in FP16 KV cache;
host-sampling check in scope (GPU argmax / top-k prefilter only if profiling shows ≥2 ms/token).

NOTE (sandbox constraint): this sandbox is Linux — nothing Metal compiles or runs here. Every phase ends with
a push to `feat/qwen36-metal` and a request to the user to build + run the phase's gates on the M2 before the
next phase proceeds. Host C++ gets a `g++ -fsyntax-only` pass where headers allow it; MSL is checked by
careful mirroring of proven kernels and by the Mac compile test.

## Phase A — Split-K attention, uncap the context window

(pushed through dd0b0b8c; Mac round 2 results: **compile gate passed** — all 30 pipelines on the M2
(macOS 26.6.2); **split-K attention 10/10 token-identical** vs k_attn — Phase A verified; **32K ctx runs and
generates** (READY 32768, no crash). **Chunked prefill 0/10 — every prompt differs at generated token 0**: a
systematic bug in one of the chunk kernels. Also learned: the first request after load pays seconds of
shader-JIT/clock warmup (p00's prompt 12.5 s vs p01's 1.9 s — the CUDA engine's warmup() exists for this;
add one). Short parity prompts (~30 tokens) can't show the chunk speedup either way — the speed gate needs
2K+ prompts once parity lands. Next: `metal/tests/q36_chunk_parity.cpp` (pushed) runs every chunk stage
against its decode kernel and names the diverging stage in one Mac round.)

- [x] Add `k_attn_part` + `k_attn_merge` to `metal/kernels/q36.metal`, copying the partial layout
      (256-cell chunks, acc/m/l) from `src/qwen35/q36_engine.cu` (~L298-355) — `k_attn_sk_part` /
      `k_attn_sk_merge` pushed in 65037a91
- [x] Wire dispatch in `metal/src/engine/q36_metal.cpp`: split-K only when `n_pos > 8192`; keep `k_attn`
      default below; add a test-only env knob to force split-K at short ctx — `STRATA_Q36M_ATTN_SK=1`
- [x] Allocate partials/merge buffers in `State`; row-block the partials so prompt use stays bounded —
      `SK_ROWS_MAX = 256` rows per dispatch, decode uses `n_head`
- [x] Startup ctx validation: refuse a context that KV + S + conv state + scratch cannot hold, printing the
      numbers; leave `READY <ctx> stop` and `--ctx` parsing unchanged — enforced only above 8,192;
      `STRATA_Q36M_MEM_OK=1` overrides
- [x] Extend `metal/tests/q36_compile_test.cpp` with the new pipelines
- [~] Verify on the Mac (user runs): 10 parity prompts greedy 128 unchanged at ctx ≤ 8192; forced split-K
      token-identical to `k_attn` at 4K; 2K-token prompt + 128 gen at ctx 16384 and 32768; record per-step
      GPU ms at 4K/16K/32K
- [ ] Verify on the Mac: 10 parity prompts greedy 128 unchanged at ctx ≤ 8192; forced split-K token-identical
      to `k_attn` at 4K; 2K-token prompt + 128 gen at ctx 16384 and 32768; record per-step GPU ms at 4K/16K/32K

## Phase B — Chunked prompt prefill

- [x] Batched dense GEMM kernels (T rows, weights read once per chunk) — implemented as weight-stationary
      batched GEMVs (`k_gemv_c`, `k_gemv_lpr8_c` for n_in=2048) with the token loop inside the threadgroup:
      simpler than a tiled GEMM and each (token, output) keeps the decode GEMV's exact summation order, so
      chunk results are bit-identical to token-by-token rather than merely close
- [x] Grouped MoE chunk GEMM: entry tables on GPU (`k_zero`/`k_hist`/`k_exp_off`/`k_place`, device atomics),
      per-expert runs walked by `k_expc` (each expert read once per chunk), combine via `k_moe_sum_chunk`
      reading through the per-token slot table (fixed slot order, so placement order cannot change digits)
- [x] GDN chunk kernels: `k_conv_chunk` (16-token tiles side by side, tile 0 stores the history — the CUDA
      engine's k_conv_seq tile mode, k_conv's math) + `k_gdn_chunk` (delta rule sequential per token in one
      launch, k_gdn's math, states L1-hot) + `k_gated_norm_chunk`
- [x] QSA chunk attention: `k_attn_prep_chunk` (rows in the split-K kernels' row order) + the Phase A
      split-K kernels row-blocked (SK_ROWS_MAX rows per dispatch); split-K kernels gained a `row0` arg
- [x] `forward_prompt_chunk(T)` + `--prefill N` / `STRATA_Q36M_PREFILL=1` (512); prompts shorter than the
      chunk run as one partial chunk; without the flag the engine behaves exactly as before (the default
      flip lands only after the Mac parity + speed gates pass — reviewer: do what upstream did)
- [x] Head/logits only via the last prompt token's decode window; prefix-cache state reuse untouched
- [~] Parity + measurement (user, on the Mac): **round 7 verdict: the unit/chain parity is 34/34 EXACT (all
  kernels + the full MoE chain one-graph-vs-many + k_tables-vs-old-chain) — yet the engine ladder stays 0/10
  (--prefill 512 diverges at token 0 on ~33-token partial chunks).** Static audit of forward_prompt_chunk vs
  forward_token found NO glue bug: state handoff correct (chunk ingests [0,last), the last token runs the
  decode step once — no double update), every arg order matches the stage-tested shapes, conv/gdn/gated-norm
  are T-robust, push_c_gemv/push_c_expc mirror the tested dispatch shapes, k_route_chunk ≡ k_topk_router
  algorithmically, run_graph has no dispatch caps, no device atomics left in the chunk path. So round 8 =
  LOCALIZE with the new per-layer trace (f54f90ad): STRATA_Q36M_TRACE=1 runs per-layer sub-graphs in BOTH
  paths and prints row-sum checksums (TRACE D per prompt token, TRACE C per chunk, %.17g doubles summed in
  identical order — equal states ⇒ bit-equal sums). Tools: `tools/qwen35/q36_trace_cmp.py` diffs the two
  logs and names the first diverging layer. Round 8a: the trace never fired because the gate sends the
  ENGINE's stderr to /dev/null without `--log` (q36_harness.py spawns the engine with stderr=DEVNULL by
  default; the shell's 2> only catches the gate's own) — env propagates fine (c2477455 fixed the script's
  hints). **Round 8b results (Mac): LAYER 0 ALREADY DIVERGES, hugely (chunk x=19.74 vs decode x=8.85 — not
  ulp drift; 40/40 layers differ from there on).** And a SECOND structural fact: the TRACE runs (per-layer
  sub-graphs) produce DIFFERENT chunk texts than round 7's one-graph runs (p01 differs at 0 now, at 3
  before; several stop lengths changed) — so the one-graph chunk path ALSO has a cross-dispatch hazard the
  per-layer structure doesn't, OR vice versa; two bugs may be stacked. The blind spot identified: stage 14
  compares the chain against ITSELF, so a shared-kernel bug at new grid shapes (p_rms at grid nt, p_add,
  the lpr8 gemv at grid[1]=nt with real types/dims, k_emb_chunk with the REAL emb type — only Q8_0 was
  stage-tested) passes every stage while corrupting both sides identically. Checked and cleared by reading:
  k_rms is grid-correct (x+gid*n), k_row/k_emb_chunk write identical values, k_route_chunk ≡ k_topk_router.
  **Round 9 = cut-level trace (5b0c1d54):** TRACE cuts G0-G10 inside layer 0 of BOTH paths (emb → the 4
  projections → conv+state → gdn+S → norm/out → add → post-norm → router/route → routed-swiglu →
  down/shared/combine → moe), q36_trace_cmp.py now compares cuts (state fields cst/sst snapshotted at the
  chunk's last token; activations summed in identical order; embtype printed on the G0 lines = the
  token_embd GGUF type id). **Mac round 9, ask user:** `git pull && cmake --build build-metal`, rerun the
  two trace gates (WITH --log) + trace_cmp (now prints LAYERS: and CUTS: verdicts) + the json cmp. The
  first diverging cut names the dispatch; if all cuts match, the bug is one-graph-structural only.
  **Round 9 verdict (decoded):** the script prints only diverging fields, and the chunk log's SECOND G
  sequence is the last-token decode window's cuts (1-token values against the whole-chunk reference -
  expected noise, ignore).  The chunk path's cuts: G1-G7 bit-exact; **G8 (routed experts) diverges ~5e-7**
  (5.9148414304 vs 5.9148444452) - last-ulp arithmetic, not placement; G9/G10 huge (downstream).  **G0 (the
  embedding) also diverges (-2.0214 vs -0.8025) while G1-G7 match** - paradoxical (the rms cannot remove a
  scale bit-exactly: eps sits inside the rsqrt argument).  The engine's G0 line now also prints the
  chunk's first/last row sums (r0/rl) and trace_cmp prints a G0 note (both embtypes, the decode's
  first/last token values, the token count) to settle whether the rows differ or the comparison is off.
  **Root cause of G8 (proven by the dispatch audit):** the decode's gate/up experts at n_in=2048 run
  k_gemv_lpr8 (8 lanes x deq_dot_block over threadgroup-cached x + an 8-lane butterfly); the chunk's
  k_expc mirrors the PLAIN k_gemv (32-lane strided units + simd_sum).  No stage ever compared expc vs
  lpr8 (stage 5 = lpr8 vs lpr8; stages 7/7b = expc vs plain gemv at n_in=256).  push_c_gemv's comment
  already documented this class for the dense products (round 5) - the routed experts were missed.
  **The fix (round 10):** k_gather_t (pure row copies: the entries' input rows in (t,k) order) + the
  decode k_gemv_lpr8 per entry (grid[1]=n_ent, ids_c) for the lpr8-typed gate/up -> eg/eu/eh come out
  (t,k)-ordered with the decode's exact arithmetic; the down expc then reads eh through srt_in_d (the
  entry index), which is CORRECT for the (t,k) layout - the old placed-ordered eh vs entry-index lookup
  was a second bug.  Non-lpr8 gate/up types keep k_expc and the down reads via the new identity map
  (s.ident_c); FF==2048 with an lpr8-typed down goes per-entry too (tables skipped, the combine reads
  through the identity); the impossible mix (lpr8-typed down at n_ff 2048 with a non-lpr8 gate/up) is
  refused at startup.  Perf note: the gate/up re-read their weights per entry (the decode's traffic) -
  the down expc keeps its once-per-chunk read; if the long-chunk numbers demand it, an exact grouped
  k_expc_lpr8 is the measured follow-up.  **Stage 16 added** (the full MoE at n_in=2048 with the real
  expert formats: the chunk chain vs the decode chain per token) - would have caught this 9 rounds ago.
  **Mac round 10a:** compile gate green (k_gather_t in), stages 1-15 + **moe2048 eh OK** - the gather +
  per-entry decode-lpr8 gate/up are bit-exact (the G8 root cause fix works).  moe2048 x FAILED (16384
  diffs, first at 0): the stage's OWN sy0/gs0 buffers were one row/one float while the chunk combine
  indexes ys[t*n+i]/gs[t] for all T tokens - threads t>=1 read ~57KB past the allocation, the GPU fault
  discarded the whole combine dispatch's writes, x2 kept its initial values.  Fixed (868a3ca4) + an ey
  check added (decode (t,k) rows vs the chunk's placed rows through pos) to name down-vs-combine.  Also
  fixed: the compile test still listed the round-5-deleted k_gemv_lpr8_c_* (it died before its verdict).
  The ENGINE path is unaffected by that bug (its sgate_c/sy_c are CT-sized).  **Round 10b:** compile gate
  green (all 55 pipelines, incl. the four k_gemv_lpr8_*); parity 36/37 OK incl. **moe2048 eh + ey** (the ey
  check exonerates the down expc: all 65536 values match through pos2) - but moe2048 x STILL fails with
  byte-identical values to round 10a (first at 0: -0.13756679 vs -0.000248415978).  That identity means the
  old OOB reads were returning zeros (benign) and the real bug is elsewhere; and with the ey check forcing
  both combines' inputs term-identical (verified buffers, shared wts2, same slot order, kernel bodies read
  and confirmed), the two sums mathematically cannot differ - so one combine's writes never land and the
  mechanism is not visible statically.  Diagnostic added (1ec7a8e4): x3 = the pristine initial (neither
  combine touches it) + the host recomputes both sides' MoE sums in double from the verified buffers
  (cpuD/cpuC, identical by the ey check) and prints them against the GPU deltas (x - x3) for t=0 and t=7.
  Whichever delta is zero (or wrong) names the dead combine; the initial line also settles which side is
  the unmoved one.  Round 10c: rerun ONLY ./build-metal/q36_chunk_parity and paste the moe2048 block.

## Phase C — MTP speculative decoding

(Paused at the phase gate: A+B await the Mac verification round before MTP builds on the chunk machinery.
The design is settled — CUDA's spec_step mirrored: redo-row draft call (redo = the last window's accepted
drafts, seeded by that window's hmain rows), chained draft seeded by the block's own shnorm output, verify
window [x, d1..d2] with per-row head + argmax, accept-prefix rewind = checkpoint S/conv + re-run kept rows
through the chunk path (bit-identical), the MTP block as an attention layer with its own KV cache.
Greedy-only, `--mtp 2`, off by default; drafts skipped when produced + 3 > max_new or near ctx end, so a
burst is never cut mid-way and no stop-race rewind is needed in v1.)

- [ ] Load `blk.40` nextn weights (`nextn.enorm/hnorm/shared_head_norm`, `eh_proj`, block layer weights) when
      `--mtp > 0`; keep the current skip when off
- [ ] MTP draft step: one MTP-layer forward proposing up to 2 tokens
- [ ] Verify window (≤3 rows) through the chunk machinery: accept prefix, resample at first mismatch;
      stop tokens, `STOP` cancel, length and ctx-end guards respected
- [ ] Gate: `--mtp 2` greedy text identical to `--mtp 0` on the 10 parity prompts; measured decode tok/s ratio
      on the same Mac/day; keep opt-in unless ≥1.2x measured

## Phase D — Measured follow-ups (decide on numbers)

- [ ] Profile host sampling per token (greedy vs temp+top-p, ~248K vocab); if ≥2 ms/token, add GPU argmax /
      top-k prefilter preserving the RNG stream and chosen tokens, re-verify sampled parity with fixed seed
- [ ] Optional opt-in FP16 KV cache (`--kv-f16`) with fp32 unpack in the attention kernels; document the text
      difference and measured memory numbers; never a default
- [ ] Fix or restore the `./run-q36.sh` reference in docs/QWEN36_METAL.md §5

## Wrap-up

- [ ] Update docs/QWEN36_METAL.md: measured Metal numbers (what, hardware, date) replacing the hand-off's
      projections; note the associative-scan and AMX out-of-scope rationale
- [ ] Full gate run on the Mac green (build, `q36_compile_test`, parity gates through `tools/qwen35/`) before
      review; confirm CUDA/HIP builds untouched (or rebuilt if any shared header was edited)
