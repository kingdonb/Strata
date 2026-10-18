# Design: Metal MTP Speculative Decoding (--mtp 2)

## The CUDA reference being mirrored

All semantics come from `src/qwen35/q36_engine.cu` (the verified CUDA engine for this family):

- **Weights** (`load()`, L1024–1096): `has_mtp = n_mtp_layers > 0` (from the required
  `qwen35moe.nextn_predict_layers` metadata; `g.n_layers = block_count - n_mtp_layers`, so the
  MTP block is `blk.40`). The block loads like any layer but is **forced to the full-attention
  path** (`y.attn = is_mtp`): `attn_q` (fused query + output gate), `attn_k/v/output`, `q/k_norm`,
  the MoE tensors, plus `nextn.eh_proj.weight` (2N×N), `nextn.enorm/hnorm/shared_head_norm` (N).
  It has **no recurrent state** — its state is an ordinary KV cache.
- **Draft call** (`forward_mtp`, L2133): per row — embed the token, RMSNorm by `enorm` into
  `cat[0:N]`, RMSNorm the seed hidden by `hnorm` into `cat[N:2N]`, `eh_proj` GEMV, RMSNorm by the
  block's `attn_norm`, run the block (attention at the row's position, writing its own KV cache,
  then MoE), RMSNorm the result by `shared_head_norm` → the block's output hidden (`hmtp`).
- **`spec_step(x, fix, D)`** (L2688–2747):
  1. *Redo-row draft call*: `T0 = |fix| + 1` rows — the drafts the last window accepted
     (`fix[i]`, seeded by that window's main-model hidden rows 0..|fix|-1) re-run at their
     original positions, then `(hlast, x)` at `pos`. Draft `d1` = argmax of the head on the last
     row's `hmtp`.
  2. *Chained drafts* `d2..dD`: one row each (`d_{i-1}` at `pos+i`), seeded by the **block's own
     output** hidden of the previous call's last row.
  3. *Verify window*: `[x, d1..dD]` = D+1 rows through the **main model**, where the GDN layers
     run in verify mode (S read, not written) and every row gets logits + greedy pick.
  4. *Accept*: `a` = the longest prefix of picks matching the drafts. `commit(a+1)` re-runs the
     kept rows' conv + recurrence into the real S (over the window's saved per-row qkv /
     alpha / beta inputs, `commit()` L2149). `hlast` = the kept last row's normed hidden;
     `pos += a+1`. Returns `a`; the caller emits the accepted drafts and feeds `picks[a]` as the
     next token (CLI loop L3932–3955).
- The window's rows compute bitwise the same values as per-token decode ("one token, a window or
  a chunk: the same kernels", L1907) — that is why MTP greedy text equals plain greedy text.

## Metal implementation (all inside `metal/`)

Files: `metal/src/engine/q36_metal.cpp` (host), `metal/kernels/q36.metal` (only if a kernel
tweak proves unavoidable — none expected), `docs/QWEN36_METAL.md` (numbers).

- **Flag** (`run()`, arg block L1230): `--mtp N` → `mtp_d`; 0/absent = off; refuse `N > 0` when
  `g.n_mtp_layers == 0` (the geometry reader already parses `nextn_predict_layers`).
- **Weights**: today the load loop skips every `blk.40` tensor (L1299). When `mtp_d > 0`, stop
  skipping: same no-copy wrapping for 2-D tensors, F32 conversion for 1-D vectors (enorm, hnorm,
  shared_head_norm, the block's norms). Metal has no hybrid-expert mode, so unlike CUDA the
  weights load only when drafts are on — the default path's memory and behavior stay identical.
- **State** (allocated when `mtp_d > 0`):
  - `mtp_kc` / `mtp_vc`: the draft block's KV cache (like a QSA layer's, `n_head_kv × ctx ×
    head_dim` F32, zeroed at alloc and by `reset()`).
  - Window scratch recurrent states `S_win[l]` / `conv_win[l]` for every GDN layer — the verify
    window's GDN/conv dispatches run against these, so the **real S/conv are never written by
    the window**: they stay at the accept point (this is the "checkpoint"), and a partial accept
    needs no restore copy.
  - Scratch: `mtp_cat` (2N concat), `hmtp` ((D+1)×N block outputs), `hlast` (N; initialized from
    the prompt's final normed hidden `s.h`), `hmain_w` ((D+1)×N window rows' out-normed hiddens —
    the next step's redo seeds), `mtp_logits` (V), `logits_w` ((D+1)×V).
  - The chunk scratch (`init_prompt`, L265, buffers L1424–1428) is created when `mtp_d > 0` too
    (CT = max(prefill_chunk, D+1)) — the verify window reuses the chunk kernels; `reset()` and
    `fill_scratch()` cover the new buffers.
- **`forward_mtp_row(tok, pos, seed)`**: one row through the block — the exact dispatch sequence
  of `forward_token`'s QSA branch (L529–557) prefixed by the enorm/hnorm/concat/eh_proj prologue
  and suffixed by the `shared_head_norm` RMSNorm, attention against `mtp_kc/vc` at `pos`, no
  `s.pos` advance. Rows of a call run sequentially (T0 ≤ 3), each writing its cache cell before
  the next row reads it.
- **Draft** (`mtp_draft(nx, fix, D)`): the redo-row call (rows seeded from `hmain_w` rows
  0..|fix|-1 and `hlast`), then the chained single-row calls seeded from `hmtp`. Head GEMV on the
  last row only, host argmax (same tie rule as `sample_next`: first max wins) → `drafts`.
- **Verify** (`mtp_verify(drafts)`): run `forward_prompt_chunk([x, d1..dD], D+1, pos)` with the
  GDN/conv dispatches pointed at the window scratch states; then one `p_rms` pass (grid = D+1)
  into `hmain_w` and a (D+1)-row head GEMV (`push_c_gemv` with `output.weight`) into `logits_w`;
  host argmax per row → `picks`, and `a` = the matching draft prefix. Reusing the chunk path is
  the load-bearing decision: its kernels are bitwise stage-identical to the decode path (T2's
  38-stage FNV parity), so a 3-row window costs ≈ one decode step and its picks equal the decode
  picks — that is what makes gate 1 hold.
- **Commit** (`mtp_commit(n_keep)`), mirroring CUDA `commit()`: for every GDN layer, re-run the
  kept rows (0..n_keep-1) of the window's saved inputs — `p_conv_c` over the saved `qkv_c` rows
  writing the real conv history, then `p_gdn_c` over the fresh conv outputs + saved
  `alpha_c`/`beta_c` rows writing the real S. Cheap (no projections/attention/MoE re-run; ≤ 3
  rows). Always run it (even on full accept) so the kept state is by construction exactly what a
  sequential decode would hold. Then `hlast = hmain_w[n_keep-1]`, `pos += n_keep`. The window's
  scratch states are simply left stale (every window rewrites them).
- **Decode loops** (CLI L1768 and serve L1665, greedy only — serve requests with
  `temperature > 0.01` keep the per-token path): the CUDA flow — emit the current token; if
  `mtp_d > 0` and `pos + mtp_d + 1 ≤ ctx`: draft → verify → **kept = the rows actually emitted**
  (stop rules checked per emitted token: EOS ids, `stop_flag`, `max_new`, ctx) → commit(kept) →
  `fix` = the emitted drafts, next token = `picks[kept-1]`; else plain `forward_token` step.
  Never emit past `max_new`; the shown-token state contract (criterion 5) follows from
  kept = emitted. Report `offered/accepted` on stderr at the end.

## Key decisions

1. **Mirror CUDA, don't invent** — the redo-row call, chained seeds, verify window and commit
   exist on CUDA with parity green; the Metal port copies their semantics (and their reasons),
   not just their shape.
2. **Window via the chunk kernels** — the only T-wide path; bitwise decode-exact per the T2
   parity harness; running 3 per-token steps instead would make MTP ~3× slower, not faster.
3. **Scratch-state checkpoint instead of memcpy** — the window writes `S_win`/`conv_win`, leaving
   the real S/conv at the accept point; no ~65–100 MB state copy per step (an M2-step-scale cost)
   and no kernel no-write flag needed. The commit re-run rebuilds the kept state from the
   pre-window states + the window's saved per-row inputs, so it is decode-exact even if the
   window's own state writes were not.
4. **Weights load only when `--mtp > 0`** — unlike CUDA's always-load (whose reason, the hybrid
   slot tables, does not exist on Metal); keeps the default path byte-identical.
5. **Host-side argmax for picks** — matches `sample_next`'s tie behavior; avoids a new kernel.
6. **The block's prompt-region cache cells stay zero** — mirrors CUDA (the block never runs
   during the prompt pass; drafts are proposals, the verify window decides).
7. **One change per PR** — everything in `metal/` + one docs section; no shared-header or
   CUDA/HIP/SYCL edits expected.

## Notes for the implementer (what recon found)

- `push_gemv`/`push_c_gemv` take dims from the tensor at dispatch (`GemvParam`), so `eh_proj`
  (n_in = 2N) needs **no new pipeline**; `mtl::Arg::B(buf, offset)` writes the two RMSNorm halves
  of `mtp_cat` without a concat kernel.
- `forward_prompt_chunk` already: writes KV per row at `pos0+t`, runs the decode path's per-row
  attention dispatch sequence, updates S/conv per token, and leaves `xc` = per-row pre-out-norm
  residual; it does **not** advance `s.pos` and does not run the head — both stay with the caller.
- The block's per-row forward can be assembled verbatim from `forward_token`'s QSA branch with
  `l = n_layers` indexing into the new `mtp_kc/vc`; the redo rows and chained rows differ only in
  their position/seed arguments.
- The `--mtp` flag is this family's **integer** draft count (the Flash-Next "draft dir" meaning
  belongs to the other engine; `src/program/generate.cpp` routes qwen35moe argv to
  `strata::qwen35::run` before its own parser — verified L2252).
- Extend the startup working-set refusal (L1459) by the MTP block's weights + cache + scratch so
  the gate stays honest.
- Old `metal/src/engine/window.cpp` MTP code (`mtp_draft`) is the **other** engine's (Flash-Next,
  directory-based `--mtp <dir>`); unrelated to this task — do not touch it.

## Gate commands

```bash
# build + compile test (M2)
cmake -S metal -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release -DMETAL_CPP_DIR=... && ninja -C build-metal
./build-metal/q36_compile_test

# base and MTP runs over the 10 parity prompts (serve protocol, greedy, 128 tokens)
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096" --out /tmp/q36-base.json
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096 --mtp 2" --out /tmp/q36-mtp.json
python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-base.json /tmp/q36-mtp.json   # 10/10 identical

# speedup: per-prompt decode tok/s from the two runs' DONE lines / solo_gate output, same Mac+day
```
