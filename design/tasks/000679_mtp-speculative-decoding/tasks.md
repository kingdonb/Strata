# Implementation Tasks: Metal MTP Speculative Decoding (--mtp 2)

All work in `metal/` + one section of `docs/QWEN36_METAL.md`. Reference:
`src/qwen35/q36_engine.cu` (`spec_step` L2688, `forward_mtp` L2133, `commit` L2149),
`metal/src/engine/q36_metal.cpp`.

- [x] Parse `--mtp N` in `run()` (integer drafts, 0/absent = off, cap 16); refuse `--mtp N > 0` when `g.n_mtp_layers == 0` with a clear message
- [x] Load the `blk.40` MTP block when `mtp_d > 0`: stop skipping it in the wrap loop, wrap 2-D tensors no-copy, F32-convert the 1-D vectors (enorm, hnorm, shared_head_norm, block norms); fail loudly on a missing tensor (`Model::W`/`F` exit with the name)
- [x] Allocate MTP state when `mtp_d > 0`: the block's `mtp_kc`/`mtp_vc` cache, per-GDN-layer window scratch states `S_win`/`conv_win`, scratch buffers (`mtp_cat` 2N, `hmtp`, `hlast`, `hmain_w`, `mtp_logits`, `logits_w`); create the chunk scratch via `init_prompt` with CT = max(prefill_chunk, D+1)
- [x] Extend `reset()` and `fill_scratch()` to the new buffers; add the MTP block's bytes to the startup working-set refusal estimate
- [x] Implement `forward_mtp_row(tok, pos, seed)`: enorm/hnorm RMSNorms into the 2N concat (via buffer-offset args), `eh_proj` GEMV, `attn_norm` RMSNorm, then `forward_token`'s QSA dispatch sequence against `mtp_kc`/`mtp_vc` at `pos`, then `shared_head_norm` RMSNorm; no `s.pos` advance
- [x] Implement `mtp_draft(nx, fix, D)`: the redo-row call (fix rows seeded from `hmain_w` 0..|fix|-1, then (hlast, nx) at pos) + chained single-row calls seeded from the block's own output; head GEMV + host argmax on the last row → drafts
- [x] Implement `mtp_verify(drafts)`: the checkpoint copies (real S/conv → `S_win`/`conv_win` via `k_gather_t` flat copies), the D+1-row window through `forward_prompt_chunk` with the GDN/conv dispatches on the window scratch states, then the per-row out-norm + (D+1)-row head GEMV + host argmax → picks and the accepted count a
- [x] Implement `mtp_commit(n_keep)`: for every GDN layer re-run the kept rows' conv history + recurrence (`p_conv_c`/`p_gdn_c` over the window's saved qkv/alpha/beta rows) into the real S/conv; set `hlast` = `hmain_w[n_keep-1]`; `pos += n_keep`
- [x] Wire the CLI generation loop (`--tokens`): the emit → draft → verify → kept = emitted → commit flow, `pos + D + 1 <= ctx` guard with plain-decode fallback, gen clamp, and the offered/accepted stderr line
- [x] Wire the serve decode loop: greedy requests only (temperature ≤ 0.01), per-emitted-token stop checks (EOS ids, cancel, max_new, ctx), kept = emitted, `DONE` line format unchanged
- [ ] Verify the base path is byte-identical on the M2: `--mtp 0`/absent tokens equal the pre-change build's base run on the 10 parity prompts (code-side the no-MTP path is untouched: same skip, same loops, same buffers, same dispatch sites; the run confirms it)
- [ ] Run the gates on the M2 (same Mac, same day, same build): `q36_compile_test`; `q36_chunk_parity` all-OK; `--mtp 2` vs `--mtp 0` token-identical via `q36_solo_gate.py` + `q36_cmp_gate.py`, 10/10, run twice (commands in design.md)
- [~] Measure decode tok/s base vs `--mtp 2` and the draft accept rate on the M2; the docs section (method + commands) is written in `docs/QWEN36_METAL.md` — the measured numbers land there when the gates run; the flag stays opt-in (no default flip in this PR)
