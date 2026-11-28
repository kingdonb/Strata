# Two picture readers: what the fork gives up for upstream's

*Date: November 28, 2026*  
*Authors: Kingdon Barrett & Claude (Opus 5.5)*  

---

The fork is retiring its own CUDA picture reader for Inkling-2-Mini, because upstream Strata 0.2.23 ships one that does the same job. The two readers run the same network and agree on what a picture says. Where they differ is when they take memory from the graphics card, and that difference lands on exactly the kind of machine the fork was built for: a 16 GB card.

The fork wrote its reader in November for chudmax, an RTX 5060 Ti 16 GB with an i5-10400. It shipped in four fork releases, from v0.2.20-mbp.rtx.3 to v0.2.22-mbp.rtx.1. On 2026-11-28 the owner chose to follow upstream, so the fork now carries upstream's Inkling engine unchanged. This is the accounting of what goes with it.

## Why the fork built one

In November, upstream's CUDA engine had no picture reader at all. Every picture went to the CPU reference reader, inside the engine's only request slot. On chudmax one 2,000×1,088 screenshot took more than 468 seconds on all 12 threads, and nothing else could be answered meanwhile. The agent harness saw that as a hang.

So the fork wrote `cuda/ink_cuda_vision.cuh`, about 550 lines of CUDA and cuBLAS. It shipped in engine 0.1.66.1 and was carried forward in 0.1.67.1 and 0.1.69.1. The same screenshot then read in 1.4 seconds on the card.

## Side by side

Both are a 23-layer vision transformer on cuBLAS, and both hand a picture to the CPU rather than end the engine. They part ways on where the weights live and who decides to use the card.

| | The fork's (`ink_cuda_vision.cuh`, 0.1.66.1–0.1.69.1) | Upstream's (`ink_vision_cuda.cuh`, 0.1.70) |
| --- | --- | --- |
| Weights between pictures | In host RAM, read once; copied to the card for each picture, then freed | On the card from the first picture; taken off after 120 s without one (one GPU) |
| Card memory at start | None held back | Weights plus a 1,024-token picture's buffers plus 64 MiB, held back from the experts (about 1.1–1.3 GB) |
| Who picks the card | Every picture, if `cudaMemGetInfo` shows room plus a 256 MiB margin | A rule at start: 24 GB or more, or `--vision-reader gpu` |
| Matrix maths | BF16 cuBLAS `GemmEx`, FP32 accumulate | `tc2` by default on sm_80+ (two BF16 halves, about 1e-5 of FP32); `bf16` or `fp32` by choice, `fp32` below sm_80 |
| Attention | FP32, chunked to a 128 MiB budget | FP32, chunked to about 64 MiB of scores |
| When the card can't | CPU, after scaling to at most 256 tokens in the engine (`--vision-cpu-max-tokens`) | CPU, after Control scales it to at most 0.8 MP |
| Multi-GPU | Off (CPU) | Stays loaded once on the card |
| Parity check | `strata-inkling-vision-check`: cosine ≥ 0.99994, about 1% relative L2 | `strata-inkling-vit-check` (`ink_vit_check.cu`), plus a test of the 24 GB rule |

## The real difference: borrow or reserve

The fork's reader borrows. It costs nothing until a picture arrives. Then it checks free card memory, copies 0.63 GB of weights up, reads the picture and gives everything back. If the room isn't there, the picture goes to the CPU, made small first. The card belongs to the model at all other times.

Upstream's reader reserves. If it will use the card, the engine sets about 1.1–1.3 GB aside at start, out of what the experts would have had. That guarantees the room, so a picture never fails for lack of memory. Inkling's experts are the part of the model that runs fastest when they all sit on the card. On a 16 GB card, upstream judges that 1.3 GB is better spent on experts, so it reads pictures on the CPU there by default.

The fork's design came from a fact about chudmax: Inkling 0.1.66 grows its KV cache into whatever VRAM is free. A standing reservation would come straight out of the context or the experts. Borrowing meant pictures were nearly free on the card, and the model kept every byte the rest of the time. The cost was a rare picture read on the CPU when the cache had filled the card. Lowering the margin from 512 to 256 MiB fixed the case that had sent a 1,400-token picture to the CPU with the cache full.

## What chudmax gives up, and what it gets

With upstream's defaults, chudmax goes back to reading pictures on the CPU, at a smaller size. The fork's numbers were measured there on 2026-11-23, Inkling SQ-10 serving:

| Picture | Tokens | Fork: read on the card | Fork: answered, through the API |
| --- | --- | --- | --- |
| 2,880×1,800, 80k-token context already full | 3,240 | 4.6 s | 7.1 s |
| 2,000×1,088 screenshot | 1,400 | 1.4 s | 2.7 s |
| 1,000×544 | 350 | 0.2 s | 1.0 s |

When the card was full, the fork read on the CPU at 256 tokens, in about 27–30 s. Before the fork's reader, the 2,000×1,088 screenshot took more than 468 s.

Upstream's two choices on this card have not been measured here yet:

- **The default, the CPU reader.** Control first scales each picture to 0.8 MP, so a 2,000×1,088 screenshot arrives at about 1,210×660. Upstream's own #7043 measurement put 0.8 MP at about a minute, on a 6-core PC. Upstream's 0.2.23 MODELS.md says "a few seconds each". Which one holds on an i5-10400 is open.
- **`--vision-reader gpu`.** Pictures should read on the card at about upstream's 0.1–0.5 s, at full size. The price is about 1.3 GB less for the experts and the cache, every minute the model runs, picture or not. Whether SQ-10 at 80k, k8v4, 4 slots still keeps all its experts on the card is open.

What chudmax gets either way: one reader that upstream maintains, tests and fixes on every backend. The fork no longer has to rebuild the Inkling engine for each release.

## What upstream's has that ours never did

- **Warm weights.** It keeps the 0.63 GB of weights on the card between pictures, for 120 s. The fork copied them up for every picture, which was part of its 1.4 s.
- **Closer to the CPU reference.** Its `tc2` mode splits each value into two BF16 halves, landing within about 1e-5 of FP32. The fork's single BF16 pass was about 1% off in relative L2, though still at cosine 0.99994.
- **Older cards.** Below sm_80 it runs a plain FP32 path, so the same code serves Turing and older.
- **One rule on every backend.** NVIDIA, AMD, Intel and the Mac follow the same policy as Qwen3.8-Flash-Next. The engine reports its choice to Control (`INFO vision_reader=gpu|cpu`), and Control sizes and times pictures to match.
- **A choice you can see.** The engine logs which reader it picked and why, and `--vision gpu|cpu` in `strata launch` and the app maps straight onto it.
- **Several cards.** Once loaded, it stays on a multi-GPU setup. The fork's sent every picture to the CPU there.

## Worth keeping

The code goes, but two ideas are worth carrying forward. The fork's last version stays in git history at tag `v0.2.22-mbp.rtx.1`.

- **Borrowing as a third mode.** For a card under 24 GB, a mode between upstream's two: read on the card when there is room right now, else on the CPU, with no standing reservation. Upstream's reader already has the parts, `load()`, `encode()` and `release()`; it lacks the per-picture memory check in front of them.
- **Shrinking in the engine.** The fork capped CPU pictures by tokens inside the engine, which bounds the CPU reader's time directly. Upstream caps by pixels in the server, and its engine on its own only scales pictures above 4,096×4,096; a token cap ties the limit to the work the reader actually does.

What to measure on chudmax next, before deciding whether either is worth proposing:

- [ ] A 2,000×1,088 screenshot on upstream's default CPU reader: seconds from request to first token.
- [ ] The same with `--vision-reader gpu`, plus SQ-10's decode speed at 80k, k8v4 and 4 slots, against today's.
- [ ] Whether `--vision-reader gpu` still leaves every expert on the card in that preset.
