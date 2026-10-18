"""q36_prefill_bench.py --exe E --args "..." --out F.json [--sizes 2048,4096,8192] [--chunks 0,512,1024,2048]
[--rounds 3] [--gen 16] [--timeout 1800]: TTFT and prompt tok/s of the q36 engine's prompt reading,
token-by-token (--prefill 0) vs the chunk sizes, one engine start per chunk value.

Per (size, round) a deterministic seeded id draw - the engine's stop ids avoided, the seed differs per size
and round so the prefix matcher never reuses a prompt and hides the read (the tool refuses a cell whose DONE
reports a reused prefix).  One discarded warm-up per start (parity prompt 0: the documented ~10 s cold first
request).  TTFT = request sent -> first T line (client side).  Prompt tok/s = the DONE line's fresh tokens /
prompt_ms (end-of-prompt, the same number the server's request line reports as `prefill X tok/s`).  Prints a
markdown table, the per-size ratios against that size's token-by-token row and the >=5x gate verdict (the
no-flag `--prefill` default flips only if every measured size clears it - docs/QWEN36_METAL.md), and writes
every raw run to the JSON.  A start the memory gate refuses before READY is recorded as a row note.

  python3 tools/qwen35/q36_prefill_bench.py --exe ./build-metal/strata \
    --args "--native $Q36_MODEL --ctx 32768" --sizes 2048,4096,8192 --chunks 0,512,1024,2048 \
    --rounds 3 --gen 16 --out bench/results/2026-XX-XX-m2-q36-prefill/bench.json

Runtime on the M2: every cell reads its whole prompt, and the token-by-token rows run ~17 tok/s (an 8K
prompt is ~8 min), so the full default matrix is a couple of hours - cut --rounds or the size list for a
first look.
"""
from pathlib import Path
import argparse, datetime, json, random, shlex, statistics, sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from q36_harness import Eng, load_prompts

STOP_IDS = {248046, 248044, 151645, 151643}   # q36_metal.cpp's finish="stop" tokens

GATE = 5.0   # the flip gate: chunked must read >= GATE x the token-by-token prompt tok/s on every size


def gen_ids(size, seed):
    """Deterministic ids for one (size, round) cell.  The values do not change what the chunk kernels
    cost; what matters is that both paths read the same ids and no two cells share a prefix."""
    r = random.Random(seed)
    out = []
    while len(out) < size:
        t = r.randrange(1000, 248043)
        if t not in STOP_IDS:
            out.append(t)
    return out


def request(eng, ids, n, timeout):
    """GEN one request -> its numbers.  The DONE line is
    `DONE <produced> <prompt_len> <prompt_ms> <decode_ms> <finish> 0 0 <reuse> 0 0 0 0 0.0 <fresh> 0`."""
    t0 = time.time()
    eng.send("GEN %d %s" % (n, ",".join(map(str, ids))))
    ttft, pp = None, []
    while True:
        l = eng.line(timeout)
        if l is None:
            raise SystemExit("engine died mid-request (see the start's stderr log)")
        if l.startswith("T "):
            if ttft is None:
                ttft = time.time() - t0
        elif l.startswith("PP "):
            pp.append(l)
        elif l.startswith("DONE"):
            f = l.split()
            return {"produced": int(f[1]), "prompt_len": int(f[2]), "prompt_ms": float(f[3]),
                    "decode_ms": float(f[4]), "finish": f[5], "reuse": int(f[8]), "fresh": int(f[14]),
                    "ttft_s": round(ttft, 3) if ttft is not None else None, "pp": pp}
        elif l.startswith("ERR"):
            raise SystemExit("ERR: " + l)


def medians(rows, chunk, size):
    rs = [r for r in rows if r.get("chunk") == chunk and r.get("size") == size and not r.get("refused")]
    if not rs:
        return None
    return {"ttft_s": statistics.median([r["ttft_s"] for r in rs]),
            "prompt_tok_s": statistics.median([r["prompt_tok_s"] for r in rs])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--args", default="", help="engine args; this tool appends --prefill <chunk> itself")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="2048,4096,8192")
    ap.add_argument("--chunks", default="0,512,1024,2048", help="0 = the token-by-token baseline")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--gen", type=int, default=16, help="generated tokens per run (TTFT needs one)")
    ap.add_argument("--timeout", type=int, default=1800, help="per request, seconds")
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parent / "parity_prompts.ids"),
                    help="the warm-up prompt's source (prompt 0)")
    ap.add_argument("--log-dir", default=None, help="engine stderr logs (default: beside --out)")
    a = ap.parse_args()

    sizes = [int(x) for x in a.sizes.split(",")]
    chunks = [int(x) for x in a.chunks.split(",")]
    base_args = shlex.split(a.args)
    if "--prefill" in base_args:
        raise SystemExit("this tool appends --prefill itself; drop it from --args")
    ctx = next((int(base_args[i + 1]) for i, x in enumerate(base_args)
                if x in ("--ctx", "--max-context") and i + 1 < len(base_args)), None)
    if ctx is not None and max(sizes) + a.gen >= ctx:
        raise SystemExit("largest size (%d) + gen (%d) does not fit --ctx (%d): raise --ctx"
                         % (max(sizes), a.gen, ctx))

    warm = load_prompts(a.prompts)[0]
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(a.log_dir) if a.log_dir else out_path.parent
    rows = []

    def save():
        json.dump({"exe": a.exe, "args": a.args, "sizes": sizes, "chunks": chunks, "rounds": a.rounds,
                   "gen": a.gen, "gate": GATE, "date": datetime.date.today().isoformat(), "rows": rows},
                  open(out_path, "w"), indent=1)

    for c in chunks:
        logf = log_dir / ("engine-prefill-%d.log" % c)
        try:
            eng = Eng(a.exe, base_args + ["--prefill", str(c)], log=str(logf))
        except SystemExit as e:   # started but never reached READY (e.g. the startup memory gate)
            rows.append({"chunk": c, "refused": True, "note": str(e), "log": str(logf)})
            save()
            print("chunk %d: REFUSED before READY (engine-prefill-%d.log)" % (c, c), flush=True)
            continue
        request(eng, warm, 8, a.timeout)          # the cold first request, discarded
        print("chunk %d: warm-up done (load %.1fs)" % (c, eng.load_s), flush=True)
        for size in sizes:
            for rnd in range(a.rounds):
                ids = gen_ids(size, (size << 8) | rnd)
                r = request(eng, ids, a.gen, a.timeout)
                if r["reuse"] != 0 or r["fresh"] != len(ids):
                    raise SystemExit("cell (size %d round %d): the engine reused %d of %d prompt tokens - a "
                                     "repeated prompt reached the prefix matcher; rerun this cell"
                                     % (size, rnd, r["reuse"], len(ids)))
                row = dict(r, chunk=c, size=size, round=rnd,
                           prompt_tok_s=r["fresh"] / (r["prompt_ms"] / 1000.0) if r["prompt_ms"] > 0 else 0.0,
                           decode_tok_s=r["produced"] / (r["decode_ms"] / 1000.0) if r["decode_ms"] > 0 else 0.0)
                rows.append(row)
                save()
                print("chunk %d size %d round %d: ttft %.1f s, prompt %.1f tok/s, decode %.1f tok/s"
                      % (c, size, rnd, r["ttft_s"] or 0.0, row["prompt_tok_s"], row["decode_tok_s"]), flush=True)
        eng.close()

    # the table: medians of the rounds, ratio against the same size's chunk-0 row
    if 0 not in chunks or medians(rows, 0, sizes[0]) is None:
        print("no token-by-token baseline: the >=%.0fx gate cannot be evaluated this run" % GATE, flush=True)
    L = ["| Prompt tokens | --prefill | TTFT ms | prompt tok/s | vs token-by-token |",
         "|---:|---:|---:|---:|---:|"]
    for size in sizes:
        base = medians(rows, 0, size)
        for c in chunks:
            m = medians(rows, c, size)
            if m is None:
                continue
            ratio = m["prompt_tok_s"] / base["prompt_tok_s"] if base and base["prompt_tok_s"] > 0 else None
            label = "0 (token-by-token)" if c == 0 else str(c)
            L.append("| %d | %s | %.0f | %.1f | %s |"
                     % (size, label, m["ttft_s"] * 1000.0, m["prompt_tok_s"],
                        "%.2fx" % ratio if ratio else "-"))
        if base is None:
            L.append("| %d | (baseline refused/absent) | - | - | - |" % size)
    table = "\n".join(L)
    print("", table, sep="\n", flush=True)

    # the gate, per chunk: the no-flag default may flip to a chunk only if that chunk clears >= GATE x
    # on every measured size (and passes the 10-prompt parity gate - checked separately, docs/QWEN36_METAL.md)
    chunk_ratios, notes = {}, []
    for c in chunks:
        if c == 0:
            continue
        ms = [medians(rows, c, s) for s in sizes]
        bs = [medians(rows, 0, s) for s in sizes]
        if any(m is None or b is None or b["prompt_tok_s"] <= 0 for m, b in zip(ms, bs)):
            notes.append("chunk %d: incomplete (a row is missing, its start was refused, or the baseline "
                         "is absent) - not a flip candidate this run" % c)
            continue
        rs = {s: m["prompt_tok_s"] / b["prompt_tok_s"] for s, m, b in zip(sizes, ms, bs)}
        chunk_ratios[str(c)] = rs
        notes.append("chunk %d: %s -> %s" % (c, ", ".join("%d: %.2fx" % (s, rs[s]) for s in sizes),
                    "clears >=%.0fx on every size" % GATE if all(v >= GATE for v in rs.values())
                    else "does not clear - the default stays opt-in"))
    verdict = "; ".join(notes) if notes else "no chunk completed all sizes"
    print("", "Gate (>= %.0fx on every measured size): %s" % (GATE, verdict), flush=True)
    json.dump({**json.load(open(out_path)), "table_md": table, "chunk_ratios": chunk_ratios,
               "gate_verdict": verdict}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
