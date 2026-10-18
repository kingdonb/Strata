#!/usr/bin/env python3
# q36_trace_cmp.py <plain-log> <chunk-log>
#
# Localizes where the chunked-prefill state first leaves the token-by-token state, from two
# STRATA_Q36M_TRACE=1 runs of the same model/prompt set.  (--log is REQUIRED on the gate: the harness
# sends the ENGINE's stderr to /dev/null without it; the shell's 2> only catches the gate's own.)
#
#   STRATA_Q36M_TRACE=1 python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
#       --args "--native $Q36_MODEL --ctx 4096" --out /tmp/plain.json --log /tmp/trace-plain.log
#   STRATA_Q36M_TRACE=1 python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
#       --args "--native $Q36_MODEL --ctx 4096 --prefill 512" --out /tmp/chunk.json --log /tmp/trace-chunk.log
#   python3 tools/qwen35/q36_trace_cmp.py /tmp/trace-plain.log /tmp/trace-chunk.log
#
# Two comparisons, both bit-exact when the states match (rows summed in index order; the decode side
# prints per (token, cut), the script sums those over the chunk's token range in token order - the same
# accumulation the chunk side does):
#   * per LAYER (TRACE D / TRACE C lines): the first layer whose whole-layer state differs;
#   * per CUT inside layer 0 (TRACE G lines): emb -> projections -> conv -> delta rule -> norm/out ->
#     add -> post-norm -> router/route -> experts -> down/combine -> moe - the dispatch that diverges.
import re
import sys

D_RE = re.compile(r"TRACE D pos=(-?\d+) L=(-?\d+) x=(-?[\d.eE+-]+) h=(-?[\d.eE+-]+)")
C_RE = re.compile(r"TRACE C nt=(\d+) pos0=(-?\d+) L=(-?\d+) x=(-?[\d.eE+-]+) h=(-?[\d.eE+-]+)")
G_RE = re.compile(r"TRACE G(\d+) pos0?=(-?\d+) L=(-?\d+)(.*)")
SNAP_FIELDS = {"cst", "sst"}     # state snapshots: compared at the chunk's LAST token, not summed
SKIP_FIELDS = {"embtype", "r0", "rl"}   # printed for information on both sides, not a sum


def parse(path):
    reqs, cur = [], None
    for line in open(path, errors="replace"):
        if "TRACE BEGIN" in line:
            cur = []
            reqs.append(cur)
            continue
        m = D_RE.match(line)
        if m and cur is not None:
            cur.append(("D", int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))))
            continue
        m = C_RE.match(line)
        if m and cur is not None:
            cur.append(("C", int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)),
                        float(m.group(5))))
            continue
        m = G_RE.match(line)
        if m and cur is not None:
            fields = {}
            for tok in m.group(4).split():
                name, _, val = tok.partition("=")
                try:
                    fields[name] = float(val)
                except ValueError:
                    pass
            cur.append(("G", int(m.group(1)), int(m.group(2)), int(m.group(3)), fields))
    return reqs


def layer_compare(r0, c0):
    r0d = [e for e in r0 if e[0] == "D"]
    c0c = [e for e in c0 if e[0] == "C"]
    if not r0d or not c0c:
        print("layer compare: missing TRACE D/C lines (skipped)")
        return
    nt, pos0 = c0c[0][1], c0c[0][2]
    ref = {}
    for _, pos, l, x, h in r0d:
        if pos0 <= pos < pos0 + nt:
            a = ref.setdefault(l, [0.0, 0.0])
            a[0] += x
            a[1] += h
    bad = []
    for _, cnt, p0, l, x, h in c0c:
        a = ref.get(l)
        if a is None or x != a[0] or h != a[1]:
            bad.append(l)
    if bad:
        print(f"LAYERS: first diverging layer {bad[0]} ({len(bad)} of {len(c0c)} differ)")
    else:
        print("LAYERS: all bit-identical")


def cut_compare(r0, c0):
    c0c = [e for e in c0 if e[0] == "C"]
    cg = [e for e in c0 if e[0] == "G"]
    if not cg:
        print("cut compare: no TRACE G lines in the chunk log (skipped)")
        return
    if c0c:
        nt, pos0 = c0c[0][1], c0c[0][2]
    else:
        nt, pos0 = 10**9, 0
    last = pos0 + nt - 1

    # G0 (embedding) diagnostics: the chunk's first/last row sums against the decode's first/last token
    # values, plus the token_embd types - settles whether the embedding rows themselves differ (real) or
    # the compared row count / alignment is off (artifact).  NOTE: the chunk log's LAST-token decode window
    # also prints G cuts (pos = the last token); those are a 1-token value against a whole-chunk reference
    # and are expected to "diverge" - only the chunk-path cuts (the first G sequence, pos0-tagged) count.
    g0c = [e for e in cg if e[1] == 0]
    g0r = [e for e in r0 if e[0] == "G" and e[1] == 0 and pos0 <= e[2] <= last]
    if g0c and g0r:
        cf, rf = g0c[0][4], g0r[0][4]
        ct_ = int(cf["embtype"]) if "embtype" in cf else -1
        rt_ = int(rf["embtype"]) if "embtype" in rf else -1
        print(f"  G0 note: chunk embtype={ct_} x={cf.get('x', 0):.17g} r0={cf.get('r0', 0):.17g} "
              f"rl={cf.get('rl', 0):.17g}")
        print(f"  G0 note: decode embtype={rt_} first(pos={g0r[0][2]}) x={rf.get('x', 0):.17g} "
              f"last(pos={g0r[-1][2]}) x={g0r[-1][4].get('x', 0):.17g} tokens={len(g0r)}")

    acc, snap = {}, {}
    for e in r0:
        if e[0] != "G":
            continue
        _, tag, pos, _l, fields = e
        if not (pos0 <= pos <= last):
            continue
        for f, v in fields.items():
            if f in SKIP_FIELDS:
                continue
            if f in SNAP_FIELDS:
                if pos == last:
                    snap[(tag, f)] = v
            else:
                acc[(tag, f)] = acc.get((tag, f), 0.0) + v

    bad = []
    for e in cg:
        _, tag, _pos, _l, fields = e
        for f, v in fields.items():
            if f in SKIP_FIELDS:
                continue
            ref = snap.get((tag, f)) if f in SNAP_FIELDS else acc.get((tag, f))
            if ref is None:
                print(f"  cut G{tag} {f}: no decode reference")
                bad.append((tag, f))
            elif ref != v:
                print(f"  cut G{tag} {f}: DIVERGES  chunk {v:.17g}  decode {ref:.17g}")
                bad.append((tag, f))
    if not bad:
        print("CUTS: all bit-identical inside layer 0 - the divergence starts later (or downstream)")
        return
    cuts = sorted({t for t, _ in bad})
    print(f"CUTS: first diverging cut G{cuts[0]} ({len(bad)} fields differ)")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    plain, chunk = parse(sys.argv[1]), parse(sys.argv[2])
    if not plain or not chunk:
        sys.exit("no TRACE blocks - run the gate with STRATA_Q36M_TRACE=1 AND --log <file> (the engine's "
                 "stderr goes to /dev/null without --log; check the file for 'TRACE BEGIN' lines)")
    print(f"plain requests traced: {len(plain)}, chunk requests traced: {len(chunk)}")
    layer_compare(plain[0], chunk[0])
    cut_compare(plain[0], chunk[0])


if __name__ == "__main__":
    main()
