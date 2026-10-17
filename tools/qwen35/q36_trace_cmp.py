#!/usr/bin/env python3
# q36_trace_cmp.py <plain-log> <chunk-log>
#
# Localizes the first layer where the chunked-prefill state leaves the token-by-token state, from two
# STRATA_Q36M_TRACE=1 runs of the same model/prompt set (stderr redirected to the logs):
#
#   STRATA_Q36M_TRACE=1 python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
#       --args "--native $Q36_MODEL --ctx 4096" --out /tmp/plain.json --log /tmp/trace-plain.log
#   STRATA_Q36M_TRACE=1 python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
#       --args "--native $Q36_MODEL --ctx 4096 --prefill 512" --out /tmp/chunk.json --log /tmp/trace-chunk.log
#   python3 tools/qwen35/q36_trace_cmp.py /tmp/trace-plain.log /tmp/trace-chunk.log
#
# (--log is REQUIRED: the harness sends the ENGINE's stderr to /dev/null without it; the shell's 2> only
# catches the gate's own stderr.)
#
# The decode side prints one line per (prompt token, layer): its row sums, accumulated in index order.
# The chunk side prints one line per (chunk, layer): every row summed in row order.  Summing the decode
# lines over the chunk's token range in token order reproduces the chunk-side accumulation exactly, so
# equal states give bit-equal doubles and the first mismatching layer is the buggy one.
import re
import sys

D_RE = re.compile(r"TRACE D pos=(-?\d+) L=(-?\d+) x=(-?[\d.eE+-]+) h=(-?[\d.eE+-]+)")
C_RE = re.compile(r"TRACE C nt=(\d+) pos0=(-?\d+) L=(-?\d+) x=(-?[\d.eE+-]+) h=(-?[\d.eE+-]+)")


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
    return reqs


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    plain, chunk = parse(sys.argv[1]), parse(sys.argv[2])
    if not plain or not chunk:
        sys.exit("no TRACE blocks - run the gate with STRATA_Q36M_TRACE=1 AND --log <file> (the engine's "
                 "stderr goes to /dev/null without --log; check the file for 'TRACE BEGIN' lines)")
    r0 = [e for e in plain[0] if e[0] == "D"]
    c0 = [e for e in chunk[0] if e[0] == "C"]
    if not r0 or not c0:
        sys.exit(f"empty blocks: plain {len(r0)} decode lines, chunk {len(c0)} chunk lines")

    nt, pos0 = c0[0][1], c0[0][2]

    # per-layer sums of the decode side's row sums over exactly the chunk's token range, in token order
    ref = {}
    for _, pos, l, x, h in r0:
        if pos0 <= pos < pos0 + nt:
            a = ref.setdefault(l, [0.0, 0.0])
            a[0] += x
            a[1] += h

    bad = []
    print(f"chunk: nt={nt} pos0={pos0}, {len(c0)} layers; decode reference over {len(ref)} layers")
    for _, cnt, p0, l, x, h in c0:
        a = ref.get(l)
        if a is None:
            print(f"  L{l:>3}  no decode reference")
            bad.append(l)
            continue
        if x == a[0] and h == a[1]:
            print(f"  L{l:>3}  OK")
        else:
            print(f"  L{l:>3}  DIVERGES  chunk x={x:.17g} h={h:.17g}  decode x={a[0]:.17g} h={a[1]:.17g}")
            bad.append(l)
    if bad:
        print(f"first diverging layer: {bad[0]} ({len(bad)} of {len(c0)} layers differ)")
        sys.exit(1)
    print("all layers bit-identical - the chunked state matches token-by-token; the divergence is "
          "downstream (final norm / head / sampling)")


if __name__ == "__main__":
    main()
