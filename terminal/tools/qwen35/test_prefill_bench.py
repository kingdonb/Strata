"""test_prefill_bench.py: runs q36_prefill_bench.py against a stub engine that speaks the serve protocol
(READY / PP / T / DONE with made-up timings) - no Metal, no GPU, runs anywhere.  Checks the JSON's rows
(reuse == 0, fresh == size), the table, the per-chunk ratios and both >=5x verdict outcomes, the
refused-start row and the --ctx guard.
  python3 tools/qwen35/test_prefill_bench.py
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "q36_prefill_bench.py"

STUB = r'''#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
def val(flag, default):
    return args[args.index(flag) + 1] if flag in args else default
chunk = int(val("--prefill", "0"))
ctx = int(val("--ctx", "4096"))
if os.environ.get("STRATA_FAKE_REFUSE") and int(os.environ["STRATA_FAKE_REFUSE"]) == chunk:
    sys.stderr.write("strata metal q36: ctx %d takes ~9.99 GB allocated (+0.25 reserve) past the "
                     "recommended working set 9.00 GB - lower --ctx (or --prefill) or set "
                     "STRATA_Q36M_MEM_OK=1\n" % ctx)
    sys.exit(1)
rate = 17.0 if chunk == 0 else 17.0 * float(os.environ.get("STRATA_FAKE_MULT", "7"))
print("INFO kv=fp32", flush=True)
print("READY %d stop" % ctx, flush=True)
while True:
    line = sys.stdin.readline()
    if not line:
        break
    f = line.split()
    if not f or f[0] != "GEN":
        continue
    n, ids = int(f[1]), [int(x) for x in f[2].split(",") if x]
    size, step = len(ids), chunk if chunk > 0 else 512
    prompt_ms = 1000.0 * size / rate
    done = 0
    while done < size - 1:
        done = min(size - 1, done + step)
        print("PP %d %d %.0f %.1f" % (done, size, prompt_ms * done / max(1, size - 1), rate), flush=True)
    for t in range(n):
        print("T %d" % (100 + t), flush=True)
    print("DONE %d %d %.1f %.1f length 0 0 0 0 0 0 0 0.0 %d 0"
          % (n, size, prompt_ms, 1000.0 * n / 16.0, size), flush=True)
'''


def run_tool(env_extra, tmp, extra_args=()):
    stub = tmp / "stub_engine.py"
    stub.write_text(STUB)
    stub.chmod(0o755)
    env = dict(os.environ, **env_extra)
    return subprocess.run(
        [sys.executable, str(TOOL), "--exe", str(stub), "--args", "--native fake.gguf --ctx 32768",
         "--out", str(tmp / "bench.json"), "--rounds", "2", *extra_args],
        capture_output=True, text=True, env=env)


def main():
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # 1. the clear case: the chunk path reads 7x (all sizes), chunk 1024 refused by the gate
        p = run_tool({"STRATA_FAKE_MULT": "7", "STRATA_FAKE_REFUSE": "1024"}, tmp,
                     ["--sizes", "2048,4096,8192", "--chunks", "0,512,1024"])
        j = json.loads((tmp / "bench.json").read_text())
        alive = [r for r in j["rows"] if not r.get("refused")]
        refused = [r for r in j["rows"] if r.get("refused")]
        ok = p.returncode == 0
        ok &= len(alive) == 2 * 3 * 2 and len(refused) == 1 and refused[0]["chunk"] == 1024
        ok &= all(r["reuse"] == 0 and r["fresh"] == r["prompt_len"] for r in alive)
        ok &= set(j["chunk_ratios"]) == {"512"} and all(abs(v - 7.0) < 1e-3 for v in j["chunk_ratios"]["512"].values()) \
            and set(j["chunk_ratios"]["512"]) == {"2048", "4096", "8192"}
        ok &= "clears >=5x on every size" in j["gate_verdict"] and "1024" in j["gate_verdict"]
        ok &= "| 4096 | 512 |" in j["table_md"] and "| 8192 | 0 (token-by-token) |" in j["table_md"]
        all_ok &= ok
        print("clear case + refused start:", "PASS" if ok else "FAIL")
        if not ok:
            print(p.stdout, p.stderr)

        # 2. the not-met case: the chunk path reads 2x
        p = run_tool({"STRATA_FAKE_MULT": "2"}, tmp, ["--sizes", "2048", "--chunks", "0,512"])
        j = json.loads((tmp / "bench.json").read_text())
        ok = p.returncode == 0 and abs(j["chunk_ratios"]["512"]["2048"] - 2.0) < 1e-3 \
            and "does not clear" in j["gate_verdict"]
        all_ok &= ok
        print("not-met case:", "PASS" if ok else "FAIL")
        if not ok:
            print(p.stdout, p.stderr)

        # 3. the baseline refused: the gate cannot be evaluated
        p = run_tool({"STRATA_FAKE_REFUSE": "0"}, tmp, ["--sizes", "2048", "--chunks", "0,512"])
        j = json.loads((tmp / "bench.json").read_text())
        ok = p.returncode == 0 and not j["chunk_ratios"] and "incomplete" in j["gate_verdict"]
        all_ok &= ok
        print("baseline-refused case:", "PASS" if ok else "FAIL")
        if not ok:
            print(p.stdout, p.stderr)

        # 4. the --ctx guard: a prompt that does not fit the engine's ctx must not run
        p = run_tool({}, tmp, ["--sizes", "33000", "--chunks", "512"])
        ok = p.returncode != 0 and "does not fit --ctx" in p.stderr
        all_ok &= ok
        print("ctx guard:", "PASS" if ok else "FAIL")
        if not ok:
            print(p.stdout, p.stderr)

    print("all pass" if all_ok else "FAILURES")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
