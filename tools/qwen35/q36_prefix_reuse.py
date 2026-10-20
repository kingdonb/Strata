"""q36_prefix_reuse.py --exe E --args "..." --out F.json [--n 48] [--mtp] : the T8-r1 prefix-cache behavior
measurement.  One engine process per configuration, greedy, the scenario set in sequence; every request's DONE
line gives the counts (f[8] reused = the state position the evaluation resumed from, f[14] = the prompt tokens
actually read):

  R1  prompt P                        read all            reused 0
  R2  P + G1 + suffix                 strict extension    reused |P|+|G1|   (the T0 delta-only path)
  R3  P + G1[:m] + changed tail       divergence          reused |P|        (the T8-r1 restore)
  R4  P                               exact re-send       reused |P|, read 0 (the re-roll: R4's tokens == R1's)
  R5  P with an early token changed   deep divergence     reused 0          (the fallback full reset)

R3 runs again alone on a fresh process: its tokens must match the in-session R3 byte for byte (the restore
rebuilds exact state).  With --mtp the same set runs on a second process (--mtp 2 --draft-vocab 8192): the kept
tokens must still match the fresh full evaluation (a draft only proposes; the verify window's main-model pick
pick decides), and R4 must still reproduce R1 (the empty-tail restore keeps the checkpoint's draft seed).  A
third process runs the whole set with STRATA_Q36M_PREFIX_RESET=1 - the old behavior's A/B: the divergence rows
(R3/R4/R5) read fully again (reused 0; R2 keeps the T0 extension), and the A/B's R3 must match the fresh R3
(reset and restore agree on the tokens).

R2's `reused` counts the phantom stop token: live_tokens ends one token past the consumed state (the final
sampled token is never fed through the model), so a following extension resumes at the phantom's slot.  That
count signature is the follow-up ticket's evidence; this script only records it.

A behavior measurement, not a benchmark: no tok/s numbers, no claims.  Run it on the Mac in the M2 session."""
from pathlib import Path
import argparse, json, shlex, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from q36_harness import Eng, load_prompts

ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True)
ap.add_argument("--args", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=48)   # generated tokens per request
ap.add_argument("--mtp", action="store_true", help="also run the set with --mtp 2 --draft-vocab 8192")
ap.add_argument("--prompts", default=str(Path(__file__).resolve().parent / "parity_prompts.ids"))
ap.add_argument("--log", default=None)
a = ap.parse_args()

PARITY = load_prompts(a.prompts)
P = PARITY[0] + PARITY[1]              # the shared prefix (two real prompts concatenated: a countable read)
SUFFIX2 = PARITY[2][:12]               # R2's strict-extension tail
ALT = PARITY[-1]                       # the tail-rotation fallback's token source


def counts(done):
    f = done.split()
    return {"generated": int(f[1]), "prompt_tokens": int(f[2]), "finish": f[5],
            "reused": int(f[8]), "read": int(f[14])}


def divergent_tail(live, at, base):
    """A tail whose first token differs from live[at], so the divergence lands exactly there."""
    tail = list(base)
    for _ in range(len(tail)):
        if tail[0] != live[at]:
            return tail
        tail = tail[1:] + tail[:1]
    for t in ALT:
        if t != live[at]:
            return [t] + tail[:-1]
    raise SystemExit("no divergent tail token found")


def run_config(name, env=None, extra_args=None, fresh_check=False):
    log = (a.log + "." + name) if a.log else None
    eng = Eng(a.exe, shlex.split(a.args) + (extra_args or []), env=env, log=log)
    rows = []

    def req(rname, ids):
        toks, done, _, _ = eng.solo(ids, a.n)
        row = counts(done)
        row.update(name=rname, prompt_tokens=len(ids), tokens=toks)
        rows.append(row)
        return toks

    r1 = req("R1-prompt", P)
    live = P + r1
    m = max(1, len(r1) // 2)
    tail = divergent_tail(live, len(P) + m, PARITY[2][12:24])
    req("R2-extension", P + r1 + SUFFIX2)
    r3 = req("R3-restore", P + r1[:m] + tail)
    r4 = req("R4-reroll", P)
    req("R5-deep-divergence", P[:5] + [P[5] ^ 1] + P[6:])

    checks = {
        "R2_delta_only": rows[1]["reused"] == len(P) + len(r1),
        "R3_restored_at_prompt_end": rows[2]["reused"] == len(P) and
                                     rows[2]["read"] == len(P) + len(r1[:m]) + len(tail) - len(P),
        "R4_reroll_free": rows[3]["reused"] == len(P) and rows[3]["read"] == 0 and r4 == r1,
        "R5_fallback_reset": rows[0]["reused"] == 0 and rows[4]["reused"] == 0,
    }
    eng.close()
    if fresh_check:
        eng2 = Eng(a.exe, shlex.split(a.args) + (extra_args or []), env=env, log=(log + ".fresh") if log else None)
        ftoks, fdone, _, _ = eng2.solo(P + r1[:m] + tail, a.n)
        eng2.close()
        checks["R3_bit_identical_to_fresh"] = ftoks == r3
    return {"name": name, "rows": rows, "checks": checks}


configs = [run_config("plain", fresh_check=True)]
if a.mtp:
    configs.append(run_config("mtp", extra_args=["--mtp", "2", "--draft-vocab", "8192"], fresh_check=True))
configs.append(run_config("ab-reset", env={"STRATA_Q36M_PREFIX_RESET": "1"}))
for cfg in configs[-1:]:   # the A/B: the divergence rows read fully again (R2 keeps the T0 extension); its R3
    cfg["checks"]["divergence_rows_fully_read"] = all(r["reused"] == 0 for r in cfg["rows"]
                                                      if r["name"] != "R2-extension")   # must equal the fresh R3
    cfg["checks"]["R3_matches_plain_fresh"] = cfg["rows"][2]["tokens"] == configs[0]["rows"][2]["tokens"]

print("%-10s %-20s %7s %7s %6s %6s  %s" % ("config", "request", "prompt", "reused", "read", "gen", "finish"))
for cfg in configs:
    for r in cfg["rows"]:
        print("%-10s %-20s %7d %7d %6d %6d  %s" %
              (cfg["name"], r["name"], r["prompt_tokens"], r["reused"], r["read"], r["generated"], r["finish"]))
    for k, v in cfg["checks"].items():
        print("  %-10s %s %s" % (cfg["name"], "ok  " if v else "FAIL", k))
json.dump({"args": a.args, "n": a.n, "configs": configs}, open(a.out, "w"), indent=1)
print("wrote", a.out)
