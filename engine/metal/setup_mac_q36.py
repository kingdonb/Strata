"""Qwen3.6-35B-A3B on a Mac: `./setup.sh --family qwen36` (this fork's Metal engine path, docs/QWEN36_METAL.md).

metal/setup_mac.py hands a run that names `--family qwen36` to main() here, before its own parser runs.

It is a file of its own, not a branch inside setup_mac.py, for two reasons:

  * The two flows share little after the Mac check and the engine build.  Qwen3.6 is ONE GGUF the engine reads in
    place: no pack, no n-gram table, no repack, no separate MTP download (the draft block is in the file).  Only the
    tokenizer is prepared, read from the GGUF.
  * Unsloth's size names are not unique.  UD-Q2_K_XL, UD-Q4_K_XL and UD-Q6_K_XL are Qwen3.8-Flash-Next sets in
    setup.py's MODELS and in strata_control/mac_fit.py's MAC_MODELS, and they are Qwen3.6 sizes too.  A Qwen3.6 size
    is therefore reached only through its family and never by its bare name.

The sizes the Mac runs are in strata_control/mac_q36.py (UD-Q2_K_XL).  They are not added to setup.py's
OWN_PATH_FAMILIES["qwen36"], which lists the sizes upstream's CUDA / HIP path runs and would offer them on every PC.
The family's other facts (title, repository, architecture, drafts per step) are read from that table.

The run config follows the one setup.py writes for this family on a PC ("arch": "qwen35moe", --max-context, --mtp as
a draft count), so the rest of Strata treats it as a Qwen3.6 config and not as a Flash-Next one.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # engine/
TERMINAL = ROOT.parent / "terminal"                     # setup.py, strata_control and the tools are the Terminal's
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))
import setup as S  # noqa: E402
from strata_control.mac_fit import mac_info, mac_problems, recommend_context  # noqa: E402
from strata_control.mac_q36 import DEFAULT as DEFAULT_MODEL, FAMILY, MAC_SIZES  # noqa: E402

DRAFT_VOCAB = 8192          # the draft head's rows (docs/QWEN36_METAL.md, "The draft-vocab subset head")
# The engine's opt-in kernels this path was validated with (each one's section in docs/QWEN36_METAL.md names its gate).
ENGINE_ENV = {"STRATA_METAL_STREAM": "1", "STRATA_Q36M_PREFILL": "1", "STRATA_Q36M_ATTN_GQA": "1",
              "STRATA_Q36M_GDN_WY": "1", "STRATA_Q36M_DENSE_MMA": "1", "STRATA_RESIDENT_HEADROOM_GIB": "5"}


def wanted(argv) -> bool:
    """Does this command line ask for the Qwen3.6 family?  (`--family qwen36` or `--family=qwen36`.)"""
    argv = list(argv or [])
    return any((x == "--family" and i + 1 < len(argv) and argv[i + 1] == FAMILY) or x == f"--family={FAMILY}"
               for i, x in enumerate(argv))


def family() -> dict:
    return S.OWN_PATH_FAMILIES[FAMILY]


def sizes() -> dict:
    """Every size --model takes: the Mac's own (MAC_SIZES), then the family's other sizes, which were not run on a Mac."""
    fam = family()
    out = {m: dict(d, mac=True) for m, d in MAC_SIZES.items()}
    for m, d in fam["models"].items():
        if m not in out:
            name = fam["file"].format(q=m)
            size, sha = fam["sha256"][name]
            out[m] = {"about": d["about"], "file": name, "size": size, "sha256": sha, "ram_gb": d["ram_gb"], "mac": False}
    return out


def gguf_name(model: str) -> str:
    return sizes()[model]["file"]


def find_gguf(name: str, gguf_dir, models_root: Path, tag: str, roots: list) -> Path | None:
    """The model file when it is already on this Mac: --gguf-dir first, then the models folder (the family's own folder,
    the short `qwen36` folder earlier setups used, the folder itself), then the same places under each data folder."""
    places = []
    if gguf_dir:
        places.append(Path(gguf_dir))
    for base in [models_root, *(Path(r) / "models" for r in roots)]:
        places += [base / tag, base / FAMILY, base]
    seen = set()
    for folder in places:
        if folder in seen:
            continue
        seen.add(folder)
        if (folder / name).is_file():
            return folder / name
    return None


def engine_args(gguf: Path, ctx: int) -> list:
    return ["--native", str(gguf), "--max-context", str(ctx), "--prefill", "auto", "--mtp", str(family().get("mtp", 2)),
            "--draft-vocab", str(DRAFT_VOCAB), "--pool-workers", "6", "--vram-reserve-mib", "1536",
            "--resident-budget-gib", "9"]


def run_config(exe: Path, gguf: Path, tokenizer: Path, model: str, ctx: int, port: int, log: Path) -> dict:
    """The run config of one Qwen3.6 size on the Metal engine."""
    fam = family()
    return {"exe": str(exe), "args": engine_args(gguf, ctx), "cwd": str(ROOT), "tokenizer": str(tokenizer),
            "model_name": f"{fam['name']}-{model.lower()}", "arch": fam["arch"], "backend": "metal", "log": str(log),
            "port": port, "env": dict(ENGINE_ENV), "vram_reserve_mib": 1536,
            "ui_resource_baseline": {"workers": 0, "reserve": 0, "ramBudget": True, "ramHeadroom": True},
            "ui_resource_profile": "work"}


def main(argv, build_engine, start) -> int:
    """The Qwen3.6 setup.  `build_engine` and `start` are metal/setup_mac.py's (the engine build and the server start
    are the same for every model on a Mac)."""
    fam = family()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", choices=[FAMILY])
    ap.add_argument("--model", choices=list(sizes()))
    ap.add_argument("--context", type=int)
    ap.add_argument("--port", type=int)
    ap.add_argument("--host")
    ap.add_argument("--api-key")
    ap.add_argument("--no-browser", dest="browser", action="store_false", default=None)
    ap.add_argument("--data-dir")
    ap.add_argument("--move-data", action="store_true")
    ap.add_argument("--set-default-data-dir", action="store_true")
    ap.add_argument("--models-dir")
    ap.add_argument("--gguf-dir")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--source", choices=S.SOURCES, default=None)
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--no-start", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--vision", choices=["yes", "no", "gpu", "cpu"])
    a, rest = ap.parse_known_args(argv)
    if rest:
        S.warn("not used for Qwen3.6 on a Mac: " + " ".join(rest))
    if a.source:
        os.environ["STRATA_SOURCE"] = a.source
    S.say(f"Strata on a Mac (Apple Silicon, Metal): {fam['title']}")
    data, elsewhere = S.data_folder_for(a)
    roots = [data, *elsewhere]
    models_root = Path(a.models_dir) if a.models_dir else data / "models"

    # ---- 1. the Mac
    S.step(1, "checking your Mac")
    info = mac_info()
    S.ok(f"{info['chip']}, {info['ram_gb']:.0f} GB unified memory, macOS {info['macos'] or '?'}")
    problems = mac_problems(info)
    for p in problems:
        S.warn(p)
    free = S.free_gb(data if data.exists() else ROOT)
    S.ok(f"free disk: {free:.0f} GB at {data}")
    if a.check:
        return 1 if problems else 0
    if problems:
        S.fail(problems[0])

    # ---- 2. the choices
    S.step(2, "your choices")
    model = a.model or DEFAULT_MODEL
    size = sizes()[model]
    if not size["mac"]:
        S.warn(f"{model} was not run on a Mac (only {', '.join(MAC_SIZES)} was): kept as you chose")
    need_ram = size["ram_gb"]
    if info["ram_gb"] < need_ram:
        S.warn(f"{model} needs about {need_ram} GB of memory and this Mac has {info['ram_gb']:.0f} GB: kept as you "
               "chose; it may not start")
    if a.vision in ("yes", "gpu", "cpu"):
        S.warn("pictures are not available for Qwen3.6 on a Mac yet: the model is set up for text")
    ctx = a.context or recommend_context(info["ram_gb"])
    S.ok(f"{model}, context {ctx}")
    tag = fam["tag"] + model
    name = gguf_name(model)
    found = find_gguf(name, a.gguf_dir, models_root, tag, roots)
    need = (0 if found else size["size"] / 1e9) + 2
    if free < need:
        S.warn(f"about {need:.0f} GB are needed on the disk ({free:.0f} GB free): kept as you chose; it may run out")

    # ---- 3. Python packages
    S.step(3, "Python packages")
    S.pip_install(S.requirement_lines() if S.REQUIREMENTS.exists() else S.PY_PACKAGES, "numpy, jinja2, ...")

    # ---- 4. the engine
    S.step(4, "the Strata engine (Metal)")
    llama = S.llama_override()                         # None: ggml is the vendored third_party/strata-ggml (STRATA_GGML_DIR overrides)
    S.ok(f"llama.cpp {S.LLAMA_CPP_COMMIT[:7]} (ggml, gguf-py)")
    if a.build:
        (ROOT / "build-metal" / "BUILD.json").unlink(missing_ok=True)
    exe = build_engine(llama)
    if a.update:
        if not found:
            S.warn(f"{model} is not installed here, and an update does not download model files: nothing was installed")
            S.say(f"  to download and prepare {model}: ./setup.sh --setup --family {FAMILY}")
        return 0

    # ---- 5. the model: one GGUF, read in place
    S.step(5, f"downloading {fam['title']} {model}")
    if found:
        gguf = found
        S.ok(f"{gguf.name} already here: {gguf.parent}")
    else:
        if a.gguf_dir:
            S.fail(f"{name} is not in {a.gguf_dir}", "put the file there, or run without --gguf-dir to download it")
        gguf = models_root / tag / name
        gguf.parent.mkdir(parents=True, exist_ok=True)
        S.download(fam["hf"] + name, gguf)
    S.verify_sha256(gguf, size["size"], size["sha256"], own=bool(found))      # the pinned size and SHA-256 (a file of your own is checked once)
    S.ok("model file present")

    # ---- 6. the tokenizer (the engine reads everything else from the GGUF itself)
    S.step(6, "preparing the model for Strata")
    pack = S.find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    if not (pack / "tokenizer" / "vocab.json").exists():
        pack.mkdir(parents=True, exist_ok=True)
        S.run([sys.executable, str(TERMINAL / "tools" / "strata_tokenizer.py"), "--gguf", str(gguf), "--out", str(pack)],
              env=dict(os.environ, **S.gguf_py_env()))
    S.ok(f"tokenizer: {pack / 'tokenizer'}")

    # ---- 7. the config, the start script
    S.step(7, "writing the start script")
    port = a.port or 8080
    cfg = run_config(exe, gguf, pack / "tokenizer", model, ctx, port, data / f"strata-{tag.lower()}.log")
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if a.browser is not None:
        cfg["open_browser"] = a.browser
    cfg_path = ROOT / f"strata-{tag.lower()}.json"
    S.write_config(cfg_path, cfg)
    script = S.write_run_script(tag, cfg_path, port, cfg.get("open_browser") is not False)
    S.ok(f"start script: {script.name}")
    S.say()
    S.say("All set.")
    S.say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1")
    S.say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    S.say(f"  Next time:        {script.name} starts it right away")
    if a.no_start:
        return 0
    return start(cfg_path, port, a.browser is not False)
