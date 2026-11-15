#!/usr/bin/env python3
"""The fork's release helper (kingdonb/Strata). docs/FORK.md, "Releasing from the fork", is the guide.

The tree keeps upstream's addresses and upstream's key, so it stays close to upstream and upstream's tests hold.
A fork release differs in two places only, and both are made here, at build time:

  bake TREE     Write the fork's release address and the fork's public feed key into a copy of the sources that is
                about to be packaged (the build box's checkout, never the checkout you work in). It is the same idea
                as packaging/make_local_feed.py, which bakes a local address into install.sh.
  carry         Take the engine and drafter entries of upstream's signed feed into entry files for build-feed, with
                their addresses made absolute. The fork's feed then names upstream's packages where they are, with
                upstream's sizes and SHA-256; nothing of upstream's is built or uploaded again. Upstream's feed is
                checked against upstream's key first.
  sequence      Print the feed sequence of a fork release: upstream's sequence * 1000 + N, so it is higher than the
                one a PC accepted from upstream and stays below the fork's sequence on the next upstream release.

Nothing here reads a private key and nothing uploads. Standard library plus this tree's strata_control.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UPSTREAM = "Niko1221/Strata"
FORK = "kingdonb/Strata"
FORK_KEY = ROOT / "packaging" / "fork" / "feed-key.pub.json"
#: what a Control package and the one-line installers are made from (packaging/package_terminal.py PAYLOAD)
BAKE_DIRS = ("strata_cli", "strata_control", "serve")
BAKE_FILES = ("strata.py", "packaging/install.sh", "packaging/install.ps1")
BAKE_SUFFIXES = {".py", ".html", ".js", ".mjs", ".sh", ".ps1", ".md", ".txt"}


def _bake_targets(tree: Path):
    for name in BAKE_FILES:
        if (tree / name).is_file():
            yield tree / name
    for folder in BAKE_DIRS:
        for p in sorted((tree / folder).rglob("*")):
            if not p.is_file() or p.suffix not in BAKE_SUFFIXES:
                continue
            if "fixtures" in p.parts or p.name.startswith("test_") or p.name.endswith("_test.py"):
                continue                                  # the tests and their recorded texts keep upstream's words
            yield p


def bake(tree: Path, *, force: bool = False) -> list[str]:
    tree = tree.resolve()
    if not force:
        dirty = subprocess.run(["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True)
        if dirty.returncode == 0 and dirty.stdout.strip():
            raise SystemExit(f"{tree} has uncommitted changes: bake a clean checkout made for the build (or pass --force)")
    changed = []
    for p in _bake_targets(tree):
        raw = p.read_bytes()
        new = raw.replace(UPSTREAM.encode(), FORK.encode())
        if new != raw:
            p.write_bytes(new)
            changed.append(str(p.relative_to(tree)))
    keys_path = tree / "strata_control" / "update_keys.json"
    keys = json.loads(keys_path.read_text(encoding="utf-8-sig"))
    fork_key = json.loads(FORK_KEY.read_text(encoding="utf-8"))["keys"][0]
    if fork_key["key_id"] not in {k["key_id"] for k in keys["keys"]}:
        keys["keys"].append(fork_key)                     # upstream's key stays: its engine packages carry its signature
        keys_path.write_text(json.dumps(keys, indent=2) + "\n", encoding="utf-8")
        changed.append("strata_control/update_keys.json")
    feed_line = (tree / "strata_cli" / "update_cmd.py").read_text(encoding="utf-8")
    want = f'DEFAULT_FEED = "https://github.com/{FORK}/releases/latest/download/strata-feed.json"'
    if want not in feed_line:
        raise SystemExit("bake did not find DEFAULT_FEED in strata_cli/update_cmd.py: upstream moved it; update scripts/fork_release.py")
    for name in ("packaging/install.sh", "packaging/install.ps1"):
        if f"https://github.com/{FORK}/releases/latest/download" not in (tree / name).read_text(encoding="utf-8"):
            raise SystemExit(f"bake did not find the release address in {name}: upstream moved it; update scripts/fork_release.py")
    return changed


def carry(feed_path: Path, sig_path: Path, tag: str, out: Path, *, skip_platforms: tuple = ("macos-arm64",)) -> dict:
    from strata_control import feed as feedmod, signing
    payload, sig = feed_path.read_bytes(), sig_path.read_bytes()
    upstream_keys = {k: v for k, v in signing.load_trusted_keys(environ={}).items()}
    feedmod.verify_and_parse(payload, sig, source=str(feed_path), trusted=upstream_keys)     # raises when the signature is not upstream's
    doc = json.loads(payload)
    base = f"https://github.com/{UPSTREAM}/releases/download/{tag}/"
    done = {"engine": [], "drafters": [], "skipped": []}
    for member in ("engine", "drafters"):
        folder = out / member
        folder.mkdir(parents=True, exist_ok=True)
        for e in (doc.get(member) or {}).get("packages", []):
            if member == "engine" and e.get("platform") in skip_platforms:
                done["skipped"].append(f"{e['platform']} {e.get('backend')} {e.get('version')}")
                continue
            e = dict(e)
            if "://" not in e["url"]:
                e["url"] = base + e["url"]
            name = e["url"].rsplit("/", 1)[1]
            (folder / (name + ".entry.json")).write_text(json.dumps(e, indent=1) + "\n", encoding="utf-8")
            done[member].append(name)
    done["upstream_sequence"] = doc["sequence"]
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bake")
    b.add_argument("tree", type=Path)
    b.add_argument("--force", action="store_true", help="bake a tree that has uncommitted changes")
    c = sub.add_parser("carry")
    c.add_argument("--upstream-feed", type=Path, required=True)
    c.add_argument("--upstream-sig", type=Path, required=True)
    c.add_argument("--upstream-tag", required=True, help="the upstream release the feed was downloaded from, e.g. v0.2.14")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--with-mac-engine", action="store_true", help="also carry upstream's Metal engine (it has no Qwen3.6 path)")
    s = sub.add_parser("sequence")
    s.add_argument("--upstream-feed", type=Path, required=True)
    s.add_argument("-n", type=int, required=True, help="the N of -mbp.rtx.N")
    a = ap.parse_args(argv)
    if a.cmd == "bake":
        for name in bake(a.tree, force=a.force):
            print("baked", name)
    elif a.cmd == "carry":
        done = carry(a.upstream_feed, a.upstream_sig, a.upstream_tag, a.out, skip_platforms=() if a.with_mac_engine else ("macos-arm64",))
        print(json.dumps(done, indent=1))
    else:
        if not 0 < a.n < 1000:
            raise SystemExit("N is 1 to 999")
        print(json.loads(a.upstream_feed.read_text(encoding="utf-8"))["sequence"] * 1000 + a.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
