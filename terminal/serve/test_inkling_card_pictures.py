"""The kingdonb fork: its CUDA strata-inkling reads pictures on the card, so upstream's CPU-reader picture cap (#7043) stays off for it.

Upstream's CUDA engine has no picture reader, so serve scales every picture to 0.8 MP on CUDA.  The fork's engine (tools/fork/inkling_release.sh,
0.1.66.1 and later) reads a 2,000 x 1,088 screenshot on the card in about 1.4 s, and scales a picture down itself when the card is too full.  Its
BUILD.json, beside the engine, says so; inkling_kind.card_picture_reader reads it.

    python -m pytest serve/test_inkling_card_pictures.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import inkling_kind as K  # noqa: E402


def engine_with(build: dict | str | None) -> str:
    d = Path(tempfile.mkdtemp())
    exe = d / "strata-inkling"
    exe.write_bytes(b"")
    if build is not None:
        (d / "BUILD.json").write_text(build if isinstance(build, str) else json.dumps(build), encoding="utf-8")
    return str(exe)


FORK = {"shape": "inkling2", "source": K.FORK_SOURCE, "cuda": "13.0", "portable": True}


class CardPictureReader(unittest.TestCase):
    def test_the_flag_names_the_card_reader(self):
        self.assertTrue(K.card_picture_reader(engine_with({**FORK, "version": "0.1.69.1", "picture_reader": "card"})))

    def test_the_fork_packages_before_the_flag_count(self):
        self.assertTrue(K.card_picture_reader(engine_with({**FORK, "version": "0.1.66.1"})))
        self.assertTrue(K.card_picture_reader(engine_with({**FORK, "version": "0.1.67.1"})))

    def test_the_fork_package_before_the_reader_does_not(self):
        self.assertFalse(K.card_picture_reader(engine_with({**FORK, "version": "0.1.65.1"})))

    def test_upstreams_engine_does_not(self):
        self.assertFalse(K.card_picture_reader(engine_with(None)))                                   # no BUILD.json
        self.assertFalse(K.card_picture_reader(engine_with({"shape": "inkling2", "version": "0.1.69"})))
        self.assertFalse(K.card_picture_reader(engine_with("not json")))
        self.assertFalse(K.card_picture_reader(engine_with({**FORK, "shape": "qwen", "version": "0.1.69.1", "picture_reader": "card"})))
        self.assertFalse(K.card_picture_reader("/nowhere/strata-inkling"))

    def test_the_card_reader_gets_no_default_cap(self):
        self.assertEqual(K.image_cap_pixels({"backend": "cuda-card"}, {}), 0)
        self.assertFalse(K.cpu_picture_reader({"backend": "cuda-card"}))

    def test_a_cap_the_person_sets_still_applies(self):
        self.assertEqual(K.image_cap_pixels({"backend": "cuda-card", "inkling_image_max_mp": 1.5}, {}), 1_500_000)
        self.assertEqual(K.image_cap_pixels({"backend": "cuda-card"}, {"STRATA_INKLING_IMAGE_MAX_MP": "2"}), 2_000_000)

    def test_upstreams_cuda_default_is_unchanged(self):
        self.assertEqual(K.image_cap_pixels({"backend": "cuda"}, {}), K.DEFAULT_MAX_PIXELS)
        self.assertEqual(K.image_cap_pixels({}, {}), K.DEFAULT_MAX_PIXELS)


class TheReleaseScriptWritesTheFlag(unittest.TestCase):
    def test_build_json_says_card(self):
        script = (ROOT.parent / "tools" / "fork" / "inkling_release.sh").read_text(encoding="utf-8")
        self.assertIn('\\"picture_reader\\": \\"card\\"', script)
        self.assertIn(K.FORK_SOURCE, script)


if __name__ == "__main__":
    unittest.main()
