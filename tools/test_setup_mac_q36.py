"""Qwen3.6 setup on a Mac (metal/setup_mac_q36.py): chosen by its family, never by a bare size name.

    python tools/test_setup_mac_q36.py

No Mac, no GPU, no download: the pure parts (the command-line test, the file search, the run config) and the hand-over
from metal/setup_mac.py.
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(name):
    spec = importlib.util.spec_from_file_location(f"t_{name}", ROOT / "metal" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Q = load("setup_mac_q36")
M = load("setup_mac")
S = Q.S


class Chosen(unittest.TestCase):
    def test_only_the_family_selects_the_qwen36_flow(self):
        self.assertTrue(Q.wanted(["--setup", "--family", "qwen36", "--model", "UD-Q2_K_XL"]))
        self.assertTrue(Q.wanted(["--family=qwen36"]))
        for argv in (["--model", "UD-Q2_K_XL"], ["--family", "unsloth", "--model", "UD-Q2_K_XL"], ["--family"], [], None,
                     ["--model", "qwen36"]):
            self.assertFalse(Q.wanted(argv), argv)

    def test_setup_mac_hands_a_qwen36_run_over_and_nothing_else(self):
        self.assertIsNone(M.qwen36_setup(["--setup", "--yes", "--model", "UD-Q2_K_XL"]))
        self.assertIsNone(M.qwen36_setup(["--family", "unsloth", "--model", "UD-Q2_K_XL"]))
        self.assertIsNone(M.qwen36_setup(["--family"]))
        module = M.qwen36_setup(["--yes", "--family", "qwen36"])
        self.assertEqual(module.FAMILY, "qwen36")
        with mock.patch.object(M, "qwen36_setup") as pick:
            pick.return_value.main.return_value = 7
            self.assertEqual(M.main(["--family", "qwen36", "--no-start"]), 7)
            args, kwargs = pick.return_value.main.call_args
            self.assertEqual(args[0], ["--family", "qwen36", "--no-start"])
            self.assertIs(kwargs["build_engine"], M.build_engine)
            self.assertIs(kwargs["start"], M.start)

    def test_the_shared_size_names_stay_flash_next_in_the_bare_name_tables(self):
        """UD-Q2_K_XL, UD-Q4_K_XL and UD-Q6_K_XL are Flash-Next sets where a table is keyed by the bare name."""
        for name in ("UD-Q2_K_XL", "UD-Q4_K_XL", "UD-Q6_K_XL"):
            self.assertNotIn("qwen36", S.MODELS[name].get("families", ()), name)
            self.assertEqual(M.MAC_MODELS[name].get("family"), "unsloth", name)

    def test_the_mac_size_is_not_offered_on_a_pc(self):
        """setup.py's family table is the family as the CUDA / HIP path runs it: a size added there is offered, and for little RAM recommended, on every PC."""
        self.assertEqual(list(S.OWN_PATH_FAMILIES["qwen36"]["models"]), ["UD-Q4_K_XL", "UD-Q6_K_XL", "Q8_0"])
        self.assertNotIn(Q.gguf_name("UD-Q2_K_XL"), S.OWN_PATH_FAMILIES["qwen36"]["sha256"])
        self.assertEqual(Q.DEFAULT_MODEL, "UD-Q2_K_XL")
        sizes = Q.sizes()
        self.assertEqual(list(sizes), ["UD-Q2_K_XL", "UD-Q4_K_XL", "UD-Q6_K_XL", "Q8_0"], "the Mac's own size first, then the family's")
        self.assertEqual([m for m, d in sizes.items() if d["mac"]], ["UD-Q2_K_XL"])

    def test_every_size_is_pinned_by_size_and_sha256(self):
        for model, d in Q.sizes().items():
            self.assertEqual(d["file"], f"Qwen3.6-35B-A3B-{model}.gguf")
            self.assertGreater(d["size"], 10 ** 10, model)
            self.assertRegex(d["sha256"], r"^[0-9a-f]{64}$", model)
        mac = Q.MAC_SIZES["UD-Q2_K_XL"]
        self.assertEqual(round(mac["size"] / 1e9, 1), mac["download_gb"])


class Files(unittest.TestCase):
    def test_the_model_file_is_found_where_earlier_setups_put_it(self):
        name = Q.gguf_name("UD-Q2_K_XL")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            models, data, other = tmp / "models", tmp / "data", tmp / "other"
            tag = "qwen36-UD-Q2_K_XL"
            self.assertIsNone(Q.find_gguf(name, None, models, tag, [data]))
            (data / "models" / "qwen36").mkdir(parents=True)
            (data / "models" / "qwen36" / name).write_bytes(b"GGUF")
            self.assertEqual(Q.find_gguf(name, None, models, tag, [data]), data / "models" / "qwen36" / name)
            (models / tag).mkdir(parents=True)
            (models / tag / name).write_bytes(b"GGUF")
            self.assertEqual(Q.find_gguf(name, None, models, tag, [data]), models / tag / name, "the models folder first")
            other.mkdir()
            (other / name).write_bytes(b"GGUF")
            self.assertEqual(Q.find_gguf(name, str(other), models, tag, [data]), other / name, "--gguf-dir before everything")


class Config(unittest.TestCase):
    def test_the_run_config_is_a_qwen36_config_and_not_a_flash_next_one(self):
        cfg = Q.run_config(Path("/e/strata"), Path("/m/q.gguf"), Path("/p/tokenizer"), "UD-Q2_K_XL", 32768, 8095, Path("/d/x.log"))
        self.assertEqual(cfg["arch"], "qwen35moe")
        self.assertTrue(S.own_path_config(cfg), "setup reads --mtp as a draft count for this config")
        self.assertEqual(cfg["model_name"], "qwen3.6-35b-a3b-ud-q2_k_xl")
        self.assertEqual(cfg["backend"], "metal")
        args = cfg["args"]
        self.assertEqual(args[:4], ["--native", str(Path("/m/q.gguf")), "--max-context", "32768"])
        self.assertEqual(args[args.index("--mtp") + 1], str(S.OWN_PATH_FAMILIES["qwen36"]["mtp"]))
        self.assertEqual(args[args.index("--draft-vocab") + 1], "8192")
        self.assertNotIn("--pack", args)
        self.assertNotIn("--ple-gguf", args)
        for knob in ("STRATA_Q36M_PREFILL", "STRATA_Q36M_ATTN_GQA", "STRATA_Q36M_GDN_WY", "STRATA_Q36M_DENSE_MMA"):
            self.assertEqual(cfg["env"][knob], "1", knob)
        other = Q.run_config(Path("/e/strata"), Path("/m/q.gguf"), Path("/p/tokenizer"), "UD-Q2_K_XL", 4096, 8080, Path("/d/x.log"))
        other["env"]["X"] = "1"
        self.assertNotIn("X", Q.ENGINE_ENV, "each config gets its own copy of the environment")

    def test_setup_reads_the_choices_back_from_the_config_it_wrote(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strata-qwen36-ud-q2_k_xl.json"
            S.write_config(path, Q.run_config(Path("/e/strata"), Path("/m/q.gguf"), Path("/p/tokenizer"), "UD-Q2_K_XL", 32768, 8095,
                                              Path("/d/x.log")))
            choices = S.choices_from_config(path)
            # (the size is not one of setup.py's own for this family, so it reads back as None: on a Mac the Qwen3.6 flow takes its default)
            self.assertEqual((choices["family"], choices["model"], choices["context"]), ("qwen36", None, 32768))


if __name__ == "__main__":
    unittest.main()
