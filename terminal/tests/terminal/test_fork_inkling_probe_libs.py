"""The kingdonb fork: the Inkling drafting probe starts the engine the way the serving start does.

On a Linux CUDA PC the engine package carries its CUDA runtime (libcudart.so.13) beside the binary, and the run config's "lib_dirs" is []. The
probe used child_env(cfg) on that bare config, so its engine exited 127 before READY, the probe recorded "off", and drafting stayed off for that
placement: on an RTX 5060 Ti, SQ-10 decoded 56 tok/s where the drafter (then DSpark) gave 78-82 (prose) and 138-151 (code). Now the probe resolves the engine
with serve.server.with_component_engine (the package folder first on the library path), and keeps the engine's last stderr lines in a
failed result.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.runner import Runner
from strata_control import inkling_family, inkling_probe as P
from tests.terminal.launch_support import HW_12, Home, catalog
from tests.terminal.test_inkling_probe import FakeEngine

CFG = {"exe": "component:strata-inkling", "arch": "inkling2_loop", "backend": "cuda", "parallel": 1, "lib_dirs": [],
       "args": ["--native", "m.strata", "--max-context", "32768"], "cwd": "."}
SPEEDS = {"off": {"code": 56.0, "chat": 56.0}, "on": {"code": 140.0, "chat": 80.0}}


class TheProbeGetsThePackageLibraries(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        t = tempfile.TemporaryDirectory(prefix="strata-probe-libs-")
        self.addCleanup(t.cleanup)
        self.path = Path(t.name) / "launch.json"
        self.path.write_text(json.dumps(CFG), encoding="utf-8")
        sidecar = mock.patch.object(inkling_family, "dflash2_sidecar", return_value=Path("dflash2.safetensors"))   # 0.2.23: the probe runs only with the DFlash 2 file
        sidecar.start()
        self.addCleanup(sidecar.stop)

    def test_the_env_is_built_from_the_resolved_config(self):
        envs, started = [], []

        def resolve(cfg):
            return {**cfg, "exe": "/pkg/strata-inkling", "lib_dirs": ["/pkg", *(cfg.get("lib_dirs") or [])]}

        def child_env(cfg, **kw):
            envs.append(cfg)
            return {"LD_LIBRARY_PATH": ":".join(cfg.get("lib_dirs") or [])}

        def open_engine(exe, args, env=None, cwd=None):
            started.append((exe, env))
            return FakeEngine(args, SPEEDS)

        r = Runner(S.LaunchSettings(family="inkling-2-mini", model="SQ-10", context=32768), catalog=catalog(HW_12))
        r.config_path = self.path
        r.emit_cb = lambda e: None
        with mock.patch("serve.server.with_component_engine", resolve), mock.patch("serve.server.child_env", child_env), \
                mock.patch.object(P, "Engine", open_engine):
            r.probe_drafting()
        self.assertEqual(envs[0]["lib_dirs"], ["/pkg"])
        self.assertTrue(started and all(exe == "/pkg/strata-inkling" and env["LD_LIBRARY_PATH"] == "/pkg" for exe, env in started))
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["drafting_probe"]["decision"], "on")
        self.assertEqual(saved["lib_dirs"], [])                    # the run config itself is not rewritten with the resolved paths
        self.assertEqual(saved["exe"], "component:strata-inkling")


class AFailedProbeSaysWhy(unittest.TestCase):
    def test_the_engines_stderr_is_kept(self):
        class Silent(FakeEngine):
            def stderr_tail(self):
                return "strata-inkling: error while loading shared libraries: libcudart.so.13: cannot open shared object file"

        res = P.probe(lambda a: Silent(a, SPEEDS, fail=True), ["--native", "m.strata"])
        self.assertEqual(res["decision"], "off")                   # upstream's rule stays: no measurement, no drafting
        self.assertIn("libcudart.so.13", res["engine_said"])

    def test_a_successful_probe_has_no_engine_said(self):
        res = P.probe(lambda a: FakeEngine(a, SPEEDS), ["--native", "m.strata"])
        self.assertNotIn("engine_said", res)
        self.assertEqual(res["decision"], "on")

    @unittest.skipIf(sys.platform == "win32", "a POSIX shell script stands in for the engine")
    def test_the_real_engine_class_reads_stderr(self):
        d = Path(tempfile.mkdtemp())
        exe = d / "fake-engine"
        exe.write_text("#!/bin/sh\necho 'cannot open libcudart.so.13' >&2\nexit 127\n", encoding="utf-8")
        exe.chmod(0o755)
        eng = P.Engine(str(exe), [])
        self.assertIsNone(eng.readline(10))                        # stdout closed: the engine ended
        eng.p.wait(timeout=10)
        for _ in range(50):
            if eng.err:
                break
            import time
            time.sleep(0.05)
        self.assertIn("libcudart.so.13", eng.stderr_tail())
        eng.close()


if __name__ == "__main__":
    unittest.main()
