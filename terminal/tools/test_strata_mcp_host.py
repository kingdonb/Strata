"""The Strata tools when a Strata server starts them for its own chat (this fork: serve/mcp.py builtin_strata_server
sets STRATA_MCP_HOST_PORT / _CONFIG / _KEY).  The tools must say which server is the chat's own, find its config
and log outside the code folder, use its key, and refuse to stop it.

A small stand-in server answers /health, /status, /v1/status and /unload on a loopback port; nothing real starts.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import strata_mcp as M  # noqa: E402

KEY = "sk-host-test"


class Fake(BaseHTTPRequestHandler):
    busy = False
    unloads = 0

    def log_message(self, *a):
        pass

    def _send(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _keyed(self):
        if self.headers.get("Authorization") == f"Bearer {KEY}":
            return True
        self._send(401, {"error": {"message": "missing or wrong API key"}})
        return False

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"service": "strata", "model": "qwen-test", "max_context": 131072, "images": False,
                                    "loaded": True, "api_key": True})
        if not self._keyed():
            return
        if self.path == "/v1/status":
            return self._send(200, {"engine": "0.1.60", "uptime_s": 5, "activity": {"requests": 3},
                                    "last_timings": {"prompt_n": 100, "prompt_per_second": 900.0, "predicted_n": 50,
                                                     "predicted_per_second": 58.5}})
        if self.path == "/status":
            return self._send(200, {"busy": type(self).busy})
        self._send(404, {})

    def do_POST(self):
        if not self._keyed():
            return
        if self.path == "/unload":
            type(self).unloads += 1
        self._send(200, {"status": "unloaded"})


class HostedTools(unittest.TestCase):
    def setUp(self):
        Fake.busy, Fake.unloads = False, 0
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.code = Path(tmp.name) / "code"              # where the Terminal's code is: no config lives here
        self.data = Path(tmp.name) / "data"              # where the running model's config and log are
        self.code.mkdir()
        self.data.mkdir()
        self.log = self.data / "strata-q2_0.log"
        self.log.write_text("engine line 1\nengine line 2\n", encoding="utf-8")
        self.config = self.data / "strata-q2_0.json"
        self.config.write_text(json.dumps({"port": self.port, "log": str(self.log), "model_name": "qwen-test"}), encoding="utf-8")

    def tools(self, hosted=True, key=KEY):
        env = {M.ENV_HOST_PORT: str(self.port), M.ENV_HOST_CONFIG: str(self.config), M.ENV_HOST_KEY: key} if hosted else {}
        with mock.patch.dict(os.environ, env):
            for k in (() if hosted else (M.ENV_HOST_PORT, M.ENV_HOST_CONFIG, M.ENV_HOST_KEY)):
                os.environ.pop(k, None)
            s = M.Strata(self.code, ROOT)
        s.health_timeout = 3
        return M.Tools(s), s

    def test_the_environment_is_read_strictly(self):
        self.assertIsNone(M.host_from_env({}))
        self.assertIsNone(M.host_from_env({M.ENV_HOST_PORT: "eighty"}))
        self.assertIsNone(M.host_from_env({M.ENV_HOST_PORT: "70000"}))
        host = M.host_from_env({M.ENV_HOST_PORT: "8080", M.ENV_HOST_CONFIG: str(self.code / "missing.json")})
        self.assertEqual(host, {"port": 8080, "config": None, "key": None})

    def test_the_chats_own_server_is_found_with_its_key_and_its_config(self):
        t, s = self.tools()
        self.assertEqual(s.ports(None)[0], self.port, "the chat's own port is looked at first")
        self.assertEqual(s.api_key_for(self.port), KEY)
        self.assertEqual(s.configs(), [self.config], "the config is found although it is not in the code folder")
        info = s.probe(self.port)
        self.assertNotIn("note", info)                   # no "wants an API key this folder's configs do not have"
        self.assertEqual(info["last_request"]["predicted_per_second"], 58.5)

    def test_status_names_the_chats_own_server_and_tells_busy_from_idle(self):
        t, s = self.tools()
        with mock.patch.object(s, "installed", return_value={"models": []}), mock.patch.object(s, "install_job", return_value=None):
            idle = t.strata_status(hardware=False)
            Fake.busy = True
            busy = t.strata_status(hardware=False)
        row = idle["running"][0]
        self.assertTrue(row["this_chat"])
        self.assertEqual(row["started_by"], M.HOST_WORDS)
        self.assertIn(", idle (this is the server and the model this chat runs on)", idle["summary"])
        self.assertIn(", busy (this is the server and the model this chat runs on; the request", busy["summary"])

    def launch_config(self):
        """A config as `strata launch` writes it: its own file name, and the engine named as a package."""
        cfg = self.data / "q2_0-128k.json"
        cfg.write_text(json.dumps({"port": self.port, "log": str(self.log), "model_name": "qwen-test",
                                   "exe": "component:strata-engine", "args": []}), encoding="utf-8")
        self.config.unlink()
        self.config = cfg

    def hy4_config(self):
        """A Hy4-Flash config as `strata launch` writes it (0.2.18): the shape engine as a package, and three .strata files."""
        models = self.data / "models" / "hy4-flash"
        models.mkdir(parents=True)
        files = {}
        for name, size in (("Hy4-Flash-SQ-64.strata", 4000), ("Hy4-Flash-Engram-Q8.strata", 5000), ("Hy4-Flash-Vision.strata", 300)):
            f = models / name
            f.write_bytes(b"\0" * size)
            f.with_name(name + ".done").write_text("{}", encoding="utf-8")   # Strata downloaded it
            files[name] = f
        cfg = self.data / "sq-64-128k.json"
        cfg.write_text(json.dumps({"port": self.port, "log": str(self.log), "model_name": "hy4-flash-sq-64", "exe": "component:strata-hy4",
                                   "args": ["--native", str(files["Hy4-Flash-SQ-64.strata"]), "--engram", str(files["Hy4-Flash-Engram-Q8.strata"]),
                                            "--max-context", "131072", "--vision", str(files["Hy4-Flash-Vision.strata"]),
                                            "--vram-experts-gb", "2.31"]}), encoding="utf-8")
        self.config.unlink()
        self.config = cfg
        return files

    def test_a_hy4_model_counts_its_engram_and_picture_files(self):
        files = self.hy4_config()
        t, s = self.tools()
        usage = s.disk_usage()["models"]["sq-64-128k"]
        self.assertEqual((usage["files"], usage["bytes"]), (3, 9300), "the Engram table and the picture reader are the model's too")
        plan = s.removal_plan(self.config)
        self.assertEqual(sorted({Path(r["path"]).name for r in plan["remove"]} & set(files)), sorted(files), "Delete takes all three files")

    def test_a_hy4_model_names_its_shape_engine_package(self):
        self.hy4_config()
        t, s = self.tools()
        found = mock.Mock(root=Path("/c/shapes/hy4/engine/0.1.64-linux-x86_64-cuda-cuda13"), version="0.1.64", backend="cuda")
        from strata_control import shape_engine
        with mock.patch.object(shape_engine, "find", return_value=found) as find, \
                mock.patch.object(s, "component_engine", side_effect=AssertionError("not the Strata engine")):
            row = s.describe_config(self.config)
            with mock.patch.object(s, "installed", return_value={"models": []}), mock.patch.object(s, "install_job", return_value=None):
                status = t.strata_status(hardware=False)
        find.assert_called_once_with("hy4")                  # looked up once, then cached
        self.assertEqual(row["engine"], {"component": "strata-hy4", "version": "0.1.64", "backend": "cuda"})
        own = status["running"][0]
        self.assertTrue(own["this_chat"])
        self.assertEqual(own["engine_version"], "0.1.60", "what the engine calls itself is kept")
        self.assertEqual(own["engine_package"]["component"], "strata-hy4")
        self.assertEqual(own["engine_package"]["version"], "0.1.64")

    def test_every_launch_config_is_listed_and_its_side_files_are_not(self):
        """`strata launch` keeps <name>.json, <name>.choices.json, <name>.tune.json and <name>.engine.log in one folder (chudmax, 0.2.18)."""
        self.hy4_config()
        other = self.data / "q2_0-256k-2.json"
        other.write_text(json.dumps({"exe": "component:strata-engine", "args": ["--max-context", "262144"], "port": 8080}), encoding="utf-8")
        os.utime(other, (2e9, 2e9))                      # the newest file: the chat's own config is still listed first
        for name, body in (("q2_0-256k-2.choices.json", {"exe": "x", "args": []}), ("q2_0-256k-2.tune.json", {"state": "done"}),
                           ("sq-64-128k.choices.json", {"model": "sq-64"}), ("notes.json", {"exe": "x"}), ("list.json", [1, 2]),
                           ("broken.json", None)):
            (self.data / name).write_text("{not json" if body is None else json.dumps(body), encoding="utf-8")
        self.code = self.data                            # serve/mcp.py starts the hosted tools with --root = the config's folder
        t, s = self.tools()
        self.assertEqual([p.name for p in s.configs()], ["sq-64-128k.json", "q2_0-256k-2.json"])
        self.assertEqual(s.find_config(None).resolve(), self.config.resolve())
        self.assertEqual(s.find_config("q2_0-256k-2").resolve(), other.resolve())
        self.assertEqual(sorted(s.disk_usage()["models"]), ["q2_0-256k-2", "sq-64-128k"])

    def test_a_launch_config_keeps_its_name_and_is_ready_with_the_engine_package(self):
        self.assertEqual(M.config_tag(Path("strata-q2_0.json")), "q2_0")
        self.assertEqual(M.config_tag(Path("q2_0-128k.json")), "q2_0-128k", "not cut by the length of 'strata-'")
        self.launch_config()
        t, s = self.tools()
        engine = self.data / "engine" / "strata"
        engine.parent.mkdir()
        engine.write_text("", encoding="utf-8")
        found = {"exe": str(engine), "root": str(engine.parent), "version": "0.1.60", "backend": "cuda"}
        # describe_config asks upstream's engine_here (0.2.16) whether the component is installed; engine_info asks the fork's lookup
        with mock.patch.object(s, "component_engine", return_value=found), \
                mock.patch.object(M.Strata, "engine_here", staticmethod(lambda reference, cfg: True)):
            d = s.describe_config(self.config)
            info = s.engine_info()
        self.assertEqual((d["model"], d["ready"], d["missing_files"]), ("q2_0-128k", True, []))
        self.assertEqual((info["version"], info["folder"], info["backend"]), ("0.1.60", str(engine.parent), "cuda"))
        with mock.patch.object(s, "component_engine", return_value=None):
            d = s.describe_config(self.config)
            self.assertEqual(s.engine_info(), {"folder": None, "version": None})
        self.assertEqual((d["ready"], d["missing_files"]), (False, ["component:strata-engine"]), "no engine package: said as before")

    def test_the_engine_package_lookup_never_raises(self):
        t, s = self.tools()
        with mock.patch.dict(os.environ, {"STRATA_COMPONENTS_DIR": str(self.data / "none")}):
            self.assertIsNone(s.component_engine())

    def test_asking_for_the_server_log_of_the_chats_own_server_gives_its_engine_log(self):
        t, s = self.tools()
        out = t.strata_logs(source="server")
        self.assertEqual(out["log"], str(self.log))
        self.assertIn("was not started by these tools", out["summary"])
        self.assertEqual(out["lines"][-1], "engine line 2")
        t2, _ = self.tools(hosted=False)
        self.assertEqual(t2.strata_logs(source="server")["lines"], [], "started by hand: the tools' own (empty) server log, as before")

    def test_without_a_host_nothing_changes(self):
        t, s = self.tools(hosted=False)
        self.assertIsNone(s.host)
        self.assertEqual(s.configs(), [])
        self.assertIsNone(s.api_key_for(self.port))
        self.assertEqual(M.McpServer(s).host_words(), "")

    def test_the_instructions_say_which_server_is_the_chats_own(self):
        t, s = self.tools()
        words = M.McpServer(s).host_words()
        self.assertIn(f"port {self.port}", words)
        self.assertIn("this_chat", words)
        self.assertNotIn(KEY, words)

    def test_the_log_of_the_chats_own_server_is_the_default(self):
        t, s = self.tools()
        out = t.strata_logs()
        self.assertEqual(out["log"], str(self.log))
        self.assertEqual(out["lines"][-1], "engine line 2")

    def test_stop_refuses_the_chats_own_server_even_with_force(self):
        t, s = self.tools()
        for force in (False, True):
            with self.assertRaises(M.ToolError) as e:
                t.strata_stop(force=force)
            self.assertIn("this chat runs on", str(e.exception))
        self.assertEqual(Fake.unloads, 0)

    def test_a_speed_test_from_inside_the_chats_own_turn_reports_the_last_request(self):
        Fake.busy = True                                 # the running request is the chat's own turn
        t, s = self.tools()
        out = t.strata_benchmark()
        self.assertTrue(out["this_chat"])
        self.assertEqual(out["last_request"]["predicted_per_second"], 58.5)
        t2, _ = self.tools(hosted=False)
        with self.assertRaises(M.ToolError) as e:        # started by hand, with no key: what the chat saw before (HTTP 401)
            t2.strata_benchmark(port=self.port)
        self.assertIn("401", str(e.exception))

    def test_no_answer_carries_the_key(self):
        t, s = self.tools()
        self.assertNotIn(KEY, json.dumps(t.strata_logs()) + json.dumps(s.probe(self.port)) + json.dumps(s.describe_config(self.config)))


if __name__ == "__main__":
    unittest.main()
