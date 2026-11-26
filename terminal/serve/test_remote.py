"""serve/test_remote.py - the `--engine remote` front-end against an in-process Strata server (mock
engine): streaming and non-streaming parity on all three APIs, client-disconnect cancellation
reaching the upstream's engine, the local MCP tool loop, and the honest unreachable-upstream
behavior.  No GPU, no pack, no network beyond loopback.

    python -m pytest serve/test_remote.py -v
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import CALL_END, CALL_START, THINK_END, ChatTemplate  # noqa: E402
from serve.mcp import McpHub  # noqa: E402
from serve.remote import RemoteEngine, redact  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "serve" / "server.py"
FAKE_MCP = ROOT / "serve" / "mcp_fake_server.py"
CTX = 8192
GREETING = f"Thinking about it.{THINK_END}\n\nHello from the mock engine."
LONG_ANSWER = "The answer goes on and on. " * 200


def tool_call_script(name, **params):
    """The model's text for one tool call, as the template's tool-call spelling writes it."""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"{THINK_END}\n\nLet me check.\n\n{CALL_START}\n<function={name}>\n{body}</function>\n{CALL_END}"


class RecordingMock(MockEngine):
    """A mock engine that remembers the cancel Event and the prompt of every generate() call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_cancel = None
        self.prompts: list[str] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_cancel = cancel
        self.prompts.append("".join(chr(b) for b in ids if 0 <= b < 256))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


def make_server(engine, tok, *, api_key=None, hub=None):
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), fit_max_tokens=True)
    if api_key:
        svc.api_key = api_key
    if hub is not None:
        svc.mcp = hub
    httpd = serve(svc, port=0)
    return svc, httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def post(base, path, body, timeout=30):
    """POST JSON; returns (status, parsed body).  HTTPError answers (code, its JSON body)."""
    request = urllib.request.Request(base + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post_stream(base, path, body, lines=None, timeout=30):
    """Open an SSE stream and return its `data:` payloads (the last chunk closed the connection when
    `lines` is None, else exactly `lines` payloads are read and the connection is closed)."""
    request = urllib.request.Request(base + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(request, timeout=timeout)
    out = []
    try:
        while lines is None or len(out) < lines:
            raw = resp.readline()
            if not raw:
                break
            text = raw.decode("utf-8", "replace").strip("\r\n")
            if text.startswith("data:"):
                out.append(text[len("data:"):].strip())
    finally:
        if lines is not None:
            resp.close()
    return out


def sse_items(payloads):
    """The parsed (event name or None, JSON object or raw text) of an SSE stream's data payloads."""
    items = []
    for text in payloads:
        if text == "[DONE]":
            items.append((None, "[DONE]"))
            continue
        try:
            data = json.loads(text)
        except ValueError:
            items.append((None, text))
            continue
        items.append((data.get("type"), data) if isinstance(data.get("type"), str) else (None, data))
    return items


def dead_port():
    """A loopback port nothing is listening on (bound once, then closed)."""
    with __import__("socket").socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class RemoteBase(unittest.TestCase):
    """One mock-engine Strata server as the upstream, one front-end in front of it."""

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.upstream_engine = MockEngine(cls.tok, [GREETING], max_context=CTX)
        cls.up_svc, cls.upstream_httpd, cls.upstream_base = make_server(cls.upstream_engine, cls.tok)
        cls.front_engine = RemoteEngine(cls.upstream_base)
        cls.front_svc, cls.front_httpd, cls.base = make_server(cls.front_engine, cls.tok)

    @classmethod
    def tearDownClass(cls):
        cls.front_httpd.shutdown()
        cls.front_httpd.server_close()
        cls.upstream_httpd.shutdown()
        cls.upstream_httpd.server_close()


class ChatParity(RemoteBase):
    def test_non_streaming_chat(self):
        status, body = post(self.base, "/v1/chat/completions",
                            {"model": "x", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        message = body["choices"][0]["message"]
        self.assertEqual(message["content"], "Hello from the mock engine.")
        self.assertEqual(message["reasoning_content"], "Thinking about it.")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["usage"]["completion_tokens"], body["usage"]["total_tokens"]
                         - body["usage"]["prompt_tokens"])

    def test_streaming_chat(self):
        payloads = post_stream(self.base, "/v1/chat/completions",
                               {"messages": [{"role": "user", "content": "hi"}], "stream": True,
                                "stream_options": {"include_usage": True}})
        items = sse_items(payloads)
        self.assertEqual(items[-1], (None, "[DONE]"))
        deltas = [data["choices"][0]["delta"] for name, data in items
                  if name is None and data != "[DONE]" and data.get("choices")]
        self.assertIn({"role": "assistant", "content": ""}, deltas[:2])
        contents = "".join(d.get("content") or "" for d in deltas)
        reasonings = "".join(d.get("reasoning_content") or "" for d in deltas)
        self.assertEqual(contents, "Hello from the mock engine.")
        self.assertEqual(reasonings, "Thinking about it.")
        finish = [data["choices"][0]["finish_reason"] for name, data in items if name is None
                  and data != "[DONE]" and data.get("choices") and data["choices"][0]["finish_reason"]]
        self.assertEqual(finish, ["stop"])
        usage = [data.get("usage") for name, data in items
                 if name is None and isinstance(data, dict) and data.get("usage")]
        self.assertEqual(len(usage), 1)
        self.assertGreater(usage[0]["prompt_tokens"], 0)      # the engine's own count, not a guess

    def test_streaming_without_usage_option_keeps_the_chunk_unchanged(self):
        payloads = post_stream(self.base, "/v1/chat/completions",
                               {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        chunks = [json.loads(p) for p in payloads[:-1] if p != "[DONE]"]
        # the finish chunk carries the usage; without include_usage it is never a separate choices:[] chunk
        self.assertTrue(all(c.get("choices") for c in chunks))
        self.assertIsNotNone(chunks[-1].get("usage"))

    def test_health_reports_the_engine(self):
        with urllib.request.urlopen(self.base + "/health", timeout=10) as r:
            body = json.loads(r.read())
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["engine"]["mode"], "remote")
        self.assertTrue(body["engine"]["reachable"])
        self.assertEqual(body["engine"]["url"], self.upstream_base)
        self.assertEqual(body["model"], "qwen3.8-flash-next")

    def test_mcp_opt_in_without_a_tool_call_answers_normally(self):
        status, body = post(self.base, "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "hi"}], "strata_mcp": True})
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], "Hello from the mock engine.")


class AnthropicParity(RemoteBase):
    def test_non_streaming(self):
        ask = {"model": "x", "max_tokens": 200, "messages": [{"role": "user", "content": "hi"}]}
        status, body = post(self.base, "/v1/messages", ask)
        self.assertEqual(status, 200)
        _, up = post(self.upstream_base, "/v1/messages", ask)     # parity: the engine host's own answer
        self.assertEqual(body["stop_reason"], up["stop_reason"])
        self.assertGreater(body["usage"]["input_tokens"], 0)
        text = lambda b: "".join(blk.get("text") or "" for blk in b["content"])
        self.assertEqual(text(body), text(up))
        self.assertIn("Hello from the mock engine.", text(body))

    def test_streaming(self):
        payloads = post_stream(self.base, "/v1/messages",
                               {"max_tokens": 200, "messages": [{"role": "user", "content": "hi"}], "stream": True})
        items = sse_items(payloads)
        starts = [data for name, data in items if name == "message_start"]
        self.assertEqual(len(starts), 1)
        self.assertGreater(starts[0]["message"]["usage"]["input_tokens"], 0)
        stops = [data for name, data in items if name == "message_stop"]
        self.assertEqual(len(stops), 1)
        deltas = [data["delta"] for name, data in items if name == "content_block_delta"]
        self.assertIn("Hello from the mock engine.", "".join(d.get("text") or "" for d in deltas))
        reasons = [data["delta"]["stop_reason"] for name, data in items if name == "message_delta"]
        self.assertEqual(reasons, ["end_turn"])

    def test_count_tokens_is_delegated(self):
        status, body = post(self.base, "/v1/messages/count_tokens",
                            {"model": "x", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertGreater(body["input_tokens"], 0)


class ResponsesParity(RemoteBase):
    def test_non_streaming(self):
        status, body = post(self.base, "/v1/responses", {"model": "x", "input": "hi"})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "completed")
        self.assertGreater(body["usage"]["input_tokens"], 0)
        kinds = [item["type"] for item in body["output"]]
        self.assertEqual(kinds, ["reasoning", "message"])
        self.assertEqual(body["output"][1]["content"][0]["text"], "Hello from the mock engine.")

    def test_streaming(self):
        payloads = post_stream(self.base, "/v1/responses", {"model": "x", "input": "hi", "stream": True})
        items = sse_items(payloads)
        types = [name for name, _ in items]
        self.assertEqual(types[0], "response.created")
        self.assertEqual(types[-1], "response.completed")
        text = "".join(data["delta"] for name, data in items if name == "response.output_text.delta")
        self.assertEqual(text, "Hello from the mock engine.")
        thinking = "".join(data["delta"] for name, data in items if name == "response.reasoning_text.delta")
        self.assertEqual(thinking, "Thinking about it.")
        completed = [data for name, data in items if name == "response.completed"][0]
        self.assertGreater(completed["response"]["usage"]["input_tokens"], 0)


class Cancellation(RemoteBase):
    """A client that hangs up mid-stream ends the upstream's generation too."""

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.upstream_engine = RecordingMock(cls.tok, LONG_ANSWER, max_context=CTX, delay_s=0.02)
        cls.up_svc, cls.upstream_httpd, cls.upstream_base = make_server(cls.upstream_engine, cls.tok)
        cls.front_engine = RemoteEngine(cls.upstream_base)
        cls.front_svc, cls.front_httpd, cls.base = make_server(cls.front_engine, cls.tok)

    def test_client_disconnect_cancels_the_upstream_engine(self):
        payload_lines = post_stream(self.base, "/v1/chat/completions",
                                    {"messages": [{"role": "user", "content": "hi"}], "stream": True},
                                    lines=3)
        self.assertTrue(payload_lines)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (self.upstream_engine.last_cancel
                                                   and self.upstream_engine.last_cancel.is_set()):
            time.sleep(0.1)
        self.assertIsNotNone(self.upstream_engine.last_cancel)
        self.assertTrue(self.upstream_engine.last_cancel.is_set())
        with urllib.request.urlopen(self.base + "/health", timeout=10) as r:
            body = json.loads(r.read())
        self.assertEqual(body["status"], "ok")                    # the front-end itself is fine


class McpToolLoop(unittest.TestCase):
    """strata_mcp: true on the front-end: the front-end's own hub discovers and runs the tool, the
    engine host never sees the flag and never needs the MCP server."""

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.upstream_engine = RecordingMock(cls.tok, [tool_call_script("fake__echo", text="hello from the tool"),
                                                      f"Round two.{THINK_END}\n\nThe final answer."],
                                            max_context=CTX)
        cls.up_svc, cls.upstream_httpd, cls.upstream_base = make_server(cls.upstream_engine, cls.tok)
        cls.front_engine = RemoteEngine(cls.upstream_base)
        hub = McpHub({"fake": {"command": sys.executable, "args": [str(FAKE_MCP)]}})
        cls.hub = hub
        hub.start()
        cls.front_svc, cls.front_httpd, cls.base = make_server(cls.front_engine, cls.tok, hub=hub)

    @classmethod
    def tearDownClass(cls):
        cls.front_httpd.shutdown()
        cls.front_httpd.server_close()
        cls.upstream_httpd.shutdown()
        cls.upstream_httpd.server_close()
        cls.hub.close()

    def test_tool_loop_runs_on_the_front_end(self):
        payloads = post_stream(self.base, "/v1/chat/completions",
                               {"messages": [{"role": "user", "content": "hi"}], "stream": True,
                                "strata_mcp": True})
        items = sse_items(payloads)
        events = [data["strata_mcp"] for _, data in items
                  if isinstance(data, dict) and isinstance(data.get("strata_mcp"), dict)]
        kinds = [e["event"] for e in events]
        self.assertIn("start", kinds)
        self.assertIn("call", kinds)
        self.assertIn("result", kinds)
        result = next(e for e in events if e["event"] == "result")
        self.assertTrue(result["ok"])
        self.assertIn("hello from the tool", result["text"])
        contents = "".join(data["choices"][0]["delta"].get("content") or ""
                           for _, data in items if data != "[DONE]" and isinstance(data, dict)
                           and data.get("choices") and data["choices"][0].get("delta"))
        # the round-1 text beside the call streams through, as run_with_mcp does locally; the final answer ends it
        self.assertIn("Let me check.", contents)
        self.assertTrue(contents.endswith("The final answer."))
        # the engine host saw the tool result in the conversation of the second round, and never the flag
        self.assertEqual(len(self.upstream_engine.prompts), 2)
        self.assertIn("hello from the tool", self.upstream_engine.prompts[1])
        self.assertNotIn("strata_mcp", self.upstream_engine.prompts[0])
        self.assertNotIn("strata_mcp", self.upstream_engine.prompts[1])


class UnreachableUpstream(unittest.TestCase):
    """Nothing is answered locally when the engine cannot be reached: the status surfaces say so."""

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.front_engine = RemoteEngine(f"http://127.0.0.1:{dead_port()}")
        cls.front_svc, cls.front_httpd, cls.base = make_server(cls.front_engine, cls.tok)

    @classmethod
    def tearDownClass(cls):
        cls.front_httpd.shutdown()
        cls.front_httpd.server_close()

    def test_health_is_degraded_and_names_the_reason(self):
        with urllib.request.urlopen(self.base + "/health", timeout=10) as r:
            body = json.loads(r.read())
        self.assertEqual(body["status"], "degraded")
        self.assertFalse(body["engine"]["reachable"])
        self.assertIn("did not answer", body["engine"]["error"])

    def test_generation_fails_on_both_paths(self):
        status, body = post(self.base, "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 503)
        self.assertIn("unreachable", body["error"]["message"])
        payloads = post_stream(self.base, "/v1/chat/completions",
                               {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        items = sse_items(payloads)
        errors = [data for _, data in items if isinstance(data, dict) and data.get("error")]
        self.assertTrue(errors)
        self.assertIn("unreachable", errors[0]["error"]["message"])
        self.assertEqual(items[-1], (None, "[DONE]"))             # the stream still ends properly

    def test_delegated_status_routes_say_503(self):
        for path, body in (("/props", None), ("/v1/models", None), ("/slots", None)):
            request = urllib.request.Request(self.base + path)
            try:
                with urllib.request.urlopen(request, timeout=10) as r:
                    code, data = r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                code, data = e.code, json.loads(e.read())
            self.assertEqual(code, 503, path)
            self.assertIn("did not answer", data["error"]["message"])
        status, body = post(self.base, "/v1/messages/count_tokens",
                            {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 503)

    def test_control_routes_are_refused(self):
        for path, body in (("/unload", None), ("/load", None), ("/v1/unload", {}), ("/v1/load", {}),
                           ("/v1/vram", {"reserve_mib": 0}), ("/v1/images/generations", {"prompt": "x"})):
            request = urllib.request.Request(self.base + path, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=10) as r:
                    code, data = r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                code, data = e.code, json.loads(e.read())
            self.assertEqual(code, 503, path)
            self.assertIn("engine host", data["error"]["message"])

    def test_metrics_still_render_with_engine_up_zero(self):
        with urllib.request.urlopen(self.base + "/metrics?format=prometheus", timeout=10) as r:
            text = r.read().decode()
        self.assertIsNotNone(re.search(r"strata:engine_up\{[^}]*\} 0", text), text[-400:])
        self.assertIn("vllm:num_requests_running", text)
        with urllib.request.urlopen(self.base + "/metrics/prometheus", timeout=10) as r:
            text = r.read().decode()
        self.assertIsNotNone(re.search(r"strata:engine_up 0", text), text[-400:])


class UpstreamAuth(unittest.TestCase):
    """The upstream's bearer key is used on every delegated call and never echoed."""

    KEY = "upstream-test-key"

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.upstream_engine = MockEngine(cls.tok, [GREETING], max_context=CTX)
        cls.up_svc, cls.upstream_httpd, cls.upstream_base = make_server(cls.upstream_engine, cls.tok,
                                                                        api_key=cls.KEY)
        cls.ok_engine = RemoteEngine(cls.upstream_base, api_key=cls.KEY)
        cls.ok_svc, cls.ok_httpd, cls.ok_base = make_server(cls.ok_engine, cls.tok)
        cls.bad_engine = RemoteEngine(cls.upstream_base, api_key="not-the-key")
        cls.bad_svc, cls.bad_httpd, cls.bad_base = make_server(cls.bad_engine, cls.tok)

    @classmethod
    def tearDownClass(cls):
        for httpd in (cls.ok_httpd, cls.bad_httpd, cls.upstream_httpd):
            httpd.shutdown()
            httpd.server_close()

    def test_the_right_key_is_delegated(self):
        status, body = post(self.ok_base, "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], "Hello from the mock engine.")

    def test_the_wrong_key_is_passed_through_as_401(self):
        status, body = post(self.bad_base, "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 401)
        self.assertIn("error", body)

    def test_the_key_never_appears_in_a_response(self):
        for base in (self.ok_base, self.bad_base):
            with urllib.request.urlopen(base + "/health", timeout=10) as r:
                text = r.read().decode()
            self.assertNotIn(self.KEY, text)
            with urllib.request.urlopen(base + "/metrics?format=prometheus", timeout=10) as r:
                self.assertNotIn(self.KEY, r.read().decode())


class StalledUpstream(unittest.TestCase):
    """engine_silence_s bounds the upstream's silence, as it bounds a local engine's."""

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.upstream_engine = RecordingMock(cls.tok, LONG_ANSWER, max_context=CTX, delay_s=3.0)
        cls.up_svc, cls.upstream_httpd, cls.upstream_base = make_server(cls.upstream_engine, cls.tok)
        cls.front_engine = RemoteEngine(cls.upstream_base, silence_s=1.0)
        cls.front_svc, cls.front_httpd, cls.base = make_server(cls.front_engine, cls.tok)

    @classmethod
    def tearDownClass(cls):
        cls.front_httpd.shutdown()
        cls.front_httpd.server_close()
        cls.upstream_httpd.shutdown()
        cls.upstream_httpd.server_close()

    def test_a_silent_upstream_ends_the_request(self):
        started = time.monotonic()
        status, body = post(self.base, "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "hi"}]}, timeout=30)
        self.assertEqual(status, 503)
        self.assertIn("stopped answering", body["error"]["message"])
        self.assertLess(time.monotonic() - started, 20)     # it did not wait out the whole answer


class McpSkipRule(unittest.TestCase):
    """A turn that also calls one of the client's own tools runs no MCP call (run_with_mcp's rule)."""

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        both_calls = tool_call_script("fake__echo", text="must not run") + "\n" + \
            tool_call_script("own_tool", q="answer me")
        cls.upstream_engine = RecordingMock(cls.tok, [both_calls, GREETING], max_context=CTX)
        cls.up_svc, cls.upstream_httpd, cls.upstream_base = make_server(cls.upstream_engine, cls.tok)
        cls.front_engine = RemoteEngine(cls.upstream_base)
        cls.hub = McpHub({"fake": {"command": sys.executable, "args": [str(FAKE_MCP)]}})
        cls.hub.start()
        cls.front_svc, cls.front_httpd, cls.base = make_server(cls.front_engine, cls.tok, hub=cls.hub)

    @classmethod
    def tearDownClass(cls):
        cls.front_httpd.shutdown()
        cls.front_httpd.server_close()
        cls.upstream_httpd.shutdown()
        cls.upstream_httpd.server_close()
        cls.hub.close()

    def test_the_mcp_call_is_skipped_and_the_own_call_passes_through(self):
        payloads = post_stream(self.base, "/v1/chat/completions",
                               {"messages": [{"role": "user", "content": "hi"}], "stream": True,
                                "strata_mcp": True,
                                "tools": [{"type": "function",
                                           "function": {"name": "own_tool", "parameters": {"type": "object",
                                                                                           "properties": {}}}}]})
        items = sse_items(payloads)
        mcp_events = [data["strata_mcp"] for _, data in items
                      if isinstance(data, dict) and isinstance(data.get("strata_mcp"), dict)]
        results = [e for e in mcp_events if e.get("event") == "result"]
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].get("skipped"))            # announced, never run: an own call shares the turn
        calls = [c for _, data in items if isinstance(data, dict)
                 for c in ((data["choices"][0]["delta"].get("tool_calls") or [])
                           if data.get("choices") and data["choices"][0].get("delta") else [])]
        names = [c["function"]["name"] for c in calls if c.get("function", {}).get("name")]
        self.assertIn("own_tool", names)                      # the client's tool passes through to be answered
        self.assertNotIn("fake__echo", names)                 # the MCP call is not handed to the client
        finishes = [data["choices"][0]["finish_reason"] for _, data in items if isinstance(data, dict)
                    and data.get("choices") and data["choices"][0]["finish_reason"]]
        self.assertEqual(finishes, ["tool_calls"])


class CliPlan(unittest.TestCase):
    """strata serve --engine remote --plan: the URL rides in the plan, the key never does."""

    def test_plan_carries_the_url_and_not_the_key(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            cfg = Path(temp) / "remote-model.json"
            cfg.write_text(json.dumps({"exe": "echo", "args": [], "model_name": "remote-model",
                                       "tokenizer": "pack/full/tokenizer",
                                       "remote_url": "http://chudmax-engine:8080"}), encoding="utf-8")
            done = subprocess.run([sys.executable, str(ROOT / "strata.py"), "serve", "--profile", str(cfg),
                                   "--engine", "remote", "--engine-url", "http://chudmax-engine:8080",
                                   "--engine-api-key", "secret-plan-key", "--plan", "--output", "json"],
                                  capture_output=True, text=True, timeout=60, cwd=str(ROOT))
        self.assertEqual(done.returncode, 0, done.stderr)
        plan = json.loads(done.stdout)["plan"]
        argv = plan["server_argv"]
        self.assertIn("--engine", argv)
        self.assertIn("remote", argv)
        self.assertIn("--engine-url", argv)
        self.assertIn("http://chudmax-engine:8080", argv)
        self.assertNotIn("secret-plan-key", done.stdout)      # the bearer token rides in the environment


class LaunchFlags(unittest.TestCase):
    """--engine remote validates its URL and refuses the engine host's knobs."""

    def run_server(self, *flags):
        return subprocess.run([sys.executable, str(SERVER), "--engine", "remote", "--host", "127.0.0.1",
                               "--port", str(dead_port()), *flags],
                              capture_output=True, text=True, timeout=60, cwd=str(ROOT))

    def test_missing_url_is_refused(self):
        done = self.run_server()
        self.assertEqual(done.returncode, 2)
        self.assertIn("--engine-url", done.stderr)

    def test_a_non_http_url_is_refused(self):
        done = self.run_server("--engine-url", "ftp://example-host:8080")
        self.assertEqual(done.returncode, 2)
        self.assertIn("http://", done.stderr)

    def test_the_engine_hosts_flags_are_refused(self):
        done = self.run_server("--engine-url", "http://127.0.0.1:8080", "--parallel", "2")
        self.assertEqual(done.returncode, 2)
        self.assertIn("--parallel", done.stderr)

    def test_the_config_remote_url_is_accepted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            cfg = Path(temp) / "remote-model.json"
            cfg.write_text(json.dumps({"tokenizer": "pack/full/tokenizer",
                                       "remote_url": "http://chudmax-engine:8080"}), encoding="utf-8")
            done = subprocess.run([sys.executable, str(SERVER), "--engine", "remote", "--config", str(cfg),
                                   "--host", "127.0.0.1", "--port", str(dead_port()), "--parallel", "2"],
                                  capture_output=True, text=True, timeout=60, cwd=str(ROOT))
        # --parallel is refused AFTER the URL resolved from the config: the config fallback works
        self.assertEqual(done.returncode, 2)
        self.assertIn("--parallel", done.stderr)

    def test_redact_strips_userinfo(self):
        self.assertEqual(redact("http://chudmax-engine:8080"), "http://chudmax-engine:8080")
        self.assertEqual(redact("http://user:secret@chudmax-engine:8080/v1"), "http://chudmax-engine:8080/v1")

    def test_no_rfc1918_addresses_in_the_new_sources(self):
        pattern = re.compile(r"\b(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d+\.\d+\b")
        for path in (ROOT / "serve" / "remote.py", Path(__file__)):
            self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")), f"RFC1918 literal in {path.name}")


if __name__ == "__main__":
    unittest.main()
