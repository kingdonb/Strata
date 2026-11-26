"""serve/remote.py - `--engine remote`: a stateless front-end that delegates generation to a resident
Strata server over HTTP (--engine-url, bearer with --engine-api-key / $STRATA_ENGINE_API_KEY, or
"remote_url" in the config).

The front-end keeps everything that needs no model: the web UI, the MCP hub (its own --mcp-config),
the Host/Origin guard, its own API keys, admission and its metrics.  What a model is needed for is
delegated to the upstream - today's server, unchanged - which owns the template, tokenizer, reasoning
parsing and tool-call parsing and answers with structured deltas:

  POST /v1/chat/completions  -> upstream /v1/chat/completions   (the MCP tool loop runs here, locally)
  POST /v1/messages          -> upstream /v1/messages           (+ /v1/messages/count_tokens)
  POST /v1/responses         -> upstream /v1/responses
  GET  /health /props /v1/models /slots -> the same upstream routes

Each upstream round is streamed (even for a non-streaming client: a cancelled read is a cancelled
generation, and the upstream's own disconnect watch turns the closed connection into an engine
cancel).  The upstream's deltas are mapped into the event protocol the local pipeline yields
(serve.frontend.Event), so the route handlers and chunk builders are shared with the local engines.
Nothing is answered locally when the upstream cannot be reached: every delegated route fails with
the dialect's error shape, and the status surfaces report the outage instead of guessing.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlsplit, urlunsplit

from serve.frontend import Event, ToolCall
from serve.mcp import McpCancelled
from serve.server_parts.engines import EngineDied

PROBE_TTL_S = 5.0            # how long a health answer is reused (status surfaces poll with this)
PROBE_TIMEOUT_S = 3.0        # one health probe
CALL_TIMEOUT_S = 15.0        # connect and the small delegated GET/POST calls
PING_S = 15.0                # an SSE keep-alive while the upstream sends nothing
OPENAI_ = "/v1/chat/completions"
ANTHROPIC_ = "/v1/messages"
RESPONSES_ = "/v1/responses"


def is_remote(svc) -> bool:
    """True when this server delegates (its engine is a serve.remote.RemoteEngine)."""
    return bool(getattr(getattr(svc, "engine", None), "remote", False))


def redact(url: str) -> str:
    """The URL without any userinfo (a key in the address is never printed anywhere)."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc.split("@", 1)[-1], parts.path, "", ""))


class UpstreamError(RuntimeError):
    """The upstream engine failed this request.  `status` is what the client is told; `body`, when
    set, is the upstream's own (dialect's) error body, passed on unchanged."""

    def __init__(self, message: str, status: int = 503, body=None):
        super().__init__(message)
        self.status = status
        self.body = body


class UpstreamUnreachable(UpstreamError):
    """The upstream could not be reached (refused, reset, timed out): a 503, never a local fallback."""


class UpstreamStatus(UpstreamError):
    """The upstream answered with a non-200 status; its body (already in the dialect's shape) passes
    through.  Built as (status, body, message) at every call site."""

    def __init__(self, status, body=None, message=""):
        self.status = int(status)
        self.body = body if isinstance(body, dict) else None
        super().__init__(message or f"the engine answered {self.status}", self.status, self.body)


class RemoteEngine:
    """The engine shell for a `--engine remote` front-end: health probing and the bookkeeping surface
    Service reads (max_context, batch, info, alive).  Generation is never asked of it - the request
    routes delegate over HTTP - so generate() failing loudly is a bug guard, not a code path."""

    remote = True

    def __init__(self, url: str, api_key: str = "", silence_s: float = 300.0):
        self.url = url.strip().rstrip("/")
        self.api_key = api_key or ""
        self.silence_s = silence_s                      # the idle-read watchdog, as engine_silence_s is for a local engine
        self.max_context = 0
        self.batch = 1
        self.info: dict = {"remote": True, "url": redact(self.url)}
        self.model_name = None
        self.family = None
        self.images = False                             # the upstream's picture reader, when its health says so
        self.remote_loaded = False
        self.last_error = None                          # the last probe's failure, for the status surfaces
        self.on_probe = None                            # set by main(): learns the upstream's model name
        self._lock = threading.Lock()
        self._probed_at = 0.0
        self._health = None                             # the last successful /health body

    # -- the probe ---------------------------------------------------------------

    def redacted_url(self) -> str:
        return redact(self.url)

    def probe(self, force: bool = False) -> dict | None:
        """The upstream's /health, cached for PROBE_TTL_S.  The health body, or None when the engine
        did not answer (then last_error says why)."""
        with self._lock:
            if not force and self._health is not None and time.monotonic() - self._probed_at < PROBE_TTL_S:
                return self._health
            try:
                _, health = http_json(self, "/health", timeout=PROBE_TIMEOUT_S)
            except UpstreamError as e:
                self._reachable(False, str(e))
                return None
            self._learn_health(health)
            return self._health

    def _learn_health(self, health: dict) -> None:
        self._health = health if isinstance(health, dict) else {}
        self._probed_at = time.monotonic()
        self.last_error = None
        self.max_context = int(self._health.get("max_context") or 0)
        self.model_name = self._health.get("model") or self.model_name
        self.family = self._health.get("family")
        self.images = bool(self._health.get("images"))
        self.remote_loaded = bool(self._health.get("loaded"))
        self.info = {**self.info, "model": self.model_name, "family": self.family}
        if callable(self.on_probe):
            try:
                self.on_probe(self._health)
            except Exception:                           # noqa: BLE001 - a listener must never break a probe
                pass

    def _reachable(self, up: bool, error: str | None = None) -> None:
        self._probed_at = time.monotonic()
        self._health = None
        if not up:
            self.last_error = error
            self.remote_loaded = False

    def alive(self) -> bool:
        """The upstream answered its health probe and says its model is loaded."""
        health = self.probe()
        return bool(health) and bool(health.get("loaded", True))

    def reachable(self) -> bool:
        return self.probe() is not None

    def learn_props(self, props: dict) -> None:
        """What the /props proxy saw: the engine's slots and context."""
        if not isinstance(props, dict):
            return
        self.batch = max(1, int(props.get("total_slots") or 1))
        ctx = ((props.get("default_generation_settings") or {}).get("n_ctx"))
        if isinstance(ctx, int) and ctx > 0:
            self.max_context = ctx
        if isinstance(props.get("model_alias"), str) and props.get("model_alias"):
            self.model_name = props["model_alias"]

    # -- everything else is delegated --------------------------------------------

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        raise EngineDied("the engine is remote (--engine remote): generation is delegated to the "
                         "upstream server; a request route must never call the engine directly")


# -- the HTTP client -------------------------------------------------------------


def _headers(engine: RemoteEngine, extra: dict | None = None) -> dict:
    h = {"Content-Type": "application/json", "Accept": "text/event-stream, application/json"}
    if engine.api_key:
        h["Authorization"] = "Bearer " + engine.api_key
    if extra:
        h.update(extra)
    return h


def http_json(engine: RemoteEngine, path: str, body: dict | None = None, timeout: float = CALL_TIMEOUT_S,
              query: str = "") -> tuple[int, object]:
    """One delegated JSON call: -> (status, parsed body).  Raises UpstreamUnreachable (transport) or
    UpstreamStatus (a non-200 with the upstream's own error body)."""
    url = engine.url + path + (("?" + query) if query else "")
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, headers=_headers(engine), method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        raw = b""
        try:
            raw = e.read()
        except Exception:                               # noqa: BLE001 - the body is best-effort
            pass
        raise UpstreamStatus(e.code, _json_of(raw), _body_message(raw, e.code)) from None
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise UpstreamUnreachable(f"the engine at {engine.redacted_url()} did not answer: "
                                  f"{type(e).__name__}") from None
    return status, _json_of(raw)


def _json_of(raw):
    try:
        return json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", "replace"))
    except ValueError:
        return None


def _body_message(raw: bytes, status: int) -> str:
    """A sentence for the client's error from the upstream's body (its own message when there is one)."""
    data = _json_of(raw)
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
        if data.get("message"):
            return str(data["message"])
        if data.get("type") == "error" and isinstance(data.get("error"), dict):
            return str((data.get("error") or {}).get("message") or f"the engine answered {status}")
    return f"the engine answered {status}"


class _UpstreamStream:
    """One streaming upstream call.  A worker thread reads the SSE into a bounded queue so the
    consumer can poll with a short timeout: a cancelled or stalled read is ended by closing the
    response, which is also how a client disconnect reaches the upstream as a cancellation."""

    def __init__(self, engine: RemoteEngine, path: str, body: dict, silence_s: float):
        self.engine, self.path, self.body = engine, path, body
        # the per-read watchdog is the engine's silence (0 = wait, as a local engine does); the health probe
        # the routes run right before bounds the connect
        self.timeout = silence_s if silence_s and silence_s > 0 else None
        self.resp = None
        self.cancel: threading.Event | None = None
        self._q: queue.Queue = queue.Queue(maxsize=1024)
        self._thread = None
        self._closed = False

    def start(self) -> "_UpstreamStream":
        request = urllib.request.Request(self.engine.url + self.path, data=json.dumps(self.body).encode("utf-8"),
                                         headers=_headers(self.engine), method="POST")
        try:
            self.resp = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raw = b""
            try:
                raw = e.read()
            except Exception:                           # noqa: BLE001
                pass
            raise UpstreamStatus(e.code, _json_of(raw), _body_message(raw, e.code)) from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise UpstreamUnreachable(f"the engine at {self.engine.redacted_url()} did not answer: "
                                      f"{type(e).__name__}") from None
        self._thread = threading.Thread(target=self._read, daemon=True, name="strata-remote-read")
        self._thread.start()
        return self

    def _read(self) -> None:
        event_name = None
        try:
            while True:
                line = self.resp.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").strip("\r\n")
                if not text or text.startswith(":"):
                    continue                             # a keep-alive comment
                if text.startswith("event:"):
                    event_name = text[len("event:"):].strip()
                    continue
                if text.startswith("data:"):
                    self._q.put(("data", text[len("data:"):].strip(), event_name))
                    event_name = None
        except Exception as e:                          # noqa: BLE001 - reported through the queue
            if not self._closed:
                self._q.put(("error", e, None))
        finally:
            self._q.put(None)

    def events(self, cancel: threading.Event | None = None):
        """The SSE items as (event name or None, data text); an SSE keep-alive of its own every PING_S
        while the upstream sends nothing.  Raises UpstreamUnreachable when the read failed."""
        self.cancel = cancel
        next_ping = time.monotonic() + PING_S
        while True:
            if cancel is not None and cancel.is_set():
                self.close()
                return
            try:
                item = self._q.get(timeout=0.25)
            except queue.Empty:
                if time.monotonic() >= next_ping:
                    yield ("ping", None)
                    next_ping = time.monotonic() + PING_S
                continue
            if item is None:
                return
            kind, data, name = item
            if kind == "error":
                if cancel is not None and cancel.is_set():
                    return
                raise UpstreamUnreachable(f"the engine at {self.engine.redacted_url()} stopped answering "
                                          f"mid-stream ({type(data).__name__})")
            yield (name, data)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.resp is not None:
            try:
                self.resp.close()
            except Exception:                           # noqa: BLE001
                pass


# -- the orchestrator ------------------------------------------------------------


def _upstream_body(req: dict, dialect: str, messages=None, tools=None) -> dict:
    """The body forwarded upstream: the client's own, `strata_mcp` stripped (the upstream's hub must
    not engage - the front-end runs the tools), streamed so a cancelled read cancels the generation."""
    body = {k: v for k, v in req.items() if k != "strata_mcp"}
    body["stream"] = True
    if dialect == "openai":
        if messages is not None:
            body["messages"] = messages
        if tools is None:
            body.pop("tools", None)                     # the finalizing round: no tools are offered
            body.pop("tool_choice", None)
        elif tools:
            body["tools"] = tools
        so = body.get("stream_options")
        so = dict(so) if isinstance(so, dict) else {}
        so["include_usage"] = True                      # the front-end records the engine's real usage
        body["stream_options"] = so
    return body


def _finish_of(finish: str | None, cancel: threading.Event) -> str:
    """The done dict's finish for the OpenAI dialect: a turn that ended in tool calls reports "stop"
    (with the calls as events) so openai_chunks derives "tool_calls" from them; an unknown reason
    passes through ("content_filter" is a real OpenAI finish_reason)."""
    if cancel.is_set():
        return "cancel"
    return "stop" if finish in (None, "tool_calls", "function_call") else (finish or "stop")


def _tool_arguments(text: str) -> dict:
    try:
        parsed = json.loads(text) if text.strip() else {}
    except ValueError:
        return {}                                       # the upstream validated the call; never invent one
    return parsed if isinstance(parsed, dict) else {}


def accounted(svc, run, cancel: threading.Event):
    """The request's own accounting while the events pass through: the live status, then the totals,
    history record and latency histogram from the upstream's usage (the local engines do this inside
    Service._run)."""
    started = time.time()
    request_id = uuid.uuid4().hex[:12]
    first, n, done = None, 0, None
    try:
        with svc.status_lock:
            svc.status.update(busy=True, phase="delegating to the engine", prompt_tokens=None,
                              generated=0, started=started, first_token=None, tool=None, tail="")
        for kind, x in run:
            if kind == "event" and getattr(x, "kind", None) in ("content", "reasoning") and x.text:
                if first is None:
                    first = time.time()
                    with svc.status_lock:
                        svc.status["first_token"] = first
                n += 1                          # a piece of text; the usage in "done" is the real count
                with svc.status_lock:
                    svc.status["generated"] = n
            elif kind == "done":
                done = x
            yield kind, x
    finally:
        now = time.time()
        pt = int((done or {}).get("prompt_tokens") or 0)
        ct = int((done or {}).get("completion_tokens") or 0)
        finish = (done or {}).get("finish") or ("cancel" if cancel.is_set() else "error")
        with svc.status_lock:
            t = svc.totals
            t["requests"] += 1
            t["prompt_tokens"] += pt
            t["output_tokens"] += ct
            svc.latencies.observe(first - started if first else None, ct, None, now - started)
            svc.history.append({"id": request_id, "projection": None, "time": started,
                                "duration_s": round(now - started, 1), "finish": finish, "gen": None,
                                "prompt_tokens": pt, "reused": None, "output_tokens": ct,
                                "first_engine_token_s": first - started if first else None,
                                "prompt_total": pt, "prompt_read": None, "reasoning_recoveries": 0,
                                "reasoning_repeat_coverage": None, "engine_generated": None,
                                "prompt_ms": None, "decode_ms": None, "decode_tok_s": None,
                                "hit_rate": None, "pcie_share": None, "ram_blobs": None,
                                "file_blobs": None, "file_mb": None,
                                "drafts_offered": None, "drafts_accepted": None})
            svc.status["prompt_tokens"] = pt
            svc.status["busy"] = False
            svc.status.pop("tail", None)
            svc.status.pop("tool", None)
        svc.last_request_at = time.time()
        print(f"[strata] done: {ct} tokens in {now - started:.0f} s (delegated, {finish}, "
              f"cancel={cancel.is_set()})", flush=True)


def remote_openai(svc, req: dict, tools, mcp_names, cancel: threading.Event):
    """The events of one /v1/chat/completions request, delegated.  `tools`: the request's own tools
    plus the MCP schemas (injected by the route); `mcp_names`: which of them this front-end runs.
    Without MCP names this is one upstream round; with them, the local tool loop."""
    engine: RemoteEngine = svc.engine
    if engine.probe(force=True) is None:
        raise UpstreamUnreachable(f"the engine at {engine.redacted_url()} is unreachable"
                                  + (f" ({engine.last_error})" if engine.last_error else ""))
    hub = svc.mcp
    max_rounds = int(hub.settings["max_rounds"]) if (mcp_names and hub) else 1
    messages = list(req.get("messages") or [])
    total_ct, last_pt, reasoning_tokens = 0, 0, 0
    timings, flags = None, {}
    rounds, finalizing = 0, False
    while True:
        body = _upstream_body(req, "openai", messages=messages, tools=None if finalizing else tools)
        stream = _UpstreamStream(engine, OPENAI_, body, engine.silence_s).start()
        try:
            outcome = yield from _openai_round(stream, cancel, mcp_names or frozenset())
        finally:
            stream.close()
        if cancel.is_set():
            yield "done", {"finish": "cancel", "completion_tokens": total_ct, "prompt_tokens": last_pt,
                           "reasoning_tokens": reasoning_tokens}
            return
        if outcome.usage is not None:
            last_pt = int(outcome.usage.get("prompt_tokens") or 0)
            total_ct += int(outcome.usage.get("completion_tokens") or 0)
            reasoning_tokens = int((outcome.usage.get("completion_tokens_details") or {})
                                   .get("reasoning_tokens") or 0)
        timings = outcome.timings or timings
        flags.update(outcome.flags)
        run_them = bool(outcome.mcp_calls) and not outcome.own_calls and not finalizing \
            and outcome.finish in (None, "stop", "tool_calls", "function_call") and not cancel.is_set()
        if not run_them:
            for c in outcome.mcp_calls:                 # announced, never run: close them in the client's view
                yield "mcp", {"event": "result", "id": c.id, "ok": False, "skipped": True, "text": "not run",
                              "chars": 0, "truncated": False, "ms": 0}
            done = {"finish": _finish_of(outcome.finish, cancel), "completion_tokens": total_ct,
                    "prompt_tokens": last_pt, "reasoning_tokens": reasoning_tokens}
            done.update({"timings": timings} if timings else {})
            done.update(flags)
            yield "done", done
            return
        rounds += 1
        results = []
        for c in outcome.mcp_calls:
            server, tool = hub.routes().get(c.name, (None, c.name))
            yield "mcp", {"event": "call", "id": c.id, "name": c.name, "server": server.name if server else None,
                          "tool": tool, "arguments": c.arguments, "round": rounds}
            box = {}

            def work(c=c, box=box):
                try:
                    box["r"] = hub.call(c.name, c.arguments, cancel)
                except McpCancelled:
                    box["cancelled"] = True

            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            try:
                while worker.is_alive():
                    worker.join(1.0)
                    if worker.is_alive():
                        yield "ping", None              # a keep-alive while a slow tool runs
            except GeneratorExit:
                cancel.set()                            # the client is gone: stop the tool too
                raise
            if "r" not in box:
                break
            r = box["r"]
            print(f"[strata] tool {c.name}: {'ok' if r['ok'] else 'error'}, {r['chars']:,} characters in "
                  f"{r['ms'] / 1000:.1f} s{' (truncated for the model)' if r['truncated'] else ''}", flush=True)
            results.append(r["text"])
            yield "mcp", {"event": "result", "id": c.id,
                          **{k: r[k] for k in ("ok", "text", "chars", "truncated", "ms")}}
        if cancel.is_set() or len(results) < len(outcome.mcp_calls):
            yield "done", {"finish": "cancel", "completion_tokens": total_ct, "prompt_tokens": last_pt,
                           "reasoning_tokens": reasoning_tokens}
            return
        messages.append({"role": "assistant", "content": outcome.text.strip(),
                         **({"reasoning_content": outcome.reasoning.strip()} if outcome.reasoning else {}),
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.name,
                                                      "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                                        for c in outcome.mcp_calls]})
        messages += [{"role": "tool", "tool_call_id": c.id, "content": r}
                     for c, r in zip(outcome.mcp_calls, results)]
        if rounds >= max_rounds:
            finalizing = True
            system_end = next((i for i, m in enumerate(messages)
                               if m.get("role") not in ("system", "developer")), len(messages))
            messages.insert(system_end, {"role": "system", "content":
                f"The tool-use limit of {max_rounds} rounds has been reached. No tools are available for this final "
                "reply. Answer the user's original request in their language using the results already obtained. "
                "Explain any gaps in the available evidence. Do not request or emit further tool calls."})
            yield "mcp", {"event": "limit", "max_rounds": max_rounds}


class _RoundOutcome:
    """What one upstream round said, for the orchestrator's loop."""

    def __init__(self, finish=None, usage=None, timings=None, flags=None, mcp_calls=None, own_calls=None,
                 text="", reasoning=""):
        self.finish = finish
        self.usage = usage
        self.timings = timings
        self.flags = flags or {}
        self.mcp_calls = mcp_calls or []
        self.own_calls = own_calls or []
        self.text = text
        self.reasoning = reasoning


def _openai_round(stream: _UpstreamStream, cancel: threading.Event, mcp_names):
    """One upstream chat-completions round: its deltas as events, its tool calls collected by index.
    Returns a _RoundOutcome."""
    streamed = {}                                   # upstream index -> {"call", "args", "mcp"}
    mcp_calls, own_calls = [], []
    text, reasoning = [], []
    finish, usage, timings = None, None, None
    flags = {}
    for name, data in stream.events(cancel):
        if data is None:                                # ("ping", None): a keep-alive for the chunk layer
            yield "ping", None
            continue
        if data == "[DONE]":
            break
        chunk = _json_of(data.encode("utf-8"))
        if not isinstance(chunk, dict):
            continue
        if chunk.get("strata_mcp"):
            continue                                # the upstream's hub never runs here; ignore if it ever did
        if chunk.get("error"):
            err = chunk["error"]
            raise UpstreamStatus(502, None, err.get("message") if isinstance(err, dict) else str(err))
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        if chunk.get("timings"):
            timings = chunk["timings"]
        for key in ("strata_rejected_tool_calls", "strata_unread_tool_markup"):
            if chunk.get(key):
                flags[key] = chunk[key]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
        delta = choice.get("delta") or {}
        rc = delta.get("reasoning_content")
        if rc:
            reasoning.append(rc)
            yield "event", Event("reasoning", rc)
        content = delta.get("content")
        if content:
            text.append(content)
            yield "event", Event("content", content)
        for tc in delta.get("tool_calls") or []:
            index = tc.get("index")
            fn = tc.get("function") or {}
            slot = streamed.get(index)
            if slot is None:
                call = ToolCall(name=fn.get("name") or "", arguments={},
                                id=tc.get("id") or ("call_" + uuid.uuid4().hex[:24]))
                slot = streamed[index] = {"call": call, "args": "", "mcp": call.name in mcp_names}
                if slot["mcp"]:
                    yield "mcp", {"event": "start", "id": call.id, "name": call.name}
                else:
                    yield "event", Event("tool_start", call=call)
            if fn.get("arguments"):
                piece = fn["arguments"]
                slot["args"] += piece
                if not slot["mcp"]:                 # the front-end runs the MCP calls; their pieces are not streamed
                    yield "event", Event("tool_args", text=piece, call=slot["call"])
    for index in sorted(streamed, key=lambda x: (x is None, x)):
        slot = streamed[index]
        call = ToolCall(name=slot["call"].name, arguments=_tool_arguments(slot["args"]), id=slot["call"].id)
        if slot["mcp"]:
            mcp_calls.append(call)
        else:
            own_calls.append(call)
            yield "event", Event("tool_call", call=call)
    return _RoundOutcome(finish, usage, timings, flags, mcp_calls, own_calls,
                         "".join(text), "".join(reasoning))


def remote_anthropic(svc, req: dict, cancel: threading.Event):
    """The events of one /v1/messages request, delegated (one upstream round; the MCP loop is an
    OpenAI-route feature here, as it is locally)."""
    engine: RemoteEngine = svc.engine
    if engine.probe(force=True) is None:
        raise UpstreamUnreachable(f"the engine at {engine.redacted_url()} is unreachable"
                                  + (f" ({engine.last_error})" if engine.last_error else ""))
    stream = _UpstreamStream(engine, ANTHROPIC_, _upstream_body(req, "anthropic"), engine.silence_s).start()
    try:
        outcome = yield from _anthropic_round(stream, cancel)
    finally:
        stream.close()
    yield "done", {"finish": _finish_of(outcome.finish, cancel),
                   "prompt_tokens": outcome.input_tokens, "completion_tokens": outcome.output_tokens}


class _AnthropicOutcome:
    def __init__(self, finish=None, input_tokens=0, output_tokens=0):
        self.finish = finish
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def _anthropic_round(stream: _UpstreamStream, cancel: threading.Event):
    """One upstream /v1/messages round: its typed blocks as events.  Returns an _AnthropicOutcome."""
    blocks = {}                                     # index -> {"kind", "id", "name", "args"}
    input_tokens, output_tokens, stop_reason = 0, 0, None
    for name, data in stream.events(cancel):
        if data is None:                                # ("ping", None): a keep-alive for the chunk layer
            yield "ping", None
            continue
        ev = _json_of(data.encode("utf-8"))
        if not isinstance(ev, dict):
            continue
        kind = ev.get("type")
        if kind == "ping":
            continue
        if kind == "message_start":
            message = ev.get("message") or {}
            input_tokens = int((message.get("usage") or {}).get("input_tokens") or 0)
        elif kind == "content_block_start":
            block = ev.get("content_block") or {}
            if block.get("type") == "tool_use":
                entry = blocks[ev.get("index")] = {"kind": "tool_use", "id": block.get("id")
                                                   or ("call_" + uuid.uuid4().hex[:24]),
                                                   "name": block.get("name") or "", "args": ""}
                yield "event", Event("tool_start", call=ToolCall(name=entry["name"], arguments={}, id=entry["id"]))
            else:
                blocks[ev.get("index")] = {"kind": block.get("type")}
        elif kind == "content_block_delta":
            entry = blocks.get(ev.get("index"))
            delta = ev.get("delta") or {}
            if delta.get("type") == "thinking_delta" and delta.get("thinking"):
                yield "event", Event("reasoning", delta["thinking"])
            elif delta.get("type") == "text_delta" and delta.get("text"):
                yield "event", Event("content", delta["text"])
            elif delta.get("type") == "input_json_delta" and delta.get("partial_json") \
                    and entry and entry["kind"] == "tool_use":
                entry["args"] += delta["partial_json"]
                yield "event", Event("tool_args", text=delta["partial_json"],
                                     call=ToolCall(name=entry["name"], arguments={}, id=entry["id"]))
        elif kind == "content_block_stop":
            entry = blocks.get(ev.get("index"))
            if entry and entry["kind"] == "tool_use":
                yield "event", Event("tool_call", call=ToolCall(name=entry["name"],
                                                                arguments=_tool_arguments(entry["args"]),
                                                                id=entry["id"]))
        elif kind == "message_delta":
            stop_reason = (ev.get("delta") or {}).get("stop_reason") or stop_reason
            output_tokens = int((ev.get("usage") or {}).get("output_tokens") or output_tokens)
        elif kind == "error":
            err = ev.get("error") or {}
            raise UpstreamStatus(502, None, str(err.get("message") or "the engine reported an error"))
        elif kind == "message_stop":
            break
    # the local protocol's vocabulary (anthropic_events maps it to Anthropic's stop_reason):
    # end_turn and its cousins are a normal stop, tool_use ends the turn with the call events out
    finish = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "stop",
              "pause_turn": "stop", "refusal": "stop"}.get(stop_reason, "stop")
    return _AnthropicOutcome(finish, input_tokens, output_tokens)


def remote_responses(svc, req: dict, cancel: threading.Event):
    """The events of one /v1/responses request, delegated (one upstream round; the front-end's own
    Assembler rebuilds the typed events, so both modes keep the same items)."""
    engine: RemoteEngine = svc.engine
    if engine.probe(force=True) is None:
        raise UpstreamUnreachable(f"the engine at {engine.redacted_url()} is unreachable"
                                  + (f" ({engine.last_error})" if engine.last_error else ""))
    stream = _UpstreamStream(engine, RESPONSES_, _upstream_body(req, "responses"), engine.silence_s).start()
    try:
        finish, usage = yield from _responses_round(stream, cancel)
    finally:
        stream.close()
    done = {"finish": _finish_of(finish, cancel), "prompt_tokens": 0, "completion_tokens": 0}
    if isinstance(usage, dict):
        done["prompt_tokens"] = int(usage.get("input_tokens") or 0)
        done["completion_tokens"] = int(usage.get("output_tokens") or 0)
        done["reasoning_tokens"] = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
        done["reused"] = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    yield "done", done


def _responses_round(stream: _UpstreamStream, cancel: threading.Event):
    """One upstream /v1/responses round: its deltas as events.  Returns (finish, usage dict or None)."""
    items = {}                                      # output index -> {"kind", "id", "name", "args"}
    usage, finish = None, None
    for name, data in stream.events(cancel):
        if data is None:                                # ("ping", None): a keep-alive for the chunk layer
            yield "ping", None
            continue
        ev = _json_of(data.encode("utf-8"))
        if not isinstance(ev, dict):
            continue
        kind = ev.get("type")
        if kind == "response.output_item.added":
            item = ev.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id") or item.get("id") or ("call_" + uuid.uuid4().hex[:24])
                entry = items[ev.get("output_index")] = {"kind": "function_call", "id": call_id,
                                                         "name": item.get("name") or "", "args": ""}
                yield "event", Event("tool_start", call=ToolCall(name=entry["name"], arguments={}, id=call_id))
        elif kind == "response.function_call_arguments.delta":
            entry = items.get(ev.get("output_index"))
            if entry:
                piece = ev.get("delta") or ""
                entry["args"] += piece
                yield "event", Event("tool_args", text=piece,
                                     call=ToolCall(name=entry["name"], arguments={}, id=entry["id"]))
        elif kind == "response.output_text.delta" and ev.get("delta"):
            yield "event", Event("content", ev["delta"])
        elif kind == "response.reasoning_text.delta" and ev.get("delta"):
            yield "event", Event("reasoning", ev["delta"])
        elif kind == "response.output_item.done":
            item = ev.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id") or item.get("id")
                yield "event", Event("tool_call", call=ToolCall(name=item.get("name") or "",
                                                                arguments=_tool_arguments(item.get("arguments") or ""),
                                                                id=call_id))
        elif kind in ("response.completed", "response.incomplete"):
            usage = (ev.get("response") or {}).get("usage") or usage
            finish = "length" if kind == "response.incomplete" else "stop"
        elif kind == "response.failed":
            err = ((ev.get("response") or {}).get("error") or {})
            raise UpstreamStatus(502, None, str(err.get("message") or "the engine failed the response"))
    return finish, usage


def count_tokens(engine: RemoteEngine, req: dict) -> int:
    """The upstream's count for the same Anthropic body (the model does not run): message_start's
    input_tokens.  Raises UpstreamError when the engine cannot be asked."""
    _, data = http_json(engine, "/v1/messages/count_tokens",
                        {k: v for k, v in req.items() if k != "strata_mcp"})
    return int((data or {}).get("input_tokens") or 0) if isinstance(data, dict) else 0


def proxy_get(engine: RemoteEngine, path: str, query: str = "", timeout: float = CALL_TIMEOUT_S):
    """A delegated GET (/props, /v1/models, /slots): (status, parsed body).  Raises UpstreamError."""
    return http_json(engine, path if path.startswith("/") else "/" + path, timeout=timeout, query=query)
