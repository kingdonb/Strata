# Autonomous Co-Piloting, Rootless Apple Silicon Telemetry, and the Tale of the Recycled Thread ID

*Date: October 25, 2026*  
*Authors: Kingdon Barrett & Gemini Antigravity (with Helix Chief of Staff)*  

---

## 1. Introduction: Pair Programming Across Real Silicon and Autonomous Agents

When building sovereign local AI stacks, there is no substitute for physical silicon. Simulators and synthetic test harnesses are valuable, but the real world always introduces friction: thermal limits on unified memory, OS scheduling quirks across thread pools, and process lifecycle edge cases.

Over the past week, we developed, stabilized, and validated **Strata** (our high-performance fork serving 35B+ MoE models like Qwen3.6-35B-A3B on Apple Silicon and commodity PC GPUs). What made this development cycle unique was the collaboration architecture:

- **The Human Operator (Kingdon)**: Steering architectural direction, approving release gates, and testing on real Apple Silicon hardware.
- **The Local Hardware Agent (Gemini Antigravity)**: Living in the local macOS terminal with access to compilers, Metal toolchains, unit tests, and live process metrics.
- **The Project Agent (Helix Chief of Staff)**: Operating in a remote orchestration environment, running automated CI matrix verifications, maintaining the spec ledger (`helix-specs`), and providing delegated PR reviews.
- **Direct Agent-to-Agent Communication**: Antigravity and Chief of Staff communicating directly over the Helix CLI (`helix org bots chat chief-of-staff`), coordinating git pull requests, verifying test results, and agreeing on strategic upstream contributions.

Here is the story of how we brought rootless GPU temperature telemetry to Apple Silicon, solved a classic process-watcher race condition, debugged an obscure Linux thread-ID recycling flake, and achieved a 100% green CI gate across 1,036 test cases.

---

## 2. Unlocking Rootless Apple Silicon GPU Thermals

### The Blank Gauge Problem
In Strata’s Terminal Overview (`strata monitor`) and Web UI, the GPU temperature gauge on Apple Silicon was conspicuously empty:
```
Hardware
GPU 0   Apple M2             0.9/17.8 GiB        20%    -- C       --
```
Under macOS, GPU telemetry conventionally relies on `powermetrics`, which requires `sudo` privileges. Because Strata is designed to run cleanly as an unprivileged user process without root access, temperature polling was previously skipped for the Metal backend.

### The Solution: IOHIDEventSystemClient
Apple Silicon devices expose an extensive array of hardware thermal sensors (such as `AppleARMPMUTempSensor`) through the private `IOHIDEventSystemClient` API in `IOKit` and `CoreFoundation`. These sensors report die-level temperatures across performance cores, efficiency cores, and the integrated GPU—and crucially, reading them does **not** require root privileges.

Using Python's standard `ctypes`, we implemented a lightweight bridge in `serve/telemetry.py` that queries IOHID thermal sensors dynamically:
```python
# Matching primary usage page 0xff00 (Vendor-defined) and usage 5 (Temperature)
client = iokit.IOHIDEventSystemClientCreate(None)
matching = cf.CFDictionaryCreateMutable(...)
# Discover PMU tdie* and tdev* sensors
...
```
Now, every second, `Telemetry.sample()` captures the peak active temperature across the Apple Silicon GPU and SoC. The result in `strata monitor`:
```
Hardware
GPU 0   Apple M2             0.9/17.8 GiB        20%    39 C       --
```
With real-time sparkline rolling history rendered right in the terminal and web dashboard.

---

## 3. The Lifeline Process Watcher & The State-Letter Race

In a local LLM server, the engine process must never outlive its server. If an inference server crashes hard (`SIGKILL`), any lingering engine process holding 20 GB of unified memory becomes an orphan that blocks further requests and drains battery.

Strata uses a portable **lifeline watcher** process that monitors a pipe connected to the parent server. When the parent dies, the pipe closes, and the watcher ensures the child process is terminated. To prevent accidentally killing an unrelated process if a PID is rapidly reused by the OS, the watcher takes an **identity token** at spawn time and verifies it before issuing any kill signal.

### The Bug
The identity token was originally constructed using:
```bash
ps -o lstart=,stat= -p <pid>
```
Notice the `stat=` column! In Unix, the process state letter dynamically flips between `R` (running) when spinning on prompt ingestion and `S` (sleeping) when waiting on I/O or Metal command buffers.

In CI and local testing:
1. The token was captured at spawn time while the process was in state `R`.
2. When the watcher evaluated the child after the pipe closed, the child was in state `S`.
3. The watcher compared the token: `"13:00:01 S" != "13:00:01 R"`.
4. Concluding the PID had been reused, the watcher failed open and exited without killing the child!

The fix was conceptually pure: **process identity is start time alone**. By switching to `ps -o lstart=` (and `/proc/<pid>/stat` field 19 on Linux), the token remains strictly invariant throughout the entire lifecycle of the process.

---

## 4. The Heisenbug: Linux `pthread` ID Recycling in Full-Suite CI

With the lifeline fix committed, we ran the full server test suite (`python -m unittest discover -s serve`). The test suite passed standalone, but inside GitHub Actions running all 1,036 tests, one test deterministically flaked:
```
FAIL: test_server_close_ends_the_sampler (test_responses.NoLeakedSampler)
AssertionError: 0 != 1 : the server did not start its hardware sampler
```

### The Investigation
The test was designed to ensure that `Telemetry` background sampler threads do not leak across tests. Upstream had previously attempted to fix test flakiness by comparing thread identities:
```python
def samplers() -> set:
    return {t.ident for t in threading.enumerate() if t.name.endswith("(_loop)")}
before = samplers()
httpd = serve(...)
mine = samplers() - before
self.assertEqual(len(mine), 1, "the server did not start its hardware sampler")
```

Why did `len(mine)` evaluate to `0`?
In a test runner executing over a thousand tests, a previous test's server sampler thread can be in the final milliseconds of its exit when `before = samplers()` runs.
1. `before` records the thread ID of the exiting sampler: `{140234567890}`.
2. The dying thread terminates.
3. The new `httpd = serve(...)` spawns a new sampler thread.
4. The Linux `pthread` allocator immediately **recycles that exact thread ID** for the new thread!
5. `samplers() - before` performs `{140234567890} - {140234567890} = set()`.
6. `len(mine) == 0`!

### The Fix: Identity by Reference
Global process scanning and thread-ID math is inherently susceptible to recycling races. We refactored `Telemetry` to simply retain a reference to its own thread:
```python
self._thread = threading.Thread(target=self._loop, daemon=True)
self._thread.start()
```
And updated the test to assert directly on the service's own thread handle:
```python
thread = httpd.svc.telemetry._thread
self.assertTrue(thread.is_alive())
httpd.shutdown()
httpd.server_close()
thread.join(timeout=5.0)
self.assertFalse(thread.is_alive())
```
No global enumeration. No string-matching on thread names. Zero race conditions. The full 1,036-test suite passed in 347 seconds flat.

---

## 5. Multi-Agent Governance: Antigravity Meets Helix Chief of Staff

Throughout this sprint, changes were governed under strict protocols:
- **No Direct Pushes to `main`**: All features and fixes must land via PRs with green CI checks.
- **Verify-Then-Act**: Every claim made by an agent must be backed by reproducible terminal output, Git SHAs, or GitHub API checks.
- **Upstream Collaboration**: Rather than racing upstream PRs with competing forks, we cherry-picked and refined upstream PR #3590 by `@justindressler`, validated it on real Apple Silicon, and contributed our findings (the pthread recycling fix and Darwin mock guard) back to upstream.

When PR #27 was opened, Chief of Staff reviewed the diff via GitHub API, checked the fail-open semantics, and approved the merge:
> *"PR #27 review: APPROVED... Production path untouched... fail-open semantics preserved... `NoLeakedSampler` goes from name-suffix thread-counting with polling to identity-by-reference... Deterministic, simpler, same contract proven."*

---

## 6. What’s Next: M2 Validation Session #4

With `main` at `af38e089` and all three PRs (#25, #26, #27) merged, we are kicking off **M2 Validation Session #4**:
1. **4K+ Prompt Prefill**: Running a 4,200+ token context through Qwen3.6-35B-A3B without expert-cache loans or Metal aborts.
2. **Sustained Thermal Telemetry**: Observing die thermals in real-time under continuous generation.
3. **Lifeline Watcher Validation**: Exercising mid-decode hard kills to guarantee zero orphaned engine memory.

All raw logs, gate outputs, and metrics will land in `docs/m2-sessions/2026-10-25-session-4/`, establishing our first live archive-backed session.

*The future of software development isn’t just agents writing code—it’s specialized agents cooperating across real hardware, verifying ground truth, and learning together.*
