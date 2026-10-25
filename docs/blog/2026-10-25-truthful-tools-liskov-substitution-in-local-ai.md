# Truthful Tools: What Would Liskov Do? Bringing Real Function Calling to Apple Silicon Local Models

*Date: October 25, 2026*  
*Authors: Kingdon Barrett & Gemini Antigravity (with Helix Chief of Staff)*  

---

## 1. The Observation: "I Don't Have Any Tools"

During hands-on playtesting of our local Apple Silicon setup (running `Qwen3.6-35B-A3B` on Metal with rootless hardware telemetry and speculative decoding), Kingdon noticed a glaring contradiction in the user interface:

The Host Web UI proudly rendered a "Tool activity" panel with example proposal and approval cards, but buried inside was a hardcoded disclaimer:
> *"Tool execution is not connected in this build. Recorded proposals and approvals are examples."*

Worse, when the model was asked directly in chat:
> *"What tools do you have available?"*

The model answered honestly and plainly:
> *"I do not have access to any tools or external functions. I cannot execute commands or browse the web."*

Meanwhile, the UI offered a toggle for "Skills." Looking under the hood, these "Skills" weren't function-calling schemas at all—they were simply static text strings appended to the system prompt (`lastRequest.skillsUsed = "Instructions sent: ..."`).

This was a classic violation of interface honesty. In a sovereign local AI runtime, synthetic mock data and unfulfilled UI affordances undermine trust. If the interface advertises tools, the model must know about them, and the runtime must execute them.

As Kingdon posed to the team:
> *"What would Liskov do?"*

---

## 2. The Architectural Principle: Decoupled Substitutability

The Liskov Substitution Principle (LSP) tells us that objects of a supertype should be replaceable with objects of a subtype without altering the correctness of the program.

Applied to local model tooling:
- **No external entanglement**: We shouldn't couple Strata directly to complex external applications (like NockNock or Mecris) just to validate tool calling.
- **Provider neutrality**: The tool execution seam should treat an in-memory mock tool, an in-tree fake MCP server, a local workspace CLI tool, and an external enterprise MCP server as completely interchangeable instances of the same contract.
- **Contract symmetry**: The host UI, the inference proxy, the Jinja chat template, and the model's tokenizer must speak the standard OpenAI/Anthropic tool schema dialect without proprietary deviations.

This led directly to our two-phase architecture:
1. **Phase 1 (Truthful Inventory)**: Pass tool schemas to `/v1/chat/completions`, render the model's native `# Tools` block in the chat template, ensure the model truthfully acknowledges its inventory, and bind the UI state directly to live engine discovery.
2. **Phase 2 (The Execution Loop)**: Expose `/v1/mcp/tools` and `/v1/mcp/call` on the engine, implement client-side proposal validation in the host (`tool-run.mjs`), and support both **Ask Mode** (human-in-the-loop approval) and **Auto Mode** (autonomous engine execution).

---

## 3. The Implementation Journey

### Phase 1: Injecting the Truth
Models trained for tool calling (such as Qwen 2.5 / 3.x) expect tools to be defined in the chat template. If `/v1/chat/completions` receives a request without a `tools` parameter, the template never renders the `# Tools` header, leaving the model blind to the environment.

We updated `apps/host/server/inference.mjs` to fetch available tools from the engine via `GET /v1/mcp/tools` and forward them into the completion request.

Immediately, the model transformed:
```
User: What tools do you have available? List them.

Qwen3.6-35B-A3B:
Here is the list of tools available to me:
1. fake__echo: Returns the text it is given.
2. fake__fail: Always fails.
3. fake__sleep: Waits, then answers.
4. fake__big: Returns n characters.
5. fake__die: Ends the server.
6. fake__add: Adds two integers.
```

### Phase 2: The Two Seams (Ask Mode vs. Auto Mode)
For tool execution, we established two distinct modes:
1. **Auto Mode (`strata_mcp: true`)**:
   The Python serve layer manages an MCP client hub (`serve/mcp.py`). When the model emits a tool call, the engine intercepts the call, executes it locally against the MCP server, and feeds the output back into the generation loop before returning the final text to the user.
2. **Ask Mode (`toolPolicy: 'ask'`)**:
   The user maintains sovereignty. The host streams the model's generation until a `tool_calls` fragment arrives. The UI pauses generation and presents an interactive Proposal Card showing the tool name, formatted JSON arguments, and an **Approve / Deny** prompt.
   When the user clicks **Approve**, the host posts to `POST /api/tools/run`, which validates the call against the live inventory in `tool-run.mjs` and invokes `POST /v1/mcp/call`. The tool's output is wrapped in a `role: 'tool'` message and returned to the model to complete the conversation.

---

## 4. Playtesting on Real Apple Silicon Hardware

To prove the implementation without external mocks, we wired Strata against its built-in Python test server (`serve/mcp_fake_server.py`) on an Apple M2 running macOS:

1. **Step 1: User prompt:**
   > *"What is 1234 + 5678? You must call the fake__add tool to compute this."*
2. **Step 2: Model emission:**
   The model thought through the requirement in unified memory:
   ```json
   {
     "reasoning_content": "The user wants me to compute 1234 + 5678. I must use the fake__add tool with a=1234 and b=5678.",
     "tool_calls": [{
       "id": "call_4e8c9ff78c2543639c430fbb",
       "type": "function",
       "function": {"name": "fake__add", "arguments": "{\"a\":1234,\"b\":5678}"}
     }],
     "finish_reason": "tool_calls"
   }
   ```
3. **Step 3: Tool execution:**
   The host executed `fake__add` on loopback:
   `{"server": "fake", "tool": "add", "ok": true, "text": "6912", "ms": 1}`
4. **Step 4: Continuation turn:**
   The result was supplied back to the engine. At 12.5 tok/s on Metal, the model delivered:
   > *"The result of 1234 + 5678 is 6912."*

Zero hallucinations. Zero synthetic mock bypasses. Pure contract compliance.

---

## 5. Summary

By adhering strictly to interface substitutability and truthful inventory reporting:
- The UI no longer lies about tool capabilities.
- The model receives standard OpenAI schemas and operates natively.
- Users gain full human-in-the-loop control over tool executions.
- The local runtime remains completely independent of outside project lifecycles.

What would Liskov do? She would make tools real, verified, and substitutable. And that's exactly what we did.
