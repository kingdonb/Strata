/**
 * The chat's ask mode: the two engine seams a plain chat needs for connected tools.
 *
 * Both speak to the engine's own MCP endpoints (serve/server.py), which run the same tested hub
 * (serve/mcp.py) that the web app's auto mode uses — the host never starts or speaks MCP itself.
 *   GET  /v1/mcp/tools  the engine's MCP tools as OpenAI function schemas (an empty list when none
 *                       are configured or the engine does not answer: the turn then simply runs
 *                       without tools)
 *   POST /v1/mcp/call   runs one call the user approved; answers the hub's result
 *                       {ok, text, chars, truncated, ms, server, tool}
 */
export async function connectedTools({ baseUrl, key, timeoutMs = 3000, fetchImpl = fetch } = {}) {
  try {
    const res = await fetchImpl(`${baseUrl}/mcp/tools`, {
      signal: AbortSignal.timeout(timeoutMs),
      ...(key ? { headers: { authorization: `Bearer ${key}` } } : {}),
    });
    const tools = await res.json();
    return Array.isArray(tools) ? tools : [];
  } catch {
    return [];
  }
}

export async function runConnectedTool({ baseUrl, key, name, arguments: args, fetchImpl = fetch, signal } = {}) {
  if (typeof name !== 'string' || !name) return { ok: false, text: 'error: no tool was named', chars: 0, truncated: false, ms: 0 };
  /* The name must be one the engine currently offers: nothing else is executed, whatever the request says. */
  const tools = await connectedTools({ baseUrl, key, fetchImpl });
  if (!tools.some((t) => t?.function?.name === name)) {
    return { ok: false, text: `error: ${name} is not among the tools this engine offers`, chars: 0, truncated: false, ms: 0 };
  }
  try {
    const res = await fetchImpl(`${baseUrl}/mcp/call`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...(key ? { authorization: `Bearer ${key}` } : {}) },
      body: JSON.stringify({ name, arguments: args && typeof args === 'object' && !Array.isArray(args) ? args : {} }),
      ...(signal ? { signal } : {}),
    });
    if (!res.ok) return { ok: false, text: `error: the engine refused the call (${res.status})`, chars: 0, truncated: false, ms: 0 };
    const r = await res.json();
    return typeof r === 'object' && r !== null && typeof r.ok === 'boolean'
      ? r
      : { ok: false, text: 'error: the engine answered in a form the chat cannot read', chars: 0, truncated: false, ms: 0 };
  } catch (e) {
    return { ok: false, text: `error: the tool call did not finish (${e?.name === 'TimeoutError' || e?.name === 'AbortError' ? 'it was stopped' : 'the engine could not be reached'})`, chars: 0, truncated: false, ms: 0 };
  }
}
