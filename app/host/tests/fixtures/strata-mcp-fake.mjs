/**
 * A fake Strata MCP server for the agent tests: a stdio stand-in for tools/strata_mcp.py that needs no
 * Python, no Strata folder and no engine. Same wire shape (newline-delimited JSON-RPC, one message per
 * line), same three tool names the gate rules speak about, honest fake answers.
 */
import { createInterface } from 'node:readline';

const TOOLS = [
  { name: 'strata_status', description: 'Is Strata running, and what is installed? (fake)', inputSchema: { type: 'object', additionalProperties: false, properties: {} } },
  { name: 'strata_install', description: 'Install a Strata model; without confirm it only returns the plan. (fake)', inputSchema: { type: 'object', additionalProperties: false, properties: { confirm: { type: 'boolean' }, cancel: { type: 'boolean' } } } },
  { name: 'strata_stop', description: 'Stop the running Strata model. (fake)', inputSchema: { type: 'object', additionalProperties: false, properties: {} } },
];

const send = message => { process.stdout.write(`${JSON.stringify(message)}\n`); };

async function handle(line) {
  let msg;
  try { msg = JSON.parse(line); } catch { return; }
  const { id, method, params } = msg;
  if (id === undefined) return;                                  // notifications need no answer
  if (method === 'initialize') {
    return send({ id, result: { protocolVersion: params?.protocolVersion ?? '2024-11-05', capabilities: { tools: { listChanged: false } }, serverInfo: { name: 'strata', version: '0.0.0-fake' } } });
  }
  if (method === 'ping') return send({ id, result: {} });
  if (method === 'tools/list') return send({ id, result: { tools: TOOLS } });
  if (method === 'tools/call') {
    const name = String(params?.name ?? ''), args = params?.arguments ?? {};
    let text = `no such tool ${name}`;
    if (name === 'strata_status') text = 'Strata engine idle (fake): no model running, .venv present';
    else if (name === 'strata_install') text = args?.confirm === true ? 'install launched (fake)' : 'Plan: install Q2_0, 66 GB download. Nothing was done yet (fake).';
    else if (name === 'strata_stop') text = 'Strata stopped (fake)';
    return send({ id, result: { content: [{ type: 'text', text }] } });
  }
  return send({ id, error: { code: -32601, message: `Method not found: ${method}` } });
}

createInterface({ input: process.stdin }).on('line', line => void handle(line));
