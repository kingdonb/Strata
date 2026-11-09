/**
 * The Strata engine tools in a real Harness (needs STRATA_HARNESS_DIR; the model is scripted and the
 * MCP server is a stdio fake, so no Python, Strata folder or engine is needed): the built-in tools
 * reach the agent as `mcp__strata__*`, a read runs unasked in Manual, a confirmed install asks and a
 * deny runs nothing, and Plan keeps the reads while refusing a stop.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { createAgentTools } from '../server/agent/tools-settings.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const FAKE = fileURLToPath(new URL('./fixtures/strata-mcp-fake.mjs', import.meta.url));

const tmp = () => mkdtempSync(join(tmpdir(), 'strata-mcp-'));
const cleanup = dir => { try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } };

async function rig(t) {
  const worker = await startScriptedWorker({});
  const dir = tmp();
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const tools = createAgentTools({
    read: () => ({}), write: () => {}, dataDir: dir,
    strata: () => ({ found: true, spec: { command: process.execPath, args: [FAKE, '--root', dir], cwd: dir } }),
  });
  const local = [{ id: 'inst-1', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], tools: () => tools.profileView(), env: process.env, startTimeoutMs: 120_000 });
  t.after(async () => { await bridge.close(); await worker.close(); cleanup(dir); });
  const run = async (chatId, tool, args, { mode = 'bypass', answer } = {}) => {
    const frames = [];
    await bridge.runTurn({ chatId, message: `SCRIPT:call ${tool} ${JSON.stringify(args)}`, workspacePath: ws, mode, onFrame: f => {
      frames.push(f);
      if (answer && f.type === 'agent-tool' && f.tool.state === 'proposed') void bridge.approve({ chatId, callId: f.tool.id, decision: answer }).catch(() => {});
    } });
    return frames.filter(f => f.type === 'agent-tool').map(f => f.tool);
  };
  return { bridge, dir, ws, run, offered: () => [...worker.requests].reverse().find(q => q.toolNames.length)?.toolNames.sort() };
}

test('real Harness: the Strata engine tools are offered; a read runs unasked in Manual; a confirmed install asks and a deny runs nothing', { skip, timeout: 240000 }, async (t) => {
  const r = await rig(t);
  const read = await r.run('s1', 'mcp__strata__strata_status', {}, { mode: 'manual' });
  assert.ok(r.offered().includes('mcp__strata__strata_status'), 'the built-in engine tools are offered');
  assert.ok(!read.some(c => c.state === 'proposed'), 'a read never asks');
  assert.equal(read.at(-1).state, 'done');
  assert.match(read.at(-1).result, /fake/);
  const ask = await r.run('s2', 'mcp__strata__strata_install', { confirm: true }, { mode: 'manual' });
  assert.ok(ask.some(c => c.state === 'proposed'), 'a confirmed install asks first');
  const denied = await r.run('s3', 'mcp__strata__strata_install', { confirm: true }, { mode: 'manual', answer: 'deny' });
  assert.ok(denied.some(c => c.state === 'proposed') && !denied.some(c => c.state === 'done' && /launched/.test(c.result ?? '')), 'a No runs nothing');
  const approved = await r.run('s4', 'mcp__strata__strata_install', { confirm: true }, { mode: 'manual', answer: 'approve' });
  assert.equal(approved.at(-1).state, 'done');
  assert.match(approved.at(-1).result, /launched/);
});

test('real Harness: Plan keeps the reads and refuses a stop', { skip, timeout: 240000 }, async (t) => {
  const r = await rig(t);
  const read = await r.run('p1', 'mcp__strata__strata_status', {}, { mode: 'plan' });
  assert.equal(read.at(-1).state, 'done');
  assert.match(read.at(-1).result, /fake/);
  const stop = await r.run('p2', 'mcp__strata__strata_stop', {}, { mode: 'plan' });
  assert.ok(stop.some(c => ['denied', 'error'].includes(c.state)) && !stop.some(c => c.state === 'done'), 'Plan refuses the stop');
});
