/** The chat's ask mode, through the real store: an approved connected tool continues as the OpenAI shape
    the engine writes itself — the assistant turn with its calls, one result per resolved call, and no user
    words of its own. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));

async function bundleStore(mockStreamPost) {
  const result = await build({
    entryPoints: [here('../src/lib/store.ts')], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
    alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' },
    plugins: [{
      name: 'mock-client',
      setup(b) {
        b.onResolve({ filter: /\/client$/ }, () => ({ path: 'mock-client', namespace: 'mock' }));
        b.onLoad({ filter: /.*/, namespace: 'mock' }, () => ({
          loader: 'js',
          contents: `
            export class ApiError extends Error {}
            const reject = async () => { throw new Error('offline'); };
            export const api = { get: reject, post: async () => ({}), put: reject, patch: reject, del: reject, delete: reject };
            export function onEvent() { return () => {}; }
            export async function streamPost(path, body, onFrame) {
              mockTurns.push({ path, body });
              onFrame({ type: 'start', model: 'm', contextMax: 8192, requestId: 'r2' });
              onFrame({ type: 'delta', text: 'The sum is 42.' });
              onFrame({ type: 'done', tokensIn: 9, tokensOut: 5, durationMs: 100, usageSource: 'reported' });
            }`,
        }));
      },
    }],
  });
  globalThis.mockTurns = [];
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}

test('a tool continuation sends the exchange in the engine shape, with no user words', async () => {
  globalThis.window ??= globalThis;
  globalThis.document ??= { addEventListener() {}, visibilityState: 'visible', documentElement: { dataset: {}, style: { setProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } } };
  const memory = new Map();
  globalThis.localStorage = { getItem: (k) => memory.get(k) ?? null, setItem: (k, v) => memory.set(k, v), removeItem: (k) => memory.delete(k) };
  globalThis.navigator ??= { language: 'en' };
  globalThis.matchMedia ??= () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });

  const store = await bundleStore();
  Object.assign(store.getState(), {
    connection: 'live',
    activeSession: 's1',
    remoteModel: null,
    library: { mode: 'real', variants: [], loaded: 'fixture' },
    sessions: [{
      id: 's1', title: 'Tools', updatedAt: '1', preferences: { useTools: true, mode: 'manual' },
      messages: [
        { id: 'u1', role: 'user', content: 'add 20 and 22', at: '2026-10-25T00:00:00Z' },
        { id: 'a1', role: 'assistant', content: 'Let me check.', at: '2026-10-25T00:00:01Z', tools: [{ id: 'c9', name: 'fake__add', args: { a: 20, b: 22 }, preview: '{"a":20,"b":22}', effect: 'Connected tool. It runs when you approve it.', scope: 'connected', state: 'done', result: '42', approved: true }] },
      ],
    }],
  });
  await store.sendMessage('', { toolContinuation: { assistantId: 'a1' } });

  const turns = globalThis.mockTurns;
  assert.equal(turns.length, 1);
  const body = turns[0].body;
  assert.equal(body.message, '', 'the continuation brings no user words');
  assert.equal(body.toolContinuation, true);
  assert.equal(body.agent, false, 'the agent path does not take a plain-chat continuation');
  assert.equal(body.useTools, true);
  assert.equal(body.toolPolicy, undefined, 'the separate policy preference is retired');
  assert.equal(body.mode, 'manual', 'the Mode selector governs the posture');
  assert.deepEqual(body.history, [
    { role: 'user', content: 'add 20 and 22' },
    { role: 'assistant', content: 'Let me check.', tool_calls: [{ function: { name: 'fake__add', arguments: '{"a":20,"b":22}' } }] },
    { role: 'tool', content: '42' },
  ]);
  const session = store.getState().sessions.find((s) => s.id === 's1');
  const answers = session.messages.filter((m) => m.role === 'assistant');
  assert.equal(answers.length, 2, 'the continuation answers as its own reply');
  assert.match(answers[1].content, /42/);
  assert.equal(answers[1].streaming, false);
});
