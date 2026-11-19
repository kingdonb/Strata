import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { connectedTools, runConnectedTool } from '../server/tool-run.mjs';

test('connectedTools reads the engine schemas and answers [] when the engine will not speak', async () => {
  const server = http.createServer((req, res) => {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify([{ type: 'function', function: { name: 'fake__echo', parameters: { type: 'object' } } }]));
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  try {
    const base = `http://127.0.0.1:${server.address().port}/v1`;
    const tools = await connectedTools({ baseUrl: base, key: 'private-key' });
    assert.equal(tools.length, 1);
    assert.equal(tools[0].function.name, 'fake__echo');
    const none = await connectedTools({ baseUrl: base, key: 'private-key', fetchImpl: async () => { throw new Error('down'); } });
    assert.deepEqual(none, []);                       // a turn then simply runs without tools
  } finally { server.closeAllConnections(); await new Promise(r => server.close(r)); }
});

test('runConnectedTool refuses an unlisted name before anything reaches the engine', async () => {
  const seen = [];
  const server = http.createServer(async (req, res) => {
    let text = ''; for await (const c of req) text += c;
    seen.push({ path: req.url, auth: req.headers.authorization, body: text ? JSON.parse(text) : null });
    if (req.url === '/v1/mcp/tools') {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify([{ type: 'function', function: { name: 'fake__add', parameters: {} } }]));
      return;
    }
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ ok: true, text: '42', chars: 2, truncated: false, ms: 4, server: 'fake', tool: 'add' }));
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  try {
    const base = `http://127.0.0.1:${server.address().port}/v1`;
    const refused = await runConnectedTool({ baseUrl: base, key: 'private-key', name: 'nope__nothing', arguments: {} });
    assert.equal(refused.ok, false);
    assert.match(refused.text, /not among the tools/);
    assert.equal(seen.length, 1);                     // only the schema read: the call itself never ran
    const good = await runConnectedTool({ baseUrl: base, key: 'private-key', name: 'fake__add', arguments: { a: 20, b: 22 } });
    assert.equal(good.ok, true);
    assert.equal(good.text, '42');
    const call = seen.find(s => s.path === '/v1/mcp/call');
    assert.deepEqual(call.body, { name: 'fake__add', arguments: { a: 20, b: 22 } });
    assert.equal(call.auth, 'Bearer private-key');
  } finally { server.closeAllConnections(); await new Promise(r => server.close(r)); }
});

test('runConnectedTool reports an engine refusal or a dropped call as a failed result', async () => {
  const server = http.createServer((req, res) => {
    if (req.url === '/v1/mcp/tools') {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify([{ type: 'function', function: { name: 'fake__x', parameters: {} } }]));
      return;
    }
    res.writeHead(404, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: { message: 'no MCP servers are configured' } }));
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  try {
    const r = await runConnectedTool({ baseUrl: `http://127.0.0.1:${server.address().port}/v1`, name: 'fake__x', arguments: {} });
    assert.equal(r.ok, false);
    assert.match(r.text, /refused the call/);
  } finally { server.closeAllConnections(); await new Promise(r => server.close(r)); }
});
