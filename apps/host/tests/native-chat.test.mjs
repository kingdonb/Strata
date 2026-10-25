import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import {proxyCompletion} from '../server/inference.mjs';
test('native chat sends image content, auth and tool opt-in and exposes reported MCP events',async()=>{
 let received;
 const server=http.createServer(async(req,res)=>{let text='';for await(const c of req)text+=c;received={body:JSON.parse(text),key:req.headers.authorization};res.writeHead(200,{'content-type':'text/event-stream'});res.end('data: {"strata_mcp":{"event":"call","id":"c1","name":"read_file","arguments":{"path":"example.txt"}}}\n\ndata: {"strata_mcp":{"event":"result","id":"c1","ok":true,"text":"Contents","ms":5}}\n\ndata: {"choices":[{"delta":{"content":"A picture"}}]}\n\ndata: {"usage":{"prompt_tokens":44,"completion_tokens":2},"timings":{"predicted_per_second":20,"prompt_per_second":800}}\n\ndata: [DONE]\n\n');});
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 try{const events=[];const result=await proxyCompletion({baseUrl:`http://127.0.0.1:${server.address().port}/v1`,modelId:'iq2_xs',authorization:'private-test-key',message:'Describe',images:[{url:'data:image/png;base64,aGVsbG8='}],extra:{strata_mcp:true,reasoning_effort:'none'},onFrame:f=>events.push(f),signal:new AbortController().signal});
 assert.equal(received.key,'Bearer private-test-key');assert.equal(received.body.strata_mcp,true);assert.equal(received.body.messages.at(-1).content[1].image_url.url,'data:image/png;base64,aGVsbG8=');assert.equal(received.body.model,'iq2_xs');assert.deepEqual(events.filter(e=>e.type==='native-tool').map(e=>e.event.event),['call','result']);assert.equal(result.tokensIn,44);assert.equal(result.tokensOut,2);assert.equal(result.usageSource,'reported');
 }finally{server.closeAllConnections();await new Promise(r=>server.close(r));}
});

test('tool history passes through and a continuation brings no user message',async()=>{
 let received;
 const server=http.createServer(async(req,res)=>{let text='';for await(const c of req)text+=c;received=JSON.parse(text);res.writeHead(200,{'content-type':'text/event-stream'});res.end('data: {"choices":[{"delta":{"content":"42"}}]}\n\ndata: {"usage":{"prompt_tokens":9,"completion_tokens":1}}\n\ndata: [DONE]\n\n');});
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const history=[{role:'user',content:'add 20 and 22'},{role:'assistant',content:'Let me check.',tool_calls:[{function:{name:'fake__add',arguments:'{"a":20,"b":22}'}}]},{role:'tool',content:'42'}];
 try{const events=[];await proxyCompletion({baseUrl:`http://127.0.0.1:${server.address().port}/v1`,modelId:'iq2_xs',message:'',history,extra:{tools:[{type:'function',function:{name:'fake__add',parameters:{}}}]},onFrame:f=>events.push(f),signal:new AbortController().signal});
 assert.deepEqual(received.messages,[{role:'user',content:'add 20 and 22'},{role:'assistant',content:'Let me check.',tool_calls:[{function:{name:'fake__add',arguments:'{"a":20,"b":22}'}}]},{role:'tool',content:'42'}]);
 assert.equal(received.tools.length,1);assert.equal(received.tools[0].function.name,'fake__add');assert.equal(received.strata_mcp,undefined);
 }finally{server.closeAllConnections();await new Promise(r=>server.close(r));}
});

test('streamed tool_calls fragments assemble into one proposal frame',async()=>{
 const server=http.createServer(async(req,res)=>{res.writeHead(200,{'content-type':'text/event-stream'});res.end(
  'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c9","function":{"name":"fake__echo","arguments":""}}]}}]}\n\n'+
  'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\\"text\\\": \\"he"}}]}}]}\n\n'+
  'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"llo\\\"}"}}]}}]}\n\n'+
  'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\ndata: {"usage":{"prompt_tokens":5,"completion_tokens":7}}\n\ndata: [DONE]\n\n');});
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 try{const events=[];const result=await proxyCompletion({baseUrl:`http://127.0.0.1:${server.address().port}/v1`,modelId:'iq2_xs',message:'echo hello',onFrame:f=>events.push(f),signal:new AbortController().signal});
 const proposals=events.filter(e=>e.type==='tool');
 assert.equal(proposals.length,1);
 const tool=proposals[0].tool;
 assert.equal(tool.id,'c9');assert.equal(tool.name,'fake__echo');assert.deepEqual(tool.args,{text:'hello'});
 assert.equal(tool.scope,'connected');assert.equal(result.finishReason,'tool_calls');
 }finally{server.closeAllConnections();await new Promise(r=>server.close(r));}
});
