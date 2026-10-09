import test from 'node:test';
import assert from 'node:assert/strict';
import {loadConversations, saveConversations, newConversation, memoryContext, precedingHistory, replaceTurn} from '../../frontend/conversations.js';
import {apiError} from '../../frontend/api.js';

function storage(value) { return {getItem: () => value, setItem: (_, updated) => { value = updated; }}; }
test('legacy history migrates safely and memory requires explicit opt-in', () => {
  const store = storage(JSON.stringify([{id:'old',memory:'private notes',messages:[{role:'user',content:'hello'},{role:'system',content:'discard'},{role:'assistant',content:'world'}]}]));
  const [chat] = loadConversations(store);
  assert.equal(chat.messages.length,2);
  assert.notEqual(chat.messages[0].id, chat.messages[1].id);
  assert.equal(memoryContext(chat),'');
  chat.memoryEnabled=true; assert.equal(memoryContext(chat),'private notes');
  saveConversations(store,[chat]); assert.equal(memoryContext(loadConversations(store)[0]),'private notes');
  chat.memoryEnabled=false; assert.equal(memoryContext(chat),'');
  assert.deepEqual(loadConversations(storage('invalid JSON')),[]);
});
test('editing replaces only the selected turn and following history after a real response', () => {
  const chat = newConversation();
  chat.messages = [{id:'a',role:'user',content:'first'},{id:'b',role:'assistant',content:'answer'},{id:'c',role:'user',content:'old'}];
  const original = JSON.stringify(chat);
  const updated = replaceTurn(chat,2,'edited',{text:'new answer',generated_tokens:3,tokens_per_second:12,prompt_tokens:4,stop_reason:'eos'});
  assert.equal(JSON.stringify(chat), original);
  assert.equal(updated[0], chat.messages[0]);
  assert.deepEqual(updated.map(message => message.content),['first','answer','edited','new answer']);
  assert.equal(updated[3].metrics.generated_tokens,3);
  assert.deepEqual(precedingHistory(chat,2,false),[]);
  assert.deepEqual(precedingHistory(chat,2,true),[{role:'user',content:'first'},{role:'assistant',content:'answer'}]);
});
test('validation errors expose field guidance, not raw input objects or stack traces', () => {
  assert.equal(apiError({detail:[{loc:['body','temperature'],msg:'Must be nonnegative',input:'private input'}]},422),'temperature: Must be nonnegative');
  assert.ok(!apiError({detail:'Traceback /secret/path'},500).includes('Traceback'));
});
