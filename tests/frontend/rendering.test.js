import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
const dom = new JSDOM('<!doctype html><div id="messages"></div>', {url:'http://localhost:3000/'});
globalThis.window = dom.window; globalThis.document = dom.window.document; globalThis.location = dom.window.location;
const {renderMarkdown} = await import('../../frontend/markdown.js');
const {createMessageView} = await import('../../frontend/messages.js');
const {icon} = await import('../../frontend/ui.js');

test('Markdown supports formatting but rejects executable HTML, unsafe URLs and remote images', () => {
  const node = renderMarkdown('# Title\n\n**bold** and [safe](https://example.com)\n\n<script>window.pwned=true</script>\n<img src=x onerror=alert(1)>\n\n[bad](javascript:alert(1))\n![tracker](https://example.com/track.png)');
  assert.equal(node.querySelector('h1').textContent,'Title');
  assert.equal(node.querySelector('strong').textContent,'bold');
  assert.equal(node.querySelector('a').rel,'noopener noreferrer');
  assert.equal(node.querySelectorAll('script,img,iframe,svg,style,input').length,0);
  assert.equal(node.querySelectorAll('[onerror],a[href^="javascript:"]').length,0);
});
test('code blocks preserve literal code and highlight a supported language', async () => {
  const node = renderMarkdown('```python\nprint("<script>literal</script>")\n```');
  assert.equal(node.querySelector('code').textContent,'print("<script>literal</script>")\n');
  assert.equal(node.querySelector('button').getAttribute('aria-label'),'Copy code');
  for(let count=0;count<50 && !node.querySelector('.hljs');count++) await new Promise(resolve => setTimeout(resolve,10));
  assert.ok(node.querySelector('.hljs-string'));
  assert.equal(node.querySelectorAll('script').length,0);
});
test('appending a message retains earlier DOM nodes; message actions target their turn', () => {
  const list = document.getElementById('messages'), actions=[];
  const view = createMessageView(list,{edit:index => actions.push(['edit',index]),regenerate:index => actions.push(['regenerate',index])});
  const first={id:'first',role:'user',content:'<b>untrusted user text</b>'};
  view.sync([first]); const original=list.querySelector('article');
  view.sync([first,{id:'second',role:'assistant',content:'**response**'}]);
  assert.equal(list.querySelector('article'),original);
  assert.equal(original.querySelector('b'),null);
  original.querySelector('[data-model-action]').click();
  list.querySelectorAll('[data-model-action]')[1].click();
  assert.deepEqual(actions,[['edit',0],['regenerate',1]]);
  view.sync([first]); assert.equal(list.querySelectorAll('article').length,1);
});
test('Font Awesome icon names resolve to locally vendored symbols', async () => {
  const sprite = await readFile(new URL('../../frontend/vendor/fontawesome.svg',import.meta.url),'utf8');
  const html = await readFile(new URL('../../frontend/index.html',import.meta.url),'utf8');
  for (const [,name] of html.matchAll(/data-icon="([a-z-]+)"/g)) assert.ok(sprite.includes(`id="fa-${name}"`),name);
  assert.equal(icon('copy').querySelector('use').getAttribute('href'),'./vendor/fontawesome.svg#fa-copy');
  assert.ok(sprite.includes('Font Awesome Free 7.3.1'));
});
