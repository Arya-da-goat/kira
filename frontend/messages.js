import {icon, button, copyText} from './ui.js';
import {renderMarkdown} from './markdown.js';

export function createMessageView(list, {edit, regenerate}) {
  const rendered = new Map();
  let limit = 80;
  const older = document.createElement('button'); older.className = 'load-older'; older.textContent = 'Show earlier messages';
  let latest = [];
  older.onclick = () => { limit += 80; sync(latest); };
  function make(message, index) {
    const article = document.createElement('article'); article.className = `message ${message.role}`;
    article.dataset.messageId = message.id;
    const avatar = document.createElement('span'); avatar.className = 'avatar';
    if (message.role === 'user') avatar.append(icon('user')); else avatar.textContent = 'K';
    const body = document.createElement('div'); body.className = 'message-body';
    const header = document.createElement('header'); header.textContent = message.role === 'user' ? 'You' : 'Kira';
    const label = document.createElement('span'); label.textContent = message.role === 'assistant' ? 'Local model' : '';
    header.append(label);
    const content = message.role === 'assistant' ? renderMarkdown(message.content || '*No visible text generated.*') : document.createElement('div');
    if (message.role === 'user') { content.className = 'user-text'; content.textContent = message.content; }
    const footer = document.createElement('footer'); footer.className = 'message-actions';
    footer.append(button('copy', 'Copy message', () => copyText(message.content)));
    const action = message.role === 'user'
      ? button('pen', 'Edit this message', () => edit(index))
      : button('rotate', 'Regenerate with current settings and context', () => regenerate(index));
    action.dataset.modelAction = 'true'; footer.append(action);
    if (message.metrics) {
      const m = message.metrics; const measurements = document.createElement('span'); measurements.className = 'message-metrics';
      measurements.textContent = `${m.generated_tokens} tokens · ${m.tokens_per_second.toFixed(1)} tok/s · ${m.prompt_tokens} context · ${m.stop_reason}`;
      footer.append(measurements);
    }
    body.append(header, content, footer); article.append(avatar, body);
    return article;
  }
  function sync(messages) {
    latest = messages;
    const start = Math.max(0, messages.length - limit);
    const visible = messages.slice(start), ids = new Set(visible.map(message => message.id));
    for (const [id, node] of rendered) if (!ids.has(id)) { node.remove(); rendered.delete(id); }
    older.hidden = start === 0;
    if (!list.contains(older)) list.prepend(older);
    let previous = older;
    visible.forEach((message, offset) => {
      if (!rendered.has(message.id)) rendered.set(message.id, make(message, start + offset));
      const node = rendered.get(message.id);
      if (previous.nextElementSibling !== node) previous.after(node);
      previous = node;
    });
  }
  return {sync, reset: () => { limit = 80; }};
}
