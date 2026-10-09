// Browser-owned memory is optional context. It never changes neural weights.
const KEY = 'kira-local-lab-v2';
export const newConversation = () => ({id: crypto.randomUUID(), messages: [], memory: '', memoryEnabled: false});
export function loadConversations(storage) {
  try {
    const entries = JSON.parse(storage.getItem(KEY) || '[]');
    if (!Array.isArray(entries)) return [];
    return entries.filter(chat => chat && typeof chat.id === 'string' && Array.isArray(chat.messages)).map(chat => ({
      id: chat.id, memory: typeof chat.memory === 'string' ? chat.memory.slice(0, 8000) : '',
      memoryEnabled: chat.memoryEnabled === true,
      messages: chat.messages.filter(message => message && ['user', 'assistant'].includes(message.role) && typeof message.content === 'string')
        .map(message => ({id: crypto.randomUUID(), role: message.role, content: message.content,
          ...(message.metrics && Number.isFinite(message.metrics.generated_tokens) && Number.isFinite(message.metrics.tokens_per_second) ? {metrics: message.metrics} : {})}))
    }));
  } catch { return []; }
}
export function saveConversations(storage, conversations) { storage.setItem(KEY, JSON.stringify(conversations)); }
export function memoryContext(chat) { return chat.memoryEnabled ? chat.memory : ''; }
export function precedingHistory(chat, index, enabled) {
  return enabled ? chat.messages.slice(0, index).slice(-40).map(({role, content}) => ({role, content})) : [];
}
export function replaceTurn(chat, index, prompt, response) {
  // Commit edits only after successful generation, preserving earlier turns on failure.
  return [...chat.messages.slice(0, index), {id: crypto.randomUUID(), role: 'user', content: prompt},
    {id: crypto.randomUUID(), role: 'assistant', content: response.text, metrics: {
      generated_tokens: response.generated_tokens, tokens_per_second: response.tokens_per_second,
      prompt_tokens: response.prompt_tokens, stop_reason: response.stop_reason
    }}];
}
