const URL_KEY = 'kira-backend-url';
let backendUrl = '', accessToken = '', revision = 0;
try { backendUrl = localStorage.getItem(URL_KEY) || ''; } catch { /* Storage is optional. */ }
export const connectionURL = () => backendUrl;
export function setConnection(url, token) {
  backendUrl = url; accessToken = token; revision += 1;
  try { localStorage.setItem(URL_KEY, backendUrl); } catch { /* Token is never persisted. */ }
}
export function apiError(data, status) {
  if (status >= 500 && status !== 503 && status !== 502) return 'The Python backend could not complete this request. Check its terminal and try again.';
  if (Array.isArray(data.detail)) return data.detail.map(item => `${item.loc?.slice(1).join(' / ') || 'Input'}: ${item.msg || 'Invalid value'}`).join('; ');
  return typeof data.detail === 'string' ? data.detail.slice(0, 600) : `Request failed (${status}). Check your input and backend connection.`;
}
export async function api(path, body) {
  if (!backendUrl && location.hostname.endsWith('.github.io')) throw new Error('GitHub Pages hosts the interface. Open Connect to link your running Python backend.');
  const currentRevision = revision;
  const headers = {};
  if (accessToken) headers.Authorization = `Bearer ${accessToken}`;
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const encoded = body === undefined ? undefined : JSON.stringify(body);
  if (encoded && new TextEncoder().encode(encoded).length > 128 * 1024) throw new Error('This request is too large. Shorten the prompt, history, memory or attachments.');
  let response;
  try {
    response = await fetch(`${backendUrl}${path}`, {method: body === undefined ? 'GET' : 'POST', headers,
      body: encoded, credentials: 'omit', redirect: 'error', signal: AbortSignal.timeout(120000)});
  } catch (error) {
    if (error.name === 'TimeoutError') throw new Error('The backend took too long to respond. Check its terminal before retrying; a training job may still be running.');
    throw new Error('Cannot reach the Python backend. Open Connect, check its address, and make sure it is running.');
  }
  if (revision !== currentRevision) throw new Error('Connection changed during this request. Retry with the new backend.');
  let data;
  try { data = await response.json(); }
  catch { throw new Error('This address did not return the Kira API. Check the backend URL and forwarded port.'); }
  if (!response.ok) throw new Error(apiError(data, response.status));
  return data;
}
