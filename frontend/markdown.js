import {Marked} from './vendor/marked.js';
import DOMPurify from './vendor/purify.js';
import {button, copyText} from './ui.js';
const escape = text => text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const parser = new Marked({gfm: true, breaks: true, renderer: {
  html: ({text}) => escape(text),
  image: ({text}) => `[Image omitted: ${escape(text)}]`,
}});
let highlighting;
async function highlighter() {
  if (!highlighting) highlighting = Promise.all([
    import('./vendor/highlight.js'), ...['python','javascript','json','bash','css','xml'].map(language => import(`./vendor/language-${language}.js`))
  ]).then(([core, ...languages]) => {
    ['python','javascript','json','bash','css','xml'].forEach((language, i) => core.default.registerLanguage(language, languages[i].default));
    return core.default;
  });
  return highlighting;
}
export function renderMarkdown(text) {
  const container = document.createElement('div'); container.className = 'markdown';
  container.innerHTML = DOMPurify.sanitize(parser.parse(text), {
    ALLOWED_TAGS: ['p','br','strong','em','del','blockquote','ul','ol','li','h1','h2','h3','h4','h5','h6','pre','code','a','hr','table','thead','tbody','tr','th','td'],
    ALLOWED_ATTR: ['href','title','class','start'], ALLOW_DATA_ATTR: false, ALLOW_ARIA_ATTR: false,
  });
  for (const link of container.querySelectorAll('a')) {
    try {
      const url = new URL(link.getAttribute('href'), location.href);
      if (!['http:', 'https:', 'mailto:'].includes(url.protocol)) throw new Error();
      link.href = url.href; link.target = '_blank'; link.rel = 'noopener noreferrer';
    } catch { link.removeAttribute('href'); }
  }
  for (const pre of container.querySelectorAll('pre')) {
    const code = pre.querySelector('code'); if (!code) continue;
    const content = code.textContent;
    const language = [...code.classList].find(name => name.startsWith('language-'))?.slice(9) || 'text';
    const bar = document.createElement('div'); bar.className = 'code-bar';
    const label = document.createElement('span'); label.textContent = language;
    bar.append(label, button('copy', 'Copy code', () => copyText(content)));
    pre.prepend(bar);
    highlighter().then(highlight => {
      if (!highlight.getLanguage(language)) return;
      code.innerHTML = DOMPurify.sanitize(highlight.highlight(content, {language, ignoreIllegals: true}).value, {ALLOWED_TAGS: ['span'], ALLOWED_ATTR: ['class']});
      code.classList.add('hljs');
    }).catch(() => { /* Readable escaped code remains available if highlighting fails. */ });
  }
  return container;
}
