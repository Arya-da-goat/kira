// Pin packages in package-lock.json; keep the browser runtime usable without npm/CDNs.
import {readFile, writeFile, mkdir, copyFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
const require = createRequire(import.meta.url);
const destination = new URL('../frontend/vendor/', import.meta.url);
await mkdir(destination, {recursive: true});
const files = {
  'marked.js': 'marked/lib/marked.esm.js',
  'purify.js': 'dompurify/dist/purify.es.mjs',
  'highlight.js': '@highlightjs/cdn-assets/es/core.min.js',
  'MARKED-LICENSE': 'marked/LICENSE',
  'DOMPURIFY-LICENSE': 'dompurify/LICENSE',
  'HIGHLIGHT-LICENSE': '@highlightjs/cdn-assets/LICENSE',
  'FONTAWESOME-LICENSE': '@fortawesome/free-solid-svg-icons/LICENSE.txt',
};
for (const language of ['python', 'javascript', 'json', 'bash', 'css', 'xml']) {
  files[`language-${language}.js`] = `@highlightjs/cdn-assets/es/languages/${language}.min.js`;
}
const hashes = {};
for (const [output, source] of Object.entries(files)) {
  const input = new URL(`../node_modules/${source}`, import.meta.url);
  await copyFile(input, new URL(output, destination));
  hashes[output] = createHash('sha256').update(await readFile(input)).digest('hex');
}
const names = ['bars','xmark','plus','magnifying-glass','gear','paperclip','arrow-up','copy','rotate','pen','trash','file','download','upload','user','brain','house','ellipsis','check','triangle-exclamation','circle-exclamation','chevron-left','chevron-down','bolt','flask','sliders','server','circle-nodes','arrow-right','link','globe','message','code','book-open','circle-info','circle','stop','arrow-down'];
const fa = require('@fortawesome/free-solid-svg-icons');
const symbols = names.map(name => {
  const entry = Object.values(fa).find(item => item?.iconName === name && item.icon);
  if (!entry) throw new Error(`Unknown Font Awesome icon: ${name}`);
  const [width, height, , , path] = entry.icon;
  return `<symbol id="fa-${name}" viewBox="0 0 ${width} ${height}"><path d="${path}"/></symbol>`;
});
const sprite = `<!-- Font Awesome Free 7.3.1 by @fontawesome — https://fontawesome.com; icons CC BY 4.0. See FONTAWESOME-LICENSE. -->\n<svg xmlns="http://www.w3.org/2000/svg"><defs>${symbols.join('\n')}</defs></svg>\n`;
await writeFile(new URL('fontawesome.svg', destination), sprite);
hashes['fontawesome.svg'] = createHash('sha256').update(sprite).digest('hex');
await writeFile(new URL('manifest.json', destination), JSON.stringify(hashes, null, 2) + '\n');
console.log(`Prepared ${Object.keys(hashes).length} pinned, local browser assets.`);
