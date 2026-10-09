# Browser dependencies

These libraries render the interface; none performs model inference or sends
prompts to a hosted model. Selected browser distributions are intentionally
vendored as static runtime dependencies so a fresh clone and GitHub Pages work
without npm installation, a CDN or an application build.

| Dependency | Pinned version | Purpose | License |
| --- | --- | --- | --- |
| [Font Awesome Free](https://fontawesome.com/) by Fonticons, Inc. | 7.3.1 | Selected SVG icons assembled into a local symbol sprite; original paths retained | Icons: CC BY 4.0; included upstream license covers other package assets |
| [Marked](https://github.com/markedjs/marked) | 18.1.0 | Markdown parsing | MIT |
| [DOMPurify](https://github.com/cure53/DOMPurify) | 3.4.16 | Sanitizing rendered Markdown and highlighting | Apache 2.0 (dual-licensed alternative MPL 2.0) |
| [Highlight.js](https://highlightjs.org/) | 11.11.1 | Lazy syntax highlighting; six selected grammars | BSD 3-Clause |

Licenses are in `frontend/vendor/*-LICENSE`. Copyright/license headers are
retained in the upstream JavaScript files and SVG sprite. `package-lock.json`
records npm package integrity; `frontend/vendor/manifest.json` records SHA256
of each served file/license. `scripts/vendor_frontend.mjs` copies upstream
browser files and assembles the icon subset from the pinned package.

For maintenance, run `npm ci --ignore-scripts`, `npm run vendor`, `npm test`
and `npm run check`. Commit updated versions, lockfile, licenses and manifest
together. `jsdom` is used only for DOM-based development tests and is never
loaded by the application. The handwritten Kira application does not require
a bundler or generate a distribution directory.
