export function plotLoss(history, svg, legend) {
  svg.replaceChildren();
  const values = history.flatMap(row => [row.train_loss, row.validation_loss].filter(Number.isFinite));
  if (!values.length) { legend.textContent = 'No training measurements loaded.'; return; }
  const min = Math.min(...values), max = Math.max(...values), last = history.at(-1).step;
  for (const [key, color] of [['train_loss', '#60a5fa'], ['validation_loss', '#34d399']]) {
    const points = history.filter(row => Number.isFinite(row[key])).map(row => `${30 + 540 * row.step / last},${190 - 160 * (row[key] - min) / Math.max(max - min, .01)}`).join(' ');
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    line.setAttribute('points', points); line.setAttribute('fill', 'none'); line.setAttribute('stroke', color); line.setAttribute('stroke-width', '2'); svg.append(line);
  }
  legend.textContent = `Blue: training · Green: validation. Steps 1–${last}; loss range ${min.toFixed(3)}–${max.toFixed(3)}. Measured optimizer updates only.`;
}
