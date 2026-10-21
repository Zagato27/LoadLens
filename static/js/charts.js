// SPDX-License-Identifier: Apache-2.0
// Shared Chart.js helpers: series palette, time lookup on category axes, marker and threshold
// plugins, synchronized hover, drag-to-zoom and legend chips, plus the SVG load profile (stepProfile,
// no Chart.js). Loaded after ui.js on pages that
// include Chart.js. The x axis stays a category axis whose labels are sorted ISO strings or epoch
// milliseconds; every time-based helper maps a timestamp to a label index by binary search.
(function () {
  const LL = window.LoadLens = window.LoadLens || {};
  const charts = LL.charts = LL.charts || {};

  // JetBrains Mono, the --mono stack of app.css; ui.js puts it into Chart.defaults.font.family.
  const FONT_FALLBACK = "'JetBrains Mono', Consolas, 'Courier New', monospace";
  // Used only when the stylesheet is not available (the CSS variables are the source of truth).
  const SERIES_FALLBACK = ['#268beb', '#c35cb4', '#908e30', '#199eaa', '#894f9a', '#8c5e1a', '#9d86d0', '#2569a8'];
  const SERIES_COUNT = 8;
  const TONES = {
    accent: ['--accent', '#6c5ce7'],
    ok: ['--ok', '#52c483'],
    warn: ['--warn', '#e4a93b'],
    fail: ['--fail', '#e4574f'],
    muted: ['--dim', '#84837d']
  };
  const raf = (cb) => (window.requestAnimationFrame ? window.requestAnimationFrame(cb) : setTimeout(cb, 16));
  const caf = (id) => (window.cancelAnimationFrame ? window.cancelAnimationFrame(id) : clearTimeout(id));

  // ---- theme tokens -------------------------------------------------------
  function readTheme() {
    let css = null;
    try { css = getComputedStyle(document.documentElement); } catch (e) { css = null; }
    const get = (name, fallback) => {
      const value = css ? String(css.getPropertyValue(name) || '').trim() : '';
      return value || fallback;
    };
    return {
      get,
      tone(name) { const spec = TONES[name] || TONES.accent; return get(spec[0], spec[1]); },
      text: get('--text', '#e9e7e2'),
      background: get('--bg', '#131416'),
      border: get('--line-2', 'rgba(255, 255, 255, 0.18)')
    };
  }
  function fontFamily() {
    const d = window.Chart && window.Chart.defaults;
    return (d && d.font && d.font.family) || FONT_FALLBACK;
  }
  function isLightTheme() {
    if (LL.ui && LL.ui.theme && typeof LL.ui.theme.current === 'function') return LL.ui.theme.current() === 'light';
    return document.documentElement.getAttribute('data-theme') === 'light';
  }

  // ---- colors ---------------------------------------------------------------
  function parseColor(value) {
    const text = String(value || '').trim().toLowerCase();
    let m = /^#([0-9a-f]{3})$/.exec(text);
    if (m) return { r: parseInt(m[1][0] + m[1][0], 16), g: parseInt(m[1][1] + m[1][1], 16), b: parseInt(m[1][2] + m[1][2], 16) };
    m = /^#([0-9a-f]{6})([0-9a-f]{2})?$/.exec(text);
    if (m) return { r: parseInt(m[1].slice(0, 2), 16), g: parseInt(m[1].slice(2, 4), 16), b: parseInt(m[1].slice(4, 6), 16) };
    m = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/.exec(text);
    if (m) return { r: Math.round(Number(m[1])), g: Math.round(Number(m[2])), b: Math.round(Number(m[3])) };
    return null;
  }
  function toHex(rgb) {
    const part = (n) => Math.max(0, Math.min(255, Math.round(n))).toString(16).padStart(2, '0');
    return `#${part(rgb.r)}${part(rgb.g)}${part(rgb.b)}`;
  }
  function rgbToHsl({ r, g, b }) {
    const rn = r / 255; const gn = g / 255; const bn = b / 255;
    const max = Math.max(rn, gn, bn); const min = Math.min(rn, gn, bn);
    const l = (max + min) / 2;
    if (max === min) return { h: 0, s: 0, l };
    const d = max - min;
    const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    let h;
    if (max === rn) h = (gn - bn) / d + (gn < bn ? 6 : 0);
    else if (max === gn) h = (bn - rn) / d + 2;
    else h = (rn - gn) / d + 4;
    return { h: h / 6, s, l };
  }
  function hslToRgb({ h, s, l }) {
    if (s === 0) return { r: l * 255, g: l * 255, b: l * 255 };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    const channel = (t0) => {
      let t = t0;
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    return { r: channel(h + 1 / 3) * 255, g: channel(h) * 255, b: channel(h - 1 / 3) * 255 };
  }
  // Later palette cycles keep the hue and move the lightness toward an end of the range: lighter first
  // on the dark theme, darker first on the light theme, alternating, a bit further with every cycle.
  // Moving a fraction of the remaining distance never clamps, so cycles stay distinct.
  function shiftForCycle(color, cycle, light) {
    const rgb = parseColor(color);
    if (!rgb || cycle <= 0) return color;
    const hsl = rgbToHsl(rgb);
    const step = Math.ceil(cycle / 2);
    const odd = cycle % 2 === 1;
    const toward = light ? (odd ? 0.12 : 0.78) : (odd ? 0.88 : 0.22);
    const share = 1 - Math.pow(0.65, step);
    hsl.l += (toward - hsl.l) * share;
    return toHex(hslToRgb(hsl));
  }

  // palette(i): color of series slot i from --series-1..8 of the current theme; cycles beyond 8.
  charts.palette = function palette(i) {
    const n = Math.max(0, Math.floor(Number(i) || 0));
    const slot = n % SERIES_COUNT;
    const cycle = Math.floor(n / SERIES_COUNT);
    const base = readTheme().get(`--series-${slot + 1}`, SERIES_FALLBACK[slot]);
    return cycle ? shiftForCycle(base, cycle, isLightTheme()) : base;
  };
  // paletteFor(names): Map(name -> color) in the given order; repeated names keep their first color.
  charts.paletteFor = function paletteFor(names) {
    const map = new Map();
    let next = 0;
    (Array.isArray(names) ? names : []).forEach((name) => {
      if (map.has(name)) return;
      map.set(name, charts.palette(next));
      next += 1;
    });
    return map;
  };
  // Datasets that carry a numeric `llSeries` slot get their color back from the palette; ui.restyleCharts
  // calls this on a theme change.
  charts.recolor = function recolor(chart) {
    const datasets = chart && chart.data && Array.isArray(chart.data.datasets) ? chart.data.datasets : [];
    datasets.forEach((ds) => {
      if (!ds || !Number.isInteger(ds.llSeries)) return;
      const color = charts.palette(ds.llSeries);
      ds.borderColor = color;
      ds.backgroundColor = color;
    });
  };

  // ---- time lookup ----------------------------------------------------------
  function toMs(value) {
    if (value instanceof Date) return value.getTime();
    if (typeof value === 'number') return Number.isFinite(value) ? value : NaN;
    if (typeof value === 'string' && value.trim()) return Date.parse(value.trim());
    return NaN;
  }
  charts.toMs = toMs;

  // Parsed label times are cached per labels array; the cache is checked against length and both ends.
  const timeCache = new WeakMap();
  function timesOf(labels) {
    if (!Array.isArray(labels) || !labels.length) return null;
    const n = labels.length;
    const cached = timeCache.get(labels);
    if (cached && cached.length === n && cached.first === labels[0] && cached.last === labels[n - 1]) return cached;
    let times = new Float64Array(n);
    for (let i = 0; i < n; i += 1) {
      const t = toMs(labels[i]);
      if (!Number.isFinite(t)) { times = null; break; }
      times[i] = t;
    }
    let halfStep = 0;
    if (times && n > 1) {
      const steps = [];
      for (let i = 1; i < n; i += 1) steps.push(times[i] - times[i - 1]);
      steps.sort((a, b) => a - b);
      const mid = steps.length >> 1;
      const median = steps.length % 2 ? steps[mid] : (steps[mid - 1] + steps[mid]) / 2;
      halfStep = Math.max(0, median / 2);
    }
    const entry = { length: n, first: labels[0], last: labels[n - 1], times, halfStep };
    timeCache.set(labels, entry);
    return entry;
  }
  function lowerBound(times, v) {
    let lo = 0; let hi = times.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (times[mid] < v) lo = mid + 1; else hi = mid; }
    return lo;
  }
  function upperBound(times, v) {
    let lo = 0; let hi = times.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (times[mid] <= v) lo = mid + 1; else hi = mid; }
    return lo;
  }

  // indexOfTime(labels, value): nearest label index, or -1 when the array is empty, the labels are not
  // times, or value lies outside [first - halfStep, last + halfStep] (halfStep = median spacing / 2).
  charts.indexOfTime = function indexOfTime(labels, value) {
    const entry = timesOf(labels);
    if (!entry || !entry.times) return -1;
    const v = toMs(value);
    if (!Number.isFinite(v)) return -1;
    const t = entry.times;
    const n = t.length;
    if (v < t[0] - entry.halfStep || v > t[n - 1] + entry.halfStep) return -1;
    const lo = Math.min(lowerBound(t, v), n - 1);
    if (lo > 0 && v - t[lo - 1] <= t[lo] - v) return lo - 1;
    return lo;
  };
  // timeAt(labels, index): epoch ms of a label, NaN when absent.
  charts.timeAt = function timeAt(labels, index) {
    const entry = timesOf(labels);
    if (!entry || !entry.times || !(index >= 0 && index < entry.times.length)) return NaN;
    return entry.times[Math.floor(index)];
  };
  // Fractional label position of a time, so markers between buckets land between the ticks.
  function fractionalIndex(entry, v) {
    const t = entry.times;
    const n = t.length;
    if (n === 1) return 0;
    if (v <= t[0]) return (v - t[0]) / ((t[1] - t[0]) || 1);
    if (v >= t[n - 1]) return n - 1 + (v - t[n - 1]) / ((t[n - 1] - t[n - 2]) || 1);
    const i = upperBound(t, v) - 1;
    return i + (v - t[i]) / ((t[i + 1] - t[i]) || 1);
  }
  function xForTime(chart, entry, v) {
    const scale = chart.scales && chart.scales.x;
    if (!scale) return NaN;
    return scale.getPixelForValue(fractionalIndex(entry, v));
  }
  function indexRange(labels, from, to) {
    const entry = timesOf(labels);
    if (!entry || !entry.times) return null;
    const t = entry.times;
    const n = t.length;
    const h = entry.halfStep;
    if (!(to >= t[0] - h) || !(from <= t[n - 1] + h)) return null;
    let i0 = Math.min(Math.max(lowerBound(t, from - h), 0), n - 1);
    let i1 = Math.min(Math.max(upperBound(t, to + h) - 1, 0), n - 1);
    if (i1 <= i0) {
      if (i0 < n - 1) i1 = i0 + 1;
      else if (i0 > 0) i0 = i1 - 1;
      else return null;
    }
    return [i0, i1];
  }

  // ---- shared plugin helpers --------------------------------------------------
  function alive(chart) { return !!(chart && chart.canvas && chart.ctx); }
  // Plugin hooks read the raw config, so `chart.options.plugins.llMarkers.highlight = …; chart.draw()`
  // takes effect without a full update (Chart.js caches the resolved plugin options until update()).
  function pluginConfig(chart, id, resolved) {
    const raw = chart && chart.config && chart.config.options && chart.config.options.plugins;
    return (raw && raw[id]) || resolved || {};
  }
  function roundRectPath(ctx, x, y, w, h, r) {
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') { ctx.roundRect(x, y, w, h, r); return; }
    ctx.rect(x, y, w, h);
  }
  function fitText(ctx, text, maxWidth) {
    const value = String(text || '');
    if (ctx.measureText(value).width <= maxWidth) return value;
    let lo = 0; let hi = value.length;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (ctx.measureText(`${value.slice(0, mid)}…`).width <= maxWidth) lo = mid; else hi = mid - 1;
    }
    return lo ? `${value.slice(0, lo)}…` : '';
  }
  const PILL_H = 16;
  const PILL_PAD = 5;
  const PILL_BAR = 3;
  function pillWidth(ctx, text) { return ctx.measureText(text).width + PILL_PAD * 2 + PILL_BAR; }
  // Label on the canvas: chart-background pill, tone-colored bar on the left, text in --text.
  function drawPill(ctx, x, y, text, toneColor, theme) {
    const w = pillWidth(ctx, text);
    ctx.save();
    ctx.globalAlpha = 0.92;
    ctx.fillStyle = theme.background;
    roundRectPath(ctx, x, y, w, PILL_H, 4);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
    ctx.strokeStyle = theme.border;
    roundRectPath(ctx, x + 0.5, y + 0.5, w - 1, PILL_H - 1, 4);
    ctx.stroke();
    ctx.fillStyle = toneColor;
    ctx.fillRect(x + 2, y + 3, PILL_BAR, PILL_H - 6);
    ctx.fillStyle = theme.text;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x + PILL_BAR + PILL_PAD + 1, y + PILL_H / 2 + 0.5);
    ctx.restore();
    return w;
  }
  function labelFont() { return `600 11px ${fontFamily()}`; }

  // ---- markers plugin ----------------------------------------------------------
  // options.plugins.llMarkers = { items: [{ t, to?, kind: 'line'|'band'|'tick', label?, tone }], highlight: null|{t, to?, tone?} }
  function bandPixels(chart, entry, from, to) {
    let a = toMs(from);
    let b = toMs(to);
    if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
    if (b < a) { const tmp = a; a = b; b = tmp; }
    const t = entry.times;
    const n = t.length;
    if (b < t[0] - entry.halfStep || a > t[n - 1] + entry.halfStep) return null;
    a = Math.max(a, t[0]);
    b = Math.min(b, t[n - 1]);
    const area = chart.chartArea;
    let x0 = Math.max(xForTime(chart, entry, a), area.left);
    let x1 = Math.min(xForTime(chart, entry, b), area.right);
    if (!Number.isFinite(x0) || !Number.isFinite(x1) || x1 < x0) return null;
    if (x1 - x0 < 2) {
      const mid = (x0 + x1) / 2;
      x0 = Math.max(area.left, mid - 1);
      x1 = Math.min(area.right, mid + 1);
    }
    return { x0, x1 };
  }
  function linePixel(chart, entry, value) {
    const v = toMs(value);
    if (!Number.isFinite(v)) return NaN;
    const t = entry.times;
    if (v < t[0] - entry.halfStep || v > t[t.length - 1] + entry.halfStep) return NaN;
    const x = xForTime(chart, entry, v);
    const area = chart.chartArea;
    return x >= area.left - 0.5 && x <= area.right + 0.5 ? x : NaN;
  }
  function markerItems(cfg) { return Array.isArray(cfg.items) ? cfg.items.filter(Boolean) : []; }
  function hasRange(item) { return item && item.to !== undefined && item.to !== null && item.to !== ''; }

  charts.markers = {
    id: 'llMarkers',
    beforeDatasetsDraw(chart, args, options) {
      const cfg = pluginConfig(chart, 'llMarkers', options);
      const area = chart.chartArea;
      const entry = timesOf(chart.data && chart.data.labels);
      if (!area || !entry || !entry.times) return;
      const theme = readTheme();
      const ctx = chart.ctx;
      ctx.save();
      ctx.beginPath();
      ctx.rect(area.left, area.top, area.right - area.left, area.bottom - area.top);
      ctx.clip();
      markerItems(cfg).forEach((item) => {
        if (item.kind !== 'band') return;
        const px = bandPixels(chart, entry, item.t, item.to);
        if (!px) return;
        ctx.globalAlpha = item.tone === 'muted' ? 0.1 : 0.14;
        ctx.fillStyle = theme.tone(item.tone);
        ctx.fillRect(px.x0, area.top, px.x1 - px.x0, area.bottom - area.top);
      });
      const hl = cfg.highlight;
      if (hl && hasRange(hl)) {
        const px = bandPixels(chart, entry, hl.t, hl.to);
        if (px) {
          const color = theme.tone(hl.tone || 'accent');
          ctx.globalAlpha = 0.24;
          ctx.fillStyle = color;
          ctx.fillRect(px.x0, area.top, px.x1 - px.x0, area.bottom - area.top);
          ctx.globalAlpha = 0.85;
          ctx.strokeStyle = color;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(px.x0 + 0.5, area.top); ctx.lineTo(px.x0 + 0.5, area.bottom);
          ctx.moveTo(px.x1 - 0.5, area.top); ctx.lineTo(px.x1 - 0.5, area.bottom);
          ctx.stroke();
        }
      }
      ctx.restore();
    },
    afterDatasetsDraw(chart, args, options) {
      const cfg = pluginConfig(chart, 'llMarkers', options);
      const area = chart.chartArea;
      const entry = timesOf(chart.data && chart.data.labels);
      if (!area || !entry || !entry.times) return;
      const theme = readTheme();
      const ctx = chart.ctx;
      const labels = [];
      ctx.save();
      markerItems(cfg).forEach((item) => {
        const kind = item.kind || 'line';
        const color = theme.tone(item.tone || (kind === 'tick' ? 'muted' : 'accent'));
        if (kind === 'band') {
          const px = bandPixels(chart, entry, item.t, item.to);
          if (px && item.label) labels.push({ x: px.x0, text: String(item.label), color });
          return;
        }
        const x = linePixel(chart, entry, item.t);
        if (!Number.isFinite(x)) return;
        ctx.setLineDash(kind === 'tick' ? [] : [5, 4]);
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(x, kind === 'tick' ? area.bottom - 7 : area.top);
        ctx.lineTo(x, area.bottom);
        ctx.stroke();
        if (item.label) labels.push({ x, text: String(item.label), color });
      });
      const hl = cfg.highlight;
      if (hl && !hasRange(hl)) {
        const x = linePixel(chart, entry, hl.t);
        if (Number.isFinite(x)) {
          ctx.setLineDash([]);
          ctx.lineWidth = 2;
          ctx.strokeStyle = theme.tone(hl.tone || 'accent');
          ctx.beginPath();
          ctx.moveTo(x, area.top);
          ctx.lineTo(x, area.bottom);
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);
      // Labels along the top of the chart area; a label that would overlap moves to the next row (max 3).
      ctx.font = labelFont();
      const maxWidth = Math.max(40, Math.min(260, area.right - area.left - 8));
      const rowsRight = [-Infinity, -Infinity, -Infinity];
      labels.sort((a, b) => a.x - b.x).forEach((label) => {
        const text = fitText(ctx, label.text, maxWidth - PILL_PAD * 2 - PILL_BAR);
        if (!text) return;
        const w = pillWidth(ctx, text);
        let left = label.x + 3;
        if (left + w > area.right - 2) left = Math.max(area.left + 2, label.x - 3 - w);
        let row = rowsRight.findIndex((right) => right + 4 <= left);
        if (row < 0) row = rowsRight.indexOf(Math.min(...rowsRight));
        rowsRight[row] = Math.max(rowsRight[row], left + w);
        drawPill(ctx, left, area.top + 3 + row * (PILL_H + 2), text, label.color, theme);
      });
      ctx.restore();
    }
  };
  // setHighlight(chart|charts, {t, to?}|null): sets llMarkers.highlight and redraws without an update.
  charts.setHighlight = function setHighlight(target, range) {
    (Array.isArray(target) ? target : [target]).forEach((chart) => {
      if (!alive(chart)) return;
      const options = chart.config.options || (chart.config.options = {});
      options.plugins = options.plugins || {};
      const cfg = options.plugins.llMarkers || (options.plugins.llMarkers = {});
      cfg.highlight = range || null;
      chart.draw();
    });
  };

  // ---- thresholds plugin -----------------------------------------------------
  // options.plugins.llThresholds = { items: [{ value, label, tone, scaleID? }] }
  function thresholdItems(chart, options, scaleId) {
    const cfg = pluginConfig(chart, 'llThresholds', options);
    const items = Array.isArray(cfg.items) ? cfg.items : [];
    return items.filter((item) => item && (item.scaleID || 'y') === scaleId && item.value !== null && item.value !== '' && Number.isFinite(Number(item.value)));
  }
  charts.thresholds = {
    id: 'llThresholds',
    afterDataLimits(chart, args, options) {
      const scale = args && args.scale;
      if (!scale || !scale.id) return;
      const items = thresholdItems(chart, options, scale.id);
      if (!items.length) return;
      const values = items.map((item) => Number(item.value));
      const hi = Math.max(...values);
      const lo = Math.min(...values);
      const opts = scale.options || {};
      const userMax = opts.max !== undefined && opts.max !== null;
      const userMin = opts.min !== undefined && opts.min !== null;
      const min = Number.isFinite(scale.min) ? Math.min(scale.min, lo) : lo;
      const max = Number.isFinite(scale.max) ? Math.max(scale.max, hi) : hi;
      const span = (max - min) || Math.abs(max) || 1;
      if (!userMax && (!Number.isFinite(scale.max) || hi + span * 0.05 > scale.max)) scale.max = hi + span * 0.05;
      if (!userMin && (!Number.isFinite(scale.min) || lo < scale.min)) scale.min = lo - span * 0.05;
    },
    afterDatasetsDraw(chart, args, options) {
      const area = chart.chartArea;
      if (!area) return;
      const cfg = pluginConfig(chart, 'llThresholds', options);
      const items = Array.isArray(cfg.items) ? cfg.items : [];
      if (!items.length) return;
      const theme = readTheme();
      const ctx = chart.ctx;
      const placed = [];
      ctx.save();
      ctx.font = labelFont();
      items.forEach((item) => {
        if (!item || item.value === null || item.value === '' || !Number.isFinite(Number(item.value))) return;
        const scale = chart.scales && chart.scales[item.scaleID || 'y'];
        if (!scale) return;
        const y = scale.getPixelForValue(Number(item.value));
        if (!Number.isFinite(y) || y < area.top - 0.5 || y > area.bottom + 0.5) return;
        const color = theme.tone(item.tone || 'fail');
        ctx.setLineDash([6, 4]);
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(area.left, y);
        ctx.lineTo(area.right, y);
        ctx.stroke();
        ctx.setLineDash([]);
        if (!item.label) return;
        const text = fitText(ctx, String(item.label), Math.max(40, Math.min(260, area.right - area.left - 8)) - PILL_PAD * 2 - PILL_BAR);
        if (!text) return;
        const w = pillWidth(ctx, text);
        const left = area.right - 4 - w;
        let top = y - 3 - PILL_H;
        const overlaps = (t) => placed.some((p) => t < p + PILL_H + 1 && t + PILL_H + 1 > p);
        if (top < area.top + 1 || overlaps(top)) top = y + 3;
        if (top + PILL_H > area.bottom || overlaps(top)) return;
        placed.push(top);
        drawPill(ctx, left, top, text, color, theme);
      });
      ctx.restore();
    }
  };

  // ---- shared tooltip ---------------------------------------------------------------
  // topTooltip({ max = 12 }): options.plugins.tooltip for `interaction: { mode: 'index' }` charts.
  // Rows are sorted by value (largest first), at most `max` are shown and the footer says «ещё N».
  // Chart.js filters before sorting, so the filter itself keeps the `max` largest values.
  function tooltipValue(item) {
    const y = item && item.parsed ? Number(item.parsed.y) : NaN;
    return Number.isFinite(y) ? y : -Infinity;
  }
  function byTooltipValue(a, b) {
    return (tooltipValue(b) - tooltipValue(a)) || ((a && a.datasetIndex) - (b && b.datasetIndex)) || 0;
  }
  // topIndices(values, max): positions of the `max` largest finite values (ties by position).
  charts.topIndices = function topIndices(values, max) {
    const limit = Math.max(1, Math.floor(Number(max) || 12));
    const list = Array.isArray(values) ? values : [];
    const rank = (v) => { const n = Number(v); return v === null || v === undefined || v === '' || !Number.isFinite(n) ? -Infinity : n; };
    return list.map((v, i) => i).sort((i, j) => (rank(list[j]) - rank(list[i])) || (i - j)).slice(0, limit).sort((i, j) => i - j);
  };
  // visibleValueCount(chart, index): visible datasets with a number at the label index.
  function visibleValueCount(chart, index) {
    if (!alive(chart)) return 0;
    let total = 0;
    (chart.data.datasets || []).forEach((ds, di) => {
      if (!chart.isDatasetVisible(di)) return;
      const raw = ds && Array.isArray(ds.data) ? ds.data[index] : null;
      const value = raw && typeof raw === 'object' ? raw.y : raw;
      if (value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value))) total += 1;
    });
    return total;
  }
  charts.topTooltip = function topTooltip(opts) {
    const options = opts || {};
    const max = Math.max(1, Math.floor(Number(options.max) || 12));
    const kept = new WeakMap();
    return {
      itemSort: byTooltipValue,
      filter(item, index, items) {
        if (!Array.isArray(items) || items.length <= max) return true;
        let keep = kept.get(items);
        if (!keep) {
          keep = new Set(items.map((it, i) => i).sort((i, j) => byTooltipValue(items[i], items[j])).slice(0, max));
          kept.set(items, keep);
        }
        return keep.has(index);
      },
      callbacks: {
        footer(items) {
          if (!Array.isArray(items) || !items.length || !items[0].chart) return '';
          const more = visibleValueCount(items[0].chart, items[0].dataIndex) - items.length;
          return more > 0 ? LL.t('charts.tooltip.more', { n: more }) : '';
        }
      }
    };
  };

  // ---- cursor tooltip ------------------------------------------------------------------
  // htmlTooltip({ max = 8 }): options.plugins.tooltip that draws an HTML box next to the pointer,
  // right and below it, flipped left / above near the viewport edges. Unlike the canvas tooltip it may
  // leave the chart, so the lines around the cursor stay visible. Rows and «ещё N» as in topTooltip;
  // pages add their own callbacks.title.
  const TIP_GAP = 14;
  const TIP_MARGIN = 8;
  const pointers = new WeakMap();
  let tipBox = null;
  let tipOwner = null;

  // tipPosition(px, py, w, h, vw, vh): top-left corner of a w×h box for a pointer at (px, py) in a
  // vw×vh viewport.
  charts.tipPosition = function tipPosition(px, py, w, h, vw, vh) {
    let left = px + TIP_GAP;
    if (left + w > vw - TIP_MARGIN) left = px - TIP_GAP - w;
    let top = py + TIP_GAP;
    if (top + h > vh - TIP_MARGIN) top = py - TIP_GAP - h;
    left = Math.max(TIP_MARGIN, Math.min(left, vw - TIP_MARGIN - w));
    top = Math.max(TIP_MARGIN, Math.min(top, vh - TIP_MARGIN - h));
    return { left: Math.round(left), top: Math.round(top) };
  };
  // valueAt(dataset, index): the number at a label index, or NaN.
  function valueAt(ds, index) {
    const raw = ds && Array.isArray(ds.data) ? ds.data[index] : null;
    const value = raw && typeof raw === 'object' ? raw.y : raw;
    if (value === null || value === undefined || value === '') return NaN;
    const n = Number(value);
    return Number.isFinite(n) ? n : NaN;
  }
  charts.valueAt = valueAt;
  function tipElement() {
    if (tipBox && tipBox.isConnected) return tipBox;
    tipBox = document.createElement('div');
    tipBox.className = 'll-chart-tip';
    // The values are also shown in the legend chips under the chart.
    tipBox.setAttribute('aria-hidden', 'true');
    tipBox.hidden = true;
    document.body.appendChild(tipBox);
    return tipBox;
  }
  function hideTip(owner) {
    if (!tipBox || (owner && tipOwner !== owner)) return;
    tipBox.hidden = true;
    tipOwner = null;
  }
  charts.hideTooltip = function hideTooltip() { hideTip(null); };
  function tipLine(className, text) {
    const div = document.createElement('div');
    div.className = className;
    div.textContent = text;
    return div;
  }
  function renderTip(context) {
    const chart = context && context.chart;
    const tooltip = context && context.tooltip;
    const pos = chart ? pointers.get(chart) : null;
    const points = tooltip && Array.isArray(tooltip.dataPoints) ? tooltip.dataPoints : [];
    // Only the chart under the pointer shows a tooltip; charts synchronized by syncHover do not.
    if (!tooltip || tooltip.opacity === 0 || !points.length || !pos || !pos.inside) { hideTip(chart); return; }
    const box = tipElement();
    box.textContent = '';
    (tooltip.title || []).filter(Boolean).forEach((line) => box.appendChild(tipLine('ll-chart-tip-title', line)));
    points.forEach((point) => {
      const ds = point.dataset || (chart.data.datasets || [])[point.datasetIndex] || {};
      const row = document.createElement('div');
      row.className = 'll-chart-tip-row';
      const swatch = document.createElement('span');
      swatch.className = 'll-chart-tip-swatch';
      if (Array.isArray(ds.borderDash) && ds.borderDash.length) swatch.classList.add('is-dashed');
      swatch.style.setProperty('--tip-color', datasetColor(ds));
      row.appendChild(swatch);
      row.appendChild(tipLine('ll-chart-tip-name', String(ds.label || '')));
      row.appendChild(tipLine('ll-chart-tip-value', formatAverage(point.parsed ? Number(point.parsed.y) : NaN)));
      box.appendChild(row);
    });
    (tooltip.footer || []).filter(Boolean).forEach((line) => box.appendChild(tipLine('ll-chart-tip-footer', line)));
    box.hidden = false;
    tipOwner = chart;
    const rect = box.getBoundingClientRect();
    const vw = window.innerWidth || document.documentElement.clientWidth;
    const vh = window.innerHeight || document.documentElement.clientHeight;
    const at = charts.tipPosition(pos.clientX, pos.clientY, rect.width, rect.height, vw, vh);
    box.style.left = `${at.left}px`;
    box.style.top = `${at.top}px`;
  }
  // A positioner that follows the pointer: Chart.js calls `external` only when the tooltip position
  // changes, and the default positioners stay put while the cursor moves inside one time bucket.
  const POINTER_POSITIONER = 'llPointer';
  const tooltipPlugin = window.Chart && window.Chart.Tooltip;
  if (tooltipPlugin && tooltipPlugin.positioners) {
    tooltipPlugin.positioners[POINTER_POSITIONER] = function followPointer(items, eventPosition) {
      if (!eventPosition || !Number.isFinite(eventPosition.x) || !Number.isFinite(eventPosition.y)) return false;
      return { x: eventPosition.x, y: eventPosition.y };
    };
  }
  charts.htmlTooltip = function htmlTooltip(opts) {
    const options = opts || {};
    const tip = charts.topTooltip({ max: options.max || 8 });
    tip.enabled = false;
    tip.external = renderTip;
    if (tooltipPlugin && tooltipPlugin.positioners && tooltipPlugin.positioners[POINTER_POSITIONER]) tip.position = POINTER_POSITIONER;
    return tip;
  };
  // Global, draws nothing: remembers where the pointer is over each chart. beforeEvent runs before
  // Chart.js updates the tooltip, so the box follows the current pointer position.
  const pointerTracker = {
    id: 'llPointer',
    beforeEvent(chart, args) {
      const evt = args && args.event;
      if (!evt || !chart.canvas) return;
      if (evt.type === 'mouseout') { pointers.set(chart, { inside: false }); return; }
      if (!Number.isFinite(evt.x) || !Number.isFinite(evt.y)) return;
      const rect = chart.canvas.getBoundingClientRect();
      const sx = chart.width ? rect.width / chart.width : 1;
      const sy = chart.height ? rect.height / chart.height : 1;
      pointers.set(chart, { inside: true, clientX: rect.left + evt.x * sx, clientY: rect.top + evt.y * sy });
    },
    afterEvent(chart, args) {
      if (args && args.event && args.event.type === 'mouseout') hideTip(chart);
    },
    afterDestroy(chart) { hideTip(chart); }
  };
  charts.pointerTracker = pointerTracker;
  if (window.Chart && typeof window.Chart.register === 'function') {
    try { window.Chart.register(pointerTracker); } catch (e) { /* older Chart.js */ }
  }
  if (typeof window.addEventListener === 'function') {
    // A fixed box would stay behind while the page scrolls; printing must not catch it either.
    window.addEventListener('scroll', () => hideTip(null), { passive: true, capture: true });
    window.addEventListener('beforeprint', () => hideTip(null));
  }

  // ---- crosshair and live legend values ------------------------------------------------
  // Plugin, passed per chart: a vertical line at the active index (the hovered chart and the charts
  // synchronized by syncHover) and live values in the chart's legend chips (chart.$llLegend).
  function activeElements(chart) {
    const active = typeof chart.getActiveElements === 'function' ? chart.getActiveElements() : [];
    return Array.isArray(active) ? active : [];
  }
  charts.crosshair = {
    id: 'llCrosshair',
    afterDatasetsDraw(chart) {
      const active = activeElements(chart);
      const area = chart.chartArea;
      const el = active.length ? active[0].element : null;
      if (!area || !el || !Number.isFinite(el.x) || el.x < area.left - 0.5 || el.x > area.right + 0.5) return;
      const ctx = chart.ctx;
      const x = Math.round(el.x) + 0.5;
      ctx.save();
      ctx.strokeStyle = readTheme().tone('muted');
      ctx.globalAlpha = 0.7;
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(x, area.top);
      ctx.lineTo(x, area.bottom);
      ctx.stroke();
      ctx.restore();
    },
    afterDraw(chart) {
      const legend = chart.$llLegend;
      if (!legend || typeof legend.showIndex !== 'function') return;
      const active = activeElements(chart);
      const index = active.length ? active[0].index : -1;
      if (chart.$llLegendIndex === index) return;
      chart.$llLegendIndex = index;
      legend.showIndex(index);
    }
  };

  // ---- pointer helpers ------------------------------------------------------------
  function relativePos(chart, evt) {
    const rect = chart.canvas.getBoundingClientRect();
    const sx = rect.width ? chart.width / rect.width : 1;
    const sy = rect.height ? chart.height / rect.height : 1;
    return { x: (evt.clientX - rect.left) * sx, y: (evt.clientY - rect.top) * sy };
  }
  function inArea(area, pos) {
    return !!area && pos.x >= area.left && pos.x <= area.right && pos.y >= area.top && pos.y <= area.bottom;
  }

  // ---- synchronized hover ------------------------------------------------------------
  // syncHover(charts): hovering one chart activates the same timestamp on the others: their points,
  // the crosshair and live legend values, but not a tooltip, which would cover those charts.
  charts.syncHover = function syncHover(list) {
    const group = (Array.isArray(list) ? list : []).filter(Boolean);
    const lastIndex = new Map();
    const bound = [];
    let frame = 0;
    let pending = null;

    function activate(chart, idx) {
      if (!alive(chart)) return;
      // The visible range is part of the key: after a zoom the same index sits at another pixel.
      const sx = chart.scales && chart.scales.x;
      const key = idx < 0 ? '-1' : `${idx}:${sx ? sx.min : ''}:${sx ? sx.max : ''}`;
      const previous = lastIndex.has(chart) ? lastIndex.get(chart) : '-1';
      if (previous === key) return;
      lastIndex.set(chart, key);
      try {
        if (idx < 0) {
          chart.setActiveElements([]);
          chart.update('none');
          return;
        }
        const elements = [];
        (chart.data.datasets || []).forEach((ds, di) => {
          if (!chart.isDatasetVisible(di)) return;
          const meta = chart.getDatasetMeta(di);
          if (!meta || !meta.data || !meta.data[idx]) return;
          const raw = ds.data ? ds.data[idx] : null;
          const value = raw && typeof raw === 'object' ? raw.y : raw;
          if (value === null || value === undefined || Number.isNaN(Number(value))) return;
          elements.push({ datasetIndex: di, index: idx });
        });
        chart.setActiveElements(elements);
        chart.update('none');
      } catch (e) { /* chart may be mid-destroy */ }
    }
    function clearOthers(source) {
      group.forEach((chart) => { if (chart !== source) activate(chart, -1); });
    }
    function flush() {
      frame = 0;
      const job = pending;
      pending = null;
      if (!job || !alive(job.chart)) return;
      const source = job.chart;
      lastIndex.delete(source);
      const pos = relativePos(source, job.evt);
      const xs = source.scales && source.scales.x;
      let time = NaN;
      if (xs && inArea(source.chartArea, pos)) {
        const idx = Math.round(xs.getValueForPixel(pos.x));
        if (idx >= xs.min && idx <= xs.max) time = charts.timeAt(source.data.labels, idx);
      }
      group.forEach((chart) => {
        if (chart === source || !alive(chart)) return;
        if (!Number.isFinite(time)) { activate(chart, -1); return; }
        const sx = chart.scales && chart.scales.x;
        const j = charts.indexOfTime(chart.data.labels, time);
        activate(chart, sx && j >= 0 && j >= sx.min && j <= sx.max ? j : -1);
      });
    }
    group.forEach((chart) => {
      const canvas = chart.canvas;
      if (!canvas) return;
      const onMove = (evt) => {
        pending = { chart, evt: { clientX: evt.clientX, clientY: evt.clientY } };
        if (!frame) frame = raf(flush);
      };
      const onLeave = () => {
        pending = null;
        if (frame) { caf(frame); frame = 0; }
        clearOthers(chart);
      };
      canvas.addEventListener('mousemove', onMove);
      canvas.addEventListener('mouseleave', onLeave);
      bound.push({ canvas, onMove, onLeave });
    });
    return {
      destroy() {
        if (frame) { caf(frame); frame = 0; }
        pending = null;
        bound.forEach(({ canvas, onMove, onLeave }) => {
          canvas.removeEventListener('mousemove', onMove);
          canvas.removeEventListener('mouseleave', onLeave);
        });
        bound.length = 0;
        clearOthers(null);
      }
    };
  };

  // ---- drag-to-zoom ------------------------------------------------------------------
  // The selection rectangle is drawn by a global no-op-by-default plugin that reads chart.$llZoomSel.
  const zoomSelection = {
    id: 'llZoomSelection',
    afterDraw(chart) {
      const sel = chart.$llZoomSel;
      const area = chart.chartArea;
      if (!sel || !area) return;
      const color = readTheme().tone('accent');
      const x0 = Math.max(area.left, Math.min(sel.x0, sel.x1));
      const x1 = Math.min(area.right, Math.max(sel.x0, sel.x1));
      const ctx = chart.ctx;
      ctx.save();
      ctx.globalAlpha = 0.14;
      ctx.fillStyle = color;
      ctx.fillRect(x0, area.top, x1 - x0, area.bottom - area.top);
      ctx.globalAlpha = 0.8;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0 + 0.5, area.top); ctx.lineTo(x0 + 0.5, area.bottom);
      ctx.moveTo(x1 - 0.5, area.top); ctx.lineTo(x1 - 0.5, area.bottom);
      ctx.stroke();
      ctx.restore();
    }
  };
  charts.zoomSelection = zoomSelection;
  if (window.Chart && typeof window.Chart.register === 'function') {
    try { window.Chart.register(zoomSelection); } catch (e) { /* older Chart.js */ }
  }
  const DRAG_MIN_PX = 8;

  // zoom(charts, { onChange }): drag selects a time range on every chart; double-click resets.
  charts.zoom = function zoom(list, opts) {
    const group = (Array.isArray(list) ? list : []).filter(Boolean);
    const options = opts || {};
    const bound = [];
    let current = null;
    let drag = null;
    let drawFrame = 0;
    let suppressClickOn = null;

    function rawScaleX(chart) {
      const o = chart.config.options || (chart.config.options = {});
      o.scales = o.scales || {};
      o.scales.x = o.scales.x || {};
      return o.scales.x;
    }
    function apply(range) {
      current = range ? { from: range.from, to: range.to } : null;
      group.forEach((chart) => {
        if (!alive(chart)) return;
        const raw = rawScaleX(chart);
        const r = current ? indexRange(chart.data.labels, current.from, current.to) : null;
        if (r) { raw.min = r[0]; raw.max = r[1]; } else { delete raw.min; delete raw.max; }
        try {
          // Hover state from before the zoom points at old pixels; the next mouse move restores it.
          chart.setActiveElements([]);
          if (chart.tooltip) chart.tooltip.setActiveElements([], { x: 0, y: 0 });
          chart.update('none');
        } catch (e) { /* destroyed */ }
      });
      if (typeof options.onChange === 'function') {
        try { options.onChange(current ? { from: current.from, to: current.to } : null); } catch (e) { /* page callback error */ }
      }
    }
    function redrawSelection() {
      drawFrame = 0;
      if (drag && alive(drag.chart)) drag.chart.draw();
    }
    function endDrag() {
      window.removeEventListener('mousemove', onWindowMove);
      window.removeEventListener('mouseup', onWindowUp);
      window.removeEventListener('keydown', onWindowKey);
      if (drawFrame) { caf(drawFrame); drawFrame = 0; }
      const d = drag;
      drag = null;
      if (d && alive(d.chart) && d.chart.$llZoomSel) {
        delete d.chart.$llZoomSel;
        d.chart.draw();
      }
      return d;
    }
    function onWindowMove(evt) {
      if (!drag || !alive(drag.chart)) return;
      const area = drag.chart.chartArea;
      const pos = relativePos(drag.chart, evt);
      drag.x1 = Math.max(area.left, Math.min(area.right, pos.x));
      if (!drag.active && Math.abs(drag.x1 - drag.x0) >= DRAG_MIN_PX) drag.active = true;
      if (drag.active) {
        drag.chart.$llZoomSel = { x0: drag.x0, x1: drag.x1 };
        if (!drawFrame) drawFrame = raf(redrawSelection);
      }
    }
    function onWindowUp() {
      const d = endDrag();
      if (!d || !d.active || !alive(d.chart)) return;
      suppressClickOn = d.chart;
      setTimeout(() => { suppressClickOn = null; }, 0);
      const xs = d.chart.scales.x;
      const clampIdx = (v) => Math.max(xs.min, Math.min(xs.max, Math.round(v)));
      const i0 = clampIdx(xs.getValueForPixel(Math.min(d.x0, d.x1)));
      const i1 = clampIdx(xs.getValueForPixel(Math.max(d.x0, d.x1)));
      if (!(i1 > i0)) return;
      const from = charts.timeAt(d.chart.data.labels, i0);
      const to = charts.timeAt(d.chart.data.labels, i1);
      if (!Number.isFinite(from) || !Number.isFinite(to)) return;
      apply({ from, to });
    }
    function onWindowKey(evt) {
      if (evt.key === 'Escape') endDrag();
    }
    group.forEach((chart) => {
      const canvas = chart.canvas;
      if (!canvas) return;
      canvas.classList.add('ll-zoomable');
      const onDown = (evt) => {
        if (evt.button !== 0 || drag || !alive(chart)) return;
        const pos = relativePos(chart, evt);
        if (!inArea(chart.chartArea, pos)) return;
        evt.preventDefault();
        drag = { chart, x0: pos.x, x1: pos.x, active: false };
        window.addEventListener('mousemove', onWindowMove);
        window.addEventListener('mouseup', onWindowUp);
        window.addEventListener('keydown', onWindowKey);
      };
      const onDouble = (evt) => {
        evt.preventDefault();
        api.reset();
      };
      const onClick = (evt) => {
        if (suppressClickOn === chart) { evt.stopImmediatePropagation(); suppressClickOn = null; }
      };
      canvas.addEventListener('mousedown', onDown);
      canvas.addEventListener('dblclick', onDouble);
      canvas.addEventListener('click', onClick, true);
      bound.push({ canvas, onDown, onDouble, onClick });
    });
    const api = {
      reset() { apply(null); },
      range() { return current ? { from: current.from, to: current.to } : null; },
      destroy() {
        endDrag();
        bound.forEach(({ canvas, onDown, onDouble, onClick }) => {
          canvas.removeEventListener('mousedown', onDown);
          canvas.removeEventListener('dblclick', onDouble);
          canvas.removeEventListener('click', onClick, true);
          canvas.classList.remove('ll-zoomable');
        });
        bound.length = 0;
      }
    };
    return api;
  };

  // ---- legend chips -----------------------------------------------------------------------
  function numericValues(data) {
    const out = [];
    (Array.isArray(data) ? data : []).forEach((raw) => {
      const value = raw && typeof raw === 'object' ? raw.y : raw;
      if (value === null || value === undefined || value === '') return;
      const n = Number(value);
      if (Number.isFinite(n)) out.push(n);
    });
    return out;
  }
  function average(data) {
    const values = numericValues(data);
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : NaN;
  }
  function formatAverage(value) {
    if (!Number.isFinite(value)) return '—';
    const abs = Math.abs(value);
    const opts = abs >= 100 ? { maximumFractionDigits: 0 } : abs >= 10 ? { maximumFractionDigits: 1 } : abs >= 1 ? { maximumFractionDigits: 2 } : { maximumSignificantDigits: 3 };
    try { return LL.fmt.number(value, opts); } catch (e) { return String(Math.round(value * 100) / 100); }
  }
  charts.formatAverage = formatAverage;
  function datasetColor(ds) {
    if (ds && typeof ds.borderColor === 'string' && ds.borderColor) return ds.borderColor;
    if (ds && typeof ds.backgroundColor === 'string' && ds.backgroundColor) return ds.backgroundColor;
    return '#888888';
  }
  function pngName(filename) {
    const raw = typeof filename === 'function' ? filename() : filename;
    const base = String(raw || 'chart').trim().replace(/[\\/:*?"<>|\u0000-\u001f]+/g, '_') || 'chart';
    return /\.png$/i.test(base) ? base : `${base}.png`;
  }

  // downloadPng(chart, { filename }): the chart canvas (markers and thresholds included) plus a
  // legend of the visible series, on a background of --bg.
  charts.downloadPng = function downloadPng(chart, opts) {
    if (!alive(chart)) return false;
    const options = opts || {};
    const canvas = chart.canvas;
    const ratio = chart.currentDevicePixelRatio || (chart.width ? canvas.width / chart.width : 1) || 1;
    const rows = (chart.data.datasets || []).map((ds, i) => ({
      label: String((ds && ds.label) || ''),
      color: datasetColor(ds),
      avg: average(ds && ds.data),
      visible: chart.isDatasetVisible(i)
    })).filter((row) => row.visible);
    const theme = readTheme();
    const family = fontFamily();
    const cssWidth = canvas.width / ratio;
    const pad = 16; const rowH = 22; const titleH = 24;
    const legendH = rows.length ? pad + titleH + rows.length * rowH + pad / 2 : 0;
    const out = document.createElement('canvas');
    out.width = canvas.width;
    out.height = canvas.height + Math.ceil(legendH * ratio);
    const ctx = out.getContext('2d');
    ctx.fillStyle = theme.background;
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(canvas, 0, 0);
    if (rows.length) {
      ctx.save();
      ctx.scale(ratio, ratio);
      let y = canvas.height / ratio + pad;
      ctx.textBaseline = 'middle';
      ctx.fillStyle = theme.text;
      ctx.font = `600 14px ${family}`;
      ctx.fillText(LL.t('charts.legend.title'), pad, y + 8);
      y += titleH;
      ctx.font = `13px ${family}`;
      rows.forEach((row) => {
        const mid = y + rowH / 2 - 2;
        ctx.fillStyle = row.color;
        ctx.fillRect(pad, mid - 6, 12, 12);
        const avgText = LL.t('charts.legend.average', { value: formatAverage(row.avg) });
        const avgWidth = ctx.measureText(avgText).width;
        ctx.fillStyle = theme.text;
        ctx.textAlign = 'left';
        ctx.fillText(fitText(ctx, row.label, Math.max(40, cssWidth - pad * 3 - 20 - avgWidth)), pad + 20, mid);
        ctx.textAlign = 'right';
        ctx.fillText(avgText, cssWidth - pad, mid);
        y += rowH;
      });
      ctx.restore();
    }
    const link = document.createElement('a');
    link.href = out.toDataURL('image/png');
    link.download = pngName(options.filename);
    document.body.appendChild(link);
    link.click();
    link.remove();
    return true;
  };

  // legendChips(chart, host, { filename, maxVisible = 12, onToggle, separator = '· ' }): chips under a chart.
  // separator goes before the value of a chip; '' renders a list entry "dash, name, value" (the report legend).
  charts.legendChips = function legendChips(chart, host, opts) {
    const options = opts || {};
    const maxVisible = Number.isFinite(Number(options.maxVisible)) ? Math.max(1, Number(options.maxVisible)) : 12;
    const separator = typeof options.separator === 'string' ? options.separator : '· ';
    if (!host) return { refresh() {}, destroy() {} };
    host.classList.add('legend-chips');
    host.innerHTML = '';
    const list = document.createElement('div');
    list.className = 'legend-chips-list';
    list.setAttribute('role', 'group');
    list.setAttribute('aria-label', LL.t('charts.legend.series_aria'));
    const tools = document.createElement('div');
    tools.className = 'legend-chips-tools';
    tools.setAttribute('role', 'group');
    tools.setAttribute('aria-label', LL.t('charts.legend.tools_aria'));
    const search = document.createElement('input');
    search.type = 'search';
    search.className = 'legend-chips-search';
    search.placeholder = LL.t('charts.legend.search_placeholder');
    search.setAttribute('aria-label', LL.t('charts.legend.search_aria'));
    search.hidden = true;
    const toolButton = (act, text, title) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn btn-sm btn-ghost';
      btn.dataset.act = act;
      btn.textContent = text;
      if (title) btn.title = title;
      return btn;
    };
    const btnAll = toolButton('all', LL.t('charts.legend.all'), LL.t('charts.legend.all_title'));
    const btnNone = toolButton('none', LL.t('charts.legend.none'), LL.t('charts.legend.none_title'));
    const btnPng = toolButton('png', 'PNG', LL.t('charts.legend.png_title'));
    const empty = document.createElement('div');
    empty.className = 'legend-chips-empty';
    empty.textContent = LL.t('charts.legend.empty');
    empty.hidden = true;
    tools.appendChild(search);
    tools.appendChild(btnAll);
    tools.appendChild(btnNone);
    tools.appendChild(btnPng);
    host.appendChild(list);
    host.appendChild(tools);

    function chips() { return Array.from(list.querySelectorAll('.legend-chip')); }
    function syncPressed() {
      chips().forEach((chip) => {
        const idx = Number(chip.dataset.index);
        chip.setAttribute('aria-pressed', alive(chart) && chart.isDatasetVisible(idx) ? 'true' : 'false');
      });
    }
    function applyFilter() {
      const q = search.value.trim().toLowerCase();
      let shown = 0;
      chips().forEach((chip) => {
        const match = !q || String(chip.dataset.name || '').toLowerCase().includes(q);
        chip.hidden = !match;
        if (match) shown += 1;
      });
      empty.hidden = !q || shown > 0;
    }
    function render() {
      list.innerHTML = '';
      const datasets = alive(chart) && Array.isArray(chart.data.datasets) ? chart.data.datasets : [];
      const many = datasets.length > maxVisible;
      host.classList.toggle('is-many', many);
      search.hidden = !many;
      if (!many) search.value = '';
      datasets.forEach((ds, i) => {
        const name = String((ds && ds.label) || LL.t('charts.legend.series_n', { n: i + 1 }));
        const avgText = formatAverage(average(ds && ds.data));
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'legend-chip';
        chip.dataset.index = String(i);
        chip.dataset.name = name;
        chip.dataset.avg = avgText;
        chip.title = LL.t('charts.legend.chip_title', { name, avg: avgText });
        chip.setAttribute('aria-label', LL.t('charts.legend.chip_aria', { name, avg: avgText }));
        chip.style.setProperty('--chip-color', datasetColor(ds));
        const swatch = document.createElement('span');
        swatch.className = 'legend-chip-swatch';
        swatch.setAttribute('aria-hidden', 'true');
        const label = document.createElement('span');
        label.className = 'legend-chip-name';
        label.textContent = name;
        const avg = document.createElement('span');
        avg.className = 'legend-chip-avg';
        avg.setAttribute('aria-hidden', 'true');
        avg.textContent = `${separator}${avgText}`;
        chip.appendChild(swatch);
        chip.appendChild(label);
        chip.appendChild(avg);
        list.appendChild(chip);
      });
      list.appendChild(empty);
      btnAll.disabled = !datasets.length;
      btnNone.disabled = !datasets.length;
      btnPng.disabled = !alive(chart);
      syncPressed();
      applyFilter();
      if (liveIndex >= 0) showIndex(liveIndex);
    }
    // showIndex(index): chips show the value at a label index (hover, called by charts.crosshair);
    // -1 brings back the averages.
    let liveIndex = -1;
    function showIndex(index) {
      liveIndex = Number.isInteger(index) && index >= 0 && alive(chart) ? index : -1;
      const live = liveIndex >= 0;
      host.classList.toggle('is-live', live);
      const datasets = alive(chart) && Array.isArray(chart.data.datasets) ? chart.data.datasets : [];
      chips().forEach((chip) => {
        const span = chip.querySelector('.legend-chip-avg');
        if (!span) return;
        span.textContent = `${separator}${live ? formatAverage(valueAt(datasets[Number(chip.dataset.index)], liveIndex)) : chip.dataset.avg}`;
      });
    }
    function setVisible(indices, visible) {
      if (!alive(chart)) return;
      indices.forEach((i) => chart.setDatasetVisibility(i, visible));
      chart.update('none');
      syncPressed();
      if (typeof options.onToggle === 'function') {
        try { options.onToggle(chart); } catch (e) { /* page callback error */ }
      }
    }
    // «All» / «None» act on the chips that match the search, or on all chips without a search.
    function targetIndices() {
      return chips().filter((chip) => !chip.hidden).map((chip) => Number(chip.dataset.index));
    }
    const onListClick = (evt) => {
      const chip = evt.target.closest('.legend-chip');
      if (!chip || !list.contains(chip) || !alive(chart)) return;
      const idx = Number(chip.dataset.index);
      setVisible([idx], !chart.isDatasetVisible(idx));
    };
    const onListKey = (evt) => {
      const chip = evt.target.closest('.legend-chip');
      if (!chip) return;
      const visible = chips().filter((c) => !c.hidden);
      const pos = visible.indexOf(chip);
      let next = null;
      if (evt.key === 'ArrowRight' || evt.key === 'ArrowDown') next = visible[Math.min(visible.length - 1, pos + 1)];
      else if (evt.key === 'ArrowLeft' || evt.key === 'ArrowUp') next = visible[Math.max(0, pos - 1)];
      else if (evt.key === 'Home') next = visible[0];
      else if (evt.key === 'End') next = visible[visible.length - 1];
      if (next) { evt.preventDefault(); next.focus(); }
    };
    const onToolsClick = (evt) => {
      const btn = evt.target.closest('button[data-act]');
      if (!btn) return;
      if (btn.dataset.act === 'all') setVisible(targetIndices(), true);
      else if (btn.dataset.act === 'none') setVisible(targetIndices(), false);
      else if (btn.dataset.act === 'png') charts.downloadPng(chart, { filename: options.filename });
    };
    const onSearch = () => applyFilter();
    const onSearchKey = (evt) => {
      if (evt.key === 'Escape' && search.value) { evt.preventDefault(); search.value = ''; applyFilter(); }
    };
    list.addEventListener('click', onListClick);
    list.addEventListener('keydown', onListKey);
    tools.addEventListener('click', onToolsClick);
    search.addEventListener('input', onSearch);
    search.addEventListener('keydown', onSearchKey);
    // Swatches follow dataset colors, which ui.restyleCharts may change on a theme switch.
    let themeFrame = 0;
    const unsubscribeTheme = LL.ui && LL.ui.theme && typeof LL.ui.theme.onChange === 'function'
      ? LL.ui.theme.onChange(() => { if (!themeFrame) themeFrame = raf(() => { themeFrame = 0; render(); }); })
      : null;
    render();
    const api = {
      refresh: render,
      showIndex,
      destroy() {
        if (chart && chart.$llLegend === api) { delete chart.$llLegend; delete chart.$llLegendIndex; }
        if (themeFrame) { caf(themeFrame); themeFrame = 0; }
        if (typeof unsubscribeTheme === 'function') unsubscribeTheme();
        list.removeEventListener('click', onListClick);
        list.removeEventListener('keydown', onListKey);
        tools.removeEventListener('click', onToolsClick);
        search.removeEventListener('input', onSearch);
        search.removeEventListener('keydown', onSearchKey);
        host.innerHTML = '';
        host.classList.remove('legend-chips', 'is-many', 'is-live');
      }
    };
    // charts.crosshair finds the legend here to show live values while the chart is hovered.
    if (alive(chart)) chart.$llLegend = api;
    return api;
  };

  // ---- load profile: SVG staircase of the load steps (no Chart.js) ---------------------------------------
  // charts.stepProfileModel(opts) -> the geometry below, or null when no step has a time range and an RPS.
  // opts (the same object charts.stepProfile takes, plus an explicit width):
  //   steps    sla_details.load_step_table.steps: [{index, label, start_iso, end_iso, plateau_start_iso,
  //            plateau_end_iso, rps, selected, after_drop, dip}]; a step without a range or a finite rps is skipped.
  //   window   SLA zone, sla_details.stable_window ({start|start_iso, end|end_iso}); without it a marker band of
  //            role 'sla', else the plateau of the selected step.
  //   markers  items of the report page's reportMarkers(): {kind: 'band'|'line'|'tick', t, to, label, tone, role}
  //            (t and to in epoch ms or ISO). Lines (degradation, rps_drop) become dashed verticals with their
  //            label; bands of other roles (hold, spike) are drawn in their tone; ticks are ignored, the staircase
  //            shows the step starts. Markers outside the time range are skipped, bands are clipped to it.
  //   target   target RPS: a dashed line with "200 · target".
  //   width, height  size in px (1 unit = 1 px); 560×150 compact and 1000×190 full by default.
  //   compact  dashboard variant: level labels on the steps and a short "SLA" caption instead of the y grid.
  //   offsetHours  clock of the time labels; by default the offset of the steps' ISO strings (+3 without one,
  //            like the report markers).
  //   highlight  index of a step (load_step_table index) to shade, e.g. the step under the pointer in the step row.
  // Result: {width, height, compact, plot: {top, bottom}, x0, x1, yMax, maxRps, offsetHours, path,
  //   steps: [{x, w, y, rps, index, label, start, end, selected, afterDrop, level: {x, y, text, anchor}|null}],
  //   bands: [{x, w, label, labelX, labelY, tone, role}], lines: [{x, t, label, labelX, labelY, anchor, tone, role}],
  //   target: {y, value, text, x, anchor}|null, grid: [{y, value, text}], ticks: [{x, t, text, anchor}],
  //   sla: {index, rps, start, end, startText, endText}|null, highlight: {index, x, w, y, path}|null, aria}.
  const PROFILE_CHAR_W = 6.7; // JetBrains Mono 11px advance, for fitting labels
  const PROFILE_TONES = ['accent', 'ok', 'warn', 'fail', 'muted'];
  function profileMs(value) {
    if (typeof value === 'number') return Number.isFinite(value) ? value : null;
    if (value instanceof Date) return Number.isFinite(value.getTime()) ? value.getTime() : null;
    let text = typeof value === 'string' ? value.trim() : '';
    if (!text) return null;
    // A naive ISO date-time is UTC, as in the report markers.
    text = text.replace(' ', 'T');
    if (/T\d/.test(text) && !/(?:[zZ]|[+-]\d{2}:?\d{2})$/.test(text)) text += 'Z';
    const ms = Date.parse(text);
    return Number.isFinite(ms) ? ms : null;
  }
  function profileOffsetHours(steps) {
    const sample = steps.map((s) => s.end_iso || s.start_iso).find((v) => typeof v === 'string' && v) || '';
    const match = String(sample).match(/([+-])(\d{2}):(\d{2})$/);
    if (!match) return 3;
    return (match[1] === '-' ? -1 : 1) * (Number(match[2]) + Number(match[3]) / 60);
  }
  function profileClock(ms, offsetHours) {
    const shifted = new Date(ms + offsetHours * 3600000);
    return `${String(shifted.getUTCHours()).padStart(2, '0')}:${String(shifted.getUTCMinutes()).padStart(2, '0')}`;
  }
  // Smallest "round" number >= value: units × a power of ten (the fine list keeps the top of the scale close).
  const ROUND_UNITS = [1, 2, 2.5, 5, 10];
  const FINE_UNITS = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];
  function niceCeil(value, units) {
    if (!(value > 0)) return 1;
    const power = 10 ** Math.floor(Math.log10(value));
    const unit = (units || ROUND_UNITS).find((m) => m * power >= value - 1e-9);
    return unit * power;
  }
  function rpsText(value) {
    const digits = Math.abs(value) >= 100 || Number.isInteger(value) ? 0 : 1;
    if (LL.fmt && typeof LL.fmt.number === 'function') return LL.fmt.number(value, { maximumFractionDigits: digits });
    return String(Number(value.toFixed(digits)));
  }
  const round1 = (v) => Math.round(v * 10) / 10;
  function profileTone(tone, fallback) { return PROFILE_TONES.indexOf(tone) >= 0 ? tone : fallback; }
  // The shaded step of a model: its column and its level drawn again over the staircase; null for no or an unknown step.
  function profileHighlight(steps, target) {
    if (target === null || target === undefined || target === '' || typeof target === 'object') return null;
    const step = (steps || []).find((s) => s.index !== null && s.index !== undefined && String(s.index) === String(target));
    if (!step) return null;
    return { index: step.index, x: step.x, w: step.w, y: step.y, path: `M${step.x} ${step.y}H${round1(step.x + step.w)}` };
  }
  function textWidth(text) { return String(text || '').length * PROFILE_CHAR_W; }
  charts.stepProfileModel = function stepProfileModel(opts) {
    const o = opts || {};
    const rawSteps = (Array.isArray(o.steps) ? o.steps : []).filter((s) => s && typeof s === 'object');
    const steps = rawSteps.map((s) => {
      const start = profileMs(s.start_iso);
      const end = profileMs(s.end_iso);
      const rps = s.rps === null || s.rps === undefined || s.rps === '' ? NaN : Number(s.rps);
      if (start === null || end === null || !(end > start) || !Number.isFinite(rps)) return null;
      return {
        index: s.index === undefined ? null : s.index,
        label: typeof s.label === 'string' ? s.label : '',
        start,
        end,
        plateauStart: profileMs(s.plateau_start_iso),
        plateauEnd: profileMs(s.plateau_end_iso),
        rps,
        selected: s.selected === true,
        afterDrop: s.after_drop === true,
        dip: s.dip === true
      };
    }).filter(Boolean).sort((a, b) => a.start - b.start);
    if (!steps.length) return null;
    const compact = !!o.compact;
    const width = Math.max(120, Number(o.width) > 0 ? Number(o.width) : (compact ? 560 : 1000));
    const height = Math.max(80, Number(o.height) > 0 ? Number(o.height) : (compact ? 150 : 190));
    const top = compact ? 8 : 10;
    const bottom = height - (compact ? 22 : 34);
    const offsetHours = Number.isFinite(Number(o.offsetHours)) && o.offsetHours !== null && o.offsetHours !== ''
      ? Number(o.offsetHours) : profileOffsetHours(rawSteps);
    const x0 = steps[0].start;
    const x1 = Math.max(...steps.map((s) => s.end));
    const maxRps = Math.max(...steps.map((s) => s.rps));
    const target = Number(o.target) > 0 ? Number(o.target) : null;
    const yMax = niceCeil(Math.max(maxRps, target || 0, 1) * 1.12, FINE_UNITS);
    const xOf = (t) => round1(((t - x0) / (x1 - x0)) * width);
    const yOf = (v) => round1(bottom - (Math.max(0, v) / yMax) * (bottom - top));
    const inRange = (t) => t !== null && t >= x0 && t <= x1;

    // Staircase: every step runs to the start of the next one (across a pause too) or, the last, to its end.
    let path = '';
    const outSteps = steps.map((s, i) => {
      const next = steps[i + 1];
      const end = next ? next.start : s.end;
      const x = xOf(s.start);
      const w = round1(xOf(end) - x);
      const y = yOf(s.rps);
      path += i === 0 ? `M${x} ${y}H${round1(x + w)}` : `V${y}H${round1(x + w)}`;
      const text = `${s.afterDrop || s.dip ? '~' : ''}${rpsText(s.rps)}`;
      let level = null;
      if (compact && w >= textWidth(text) + 8) {
        const below = y + 14 <= bottom - 3;
        level = { x: round1(x + w / 2), y: below ? round1(y + 14) : round1(y - 6), text, anchor: 'middle' };
      }
      return { x, w, y, rps: s.rps, index: s.index, label: s.label, start: s.start, end, selected: s.selected, afterDrop: s.afterDrop, level };
    });

    // SLA zone: the window, else the SLA band of the markers, else the plateau of the selected step.
    const markers = (Array.isArray(o.markers) ? o.markers : []).filter((m) => m && typeof m === 'object');
    const win = o.window && typeof o.window === 'object' ? o.window : null;
    let slaRange = null;
    let slaLabel = '';
    if (win) {
      const a = profileMs(win.start !== undefined ? win.start : win.start_iso);
      const b = profileMs(win.end !== undefined ? win.end : win.end_iso);
      if (a !== null && b !== null && b > a) slaRange = [a, b];
    }
    const slaMarker = markers.find((m) => m.kind === 'band' && m.role === 'sla');
    if (!slaRange && slaMarker) {
      const a = profileMs(slaMarker.t);
      const b = profileMs(slaMarker.to);
      if (a !== null && b !== null && b > a) { slaRange = [a, b]; slaLabel = String(slaMarker.label || ''); }
    }
    const selected = steps.find((s) => s.selected) || null;
    if (!slaRange && selected) slaRange = [selected.plateauStart !== null ? selected.plateauStart : selected.start, selected.plateauEnd !== null ? selected.plateauEnd : selected.end];
    const slaFrom = slaRange ? Math.max(slaRange[0], x0) : null;
    const slaStep = selected || (slaRange ? steps.find((s) => s.start <= slaFrom && slaFrom < s.end) || null : null);
    const bands = [];
    const addBand = (range, label, tone, role) => {
      const a = Math.max(range[0], x0);
      const b = Math.min(range[1], x1);
      if (!(b > a)) return null;
      const x = xOf(a);
      const w = Math.max(1, round1(xOf(b) - x));
      const band = { x, w, label, labelX: round1(x + 4), labelY: compact ? round1(bottom - 6) : round1(bottom + 14), tone, role };
      bands.push(band);
      return band;
    };
    let sla = null;
    if (slaRange) {
      let label = LL.t('charts.profile.sla');
      if (!compact) label = slaLabel || (slaStep && slaStep.index !== null && slaStep.index !== '' ? LL.t('charts.profile.sla_step', { index: slaStep.index }) : label);
      if (addBand(slaRange, label, 'accent', 'sla')) {
        sla = {
          index: slaStep ? slaStep.index : null,
          rps: slaStep ? slaStep.rps : null,
          start: Math.max(slaRange[0], x0),
          end: Math.min(slaRange[1], x1),
          startText: profileClock(Math.max(slaRange[0], x0), offsetHours),
          endText: profileClock(Math.min(slaRange[1], x1), offsetHours)
        };
      }
    }
    markers.forEach((m) => {
      if (m.kind !== 'band' || m.role === 'sla') return;
      const a = profileMs(m.t);
      const b = profileMs(m.to);
      if (a === null || b === null || !(b > a)) return;
      addBand([a, b], String(m.label || ''), profileTone(m.tone, 'muted'), String(m.role || ''));
    });

    // Event lines with their labels in rows, so neighbours do not overlap.
    const lines = [];
    const rows = [];
    markers.filter((m) => m.kind === 'line').map((m) => ({ m, t: profileMs(m.t) }))
      .filter((item) => inRange(item.t)).sort((a, b) => a.t - b.t).forEach(({ m, t }) => {
        const x = xOf(t);
        const label = String(m.label || '');
        const w = textWidth(label);
        const anchor = x + 5 + w > width ? 'end' : 'start';
        const from = anchor === 'start' ? x + 5 : x - 5 - w;
        const to = from + w;
        let row = 0;
        while (rows[row] !== undefined && rows[row] > from - 8) row += 1;
        rows[row] = to;
        lines.push({ x, t, label, labelX: round1(anchor === 'start' ? x + 5 : x - 5), labelY: round1(top + 10 + row * 13), anchor, tone: profileTone(m.tone, 'fail'), role: String(m.role || '') });
      });

    const valueText = target !== null ? rpsText(target) : '';
    const targetLine = target !== null ? {
      y: yOf(target),
      value: target,
      text: LL.t('charts.profile.target', { value: valueText }),
      x: compact ? 4 : round1(width - 2),
      anchor: compact ? 'start' : 'end'
    } : null;

    // Y grid of the full variant: two or three round levels.
    const grid = [];
    if (!compact) {
      const step = niceCeil(yMax / 3);
      for (let v = step; v < yMax - 1e-9; v += step) grid.push({ y: yOf(v), value: v, text: rpsText(v) });
    }

    // Time labels: both ends plus round clock times that keep clear of them.
    const ticks = [{ x: 0, t: x0, text: profileClock(x0, offsetHours), anchor: 'start' }];
    const minGap = compact ? 70 : 60;
    const spanMin = (x1 - x0) / 60000;
    const maxTicks = Math.max(2, Math.floor(width / (compact ? 120 : 100)));
    const stepMin = [1, 2, 5, 10, 15, 20, 30, 60, 120, 180, 360, 720, 1440].find((m) => spanMin / m <= maxTicks) || 1440;
    const offsetMs = offsetHours * 3600000;
    const stepMs = stepMin * 60000;
    for (let t = Math.ceil((x0 + offsetMs) / stepMs) * stepMs - offsetMs; t < x1; t += stepMs) {
      const x = xOf(t);
      if (x - 0 < minGap || width - x < minGap) continue;
      ticks.push({ x, t, text: profileClock(t, offsetHours), anchor: 'middle' });
    }
    ticks.push({ x: width, t: x1, text: profileClock(x1, offsetHours), anchor: 'end' });

    return {
      width,
      height,
      compact,
      plot: { top, bottom },
      x0,
      x1,
      yMax,
      maxRps,
      offsetHours,
      path,
      steps: outSteps,
      bands,
      lines,
      target: targetLine,
      grid,
      ticks,
      sla,
      highlight: profileHighlight(outSteps, o.highlight),
      aria: LL.tn('charts.profile.aria', steps.length, { max: rpsText(maxRps) })
    };
  };

  function profileEsc(value) {
    return String(value === undefined || value === null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function profileSvg(model) {
    const { width, height, plot } = model;
    const text = (x, y, value, cls, anchor) => `<text x="${x}" y="${y}"${anchor && anchor !== 'start' ? ` text-anchor="${anchor}"` : ''}${cls ? ` class="${cls}"` : ''}>${profileEsc(value)}</text>`;
    const parts = [];
    const hl = model.highlight;
    model.bands.forEach((b) => parts.push(`<rect class="pf-band tone-${b.tone}" x="${b.x}" y="${plot.top}" width="${b.w}" height="${round1(plot.bottom - plot.top)}"/>`));
    if (hl) parts.push(`<rect class="pf-hl" x="${hl.x}" y="${plot.top}" width="${hl.w}" height="${round1(plot.bottom - plot.top)}"/>`);
    model.grid.forEach((g) => {
      parts.push(`<line class="pf-grid" x1="0" x2="${width}" y1="${g.y}" y2="${g.y}"/>`);
      parts.push(text(0, round1(g.y - 4), g.text));
    });
    parts.push(`<line class="pf-axis" x1="0" x2="${width}" y1="${plot.bottom + 0.5}" y2="${plot.bottom + 0.5}"/>`);
    if (model.target) {
      const t = model.target;
      parts.push(`<line class="pf-target" x1="0" x2="${width}" y1="${t.y}" y2="${t.y}"/>`);
      parts.push(text(t.x, round1(t.y - 5), t.text, 'pf-target-label', t.anchor));
    }
    parts.push(`<path class="pf-path" d="${model.path}"/>`);
    if (hl) parts.push(`<path class="pf-hl-path" d="${hl.path}"/>`);
    model.steps.forEach((s) => {
      if (!s.level) return;
      const cls = `pf-level${s.selected ? ' is-selected' : ''}${hl && hl.index === s.index ? ' is-highlighted' : ''}`;
      parts.push(text(s.level.x, s.level.y, s.level.text, cls, s.level.anchor));
    });
    model.lines.forEach((l) => {
      parts.push(`<line class="pf-mark tone-${l.tone}" x1="${l.x + 0.5}" x2="${l.x + 0.5}" y1="${plot.top}" y2="${plot.bottom}"/>`);
      if (l.label) parts.push(text(l.labelX, l.labelY, l.label, `pf-mark-label tone-${l.tone}`, l.anchor));
    });
    model.bands.forEach((b) => { if (b.label) parts.push(text(b.labelX, b.labelY, b.label, 'pf-band-label')); });
    model.ticks.forEach((t) => parts.push(text(t.x, height - 4, t.text, '', t.anchor)));
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}" preserveAspectRatio="none" role="img" aria-label="${profileEsc(model.aria)}">${parts.join('')}</svg>`;
  }
  charts.stepProfileSvg = profileSvg;

  // charts.stepProfile(host, opts): draws the profile into host (class ll-profile) at the host's width and
  // redraws when the width changes; with nothing to draw the host is emptied and hidden. opts as for
  // stepProfileModel. Returns {model, update(nextOpts), highlight(stepIndex|null), destroy()}; highlight shades one
  // step (the report does it while a cell of its step row is hovered or focused) and keeps it over redraws.
  charts.stepProfile = function stepProfile(host, opts) {
    if (!host) return null;
    const previous = host.$llProfile;
    if (previous && typeof previous.destroy === 'function') previous.destroy();
    const options = Object.assign({}, opts || {});
    let observer = null;
    let lastWidth = 0;
    const measure = () => {
      let w = Number(host.clientWidth) || 0;
      if (!w && typeof host.getBoundingClientRect === 'function') w = Number(host.getBoundingClientRect().width) || 0;
      return Math.round(w);
    };
    const handle = {
      model: null,
      update(next) {
        Object.assign(options, next || {});
        lastWidth = measure();
        const model = charts.stepProfileModel(Object.assign({}, options, { width: options.width || lastWidth || undefined }));
        if (host.classList) host.classList.add('ll-profile');
        if (!model) {
          host.innerHTML = '';
          host.hidden = true;
        } else {
          host.hidden = false;
          host.innerHTML = profileSvg(model);
        }
        handle.model = model;
        return handle;
      },
      highlight(target) {
        options.highlight = target === undefined ? null : target;
        if (!handle.model) return handle;
        const next = profileHighlight(handle.model.steps, options.highlight);
        const current = handle.model.highlight;
        if ((next && current && next.index === current.index) || (!next && !current)) return handle;
        handle.model = Object.assign({}, handle.model, { highlight: next });
        host.innerHTML = profileSvg(handle.model);
        return handle;
      },
      destroy() {
        if (observer) { try { observer.disconnect(); } catch (e) { /* already gone */ } observer = null; }
        if (host.$llProfile === handle) delete host.$llProfile;
      }
    };
    handle.update();
    if (!options.width && typeof window.ResizeObserver === 'function') {
      observer = new window.ResizeObserver(() => {
        const w = measure();
        if (w && w !== lastWidth) handle.update();
      });
      observer.observe(host);
    }
    host.$llProfile = handle;
    return handle;
  };
})();
