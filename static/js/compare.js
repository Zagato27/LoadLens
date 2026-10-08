// Compare page: run pickers with search, baseline selection, verdict panel, per-domain metric tables with inline charts.
(function () {
  const LL = window.LoadLens;
  const ui = LL.ui;
  const $ = (id) => document.getElementById(id);
  const esc = ui.escapeHtml;

  const AGG_LABELS = { p95: 'p95', avg: 'среднее', max: 'максимум' };
  const AGG_NOTES = {
    p95: 'p95 по всем точкам метрики за окно теста: показывает «хвост» без единичных выбросов.',
    avg: 'Среднее по всем точкам метрики за окно теста.',
    max: 'Максимум по всем точкам метрики за окно теста; чувствителен к кратким пикам.'
  };
  const HIGHER_BETTER = ['rps', 'request count', 'checks', 'throughput', 'fetch rate', 'requests'];
  const LOWER_BETTER = ['latency', 'duration', 'time', 'p95', 'p99', 'lag', 'cpu', 'memory', 'mem', 'disk', 'error', 'ошиб', 'gc', 'pause', 'heap', 'threads'];

  const state = {
    runA: null,
    runB: null,
    agg: 'p95',
    schema: {},
    charts: {},
    pickers: {}
  };

  // Background plugin for chart area
  const backgroundPlugin = {
    id: 'customBackground',
    beforeDraw(c, args, opts) {
      const { ctx, chartArea } = c;
      if (!chartArea) return;
      ctx.save();
      ctx.fillStyle = (opts && opts.color) || ui.chartColors().background;
      ctx.fillRect(chartArea.left, chartArea.top, chartArea.right - chartArea.left, chartArea.bottom - chartArea.top);
      ctx.restore();
    }
  };
  if (window.Chart && Chart.register) Chart.register(backgroundPlugin);

  // ---- run pickers ---------------------------------------------------------
  function runMeta(r) {
    const bits = [];
    if (r.service) bits.push(r.service);
    if (r.end_time || r.start_time) bits.push(LL.formatDateTime(r.end_time || r.start_time));
    if (r.test_type) bits.push(LL.testTypeLabel(r.test_type));
    if (r.verdict) bits.push(r.verdict);
    return bits.join(' · ');
  }
  async function fetchRunOptions(query) {
    const u = new URL('/runs', location.origin);
    if (query) u.searchParams.set('q', query);
    u.searchParams.set('limit', '50');
    const resp = await fetch(u);
    const rows = await resp.json();
    if (!resp.ok) throw new Error(rows.error || `HTTP ${resp.status}`);
    return rows.map((r) => ({ value: r.run_name, label: r.run_name, meta: runMeta(r), run: r }));
  }
  async function findRun(runName) {
    const options = await fetchRunOptions(runName);
    return options.find((o) => o.value === runName) || { value: runName, label: runName, meta: '', run: { run_name: runName } };
  }
  function setRun(which, item) {
    state[which] = item;
    $(which === 'runA' ? 'runAMeta' : 'runBMeta').textContent = item ? item.meta : '';
    const both = !!(state.runA && state.runB);
    $('compareBtn').disabled = !both;
    $('baselineSuccessBtn').disabled = !state.runA;
    $('baselinePrevBtn').disabled = !state.runA;
    $('compareHint').textContent = both
      ? (state.runA.value === state.runB.value ? 'Выбран один и тот же прогон.' : 'Нажмите «Сравнить».')
      : 'Выберите оба прогона.';
    if (which === 'runB' && item && !item.fromBaseline) $('baselineNote').textContent = '';
  }
  async function pickBaseline(mode, silent) {
    if (!state.runA) return false;
    const note = $('baselineNote');
    note.textContent = 'Ищем…';
    try {
      const u = new URL('/compare_baseline', location.origin);
      u.searchParams.set('run_name', state.runA.value);
      u.searchParams.set('mode', mode);
      const resp = await fetch(u);
      const data = await resp.json();
      if (resp.status === 404) {
        note.textContent = data.error || 'Не найдено';
        if (!silent) ui.toast(data.error || 'Подходящий прогон не найден', { tone: 'info' });
        return false;
      }
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      const item = { value: data.run_name, label: data.run_name, meta: [data.service, data.created_at ? LL.formatDateTime(data.created_at) : '', data.verdict].filter(Boolean).join(' · '), fromBaseline: true };
      state.pickers.runB.setItem(item);
      setRun('runB', item);
      note.textContent = mode === 'previous_success' ? 'B: предыдущий успешный прогон сервиса — можно изменить.' : 'B: предыдущий прогон сервиса — можно изменить.';
      return true;
    } catch (e) {
      note.textContent = `Ошибка: ${e.message}`;
      return false;
    }
  }

  // ---- verdict panel -------------------------------------------------------
  async function loadLlmRows(runName) {
    const resp = await fetch('/llm_reports?run_name=' + encodeURIComponent(runName));
    const rows = await resp.json();
    if (!resp.ok) throw new Error(rows.error || `HTTP ${resp.status}`);
    const byDomain = {};
    (Array.isArray(rows) ? rows : []).forEach((r) => { if (r && r.domain && !byDomain[r.domain]) byDomain[r.domain] = r; });
    return byDomain;
  }
  function parsedOf(row) {
    if (!row) return {};
    const p = row.parsed;
    if (p && typeof p === 'object') return p;
    if (typeof p === 'string') { try { return JSON.parse(p); } catch (e) { return {}; } }
    return {};
  }
  function pill(verdict) {
    if (!verdict) return '<span class="dim">—</span>';
    return `<span class="pill ${LL.verdictClass(verdict)}">${esc(verdict)}</span>`;
  }
  function durationText(row) {
    if (!row || !Number.isFinite(Number(row.start_ms)) || !Number.isFinite(Number(row.end_ms))) return '—';
    const minutes = Math.round((Number(row.end_ms) - Number(row.start_ms)) / 60000);
    return `${minutes} мин`;
  }
  function renderVerdicts(a, b) {
    const order = ['final', 'lt_framework', 'microservices', 'jvm', 'database', 'kafka', 'hard_resources', 'application_logs'];
    const domains = order.filter((d) => a[d] || b[d]);
    const finalA = a.final; const finalB = b.final;
    const peakA = parsedOf(finalA).peak_performance || {}; const peakB = parsedOf(finalB).peak_performance || {};
    const cells = [
      '<div class="vg-head">Показатель</div>', `<div class="vg-head">A · ${esc(state.runA.value)}</div>`, `<div class="vg-head">B · ${esc(state.runB.value)}</div>`
    ];
    cells.push('<div>Итоговый вердикт</div>', `<div>${pill(finalA && finalA.verdict)}</div>`, `<div>${pill(finalB && finalB.verdict)}</div>`);
    cells.push('<div>Вердикт по SLA</div>', `<div>${pill(finalA && finalA.sla_verdict)}</div>`, `<div>${pill(finalB && finalB.sla_verdict)}</div>`);
    cells.push(`<div>${ui.term('stable_max', 'Максимальный RPS')}</div>`, `<div>${esc(peakA.max_rps !== undefined && peakA.max_rps !== null ? peakA.max_rps : '—')}</div>`, `<div>${esc(peakB.max_rps !== undefined && peakB.max_rps !== null ? peakB.max_rps : '—')}</div>`);
    cells.push('<div>Тип теста</div>', `<div>${esc(LL.testTypeLabel(finalA && finalA.test_type))}</div>`, `<div>${esc(LL.testTypeLabel(finalB && finalB.test_type))}</div>`);
    cells.push('<div>Длительность окна</div>', `<div>${esc(durationText(finalA))}</div>`, `<div>${esc(durationText(finalB))}</div>`);
    domains.filter((d) => d !== 'final').forEach((d) => {
      cells.push(`<div>${esc(LL.domainTitle(d))}</div>`, `<div>${pill(a[d] && (a[d].verdict || a[d].llm_verdict))}</div>`, `<div>${pill(b[d] && (b[d].verdict || b[d].llm_verdict))}</div>`);
    });
    $('verdictGrid').innerHTML = cells.join('');
    const findingsHtml = (row, label) => {
      const findings = Array.isArray(parsedOf(row).findings) ? parsedOf(row).findings : [];
      const items = findings.map((f) => `<li><span class="sev ${esc(String(f.severity || 'low').toLowerCase())}">${esc(f.severity || 'low')}</span>${esc(f.summary || f.title || '')}${f.component ? ` <span class="dim">· ${esc(f.component)}</span>` : ''}</li>`).join('');
      return `<div class="findings-col"><h4>${esc(label)}</h4>${items ? `<ul class="findings-list">${items}</ul>` : '<div class="dim">Находок нет или итоговый анализ отсутствует.</div>'}</div>`;
    };
    $('findingsCompare').innerHTML = findingsHtml(finalA, `A · ${state.runA.value}`) + findingsHtml(finalB, `B · ${state.runB.value}`);
    ui.applyTerms($('verdictCompare'));
    $('verdictCompare').hidden = false;
  }

  // ---- metrics -------------------------------------------------------------
  function metricDirection(label) {
    const l = String(label || '').toLowerCase();
    if (HIGHER_BETTER.some((k) => l.includes(k)) && !LOWER_BETTER.some((k) => l.includes(k) && k !== 'time')) return 'higher';
    if (LOWER_BETTER.some((k) => l.includes(k))) return 'lower';
    return 'neutral';
  }
  function trendHtml(trendPct, label) {
    if (typeof trendPct !== 'number' || !Number.isFinite(trendPct)) return '<span class="trend neutral">—</span>';
    const dir = metricDirection(label);
    const abs = Math.abs(trendPct);
    const arrow = trendPct >= 0 ? '↑' : '↓';
    if (abs < 0.05) return '<span class="trend neutral">без изменений</span>';
    let cls = 'neutral'; let word = '';
    if (dir === 'higher') { cls = trendPct > 0 ? 'better' : 'worse'; word = trendPct > 0 ? 'лучше' : 'хуже'; }
    else if (dir === 'lower') { cls = trendPct < 0 ? 'better' : 'worse'; word = trendPct < 0 ? 'лучше' : 'хуже'; }
    return `<span class="trend ${cls}" title="${dir === 'neutral' ? 'Направление не определено по названию метрики' : (dir === 'higher' ? 'Рост считается улучшением' : 'Рост считается ухудшением')}">${arrow} ${abs.toFixed(1)} %${word ? ` · ${word}` : ''}</span>`;
  }
  const fmt = (v) => (typeof v === 'number' && Number.isFinite(v) ? (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(2)) : '—');

  async function loadSchema() {
    const resp = await fetch('/domains_schema');
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    state.schema = data || {};
  }

  function destroyCharts(domain) {
    Object.keys(state.charts).forEach((key) => {
      if (!domain || key.startsWith(`${domain}::`)) { try { state.charts[key].destroy(); } catch (e) { /* already gone */ } delete state.charts[key]; }
    });
  }

  async function renderDomainTable(domain, panel) {
    panel.innerHTML = `<div class="card">${ui.skeleton(4)}</div>`;
    destroyCharts(domain);
    const u = new URL('/compare_summary', location.origin);
    u.searchParams.set('run_a', state.runA.value);
    u.searchParams.set('run_b', state.runB.value);
    u.searchParams.set('domain', domain);
    u.searchParams.set('agg', state.agg);
    let rows;
    try {
      const resp = await fetch(u);
      rows = await resp.json();
      if (!resp.ok) throw new Error(rows.error || `HTTP ${resp.status}`);
    } catch (e) {
      panel.innerHTML = `<div class="card"><div class="empty-state">Не удалось загрузить сводку: ${esc(e.message)}</div></div>`;
      return;
    }
    const aggLabel = AGG_LABELS[state.agg];
    const table = document.createElement('table');
    table.className = 'data-table';
    table.innerHTML = `<thead><tr><th>Метрика</th><th class="num">A (${esc(aggLabel)})</th><th class="num">B (${esc(aggLabel)})</th><th>Изменение B к A</th></tr></thead>`;
    const tbody = document.createElement('tbody');
    if (!rows.length) tbody.innerHTML = '<tr class="empty-row"><td colspan="4">В этом домене нет общих метрик у выбранных прогонов.</td></tr>';
    rows.forEach((r) => {
      const tr = document.createElement('tr');
      tr.className = 'metric-row clickable';
      tr.innerHTML = `<td class="metric-label">${esc(r.query_label)}<div class="dim">нажмите, чтобы открыть график</div></td><td class="num">${fmt(r.value_a)}</td><td class="num">${fmt(r.value_b)}</td><td>${trendHtml(r.trend_pct, r.query_label)}</td>`;
      tr.addEventListener('click', () => toggleChartRow(domain, r.query_label, tr, tbody));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    const card = document.createElement('div');
    card.className = 'card';
    const wrap = document.createElement('div');
    wrap.className = 'table-wrap';
    wrap.appendChild(table);
    card.appendChild(wrap);
    panel.innerHTML = '';
    panel.appendChild(card);
  }

  async function toggleChartRow(domain, queryLabel, tr, tbody) {
    const existing = tbody.querySelector('tr.metric-chart-row');
    const wasOpen = existing && existing.dataset.label === queryLabel;
    if (existing) { existing.remove(); destroyCharts(domain); }
    tbody.querySelectorAll('tr.metric-row.open').forEach((r) => r.classList.remove('open'));
    if (wasOpen) return;
    tr.classList.add('open');
    const chartRow = document.createElement('tr');
    chartRow.className = 'metric-chart-row';
    chartRow.dataset.label = queryLabel;
    const td = document.createElement('td');
    td.colSpan = 4;
    td.innerHTML = `<div class="chart-toolbar"><strong>${esc(queryLabel)}</strong><span class="dim">A — сплошная, B — пунктир; время от старта прогона</span><button type="button" class="btn btn-sm chart-png">Скачать PNG</button></div><div class="cmp-chart-wrap" style="margin-top:0"><div class="cmp-chart-canvas"><canvas height="140"></canvas></div><div class="cmp-legend-panel"><h4>Серии</h4><div class="series-legend-host">${ui.skeleton(3)}</div></div></div><div class="series-table-wrap"></div>`;
    chartRow.appendChild(td);
    tr.after(chartRow);
    await drawOverlay(domain, queryLabel, chartRow);
    await renderSeriesTable(domain, queryLabel, chartRow.querySelector('.series-table-wrap'));
  }

  async function drawOverlay(domain, queryLabel, chartRow) {
    const canvas = chartRow.querySelector('canvas');
    const legendHost = chartRow.querySelector('.series-legend-host');
    const u = new URL('/compare_series', location.origin);
    u.searchParams.set('run_a', state.runA.value);
    u.searchParams.set('run_b', state.runB.value);
    u.searchParams.set('domain', domain);
    u.searchParams.set('query_label', queryLabel);
    u.searchParams.set('series_key', 'auto');
    u.searchParams.set('align', 'offset');
    let data;
    try {
      const resp = await fetch(u);
      data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    } catch (e) {
      legendHost.innerHTML = `<div class="dim">Ошибка загрузки: ${esc(e.message)}</div>`;
      return;
    }
    if (!data.points || !data.points.length) { legendHost.innerHTML = '<div class="dim">Нет данных для выбранной метрики</div>'; return; }
    const xs = new Set();
    const seriesMap = new Map();
    data.points.forEach((p) => {
      xs.add(p.t_offset_sec);
      const key = `${p.run_name}\u0000${p.series}`;
      if (!seriesMap.has(key)) seriesMap.set(key, new Map());
      seriesMap.get(key).set(p.t_offset_sec, p.value);
    });
    const labels = Array.from(xs).sort((a, b) => a - b);
    const datasets = [];
    seriesMap.forEach((m, key) => {
      const [runName, series] = key.split('\u0000');
      const isB = runName === state.runB.value && runName !== state.runA.value;
      const color = LL.colorFor(series);
      datasets.push({ label: `${isB ? 'B' : 'A'} · ${series}`, series, isB, data: labels.map((x) => (m.has(x) ? m.get(x) : null)), borderColor: color, backgroundColor: color, pointRadius: 0, borderWidth: 2, spanGaps: true, borderDash: isB ? [6, 4] : [] });
    });
    const colors = ui.chartColors();
    // eslint-disable-next-line no-undef
    const chart = new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        interaction: { mode: 'nearest', intersect: false },
        plugins: { legend: { display: false }, customBackground: { color: colors.background } },
        scales: {
          x: { title: { display: true, text: 'Время от старта (чч:мм)', color: colors.text }, ticks: { color: colors.text, callback(v) { const sec = parseInt(this.getLabelForValue(v), 10) || 0; return `${String(Math.floor(sec / 3600)).padStart(2, '0')}:${String(Math.floor((sec % 3600) / 60)).padStart(2, '0')}`; } }, grid: { color: colors.grid } },
          y: { ticks: { color: colors.text }, grid: { color: colors.grid } }
        }
      }
    });
    state.charts[`${domain}::${queryLabel}`] = chart;
    renderLegend(chart, legendHost);
    chartRow.querySelector('.chart-png').addEventListener('click', () => {
      const a = document.createElement('a');
      a.href = chart.toBase64Image();
      a.download = `compare-${queryLabel.replace(/[^\w\-]+/g, '_')}.png`;
      document.body.appendChild(a); a.click(); a.remove();
    });
  }

  function renderLegend(chart, host) {
    const rows = chart.data.datasets.map((ds, i) => {
      const vals = ds.data.filter((v) => v !== null && Number.isFinite(v));
      const avg = vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
      return { idx: i, ds, avg, visible: chart.isDatasetVisible(i) };
    });
    host.innerHTML = `<table class="series-legend"><thead><tr><th>Серия</th><th class="num">Среднее</th></tr></thead><tbody>${rows.map((r) => `<tr data-idx="${r.idx}" class="${r.visible ? '' : 'off'}" title="Нажмите, чтобы скрыть/показать"><td><span class="swatch ${r.ds.isB ? 'dashed' : ''}" style="background:${r.ds.isB ? 'none' : r.ds.borderColor};color:${r.ds.borderColor}"></span>${esc(r.ds.label)}</td><td class="num">${fmt(r.avg)}</td></tr>`).join('')}</tbody></table>`;
    host.querySelectorAll('tr[data-idx]').forEach((tr) => {
      tr.style.cursor = 'pointer';
      tr.addEventListener('click', () => {
        const idx = Number(tr.dataset.idx);
        const visible = chart.isDatasetVisible(idx);
        chart.setDatasetVisibility(idx, !visible);
        chart.update();
        tr.classList.toggle('off', visible);
      });
    });
  }

  async function renderSeriesTable(domain, queryLabel, host) {
    const u = new URL('/compare_metric_summary', location.origin);
    u.searchParams.set('run_a', state.runA.value);
    u.searchParams.set('run_b', state.runB.value);
    u.searchParams.set('domain', domain);
    u.searchParams.set('query_label', queryLabel);
    u.searchParams.set('agg', state.agg);
    try {
      const resp = await fetch(u);
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      const rows = Array.isArray(data.rows) ? data.rows : [];
      if (!rows.length) { host.innerHTML = ''; return; }
      const aggLabel = AGG_LABELS[state.agg];
      host.innerHTML = `<table class="data-table" style="margin-top:8px"><thead><tr><th>Серия</th><th class="num">A (${esc(aggLabel)})</th><th class="num">B (${esc(aggLabel)})</th><th>Изменение</th></tr></thead><tbody>${rows.map((r) => `<tr><td>${esc(r.series)}</td><td class="num">${fmt(r.value_a)}</td><td class="num">${fmt(r.value_b)}</td><td>${trendHtml(r.trend_pct, queryLabel)}</td></tr>`).join('')}</tbody></table>`;
    } catch (e) {
      host.innerHTML = `<div class="dim">Не удалось загрузить разбивку по сериям: ${esc(e.message)}</div>`;
    }
  }

  function renderDomainTabs() {
    const order = ['lt_framework', 'microservices', 'jvm', 'database', 'kafka', 'hard_resources'];
    const domains = Object.keys(state.schema).sort((a, b) => { const ia = order.indexOf(a); const ib = order.indexOf(b); return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib); });
    const nav = $('domainTabs');
    const panels = $('domainPanels');
    nav.innerHTML = '';
    panels.innerHTML = '';
    if (!domains.length) { panels.innerHTML = '<div class="card empty-state">В базе нет сохранённых метрик.</div>'; return; }
    domains.forEach((d, i) => {
      const btn = document.createElement('button');
      btn.className = `tab ${i === 0 ? 'active' : ''}`;
      btn.dataset.target = `cmp-panel-${i}`;
      btn.textContent = `${LL.domainTitle(d)} (${(state.schema[d] || []).length})`;
      nav.appendChild(btn);
      const panel = document.createElement('div');
      panel.id = `cmp-panel-${i}`;
      panel.className = `tabpanel ${i === 0 ? 'active' : ''}`;
      panel.dataset.domain = d;
      panels.appendChild(panel);
    });
    ui.tabs(nav, { onChange: (target) => { const panel = $(target); if (panel && !panel.dataset.loaded) { panel.dataset.loaded = '1'; renderDomainTable(panel.dataset.domain, panel); } } });
  }

  async function runCompare() {
    if (!state.runA || !state.runB) return;
    const btn = $('compareBtn');
    btn.disabled = true;
    $('compareHint').textContent = 'Загружаем…';
    try {
      const [a, b] = await Promise.all([loadLlmRows(state.runA.value), loadLlmRows(state.runB.value)]);
      renderVerdicts(a, b);
      await loadSchema();
      $('metricsCompare').hidden = false;
      $('aggNote').textContent = AGG_NOTES[state.agg];
      renderDomainTabs();
      try { history.replaceState(null, '', `/compare?run_a=${encodeURIComponent(state.runA.value)}&run_b=${encodeURIComponent(state.runB.value)}`); } catch (e) { /* ignore */ }
      $('compareHint').textContent = '';
    } catch (e) {
      ui.toast(`Не удалось выполнить сравнение: ${e.message}`, { tone: 'error' });
      $('compareHint').textContent = e.message;
    } finally {
      btn.disabled = false;
    }
  }

  function wireAgg() {
    document.querySelectorAll('#aggGroup .seg-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.agg = btn.dataset.agg;
        document.querySelectorAll('#aggGroup .seg-btn').forEach((b) => { const on = b === btn; b.classList.toggle('active', on); b.setAttribute('aria-pressed', on ? 'true' : 'false'); });
        $('aggNote').textContent = AGG_NOTES[state.agg];
        document.querySelectorAll('#domainPanels .tabpanel').forEach((panel) => { panel.dataset.loaded = ''; });
        const active = document.querySelector('#domainPanels .tabpanel.active');
        if (active) { active.dataset.loaded = '1'; renderDomainTable(active.dataset.domain, active); }
      });
    });
  }

  document.addEventListener('DOMContentLoaded', async () => {
    state.pickers.runA = ui.combobox($('runAPicker'), { placeholder: 'Начните вводить название запуска…', fetchOptions: fetchRunOptions, onSelect: (item) => setRun('runA', item) });
    state.pickers.runB = ui.combobox($('runBPicker'), { placeholder: 'Название запуска или подберите автоматически', fetchOptions: fetchRunOptions, onSelect: (item) => setRun('runB', item) });
    $('baselineSuccessBtn').addEventListener('click', () => pickBaseline('previous_success', false));
    $('baselinePrevBtn').addEventListener('click', () => pickBaseline('previous', false));
    $('compareBtn').addEventListener('click', runCompare);
    wireAgg();
    ui.theme.onChange(() => ui.restyleCharts());

    const params = new URLSearchParams(location.search);
    const ra = params.get('run_a'); const rb = params.get('run_b');
    try {
      if (ra) { const item = await findRun(ra); state.pickers.runA.setItem(item); setRun('runA', item); }
      if (rb) { const item = await findRun(rb); state.pickers.runB.setItem(item); setRun('runB', item); }
      else if (ra) { if (!(await pickBaseline('previous_success', true))) await pickBaseline('previous', true); }
      if (state.runA && state.runB) runCompare();
    } catch (e) {
      ui.toast(`Не удалось подставить прогоны из ссылки: ${e.message}`, { tone: 'error' });
    }
  });
})();
