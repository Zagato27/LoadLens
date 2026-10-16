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
  // Lines in the overlay tooltip next to the cursor; the rest is «ещё N», all values are in the legend.
  const TOOLTIP_MAX = 8;

  const state = {
    runA: null,
    runB: null,
    agg: 'p95',
    schema: {},
    charts: {},
    // Legend chips and zoom handles of the open overlay chart, by the same key as `charts`.
    chartAddons: {},
    pickers: {},
    // Domain titles stored with the reports (run A wins when both have one).
    domainTitles: {}
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
    return LL.formatDuration(Number(row.end_ms) - Number(row.start_ms));
  }
  // Peak for step/stress, held RPS for soak, spike level for spike.
  function keyRpsText(row, peak) {
    const parsed = parsedOf(row);
    const has = (v) => v !== undefined && v !== null && v !== '';
    if (has(peak.max_rps)) return parsed.saturation ? `${peak.max_rps} (точка насыщения)` : String(peak.max_rps);
    const held = (parsed.stability_under_load || {}).actual_rps;
    if (has(held)) return `${held} (удержано)`;
    const spike = (parsed.spike_analysis || {}).max_spike_rps;
    if (has(spike)) return `${spike} (всплеск)`;
    return '—';
  }
  function renderVerdicts(a, b) {
    const order = ['final', 'lt_framework', 'microservices', 'jvm', 'database', 'kafka', 'hard_resources', 'lt_generators', 'application_logs'];
    const domains = order.filter((d) => a[d] || b[d]);
    const finalA = a.final; const finalB = b.final;
    const peakA = parsedOf(finalA).peak_performance || {}; const peakB = parsedOf(finalB).peak_performance || {};
    const cells = [
      '<div class="vg-head">Показатель</div>', `<div class="vg-head">A · ${esc(state.runA.value)}</div>`, `<div class="vg-head">B · ${esc(state.runB.value)}</div>`
    ];
    cells.push('<div>Итоговый вердикт</div>', `<div>${pill(finalA && finalA.verdict)}</div>`, `<div>${pill(finalB && finalB.verdict)}</div>`);
    cells.push('<div>Вердикт по SLA</div>', `<div>${pill(finalA && finalA.sla_verdict)}</div>`, `<div>${pill(finalB && finalB.sla_verdict)}</div>`);
    cells.push(`<div>${ui.term('stable_max', 'Макс. RPS / ключевой RPS типа теста')}</div>`, `<div>${esc(keyRpsText(finalA, peakA))}</div>`, `<div>${esc(keyRpsText(finalB, peakB))}</div>`);
    cells.push('<div>Тип теста</div>', `<div>${esc(LL.testTypeLabel(finalA && finalA.test_type))}</div>`, `<div>${esc(LL.testTypeLabel(finalB && finalB.test_type))}</div>`);
    cells.push('<div>Длительность окна</div>', `<div>${esc(durationText(finalA))}</div>`, `<div>${esc(durationText(finalB))}</div>`);
    domains.filter((d) => d !== 'final').forEach((d) => {
      cells.push(`<div>${esc(LL.domainTitleFor(d, state.domainTitles))}</div>`, `<div>${pill(a[d] && (a[d].verdict || a[d].llm_verdict))}</div>`, `<div>${pill(b[d] && (b[d].verdict || b[d].llm_verdict))}</div>`);
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
      if (domain && !key.startsWith(`${domain}::`)) return;
      (state.chartAddons[key] || []).forEach((addon) => { try { addon.destroy(); } catch (e) { /* already gone */ } });
      delete state.chartAddons[key];
      try { state.charts[key].destroy(); } catch (e) { /* already gone */ }
      delete state.charts[key];
    });
  }

  function metricsTableHead() {
    const aggLabel = esc(AGG_LABELS[state.agg]);
    return `<thead><tr><th>Метрика</th><th class="num">A (${aggLabel})</th><th class="num">B (${aggLabel})</th><th>Изменение B к A</th></tr></thead>`;
  }

  async function renderDomainTable(domain, panel) {
    panel.innerHTML = `<div class="card"><div class="table-wrap"><table class="data-table">${metricsTableHead()}<tbody>${ui.skeletonRows(4, 4)}</tbody></table></div></div>`;
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
    const table = document.createElement('table');
    table.className = 'data-table';
    table.innerHTML = metricsTableHead();
    const tbody = document.createElement('tbody');
    if (!rows.length) tbody.innerHTML = ui.emptyRow(4, 'В этом домене нет общих метрик у выбранных прогонов.', { icon: 'columns' });
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
    td.innerHTML = [
      `<div class="chart-toolbar"><strong>${esc(queryLabel)}</strong><span class="dim">A — сплошная, B — пунктир; время от старта прогона. Выделите участок мышью, чтобы приблизить</span>`,
      `<button type="button" class="chart-zoom-reset" hidden>${ui.icon('x')}Сбросить масштаб</button></div>`,
      '<div class="cmp-chart-wrap cmp-overlay">',
      `<div class="cmp-overlay-canvas"><canvas role="img" aria-label="${esc(`График: ${queryLabel}, прогоны A и B`)}"></canvas><div class="cmp-overlay-pending">${ui.skeleton(3)}</div></div>`,
      '<div class="cmp-legend"></div>',
      '</div><div class="series-table-wrap"></div>'
    ].join('');
    chartRow.appendChild(td);
    tr.after(chartRow);
    await drawOverlay(domain, queryLabel, chartRow);
    await renderSeriesTable(domain, queryLabel, chartRow.querySelector('.series-table-wrap'));
  }

  async function drawOverlay(domain, queryLabel, chartRow) {
    const canvas = chartRow.querySelector('canvas');
    const pending = chartRow.querySelector('.cmp-overlay-pending');
    const legendHost = chartRow.querySelector('.cmp-legend');
    const showProblem = (text) => {
      pending.innerHTML = `<div class="dim">${esc(text)}</div>`;
      pending.classList.add('is-message');
    };
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
      showProblem(`Ошибка загрузки: ${e.message}`);
      return;
    }
    if (!data.points || !data.points.length) { showProblem('Нет данных для выбранной метрики'); return; }
    // The row may have been closed while the series were loading.
    if (!chartRow.isConnected) return;
    const { labels, datasets } = overlayDatasets(data.points);
    const colors = ui.chartColors();
    const offsetLabel = (sec) => `${String(Math.floor(sec / 3600)).padStart(2, '0')}:${String(Math.floor((sec % 3600) / 60)).padStart(2, '0')}`;
    // The tooltip sits next to the cursor (largest values first, «ещё N»); the legend shows every value.
    const tooltip = LL.charts.htmlTooltip({ max: TOOLTIP_MAX });
    tooltip.callbacks = Object.assign({}, tooltip.callbacks, {
      title: (items) => (items.length ? `+${offsetLabel(Number(items[0].label) || 0)} от старта` : '')
    });
    // eslint-disable-next-line no-undef
    const chart = new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          customBackground: { color: colors.background },
          tooltip: tooltip
        },
        scales: {
          x: { title: { display: true, text: 'Время от старта (чч:мм)', color: colors.text }, ticks: { color: colors.text, maxRotation: 0, autoSkipPadding: 12, callback(v) { return offsetLabel(parseInt(this.getLabelForValue(v), 10) || 0); } }, grid: { color: colors.grid } },
          y: { ticks: { color: colors.text }, grid: { color: colors.grid } }
        }
      },
      plugins: [LL.charts.crosshair]
    });
    pending.remove();
    const key = `${domain}::${queryLabel}`;
    state.charts[key] = chart;
    const legend = LL.charts.legendChips(chart, legendHost, { filename: `compare-${queryLabel}` });
    const resetBtn = chartRow.querySelector('.chart-zoom-reset');
    const zoom = LL.charts.zoom([chart], { onChange: (range) => { resetBtn.hidden = !range; } });
    resetBtn.addEventListener('click', () => zoom.reset());
    state.chartAddons[key] = [legend, zoom];
  }

  // Datasets of the A/B overlay: one color per series name (shared by A and B, palette slot in
  // llSeries so a theme switch recolors them), A solid and B dashed, A and B of a series side by side.
  function overlayDatasets(points) {
    const xs = new Set();
    const seriesMap = new Map();
    const names = [];
    points.forEach((p) => {
      xs.add(p.t_offset_sec);
      const key = `${p.run_name}\u0000${p.series}`;
      if (!seriesMap.has(key)) seriesMap.set(key, new Map());
      seriesMap.get(key).set(p.t_offset_sec, p.value);
      if (!names.includes(p.series)) names.push(p.series);
    });
    const labels = Array.from(xs).sort((a, b) => a - b);
    const datasets = [];
    seriesMap.forEach((m, key) => {
      const [runName, series] = key.split('\u0000');
      const isB = runName === state.runB.value && runName !== state.runA.value;
      const slot = names.indexOf(series);
      const color = LL.charts.palette(slot);
      datasets.push({ label: `${isB ? 'B' : 'A'} · ${series}`, series, isB, llSeries: slot, data: labels.map((x) => (m.has(x) ? m.get(x) : null)), borderColor: color, backgroundColor: color, pointRadius: 0, borderWidth: 2, spanGaps: true, borderDash: isB ? [6, 4] : [] });
    });
    datasets.sort((a, b) => (a.llSeries - b.llSeries) || (Number(a.isB) - Number(b.isB)));
    return { labels, datasets };
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
      host.innerHTML = `<div class="table-wrap"><table class="data-table" style="margin-top:8px"><thead><tr><th>Серия</th><th class="num">A (${esc(aggLabel)})</th><th class="num">B (${esc(aggLabel)})</th><th>Изменение</th></tr></thead><tbody>${rows.map((r) => `<tr><td>${esc(r.series)}</td><td class="num">${fmt(r.value_a)}</td><td class="num">${fmt(r.value_b)}</td><td>${trendHtml(r.trend_pct, queryLabel)}</td></tr>`).join('')}</tbody></table></div>`;
    } catch (e) {
      host.innerHTML = `<div class="dim">Не удалось загрузить разбивку по сериям: ${esc(e.message)}</div>`;
    }
  }

  function renderDomainTabs() {
    const order = ['lt_framework', 'microservices', 'jvm', 'database', 'kafka', 'hard_resources', 'lt_generators'];
    const domains = Object.keys(state.schema).sort((a, b) => { const ia = order.indexOf(a); const ib = order.indexOf(b); return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib); });
    const nav = $('domainTabs');
    const panels = $('domainPanels');
    nav.innerHTML = '';
    panels.innerHTML = '';
    if (!domains.length) {
      panels.innerHTML = `<div class="card">${ui.emptyState({ icon: 'columns', title: 'Метрик для сравнения нет', text: 'В базе нет сохранённых метрик: они появляются после создания отчёта по прогону.', actionHtml: '<a class="btn btn-primary" href="/new" data-min-role="engineer">Новый отчёт</a>' })}</div>`;
      return;
    }
    domains.forEach((d, i) => {
      const btn = document.createElement('button');
      btn.className = `tab ${i === 0 ? 'active' : ''}`;
      btn.dataset.target = `cmp-panel-${i}`;
      btn.textContent = `${LL.domainTitleFor(d, state.domainTitles)} (${(state.schema[d] || []).length})`;
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
      state.domainTitles = Object.assign({}, LL.reportDomainTitles(Object.values(b)), LL.reportDomainTitles(Object.values(a)));
      renderVerdicts(a, b);
      await loadSchema();
      $('metricsCompare').hidden = false;
      $('aggNote').textContent = AGG_NOTES[state.agg];
      if ($('compareEmpty')) $('compareEmpty').hidden = true;
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

  // Empty-state action: the run with the newest report as A, the previous (successful if any) run of its
  // service as B. /compare_baseline looks for reports created earlier, hence the sort by report time.
  async function compareLatest(btn) {
    btn.disabled = true;
    try {
      const u = new URL('/runs', location.origin);
      u.searchParams.set('limit', '1');
      u.searchParams.set('sort', 'report_created_at');
      const resp = await fetch(u);
      const rows = await resp.json();
      if (!resp.ok) throw new Error(rows.error || `HTTP ${resp.status}`);
      if (!Array.isArray(rows) || !rows.length) { ui.toast('Прогонов пока нет: сначала создайте отчёт', { tone: 'info' }); return; }
      const r = rows[0];
      const item = { value: r.run_name, label: r.run_name, meta: runMeta(r), run: r };
      state.pickers.runA.setItem(item);
      setRun('runA', item);
      const found = (await pickBaseline('previous_success', true)) || (await pickBaseline('previous', true));
      if (found) await runCompare();
      else ui.toast(`У прогона «${r.run_name}» нет предыдущего прогона того же сервиса: выберите B вручную`, { tone: 'info' });
    } catch (e) {
      ui.toast(`Не удалось подобрать прогоны: ${e.message}`, { tone: 'error' });
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

    const params = new URLSearchParams(location.search);
    const ra = params.get('run_a'); const rb = params.get('run_b');
    // Empty state until the first comparison; hidden while runs from the link are being resolved.
    const empty = document.createElement('div');
    empty.id = 'compareEmpty';
    empty.className = 'card compare-empty';
    empty.hidden = !!ra;
    empty.innerHTML = ui.emptyState({
      icon: 'columns',
      title: 'Выберите два прогона',
      text: 'A — прогон, который нужно оценить; B — с чем сравнить: после выбора A его можно подставить кнопкой «Предыдущий успешный». Можно сразу сравнить последний прогон с предыдущим прогоном того же сервиса.',
      actionHtml: '<button type="button" class="btn" id="compareLatestBtn">Сравнить последний прогон</button>'
    });
    $('verdictCompare').before(empty);
    $('compareLatestBtn').addEventListener('click', (event) => compareLatest(event.currentTarget));
    try {
      if (ra) { const item = await findRun(ra); state.pickers.runA.setItem(item); setRun('runA', item); }
      if (rb) { const item = await findRun(rb); state.pickers.runB.setItem(item); setRun('runB', item); }
      else if (ra) { if (!(await pickBaseline('previous_success', true))) await pickBaseline('previous', true); }
      if (state.runA && state.runB) runCompare();
      else empty.hidden = false;
    } catch (e) {
      empty.hidden = false;
      ui.toast(`Не удалось подставить прогоны из ссылки: ${e.message}`, { tone: 'error' });
    }
  });
})();
