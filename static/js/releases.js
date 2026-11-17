// SPDX-License-Identifier: Apache-2.0
// Releases: the list (/releases, releases.html) and the page of one release (/releases/<name>, release.html).
// Data: GET /releases_overview, GET/PUT /releases_data/<name>, GET /release_trend, GET /release_publications/<name>.
// Publication jobs: POST /releases/<name>/confluence.
// Marks of runs go through the routes of the releases blueprint: PUT /runs/<run>/release with accepted: true
// (accept a run in the release), PUT/DELETE /runs/<run>/baseline (pin or unpin the baseline of the service).
// The trend of a service is three Chart.js charts (p95, errors, load) over the release names; the selection lives in
// the address (?service=&type=). Pure helpers are exported as LL.releases._internals for the tests.
(function () {
  const LL = window.LoadLens;
  const ui = LL.ui;
  const esc = (value) => ui.escapeHtml(value);
  const $ = (id) => document.getElementById(id);
  const METRICS = ['p95', 'errors', 'capacity'];
  const TREND_TYPES = ['step', 'stress', 'soak', 'spike'];
  const TREND_LIMIT = 12;
  const PUBLISH_POLLS = 180;
  const PUBLISH_POLL_MS = 1500;
  const TICKET_RE = /^[A-Z][A-Z0-9_]*-\d+$/;
  const NOTE_MAX = 1000;
  const LIST_COLUMNS = 6;
  const PAGE_COLUMNS = 9;

  // ---- pure helpers ------------------------------------------------------------------------------

  // Release name of a /releases/<name> address ('' for the list).
  function releaseNameFromPath(pathname) {
    const match = /^\/releases\/(.+)$/.exec(String(pathname || ''));
    if (!match) return '';
    try { return decodeURIComponent(match[1]); } catch (e) { return match[1]; }
  }
  function releaseUrl(name) { return `/releases/${encodeURIComponent(name)}`; }
  function dataUrl(name) { return `/releases_data/${encodeURIComponent(name)}`; }

  // ?service=&type= of the trend.
  function parseQuery(search) {
    const params = new URLSearchParams(search || '');
    const type = (params.get('type') || '').trim().toLowerCase();
    return { service: (params.get('service') || '').trim(), type: TREND_TYPES.includes(type) ? type : '' };
  }
  function buildQuery(state) {
    const params = new URLSearchParams();
    if (state && state.service) params.set('service', state.service);
    if (state && state.type) params.set('type', state.type);
    const text = params.toString();
    return text ? `?${text}` : '';
  }

  function statusLabel(status) {
    return status === 'incomplete' ? LL.t('releases.status.incomplete') : LL.verdictLabel(status);
  }
  function statusClass(status) {
    return status === 'incomplete' ? 'rel-incomplete' : LL.verdictClass(status);
  }

  function finite(value) {
    if (value === null || value === undefined || value === '' || typeof value === 'boolean') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  // Signed change of a delta item: «+12,5 %», «+0,3 п.п.», «−8 %» (ru); '—' without a number.
  function deltaText(item) {
    const change = finite(item && item.change);
    if (change === null) return '—';
    const rounded = Math.round(change * 10) / 10;
    const sign = rounded > 0 ? '+' : (rounded < 0 ? '−' : '');
    const unit = item.unit === 'pp' ? LL.t('releases.unit.pp') : '%';
    return `${sign}${LL.fmt.number(Math.abs(rounded), { maximumFractionDigits: 1 })} ${unit}`;
  }
  // Tone of a delta against the service thresholds: ok | warn | fail, else neutral (no thresholds).
  function deltaTone(item) {
    const tone = item && item.tone;
    return ['ok', 'warn', 'fail'].includes(tone) ? tone : 'neutral';
  }
  // Value of a metric with its unit: p95 in ms, errors in %, load in RPS.
  function valueText(metric, value) {
    const number = finite(value);
    if (number === null) return '—';
    if (metric === 'p95') return LL.t('releases.value.ms', { value: LL.fmt.number(number, { maximumFractionDigits: number >= 100 ? 0 : 1 }) });
    if (metric === 'errors') return LL.t('releases.value.pct', { value: LL.fmt.number(number, { maximumFractionDigits: 2 }) });
    return LL.t('releases.value.rps', { value: LL.fmt.number(number, { maximumFractionDigits: 0 }) });
  }
  // «412 ms → 380 ms» for the title of a delta cell.
  function deltaTitle(item) {
    if (!item) return '';
    const metric = item.metric === 'p95' && item.stat === 'p99' ? 'p95' : item.metric;
    const stat = item.stat === 'p99' ? ` (${LL.t('releases.value.p99')})` : '';
    return `${valueText(metric, item.baseline)} → ${valueText(metric, item.current)}${stat}`;
  }
  // Where the two runs were compared: «на 600 RPS», «на потолке 540 RPS»…; '' without a level.
  function levelText(deltas) {
    const level = finite(deltas && deltas.level);
    if (level === null) return '';
    const value = LL.fmt.number(level, { maximumFractionDigits: 0 });
    const keys = { target: 'releases.level.target', ceiling: 'releases.level.ceiling', hold: 'releases.level.hold', spike: 'releases.level.spike' };
    const key = keys[deltas.kind];
    return key ? LL.t(key, { level: value }) : LL.t('releases.level.plain', { level: value });
  }
  // Why a row has no deltas.
  function reasonText(reason) {
    const keys = {
      no_previous: 'releases.reason.no_previous',
      no_accepted: 'releases.reason.no_accepted',
      mode_unsupported: 'releases.reason.mode_unsupported',
      no_checks: 'releases.reason.no_checks',
      no_baseline_sla: 'releases.reason.no_checks',
      level_mismatch: 'releases.reason.level_mismatch',
      no_step_at_level: 'releases.reason.level_mismatch',
      no_level: 'releases.reason.level_mismatch'
    };
    return LL.t(keys[reason] || 'releases.reason.unknown');
  }

  // Cells of the release table: one model per row of GET /releases_data/<name>.
  function rowsModel(rows) {
    return (Array.isArray(rows) ? rows : []).filter((row) => row && typeof row === 'object').map((row) => {
      const run = row.run || null;
      const deltas = row.deltas || {};
      const items = {};
      (Array.isArray(deltas.items) ? deltas.items : []).forEach((item) => { if (item && item.metric) items[item.metric] = item; });
      const cells = {};
      METRICS.forEach((metric) => {
        const item = items[metric];
        cells[metric] = item
          ? { text: deltaText(item), tone: deltaTone(item), title: deltaTitle(item) }
          : { text: '—', tone: 'neutral', title: '' };
      });
      const previous = row.previous || null;
      const stored = row.stored_regression && typeof row.stored_regression === 'object' ? row.stored_regression : null;
      const storedBase = stored ? String(stored.baseline_run_name || stored.baseline_run || '') : '';
      const attempts = Array.isArray(row.attempts) ? row.attempts : [];
      return {
        key: `${row.service}\u0000${row.test_type}`,
        service: String(row.service || ''),
        testType: String(row.test_type || ''),
        run,
        verdict: run ? LL.verdictCode(run.verdict) : null,
        manual: !!(run && run.manual_verdict),
        baseline: !!row.baseline,
        reportUrl: row.report_url || (run ? LL.reportUrl(run) : ''),
        compareUrl: row.compare_url || '',
        available: !!deltas.available,
        reason: deltas.available ? '' : reasonText(deltas.reason),
        level: deltas.available ? levelText(deltas) : '',
        cells,
        previous,
        // The report was checked against another baseline when it was built (the page compares live).
        staleNote: !!(storedBase && previous && previous.run_name && storedBase !== previous.run_name),
        storedBase,
        attempts
      };
    });
  }

  // Series of the trend charts: values in report order, null where a run has none (the line skips it).
  function trendModel(points) {
    const list = (Array.isArray(points) ? points : []).filter((point) => point && typeof point === 'object');
    const series = {};
    METRICS.forEach((metric) => { series[metric] = list.map((point) => finite(point[metric])); });
    return {
      labels: list.map((point) => String(point.release || point.run_name || '')),
      runs: list.map((point) => String(point.run_name || '')),
      verdicts: list.map((point) => LL.verdictCode(point.verdict)),
      baseline: list.map((point) => !!point.baseline),
      levels: list.map((point) => ({ kind: point.kind || null, level: finite(point.level) })),
      series,
      empty: METRICS.every((metric) => series[metric].every((value) => value === null))
    };
  }

  // ---- shared: requests ----------------------------------------------------------------------------

  async function api(method, url, body) {
    const init = { method };
    if (body !== undefined) {
      init.headers = { 'Content-Type': 'application/json' };
      init.body = JSON.stringify(body);
    }
    const resp = await fetch(url, init);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const error = new Error(data.error || LL.t('releases.error.status', { status: resp.status }));
      error.status = resp.status;
      throw error;
    }
    return data;
  }

  // ---- shared: the trend -------------------------------------------------------------------------

  const trend = { charts: [], model: null, request: 0 };

  function destroyCharts() {
    trend.charts.forEach((chart) => { try { chart.destroy(); } catch (e) { /* already gone */ } });
    trend.charts = [];
  }

  function toneColor(colors, code) {
    return { pass: colors.statusOk, risk: colors.statusWarn, fail: colors.statusFail }[code] || colors.statusNa;
  }

  // Three charts (p95, errors, load) into host; points are coloured by the verdict, the pinned baseline is a diamond.
  function trendCharts(host, model) {
    destroyCharts();
    host.innerHTML = '';
    if (!model || !model.labels.length || model.empty) {
      host.innerHTML = ui.emptyState({ icon: 'trending', text: LL.t('releases.trend.empty') });
      return;
    }
    if (!window.Chart) {
      host.innerHTML = `<p class="dim">${esc(LL.t('releases.trend.no_charts'))}</p>`;
      return;
    }
    const colors = ui.chartColors();
    const titles = { p95: LL.t('releases.trend.p95'), errors: LL.t('releases.trend.errors'), capacity: LL.t('releases.trend.capacity') };
    const grid = document.createElement('div');
    grid.className = 'release-trend-grid';
    host.appendChild(grid);
    METRICS.forEach((metric) => {
      const box = document.createElement('figure');
      box.className = 'release-trend-chart';
      box.innerHTML = `<figcaption>${esc(titles[metric])}</figcaption><div class="release-trend-canvas"><canvas role="img" aria-label="${esc(titles[metric])}"></canvas></div>`;
      grid.appendChild(box);
      const tooltip = LL.charts && LL.charts.htmlTooltip ? LL.charts.htmlTooltip({ max: 1 }) : {};
      tooltip.callbacks = Object.assign({}, tooltip.callbacks, {
        title: (items) => {
          if (!items.length) return '';
          const index = items[0].dataIndex;
          const lines = [model.labels[index], `${model.runs[index]} · ${LL.verdictLabel(model.verdicts[index])}`];
          const level = levelText(model.levels[index]);
          if (level) lines.push(level);
          if (model.baseline[index]) lines.push(LL.t('releases.trend.baseline'));
          return lines;
        }
      });
      // eslint-disable-next-line no-undef
      const chart = new Chart(box.querySelector('canvas').getContext('2d'), {
        type: 'line',
        data: {
          labels: model.labels.slice(),
          datasets: [{
            label: titles[metric],
            data: model.series[metric].slice(),
            borderColor: colors.accent,
            backgroundColor: colors.accent,
            borderWidth: 2,
            spanGaps: true,
            pointRadius: model.baseline.map((pinned) => (pinned ? 7 : 4)),
            pointHoverRadius: model.baseline.map((pinned) => (pinned ? 9 : 6)),
            pointStyle: model.baseline.map((pinned) => (pinned ? 'rectRot' : 'circle')),
            pointBackgroundColor: model.verdicts.map((code) => toneColor(colors, code)),
            pointBorderColor: model.baseline.map((pinned, i) => (pinned ? colors.ink : toneColor(colors, model.verdicts[i])))
          }]
        },
        options: {
          // Tick numbers in the interface language, not the browser's.
          locale: LL.intl,
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          interaction: { mode: 'index', intersect: false },
          plugins: { legend: { display: false }, tooltip },
          scales: {
            x: { ticks: { color: colors.text, maxRotation: 0, autoSkip: true, autoSkipPadding: 8 }, grid: { color: colors.grid } },
            y: { beginAtZero: metric !== 'p95', ticks: { color: colors.text }, grid: { color: colors.grid } }
          }
        },
        plugins: LL.charts && LL.charts.crosshair ? [LL.charts.crosshair] : []
      });
      trend.charts.push(chart);
    });
  }

  async function loadTrend(service, type) {
    const host = $('trendCharts');
    if (!host) return;
    const seq = ++trend.request;
    if (!service) {
      destroyCharts();
      host.innerHTML = ui.emptyState({ icon: 'trending', text: LL.t('releases.trend.pick_service') });
      return;
    }
    host.innerHTML = ui.skeleton(3);
    try {
      const params = new URLSearchParams({ service, test_type: type || 'step', limit: String(TREND_LIMIT) });
      const data = await api('GET', `/release_trend?${params.toString()}`);
      if (seq !== trend.request) return;
      trend.model = trendModel(data.points);
      trendCharts(host, trend.model);
    } catch (e) {
      if (seq !== trend.request) return;
      destroyCharts();
      host.innerHTML = `<p class="dim">${esc(LL.t('releases.trend.load_failed', { error: e.message }))}</p>`;
    }
  }

  function fillSelect(select, options, value) {
    select.innerHTML = options.map((option) => `<option value="${esc(option.value)}"${option.value === value ? ' selected' : ''}>${esc(option.label)}</option>`).join('');
  }

  // Service and type pickers of the trend; `pairs` limits the types to those a service has (release page).
  function setupTrend(services, pairs) {
    const section = $('trendSection');
    const serviceSelect = $('trendService');
    const typeSelect = $('trendType');
    if (!section || !serviceSelect || !typeSelect) return;
    const query = parseQuery(location.search);
    const state = {
      service: services.some((item) => item.value === query.service) ? query.service : ((services[0] || {}).value || ''),
      type: query.type || ''
    };
    const typesOf = (service) => {
      const own = pairs ? pairs.filter((pair) => pair.service === service).map((pair) => pair.type) : TREND_TYPES;
      const list = own.filter((type, i) => own.indexOf(type) === i);
      return list.length ? list : ['step'];
    };
    const sync = (push) => {
      const types = typesOf(state.service);
      if (!types.includes(state.type)) state.type = types[0];
      fillSelect(serviceSelect, services, state.service);
      fillSelect(typeSelect, types.map((type) => ({ value: type, label: LL.testTypeLabel(type) })), state.type);
      if (push) history.replaceState(null, '', `${location.pathname}${buildQuery(state)}`);
      loadTrend(state.service, state.type);
    };
    serviceSelect.onchange = () => { state.service = serviceSelect.value; sync(true); };
    typeSelect.onchange = () => { state.type = typeSelect.value; sync(true); };
    section.hidden = !services.length && !!pairs;
    sync(false);
  }

  // The charts take the colours of the theme when they are drawn: redraw them on a theme change.
  if (ui.theme && typeof ui.theme.onChange === 'function') {
    ui.theme.onChange(() => {
      const host = $('trendCharts');
      if (host && trend.model) trendCharts(host, trend.model);
    });
  }

  // ---- the release list --------------------------------------------------------------------------

  function verdictCountsHtml(verdicts) {
    const counts = verdicts && typeof verdicts === 'object' ? verdicts : {};
    const parts = LL.VERDICTS.filter((code) => Number(counts[code]) > 0)
      .map((code) => `<span class="st ${LL.verdictClass(code)}" title="${esc(LL.verdictLabel(code))}">${esc(String(counts[code]))}</span>`);
    return parts.length ? `<span class="release-verdict-counts">${parts.join('')}</span>` : '<span class="dim">—</span>';
  }

  function renderList(items) {
    const body = $('releaseRows');
    body.innerHTML = '';
    if (!items.length) {
      body.innerHTML = ui.emptyRow(LIST_COLUMNS, LL.t('releases.list.empty'), {
        icon: 'tag',
        actionHtml: `<a class="btn" href="/reports">${esc(LL.t('releases.list.to_archive'))}</a>`
      });
      return;
    }
    items.forEach((item) => {
      const row = document.createElement('tr');
      row.className = 'data-row';
      const url = releaseUrl(item.release);
      row.innerHTML = [
        `<td class="name"><a class="run-link" href="${esc(url)}">${esc(item.release)}</a></td>`,
        `<td><span class="pill ${statusClass(item.status)}">${esc(statusLabel(item.status))}</span></td>`,
        `<td class="m">${esc(LL.t('releases.list.services_value', { tested: item.services_tested, required: item.services_required }))}</td>`,
        `<td>${verdictCountsHtml(item.verdicts)}</td>`,
        `<td class="m muted nowrap">${esc(item.last_created_at ? LL.formatDateTime(item.last_created_at) : '—')}</td>`,
        `<td class="m">${esc(item.ticket || '—')}</td>`
      ].join('');
      row.addEventListener('click', (event) => {
        if (event.target.closest && event.target.closest('a')) return;
        location.href = url;
      });
      body.appendChild(row);
    });
  }

  async function initList() {
    const body = $('releaseRows');
    body.innerHTML = ui.skeletonRows(LIST_COLUMNS, 4);
    try {
      const items = await api('GET', '/releases_overview');
      renderList(Array.isArray(items) ? items : []);
    } catch (e) {
      body.innerHTML = '';
      const hint = $('releasesHint');
      hint.hidden = false;
      hint.innerHTML = `<div class="report-note-title">${esc(LL.t('releases.list.unavailable'))}</div><p>${esc(e.message)}</p>`;
    }
    let services = [];
    try {
      const data = await api('GET', '/services');
      services = (Array.isArray(data.services) ? data.services : []).map((item) => ({ value: String(item.id), label: String(item.title || item.id) }));
    } catch (e) { services = []; }
    setupTrend(services, null);
  }

  // ---- the page of a release -----------------------------------------------------------------------

  const page = { name: '', data: null, open: new Set() };

  function menuButton(label, onClick) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = label;
    btn.addEventListener('click', (event) => { event.stopPropagation(); onClick(btn); });
    return btn;
  }

  function deltaCell(cell, extra) {
    const td = document.createElement('td');
    td.className = 'num';
    td.innerHTML = `<span class="rel-delta tone-${esc(cell.tone)}"${cell.title ? ` title="${esc(cell.title)}"` : ''}>${esc(cell.text)}</span>${extra ? `<span class="sub">${esc(extra)}</span>` : ''}`;
    return td;
  }

  function runCellHtml(model) {
    if (!model.run) return `<span class="dim">${esc(LL.t('releases.table.no_accepted'))}</span>`;
    const badges = [];
    if (model.baseline) badges.push(`<span class="rel-badge">${esc(LL.t('releases.table.baseline'))}</span>`);
    if (model.attempts.length > 1) badges.push(`<span class="rel-badge dim">${esc(LL.t('releases.table.attempts_count', { n: model.attempts.length }))}</span>`);
    const created = model.run.created_at ? LL.formatDateTime(model.run.created_at) : '';
    return `<a class="run-link" href="${esc(model.reportUrl)}">${esc(model.run.run_name)}</a>${badges.join('')}${created ? `<span class="sub">${esc(created)}</span>` : ''}`;
  }

  function previousCellHtml(model) {
    const previous = model.previous;
    if (!previous || !previous.run_name) return '<span class="dim">—</span>';
    const label = esc(previous.release || '—');
    const link = model.compareUrl ? `<a href="${esc(model.compareUrl)}" title="${esc(LL.t('releases.table.compare_title'))}">${label}</a>` : label;
    const stale = model.staleNote ? `<span class="sub rel-stale">${esc(LL.t('releases.table.stale', { run: model.storedBase }))}</span>` : '';
    return `${link}<span class="sub">${esc(previous.run_name)}</span>${stale}`;
  }

  async function acceptRun(runName) {
    try {
      await api('PUT', `/runs/${encodeURIComponent(runName)}/release`, { release: page.name, accepted: true });
      ui.toast(LL.t('releases.attempts.accepted_toast', { run: runName }), { tone: 'ok' });
      await loadRelease();
    } catch (e) {
      ui.toast(LL.t('releases.attempts.accept_failed', { error: e.message }), { tone: 'error' });
    }
  }

  async function toggleBaseline(model) {
    const runName = model.run.run_name;
    try {
      await api(model.baseline ? 'DELETE' : 'PUT', `/runs/${encodeURIComponent(runName)}/baseline`, {});
      ui.toast(LL.t(model.baseline ? 'releases.baseline.unpinned' : 'releases.baseline.pinned', { run: runName }), { tone: 'ok' });
      await loadRelease();
    } catch (e) {
      ui.toast(LL.t('releases.baseline.failed', { error: e.message }), { tone: 'error' });
    }
  }

  function actionsCell(model) {
    const td = document.createElement('td');
    td.className = 'actions';
    const menu = document.createElement('details');
    menu.className = 'menu';
    menu.addEventListener('toggle', () => {
      const wrap = menu.closest('.table-wrap');
      if (wrap) wrap.classList.toggle('menu-open', !!document.querySelector('details.menu[open]'));
    });
    const summary = document.createElement('summary');
    summary.textContent = '⋯';
    summary.title = LL.t('releases.table.actions');
    summary.setAttribute('aria-label', LL.t('releases.table.actions_for', { service: model.service }));
    menu.appendChild(summary);
    const items = document.createElement('div');
    items.className = 'menu-items';
    if (model.compareUrl) {
      items.appendChild(menuButton(LL.t('releases.table.compare_previous'), () => { location.href = model.compareUrl; }));
    }
    if (model.run && LL.can('engineer')) {
      items.appendChild(menuButton(LL.t(model.baseline ? 'releases.table.unpin_baseline' : 'releases.table.pin_baseline'), () => {
        menu.removeAttribute('open');
        toggleBaseline(model);
      }));
    }
    items.appendChild(menuButton(LL.t('releases.table.attempts', { n: model.attempts.length }), () => {
      menu.removeAttribute('open');
      if (page.open.has(model.key)) page.open.delete(model.key); else page.open.add(model.key);
      renderRows(page.data.rows);
    }));
    menu.appendChild(items);
    td.appendChild(menu);
    return td;
  }

  function attemptsRow(model) {
    const row = document.createElement('tr');
    row.className = 'release-attempts-row';
    const td = document.createElement('td');
    td.colSpan = PAGE_COLUMNS;
    const list = model.attempts.map((attempt) => {
      const accepted = attempt.accepted
        ? `<span class="rel-badge">${esc(LL.t('releases.attempts.accepted'))}</span>`
        : `<button type="button" class="btn btn-sm" data-accept="${esc(attempt.run_name)}" data-min-role="engineer">${esc(LL.t('releases.attempts.accept'))}</button>`;
      return `<li><a href="${esc(LL.reportUrl(attempt))}">${esc(attempt.run_name)}</a>`
        + ` <span class="pill ${LL.verdictClass(attempt.verdict)}">${esc(LL.verdictLabel(attempt.verdict))}</span>`
        + ` <span class="dim">${esc(attempt.created_at ? LL.formatDateTime(attempt.created_at) : '')}</span> ${accepted}</li>`;
    }).join('');
    td.innerHTML = `<div class="release-attempts"><div class="release-attempts-title">${esc(LL.t('releases.attempts.title', { service: model.service, type: LL.testTypeLabel(model.testType) }))}</div><ul>${list}</ul></div>`;
    td.querySelectorAll('[data-accept]').forEach((btn) => btn.addEventListener('click', () => {
      btn.disabled = true;
      acceptRun(btn.getAttribute('data-accept'));
    }));
    row.appendChild(td);
    return row;
  }

  function renderRows(rows) {
    const body = $('releaseRows');
    body.innerHTML = '';
    const models = rowsModel(rows);
    if (!models.length) {
      body.innerHTML = ui.emptyRow(PAGE_COLUMNS, LL.t('releases.table.empty'), { icon: 'tag' });
      return models;
    }
    models.forEach((model) => {
      const row = document.createElement('tr');
      row.className = 'data-row';
      const service = document.createElement('td');
      service.className = 'name';
      service.textContent = model.service;
      row.appendChild(service);
      const type = document.createElement('td');
      type.className = 'muted';
      type.textContent = LL.testTypeLabel(model.testType);
      row.appendChild(type);
      const run = document.createElement('td');
      run.className = 'run-name';
      run.innerHTML = runCellHtml(model);
      row.appendChild(run);
      const verdict = document.createElement('td');
      verdict.innerHTML = model.verdict
        ? `<span class="pill ${LL.verdictClass(model.verdict)}">${esc(LL.verdictLabel(model.verdict))}</span>${model.manual ? `<span class="verdict-manual-mark" title="${esc(LL.t('releases.table.manual_verdict'))}">${esc(LL.t('releases.table.manual_mark'))}</span>` : ''}`
        : '<span class="dim">—</span>';
      row.appendChild(verdict);
      if (model.available) {
        row.appendChild(deltaCell(model.cells.p95));
        row.appendChild(deltaCell(model.cells.errors));
        row.appendChild(deltaCell(model.cells.capacity, model.level));
      } else {
        const reason = document.createElement('td');
        reason.className = 'dim release-no-deltas';
        reason.colSpan = 3;
        reason.textContent = model.reason;
        row.appendChild(reason);
      }
      const previous = document.createElement('td');
      previous.innerHTML = previousCellHtml(model);
      row.appendChild(previous);
      row.appendChild(actionsCell(model));
      body.appendChild(row);
      if (page.open.has(model.key)) body.appendChild(attemptsRow(model));
    });
    return models;
  }

  function renderHeader(data) {
    const status = $('releaseStatus');
    status.className = `pill ${statusClass(data.status)}`;
    status.textContent = statusLabel(data.status);
    const tested = new Set((data.rows || []).filter((row) => row.run).map((row) => row.service)).size;
    const meta = [LL.t('releases.page.services', { tested, required: Number(data.services_required) || 0 })];
    if (data.updated_at && data.author) meta.push(LL.t('releases.page.changed_by', { author: data.author, time: LL.formatDateTime(data.updated_at) }));
    $('releaseMeta').textContent = meta.join(' · ');
    const note = $('releaseNote');
    note.hidden = !data.note;
    note.textContent = data.note || '';
    const jira = data.jira || null;
    const issues = $('jiraIssues');
    issues.hidden = !(jira && jira.issues_url);
    if (!issues.hidden) issues.href = jira.issues_url;
    const ticket = $('jiraTicket');
    ticket.hidden = !(jira && jira.ticket_url);
    if (!ticket.hidden) {
      ticket.href = jira.ticket_url;
      ticket.textContent = LL.t('releases.page.jira_ticket', { key: data.ticket });
    }
    setPublication(data.publication && data.publication.page_url ? data.publication.page_url : '');
  }

  function renderUntested(untested) {
    const section = $('untestedSection');
    const list = Array.isArray(untested) ? untested : [];
    section.hidden = !list.length;
    const launch = LL.can('engineer');
    // The launch form has no service parameter: the link opens it, the service is picked there.
    $('untested').innerHTML = list.map((service) => `<li><span class="name">${esc(service)}</span>${launch ? ` <a class="link" href="/launch">${esc(LL.t('releases.untested.launch'))}</a>` : ''}</li>`).join('');
  }

  function showProblem(text) {
    const box = $('releaseProblem');
    box.hidden = !text;
    box.innerHTML = text ? `<p>${esc(text)}</p>` : '';
  }

  async function loadRelease() {
    let data;
    try {
      data = await api('GET', dataUrl(page.name));
    } catch (e) {
      // A reload after a change that failed keeps the table on screen: the change itself was saved.
      if (page.data && e.status !== 404) {
        ui.toast(LL.t('releases.page.load_failed', { error: e.message }), { tone: 'error' });
        return;
      }
      $('releaseRows').innerHTML = '';
      showProblem(e.status === 404 ? LL.t('releases.page.not_found') : LL.t('releases.page.load_failed', { error: e.message }));
      $('releaseEdit').hidden = true;
      $('releaseConfluence').hidden = true;
      return;
    }
    showProblem('');
    page.data = data;
    renderHeader(data);
    const models = renderRows(data.rows);
    renderUntested(data.untested);
    const pairs = models.filter((model) => model.run).map((model) => ({ service: model.service, type: model.testType }));
    const services = pairs.map((pair) => pair.service).filter((service, i, all) => all.indexOf(service) === i)
      .map((service) => ({ value: service, label: service }));
    setupTrend(services, pairs);
  }

  function editDialogHtml(data) {
    return [
      '<div class="release-dialog">',
      `<label class="field"><span>${esc(LL.t('releases.edit.ticket'))}</span><input type="text" name="releaseTicket" value="${esc(data.ticket || '')}" placeholder="PROJ-123" autocomplete="off"></label>`,
      `<p class="dim">${esc(LL.t('releases.edit.ticket_hint'))}</p>`,
      `<label class="field"><span>${esc(LL.t('releases.edit.note'))}</span><textarea name="releaseNote" rows="4" maxlength="${NOTE_MAX}">${esc(data.note || '')}</textarea></label>`,
      '<div class="dialog-error" data-release-error role="alert"></div>',
      '</div>'
    ].join('');
  }

  async function openEditDialog() {
    const data = page.data || { ticket: '', note: '' };
    let busy = false;
    await ui.dialog({
      title: LL.t('releases.edit.title', { name: page.name }),
      bodyHtml: editDialogHtml(data),
      onOpen: (body) => {
        const error = body.querySelector('[data-release-error]');
        body.addEventListener('input', () => { error.textContent = ''; });
      },
      actions: [
        { label: LL.t('common.action.cancel') },
        {
          label: LL.t('common.action.save'),
          primary: true,
          onClick: async (close, body) => {
            if (busy) return;
            const ticket = (body.querySelector('input[name="releaseTicket"]').value || '').trim();
            const note = (body.querySelector('textarea[name="releaseNote"]').value || '').trim();
            const error = body.querySelector('[data-release-error]');
            if (ticket && !TICKET_RE.test(ticket)) { error.textContent = LL.t('releases.edit.ticket_invalid'); return; }
            busy = true;
            try {
              await api('PUT', dataUrl(page.name), { ticket, note });
              close(true);
              ui.toast(LL.t('releases.edit.saved'), { tone: 'ok' });
              await loadRelease();
            } catch (e) {
              error.textContent = e.message;
            } finally {
              busy = false;
            }
          }
        }
      ]
    });
  }

  function setPublication(url) {
    const link = $('releaseConfluenceLink');
    link.hidden = !url;
    if (url) link.href = url;
    $('releaseConfluence').textContent = url ? LL.t('releases.confluence.update') : LL.t('releases.confluence.add');
  }

  function jobFailure(job) {
    const headline = (job && job.message) || LL.t('releases.confluence.failed');
    return job && job.error ? `${headline}: ${job.error}` : headline;
  }

  async function publishRelease() {
    const button = $('releaseConfluence');
    const status = $('releaseConfluenceStatus');
    button.disabled = true;
    status.textContent = LL.t('releases.confluence.publishing');
    try {
      const body = await api('POST', `${releaseUrl(page.name)}/confluence`, {});
      if (!body.job_id) { status.textContent = LL.t('releases.confluence.start_error'); return; }
      const result = await ui.pollJob(body.job_id, {
        polls: PUBLISH_POLLS,
        intervalMs: PUBLISH_POLL_MS,
        onProgress: (job) => {
          const pct = Number(job.progress || 0);
          status.textContent = `${job.message || LL.t('releases.confluence.publishing')}${pct ? ` (${pct}%)` : ''}`;
        }
      });
      if (result.status === 'done') {
        setPublication(result.job.page_url || '');
        status.textContent = LL.t('releases.confluence.published');
        ui.toast(LL.t('releases.confluence.published_toast'), { tone: 'ok' });
      } else if (result.status === 'error') {
        status.textContent = jobFailure(result.job);
        ui.toast(jobFailure(result.job), { tone: 'error' });
      } else {
        status.textContent = LL.t('releases.confluence.timeout');
      }
    } catch (e) {
      status.textContent = e.message;
    } finally {
      button.disabled = false;
    }
  }

  async function initRelease() {
    page.name = releaseNameFromPath(location.pathname);
    $('releaseRows').innerHTML = ui.skeletonRows(PAGE_COLUMNS, 3);
    $('releaseEdit').addEventListener('click', openEditDialog);
    $('releaseConfluence').addEventListener('click', publishRelease);
    await loadRelease();
  }

  LL.releases = {
    _internals: {
      releaseNameFromPath, releaseUrl, parseQuery, buildQuery, statusLabel, statusClass, deltaText, deltaTone, deltaTitle,
      valueText, levelText, reasonText, rowsModel, trendModel, trendCharts, renderList, renderRows, verdictCountsHtml
    }
  };

  document.addEventListener('DOMContentLoaded', () => {
    if ($('releaseTable')) initRelease();
    else if ($('releasesTable')) initList();
  });
})();
