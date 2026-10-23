// SPDX-License-Identifier: Apache-2.0
// Archive page: runs table whose filters, sort and page live in the URL, plus multi-select for comparison.
(function () {
  const LL = window.LoadLens = window.LoadLens || {};
  const PAGE_SIZE = 20;
  const JOBS_REFRESH_MS = 5000;
  // The search applies by itself this long after the last keystroke (Enter applies at once).
  const SEARCH_DEBOUNCE_MS = 350;
  const COLUMNS = 11;
  const SORT_KEYS = ['run_name', 'service', 'test_type', 'end_time', 'report_created_at', 'verdict', 'max_rps'];
  // Dates and load levels read best newest/largest first; text columns alphabetically.
  const DESC_FIRST = new Set(['end_time', 'report_created_at', 'max_rps']);
  const DEFAULT_SORT = 'end_time';
  const MAX_PAGE = 50000; // the server caps offset at 1 000 000
  const FILTER_MAX_LENGTH = 200;

  // ---- pure helpers (LL.archive._internals) ------------------------------------------------

  function defaultDir(sort) {
    return DESC_FIRST.has(sort) ? 'desc' : 'asc';
  }

  function defaultState() {
    return { page: 1, sort: DEFAULT_SORT, dir: defaultDir(DEFAULT_SORT), q: '', service: '', verdict: '', testType: '' };
  }

  // Archive state from a query string (?q=&service=&verdict=&test_type=&sort=&dir=&page=); bad values fall back to defaults.
  function parseQuery(search) {
    const params = new URLSearchParams(search || '');
    const text = (name) => String(params.get(name) || '').trim().slice(0, FILTER_MAX_LENGTH);
    const state = defaultState();
    state.q = text('q');
    state.service = text('service');
    // A verdict code; a shared link with an old label (?verdict=Провал) selects its code.
    const verdict = LL.verdictCode(text('verdict'), null);
    if (verdict) state.verdict = verdict;
    state.testType = text('test_type').toLowerCase();
    const sort = text('sort');
    if (SORT_KEYS.includes(sort)) state.sort = sort;
    const dir = text('dir');
    state.dir = (dir === 'asc' || dir === 'desc') ? dir : defaultDir(state.sort);
    const page = Number(text('page'));
    if (Number.isInteger(page) && page >= 1 && page <= MAX_PAGE) state.page = page;
    return state;
  }

  // Query string for the address bar: only values that differ from the defaults, '' when none do.
  function buildQuery(state) {
    const params = new URLSearchParams();
    if (state.q) params.set('q', state.q);
    if (state.service) params.set('service', state.service);
    if (state.verdict) params.set('verdict', state.verdict);
    if (state.testType) params.set('test_type', state.testType);
    if (state.sort !== DEFAULT_SORT) params.set('sort', state.sort);
    if (state.dir !== defaultDir(state.sort)) params.set('dir', state.dir);
    if (state.page > 1) params.set('page', String(state.page));
    const text = params.toString();
    return text ? `?${text}` : '';
  }

  // Path of the /runs request for the current page.
  function buildRunsUrl(state, pageSize = PAGE_SIZE) {
    const params = new URLSearchParams();
    if (state.q) params.set('q', state.q);
    if (state.service) params.set('service', state.service);
    if (state.verdict) params.set('verdict', state.verdict);
    if (state.testType) params.set('test_type', state.testType);
    params.set('offset', String((Math.max(1, state.page) - 1) * pageSize));
    params.set('limit', String(pageSize));
    params.set('sort', state.sort);
    params.set('dir', state.dir);
    // The SLA n/m and p95 columns: the server reads the checks for the rows of the page only.
    params.set('include', 'sla');
    return `/runs?${params.toString()}`;
  }

  function hasFilters(state) {
    return !!(state.q || state.service || state.verdict || state.testType);
  }

  // Chips above the table, one per active filter. `labels.service(id)` and `labels.testType(key)` give display names.
  function activeChips(state, labels) {
    const names = labels || {};
    const chips = [];
    if (state.q) chips.push({ key: 'q', label: LL.t('archive.filter.search'), value: LL.t('archive.chip.query_value', { q: state.q }) });
    if (state.service) chips.push({ key: 'service', label: LL.t('archive.filter.service'), value: names.service ? names.service(state.service) : state.service });
    if (state.verdict) chips.push({ key: 'verdict', label: LL.t('archive.filter.verdict'), value: LL.verdictLabel(state.verdict) });
    if (state.testType) chips.push({ key: 'testType', label: LL.t('archive.filter.test_type'), value: names.testType ? names.testType(state.testType) : state.testType });
    return chips;
  }

  // State without one filter, back on the first page.
  function withoutFilter(state, key) {
    return { ...state, [key]: '', page: 1 };
  }

  // Sort after a click on a column header: the same column flips direction, a new one starts with its default.
  function nextSort(state, key) {
    if (state.sort === key) return { sort: key, dir: state.dir === 'asc' ? 'desc' : 'asc' };
    return { sort: key, dir: defaultDir(key) };
  }

  function runTime(run) {
    const t = Date.parse((run && (run.end_time || run.start_time || run.report_created_at)) || '');
    return Number.isFinite(t) ? t : 0;
  }

  // run_a is the newer run (the one being assessed), run_b the older baseline, as on the compare page.
  function compareTargets(runs) {
    const list = Array.isArray(runs) ? runs.filter((r) => r && r.run_name) : [];
    if (list.length !== 2) return null;
    const [first, second] = list;
    return runTime(second) > runTime(first) ? { runA: second.run_name, runB: first.run_name } : { runA: first.run_name, runB: second.run_name };
  }

  function compareUrl(runs) {
    const pair = compareTargets(runs);
    return pair ? `/compare?run_a=${encodeURIComponent(pair.runA)}&run_b=${encodeURIComponent(pair.runB)}` : '';
  }

  function selectionLabel(count) {
    return LL.t('archive.selection.count', { n: count });
  }

  // Why «Сравнить» is unavailable; '' when exactly two runs are selected.
  function selectionHint(count) {
    if (count === 2 || count <= 0) return '';
    if (count === 1) return LL.t('archive.selection.need_one_more');
    return LL.t('archive.selection.too_many');
  }

  // Digit group and decimal separators of the UI language; ru keeps the no-break space and the comma.
  const NUMBER_SEPARATORS = { ru: { group: '\u00a0', decimal: ',' }, en: { group: ',', decimal: '.' } };

  // Whole numbers from 100 up (ru «12 345», en «12,345»), one decimal below (ru «95,5», en «95.5»); '—' when unknown.
  function formatRps(value) {
    if (value === null || value === undefined || value === '' || typeof value === 'boolean') return '—';
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    const sep = NUMBER_SEPARATORS[LL.locale] || NUMBER_SEPARATORS.ru;
    if (Math.abs(n) >= 100) {
      const rounded = Math.round(n);
      const digits = String(Math.abs(rounded)).replace(/\B(?=(\d{3})+(?!\d))/g, sep.group);
      return rounded < 0 ? `-${digits}` : digits;
    }
    return (Math.round(n * 10) / 10).toFixed(1).replace(/\.0$/, '').replace('.', sep.decimal);
  }

  function sameDay(a, b) {
    return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
  }

  // Short date and time for the table in the UI language: ru «15.10 20:10», en «15/10 20:10»; the year is
  // added outside the current one; with { timeToday: true } a date of today is the time alone («20:10»).
  function shortDateTime(value, now, options) {
    const date = new Date(value);
    if (!Number.isFinite(date.getTime())) return '—';
    const today = new Date(now === undefined ? Date.now() : now);
    const time = LL.fmt.dateTime(date, { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
    if (options && options.timeToday && sameDay(date, today)) return time;
    const dayOptions = { day: '2-digit', month: '2-digit' };
    if (date.getFullYear() !== today.getFullYear()) dayOptions.year = 'numeric';
    return `${LL.fmt.date(date, dayOptions)} ${time}`;
  }

  // Short absolute date for the table, with the full date and the relative one for the tooltip.
  function timeCell(value, now, options) {
    if (!value) return { text: '—', title: '' };
    const text = shortDateTime(value, now, options);
    if (text === '—') return { text, title: '' };
    return { text, title: `${LL.formatDateTime(value)} · ${LL.relativeTime(value, now)}` };
  }

  // «2/3»: SLA criteria met of the evaluated ones (/runs?include=sla); '—' without a summary.
  function slaText(sla) {
    const total = Number(sla && sla.total);
    if (!Number.isFinite(total) || total <= 0) return '—';
    return `${Number(sla.passed) || 0}/${total}`;
  }

  // p95 of the SLA criteria in ms: whole from 10 up, grouped like the RPS column; '—' when unknown.
  function p95Text(sla) {
    const value = sla ? sla.p95_ms : null;
    if (value === null || value === undefined || value === '' || typeof value === 'boolean') return '—';
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    return formatRps(Math.abs(n) >= 10 ? Math.round(n) : n);
  }

  function pageCount(total) {
    return Math.max(1, Math.ceil((Number(total) || 0) / PAGE_SIZE));
  }

  // «страница 2 из 3» under the table.
  function pageText(page, total) {
    return LL.t('archive.pager.page', { page, pages: pageCount(total) });
  }

  // «Ступенчатый» for «Ступенчатый (max perf)»: the table column is narrow, the full label goes to the tooltip.
  function shortTypeLabel(key) {
    return String(LL.testTypeLabel(key)).replace(/\s*\([^)]*\)\s*$/, '');
  }

  function durationText(run) {
    const s = Date.parse((run && run.start_time) || '');
    const e = Date.parse((run && run.end_time) || '');
    if (!Number.isFinite(s) || !Number.isFinite(e)) return '';
    return LL.formatDuration(Math.max(0, e - s));
  }

  // Tooltip of the verdict pill: who changed it by hand, or the SLA and AI verdicts when they differ; '' otherwise.
  function verdictTitle(run) {
    const r = run || {};
    if (r.manual_verdict && r.manual_verdict.verdict) return LL.manualVerdictTitle(r.manual_verdict);
    if (r.sla_verdict && r.llm_verdict && LL.verdictCode(r.sla_verdict) !== LL.verdictCode(r.llm_verdict)) {
      return LL.t('archive.row.verdict_split', { sla: LL.verdictLabel(r.sla_verdict), llm: LL.verdictLabel(r.llm_verdict) });
    }
    return '';
  }

  // Range line under the table ("Showing 21–40 of 57") for the rows of the current page.
  function pagerText(page, total, rowsOnPage) {
    if (!total) return LL.t('archive.pager.nothing');
    const from = (page - 1) * PAGE_SIZE + 1;
    const to = Math.min(total, (page - 1) * PAGE_SIZE + rowsOnPage);
    return LL.t('archive.pager.shown', { from, to, total });
  }

  // Second line of a jobs strip row: kind, service, progress or message, time.
  function jobMeta(job) {
    const bits = [];
    if (job.kind === 'confluence') bits.push(LL.t('archive.jobs.kind_confluence'));
    if (job.kind === 'forecast_confluence') bits.push(LL.t('archive.jobs.kind_forecast_confluence'));
    if (job.service) bits.push(job.service);
    bits.push(job.status === 'running' ? `${Number(job.progress) || 0}% — ${job.message || ''}` : (job.message || ''));
    if (job.updated_at) bits.push(LL.formatDateTime(job.updated_at));
    return bits.filter(Boolean).join(' · ');
  }

  // The tab of the page from its address: '?tab=runs' opens the load test runs, anything else the reports.
  function tabFromQuery(search) {
    return new URLSearchParams(search || '').get('tab') === 'runs' ? 'runs' : 'reports';
  }

  // Id of the report a load test run produced, from its address /reports/<id>; '' without one.
  function reportIdFromUrl(url) {
    const match = /^\/reports\/([^/?#]+)$/.exec(String(url || ''));
    if (!match) return '';
    try { return decodeURIComponent(match[1]); } catch (e) { return match[1]; }
  }

  // Lookup of runs started from LoadLens: by the id of their report first (survives renaming the report),
  // then by service and name, where the newest run wins (the list comes newest first).
  function launchedIndex(runs) {
    const index = new Map();
    (Array.isArray(runs) ? runs : []).forEach((run) => {
      if (!run || !run.run_id) return;
      const reportId = reportIdFromUrl(run.report_url);
      if (reportId && !index.has(`id:${reportId}`)) index.set(`id:${reportId}`, run.run_id);
      const key = `name:${run.service || ''}|${run.run_name || ''}`;
      if (!index.has(key)) index.set(key, run.run_id);
    });
    return index;
  }

  function launchedFor(index, row) {
    if (!index || !row) return null;
    return (row.run_id && index.get(`id:${row.run_id}`)) || index.get(`name:${row.service || ''}|${row.run_name || ''}`) || null;
  }

  LL.archive = LL.archive || {};
  LL.archive._internals = {
    PAGE_SIZE, SORT_KEYS, defaultState, defaultDir, parseQuery, buildQuery, buildRunsUrl, hasFilters, activeChips,
    withoutFilter, nextSort, compareTargets, compareUrl, selectionLabel, selectionHint, formatRps, timeCell, durationText,
    shortTypeLabel, verdictTitle, pagerText, jobMeta, shortDateTime, slaText, p95Text, pageCount, pageText,
    tabFromQuery, reportIdFromUrl, launchedIndex, launchedFor,
  };

  // ---- page --------------------------------------------------------------------------------

  const $ = (id) => document.getElementById(id);
  const esc = (value) => LL.ui.escapeHtml(value);
  const state = { ...defaultState(), total: 0 };
  // Selected runs survive paging and filter changes: run_name -> run row.
  const selected = new Map();
  let jobsTimer = null;
  let searchTimer = 0;
  let loadSeq = 0;
  let columnLabels = [];
  // Runs started from the Run test page (launchedIndex), engineers only.
  let launchedRuns = null;
  // 'reports' or 'runs'; only the active tab writes the address and refreshes in the background.
  let activeTab = 'reports';
  let reportsStarted = false;

  // ---- filters ----

  function ensureOption(select, value, label) {
    if (!select || !value) return;
    if (Array.from(select.options).some((o) => o.value === value)) return;
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label || value;
    select.appendChild(opt);
  }

  function fillStaticOptions() {
    LL.VERDICTS.forEach((v) => ensureOption($('verdictFilter'), v, LL.verdictLabel(v)));
    Object.keys(LL.TEST_TYPE_LABELS).forEach((key) => ensureOption($('testTypeFilter'), key, LL.testTypeLabel(key)));
    // Values from a shared link that the lists do not offer (yet) still need an option to show as selected.
    ensureOption($('serviceFilter'), state.service, state.service);
    ensureOption($('testTypeFilter'), state.testType, LL.testTypeLabel(state.testType));
  }

  async function loadServiceOptions() {
    try {
      const resp = await fetch('/services');
      const payload = await resp.json();
      const services = Array.isArray(payload && payload.services) ? payload.services : [];
      const sel = $('serviceFilter');
      services.forEach((svc) => {
        const id = typeof svc === 'string' ? svc : String((svc && svc.id) || '');
        if (!id) return;
        const title = (svc && svc.title) || id;
        const existing = Array.from(sel.options).find((o) => o.value === id);
        if (existing) existing.textContent = title;
        else ensureOption(sel, id, title);
      });
      sel.value = state.service;
      renderChips();
    } catch (e) {
      // the service filter keeps «Все» and the value from the link
    }
  }

  function applyStateToControls() {
    $('searchInput').value = state.q;
    $('serviceFilter').value = state.service;
    $('verdictFilter').value = state.verdict;
    $('testTypeFilter').value = state.testType;
  }

  function readFilters() {
    state.q = String($('searchInput').value || '').trim().slice(0, FILTER_MAX_LENGTH);
    state.service = $('serviceFilter').value;
    state.verdict = $('verdictFilter').value;
    state.testType = $('testTypeFilter').value;
    state.page = 1;
  }

  function resetFilters() {
    Object.assign(state, { q: '', service: '', verdict: '', testType: '', page: 1 });
    applyStateToControls();
  }

  function optionLabel(selectId, value) {
    const opt = Array.from($(selectId).options).find((o) => o.value === value);
    return opt ? opt.textContent : value;
  }

  function renderChips() {
    const box = $('filterChips');
    if (!box) return;
    const chips = activeChips(state, {
      service: (id) => optionLabel('serviceFilter', id),
      testType: (key) => optionLabel('testTypeFilter', key),
    });
    box.innerHTML = '';
    chips.forEach((chip, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'filter-chip';
      btn.setAttribute('aria-label', LL.t('archive.chip.remove_label', { label: chip.label, value: chip.value }));
      btn.title = LL.t('archive.chip.remove');
      btn.innerHTML = `<span class="chip-key">${esc(chip.label)}:</span> <span class="chip-value">${esc(chip.value)}</span><svg class="icon" aria-hidden="true"><use href="#i-x"/></svg>`;
      btn.addEventListener('click', () => {
        Object.assign(state, withoutFilter(state, chip.key));
        applyStateToControls();
        loadPage();
        // The clicked chip is gone; keep keyboard focus nearby.
        const rest = box.querySelectorAll('.filter-chip');
        (rest[Math.min(idx, rest.length - 1)] || $('searchInput')).focus();
      });
      box.appendChild(btn);
    });
    if (chips.length > 1) {
      const all = document.createElement('button');
      all.type = 'button';
      all.className = 'btn btn-ghost btn-sm';
      all.textContent = LL.t('archive.chip.reset_all');
      all.addEventListener('click', () => { resetFilters(); loadPage(); $('searchInput').focus(); });
      box.appendChild(all);
    }
    box.hidden = !chips.length;
  }

  function syncUrl() {
    if (activeTab !== 'reports') return;
    try {
      history.replaceState(history.state, '', `${location.pathname}${buildQuery(state)}${location.hash}`);
    } catch (e) { /* the address bar stays as it was */ }
  }

  // ---- selection ----

  function renderSelectionBar() {
    const bar = $('selectionBar');
    if (!bar) return;
    const count = selected.size;
    bar.hidden = count === 0;
    // «выбрано 2» with the number in bold.
    $('selectionCount').innerHTML = LL.tHtml('archive.selection.count', { n: LL.html(`<b>${count}</b>`) });
    // Two runs read as «A против B» in the order of the compare page (newer first); otherwise a list.
    const box = $('selectionNames');
    const pair = compareTargets(Array.from(selected.values()));
    box.textContent = '';
    if (pair) {
      const versus = document.createElement('span');
      versus.className = 'selection-versus';
      versus.textContent = ` ${LL.t('archive.selection.versus')} `;
      box.append(pair.runA, versus, pair.runB);
      box.title = `${pair.runA} ${LL.t('archive.selection.versus')} ${pair.runB}`;
    } else {
      box.textContent = Array.from(selected.keys()).join(', ');
      box.title = box.textContent;
    }
    $('selectionHint').textContent = selectionHint(count);
    $('selectionCompare').setAttribute('aria-disabled', count === 2 ? 'false' : 'true');
  }

  function setSelected(run, on) {
    if (on) selected.set(run.run_name, run);
    else selected.delete(run.run_name);
    syncSelection();
  }

  // Checkboxes, row highlight and the bar follow the `selected` map.
  function syncSelection() {
    document.querySelectorAll('#tbody tr.data-row').forEach((tr) => {
      const isOn = selected.has(tr.dataset.runName);
      tr.classList.toggle('selected', isOn);
      const box = tr.querySelector('input.row-select');
      if (box) box.checked = isOn;
    });
    renderSelectionBar();
  }

  function compareSelected() {
    const url = compareUrl(Array.from(selected.values()));
    if (!url) {
      LL.ui.toast(selectionHint(selected.size) || LL.t('archive.selection.need_two'), { tone: 'info' });
      return;
    }
    location.href = url;
  }

  // ---- table ----

  function cell(text, className) {
    const td = document.createElement('td');
    if (className) td.className = className;
    td.textContent = text;
    return td;
  }

  // A date cell: the short date (full and relative in the tooltip) and, after it, a dim note (the duration).
  function timeTd(value, now, className, note, options) {
    const td = document.createElement('td');
    td.className = className || '';
    const info = timeCell(value, now, options);
    if (value && info.title) {
      const time = document.createElement('time');
      time.dateTime = String(value);
      time.title = info.title;
      time.textContent = info.text;
      td.appendChild(time);
    } else {
      td.textContent = info.text;
    }
    if (note) {
      const small = document.createElement('span');
      small.className = 'cell-note';
      small.textContent = note;
      td.appendChild(small);
    }
    return td;
  }

  function menuButton(label, onClick, className) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = label;
    if (className) btn.className = className;
    btn.addEventListener('click', (e) => { e.stopPropagation(); onClick(btn); });
    return btn;
  }

  function buildActionsCell(run) {
    const td = document.createElement('td');
    td.className = 'actions';
    const inner = document.createElement('div');
    inner.className = 'actions-inner';

    const menu = document.createElement('details');
    menu.className = 'menu';
    menu.addEventListener('toggle', () => {
      const wrap = menu.closest('.table-wrap');
      if (wrap) wrap.classList.toggle('menu-open', !!document.querySelector('details.menu[open]'));
    });
    const summary = document.createElement('summary');
    summary.textContent = '⋯';
    summary.title = LL.t('archive.row.actions');
    summary.setAttribute('aria-label', LL.t('archive.row.actions_for', { name: run.run_name }));
    summary.addEventListener('click', (e) => e.stopPropagation());
    const items = document.createElement('div');
    items.className = 'menu-items';
    items.addEventListener('click', (e) => e.stopPropagation());

    items.appendChild(menuButton(LL.t('archive.row.compare_with'), () => {
      location.href = '/compare?run_a=' + encodeURIComponent(run.run_name);
    }));
    const launched = launchedRun(run);
    if (launched && LL.can('engineer')) {
      items.appendChild(menuButton(LL.t('archive.row.relaunch'), () => {
        location.href = '/launch?from=' + encodeURIComponent(launched);
      }));
    }
    if (LL.can('engineer')) {
      items.appendChild(menuButton(LL.t('archive.row.rename'), async () => {
        menu.removeAttribute('open');
        await renameRun(run);
      }));
    }
    items.appendChild(menuButton(LL.t('archive.row.copy_link'), async (btn) => {
      try {
        await navigator.clipboard.writeText(location.origin + LL.reportUrl(run));
        btn.textContent = LL.t('archive.row.link_copied');
        btn.classList.add('confirmed');
        setTimeout(() => { btn.textContent = LL.t('archive.row.copy_link'); btn.classList.remove('confirmed'); menu.removeAttribute('open'); }, 1200);
      } catch (err) {
        btn.textContent = LL.t('archive.row.copy_failed');
        setTimeout(() => { btn.textContent = LL.t('archive.row.copy_link'); }, 1500);
      }
    }));
    if (LL.can('admin')) {
      items.appendChild(menuButton(LL.t('archive.row.delete'), async () => {
        menu.removeAttribute('open');
        await deleteRun(run);
      }, 'danger'));
    }

    menu.appendChild(summary);
    menu.appendChild(items);
    inner.appendChild(menu);
    td.appendChild(inner);
    return td;
  }

  async function renameRun(run) {
    const next = await LL.ui.prompt({
      title: LL.t('archive.rename.title'),
      label: LL.t('archive.rename.label'),
      value: run.run_name,
      validate: (v) => (!v.trim() ? LL.t('archive.rename.empty') : (/[\\/\n\r\t]/.test(v) ? LL.t('archive.rename.bad_chars') : ''))
    });
    if (next == null) return;
    const name = String(next).trim();
    if (!name || name === run.run_name) return;
    try {
      const resp = await fetch('/runs/' + encodeURIComponent(run.run_name), {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ new_run_name: name, service: run.service || '' })
      });
      const j = await resp.json();
      if (resp.ok) {
        if (selected.has(run.run_name)) { selected.delete(run.run_name); selected.set(name, { ...run, run_name: name }); }
        LL.ui.toast(LL.t('archive.rename.done', { name }), { tone: 'ok' });
        await loadPage();
      } else LL.ui.toast(j.error || LL.t('archive.rename.failed'), { tone: 'error' });
    } catch (err) {
      LL.ui.toast(LL.t('archive.rename.failed_detail', { error: err.message }), { tone: 'error' });
    }
  }

  async function deleteRun(run) {
    const ok = await LL.ui.confirm({
      title: LL.t('archive.delete.title'),
      message: LL.t('archive.delete.message', { name: run.run_name }),
      confirmText: LL.t('common.action.delete'),
      danger: true
    });
    if (!ok) return;
    try {
      const resp = await fetch('/runs/' + encodeURIComponent(run.run_name), { method: 'DELETE' });
      const j = await resp.json();
      if (resp.ok) {
        selected.delete(run.run_name);
        LL.ui.toast(LL.t('archive.delete.done'), { tone: 'ok' });
        await loadPage();
      } else LL.ui.toast(j.error || LL.t('archive.delete.failed'), { tone: 'error' });
    } catch (err) {
      LL.ui.toast(LL.t('archive.delete.failed_detail', { error: err.message }), { tone: 'error' });
    }
  }

  function buildSelectCell(run) {
    const td = document.createElement('td');
    td.className = 'select-cell';
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.className = 'row-select';
    box.checked = selected.has(run.run_name);
    box.setAttribute('aria-label', LL.t('archive.row.select', { name: run.run_name }));
    box.addEventListener('change', () => setSelected(run, box.checked));
    // The whole cell is the hit area; neither the cell nor the box opens the report.
    td.addEventListener('click', (e) => {
      e.stopPropagation();
      if (e.target !== box) box.click();
    });
    td.appendChild(box);
    return td;
  }

  function renderEmpty(text, actionHtml) {
    $('tbody').innerHTML = LL.ui.emptyRow(COLUMNS, text, { actionHtml });
  }

  function renderRows(rows) {
    const tbody = $('tbody');
    tbody.innerHTML = '';
    if (!rows.length) {
      if (hasFilters(state)) {
        renderEmpty(LL.t('archive.empty.filtered'), `<button type="button" class="btn btn-sm" id="emptyReset">${LL.tHtml('archive.empty.reset_filters')}</button>`);
        const reset = $('emptyReset');
        if (reset) reset.addEventListener('click', () => { resetFilters(); loadPage(); });
      } else {
        renderEmpty(LL.t('archive.empty.none'), LL.can('engineer') ? `<a class="btn btn-sm" href="/new">${LL.tHtml('archive.empty.create')}</a>` : '');
      }
      return;
    }
    const now = Date.now();
    rows.forEach((r) => {
      const tr = document.createElement('tr');
      tr.className = 'data-row';
      tr.dataset.runName = r.run_name;
      tr.classList.toggle('selected', selected.has(r.run_name));
      tr.appendChild(buildSelectCell(r));
      // The name is a real link: keyboard focus, middle click and «open in new tab» work.
      const nameCell = document.createElement('td');
      nameCell.className = 'run-name';
      nameCell.title = r.run_name;
      const link = document.createElement('a');
      link.className = 'run-link';
      link.href = LL.reportUrl(r);
      link.textContent = r.run_name;
      nameCell.appendChild(link);
      if (launchedRun(r)) {
        const badge = document.createElement('span');
        badge.className = 'pill na launched-badge';
        badge.textContent = LL.t('archive.row.launched');
        nameCell.appendChild(badge);
      }
      tr.appendChild(nameCell);
      const serviceTd = cell(r.service || '—', 'nowrap service-cell');
      serviceTd.title = r.service || '';
      tr.appendChild(serviceTd);
      const typeTd = cell(shortTypeLabel(r.test_type_key || r.test_type), 'muted nowrap');
      typeTd.title = LL.testTypeLabel(r.test_type_key || r.test_type);
      tr.appendChild(typeTd);
      tr.appendChild(timeTd(r.end_time || r.start_time, now, 'm nowrap date-cell', durationText(r)));
      const verdictTd = document.createElement('td');
      verdictTd.className = 'nowrap';
      const pill = document.createElement('span');
      pill.className = `pill ${LL.verdictClass(r.verdict)}`;
      pill.textContent = LL.verdictLabel(r.verdict);
      const title = verdictTitle(r);
      if (title) pill.title = title;
      verdictTd.appendChild(pill);
      if (r.manual_verdict && r.manual_verdict.verdict) {
        const mark = document.createElement('span');
        mark.className = 'verdict-manual-mark';
        mark.textContent = LL.t('archive.row.manual');
        mark.title = pill.title;
        verdictTd.appendChild(mark);
      }
      tr.appendChild(verdictTd);
      const slaTd = cell(slaText(r.sla), 'n');
      if (r.sla && r.sla.total) slaTd.title = LL.t('archive.row.sla_title', { passed: r.sla.passed, total: r.sla.total });
      tr.appendChild(slaTd);
      tr.appendChild(cell(formatRps(r.max_rps), 'n'));
      const p95Td = cell(p95Text(r.sla), 'n');
      if (r.sla && r.sla.p95_passed === false && r.sla.p95_ms != null) {
        p95Td.classList.add('is-over');
        p95Td.title = LL.t('archive.row.p95_over');
      }
      tr.appendChild(p95Td);
      tr.appendChild(timeTd(r.report_created_at, now, 'n nowrap', '', { timeToday: true }));
      tr.appendChild(buildActionsCell(r));
      // Phone card layout (stage 5) reads the column name from data-label.
      Array.from(tr.children).forEach((td, i) => { if (columnLabels[i]) td.setAttribute('data-label', columnLabels[i]); });
      // The rest of the row opens the report too; controls inside it handle their own clicks.
      tr.addEventListener('click', (e) => {
        if (e.target.closest && e.target.closest('a, button, input, details, summary')) return;
        location.href = LL.reportUrl(r);
      });
      tbody.appendChild(tr);
    });
  }

  // The range next to the filters («показано 1–20 из 46») and the text pager under the table (hidden on one page).
  function renderPager(rowsOnPage) {
    const to = Math.min(state.total, (state.page - 1) * PAGE_SIZE + rowsOnPage);
    $('archiveCounter').textContent = pagerText(state.page, state.total, rowsOnPage);
    $('pageInfo').textContent = pageText(state.page, state.total);
    $('prevBtn').disabled = state.page <= 1;
    $('nextBtn').disabled = to >= state.total;
    $('archivePager').hidden = pageCount(state.total) <= 1 && state.page <= 1;
  }

  function renderSortIndicators() {
    document.querySelectorAll('th[data-sort]').forEach((th) => {
      const on = th.getAttribute('data-sort') === state.sort;
      th.classList.toggle('sorted-asc', on && state.dir === 'asc');
      th.classList.toggle('sorted-desc', on && state.dir === 'desc');
      if (on) th.setAttribute('aria-sort', state.dir === 'asc' ? 'ascending' : 'descending');
      else th.removeAttribute('aria-sort');
    });
  }

  function renderBreadcrumbs() {
    const box = $('archiveBreadcrumbs');
    if (!box) return;
    if (!state.service) { box.hidden = true; box.innerHTML = ''; return; }
    box.innerHTML = `<a href="/reports">${LL.tHtml('archive.breadcrumbs.root')}</a><span class="sep" aria-hidden="true">›</span><span class="current" aria-current="page">${esc(optionLabel('serviceFilter', state.service))}</span>`;
    box.hidden = false;
  }

  function launchedRun(run) {
    return launchedFor(launchedRuns, run);
  }

  async function loadLaunchedRuns() {
    if (launchedRuns || !LL.can('engineer')) return;
    launchedRuns = new Map();
    try {
      const resp = await fetch('/load_tests/runs?limit=200');
      if (!resp.ok) return;
      const data = await resp.json();
      launchedRuns = launchedIndex(data.runs);
    } catch (e) { /* the archive works without the launch history */ }
  }

  // quiet: background refresh (jobs strip) keeps the rows on screen instead of flashing skeletons.
  async function loadPage(options) {
    const quiet = !!(options && options.quiet);
    const seq = ++loadSeq;
    syncUrl();
    renderChips();
    renderSortIndicators();
    renderBreadcrumbs();
    if (!quiet) $('tbody').innerHTML = LL.ui.skeletonRows(COLUMNS, 5);
    await loadLaunchedRuns();
    let resp;
    let rows;
    try {
      resp = await fetch(buildRunsUrl(state));
      rows = await resp.json();
    } catch (e) {
      if (seq !== loadSeq) return;
      renderLoadError(LL.t('archive.load.failed_detail', { error: e.message }));
      return;
    }
    // A newer request (filter or page change) superseded this one.
    if (seq !== loadSeq) return;
    if (!resp.ok || !Array.isArray(rows)) {
      renderLoadError((rows && rows.error) ? LL.t('archive.load.error_detail', { error: rows.error }) : LL.t('archive.load.error'));
      return;
    }
    state.total = Number(resp.headers.get('X-Total-Count') || rows.length);
    // A shared link may point past the last page after runs were deleted.
    if (!rows.length && state.total > 0 && state.page > 1) {
      state.page = Math.max(1, Math.ceil(state.total / PAGE_SIZE));
      await loadPage(options);
      return;
    }
    renderRows(rows);
    renderPager(rows.length);
  }

  function renderLoadError(text) {
    renderEmpty(text, `<button type="button" class="btn btn-sm" id="retryLoad">${LL.tHtml('archive.load.retry')}</button>`);
    const retry = $('retryLoad');
    if (retry) retry.addEventListener('click', () => loadPage());
    $('pageInfo').textContent = '';
    $('archiveCounter').textContent = '';
    $('prevBtn').disabled = true;
    $('nextBtn').disabled = true;
  }

  // ---- background jobs strip ----

  function renderJobRow(job) {
    const row = document.createElement('div');
    row.className = 'job-row';
    const pill = document.createElement('span');
    pill.className = `pill ${job.status}`;
    pill.textContent = job.status === 'running' ? LL.t('archive.jobs.running') : LL.t('archive.jobs.error');
    const info = document.createElement('div');
    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = job.run_name || LL.t('archive.jobs.untitled');
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = jobMeta(job);
    info.appendChild(name);
    info.appendChild(meta);
    const link = document.createElement('a');
    if (job.kind === 'report') {
      link.href = `/new?job=${encodeURIComponent(job.job_id)}`;
      link.textContent = job.status === 'running' ? LL.t('archive.jobs.follow') : LL.t('archive.jobs.details');
      link.setAttribute('data-min-role', 'engineer');
    } else if (job.kind === 'forecast_confluence' && job.report_url) {
      link.href = job.report_url;
      link.textContent = LL.t('archive.jobs.open_forecast');
    } else if (job.page_url && String(job.page_url).startsWith('/reports/')) {
      link.href = job.page_url;
      link.textContent = LL.t('archive.jobs.open_report');
    } else if (job.run_name) {
      link.href = LL.reportUrl({ run_name: job.run_name });
      link.textContent = LL.t('archive.jobs.open_report');
    }
    const actions = document.createElement('span');
    actions.className = 'job-actions';
    actions.appendChild(link);
    const dismiss = LL.jobDismissButton(job, loadJobsStrip);
    if (dismiss) actions.appendChild(dismiss);
    row.appendChild(pill);
    row.appendChild(info);
    row.appendChild(actions);
    return row;
  }

  // Finished rows currently shown: "Очистить завершённые" hides exactly these.
  let finishedJobIds = [];

  async function clearFinishedJobs() {
    const btn = $('jobsStripClear');
    btn.disabled = true;
    await LL.dismissJobs(finishedJobIds);
    btn.disabled = false;
    loadJobsStrip();
  }

  async function loadJobsStrip() {
    const strip = $('jobsStrip');
    const list = $('jobsStripList');
    if (!strip || !list) return;
    try {
      const resp = await fetch('/jobs?status=running,error&limit=10');
      if (!resp.ok) return;
      const payload = await resp.json();
      const jobs = Array.isArray(payload && payload.jobs) ? payload.jobs : [];
      list.innerHTML = '';
      finishedJobIds = jobs.filter((j) => j.status !== 'running').map((j) => j.job_id);
      $('jobsStripClear').hidden = !finishedJobIds.length;
      if (!jobs.length) {
        strip.style.display = 'none';
        if (jobsTimer) { clearInterval(jobsTimer); jobsTimer = null; }
        return;
      }
      jobs.forEach((job) => list.appendChild(renderJobRow(job)));
      const running = jobs.filter((j) => j.status === 'running').length;
      $('jobsStripNote').textContent = running ? LL.t('archive.jobs.running_count', { n: running }) : '';
      strip.style.display = '';
      if (running && !jobsTimer) {
        jobsTimer = setInterval(async () => {
          if (activeTab !== 'reports') return;
          await loadJobsStrip();
          await loadPage({ quiet: true });
        }, JOBS_REFRESH_MS);
      }
      if (!running && jobsTimer) { clearInterval(jobsTimer); jobsTimer = null; }
    } catch (e) {
      // strip is auxiliary
    }
  }

  // ---- wiring ----

  function sortBy(key) {
    Object.assign(state, nextSort(state, key), { page: 1 });
    loadPage();
  }

  // The search applies on Enter at once, or by itself shortly after typing stops.
  function applySearch() {
    window.clearTimeout(searchTimer);
    const q = String($('searchInput').value || '').trim().slice(0, FILTER_MAX_LENGTH);
    if (q === state.q) return;
    readFilters();
    loadPage();
  }

  function wireControls() {
    $('searchInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); applySearch(); } });
    $('searchInput').addEventListener('input', () => {
      window.clearTimeout(searchTimer);
      searchTimer = window.setTimeout(applySearch, SEARCH_DEBOUNCE_MS);
    });
    ['serviceFilter', 'verdictFilter', 'testTypeFilter'].forEach((id) => {
      $(id).addEventListener('change', () => { readFilters(); loadPage(); });
    });
    $('prevBtn').addEventListener('click', () => { if (state.page > 1) { state.page -= 1; loadPage(); } });
    $('nextBtn').addEventListener('click', () => { state.page += 1; loadPage(); });
    document.querySelectorAll('th[data-sort]').forEach((th) => {
      const key = th.getAttribute('data-sort');
      th.addEventListener('click', () => sortBy(key));
      th.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sortBy(key); }
      });
    });
    $('selectionCompare').addEventListener('click', compareSelected);
    $('selectionClear').addEventListener('click', () => {
      selected.clear();
      syncSelection();
      $('searchInput').focus();
    });
    // Close any open row menu when clicking elsewhere.
    document.addEventListener('click', (e) => {
      document.querySelectorAll('details.menu[open]').forEach((menu) => {
        if (!menu.contains(e.target)) menu.removeAttribute('open');
      });
    });
  }

  // ---- tabs ----

  // The reports tab starts on its first showing; later showings refresh it quietly (runs may have taken
  // their reports with them) and put its filters back into the address.
  async function showReports() {
    if (reportsStarted) {
      syncUrl();
      await loadPage({ quiet: true });
      loadJobsStrip();
      return;
    }
    reportsStarted = true;
    const services = loadServiceOptions();
    await loadPage();
    await services;
    $('jobsStripClear').addEventListener('click', clearFinishedJobs);
    loadJobsStrip();
  }

  function selectTab(tab, options) {
    const next = tab === 'runs' && LL.archiveRuns ? 'runs' : 'reports';
    activeTab = next;
    [['reports', 'tabReports', 'reportsPanel'], ['runs', 'tabRuns', 'runsPanel']].forEach(([name, tabId, panelId]) => {
      const on = name === next;
      const btn = $(tabId);
      btn.classList.toggle('active', on);
      btn.setAttribute('aria-selected', on ? 'true' : 'false');
      btn.tabIndex = on ? 0 : -1;
      $(panelId).hidden = !on;
    });
    const subtitle = document.querySelector('.page-header .page-subtitle');
    if (subtitle) subtitle.textContent = LL.t(next === 'runs' ? 'archive.runs.subtitle' : 'archive.subtitle');
    if (next === 'runs') {
      $('archiveBreadcrumbs').hidden = true;
      LL.archiveRuns.show();
    } else {
      showReports();
    }
    if (options && options.focus) $(next === 'runs' ? 'tabRuns' : 'tabReports').focus();
  }

  function wireTabs() {
    const tabs = [$('tabReports'), $('tabRuns')];
    tabs.forEach((btn) => btn.addEventListener('click', () => {
      if (btn.dataset.tab !== activeTab) selectTab(btn.dataset.tab);
    }));
    // Arrow keys move between the tabs (WAI-ARIA tabs pattern).
    tabs.forEach((btn, i) => btn.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      e.preventDefault();
      const other = tabs[(i + 1) % tabs.length];
      selectTab(other.dataset.tab, { focus: true });
    }));
  }

  document.addEventListener('DOMContentLoaded', async () => {
    if (!$('runsTable')) return;
    // The filters apply as they change; the form itself never submits (CSP forbids inline handlers).
    const filtersForm = $('filtersForm');
    if (filtersForm) filtersForm.addEventListener('submit', (e) => e.preventDefault());
    columnLabels = Array.from(document.querySelectorAll('#runsTable thead th')).map((th) => th.textContent.trim());
    // Filters, sort and page come from the address (?service= also arrives from report breadcrumbs).
    const tab = tabFromQuery(location.search);
    if (tab === 'reports') Object.assign(state, parseQuery(location.search));
    fillStaticOptions();
    applyStateToControls();
    wireControls();
    renderSelectionBar();
    if ($('tabRuns')) wireTabs();
    selectTab(tab);
  });
})();
