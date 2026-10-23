// SPDX-License-Identifier: Apache-2.0
// Archive page, Runs tab: load test runs started from LoadLens (GET /load_tests/history), deleted one by one
// or by checkboxes, optionally with their reports. archive.js owns the tabs and calls LL.archiveRuns.show().
(function () {
  const LL = window.LoadLens = window.LoadLens || {};
  const PAGE_SIZE = 20;
  const FILTER_MAX_LENGTH = 200;
  const SEARCH_DEBOUNCE_MS = 350;
  // Active runs change status on their own: the tab refreshes while one is on screen.
  const ACTIVE_REFRESH_MS = 10000;
  const MAX_PAGE = 50000;
  const STATUS_FILTERS = ['active', 'done', 'failed', 'cancelled'];
  // Same tones as the runs list of the Run test page.
  const STATUS_TONE = {
    starting: 'running', running: 'running', stopping: 'warn', draining: 'running', reporting: 'running',
    done: 'done', failed: 'fail', cancelled: 'na',
  };

  // ---- pure helpers (LL.archiveRuns._internals) ----------------------------------------------

  function defaultState() {
    return { page: 1, q: '', service: '', status: '' };
  }

  // Tab state from ?tab=runs&q=&service=&status=&page=; bad values fall back to defaults.
  function parseQuery(search) {
    const params = new URLSearchParams(search || '');
    const text = (name) => String(params.get(name) || '').trim().slice(0, FILTER_MAX_LENGTH);
    const state = defaultState();
    state.q = text('q');
    state.service = text('service');
    const status = text('status');
    if (STATUS_FILTERS.includes(status)) state.status = status;
    const page = Number(text('page'));
    if (Number.isInteger(page) && page >= 1 && page <= MAX_PAGE) state.page = page;
    return state;
  }

  function buildQuery(state) {
    const params = new URLSearchParams();
    params.set('tab', 'runs');
    if (state.q) params.set('q', state.q);
    if (state.service) params.set('service', state.service);
    if (state.status) params.set('status', state.status);
    if (state.page > 1) params.set('page', String(state.page));
    return `?${params.toString()}`;
  }

  function buildHistoryUrl(state, pageSize = PAGE_SIZE) {
    const params = new URLSearchParams();
    if (state.q) params.set('q', state.q);
    if (state.service) params.set('service', state.service);
    if (state.status) params.set('status', state.status);
    params.set('offset', String((Math.max(1, state.page) - 1) * pageSize));
    params.set('limit', String(pageSize));
    return `/load_tests/history?${params.toString()}`;
  }

  function hasFilters(state) {
    return !!(state.q || state.service || state.status);
  }

  function statusTone(status) {
    return STATUS_TONE[status] || 'na';
  }

  const STATUS_FILTER_KEYS = {
    active: 'archive.runs.filter.active',
    done: 'common.lt_run_status.done',
    failed: 'common.lt_run_status.failed',
    cancelled: 'common.lt_run_status.cancelled',
  };

  function statusFilterLabel(code) {
    return STATUS_FILTER_KEYS[code] ? LL.t(STATUS_FILTER_KEYS[code]) : String(code || '');
  }

  // «2 инст.» and, as a note, the tools: «k6, JMeter».
  function instancesText(run) {
    const count = Number(run && run.instances) || 0;
    const tools = Array.isArray(run && run.tools) ? run.tools.filter(Boolean) : [];
    return { text: LL.t('archive.runs.instances', { n: count }), note: tools.join(', ') };
  }

  // Run length from its start to its end, or to now while it goes on; '' before it starts.
  function runDuration(run, now) {
    const start = Date.parse((run && run.started_at) || '');
    if (!Number.isFinite(start)) return '';
    const end = run.finished_at ? Date.parse(run.finished_at) : (run.active ? (now === undefined ? Date.now() : now) : NaN);
    if (!Number.isFinite(end)) return '';
    return LL.formatDuration(Math.max(0, end - start));
  }

  // The report cell: a checked report links by id with its current name; an unchecked one (the report
  // database did not answer) keeps the stored link; no report gives null.
  function reportLink(run) {
    const report = run && run.report;
    if (!report || !report.url) return null;
    return {
      url: report.url,
      text: report.run_name || LL.t('archive.runs.row.report_link'),
      title: run.has_report === null ? LL.t('archive.runs.row.report_unchecked') : '',
    };
  }

  // What the delete dialog says and offers for these runs.
  function deletePlan(runs, isAdmin) {
    const list = Array.isArray(runs) ? runs : [];
    const withReports = list.filter((r) => r && r.has_report === true).length;
    return {
      title: list.length === 1 ? LL.t('archive.runs.delete.title_one') : LL.t('archive.runs.delete.title_many'),
      message: list.length === 1
        ? LL.t('archive.runs.delete.message_one', { name: list[0].run_name })
        : LL.t('archive.runs.delete.message_many', { n: list.length }),
      offerReports: !!isAdmin && withReports > 0,
      reports: withReports,
    };
  }

  // Toast after a delete: what went, what stayed and why (the first reason is enough).
  function deleteSummary(result) {
    const r = result || {};
    const parts = [LL.t('archive.runs.delete.done', { n: (r.deleted || []).length })];
    if ((r.reports_deleted || []).length) parts.push(LL.t('archive.runs.delete.reports_done', { n: r.reports_deleted.length }));
    const skipped = r.skipped || [];
    if (skipped.length) parts.push(LL.t('archive.runs.delete.skipped', { n: skipped.length, reason: skipped[0].message || '' }));
    const failed = r.reports_failed || [];
    if (failed.length) parts.push(failed[0].message || '');
    const tone = skipped.length || failed.length ? (r.deleted && r.deleted.length ? 'info' : 'error') : 'ok';
    return { text: parts.filter(Boolean).join(' · '), tone };
  }

  LL.archiveRuns = LL.archiveRuns || {};
  LL.archiveRuns._internals = {
    PAGE_SIZE, STATUS_FILTERS, defaultState, parseQuery, buildQuery, buildHistoryUrl, hasFilters, statusTone,
    statusFilterLabel, instancesText, runDuration, reportLink, deletePlan, deleteSummary,
  };

  // ---- page --------------------------------------------------------------------------------

  const $ = (id) => document.getElementById(id);
  const esc = (value) => LL.ui.escapeHtml(value);
  const helpers = () => (LL.archive && LL.archive._internals) || {};
  const state = { ...defaultState(), total: 0 };
  // Selected runs survive paging and filter changes: run_id -> run row.
  const selected = new Map();
  let started = false;
  let loadSeq = 0;
  let searchTimer = 0;
  let refreshTimer = 0;
  let columnLabels = [];
  // Columns of the table: viewers delete nothing, so they get no selection column.
  let columns = 9;

  function isShown() {
    const panel = $('runsPanel');
    return !!panel && !panel.hidden;
  }

  function syncUrl() {
    if (!isShown()) return;
    try {
      history.replaceState(history.state, '', `${location.pathname}${buildQuery(state)}${location.hash}`);
    } catch (e) { /* the address bar stays as it was */ }
  }

  // ---- filters ----

  function addOption(select, value, label) {
    if (!select || !value || Array.from(select.options).some((o) => o.value === value)) return;
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label || value;
    select.appendChild(opt);
  }

  async function loadServiceOptions() {
    const sel = $('ltServiceFilter');
    addOption(sel, state.service, state.service);
    try {
      const resp = await fetch('/services');
      const payload = await resp.json();
      (Array.isArray(payload && payload.services) ? payload.services : []).forEach((svc) => {
        const id = typeof svc === 'string' ? svc : String((svc && svc.id) || '');
        const existing = Array.from(sel.options).find((o) => o.value === id);
        if (existing) existing.textContent = (svc && svc.title) || id;
        else addOption(sel, id, (svc && svc.title) || id);
      });
    } catch (e) { /* the filter keeps «all» and the value from the link */ }
    sel.value = state.service;
  }

  function applyStateToControls() {
    $('ltSearchInput').value = state.q;
    $('ltServiceFilter').value = state.service;
    $('ltStatusFilter').value = state.status;
  }

  function readFilters() {
    state.q = String($('ltSearchInput').value || '').trim().slice(0, FILTER_MAX_LENGTH);
    state.service = $('ltServiceFilter').value;
    state.status = $('ltStatusFilter').value;
    state.page = 1;
  }

  function resetFilters() {
    Object.assign(state, { q: '', service: '', status: '', page: 1 });
    applyStateToControls();
  }

  // ---- selection ----

  function renderSelectionBar() {
    const count = selected.size;
    $('ltSelectionBar').hidden = count === 0;
    $('ltSelectionCount').innerHTML = LL.tHtml('archive.selection.count', { n: LL.html(`<b>${count}</b>`) });
    const names = Array.from(selected.values()).map((r) => r.run_name).join(', ');
    $('ltSelectionNames').textContent = names;
    $('ltSelectionNames').title = names;
  }

  function syncSelection() {
    document.querySelectorAll('#ltTbody tr.data-row').forEach((tr) => {
      const on = selected.has(tr.dataset.runId);
      tr.classList.toggle('selected', on);
      const box = tr.querySelector('input.row-select');
      if (box) box.checked = on;
    });
    renderSelectionBar();
  }

  function setSelected(run, on) {
    if (on) selected.set(run.run_id, run);
    else selected.delete(run.run_id);
    syncSelection();
  }

  // ---- delete ----

  // The confirm dialog; resolves to { withReports } or null when cancelled.
  function confirmDelete(runs) {
    const plan = deletePlan(runs, LL.can('admin'));
    let bodyHtml = `<p>${esc(plan.message)}</p>`;
    if (plan.offerReports) {
      bodyHtml += `<label class="checkbox-row"><input type="checkbox" id="ltDeleteReports">${esc(LL.t('archive.runs.delete.with_reports', { n: plan.reports }))}</label>`
        + `<p class="dim">${esc(LL.t('archive.runs.delete.with_reports_hint'))}</p>`;
    }
    return LL.ui.dialog({
      title: plan.title,
      bodyHtml,
      actions: [
        { label: LL.t('common.action.cancel'), value: null },
        {
          label: LL.t('common.action.delete'),
          danger: true,
          onClick: (close, body) => {
            const box = body.querySelector('#ltDeleteReports');
            close({ withReports: !!(box && box.checked) });
          },
        },
      ],
    });
  }

  async function deleteRuns(runs) {
    const list = runs.filter((r) => r && r.can_delete);
    if (!list.length) return;
    const choice = await confirmDelete(list);
    if (!choice) return;
    try {
      const resp = await fetch('/load_tests/runs/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_ids: list.map((r) => r.run_id), with_reports: choice.withReports }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        LL.ui.toast(data.error || LL.t('archive.runs.delete.failed'), { tone: 'error' });
        return;
      }
      (data.deleted || []).forEach((r) => selected.delete(r.run_id));
      const summary = deleteSummary(data);
      LL.ui.toast(summary.text, { tone: summary.tone });
      await load();
    } catch (err) {
      LL.ui.toast(LL.t('archive.runs.delete.failed_detail', { error: err.message }), { tone: 'error' });
    }
  }

  // ---- table ----

  function cell(text, className) {
    const td = document.createElement('td');
    if (className) td.className = className;
    td.textContent = text;
    return td;
  }

  function withNote(td, note) {
    if (!note) return td;
    const small = document.createElement('span');
    small.className = 'cell-note';
    small.textContent = note;
    td.appendChild(small);
    return td;
  }

  function timeTd(value, now, note) {
    const td = document.createElement('td');
    td.className = 'm nowrap date-cell';
    const info = helpers().timeCell ? helpers().timeCell(value, now) : { text: value || '—', title: '' };
    if (value && info.title) {
      const time = document.createElement('time');
      time.dateTime = String(value);
      time.title = info.title;
      time.textContent = info.text;
      td.appendChild(time);
    } else {
      td.textContent = info.text;
    }
    return withNote(td, note);
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
    const items = document.createElement('div');
    items.className = 'menu-items';
    items.addEventListener('click', (e) => e.stopPropagation());
    if (LL.can('engineer')) {
      items.appendChild(menuButton(LL.t('archive.runs.row.open'), () => { location.href = `/launch/${encodeURIComponent(run.run_id)}`; }));
      items.appendChild(menuButton(LL.t('archive.row.relaunch'), () => { location.href = `/launch?from=${encodeURIComponent(run.run_id)}`; }));
    }
    const report = reportLink(run);
    if (report) items.appendChild(menuButton(LL.t('archive.runs.row.report'), () => { location.href = report.url; }));
    const menu = document.createElement('details');
    if (run.can_delete) {
      items.appendChild(menuButton(LL.t('archive.row.delete'), async () => {
        menu.removeAttribute('open');
        await deleteRuns([run]);
      }, 'danger'));
    }
    if (!items.children.length) return td;
    const inner = document.createElement('div');
    inner.className = 'actions-inner';
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
    menu.append(summary, items);
    inner.appendChild(menu);
    td.appendChild(inner);
    return td;
  }

  function buildSelectCell(run) {
    const td = document.createElement('td');
    td.className = 'select-cell';
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.className = 'row-select';
    if (!run.can_delete) {
      // A disabled box says why, and keeps the phone card layout (a lone checkbox is not the card title).
      box.disabled = true;
      box.title = LL.t(run.active ? 'archive.runs.row.cannot_delete_active' : 'archive.runs.row.cannot_delete_foreign');
      box.setAttribute('aria-label', box.title);
      td.appendChild(box);
      return td;
    }
    box.checked = selected.has(run.run_id);
    box.setAttribute('aria-label', LL.t('archive.runs.row.select', { name: run.run_name }));
    box.addEventListener('change', () => setSelected(run, box.checked));
    td.addEventListener('click', (e) => {
      e.stopPropagation();
      if (e.target !== box) box.click();
    });
    td.appendChild(box);
    return td;
  }

  function renderEmpty(text, actionHtml) {
    $('ltTbody').innerHTML = LL.ui.emptyRow(columns, text, { actionHtml });
  }

  function renderRows(rows) {
    const tbody = $('ltTbody');
    tbody.innerHTML = '';
    if (!rows.length) {
      if (hasFilters(state)) {
        renderEmpty(LL.t('archive.runs.empty.filtered'), `<button type="button" class="btn btn-sm" id="ltEmptyReset">${LL.tHtml('archive.empty.reset_filters')}</button>`);
        const reset = $('ltEmptyReset');
        if (reset) reset.addEventListener('click', () => { resetFilters(); load(); });
      } else {
        renderEmpty(LL.t('archive.runs.empty.none'), LL.can('engineer') ? `<a class="btn btn-sm" href="/launch">${LL.tHtml('archive.runs.empty.launch')}</a>` : '');
      }
      return;
    }
    const now = Date.now();
    const engineer = LL.can('engineer');
    rows.forEach((run) => {
      const tr = document.createElement('tr');
      tr.className = 'data-row';
      tr.dataset.runId = run.run_id;
      tr.classList.toggle('selected', selected.has(run.run_id));
      if (engineer) tr.appendChild(buildSelectCell(run));
      const nameTd = document.createElement('td');
      nameTd.className = 'run-name';
      nameTd.title = run.run_name;
      // The run page shows commands and hosts: a link for engineers, plain text for viewers.
      const name = document.createElement(engineer ? 'a' : 'span');
      name.textContent = run.run_name;
      if (engineer) {
        name.className = 'run-link';
        name.href = `/launch/${encodeURIComponent(run.run_id)}`;
      }
      nameTd.appendChild(name);
      tr.appendChild(nameTd);
      const serviceTd = cell(run.service || '—', 'nowrap service-cell');
      serviceTd.title = run.service || '';
      tr.appendChild(serviceTd);
      const statusTd = document.createElement('td');
      statusTd.className = 'nowrap';
      const pill = document.createElement('span');
      pill.className = `pill ${statusTone(run.status)}`;
      pill.textContent = run.status_title || run.status;
      statusTd.appendChild(pill);
      tr.appendChild(statusTd);
      tr.appendChild(timeTd(run.started_at || run.created_at, now, runDuration(run, now)));
      const inst = instancesText(run);
      const instTd = withNote(cell(inst.text, 'nowrap lt-instances'), inst.note);
      instTd.title = inst.note;
      tr.appendChild(instTd);
      tr.appendChild(cell(run.owner || '—', 'muted nowrap'));
      const reportTd = document.createElement('td');
      reportTd.className = 'run-name lt-report';
      const report = reportLink(run);
      if (report) {
        const link = document.createElement('a');
        link.className = 'run-link';
        link.href = report.url;
        link.textContent = report.text;
        reportTd.title = report.title || report.text;
        reportTd.appendChild(link);
      } else {
        reportTd.textContent = '—';
      }
      tr.appendChild(reportTd);
      tr.appendChild(buildActionsCell(run));
      Array.from(tr.children).forEach((td, i) => { if (columnLabels[i]) td.setAttribute('data-label', columnLabels[i]); });
      tr.addEventListener('click', (e) => {
        if (e.target.closest && e.target.closest('a, button, input, details, summary')) return;
        if (engineer) location.href = `/launch/${encodeURIComponent(run.run_id)}`;
        else if (report) location.href = report.url;
      });
      tbody.appendChild(tr);
    });
  }

  function renderPager(rowsOnPage) {
    const h = helpers();
    const pages = Math.max(1, Math.ceil((Number(state.total) || 0) / PAGE_SIZE));
    const to = Math.min(state.total, (state.page - 1) * PAGE_SIZE + rowsOnPage);
    $('ltCounter').textContent = h.pagerText ? h.pagerText(state.page, state.total, rowsOnPage) : '';
    $('ltPageInfo').textContent = h.pageText ? h.pageText(state.page, state.total) : '';
    $('ltPrevBtn').disabled = state.page <= 1;
    $('ltNextBtn').disabled = to >= state.total;
    $('ltPager').hidden = pages <= 1 && state.page <= 1;
  }

  function renderLoadError(text) {
    renderEmpty(text, `<button type="button" class="btn btn-sm" id="ltRetryLoad">${LL.tHtml('archive.load.retry')}</button>`);
    const retry = $('ltRetryLoad');
    if (retry) retry.addEventListener('click', () => load());
    $('ltCounter').textContent = '';
    $('ltPager').hidden = true;
  }

  async function load(options) {
    const quiet = !!(options && options.quiet);
    const seq = ++loadSeq;
    window.clearTimeout(refreshTimer);
    syncUrl();
    if (!quiet) $('ltTbody').innerHTML = LL.ui.skeletonRows(columns, 5);
    let resp;
    let data;
    try {
      resp = await fetch(buildHistoryUrl(state));
      data = await resp.json();
    } catch (e) {
      if (seq !== loadSeq) return;
      renderLoadError(LL.t('archive.load.failed_detail', { error: e.message }));
      return;
    }
    if (seq !== loadSeq) return;
    if (!resp.ok || !data || !Array.isArray(data.runs)) {
      renderLoadError((data && data.error) ? LL.t('archive.load.error_detail', { error: data.error }) : LL.t('archive.load.error'));
      return;
    }
    state.total = Number(resp.headers.get('X-Total-Count') || data.total || data.runs.length);
    // Past the last page after deletions: go to the last one.
    if (!data.runs.length && state.total > 0 && state.page > 1) {
      state.page = Math.max(1, Math.ceil(state.total / PAGE_SIZE));
      await load(options);
      return;
    }
    renderRows(data.runs);
    renderPager(data.runs.length);
    if (data.runs.some((r) => r.active)) {
      refreshTimer = window.setTimeout(() => { if (isShown()) load({ quiet: true }); }, ACTIVE_REFRESH_MS);
    }
  }

  // ---- wiring ----

  function applySearch() {
    window.clearTimeout(searchTimer);
    const q = String($('ltSearchInput').value || '').trim().slice(0, FILTER_MAX_LENGTH);
    if (q === state.q) return;
    readFilters();
    load();
  }

  function wire() {
    $('ltFiltersForm').addEventListener('submit', (e) => e.preventDefault());
    $('ltSearchInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); applySearch(); } });
    $('ltSearchInput').addEventListener('input', () => {
      window.clearTimeout(searchTimer);
      searchTimer = window.setTimeout(applySearch, SEARCH_DEBOUNCE_MS);
    });
    ['ltServiceFilter', 'ltStatusFilter'].forEach((id) => $(id).addEventListener('change', () => { readFilters(); load(); }));
    $('ltPrevBtn').addEventListener('click', () => { if (state.page > 1) { state.page -= 1; load(); } });
    $('ltNextBtn').addEventListener('click', () => { state.page += 1; load(); });
    $('ltSelectionDelete').addEventListener('click', () => deleteRuns(Array.from(selected.values())));
    $('ltSelectionClear').addEventListener('click', () => {
      selected.clear();
      syncSelection();
      $('ltSearchInput').focus();
    });
  }

  // Called by archive.js each time the tab is shown: the first time reads the address and loads,
  // later times put the tab's filters back into the address and refresh.
  LL.archiveRuns.show = function show() {
    if (!$('ltRunsTable')) return;
    if (started) {
      load({ quiet: true });
      return;
    }
    started = true;
    // Only an address of this tab carries its filters; the reports tab uses the same parameter names.
    if (new URLSearchParams(location.search).get('tab') === 'runs') Object.assign(state, parseQuery(location.search));
    if (!LL.can('engineer')) {
      const selectHead = document.querySelector('#ltRunsTable thead th.select-col');
      if (selectHead) selectHead.remove();
    }
    const heads = Array.from(document.querySelectorAll('#ltRunsTable thead th'));
    columns = heads.length;
    columnLabels = heads.map((th) => th.textContent.trim());
    STATUS_FILTERS.forEach((code) => addOption($('ltStatusFilter'), code, statusFilterLabel(code)));
    applyStateToControls();
    wire();
    renderSelectionBar();
    loadServiceOptions();
    load();
  };
})();
