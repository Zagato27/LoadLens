// Archive page: runs table whose filters, sort and page live in the URL, plus multi-select for comparison.
(function () {
  const LL = window.LoadLens = window.LoadLens || {};
  const PAGE_SIZE = 20;
  const JOBS_REFRESH_MS = 5000;
  const COLUMNS = 9;
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
    const verdict = text('verdict');
    if ((LL.VERDICTS || []).includes(verdict)) state.verdict = verdict;
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
    return `/runs?${params.toString()}`;
  }

  function hasFilters(state) {
    return !!(state.q || state.service || state.verdict || state.testType);
  }

  // Chips above the table, one per active filter. `labels.service(id)` and `labels.testType(key)` give display names.
  function activeChips(state, labels) {
    const names = labels || {};
    const chips = [];
    if (state.q) chips.push({ key: 'q', label: 'Поиск', value: `«${state.q}»` });
    if (state.service) chips.push({ key: 'service', label: 'Сервис', value: names.service ? names.service(state.service) : state.service });
    if (state.verdict) chips.push({ key: 'verdict', label: 'Вердикт', value: state.verdict });
    if (state.testType) chips.push({ key: 'testType', label: 'Тип теста', value: names.testType ? names.testType(state.testType) : state.testType });
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
    return `Выбрано ${count}`;
  }

  // Why «Сравнить» is unavailable; '' when exactly two runs are selected.
  function selectionHint(count) {
    if (count === 2 || count <= 0) return '';
    if (count === 1) return 'Отметьте ещё один запуск, чтобы сравнить';
    return 'Сравнить можно ровно два запуска — снимите лишние отметки';
  }

  // «12 345» (digits grouped by a no-break space) from 100 up, «95,5» below; '—' when unknown.
  function formatRps(value) {
    if (value === null || value === undefined || value === '' || typeof value === 'boolean') return '—';
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    if (Math.abs(n) >= 100) {
      const rounded = Math.round(n);
      const digits = String(Math.abs(rounded)).replace(/\B(?=(\d{3})+(?!\d))/g, '\u00a0');
      return rounded < 0 ? `-${digits}` : digits;
    }
    return (Math.round(n * 10) / 10).toFixed(1).replace(/\.0$/, '').replace('.', ',');
  }

  // Relative date with the full one for the tooltip.
  function timeCell(value, now) {
    if (!value) return { text: '—', title: '' };
    const full = LL.formatDateTime(value);
    return { text: LL.relativeTime(value, now), title: full === '—' ? '' : full };
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

  LL.archive = LL.archive || {};
  LL.archive._internals = {
    PAGE_SIZE, SORT_KEYS, defaultState, defaultDir, parseQuery, buildQuery, buildRunsUrl, hasFilters, activeChips,
    withoutFilter, nextSort, compareTargets, compareUrl, selectionLabel, selectionHint, formatRps, timeCell, durationText,
    shortTypeLabel,
  };

  // ---- page --------------------------------------------------------------------------------

  const $ = (id) => document.getElementById(id);
  const esc = (value) => LL.ui.escapeHtml(value);
  const state = { ...defaultState(), total: 0 };
  // Selected runs survive paging and filter changes: run_name -> run row.
  const selected = new Map();
  let jobsTimer = null;
  let loadSeq = 0;
  let columnLabels = [];
  // Runs started from the Run test page: "service|run name" -> load test run id (engineers only).
  let launchedRuns = null;

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
    LL.VERDICTS.forEach((v) => ensureOption($('verdictFilter'), v, v));
    Object.keys(LL.TEST_TYPE_LABELS).forEach((key) => ensureOption($('testTypeFilter'), key, LL.TEST_TYPE_LABELS[key]));
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
      btn.setAttribute('aria-label', `Убрать фильтр «${chip.label}: ${chip.value}»`);
      btn.title = 'Убрать фильтр';
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
      all.textContent = 'Сбросить все';
      all.addEventListener('click', () => { resetFilters(); loadPage(); $('searchInput').focus(); });
      box.appendChild(all);
    }
    box.hidden = !chips.length;
  }

  function syncUrl() {
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
    $('selectionCount').textContent = selectionLabel(count);
    const names = Array.from(selected.keys()).join(', ');
    $('selectionNames').textContent = names;
    $('selectionNames').title = names;
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
      LL.ui.toast(selectionHint(selected.size) || 'Отметьте два запуска', { tone: 'info' });
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

  function timeTd(value, now, className, sub) {
    const td = document.createElement('td');
    td.className = className || '';
    const info = timeCell(value, now);
    if (value && info.title) {
      const time = document.createElement('time');
      time.dateTime = String(value);
      time.title = info.title;
      time.textContent = info.text;
      td.appendChild(time);
    } else {
      td.textContent = info.text;
    }
    if (sub) {
      const small = document.createElement('div');
      small.className = 'cell-sub';
      small.textContent = sub;
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
    summary.title = 'Действия';
    summary.setAttribute('aria-label', `Действия для ${run.run_name}`);
    summary.addEventListener('click', (e) => e.stopPropagation());
    const items = document.createElement('div');
    items.className = 'menu-items';
    items.addEventListener('click', (e) => e.stopPropagation());

    items.appendChild(menuButton('Сравнить с другим запуском', () => {
      location.href = '/compare?run_a=' + encodeURIComponent(run.run_name);
    }));
    const launched = launchedRun(run);
    if (launched && LL.can('engineer')) {
      items.appendChild(menuButton('Повторить запуск', () => {
        location.href = '/launch?from=' + encodeURIComponent(launched);
      }));
    }
    if (LL.can('engineer')) {
      items.appendChild(menuButton('Переименовать', async () => {
        menu.removeAttribute('open');
        await renameRun(run);
      }));
    }
    items.appendChild(menuButton('Копировать ссылку', async (btn) => {
      try {
        await navigator.clipboard.writeText(location.origin + LL.reportUrl(run));
        btn.textContent = 'Ссылка скопирована';
        btn.classList.add('confirmed');
        setTimeout(() => { btn.textContent = 'Копировать ссылку'; btn.classList.remove('confirmed'); menu.removeAttribute('open'); }, 1200);
      } catch (err) {
        btn.textContent = 'Не удалось скопировать';
        setTimeout(() => { btn.textContent = 'Копировать ссылку'; }, 1500);
      }
    }));
    if (LL.can('admin')) {
      items.appendChild(menuButton('Удалить отчёт', async () => {
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
      title: 'Переименовать отчёт',
      label: 'Новое название',
      value: run.run_name,
      validate: (v) => (!v.trim() ? 'Название не может быть пустым' : (/[\\/\n\r\t]/.test(v) ? 'Без символов / \\ и переносов строк' : ''))
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
        LL.ui.toast(`Отчёт переименован в «${name}»`, { tone: 'ok' });
        await loadPage();
      } else LL.ui.toast(j.error || 'Ошибка переименования', { tone: 'error' });
    } catch (err) {
      LL.ui.toast(`Ошибка переименования: ${err.message}`, { tone: 'error' });
    }
  }

  async function deleteRun(run) {
    const ok = await LL.ui.confirm({
      title: 'Удалить отчёт',
      message: `Отчёт «${run.run_name}» будет удалён вместе с метриками и анализом. Это действие необратимо.`,
      confirmText: 'Удалить',
      danger: true
    });
    if (!ok) return;
    try {
      const resp = await fetch('/runs/' + encodeURIComponent(run.run_name), { method: 'DELETE' });
      const j = await resp.json();
      if (resp.ok) {
        selected.delete(run.run_name);
        LL.ui.toast('Отчёт удалён', { tone: 'ok' });
        await loadPage();
      } else LL.ui.toast(j.error || 'Ошибка удаления', { tone: 'error' });
    } catch (err) {
      LL.ui.toast(`Ошибка удаления: ${err.message}`, { tone: 'error' });
    }
  }

  function buildSelectCell(run) {
    const td = document.createElement('td');
    td.className = 'select-cell';
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.className = 'row-select';
    box.checked = selected.has(run.run_name);
    box.setAttribute('aria-label', `Выбрать «${run.run_name}» для сравнения`);
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
        renderEmpty('По заданным фильтрам запусков не найдено.', '<button type="button" class="btn btn-sm" id="emptyReset">Сбросить фильтры</button>');
        const reset = $('emptyReset');
        if (reset) reset.addEventListener('click', () => { resetFilters(); loadPage(); });
      } else {
        renderEmpty('Запусков пока нет.', LL.can('engineer') ? '<a class="btn btn-sm" href="/new">Создать отчёт</a>' : '');
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
        badge.textContent = 'запущен из LoadLens';
        nameCell.appendChild(badge);
      }
      tr.appendChild(nameCell);
      tr.appendChild(cell(r.service || '—', 'nowrap'));
      const typeTd = cell(shortTypeLabel(r.test_type_key || r.test_type), 'muted nowrap');
      typeTd.title = LL.testTypeLabel(r.test_type_key || r.test_type);
      tr.appendChild(typeTd);
      tr.appendChild(timeTd(r.end_time || r.start_time, now, 'nowrap', durationText(r)));
      tr.appendChild(timeTd(r.report_created_at, now, 'muted nowrap'));
      const verdict = r.verdict || 'Недостаточно данных';
      const verdictTd = document.createElement('td');
      const pill = document.createElement('span');
      pill.className = `pill ${LL.verdictClass(verdict)}`;
      pill.textContent = verdict;
      if (r.manual_verdict && r.manual_verdict.verdict) {
        pill.title = LL.manualVerdictTitle(r.manual_verdict);
      } else if (r.sla_verdict && r.llm_verdict && r.sla_verdict !== r.llm_verdict) {
        pill.title = `По SLA: ${r.sla_verdict}; оценка ИИ: ${r.llm_verdict}`;
      }
      verdictTd.appendChild(pill);
      if (r.manual_verdict && r.manual_verdict.verdict) {
        const mark = document.createElement('span');
        mark.className = 'verdict-manual-mark';
        mark.textContent = 'вручную';
        mark.title = pill.title;
        verdictTd.appendChild(mark);
      }
      tr.appendChild(verdictTd);
      tr.appendChild(cell(formatRps(r.max_rps), 'num'));
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

  function renderPager(rowsOnPage) {
    const from = state.total ? (state.page - 1) * PAGE_SIZE + 1 : 0;
    const to = Math.min(state.total, (state.page - 1) * PAGE_SIZE + rowsOnPage);
    $('pageInfo').textContent = state.total
      ? `Показано ${from}–${to} из ${state.total}`
      : 'Ничего не найдено';
    $('prevBtn').disabled = state.page <= 1;
    $('nextBtn').disabled = to >= state.total;
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
    box.innerHTML = `<a href="/reports">Архив</a><span class="sep" aria-hidden="true">›</span><span class="current" aria-current="page">${esc(optionLabel('serviceFilter', state.service))}</span>`;
    box.hidden = false;
  }

  function launchedRun(run) {
    return launchedRuns ? launchedRuns.get(`${run.service || ''}|${run.run_name || ''}`) : null;
  }

  async function loadLaunchedRuns() {
    if (launchedRuns || !LL.can('engineer')) return;
    launchedRuns = new Map();
    try {
      const resp = await fetch('/load_tests/runs?limit=200');
      if (!resp.ok) return;
      const data = await resp.json();
      (data.runs || []).forEach((run) => launchedRuns.set(`${run.service}|${run.run_name}`, run.run_id));
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
      renderLoadError(`Не удалось загрузить список: ${e.message}`);
      return;
    }
    // A newer request (filter or page change) superseded this one.
    if (seq !== loadSeq) return;
    if (!resp.ok || !Array.isArray(rows)) {
      renderLoadError((rows && rows.error) ? `Ошибка загрузки: ${rows.error}` : 'Ошибка загрузки списка');
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
    renderEmpty(text, '<button type="button" class="btn btn-sm" id="retryLoad">Повторить</button>');
    const retry = $('retryLoad');
    if (retry) retry.addEventListener('click', () => loadPage());
    $('pageInfo').textContent = '';
    $('prevBtn').disabled = true;
    $('nextBtn').disabled = true;
  }

  // ---- background jobs strip ----

  function renderJobRow(job) {
    const row = document.createElement('div');
    row.className = 'job-row';
    const pill = document.createElement('span');
    pill.className = `pill ${job.status}`;
    pill.textContent = job.status === 'running' ? 'В работе' : 'Ошибка';
    const info = document.createElement('div');
    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = job.run_name || '(без названия)';
    const meta = document.createElement('div');
    meta.className = 'meta';
    const bits = [];
    if (job.kind === 'confluence') bits.push('публикация в Confluence');
    if (job.kind === 'forecast_confluence') bits.push('публикация прогноза в Confluence');
    if (job.service) bits.push(job.service);
    bits.push(job.status === 'running' ? `${Number(job.progress) || 0}% — ${job.message || ''}` : (job.message || ''));
    if (job.updated_at) bits.push(LL.formatDateTime(job.updated_at));
    meta.textContent = bits.filter(Boolean).join(' · ');
    info.appendChild(name);
    info.appendChild(meta);
    const link = document.createElement('a');
    if (job.kind === 'report') {
      link.href = `/new?job=${encodeURIComponent(job.job_id)}`;
      link.textContent = job.status === 'running' ? 'Следить' : 'Подробности';
      link.setAttribute('data-min-role', 'engineer');
    } else if (job.kind === 'forecast_confluence' && job.report_url) {
      link.href = job.report_url;
      link.textContent = 'К прогнозу';
    } else if (job.page_url && String(job.page_url).startsWith('/reports/')) {
      link.href = job.page_url;
      link.textContent = 'К отчёту';
    } else if (job.run_name) {
      link.href = LL.reportUrl({ run_name: job.run_name });
      link.textContent = 'К отчёту';
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
      $('jobsStripNote').textContent = running ? `в работе: ${running}` : '';
      strip.style.display = '';
      if (running && !jobsTimer) jobsTimer = setInterval(async () => { await loadJobsStrip(); await loadPage({ quiet: true }); }, JOBS_REFRESH_MS);
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

  function wireControls() {
    $('searchBtn').addEventListener('click', () => { readFilters(); loadPage(); });
    $('clearBtn').addEventListener('click', () => { resetFilters(); loadPage(); });
    $('searchInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); readFilters(); loadPage(); } });
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

  document.addEventListener('DOMContentLoaded', async () => {
    if (!$('runsTable')) return;
    // The filters apply as they change; the form itself never submits (CSP forbids inline handlers).
    const filtersForm = $('filtersForm');
    if (filtersForm) filtersForm.addEventListener('submit', (e) => e.preventDefault());
    columnLabels = Array.from(document.querySelectorAll('#runsTable thead th')).map((th) => th.textContent.trim());
    // Filters, sort and page come from the address (?service= also arrives from report breadcrumbs).
    Object.assign(state, parseQuery(location.search));
    fillStaticOptions();
    applyStateToControls();
    wireControls();
    renderSelectionBar();
    const services = loadServiceOptions();
    await loadPage();
    await services;
    $('jobsStripClear').addEventListener('click', clearFinishedJobs);
    loadJobsStrip();
  });
})();
