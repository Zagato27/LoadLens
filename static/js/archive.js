// Archive page logic
(function () {
  const LL = window.LoadLens;
  const PAGE_SIZE = 20;
  const JOBS_REFRESH_MS = 5000;

  const state = {
    page: 1,
    total: 0,
    sort: 'end_time',
    dir: 'desc',
    q: '',
    service: '',
    verdict: '',
    testType: ''
  };
  let jobsTimer = null;

  const $ = (id) => document.getElementById(id);

  function fmtDuration(startIso, endIso) {
    const s = new Date(startIso).getTime();
    const e = new Date(endIso).getTime();
    if (!Number.isFinite(s) || !Number.isFinite(e)) return '—';
    const m = Math.floor(Math.max(0, e - s) / 60000);
    const h = Math.floor(m / 60);
    return h ? `${h}ч ${String(m % 60).padStart(2, '0')}м` : `${m}м`;
  }

  function reportUrl(run) {
    if (run && run.run_id) return '/reports/' + encodeURIComponent(run.run_id);
    return '/reports/' + encodeURIComponent(run.run_name || '');
  }

  // ---- filters ----------------------------------------------------------

  async function loadFilterOptions() {
    const verdictSel = $('verdictFilter');
    LL.VERDICTS.forEach((v) => {
      const opt = document.createElement('option');
      opt.value = v; opt.textContent = v;
      verdictSel.appendChild(opt);
    });
    const typeSel = $('testTypeFilter');
    Object.keys(LL.TEST_TYPE_LABELS).forEach((key) => {
      const opt = document.createElement('option');
      opt.value = key; opt.textContent = LL.TEST_TYPE_LABELS[key];
      typeSel.appendChild(opt);
    });
    try {
      const resp = await fetch('/services');
      const payload = await resp.json();
      const services = Array.isArray(payload && payload.services) ? payload.services : [];
      const serviceSel = $('serviceFilter');
      services.forEach((svc) => {
        const id = typeof svc === 'string' ? svc : String((svc && svc.id) || '');
        if (!id) return;
        const opt = document.createElement('option');
        opt.value = id; opt.textContent = (svc && svc.title) || id;
        serviceSel.appendChild(opt);
      });
    } catch (e) {
      // service filter stays with "Все" only
    }
  }

  function readFilters() {
    state.q = String($('searchInput').value || '').trim();
    state.service = $('serviceFilter').value;
    state.verdict = $('verdictFilter').value;
    state.testType = $('testTypeFilter').value;
    state.page = 1;
  }

  function resetFilters() {
    $('searchInput').value = '';
    $('serviceFilter').value = '';
    $('verdictFilter').value = '';
    $('testTypeFilter').value = '';
    readFilters();
  }

  // ---- table ------------------------------------------------------------

  function buildRunsUrl() {
    const u = new URL('/runs', location.origin);
    if (state.q) u.searchParams.set('q', state.q);
    if (state.service) u.searchParams.set('service', state.service);
    if (state.verdict) u.searchParams.set('verdict', state.verdict);
    if (state.testType) u.searchParams.set('test_type', state.testType);
    u.searchParams.set('offset', String((state.page - 1) * PAGE_SIZE));
    u.searchParams.set('limit', String(PAGE_SIZE));
    u.searchParams.set('sort', state.sort);
    u.searchParams.set('dir', state.dir);
    return u;
  }

  function cell(text, className) {
    const td = document.createElement('td');
    if (className) td.className = className;
    td.textContent = text;
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

    const openBtn = document.createElement('button');
    openBtn.className = 'btn';
    openBtn.textContent = 'Открыть';
    openBtn.addEventListener('click', (e) => { e.stopPropagation(); location.href = reportUrl(run); });

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
    if (LL.can('engineer')) {
      items.appendChild(menuButton('Переименовать', async () => {
        menu.removeAttribute('open');
        await renameRun(run);
      }));
    }
    items.appendChild(menuButton('Копировать ссылку', async (btn) => {
      try {
        await navigator.clipboard.writeText(location.origin + reportUrl(run));
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
    inner.appendChild(openBtn);
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
      if (resp.ok) { LL.ui.toast(`Отчёт переименован в «${name}»`, { tone: 'ok' }); await loadPage(); }
      else LL.ui.toast(j.error || 'Ошибка переименования', { tone: 'error' });
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
      if (resp.ok) { LL.ui.toast('Отчёт удалён', { tone: 'ok' }); await loadPage(); }
      else LL.ui.toast(j.error || 'Ошибка удаления', { tone: 'error' });
    } catch (err) {
      LL.ui.toast(`Ошибка удаления: ${err.message}`, { tone: 'error' });
    }
  }

  function renderRows(rows) {
    const tbody = $('tbody');
    tbody.innerHTML = '';
    if (!rows.length) {
      const tr = document.createElement('tr');
      tr.className = 'empty-row';
      const td = document.createElement('td');
      td.colSpan = 8;
      const hasFilters = state.q || state.service || state.verdict || state.testType;
      td.textContent = hasFilters
        ? 'По заданным фильтрам запусков не найдено. Сбросьте фильтры или измените запрос.'
        : 'Запусков пока нет. Создайте первый отчёт на странице «Новый отчёт».';
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }
    rows.forEach((r) => {
      const tr = document.createElement('tr');
      tr.className = 'data-row';
      const nameCell = cell(r.run_name, 'run-name');
      nameCell.title = r.run_name;
      tr.appendChild(nameCell);
      tr.appendChild(cell(r.service || '—'));
      tr.appendChild(cell(LL.testTypeLabel(r.test_type), 'muted'));
      tr.appendChild(cell(LL.formatDateTime(r.end_time || r.start_time), 'nowrap'));
      tr.appendChild(cell(r.report_created_at ? LL.formatDateTime(r.report_created_at) : '—', 'muted nowrap'));
      const verdict = r.verdict || 'Недостаточно данных';
      const verdictTd = document.createElement('td');
      const pill = document.createElement('span');
      pill.className = `pill ${LL.verdictClass(verdict)}`;
      pill.textContent = verdict;
      if (r.sla_verdict && r.llm_verdict && r.sla_verdict !== r.llm_verdict) {
        pill.title = `По SLA: ${r.sla_verdict}; оценка ИИ: ${r.llm_verdict}`;
      }
      verdictTd.appendChild(pill);
      tr.appendChild(verdictTd);
      tr.appendChild(cell(fmtDuration(r.start_time, r.end_time), 'muted'));
      tr.appendChild(buildActionsCell(r));
      tr.addEventListener('click', () => { location.href = reportUrl(r); });
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
      th.classList.remove('sorted-asc', 'sorted-desc');
      if (th.getAttribute('data-sort') === state.sort) th.classList.add(state.dir === 'asc' ? 'sorted-asc' : 'sorted-desc');
    });
  }

  function renderSkeletonRows() {
    const tbody = $('tbody');
    tbody.innerHTML = '';
    for (let i = 0; i < 5; i += 1) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = 8;
      td.innerHTML = LL.ui.skeleton(1, ['']);
      tr.appendChild(td);
      tbody.appendChild(tr);
    }
  }

  function renderBreadcrumbs() {
    const box = $('archiveBreadcrumbs');
    if (!box) return;
    if (!state.service) { box.style.display = 'none'; box.innerHTML = ''; return; }
    box.innerHTML = `<a href="/reports">Архив</a><span class="sep">›</span><span class="current">${LL.ui.escapeHtml(state.service)}</span>`;
    box.style.display = '';
  }

  async function loadPage() {
    renderSkeletonRows();
    let resp;
    let rows;
    try {
      resp = await fetch(buildRunsUrl());
      rows = await resp.json();
    } catch (e) {
      $('tbody').innerHTML = '';
      $('pageInfo').textContent = `Ошибка загрузки списка: ${e.message}`;
      return;
    }
    if (!resp.ok || !Array.isArray(rows)) {
      $('tbody').innerHTML = '';
      $('pageInfo').textContent = (rows && rows.error) ? `Ошибка загрузки: ${rows.error}` : 'Ошибка загрузки списка';
      return;
    }
    state.total = Number(resp.headers.get('X-Total-Count') || rows.length);
    renderRows(rows);
    renderPager(rows.length);
    renderSortIndicators();
    renderBreadcrumbs();
  }

  // ---- background jobs strip -------------------------------------------

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
      link.href = '/reports/' + encodeURIComponent(job.run_name);
      link.textContent = 'К отчёту';
    }
    row.appendChild(pill);
    row.appendChild(info);
    row.appendChild(link);
    return row;
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
      if (!jobs.length) {
        strip.style.display = 'none';
        if (jobsTimer) { clearInterval(jobsTimer); jobsTimer = null; }
        return;
      }
      jobs.forEach((job) => list.appendChild(renderJobRow(job)));
      const running = jobs.filter((j) => j.status === 'running').length;
      $('jobsStripNote').textContent = running ? `в работе: ${running}` : '';
      strip.style.display = '';
      if (running && !jobsTimer) jobsTimer = setInterval(async () => { await loadJobsStrip(); await loadPage(); }, JOBS_REFRESH_MS);
      if (!running && jobsTimer) { clearInterval(jobsTimer); jobsTimer = null; }
    } catch (e) {
      // strip is auxiliary
    }
  }

  // ---- wiring -----------------------------------------------------------

  function wireControls() {
    $('searchBtn').addEventListener('click', () => { readFilters(); loadPage(); });
    $('clearBtn').addEventListener('click', () => { resetFilters(); loadPage(); });
    $('searchInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); readFilters(); loadPage(); } });
    ['serviceFilter', 'verdictFilter', 'testTypeFilter'].forEach((id) => {
      $(id).addEventListener('change', () => { readFilters(); loadPage(); });
    });
    $('prevBtn').addEventListener('click', () => { if (state.page > 1) { state.page--; loadPage(); } });
    $('nextBtn').addEventListener('click', () => { state.page++; loadPage(); });
    document.querySelectorAll('th[data-sort]').forEach((th) => {
      th.addEventListener('click', () => {
        const by = th.getAttribute('data-sort');
        if (state.sort === by) state.dir = (state.dir === 'asc') ? 'desc' : 'asc';
        else { state.sort = by; state.dir = (by === 'end_time' || by === 'report_created_at') ? 'desc' : 'asc'; }
        state.page = 1;
        loadPage();
      });
    });
    // Close any open row menu when clicking elsewhere.
    document.addEventListener('click', (e) => {
      document.querySelectorAll('details.menu[open]').forEach((menu) => {
        if (!menu.contains(e.target)) menu.removeAttribute('open');
      });
    });
  }

  document.addEventListener('DOMContentLoaded', async () => {
    wireControls();
    await loadFilterOptions();
    // Deep link from breadcrumbs / dashboard: /reports?service=X
    const params = new URLSearchParams(location.search);
    const preService = (params.get('service') || '').trim();
    if (preService) {
      const sel = $('serviceFilter');
      if (!Array.from(sel.options).some((o) => o.value === preService)) {
        const opt = document.createElement('option');
        opt.value = preService; opt.textContent = preService;
        sel.appendChild(opt);
      }
      sel.value = preService;
      state.service = preService;
    }
    await loadPage();
    loadJobsStrip();
  });
})();
