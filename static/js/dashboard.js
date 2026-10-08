// Dashboard page logic
(function () {
  const LL = window.LoadLens;
  const ui = LL.ui;
  const $ = (id) => document.getElementById(id);
  const esc = ui.escapeHtml;
  let statusChart = null;

  function fmtPeriod(startMs, endMs) {
    const s = Number(startMs); const e = Number(endMs);
    if (!Number.isFinite(s) && !Number.isFinite(e)) return '—';
    if (Number.isFinite(s) && Number.isFinite(e)) return `${LL.formatDateTime(s)} — ${LL.formatDateTime(e)}`;
    return LL.formatDateTime(Number.isFinite(s) ? s : e);
  }
  function pill(verdict) {
    const v = verdict || 'Недостаточно данных';
    return `<span class="pill ${LL.verdictClass(v)}">${esc(v)}</span>`;
  }
  function reportUrl(run) {
    if (run && run.run_id) return `/reports/${encodeURIComponent(run.run_id)}`;
    const name = (run && run.run_name) || '';
    return name ? `/reports/${encodeURIComponent(name)}` : '/reports';
  }

  function renderSkeletons() {
    $('servicesBody').innerHTML = `<tr><td colspan="6">${ui.skeleton(3)}</td></tr>`;
    $('recentBody').innerHTML = `<tr><td colspan="5">${ui.skeleton(3)}</td></tr>`;
  }

  function renderLastRun(last) {
    if (!last) return;
    $('sumRun').textContent = last.run_name || '—';
    $('sumService').textContent = last.service || '—';
    $('sumPeriod').textContent = fmtPeriod(last.start_ms, last.end_ms);
    $('sumVerdict').innerHTML = pill(last.verdict);
    const link = $('lastReportLink');
    if (last.run_name) { link.href = reportUrl(last); link.style.display = 'block'; }
  }

  function renderVerdictChart(counts) {
    const labels = LL.VERDICTS;
    const values = labels.map((k) => Number(counts[k] || 0));
    const total = values.reduce((a, b) => a + b, 0);
    $('verdictTotals').textContent = total ? `Всего запусков с итогом: ${total}` : 'Пока нет завершённых отчётов';
    const colors = ui.chartColors();
    const ctx = $('statusChart').getContext('2d');
    if (statusChart) statusChart.destroy();
    // eslint-disable-next-line no-undef
    statusChart = new Chart(ctx, {
      type: 'doughnut',
      data: { labels, datasets: [{ data: values, backgroundColor: ['#2ecc71', '#f1c40f', '#e74c3c', '#7f8c8d'], borderColor: colors.background, borderWidth: 2 }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: colors.text } } } }
    });
  }

  function renderServices(services) {
    const body = $('servicesBody');
    body.innerHTML = '';
    if (!services.length) {
      body.innerHTML = '<tr class="empty-row"><td colspan="6">Пока нет ни одного завершённого отчёта.</td></tr>';
      return;
    }
    services.forEach((svc) => {
      const tr = document.createElement('tr');
      const dots = (svc.recent_verdicts || []).slice().reverse().map((v) => `<span class="dot ${LL.verdictClass(v)}" title="${esc(v)}"></span>`).join('');
      tr.innerHTML = `
        <td><a href="/reports?service=${encodeURIComponent(svc.service)}"><strong>${esc(svc.service)}</strong></a></td>
        <td class="run-name" title="${esc(svc.last_run)}"><a href="${reportUrl({ run_name: svc.last_run })}">${esc(svc.last_run)}</a><div class="dim">${esc(LL.testTypeLabel(svc.test_type))} · ${esc(svc.created_at ? LL.formatDateTime(svc.created_at) : '—')}</div></td>
        <td>${pill(svc.verdict)}</td>
        <td class="num">${svc.max_rps !== null && svc.max_rps !== undefined ? esc(Number(svc.max_rps).toFixed(1)) : '—'}</td>
        <td><span class="verdict-dots" title="Слева направо: от старых к новым">${dots}</span></td>
        <td class="nowrap"><a class="btn btn-sm" href="/compare?run_a=${encodeURIComponent(svc.last_run)}">Сравнить с предыдущим</a></td>`;
      body.appendChild(tr);
    });
  }

  function renderRecent(runs) {
    const body = $('recentBody');
    body.innerHTML = '';
    if (!runs.length) {
      body.innerHTML = '<tr class="empty-row"><td colspan="5">Запусков пока нет.</td></tr>';
      return;
    }
    runs.forEach((r) => {
      const tr = document.createElement('tr');
      tr.className = 'clickable';
      tr.innerHTML = `
        <td class="run-name" title="${esc(r.run_name)}">${esc(r.run_name)}</td>
        <td>${esc(r.service || '—')}</td>
        <td class="muted">${esc(LL.testTypeLabel(r.test_type))}</td>
        <td>${pill(r.verdict)}</td>
        <td class="muted nowrap">${esc(r.created_at ? LL.formatDateTime(r.created_at) : '—')}</td>`;
      tr.addEventListener('click', () => { location.href = reportUrl(r); });
      body.appendChild(tr);
    });
  }

  async function loadJobs() {
    try {
      const resp = await fetch('/jobs?status=running&limit=5');
      if (!resp.ok) return;
      const data = await resp.json();
      const jobs = Array.isArray(data.jobs) ? data.jobs : [];
      const card = $('jobsCard');
      const list = $('jobsList');
      if (!jobs.length) { card.hidden = true; return; }
      card.hidden = false;
      list.innerHTML = jobs.map((j) => `<div class="job-row"><span class="pill running">В работе</span><div><div class="name">${esc(j.run_name || '(без названия)')}</div><div class="meta">${esc(j.service || '')} · ${Number(j.progress) || 0}% — ${esc(j.message || '')}</div></div><a href="/new?job=${encodeURIComponent(j.job_id)}" data-min-role="engineer">Следить</a></div>`).join('');
      setTimeout(loadJobs, 5000);
    } catch (e) {
      // auxiliary block
    }
  }

  async function seedDemo(button) {
    button.disabled = true;
    ui.toast('Создаём демонстрационный прогон…', { tone: 'info', timeout: 3000 });
    try {
      const resp = await fetch('/demo/seed', { method: 'POST' });
      const data = await resp.json();
      if (resp.status === 409 && data.report_url) { ui.toast('Демо-прогон уже есть — открываем', { tone: 'info' }); location.href = data.report_url; return; }
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      ui.toast('Демо-прогон создан', { tone: 'ok' });
      location.href = data.report_url;
    } catch (e) {
      ui.toast(`Не удалось создать демо-данные: ${e.message}`, { tone: 'error' });
      button.disabled = false;
    }
  }

  async function loadDashboard() {
    renderSkeletons();
    let data;
    try {
      const resp = await fetch('/dashboard_data');
      data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    } catch (e) {
      $('setupBanner').hidden = false;
      $('setupBanner').querySelector('.muted').textContent = `Не удалось загрузить данные: ${e.message}. Скорее всего, база данных недоступна — проверьте подключение в мастере.`;
      $('servicesBody').innerHTML = '<tr class="empty-row"><td colspan="6">Нет данных</td></tr>';
      $('recentBody').innerHTML = '<tr class="empty-row"><td colspan="5">Нет данных</td></tr>';
      return;
    }
    const runsTotal = Number(data.runs_total || 0);
    const hasFinal = !!(data.last_run && data.last_run.run_name);
    const empty = runsTotal === 0 && !hasFinal;
    // With a project area selected an empty dashboard means "no runs here", not "not configured".
    const area = await LL.initProjectArea();
    $('setupBanner').hidden = !empty || !!area;
    $('emptyState').hidden = !empty;
    if (empty && area) {
      $('emptyState').querySelector('h3').textContent = `В области «${area}» отчётов пока нет`;
    }
    renderLastRun(data.last_run);
    renderVerdictChart(data.verdict_counts || {});
    renderServices(Array.isArray(data.services) ? data.services : []);
    renderRecent(Array.isArray(data.recent_runs) ? data.recent_runs : []);
  }

  document.addEventListener('DOMContentLoaded', () => {
    ['bannerDemoBtn', 'emptyDemoBtn'].forEach((id) => { const b = $(id); if (b) b.addEventListener('click', () => seedDemo(b)); });
    ui.theme.onChange(() => { if (statusChart) { const c = ui.chartColors(); statusChart.data.datasets[0].borderColor = c.background; statusChart.update('none'); } });
    loadDashboard();
    loadJobs();
  });
})();
