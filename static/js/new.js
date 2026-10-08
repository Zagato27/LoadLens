// New report page logic (templates/index.html)
(function () {
  const LL = window.LoadLens;
  const POLL_INTERVAL_MS = 1500;
  const JOBS_REFRESH_MS = 5000;
  const LONG_WINDOW_WARN_MINUTES = 24 * 60;
  const SERVICE_HINT = 'Список сервисов задаётся в настройках (раздел «Сервисы»).';
  const TEST_TYPE_HINT = 'Тип теста подбирает инструкции для анализа ИИ и метод расчёта максимальной производительности.';

  // Phase boundaries mirror the progress budget in AI/pipeline.py and update_page.py.
  // The Confluence template path spends 0..40 on charts and attachments, then maps the pipeline into 40..94.
  const WEB_PHASES = [
    { key: 'prepare', from: 0 },
    { key: 'collect', from: 10 },
    { key: 'sla', from: 42 },
    { key: 'save_metrics', from: 45 },
    { key: 'llm', from: 47 },
    { key: 'final', from: 83 },
    { key: 'save', from: 95 },
    { key: 'finish', from: 98 }
  ];
  const CONFLUENCE_PHASES = [
    { key: 'prepare', from: 0 },
    { key: 'charts', from: 8 },
    { key: 'attachments', from: 28 },
    { key: 'collect', from: 45 },
    { key: 'sla', from: 63 },
    { key: 'save_metrics', from: 64 },
    { key: 'llm', from: 65 },
    { key: 'final', from: 85 },
    { key: 'save', from: 91 },
    { key: 'finish', from: 98 }
  ];
  let activePhases = WEB_PHASES;

  let progressTimer = null;
  let jobsTimer = null;
  // Options of the job currently shown in the progress card (run name, service, publication target).
  let currentJobOpts = {};

  const $ = (id) => document.getElementById(id);

  function setHint(id, text, tone) {
    const el = $(id);
    if (!el) return;
    el.textContent = text || '';
    el.classList.remove('error', 'warn', 'ok');
    if (tone) el.classList.add(tone);
  }

  function setInvalid(id, invalid) {
    const el = $(id);
    if (el) el.setAttribute('aria-invalid', invalid ? 'true' : 'false');
  }

  function toLocalInputValue(date) {
    const pad = (n) => String(n).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
  }

  // ---- services ---------------------------------------------------------

  async function loadServices() {
    const select = $('service');
    if (!select) return;
    try {
      const response = await fetch('/services');
      const payload = await response.json();
      const services = Array.isArray(payload && payload.services) ? payload.services : [];
      select.innerHTML = '';
      if (!services.length) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'Нет настроенных сервисов';
        select.appendChild(opt);
        select.disabled = true;
        setHint('serviceHint', 'Добавьте сервис в настройках (раздел «Сервисы»), затем обновите страницу.', 'warn');
        return;
      }
      const placeholder = document.createElement('option');
      placeholder.value = '';
      placeholder.textContent = 'Выберите сервис…';
      select.appendChild(placeholder);
      services.forEach((svc) => {
        const id = typeof svc === 'string' ? svc : String((svc && svc.id) || '');
        if (!id) return;
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = typeof svc === 'string' ? svc : ((svc && svc.title) || id);
        select.appendChild(opt);
      });
      if (services.length === 1) select.value = String(services[0].id || services[0]);
    } catch (error) {
      select.innerHTML = '<option value="">Не удалось загрузить сервисы</option>';
      select.disabled = true;
      setHint('serviceHint', `Ошибка загрузки списка сервисов: ${error.message}`, 'error');
    }
  }

  // ---- period -----------------------------------------------------------

  function applyPreset(minutes) {
    const end = new Date();
    end.setSeconds(0, 0);
    const start = new Date(end.getTime() - minutes * 60000);
    $('start').value = toLocalInputValue(start);
    $('end').value = toLocalInputValue(end);
    validateRange();
  }

  function validateRange() {
    const startVal = $('start').value;
    const endVal = $('end').value;
    setInvalid('start', false);
    setInvalid('end', false);
    if (!startVal || !endVal) {
      setHint('rangeHint', 'Укажите начало и окончание теста или выберите быстрый период.');
      return false;
    }
    const startMs = new Date(startVal).getTime();
    const endMs = new Date(endVal).getTime();
    if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) {
      setHint('rangeHint', 'Некорректная дата.', 'error');
      return false;
    }
    if (endMs <= startMs) {
      setInvalid('end', true);
      setHint('rangeHint', 'Окончание должно быть позже начала.', 'error');
      return false;
    }
    const minutes = Math.round((endMs - startMs) / 60000);
    const human = minutes >= 60 ? `${Math.floor(minutes / 60)} ч ${String(minutes % 60).padStart(2, '0')} мин` : `${minutes} мин`;
    if (minutes > LONG_WINDOW_WARN_MINUTES) {
      setHint('rangeHint', `Окно ${human}: длинные интервалы обрабатываются дольше и агрегируются грубее.`, 'warn');
    } else {
      setHint('rangeHint', `Длительность окна: ${human}.`);
    }
    return true;
  }

  function highlightPreset() {
    const startVal = $('start').value;
    const endVal = $('end').value;
    const diff = (startVal && endVal) ? Math.round((new Date(endVal) - new Date(startVal)) / 60000) : null;
    document.querySelectorAll('#rangePresets .seg-btn').forEach((btn) => {
      const active = diff !== null && Number(btn.dataset.minutes) === diff;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
  }

  // ---- LLM toggle -------------------------------------------------------

  function selectedLlm() {
    return $('use_llm_btn_yes').classList.contains('active');
  }

  function selectLlm(enabled) {
    const yes = $('use_llm_btn_yes');
    const no = $('use_llm_btn_no');
    yes.classList.toggle('active', enabled);
    no.classList.toggle('active', !enabled);
    yes.setAttribute('aria-pressed', enabled ? 'true' : 'false');
    no.setAttribute('aria-pressed', enabled ? 'false' : 'true');
  }

  // ---- publication target (web only / web + Confluence template) --------

  function selectedTarget() {
    const active = document.querySelector('#targetGroup .seg-btn.active');
    return active ? active.dataset.target : 'web';
  }

  function selectTarget(target) {
    document.querySelectorAll('#targetGroup .seg-btn').forEach((btn) => {
      const on = btn.dataset.target === target;
      btn.classList.toggle('active', on);
      btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
  }

  // ---- validation & submit ---------------------------------------------

  function selectedTestType() {
    const checked = document.querySelector('input[name="test_type"]:checked');
    return checked ? checked.value : '';
  }

  function validateForm() {
    let ok = validateRange();
    const service = $('service').value;
    setInvalid('service', !service);
    if (!service) {
      setHint('serviceHint', 'Выберите сервис.', 'error');
      ok = false;
    } else {
      setHint('serviceHint', SERVICE_HINT);
    }
    if (!selectedTestType()) {
      setHint('testTypeHint', 'Выберите тип теста: от него зависят инструкции анализа.', 'error');
      ok = false;
    } else {
      setHint('testTypeHint', TEST_TYPE_HINT);
    }
    setHint('runNameHint', '');
    setInvalid('run_name', false);
    return ok;
  }

  async function createReport(event) {
    event.preventDefault();
    const formError = $('formError');
    formError.textContent = '';
    if (!validateForm()) {
      formError.textContent = 'Заполните выделенные поля.';
      return;
    }
    const toConfluence = selectedTarget() === 'confluence';
    const payload = {
      start: LL.localInputToIso($('start').value),
      end: LL.localInputToIso($('end').value),
      service: $('service').value,
      project_area: LL.activeProjectArea || '',
      test_type: selectedTestType(),
      use_llm: selectedLlm(),
      run_name: ($('run_name').value || '').trim(),
      save_to_db: true,
      // The template flow also stores metrics and analysis, so the web report is available too.
      web_only: !toConfluence
    };
    const button = $('createBtn');
    button.disabled = true;
    try {
      const resp = await fetch('/create_report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const result = await resp.json();
      if (resp.ok && result.status === 'accepted' && result.job_id) {
        showProgressForJob(result.job_id, {
          runName: result.run_name || payload.run_name,
          service: payload.service,
          useLlm: payload.use_llm,
          confluence: toConfluence
        });
        return;
      }
      const message = result.message || result.error || 'Не удалось запустить задачу';
      if (/уже существует/i.test(message)) {
        setInvalid('run_name', true);
        setHint('runNameHint', message, 'error');
        $('run_name').focus();
      } else {
        formError.textContent = message;
      }
    } catch (error) {
      formError.textContent = `Ошибка создания отчёта: ${error.message}`;
    } finally {
      button.disabled = false;
    }
  }

  // ---- progress ---------------------------------------------------------

  function phaseIndexFor(pct) {
    let idx = 0;
    activePhases.forEach((phase, i) => { if (pct >= phase.from) idx = i; });
    return idx;
  }

  function renderPhases(pct, status) {
    const currentIdx = status === 'done' ? activePhases.length : phaseIndexFor(pct);
    document.querySelectorAll('#phaseList li').forEach((li) => {
      if (li.hidden) return;
      const idx = activePhases.findIndex((p) => p.key === li.dataset.phase);
      li.classList.remove('done', 'current', 'failed');
      if (status === 'done' || idx < currentIdx) li.classList.add('done');
      else if (idx === currentIdx) li.classList.add(status === 'error' ? 'failed' : 'current');
    });
  }

  function setLlmPhasesVisible(visible) {
    document.querySelectorAll('#phaseList li[data-llm-only]').forEach((li) => {
      li.hidden = !visible;
    });
  }

  function setPhaseMode(confluence) {
    activePhases = confluence ? CONFLUENCE_PHASES : WEB_PHASES;
    document.querySelectorAll('#phaseList li[data-confluence-only]').forEach((li) => {
      li.hidden = !confluence;
    });
    const list = $('phaseList');
    if (list) list.hidden = false;
  }

  function showProgressForJob(jobId, options) {
    const opts = options || {};
    currentJobOpts = opts;
    const form = $('reportForm');
    if (form) form.style.display = 'none';
    $('progressContainer').style.display = '';
    $('progressTitle').textContent = opts.runName ? `Формирование отчёта «${opts.runName}»` : 'Формирование отчёта';
    setPhaseMode(!!opts.confluence);
    if (typeof opts.useLlm === 'boolean') setLlmPhasesVisible(opts.useLlm);
    const jobUrl = `/new?job=${encodeURIComponent(jobId)}`;
    try { history.replaceState(null, '', jobUrl); } catch (e) { /* history API unavailable */ }
    const jobLink = $('jobLink');
    jobLink.href = jobUrl;
    jobLink.style.display = '';
    $('webLink').style.display = 'none';
    $('confluenceLink').style.display = 'none';
    $('newReportLink').style.display = 'none';
    $('progressErrorDetails').style.display = 'none';
    renderPhases(0, 'running');
    startPolling(jobId);
  }

  function isConfluenceUrl(url) {
    return /^https?:\/\//i.test(String(url || ''));
  }

  function showResultLinks(job) {
    const webLink = $('webLink');
    const confluenceLink = $('confluenceLink');
    const webHref = [job.page_url, job.report_url].map((value) => String(value || '')).find((url) => url.startsWith('/reports/')) || '';
    if (isConfluenceUrl(job.report_url)) {
      confluenceLink.href = job.report_url;
      confluenceLink.style.display = 'inline-block';
      if (webHref) {
        webLink.href = webHref;
        webLink.style.display = 'inline-block';
      }
    } else if (webHref || job.report_url) {
      webLink.href = webHref || job.report_url;
      webLink.style.display = 'inline-block';
    }
  }

  function applyJobState(job) {
    const pct = Math.max(0, Math.min(100, Number(job.progress) || 0));
    const bar = $('progressBarFill');
    bar.style.width = `${pct}%`;
    bar.classList.toggle('error', job.status === 'error');
    $('progressPct').textContent = `${pct}%`;
    $('progressMessage').textContent = job.message || '';
    if (job.run_name) $('progressTitle').textContent = `Формирование отчёта «${job.run_name}»`;
    renderPhases(pct, job.status);
    if (job.status === 'done') {
      showResultLinks(job);
      $('progressMessage').textContent = 'Отчёт готов.';
      $('newReportLink').style.display = '';
      return true;
    }
    if (job.status === 'error') {
      $('progressMessage').textContent = job.message || 'Ошибка выполнения задачи';
      if (job.error) {
        $('progressErrorText').textContent = job.error;
        $('progressErrorDetails').style.display = '';
      }
      $('newReportLink').style.display = '';
      return true;
    }
    return false;
  }

  function startPolling(jobId) {
    if (progressTimer) clearInterval(progressTimer);
    const tick = async () => {
      try {
        const r = await fetch(`/job_status/${encodeURIComponent(jobId)}`);
        if (r.status === 404) {
          clearInterval(progressTimer);
          $('progressMessage').textContent = 'Задача не найдена: возможно, сервер был перезапущен до сохранения статуса. Проверьте архив.';
          $('newReportLink').style.display = '';
          return;
        }
        if (!r.ok) return;
        const job = await r.json();
        if (applyJobState(job)) {
          clearInterval(progressTimer);
          loadActiveJobs();
        }
      } catch (e) {
        // transient network error: keep polling
      }
    };
    tick();
    progressTimer = setInterval(tick, POLL_INTERVAL_MS);
  }

  // ---- active jobs panel -----------------------------------------------

  function jobStatusLabel(status) {
    if (status === 'running') return 'В работе';
    if (status === 'error') return 'Ошибка';
    if (status === 'done') return 'Готово';
    return status || '';
  }

  function renderJobRow(job) {
    const row = document.createElement('div');
    row.className = 'job-row';
    const pill = document.createElement('span');
    pill.className = `pill ${job.status}`;
    pill.textContent = jobStatusLabel(job.status);
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
    if (job.status === 'running') bits.push(`${Number(job.progress) || 0}% — ${job.message || ''}`);
    else bits.push(job.message || '');
    if (job.updated_at) bits.push(LL.formatDateTime(job.updated_at));
    meta.textContent = bits.filter(Boolean).join(' · ');
    info.appendChild(name);
    info.appendChild(meta);
    const link = document.createElement('a');
    if (job.status === 'done' && job.report_url) {
      link.href = job.report_url;
      link.textContent = 'Открыть отчёт';
    } else if (job.kind === 'report') {
      link.href = `/new?job=${encodeURIComponent(job.job_id)}`;
      link.textContent = job.status === 'running' ? 'Следить' : 'Подробности';
    } else if (job.page_url) {
      link.href = job.page_url;
      link.textContent = 'Открыть страницу';
    }
    row.appendChild(pill);
    row.appendChild(info);
    row.appendChild(link);
    return row;
  }

  async function loadActiveJobs() {
    const panel = $('activeJobs');
    const list = $('activeJobsList');
    if (!panel || !list) return;
    try {
      const resp = await fetch('/jobs?status=running,error&limit=10');
      if (!resp.ok) return;
      const payload = await resp.json();
      const jobs = Array.isArray(payload && payload.jobs) ? payload.jobs : [];
      list.innerHTML = '';
      if (!jobs.length) {
        panel.style.display = 'none';
        if (jobsTimer) { clearInterval(jobsTimer); jobsTimer = null; }
        return;
      }
      jobs.forEach((job) => list.appendChild(renderJobRow(job)));
      const running = jobs.filter((j) => j.status === 'running').length;
      $('activeJobsNote').textContent = running ? `в работе: ${running}` : 'завершились с ошибкой';
      panel.style.display = '';
      if (running && !jobsTimer) jobsTimer = setInterval(loadActiveJobs, JOBS_REFRESH_MS);
      if (!running && jobsTimer) { clearInterval(jobsTimer); jobsTimer = null; }
    } catch (e) {
      // panel is auxiliary; the form keeps working without it
    }
  }

  // ---- wiring -----------------------------------------------------------

  document.addEventListener('DOMContentLoaded', async () => {
    $('tzLabel').textContent = LL.timeZoneLabel();
    await LL.initProjectArea();
    await loadServices();
    loadActiveJobs();

    document.querySelectorAll('#rangePresets .seg-btn').forEach((btn) => {
      btn.addEventListener('click', () => { applyPreset(Number(btn.dataset.minutes)); highlightPreset(); });
    });
    document.querySelectorAll('#targetGroup .seg-btn').forEach((btn) => {
      btn.addEventListener('click', () => selectTarget(btn.dataset.target));
    });
    ['start', 'end'].forEach((id) => {
      $(id).addEventListener('change', () => { validateRange(); highlightPreset(); });
    });
    $('service').addEventListener('change', () => {
      if ($('service').value) { setInvalid('service', false); setHint('serviceHint', SERVICE_HINT); }
    });
    document.querySelectorAll('input[name="test_type"]').forEach((radio) => {
      radio.addEventListener('change', () => setHint('testTypeHint', TEST_TYPE_HINT));
    });
    $('run_name').addEventListener('input', () => { setInvalid('run_name', false); setHint('runNameHint', ''); });
    $('use_llm_btn_yes').addEventListener('click', () => selectLlm(true));
    $('use_llm_btn_no').addEventListener('click', () => selectLlm(false));
    $('reportForm').addEventListener('submit', createReport);

    validateRange();

    const resumeJobId = new URLSearchParams(location.search).get('job');
    if (resumeJobId) showProgressForJob(resumeJobId, {});
  });
})();
