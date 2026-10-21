// SPDX-License-Identifier: Apache-2.0
// New report page logic (templates/index.html)
(function () {
  const LL = window.LoadLens;
  const POLL_INTERVAL_MS = 1500;
  const JOBS_REFRESH_MS = 5000;
  const LONG_WINDOW_WARN_MINUTES = 24 * 60;
  // Default hints under the service and test type fields (the same keys as templates/index.html); read lazily,
  // so the dictionary of the page is used, not the one present when the file loaded.
  const serviceHint = () => LL.t('new.service.hint');
  const testTypeHint = () => LL.t('new.test_type.hint');
  // Test window hint: how far back /lt_windows looks and how many chips are shown.
  const WINDOWS_HOURS = 6;
  const WINDOWS_MAX_CHIPS = 5;
  // Progress card: "usually about N min" is the median of this many latest finished report jobs.
  const TYPICAL_SAMPLE = 10;
  const TYPICAL_MIN_SAMPLES = 3;
  const JOB_KIND_REPORT = 'report';

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
  // Service combobox (static/js/service_picker.js); the hidden #service input keeps the value.
  let picker = null;

  const $ = (id) => document.getElementById(id);
  const pad2 = (n) => String(n).padStart(2, '0');

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
    if (id === 'service' && picker) picker.setInvalid(invalid);
  }

  function toLocalInputValue(date) {
    return `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}T${pad2(date.getHours())}:${pad2(date.getMinutes())}`;
  }

  // ---- services ---------------------------------------------------------

  async function loadServices() {
    const hidden = $('service');
    const host = $('serviceCombo');
    if (!hidden || !host || !LL.servicePicker) return;
    picker = LL.servicePicker({ host, input: hidden, inputId: 'serviceInput', describedBy: 'serviceHint' });
    try {
      const services = await picker.ready;
      if (!services.length) setHint('serviceHint', LL.t('new.service.empty'), 'warn');
    } catch (error) {
      setHint('serviceHint', LL.t('new.service.load_error', { error: error.message }), 'error');
    }
  }

  // ---- detected test windows (GET /lt_windows) ---------------------------

  let windowsSeq = 0;

  function overlapMs(a, b) {
    return Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start));
  }

  // Chips of the /lt_windows answer, newest first: stretches with RPS above zero and finished load test runs.
  // A run overlapping more than half of a stretch (or of itself) merges into it: the run's exact times and
  // name win, the RPS peak of the stretch stays.
  function mergeWindows(windows, runs, maxChips) {
    const valid = (w) => Number.isFinite(w.start) && Number.isFinite(w.end) && w.end > w.start;
    const peakOf = (value) => (value === null || value === undefined || value === '' || !Number.isFinite(Number(value)) ? null : Number(value));
    const items = (Array.isArray(windows) ? windows : []).filter(Boolean).map((w) => ({
      start: Date.parse(w.start), end: Date.parse(w.end), peak: peakOf(w.rps_peak), name: '', ongoing: !!w.ongoing
    })).filter(valid);
    (Array.isArray(runs) ? runs : []).filter(Boolean).forEach((run) => {
      const item = { start: Date.parse(run.start), end: Date.parse(run.end), peak: null, name: String(run.run_name || ''), ongoing: false };
      if (!valid(item)) return;
      const hit = items.find((w) => !w.name && overlapMs(w, item) > 0.5 * Math.min(w.end - w.start, item.end - item.start));
      if (hit) Object.assign(hit, { start: item.start, end: item.end, name: item.name, ongoing: false });
      else items.push(item);
    });
    return items.sort((a, b) => b.start - a.start).slice(0, maxChips || WINDOWS_MAX_CHIPS);
  }

  // '' for today, «yesterday», otherwise «14.10» in ru and «14/10» in en (browser time zone).
  function dayText(ms, nowMs) {
    const d = new Date(ms);
    const key = (x) => `${x.getFullYear()}-${x.getMonth()}-${x.getDate()}`;
    const today = new Date(nowMs);
    if (key(d) === key(today)) return '';
    const yesterday = new Date(nowMs);
    yesterday.setDate(yesterday.getDate() - 1);
    if (key(d) === key(yesterday)) return LL.t('new.windows.yesterday');
    return LL.t('new.windows.day_month', { day: pad2(d.getDate()), month: pad2(d.getMonth() + 1) });
  }

  // «release-2.5 · 15:57–16:57 (1 h 00 min, peak 249 RPS)»; the clock times are those the chip puts into
  // the fields (start rounded down, end rounded up to the minute).
  function windowLabel(item, nowMs) {
    const now = Number.isFinite(nowMs) ? nowMs : Date.now();
    const clock = (ms) => { const d = new Date(ms); return `${pad2(d.getHours())}:${pad2(d.getMinutes())}`; };
    const startMs = Math.floor(item.start / 60000) * 60000;
    const endMs = Math.ceil(item.end / 60000) * 60000;
    const startDay = dayText(startMs, now);
    const endDay = dayText(endMs, now);
    const range = `${startDay ? `${startDay} ` : ''}${clock(startMs)}–${endDay && endDay !== startDay ? `${endDay} ` : ''}${clock(endMs)}`;
    const bits = [LL.formatDuration(item.end - item.start)];
    if (item.peak !== null && item.peak > 0) bits.push(LL.t('new.windows.peak', { rps: item.peak >= 10 ? Math.round(item.peak) : Math.round(item.peak * 10) / 10 }));
    if (item.ongoing) bits.push(LL.t('new.windows.ongoing'));
    return `${item.name ? `${item.name} · ` : ''}${range} (${bits.join(', ')})`;
  }

  // datetime-local values of a chip: the start rounded down and the end rounded up to the minute.
  function windowInputs(item) {
    return {
      start: toLocalInputValue(new Date(Math.floor(item.start / 60000) * 60000)),
      end: toLocalInputValue(new Date(Math.ceil(item.end / 60000) * 60000))
    };
  }

  function highlightWindowChips() {
    const startVal = $('start').value;
    const endVal = $('end').value;
    document.querySelectorAll('#ltWindowsChips .lt-chip').forEach((chip) => {
      const on = !!startVal && chip.dataset.start === startVal && chip.dataset.end === endVal;
      chip.classList.toggle('active', on);
      chip.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
  }

  function renderWindows(items) {
    const host = $('ltWindowsChips');
    host.replaceChildren();
    items.forEach((item) => {
      const values = windowInputs(item);
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'lt-chip';
      chip.textContent = windowLabel(item);
      chip.title = LL.t('new.windows.chip_title');
      chip.dataset.start = values.start;
      chip.dataset.end = values.end;
      chip.setAttribute('aria-pressed', 'false');
      chip.addEventListener('click', () => {
        $('start').value = chip.dataset.start;
        $('end').value = chip.dataset.end;
        validateRange();
        highlightPreset();
      });
      host.appendChild(chip);
    });
    $('ltWindows').hidden = !items.length;
    highlightWindowChips();
  }

  // Quiet one-line status under the chips; the error text of the source goes into the tooltip.
  function windowsHint(text, tone, detail) {
    setHint('ltWindowsHint', text, tone);
    const el = $('ltWindowsHint');
    if (!el) return;
    if (detail) el.title = String(detail);
    else el.removeAttribute('title');
  }

  async function loadWindows(service) {
    const seq = ++windowsSeq;
    renderWindows([]);
    if (!service) { windowsHint(''); return; }
    windowsHint(LL.t('new.windows.loading'));
    try {
      const resp = await fetch(`/lt_windows?service=${encodeURIComponent(service)}&hours=${WINDOWS_HOURS}`);
      const data = await resp.json().catch(() => ({}));
      if (seq !== windowsSeq) return;
      if (!resp.ok) throw new Error(data.error || data.message || `HTTP ${resp.status}`);
      const items = mergeWindows(data.windows, data.runs);
      renderWindows(items);
      const message = String(data.message || '').slice(0, 200).replace(/\.$/, '');
      const lower = message ? message.charAt(0).toLowerCase() + message.slice(1) : '';
      if (data.ok === false && lower) windowsHint(LL.t('new.windows.undetected', { reason: lower }), 'warn', data.detail);
      else if (!items.length) windowsHint(LL.t('new.windows.none', { hours: data.hours || WINDOWS_HOURS }));
      else windowsHint('');
    } catch (error) {
      if (seq !== windowsSeq) return;
      windowsHint(LL.t('new.windows.unavailable'), 'warn', error.message);
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
      setHint('rangeHint', LL.t('new.range.empty'));
      return false;
    }
    const startMs = new Date(startVal).getTime();
    const endMs = new Date(endVal).getTime();
    if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) {
      setHint('rangeHint', LL.t('new.range.invalid'), 'error');
      return false;
    }
    if (endMs <= startMs) {
      setInvalid('end', true);
      setHint('rangeHint', LL.t('new.range.order'), 'error');
      return false;
    }
    const minutes = Math.round((endMs - startMs) / 60000);
    const human = minutes >= 60
      ? LL.t('new.range.hours_minutes', { hours: Math.floor(minutes / 60), minutes: String(minutes % 60).padStart(2, '0') })
      : LL.t('new.range.minutes', { minutes });
    if (minutes > LONG_WINDOW_WARN_MINUTES) {
      setHint('rangeHint', LL.t('new.range.long', { duration: human }), 'warn');
    } else {
      setHint('rangeHint', LL.t('new.range.duration', { duration: human }));
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
    highlightWindowChips();
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
      setHint('serviceHint', LL.t('new.service.required'), 'error');
      ok = false;
    } else {
      setHint('serviceHint', serviceHint());
    }
    if (!selectedTestType()) {
      setHint('testTypeHint', LL.t('new.test_type.required'), 'error');
      ok = false;
    } else {
      setHint('testTypeHint', testTypeHint());
    }
    setHint('runNameHint', '');
    setInvalid('run_name', false);
    return ok;
  }

  // /create_report answers code "run_name_taken" when the run name is in use for the service.
  function runNameTaken(result) {
    return !!result && result.code === 'run_name_taken';
  }

  async function createReport(event) {
    event.preventDefault();
    const formError = $('formError');
    formError.textContent = '';
    if (!validateForm()) {
      formError.textContent = LL.t('new.form.invalid');
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
        if (LL.servicePicker) LL.servicePicker.remember(payload.service);
        showProgressForJob(result.job_id, {
          runName: result.run_name || payload.run_name,
          service: payload.service,
          useLlm: payload.use_llm,
          confluence: toConfluence
        });
        return;
      }
      const message = result.message || result.error || LL.t('new.form.start_failed');
      if (runNameTaken(result)) {
        setInvalid('run_name', true);
        setHint('runNameHint', message, 'error');
        $('run_name').focus();
      } else {
        formError.textContent = message;
      }
    } catch (error) {
      formError.textContent = LL.t('new.form.create_error', { error: error.message });
    } finally {
      button.disabled = false;
    }
  }

  // ---- progress timing ----------------------------------------------------

  let elapsedTimer = null;
  let elapsedStart = 0;
  let jobFinished = false;

  // «2:15», «1:02:03».
  function formatElapsed(ms) {
    const total = Math.max(0, Math.floor((Number(ms) || 0) / 1000));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const seconds = total % 60;
    return hours ? `${hours}:${pad2(minutes)}:${pad2(seconds)}` : `${minutes}:${pad2(seconds)}`;
  }

  // Median run time (ms, created_at to updated_at) of the latest ``sample`` finished jobs of ``kind``
  // (the list is newest first, as /jobs returns it); null with fewer than ``minSamples`` usable jobs.
  function medianDurationMs(jobs, kind, sample, minSamples) {
    const values = (Array.isArray(jobs) ? jobs : [])
      .filter((job) => job && job.status === 'done' && (!kind || job.kind === kind))
      .slice(0, sample || TYPICAL_SAMPLE)
      .map((job) => Date.parse(job.updated_at) - Date.parse(job.created_at))
      .filter((ms) => Number.isFinite(ms) && ms > 0)
      .sort((a, b) => a - b);
    if (!values.length || values.length < (minSamples || TYPICAL_MIN_SAMPLES)) return null;
    const mid = Math.floor(values.length / 2);
    return values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
  }

  function typicalText(ms) {
    if (!Number.isFinite(ms) || ms <= 0) return '';
    return LL.t('new.progress.typical', { minutes: Math.max(1, Math.round(ms / 60000)) });
  }

  function tickElapsed() {
    const el = $('progressElapsed');
    if (el) el.textContent = LL.t('new.progress.running', { elapsed: formatElapsed(Date.now() - elapsedStart) });
  }

  function startElapsed() {
    jobFinished = false;
    elapsedStart = Date.now();
    clearInterval(elapsedTimer);
    tickElapsed();
    elapsedTimer = setInterval(tickElapsed, 1000);
    loadTypicalDuration();
  }

  // The server's creation time wins for a resumed job; one ahead of this browser's clock is ignored.
  function syncElapsed(job) {
    const created = Date.parse(job && job.created_at);
    if (Number.isFinite(created) && created <= Date.now() + 5000) elapsedStart = created;
  }

  function stopElapsed(job) {
    jobFinished = true;
    clearInterval(elapsedTimer);
    elapsedTimer = null;
    const el = $('progressElapsed');
    const typical = $('progressTypical');
    if (typical) typical.hidden = true;
    if (!el) return;
    if (!job) { el.textContent = ''; return; }
    const created = Date.parse(job.created_at);
    const updated = Date.parse(job.updated_at);
    const took = Number.isFinite(created) && Number.isFinite(updated) && updated >= created ? updated - created : Date.now() - elapsedStart;
    el.textContent = LL.t(job.status === 'done' ? 'new.progress.took' : 'new.progress.passed', { elapsed: formatElapsed(took) });
  }

  async function loadTypicalDuration() {
    const el = $('progressTypical');
    if (!el) return;
    el.hidden = true;
    try {
      const resp = await fetch(`/jobs?status=done&limit=${TYPICAL_SAMPLE * 2}`);
      if (!resp.ok) return;
      const payload = await resp.json();
      const ms = medianDurationMs(payload && payload.jobs, JOB_KIND_REPORT);
      if (ms && !jobFinished) {
        el.textContent = typicalText(ms);
        el.hidden = false;
      }
    } catch (e) {
      // the typical duration is a hint only
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
    $('progressTitle').textContent = opts.runName ? LL.t('new.progress.title_named', { name: opts.runName }) : LL.t('new.progress.title');
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
    startElapsed();
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
      confluenceLink.style.display = 'inline-flex';
      if (webHref) {
        webLink.href = webHref;
        webLink.style.display = 'inline-flex';
      }
    } else if (webHref || job.report_url) {
      webLink.href = webHref || job.report_url;
      webLink.style.display = 'inline-flex';
    }
  }

  function applyJobState(job) {
    const pct = Math.max(0, Math.min(100, Number(job.progress) || 0));
    const bar = $('progressBarFill');
    bar.style.width = `${pct}%`;
    bar.classList.toggle('error', job.status === 'error');
    $('progressPct').textContent = `${pct}%`;
    $('progressMessage').textContent = job.message || '';
    if (job.run_name) $('progressTitle').textContent = LL.t('new.progress.title_named', { name: job.run_name });
    renderPhases(pct, job.status);
    if (job.status === 'done' || job.status === 'error') stopElapsed(job);
    else syncElapsed(job);
    if (job.status === 'done') {
      showResultLinks(job);
      $('progressMessage').textContent = LL.t('new.progress.done');
      $('newReportLink').style.display = '';
      return true;
    }
    if (job.status === 'error') {
      $('progressMessage').textContent = job.message || LL.t('new.progress.failed');
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
          stopElapsed(null);
          $('progressMessage').textContent = LL.t('new.progress.not_found');
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
    if (status === 'running') return LL.t('new.job.status_running');
    if (status === 'error') return LL.t('new.job.status_error');
    if (status === 'done') return LL.t('new.job.status_done');
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
    name.textContent = job.run_name || LL.t('new.job.untitled');
    const meta = document.createElement('div');
    meta.className = 'meta';
    const bits = [];
    if (job.kind === 'confluence') bits.push(LL.t('new.job.kind_confluence'));
    if (job.kind === 'forecast_confluence') bits.push(LL.t('new.job.kind_forecast_confluence'));
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
      link.textContent = LL.t('new.action.open_report');
    } else if (job.kind === 'report') {
      link.href = `/new?job=${encodeURIComponent(job.job_id)}`;
      link.textContent = LL.t(job.status === 'running' ? 'new.job.follow' : 'new.job.details');
    } else if (job.page_url) {
      link.href = job.page_url;
      link.textContent = LL.t('new.job.open_page');
    }
    const actions = document.createElement('span');
    actions.className = 'job-actions';
    actions.appendChild(link);
    const dismiss = LL.jobDismissButton(job, loadActiveJobs);
    if (dismiss) actions.appendChild(dismiss);
    row.appendChild(pill);
    row.appendChild(info);
    row.appendChild(actions);
    return row;
  }

  // Finished rows currently shown: "Clear finished" hides exactly these.
  let finishedJobIds = [];

  async function clearFinishedJobs() {
    const btn = $('activeJobsClear');
    btn.disabled = true;
    await LL.dismissJobs(finishedJobIds);
    btn.disabled = false;
    loadActiveJobs();
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
      finishedJobIds = jobs.filter((j) => j.status !== 'running').map((j) => j.job_id);
      $('activeJobsClear').hidden = !finishedJobIds.length;
      if (!jobs.length) {
        panel.style.display = 'none';
        if (jobsTimer) { clearInterval(jobsTimer); jobsTimer = null; }
        return;
      }
      jobs.forEach((job) => list.appendChild(renderJobRow(job)));
      const running = jobs.filter((j) => j.status === 'running').length;
      $('activeJobsNote').textContent = running ? LL.t('new.jobs.running', { n: running }) : LL.t('new.jobs.failed_only');
      panel.style.display = '';
      if (running && !jobsTimer) jobsTimer = setInterval(loadActiveJobs, JOBS_REFRESH_MS);
      if (!running && jobsTimer) { clearInterval(jobsTimer); jobsTimer = null; }
    } catch (e) {
      // panel is auxiliary; the form keeps working without it
    }
  }

  // Pure helpers for tests (tests/test_ui_forms_js.py).
  LL.newReport = {
    _internals: { mergeWindows, windowLabel, windowInputs, dayText, formatElapsed, medianDurationMs, typicalText, jobStatusLabel, renderJobRow, runNameTaken }
  };

  // ---- wiring -----------------------------------------------------------

  document.addEventListener('DOMContentLoaded', async () => {
    $('tzLabel').textContent = LL.timeZoneLabel();
    await LL.initProjectArea();
    await loadServices();
    if ($('service').value) loadWindows($('service').value);
    $('activeJobsClear').addEventListener('click', clearFinishedJobs);
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
      if ($('service').value) { setInvalid('service', false); setHint('serviceHint', serviceHint()); }
      loadWindows($('service').value);
    });
    document.querySelectorAll('input[name="test_type"]').forEach((radio) => {
      radio.addEventListener('change', () => setHint('testTypeHint', testTypeHint()));
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
