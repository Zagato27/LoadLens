// Run page: status of a load test run, live RPS / errors / p95 / users, instances, stop buttons and logs.
(function () {
  const LL = window.LoadLens;
  const ui = LL.ui;
  const $ = (id) => document.getElementById(id);
  const RUN_POLL_MS = 3000;
  const LIVE_POLL_MS = 5000;
  const LIVE_ROLES = ['lt.rps_total', 'lt.error_rate_pct', 'lt.p95_total', 'lt.vus'];
  // A minute without new points: the metrics do not reach the storage (or the tool stopped writing).
  const STALE_AFTER_SEC = 60;
  // While the page shows metrics without the run tag, every this many polls a full read checks the tag again.
  const RECHECK_PLAIN_EVERY = 12;
  const LOG_POLL_MS = 5000;

  const STATUS_TONE = {
    starting: 'running', running: 'running', stopping: 'warn', draining: 'running', reporting: 'running',
    done: 'done', failed: 'fail', cancelled: 'na'
  };
  const INSTANCE_TONE = {
    pending: 'na', starting: 'running', running: 'running', finished: 'done', checks_failed: 'warn',
    failed: 'fail', stopped: 'warn', timed_out: 'warn', lost: 'fail', cancelled: 'na'
  };

  const root = $('runRoot');
  const runId = root ? root.dataset.runId : '';
  const state = { run: null, charts: {}, runTimer: null, liveTimer: null, logTimer: null, logIndex: null, liveLoaded: false, instancesSignature: '', live: freshLive(0) };

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  async function fetchJson(url, options) {
    const resp = await fetch(url, options);
    let data = {};
    try { data = await resp.json(); } catch (e) { data = {}; }
    if (!resp.ok) throw new Error(data.message || data.error || `HTTP ${resp.status}`);
    return data;
  }

  function fmtTime(iso) {
    return iso ? LL.formatDateTime(iso) : '—';
  }

  function duration(fromIso, toIso) {
    if (!fromIso) return '';
    const ms = (toIso ? new Date(toIso) : new Date()) - new Date(fromIso);
    if (!(ms >= 0)) return '';
    const total = Math.round(ms / 1000);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    return h ? `${h} ч ${m} мин` : (m ? `${m} мин ${s} с` : `${s} с`);
  }

  // ---- run -------------------------------------------------------------------------------------

  function renderRun(run) {
    state.run = run;
    document.title = `${run.run_name} · Прогон теста`;
    $('runTitle').textContent = run.run_name;
    const crumb = document.getElementById('runCrumb');
    if (crumb) crumb.textContent = run.run_name;
    const typeLabel = run.test_type ? LL.testTypeLabel(run.test_type) : 'тип не указан';
    const window = run.started_at ? `${fmtTime(run.started_at)} — ${run.finished_at ? fmtTime(run.finished_at) : 'идёт'} (${duration(run.started_at, run.finished_at)})` : `создан ${fmtTime(run.created_at)}`;
    $('runMeta').textContent = `${run.service} · ${typeLabel} · тег ${run.run_tag} · ${run.owner || 'без владельца'} · ${window}`;
    $('runMessage').textContent = run.message || '';
    const status = $('runStatus');
    status.className = `pill pill-lg ${STATUS_TONE[run.status] || 'na'}`;
    status.textContent = run.status_title;
    const stopAll = $('stopAll');
    stopAll.style.display = run.loading ? '' : 'none';
    stopAll.disabled = run.stop_requested;
    stopAll.textContent = run.stop_requested ? 'Останавливается…' : 'Остановить прогон';
    $('repeatLink').href = `/launch?from=${encodeURIComponent(run.run_id)}`;
    const report = $('reportLink');
    const reportUrl = run.report_url || (run.report && run.report.page_url);
    if (reportUrl) { report.href = reportUrl; report.style.display = ''; } else report.style.display = 'none';
    const job = $('jobLink');
    if (run.report_job_id && !reportUrl) {
      job.href = `/new?job=${encodeURIComponent(run.report_job_id)}`;
      job.textContent = run.report ? `Отчёт: ${run.report.progress}%` : 'Задача отчёта';
      job.style.display = '';
    } else job.style.display = 'none';
    renderInstances(run);
  }

  function renderInstances(run) {
    // Polling returns the same rows most of the time: redrawing them would swallow clicks and fold
    // the opened command blocks, so the table changes only when the rows do.
    const signature = JSON.stringify([run.loading, run.instances]);
    if (signature === state.instancesSignature) return;
    state.instancesSignature = signature;
    const body = $('instancesBody');
    const openCommands = new Set(Array.from(body.querySelectorAll('details[open]')).map((node) => node.dataset.index));
    body.replaceChildren();
    run.instances.forEach((inst) => {
      const tr = el('tr');
      const nameCell = el('td');
      nameCell.appendChild(el('strong', '', inst.name));
      if (inst.script) nameCell.appendChild(el('div', 'dim', inst.script));
      if (inst.command) {
        const details = el('details');
        details.dataset.index = String(inst.index);
        details.open = openCommands.has(String(inst.index));
        details.appendChild(el('summary', 'dim', inst.ci ? 'Параметры пайплайна' : 'Команда'));
        details.appendChild(el('pre', '', inst.command));
        if (inst.env_keys && inst.env_keys.length) details.appendChild(el('div', 'dim', `Переменные: ${inst.env_keys.join(', ')}`));
        nameCell.appendChild(details);
      }
      const where = el('td', '', `${inst.generator_title} · ${inst.tool_title || 'вручную'}`);
      if (inst.ci && inst.ci.ref) where.appendChild(el('div', 'dim', `ветка ${inst.ci.ref}`));
      if (inst.external_url && /^https?:\/\//.test(inst.external_url)) {
        const link = el('a', '', inst.ci && inst.ci.kind === 'jenkins' ? 'Открыть в Jenkins' : 'Открыть в GitLab');
        link.href = inst.external_url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        const line = el('div');
        line.appendChild(link);
        where.appendChild(line);
      }
      if (inst.start_delay_sec) where.appendChild(el('div', 'dim', `старт через ${inst.start_delay_sec} с`));
      const statusCell = el('td');
      statusCell.appendChild(el('span', `pill ${INSTANCE_TONE[inst.status] || 'na'}`, inst.status_title));
      if (inst.exit_code !== null && inst.exit_code !== undefined) statusCell.appendChild(el('div', 'dim', `код выхода ${inst.exit_code}`));
      if (inst.status_text && inst.active) statusCell.appendChild(el('div', 'dim', inst.status_text));
      if (inst.message) statusCell.appendChild(el('div', 'msg', inst.message));
      const timeCell = el('td', 'nowrap');
      timeCell.appendChild(el('div', '', inst.started_at ? fmtTime(inst.started_at) : '—'));
      if (inst.started_at) timeCell.appendChild(el('div', 'dim', duration(inst.started_at, inst.finished_at)));
      const actions = el('td');
      const box = el('div', 'inst-actions');
      if (inst.has_logs) {
        const logBtn = el('button', 'btn btn-sm', 'Журнал');
        logBtn.type = 'button';
        logBtn.addEventListener('click', () => openLog(inst.index));
        box.appendChild(logBtn);
      }
      if (inst.active && !inst.stop_requested && run.loading) {
        const stopBtn = el('button', 'btn btn-sm btn-danger', inst.manual ? 'Завершить' : 'Остановить');
        stopBtn.type = 'button';
        stopBtn.addEventListener('click', () => stop(inst.index, inst.name));
        box.appendChild(stopBtn);
      }
      actions.appendChild(box);
      tr.append(nameCell, where, statusCell, timeCell, actions);
      body.appendChild(tr);
    });
  }

  async function loadRun() {
    clearTimeout(state.runTimer);
    try {
      const run = await fetchJson(`/load_tests/runs/${encodeURIComponent(runId)}`);
      const wasActive = state.run && state.run.loading;
      renderRun(run);
      if (run.active) state.runTimer = setTimeout(loadRun, RUN_POLL_MS);
      if (!state.liveLoaded || (wasActive && !run.loading)) loadLive(true);
    } catch (e) {
      $('runMessage').textContent = `Прогон не загрузился: ${e.message}`;
      state.runTimer = setTimeout(loadRun, RUN_POLL_MS * 3);
    }
  }

  async function stop(index, name) {
    const whole = index === null || index === undefined;
    const ok = await ui.confirm({
      title: whole ? 'Остановить прогон' : `Остановить инстанс «${name}»`,
      message: whole
        ? 'Всем инстансам уйдёт сигнал остановки; не завершившиеся за отведённое время будут убиты. Отчёт построится по уже собранным метрикам.'
        : 'Инстансу уйдёт сигнал остановки. Остальные продолжат работу.',
      confirmText: 'Остановить',
      danger: true
    });
    if (!ok) return;
    try {
      const run = await fetchJson(`/load_tests/runs/${encodeURIComponent(runId)}/stop`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(whole ? {} : { index })
      });
      renderRun(run);
      ui.toast('Остановка запрошена', { tone: 'ok' });
      loadRun();
    } catch (e) {
      ui.toast(`Не удалось остановить: ${e.message}`, { tone: 'error' });
    }
  }

  // ---- live metrics ----------------------------------------------------------------------------

  const EVENT_COLORS = { start: '#2f9e44', finish: '#868e96', stop: '#f08c00', limit: '#e03131' };
  const INSTANCE_COLORS = ['#4dabf7', '#f783ac', '#ffd43b', '#63e6be', '#b197fc', '#ffa94d', '#74c0fc', '#e599f7'];

  function freshLive(windowMin) {
    return { window: windowMin, step: null, mode: '', series: {}, labels: {}, instances: {}, events: [], sla: {}, start: null, end: null, lastMs: null, polls: 0, rpsQuery: null };
  }

  function lastValue(points) {
    return points && points.length ? points[points.length - 1][1] : null;
  }

  function fmtNumber(value, digits) {
    if (value === null || value === undefined) return '—';
    return Number(value).toLocaleString('ru-RU', { maximumFractionDigits: digits });
  }

  function chartTime(value) {
    const d = new Date(value);
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;
  }

  // Points of a newer read replace the older ones from its first moment on; points before the window go.
  function mergePoints(old, fresh, fromMs) {
    const incoming = Array.isArray(fresh) ? fresh : [];
    const cut = incoming.length ? incoming[0][0] : null;
    const kept = (Array.isArray(old) ? old : []).filter((p) => (cut === null || p[0] < cut) && (fromMs === null || fromMs === undefined || p[0] >= fromMs));
    return kept.concat(incoming);
  }

  // «данные на 12:04:31, задержка 9 с»; a minute without new points is a warning.
  function freshness(lastMs, nowMs, active) {
    if (!lastMs) return { text: active ? 'данных пока нет' : '', stale: false };
    if (!active) return { text: `тест завершён, данные до ${chartTime(lastMs)}`, stale: false };
    const delay = Math.max(0, Math.round((nowMs - lastMs) / 1000));
    return { text: `данные на ${chartTime(lastMs)}, задержка ${delay} с`, stale: delay > STALE_AFTER_SEC };
  }

  // Threshold lines of the service SLA for each chart.
  function slaLines(sla) {
    const s = sla || {};
    const lines = { rps: [], p95: [] };
    if (s.target_rps) lines.rps.push({ axis: 'y', value: s.target_rps, label: `цель RPS ${fmtNumber(s.target_rps, 1)}`, color: '#2f9e44' });
    if (s.max_error_rate_pct) lines.rps.push({ axis: 'y1', value: s.max_error_rate_pct, label: `порог ошибок ${fmtNumber(s.max_error_rate_pct, 2)}%`, color: '#e03131' });
    if (s.max_p95_ms) lines.p95.push({ axis: 'y', value: s.max_p95_ms, label: `порог p95 ${fmtNumber(s.max_p95_ms, 0)} мс`, color: '#e03131' });
    return lines;
  }

  // An axis reaches a threshold when the data come within half of it; a far threshold stays a note on top.
  function axisRange(base, lines, maxValue) {
    const near = (lines || []).filter((line) => maxValue > 0 && line.value <= maxValue * 2);
    const top = Math.max(0, ...near.map((line) => line.value * 1.15));
    return top ? { ...base, suggestedMax: Math.max(top, base.suggestedMax || 0) } : base;
  }

  // Event and threshold lines drawn over the datasets (no annotation plugin in the bundle).
  const markersPlugin = {
    id: 'llMarkers',
    afterDatasetsDraw(chart) {
      const opts = chart.options.plugins.llMarkers || {};
      const area = chart.chartArea;
      const ctx = chart.ctx;
      if (!area) return;
      ctx.save();
      ctx.font = '11px sans-serif';
      ctx.lineWidth = 1;
      (opts.events || []).forEach((ev) => {
        const x = chart.scales.x.getPixelForValue(ev.ts);
        if (!(x >= area.left && x <= area.right)) return;
        ctx.strokeStyle = EVENT_COLORS[ev.kind] || '#868e96';
        ctx.setLineDash(ev.kind === 'limit' ? [6, 4] : [3, 3]);
        ctx.beginPath();
        ctx.moveTo(x, area.top);
        ctx.lineTo(x, area.bottom);
        ctx.stroke();
        if (opts.labels) {
          ctx.save();
          ctx.fillStyle = ctx.strokeStyle;
          ctx.translate(x + 3, area.top + 4);
          ctx.rotate(Math.PI / 2);
          ctx.fillText(ev.label, 0, 0);
          ctx.restore();
        }
      });
      let above = 0;
      (opts.lines || []).forEach((line) => {
        const scale = chart.scales[line.axis || 'y'];
        if (!scale) return;
        const y = scale.getPixelForValue(line.value);
        if (y < area.top) {
          // Far above the data: a note on top, in the middle (event labels sit at the edges).
          ctx.fillStyle = line.color;
          ctx.textAlign = 'center';
          ctx.fillText(`${line.label} (выше графика)`, (area.left + area.right) / 2, area.top + 12 + above * 14);
          ctx.textAlign = 'left';
          above += 1;
          return;
        }
        if (y > area.bottom) return;
        ctx.strokeStyle = line.color;
        ctx.fillStyle = line.color;
        ctx.setLineDash([8, 4]);
        ctx.beginPath();
        ctx.moveTo(area.left, y);
        ctx.lineTo(area.right, y);
        ctx.stroke();
        ctx.textAlign = 'right';
        ctx.fillText(line.label, area.right - 4, y - 4);
        ctx.textAlign = 'left';
      });
      ctx.restore();
    }
  };

  function chartOptions(colors, scales, markers) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'nearest', intersect: false, axis: 'x' },
      plugins: {
        legend: { labels: { color: colors.text, boxWidth: 12 } },
        tooltip: { callbacks: { title: (items) => (items.length ? chartTime(items[0].parsed.x) : '') } },
        llMarkers: markers
      },
      scales
    };
  }

  function drawChart(key, canvasId, datasets, scales, markers) {
    if (!window.Chart) return;
    const colors = ui.chartColors();
    const existing = state.charts[key];
    if (existing) {
      existing.data.datasets = datasets;
      existing.options.scales = scales;
      existing.options.plugins.llMarkers = markers;
      existing.update('none');
      return;
    }
    state.charts[key] = new window.Chart($(canvasId).getContext('2d'), {
      type: 'line',
      data: { datasets },
      options: chartOptions(colors, scales, markers),
      plugins: [markersPlugin]
    });
  }

  function renderLive(message) {
    const live = state.live;
    const active = !!(state.run && state.run.loading);
    $('tileRps').textContent = fmtNumber(lastValue(live.series['lt.rps_total']), 1);
    $('tileErr').textContent = fmtNumber(lastValue(live.series['lt.error_rate_pct']), 2);
    $('tileP95').textContent = fmtNumber(lastValue(live.series['lt.p95_total']), 0);
    $('tileVus').textContent = fmtNumber(lastValue(live.series['lt.vus']), 0);
    const fresh = freshness(live.lastMs, Date.now(), active);
    $('liveFresh').textContent = fresh.text;
    $('liveFresh').classList.toggle('stale', fresh.stale);
    const notes = [];
    if (message) notes.push(message);
    if (live.step) notes.push(`шаг ${live.step} с${live.mode === 'filtered' ? ', только метрики с тегом прогона' : ''}`);
    $('liveNote').textContent = notes.join(' · ');
    const query = $('liveQuery');
    const empty = !(live.series['lt.rps_total'] || []).length;
    query.hidden = !(empty && live.rpsQuery && live.rpsQuery.query);
    if (!query.hidden) {
      query.querySelector('summary').textContent = `Запрос RPS: ${live.rpsQuery.label}`;
      $('liveQueryText').textContent = live.rpsQuery.query;
    }
    if (!window.Chart) return;
    const colors = ui.chartColors();
    const toXY = (points) => (points || []).map((p) => ({ x: p[0], y: p[1] }));
    const x = {
      type: 'linear', min: live.start ? live.start * 1000 : undefined, max: live.end ? live.end * 1000 : undefined,
      ticks: { color: colors.text, callback: (value) => chartTime(value), maxTicksLimit: 8 }, grid: { color: colors.grid }
    };
    const axis = (title, extra) => ({ beginAtZero: true, title: { display: !!title, text: title, color: colors.text }, ticks: { color: colors.text }, grid: { color: colors.grid }, ...(extra || {}) });
    const lines = slaLines(live.sla);
    const dataMax = (...sets) => Math.max(0, ...sets.flat().map((p) => (p && Number.isFinite(p[1]) ? p[1] : 0)));
    const withLine = (base, list, axisKey, maxValue) => axisRange(base, list.filter((l) => (l.axis || 'y') === axisKey), maxValue);
    const instances = Object.keys(live.instances).map((tag) => live.instances[tag]);
    const rpsSets = [{ label: instances.length > 1 ? 'RPS, всего' : 'RPS', data: toXY(live.series['lt.rps_total']), borderColor: colors.accent, backgroundColor: colors.accentSoft, yAxisID: 'y', pointRadius: 0, borderWidth: 2, spanGaps: true }];
    if (instances.length > 1) {
      instances.forEach((inst, i) => rpsSets.push({ label: `RPS ${inst.name}`, data: toXY(inst.points), borderColor: INSTANCE_COLORS[i % INSTANCE_COLORS.length], backgroundColor: INSTANCE_COLORS[i % INSTANCE_COLORS.length], yAxisID: 'y', pointRadius: 0, borderWidth: 1.2, spanGaps: true }));
    }
    rpsSets.push({ label: 'Ошибки, %', data: toXY(live.series['lt.error_rate_pct']), borderColor: '#e5484d', backgroundColor: '#e5484d', yAxisID: 'y1', pointRadius: 0, borderWidth: 1.5, spanGaps: true });
    const rpsMax = dataMax(live.series['lt.rps_total'] || [], ...instances.map((inst) => inst.points || []));
    drawChart('rps', 'chartRps', rpsSets, {
      x,
      y: withLine(axis('RPS', { position: 'left' }), lines.rps, 'y', rpsMax),
      y1: withLine(axis('Ошибки, %', { position: 'right', suggestedMax: 5, grid: { drawOnChartArea: false } }), lines.rps, 'y1', Math.max(5, dataMax(live.series['lt.error_rate_pct'] || [])))
    }, { events: live.events, lines: lines.rps, labels: true });
    drawChart('p95', 'chartP95', [
      { label: 'p95, мс', data: toXY(live.series['lt.p95_total']), borderColor: '#f59f00', backgroundColor: '#f59f00', pointRadius: 0, borderWidth: 1.5, spanGaps: true }
    ], { x, y: withLine(axis(''), lines.p95, 'y', dataMax(live.series['lt.p95_total'] || [])) }, { events: live.events, lines: lines.p95, labels: false });
    drawChart('vus', 'chartVus', [
      { label: 'Пользователи', data: toXY(live.series['lt.vus']), borderColor: '#20c997', backgroundColor: '#20c997', pointRadius: 0, borderWidth: 1.5, stepped: true, spanGaps: true }
    ], { x, y: axis('') }, { events: live.events, lines: [], labels: false });
  }

  function applyLive(data, incremental) {
    const live = state.live;
    const fromMs = data.start ? data.start * 1000 : null;
    LIVE_ROLES.forEach((role) => {
      const fresh = data.series && data.series[role] ? data.series[role].points : [];
      live.series[role] = incremental ? mergePoints(live.series[role], fresh, fromMs) : (fresh || []);
    });
    const instances = {};
    (data.instances || []).forEach((inst) => {
      const old = live.instances[inst.tag];
      instances[inst.tag] = { name: inst.name, points: incremental && old ? mergePoints(old.points, inst.points, fromMs) : inst.points };
    });
    if (incremental) {
      Object.keys(live.instances).forEach((tag) => { if (!instances[tag]) instances[tag] = live.instances[tag]; });
    }
    live.instances = instances;
    live.step = data.step_sec || null;
    live.mode = data.mode || live.mode || '';
    live.events = data.events || [];
    live.sla = data.sla || {};
    live.start = data.start || null;
    live.end = data.end || null;
    const rps = live.series['lt.rps_total'] || [];
    live.lastMs = rps.length ? rps[rps.length - 1][0] : null;
    if (data.rps_query || !incremental) live.rpsQuery = data.rps_query || null;
  }

  async function loadLive(full) {
    clearTimeout(state.liveTimer);
    const live = state.live;
    const running = !!(state.run && state.run.loading);
    live.polls += 1;
    // Metrics without the run tag may have been someone else's at the start: look for the tag again now and then.
    const recheck = live.mode === 'plain' && live.polls % RECHECK_PLAIN_EVERY === 0;
    const incremental = !full && !recheck && running && !!live.step && !!live.lastMs;
    const params = new URLSearchParams({ window: String(live.window) });
    if (incremental) {
      params.set('since', String(live.lastMs));
      if (live.mode) params.set('mode', live.mode);
    }
    try {
      const data = await fetchJson(`/load_tests/runs/${encodeURIComponent(runId)}/live?${params}`);
      if (live !== state.live) return;  // the window changed meanwhile
      if (incremental && data.step_sec !== live.step) {
        // The run grew past a resolution step: read the whole window again.
        live.step = null;
        loadLive(true);
        return;
      }
      applyLive(data, incremental);
      state.liveLoaded = true;
      renderLive(data.ok ? data.message : (data.message || 'Метрики не загрузились'));
    } catch (e) {
      $('liveNote').textContent = `Метрики не загрузились: ${e.message}`;
    }
    if (state.run && state.run.loading) state.liveTimer = setTimeout(() => loadLive(false), LIVE_POLL_MS);
  }

  function setWindow(minutes) {
    document.querySelectorAll('#liveWindow [data-window]').forEach((btn) => {
      const on = Number(btn.dataset.window) === minutes;
      btn.classList.toggle('active', on);
      btn.setAttribute('aria-pressed', String(on));
    });
    state.live = freshLive(minutes);
    loadLive(true);
  }

  // ---- logs ------------------------------------------------------------------------------------

  async function loadLog() {
    clearTimeout(state.logTimer);
    if (state.logIndex === null) return;
    const pre = $('logText');
    try {
      const data = await fetchJson(`/load_tests/runs/${encodeURIComponent(runId)}/logs?index=${state.logIndex}`);
      const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 20;
      pre.textContent = data.text || '(журнал пуст)';
      if (atBottom) pre.scrollTop = pre.scrollHeight;
    } catch (e) {
      pre.textContent = `Журнал недоступен: ${e.message}`;
    }
    const inst = state.run && state.run.instances.find((i) => i.index === state.logIndex);
    if ($('logFollow').checked && inst && inst.active) state.logTimer = setTimeout(loadLog, LOG_POLL_MS);
  }

  function openLog(index) {
    const inst = state.run.instances.find((i) => i.index === index);
    state.logIndex = index;
    $('logTitle').textContent = `Журнал: ${inst ? inst.name : index}`;
    $('logCard').style.display = '';
    $('logText').textContent = 'Загрузка…';
    loadLog();
    $('logCard').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function init() {
    $('stopAll').addEventListener('click', () => stop(null, ''));
    $('logRefresh').addEventListener('click', loadLog);
    $('logFollow').addEventListener('change', loadLog);
    $('logClose').addEventListener('click', () => { state.logIndex = null; clearTimeout(state.logTimer); $('logCard').style.display = 'none'; });
    document.querySelectorAll('#liveWindow [data-window]').forEach((btn) => btn.addEventListener('click', () => setWindow(Number(btn.dataset.window))));
    loadRun();
  }

  LL.launchRun = { _internals: { mergePoints, freshness, slaLines, axisRange } };

  if ($('runRoot')) document.addEventListener('DOMContentLoaded', init);
})();
