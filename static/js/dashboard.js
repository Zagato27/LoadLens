// Dashboard page logic: KPI tiles, the last-run hero, verdicts per week, per-service trends and recent runs.
(function () {
  const LL = window.LoadLens;
  const ui = LL.ui;
  const $ = (id) => document.getElementById(id);
  const esc = ui.escapeHtml;
  const JOBS_POLL_MS = 5000;
  const JOBS_LIMIT = 20;
  const JOBS_SHOWN = 5;
  const CLOCK_MS = 60000;
  const ACTIVE_POLL_MS = 15000;
  const ACTIVE_IDLE_MS = 60000;
  const ACTIVE_SHOWN = 5;
  const ACTIVE_SPARK = { width: 120, height: 32, pad: 3 };
  // Pill tone of a load test run going on now (the same as on the run page).
  const RUN_TONE = { starting: 'running', running: 'running', stopping: 'warn', draining: 'running', reporting: 'running' };
  const NA_VERDICT = 'Недостаточно данных';
  // Verdicts of the weekly chart, bottom to top, with the status tone of each.
  const WEEK_SERIES = [
    { verdict: 'Успешно', tone: 'ok' },
    { verdict: 'Есть риски', tone: 'warn' },
    { verdict: 'Провал', tone: 'fail' },
    { verdict: NA_VERDICT, tone: 'na' }
  ];
  const TONE_COLORS = { ok: 'statusOk', warn: 'statusWarn', fail: 'statusFail', na: 'statusNa' };
  // What the RPS figure means for each test type: hero tile label and the caption in the services table.
  const RPS_LABELS = { soak: 'RPS удержания', spike: 'RPS всплеска', stress: 'RPS насыщения' };
  const RPS_CAPTIONS = { soak: 'удержано (среднее)', spike: 'уровень всплеска', stress: 'точка насыщения' };
  const SPARK = { width: 132, height: 36, pad: 7, r: 4, rLast: 5, hit: 10 };
  let weeksChart = null;
  let weeksCurrent = null;
  let lastData = null;
  let jobsTimer = 0;
  let runningJobs = null;
  let activeTimer = 0;
  let activeState = null;
  let lastActive = null;

  // ---- pure helpers (LL.dashboard._internals) ----------------------------------------------
  function finite(value) {
    if (value === null || value === undefined || value === '' || typeof value === 'boolean') return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  function formatRps(value) {
    const n = finite(value);
    return n === null ? '—' : n.toFixed(1);
  }
  // Runs saved without a type are step runs.
  function typeKey(testType) {
    return String(testType || '').trim().toLowerCase() || 'step';
  }
  // Change of `current` against `previous` in percent (one decimal); null unless both are numbers and previous ≠ 0.
  function rpsDelta(current, previous) {
    const cur = finite(current);
    const prev = finite(previous);
    if (cur === null || prev === null || prev === 0) return null;
    const pct = Math.round(((cur - prev) / Math.abs(prev)) * 1000) / 10;
    const direction = Math.abs(pct) < 0.5 ? 'flat' : (pct > 0 ? 'up' : 'down');
    const arrow = { up: '▲', down: '▼', flat: '≈' }[direction];
    const text = direction === 'flat' ? '0%' : `${pct > 0 ? '+' : '−'}${Math.abs(pct).toFixed(1)}%`;
    return { pct, direction, arrow, text };
  }
  // The run before the newest one of a service history (newest first) with the same test type.
  function previousOfSameType(history) {
    const list = Array.isArray(history) ? history.filter(Boolean) : [];
    if (list.length < 2) return null;
    const key = typeKey(list[0].test_type);
    return list.slice(1).find((h) => typeKey(h.test_type) === key) || null;
  }
  // Max RPS change of the newest run against previousOfSameType: rpsDelta plus `previous`.
  function historyDelta(history) {
    const list = Array.isArray(history) ? history.filter(Boolean) : [];
    const previous = previousOfSameType(list);
    if (!previous) return null;
    const delta = rpsDelta(list[0].max_rps, previous.max_rps);
    return delta ? Object.assign({ previous }, delta) : null;
  }
  // «SLA 2/3»: ok when every evaluated check passed, fail when none did, warn in between.
  function slaBadge(sla) {
    const total = Math.max(0, Math.floor(finite(sla && sla.total) || 0));
    const passed = Math.min(total, Math.max(0, Math.floor(finite(sla && sla.passed) || 0)));
    if (!total) return { text: 'SLA не задан', tone: 'na', title: 'Критерии SLA для запуска не проверялись' };
    const tone = passed === total ? 'ok' : (passed === 0 ? 'fail' : 'warn');
    return { text: `SLA ${passed}/${total}`, tone, title: `Выполнено критериев SLA: ${passed} из ${total}` };
  }
  function shareText(share) {
    const n = finite(share);
    return n === null ? '—' : `${Math.round(n * 100)}%`;
  }
  // Week starts arrive as 'YYYY-MM-DD' (UTC Mondays): label '12.10', range '12.10–18.10'.
  function parseDay(iso) {
    const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso || ''));
    return m ? new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]))) : null;
  }
  function dayLabel(d) {
    const pad = (n) => String(n).padStart(2, '0');
    return `${pad(d.getUTCDate())}.${pad(d.getUTCMonth() + 1)}`;
  }
  function weekLabel(iso) {
    const d = parseDay(iso);
    return d ? dayLabel(d) : String(iso || '');
  }
  function weekRange(iso) {
    const d = parseDay(iso);
    return d ? `${dayLabel(d)}–${dayLabel(new Date(d.getTime() + 6 * 86400000))}` : String(iso || '');
  }
  // Chart model of verdict_weeks: labels, week ranges, one value list per verdict, column totals.
  function weeksModel(weeks) {
    const list = Array.isArray(weeks) ? weeks.filter((w) => w && w.week_start) : [];
    const series = WEEK_SERIES.map((s) => ({
      verdict: s.verdict,
      tone: s.tone,
      values: list.map((w) => Math.max(0, Math.floor(finite(w.counts && w.counts[s.verdict]) || 0)))
    }));
    const totals = list.map((_, i) => series.reduce((sum, s) => sum + s.values[i], 0));
    return {
      labels: list.map((w) => weekLabel(w.week_start)),
      ranges: list.map((w) => weekRange(w.week_start)),
      series,
      totals,
      total: totals.reduce((a, b) => a + b, 0)
    };
  }
  function weeksSummary(model) {
    if (!model.total) return `Вердикты по неделям: за ${model.labels.length} нед. завершённых отчётов нет`;
    const parts = model.series.map((s) => `${s.verdict} — ${s.values.reduce((a, b) => a + b, 0)}`);
    return `Вердикты по неделям за ${model.labels.length} нед.: всего ${model.total}; ${parts.join(', ')}`;
  }
  // Index of the topmost visible non-empty segment of a stacked column; it gets the rounded end.
  function topSegmentIndex(values, visible) {
    for (let i = (values || []).length - 1; i >= 0; i -= 1) {
      if ((!visible || visible[i] !== false) && Number(values[i]) > 0) return i;
    }
    return -1;
  }
  const round1 = (n) => Math.round(n * 10) / 10;
  // Sparkline geometry, oldest run first: x spreads evenly, y spans min..max of the known values,
  // a run without RPS sits on the bottom edge.
  function sparkPoints(history, size) {
    const box = Object.assign({}, SPARK, size || {});
    const list = (Array.isArray(history) ? history.filter(Boolean) : []).slice().reverse();
    const values = list.map((h) => finite(h.max_rps));
    const known = values.filter((v) => v !== null);
    const min = known.length ? Math.min(...known) : 0;
    const max = known.length ? Math.max(...known) : 0;
    const innerW = box.width - 2 * box.pad;
    const innerH = box.height - 2 * box.pad;
    return list.map((run, i) => {
      const v = values[i];
      const x = list.length > 1 ? box.pad + (innerW * i) / (list.length - 1) : box.width / 2;
      let y = box.height - box.pad;
      if (v !== null) y = max === min ? box.pad + innerH / 2 : box.pad + innerH * (1 - (v - min) / (max - min));
      return { x: round1(x), y: round1(y), value: v, run };
    });
  }
  function sparkTitle(run, value) {
    return [
      run.run_name || '—',
      run.created_at ? LL.formatDateTime(run.created_at) : '',
      value === null ? 'RPS нет' : `${formatRps(value)} RPS`,
      run.test_type ? LL.testTypeLabel(run.test_type) : '',
      run.verdict || NA_VERDICT
    ].filter(Boolean).join(' · ');
  }
  // Inline SVG of history[].max_rps (history is newest first, drawn oldest to newest); points wear the
  // verdict color, the newest one is larger; a run without RPS is a hollow point on the bottom edge.
  // The line joins only runs of the newest run's test type (a soak level between step peaks is not a
  // drop); runs of other types stay as faded points.
  function sparkline(history) {
    const pts = sparkPoints(history);
    if (!pts.length) return '<span class="dim">—</span>';
    const mainType = typeKey(pts[pts.length - 1].run.test_type);
    const segments = [];
    let segment = [];
    pts.forEach((p) => {
      if (typeKey(p.run.test_type) !== mainType) return;
      if (p.value === null) {
        if (segment.length) segments.push(segment);
        segment = [];
      } else {
        segment.push(p);
      }
    });
    if (segment.length) segments.push(segment);
    const lines = segments.filter((s) => s.length > 1)
      .map((s) => `<polyline class="dash-spark-line" points="${s.map((p) => `${p.x},${p.y}`).join(' ')}"/>`).join('');
    const dots = pts.map((p, i) => {
      const r = i === pts.length - 1 ? SPARK.rLast : SPARK.r;
      const other = typeKey(p.run.test_type) !== mainType ? ' other' : '';
      const cls = `dash-spark-dot ${LL.verdictClass(p.run.verdict)}${p.value === null ? ' empty' : ''}${other}`;
      return `<g class="dash-spark-point"><title>${esc(sparkTitle(p.run, p.value))}</title>`
        + `<circle class="dash-spark-hit" cx="${p.x}" cy="${p.y}" r="${SPARK.hit}"/>`
        + `<circle class="${cls}" cx="${p.x}" cy="${p.y}" r="${r}"/></g>`;
    }).join('');
    const label = `Max RPS запусков от старых к новым: ${pts.map((p) => (p.value === null ? 'нет данных' : formatRps(p.value))).join(', ')}`;
    return `<svg class="dash-spark" viewBox="0 0 ${SPARK.width} ${SPARK.height}" width="${SPARK.width}" height="${SPARK.height}" role="img" aria-label="${esc(label)}">${lines}${dots}</svg>`;
  }
  // Test window: the end keeps only its time when both ends fall on the same day.
  function periodText(startMs, endMs) {
    const s = finite(startMs);
    const e = finite(endMs);
    if (s === null && e === null) return '';
    if (s === null || e === null) return LL.formatDateTime(s === null ? e : s);
    const from = LL.formatDateTime(s);
    const to = LL.formatDateTime(e);
    return from.slice(0, 10) === to.slice(0, 10) ? `${from} — ${to.slice(11)}` : `${from} — ${to}`;
  }
  // Hero tiles of the last run: max RPS against the previous run of the same type, test window, test type.
  function heroKpis(last, kpi) {
    const info = kpi || {};
    const type = typeKey(last.test_type);
    const rps = finite(last.max_rps) !== null ? last.max_rps : (info.last && info.last.max_rps);
    const previous = info.previous && info.previous.run_name ? info.previous : null;
    const delta = previous ? rpsDelta(rps, previous.max_rps) : null;
    let note = 'нет предыдущего прогона этого типа';
    if (previous) note = delta ? `к «${previous.run_name}»` : `к «${previous.run_name}»: нет данных для сравнения`;
    const start = finite(last.start_ms);
    const end = finite(last.end_ms);
    const caption = RPS_CAPTIONS[type] || (type === 'step' && last.test_type ? 'пик производительности' : '');
    return [
      {
        key: 'rps',
        label: RPS_LABELS[type] || 'Max RPS',
        value: formatRps(rps),
        delta,
        note,
        title: previous ? `Предыдущий прогон того же типа: ${previous.run_name}, ${formatRps(previous.max_rps)} RPS` : ''
      },
      { key: 'window', label: 'Окно теста', value: start !== null && end !== null && end >= start ? LL.formatDuration(end - start) : '—', note: periodText(start, end) },
      { key: 'type', label: 'Тип теста', value: LL.testTypeLabel(last.test_type), note: caption ? `RPS: ${caption}` : '' }
    ];
  }
  // Top-row tiles from /dashboard_data (the jobs tile comes from /jobs).
  function kpiTiles(data, now) {
    const kpi = (data && data.kpi) || {};
    const runs7 = finite(kpi.runs_7d);
    const runs30 = finite(kpi.runs_30d);
    const share = finite(kpi.risky_share_30d);
    const risky = finite(kpi.risky_30d);
    const last = data && data.last_run && data.last_run.run_name ? data.last_run : null;
    return {
      runs: { value: runs7 === null ? '—' : String(runs7), note: runs30 === null ? '' : `за 30 дней: ${runs30}` },
      risky: {
        value: shareText(share),
        meter: share === null ? 0 : Math.max(0, Math.min(1, share)),
        note: share === null ? 'за 30 дней запусков нет' : `${risky === null ? Math.round(share * runs30) : risky} из ${runs30}: риски и провалы`
      },
      updated: {
        value: last && last.created_at ? LL.relativeTime(last.created_at, now) : '—',
        note: last ? `отчёт «${last.run_name}»` : 'отчётов пока нет',
        title: last && last.created_at ? LL.formatDateTime(last.created_at) : ''
      }
    };
  }
  function jobsTile(jobs) {
    if (!Array.isArray(jobs)) return { value: '—', note: 'список задач недоступен' };
    if (!jobs.length) return { value: '0', note: 'нет активных задач' };
    const first = jobs[0] || {};
    return {
      value: jobs.length >= JOBS_LIMIT ? `${JOBS_LIMIT}+` : String(jobs.length),
      note: `${first.run_name || '(без названия)'} · ${Math.round(finite(first.progress) || 0)}%`
    };
  }

  // «12 мин из 30 мин»: how long the load goes against the run's limit; before the start only the limit.
  function elapsedText(startedAt, maxSeconds, now) {
    const max = finite(maxSeconds);
    const limit = max && max > 0 ? LL.formatDuration(max * 1000) : '';
    const started = startedAt ? Date.parse(startedAt) : NaN;
    if (!Number.isFinite(started)) return limit ? `лимит ${limit}` : '';
    const gone = LL.formatDuration(Math.max(0, (now === undefined ? Date.now() : now) - started));
    return limit ? `${gone} из ${limit}` : gone;
  }
  function errorsText(value) {
    const n = finite(value);
    if (n === null) return '';
    return `${n.toFixed(n > 0 && n < 1 ? 2 : 1)}%`;
  }
  // One run of /load_tests/active as the dashboard shows it.
  function activeRunSummary(run, now) {
    const r = run || {};
    const total = finite(r.instances_total) || 0;
    const going = finite(r.instances_running) || 0;
    const live = r.live && typeof r.live === 'object' ? r.live : null;
    let liveText = '';
    let liveNote = '';
    if (live && live.ok && finite(live.rps) !== null) {
      liveText = `${formatRps(live.rps)} RPS`;
      if (finite(live.errors) !== null) liveText += ` · ошибки ${errorsText(live.errors)}`;
    } else if (live && live.message) {
      liveNote = String(live.message);
    } else if (r.loading) {
      liveNote = 'Метрики появятся после старта нагрузки';
    } else {
      liveNote = String(r.message || '');
    }
    return {
      name: r.run_name || '(без названия)',
      url: r.url || `/launch/${encodeURIComponent(r.run_id || '')}`,
      status: r.status_title || r.status || '',
      tone: RUN_TONE[r.status] || 'na',
      meta: [r.service, r.owner].filter(Boolean).join(' · '),
      progress: [
        total > 1 ? `инстансы: ${going} из ${total}` : '',
        r.loading ? elapsedText(r.started_at, r.max_seconds, now) : ''
      ].filter(Boolean).join(' · '),
      liveText,
      liveNote,
      points: live && Array.isArray(live.points) ? live.points : [],
      canStop: !!r.can_stop
    };
  }
  function activeTile(data, now) {
    if (!data || !Array.isArray(data.runs)) return { value: '—', note: 'список прогонов недоступен' };
    const total = finite(data.total) || 0;
    if (!total) return { value: '0', note: 'нет идущих прогонов' };
    const first = data.runs[0];
    if (!first) return { value: String(total), note: '' };
    const elapsed = first.loading ? elapsedText(first.started_at, first.max_seconds, now) : (first.status_title || '');
    return { value: String(total), note: [first.run_name || '(без названия)', elapsed].filter(Boolean).join(' · ') };
  }
  // RPS of the last minutes as an inline SVG: a line over a filled area from zero; '' below two points.
  function liveSpark(points, width, height) {
    const pts = (Array.isArray(points) ? points : [])
      .filter((p) => Array.isArray(p) && finite(p[0]) !== null && finite(p[1]) !== null)
      .map((p) => [Number(p[0]), Number(p[1])]);
    if (pts.length < 2) return '';
    const w = width || ACTIVE_SPARK.width;
    const h = height || ACTIVE_SPARK.height;
    const pad = ACTIVE_SPARK.pad;
    const x0 = pts[0][0];
    const spanX = pts[pts.length - 1][0] - x0 || 1;
    const low = Math.min(0, ...pts.map((p) => p[1]));
    const spanY = Math.max(...pts.map((p) => p[1])) - low || 1;
    const coords = pts.map(([x, y]) => [
      (pad + ((x - x0) / spanX) * (w - 2 * pad)).toFixed(1),
      (h - pad - ((y - low) / spanY) * (h - 2 * pad)).toFixed(1)
    ]);
    const line = coords.map((c) => c.join(',')).join(' ');
    const base = (h - pad).toFixed(1);
    const area = `${line} ${coords[coords.length - 1][0]},${base} ${coords[0][0]},${base}`;
    return `<svg class="dash-live-spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true">`
      + `<polygon class="area" points="${area}"/><polyline class="line" points="${line}"/></svg>`;
  }

  // ---- rendering ----------------------------------------------------------------------------
  function pill(verdict, extra) {
    const v = verdict || NA_VERDICT;
    return `<span class="pill ${LL.verdictClass(v)}${extra ? ` ${extra}` : ''}">${esc(v)}</span>`;
  }
  function deltaHtml(delta, title) {
    if (!delta) return '';
    return `<span class="dash-delta ${delta.direction}"${title ? ` title="${esc(title)}"` : ''}>${delta.arrow} ${esc(delta.text)}</span>`;
  }
  function setText(id, text, title) {
    const el = $(id);
    if (!el) return;
    el.textContent = text === '' || text === null || text === undefined ? '\u00a0' : String(text);
    if (title !== undefined) el.title = title || '';
  }

  function renderSkeletons() {
    $('servicesBody').innerHTML = ui.skeletonRows(6, 3);
    $('recentBody').innerHTML = ui.skeletonRows(5, 3);
    $('lastKpis').innerHTML = [1, 2, 3].map(() => `<div class="dash-kpi" aria-hidden="true">${ui.skeleton(2, ['w-40', 'w-60'])}</div>`).join('');
  }

  function renderTiles(data) {
    const t = kpiTiles(data);
    setText('kpiRunsValue', t.runs.value);
    setText('kpiRunsNote', t.runs.note);
    setText('kpiRiskyValue', t.risky.value);
    setText('kpiRiskyNote', t.risky.note);
    const meter = $('kpiRiskyMeter');
    if (meter) meter.style.width = `${Math.round(t.risky.meter * 100)}%`;
    setText('kpiUpdatedValue', t.updated.value, t.updated.title);
    setText('kpiUpdatedNote', t.updated.note);
  }

  function renderHero(last, kpi, emptyTitle) {
    const has = !!(last && last.run_name);
    $('lastReportLink').hidden = !has;
    $('lastCompareLink').hidden = !has;
    if (!has) {
      setText('lastTitle', emptyTitle || 'Отчётов пока нет', '');
      setText('lastMeta', emptyTitle ? '' : 'Здесь появится итог последнего запуска: вердикт, SLA и Max RPS.');
      $('lastBadges').innerHTML = '';
      $('lastKpis').innerHTML = '';
      return;
    }
    setText('lastTitle', last.run_name, last.run_name);
    setText('lastMeta', [last.service || '—', last.created_at ? `отчёт создан ${LL.relativeTime(last.created_at)}` : ''].filter(Boolean).join(' · '));
    const sla = slaBadge(last.sla);
    $('lastBadges').innerHTML = `${pill(last.verdict, 'pill-lg')}<span class="pill pill-lg ${sla.tone}" title="${esc(sla.title)}">${esc(sla.text)}</span>`;
    $('lastKpis').innerHTML = heroKpis(last, kpi).map((t) => `
      <div class="dash-kpi dash-kpi-${t.key}"${t.title ? ` title="${esc(t.title)}"` : ''}>
        <div class="dash-kpi-label">${esc(t.label)}</div>
        <div class="dash-kpi-value">${esc(t.value)}</div>
        <div class="dash-kpi-note">${deltaHtml(t.delta)}${t.delta ? ' ' : ''}${esc(t.note || '')}</div>
      </div>`).join('');
    $('lastReportLink').href = LL.reportUrl(last);
    const previous = kpi && kpi.previous && kpi.previous.run_name ? kpi.previous : null;
    $('lastCompareLink').href = `/compare?run_a=${encodeURIComponent(last.run_name)}${previous ? `&run_b=${encodeURIComponent(previous.run_name)}` : ''}`;
  }

  // Only the topmost segment of a column gets the 4px rounded end (Chart.js scriptable option).
  function segmentRadius(ctx) {
    const chart = ctx && ctx.chart;
    if (!chart || typeof ctx.dataIndex !== 'number') return 0;
    const values = chart.data.datasets.map((ds) => (ds.data || [])[ctx.dataIndex]);
    const visible = chart.data.datasets.map((_, i) => chart.isDatasetVisible(i));
    return topSegmentIndex(values, visible) === ctx.datasetIndex ? 4 : 0;
  }
  function weeksDatasets(model, colors) {
    return model.series.map((s) => ({
      label: s.verdict,
      data: s.values,
      llTone: s.tone,
      backgroundColor: colors[TONE_COLORS[s.tone]],
      hoverBackgroundColor: colors[TONE_COLORS[s.tone]],
      // A 2px gap in the card color between stacked segments.
      borderColor: colors.surface,
      borderWidth: { top: 2, right: 0, bottom: 0, left: 0 },
      borderRadius: segmentRadius,
      maxBarThickness: 24,
      stack: 'verdicts'
    }));
  }
  function renderWeeks(weeks) {
    const model = weeksModel(weeks);
    weeksCurrent = model;
    const canvas = $('weeksChart');
    setText('weeksTotal', model.total ? `${model.total} за ${model.labels.length} нед.` : '');
    $('weeksEmpty').hidden = model.total > 0;
    canvas.setAttribute('aria-label', weeksSummary(model));
    if (!window.Chart || typeof canvas.getContext !== 'function') return;
    const colors = ui.chartColors();
    if (weeksChart) {
      weeksChart.data.labels = model.labels;
      weeksChart.data.datasets = weeksDatasets(model, colors);
      weeksChart.update();
      return;
    }
    // eslint-disable-next-line no-undef
    weeksChart = new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: { labels: model.labels, datasets: weeksDatasets(model, colors) },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false, axis: 'x' },
        scales: {
          x: { stacked: true, grid: { display: false }, border: { display: false }, ticks: { color: colors.text } },
          y: { stacked: true, beginAtZero: true, border: { display: false }, grid: { color: colors.grid }, ticks: { color: colors.text, precision: 0, maxTicksLimit: 5 } }
        },
        plugins: {
          legend: { position: 'bottom', labels: { color: colors.text, usePointStyle: true, pointStyle: 'rectRounded', boxWidth: 10, boxHeight: 10, padding: 14 } },
          tooltip: {
            filter: (item) => Number(item.raw) > 0,
            itemSort: (a, b) => b.datasetIndex - a.datasetIndex,
            callbacks: {
              title: (items) => (items.length && weeksCurrent ? `Неделя ${weeksCurrent.ranges[items[0].dataIndex] || ''}` : ''),
              label: (item) => ` ${item.dataset.label}: ${item.raw}`,
              footer: (items) => (items.length && weeksCurrent ? `Всего: ${weeksCurrent.totals[items[0].dataIndex] || 0}` : '')
            }
          }
        }
      }
    });
  }
  function recolorWeeks() {
    if (!weeksChart) return;
    const colors = ui.chartColors();
    weeksChart.data.datasets.forEach((ds) => {
      const color = colors[TONE_COLORS[ds.llTone]];
      ds.backgroundColor = color;
      ds.hoverBackgroundColor = color;
      ds.borderColor = colors.surface;
    });
  }

  // Older payloads carry only the verdicts of the service; the sparkline then shows verdict points.
  function serviceHistory(svc) {
    if (Array.isArray(svc.history) && svc.history.length) return svc.history;
    const verdicts = Array.isArray(svc.recent_verdicts) && svc.recent_verdicts.length ? svc.recent_verdicts : [svc.verdict];
    return verdicts.map((verdict, i) => (i === 0
      ? { run_name: svc.last_run, run_id: svc.run_id || '', created_at: svc.created_at, verdict, max_rps: svc.max_rps, test_type: svc.test_type }
      : { verdict, max_rps: null }));
  }
  function serviceRowHtml(svc) {
    const history = serviceHistory(svc);
    const previous = previousOfSameType(history);
    const delta = historyDelta(history);
    const rps = finite(svc.max_rps);
    const caption = RPS_CAPTIONS[typeKey(svc.test_type)];
    const deltaTitle = delta ? `К прогону «${delta.previous.run_name || '—'}» того же типа: ${formatRps(delta.previous.max_rps)} RPS` : '';
    const compare = `/compare?run_a=${encodeURIComponent(svc.last_run || '')}${previous && previous.run_name ? `&run_b=${encodeURIComponent(previous.run_name)}` : ''}`;
    const compareTitle = previous && previous.run_name
      ? `Сравнить «${svc.last_run}» с предыдущим запуском того же типа «${previous.run_name}»`
      : `Сравнить «${svc.last_run}» с предыдущим запуском`;
    const created = svc.created_at ? LL.relativeTime(svc.created_at) : '—';
    const createdFull = svc.created_at ? LL.formatDateTime(svc.created_at) : '';
    return `<tr>
        <td class="dash-service" title="${esc(svc.service)}"><a href="/reports?service=${encodeURIComponent(svc.service)}"><strong>${esc(svc.service)}</strong></a></td>
        <td class="run-name" title="${esc(svc.last_run)}"><a href="${esc(LL.reportUrl({ run_id: svc.run_id, run_name: svc.last_run }))}">${esc(svc.last_run)}</a><div class="dim">${esc(LL.testTypeLabel(svc.test_type))} · <span title="${esc(createdFull)}">${esc(created)}</span></div></td>
        <td>${pill(svc.verdict)}</td>
        <td class="num">${esc(formatRps(rps))}${rps !== null && caption ? `<div class="dim">${esc(caption)}</div>` : ''}${delta ? `<div>${deltaHtml(delta, deltaTitle)}</div>` : ''}</td>
        <td class="dash-trend">${sparkline(history)}</td>
        <td class="nowrap"><a class="btn btn-sm" href="${esc(compare)}" title="${esc(compareTitle)}" aria-label="${esc(compareTitle)}">Сравнить</a></td>
      </tr>`;
  }
  function renderServices(services) {
    const body = $('servicesBody');
    body.innerHTML = services.length ? services.map(serviceRowHtml).join('') : ui.emptyRow(6, 'Пока нет ни одного завершённого отчёта.');
  }

  function renderRecent(runs) {
    const body = $('recentBody');
    if (!runs.length) {
      body.innerHTML = ui.emptyRow(5, 'Запусков пока нет.');
      return;
    }
    body.innerHTML = '';
    runs.forEach((r) => {
      const tr = document.createElement('tr');
      tr.className = 'clickable';
      const href = LL.reportUrl(r);
      tr.innerHTML = `
        <td class="run-name" title="${esc(r.run_name)}"><a href="${esc(href)}">${esc(r.run_name)}</a></td>
        <td>${esc(r.service || '—')}</td>
        <td class="muted">${esc(LL.testTypeLabel(r.test_type))}</td>
        <td>${pill(r.verdict)}</td>
        <td class="muted nowrap" title="${esc(r.created_at ? LL.formatDateTime(r.created_at) : '')}">${esc(r.created_at ? LL.relativeTime(r.created_at) : '—')}</td>`;
      tr.addEventListener('click', (ev) => {
        if (ev.target && ev.target.closest && ev.target.closest('a')) return;
        location.href = href;
      });
      body.appendChild(tr);
    });
  }

  function jobRowHtml(j) {
    return `<div class="job-row"><span class="pill running">В работе</span><div><div class="name">${esc(j.run_name || '(без названия)')}</div><div class="meta">${esc(j.service || '')} · ${Math.round(finite(j.progress) || 0)}% — ${esc(j.message || '')}</div></div><a href="/new?job=${encodeURIComponent(j.job_id)}" data-min-role="engineer">Следить</a></div>`;
  }
  function renderJobs(jobs) {
    const tile = jobsTile(jobs);
    setText('kpiJobsValue', tile.value);
    setText('kpiJobsNote', tile.note);
    const list = Array.isArray(jobs) ? jobs : [];
    $('jobsCard').hidden = !list.length;
    const more = list.length > JOBS_SHOWN ? `<div class="dim dash-jobs-more">и ещё ${list.length - JOBS_SHOWN}</div>` : '';
    $('jobsList').innerHTML = list.slice(0, JOBS_SHOWN).map(jobRowHtml).join('') + more;
  }
  // Polls every 5 s while jobs run. A job that leaves the running list may have saved a report, so the
  // summary reloads quietly (without skeletons).
  async function loadJobs() {
    clearTimeout(jobsTimer);
    let jobs = null;
    try {
      const resp = await fetch(`/jobs?status=running&limit=${JOBS_LIMIT}`);
      if (resp.ok) {
        const data = await resp.json();
        jobs = Array.isArray(data.jobs) ? data.jobs : [];
      }
    } catch (e) {
      jobs = null; // auxiliary block
    }
    renderJobs(jobs);
    if (!jobs) return;
    const ids = jobs.map((j) => String(j.job_id || ''));
    if (runningJobs && runningJobs.some((id) => !ids.includes(id))) loadDashboard({ quiet: true });
    runningJobs = ids;
    if (jobs.length) jobsTimer = setTimeout(loadJobs, JOBS_POLL_MS);
  }

  function activeRowHtml(run, now) {
    const s = activeRunSummary(run, now);
    const live = s.liveText
      ? `<div class="dash-active-live" title="RPS и доля ошибок за последние 15 минут">${liveSpark(s.points)}<span class="dash-active-value">${esc(s.liveText)}</span></div>`
      : `<div class="dash-active-live dim">${esc(s.liveNote)}</div>`;
    const stop = s.canStop
      ? `<button type="button" class="btn btn-sm btn-danger" data-stop-run="${esc(run.run_id || '')}" data-run-name="${esc(s.name)}">Остановить</button>`
      : '';
    return `<div class="dash-active-row"><span class="pill ${s.tone}">${esc(s.status)}</span>`
      + `<div class="dash-active-main"><a class="name" href="${esc(s.url)}">${esc(s.name)}</a>`
      + `${s.meta ? `<div class="meta">${esc(s.meta)}</div>` : ''}${s.progress ? `<div class="meta">${esc(s.progress)}</div>` : ''}</div>`
      + `${live}<div class="dash-active-actions"><a class="btn btn-sm" href="${esc(s.url)}">Открыть</a>${stop}</div></div>`;
  }
  function renderActive(data) {
    const tile = activeTile(data);
    setText('kpiActiveValue', tile.value);
    setText('kpiActiveNote', tile.note);
    const section = $('activeRuns');
    if (!section) return;
    const runs = data && Array.isArray(data.runs) ? data.runs : [];
    section.hidden = !runs.length;
    $('activeList').innerHTML = runs.slice(0, ACTIVE_SHOWN).map((r) => activeRowHtml(r)).join('');
    const rest = (data ? finite(data.total) || 0 : 0) - Math.min(runs.length, ACTIVE_SHOWN);
    $('activeMore').hidden = rest <= 0;
    $('activeMore').innerHTML = rest > 0 ? `и ещё ${rest} — <a href="/launch#runsPanel">все прогоны</a>` : '';
  }
  // Polls every 15 s while runs go on, every minute otherwise. A run that changes its status or ends
  // may have started or saved a report, so report jobs and the summary reload too.
  async function loadActive() {
    clearTimeout(activeTimer);
    let data = null;
    try {
      const resp = await fetch(`/load_tests/active?limit=${ACTIVE_SHOWN}&live=1`);
      if (resp.ok) data = await resp.json();
    } catch (e) {
      data = null; // auxiliary block
    }
    lastActive = data;
    renderActive(data);
    if (data) {
      if (LL.activeRuns) LL.activeRuns.show(data.total);
      const state = (Array.isArray(data.runs) ? data.runs : []).map((r) => `${r.run_id}:${r.status}`).join(' ');
      if (activeState !== null && state !== activeState) {
        loadJobs();
        loadDashboard({ quiet: true });
      }
      activeState = state;
    }
    activeTimer = setTimeout(loadActive, data && data.total ? ACTIVE_POLL_MS : ACTIVE_IDLE_MS);
  }
  async function stopActiveRun(runId, name) {
    const ok = await ui.confirm({
      title: `Остановить прогон «${name}»`,
      message: 'Всем инстансам уйдёт сигнал остановки (CI-пайплайнам — отмена); не завершившиеся за отведённое время будут убиты. Отчёт построится по уже собранным метрикам.',
      confirmText: 'Остановить',
      danger: true
    });
    if (!ok) return;
    try {
      const resp = await fetch(`/load_tests/runs/${encodeURIComponent(runId)}/stop`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
      });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(body.error || `HTTP ${resp.status}`);
      ui.toast('Остановка запрошена', { tone: 'ok' });
    } catch (e) {
      ui.toast(`Не удалось остановить: ${e.message}`, { tone: 'error' });
    }
    loadActive();
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

  // `quiet` keeps the current render on screen while reloading (no skeletons, errors ignored).
  async function loadDashboard(options) {
    const quiet = !!(options && options.quiet);
    if (!quiet) renderSkeletons();
    let data;
    try {
      const resp = await fetch('/dashboard_data');
      data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    } catch (e) {
      if (quiet) return;
      $('setupBanner').hidden = false;
      $('setupBanner').querySelector('.muted').textContent = `Не удалось загрузить данные: ${e.message}. Скорее всего, база данных недоступна — проверьте подключение в мастере.`;
      $('servicesBody').innerHTML = ui.emptyRow(6, 'Нет данных');
      $('recentBody').innerHTML = ui.emptyRow(5, 'Нет данных');
      renderHero(null, null, 'Нет данных');
      return;
    }
    lastData = data;
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
    renderTiles(data);
    renderHero(data.last_run, data.kpi);
    renderWeeks(data.verdict_weeks);
    renderServices(Array.isArray(data.services) ? data.services : []);
    renderRecent(Array.isArray(data.recent_runs) ? data.recent_runs : []);
  }

  LL.dashboard = {
    _internals: {
      finite,
      formatRps,
      typeKey,
      rpsDelta,
      previousOfSameType,
      historyDelta,
      slaBadge,
      shareText,
      weekLabel,
      weekRange,
      weeksModel,
      weeksSummary,
      topSegmentIndex,
      sparkPoints,
      sparkline,
      periodText,
      heroKpis,
      kpiTiles,
      jobsTile,
      serviceHistory,
      elapsedText,
      errorsText,
      activeRunSummary,
      activeTile,
      liveSpark,
      activeRowHtml
    }
  };

  document.addEventListener('DOMContentLoaded', () => {
    ['bannerDemoBtn', 'emptyDemoBtn'].forEach((id) => { const b = $(id); if (b) b.addEventListener('click', () => seedDemo(b)); });
    // Status colors and the segment gap follow the theme; restyleCharts redraws ticks, grid and legend.
    ui.theme.onChange(() => { recolorWeeks(); ui.restyleCharts(); });
    loadDashboard();
    loadJobs();
    if (LL.can('engineer') && $('activeRuns')) {
      // The dashboard reads the running load tests itself and passes their number to the menu badge.
      if (LL.activeRuns) LL.activeRuns.claim();
      $('activeList').addEventListener('click', (ev) => {
        const btn = ev.target && ev.target.closest ? ev.target.closest('[data-stop-run]') : null;
        if (btn) stopActiveRun(btn.dataset.stopRun, btn.dataset.runName || '');
      });
      loadActive();
    }
    // Relative times («5 мин назад») age while the page stays open.
    setInterval(() => {
      if (!lastData) return;
      renderTiles(lastData);
      renderHero(lastData.last_run, lastData.kpi);
    }, CLOCK_MS);
    // Elapsed times of the running tests age between the polls too.
    setInterval(() => { if (lastActive) renderActive(lastActive); }, CLOCK_MS);
  });
})();
