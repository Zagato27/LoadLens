// SPDX-License-Identifier: Apache-2.0
// Dashboard page logic: the status line with the verdicts of 8 weeks, the last run (figures against the SLA
// thresholds and its load profile), per-service trends, recent runs, report jobs and running load tests.
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
  const DAY_MS = 86400000;
  // Pill tone of a load test run going on now (the same as on the run page).
  const RUN_TONE = { starting: 'running', running: 'running', stopping: 'warn', draining: 'running', reporting: 'running' };
  // Verdict codes of the weekly columns (the keys of verdict_weeks[].counts), bottom to top, with the
  // status tone of each (read when called: the labels follow the UI language).
  const weekSeries = () => LL.VERDICTS.map((code) => ({ verdict: code, label: LL.verdictLabel(code), tone: LL.verdictClass(code) }));
  // Tone classes of the last-run block (its left rule wears the verdict color).
  const RUN_TONES = ['tone-ok', 'tone-warn', 'tone-fail', 'tone-na'];
  // What the RPS figure means for each test type: the figure caption and the caption in the services table.
  const RPS_LABELS = {
    soak: () => LL.t('dashboard.rps.label_soak'),
    spike: () => LL.t('dashboard.rps.label_spike'),
    stress: () => LL.t('dashboard.rps.label_stress')
  };
  const RPS_CAPTIONS = {
    soak: () => LL.t('dashboard.rps.caption_soak'),
    spike: () => LL.t('dashboard.rps.caption_spike'),
    stress: () => LL.t('dashboard.rps.caption_stress')
  };
  const rpsLabel = (type) => (RPS_LABELS[type] ? RPS_LABELS[type]() : '');
  const rpsCaption = (type) => (RPS_CAPTIONS[type] ? RPS_CAPTIONS[type]() : '');
  // Service sparkline: verdict squares on a hairline, the newest square is larger; `hit` is the hover area.
  const SPARK = { width: 110, height: 26, pad: 4, size: 5, sizeLast: 7, hit: 9 };
  // Verdict columns of the status line (236×64): columns stand on the axis at y=48, two date labels under it.
  const WEEKS_SVG = { width: 236, height: 64, plotW: 220, base: 48, top: 2, unit: 8, label: 62, col: 0.64 };
  // A figure bar spans 1.5 thresholds, so the threshold tick sits at two thirds of it.
  const FIG_SCALE = 1.5;
  let lastData = null;
  let jobsTimer = 0;
  let runningJobs = null;
  let activeTimer = 0;
  let activeState = null;
  let lastActive = null;
  let profileHandle = null;
  let profileKey = null;

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
    if (!total) return { text: LL.t('dashboard.sla.none'), tone: 'na', title: LL.t('dashboard.sla.none_title'), passed, total };
    const tone = passed === total ? 'ok' : (passed === 0 ? 'fail' : 'warn');
    return { text: `SLA ${passed}/${total}`, tone, title: LL.t('dashboard.sla.title', { passed, total }), passed, total };
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
  // Model of verdict_weeks: labels, week ranges, one value list per verdict, column totals.
  function weeksModel(weeks) {
    const list = Array.isArray(weeks) ? weeks.filter((w) => w && w.week_start) : [];
    const series = weekSeries().map((s) => ({
      verdict: s.verdict,
      label: s.label,
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
    if (!model.total) return LL.tn('dashboard.weeks.summary_empty', model.labels.length);
    const parts = model.series.map((s) => `${s.label} — ${s.values.reduce((a, b) => a + b, 0)}`);
    return LL.tn('dashboard.weeks.summary', model.labels.length, { total: model.total, parts: parts.join(', ') });
  }
  // Finished reports over the shown weeks, next to the columns; '' without any.
  function weeksTotalText(model) {
    return model.total ? LL.tn('dashboard.weeks.total', model.labels.length, { total: model.total }) : '';
  }
  // Title and footer of a weekly column tooltip (items as Chart.js passes them: [{dataIndex}]).
  function weeksTooltipTitle(model, items) {
    return items && items.length && model ? LL.t('dashboard.weeks.tooltip_title', { range: model.ranges[items[0].dataIndex] || '' }) : '';
  }
  function weeksTooltipFooter(model, items) {
    return items && items.length && model ? LL.t('dashboard.weeks.tooltip_total', { total: model.totals[items[0].dataIndex] || 0 }) : '';
  }
  // Hover text of a weekly column: the week, its verdicts and the total, one per line.
  function weekTitle(model, index) {
    const items = [{ dataIndex: index }];
    const parts = model.series.filter((s) => s.values[index] > 0).map((s) => `${s.label}: ${s.values[index]}`);
    return [weeksTooltipTitle(model, items)].concat(parts, [weeksTooltipFooter(model, items)]).filter(Boolean).join('\n');
  }
  const round1 = (n) => Math.round(n * 10) / 10;
  // SVG of the weekly verdict columns: segments stacked from the axis in verdict order (8 px per run, less when
  // a week holds more than fits), an empty week a 2 px stub; the first and the last week start under the axis.
  function weeksSvg(model) {
    const box = WEEKS_SVG;
    const n = model && model.labels ? model.labels.length : 0;
    if (!n) return '';
    const step = box.plotW / n;
    const colW = round1(step * box.col);
    const unit = Math.min(box.unit, (box.base - box.top) / Math.max(1, ...model.totals));
    const cols = model.labels.map((_, i) => {
      const x = round1(2 + i * step);
      let y = box.base;
      const rects = [];
      model.series.forEach((s) => {
        const v = s.values[i];
        if (!(v > 0)) return;
        const h = round1(v * unit);
        y = round1(y - h);
        rects.push(`<rect class="seg ${s.tone}" x="${x}" y="${y}" width="${colW}" height="${h}"/>`);
      });
      if (!rects.length) rects.push(`<rect class="seg empty" x="${x}" y="${box.base - 2}" width="${colW}" height="2"/>`);
      return `<g class="wk"><title>${esc(weekTitle(model, i))}</title>${rects.join('')}</g>`;
    }).join('');
    const axis = `<line class="axis" x1="0" x2="${box.plotW}" y1="${box.base + 0.5}" y2="${box.base + 0.5}"/>`;
    const first = `<text x="2" y="${box.label}">${esc(model.labels[0])}</text>`;
    const last = n > 1 ? `<text x="${round1(2 + (n - 1) * step + colW)}" y="${box.label}" text-anchor="end">${esc(model.labels[n - 1])}</text>` : '';
    return `<svg class="dash-weeks-svg" viewBox="0 0 ${box.width} ${box.height}" width="${box.width}" height="${box.height}" aria-hidden="true">${cols}${axis}${first}${last}</svg>`;
  }
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
      value === null ? LL.t('dashboard.spark.no_rps') : `${formatRps(value)} RPS`,
      run.test_type ? LL.testTypeLabel(run.test_type) : '',
      LL.verdictLabel(run.verdict)
    ].filter(Boolean).join(' · ');
  }
  // Inline SVG of history[].max_rps (history is newest first, drawn oldest to newest): squares in the verdict
  // color on a hairline, the newest one larger; a run without RPS is a hollow square on the bottom edge.
  // The line joins only runs of the newest run's test type (a soak level between step peaks is not a
  // drop); runs of other types stay as faded squares.
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
      const size = i === pts.length - 1 ? SPARK.sizeLast : SPARK.size;
      const other = typeKey(p.run.test_type) !== mainType ? ' other' : '';
      const cls = `dash-spark-dot ${LL.verdictClass(p.run.verdict)}${p.value === null ? ' empty' : ''}${other}`;
      return `<g class="dash-spark-point"><title>${esc(sparkTitle(p.run, p.value))}</title>`
        + `<rect class="dash-spark-hit" x="${round1(p.x - SPARK.hit)}" y="${round1(p.y - SPARK.hit)}" width="${2 * SPARK.hit}" height="${2 * SPARK.hit}"/>`
        + `<rect class="${cls}" x="${round1(p.x - size / 2)}" y="${round1(p.y - size / 2)}" width="${size}" height="${size}"/></g>`;
    }).join('');
    const label = LL.t('dashboard.spark.aria', { values: pts.map((p) => (p.value === null ? LL.t('dashboard.spark.no_data') : formatRps(p.value))).join(', ') });
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
  // Facts of the last run: max RPS against the previous run of the same type, test window, test type.
  function heroKpis(last, kpi) {
    const info = kpi || {};
    const type = typeKey(last.test_type);
    const rps = finite(last.max_rps) !== null ? last.max_rps : (info.last && info.last.max_rps);
    const previous = info.previous && info.previous.run_name ? info.previous : null;
    const delta = previous ? rpsDelta(rps, previous.max_rps) : null;
    let note = LL.t('dashboard.hero.no_previous');
    if (previous) note = LL.t(delta ? 'dashboard.hero.vs_previous' : 'dashboard.hero.vs_previous_no_data', { name: previous.run_name });
    const start = finite(last.start_ms);
    const end = finite(last.end_ms);
    const caption = rpsCaption(type) || (type === 'step' && last.test_type ? LL.t('dashboard.rps.caption_step') : '');
    return [
      {
        key: 'rps',
        label: rpsLabel(type) || 'Max RPS',
        value: formatRps(rps),
        delta,
        note,
        title: previous ? LL.t('dashboard.hero.previous_title', { name: previous.run_name, rps: formatRps(previous.max_rps) }) : ''
      },
      { key: 'window', label: LL.t('dashboard.hero.window'), value: start !== null && end !== null && end >= start ? LL.formatDuration(end - start) : '—', note: periodText(start, end) },
      { key: 'type', label: LL.t('dashboard.hero.test_type'), value: LL.testTypeLabel(last.test_type), note: caption ? `RPS: ${caption}` : '' }
    ];
  }
  // Counters of /dashboard_data as value and note (the jobs counter comes from /jobs); the notes are the
  // tooltips of the status line.
  function kpiTiles(data, now) {
    const kpi = (data && data.kpi) || {};
    const runs7 = finite(kpi.runs_7d);
    const runs30 = finite(kpi.runs_30d);
    const share = finite(kpi.risky_share_30d);
    const risky = finite(kpi.risky_30d);
    const last = data && data.last_run && data.last_run.run_name ? data.last_run : null;
    return {
      runs: { value: runs7 === null ? '—' : String(runs7), note: runs30 === null ? '' : LL.t('dashboard.kpi.runs_note', { count: runs30 }) },
      risky: {
        value: shareText(share),
        meter: share === null ? 0 : Math.max(0, Math.min(1, share)),
        note: share === null ? LL.t('dashboard.kpi.risky_none') : LL.t('dashboard.kpi.risky_note', { risky: risky === null ? Math.round(share * runs30) : risky, total: runs30 })
      },
      updated: {
        value: last && last.created_at ? LL.relativeTime(last.created_at, now) : '—',
        note: last ? LL.t('dashboard.kpi.updated_note', { name: last.run_name }) : LL.t('dashboard.kpi.updated_none'),
        title: last && last.created_at ? LL.formatDateTime(last.created_at) : ''
      }
    };
  }
  // The status line under the title: «6 runs in 7 days», «12 in 30 days, 7 at risk» and the time since the
  // last report; every item is {value, note, title} (value '' hides the number).
  function statusLine(data, now) {
    const kpi = (data && data.kpi) || {};
    const tiles = kpiTiles(data, now);
    const runs7 = finite(kpi.runs_7d);
    const runs30 = finite(kpi.runs_30d);
    const share = finite(kpi.risky_share_30d);
    let risky = finite(kpi.risky_30d);
    if (risky === null && share !== null && runs30 !== null) risky = Math.round(share * runs30);
    const last = data && data.last_run && data.last_run.run_name ? data.last_run : null;
    const nowMs = now === undefined ? Date.now() : Number(now);
    const created = last && last.created_at ? Date.parse(last.created_at) : NaN;
    const title = [tiles.updated.title, tiles.updated.note].filter(Boolean).join(' · ');
    let updated = { value: '', note: LL.t('dashboard.kpi.updated_none'), title: '' };
    if (Number.isFinite(created) && Number.isFinite(nowMs)) {
      // Under a minute «34 s since the last report» reads oddly: the report is «just now».
      updated = nowMs - created < 60000
        ? { value: '', note: LL.t('dashboard.status.report_just_now'), title }
        : { value: LL.formatDuration(nowMs - created), note: LL.t('dashboard.status.since_report'), title };
    }
    return {
      runs: { value: tiles.runs.value, note: LL.tn('dashboard.status.runs_7d', runs7 === null ? 0 : runs7), title: '' },
      risky: {
        value: runs30 === null ? '—' : String(runs30),
        note: LL.t('dashboard.status.runs_30d'),
        warn: risky ? LL.t('dashboard.status.risky', { count: risky }) : '',
        title: tiles.risky.note
      },
      updated
    };
  }
  function jobsTile(jobs) {
    if (!Array.isArray(jobs)) return { value: '—', note: LL.t('dashboard.kpi.jobs_unavailable') };
    if (!jobs.length) return { value: '0', note: LL.t('dashboard.kpi.jobs_none') };
    const first = jobs[0] || {};
    return {
      value: jobs.length >= JOBS_LIMIT ? `${JOBS_LIMIT}+` : String(jobs.length),
      note: `${first.run_name || LL.t('dashboard.untitled')} · ${Math.round(finite(first.progress) || 0)}%`
    };
  }
  // «0 report jobs running» of the status line; the first job (or why the list is missing) is the tooltip.
  function jobsStat(jobs) {
    const tile = jobsTile(jobs);
    return { value: tile.value, note: LL.tn('dashboard.status.jobs', Array.isArray(jobs) ? jobs.length : 0), title: tile.note };
  }

  // '12 min of 30 min': how long the load goes against the run's limit; before the start only the limit.
  function elapsedText(startedAt, maxSeconds, now) {
    const max = finite(maxSeconds);
    const limit = max && max > 0 ? LL.formatDuration(max * 1000) : '';
    const started = startedAt ? Date.parse(startedAt) : NaN;
    if (!Number.isFinite(started)) return limit ? LL.t('dashboard.active.limit', { limit }) : '';
    const gone = LL.formatDuration(Math.max(0, (now === undefined ? Date.now() : now) - started));
    return limit ? LL.t('dashboard.active.elapsed_of', { elapsed: gone, limit }) : gone;
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
      if (finite(live.errors) !== null) liveText += ` · ${LL.t('dashboard.active.errors', { value: errorsText(live.errors) })}`;
    } else if (live && live.message) {
      liveNote = String(live.message);
    } else if (r.loading) {
      liveNote = LL.t('dashboard.active.metrics_after_start');
    } else {
      liveNote = String(r.message || '');
    }
    return {
      name: r.run_name || LL.t('dashboard.untitled'),
      url: r.url || `/launch/${encodeURIComponent(r.run_id || '')}`,
      status: r.status_title || r.status || '',
      tone: RUN_TONE[r.status] || 'na',
      meta: [r.service, r.owner].filter(Boolean).join(' · '),
      progress: [
        total > 1 ? LL.t('dashboard.active.instances', { running: going, total }) : '',
        r.loading ? elapsedText(r.started_at, r.max_seconds, now) : ''
      ].filter(Boolean).join(' · '),
      liveText,
      liveNote,
      points: live && Array.isArray(live.points) ? live.points : [],
      canStop: !!r.can_stop
    };
  }
  function activeTile(data, now) {
    if (!data || !Array.isArray(data.runs)) return { value: '—', note: LL.t('dashboard.kpi.active_unavailable') };
    const total = finite(data.total) || 0;
    if (!total) return { value: '0', note: LL.t('dashboard.kpi.active_none') };
    const first = data.runs[0];
    if (!first) return { value: String(total), note: '' };
    const elapsed = first.loading ? elapsedText(first.started_at, first.max_seconds, now) : (first.status_title || '');
    return { value: String(total), note: [first.run_name || LL.t('dashboard.untitled'), elapsed].filter(Boolean).join(' · ') };
  }
  // «1 test running now» of the status line (engineers and admins); the first run is the tooltip.
  function activeStat(data, now) {
    const tile = activeTile(data, now);
    const total = data && Array.isArray(data.runs) ? finite(data.total) || 0 : 0;
    return { value: tile.value, note: LL.tn('dashboard.status.active', total), title: tile.note };
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

  // ---- trend in words --------------------------------------------------------------------------
  // How long ago a run was, for «than a week ago»: the time today, yesterday, days, a week, weeks, a month,
  // months (calendar days in the browser clock); '' without a date.
  function ageWords(value, now) {
    const time = value ? Date.parse(value) : NaN;
    const nowMs = now === undefined ? Date.now() : Number(now);
    if (!Number.isFinite(time) || !Number.isFinite(nowMs)) return '';
    const midnight = (ms) => { const d = new Date(ms); d.setHours(0, 0, 0, 0); return d.getTime(); };
    const days = Math.round((midnight(nowMs) - midnight(time)) / DAY_MS);
    if (days <= 0) {
      const d = new Date(time);
      const pad = (n) => String(n).padStart(2, '0');
      return LL.t('dashboard.trend.when.time', { time: `${pad(d.getHours())}:${pad(d.getMinutes())}` });
    }
    if (days === 1) return LL.t('dashboard.trend.when.yesterday');
    if (days < 7) return LL.tn('dashboard.trend.when.days', days);
    if (days < 14) return LL.t('dashboard.trend.when.week');
    if (days < 30) return LL.tn('dashboard.trend.when.weeks', Math.floor(days / 7));
    if (days < 60) return LL.t('dashboard.trend.when.month');
    return LL.tn('dashboard.trend.when.months', Math.floor(days / 30));
  }
  // «27 %»: whole percent from 10 up, one decimal below, in the number format of the UI language.
  function pctWords(pct) {
    const abs = Math.abs(Number(pct) || 0);
    const digits = abs >= 10 ? 0 : 1;
    const value = LL.fmt && typeof LL.fmt.number === 'function'
      ? LL.fmt.number(abs, { maximumFractionDigits: digits })
      : String(Number(abs.toFixed(digits)));
    return LL.t('dashboard.trend.pct', { value });
  }
  // «27 % better» / «5.2 % worse» / «no change» of an rpsDelta (more RPS is better).
  function deltaWords(delta) {
    if (!delta) return '';
    if (delta.direction === 'up') return LL.t('dashboard.trend.better', { pct: pctWords(delta.pct) });
    if (delta.direction === 'down') return LL.t('dashboard.trend.worse', { pct: pctWords(delta.pct) });
    return LL.t('dashboard.trend.flat');
  }
  // Trend of a service in words: the max RPS of the newest run against the previous run of the same type,
  // {direction, text, since: «than a week ago» | «same as a week ago», previous}; null without a comparison.
  function trendWords(history, now) {
    const delta = historyDelta(history);
    if (!delta) return null;
    const when = ageWords(delta.previous.created_at, now);
    let since = '';
    if (when) since = delta.direction === 'flat' ? LL.t('dashboard.trend.same', { when }) : LL.t('dashboard.trend.than', { when });
    return { direction: delta.direction, text: deltaWords(delta), since, previous: delta.previous };
  }

  // ---- figures of the last run ---------------------------------------------------------------------
  function checkOf(checks, name) {
    return checks.find((c) => c.name === name) || null;
  }
  function passTone(passed) {
    return passed === true ? 'ok' : (passed === false ? 'fail' : 'na');
  }
  // A threshold or a small value: integers as they are, otherwise up to two decimals.
  function numText(value) {
    const n = finite(value);
    if (n === null) return '—';
    return Number.isInteger(n) ? String(n) : String(Number(n.toFixed(2)));
  }
  // Latency in ms: whole from 100 up, one decimal below.
  function msText(value) {
    const n = finite(value);
    if (n === null) return '—';
    return n >= 100 ? String(Math.round(n)) : String(Number(n.toFixed(1)));
  }
  // The 3px bar «fact against threshold»: the bar spans FIG_SCALE thresholds; fill and tick in percent.
  function figBar(actual, threshold, tone) {
    const a = finite(actual);
    const th = finite(threshold);
    if (a === null || th === null || th <= 0) return null;
    return { fill: round1(Math.max(0, Math.min(100, (a / (th * FIG_SCALE)) * 100))), tick: round1(100 / FIG_SCALE), tone };
  }
  // A «not above the limit» check as a figure: its value, the limit and «met» / «breached».
  function limitFigure(key, check, unit, format, caption) {
    const actual = finite(check.actual);
    const tone = passTone(check.passed);
    let word = LL.t('dashboard.fig.unknown');
    if (tone === 'ok') word = LL.t('dashboard.fig.kept');
    else if (tone === 'fail') word = LL.t('dashboard.fig.broken');
    return {
      key,
      value: actual === null ? '—' : format(actual),
      unit: actual === null ? '' : unit,
      tone: check.passed === false ? 'fail' : '',
      caption,
      word,
      wordTone: tone,
      bar: figBar(actual, check.threshold, tone),
      title: check.title || ''
    };
  }
  // Figures of the last-run block: max RPS (against the target_rps threshold when there is one), p95 and the
  // error rate against their limits, «n / m» SLA criteria with one square per check. A run without p95 and
  // error checks shows its test window instead, so the block never shrinks to a single number.
  function lastFigures(last, kpi) {
    const l = last || {};
    const info = kpi || {};
    const checks = (Array.isArray(l.checks) ? l.checks : []).filter((c) => c && typeof c === 'object');
    const tiles = heroKpis(l, info);
    const rpsTile = tiles[0];
    const windowTile = tiles[1];
    const rps = finite(l.max_rps) !== null ? finite(l.max_rps) : finite(info.last && info.last.max_rps);
    const target = checkOf(checks, 'target_rps');
    const targetValue = target ? finite(target.threshold) : null;
    const targetTone = target ? passTone(target.passed) : '';
    let targetWord = '';
    if (targetValue !== null && target.passed === true) targetWord = LL.t('dashboard.fig.reached');
    else if (targetValue !== null && target.passed === false) {
      // No step at the target held the latency and error thresholds: the RPS ran, the target is not confirmed.
      targetWord = finite(target.actual) === null ? LL.t('dashboard.fig.not_confirmed') : LL.t('dashboard.fig.not_reached');
    }
    const figs = [{
      key: 'rps',
      value: formatRps(rps),
      unit: rps === null ? '' : 'RPS',
      tone: target && target.passed === false ? 'fail' : '',
      caption: [rpsTile.label, targetValue !== null ? LL.t('dashboard.fig.target', { value: numText(targetValue) }) : ''].filter(Boolean).join(' · '),
      word: targetWord,
      wordTone: targetTone,
      bar: targetValue !== null ? figBar(rps, targetValue, targetTone) : null,
      title: rpsTile.title
    }];
    const p95 = checkOf(checks, 'p95_latency');
    if (p95) {
      const caption = finite(p95.threshold) !== null ? LL.t('dashboard.fig.p95', { value: numText(p95.threshold) }) : 'p95';
      figs.push(limitFigure('p95', p95, LL.t('dashboard.unit.ms'), msText, caption));
    }
    const errors = checkOf(checks, 'error_rate');
    if (errors) {
      const caption = finite(errors.threshold) !== null ? LL.t('dashboard.fig.errors', { value: numText(errors.threshold) }) : LL.t('dashboard.fig.errors_label');
      figs.push(limitFigure('errors', errors, '%', numText, caption));
    }
    if (!p95 && !errors) {
      figs.push({ key: 'window', value: windowTile.value, unit: '', tone: '', caption: windowTile.label, note: windowTile.note });
    }
    const sla = slaBadge(l.sla);
    if (sla.total) {
      const judged = checks.filter((c) => c.passed === true || c.passed === false);
      const squares = judged.length
        ? judged.map((c) => ({ tone: passTone(c.passed), title: c.title || c.name }))
        : Array.from({ length: sla.total }, (_, i) => ({ tone: i < sla.passed ? 'ok' : 'fail', title: '' }));
      figs.push({ key: 'sla', value: String(sla.passed), unit: `/ ${sla.total}`, tone: '', caption: LL.t('dashboard.fig.sla'), squares, title: sla.title });
    }
    return figs;
  }
  function figureHtml(fig) {
    const unit = fig.unit ? `<small>${esc(fig.unit)}</small>` : '';
    const word = fig.word ? ` <b>${esc(fig.word)}</b>` : '';
    const note = fig.note ? `<div class="dash-fig-note">${esc(fig.note)}</div>` : '';
    const bar = fig.bar
      ? `<div class="fig-bar" aria-hidden="true"><i class="${fig.bar.tone}" style="width:${fig.bar.fill}%"></i><b style="left:${fig.bar.tick}%"></b></div>`
      : '';
    const squares = fig.squares && fig.squares.length
      ? `<div class="dash-fig-squares">${fig.squares.map((s) => `<span class="sq ${s.tone}"${s.title ? ` title="${esc(s.title)}"` : ''}></span>`).join('')}</div>`
      : '';
    return `<div class="fig dash-fig-${fig.key}"${fig.title ? ` title="${esc(fig.title)}"` : ''}>`
      + `<div class="v${fig.tone ? ` ${fig.tone}` : ''}">${esc(fig.value)}${unit}</div>`
      + `<div class="k${fig.wordTone ? ` ${fig.wordTone}` : ''}">${esc(fig.caption)}${word}</div>${note}${bar}${squares}</div>`;
  }
  // The test type and, unless a window figure shows it, the test window and its length (the meta line).
  function heroPeriod(last, withWindow) {
    return heroPeriodParts(last, withWindow).join(' · ');
  }
  function heroPeriodParts(last, withWindow) {
    const l = last || {};
    const parts = [l.test_type ? LL.testTypeLabel(l.test_type) : ''];
    if (withWindow) {
      const start = finite(l.start_ms);
      const end = finite(l.end_ms);
      // The window as «2026-10-15 19:10–20:10» (the end keeps only its time on the same day).
      parts.push(periodText(start, end).replace(/ — (\d{2}:\d{2})$/, '–$1'));
      if (start !== null && end !== null && end >= start) parts.push(LL.formatDuration(end - start));
    }
    return parts.filter(Boolean);
  }
  // «previous r3 · Max RPS 27 % better» next to the actions.
  function previousNote(last, kpi) {
    const info = kpi || {};
    const previous = info.previous && info.previous.run_name ? info.previous : null;
    if (!previous) return LL.t('dashboard.hero.no_previous');
    const rps = finite(last && last.max_rps) !== null ? last.max_rps : (info.last && info.last.max_rps);
    const delta = rpsDelta(rps, previous.max_rps);
    if (!delta) return LL.t('dashboard.hero.previous_note_no_data', { name: previous.run_name });
    return LL.t('dashboard.hero.previous_note', { name: previous.run_name, change: deltaWords(delta) });
  }
  // Caption of the load profile: «SLA · step 5 · 250 RPS · 19:50–19:54» from model.sla; '' without a zone.
  function profileCaption(model) {
    const sla = model && model.sla;
    if (!sla) return '';
    const range = { from: sla.startText, to: sla.endText };
    if (sla.index !== null && sla.index !== undefined && sla.index !== '' && finite(sla.rps) !== null) {
      const rps = finite(sla.rps);
      return LL.t('dashboard.profile.sla_step', Object.assign({ index: sla.index, rps: Number.isInteger(rps) ? String(rps) : rps.toFixed(1) }, range));
    }
    return LL.t('dashboard.profile.sla_window', range);
  }

  // ---- rendering ----------------------------------------------------------------------------
  // ``manual`` ({verdict, reason, author, changed_at}) marks a verdict set by hand.
  function pill(verdict, extra, manual) {
    const title = LL.manualVerdictTitle(manual);
    const mark = title ? `<span class="verdict-manual-mark" title="${esc(title)}">${esc(LL.t('dashboard.verdict.manual_mark'))}</span>` : '';
    return `<span class="pill ${LL.verdictClass(verdict)}${extra ? ` ${extra}` : ''}"${title ? ` title="${esc(title)}"` : ''}>${esc(LL.verdictLabel(verdict))}</span>${mark}`;
  }
  function trendHtml(trend, title) {
    if (!trend) return '';
    return `<div class="dash-trend-words"${title ? ` title="${esc(title)}"` : ''}><span class="dash-delta ${trend.direction}">${esc(trend.text)}</span>`
      + `${trend.since ? `<span class="sub">${esc(trend.since)}</span>` : ''}</div>`;
  }
  function setText(id, text, title) {
    const el = $(id);
    if (!el) return;
    el.textContent = text === '' || text === null || text === undefined ? '\u00a0' : String(text);
    if (title !== undefined) el.title = title || '';
  }
  // Text that may be empty (no placeholder space, so :empty rules apply).
  function setPlain(id, text) {
    const el = $(id);
    if (el) el.textContent = text ? String(text) : '';
  }
  // One item of the status line: the number in <b> (hidden when ''), the note (text or trusted markup) and the
  // tooltip on the item.
  function setStat(boxId, valueId, noteId, item, noteHtml) {
    const value = $(valueId);
    const note = $(noteId);
    const box = $(boxId);
    if (value) {
      value.textContent = item.value;
      value.hidden = !item.value;
    }
    if (note) {
      if (noteHtml === undefined) note.textContent = item.note;
      else note.innerHTML = noteHtml;
    }
    if (box) box.title = item.title || '';
  }

  function renderSkeletons() {
    $('servicesBody').innerHTML = ui.skeletonRows(6, 3);
    $('recentBody').innerHTML = ui.skeletonRows(5, 3);
    $('lastKpis').innerHTML = [1, 2, 3].map(() => `<div class="fig dash-fig-skeleton" aria-hidden="true">${ui.skeleton(2, ['w-40', 'w-60'])}</div>`).join('');
  }

  function renderStatus(data) {
    const s = statusLine(data);
    setStat('kpiRuns', 'kpiRunsValue', 'kpiRunsNote', s.runs);
    const risky = s.risky.warn
      ? LL.tHtml('dashboard.status.runs_30d_risky', { risky: LL.html(`<span class="warn">${esc(s.risky.warn)}</span>`) })
      : esc(s.risky.note);
    setStat('kpiRisky', 'kpiRiskyValue', 'kpiRiskyNote', s.risky, risky);
    setStat('kpiUpdated', 'kpiUpdatedValue', 'kpiUpdatedNote', s.updated);
  }

  // Draws the load profile of the last run on the right of its block; hides the column without one.
  // The profile is redrawn only when the payload changes (the block re-renders every minute for the clock).
  function renderProfile(profile) {
    const box = $('lastProfileBox');
    const host = $('lastProfile');
    const card = $('lastCard');
    if (!box || !host) return;
    const p = profile && Array.isArray(profile.steps) && profile.steps.length ? profile : null;
    const charts = LL.charts && typeof LL.charts.stepProfile === 'function' ? LL.charts : null;
    const key = p && charts ? JSON.stringify(p) : '';
    if (key === profileKey) return;
    profileKey = key;
    if (profileHandle) {
      profileHandle.destroy();
      profileHandle = null;
    }
    const caption = $('lastProfileSla');
    if (!key) {
      box.hidden = true;
      host.innerHTML = '';
      if (card) card.classList.remove('has-profile');
      if (caption) caption.textContent = '';
      return;
    }
    // Visible first: the profile takes the width of its host.
    box.hidden = false;
    if (card) card.classList.add('has-profile');
    profileHandle = charts.stepProfile(host, {
      steps: p.steps, window: p.window || null, markers: Array.isArray(p.markers) ? p.markers : [], target: p.target, compact: true
    });
    const model = profileHandle && profileHandle.model;
    if (!model) {
      box.hidden = true;
      if (card) card.classList.remove('has-profile');
    }
    if (caption) caption.textContent = profileCaption(model);
  }

  function renderHero(last, kpi, emptyTitle) {
    const has = !!(last && last.run_name);
    const card = $('lastCard');
    $('lastReportLink').hidden = !has;
    $('lastCompareLink').hidden = !has;
    if (card) {
      RUN_TONES.forEach((tone) => card.classList.remove(tone));
      card.classList.add(`tone-${has ? LL.verdictClass(last.verdict) : 'na'}`);
    }
    if (!has) {
      setText('lastTitle', emptyTitle || LL.t('dashboard.empty.title'), '');
      setText('lastMeta', emptyTitle ? '' : LL.t('dashboard.hero.empty_hint'));
      setPlain('lastPeriod', '');
      setPlain('lastPrevNote', '');
      $('lastBadges').innerHTML = '';
      $('lastKpis').innerHTML = '';
      renderProfile(null);
      return;
    }
    setText('lastTitle', last.run_name, last.run_name);
    setText('lastMeta', [last.service || '—', last.created_at ? LL.t('dashboard.hero.created', { time: LL.relativeTime(last.created_at) }) : ''].filter(Boolean).join(' · '));
    const figs = lastFigures(last, kpi);
    // Each part stays whole: the line breaks only between the parts.
    $('lastPeriod').innerHTML = heroPeriodParts(last, !figs.some((f) => f.key === 'window'))
      .map((part) => `<span class="nowrap">${esc(part)}</span>`).join(' · ');
    // Without SLA criteria there is no «n / m» figure: the block says so next to the verdict.
    const sla = slaBadge(last.sla);
    const slaMark = sla.total ? '' : `<span class="pill na" title="${esc(sla.title)}">${esc(sla.text)}</span>`;
    $('lastBadges').innerHTML = `${pill(last.verdict, 'pill-lg', last.manual_verdict)}${slaMark}`;
    $('lastKpis').innerHTML = figs.map(figureHtml).join('');
    $('lastReportLink').href = LL.reportUrl(last);
    const previous = kpi && kpi.previous && kpi.previous.run_name ? kpi.previous : null;
    $('lastCompareLink').href = `/compare?run_a=${encodeURIComponent(last.run_name)}${previous ? `&run_b=${encodeURIComponent(previous.run_name)}` : ''}`;
    setPlain('lastPrevNote', previousNote(last, kpi));
    renderProfile(last.profile);
  }

  function renderWeeks(weeks) {
    const model = weeksModel(weeks);
    const host = $('weeksChart');
    setText('weeksTotal', weeksTotalText(model));
    $('weeksEmpty').hidden = model.total > 0;
    if (!host) return;
    host.hidden = !model.total;
    host.setAttribute('aria-label', weeksSummary(model));
    host.innerHTML = model.total ? weeksSvg(model) : '';
  }

  // Older payloads carry only the verdicts of the service; the sparkline then shows verdict points.
  function serviceHistory(svc) {
    if (Array.isArray(svc.history) && svc.history.length) return svc.history;
    const verdicts = Array.isArray(svc.recent_verdicts) && svc.recent_verdicts.length ? svc.recent_verdicts : [svc.verdict];
    return verdicts.map((verdict, i) => (i === 0
      ? { run_name: svc.last_run, run_id: svc.run_id || '', created_at: svc.created_at, verdict, max_rps: svc.max_rps, test_type: svc.test_type }
      : { verdict, max_rps: null }));
  }
  function serviceRowHtml(svc, now) {
    const history = serviceHistory(svc);
    const previous = previousOfSameType(history);
    const trend = trendWords(history, now);
    const rps = finite(svc.max_rps);
    const caption = rpsCaption(typeKey(svc.test_type));
    const deltaTitle = trend ? LL.t('dashboard.services.delta_title', { name: trend.previous.run_name || '—', rps: formatRps(trend.previous.max_rps) }) : '';
    const compare = `/compare?run_a=${encodeURIComponent(svc.last_run || '')}${previous && previous.run_name ? `&run_b=${encodeURIComponent(previous.run_name)}` : ''}`;
    const compareTitle = previous && previous.run_name
      ? LL.t('dashboard.services.compare_title_previous', { run: svc.last_run, previous: previous.run_name })
      : LL.t('dashboard.services.compare_title', { run: svc.last_run });
    const created = svc.created_at ? LL.relativeTime(svc.created_at) : '—';
    const createdFull = svc.created_at ? LL.formatDateTime(svc.created_at) : '';
    return `<tr>
        <td class="name dash-service" title="${esc(svc.service)}"><a href="/reports?service=${encodeURIComponent(svc.service)}">${esc(svc.service)}</a></td>
        <td class="run-name" title="${esc(svc.last_run)}"><a href="${esc(LL.reportUrl({ run_id: svc.run_id, run_name: svc.last_run }))}">${esc(svc.last_run)}</a><span class="sub">${esc(LL.testTypeLabel(svc.test_type))} · <span class="mono" title="${esc(createdFull)}">${esc(created)}</span></span></td>
        <td>${pill(svc.verdict)}</td>
        <td class="n">${esc(formatRps(rps))}${rps !== null && caption ? `<span class="sub">${esc(caption)}</span>` : ''}</td>
        <td class="dash-trend"><div class="dash-trend-cell">${sparkline(history)}${trendHtml(trend, deltaTitle)}</div></td>
        <td class="actions nowrap"><a class="link" href="${esc(compare)}" title="${esc(compareTitle)}" aria-label="${esc(compareTitle)}">${esc(LL.t('dashboard.services.compare'))}</a></td>
      </tr>`;
  }
  function renderServices(services) {
    const body = $('servicesBody');
    body.innerHTML = services.length ? services.map((svc) => serviceRowHtml(svc)).join('') : ui.emptyRow(6, LL.t('dashboard.services.empty'));
  }

  function renderRecent(runs) {
    const body = $('recentBody');
    if (!runs.length) {
      body.innerHTML = ui.emptyRow(5, LL.t('dashboard.recent.empty'));
      return;
    }
    body.innerHTML = '';
    runs.forEach((r) => {
      const tr = document.createElement('tr');
      tr.className = 'clickable';
      const href = LL.reportUrl(r);
      tr.innerHTML = `
        <td class="name run-name" title="${esc(r.run_name)}"><a href="${esc(href)}">${esc(r.run_name)}</a></td>
        <td>${esc(r.service || '—')}</td>
        <td class="muted">${esc(LL.testTypeLabel(r.test_type))}</td>
        <td>${pill(r.verdict, '', r.manual_verdict)}</td>
        <td class="n muted" title="${esc(r.created_at ? LL.formatDateTime(r.created_at) : '')}">${esc(r.created_at ? LL.relativeTime(r.created_at) : '—')}</td>`;
      tr.addEventListener('click', (ev) => {
        if (ev.target && ev.target.closest && ev.target.closest('a')) return;
        location.href = href;
      });
      body.appendChild(tr);
    });
  }

  function jobRowHtml(j) {
    return `<div class="job-row"><span class="pill running">${esc(LL.t('dashboard.jobs.in_progress'))}</span><div><div class="name">${esc(j.run_name || LL.t('dashboard.untitled'))}</div><div class="meta">${esc(j.service || '')} · ${Math.round(finite(j.progress) || 0)}% — ${esc(j.message || '')}</div></div><a href="/new?job=${encodeURIComponent(j.job_id)}" data-min-role="engineer">${esc(LL.t('dashboard.jobs.follow'))}</a></div>`;
  }
  function renderJobs(jobs) {
    setStat('kpiJobs', 'kpiJobsValue', 'kpiJobsNote', jobsStat(jobs));
    const list = Array.isArray(jobs) ? jobs : [];
    $('jobsCard').hidden = !list.length;
    const more = list.length > JOBS_SHOWN ? `<div class="dim dash-jobs-more">${esc(LL.t('dashboard.jobs.more', { count: list.length - JOBS_SHOWN }))}</div>` : '';
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
      ? `<div class="dash-active-live" title="${esc(LL.t('dashboard.active.live_title'))}">${liveSpark(s.points)}<span class="dash-active-value">${esc(s.liveText)}</span></div>`
      : `<div class="dash-active-live dim">${esc(s.liveNote)}</div>`;
    const stop = s.canStop
      ? `<button type="button" class="btn btn-sm btn-danger" data-stop-run="${esc(run.run_id || '')}" data-run-name="${esc(s.name)}">${esc(LL.t('dashboard.active.stop'))}</button>`
      : '';
    return `<div class="dash-active-row"><span class="pill ${s.tone}">${esc(s.status)}</span>`
      + `<div class="dash-active-main"><a class="name" href="${esc(s.url)}">${esc(s.name)}</a>`
      + `${s.meta ? `<div class="meta">${esc(s.meta)}</div>` : ''}${s.progress ? `<div class="meta">${esc(s.progress)}</div>` : ''}</div>`
      + `${live}<div class="dash-active-actions"><a class="btn btn-sm" href="${esc(s.url)}">${esc(LL.t('dashboard.active.open'))}</a>${stop}</div></div>`;
  }
  // The count of runs not listed plus a link to all runs, under the running tests; '' when every run is shown.
  function activeMoreHtml(rest) {
    if (!(rest > 0)) return '';
    return LL.tHtml('dashboard.active.more', {
      count: rest,
      link: LL.html(`<a href="/launch#runsPanel">${esc(LL.t('dashboard.active.more_link'))}</a>`)
    });
  }
  function renderActive(data) {
    setStat('activeTile', 'kpiActiveValue', 'kpiActiveNote', activeStat(data));
    const section = $('activeRuns');
    if (!section) return;
    const runs = data && Array.isArray(data.runs) ? data.runs : [];
    section.hidden = !runs.length;
    $('activeList').innerHTML = runs.slice(0, ACTIVE_SHOWN).map((r) => activeRowHtml(r)).join('');
    const rest = (data ? finite(data.total) || 0 : 0) - Math.min(runs.length, ACTIVE_SHOWN);
    $('activeMore').hidden = rest <= 0;
    $('activeMore').innerHTML = activeMoreHtml(rest);
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
      title: LL.t('dashboard.stop.title', { name }),
      message: LL.t('dashboard.stop.message'),
      confirmText: LL.t('dashboard.active.stop'),
      danger: true
    });
    if (!ok) return;
    try {
      const resp = await fetch(`/load_tests/runs/${encodeURIComponent(runId)}/stop`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
      });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(body.error || `HTTP ${resp.status}`);
      ui.toast(LL.t('dashboard.stop.requested'), { tone: 'ok' });
    } catch (e) {
      ui.toast(LL.t('dashboard.stop.failed', { error: e.message }), { tone: 'error' });
    }
    loadActive();
  }

  async function seedDemo(button) {
    button.disabled = true;
    ui.toast(LL.t('dashboard.demo.creating'), { tone: 'info', timeout: 3000 });
    try {
      const resp = await fetch('/demo/seed', { method: 'POST' });
      const data = await resp.json();
      if (resp.status === 409 && data.report_url) { ui.toast(LL.t('dashboard.demo.exists'), { tone: 'info' }); location.href = data.report_url; return; }
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      ui.toast(LL.t('dashboard.demo.created'), { tone: 'ok' });
      location.href = data.report_url;
    } catch (e) {
      ui.toast(LL.t('dashboard.demo.failed', { error: e.message }), { tone: 'error' });
      button.disabled = false;
    }
  }

  // The demo stack (LOADLENS_DEMO_SEED=1): a dismissible notice that the runs are sample data.
  const DEMO_BANNER_PREF = 'demoBannerDismissed';

  async function loadDemoBanner() {
    const banner = $('demoBanner');
    if (!banner || LL.prefs.get(DEMO_BANNER_PREF, false)) return;
    try {
      const resp = await fetch('/demo/status');
      if (!resp.ok) return;
      const status = await resp.json();
      banner.hidden = !(status && status.enabled && status.exists);
    } catch (e) {
      /* the banner is optional: a failed lookup leaves it hidden */
    }
  }

  function dismissDemoBanner() {
    LL.prefs.set(DEMO_BANNER_PREF, true);
    const banner = $('demoBanner');
    if (banner) banner.hidden = true;
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
      $('setupBanner').querySelector('.muted').textContent = LL.t('dashboard.load_failed', { error: e.message });
      $('servicesBody').innerHTML = ui.emptyRow(6, LL.t('dashboard.no_data'));
      $('recentBody').innerHTML = ui.emptyRow(5, LL.t('dashboard.no_data'));
      $('lastCard').hidden = false;
      renderHero(null, null, LL.t('dashboard.no_data'));
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
      $('emptyState').querySelector('h3').textContent = LL.t('dashboard.empty.area_title', { area });
    }
    // The empty state explains an empty dashboard; the last-run block would only repeat it.
    $('lastCard').hidden = empty;
    renderStatus(data);
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
      weeksTotalText,
      weeksTooltipTitle,
      weeksTooltipFooter,
      weekTitle,
      weeksSvg,
      sparkPoints,
      sparkline,
      periodText,
      heroKpis,
      kpiTiles,
      statusLine,
      jobsTile,
      jobsStat,
      verdictPill: pill,
      renderHero,
      serviceHistory,
      elapsedText,
      errorsText,
      activeRunSummary,
      activeTile,
      activeStat,
      liveSpark,
      activeRowHtml,
      activeMoreHtml,
      serviceRowHtml,
      jobRowHtml,
      ageWords,
      pctWords,
      deltaWords,
      trendWords,
      lastFigures,
      figureHtml,
      heroPeriod,
      previousNote,
      profileCaption
    }
  };

  document.addEventListener('DOMContentLoaded', () => {
    ['bannerDemoBtn', 'emptyDemoBtn'].forEach((id) => { const b = $(id); if (b) b.addEventListener('click', () => seedDemo(b)); });
    const dismiss = $('demoBannerDismiss');
    if (dismiss) dismiss.addEventListener('click', dismissDemoBanner);
    loadDashboard();
    loadDemoBanner();
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
    // Relative times ('5 min ago') age while the page stays open.
    setInterval(() => {
      if (!lastData) return;
      renderStatus(lastData);
      renderHero(lastData.last_run, lastData.kpi);
    }, CLOCK_MS);
    // Elapsed times of the running tests age between the polls too.
    setInterval(() => { if (lastActive) renderActive(lastActive); }, CLOCK_MS);
  });
})();
