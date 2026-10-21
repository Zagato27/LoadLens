// SPDX-License-Identifier: Apache-2.0
(function () {
  const LL = window.LoadLens;
  const escapeHtml = LL.ui.escapeHtml;
  const DEFAULT_HEADROOM_PCT = 20;
  const SLA_LINE_SPAN = 2.5;
  const AXIS_SPAN_CAP = 2;
  const COLORS = {
    sample: 'rgba(96, 140, 196, 0.75)',
    edge: 'rgba(150, 150, 150, 0.8)',
    stall: '#ef4444',
    step: '#f97316',
    p95: '#ec4899',
    knee: '#14b8a6',
    limit: '#b91c1c',
    sla: '#ef4444',
    zoneUnstable: 'rgba(239, 68, 68, 0.10)',
    // Beyond the tested load: a neutral shade that reads on both themes (the accent marks SLA zones only).
    zoneUntested: 'rgba(128, 128, 128, 0.12)'
  };
  // The model curve is a neutral reference line over the colored measurements (the muted text color of the theme).
  function modelColor() {
    return LL.ui.chartColors().inkMuted || '#a3a19a';
  }
  // Labels of the server codes are getters: the tables are built at load, the text is read in the page locale.
  const CONFIDENCE = {
    high: { get text() { return LL.t('forecast.confidence.high'); }, pill: 'ok' },
    medium: { get text() { return LL.t('forecast.confidence.medium'); }, pill: 'warn' },
    low: { get text() { return LL.t('forecast.confidence.low'); }, pill: 'fail' }
  };
  // Line colors of the target follow the status tokens of the theme (LL.ui.chartColors), like the scale above.
  const statusColor = (name, fallback) => (LL.ui.chartColors ? LL.ui.chartColors()[name] : '') || fallback;
  const ANSWER = {
    holds: { get text() { return LL.t('forecast.answer.holds'); }, tone: 'success', get color() { return statusColor('statusOk', '#52c483'); } },
    scale: { get text() { return LL.t('forecast.answer.scale'); }, tone: 'risk', get color() { return statusColor('statusWarn', '#e4a93b'); } },
    tight: { get text() { return LL.t('forecast.answer.tight'); }, tone: 'risk', get color() { return statusColor('statusWarn', '#e4a93b'); } },
    blocked: { get text() { return LL.t('forecast.answer.blocked'); }, tone: 'fail', get color() { return statusColor('statusFail', '#e4574f'); } }
  };
  const CEILING_SOURCES = {
    get sla() { return LL.t('forecast.ceiling_source.sla'); },
    get default() { return LL.t('forecast.ceiling_source.default'); },
    get request() { return LL.t('forecast.ceiling_source.request'); }
  };
  const SCALING = {
    helps: { get text() { return LL.t('forecast.scaling.helps'); }, tone: 'success' },
    partly: { get text() { return LL.t('forecast.scaling.partly'); }, tone: 'risk' },
    no: { get text() { return LL.t('forecast.scaling.no'); }, tone: 'fail' }
  };
  const EXPLAIN_ITEMS = 3;
  const SCALE_TARGET_TONES = { holds: 'ok', scale: 'warn', tight: 'warn', blocked: 'fail' };
  const SCALE_LABEL_GAP_PCT = 22;
  const SCALE_EDGE_PCT = 10;
  const SCALE_ZONE_TEXT_PCT = 10;
  const SCALE_ROW_PX = 18;
  const PUBLISH_POLLS = 180;
  const PUBLISH_POLL_MS = 1500;
  const RESOURCE_ROWS = 5;
  // Words for the parts of the test the model is built on: detected load steps or plain time intervals.
  // title: legend heading, one/many: name before numbers, at: "at step N", by(n): "over n steps".
  const STEP_WORDS = {
    step_detector: {
      get title() { return LL.t('forecast.unit.step.title'); },
      get many() { return LL.t('forecast.unit.step.names'); },
      get one() { return LL.t('forecast.unit.step.name'); },
      get at() { return LL.t('forecast.unit.step.at'); },
      by: (count) => LL.tn('forecast.unit.step.by', count)
    },
    time_buckets: {
      get title() { return LL.t('forecast.unit.interval.title'); },
      get many() { return LL.t('forecast.unit.interval.names'); },
      get one() { return LL.t('forecast.unit.interval.name'); },
      get at() { return LL.t('forecast.unit.interval.at'); },
      by: (count) => LL.tn('forecast.unit.interval.by', count)
    }
  };
  const ROLE_TITLES = {
    get fit() { return LL.t('forecast.role.fit'); },
    get stall() { return LL.t('forecast.role.stall'); },
    get edge() { return LL.t('forecast.role.edge'); },
    get after_cutoff() { return LL.t('forecast.role.after_cutoff'); }
  };
  const REASONS = {
    low_r2: {
      reason: (data) => LL.t('forecast.reason.low_r2', { share: formatPct(Math.max(0, data.fit.r2) * 100), r2: formatNumber(data.fit.r2, 2) }),
      action: (data) => (data.instability ? '' : LL.t('forecast.reason.low_r2_action'))
    },
    instability: {
      reason: (data) => LL.t('forecast.reason.instability', { rps: formatApproxRps(data.instability.onset_rps) }),
      action: (data) => instabilityAction(data.instability)
    },
    extrapolation: {
      reason: () => LL.t('forecast.reason.extrapolation'),
      action: (data) => LL.t('forecast.reason.verify_rps', { rps: formatRps(data.target.rps) })
    },
    cpu_fit: {
      reason: (data) => LL.t('forecast.reason.cpu_fit', { services: decidingServices(data.resources).filter((item) => !item.reliable).map((item) => item.name).join(', ') }),
      action: () => LL.t('forecast.reason.cpu_fit_action')
    },
    cpu_extrapolation: {
      reason: (data) => LL.t('forecast.reason.cpu_extrapolation', { ratio: formatNumber(data.resources.target_rps / data.tested_rps, 1) }),
      action: (data) => LL.t('forecast.reason.verify_rps', { rps: formatRps(data.resources.target_rps) })
    },
    scaling_untested: {
      reason: () => LL.t('forecast.reason.scaling_untested'),
      action: (data) => LL.t('forecast.reason.scaling_untested_action', {
        services: scaledServices(data.resources).map((item) => `${item.name} — ${podsText(item.instances_needed)}`).join(', '),
        rps: formatRps(data.resources.target_rps)
      })
    }
  };
  let runId = '';
  let charts = [];
  let refreshTimer = 0;
  let current = null;
  let configuredCeiling = null;

  const zonesPlugin = {
    id: 'forecastZones',
    beforeDatasetsDraw(chart, args, options) {
      const area = chart.chartArea;
      const scale = chart.scales.x;
      (options.zones || []).forEach((zone) => {
        const left = Math.max(area.left, scale.getPixelForValue(zone.from));
        const right = Math.min(area.right, scale.getPixelForValue(zone.to));
        if (!(right > left)) return;
        chart.ctx.save();
        chart.ctx.fillStyle = zone.color;
        chart.ctx.fillRect(left, area.top, right - left, area.bottom - area.top);
        chart.ctx.restore();
      });
    }
  };
  const stepNumbersPlugin = {
    id: 'forecastStepNumbers',
    afterDatasetsDraw(chart) {
      const ctx = chart.ctx;
      chart.data.datasets.forEach((dataset, index) => {
        if (!dataset.stepNumbers || !chart.isDatasetVisible(index)) return;
        ctx.save();
        ctx.font = "600 11px 'JetBrains Mono', Consolas, 'Courier New', monospace";
        ctx.fillStyle = '#1a1f24';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        chart.getDatasetMeta(index).data.forEach((element, pointIndex) => {
          ctx.fillText(String(dataset.data[pointIndex].number), element.x, element.y);
        });
        ctx.restore();
      });
    }
  };

  function formatNumber(value, digits) {
    const n = Number(value);
    if (value === null || value === undefined || value === '' || !Number.isFinite(n)) return '—';
    return LL.fmt.number(n, { maximumFractionDigits: digits || 0 });
  }
  // Decimal separator of the UI locale, for numbers Intl does not format (κ in exponent notation).
  function decimalSeparator() {
    return LL.fmt.number(1.5, { minimumFractionDigits: 1 }).charAt(1);
  }
  function threeSignificant(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n === 0) return n;
    const step = Math.pow(10, Math.max(0, Math.floor(Math.log10(Math.abs(n))) - 2));
    return Math.round(n / step) * step;
  }
  function formatRps(value) {
    return `${formatNumber(value)} RPS`;
  }
  function formatApproxRps(value) {
    return `≈ ${formatRps(threeSignificant(value))}`;
  }
  function formatMs(value) {
    const n = Number(value);
    if (value === null || value === undefined || !Number.isFinite(n)) return '—';
    if (n >= 1000) return LL.t('forecast.format.seconds', { value: formatNumber(n / 1000, 1) });
    return LL.t('forecast.format.ms', { value: formatNumber(n, n < 10 ? 1 : 0) });
  }
  function formatMsRange(values) {
    const low = Math.min.apply(null, values);
    const high = Math.max.apply(null, values);
    if (formatMs(low) === formatMs(high)) return `≈ ${formatMs(low)}`;
    if (high < 1000) return `${formatNumber(low, high < 10 ? 1 : 0)}–${formatMs(high)}`;
    return `${formatMs(low)} – ${formatMs(high)}`;
  }
  function formatPct(value) {
    const n = Number(value);
    return LL.t('forecast.format.pct', { value: formatNumber(n, Math.abs(n) < 10 ? 1 : 0) });
  }
  function cpuPct(share) {
    return formatPct(Number(share) * 100);
  }
  function podsText(count) {
    return LL.tn('forecast.pods', count);
  }
  function minutesText(value) {
    return LL.t('forecast.format.minutes', { value: formatNumber(value, 1) });
  }
  function hhmm(value) {
    const match = String(value || '').match(/T(\d{2}:\d{2})/);
    return match ? match[1] : '—';
  }
  function numberRange(numbers) {
    const parts = [];
    let first = numbers[0];
    let last = numbers[0];
    numbers.slice(1).concat([null]).forEach((number) => {
      if (number === last + 1) {
        last = number;
        return;
      }
      parts.push(first === last ? String(first) : `${first}–${last}`);
      first = number;
      last = number;
    });
    return parts.join(', ');
  }
  function instabilityAction(info) {
    const rps = formatRps(threeSignificant(info.onset_rps));
    if (info.period_min == null) return LL.t('forecast.reason.instability_action', { rps: rps });
    return LL.t('forecast.reason.instability_action_period', { rps: rps, minutes: formatNumber(info.period_min) });
  }
  function stepWords(data) {
    const steps = data.steps || [];
    const buckets = steps.length > 0 && steps.every((step) => step.source === 'time_buckets');
    return STEP_WORDS[buckets ? 'time_buckets' : 'step_detector'];
  }
  function listHtml(items) {
    return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`;
  }
  function setHtml(id, html) {
    const node = document.getElementById(id);
    if (node) node.innerHTML = html;
  }
  function setText(id, text) {
    const node = document.getElementById(id);
    if (node) node.textContent = text;
  }
  function runIdFromPath() {
    const parts = location.pathname.split('/').filter(Boolean);
    const idx = parts.indexOf('forecasting');
    return idx >= 0 && parts.length === idx + 2 ? decodeURIComponent(parts[idx + 1]) : '';
  }

  async function fetchForecast(target, headroom, ceiling) {
    const url = new URL('/forecast/' + encodeURIComponent(runId), location.origin);
    if (target !== null && target !== undefined && target !== '') url.searchParams.set('target_rps', String(target));
    url.searchParams.set('headroom_pct', String(headroom));
    if (ceiling !== null && ceiling !== undefined) url.searchParams.set('cpu_ceiling_pct', String(ceiling));
    const resp = await fetch(url);
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) return { error: body.error || LL.t('forecast.error.build', { status: resp.status }), code: body.code || '' };
    return { data: body };
  }

  function limitText(data) {
    const limit = data.limit;
    if (limit.kind === 'instability') return LL.t('forecast.limit.instability', { rps: formatApproxRps(limit.rps) });
    if (limit.kind === 'model') return LL.t('forecast.limit.model', { rps: formatApproxRps(limit.rps) });
    return LL.t('forecast.limit.none');
  }
  function scaledServices(plan) {
    return (plan.services || []).filter((item) => item.instances_needed == null || item.instances_needed > item.instances);
  }
  function decidingServices(plan) {
    return (plan.services || []).filter((item) => item.name === plan.bottleneck || scaledServices(plan).includes(item));
  }
  function serviceCeiling(data, name) {
    const item = (data.resources.services || []).find((service) => service.name === name);
    return item ? item.cpu_ceiling : data.resources.ceiling_pct / 100;
  }
  function stuckText(item) {
    return LL.t('forecast.plan.stuck', { name: item.name, base: cpuPct(item.cpu_base), ceiling: cpuPct(item.cpu_ceiling) });
  }
  function scaleLines(data) {
    const plan = data.resources;
    const lines = scaledServices(plan).map((item) => (item.instances_needed == null
      ? stuckText(item)
      : LL.t('forecast.plan.add', {
        name: item.name,
        from: item.instances,
        to: podsText(item.instances_needed),
        after: cpuPct(item.cpu_after),
        current: cpuPct(item.cpu_at_target)
      })));
    if (plan.scaled_capacity_rps != null) {
      lines.push(LL.t('forecast.plan.after_scaling', { rps: formatApproxRps(plan.scaled_capacity_rps), service: plan.next_bottleneck }));
    }
    if (data.limit_applies && data.limit.rps != null) {
      lines.push(LL.t('forecast.plan.non_cpu_limit', { rps: formatApproxRps(data.limit.rps) }));
    }
    return lines;
  }
  function blockedText(data) {
    const plan = data.resources;
    const rps = formatApproxRps(data.limit.rps);
    if (!plan.limit_service) return LL.t('forecast.plan.blocked', { rps: rps });
    return LL.t('forecast.plan.blocked_busiest', { rps: rps, service: plan.limit_service, cpu: cpuPct(plan.limit_cpu) });
  }
  function blockedLines(data) {
    const plan = data.resources;
    const scaled = scaledServices(plan);
    const stuck = scaled.filter((item) => item.instances_needed == null).map(stuckText);
    if (plan.status !== 'ok' || !data.limit_applies) return stuck;
    const short = scaled
      .filter((item) => item.instances_needed != null)
      .map((item) => LL.t('forecast.plan.cpu_short', { name: item.name, cpu: cpuPct(item.cpu_at_target), pods: podsText(item.instances_needed) }));
    return [blockedText(data), ...stuck, ...short];
  }
  function holdsLines(data) {
    const plan = data.resources;
    const lines = [];
    if (plan.status === 'ok' && plan.services.length) {
      const top = plan.services[0];
      lines.push(LL.t('forecast.plan.busiest', { name: top.name, cpu: cpuPct(top.cpu_at_target), ceiling: cpuPct(top.cpu_ceiling) }));
    }
    if (data.limit_applies && data.target.headroom_pct != null) lines.push(LL.t('forecast.plan.headroom', { pct: formatPct(data.target.headroom_pct) }));
    return lines;
  }
  function tightText(data, headroom) {
    const params = { pct: formatPct(data.target.headroom_pct), required: formatPct(headroom) };
    return data.resources.status === 'ok' ? LL.t('forecast.plan.tight_not_cpu', params) : LL.t('forecast.plan.tight', params);
  }
  function limitLine(data) {
    const plan = data.resources;
    if (!data.limit_applies && plan.limit_cause) {
      const params = { rps: formatApproxRps(data.limit.rps), service: plan.limit_cause, cpu: cpuPct(plan.limit_cpu) };
      return data.limit.kind === 'instability' ? LL.t('forecast.plan.cause_instability', params) : LL.t('forecast.plan.cause_model', params);
    }
    const verdict = data.answer ? data.answer.verdict : '';
    if (data.limit_applies && plan.status === 'ok' && (verdict === 'blocked' || verdict === 'scale')) return '';
    return limitText(data);
  }
  function safeLine(data, headroom) {
    const safe = data.safe;
    if (!safe || safe.rps == null) return '';
    const value = formatRps(Math.floor(safe.rps));
    if (safe.source === 'cpu') return LL.t('forecast.safe.cpu', { rps: value, service: safe.service, cpu: cpuPct(serviceCeiling(data, safe.service)) });
    return LL.t('forecast.safe.headroom', { pct: formatPct(headroom), rps: value });
  }
  function latencyLine(data) {
    const target = data.target;
    if (!target || target.response_ms == null || ['scale', 'blocked'].includes(data.answer.verdict)) return '';
    const mean = formatMs(target.response_ms);
    if (target.p95_ms == null) return LL.t('forecast.latency.mean', { mean: mean });
    return LL.t('forecast.latency.mean_p95', { mean: mean, p95: formatMs(target.p95_ms) });
  }
  function answerLines(data, headroom) {
    const verdict = data.answer.verdict;
    const lines = [];
    if (verdict === 'scale') lines.push(...scaleLines(data));
    if (verdict === 'blocked') lines.push(...blockedLines(data));
    if (verdict === 'tight') lines.push(tightText(data, headroom));
    if (verdict === 'holds') lines.push(...holdsLines(data));
    lines.push(limitLine(data), safeLine(data, headroom), latencyLine(data));
    return lines;
  }
  function paragraphs(lines) {
    return lines.filter(Boolean).map((line) => `<p>${escapeHtml(line)}</p>`).join('');
  }
  function answerHtml(data, headroom) {
    const answer = data.answer;
    if (!answer) return `<div class="forecast-verdict">${paragraphs([limitLine(data), safeLine(data, headroom)])}</div>`;
    const view = ANSWER[answer.verdict];
    const confidence = CONFIDENCE[answer.confidence];
    return [
      `<div class="forecast-verdict is-${view.tone}">`,
      '<div class="forecast-verdict-head">',
      `<span class="forecast-verdict-title">${LL.tHtml('forecast.answer.title', { rps: formatRps(data.target.rps), answer: view.text })}</span>`,
      `<span class="pill ${confidence.pill}">${LL.tHtml('forecast.confidence.label', { level: confidence.text })}</span>`,
      '</div>',
      paragraphs(answerLines(data, headroom)),
      '</div>'
    ].join('');
  }

  function resourceRows(plan) {
    const scaled = scaledServices(plan);
    const others = plan.services.filter((item) => !scaled.includes(item));
    return scaled.concat(others.slice(0, Math.max(0, RESOURCE_ROWS - scaled.length)));
  }
  function resourceNotes(item, plan) {
    const notes = [];
    const changed = scaledServices(plan).includes(item);
    if (item.name === plan.bottleneck) notes.push(LL.t('forecast.resources.note_bottleneck'));
    if (item.name === plan.limit_cause && item.cpu_ceiling < plan.ceiling_pct / 100) notes.push(LL.t('forecast.resources.note_limit', { ceiling: cpuPct(item.cpu_ceiling) }));
    if (!item.reliable && (changed || item.name === plan.bottleneck)) notes.push(LL.t('forecast.resources.note_poor_fit'));
    return notes;
  }
  function resourceRow(item, plan) {
    const changed = scaledServices(plan).includes(item);
    const notes = resourceNotes(item, plan);
    const needed = item.instances_needed == null ? LL.t('forecast.resources.not_by_pods') : String(item.instances_needed);
    const after = changed && item.cpu_after != null ? ` <span class="dim">(≈ ${escapeHtml(cpuPct(item.cpu_after))})</span>` : '';
    return [
      `<tr class="${changed ? 'is-scale' : ''}">`,
      `<td>${escapeHtml(item.name)}${notes.length ? `<div class="dim">${escapeHtml(notes.join('; '))}</div>` : ''}</td>`,
      `<td class="num">${item.instances}</td>`,
      `<td class="num">${escapeHtml(cpuPct(item.cpu_measured))}</td>`,
      `<td class="num${item.cpu_at_target >= item.cpu_ceiling ? ' is-over' : ''}">${escapeHtml(cpuPct(item.cpu_at_target))}</td>`,
      `<td class="num">${changed ? `<strong>${escapeHtml(needed)}</strong>` : escapeHtml(needed)}${after}</td>`,
      '</tr>'
    ].join('');
  }
  function nodesText(plan) {
    if (plan.nodes_message) return plan.nodes_message;
    const nodes = plan.nodes || [];
    if (!nodes.length) return LL.t('forecast.nodes.no_data');
    const over = nodes.filter((node) => node.cpu_at_target >= plan.ceiling_pct / 100);
    if (over.length) {
      const list = over.slice(0, 4).map((node) => `${node.name} ≈ ${cpuPct(node.cpu_at_target)}`).join(', ');
      return LL.t('forecast.nodes.over', { nodes: list });
    }
    return LL.t('forecast.nodes.busiest', { name: nodes[0].name, cpu: cpuPct(nodes[0].cpu_at_target) });
  }
  function resourcesHtml(data) {
    const plan = data.resources;
    const title = `<h4>${LL.tHtml('forecast.resources.title', { rps: formatRps(plan.target_rps) })}</h4>`;
    if (plan.status !== 'ok') return `${title}<p>${escapeHtml(plan.message)}</p>`;
    const words = stepWords(data);
    const rows = resourceRows(plan);
    const rest = plan.services.length - rows.length;
    const note = LL.t('forecast.resources.note', {
      ceiling: formatPct(plan.ceiling_pct),
      source: CEILING_SOURCES[plan.ceiling_source],
      basis: words.by(plan.steps_used)
    });
    const restText = rest > 0
      ? `<p class="dim">${escapeHtml(LL.tn('forecast.resources.rest', rest, { cpu: cpuPct(rows[rows.length - 1].cpu_at_target) }))}</p>`
      : '';
    const head = [
      `<th>${LL.tHtml('forecast.resources.col_service')}</th>`,
      `<th class="num">${LL.tHtml('forecast.resources.col_pods')}</th>`,
      `<th class="num">${LL.tHtml('forecast.resources.col_cpu_step', { unit: words.one, number: plan.last_step })}</th>`,
      `<th class="num">${LL.tHtml('forecast.resources.col_cpu_target')}</th>`,
      `<th class="num">${LL.tHtml('forecast.resources.col_needed')}</th>`
    ].join('');
    return [
      title,
      `<p class="dim">${escapeHtml(note)}</p>`,
      '<div class="step-table-wrap"><table class="step-table forecast-table">',
      `<thead><tr>${head}</tr></thead>`,
      `<tbody>${rows.map((item) => resourceRow(item, plan)).join('')}</tbody>`,
      '</table></div>',
      restText,
      `<p>${escapeHtml(nodesText(plan))}</p>`
    ].join('');
  }

  function stableSteps(data) {
    const onset = data.instability ? data.instability.onset_step : Infinity;
    return (data.steps || []).filter((step) => step.stalls === 0 && !step.after_drop && step.number < onset);
  }
  function stableText(data, stable) {
    const words = stepWords(data);
    const rps = stable.map((step) => step.rps);
    const load = stable.length > 1
      ? `${formatNumber(Math.min.apply(null, rps))} → ${formatRps(Math.max.apply(null, rps))}`
      : formatRps(rps[0]);
    const name = stable.length > 1 ? words.many : words.one;
    return LL.t('forecast.facts.stable', {
      unit: name,
      numbers: numberRange(stable.map((step) => step.number)),
      load: load,
      mean: formatMsRange(stable.map((step) => step.response_ms))
    });
  }
  function instabilityText(data) {
    const info = data.instability;
    const params = {
      time: hhmm(info.onset_iso),
      label: info.onset_label,
      rps: formatApproxRps(info.onset_rps),
      points: LL.tn('forecast.facts.stall_points', info.samples, { stalls: info.stalls, minutes: formatNumber(data.bin_minutes, 1) }),
      mean: formatMs(info.worst_response_ms),
      p95: formatMs(info.worst_p95_ms)
    };
    const text = info.worst_p95_ms != null ? LL.t('forecast.facts.instability_p95', params) : LL.t('forecast.facts.instability', params);
    if (info.period_min == null) return text;
    return `${text} ${LL.t('forecast.facts.instability_period', { minutes: formatNumber(info.period_min) })}`;
  }
  function bestText(data) {
    const steps = (data.steps || []).filter((step) => !step.after_drop);
    const bin = minutesText(data.bin_minutes);
    if (!steps.length) return LL.t('forecast.facts.peak', { bin: bin, peak: formatNumber(data.peak_rps) });
    const best = steps.reduce((top, step) => (step.rps > top.rps ? step : top));
    return LL.t('forecast.facts.best', {
      unit: stepWords(data).one,
      best: formatNumber(best.rps),
      label: best.label,
      bin: bin,
      peak: formatNumber(data.peak_rps)
    });
  }
  function cpuFact(data) {
    const plan = data.resources;
    if (plan.status !== 'ok' || !plan.services.length) return '';
    const top = plan.services.reduce((best, item) => (item.cpu_measured > best.cpu_measured ? item : best));
    return LL.t('forecast.facts.cpu', {
      name: top.name,
      cpu: cpuPct(top.cpu_measured),
      unit: stepWords(data).at,
      number: plan.last_step,
      pods: podsText(top.instances)
    });
  }
  function factItems(data) {
    const items = [];
    const stable = stableSteps(data);
    if (stable.length) items.push(stableText(data, stable));
    const cpu = cpuFact(data);
    if (cpu) items.push(cpu);
    if (data.instability) items.push(instabilityText(data));
    else if (data.excluded_stalls) items.push(LL.t('forecast.facts.spikes', { count: data.excluded_stalls }));
    items.push(bestText(data));
    if (data.cutoff_kind === 'rps_drop') items.push(LL.t('forecast.facts.rps_drop', { time: hhmm(data.cutoff_iso) }));
    return items;
  }
  function uslItems(data) {
    const fit = data.fit || {};
    const ceiling = fit.x_max != null
      ? LL.t('forecast.method.usl_ceiling', { rps: formatApproxRps(fit.x_max) })
      : LL.t('forecast.method.usl_no_ceiling');
    const knee = data.knee
      ? LL.t('forecast.method.knee', { rps: formatApproxRps(data.knee.rps) })
      : LL.t('forecast.method.no_knee');
    return [ceiling, knee];
  }
  function confidentBasis(data) {
    const r2 = `R² = ${formatNumber(data.fit.r2, 2)}`;
    if (data.resources.status === 'ok' && !data.limit_applies) return LL.t('forecast.confidence.basis_cpu');
    if (data.answer.verdict === 'blocked') return LL.t('forecast.confidence.basis_limit', { r2: r2, rps: formatApproxRps(data.limit.rps) });
    return LL.t('forecast.confidence.basis_range', { r2: r2 });
  }
  function confidenceHtml(data) {
    const answer = data.answer;
    if (!answer) return '';
    const title = `<h4>${LL.tHtml('forecast.confidence.label', { level: CONFIDENCE[answer.confidence].text })}</h4>`;
    const codes = answer.confidence_reasons || [];
    if (!codes.length) return `${title}<p>${escapeHtml(confidentBasis(data))}</p>`;
    const actions = codes.map((code) => REASONS[code].action(data)).filter(Boolean);
    const actionsHtml = actions.length ? `<h5>${LL.tHtml('forecast.confidence.actions')}</h5>${listHtml(actions)}` : '';
    return `${title}${listHtml(codes.map((code) => REASONS[code].reason(data)))}${actionsHtml}`;
  }
  function exclusionText(data) {
    const parts = [];
    if (data.excluded_edges) parts.push(LL.t('forecast.method.excluded_edges', { count: data.excluded_edges }));
    if (data.excluded_stalls) parts.push(LL.t('forecast.method.excluded_stalls', { count: data.excluded_stalls }));
    const after = (data.points || []).filter((point) => point.role === 'after_cutoff').length;
    if (data.cutoff_kind === 'rps_drop' && after) parts.push(LL.t('forecast.method.excluded_after_drop', { count: after }));
    return parts.length ? LL.t('forecast.method.excluded', { parts: parts.join(', ') }) : LL.t('forecast.method.all_included');
  }
  function kappaText(fit) {
    const kappa = Number(fit.kappa) || 0;
    if (kappa <= 0) return LL.t('forecast.method.kappa_zero');
    const text = LL.t('forecast.method.kappa', { kappa: kappa.toExponential(2).replace('.', decimalSeparator()) });
    if (fit.n_peak == null) return text;
    return `${text} ${LL.t('forecast.method.kappa_peak', { concurrency: formatNumber(threeSignificant(fit.n_peak)) })}`;
  }
  function methodItems(data) {
    const fit = data.fit || {};
    const end = data.cutoff_kind === 'rps_drop' ? LL.t('forecast.method.end_rps_drop') : LL.t('forecast.method.end_test');
    return [
      LL.t('forecast.method.pods'),
      ...uslItems(data),
      LL.t('forecast.method.window', {
        points: LL.tn('forecast.method.points', data.samples_used),
        minutes: formatNumber(data.bin_minutes, 1),
        start: hhmm(data.window_start_iso),
        end: hhmm(data.cutoff_iso),
        reason: end
      }),
      exclusionText(data),
      data.load_model === 'closed'
        ? LL.t('forecast.method.closed', { seconds: formatNumber(data.think_time_s, 2) })
        : LL.t('forecast.method.open'),
      LL.t('forecast.method.sigma', { sigma: formatPct((Number(fit.sigma) || 0) * 100) }),
      kappaText(fit),
      LL.t('forecast.method.r2', { r2: formatNumber(fit.r2, 2) })
    ].concat(data.notes || []);
  }

  function isFiniteNumber(value) {
    return value !== null && value !== undefined && Number.isFinite(Number(value));
  }
  function maxOf(values) {
    const finite = values.filter(isFiniteNumber).map(Number);
    return finite.length ? Math.max.apply(null, finite) : 0;
  }
  function minOf(values) {
    const finite = values.filter(isFiniteNumber).map(Number);
    return finite.length ? Math.min.apply(null, finite) : 0;
  }
  function niceMin(value) {
    if (!(value > 0)) return 0;
    const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
    return Math.floor(value / magnitude) * magnitude;
  }
  function niceMax(value) {
    if (!(value > 0)) return 1;
    const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
    const factor = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find((step) => step * magnitude >= value);
    return factor * magnitude;
  }
  function cappedMax(dataMax, references) {
    return niceMax(1.08 * Math.min(Math.max(dataMax, maxOf(references)), AXIS_SPAN_CAP * dataMax));
  }
  function byRole(points, role) {
    return points.filter((point) => point.role === role);
  }
  function clipped(point, xMax, yMax) {
    if (point.x <= xMax && point.y <= yMax) return point;
    return Object.assign({}, point, { x: Math.min(point.x, xMax), y: Math.min(point.y, yMax), clipped: true });
  }
  function clippedCount(datasets) {
    return datasets.reduce((sum, dataset) => sum + dataset.data.filter((point) => point.clipped).length, 0);
  }
  function limitTitle(limit) {
    return limit.kind === 'instability' ? LL.t('forecast.chart.limit_instability') : LL.t('forecast.chart.limit_model');
  }

  function sampleTip(point) {
    const lines = [`${hhmm(point.time_iso)} · ${ROLE_TITLES[point.role]}`, formatRps(point.rps), LL.t('forecast.point.mean', { value: formatMs(point.response_ms) })];
    if (point.p95_ms != null) lines.push(`p95 ${formatMs(point.p95_ms)}`);
    lines.push(LL.t('forecast.point.concurrency', { value: formatNumber(point.concurrency, 1) }));
    if (point.vus != null) lines.push(`VU ${formatNumber(point.vus)}`);
    return lines;
  }
  function samplePoint(x, y, point) {
    return { x: x, y: y, tip: sampleTip(point) };
  }
  function stepPoint(x, y, step) {
    const lines = [step.label, LL.t('forecast.point.rps_mean', { rps: formatRps(step.rps) }), LL.t('forecast.point.mean', { value: formatMs(step.response_ms) })];
    if (step.p95_ms != null) lines.push(LL.t('forecast.point.worst_p95', { value: formatMs(step.p95_ms) }));
    lines.push(step.stalls ? LL.t('forecast.point.stalls', { stalls: step.stalls, samples: step.samples }) : LL.t('forecast.point.no_stalls'));
    return { x: x, y: y, number: step.number, tip: lines };
  }
  function modelPoint(x, y, point) {
    const lines = [LL.t('forecast.point.model', { rps: formatRps(point.rps) }), LL.t('forecast.point.mean', { value: formatMs(point.response_ms) })];
    if (point.p95_ms != null) lines.push(`p95 ${formatMs(point.p95_ms)}`);
    lines.push(LL.t('forecast.point.concurrency', { value: formatNumber(point.concurrency, 1) }));
    return { x: x, y: y, tip: lines };
  }
  function verticalLine(x, height, tip) {
    return [{ x: x, y: 0, tip: tip }, { x: x, y: height, tip: tip }];
  }
  function horizontalLine(y, width, tip) {
    return [{ x: 0, y: y, tip: tip }, { x: width, y: y, tip: tip }];
  }

  function clippedStyle(style) {
    return (context) => (context.raw && context.raw.clipped ? 'triangle' : style);
  }
  function pointSet(label, data, color, radius, order) {
    return {
      label: label,
      data: data,
      order: order,
      showLine: false,
      pointRadius: radius,
      pointHoverRadius: radius + 2,
      backgroundColor: color,
      borderColor: color,
      pointStyle: clippedStyle('circle'),
      clip: false
    };
  }
  function lineSet(label, data, color, dashed, order) {
    return {
      label: label,
      data: data,
      order: order,
      showLine: true,
      pointRadius: 0,
      pointHitRadius: 6,
      borderWidth: 2,
      borderColor: color,
      backgroundColor: color,
      borderDash: dashed ? [6, 4] : [],
      pointStyle: 'line'
    };
  }
  function stepSet(label, data) {
    return Object.assign(pointSet(label, data, COLORS.step, 9, 1), { stepNumbers: true, borderColor: 'rgba(0, 0, 0, 0.35)', borderWidth: 1 });
  }
  function stepP95Set(label, data) {
    return Object.assign(pointSet(label, data, 'transparent', 6, 2), { borderColor: COLORS.step, borderWidth: 2, pointStyle: clippedStyle('rectRot') });
  }
  function edgeSet(data) {
    return Object.assign(pointSet(LL.t('forecast.chart.edges'), data, 'transparent', 3.5, 5), { borderColor: COLORS.edge, borderWidth: 1.5 });
  }
  function kneeSet(knee) {
    const tip = [LL.t('forecast.point.knee', { rps: formatRps(knee.rps) }), LL.t('forecast.point.mean', { value: formatMs(knee.response_ms) })];
    return Object.assign(pointSet(LL.t('forecast.chart.knee'), [{ x: knee.rps, y: knee.response_ms, tip: tip }], COLORS.knee, 7, 0), { pointStyle: 'rectRot' });
  }

  function latencyBounds(data, points, steps, curve) {
    const measured = byRole(points, 'fit');
    const stable = steps.filter((step) => step.stalls === 0);
    const base = maxOf([].concat(
      measured.map((point) => point.response_ms),
      curve.map((point) => point.response_ms),
      curve.map((point) => point.p95_ms),
      stable.map((step) => step.response_ms),
      stable.map((step) => step.p95_ms)
    ));
    const sla = Number(data.sla_p95_ms);
    const hasP95 = curve.some((point) => point.p95_ms != null) || steps.some((step) => step.p95_ms != null);
    const slaKnown = Number.isFinite(sla) && sla > 0;
    const slaShown = slaKnown && hasP95 && sla <= SLA_LINE_SPAN * base;
    const loads = [].concat(measured.concat(byRole(points, 'stall')).map((point) => point.rps), steps.map((step) => step.rps), curve.map((point) => point.rps));
    const dataX = maxOf(loads);
    return {
      xMin: niceMin(0.9 * minOf(loads.concat([data.target ? data.target.rps : null]))),
      xMax: cappedMax(dataX, [data.target ? data.target.rps : null, data.limit.rps]),
      yMax: niceMax(1.25 * Math.max(base, slaShown ? sla : 0)),
      sla: slaShown ? sla : null,
      slaNote: slaNote(slaKnown && !slaShown ? sla : null, hasP95)
    };
  }
  function slaNote(sla, hasP95) {
    if (sla == null) return '';
    if (!hasP95) return LL.t('forecast.chart.sla_no_p95', { value: formatMs(sla) });
    return LL.t('forecast.chart.sla_above', { value: formatMs(sla) });
  }
  function latencyReferences(data, bounds) {
    const sets = [];
    if (bounds.sla != null) sets.push(lineSet('SLA p95', horizontalLine(bounds.sla, bounds.xMax, [`SLA p95: ${formatMs(bounds.sla)}`]), COLORS.sla, true, 3));
    if (data.target) {
      const tip = [LL.t('forecast.point.target', { rps: formatRps(data.target.rps) })];
      sets.push(lineSet(LL.t('forecast.chart.target'), verticalLine(data.target.rps, bounds.yMax, tip), ANSWER[data.answer.verdict].color, true, 3));
    }
    if (data.limit.rps != null) {
      const tip = [`${limitTitle(data.limit)}: ${formatApproxRps(data.limit.rps)}`];
      sets.push(lineSet(limitTitle(data.limit), verticalLine(data.limit.rps, bounds.yMax, tip), COLORS.limit, false, 3));
    }
    if (data.knee) sets.push(kneeSet(data.knee));
    return sets;
  }
  function latencyZones(data, xMax) {
    if (data.instability) return [{ from: data.instability.onset_rps, to: xMax, color: COLORS.zoneUnstable }];
    return data.tested_rps ? [{ from: data.tested_rps, to: xMax, color: COLORS.zoneUntested }] : [];
  }
  function latencyCaption(data, datasets, bounds) {
    const parts = [data.instability
      ? LL.t('forecast.chart.zone_unstable')
      : LL.t('forecast.chart.zone_untested')];
    const hidden = clippedCount(datasets);
    if (hidden) parts.push(LL.t('forecast.chart.clipped_top', { count: hidden }));
    if (bounds.slaNote) parts.push(bounds.slaNote);
    if (data.target && data.target.rps > bounds.xMax) parts.push(LL.t('forecast.chart.target_right', { rps: formatRps(data.target.rps) }));
    return parts.join(' ');
  }
  function latencyView(data) {
    const points = data.points || [];
    const steps = (data.steps || []).filter((step) => !step.after_drop);
    const peak = data.fit && data.fit.n_peak != null ? data.fit.n_peak : Infinity;
    const curve = (data.curve || []).filter((point) => point.concurrency <= peak);
    const bounds = latencyBounds(data, points, steps, curve);
    const clip = (point) => clipped(point, bounds.xMax, bounds.yMax);
    const words = stepWords(data);
    const datasets = [
      pointSet(LL.t('forecast.chart.points', { minutes: formatNumber(data.bin_minutes, 1) }), byRole(points, 'fit').map((p) => clip(samplePoint(p.rps, p.response_ms, p))), COLORS.sample, 2.5, 6),
      pointSet(LL.t('forecast.chart.stalls'), byRole(points, 'stall').map((p) => clip(samplePoint(p.rps, p.response_ms, p))), COLORS.stall, 3.5, 5),
      stepSet(LL.t('forecast.chart.unit_mean', { unit: words.title }), steps.map((step) => clip(stepPoint(step.rps, step.response_ms, step)))),
      stepP95Set(`${words.title}: p95`, steps.filter((step) => step.p95_ms != null).map((step) => clip(stepPoint(step.rps, step.p95_ms, step)))),
      lineSet(LL.t('forecast.chart.model_mean'), curve.map((point) => modelPoint(point.rps, point.response_ms, point)), modelColor(), false, 4),
      lineSet(LL.t('forecast.chart.model_p95'), curve.filter((point) => point.p95_ms != null).map((point) => modelPoint(point.rps, point.p95_ms, point)), COLORS.p95, true, 4)
    ].concat(latencyReferences(data, bounds));
    return {
      datasets: datasets,
      xMin: bounds.xMin,
      xMax: bounds.xMax,
      yMax: bounds.yMax,
      zones: latencyZones(data, bounds.xMax),
      caption: latencyCaption(data, datasets, bounds)
    };
  }
  function throughputView(data) {
    const points = (data.points || []).filter((point) => point.role !== 'after_cutoff');
    const curve = data.curve || [];
    const target = data.target;
    const xMax = niceMax(1.1 * maxOf([].concat(byRole(points, 'fit').map((p) => p.concurrency), curve.map((p) => p.concurrency), [target ? target.concurrency : null])));
    const dataY = maxOf([].concat(points.map((p) => p.rps), curve.map((p) => p.rps)));
    const yMax = cappedMax(dataY, [target ? target.rps : null, data.limit.rps]);
    const sample = (p) => clipped(samplePoint(p.concurrency, p.rps, p), xMax, yMax);
    const datasets = [
      pointSet(LL.t('forecast.chart.model_points'), byRole(points, 'fit').map(sample), COLORS.sample, 2.5, 6),
      pointSet(LL.t('forecast.chart.stalls'), byRole(points, 'stall').map(sample), COLORS.stall, 3.5, 5),
      edgeSet(byRole(points, 'edge').map(sample)),
      lineSet(LL.t('forecast.chart.model'), curve.map((point) => modelPoint(point.concurrency, point.rps, point)), modelColor(), false, 4)
    ];
    if (target) {
      const tip = [LL.t('forecast.point.target', { rps: formatRps(target.rps) })];
      datasets.push(lineSet(LL.t('forecast.chart.target'), horizontalLine(target.rps, xMax, tip), ANSWER[data.answer.verdict].color, true, 3));
    }
    if (data.limit.rps != null) {
      const tip = [`${limitTitle(data.limit)}: ${formatApproxRps(data.limit.rps)}`];
      datasets.push(lineSet(limitTitle(data.limit), horizontalLine(data.limit.rps, xMax, tip), COLORS.limit, false, 3));
    }
    const hidden = clippedCount(datasets);
    const caption = hidden
      ? LL.t('forecast.chart.clipped_edge', { count: hidden })
      : '';
    return { datasets: datasets, xMin: 0, xMax: xMax, yMax: yMax, zones: [], caption: caption };
  }

  function forecastAxis(title, colors, min, max) {
    return {
      type: 'linear',
      min: min,
      max: max,
      title: { display: true, text: title, color: colors.text },
      ticks: { color: colors.text },
      grid: { color: colors.grid }
    };
  }
  function createChart(canvasId, view, xTitle, yTitle) {
    const canvas = document.getElementById(canvasId);
    const colors = LL.ui.chartColors();
    return new Chart(canvas.getContext('2d'), {
      type: 'scatter',
      data: { datasets: view.datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        locale: LL.intl,
        plugins: {
          legend: {
            labels: {
              color: colors.text,
              usePointStyle: true,
              sort: (left, right) => left.datasetIndex - right.datasetIndex,
              filter: (item, chartData) => chartData.datasets[item.datasetIndex].data.length > 0
            }
          },
          tooltip: { callbacks: { label: (item) => (item.raw && item.raw.tip) || item.dataset.label || '' } },
          forecastZones: { zones: view.zones }
        },
        scales: { x: forecastAxis(xTitle, colors, view.xMin, view.xMax), y: forecastAxis(yTitle, colors, 0, view.yMax) }
      },
      plugins: [zonesPlugin, stepNumbersPlugin]
    });
  }
  function clearCharts() {
    charts.forEach((chart) => chart.destroy());
    charts = [];
  }
  function paintCharts(data) {
    clearCharts();
    if (!window.Chart) return;
    const latency = latencyView(data);
    charts.push(createChart('forecastLatency', latency, 'RPS', LL.t('forecast.chart.axis_latency')));
    setText('forecastLatencyCaption', latency.caption);
    const throughput = throughputView(data);
    charts.push(createChart('forecastThroughput', throughput, data.load_model === 'closed' ? 'VU' : LL.t('forecast.chart.axis_concurrency'), 'RPS'));
    setText('forecastThroughputCaption', throughput.caption);
  }

  function presetBase(data) {
    const sla = Number(data.sla_target_rps);
    if (Number.isFinite(sla) && sla > 0) return { rps: sla, base: 'sla' };
    return { rps: Number(data.tested_rps) || 0, base: 'tested' };
  }
  function presetValue(name) {
    if (!current) return null;
    const base = presetBase(current).rps;
    if (name === 'sla') return Number(current.sla_target_rps);
    if (name === 'plus25') return Math.round(base * 1.25);
    if (name === 'plus50') return Math.round(base * 1.5);
    if (name === 'safe') return current.safe.rps == null ? null : Math.floor(current.safe.rps);
    return null;
  }
  function controlsHtml(data, target) {
    const overSla = presetBase(data).base === 'sla';
    const plus = (pct) => (overSla ? LL.tHtml('forecast.preset.plus_sla', { pct: pct }) : LL.tHtml('forecast.preset.plus_tested', { pct: pct }));
    const presets = [
      data.sla_target_rps ? `<button type="button" class="btn" data-preset="sla">${LL.tHtml('forecast.preset.sla')}</button>` : '',
      `<button type="button" class="btn" data-preset="plus25">${plus(25)}</button>`,
      `<button type="button" class="btn" data-preset="plus50">${plus(50)}</button>`,
      `<button type="button" class="btn" data-preset="safe" id="forecastPresetSafe">${LL.tHtml('forecast.preset.safe')}</button>`
    ].join('');
    return [
      '<div class="forecast-controls">',
      `<label>${LL.tHtml('forecast.controls.target')}<input id="forecastTarget" type="number" min="1" step="1" value="${target == null ? '' : escapeHtml(target)}" /></label>`,
      `<label title="${LL.tHtml('forecast.controls.ceiling_hint')}">${LL.tHtml('forecast.controls.ceiling')}<input id="forecastCpuCeiling" type="number" min="10" max="100" step="1" value="${escapeHtml(Math.round(data.resources.ceiling_pct))}" /></label>`,
      `<label title="${LL.tHtml('forecast.controls.headroom_hint')}">${LL.tHtml('forecast.controls.headroom')}<input id="forecastHeadroom" type="number" min="0" max="90" step="1" value="${DEFAULT_HEADROOM_PCT}" /></label>`,
      '<div class="forecast-presets" role="group" aria-labelledby="forecastPresetsCaption">',
      `<span class="forecast-presets-caption" id="forecastPresetsCaption">${LL.tHtml('forecast.controls.presets')}</span>`,
      `<div class="forecast-presets-row">${presets}</div>`,
      '</div>',
      '</div>'
    ].join('');
  }
  function shellHtml(data, target) {
    return [
      '<div class="card forecast-card">',
      '<div id="forecastVerdict"></div>',
      controlsHtml(data, target),
      '<section class="forecast-block forecast-overview">',
      `<h4>${LL.tHtml('forecast.section.scale')}</h4>`,
      '<div id="forecastScale"></div>',
      '<p class="dim" id="forecastScaleCaption"></p>',
      `<h4 class="forecast-overview-steps">${LL.tHtml('forecast.section.steps')}</h4>`,
      '<div id="forecastSteps"></div>',
      '</section>',
      '<section class="forecast-block forecast-resources" id="forecastResources"></section>',
      '<section class="forecast-block forecast-explain" id="forecastExplain" hidden></section>',
      '<div class="forecast-blocks">',
      `<section class="forecast-block"><h4>${LL.tHtml('forecast.section.facts')}</h4><div id="forecastFacts"></div></section>`,
      '<section class="forecast-block" id="forecastConfidence"></section>',
      '</div>',
      '<details class="report-collapsible forecast-details">',
      `<summary>${LL.tHtml('forecast.section.details')}</summary>`,
      '<div class="report-collapsible-body">',
      '<figure class="forecast-figure">',
      `<figcaption class="forecast-chart-title">${LL.tHtml('forecast.section.latency_chart')}</figcaption>`,
      '<div class="forecast-chart forecast-chart-main"><canvas id="forecastLatency"></canvas></div>',
      '<p class="dim" id="forecastLatencyCaption"></p>',
      '</figure>',
      '</div>',
      '</details>',
      '<details class="report-collapsible forecast-details">',
      `<summary>${LL.tHtml('forecast.section.method')}</summary>`,
      '<div class="report-collapsible-body">',
      '<div id="forecastMethod"></div>',
      '<figure class="forecast-figure">',
      `<figcaption class="forecast-chart-title">${LL.tHtml('forecast.section.throughput_chart')}</figcaption>`,
      '<div class="forecast-chart forecast-chart-secondary"><canvas id="forecastThroughput"></canvas></div>',
      '<p class="dim" id="forecastThroughputCaption"></p>',
      '</figure>',
      '</div>',
      '</details>',
      '</div>'
    ].join('');
  }
  function unavailable(message) {
    return `<div class="report-note"><div class="report-note-title">${LL.tHtml('forecast.unavailable')}</div><div class="report-note-body"><p>${escapeHtml(message)}</p></div></div>`;
  }

  function findingTitle(finding) {
    const domain = (LL.DOMAIN_TITLES && LL.DOMAIN_TITLES[finding.domain]) || finding.domain;
    const start = hhmm(finding.start_time);
    const end = hhmm(finding.end_time);
    const time = start === '—' ? '' : (end !== '—' && end !== start ? `${start}–${end}` : start);
    return [domain, time].filter(Boolean).join(' · ');
  }
  function findingItem(finding) {
    return `<li><span class="forecast-finding-meta">${escapeHtml(findingTitle(finding))}</span>${escapeHtml(finding.summary)}</li>`;
  }
  function evidenceItem(link, byRef) {
    const finding = byRef[link.finding_id];
    const title = finding ? findingTitle(finding) : LL.t('forecast.explain.finding_missing', { id: link.finding_id });
    return `<li><span class="forecast-finding-meta">${escapeHtml(title)}</span>${escapeHtml(link.role)}</li>`;
  }
  function checkText(check) {
    const parts = [];
    if (check.status === 'verified') parts.push(LL.t('forecast.explain.check_verified', { matched: check.claims_matched, total: check.claims_total }));
    else if (check.status === 'unverified') parts.push(LL.t('forecast.explain.check_unverified', { numbers: check.unmatched.join(', ') }));
    else parts.push(LL.t('forecast.explain.check_none'));
    if (check.unknown_findings.length) parts.push(LL.t('forecast.explain.check_unknown', { findings: check.unknown_findings.join(', ') }));
    return parts.join('; ');
  }
  function cappedList(items, tag) {
    const shown = items.slice(0, EXPLAIN_ITEMS).map((item) => `<li>${escapeHtml(item)}</li>`).join('');
    const rest = items.slice(EXPLAIN_ITEMS);
    const more = rest.length
      ? `<details class="forecast-more"><summary>${LL.tHtml('forecast.explain.more', { count: rest.length })}</summary><${tag} start="${EXPLAIN_ITEMS + 1}">${rest.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</${tag}></details>`
      : '';
    return items.length ? `<${tag}>${shown}</${tag}>${more}` : `<p class="dim">${LL.tHtml('forecast.explain.none')}</p>`;
  }
  function keyFactsHtml(facts) {
    if (!facts.length) return '';
    const cards = facts.map((fact) => [
      '<div class="forecast-fact">',
      `<span class="forecast-fact-value">${escapeHtml(fact.value)}</span>`,
      `<span class="forecast-fact-label">${escapeHtml(fact.label)}</span>`,
      '</div>'
    ].join('')).join('');
    return `<div class="forecast-facts">${cards}</div>`;
  }
  function explanationBody(view) {
    const result = view.explanation;
    const byRef = Object.fromEntries(view.findings.map((finding) => [finding.ref, finding]));
    const generated = view.generated_at ? LL.fmt.dateTime(view.generated_at, { dateStyle: 'short', timeStyle: 'short' }) : '';
    const meta = [view.model, generated, checkText(view.check)].filter(Boolean).join(' · ');
    const scaling = SCALING[result.scaling.verdict];
    const evidence = result.evidence.length
      ? `<details class="forecast-findings"><summary>${LL.tHtml('forecast.explain.evidence', { count: result.evidence.length })}</summary><ul>${result.evidence.map((link) => evidenceItem(link, byRef)).join('')}</ul></details>`
      : '';
    return [
      `<p class="forecast-explain-headline">${escapeHtml(result.headline)}</p>`,
      `<p class="forecast-explain-cause">${escapeHtml(result.cause)}</p>`,
      keyFactsHtml(result.key_facts),
      `<div class="forecast-explain-scaling is-${scaling.tone}"><strong>${escapeHtml(scaling.text)}.</strong> ${escapeHtml(result.scaling.reason)}</div>`,
      '<div class="forecast-explain-columns">',
      `<section><h5>${LL.tHtml('forecast.explain.risks')}</h5>${cappedList(result.scaling_risks, 'ul')}</section>`,
      `<section><h5>${LL.tHtml('forecast.explain.next_checks')}</h5>${cappedList(result.next_checks, 'ol')}</section>`,
      '</div>',
      evidence,
      `<p class="dim forecast-explain-meta">${escapeHtml(meta)}</p>`
    ].join('');
  }
  function explanationHtml(view) {
    const done = view.explanation != null;
    const button = view.llm_problem
      ? `<span class="dim">${escapeHtml(view.llm_problem)}</span>`
      : `<button type="button" class="btn btn-sm" id="forecastExplainRun">${done ? LL.tHtml('forecast.explain.refresh') : LL.tHtml('forecast.explain.run')}</button>`;
    const confidence = done ? CONFIDENCE[view.explanation.confidence] : null;
    const pill = confidence ? `<span class="pill ${confidence.pill}">${LL.tHtml('forecast.explain.confidence', { level: confidence.text })}</span>` : '';
    const action = `<span class="forecast-explain-tools">${pill}${button}</span>`;
    const intro = `<p class="dim">${LL.tHtml('forecast.explain.intro')}</p>`;
    const findings = view.findings.length
      ? `<details class="forecast-findings"${done ? '' : ' open'}><summary>${LL.tHtml('forecast.explain.findings', { count: view.findings.length })}</summary><ul>${view.findings.map(findingItem).join('')}</ul></details>`
      : `<p class="dim">${LL.tHtml('forecast.explain.no_findings')}</p>`;
    return [
      `<div class="forecast-explain-head"><h4>${LL.tHtml('forecast.explain.title')}</h4>${action}</div>`,
      view.status === 'stale' ? `<p class="dim">${escapeHtml(view.message)}</p>` : '',
      done ? explanationBody(view) : intro,
      '<div id="forecastExplainError"></div>',
      findings
    ].join('');
  }
  function renderExplanation(view) {
    const block = document.getElementById('forecastExplain');
    block.hidden = view.status === 'unavailable';
    if (block.hidden) return;
    block.innerHTML = explanationHtml(view);
    const button = document.getElementById('forecastExplainRun');
    if (button) button.addEventListener('click', runExplanation);
  }
  async function explanationRequest(method) {
    const options = method === 'POST' ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' } : {};
    const resp = await fetch('/forecast/' + encodeURIComponent(runId) + '/explanation', options);
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) return { error: body.error || LL.t('forecast.explain.error', { status: resp.status }) };
    return { view: body };
  }
  async function loadExplanation() {
    const result = await explanationRequest('GET');
    if (!result.error) {
      renderExplanation(result.view);
      return;
    }
    const block = document.getElementById('forecastExplain');
    block.hidden = false;
    block.innerHTML = `<div class="forecast-explain-head"><h4>${LL.tHtml('forecast.explain.title')}</h4></div><p class="forecast-explain-error">${escapeHtml(result.error)}</p>`;
  }
  async function runExplanation() {
    const button = document.getElementById('forecastExplainRun');
    button.disabled = true;
    button.textContent = LL.t('forecast.explain.running');
    setHtml('forecastExplainError', '');
    const result = await explanationRequest('POST');
    if (!result.error) {
      renderExplanation(result.view);
      return;
    }
    button.disabled = false;
    button.textContent = LL.t('forecast.explain.retry');
    setHtml('forecastExplainError', `<p class="forecast-explain-error">${escapeHtml(result.error)}</p>`);
  }

  function percentOf(value, end) {
    return Math.max(0, Math.min(100, (Number(value) / end) * 100));
  }
  function zoneText(data, zone) {
    if (!data.limit || data.limit.rps == null) return LL.t('forecast.scale.zone_no_stalls');
    if (zone.tone === 'fail') return data.limit.kind === 'instability' ? LL.t('forecast.scale.zone_stalls') : LL.t('forecast.scale.zone_over_ceiling');
    return zone.tone === 'warn' ? LL.t('forecast.scale.zone_tight') : LL.t('forecast.scale.zone_safe');
  }
  function markText(data, mark) {
    if (mark.kind === 'safe') return { value: formatRps(Math.floor(mark.rps)), label: LL.t('forecast.scale.mark_safe') };
    if (mark.kind === 'limit') {
      const label = data.limit.kind === 'instability' ? LL.t('forecast.scale.mark_stalls') : LL.t('forecast.scale.mark_ceiling');
      return { value: formatApproxRps(mark.rps), label: label };
    }
    return { value: formatRps(Math.round(mark.rps)), label: LL.t('forecast.scale.mark_tested') };
  }
  function placedMarks(data, end) {
    const lastByRow = [];
    return data.scale.marks.map((mark) => {
      const pct = percentOf(mark.rps, end);
      let row = 0;
      while (lastByRow[row] != null && pct - lastByRow[row] < SCALE_LABEL_GAP_PCT) row += 1;
      lastByRow[row] = pct;
      return Object.assign({ pct: pct, row: row }, markText(data, mark));
    });
  }
  function edgeClass(pct) {
    if (pct < SCALE_EDGE_PCT) return ' is-start';
    return pct > 100 - SCALE_EDGE_PCT ? ' is-end' : '';
  }
  function scaleTargetHtml(data, end) {
    if (!data.target || !data.answer) return '';
    const pct = percentOf(data.target.rps, end);
    const beyond = data.target.rps > end ? ' →' : '';
    const tone = SCALE_TARGET_TONES[data.answer.verdict];
    return `<div class="forecast-scale-target is-${tone}${edgeClass(pct)}" style="left:${pct}%"><span>${LL.tHtml('forecast.scale.target', { rps: formatRps(data.target.rps) })}${beyond}</span></div>`;
  }
  function scaleHtml(data) {
    const end = data.scale.end_rps;
    if (!(end > 0)) return '';
    const zones = data.scale.zones.map((zone) => {
      const left = percentOf(zone.from_rps, end);
      const width = percentOf(zone.to_rps, end) - left;
      const label = zoneText(data, zone);
      const text = width >= SCALE_ZONE_TEXT_PCT ? escapeHtml(label) : '';
      return `<div class="forecast-scale-zone is-${zone.tone}" style="left:${left}%;width:${width}%" title="${escapeHtml(label)}">${text}</div>`;
    }).join('');
    const tested = percentOf(data.scale.tested_rps, end);
    const untested = tested < 100 ? `<div class="forecast-scale-untested" style="left:${tested}%;width:${100 - tested}%" title="${LL.tHtml('forecast.scale.untested')}"></div>` : '';
    const marks = placedMarks(data, end);
    const rows = Math.max.apply(null, marks.map((mark) => mark.row)) + 1;
    const marksHtml = marks.map((mark) => (
      `<div class="forecast-scale-mark${edgeClass(mark.pct)}" style="left:${mark.pct}%;--row:${mark.row}"><b>${escapeHtml(mark.value)}</b> ${escapeHtml(mark.label)}</div>`
    )).join('');
    return [
      '<div class="forecast-scale">',
      scaleTargetHtml(data, end),
      `<div class="forecast-scale-bar">${zones}${untested}</div>`,
      `<div class="forecast-scale-marks" style="height:${rows * SCALE_ROW_PX + 10}px">${marksHtml}</div>`,
      '</div>'
    ].join('');
  }
  function scaleCaption(data) {
    if (data.limit.rps == null) return LL.t('forecast.scale.caption_no_limit');
    return data.limit.kind === 'instability' ? LL.t('forecast.scale.caption_instability') : LL.t('forecast.scale.caption_model');
  }
  function stepState(step, tone) {
    if (tone === 'muted') return LL.t('forecast.steps.state_after_drop');
    if (tone === 'fail') return LL.t('forecast.steps.state_stalls', { stalls: step.stalls, samples: step.samples });
    return LL.t('forecast.steps.state_ok');
  }
  function stepsHtml(data) {
    const steps = data.steps || [];
    if (!steps.length) return `<p class="dim">${LL.tHtml('forecast.steps.none')}</p>`;
    const words = stepWords(data);
    const cards = steps.map((step) => {
      const tone = step.after_drop ? 'muted' : (step.stalls ? 'fail' : 'ok');
      const state = stepState(step, tone);
      return [
        `<div class="forecast-step is-${tone}" title="${escapeHtml(step.label)}">`,
        `<span class="forecast-step-name">${escapeHtml(words.one)} ${step.number}</span>`,
        `<span class="forecast-step-rps">${escapeHtml(formatRps(Math.round(step.rps)))}</span>`,
        `<span class="forecast-step-time">${LL.tHtml('forecast.point.mean', { value: formatMs(step.response_ms) })}</span>`,
        `<span class="forecast-step-state">${escapeHtml(state)}</span>`,
        '</div>'
      ].join('');
    }).join('');
    return `<div class="forecast-steps">${cards}</div>`;
  }

  // Labels of the load scale stay inside it: a centered mark (or the target label) that would cross an edge of a
  // narrow scale is anchored to that edge (fit-start / fit-end), as the marks near the ends are (is-start / is-end).
  // Runs after the scale is drawn, on resize and when the web fonts are in.
  function fitScaleLabels(root) {
    const scale = root && typeof root.querySelector === 'function' ? root.querySelector('.forecast-scale') : null;
    if (!scale || typeof scale.getBoundingClientRect !== 'function') return;
    const box = scale.getBoundingClientRect();
    if (!box || !box.width) return;
    Array.from(scale.querySelectorAll('.forecast-scale-mark, .forecast-scale-target')).forEach((el) => {
      el.classList.remove('fit-start', 'fit-end');
      if (el.classList.contains('is-start') || el.classList.contains('is-end')) return;
      const label = el.classList.contains('forecast-scale-target') ? el.querySelector('span') : el;
      if (!label || typeof label.getBoundingClientRect !== 'function') return;
      const rect = label.getBoundingClientRect();
      if (rect.right > box.right + 0.5) el.classList.add('fit-end');
      else if (rect.left < box.left - 0.5) el.classList.add('fit-start');
    });
  }
  let fitTimer = 0;
  function refitScaleLabels() {
    clearTimeout(fitTimer);
    fitTimer = setTimeout(() => fitScaleLabels(document.getElementById('forecastScale')), 100);
  }

  function paintForecast(data, headroom) {
    current = data;
    setHtml('forecastVerdict', answerHtml(data, headroom));
    setHtml('forecastScale', scaleHtml(data));
    fitScaleLabels(document.getElementById('forecastScale'));
    setText('forecastScaleCaption', scaleCaption(data));
    setHtml('forecastSteps', stepsHtml(data));
    setHtml('forecastResources', resourcesHtml(data));
    setHtml('forecastFacts', listHtml(factItems(data)));
    const confidence = document.getElementById('forecastConfidence');
    confidence.innerHTML = confidenceHtml(data);
    confidence.hidden = confidence.innerHTML === '';
    setHtml('forecastMethod', listHtml(methodItems(data)));
    document.getElementById('forecastPresetSafe').disabled = data.safe.rps == null;
    const headroomInput = document.getElementById('forecastHeadroom');
    headroomInput.disabled = !data.limit_applies;
    headroomInput.title = data.limit_applies ? '' : LL.t('forecast.controls.headroom_disabled');
    paintCharts(data);
  }
  function numberInput(id, fallback) {
    const input = document.getElementById(id);
    const value = input.value === '' ? fallback : Number(input.value);
    return Number.isFinite(value) ? value : fallback;
  }
  async function refreshForecast() {
    const target = numberInput('forecastTarget', null);
    const headroom = numberInput('forecastHeadroom', DEFAULT_HEADROOM_PCT);
    const ceiling = numberInput('forecastCpuCeiling', null);
    const result = await fetchForecast(target, headroom, ceiling === configuredCeiling ? null : ceiling);
    if (result.error) {
      clearCharts();
      setHtml('forecastVerdict', unavailable(result.error));
      return;
    }
    paintForecast(result.data, headroom);
  }
  function bindControls() {
    const targetInput = document.getElementById('forecastTarget');
    const schedule = () => {
      window.clearTimeout(refreshTimer);
      refreshTimer = window.setTimeout(refreshForecast, 400);
    };
    ['forecastTarget', 'forecastCpuCeiling', 'forecastHeadroom'].forEach((id) => {
      document.getElementById(id).addEventListener('input', schedule);
    });
    document.querySelectorAll('#forecastPanel [data-preset]').forEach((button) => {
      button.addEventListener('click', () => {
        const value = presetValue(button.dataset.preset);
        if (!Number.isFinite(value) || value <= 0) return;
        targetInput.value = String(value);
        refreshForecast();
      });
    });
  }
  function setPublication(url) {
    const link = document.getElementById('forecastConfluenceLink');
    link.hidden = !url;
    if (url) link.href = url;
    document.getElementById('forecastConfluence').textContent = url ? LL.t('forecast.confluence.update') : LL.t('forecast.confluence.add');
  }
  async function loadPublication() {
    const resp = await fetch('/forecast/' + encodeURIComponent(runId) + '/confluence');
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      setText('forecastConfluenceStatus', body.error || LL.t('forecast.confluence.status_error', { status: resp.status }));
      return;
    }
    setPublication(body.page_url || '');
  }
  function jobFailure(job) {
    const headline = job.message || LL.t('forecast.confluence.failed');
    return job.error ? `${headline}: ${job.error}` : headline;
  }
  async function pollPublication(jobId) {
    for (let attempt = 0; attempt < PUBLISH_POLLS; attempt += 1) {
      const resp = await fetch('/job_status/' + encodeURIComponent(jobId));
      const job = await resp.json().catch(() => ({}));
      if (job.status === 'done') {
        setPublication(job.page_url || '');
        setText('forecastConfluenceStatus', LL.t('forecast.confluence.published'));
        LL.ui.toast(LL.t('forecast.confluence.published_toast'), { tone: 'ok' });
        return;
      }
      if (!resp.ok || job.status === 'error' || job.status === 'not_found') {
        setText('forecastConfluenceStatus', jobFailure(job));
        LL.ui.toast(jobFailure(job), { tone: 'error' });
        return;
      }
      const pct = Number(job.progress || 0);
      setText('forecastConfluenceStatus', `${job.message || LL.t('forecast.confluence.publishing')}${pct ? ` (${pct}%)` : ''}`);
      await new Promise((resolve) => setTimeout(resolve, PUBLISH_POLL_MS));
    }
    setText('forecastConfluenceStatus', LL.t('forecast.confluence.timeout'));
  }
  async function publishForecast() {
    const button = document.getElementById('forecastConfluence');
    const ceiling = numberInput('forecastCpuCeiling', null);
    const payload = {
      run_name: document.getElementById('forecastTitle').textContent,
      target_rps: numberInput('forecastTarget', null),
      headroom_pct: numberInput('forecastHeadroom', DEFAULT_HEADROOM_PCT),
      cpu_ceiling_pct: ceiling === configuredCeiling ? null : ceiling
    };
    button.disabled = true;
    setText('forecastConfluenceStatus', LL.t('forecast.confluence.publishing'));
    try {
      const resp = await fetch('/forecast/' + encodeURIComponent(runId) + '/confluence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok || !body.job_id) {
        setText('forecastConfluenceStatus', body.error || LL.t('forecast.confluence.start_error', { status: resp.status }));
        return;
      }
      await pollPublication(body.job_id);
    } finally {
      button.disabled = false;
    }
  }
  function enablePublishing() {
    const button = document.getElementById('forecastConfluence');
    button.hidden = false;
    button.addEventListener('click', publishForecast);
    loadPublication();
  }
  function defaultTarget(data) {
    const sla = Number(data.sla_target_rps);
    if (Number.isFinite(sla) && sla > 0) return sla;
    const tested = Number(data.tested_rps);
    return Number.isFinite(tested) && tested > 0 ? Math.round(tested * 1.25) : null;
  }
  async function start() {
    runId = runIdFromPath();
    const box = document.getElementById('forecastPanel');
    if (!runId || !box) return;
    const refResp = await fetch('/report_ref/' + encodeURIComponent(runId));
    if (!refResp.ok) {
      document.getElementById('forecastTitle').textContent = LL.t('forecast.page.not_found');
      box.innerHTML = unavailable(LL.t('forecast.page.not_found'));
      return;
    }
    const ref = await refResp.json();
    const name = ref.run_name || LL.t('forecast.page.report');
    document.getElementById('forecastTitle').textContent = name;
    document.title = LL.t('forecast.page.document_title', { name: name });
    const crumbs = document.getElementById('forecastBreadcrumbs');
    crumbs.innerHTML = `<a href="/forecasting">${LL.tHtml('forecast.list.title')}</a><span class="sep">›</span><span class="current">${escapeHtml(name)}</span>`;
    const link = document.getElementById('forecastReportLink');
    link.href = '/reports/' + encodeURIComponent(runId);
    link.hidden = false;
    box.innerHTML = `<div class="card">${LL.ui.skeleton(4)}</div>`;
    const first = await fetchForecast(null, DEFAULT_HEADROOM_PCT);
    if (first.error) {
      box.innerHTML = unavailable(first.error);
      return;
    }
    const target = defaultTarget(first.data);
    const result = target ? await fetchForecast(target, DEFAULT_HEADROOM_PCT) : first;
    if (result.error) {
      box.innerHTML = unavailable(result.error);
      return;
    }
    configuredCeiling = Math.round(result.data.resources.ceiling_pct);
    box.innerHTML = shellHtml(result.data, target);
    bindControls();
    paintForecast(result.data, DEFAULT_HEADROOM_PCT);
    window.addEventListener('resize', refitScaleLabels);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(refitScaleLabels, () => {});
    loadExplanation();
    enablePublishing();
  }
  // Pure builders of the page for tests (tests/test_i18n_forecast.py).
  LL.forecastReport = {
    _internals: {
      formatNumber, formatMs, formatPct, podsText, kappaText, answerHtml, resourcesHtml, factItems, confidenceHtml,
      methodItems, scaleHtml, scaleCaption, stepsHtml, latencyView, throughputView, controlsHtml, shellHtml,
      unavailable, explanationHtml, fitScaleLabels
    }
  };
  document.addEventListener('DOMContentLoaded', start);
})();
