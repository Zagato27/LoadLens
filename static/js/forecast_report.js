(function () {
  const LL = window.LoadLens;
  const escapeHtml = LL.ui.escapeHtml;
  const LOCALE = 'ru-RU';
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
    zoneUnstable: 'rgba(239, 68, 68, 0.10)'
  };
  function accentColor() {
    return LL.ui.chartColors().accent || '#6200ee';
  }
  function accentZone() {
    return LL.ui.chartColors().accentSoft || 'rgba(98, 0, 238, 0.16)';
  }
  const CONFIDENCE = {
    high: { text: 'высокая', pill: 'ok' },
    medium: { text: 'средняя', pill: 'warn' },
    low: { text: 'низкая', pill: 'fail' }
  };
  const ANSWER = {
    holds: { text: 'выдержит', tone: 'success', color: '#22c55e' },
    scale: { text: 'выдержит после масштабирования', tone: 'risk', color: '#eab308' },
    tight: { text: 'на пределе', tone: 'risk', color: '#eab308' },
    blocked: { text: 'не выдержит', tone: 'fail', color: '#ef4444' }
  };
  const CEILING_SOURCES = { sla: 'из SLA', default: 'по умолчанию, в SLA не задан', request: 'задан вручную' };
  const SCALING = {
    helps: { text: 'Добавление подов поможет', tone: 'success' },
    partly: { text: 'Добавление подов поможет частично', tone: 'risk' },
    no: { text: 'Добавление подов не снимет причину', tone: 'fail' }
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
  const STEP_WORDS = {
    step_detector: { title: 'Ступени', many: 'ступени', one: 'ступень', at: 'ступени', by: ['ступени', 'ступеням', 'ступеням'] },
    time_buckets: { title: 'Интервалы', many: 'интервалы', one: 'интервал', at: 'интервале', by: ['интервалу', 'интервалам', 'интервалам'] }
  };
  const ROLE_TITLES = {
    fit: 'в модели',
    stall: 'провал',
    edge: 'разгон или спад нагрузки',
    after_cutoff: 'после окна модели'
  };
  const REASONS = {
    low_r2: {
      reason: (data) => `Модель объясняет ${formatPct(Math.max(0, data.fit.r2) * 100)} разброса данных (R² = ${formatNumber(data.fit.r2, 2)}).`,
      action: (data) => (data.instability ? '' : 'Повторить тест с более длинными ступенями: точки сильно разбросаны.')
    },
    instability: {
      reason: (data) => `Провалы не укладываются в плавную модель: выше ${formatApproxRps(data.instability.onset_rps)} прогноз неприменим.`,
      action: (data) => `Разобраться с провалами${periodText(data.instability, ' (повторяются примерно каждые ', ' мин)')} и повторить ступени около ${formatRps(threeSignificant(data.instability.onset_rps))}.`
    },
    extrapolation: {
      reason: () => 'Цель за пределами проверенного диапазона: прогноз — экстраполяция.',
      action: (data) => `Проверить ${formatRps(data.target.rps)} отдельным тестом.`
    },
    cpu_fit: {
      reason: (data) => `CPU ${decidingServices(data.resources).filter((item) => !item.reliable).map((item) => item.name).join(', ')} плохо ложится на прямую: число подов приблизительное.`,
      action: () => 'Проверить число подов тестом.'
    },
    cpu_extrapolation: {
      reason: (data) => `Цель в ${formatNumber(data.resources.target_rps / data.tested_rps, 1)} раза выше проверенной нагрузки: CPU посчитан продолжением прямой.`,
      action: (data) => `Проверить ${formatRps(data.resources.target_rps)} отдельным тестом.`
    },
    scaling_untested: {
      reason: () => 'Конфигурацию с новым числом подов тест не проверял.',
      action: (data) => `Проверить тестом: ${scaledServices(data.resources).map((item) => `${item.name} — ${podsText(item.instances_needed)}`).join(', ')} на ${formatRps(data.resources.target_rps)}.`
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
        ctx.font = "700 11px 'Montserrat', Arial, sans-serif";
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

  const numberFormats = {};
  function formatNumber(value, digits) {
    const n = Number(value);
    if (value === null || value === undefined || value === '' || !Number.isFinite(n)) return '—';
    const key = digits || 0;
    if (!numberFormats[key]) numberFormats[key] = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: key });
    return numberFormats[key].format(n);
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
    if (n >= 1000) return `${formatNumber(n / 1000, 1)} с`;
    return `${formatNumber(n, n < 10 ? 1 : 0)} мс`;
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
    return `${formatNumber(n, Math.abs(n) < 10 ? 1 : 0)} %`;
  }
  function cpuPct(share) {
    return formatPct(Number(share) * 100);
  }
  function podsText(count) {
    return `${count} ${plural(count, 'под', 'пода', 'подов')}`;
  }
  function plural(count, one, few, many) {
    const tail = count % 100;
    if (tail % 10 === 1 && tail !== 11) return one;
    if (tail % 10 >= 2 && tail % 10 <= 4 && (tail < 10 || tail >= 20)) return few;
    return many;
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
  function periodText(instability, before, after) {
    return instability.period_min == null ? '' : `${before}${formatNumber(instability.period_min)}${after}`;
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
    if (!resp.ok) return { error: body.error || `Не удалось построить прогноз (HTTP ${resp.status})`, code: body.code || '' };
    return { data: body };
  }

  function limitText(data) {
    const limit = data.limit;
    if (limit.kind === 'instability') return `Предел ${formatApproxRps(limit.rps)}: с этой нагрузки в тесте начались провалы.`;
    if (limit.kind === 'model') return `Предел ${formatApproxRps(limit.rps)}: потолок пропускной способности по модели.`;
    return 'Предел по данным не виден: в проверенном диапазоне RPS растёт почти линейно.';
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
    return `${item.name}: CPU пода без роста нагрузки ≈ ${cpuPct(item.cpu_base)} — выше потолка ${cpuPct(item.cpu_ceiling)}, подами не решается: нужен больший лимит CPU.`;
  }
  function scaleLines(data) {
    const plan = data.resources;
    const lines = scaledServices(plan).map((item) => (item.instances_needed == null
      ? stuckText(item)
      : `Добавить: ${item.name} ${item.instances} → ${podsText(item.instances_needed)}, CPU пода при цели ≈ ${cpuPct(item.cpu_after)} вместо ${cpuPct(item.cpu_at_target)}.`));
    if (plan.scaled_capacity_rps != null) {
      lines.push(`После этого CPU хватит до ${formatApproxRps(plan.scaled_capacity_rps)} — дальше первым упрётся ${plan.next_bottleneck}.`);
    }
    if (data.limit_applies && data.limit.rps != null) {
      lines.push(`Выше ${formatApproxRps(data.limit.rps)} упрётся в предел, не связанный с CPU сервисов.`);
    }
    return lines;
  }
  function blockedText(data) {
    const plan = data.resources;
    const busiest = plan.limit_service ? ` При этой нагрузке самый загруженный сервис — ${plan.limit_service}, CPU пода ≈ ${cpuPct(plan.limit_cpu)}.` : '';
    return `Предел ${formatApproxRps(data.limit.rps)} не связан с CPU сервисов.${busiest} Добавление подов не поможет: ищите блокировки, пулы соединений, GC, внешние вызовы.`;
  }
  function blockedLines(data) {
    const plan = data.resources;
    const scaled = scaledServices(plan);
    const stuck = scaled.filter((item) => item.instances_needed == null).map(stuckText);
    if (plan.status !== 'ok' || !data.limit_applies) return stuck;
    const short = scaled
      .filter((item) => item.instances_needed != null)
      .map((item) => `CPU тоже не хватит: ${item.name} при цели ≈ ${cpuPct(item.cpu_at_target)} — нужно ${podsText(item.instances_needed)}, но предел это не снимет.`);
    return [blockedText(data), ...stuck, ...short];
  }
  function holdsLines(data) {
    const plan = data.resources;
    const lines = [];
    if (plan.status === 'ok' && plan.services.length) {
      const top = plan.services[0];
      lines.push(`Самый загруженный — ${top.name}: CPU пода при цели ≈ ${cpuPct(top.cpu_at_target)}, потолок ${cpuPct(top.cpu_ceiling)}.`);
    }
    if (data.limit_applies && data.target.headroom_pct != null) lines.push(`Запас до предела ${formatPct(data.target.headroom_pct)}.`);
    return lines;
  }
  function tightText(data, headroom) {
    const cause = data.resources.status === 'ok' ? ': предел не связан с CPU сервисов' : '';
    return `Запас до предела ${formatPct(data.target.headroom_pct)} при требуемых ${formatPct(headroom)}${cause}.`;
  }
  function limitLine(data) {
    const plan = data.resources;
    if (!data.limit_applies && plan.limit_cause) {
      const subject = data.limit.kind === 'instability'
        ? `Провалы с ${formatApproxRps(data.limit.rps)} объясняются`
        : `Потолок модели ${formatApproxRps(data.limit.rps)} объясняется`;
      return `${subject} CPU: ${plan.limit_cause} при этой нагрузке ≈ ${cpuPct(plan.limit_cpu)} пода.`;
    }
    const verdict = data.answer ? data.answer.verdict : '';
    if (data.limit_applies && plan.status === 'ok' && (verdict === 'blocked' || verdict === 'scale')) return '';
    return limitText(data);
  }
  function safeLine(data, headroom) {
    const safe = data.safe;
    if (!safe || safe.rps == null) return '';
    const value = formatRps(Math.floor(safe.rps));
    if (safe.source === 'cpu') return `Безопасный максимум текущей конфигурации: ${value} — дальше ${safe.service} выше ${cpuPct(serviceCeiling(data, safe.service))} CPU.`;
    return `Безопасный максимум с запасом ${formatPct(headroom)} до предела: ${value}.`;
  }
  function latencyLine(data) {
    const target = data.target;
    if (!target || target.response_ms == null || ['scale', 'blocked'].includes(data.answer.verdict)) return '';
    const p95 = target.p95_ms != null ? `, p95 ≈ ${formatMs(target.p95_ms)}` : '';
    return `Время отклика при цели по модели: среднее ≈ ${formatMs(target.response_ms)}${p95}.`;
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
      `<span class="forecast-verdict-title">Цель ${escapeHtml(formatRps(data.target.rps))}: ${view.text}</span>`,
      `<span class="pill ${confidence.pill}">Достоверность: ${confidence.text}</span>`,
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
    if (item.name === plan.bottleneck) notes.push('упрётся первым');
    if (item.name === plan.limit_cause && item.cpu_ceiling < plan.ceiling_pct / 100) notes.push(`потолок ${cpuPct(item.cpu_ceiling)}: на этой загрузке тест упёрся в предел`);
    if (!item.reliable && (changed || item.name === plan.bottleneck)) notes.push('CPU плохо ложится на прямую');
    return notes;
  }
  function resourceRow(item, plan) {
    const changed = scaledServices(plan).includes(item);
    const notes = resourceNotes(item, plan);
    const needed = item.instances_needed == null ? 'подами не решается' : String(item.instances_needed);
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
    if (!nodes.length) return 'Узлы: на стабильных ступенях нет данных CPU.';
    const over = nodes.filter((node) => node.cpu_at_target >= plan.ceiling_pct / 100);
    if (over.length) {
      const list = over.slice(0, 4).map((node) => `${node.name} ≈ ${cpuPct(node.cpu_at_target)}`).join(', ');
      return `Узлы выше потолка при цели: ${list} — добавленным подам нужно место на других узлах.`;
    }
    return `Узлы: самый загруженный при цели — ${nodes[0].name} ≈ ${cpuPct(nodes[0].cpu_at_target)}, ниже потолка.`;
  }
  function resourcesHtml(data) {
    const plan = data.resources;
    const title = `<h4>Поды при ${escapeHtml(formatRps(plan.target_rps))}</h4>`;
    if (plan.status !== 'ok') return `${title}<p>${escapeHtml(plan.message)}</p>`;
    const words = stepWords(data);
    const rows = resourceRows(plan);
    const rest = plan.services.length - rows.length;
    const basis = `${plan.steps_used} ${plural(plan.steps_used, words.by[0], words.by[1], words.by[2])}`;
    const note = `Потолок CPU пода ${formatPct(plan.ceiling_pct)} (${CEILING_SOURCES[plan.ceiling_source]}); CPU пода — прямая по ${basis} без провалов.`;
    const restText = rest > 0
      ? `<p class="dim">Ещё ${rest} ${plural(rest, 'сервис', 'сервиса', 'сервисов')} при цели не выше ${escapeHtml(cpuPct(rows[rows.length - 1].cpu_at_target))} CPU пода.</p>`
      : '';
    return [
      title,
      `<p class="dim">${escapeHtml(note)}</p>`,
      '<div class="step-table-wrap"><table class="step-table forecast-table">',
      `<thead><tr><th>Сервис</th><th class="num">Подов</th><th class="num">CPU пода, ${escapeHtml(words.one)} ${plan.last_step}</th><th class="num">CPU пода при цели</th><th class="num">Нужно подов</th></tr></thead>`,
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
    return `Без провалов: ${name} ${numberRange(stable.map((step) => step.number))}, ${load}, среднее ${formatMsRange(stable.map((step) => step.response_ms))}.`;
  }
  function instabilityText(data) {
    const info = data.instability;
    const p95 = info.worst_p95_ms != null ? `, p95 до ${formatMs(info.worst_p95_ms)}` : '';
    const points = `${info.stalls} из ${info.samples} ${plural(info.samples, 'точки', 'точек', 'точек')} по ${formatNumber(data.bin_minutes, 1)} мин`;
    return `С ${hhmm(info.onset_iso)} (${info.onset_label}, ${formatApproxRps(info.onset_rps)}) начались провалы: ${points}, среднее до ${formatMs(info.worst_response_ms)}${p95}.${periodText(info, ' Повторяются примерно каждые ', ' мин.')}`;
  }
  function bestText(data) {
    const steps = (data.steps || []).filter((step) => !step.after_drop);
    const bin = `${formatNumber(data.bin_minutes, 1)} мин`;
    if (!steps.length) return `Максимум RPS за ${bin}: ${formatNumber(data.peak_rps)}.`;
    const best = steps.reduce((top, step) => (step.rps > top.rps ? step : top));
    return `Максимум RPS: в среднем за ${stepWords(data).one} — ${formatNumber(best.rps)} (${best.label}), за ${bin} — ${formatNumber(data.peak_rps)}.`;
  }
  function cpuFact(data) {
    const plan = data.resources;
    if (plan.status !== 'ok' || !plan.services.length) return '';
    const top = plan.services.reduce((best, item) => (item.cpu_measured > best.cpu_measured ? item : best));
    return `Больше всего CPU: ${top.name} — ${cpuPct(top.cpu_measured)} пода на ${stepWords(data).at} ${plan.last_step}, ${podsText(top.instances)}.`;
  }
  function factItems(data) {
    const items = [];
    const stable = stableSteps(data);
    if (stable.length) items.push(stableText(data, stable));
    const cpu = cpuFact(data);
    if (cpu) items.push(cpu);
    if (data.instability) items.push(instabilityText(data));
    else if (data.excluded_stalls) items.push(`Отдельные всплески времени отклика: ${data.excluded_stalls} — в модель не вошли.`);
    items.push(bestText(data));
    if (data.cutoff_kind === 'rps_drop') items.push(`В ${hhmm(data.cutoff_iso)} RPS упал — точки после этого в модель не вошли.`);
    return items;
  }
  function uslItems(data) {
    const fit = data.fit || {};
    const ceiling = fit.x_max != null
      ? `Потолок пропускной способности по модели USL ${formatApproxRps(fit.x_max)}.`
      : 'Потолок по модели USL не виден: в проверенном диапазоне RPS растёт почти линейно.';
    const knee = data.knee
      ? `Колено ${formatApproxRps(data.knee.rps)}: дальше время отклика растёт быстрее нагрузки.`
      : 'Колена до предела нет: время отклика растёт не быстрее нагрузки.';
    return [ceiling, knee];
  }
  function confidentBasis(data) {
    const r2 = `R² = ${formatNumber(data.fit.r2, 2)}`;
    if (data.resources.status === 'ok' && !data.limit_applies) return 'Расчёт опирается на измеренные ступени: CPU сервисов хорошо ложится на прямую.';
    if (data.answer.verdict === 'blocked') return `Модель хорошо описывает данные (${r2}): предел ${formatApproxRps(data.limit.rps)} виден уверенно.`;
    return `Цель внутри проверенного диапазона, модель хорошо описывает данные (${r2}).`;
  }
  function confidenceHtml(data) {
    const answer = data.answer;
    if (!answer) return '';
    const title = `<h4>Достоверность: ${CONFIDENCE[answer.confidence].text}</h4>`;
    const codes = answer.confidence_reasons || [];
    if (!codes.length) return `${title}<p>${escapeHtml(confidentBasis(data))}</p>`;
    const actions = codes.map((code) => REASONS[code].action(data)).filter(Boolean);
    const actionsHtml = actions.length ? `<h5>Что делать</h5>${listHtml(actions)}` : '';
    return `${title}${listHtml(codes.map((code) => REASONS[code].reason(data)))}${actionsHtml}`;
  }
  function exclusionText(data) {
    const parts = [];
    if (data.excluded_edges) parts.push(`разгон и спад нагрузки — ${data.excluded_edges}`);
    if (data.excluded_stalls) parts.push(`провалы — ${data.excluded_stalls}`);
    const after = (data.points || []).filter((point) => point.role === 'after_cutoff').length;
    if (data.cutoff_kind === 'rps_drop' && after) parts.push(`после падения RPS — ${after}`);
    return parts.length ? `Не вошли в модель: ${parts.join(', ')}.` : 'В модель вошли все точки окна.';
  }
  function kappaText(fit) {
    const kappa = Number(fit.kappa) || 0;
    if (kappa <= 0) return 'κ = 0 — спада RPS после пика модель не видит.';
    const peak = fit.n_peak != null ? ` Пик — при параллелизме ≈ ${formatNumber(threeSignificant(fit.n_peak))}.` : '';
    return `κ = ${kappa.toExponential(2).replace('.', ',')} — потери на согласование параллельных запросов: из-за κ после пика RPS падает.${peak}`;
  }
  function methodItems(data) {
    const fit = data.fit || {};
    const end = data.cutoff_kind === 'rps_drop' ? 'первое падение RPS' : 'конец теста';
    const count = `${data.samples_used} ${plural(data.samples_used, 'точке', 'точкам', 'точкам')}`;
    return [
      'Поды: CPU пода = база + стоимость × RPS на под — прямая по стабильным ступеням для каждого сервиса. Нужно подов = стоимость × цель / (потолок − база). Нагрузка делится между подами поровну, смесь запросов — как в тесте.',
      ...uslItems(data),
      `Модель USL (универсальный закон масштабируемости) по ${count} по ${formatNumber(data.bin_minutes, 1)} мин: с ${hhmm(data.window_start_iso)} до ${hhmm(data.cutoff_iso)} (${end}).`,
      exclusionText(data),
      data.load_model === 'closed'
        ? `Модель нагрузки закрытая: параллелизм — число VU, паузы в сценарии ≈ ${formatNumber(data.think_time_s, 2)} с.`
        : 'Модель нагрузки открытая: параллелизм — RPS × среднее время отклика.',
      `σ = ${formatPct((Number(fit.sigma) || 0) * 100)} — потери на конкуренцию за общий ресурс: чем больше σ, тем раньше замедляется рост RPS.`,
      kappaText(fit),
      `R² = ${formatNumber(fit.r2, 2)} — доля разброса RPS, которую объясняет модель.`
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
    return limit.kind === 'instability' ? 'Предел: провалы' : 'Предел: потолок модели';
  }

  function sampleTip(point) {
    const lines = [`${hhmm(point.time_iso)} · ${ROLE_TITLES[point.role]}`, formatRps(point.rps), `среднее ${formatMs(point.response_ms)}`];
    if (point.p95_ms != null) lines.push(`p95 ${formatMs(point.p95_ms)}`);
    lines.push(`параллелизм ${formatNumber(point.concurrency, 1)}`);
    if (point.vus != null) lines.push(`VU ${formatNumber(point.vus)}`);
    return lines;
  }
  function samplePoint(x, y, point) {
    return { x: x, y: y, tip: sampleTip(point) };
  }
  function stepPoint(x, y, step) {
    const lines = [step.label, `${formatRps(step.rps)} в среднем`, `среднее ${formatMs(step.response_ms)}`];
    if (step.p95_ms != null) lines.push(`p95 худшей точки ${formatMs(step.p95_ms)}`);
    lines.push(step.stalls ? `провалы: ${step.stalls} из ${step.samples}` : 'без провалов');
    return { x: x, y: y, number: step.number, tip: lines };
  }
  function modelPoint(x, y, point) {
    const lines = [`Модель: ${formatRps(point.rps)}`, `среднее ${formatMs(point.response_ms)}`];
    if (point.p95_ms != null) lines.push(`p95 ${formatMs(point.p95_ms)}`);
    lines.push(`параллелизм ${formatNumber(point.concurrency, 1)}`);
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
    return Object.assign(pointSet('Разгон и спад нагрузки', data, 'transparent', 3.5, 5), { borderColor: COLORS.edge, borderWidth: 1.5 });
  }
  function kneeSet(knee) {
    const tip = [`Колено: ${formatRps(knee.rps)}`, `среднее ${formatMs(knee.response_ms)}`];
    return Object.assign(pointSet('Колено', [{ x: knee.rps, y: knee.response_ms, tip: tip }], COLORS.knee, 7, 0), { pointStyle: 'rectRot' });
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
    if (!hasP95) return `Порог SLA p95 (${formatMs(sla)}) не показан: в данных прогноза нет p95.`;
    return `Порог SLA p95 (${formatMs(sla)}) выше графика.`;
  }
  function latencyReferences(data, bounds) {
    const sets = [];
    if (bounds.sla != null) sets.push(lineSet('SLA p95', horizontalLine(bounds.sla, bounds.xMax, [`SLA p95: ${formatMs(bounds.sla)}`]), COLORS.sla, true, 3));
    if (data.target) {
      const tip = [`Цель: ${formatRps(data.target.rps)}`];
      sets.push(lineSet('Цель', verticalLine(data.target.rps, bounds.yMax, tip), ANSWER[data.answer.verdict].color, true, 3));
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
    return data.tested_rps ? [{ from: data.tested_rps, to: xMax, color: accentZone() }] : [];
  }
  function latencyCaption(data, datasets, bounds) {
    const parts = [data.instability
      ? 'Красная заливка — нагрузка, с которой в тесте начались провалы: модель там не строится.'
      : 'Заливка — нагрузка выше проверенной: там только прогноз.'];
    const hidden = clippedCount(datasets);
    if (hidden) parts.push(`Значения выше графика (${hidden}) — треугольники у верхнего края, точные числа в подсказке.`);
    if (bounds.slaNote) parts.push(bounds.slaNote);
    if (data.target && data.target.rps > bounds.xMax) parts.push(`Цель ${formatRps(data.target.rps)} правее графика.`);
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
      pointSet(`Точки по ${formatNumber(data.bin_minutes, 1)} мин`, byRole(points, 'fit').map((p) => clip(samplePoint(p.rps, p.response_ms, p))), COLORS.sample, 2.5, 6),
      pointSet('Провалы', byRole(points, 'stall').map((p) => clip(samplePoint(p.rps, p.response_ms, p))), COLORS.stall, 3.5, 5),
      stepSet(`${words.title}: среднее`, steps.map((step) => clip(stepPoint(step.rps, step.response_ms, step)))),
      stepP95Set(`${words.title}: p95`, steps.filter((step) => step.p95_ms != null).map((step) => clip(stepPoint(step.rps, step.p95_ms, step)))),
      lineSet('Модель: среднее', curve.map((point) => modelPoint(point.rps, point.response_ms, point)), accentColor(), false, 4),
      lineSet('Модель: p95', curve.filter((point) => point.p95_ms != null).map((point) => modelPoint(point.rps, point.p95_ms, point)), COLORS.p95, true, 4)
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
      pointSet('Точки модели', byRole(points, 'fit').map(sample), COLORS.sample, 2.5, 6),
      pointSet('Провалы', byRole(points, 'stall').map(sample), COLORS.stall, 3.5, 5),
      edgeSet(byRole(points, 'edge').map(sample)),
      lineSet('Модель', curve.map((point) => modelPoint(point.concurrency, point.rps, point)), accentColor(), false, 4)
    ];
    if (target) datasets.push(lineSet('Цель', horizontalLine(target.rps, xMax, [`Цель: ${formatRps(target.rps)}`]), ANSWER[data.answer.verdict].color, true, 3));
    if (data.limit.rps != null) {
      const tip = [`${limitTitle(data.limit)}: ${formatApproxRps(data.limit.rps)}`];
      datasets.push(lineSet(limitTitle(data.limit), horizontalLine(data.limit.rps, xMax, tip), COLORS.limit, false, 3));
    }
    const hidden = clippedCount(datasets);
    const caption = hidden
      ? `Точки за краем графика (${hidden}) — треугольники у края: в провалах и на разгоне параллелизм по закону Литтла резко растёт.`
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
        locale: LOCALE,
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
    charts.push(createChart('forecastLatency', latency, 'RPS', 'Время отклика, мс'));
    setText('forecastLatencyCaption', latency.caption);
    const throughput = throughputView(data);
    charts.push(createChart('forecastThroughput', throughput, data.load_model === 'closed' ? 'VU' : 'Параллелизм, запросов в системе', 'RPS'));
    setText('forecastThroughputCaption', throughput.caption);
  }

  function presetBase(data) {
    const sla = Number(data.sla_target_rps);
    if (Number.isFinite(sla) && sla > 0) return { rps: sla, suffix: 'к SLA' };
    return { rps: Number(data.tested_rps) || 0, suffix: 'к проверенному' };
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
    const base = presetBase(data);
    const presets = [
      data.sla_target_rps ? '<button type="button" class="btn" data-preset="sla">Цель SLA</button>' : '',
      `<button type="button" class="btn" data-preset="plus25">+25 % ${base.suffix}</button>`,
      `<button type="button" class="btn" data-preset="plus50">+50 % ${base.suffix}</button>`,
      '<button type="button" class="btn" data-preset="safe" id="forecastPresetSafe">Безопасный максимум</button>'
    ].join('');
    return [
      '<div class="forecast-controls">',
      `<label>Целевая нагрузка, RPS<input id="forecastTarget" type="number" min="1" step="1" value="${target == null ? '' : escapeHtml(target)}" /></label>`,
      `<label title="Загрузка CPU одного пода, выше которой нужен ещё под">Потолок CPU пода, %<input id="forecastCpuCeiling" type="number" min="10" max="100" step="1" value="${escapeHtml(Math.round(data.resources.ceiling_pct))}" /></label>`,
      `<label title="Какую долю предела оставить в резерве">Запас до предела, %<input id="forecastHeadroom" type="number" min="0" max="90" step="1" value="${DEFAULT_HEADROOM_PCT}" /></label>`,
      '<div class="forecast-presets" role="group" aria-labelledby="forecastPresetsCaption">',
      '<span class="forecast-presets-caption" id="forecastPresetsCaption">Быстрый выбор цели</span>',
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
      '<h4>Где цель на шкале нагрузки</h4>',
      '<div id="forecastScale"></div>',
      '<p class="dim" id="forecastScaleCaption"></p>',
      '<h4 class="forecast-overview-steps">Как прошли ступени теста</h4>',
      '<div id="forecastSteps"></div>',
      '</section>',
      '<section class="forecast-block forecast-resources" id="forecastResources"></section>',
      '<section class="forecast-block forecast-explain" id="forecastExplain" hidden></section>',
      '<div class="forecast-blocks">',
      '<section class="forecast-block"><h4>Что было в тесте</h4><div id="forecastFacts"></div></section>',
      '<section class="forecast-block" id="forecastConfidence"></section>',
      '</div>',
      '<details class="report-collapsible forecast-details">',
      '<summary>Подробный график — для инженеров</summary>',
      '<div class="report-collapsible-body">',
      '<figure class="forecast-figure">',
      '<figcaption class="forecast-chart-title">Время отклика от нагрузки</figcaption>',
      '<div class="forecast-chart forecast-chart-main"><canvas id="forecastLatency"></canvas></div>',
      '<p class="dim" id="forecastLatencyCaption"></p>',
      '</figure>',
      '</div>',
      '</details>',
      '<details class="report-collapsible forecast-details">',
      '<summary>Как считается</summary>',
      '<div class="report-collapsible-body">',
      '<div id="forecastMethod"></div>',
      '<figure class="forecast-figure">',
      '<figcaption class="forecast-chart-title">Пропускная способность от параллелизма</figcaption>',
      '<div class="forecast-chart forecast-chart-secondary"><canvas id="forecastThroughput"></canvas></div>',
      '<p class="dim" id="forecastThroughputCaption"></p>',
      '</figure>',
      '</div>',
      '</details>',
      '</div>'
    ].join('');
  }
  function unavailable(message) {
    return `<div class="report-note"><div class="report-note-title">Прогноз недоступен</div><div class="report-note-body"><p>${escapeHtml(message)}</p></div></div>`;
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
    const title = finding ? findingTitle(finding) : `${link.finding_id} — нет в отчёте`;
    return `<li><span class="forecast-finding-meta">${escapeHtml(title)}</span>${escapeHtml(link.role)}</li>`;
  }
  function checkText(check) {
    const parts = [];
    if (check.status === 'verified') parts.push(`числа сверены с данными: ${check.claims_matched} из ${check.claims_total}`);
    else if (check.status === 'unverified') parts.push(`есть числа, которых нет в данных: ${check.unmatched.join(', ')}`);
    else parts.push('чисел в ответе нет');
    if (check.unknown_findings.length) parts.push(`ссылки на несуществующие находки: ${check.unknown_findings.join(', ')}`);
    return parts.join('; ');
  }
  function cappedList(items, tag) {
    const shown = items.slice(0, EXPLAIN_ITEMS).map((item) => `<li>${escapeHtml(item)}</li>`).join('');
    const rest = items.slice(EXPLAIN_ITEMS);
    const more = rest.length
      ? `<details class="forecast-more"><summary>Ещё ${rest.length}</summary><${tag} start="${EXPLAIN_ITEMS + 1}">${rest.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</${tag}></details>`
      : '';
    return items.length ? `<${tag}>${shown}</${tag}>${more}` : '<p class="dim">Нет.</p>';
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
    const generated = view.generated_at ? new Date(view.generated_at).toLocaleString(LOCALE, { dateStyle: 'short', timeStyle: 'short' }) : '';
    const meta = [view.model, generated, checkText(view.check)].filter(Boolean).join(' · ');
    const scaling = SCALING[result.scaling.verdict];
    const evidence = result.evidence.length
      ? `<details class="forecast-findings"><summary>На чём основано (${result.evidence.length})</summary><ul>${result.evidence.map((link) => evidenceItem(link, byRef)).join('')}</ul></details>`
      : '';
    return [
      `<p class="forecast-explain-headline">${escapeHtml(result.headline)}</p>`,
      `<p class="forecast-explain-cause">${escapeHtml(result.cause)}</p>`,
      keyFactsHtml(result.key_facts),
      `<div class="forecast-explain-scaling is-${scaling.tone}"><strong>${scaling.text}.</strong> ${escapeHtml(result.scaling.reason)}</div>`,
      '<div class="forecast-explain-columns">',
      `<section><h5>Риски</h5>${cappedList(result.scaling_risks, 'ul')}</section>`,
      `<section><h5>Что сделать</h5>${cappedList(result.next_checks, 'ol')}</section>`,
      '</div>',
      evidence,
      `<p class="dim forecast-explain-meta">${escapeHtml(meta)}</p>`
    ].join('');
  }
  function explanationHtml(view) {
    const done = view.explanation != null;
    const button = view.llm_problem
      ? `<span class="dim">${escapeHtml(view.llm_problem)}</span>`
      : `<button type="button" class="btn btn-sm" id="forecastExplainRun">${done ? 'Обновить разбор' : 'Разобрать с ИИ'}</button>`;
    const confidence = done ? CONFIDENCE[view.explanation.confidence] : null;
    const pill = confidence ? `<span class="pill ${confidence.pill}">Уверенность модели: ${confidence.text}</span>` : '';
    const action = `<span class="forecast-explain-tools">${pill}${button}</span>`;
    const intro = '<p class="dim">ИИ свяжет предел с находками отчёта ниже и оценит риски масштабирования. Числа ответа сверяются с данными.</p>';
    const findings = view.findings.length
      ? `<details class="forecast-findings"${done ? '' : ' open'}><summary>Связанные находки отчёта (${view.findings.length})</summary><ul>${view.findings.map(findingItem).join('')}</ul></details>`
      : '<p class="dim">Находок отчёта рядом с пределом нет.</p>';
    return [
      `<div class="forecast-explain-head"><h4>Почему упёрлось</h4>${action}</div>`,
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
    if (!resp.ok) return { error: body.error || `Разбор недоступен (HTTP ${resp.status})` };
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
    block.innerHTML = `<div class="forecast-explain-head"><h4>Почему упёрлось</h4></div><p class="forecast-explain-error">${escapeHtml(result.error)}</p>`;
  }
  async function runExplanation() {
    const button = document.getElementById('forecastExplainRun');
    button.disabled = true;
    button.textContent = 'Разбираю… до минуты';
    setHtml('forecastExplainError', '');
    const result = await explanationRequest('POST');
    if (!result.error) {
      renderExplanation(result.view);
      return;
    }
    button.disabled = false;
    button.textContent = 'Повторить разбор';
    setHtml('forecastExplainError', `<p class="forecast-explain-error">${escapeHtml(result.error)}</p>`);
  }

  function percentOf(value, end) {
    return Math.max(0, Math.min(100, (Number(value) / end) * 100));
  }
  function zoneText(data, zone) {
    if (!data.limit || data.limit.rps == null) return 'без провалов';
    if (zone.tone === 'fail') return data.limit.kind === 'instability' ? 'провалы в тесте' : 'выше потолка модели';
    return zone.tone === 'warn' ? 'без запаса' : 'с запасом';
  }
  function markText(data, mark) {
    if (mark.kind === 'safe') return { value: formatRps(Math.floor(mark.rps)), label: 'безопасный максимум' };
    if (mark.kind === 'limit') return { value: formatApproxRps(mark.rps), label: data.limit.kind === 'instability' ? 'начались провалы' : 'потолок модели' };
    return { value: formatRps(Math.round(mark.rps)), label: 'проверено в тесте' };
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
    return `<div class="forecast-scale-target is-${tone}${edgeClass(pct)}" style="left:${pct}%"><span>Цель ${escapeHtml(formatRps(data.target.rps))}${beyond}</span></div>`;
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
    const untested = tested < 100 ? `<div class="forecast-scale-untested" style="left:${tested}%;width:${100 - tested}%" title="Выше проверенной нагрузки — только расчёт"></div>` : '';
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
    if (data.limit.rps == null) return 'Зелёная полоса — нагрузка, которую тест прошёл без провалов; штриховка — выше проверенной нагрузки, там только расчёт.';
    const over = data.limit.kind === 'instability' ? 'красная — с этой нагрузки в тесте начались провалы' : 'красная — выше потолка модели';
    return `Зелёная зона — нагрузка с запасом, жёлтая — без запаса, ${over}. Штриховка — выше проверенной нагрузки: там только расчёт.`;
  }
  function stepsHtml(data) {
    const steps = data.steps || [];
    if (!steps.length) return '<p class="dim">Ступеней в данных нет.</p>';
    const words = stepWords(data);
    const cards = steps.map((step) => {
      const tone = step.after_drop ? 'muted' : (step.stalls ? 'fail' : 'ok');
      const state = { muted: '· после падения RPS', fail: `✗ провалы: ${step.stalls} из ${step.samples}`, ok: '✓ без провалов' }[tone];
      return [
        `<div class="forecast-step is-${tone}" title="${escapeHtml(step.label)}">`,
        `<span class="forecast-step-name">${escapeHtml(words.one)} ${step.number}</span>`,
        `<span class="forecast-step-rps">${escapeHtml(formatRps(Math.round(step.rps)))}</span>`,
        `<span class="forecast-step-time">среднее ${escapeHtml(formatMs(step.response_ms))}</span>`,
        `<span class="forecast-step-state">${escapeHtml(state)}</span>`,
        '</div>'
      ].join('');
    }).join('');
    return `<div class="forecast-steps">${cards}</div>`;
  }

  function paintForecast(data, headroom) {
    current = data;
    setHtml('forecastVerdict', answerHtml(data, headroom));
    setHtml('forecastScale', scaleHtml(data));
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
    headroomInput.title = data.limit_applies ? '' : 'Предел объясняется CPU сервиса: запас задаёт потолок CPU пода';
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
    document.getElementById('forecastConfluence').textContent = url ? 'Обновить отчёт в Confluence' : 'Добавить отчёт в Confluence';
  }
  async function loadPublication() {
    const resp = await fetch('/forecast/' + encodeURIComponent(runId) + '/confluence');
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      setText('forecastConfluenceStatus', body.error || `Не удалось узнать публикацию в Confluence (HTTP ${resp.status})`);
      return;
    }
    setPublication(body.page_url || '');
  }
  function jobFailure(job) {
    const headline = job.message || 'Ошибка публикации';
    return job.error ? `${headline}: ${job.error}` : headline;
  }
  async function pollPublication(jobId) {
    for (let attempt = 0; attempt < PUBLISH_POLLS; attempt += 1) {
      const resp = await fetch('/job_status/' + encodeURIComponent(jobId));
      const job = await resp.json().catch(() => ({}));
      if (job.status === 'done') {
        setPublication(job.page_url || '');
        setText('forecastConfluenceStatus', 'Опубликовано');
        LL.ui.toast('Прогноз опубликован в Confluence', { tone: 'ok' });
        return;
      }
      if (!resp.ok || job.status === 'error' || job.status === 'not_found') {
        setText('forecastConfluenceStatus', jobFailure(job));
        LL.ui.toast(jobFailure(job), { tone: 'error' });
        return;
      }
      const pct = Number(job.progress || 0);
      setText('forecastConfluenceStatus', `${job.message || 'Публикация…'}${pct ? ` (${pct}%)` : ''}`);
      await new Promise((resolve) => setTimeout(resolve, PUBLISH_POLL_MS));
    }
    setText('forecastConfluenceStatus', 'Публикация занимает слишком долго, проверьте статус позже');
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
    setText('forecastConfluenceStatus', 'Публикация…');
    try {
      const resp = await fetch('/forecast/' + encodeURIComponent(runId) + '/confluence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok || !body.job_id) {
        setText('forecastConfluenceStatus', body.error || `Не удалось начать публикацию (HTTP ${resp.status})`);
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
      document.getElementById('forecastTitle').textContent = 'Отчёт не найден';
      box.innerHTML = unavailable('Отчёт не найден');
      return;
    }
    const ref = await refResp.json();
    const name = ref.run_name || 'Отчёт';
    document.getElementById('forecastTitle').textContent = name;
    document.title = `Прогноз: ${name}`;
    const crumbs = document.getElementById('forecastBreadcrumbs');
    crumbs.innerHTML = `<a href="/forecasting">Прогнозирование</a><span class="sep">›</span><span class="current">${escapeHtml(name)}</span>`;
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
    loadExplanation();
    enablePublishing();
  }
  document.addEventListener('DOMContentLoaded', start);
})();
