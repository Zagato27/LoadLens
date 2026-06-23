// Reports page logic
(function () {
  // Chart background plugin
  const cmpBgPlugin = { id: 'cmpBg', beforeDraw(chart, args, opts) { const { ctx, chartArea } = chart; if (!chartArea) return; ctx.save(); ctx.fillStyle = (opts && opts.color) || '#151515'; ctx.fillRect(chartArea.left, chartArea.top, chartArea.right - chartArea.left, chartArea.bottom - chartArea.top); ctx.restore(); } };
  if (window.Chart && Chart.register) Chart.register(cmpBgPlugin);

  function randColor(alpha = 0.7) {
    if (window.LoadLens && typeof window.LoadLens.randColor === 'function') {
      return window.LoadLens.randColor(alpha);
    }
    const r = Math.floor(100 + Math.random() * 155);
    const g = Math.floor(100 + Math.random() * 155);
    const b = Math.floor(100 + Math.random() * 155);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  function getRunFromPath() {
    try {
      const parts = location.pathname.split('/').filter(Boolean);
      const idx = parts.indexOf('reports');
      if (idx >= 0) {
        if (parts[idx + 2]) return decodeURIComponent(parts[idx + 2]);
        if (parts[idx + 1]) return decodeURIComponent(parts[idx + 1]);
      }
    } catch (e) {}
    return null;
  }

  function renamedReportPath(newRunName) {
    try {
      const parts = location.pathname.split('/').filter(Boolean);
      const idx = parts.indexOf('reports');
      if (idx >= 0 && parts[idx + 2]) {
        parts[idx + 2] = encodeURIComponent(newRunName);
        return '/' + parts.join('/');
      }
      if (idx >= 0 && parts[idx + 1]) {
        parts[idx + 1] = encodeURIComponent(newRunName);
        return '/' + parts.join('/');
      }
    } catch (e) {}
    return '/reports/' + encodeURIComponent(newRunName);
  }

  function updateReportTitle(run) {
    const title = document.getElementById('pageTitle');
    if (title) title.textContent = run ? `Отчет по тесту ${run}` : 'Отчет по тесту';
    try { document.title = run ? `Отчёт ${run}` : 'Отчёт'; } catch (e) {}
  }

  function wireRenameReport() {
    const btn = document.getElementById('renameRunBtn');
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const run = getRunFromPath();
      if (!run) return;
      const nextName = prompt('Новое название отчёта:', run);
      if (nextName === null) return;
      const trimmed = String(nextName || '').trim();
      if (!trimmed || trimmed === run) return;
      try {
        const resp = await fetch('/runs/' + encodeURIComponent(run), {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ new_run_name: trimmed })
        });
        const j = await resp.json();
        if (resp.ok) {
          location.href = renamedReportPath(trimmed);
        } else {
          alert(j.error || 'Ошибка переименования');
        }
      } catch (e) {
        alert('Ошибка переименования');
      }
    });
  }

  async function reportsLoadSchema() {
    try {
      const r = await fetch('/domains_schema');
      return await r.json();
    } catch (e) {
      return {};
    }
  }

  function safeStr(x) { return (x === undefined || x === null) ? '' : String(x).trim(); }
  function pct(x) { try { if (x === undefined || x === null) return '—'; const v = Number(x); if (!isFinite(v)) return '—'; return `${Math.round(v * 100)}%`; } catch (e) { return '—'; } }
  function parseSystemContextValue(value) {
    if (!value) return null;
    if (typeof value === 'string') {
      try { return JSON.parse(value); } catch (e) { return null; }
    }
    return (typeof value === 'object') ? value : null;
  }
  function hasMeaningfulSystemContext(value) {
    if (typeof value === 'string') return Boolean(value.trim());
    if (Array.isArray(value)) return value.some((item) => hasMeaningfulSystemContext(item));
    if (value && typeof value === 'object') {
      if (value.enabled === false) return false;
      return Object.keys(value).some((key) => !['schema_version', 'enabled'].includes(key) && hasMeaningfulSystemContext(value[key]));
    }
    return false;
  }
  function systemContextMarkdown(ctx) {
    if (!ctx || typeof ctx !== 'object' || !hasMeaningfulSystemContext(ctx)) return '';
    const lines = [];
    const system = (ctx.system && typeof ctx.system === 'object') ? ctx.system : {};
    const architecture = (ctx.architecture && typeof ctx.architecture === 'object') ? ctx.architecture : {};
    const loadModel = (ctx.load_model && typeof ctx.load_model === 'object') ? ctx.load_model : {};
    const operational = (ctx.operational_context && typeof ctx.operational_context === 'object') ? ctx.operational_context : {};
    if (safeStr(system.name)) lines.push(`- Система: ${safeStr(system.name)}`);
    if (safeStr(system.domain)) lines.push(`- Домен: ${safeStr(system.domain)}`);
    if (safeStr(system.description)) lines.push(`- Описание: ${safeStr(system.description)}`);
    if (safeStr(system.test_goal)) lines.push(`- Цель теста: ${safeStr(system.test_goal)}`);
    if (safeStr(architecture.style)) lines.push(`- Архитектура: ${safeStr(architecture.style)}`);

    const components = Array.isArray(architecture.components) ? architecture.components : [];
    if (components.length) {
      lines.push('');
      lines.push('#### Компоненты');
      components.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const name = safeStr(item.name || item.id);
        const role = safeStr(item.role);
        const criticality = safeStr(item.criticality);
        const technologies = Array.isArray(item.technologies) ? item.technologies.map((x) => safeStr(x)).filter(Boolean) : [];
        const meta = [];
        if (role) meta.push(`роль: ${role}`);
        if (criticality) meta.push(`criticality: ${criticality}`);
        if (technologies.length) meta.push(`tech: ${technologies.join(', ')}`);
        if (name) lines.push(meta.length ? `- ${name} (${meta.join('; ')})` : `- ${name}`);
      });
    }

    const dependencies = Array.isArray(architecture.dependencies) ? architecture.dependencies : [];
    if (dependencies.length) {
      lines.push('');
      lines.push('#### Зависимости');
      dependencies.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const from = safeStr(item.from);
        const to = safeStr(item.to);
        const kind = safeStr(item.kind);
        const purpose = safeStr(item.purpose);
        const meta = [];
        if (kind) meta.push(`type: ${kind}`);
        if (purpose) meta.push(`purpose: ${purpose}`);
        if (from || to) lines.push(meta.length ? `- ${from} -> ${to} (${meta.join('; ')})` : `- ${from} -> ${to}`);
      });
    }

    const dataStores = Array.isArray(architecture.data_stores) ? architecture.data_stores : [];
    if (dataStores.length) {
      lines.push('');
      lines.push('#### Хранилища');
      dataStores.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const name = safeStr(item.id);
        const type = safeStr(item.type);
        const usedBy = Array.isArray(item.used_by) ? item.used_by.map((x) => safeStr(x)).filter(Boolean) : [];
        const purpose = safeStr(item.purpose);
        const meta = [];
        if (type) meta.push(`type: ${type}`);
        if (usedBy.length) meta.push(`used_by: ${usedBy.join(', ')}`);
        if (purpose) meta.push(`purpose: ${purpose}`);
        if (name) lines.push(meta.length ? `- ${name} (${meta.join('; ')})` : `- ${name}`);
      });
    }

    const flows = Array.isArray(loadModel.critical_user_flows) ? loadModel.critical_user_flows : [];
    if (flows.length) {
      lines.push('');
      lines.push('#### Критичные потоки');
      flows.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const name = safeStr(item.name || item.id);
        const steps = Array.isArray(item.steps) ? item.steps.map((x) => safeStr(x)).filter(Boolean) : [];
        const successSignals = Array.isArray(item.success_signals) ? item.success_signals.map((x) => safeStr(x)).filter(Boolean) : [];
        if (name) lines.push(`- ${name}`);
        if (steps.length) lines.push(`  путь: ${steps.join(' -> ')}`);
        if (successSignals.length) lines.push(`  сигналы успеха: ${successSignals.join(', ')}`);
      });
    }

    const entrypoints = Array.isArray(loadModel.entrypoints) ? loadModel.entrypoints : [];
    if (entrypoints.length) {
      lines.push('');
      lines.push('#### Точки входа нагрузки');
      entrypoints.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const name = safeStr(item.name || item.id);
        const kind = safeStr(item.kind);
        const priority = safeStr(item.business_priority);
        const meta = [];
        if (kind) meta.push(`type: ${kind}`);
        if (priority) meta.push(`priority: ${priority}`);
        if (name) lines.push(meta.length ? `- ${name} (${meta.join('; ')})` : `- ${name}`);
      });
    }

    const hotspots = Array.isArray(loadModel.expected_hotspots) ? loadModel.expected_hotspots.map((x) => safeStr(x)).filter(Boolean) : [];
    if (hotspots.length) {
      lines.push('');
      lines.push('#### Ожидаемые hotspots');
      hotspots.forEach((item) => lines.push(`- ${item}`));
    }

    const focus = Array.isArray(operational.analysis_focus) ? operational.analysis_focus.map((x) => safeStr(x)).filter(Boolean) : [];
    if (focus.length) {
      lines.push('');
      lines.push('#### Фокус анализа');
      focus.forEach((item) => lines.push(`- ${item}`));
    }

    const risks = Array.isArray(operational.known_risks) ? operational.known_risks.map((x) => safeStr(x)).filter(Boolean) : [];
    if (risks.length) {
      lines.push('');
      lines.push('#### Известные риски');
      risks.forEach((item) => lines.push(`- ${item}`));
    }

    const constraints = Array.isArray(operational.known_constraints) ? operational.known_constraints.map((x) => safeStr(x)).filter(Boolean) : [];
    if (constraints.length) {
      lines.push('');
      lines.push('#### Ограничения');
      constraints.forEach((item) => lines.push(`- ${item}`));
    }
    return lines.join('\n');
  }
  function escapeHtml(value) {
    return String(value || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
  function standardizeVerdict(vRaw) {
    const v = (safeStr(vRaw) || '').toLowerCase();
    if (!v) return 'Недостаточно данных';
    const ok = ['ok', 'okay', 'успех', 'успешно', 'success', 'passed', 'green'];
    const warn = ['warn', 'warning', 'есть риски', 'риски', 'risk', 'risks', 'degrad', 'degraded', 'предупреждение'];
    const crit = ['critical', 'критично', 'fail', 'failed', 'ошибка', 'error', 'red', 'провал'];
    const na = ['insufficient', 'нет данных', 'недостаточно', 'no data', 'unknown', 'n/a'];
    if (ok.some((x) => v.includes(x))) return 'Успешно';
    if (warn.some((x) => v.includes(x))) return 'Есть риски';
    if (crit.some((x) => v.includes(x))) return 'Провал';
    if (na.some((x) => v.includes(x))) return 'Недостаточно данных';
    return 'Недостаточно данных';
  }
  function markdownToSafeHtml(md) {
    const source = safeStr(md);
    if (!source) return '';
    let html = (window.marked && typeof marked.parse === 'function')
      ? window.marked.parse(source)
      : escapeHtml(source).replace(/\n/g, '<br>');
    try { if (window.DOMPurify) html = window.DOMPurify.sanitize(html); } catch (e) {}
    return html;
  }
  function verdictTone(verdict) {
    const text = standardizeVerdict(verdict);
    if (text === 'Успешно') return { text, className: 'report-verdict-success' };
    if (text === 'Есть риски') return { text, className: 'report-verdict-risk' };
    if (text === 'Провал') return { text, className: 'report-verdict-fail' };
    return { text, className: 'report-verdict-na' };
  }
  function renderListCell(items, emptyText, formatter) {
    const values = (Array.isArray(items) ? items : [])
      .map((item) => {
        try { return formatter ? formatter(item) : safeStr(item); } catch (e) { return ''; }
      })
      .map((item) => safeStr(item))
      .filter(Boolean);
    if (!values.length) return `<div class="report-empty">${escapeHtml(emptyText)}</div>`;
    return `<ul class="report-list">${values.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`;
  }
  function renderKeyValueTable(title, rows) {
    const extraClass = (rows && rows._tableClass) ? ` ${rows._tableClass}` : '';
    const body = (Array.isArray(rows) ? rows : []).map((row) => [
      '<tr>',
      `<th scope="row">${escapeHtml(row.label || '')}</th>`,
      `<td>${row.html || `<span class="report-empty">${escapeHtml('—')}</span>`}</td>`,
      '</tr>'
    ].join('')).join('');
    return [
      '<section>',
      title ? `<div class="report-block-title">${escapeHtml(title)}</div>` : '',
      '<div class="report-table-shell">',
      `<table class="report-table${extraClass}"><tbody>`,
      body,
      '</tbody></table>',
      '</div>',
      '</section>'
    ].join('');
  }
  function renderProblemRecommendationRows(rows) {
    const body = rows.map((row) => [
      '<tr>',
      `<td>${row.problemHtml || `<span class="report-empty">${escapeHtml('—')}</span>`}</td>`,
      `<td>${row.recommendationHtml || `<span class="report-empty">${escapeHtml('—')}</span>`}</td>`,
      '</tr>'
    ].join('')).join('');
    return [
      '<section>',
      '<div class="report-block-title">Проблемы и рекомендации</div>',
      '<div class="report-table-shell">',
      '<table class="report-table report-two-col-table">',
      '<thead><tr>',
      '<th>Проблемы</th>',
      '<th>Рекомендации по устранению</th>',
      '</tr></thead>',
      `<tbody>${body}</tbody>`,
      '</table>',
      '</div>',
      '</section>'
    ].join('');
  }
  function renderTwoColumnTable(title, leftTitle, rightTitle, leftHtml, rightHtml) {
    return [
      '<section>',
      title ? `<div class="report-block-title">${escapeHtml(title)}</div>` : '',
      '<div class="report-table-shell">',
      '<table class="report-table report-two-col-table">',
      '<thead><tr>',
      `<th>${escapeHtml(leftTitle)}</th>`,
      `<th>${escapeHtml(rightTitle)}</th>`,
      '</tr></thead>',
      `<tbody><tr><td>${leftHtml}</td><td>${rightHtml}</td></tr></tbody>`,
      '</table>',
      '</div>',
      '</section>'
    ].join('');
  }
  function renderRationaleBox(text) {
    const contentHtml = markdownToSafeHtml(text) || '<p class="report-empty">Нет пояснения.</p>';
    return [
      '<section class="report-note">',
      '<div class="report-note-title">Обоснование вердикта</div>',
      `<div class="report-note-body report-markdown-block">${contentHtml}</div>`,
      '</section>'
    ].join('');
  }
  function normalizeEvidenceItems(value) {
    const raw = Array.isArray(value) ? value : (value ? [value] : []);
    return raw.map((item) => {
      if (item && typeof item === 'object') {
        return {
          metric: safeStr(item.metric || item.name || item.label),
          observedValue: safeStr(item.observed_value || item.value || item.actual),
          threshold: safeStr(item.threshold || item.limit || item.baseline),
          note: safeStr(item.note || item.details || item.evidence)
        };
      }
      return { metric: '', observedValue: '', threshold: '', note: safeStr(item) };
    }).filter((item) => item.metric || item.observedValue || item.threshold || item.note);
  }
  function renderMetaGrid(entries) {
    const items = (Array.isArray(entries) ? entries : [])
      .map((entry) => ({
        label: safeStr(entry && entry.label),
        value: safeStr(entry && entry.value)
      }))
      .filter((entry) => entry.label && entry.value)
      .map((entry) => [
        '<div class="report-item-meta-entry">',
        `<span class="report-item-meta-label">${escapeHtml(entry.label)}</span>`,
        `<span class="report-item-meta-value">${escapeHtml(entry.value)}</span>`,
        '</div>'
      ].join(''));
    if (!items.length) return '';
    return `<div class="report-item-meta-grid">${items.join('')}</div>`;
  }
  function renderFindingEvidence(item) {
    if (!item || typeof item !== 'object') return '';
    const evidenceSummary = safeStr(item.evidence_summary || item.evidence);
    const evidenceItems = normalizeEvidenceItems(item.evidence_items || item.evidence_list || item.evidence_rows);
    if (!evidenceSummary && !evidenceItems.length) return '';
    const lines = [];
    if (evidenceSummary) {
      lines.push(`<div class="report-item-evidence-text">${escapeHtml(evidenceSummary)}</div>`);
    }
    if (evidenceItems.length) {
      lines.push(
        `<ul class="report-item-evidence-list">${evidenceItems.map((evidenceItem) => {
          const bits = [];
          if (evidenceItem.metric) bits.push(evidenceItem.metric);
          if (evidenceItem.observedValue) bits.push(`значение: ${evidenceItem.observedValue}`);
          if (evidenceItem.threshold) bits.push(`порог: ${evidenceItem.threshold}`);
          if (evidenceItem.note) bits.push(evidenceItem.note);
          return `<li>${escapeHtml(bits.join(' | '))}</li>`;
        }).join('')}</ul>`
      );
    }
    return `<div class="report-item-evidence-inline">${lines.join('')}</div>`;
  }
  function renderFindingDetails(item) {
    if (!item || typeof item !== 'object') return escapeHtml(safeStr(item));
    const summary = safeStr(item.summary || item.title || item.text);
    const metaHtml = renderMetaGrid([
      { label: 'Критичность', value: safeStr(item.severity) },
      { label: 'Компонент', value: safeStr(item.component) }
    ]);
    const evidenceHtml = renderFindingEvidence(item);
    return [
      '<div class="report-item-card">',
      `<div class="report-item-summary">${escapeHtml(summary || '—')}</div>`,
      evidenceHtml,
      metaHtml,
      '</div>'
    ].join('');
  }
  function renderActionDetails(item) {
    if (!item || typeof item !== 'object') return escapeHtml(safeStr(item));
    const summary = safeStr(item.summary || item.action || item.text);
    const details = safeStr(item.details || item.description || item.implementation_details);
    const detailsHtml = details
      ? `<div class="report-item-details report-markdown-block">${markdownToSafeHtml(details)}</div>`
      : '';
    return [
      '<div class="report-item-card">',
      `<div class="report-item-summary">${escapeHtml(summary || '—')}</div>`,
      detailsHtml,
      '</div>'
    ].join('');
  }
  function renderActionEntries(items, emptyText) {
    const values = (Array.isArray(items) ? items : []).filter(Boolean);
    if (!values.length) return `<span class="report-empty">${escapeHtml(emptyText)}</span>`;
    return `<div class="report-cell-stack">${values.map((item) => renderActionDetails(item)).join('')}</div>`;
  }
  function normalizeFindingLinkId(value, fallback) {
    const normalized = safeStr(value)
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, '_')
      .replace(/_+/g, '_')
      .replace(/^_+|_+$/g, '');
    return normalized || fallback;
  }
  function normalizeFindingLinkIds(value) {
    const items = Array.isArray(value) ? value : (safeStr(value) ? [value] : []);
    const ids = [];
    items.forEach((item, idx) => {
      const normalized = normalizeFindingLinkId(item, '');
      if (normalized && ids.indexOf(normalized) < 0) ids.push(normalized);
    });
    return ids;
  }
  function normalizeFindingEntry(item, idx) {
    const rawItem = (item && typeof item === 'object')
      ? item
      : { summary: safeStr(item) };
    const summary = safeStr(rawItem.summary || rawItem.title || rawItem.text);
    const components = [];
    const fallbackId = `finding_${idx + 1}`;
    if (rawItem && typeof rawItem === 'object') {
      const component = safeStr(rawItem.component).toLowerCase();
      if (component) components.push(component);
      const affected = Array.isArray(rawItem.affected_components)
        ? rawItem.affected_components.map((x) => safeStr(x).toLowerCase()).filter(Boolean)
        : [];
      affected.forEach((x) => { if (components.indexOf(x) < 0) components.push(x); });
      return {
        idx,
        id: normalizeFindingLinkId(rawItem.id || rawItem.finding_id || rawItem.key, fallbackId),
        summary,
        components,
        item: rawItem
      };
    }
    return { idx, id: fallbackId, summary, components, item: rawItem };
  }
  function normalizeActionEntry(item, idx) {
    const rawItem = (item && typeof item === 'object')
      ? item
      : { summary: safeStr(item) };
    const summary = safeStr(rawItem.summary || rawItem.action || rawItem.text);
    const components = [];
    const forFindingIds = [];
    if (rawItem && typeof rawItem === 'object') {
      const single = safeStr(rawItem.component).toLowerCase();
      if (single) components.push(single);
      const affected = Array.isArray(rawItem.affected_components)
        ? rawItem.affected_components.map((x) => safeStr(x).toLowerCase()).filter(Boolean)
        : [];
      affected.forEach((x) => { if (components.indexOf(x) < 0) components.push(x); });
      normalizeFindingLinkIds(
        rawItem.for_finding_ids || rawItem.for_findings || rawItem.finding_ids || rawItem.related_findings || rawItem.for_finding_id || rawItem.finding_id
      ).forEach((id) => { if (forFindingIds.indexOf(id) < 0) forFindingIds.push(id); });
    }
    return { idx, summary, components, forFindingIds, item: rawItem };
  }
  function matchLegacyActionForFinding(finding, actionEntries, unusedActionIds) {
    const legacyActions = actionEntries.filter((action) => !action.forFindingIds.length && unusedActionIds.has(action.idx));
    let matchedAction = null;
    if (finding.components.length) {
      matchedAction = legacyActions.find((action) => (
        action.components.some((component) => finding.components.indexOf(component) >= 0)
      )) || null;
    }
    if (!matchedAction && unusedActionIds.has(finding.idx)) {
      matchedAction = legacyActions.find((action) => action.idx === finding.idx) || null;
    }
    if (!matchedAction) {
      matchedAction = legacyActions[0] || null;
    }
    if (matchedAction) unusedActionIds.delete(matchedAction.idx);
    return matchedAction;
  }
  function pairFindingsWithActions(findings, actions) {
    const findingEntries = (Array.isArray(findings) ? findings : [])
      .map((item, idx) => normalizeFindingEntry(item, idx))
      .filter((item) => safeStr(item.summary));
    const actionEntries = (Array.isArray(actions) ? actions : [])
      .map((item, idx) => normalizeActionEntry(item, idx))
      .filter((item) => safeStr(item.summary));
    const linkedFindingIds = new Set(findingEntries.map((item) => item.id));
    const unusedLegacyActions = new Set(
      actionEntries
        .filter((item) => !item.forFindingIds.length)
        .map((item) => item.idx)
    );
    const rows = findingEntries.map((finding) => {
      const explicitMatches = actionEntries
        .filter((action) => action.forFindingIds.indexOf(finding.id) >= 0)
        .map((action) => action.item);
      const legacyMatch = explicitMatches.length
        ? null
        : matchLegacyActionForFinding(finding, actionEntries, unusedLegacyActions);
      const recommendationTexts = explicitMatches.length
        ? explicitMatches
        : (legacyMatch ? [legacyMatch.item] : []);
      return {
        problemHtml: renderFindingDetails(finding.item),
        recommendationHtml: renderActionEntries(recommendationTexts, 'Нет рекомендации.')
      };
    });
    actionEntries
      .filter((action) => (
        (!action.forFindingIds.length && unusedLegacyActions.has(action.idx))
        || (action.forFindingIds.length && !action.forFindingIds.some((id) => linkedFindingIds.has(id)))
      ))
      .forEach((action) => {
        rows.push({
          problemHtml: '<span class="report-empty">Дополнительная рекомендация</span>',
          recommendationHtml: renderActionEntries([action.item], 'Нет рекомендации.')
        });
      });
    if (!rows.length) {
      rows.push({
        problemHtml: '<span class="report-empty">Нет существенных проблем.</span>',
        recommendationHtml: renderActionEntries(actionEntries.map((action) => action.item), 'Нет рекомендаций.')
      });
    }
    return rows;
  }
  function renderSystemContextBox(ctx) {
    const box = document.getElementById('systemContextBox');
    if (!box) return;
    if (!ctx || typeof ctx !== 'object' || !hasMeaningfulSystemContext(ctx)) {
      box.style.display = 'none';
      box.innerHTML = '';
      return;
    }
    const html = markdownToSafeHtml(systemContextMarkdown(ctx));
    box.innerHTML = [
      '<details class="report-collapsible">',
      '<summary>',
      '<span>Контекст тестируемой системы</span>',
      '<small class="report-collapsible-subtitle">Снимок на момент генерации отчета</small>',
      '</summary>',
      `<div class="report-collapsible-body report-markdown-block">${html}</div>`,
      '</details>'
    ].join('');
    box.style.display = '';
  }
  function renderStructuredReportHtml(report, extraHtml) {
    const verdict = verdictTone((report || {}).verdict || '');
    const verdictRationale = safeStr((report || {}).verdict_rationale || (report || {}).verdict_reason || (report || {}).rationale)
      .replace(/^([^\n]+)\n(?=- )/, '$1\n\n');
    const peak = ((report || {}).peak_performance || (report || {}).peak_perfomance || {});
    const testProfile = ((report || {}).test_profile || {});
    const stability = ((report || {}).stability_under_load || {});
    const isStabilityReport = Boolean(
      peak.not_applicable
      || peak.notApplicable
      || safeStr(testProfile.mode).toLowerCase() === 'stability'
      || ['soak', 'stability', 'endurance'].includes(safeStr(testProfile.test_type).toLowerCase())
    );
    const findings = (report || {}).findings || [];
    const actions = (report || {}).recommended_actions || (report || {}).actions || [];
    const verdictRows = [
      {
        label: 'Вердикт по тесту',
        html: `<span class="report-verdict-pill ${verdict.className}">${escapeHtml(verdict.text)}</span>`
      }
    ];
    verdictRows._tableClass = 'report-aligned-value-table';
    const peakRows = [
      { label: 'Максимальный RPS', html: escapeHtml(safeStr(peak.max_rps) || '—') },
      { label: 'Время пиковой производительности', html: escapeHtml(safeStr(peak.max_time) || '—') },
      { label: 'Время деградации', html: escapeHtml(safeStr(peak.drop_time) || '—') }
    ];
    peakRows._tableClass = 'report-aligned-value-table';
    const stabilityRows = [
      { label: 'Тип теста', html: escapeHtml(safeStr(testProfile.test_type) || 'stability/soak') },
      { label: 'Фокус оценки', html: escapeHtml(safeStr(stability.focus || testProfile.focus) || 'Удержание нагрузки без накопления деградации') },
      { label: 'Целевой RPS', html: escapeHtml(safeStr(stability.target_rps) || '—') },
      { label: 'Фактический RPS', html: escapeHtml(safeStr(stability.actual_rps) || '—') },
      { label: 'Итог SLA', html: escapeHtml(safeStr(stability.sla_summary) || '—') }
    ];
    stabilityRows._tableClass = 'report-aligned-value-table';
    const sections = [
      renderKeyValueTable('', verdictRows),
      renderRationaleBox(verdictRationale),
      isStabilityReport
        ? renderKeyValueTable('Стабильность под нагрузкой', stabilityRows)
        : renderKeyValueTable('Пиковая производительность', peakRows),
      renderProblemRecommendationRows(pairFindingsWithActions(findings, actions)),
      extraHtml || ''
    ].filter(Boolean);
    return `<div class="report-stack">${sections.join('')}</div>`;
  }
  function judgeDetailsMarkdown(scores) {
    try {
      const s = scores || {};
      if (!Object.keys(s).length) return '';
      const j = (s.judge) || {};
      const rubric = (j && typeof j.rubric === 'object' && j.rubric) ? j.rubric : {};
      const meta = (s.judge_meta && typeof s.judge_meta === 'object') ? s.judge_meta : {};
      const dataDetails = (s.data_score_details && typeof s.data_score_details === 'object') ? s.data_score_details : {};
      const overall = j.overall, factual = j.factual, completeness = j.completeness, specificity = j.specificity;
      const dataScore = s.data_score, finalScore = s.final_score, conf = s.confidence;
      const level = (x) => {
        const v = Number(x);
        if (!Number.isFinite(v) || v <= 0) return 'нет данных';
        if (v >= 0.9) return 'очень высокая';
        if (v >= 0.75) return 'высокая';
        if (v >= 0.55) return 'умеренная';
        return 'низкая';
      };
      const num = (x) => {
        const v = Number(x);
        return Number.isFinite(v) ? v : null;
      };
      const summary = (() => {
        const judge = num(overall);
        const data = num(dataScore);
        const final = num(finalScore);
        const numericGrounding = num(dataDetails.numeric_grounding);
        const numericClaimsTotal = Number(dataDetails.numeric_claims_total || 0);
        const labelGrounding = num(dataDetails.label_grounding);
        const peakChecked = Boolean(dataDetails.peak_checked);
        const peakConsistency = num(dataDetails.peak_consistency);
        if (judge !== null && data !== null) {
          if (
            numericClaimsTotal > 0
            && numericGrounding !== null
            && numericGrounding >= 0.65
            && labelGrounding !== null
            && labelGrounding < 0.35
          ) {
            if (peakChecked && peakConsistency !== null && peakConsistency < 0.35) {
              return 'Числа в тексте в целом сходятся с данными, но в findings мало прямых ссылок на имена метрик и серий, а peak_performance не подтвердился.';
            }
            return 'Числа в тексте в целом сходятся с данными, но в findings мало прямых ссылок на имена метрик и серий из контекста.';
          }
          if (numericClaimsTotal > 0 && numericGrounding !== null && numericGrounding < 0.35) {
            return 'Судья оценивает ответ высоко, но числовые утверждения из текста подтверждаются данными слабо.';
          }
          if (
            labelGrounding !== null
            && labelGrounding < 0.35
            && (numericClaimsTotal === 0 || numericGrounding === null || numericGrounding < 0.55)
          ) {
            return 'Судья оценивает ответ высоко, но текст слабо привязан к конкретным метрикам и сериям из контекста.';
          }
          if (peakChecked && peakConsistency !== null && peakConsistency < 0.2 && data >= 0.4) {
            return 'Основные выводы частично подтверждаются, но значение peak_performance заметно расходится с расчётом по данным.';
          }
          if (judge >= 0.8 && data >= 0.7) return 'Ответ выглядит надежным: и судья, и эвристическая проверка по данным оценивают его высоко.';
          if (judge >= 0.8 && data < 0.5) return 'Судья оценивает ответ высоко, но эвристическая проверка по данным подтверждает его только частично.';
          if (judge < 0.6 && data >= 0.7) return 'По данным ответ подтверждается лучше, чем по оценке судьи: вероятно, ему не хватило полноты или конкретики.';
          if (judge < 0.6 && data < 0.6) return 'И судья, и эвристическая проверка по данным оценивают ответ сдержанно.';
          if (Math.abs(judge - data) < 0.12) return 'Судья и эвристическая проверка по данным дают близкие оценки.';
        }
        if (judge !== null && final !== null && final + 0.15 < judge) {
          return 'Итоговая оценка заметно ниже оценки судьи, потому что проверка по данным нашла мало подтверждений.';
        }
        return 'Это служебная оценка качества ответа: она помогает выбрать лучший кандидат среди нескольких вариантов.';
      })();
      const lines = [];
      if (summary) lines.push(summary);
      lines.push('');
      lines.push('#### Главное');
      lines.push(`- Оценка текста судьей: ${pct(overall)} (${level(overall)})`);
      lines.push(`- Эвристическая проверка по данным: ${pct(dataScore)} (${level(dataScore)})`);
      lines.push(`- Итог для выбора кандидата: ${pct(finalScore)} (${level(finalScore)})`);
      if (typeof conf === 'number') lines.push(`- Уверенность модели: ${pct(conf)} (${level(conf)})`);
      lines.push('');
      lines.push('#### За что поставлена оценка');
      lines.push(`- Точность относительно данных: ${pct(factual)}`);
      lines.push(`- Полнота покрытия важных наблюдений: ${pct(completeness)}`);
      lines.push(`- Конкретика по метрикам и компонентам: ${pct(specificity)}`);
      if (Object.keys(rubric).length) {
        lines.push('');
        lines.push('#### Детальная проверка');
        lines.push(`- Опора на данные: ${pct(rubric.evidence_grounding)}`);
        lines.push(`- Покрытие важных проблем: ${pct(rubric.issue_coverage)}`);
        lines.push(`- Насыщенность конкретикой: ${pct(rubric.specificity)}`);
        lines.push(`- Учет SLA и рисков: ${pct(rubric.sla_alignment)}`);
        lines.push(`- Полезность рекомендаций: ${pct(rubric.actionability)}`);
      }
      if (Object.keys(dataDetails).length) {
        lines.push('');
        lines.push('#### Что подтвердилось по данным');
        lines.push(`- Прямые ссылки на имена метрик и серий: ${pct(dataDetails.label_grounding)}`);
        if (Number(dataDetails.numeric_claims_total || 0) > 0) {
          lines.push(`- Совпадение чисел из текста: ${pct(dataDetails.numeric_grounding)} (подтверждено ${Number(dataDetails.numeric_claims_supported || 0)} из ${Number(dataDetails.numeric_claims_total || 0)})`);
        } else {
          lines.push('- Совпадение чисел из текста: не проверялось, в findings не найдено явных числовых утверждений.');
        }
        if (dataDetails.peak_checked) {
          lines.push(`- Совпадение peak_performance: ${pct(dataDetails.peak_consistency)}`);
        } else {
          lines.push('- Совпадение peak_performance: не проверялось.');
        }
        lines.push(`- Опора рекомендаций на подтвержденные наблюдения: ${pct(dataDetails.recommendation_grounding)}`);
      }
      if (meta.used_safe_path || meta.context_truncated || Number(meta.truncated_candidates || 0) > 0) {
        lines.push('');
        lines.push('#### Особенности оценки');
        if (meta.used_safe_path) lines.push('- Для этого ответа использовался совместимый fallback-режим без доменной rubric.');
        if (meta.context_truncated) lines.push('- Для этого отчета judge работал с укороченной версией контекста.');
        if (Number(meta.truncated_candidates || 0) > 0) lines.push(`- В этом отчете часть candidate-ответов была укорочена перед оценкой: ${Number(meta.truncated_candidates || 0)}.`);
      }
      lines.push('');
      lines.push('_Проверка по данным остаётся эвристической: она оценивает привязку текста к метрикам, совпадение чисел и согласованность peak_performance, а не выполняет полноценный факт-чекинг каждой фразы._');
      return lines.join('\n');
    } catch (e) { return ''; }
  }
  function judgeSectionHtml(scores) {
    const md = judgeDetailsMarkdown(scores);
    if (!safeStr(md)) return '';
    return [
      '<details class="report-collapsible">',
      '<summary><span>Оценка ответа</span></summary>',
      `<div class="report-collapsible-body report-markdown-block">${markdownToSafeHtml(md)}</div>`,
      '</details>'
    ].join('');
  }

  function slaMarkdown(row) {
    try {
      if (!row || row.domain !== 'final') return '';
      const verdict = safeStr(row.sla_verdict);
      let details = row.sla_details;
      if (details && typeof details === 'string') {
        try { details = JSON.parse(details); } catch (e) { details = null; }
      }
      const checks = Array.isArray(details && details.checks) ? details.checks : [];
      const summary = safeStr(details && details.summary);
      if (!verdict && !checks.length && !summary) return '';
      const lines = ['', '#### SLA'];
      if (verdict) lines.push(`- Вердикт SLA: ${verdict}`);
      if (summary) lines.push(`- Сводка: ${summary}`);
      checks.forEach((check) => {
        if (!check || typeof check !== 'object') return;
        const name = safeStr(check.name);
        const actual = safeStr(check.actual);
        const threshold = safeStr(check.threshold);
        const passed = check.passed === true ? 'OK' : (check.passed === false ? 'FAIL' : 'N/A');
        const msg = safeStr(check.message);
        lines.push(`- ${name || 'check'}: ${passed}; actual=${actual || '—'}; threshold=${threshold || '—'}${msg ? `; ${msg}` : ''}`);
      });
      return '\n' + lines.join('\n');
    } catch (e) {
      return '';
    }
  }

  function looksLikeBrokenStructuredResponse(raw) {
    const text = safeStr(raw);
    if (!text) return false;
    const looksStructured = text.startsWith('{') || text.startsWith('```json') || text.includes('"verdict"') || text.includes('"findings"');
    if (!looksStructured) return false;
    try {
      JSON.parse(text);
      return false;
    } catch (e) {
      return true;
    }
  }

  function invalidStructuredResponseMarkdown(raw) {
    const text = safeStr(raw);
    const excerpt = (text.length > 500 ? `${text.slice(0, 500)}...` : text).replace(/```/g, '` ` `');
    const lines = [
      '### Ошибка структуры ответа LLM',
      '- Ответ модели похож на JSON, но не прошёл валидацию и не был показан как полноценный отчёт.',
      '- Проверьте лимит токенов, prompt и повторите генерацию.',
    ];
    if (excerpt) {
      lines.push('');
      lines.push('```json');
      lines.push(excerpt);
      lines.push('```');
    }
    return lines.join('\n');
  }

  async function reportsLoadLlm() {
    const run = getRunFromPath(); if (!run) return;
    const box = document.getElementById('rep-llm-tabs'); if (!box) return;
    box.textContent = 'Загрузка…';
    const resp = await fetch('/llm_reports?run_name=' + encodeURIComponent(run));
    let arr = await resp.json();
    if (!resp.ok) {
      box.textContent = safeStr(arr && (arr.error || arr.message)) || 'Не удалось загрузить LLM-отчёт';
      return;
    }
    try {
      const title = document.getElementById('pageTitle');
      if (title) title.textContent = `Отчет по тесту ${run}`;
      const list = (Array.isArray(arr) ? arr : []);
      const parsedContexts = list
        .map((item) => parseSystemContextValue(item && item.system_context))
        .filter((item) => item && typeof item === 'object');
      const systemContext = parsedContexts.find((item) => hasMeaningfulSystemContext(item)) || parsedContexts[0] || null;
      renderSystemContextBox(systemContext);
      const starts = list.map((x) => parseInt(x.start_ms, 10)).filter((v) => Number.isFinite(v));
      const ends = list.map((x) => parseInt(x.end_ms, 10)).filter((v) => Number.isFinite(v));
      let startStr = '—', endStr = '—';
      if (starts.length) {
        const ms = Math.min.apply(null, starts);
        const d = new Date(ms);
        const yyyy = d.getFullYear();
        const MM = String(d.getMonth() + 1).padStart(2, '0');
        const dd = String(d.getDate()).padStart(2, '0');
        const hh = String(d.getHours()).padStart(2, '0');
        const mm = String(d.getMinutes()).padStart(2, '0');
        startStr = `${yyyy}-${MM}-${dd} ${hh}:${mm}`;
      }
      if (ends.length) {
        const me = Math.max.apply(null, ends);
        const de = new Date(me);
        const yyyy = de.getFullYear();
        const MM = String(de.getMonth() + 1).padStart(2, '0');
        const dd = String(de.getDate()).padStart(2, '0');
        const hh = String(de.getHours()).padStart(2, '0');
        const mm = String(de.getMinutes()).padStart(2, '0');
        endStr = `${yyyy}-${MM}-${dd} ${hh}:${mm}`;
      }
      const range = document.getElementById('pageTimeRange');
      if (range) range.textContent = `Время теста: ${startStr} - ${endStr}`;
    } catch (e) {}
    if (!Array.isArray(arr)) { renderSystemContextBox(null); box.textContent = 'Нет данных'; return; }
    arr = arr.filter((x) => (x && x.domain && x.domain !== 'engineer'));
    const domainsOrder = ['final', 'jvm', 'database', 'kafka', 'microservices', 'hard_resources', 'lt_framework'];
    arr.sort((a, b) => domainsOrder.indexOf(a.domain) - domainsOrder.indexOf(b.domain));
    const tabsNav = document.createElement('div'); tabsNav.className = 'app-nav-inner';
    const tabsBody = document.createElement('div');
    const idBase = 'rep-llm-tab-';
    arr.forEach((x, idx) => {
      const btn = document.createElement('button'); btn.className = 'nav-btn' + (idx === 0 ? ' active' : ''); btn.textContent = (x.domain === 'final' ? 'Итог' : x.domain);
      btn.dataset.target = idBase + idx;
      tabsNav.appendChild(btn);
      const pane = document.createElement('div'); pane.id = idBase + idx; pane.className = 'panel' + (idx === 0 ? ' active' : '');
      let md = '';
      let html = '';
      let hasStructuredData = false;
      let structuredReport = null;
      let sc = x ? x.scores : null; if (sc && typeof sc === 'string') { try { sc = JSON.parse(sc); } catch (e) {} }
      try {
        let parsed = x ? x.parsed : null;
        if (parsed && typeof parsed === 'string') { try { parsed = JSON.parse(parsed); } catch (e) {} }
        if (parsed && typeof parsed === 'object') {
          hasStructuredData = true;
          structuredReport = parsed;
        } else {
          const raw = String(x && x.text ? x.text : '');
          let fallbackParsed = null;
          if (raw.trim().startsWith('{')) { try { fallbackParsed = JSON.parse(raw); } catch (e) {} }
          if (!fallbackParsed && raw.includes('\"verdict\"')) {
            try { const start = raw.indexOf('{'); const end = raw.lastIndexOf('}'); if (start >= 0 && end > start) { fallbackParsed = JSON.parse(raw.slice(start, end + 1)); } } catch (e) {}
          }
          if (fallbackParsed && typeof fallbackParsed === 'object') {
            hasStructuredData = true;
            structuredReport = fallbackParsed;
          } else if (looksLikeBrokenStructuredResponse(raw)) {
            md = invalidStructuredResponseMarkdown(raw);
          } else {
            md = raw;
          }
        }
      } catch (e) { md = String(x && x.text ? x.text : ''); }
      if (hasStructuredData) {
        const extraSections = [];
        try {
          const slaMd = slaMarkdown(x);
          if (safeStr(slaMd)) {
            extraSections.push(`<div class="report-markdown-block">${markdownToSafeHtml(slaMd)}</div>`);
          }
        } catch (e) {}
        try {
          const judgeHtml = judgeSectionHtml(sc);
          if (safeStr(judgeHtml)) extraSections.push(judgeHtml);
        } catch (e) {}
        html = renderStructuredReportHtml(structuredReport || {}, extraSections.join(''));
      } else {
        try { md += slaMarkdown(x); } catch (e) {}
        html = markdownToSafeHtml(md);
      }
      pane.innerHTML = `<div class="report-analysis-card">${html}</div>`;
      tabsBody.appendChild(pane);
    });
    box.innerHTML = ''; box.appendChild(tabsNav); box.appendChild(tabsBody);
    tabsNav.querySelectorAll('.nav-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        tabsNav.querySelectorAll('.nav-btn').forEach((b) => b.classList.remove('active'));
        tabsBody.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
        btn.classList.add('active');
        const t = btn.dataset.target; const pane = tabsBody.querySelector('#' + t); if (pane) pane.classList.add('active');
        scheduleLegendSync();
      });
    });
  }

  async function reportsDrawOne(run, domain, ql, canvasId, legendRoot) {
    const u = new URL('/run_series', location.origin);
    u.searchParams.set('run_name', run);
    u.searchParams.set('domain', domain);
    u.searchParams.set('query_label', ql);
    u.searchParams.set('series_key', 'auto');
    u.searchParams.set('align', 'absolute');
    const resp = await fetch(u);
    const data = await resp.json();
    if (!data || !data.points || !data.points.length) {
      try { const tbl = legendRoot.querySelector('.table'); if (tbl) tbl.innerHTML = 'Нет данных'; } catch (e) {}
      return;
    }
    const labels = []; const map = {};
    data.points.forEach((p) => {
      const t = p.t;
      if (labels.indexOf(t) < 0) labels.push(t);
      const k = p.series; if (!map[k]) map[k] = new Map();
      map[k].set(t, p.value);
    });
    labels.sort((a, b) => new Date(a) - new Date(b));
    const datasets = Object.keys(map).map((k) => { const color = randColor(); return { label: k, data: labels.map((t) => (map[k].has(t) ? map[k].get(t) : null)), borderColor: color, backgroundColor: color, pointRadius: 0, borderWidth: 2, spanGaps: true }; });
    const ctx = document.getElementById(canvasId).getContext('2d');
    // eslint-disable-next-line no-undef
    const chart = new Chart(ctx, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        interaction: { mode: 'nearest', intersect: false },
        plugins: { legend: { display: false }, cmpBg: { color: '#151515' } },
        scales: {
          x: {
            title: { display: true, text: 'Время', color: '#ccc' },
            ticks: {
              color: '#bbb',
              callback(value) {
                const raw = this.getLabelForValue(value);
                const d = new Date(raw);
                const hh = String(d.getHours()).padStart(2, '0');
                const mm = String(d.getMinutes()).padStart(2, '0');
                return `${hh}:${mm}`;
              }
            },
            grid: { color: '#2f2f2f', drawBorder: true, borderColor: '#444' }
          },
          y: { ticks: { color: '#bbb' }, grid: { color: '#2f2f2f', drawBorder: true, borderColor: '#444' } }
        }
      }
    });
    try { legendRoot.dataset.domain = domain; legendRoot.dataset.queryLabel = ql; } catch (e) {}
    buildLegendFor(chart, legendRoot);
    try {
      const canvasEl = document.getElementById(canvasId);
      legendRoot.style.height = 'auto';
      const rect = canvasEl.getBoundingClientRect();
      const attrH = (canvasEl.height || parseInt(canvasEl.getAttribute('height') || '0', 10)) || 0;
      const h = Math.max(160, Math.floor((rect && rect.height) || attrH || 0));
      legendRoot.style.height = h + 'px';
      if (window.requestAnimationFrame) requestAnimationFrame(() => {
        const rect2 = canvasEl.getBoundingClientRect();
        const attrH2 = (canvasEl.height || parseInt(canvasEl.getAttribute('height') || '0', 10)) || 0;
        const h2 = Math.max(160, Math.floor((rect2 && rect2.height) || attrH2 || 0));
        legendRoot.style.height = h2 + 'px';
      });
    } catch (e) {}
    scheduleLegendSync();
  }

  function downloadChartWithLegend(currentChart, root) {
    if (!currentChart) return null;
    const canvas = currentChart.canvas;
    const chartW = canvas.width;
    const chartH = canvas.height;
    const rows = (currentChart.data?.datasets || []).map((ds, i) => {
      const vals = (ds.data || []).filter((v) => v != null && !isNaN(v));
      const avg = vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length) : 0;
      return { idx: i, label: ds.label, color: ds.borderColor, avg, visible: currentChart.isDatasetVisible(i) };
    }).filter((r) => r.visible);
    const pad = 16, rowH = 24, titleH = 22, hdrH = rows.length ? (titleH + 10) : 0;
    const legendH = rows.length ? (hdrH + rows.length * rowH + pad) : 0;
    const outH = chartH + (legendH ? (legendH + pad) : 0);
    const out = document.createElement('canvas');
    out.width = chartW; out.height = outH;
    const ctx = out.getContext('2d');
    ctx.fillStyle = '#0f0f0f';
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(canvas, 0, 0);
    if (rows.length) {
      let y = chartH + pad;
      ctx.fillStyle = '#ddd'; ctx.font = '16px Montserrat, Arial, sans-serif';
      ctx.fillText('Легенда', pad, y);
      y += titleH;
      ctx.font = '13px Montserrat, Arial, sans-serif';
      rows.forEach((r) => {
        ctx.fillStyle = r.color || '#888';
        ctx.fillRect(pad, y - 12, 14, 14);
        ctx.strokeStyle = '#444'; ctx.strokeRect(pad, y - 12, 14, 14);
        ctx.fillStyle = '#ddd';
        const label = String(r.label || '');
        ctx.fillText(label, pad + 20, y);
        const avgStr = Number.isFinite(r.avg) ? r.avg.toFixed(2) : '—';
        const right = out.width - pad;
        const text = `Среднее: ${avgStr}`;
        const tw = ctx.measureText(text).width;
        ctx.fillText(text, right - tw, y);
        y += rowH;
      });
    }
    const a = document.createElement('a');
    a.href = out.toDataURL('image/png');
    const nameParts = [];
    try { const d = root?.dataset?.domain; if (d) nameParts.push(d); const q = root?.dataset?.queryLabel; if (q) nameParts.push(q); } catch (e) {}
    a.download = `report-${(nameParts.join('-') || 'chart')}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    return true;
  }
  let repLegendSortBy = 'avg'; let repLegendSortDir = 'desc';
  function buildLegendFor(chart, root) {
    const box = root.querySelector('.table'); if (!box) return; box.innerHTML = '';
    const tbl = document.createElement('table'); tbl.className = 'cmp-legend-table';
    const thead = document.createElement('thead'); thead.innerHTML = '<tr><th data-sort=\"name\" style=\"cursor:pointer\">Серия</th><th>Цвет</th><th data-sort=\"avg\" style=\"cursor:pointer\">Среднее</th><th>Вкл</th></tr>';
    tbl.appendChild(thead); const tbody = document.createElement('tbody');
    let rows = (chart?.data?.datasets || []).map((ds, i) => {
      const vals = (ds.data || []).filter((v) => v != null && !isNaN(v));
      const avg = vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length) : 0;
      return { idx: i, label: ds.label, color: ds.borderColor, avg: avg, visible: chart.isDatasetVisible(i) };
    });
    rows.sort((a, b) => repLegendSortBy === 'name' ? (repLegendSortDir === 'asc' ? a.label.localeCompare(b.label) : b.label.localeCompare(a.label)) : (repLegendSortDir === 'asc' ? (a.avg - b.avg) : (b.avg - a.avg)));
    rows.forEach((row) => {
      const tr = document.createElement('tr'); tr.className = 'cmp-legend-row' + (row.visible ? '' : ' hidden cmp-off');
      const tdName = document.createElement('td'); tdName.textContent = row.label;
      const tdColor = document.createElement('td'); const sw = document.createElement('span'); sw.className = 'cmp-legend-color'; sw.style.background = row.color; tdColor.appendChild(sw);
      const tdAvg = document.createElement('td'); tdAvg.textContent = isFinite(row.avg) ? row.avg.toFixed(2) : '—';
      const tdToggle = document.createElement('td'); tdToggle.textContent = row.visible ? '✓' : '✕';
      tr.appendChild(tdName); tr.appendChild(tdColor); tr.appendChild(tdAvg); tr.appendChild(tdToggle);
      tr.addEventListener('click', () => {
        const vis = chart.isDatasetVisible(row.idx);
        chart.setDatasetVisibility(row.idx, !vis);
        chart.update();
        buildLegendFor(chart, root);
      });
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody); box.appendChild(tbl);
    thead.addEventListener('click', (e) => {
      const th = e.target.closest('[data-sort]'); if (!th) return;
      const by = th.getAttribute('data-sort');
      if (repLegendSortBy === by) { repLegendSortDir = (repLegendSortDir === 'asc') ? 'desc' : 'asc'; }
      else { repLegendSortBy = by; repLegendSortDir = (by === 'avg') ? 'desc' : 'asc'; }
      buildLegendFor(chart, root);
    });
    const hideBtn = root.querySelector('.hideAll'); const showBtn = root.querySelector('.showAll');
    if (hideBtn) hideBtn.onclick = () => { for (let i = 0; i < chart.data.datasets.length; i++) { chart.setDatasetVisibility(i, false); } chart.update(); buildLegendFor(chart, root); };
    if (showBtn) showBtn.onclick = () => { for (let i = 0; i < chart.data.datasets.length; i++) { chart.setDatasetVisibility(i, true); } chart.update(); buildLegendFor(chart, root); };
    const dlBtn = root.querySelector('.downloadPng'); if (dlBtn) dlBtn.onclick = () => { downloadChartWithLegend(chart, root); };
  }

  function syncLegends() {
    try {
      const doSync = () => {
        // Определяем эталонную высоту по первому видимому канвасу
        let referenceH = 0;
        document.querySelectorAll('.cmp-chart-wrap').forEach((w) => {
          if (referenceH > 0) return;
          // считаем видимым, если элемент участвует в раскладке
          if (!w || w.offsetParent === null) return;
          const canvas = w.querySelector('canvas');
          if (!canvas) return;
          const rect = canvas.getBoundingClientRect();
          const attrH = (canvas.height || parseInt(canvas.getAttribute('height') || '0', 10)) || 0;
          const h = Math.max(0, Math.floor((rect && rect.height) || 0), attrH);
          if (h > 0) referenceH = h;
        });
        // Фолбэк по умолчанию
        if (referenceH <= 0) referenceH = 160;

        document.querySelectorAll('.cmp-chart-wrap').forEach((w) => {
          const canvas = w.querySelector('canvas');
          const legend = w.querySelector('.cmp-legend-panel');
          if (!canvas || !legend) return;
          legend.style.height = 'auto';
          const rect = canvas.getBoundingClientRect();
          const attrH = (canvas.height || parseInt(canvas.getAttribute('height') || '0', 10)) || 0;
          let h = Math.max(Math.floor((rect && rect.height) || 0), attrH, referenceH);
          h = Math.max(160, h);
          legend.style.height = h + 'px';
        });
      };
      doSync();
      if (window.requestAnimationFrame) {
        requestAnimationFrame(() => doSync());
        // Дополнительная попытка после финального ресайза графиков
        requestAnimationFrame(() => requestAnimationFrame(() => doSync()));
      }
      // Фолбэк таймером для случаев скрытых табов
      setTimeout(doSync, 60);
      setTimeout(doSync, 120);
      setTimeout(doSync, 250);
    } catch (e) {}
  }
  window.addEventListener('resize', syncLegends);

  // Планировщик повторного пересчёта (надёжнее при скрытых табах и ленивой отрисовке)
  function scheduleLegendSync() {
    try {
      try {
        if (window.Chart && typeof Chart.getChart === 'function') {
          document.querySelectorAll('.cmp-chart-wrap canvas').forEach((c) => {
            const inst = Chart.getChart(c);
            if (inst && typeof inst.resize === 'function') {
              try { inst.resize(); } catch (e) {}
            }
          });
        }
      } catch (e) {}
      syncLegends();
      if (window.requestAnimationFrame) {
        requestAnimationFrame(syncLegends);
        requestAnimationFrame(() => requestAnimationFrame(syncLegends));
      }
      setTimeout(syncLegends, 0);
      setTimeout(syncLegends, 60);
      setTimeout(syncLegends, 120);
      setTimeout(syncLegends, 250);
    } catch (e) {}
  }

  async function engineerLoad() {
    try {
      const run = getRunFromPath(); if (!run) return;
      const r = await fetch('/engineer_summary?run_name=' + encodeURIComponent(run));
      if (!r.ok) return;
      const j = await r.json();
      const ed = document.getElementById('engineerEditor');
      const updated = document.getElementById('engineerUpdated');
      let html = String(j.content_html || '');
      try { if (window.DOMPurify) html = window.DOMPurify.sanitize(html); } catch (e) {}
      ed.innerHTML = html || '<p style=\"color:#888\">Добавьте итоговый комментарий…</p>';
      updated.textContent = j.created_at ? ('Обновлено: ' + j.created_at.replace('T', ' ').slice(0, 16)) : '';
    } catch (e) {}
  }
  function editorExec(cmd, val) {
    try {
      if (cmd === 'h3' || cmd === 'h4') { document.execCommand('formatBlock', false, cmd.toUpperCase()); return; }
      document.execCommand(cmd, false, val || null);
    } catch (e) {}
  }
  function wireEngineerEditor() {
    const tb = document.getElementById('engineerToolbar');
    if (tb) {
      tb.addEventListener('click', (e) => {
        const btn = e.target.closest('button'); if (!btn) return;
        const cmd = btn.getAttribute('data-cmd'); if (cmd) editorExec(cmd);
      });
    }
    const linkBtn = document.getElementById('cmdLink');
    if (linkBtn) { linkBtn.addEventListener('click', () => { const url = prompt('URL ссылки:', 'https://'); if (url) { editorExec('createLink', url); } }); }
    const saveBtn = document.getElementById('saveEngineer');
    const editBtn = document.getElementById('editEngineer');
    const editor = document.getElementById('engineerEditor');
    function setEngineerEditing(on) {
      try {
        if (editor) editor.setAttribute('contenteditable', on ? 'true' : 'false');
        if (tb) tb.style.display = on ? '' : 'none';
        if (saveBtn) saveBtn.style.display = on ? '' : 'none';
        if (editBtn) editBtn.textContent = on ? 'Завершить' : 'Редактировать';
      } catch (e) {}
    }
    if (editBtn) {
      editBtn.addEventListener('click', () => {
        const isOn = editor && editor.getAttribute('contenteditable') === 'true';
        setEngineerEditing(!isOn);
      });
    }
    setEngineerEditing(false);
    if (saveBtn) {
      saveBtn.addEventListener('click', async () => {
        const run = getRunFromPath(); if (!run) return;
        const html = document.getElementById('engineerEditor').innerHTML;
        const st = document.getElementById('engineerStatus');
        st.textContent = 'Сохранение…';
        try {
          const resp = await fetch('/engineer_summary', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ run_name: run, content_html: html }) });
          const j = await resp.json();
          if (resp.ok) { st.textContent = 'Сохранено'; setEngineerEditing(false); await engineerLoad(); setTimeout(() => { st.textContent = ''; }, 1500); }
          else { st.textContent = j.error || 'Ошибка сохранения'; }
        } catch (e) { st.textContent = 'Ошибка сохранения'; }
      });
    }
  }

  async function reportsRenderDomains(schema) {
    const run = getRunFromPath(); if (!run) return;
    const root = document.getElementById('rep-domain-tabs'); if (!root) return;
    root.innerHTML = 'Загрузка…';
    const domains = Object.keys(schema || {});
    const tabsNav = document.createElement('div'); tabsNav.className = 'app-nav-inner';
    const tabsBody = document.createElement('div');
    const idBase = 'rep-dom-tab-';
    domains.forEach((domain, di) => {
      const btn = document.createElement('button'); btn.className = 'nav-btn' + (di === 0 ? ' active' : ''); btn.textContent = domain; btn.dataset.target = idBase + di; tabsNav.appendChild(btn);
      const pane = document.createElement('div'); pane.id = idBase + di; pane.className = 'panel' + (di === 0 ? ' active' : '');
      const list = (schema[domain] || []).map((x) => x.query_label);
      list.forEach((ql, qi) => {
        const wrap = document.createElement('div'); wrap.className = 'cmp-chart-wrap'; wrap.style.marginTop = '12px';
        const canvasBox = document.createElement('div'); canvasBox.className = 'cmp-chart-canvas';
        const title = document.createElement('div'); title.style.padding = '8px 0'; title.style.fontWeight = '600'; title.textContent = ql;
        const cnv = document.createElement('canvas'); cnv.id = `rep-chart-${di}-${qi}`; cnv.height = 120; canvasBox.appendChild(title); canvasBox.appendChild(cnv);
        const legend = document.createElement('div'); legend.className = 'cmp-legend-panel'; legend.innerHTML = '<h4>Легенда</h4><div class=\"cmp-legend-controls\"><button class=\"hideAll\">Выключить все</button><button class=\"showAll\">Включить все</button><button class=\"downloadPng\">Скачать PNG</button></div><div class=\"table\"></div>';
        wrap.appendChild(canvasBox); wrap.appendChild(legend); pane.appendChild(wrap);
        setTimeout(async () => { await reportsDrawOne(run, domain, ql, cnv.id, legend); }, 0);
      });
      tabsBody.appendChild(pane);
    });
    root.innerHTML = ''; root.appendChild(tabsNav); root.appendChild(tabsBody);
    tabsNav.querySelectorAll('.nav-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        tabsNav.querySelectorAll('.nav-btn').forEach((b) => b.classList.remove('active'));
        tabsBody.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
        btn.classList.add('active');
        const t = btn.dataset.target; const pane = tabsBody.querySelector('#' + t); if (pane) pane.classList.add('active');
        scheduleLegendSync();
      });
    });
    if (!domains.length) { root.textContent = 'Нет данных'; }
  }

  // Наблюдатели за переключением классов панелей (пересчёт при показе)
  function attachPanelObservers() {
    try {
      const targets = ['rep-llm-tabs', 'rep-domain-tabs'];
      targets.forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;
        const mo = new MutationObserver(() => { scheduleLegendSync(); });
        mo.observe(el, { attributes: true, subtree: true, attributeFilter: ['class'] });
      });
    } catch (e) {}
  }

  document.addEventListener('DOMContentLoaded', async () => {
    try { await window.LoadLens.initProjectArea(); } catch (e) {}
    updateReportTitle(getRunFromPath());
    wireRenameReport();
    const schema = await reportsLoadSchema();
    await reportsLoadLlm();
    await reportsRenderDomains(schema);
    wireEngineerEditor();
    await engineerLoad();
    attachPanelObservers();
    scheduleLegendSync();
  });
})();


