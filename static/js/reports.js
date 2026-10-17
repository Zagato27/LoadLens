// Reports page logic
(function () {
  // Chart background plugin: the plot area gets the color passed in options.plugins.cmpBg (ui.restyleCharts
  // keeps it in sync with the theme), else --bg-chart of the current theme.
  function chartBackground() {
    try { return getComputedStyle(document.documentElement).getPropertyValue('--bg-chart').trim() || '#12151c'; } catch (e) { return '#12151c'; }
  }
  const cmpBgPlugin = { id: 'cmpBg', beforeDraw(chart, args, opts) { const { ctx, chartArea } = chart; if (!chartArea) return; ctx.save(); ctx.fillStyle = (opts && opts.color) || chartBackground(); ctx.fillRect(chartArea.left, chartArea.top, chartArea.right - chartArea.left, chartArea.bottom - chartArea.top); ctx.restore(); } };
  if (window.Chart && Chart.register) Chart.register(cmpBgPlugin);

  const LL = window.LoadLens;
  // Report-wide chart state: markers and thresholds of the final row, every chart drawn so far and the
  // per-tab chart groups (synchronized hover, drag zoom, legend chips).
  const reportState = { finalRow: null, stepTable: null, markers: [], thresholds: [], queryLabels: [], charts: [], groups: {}, activePane: '', paneRenderers: {} };
  const MAX_VISIBLE_SERIES = 12;
  // Rows in the hover tooltip next to the cursor; the rest is «ещё N», all values are in the legend.
  const TOOLTIP_ROWS = 8;

  let feedbackByKey = {};
  // Domain titles stored with the report (query packs title «jvm» as «Go runtime» and the like).
  let reportDomainTitles = {};

  function feedbackKey(domain, target, findingId) {
    return `${domain || ''}|${target || ''}|${findingId || ''}`;
  }
  function currentVote(domain, target, findingId) {
    const row = feedbackByKey[feedbackKey(domain, target, findingId)];
    return row && row.vote ? row.vote : '';
  }
  function setVote(domain, target, findingId, vote, comment) {
    feedbackByKey[feedbackKey(domain, target, findingId)] = {
      domain, target, finding_id: findingId || '', vote, comment: comment || ''
    };
  }
  function indexFeedback(rows) {
    feedbackByKey = {};
    (Array.isArray(rows) ? rows : []).forEach((row) => {
      if (!row || !row.domain || !row.target || !row.vote) return;
      setVote(row.domain, row.target, row.finding_id, row.vote, row.comment);
    });
  }
  function renderVoteGroup(domain, target, findingId, agreeLabel, disagreeLabel) {
    if (!domain || !target) return '';
    const current = currentVote(domain, target, findingId);
    return [
      `<div class="feedback-votes" data-feedback-domain="${escapeHtml(domain)}" data-feedback-target="${escapeHtml(target)}" data-feedback-id="${escapeHtml(findingId || '')}">`,
      `<button type="button" class="btn btn-sm feedback-vote${current === 'agree' ? ' is-agree' : ''}" data-vote="agree">${escapeHtml(agreeLabel)}</button>`,
      `<button type="button" class="btn btn-sm feedback-vote${current === 'disagree' ? ' is-disagree' : ''}" data-vote="disagree">${escapeHtml(disagreeLabel)}</button>`,
      '</div>'
    ].join('');
  }
  function paintVoteGroup(group) {
    const current = currentVote(group.dataset.feedbackDomain, group.dataset.feedbackTarget, group.dataset.feedbackId);
    group.querySelectorAll('.feedback-vote').forEach((btn) => {
      btn.classList.toggle('is-agree', btn.dataset.vote === 'agree' && current === 'agree');
      btn.classList.toggle('is-disagree', btn.dataset.vote === 'disagree' && current === 'disagree');
    });
  }
  async function onFeedbackVote(btn) {
    const group = btn.closest('.feedback-votes');
    if (!group || group.dataset.busy === '1') return;
    const vote = btn.dataset.vote;
    const domain = group.dataset.feedbackDomain;
    const target = group.dataset.feedbackTarget;
    const findingId = group.dataset.feedbackId || '';
    const run = getRunFromPath();
    if (!run || !domain || !target || !vote) return;
    let comment = '';
    if (vote === 'disagree') {
      const typed = await LL.ui.prompt({
        title: target === 'verdict' ? 'Почему вердикт неверен?' : 'Почему находка неверна?',
        message: 'Короткий комментарий поможет сравнить правки промптов.',
        label: 'Комментарий',
        confirmText: 'Сохранить',
        cancelText: 'Отмена',
        multiline: true
      });
      if (typed === null) return;
      comment = String(typed || '').trim();
    }
    group.dataset.busy = '1';
    try {
      const resp = await fetch('/llm_feedback', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_name: run, domain, target, finding_id: findingId, vote, comment })
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error((data && data.error) || 'Не удалось сохранить оценку');
      setVote(domain, target, findingId, (data && data.vote) || vote, (data && data.comment) || comment);
      document.querySelectorAll('.feedback-votes').forEach(paintVoteGroup);
    } catch (e) {
      LL.ui.alert({ title: 'Оценка не сохранена', message: e.message || String(e) });
    } finally {
      delete group.dataset.busy;
    }
  }
  function bindFeedbackVotes(root) {
    if (!root) return;
    root.querySelectorAll('.feedback-vote').forEach((btn) => {
      btn.addEventListener('click', () => onFeedbackVote(btn));
    });
  }
  function formatTokenCount(n) {
    if (n === null || n === undefined || n === '') return null;
    const v = Number(n);
    if (!Number.isFinite(v)) return null;
    if (v >= 1000) {
      const k = v / 1000;
      const text = k >= 10 ? String(Math.round(k)) : k.toFixed(1).replace(/\.0$/, '');
      return `${text}k`;
    }
    return String(Math.round(v));
  }
  function collectRunUsage(finalRow, allRows) {
    const scores = parseJsonish(finalRow && finalRow.scores) || {};
    const run = scores.run_usage || {};
    let calls = Number(run.calls);
    let prompt = run.prompt_tokens;
    let completion = run.completion_tokens;
    if (!Number.isFinite(calls) || calls <= 0) {
      calls = 0;
      prompt = null;
      completion = null;
      (Array.isArray(allRows) && allRows.length ? allRows : [finalRow]).forEach((row) => {
        const sc = parseJsonish(row && row.scores) || {};
        const u = sc.run_usage || sc.usage || {};
        calls += Number(u.calls) || 0;
        if (u.prompt_tokens != null && Number.isFinite(Number(u.prompt_tokens))) prompt = (prompt || 0) + Number(u.prompt_tokens);
        if (u.completion_tokens != null && Number.isFinite(Number(u.completion_tokens))) completion = (completion || 0) + Number(u.completion_tokens);
      });
    }
    return { calls, prompt, completion };
  }
  function formatUsageLine(usage) {
    if (!usage || !Number.isFinite(Number(usage.calls)) || Number(usage.calls) <= 0) return '';
    const prompt = formatTokenCount(usage.prompt);
    const completion = formatTokenCount(usage.completion);
    if (prompt != null && completion != null) return `${usage.calls} вызовов · ${prompt} / ${completion} токенов`;
    return `${usage.calls} вызовов`;
  }
  function pluralFindings(count) {
    const mod10 = count % 10;
    const mod100 = count % 100;
    if (mod10 === 1 && mod100 !== 11) return 'находка';
    if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return 'находки';
    return 'находок';
  }
  function isUnverifiedFinding(item) {
    const verification = item && item.verification;
    return !!(verification && typeof verification === 'object' && verification.status === 'unverified');
  }
  function renderUnverifiedPill(item) {
    if (!isUnverifiedFinding(item)) return '';
    const unmatched = Array.isArray(item.verification.unmatched) ? item.verification.unmatched.map(safeStr).filter(Boolean) : [];
    const tip = `${(LL.GLOSSARY && LL.GLOSSARY.unverified) || ''}${unmatched.length ? ` Не найдены в данных: ${unmatched.join(', ')}.` : ''}`.trim();
    return `<abbr class="term" data-term="unverified" data-tip="${escapeHtml(tip)}" title="${escapeHtml(tip)}" tabindex="0" style="text-decoration:none"><span class="pill warn finding-unverified">Не подтверждено данными</span></abbr>`;
  }


  let reportRef = { run_id: '', run_name: '', service: '' };

  function reportIdFromPath() {
    try {
      const parts = location.pathname.split('/').filter(Boolean);
      const idx = parts.indexOf('reports');
      if (idx >= 0 && parts.length === idx + 2) return decodeURIComponent(parts[idx + 1]);
    } catch (e) {}
    return '';
  }

  function getRunFromPath() {
    return reportRef.run_name || '';
  }

  function getServiceFromPath() {
    return reportRef.service || '';
  }

  async function loadReportRef() {
    const id = reportIdFromPath();
    if (!id) return;
    const resp = await fetch('/report_ref/' + encodeURIComponent(id));
    if (!resp.ok) return;
    const data = await resp.json();
    if (data && data.run_name) reportRef = data;
  }

  function setPageTitle(run) {
    const title = document.getElementById('pageTitle');
    if (title) title.textContent = run ? `Отчет по тесту ${run}` : 'Отчет по тесту';
    document.title = run ? `Отчёт: ${run}` : 'Отчёт';
    renderBreadcrumbs(getServiceFromPath(), run);
  }

  function renderBreadcrumbs(service, run) {
    const nav = document.getElementById('reportBreadcrumbs');
    if (!nav) return;
    const parts = ['<a href="/reports">Архив</a>'];
    if (service) parts.push(`<span class="sep">›</span><a href="/reports?service=${encodeURIComponent(service)}">${escapeHtml(service)}</a>`);
    if (run) parts.push(`<span class="sep">›</span><span class="current">${escapeHtml(run)}</span>`);
    nav.innerHTML = parts.join('');
  }

  function wireRenameReport() {
    const btn = document.getElementById('renameReport');
    const form = document.getElementById('renameReportForm');
    const input = document.getElementById('renameReportInput');
    const save = document.getElementById('renameReportSave');
    const cancel = document.getElementById('renameReportCancel');
    const status = document.getElementById('renameReportStatus');
    if (!btn || !form || !input || !save || !cancel) return;

    function openForm() {
      input.value = getRunFromPath() || '';
      form.style.display = 'flex';
      btn.style.display = 'none';
      if (status) status.textContent = '';
      input.focus();
      input.select();
    }

    function closeForm() {
      form.style.display = 'none';
      btn.style.display = '';
      if (status) status.textContent = '';
    }

    async function submit() {
      const run = getRunFromPath();
      const next = String(input.value || '').trim();
      if (!run) {
        if (status) status.textContent = 'Не удалось определить запуск';
        return;
      }
      if (!next) {
        if (status) status.textContent = 'Название не может быть пустым';
        return;
      }
      save.disabled = true;
      if (status) status.textContent = 'Сохранение…';
      try {
        const resp = await fetch('/runs/' + encodeURIComponent(run), {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            new_run_name: next,
            service: getServiceFromPath()
          })
        });
        const data = await resp.json();
        if (!resp.ok) {
          if (status) status.textContent = data.error || 'Ошибка переименования';
          save.disabled = false;
          return;
        }
        const url = data.page_url || (reportRef.run_id ? ('/reports/' + encodeURIComponent(reportRef.run_id)) : location.pathname);
        LL.ui.toast(`Отчёт переименован в «${data.run_name}»`, { tone: 'ok' });
        location.replace(url);
      } catch (e) {
        if (status) status.textContent = 'Ошибка переименования';
        save.disabled = false;
      }
    }

    btn.addEventListener('click', openForm);
    cancel.addEventListener('click', closeForm);
    save.addEventListener('click', submit);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        submit();
      } else if (e.key === 'Escape') {
        closeForm();
      }
    });
  }

  function setConfluenceLink(url) {
    const link = document.getElementById('confluencePageLink');
    const btn = document.getElementById('publishConfluence');
    if (!link) return;
    if (url) {
      link.href = url;
      link.style.display = '';
      if (btn) btn.textContent = 'Обновить отчет в Confluence';
    } else {
      link.style.display = 'none';
      if (btn) btn.textContent = 'Добавить отчет в Confluence';
    }
  }

  async function loadConfluencePublication() {
    const run = getRunFromPath();
    if (!run) return;
    try {
      const resp = await fetch('/confluence_publication?run_name=' + encodeURIComponent(run));
      const data = await resp.json();
      if (resp.ok && data && data.page_url) setConfluenceLink(data.page_url);
    } catch (e) {}
  }

  async function pollConfluenceJob(jobId) {
    const statusEl = document.getElementById('confluenceStatus');
    const btn = document.getElementById('publishConfluence');
    for (let i = 0; i < 180; i += 1) {
      const resp = await fetch('/job_status/' + encodeURIComponent(jobId));
      const job = await resp.json();
      if (statusEl) {
        const pct = Number(job.progress || 0);
        statusEl.textContent = (job.message || 'Публикация…') + (pct ? ` (${pct}%)` : '');
      }
      if (job.status === 'done') {
        setConfluenceLink(job.page_url || '');
        if (statusEl) statusEl.textContent = 'Опубликовано';
        if (btn) btn.disabled = false;
        LL.ui.toast('Отчёт опубликован в Confluence', { tone: 'ok' });
        setTimeout(() => { if (statusEl && statusEl.textContent === 'Опубликовано') statusEl.textContent = ''; }, 2500);
        return;
      }
      if (job.status === 'error' || job.status === 'not_found') {
        const headline = job.message || 'Ошибка публикации';
        const text = job.error ? `${headline}: ${job.error}` : headline;
        if (statusEl) statusEl.textContent = text;
        LL.ui.toast(text, { tone: 'error' });
        if (btn) btn.disabled = false;
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 1500));
    }
    if (statusEl) statusEl.textContent = 'Публикация занимает слишком долго, проверьте статус позже';
    if (btn) btn.disabled = false;
  }

  function wireConfluencePublish() {
    const btn = document.getElementById('publishConfluence');
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const run = getRunFromPath();
      const statusEl = document.getElementById('confluenceStatus');
      if (!run) {
        if (statusEl) statusEl.textContent = 'Не удалось определить запуск';
        return;
      }
      btn.disabled = true;
      if (statusEl) statusEl.textContent = 'Публикация…';
      try {
        const resp = await fetch('/publish_confluence', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            run_name: run,
            service: getServiceFromPath(),
            source_url: window.location.href
          })
        });
        const data = await resp.json();
        if (!resp.ok || !data.job_id) {
          if (statusEl) statusEl.textContent = data.error || data.message || 'Ошибка запуска публикации';
          btn.disabled = false;
          return;
        }
        await pollConfluenceJob(data.job_id);
      } catch (e) {
        if (statusEl) statusEl.textContent = 'Ошибка запуска публикации';
        btn.disabled = false;
      }
    });
  }

  async function reportsLoadSchema(run) {
    try {
      const u = new URL('/domains_schema', location.origin);
      if (run) u.searchParams.set('run_name', run);
      const r = await fetch(u);
      return await r.json();
    } catch (e) {
      return {};
    }
  }

  function safeStr(x) { return (x === undefined || x === null) ? '' : String(x).trim(); }
  function pct(x) { try { if (x === undefined || x === null) return '—'; const v = Number(x); if (!isFinite(v)) return '—'; return `${Math.round(v * 100)}%`; } catch (e) { return '—'; } }
  // JSONB columns arrive as objects; legacy rows may hold JSON text.
  function parseJsonish(value) {
    if (!value) return null;
    if (typeof value === 'object') return value;
    if (typeof value === 'string') {
      try {
        const parsed = JSON.parse(value);
        return (parsed && typeof parsed === 'object') ? parsed : null;
      } catch (e) {
        return null;
      }
    }
    return null;
  }
  function parseSystemContextValue(value) {
    return parseJsonish(value);
  }
  function hasMeaningfulSystemContext(value) {
    if (typeof value === 'string') return !!value.trim();
    if (Array.isArray(value)) return value.some((item) => hasMeaningfulSystemContext(item));
    if (value && typeof value === 'object') {
      if (value.enabled === false) return false;
      return Object.keys(value).some((key) => !['schema_version', 'enabled'].includes(key) && hasMeaningfulSystemContext(value[key]));
    }
    return false;
  }
  function systemContextMarkdown(ctx) {
    if (!ctx || typeof ctx !== 'object') return '';
    const system = (ctx.system && typeof ctx.system === 'object') ? ctx.system : {};
    const architecture = (ctx.architecture && typeof ctx.architecture === 'object') ? ctx.architecture : {};
    const loadModel = (ctx.load_model && typeof ctx.load_model === 'object') ? ctx.load_model : {};
    const operational = (ctx.operational_context && typeof ctx.operational_context === 'object') ? ctx.operational_context : {};
    const lines = [];
    const sysBits = [];
    if (safeStr(system.name)) sysBits.push(`**Система:** ${safeStr(system.name)}`);
    if (safeStr(system.domain)) sysBits.push(`**Домен:** ${safeStr(system.domain)}`);
    if (safeStr(system.description)) sysBits.push(safeStr(system.description));
    if (safeStr(system.test_goal)) sysBits.push(`**Цель теста:** ${safeStr(system.test_goal)}`);
    if (sysBits.length) {
      lines.push('');
      sysBits.forEach((item) => lines.push(`- ${item}`));
    }
    if (safeStr(architecture.style)) {
      lines.push('');
      lines.push(`- **Архитектура:** ${safeStr(architecture.style)}`);
    }
    const components = Array.isArray(architecture.components) ? architecture.components : [];
    if (components.length) {
      lines.push('');
      lines.push('#### Компоненты');
      components.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const name = safeStr(item.name || item.id);
        if (!name) return;
        const bits = [];
        if (safeStr(item.role)) bits.push(`роль: ${safeStr(item.role)}`);
        if (safeStr(item.criticality)) bits.push(`criticality: ${safeStr(item.criticality)}`);
        const technologies = Array.isArray(item.technologies) ? item.technologies.map((x) => safeStr(x)).filter(Boolean) : [];
        if (technologies.length) bits.push(`tech: ${technologies.join(', ')}`);
        lines.push(bits.length ? `- **${name}** (${bits.join('; ')})` : `- **${name}**`);
      });
    }
    const deps = Array.isArray(architecture.dependencies) ? architecture.dependencies : [];
    if (deps.length) {
      lines.push('');
      lines.push('#### Зависимости');
      deps.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const from = safeStr(item.from);
        const to = safeStr(item.to);
        if (!from && !to) return;
        const kind = safeStr(item.kind);
        const purpose = safeStr(item.purpose);
        const suffix = [kind && `тип: ${kind}`, purpose && purpose].filter(Boolean).join('; ');
        lines.push(`- ${from || 'unknown'} -> ${to || 'unknown'}${suffix ? ` (${suffix})` : ''}`);
      });
    }
    const flows = Array.isArray(loadModel.critical_user_flows) ? loadModel.critical_user_flows : [];
    if (flows.length) {
      lines.push('');
      lines.push('#### Критичные потоки');
      flows.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const name = safeStr(item.name || item.id);
        if (!name) return;
        const steps = Array.isArray(item.steps) ? item.steps.map((x) => safeStr(x)).filter(Boolean) : [];
        const successSignals = Array.isArray(item.success_signals) ? item.success_signals.map((x) => safeStr(x)).filter(Boolean) : [];
        lines.push(`- **${name}**${steps.length ? `: ${steps.join(' -> ')}` : ''}`);
        if (successSignals.length) lines.push(`  Успех: ${successSignals.join(', ')}`);
      });
    }
    const entrypoints = Array.isArray(loadModel.entrypoints) ? loadModel.entrypoints : [];
    if (entrypoints.length) {
      lines.push('');
      lines.push('#### Точки входа');
      entrypoints.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const name = safeStr(item.name || item.id);
        if (!name) return;
        const kind = safeStr(item.kind);
        const priority = safeStr(item.business_priority);
        const suffix = [kind && `тип: ${kind}`, priority && `priority: ${priority}`].filter(Boolean).join('; ');
        lines.push(`- ${name}${suffix ? ` (${suffix})` : ''}`);
      });
    }
    const hotspots = Array.isArray(loadModel.expected_hotspots) ? loadModel.expected_hotspots.map((x) => safeStr(x)).filter(Boolean) : [];
    if (hotspots.length) {
      lines.push('');
      lines.push(`- **Ожидаемые hotspots:** ${hotspots.join(', ')}`);
    }
    const pushSimpleList = (title, items) => {
      const clean = Array.isArray(items) ? items.map((x) => safeStr(x)).filter(Boolean) : [];
      if (!clean.length) return;
      lines.push('');
      lines.push(`#### ${title}`);
      clean.forEach((item) => lines.push(`- ${item}`));
    };
    pushSimpleList('Известные ограничения', operational.known_constraints);
    pushSimpleList('Известные риски', operational.known_risks);
    pushSimpleList('Нормальные правила деградации', operational.normal_degradation_rules);
    pushSimpleList('Фокус анализа', operational.analysis_focus);
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
  // Model output and engineer notes are untrusted: only formatting survives. No images (they would
  // load from any host and leak data), styles, forms or event handlers; links to http(s), mailto,
  // this site or an anchor. Without DOMPurify the text is shown escaped, never as markup.
  const SAFE_URI = new RegExp('^(?:https?:|mailto:|/(?!/)|#)', 'i');
  const MARKDOWN_PURIFY = {
    ALLOWED_TAGS: ['p', 'br', 'b', 'strong', 'i', 'em', 'u', 's', 'del', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'blockquote', 'pre', 'code', 'hr', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'a', 'span'],
    ALLOWED_ATTR: ['href', 'title', 'align'],
    ALLOWED_URI_REGEXP: SAFE_URI,
  };
  const NOTE_PURIFY = {
    ALLOWED_TAGS: ['p', 'div', 'br', 'b', 'strong', 'i', 'em', 'u', 's', 'ul', 'ol', 'li', 'h3', 'h4', 'blockquote', 'pre', 'code', 'a'],
    ALLOWED_ATTR: ['href', 'rel'],
    ALLOWED_URI_REGEXP: SAFE_URI,
  };
  function sanitizeOrEscape(html, config, fallbackText) {
    try {
      if (window.DOMPurify && typeof window.DOMPurify.sanitize === 'function') return window.DOMPurify.sanitize(html, config);
    } catch (e) {}
    return escapeHtml(fallbackText);
  }
  function markdownToSafeHtml(md) {
    const source = safeStr(md);
    if (!source) return '';
    if (!(window.marked && typeof marked.parse === 'function')) return escapeHtml(source).replace(/\n/g, '<br>');
    return sanitizeOrEscape(window.marked.parse(source), MARKDOWN_PURIFY, source);
  }
  // A response that looks like JSON but does not parse: show a clear error instead of raw text.
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
      '- Проверьте лимит токенов, prompt и повторите генерацию.'
    ];
    if (excerpt) {
      lines.push('');
      lines.push('```json');
      lines.push(excerpt);
      lines.push('```');
    }
    return lines.join('\n');
  }
  function verdictTone(verdict) {
    const text = standardizeVerdict(verdict);
    if (text === 'Успешно') return { text, className: 'report-verdict-success' };
    if (text === 'Есть риски') return { text, className: 'report-verdict-risk' };
    if (text === 'Провал') return { text, className: 'report-verdict-fail' };
    return { text, className: 'report-verdict-na' };
  }
  // ---- manual verdict ----------------------------------------------------------------------------
  const VERDICT_CHOICES = ['Успешно', 'Есть риски', 'Провал', 'Недостаточно данных'];
  function manualOf(row) {
    const manual = row && row.manual_verdict;
    return manual && manual.verdict ? manual : null;
  }
  // The line under the verdict pill: who set the verdict and from what.
  function verdictSourceText(finalRow) {
    const row = finalRow || {};
    const manual = manualOf(row);
    if (manual) {
      const who = [safeStr(manual.author), manual.changed_at ? LL.formatDateTime(manual.changed_at) : ''].filter(Boolean).join(', ');
      const auto = standardizeVerdict(row.auto_verdict || row.sla_verdict || row.llm_verdict);
      return `Изменён вручную${who ? `: ${who}` : ''}. Автоматически: ${auto}.`;
    }
    const sla = safeStr(row.sla_verdict);
    const llm = safeStr(row.llm_verdict);
    if (sla) {
      // Reports built before the SLA verdict replaced the model's one keep a different model verdict.
      return llm && standardizeVerdict(llm) !== standardizeVerdict(sla)
        ? `Определён по SLA-критериям. Оценка ИИ: ${standardizeVerdict(llm)}.`
        : 'Определён по SLA-критериям.';
    }
    return standardizeVerdict(row.auto_verdict || row.verdict) === 'Недостаточно данных'
      ? 'SLA-критерии не заданы, поэтому автоматический вердикт — «Недостаточно данных». Задайте SLA сервиса или поставьте вердикт вручную.'
      : 'SLA-критерии не заданы: вердикт определён оценкой ИИ.';
  }
  function verdictHistoryItem(entry) {
    const e = entry || {};
    const head = [e.changed_at ? LL.formatDateTime(e.changed_at) : '', safeStr(e.author)].filter(Boolean).join(' · ');
    const change = e.verdict
      ? `${standardizeVerdict(e.previous_verdict)} → ${standardizeVerdict(e.verdict)}`
      : `возвращён автоматический: ${standardizeVerdict(e.auto_verdict)}`;
    const reason = safeStr(e.reason);
    return `<li>${head ? `<span class="dim">${escapeHtml(head)}</span> ` : ''}${escapeHtml(change)}`
      + `${reason ? `<div class="verdict-history-reason">${escapeHtml(reason)}</div>` : ''}</li>`;
  }
  function verdictHistoryHtml(history) {
    const items = Array.isArray(history) ? history : [];
    if (!items.length) return '';
    return `<details class="verdict-history"><summary>История изменений (${items.length})</summary>`
      + `<ul>${items.map(verdictHistoryItem).join('')}</ul></details>`;
  }
  // Body of the «Изменить вердикт» dialog from GET /runs/<run>/verdict.
  function verdictDialogHtml(state) {
    const s = state || {};
    const current = standardizeVerdict(s.verdict);
    const auto = standardizeVerdict(s.automatic);
    const max = Number(s.reason_max) || 1000;
    const choices = (Array.isArray(s.choices) && s.choices.length ? s.choices : VERDICT_CHOICES).map((v) => {
      const same = v === current;
      return `<label class="verdict-choice${same ? ' is-current' : ''}"><input type="radio" name="verdictChoice" value="${escapeHtml(v)}"${same ? ' disabled' : ''}>`
        + `<span class="report-verdict-pill ${verdictTone(v).className}">${escapeHtml(v)}</span>${same ? '<span class="dim">сейчас</span>' : ''}</label>`;
    }).join('');
    const now = s.manual
      ? `Сейчас «${current}», изменён вручную; автоматический — «${auto}».`
      : `Сейчас «${current}», определён автоматически.`;
    return [
      '<div class="verdict-dialog">',
      `<p class="muted">${escapeHtml(now)} Ручной вердикт действует на дашборде, в архиве, сравнении и при выборе базового прогона; анализ отчёта не меняется.</p>`,
      `<fieldset class="verdict-choices"><legend>Новый вердикт</legend>${choices}</fieldset>`,
      `<label class="verdict-reason"><span>Причина</span><textarea name="verdictReason" rows="3" maxlength="${max}" placeholder="Например: провал из-за сбоя генератора, сервис держал целевую нагрузку"></textarea></label>`,
      '<div class="dialog-error" data-verdict-error role="alert"></div>',
      verdictHistoryHtml(s.history),
      '</div>'
    ].join('');
  }
  function verdictFormProblem(verdict, reason, state) {
    if (!verdict) return 'Выберите новый вердикт';
    if (!reason) return 'Укажите причину: её увидят в отчёте, архиве и Confluence';
    const max = Number(state && state.reason_max) || 1000;
    if (reason.length > max) return `Причина — не длиннее ${max} символов`;
    return '';
  }
  async function verdictRequest(method, body) {
    const url = `/runs/${encodeURIComponent(getRunFromPath())}/verdict`;
    const init = method === 'GET' ? {} : { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) };
    const resp = await fetch(url, init);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    return data;
  }
  // A changed verdict redraws the hero; a published page keeps the old one until it is updated.
  function applyVerdictState(state, doneText) {
    const row = reportState.finalRow;
    if (!row) return;
    row.manual_verdict = state.manual || null;
    row.verdict = state.verdict;
    row.auto_verdict = state.automatic;
    renderHero(row, reportState.rows || [row]);
    const link = document.getElementById('confluencePageLink');
    const published = !!(link && link.style.display !== 'none' && link.getAttribute('href') && link.getAttribute('href') !== '#');
    if (published) {
      LL.ui.toast(`${doneText}. Страница в Confluence показывает прежний вердикт — нажмите «Обновить отчет в Confluence».`, { tone: 'info', timeout: 10000 });
    } else {
      LL.ui.toast(doneText, { tone: 'ok' });
    }
  }
  async function openVerdictDialog() {
    let state;
    try {
      state = await verdictRequest('GET');
    } catch (e) {
      LL.ui.toast(`Вердикт не загрузился: ${e.message}`, { tone: 'error' });
      return;
    }
    let busy = false;
    await LL.ui.dialog({
      title: 'Изменить вердикт',
      bodyHtml: verdictDialogHtml(state),
      // A message about a missing choice or reason goes away as soon as the form changes.
      onOpen: (body) => {
        const error = body.querySelector('[data-verdict-error]');
        body.addEventListener('input', () => { error.textContent = ''; });
        body.addEventListener('change', () => { error.textContent = ''; });
      },
      actions: [
        { label: 'Отмена' },
        {
          label: 'Сохранить',
          primary: true,
          onClick: async (close, body) => {
            if (busy) return;
            const picked = body.querySelector('input[name="verdictChoice"]:checked');
            const reason = (body.querySelector('textarea[name="verdictReason"]').value || '').trim();
            const error = body.querySelector('[data-verdict-error]');
            const problem = verdictFormProblem(picked ? picked.value : '', reason, state);
            error.textContent = problem;
            if (problem) return;
            busy = true;
            try {
              const next = await verdictRequest('PUT', { verdict: picked.value, reason });
              close(next);
              applyVerdictState(next, 'Вердикт изменён');
            } catch (e) {
              error.textContent = e.message;
            } finally {
              busy = false;
            }
          }
        }
      ]
    });
  }
  async function resetVerdict() {
    const row = reportState.finalRow || {};
    const ok = await LL.ui.confirm({
      title: 'Вернуть автоматический вердикт',
      message: `Отчёт снова покажет «${standardizeVerdict(row.auto_verdict)}» — вердикт по SLA. История изменений сохранится.`,
      confirmText: 'Вернуть'
    });
    if (!ok) return;
    try {
      applyVerdictState(await verdictRequest('DELETE', {}), 'Возвращён автоматический вердикт');
    } catch (e) {
      LL.ui.toast(`Не удалось вернуть вердикт: ${e.message}`, { tone: 'error' });
    }
  }
  function bindVerdictActions(root) {
    root.querySelectorAll('[data-verdict-action]').forEach((btn) => {
      btn.addEventListener('click', () => (btn.dataset.verdictAction === 'reset' ? resetVerdict() : openVerdictDialog()));
    });
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
  function formatFindingWindow(item) {
    const start = safeStr(item && (item.start_time || item.start || item.window_start || item.from));
    const end = safeStr(item && (item.end_time || item.end || item.window_end || item.to));
    if (start && end) return `${start} - ${end}`;
    return start || end || '';
  }
  function normalizeEvidenceItems(value) {
    const items = Array.isArray(value) ? value : (value && typeof value === 'object' ? [value] : []);
    return items
      .map((item) => {
        if (!item || typeof item !== 'object') return null;
        const metric = safeStr(item.metric || item.name || item.label);
        const observedValue = safeStr(item.observed_value || item.value || item.actual);
        const threshold = safeStr(item.threshold || item.limit || item.baseline);
        const note = safeStr(item.note || item.details || item.evidence);
        return (metric || observedValue || threshold || note) ? {
          metric, observedValue, threshold, note
        } : null;
      })
      .filter(Boolean);
  }
  function renderMetaGrid(rows) {
    const items = (Array.isArray(rows) ? rows : [])
      .filter((row) => row && safeStr(row.value))
      .map((row) => [
        '<div class="report-item-meta-entry">',
        `<span class="report-item-meta-label">${escapeHtml(row.label || '')}</span>`,
        `<span class="report-item-meta-value">${escapeHtml(row.value || '')}</span>`,
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
  function renderFindingDetails(item, domain) {
    if (!item || typeof item !== 'object') return escapeHtml(safeStr(item));
    const summary = safeStr(item.summary || item.title || item.text);
    const metaHtml = renderMetaGrid([
      { label: 'Критичность', value: safeStr(item.severity) },
      { label: 'Компонент', value: safeStr(item.component) }
    ]);
    const evidenceHtml = renderFindingEvidence(item);
    const findingId = normalizeFindingLinkId(item.id || item.finding_id || item.key, '');
    const voteHtml = (domain && findingId) ? renderVoteGroup(domain, 'finding', findingId, 'Верно', 'Неверно') : '';
    return [
      `<div class="report-item-card${isUnverifiedFinding(item) ? ' report-item-unverified' : ''}">`,
      `<div class="report-item-summary">${escapeHtml(summary || '—')} ${renderUnverifiedPill(item)}</div>`,
      evidenceHtml,
      metaHtml,
      voteHtml,
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
    items.forEach((item) => {
      const normalized = normalizeFindingLinkId(item, '');
      if (normalized && ids.indexOf(normalized) < 0) ids.push(normalized);
    });
    return ids;
  }
  function renderActionTexts(texts, emptyText) {
    const values = (Array.isArray(texts) ? texts : []).map((item) => safeStr(item)).filter(Boolean);
    if (!values.length) return `<span class="report-empty">${escapeHtml(emptyText)}</span>`;
    if (values.length === 1) return escapeHtml(values[0]);
    return `<ul class="report-list">${values.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`;
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
  function pairFindingsWithActions(findings, actions, domain) {
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
        problemHtml: renderFindingDetails(finding.item, domain),
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
  function renderAnalysisError(report) {
    const error = report && report.analysis_error;
    if (!error || typeof error !== 'object') return '';
    const message = safeStr(error.message) || 'Анализ ИИ не выполнен.';
    const bits = [];
    if (error.provider) bits.push(String(error.provider));
    if (error.max_tokens) bits.push('лимит ' + error.max_tokens);
    if (error.output_tokens != null && error.output_tokens !== '') bits.push('выдано ' + error.output_tokens);
    const meta = bits.length ? `<p class="dim">${escapeHtml(bits.join(' · '))}</p>` : '';
    const excerpt = error.excerpt ? `<pre>${escapeHtml(String(error.excerpt))}</pre>` : '';
    return `<div class="report-note"><div class="report-note-title">Анализ ИИ не выполнен</div><div class="report-note-body"><p>${escapeHtml(message)}</p>${meta}${excerpt}</div></div>`;
  }
  function peakTableApplies(domain, peak) {
    if (domain !== 'final' && domain !== 'lt_framework') return false;
    if (!peak || peak.not_applicable || peak.notApplicable) return false;
    return Boolean(safeStr(peak.max_rps) || safeStr(peak.max_time) || safeStr(peak.drop_time));
  }
  // Test-type section of the final report; mirrors AI/report_sections.py.
  function reportMode(report) {
    const r = report || {};
    if (r.spike_analysis && typeof r.spike_analysis === 'object') return 'spike';
    if (r.saturation && typeof r.saturation === 'object') return 'stress';
    if (r.stability_under_load && typeof r.stability_under_load === 'object') return 'stability';
    const profile = (r.test_profile && typeof r.test_profile === 'object') ? r.test_profile : {};
    const mode = safeStr(profile.mode).toLowerCase();
    if (mode) return mode;
    const peak = r.peak_performance || r.peak_perfomance || {};
    return (peak.not_applicable || peak.notApplicable) ? 'stability' : 'capacity';
  }
  function normalizedTestType(report, fallback) {
    const profile = ((report || {}).test_profile && typeof report.test_profile === 'object') ? report.test_profile : {};
    const explicit = safeStr(profile.normalized_test_type).toLowerCase();
    if (explicit) return explicit;
    const mode = reportMode(report);
    const byMode = { stability: 'soak', spike: 'spike', stress: 'stress' }[mode];
    if (byMode) return byMode;
    const raw = safeStr(fallback || profile.test_type).toLowerCase();
    return LL.TEST_TYPE_LABELS && LL.TEST_TYPE_LABELS[raw] ? raw : 'step';
  }
  function hhmm(value) {
    return hhmmFromIso(value) || '—';
  }
  function signed(value, unit) {
    const n = Number(value);
    if (value === null || value === undefined || value === '' || !Number.isFinite(n)) return '';
    return (n > 0 ? '+' : '') + formatNumber(n) + unit;
  }
  const TREND_TITLES = { rps: 'RPS', p95: 'p95 latency', error_rate: 'доля ошибок', cpu: 'CPU', memory: 'память' };
  const RESOURCE_TITLES = { cpu: 'CPU', memory: 'память' };
  function stabilityRows(report) {
    const st = report.stability_under_load || {};
    const profile = report.test_profile || {};
    const rows = [
      { label: 'Тип теста', value: safeStr(profile.label_ru || profile.test_type) || 'stability/soak' },
      { label: 'Фокус оценки', value: safeStr(st.focus || profile.focus) || 'Удержание нагрузки без накопления деградации' },
      { label: 'Целевой RPS', value: formatNumber(st.target_rps) },
      { label: 'Фактический RPS', value: formatNumber(st.actual_rps) }
    ];
    const hold = (st.hold && typeof st.hold === 'object') ? st.hold : null;
    if (hold) {
      rows.push({ label: 'Окно удержания', value: hhmm(hold.start_iso) + '–' + hhmm(hold.end_iso) + (hold.duration_min != null ? ` (${formatNumber(hold.duration_min)} мин)` : '') });
      if (hold.min_bucket_rps != null) {
        rows.push({ label: 'Минимальный интервал', value: `${safeStr(hold.min_bucket_label) || 'интервал'}: ${formatNumber(hold.min_bucket_rps)} RPS` + (hold.drop_pct ? ` (−${formatNumber(hold.drop_pct)} % от среднего)` : '') });
      }
      if (hold.early_drop_iso) rows.push({ label: 'Падение RPS до конца теста', value: hhmm(hold.early_drop_iso) });
      if (safeStr(hold.note)) rows.push({ label: 'Примечание', value: safeStr(hold.note) });
    }
    rows.push({ label: 'Итог SLA', value: safeStr(st.sla_summary) || '—' });
    return rows;
  }
  function spikeRows(report) {
    const sa = report.spike_analysis || {};
    const profile = report.test_profile || {};
    const rows = [{ label: 'Тип теста', value: safeStr(profile.label_ru || profile.test_type) || 'spike' }];
    if (sa.baseline_rps != null) rows.push({ label: 'Фоновая нагрузка', value: formatNumber(sa.baseline_rps) + ' RPS' });
    if (!sa.detected) {
      rows.push({ label: 'Всплеск', value: safeStr(sa.reason) || 'не обнаружен' });
    } else {
      if (sa.target_rps != null) {
        const reached = sa.target_reached === true ? ' — достигнут' : (sa.target_reached === false ? ' — не достигнут' : '');
        rows.push({ label: 'Целевой RPS всплеска', value: formatNumber(sa.target_rps) + reached });
      }
      (Array.isArray(sa.spikes) ? sa.spikes : []).forEach((spike) => {
        const pre = spike.pre || {};
        const during = spike.during || {};
        const bits = [`${hhmm(spike.start_iso)}–${hhmm(spike.end_iso)} (${formatNumber(spike.duration_sec)} с)`];
        if (pre.p95 != null && during.p95 != null) {
          const over = signed(spike.overshoot_p95_pct, ' %');
          bits.push(`p95 ${formatNumber(pre.p95)} → ${formatNumber(during.p95)}` + (over ? ` (${over})` : ''));
        }
        const errors = signed(spike.overshoot_error_rate_pp, ' п.п.');
        if (errors) bits.push('ошибки ' + errors);
        if (spike.recovered === true) bits.push(`восстановление за ${formatNumber(spike.recovery_time_sec)} с`);
        else if (spike.recovered === false) bits.push('задержки не вернулись к фону');
        if (safeStr(spike.note)) bits.push(safeStr(spike.note));
        rows.push({ label: safeStr(spike.label) || 'всплеск', value: bits.join(', ') });
      });
    }
    rows.push({ label: 'Итог SLA', value: safeStr(sa.sla_summary) || '—' });
    return rows;
  }
  function resourceText(resource) {
    const title = RESOURCE_TITLES[safeStr(resource.kind)] || safeStr(resource.kind);
    if (resource.evidence === 'threshold') {
      return `${title}: p95 ${formatNumber(resource.value)} > порог ${formatNumber(resource.threshold)}` + (safeStr(resource.series) ? `, серия «${safeStr(resource.series)}»` : '');
    }
    return `${title}: рост ${signed(resource.growth_pct, ' %')} от первой ступени (p95 ${formatNumber(resource.value)})`;
  }
  function saturationRows(report) {
    const sat = report.saturation || {};
    const peak = report.peak_performance || report.peak_perfomance || {};
    const rows = [{ label: 'Точка насыщения', value: formatNumber(sat.saturation_rps != null ? sat.saturation_rps : peak.max_rps) + ' RPS' }];
    if (safeStr(sat.saturation_step_label)) rows.push({ label: 'Ступень насыщения', value: safeStr(sat.saturation_step_label) });
    if (safeStr(peak.max_time)) rows.push({ label: 'Время пиковой производительности', value: safeStr(peak.max_time) });
    if (safeStr(sat.degraded_step_label) || sat.first_degraded_rps != null) {
      rows.push({ label: 'Первая деградировавшая ступень', value: safeStr(sat.degraded_step_label) || `≈${formatNumber(sat.first_degraded_rps)} RPS` });
    }
    const checks = Array.isArray(sat.degraded_checks) ? sat.degraded_checks : [];
    if (checks.length) rows.push({ label: 'Нарушенные критерии', value: checks.map((name) => SLA_CHECK_LABELS[name] || name).join(', ') });
    rows.push({ label: 'Характер деградации', value: safeStr(sat.failure_mode_title || sat.failure_mode) || '—' });
    rows.push({ label: 'Ограничивающий ресурс', value: (sat.limiting_resource && typeof sat.limiting_resource === 'object') ? resourceText(sat.limiting_resource) : 'не определён' });
    if (safeStr(peak.drop_time || sat.rps_drop_iso)) rows.push({ label: 'Время деградации', value: safeStr(peak.drop_time || sat.rps_drop_iso) });
    if (safeStr(sat.note)) rows.push({ label: 'Примечание', value: safeStr(sat.note) });
    return rows;
  }
  function testTypeSection(report) {
    const r = report || {};
    const mode = reportMode(r);
    if (mode === 'stability') return { title: 'Стабильность под нагрузкой', rows: stabilityRows(r) };
    if (mode === 'spike') return { title: 'Всплески нагрузки', rows: spikeRows(r) };
    if (mode === 'stress' && r.saturation && typeof r.saturation === 'object') return { title: 'Точка насыщения', rows: saturationRows(r) };
    return null;
  }
  function renderTrendsTable(report) {
    const st = (report || {}).stability_under_load || {};
    const trends = Array.isArray(st.trends) ? st.trends : [];
    if (!trends.length) return '';
    const rows = trends.map((trend) => [
      `<tr class="${trend.flagged ? 'is-unstable' : ''}">`,
      `<td>${escapeHtml(safeStr(trend.title) || TREND_TITLES[safeStr(trend.metric_kind)] || safeStr(trend.metric_kind))}${trend.flagged ? '<span class="step-flag">риск</span>' : ''}<div class="sla-check-note">${escapeHtml(safeStr(trend.label))}</div></td>`,
      `<td class="num">${escapeHtml(formatNumber(trend.first))}</td>`,
      `<td class="num">${escapeHtml(formatNumber(trend.last))}</td>`,
      `<td class="num">${escapeHtml(signed(trend.change_pct, ' %') || '—')}</td>`,
      `<td class="num">${escapeHtml(formatNumber(trend.slope_per_hour))}</td>`,
      '</tr>'
    ].join('')).join('');
    const threshold = st.trend_threshold_pct != null ? ` Риск — неблагоприятное изменение больше ${formatNumber(st.trend_threshold_pct)} %.` : '';
    return [
      '<section>',
      `<div class="report-block-title">${LL.ui.term('trend_drift', 'Тренды за окно удержания')}</div>`,
      '<div class="step-table-wrap"><table class="step-table">',
      '<thead><tr><th>Метрика</th><th class="num">Начало</th><th class="num">Конец</th><th class="num">Изменение</th><th class="num">Наклон в час</th></tr></thead>',
      `<tbody>${rows}</tbody></table></div>`,
      `<p class="step-note">Среднее первой и последней четверти окна удержания.${escapeHtml(threshold)}</p>`,
      '</section>'
    ].join('');
  }
  function renderTestTypeSection(report, domain) {
    if (domain !== 'final') return '';
    const section = testTypeSection(report);
    if (!section) return '';
    const rows = section.rows.map((row) => ({ label: row.label, html: escapeHtml(safeStr(row.value) || '—') }));
    rows._tableClass = 'report-aligned-value-table';
    return renderKeyValueTable(section.title, rows) + (reportMode(report) === 'stability' ? renderTrendsTable(report) : '');
  }
  function renderStructuredReportHtml(report, extraHtml, domain) {
    const analysisError = renderAnalysisError(report);
    if (analysisError) {
      return `<div class="report-stack">${analysisError}${extraHtml || ''}</div>`;
    }
    const verdict = verdictTone((report || {}).verdict || '');
    const verdictRationale = safeStr((report || {}).verdict_rationale || (report || {}).verdict_reason || (report || {}).rationale)
      .replace(/^([^\n]+)\n(?=- )/, '$1\n\n');
    const peak = ((report || {}).peak_performance || (report || {}).peak_perfomance || {});
    const findings = (report || {}).findings || [];
    const actions = (report || {}).recommended_actions || (report || {}).actions || [];
    const verdictRows = [
      {
        label: 'Вердикт ИИ',
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
    const typeSection = renderTestTypeSection(report, domain);
    const sections = [
      renderKeyValueTable('', verdictRows),
      renderRationaleBox(verdictRationale),
      typeSection || (peakTableApplies(domain, peak) ? renderKeyValueTable('Пиковая производительность', peakRows) : ''),
      renderProblemRecommendationRows(pairFindingsWithActions(findings, actions, domain)),
      extraHtml || ''
    ].filter(Boolean);
    return `<div class="report-stack">${sections.join('')}</div>`;
  }
  // Where the reliability level comes from: the judge, the data check alone (one answer), or the data
  // check and confidence because the judge gave no usable scores. Reports saved before that case was
  // handled counted a missing judge as 0; it is recognised here and the result is recomputed without it.
  function answerScoreState(scores) {
    const s = scores || {};
    const j = (s.judge && typeof s.judge === 'object') ? s.judge : {};
    const meta = (s.judge_meta && typeof s.judge_meta === 'object') ? s.judge_meta : {};
    const num = (x) => { const v = Number(x); return x === null || x === undefined || !Number.isFinite(v) ? null : v; };
    const skipped = Boolean(meta.skipped);
    const allZero = Object.keys(j).length > 0 && ['overall', 'factual', 'completeness', 'specificity'].every((k) => !(num(j[k]) > 0));
    const unavailable = !skipped && (Boolean(meta.unavailable) || Boolean(meta.error) || meta.scores_returned === 0 || allZero);
    const data = num(s.data_score);
    const conf = num(s.confidence);
    let final = num(s.final_score);
    // Same weights as AI/scoring.py: data 0.35 and confidence 0.05, renormalized without the judge.
    if (unavailable && !meta.unavailable && data !== null) final = (0.35 * data + 0.05 * (conf || 0)) / 0.4;
    if (skipped) final = data;
    let reason = '';
    if (unavailable) {
      if (meta.reason === 'partial_scores') reason = 'он оценил не все варианты ответа';
      else if (meta.reason === 'error' || meta.error) reason = 'вызов модели завершился ошибкой, подробности в журнале приложения';
      else reason = 'его ответ не удалось разобрать';
    }
    return { skipped, unavailable, reason, final };
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
      const dataScore = s.data_score, conf = s.confidence;
      const scoreState = answerScoreState(s);
      // One answer from the model: there was nothing to choose from, so the judge did not run.
      const judgeSkipped = scoreState.skipped;
      // The judge ran but gave no usable scores: the result rests on the data check and confidence.
      const judgeUnavailable = scoreState.unavailable;
      const finalScore = judgeUnavailable ? scoreState.final : s.final_score;
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
        if (judgeSkipped || judgeUnavailable) {
          const intro = judgeSkipped
            ? 'Модель дала один вариант ответа, поэтому судья не запускался: выбирать было не из чего.'
            : `Судья не дал оценку: ${scoreState.reason}. Итог посчитан по проверке данных и уверенности модели.`;
          if (data === null) return intro;
          // Same thresholds as the level in the section title.
          if (data >= 0.75) return `${intro} Проверка по данным хорошо подтверждает выводы.`;
          if (data >= 0.55) return `${intro} Проверка по данным в целом подтверждает выводы.`;
          return `${intro} Проверка по данным подтверждает выводы слабо: сверяйте их с графиками.`;
        }
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
      if (judgeSkipped) {
        lines.push('- Судья: не запускался, модель дала один вариант ответа');
        lines.push(`- Эвристическая проверка по данным: ${pct(dataScore)} (${level(dataScore)})`);
        if (typeof conf === 'number') lines.push(`- Уверенность модели: ${pct(conf)} (${level(conf)})`);
      } else if (judgeUnavailable) {
        lines.push('- Судья: не дал оценку');
        lines.push(`- Эвристическая проверка по данным: ${pct(dataScore)} (${level(dataScore)})`);
        lines.push(`- Итог без судьи: ${pct(finalScore)} (${level(finalScore)})`);
        if (typeof conf === 'number') lines.push(`- Уверенность модели: ${pct(conf)} (${level(conf)})`);
      } else {
        lines.push(`- Оценка текста судьей: ${pct(overall)} (${level(overall)})`);
        lines.push(`- Эвристическая проверка по данным: ${pct(dataScore)} (${level(dataScore)})`);
        lines.push(`- Итог для выбора кандидата: ${pct(finalScore)} (${level(finalScore)})`);
        if (typeof conf === 'number') lines.push(`- Уверенность модели: ${pct(conf)} (${level(conf)})`);
        lines.push('');
        lines.push('#### За что поставлена оценка');
        lines.push(`- Точность относительно данных: ${pct(factual)}`);
        lines.push(`- Полнота покрытия важных наблюдений: ${pct(completeness)}`);
        lines.push(`- Конкретика по метрикам и компонентам: ${pct(specificity)}`);
      }
      if (!judgeSkipped && !judgeUnavailable && Object.keys(rubric).length) {
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
  function confidenceLevelLabel(scores) {
    const s = scores || {};
    const judge = s.judge || {};
    const state = answerScoreState(s);
    const raw = state.final !== null ? state.final : Number(judge.overall);
    if (!Number.isFinite(raw) || raw <= 0) return '';
    if (raw >= 0.9) return 'очень высокая';
    if (raw >= 0.75) return 'высокая';
    if (raw >= 0.55) return 'умеренная';
    return 'низкая';
  }
  function answerSubtitle(scores) {
    const state = answerScoreState(scores);
    if (state.skipped) return 'по проверке данных, без судьи';
    if (state.unavailable) return 'по проверке данных: судья не дал оценку';
    return 'как выбран и проверен ответ ИИ';
  }
  function judgeSectionHtml(scores) {
    const md = judgeDetailsMarkdown(scores);
    if (!safeStr(md)) return '';
    const level = confidenceLevelLabel(scores);
    const titleHtml = level
      ? `${LL.ui.term('confidence', 'Достоверность ответа')}: ${escapeHtml(level)}`
      : LL.ui.term('confidence', 'Оценка ответа');
    return [
      '<details class="report-collapsible">',
      `<summary><span>${titleHtml}</span><small class="report-collapsible-subtitle">${answerSubtitle(scores)}</small></summary>`,
      `<div class="report-collapsible-body report-markdown-block">${markdownToSafeHtml(md)}</div>`,
      '</details>'
    ].join('');
  }

  const SLA_CHECK_LABELS = {
    target_rps: 'Целевой RPS',
    error_rate: 'Доля ошибок, %',
    p95_latency: 'P95 latency, мс',
    p99_latency: 'P99 latency, мс',
    cpu_usage: 'CPU, %',
    memory_usage: 'Память, %',
    recovery: 'Восстановление после всплеска, мин'
  };
  function formatNumber(value) {
    const n = Number(value);
    if (value === null || value === undefined || value === '' || !Number.isFinite(n)) return '—';
    return Number.isInteger(n) ? String(n) : n.toFixed(2).replace(/\.?0+$/, '');
  }
  // Mini bar of the «Факт» column: the track is 0…1.5× the threshold, the mark sits at the threshold
  // (2/3 of the track), the fill is the actual value colored by the check status. For target_rps a
  // longer bar is better; for the other checks a bar past the mark is a breach.
  const SLA_BAR_SPAN = 1.5;
  function slaBarModel(check) {
    const actual = finiteOrNull(check && check.actual);
    const threshold = finiteOrNull(check && check.threshold);
    if (actual === null || threshold === null || threshold <= 0 || actual < 0) return null;
    const ratio = Math.min(actual / threshold, SLA_BAR_SPAN);
    const passed = check.passed;
    return {
      fill: Math.round((ratio / SLA_BAR_SPAN) * 1000) / 10,
      mark: Math.round((1 / SLA_BAR_SPAN) * 1000) / 10,
      tone: passed === true ? 'ok' : (passed === false ? 'fail' : 'na'),
      higherBetter: safeStr(check.name) === 'target_rps',
      percent: Math.round((actual / threshold) * 100)
    };
  }
  function slaBarHtml(check) {
    const bar = slaBarModel(check);
    if (!bar) return '';
    const title = `${bar.percent} % от порога${bar.higherBetter ? ' (больше — лучше)' : ''}`;
    return `<div class="sla-bar ${bar.tone}${bar.higherBetter ? ' is-higher-better' : ''}" title="${escapeHtml(title)}" aria-hidden="true">`
      + `<span class="sla-bar-fill" style="width:${bar.fill}%"></span><span class="sla-bar-mark" style="left:${bar.mark}%"></span></div>`;
  }
  function renderSlaChecklist(finalRow) {
    const slaDetails = parseJsonish(finalRow.sla_details) || {};
    const checks = Array.isArray(slaDetails.checks) ? slaDetails.checks : [];
    if (!checks.length) {
      return [
        '<div class="hero-label">SLA-критерии</div>',
        '<div class="hero-empty">SLA-критерии для этого сервиса не заданы, поэтому вердикт определён оценкой ИИ. ',
        'Задать пороги можно в <a href="/settings">настройках</a> (раздел «SLA-критерии»).</div>'
      ].join('');
    }
    const rows = checks.map((check) => {
      const name = safeStr(check && check.name);
      const plainLabel = SLA_CHECK_LABELS[name] || name || '—';
      const termKey = name === 'target_rps' ? 'target_rps' : (name === 'error_rate' ? 'error_rate' : (name.startsWith('p95') ? 'p95' : ''));
      const label = termKey ? LL.ui.term(termKey, plainLabel) : escapeHtml(plainLabel);
      const passed = check ? check.passed : null;
      const statusClass = passed === true ? 'ok' : (passed === false ? 'fail' : 'na');
      const statusMark = passed === true ? '✓' : (passed === false ? '✗' : '—');
      const statusTitle = passed === true ? 'Порог соблюдён' : (passed === false ? 'Порог нарушен' : 'Нет данных для проверки');
      const note = safeStr(check && check.message);
      return [
        '<tr>',
        `<td><span class="sla-status ${statusClass}" title="${escapeHtml(statusTitle)}">${statusMark}</span></td>`,
        `<td>${label}${note ? `<div class="sla-check-note">${escapeHtml(note)}</div>` : ''}</td>`,
        `<td class="num">${escapeHtml(formatNumber(check && check.threshold))}</td>`,
        `<td class="num sla-actual">${escapeHtml(formatNumber(check && check.actual))}${slaBarHtml(check)}</td>`,
        '</tr>'
      ].join('');
    }).join('');
    const summary = safeStr(slaDetails.summary);
    return [
      '<div class="hero-label">SLA-критерии</div>',
      '<table class="sla-table"><thead><tr><th></th><th>Критерий</th><th class="num">Порог</th><th class="num">Факт</th></tr></thead>',
      `<tbody>${rows}</tbody></table>`,
      summary ? `<div class="hero-sub">${escapeHtml(summary)}</div>` : '',
      slaChecklistNote(finalRow)
    ].join('');
  }
  function renderHeroMeta(entries) {
    const items = entries
      .filter((entry) => entry && safeStr(entry.value))
      .map((entry) => [
        '<div class="hero-meta-entry">',
        `<span class="hero-label">${entry.labelHtml || escapeHtml(entry.label)}</span>`,
        `<div class="hero-meta-value">${escapeHtml(entry.value)}</div>`,
        entry.note ? `<div class="hero-meta-note">${escapeHtml(entry.note)}</div>` : '',
        '</div>'
      ].join(''));
    return items.length ? `<div class="hero-meta">${items.join('')}</div>` : '';
  }
  function hhmmFromIso(value) {
    const match = String(value || '').match(/T(\d{2}:\d{2})/);
    return match ? match[1] : '';
  }
  function stepInterval(step) {
    const start = hhmmFromIso(step && (step.plateau_start_iso || step.start_iso));
    const end = hhmmFromIso(step && (step.plateau_end_iso || step.end_iso));
    if (start && end) return start + '–' + end;
    return start || end || '—';
  }
  function stepTablePayload(value) {
    if (!value || typeof value !== 'object' || !Array.isArray(value.steps) || !value.steps.length) return null;
    return value;
  }
  function embeddedLoadStepTable(finalRow) {
    if (!finalRow) return null;
    const sla = parseJsonish(finalRow.sla_details) || {};
    const parsed = parseJsonish(finalRow.parsed) || {};
    return stepTablePayload(sla.load_step_table) || stepTablePayload(parsed.load_step_table);
  }
  function hintFromCheck(checks, name) {
    const check = (checks || []).find((item) => item && item.name === name);
    const message = safeStr(check && check.message);
    const label = (message.match(/label='([^']*)'/) || message.match(/запрос «([^»]*)»/) || [])[1] || '';
    const series = (message.match(/series='([^']*)'/) || message.match(/серия «([^»]*)»/) || [])[1] || '';
    return { label: label, series: series, actual: check ? check.actual : null };
  }
  function findStepSection(sections, label) {
    const target = safeStr(label).toLowerCase();
    if (!target) return null;
    const exact = sections.filter((section) => safeStr(section && section.label).toLowerCase() === target);
    if (exact.length === 1) return exact[0];
    const partial = sections.filter((section) => safeStr(section && section.label).toLowerCase().indexOf(target) >= 0);
    return partial.length === 1 ? partial[0] : null;
  }
  function isAggregateLabel(label) {
    const text = safeStr(label).toLowerCase();
    return ['sum', 'all groups', 'total', 'общ', 'итого'].some((word) => text.indexOf(word) >= 0);
  }
  function sectionPeak(section) {
    const rows = ((section && section.rows) || []).filter((row) => row && typeof row === 'object');
    return rows.reduce((max, row) => Math.max(max, Number(row.overall_max) || 0), 0);
  }
  function pickStepSection(sections, hint, keywords, skipLabel) {
    const hinted = findStepSection(sections, hint && hint.label);
    if (hinted && safeStr(hinted.label).toLowerCase() !== safeStr(skipLabel).toLowerCase()) return hinted;
    const skip = safeStr(skipLabel).toLowerCase();
    const matched = sections.filter((section) => {
      const label = safeStr(section && section.label).toLowerCase();
      return label !== skip && keywords.some((word) => label.indexOf(word) >= 0);
    });
    if (!matched.length) return null;
    const aggregate = matched.filter((section) => isAggregateLabel(section.label));
    const pool = aggregate.length ? aggregate : matched;
    return pool.slice().sort((a, b) => sectionPeak(b) - sectionPeak(a))[0];
  }
  function pickStepRow(section, seriesName) {
    const rows = ((section && section.rows) || []).filter((row) => row && typeof row === 'object');
    if (!rows.length) return null;
    const named = seriesName ? rows.find((row) => safeStr(row.series) === seriesName) : null;
    if (named) return named;
    return rows.slice().sort((a, b) => Number(b.overall_max || 0) - Number(a.overall_max || 0))[0];
  }
  function perStepField(row, index, field) {
    const items = (row && row.per_step) || [];
    const byNumber = items.find((item) => item && Number(item.step) === Number(index));
    const positional = items[index - 1];
    const item = byNumber || (positional && positional.step == null ? positional : null);
    if (!item || item[field] == null || item[field] === '') return null;
    const number = Number(item[field]);
    return Number.isFinite(number) ? number : null;
  }
  function sectionStepP95(section, seriesName, index) {
    const rows = ((section && section.rows) || []).filter((row) => row && typeof row === 'object');
    if (seriesName) {
      const named = rows.find((row) => safeStr(row.series) === seriesName);
      if (named) return perStepField(named, index, 'p95');
    }
    let best = null;
    rows.forEach((row) => {
      const value = perStepField(row, index, 'p95');
      if (value != null && (best == null || value > best)) best = value;
    });
    return best;
  }
  function markSelectedStep(steps, level) {
    if (level === null || level === undefined || level === '') return;
    const target = Number(level);
    if (!Number.isFinite(target)) return;
    const matches = steps.filter((step) => Number.isFinite(Number(step.rps)) && Math.abs(Number(step.rps) - target) <= 0.51);
    const stable = matches.filter((step) => step.stable !== false);
    const chosen = (stable.length ? stable : matches).slice(-1)[0];
    if (chosen) chosen.selected = true;
  }
  function stepTableFromPack(pack, checks) {
    const source = (pack && typeof pack === 'object') ? pack : {};
    const rawSteps = Array.isArray(source.load_steps) ? source.load_steps : [];
    if (!rawSteps.length) return null;
    const domains = (source.domains && typeof source.domains === 'object') ? source.domains : {};
    const lt = (domains.lt_framework && typeof domains.lt_framework === 'object') ? domains.lt_framework : {};
    const sections = Array.isArray(source.step_table) ? source.step_table : (Array.isArray(lt.step_table) ? lt.step_table : []);
    const rpsHint = hintFromCheck(checks, 'target_rps');
    const p95Hint = hintFromCheck(checks, 'p95_latency');
    const latencyHint = p95Hint.label || p95Hint.series ? p95Hint : hintFromCheck(checks, 'p99_latency');
    const rpsSection = pickStepSection(sections, rpsHint, ['rps', 'throughput', 'нагруз'], '');
    const latencySection = pickStepSection(sections, latencyHint, ['p95', 'latency', 'response', 'отклик'], rpsSection && rpsSection.label);
    const rpsRow = pickStepRow(rpsSection, rpsHint.series);
    const steps = rawSteps.filter((step) => step && typeof step === 'object').map((step, position) => {
      const index = Number(step.index) || (position + 1);
      const hasLevel = step.rps_level !== null && step.rps_level !== undefined && step.rps_level !== '';
      const level = Number(step.rps_level);
      const mean = perStepField(rpsRow, index, 'mean');
      return {
        index: index,
        start_iso: step.start_iso,
        end_iso: step.end_iso,
        plateau_start_iso: step.plateau_start_iso,
        plateau_end_iso: step.plateau_end_iso,
        stable: step.stable,
        after_drop: step.after_drop === true,
        dip: step.dip === true,
        rps: hasLevel && Number.isFinite(level) ? level : mean,
        latency_p95: sectionStepP95(latencySection, latencyHint.series, index),
        selected: false
      };
    });
    if (!steps.length) return null;
    markSelectedStep(steps, rpsHint.actual);
    return {
      rps_query: rpsSection ? rpsSection.label : '',
      latency_query: latencySection ? latencySection.label : '',
      latency_series: safeStr(latencyHint.series),
      steps: steps
    };
  }
  function momentMs(value) {
    if (value == null || value === '') return null;
    const ms = Date.parse(String(value).trim().replace(' ', 'T'));
    return Number.isFinite(ms) ? ms : null;
  }
  function stepCovers(step, ms) {
    if (ms == null) return false;
    const start = momentMs(step.plateau_start_iso || step.start_iso);
    const end = momentMs(step.plateau_end_iso || step.end_iso);
    return start != null && end != null && start <= ms && ms <= end;
  }
  function reportOffsetHours(steps) {
    const sample = (steps || []).map((step) => step.end_iso || step.start_iso).find(Boolean) || '';
    const match = String(sample).match(/([+-])(\d{2}):(\d{2})$/);
    if (!match) return 3;
    const sign = match[1] === '-' ? -1 : 1;
    return sign * (Number(match[2]) + Number(match[3]) / 60);
  }
  function clockAt(ms, offsetHours) {
    const shifted = new Date(ms + offsetHours * 3600000);
    return String(shifted.getUTCHours()).padStart(2, '0') + ':' + String(shifted.getUTCMinutes()).padStart(2, '0');
  }
  function slaWindowOf(finalRow) {
    const sla = parseJsonish(finalRow && finalRow.sla_details) || {};
    if (sla.stable_window && typeof sla.stable_window === 'object') return sla.stable_window;
    const parsed = parseJsonish(finalRow && finalRow.parsed) || {};
    const det = parsed.deterministic_sla;
    return det && typeof det.stable_window === 'object' ? det.stable_window : null;
  }
  function slaChecksOf(finalRow) {
    const sla = parseJsonish(finalRow && finalRow.sla_details) || {};
    if (Array.isArray(sla.checks) && sla.checks.length) return sla.checks;
    const parsed = parseJsonish(finalRow && finalRow.parsed) || {};
    const det = parsed.deterministic_sla;
    return det && Array.isArray(det.checks) ? det.checks : [];
  }
  function slaStatusWord(passed) {
    if (passed === true) return 'пройден';
    if (passed === false) return 'не пройден';
    return 'нет данных';
  }
  function slaStepSentence(chosen, checks, window, steps) {
    if (!chosen || !checks.length) return '';
    const offset = reportOffsetHours(steps);
    const startMs = momentMs(window && (window.start || window.start_iso));
    const endMs = momentMs(window && (window.end || window.end_iso));
    const windowText = startMs != null && endMs != null ? clockAt(startMs, offset) + '–' + clockAt(endMs, offset) : '';
    const level = Number(window && window.level);
    const levelText = Number.isFinite(level) ? formatNumber(level) : '';
    const stepRps = Number(chosen.rps);
    const differs = levelText && Number.isFinite(stepRps) && Math.abs(stepRps - level) > 1;
    const head = 'SLA проверено на ступени ' + (chosen.index || '') + ' (' + stepInterval(chosen) + ')';
    const windowBit = windowText
      ? ', окно ' + windowText + (levelText ? ', RPS окна ' + levelText : '')
      : '';
    const meanBit = differs ? '. RPS в таблице — среднее всего интервала, пороги считаются по этому окну' : '';
    return head + windowBit + meanBit + '. ' + slaChecksText(checks) + '.';
  }
  function slaChecksText(checks) {
    return checks.map((check) => {
      const label = SLA_CHECK_LABELS[safeStr(check && check.name)] || safeStr(check && check.name) || 'критерий';
      const actual = formatNumber(check && check.actual);
      return label + (actual === '—' ? '' : ' ' + actual) + ' — ' + slaStatusWord(check && check.passed);
    }).join('; ');
  }
  function linkSlaToSteps(table, finalRow) {
    const steps = (table.steps || []).map((step) => Object.assign({}, step));
    const checks = slaChecksOf(finalRow);
    const window = slaWindowOf(finalRow);
    const kind = safeStr(window && window.kind);
    if (kind === 'hold') {
      steps.forEach((step) => { step.selected = false; });
      const offset = reportOffsetHours(steps);
      const startMs = momentMs(window.start || window.start_iso);
      const endMs = momentMs(window.end || window.end_iso);
      const span = startMs != null && endMs != null ? ' ' + clockAt(startMs, offset) + '–' + clockAt(endMs, offset) : '';
      const level = Number(window.level);
      const head = 'SLA проверено на окне удержания' + span + (Number.isFinite(level) ? ', средний RPS ' + formatNumber(level) : '');
      return { steps: steps, slaNote: checks.length ? head + '. ' + slaChecksText(checks) + '.' : head + '.' };
    }
    if (kind === 'spike') {
      const moment = momentMs(window.start || window.start_iso);
      steps.forEach((step) => { step.selected = false; });
      const covered = steps.find((step) => stepCovers(step, moment));
      if (covered) covered.selected = true;
      const head = 'SLA проверено в окнах всплесков: по каждому порогу — худший всплеск';
      return { steps: steps, slaNote: checks.length ? head + '. ' + slaChecksText(checks) + '.' : head + '.' };
    }
    if (!steps.some((step) => step.selected)) {
      const moment = momentMs(window && (window.start || window.start_iso));
      const covered = steps.find((step) => stepCovers(step, moment));
      if (covered) covered.selected = true;
      else markSelectedStep(steps, hintFromCheck(checks, 'target_rps').actual);
    }
    const chosen = steps.find((step) => step.selected) || null;
    return { steps: steps, slaNote: slaStepSentence(chosen, checks, window, steps) };
  }
  function slaChecklistNote(finalRow) {
    const table = embeddedLoadStepTable(finalRow);
    if (!table) return '';
    const linked = linkSlaToSteps(table, finalRow);
    return linked.slaNote ? `<div class="hero-sub">${escapeHtml(linked.slaNote)}</div>` : '';
  }
  function renderLoadStepTableHtml(table) {
    const rows = table.steps.map((step) => {
      const classes = [];
      if (step.selected) classes.push('is-selected');
      if (step.stable === false) classes.push('is-unstable');
      const badge = step.selected ? '<span class="step-badge">SLA</span>' : '';
      let flagText = '';
      if (step.after_drop === true) flagText = 'после просадки';
      else if (step.dip === true) flagText = 'кратковременная просадка';
      else if (step.stable === false) flagText = 'нестабильная';
      const drop = flagText ? `<span class="step-flag">${flagText}</span>` : '';
      // The interval shown in the row (plateau when known) is what the charts highlight on hover.
      const range = stepRangeMs(step);
      const rangeAttrs = range[0] !== null && range[1] !== null ? ` data-step-from="${range[0]}" data-step-to="${range[1]}"` : '';
      return [
        `<tr class="${classes.join(' ')}"${rangeAttrs}>`,
        `<td>${escapeHtml(String(step.index || ''))}${badge}${drop}</td>`,
        `<td>${escapeHtml(stepInterval(step))}</td>`,
        `<td class="num">${escapeHtml(formatNumber(step.rps))}</td>`,
        `<td class="num">${escapeHtml(formatNumber(step.latency_p95))}</td>`,
        '</tr>'
      ].join('');
    }).join('');
    const sourceBits = [];
    if (safeStr(table.latency_query)) sourceBits.push('запрос «' + safeStr(table.latency_query) + '»');
    if (safeStr(table.latency_series)) sourceBits.push('серия «' + safeStr(table.latency_series) + '»');
    else if (safeStr(table.latency_query)) sourceBits.push('максимум по сериям запроса');
    const source = sourceBits.length ? ' ' + escapeHtml(sourceBits.join(', ')) + '.' : '';
    return [
      '<details class="report-collapsible">',
      '<summary><span>Ступени теста</span><small class="report-collapsible-subtitle">RPS и время отклика</small></summary>',
      '<div class="report-collapsible-body">',
      '<div class="step-table-wrap"><table class="step-table">',
      '<thead><tr><th>Ступень</th><th>Интервал</th><th class="num">RPS</th><th class="num">Время отклика, p95</th></tr></thead>',
      `<tbody>${rows}</tbody></table></div>`,
      `<p class="step-note">RPS — уровень ступени. Время отклика — p95 на плато, без хвоста после просадки.${source} Наведите на строку, чтобы подсветить ступень на графиках.</p>`,
      table.slaNote ? `<p class="step-sla">${escapeHtml(table.slaNote)}</p>` : '',
      '</div></details>'
    ].join('');
  }
  // ---- chart markers, SLA thresholds and series order ------------------------------------------------
  // Mirrors AI/report_markers.py (markers_for, thresholds_for, order_series), which draws the same items
  // on the Confluence PNGs; tests/test_report_markers.py runs both sides on the same rows. Only
  // deterministic data is used: the step table, the SLA window and the test-type payloads, never the
  // LLM text of peak_performance.max_time / drop_time.
  const THRESHOLD_SPECS = {
    target_rps: ['цель ≥ {} RPS', 'accent'],
    p95_latency: ['p95 ≤ {} мс', 'fail'],
    p99_latency: ['p99 ≤ {} мс', 'fail'],
    error_rate: ['ошибки ≤ {} %', 'fail'],
    cpu_usage: ['CPU ≤ {} %', 'fail'],
    memory_usage: ['память ≤ {} %', 'fail']
  };
  const DROP_MERGE_MS = 60000;
  function isPlainObject(value) { return !!value && typeof value === 'object' && !Array.isArray(value); }
  function finiteOrNull(value) {
    if (value === null || value === undefined || typeof value === 'boolean' || typeof value === 'object') return null;
    if (typeof value === 'string' && !value.trim()) return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  // Epoch ms of an ISO string; a string without a zone is UTC (as on the server).
  function utcMoment(value) {
    let text = safeStr(value);
    if (!text) return null;
    text = text.replace(' ', 'T');
    if (/T\d/.test(text) && !/(?:[zZ]|[+-]\d{2}:?\d{2})$/.test(text)) text += 'Z';
    const ms = Date.parse(text);
    return Number.isFinite(ms) ? ms : null;
  }
  function stepRangeMs(step) {
    return [
      utcMoment(step && (step.plateau_start_iso || step.start_iso)),
      utcMoment(step && (step.plateau_end_iso || step.end_iso))
    ];
  }
  function markerSlaWindow(finalRow) {
    const sla = parseJsonish(finalRow && finalRow.sla_details) || {};
    let window = sla.stable_window;
    if (!isPlainObject(window)) {
      const parsed = parseJsonish(finalRow && finalRow.parsed) || {};
      window = isPlainObject(parsed.deterministic_sla) ? parsed.deterministic_sla.stable_window : null;
    }
    return isPlainObject(window) ? window : {};
  }
  function markerChecks(finalRow) {
    const sla = parseJsonish(finalRow && finalRow.sla_details) || {};
    let checks = sla.checks;
    if (!(Array.isArray(checks) && checks.length)) {
      const parsed = parseJsonish(finalRow && finalRow.parsed) || {};
      checks = isPlainObject(parsed.deterministic_sla) ? parsed.deterministic_sla.checks : null;
    }
    return Array.isArray(checks) ? checks.filter(isPlainObject) : [];
  }
  function markerTable(finalRow, table) {
    if (isPlainObject(table) && Array.isArray(table.steps) && table.steps.length) return table;
    return embeddedLoadStepTable(finalRow) || {};
  }
  function markerMode(parsed, window) {
    const kind = safeStr(window.kind).toLowerCase();
    if (isPlainObject(parsed.spike_analysis) || kind === 'spike') return 'spike';
    const profile = isPlainObject(parsed.test_profile) ? parsed.test_profile : {};
    if (isPlainObject(parsed.stability_under_load) || kind === 'hold' || ['stability', 'soak'].indexOf(safeStr(profile.mode).toLowerCase()) >= 0) return 'hold';
    return 'step';
  }
  function markerItem(kind, t, to, label, tone, role) {
    return { kind: kind, t: t, to: to, label: label, tone: tone, role: role };
  }
  function bandItem(start, end, label, tone, role) {
    let a = utcMoment(start);
    let b = utcMoment(end);
    if (a === null || b === null) return null;
    if (b < a) { const swap = a; a = b; b = swap; }
    return markerItem('band', a, b, label, tone, role);
  }
  function checkNamed(checks, name) {
    return checks.find((check) => safeStr(check.name) === name) || {};
  }
  // Step the SLA was checked on: the marked one, the one under the window start, else by target RPS.
  function chosenSlaStep(steps, window, checks) {
    const selected = steps.find((step) => step.selected);
    if (selected) return selected;
    const start = utcMoment(window.start || window.start_iso);
    const covered = steps.find((step) => {
      const range = stepRangeMs(step);
      return start !== null && range[0] !== null && range[1] !== null && range[0] <= start && start <= range[1];
    });
    if (covered) return covered;
    const target = finiteOrNull(checkNamed(checks, 'target_rps').actual);
    if (target === null) return null;
    const matches = steps.filter((step) => finiteOrNull(step.rps) !== null && Math.abs(Number(step.rps) - target) <= 0.51);
    const stable = matches.filter((step) => step.stable !== false);
    const pool = stable.length ? stable : matches;
    return pool.length ? pool[pool.length - 1] : null;
  }
  function stepPosition(steps, step) {
    if (!step) return -1;
    const same = (a, b) => (a === undefined ? null : a) === (b === undefined ? null : b);
    return steps.findIndex((item) => same(item.index, step.index) && same(item.start_iso, step.start_iso));
  }
  function stepMarkers(parsed, window, checks, steps) {
    const out = [];
    const offset = reportOffsetHours(steps);
    const saturation = isPlainObject(parsed.saturation) ? parsed.saturation : null;
    const chosen = chosenSlaStep(steps, window, checks);
    const kind = safeStr(window.kind).toLowerCase() || 'step';
    const hasWindow = Object.keys(window).length > 0;
    const label = chosen && safeStr(chosen.index) ? `SLA: ступень ${safeStr(chosen.index)}` : 'SLA: окно';
    let band = null;
    if (hasWindow && kind === 'step') band = bandItem(window.start || window.start_iso, window.end || window.end_iso, label, 'accent', 'sla');
    if (!band && chosen) band = bandItem(chosen.plateau_start_iso || chosen.start_iso, chosen.plateau_end_iso || chosen.end_iso, label, 'accent', 'sla');
    if (!band && saturation) {
      const satWindow = isPlainObject(saturation.saturation_window) ? saturation.saturation_window : {};
      band = bandItem(satWindow.start_iso, satWindow.end_iso, label, 'accent', 'sla');
    }
    if (band) out.push(band);
    steps.slice(1).forEach((step) => {
      const t = utcMoment(step.start_iso);
      if (t !== null) out.push(markerItem('tick', t, null, '', 'muted', 'step'));
    });
    let degradationMs = null;
    const degradedChecks = window.degraded_checks;
    const degraded = (window.degraded_level !== null && window.degraded_level !== undefined)
      || (Array.isArray(degradedChecks) ? degradedChecks.length > 0 : !!degradedChecks);
    if (hasWindow && kind === 'step' && degraded) {
      const position = stepPosition(steps, chosen);
      const following = position >= 0 && position < steps.length - 1 ? steps[position + 1] : null;
      degradationMs = following ? utcMoment(following.start_iso) : utcMoment(window.end || window.end_iso);
      if (degradationMs !== null) out.push(markerItem('line', degradationMs, null, `деградация ${clockAt(degradationMs, offset)}`, 'fail', 'degradation'));
    }
    let dropMs = saturation ? utcMoment(saturation.rps_drop_iso) : null;
    if (dropMs === null) {
      const afterDrop = steps.find((step) => step.after_drop === true);
      dropMs = afterDrop ? utcMoment(afterDrop.start_iso) : null;
    }
    if (dropMs !== null && (degradationMs === null || Math.abs(dropMs - degradationMs) > DROP_MERGE_MS)) {
      out.push(markerItem('line', dropMs, null, `падение RPS ${clockAt(dropMs, offset)}`, 'fail', 'rps_drop'));
    }
    return out;
  }
  function holdMarkers(parsed, window) {
    const stability = isPlainObject(parsed.stability_under_load) ? parsed.stability_under_load : {};
    const hold = isPlainObject(stability.hold) ? stability.hold : {};
    let band = bandItem(hold.start_iso, hold.end_iso, 'окно удержания', 'accent', 'hold');
    if (!band && safeStr(window.kind).toLowerCase() === 'hold') {
      band = bandItem(window.start || window.start_iso, window.end || window.end_iso, 'окно удержания', 'accent', 'hold');
    }
    return band ? [band] : [];
  }
  function spikeMarkers(parsed, window) {
    const out = [];
    const analysis = isPlainObject(parsed.spike_analysis) ? parsed.spike_analysis : {};
    (Array.isArray(analysis.spikes) ? analysis.spikes : []).forEach((spike, position) => {
      if (!isPlainObject(spike)) return;
      const label = safeStr(spike.label) || `всплеск ${safeStr(spike.index) || position + 1}`;
      const band = bandItem(spike.start_iso, spike.end_iso, label, 'warn', 'spike');
      if (band) out.push(band);
    });
    if (!out.length && safeStr(window.kind).toLowerCase() === 'spike') {
      const band = bandItem(window.start || window.start_iso, window.end || window.end_iso, safeStr(window.title) || 'всплеск', 'warn', 'spike');
      if (band) out.push(band);
    }
    return out;
  }
  // reportMarkers(finalRow, table): markers drawn on every chart (see AI/report_markers.markers_for).
  function reportMarkers(finalRow, table) {
    if (!finalRow) return [];
    const parsed = parseJsonish(finalRow.parsed) || {};
    const window = markerSlaWindow(finalRow);
    const mode = markerMode(parsed, window);
    if (mode === 'spike') return spikeMarkers(parsed, window);
    if (mode === 'hold') return holdMarkers(parsed, window);
    const steps = (markerTable(finalRow, table).steps || []).filter(isPlainObject);
    return stepMarkers(parsed, window, markerChecks(finalRow), steps);
  }
  function messageHint(message) {
    const label = (message.match(/label='([^']*)'/) || message.match(/запрос «([^»]*)»/) || [])[1] || '';
    const series = (message.match(/series='([^']*)'/) || message.match(/серия «([^»]*)»/) || [])[1] || '';
    return { label: label, series: series };
  }
  // Query label and series of a check: explicit fields, then the message, then the SLA window and the step table.
  function checkSource(check, window, table) {
    const name = safeStr(check.name);
    let label = safeStr(check.query_label);
    let series = safeStr(check.series);
    if (!label) {
      const hint = messageHint(safeStr(check.message));
      label = hint.label;
      series = series || hint.series;
    }
    if (!label && name === 'target_rps') {
      label = safeStr(window.label) || safeStr(table.rps_query);
      series = series || (safeStr(window.label) ? safeStr(window.series) : safeStr(table.rps_series));
    }
    if (!label && (name === 'p95_latency' || name === 'p99_latency')) {
      label = safeStr(table.latency_query);
      series = series || safeStr(table.latency_series);
    }
    return { label: label, series: series };
  }
  // slaThresholds(finalRow, table): [{name, query_label, series, value, label, tone}] (AI/report_markers.thresholds_for).
  function slaThresholds(finalRow, table) {
    if (!finalRow) return [];
    const window = markerSlaWindow(finalRow);
    const source = markerTable(finalRow, table);
    const out = [];
    markerChecks(finalRow).forEach((check) => {
      const name = safeStr(check.name);
      const spec = THRESHOLD_SPECS[name];
      const value = finiteOrNull(check.threshold);
      if (!spec || value === null) return;
      const where = checkSource(check, window, source);
      if (!where.label) return;
      out.push({ name: name, query_label: where.label, series: where.series, value: value, label: spec[0].replace('{}', formatNumber(value)), tone: spec[1] });
    });
    return out;
  }
  // Chart label a check label refers to: the exact one, else the only label containing it (case-insensitive,
  // as AI/sla_evaluator finds the SLA query), else the label itself (AI/report_markers.resolve_query_label).
  function resolveQueryLabel(label, allLabels) {
    const text = safeStr(label);
    const labels = (allLabels || []).map(safeStr).filter(Boolean);
    if (!text || !labels.length) return text;
    const lower = text.toLowerCase();
    const exact = labels.filter((item) => item.toLowerCase() === lower);
    if (exact.length === 1) return exact[0];
    const partial = labels.filter((item) => item.toLowerCase().indexOf(lower) >= 0);
    return partial.length === 1 ? partial[0] : text;
  }
  // Items for options.plugins.llThresholds of the chart of `queryLabel`; `allLabels` are the run's chart labels.
  function thresholdsForQuery(thresholds, queryLabel, allLabels) {
    const target = safeStr(queryLabel);
    return (thresholds || []).filter((item) => target && resolveQueryLabel(item.query_label, allLabels) === target)
      .map((item) => ({ value: item.value, label: item.label, tone: item.tone }));
  }
  function seriesMean(values) {
    let total = 0;
    let count = 0;
    (values || []).forEach((value) => {
      const n = finiteOrNull(value);
      if (n === null) return;
      total += n;
      count += 1;
    });
    return count ? Math.abs(total / count) : -1;
  }
  // orderSeries({name: values[]}): names by the absolute mean, largest first, then by name; series i gets palette slot i.
  function orderSeries(valuesBySeries) {
    return Object.keys(valuesBySeries || {})
      .map((name) => ({ name: name, mean: seriesMean(valuesBySeries[name]) }))
      .sort((a, b) => (b.mean - a.mean) || (a.name < b.name ? -1 : (a.name > b.name ? 1 : 0)))
      .map((item) => item.name);
  }
  function highlightCharts(range) {
    if (!LL.charts || typeof LL.charts.setHighlight !== 'function') return;
    LL.charts.setHighlight(reportState.charts.filter((chart) => chart && chart.canvas), range);
  }
  async function fetchContextPack(run, domain) {
    const resp = await fetch('/llm_context?run_name=' + encodeURIComponent(run) + '&domain=' + encodeURIComponent(domain));
    if (!resp.ok) return null;
    const data = await resp.json();
    const row = Array.isArray(data) ? data.find((item) => item && item.domain === domain) : null;
    return parseJsonish(row && row.context);
  }
  function stepsHost() {
    let box = document.getElementById('reportSteps');
    if (box) return box;
    const anchor = document.getElementById('rep-llm-tabs');
    if (!anchor || !anchor.parentNode) return null;
    box = document.createElement('div');
    box.id = 'reportSteps';
    box.className = 'report-steps';
    box.hidden = true;
    anchor.parentNode.insertBefore(box, anchor);
    return box;
  }
  // Step table saved with the report, else rebuilt from the stored model context (as Confluence does).
  async function resolveStepTable(finalRow) {
    const embedded = embeddedLoadStepTable(finalRow);
    if (embedded) return embedded;
    const run = getRunFromPath();
    if (!finalRow || !run) return null;
    try {
      const sla = parseJsonish(finalRow.sla_details) || {};
      const checks = Array.isArray(sla.checks) ? sla.checks : [];
      let pack = await fetchContextPack(run, 'lt_framework');
      if (!pack || !Array.isArray(pack.load_steps) || !pack.load_steps.length) {
        pack = await fetchContextPack(run, 'final');
      }
      return stepTableFromPack(pack, checks);
    } catch (e) {
      return null;
    }
  }
  // Hovering a row of the step table highlights that step on every chart of the report.
  function bindStepHighlight(box) {
    const rows = Array.from(box.querySelectorAll('tr[data-step-from]'));
    const clear = () => highlightCharts(null);
    rows.forEach((row) => {
      const range = { t: Number(row.dataset.stepFrom), to: Number(row.dataset.stepTo), tone: 'accent' };
      row.addEventListener('mouseenter', () => highlightCharts(range));
      row.addEventListener('mouseleave', clear);
    });
  }
  async function renderLoadSteps(finalRow) {
    const box = stepsHost();
    const table = await resolveStepTable(finalRow);
    if (!box) return table;
    if (!table) {
      box.hidden = true;
      box.innerHTML = '';
      return null;
    }
    const linked = linkSlaToSteps(table, finalRow);
    box.hidden = false;
    box.innerHTML = renderLoadStepTableHtml(Object.assign({}, table, linked));
    bindStepHighlight(box);
    return table;
  }
  async function renderForecastLink() {
    const link = document.getElementById('forecastLink');
    if (!link || !reportRef.run_id) return;
    link.hidden = true;
    try {
      const resp = await fetch('/forecast/' + encodeURIComponent(reportRef.run_id) + '/status');
      const data = await resp.json();
      if (resp.ok && data.available) {
        link.href = '/forecasting/' + encodeURIComponent(reportRef.run_id);
        link.hidden = false;
      }
    } catch (e) {}
  }

  // Hero meta entries of the test type that are not KPI tiles (heroKpis shows the numbers).
  function heroTypeEntries(parsed, peak) {
    const mode = reportMode(parsed);
    if (mode === 'stability') {
      const st = parsed.stability_under_load || {};
      const hold = (st.hold && typeof st.hold === 'object') ? st.hold : {};
      return [
        { label: 'Мин. интервал RPS', value: hold.min_bucket_rps != null ? formatNumber(hold.min_bucket_rps) : '', note: hold.drop_pct ? `−${formatNumber(hold.drop_pct)} % от среднего` : '' }
      ];
    }
    if (mode === 'spike') return [];
    if (mode === 'stress') {
      const sat = parsed.saturation || {};
      const resource = (sat.limiting_resource && typeof sat.limiting_resource === 'object') ? resourceText(sat.limiting_resource) : '';
      return [
        { label: 'Характер деградации', value: safeStr(sat.failure_mode_title) || '—', note: safeStr(sat.degraded_step_label) ? 'с ' + safeStr(sat.degraded_step_label) : '' },
        { label: 'Ограничивающий ресурс', labelHtml: LL.ui.term('limiting_resource', 'Ограничивающий ресурс'), value: resource || 'не определён' }
      ];
    }
    if (!peakTableApplies('final', peak)) return [];
    return [
      { label: 'Время пика', value: safeStr(peak.max_time) || '—' },
      { label: 'Время деградации', value: safeStr(peak.drop_time) || '—' }
    ];
  }
  function kpiTone(passed) {
    if (passed === true) return 'ok';
    if (passed === false) return 'fail';
    return '';
  }
  function kpiThresholdNote(check, op, unit, words) {
    if (!check || finiteOrNull(check.threshold) === null) return '';
    const state = check.passed === true ? words[0] : (check.passed === false ? words[1] : '');
    return `${op} ${formatNumber(check.threshold)}${unit}${state ? ' · ' + state : ''}`;
  }
  // heroKpis(finalRow): up to three number tiles {key, label, term, value, unit, note, tone} above the meta.
  // Step tests: max RPS, p95 and errors with their SLA thresholds; soak, spike and stress take their own
  // numbers (the meta then skips them, see heroTypeEntries).
  function heroKpis(finalRow) {
    if (!finalRow) return [];
    const parsed = parseJsonish(finalRow.parsed) || {};
    const peak = isPlainObject(parsed.peak_performance) ? parsed.peak_performance : {};
    const checks = slaChecksOf(finalRow).filter(isPlainObject);
    const byName = (name) => checks.find((check) => safeStr(check.name) === name) || null;
    const target = byName('target_rps');
    const latency = byName('p95_latency') || byName('p99_latency');
    const errors = byName('error_rate');
    const mode = reportMode(parsed);
    const tile = (key, label, term, value, unit, note, tone) => {
      const text = formatNumber(value);
      return { key: key, label: label, term: term, value: text, unit: text === '—' ? '' : unit, note: note || '', tone: tone || '' };
    };
    const textTile = (key, label, term, value, note, tone) => ({ key: key, label: label, term: term, value: value, unit: '', note: note || '', tone: tone || '' });
    const latencyTile = latency ? tile('latency', latency.name === 'p99_latency' ? 'p99' : 'p95', latency.name === 'p99_latency' ? '' : 'p95', latency.actual, ' мс', kpiThresholdNote(latency, 'порог ≤', ' мс', ['соблюдён', 'нарушен']), kpiTone(latency.passed)) : null;
    const errorTile = errors ? tile('errors', 'Ошибки', 'error_rate', errors.actual, ' %', kpiThresholdNote(errors, 'порог ≤', ' %', ['соблюдён', 'нарушен']), kpiTone(errors.passed)) : null;
    const targetNote = kpiThresholdNote(target, 'цель ≥', ' RPS', ['достигнута', 'не достигнута']);
    if (mode === 'spike') {
      const sa = parsed.spike_analysis || {};
      const spikes = Array.isArray(sa.spikes) ? sa.spikes : [];
      const worst = spikes.find((item) => item && item.index === sa.worst_spike_index) || spikes[0] || {};
      const notRecovered = spikes.filter((item) => item && item.recovered === false).length;
      const times = spikes.map((item) => Number(item && item.recovery_time_sec)).filter((n) => Number.isFinite(n));
      const recovery = notRecovered ? `не восстановилось: ${notRecovered} из ${spikes.length}` : (times.length ? `${formatNumber(Math.max.apply(null, times))} с` : '—');
      const recoveryCheck = byName('recovery');
      const baseline = sa.baseline_rps != null ? `фон ${formatNumber(sa.baseline_rps)} RPS` : '';
      return [
        sa.detected
          ? tile('spike', 'Всплеск', 'spike_window', sa.max_spike_rps, ' RPS', [baseline, `всплесков: ${spikes.length}`, targetNote].filter(Boolean).join(' · '), kpiTone(target && target.passed))
          : textTile('spike', 'Всплеск', 'spike_window', 'не обнаружен', baseline, ''),
        textTile('overshoot', 'Рост p95', 'overshoot', signed(worst.overshoot_p95_pct, ' %') || '—', 'худший всплеск', kpiTone(latency && latency.passed)),
        textTile('recovery', 'Восстановление', 'recovery_time', recovery, recoveryCheck && finiteOrNull(recoveryCheck.threshold) !== null ? `предел ${formatNumber(recoveryCheck.threshold)} мин` : '', kpiTone(recoveryCheck && recoveryCheck.passed))
      ];
    }
    if (mode === 'stability') {
      const st = parsed.stability_under_load || {};
      const hold = isPlainObject(st.hold) ? st.hold : {};
      const span = hold.start_iso ? `${hhmm(hold.start_iso)}–${hhmm(hold.end_iso)}` : '';
      const goal = finiteOrNull(st.target_rps) !== null ? `цель ≥ ${formatNumber(st.target_rps)} RPS${target && target.passed === true ? ' · удержана' : (target && target.passed === false ? ' · не удержана' : '')}` : targetNote;
      return [
        tile('rps', 'Средний RPS', 'hold_window', st.actual_rps, ' RPS', [goal, span].filter(Boolean).join(' · '), kpiTone(target && target.passed)),
        latencyTile,
        errorTile
      ].filter(Boolean);
    }
    if (mode === 'stress') {
      const sat = parsed.saturation || {};
      const point = finiteOrNull(sat.saturation_rps) !== null ? sat.saturation_rps : peak.max_rps;
      return [
        tile('rps', 'Точка насыщения', 'saturation_point', point, ' RPS', safeStr(sat.saturation_step_label), ''),
        latencyTile,
        errorTile
      ].filter(Boolean);
    }
    const peakRps = finiteOrNull(peak.max_rps) !== null ? peak.max_rps : (target ? target.actual : null);
    const method = safeStr(peak.method);
    const rpsTerm = /stable/i.test(method) ? 'stable_max' : 'peak_max';
    const rpsTile = finiteOrNull(peakRps) !== null || target
      ? tile('rps', 'Максимальный RPS', rpsTerm, peakRps, ' RPS', targetNote || (method ? `метод: ${method}` : ''), kpiTone(target && target.passed))
      : null;
    return [rpsTile, latencyTile, errorTile].filter(Boolean);
  }
  function renderHeroKpis(tiles) {
    if (!tiles.length) return '';
    const items = tiles.map((item) => {
      const label = item.term ? LL.ui.term(item.term, item.label) : escapeHtml(item.label);
      const unit = item.unit ? `<span class="report-kpi-unit">${escapeHtml(item.unit)}</span>` : '';
      const tone = item.tone ? ` is-${item.tone}` : '';
      return [
        `<div class="report-kpi${tone}${item.unit ? '' : ' is-text'}" role="listitem">`,
        `<div class="report-kpi-label">${label}</div>`,
        `<div class="report-kpi-value">${escapeHtml(item.value)}${unit}</div>`,
        item.note ? `<div class="report-kpi-note">${escapeHtml(item.note)}</div>` : '',
        '</div>'
      ].join('');
    });
    return `<div class="report-kpis" role="list" aria-label="Ключевые показатели">${items.join('')}</div>`;
  }
  function renderHero(finalRow, allRows) {
    const hero = document.getElementById('reportHero');
    if (!hero) return;
    if (!finalRow) {
      hero.innerHTML = '<div class="hero-card hero-empty">Итоговый отчёт для этого запуска ещё не сформирован. Если генерация идёт, обновите страницу позже; статус задачи виден в архиве.</div>';
      return;
    }
    const effective = verdictTone(finalRow.verdict);
    const parsed = parseJsonish(finalRow.parsed) || {};
    const peak = (parsed.peak_performance && typeof parsed.peak_performance === 'object') ? parsed.peak_performance : {};
    const service = safeStr(finalRow.service) || getServiceFromPath();
    const startMs = Number(finalRow.start_ms);
    const endMs = Number(finalRow.end_ms);
    const period = (Number.isFinite(startMs) && Number.isFinite(endMs))
      ? `${LL.formatDateTime(startMs)} — ${LL.formatDateTime(endMs)}`
      : '';
    const durationMin = (Number.isFinite(startMs) && Number.isFinite(endMs)) ? Math.round((endMs - startMs) / 60000) : null;
    const manual = manualOf(finalRow);
    const verdictSource = verdictSourceText(finalRow);
    const verdictTermKey = { 'Успешно': 'verdict_success', 'Есть риски': 'verdict_risk', 'Провал': 'verdict_fail' }[effective.text] || 'verdict_na';
    const testType = normalizedTestType(parsed, finalRow.test_type);
    const typeTermKey = LL.GLOSSARY['test_type_' + testType] ? 'test_type_' + testType : 'step_profile';
    hero.innerHTML = [
      '<div class="hero-card">',
      `<div class="hero-label">${LL.ui.term('sla_first', 'Вердикт по тесту')}</div>`,
      `<div class="hero-verdict-row"><abbr class="term" data-term="${verdictTermKey}" data-tip="${escapeHtml(LL.GLOSSARY[verdictTermKey] || '')}" title="${escapeHtml(LL.GLOSSARY[verdictTermKey] || '')}" tabindex="0" style="text-decoration:none"><span class="report-verdict-pill ${effective.className}">${escapeHtml(effective.text)}</span></abbr>${manual ? '<span class="verdict-manual-tag">вручную</span>' : ''}</div>`,
      `<div class="hero-sub">${escapeHtml(verdictSource)}</div>`,
      manual && safeStr(manual.reason) ? `<div class="hero-sub hero-manual-reason">Причина: ${escapeHtml(safeStr(manual.reason))}</div>` : '',
      '<div class="hero-verdict-actions" data-min-role="engineer">'
        + '<button type="button" class="btn btn-sm" data-verdict-action="change">Изменить вердикт</button>'
        + (manual ? '<button type="button" class="btn btn-sm" data-verdict-action="reset">Вернуть автоматический</button>' : '')
        + '</div>',
      (function () {
        const verification = (parsed.verification_summary && typeof parsed.verification_summary === 'object') ? parsed.verification_summary : null;
        const unverifiedCount = verification ? Number(verification.unverified) : 0;
        if (!verification || !(unverifiedCount > 0)) return '';
        const note = `${unverifiedCount} ${pluralFindings(unverifiedCount)} не подтверждены данными и не учтены в обосновании вердикта${verification.revised_by_model ? '' : ' (верификационный проход модели не выполнялся)'}.`;
        return `<div class="hero-sub hero-unverified">${LL.ui.term('unverified', note)}</div>`;
      })(),
      (function () {
        const usageLine = formatUsageLine(collectRunUsage(finalRow, allRows));
        return usageLine ? `<div class="hero-sub hero-usage">${escapeHtml(usageLine)}</div>` : '';
      })(),
      renderHeroKpis(heroKpis(finalRow)),
      renderHeroMeta([
        { label: 'Сервис', value: service || '—' },
        { label: 'Тип теста', labelHtml: LL.ui.term(typeTermKey, 'Тип теста'), value: LL.testTypeLabel(finalRow.test_type) },
        { label: 'Период', value: period || '—', note: period ? `${durationMin} мин · ${LL.timeZoneLabel()}` : '' },
        ...heroTypeEntries(parsed, peak)
      ]),
      (function () {
        const votes = renderVoteGroup('final', 'verdict', '', 'Согласен с вердиктом', 'Не согласен');
        return votes ? `<div class="hero-card-foot" data-min-role="engineer"><span class="hero-card-foot-label">Вердикт верен?</span>${votes}</div>` : '';
      })(),
      '</div>',
      `<div class="hero-card">${renderSlaChecklist(finalRow)}</div>`
    ].join('');
    bindFeedbackVotes(hero);
    bindVerdictActions(hero);
  }

  async function reportsLoadLlm() {
    const run = getRunFromPath(); if (!run) return;
    const box = document.getElementById('rep-llm-tabs'); if (!box) return;
    box.innerHTML = `<div class="report-analysis-card">${LL.ui.skeleton(4)}</div>`;
    const [resp, feedbackResp] = await Promise.all([
      fetch('/llm_reports?run_name=' + encodeURIComponent(run)),
      fetch('/llm_feedback?run_name=' + encodeURIComponent(run)).catch(() => null)
    ]);
    let arr = await resp.json();
    let feedbackRows = [];
    try { feedbackRows = feedbackResp ? await feedbackResp.json() : []; } catch (e) { feedbackRows = []; }
    indexFeedback(Array.isArray(feedbackRows) ? feedbackRows : []);
    if (!Array.isArray(arr)) {
      renderSystemContextBox(null);
      renderHero(null);
      await renderLoadSteps(null);
      renderForecastLink();
      box.textContent = 'Нет данных';
      return;
    }
    // Rows are ordered by created_at DESC: keep only the latest row per domain.
    const seenDomains = new Set();
    arr = arr.filter((x) => {
      if (!x || !x.domain || x.domain === 'engineer') return false;
      if (seenDomains.has(x.domain)) return false;
      seenDomains.add(x.domain);
      return true;
    });
    reportDomainTitles = LL.reportDomainTitles(arr);
    try {
      setPageTitle(run);
      const parsedContexts = arr
        .map((item) => parseSystemContextValue(item && item.system_context))
        .filter((item) => item && typeof item === 'object');
      const systemContext = parsedContexts.find((item) => hasMeaningfulSystemContext(item)) || parsedContexts[0] || null;
      renderSystemContextBox(systemContext);
      const starts = arr.map((x) => parseInt(x.start_ms, 10)).filter((v) => Number.isFinite(v));
      const ends = arr.map((x) => parseInt(x.end_ms, 10)).filter((v) => Number.isFinite(v));
      const startStr = starts.length ? LL.formatDateTime(Math.min.apply(null, starts)) : '—';
      const endStr = ends.length ? LL.formatDateTime(Math.max.apply(null, ends)) : '—';
      const range = document.getElementById('pageTimeRange');
      if (range) range.textContent = `Время теста: ${startStr} — ${endStr} · ${LL.timeZoneLabel()}`;
    } catch (e) {}
    const finalRow = arr.find((x) => x.domain === 'final') || null;
    reportState.finalRow = finalRow;
    reportState.rows = arr;
    renderHero(finalRow, arr);
    // Charts read the markers and thresholds, so the step table is resolved before they are drawn.
    const stepTable = await renderLoadSteps(finalRow);
    reportState.finalRow = finalRow;
    reportState.stepTable = stepTable;
    try {
      reportState.markers = reportMarkers(finalRow, stepTable);
      reportState.thresholds = slaThresholds(finalRow, stepTable);
    } catch (e) {
      reportState.markers = [];
      reportState.thresholds = [];
    }
    renderForecastLink();
    const domainsOrder = ['final', 'jvm', 'database', 'kafka', 'microservices', 'hard_resources', 'lt_framework', 'lt_generators', 'application_logs'];
    arr.sort((a, b) => domainsOrder.indexOf(a.domain) - domainsOrder.indexOf(b.domain));
    const tabsNav = document.createElement('div'); tabsNav.className = 'tabs'; tabsNav.setAttribute('aria-label', 'Анализ по доменам');
    const tabsBody = document.createElement('div');
    const idBase = 'rep-llm-tab-';
    arr.forEach((x, idx) => {
      const btn = document.createElement('button'); btn.className = 'tab' + (idx === 0 ? ' active' : '');
      const dot = document.createElement('span');
      const verdictForTab = x.verdict || x.llm_verdict || '';
      dot.className = `tab-dot ${LL.verdictClass(verdictForTab)}`;
      dot.setAttribute('aria-hidden', 'true');
      btn.title = `Вердикт: ${standardizeVerdict(verdictForTab)}`;
      btn.appendChild(dot);
      btn.appendChild(document.createTextNode(LL.domainTitleFor(x.domain, reportDomainTitles)));
      btn.dataset.target = idBase + idx;
      tabsNav.appendChild(btn);
      const pane = document.createElement('div'); pane.id = idBase + idx; pane.className = 'tabpanel' + (idx === 0 ? ' active' : '');
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
          const judgeHtml = judgeSectionHtml(sc);
          if (safeStr(judgeHtml)) extraSections.push(judgeHtml);
        } catch (e) {}
        html = renderStructuredReportHtml(structuredReport || {}, extraSections.join(''), x.domain);
      } else {
        html = markdownToSafeHtml(md);
      }
      const aiBadge = '<div class="ai-badge" title="Текст ниже сгенерирован языковой моделью по агрегированным метрикам">Сгенерировано ИИ · сверяйте выводы с графиками</div>';
      // Print shows every tab panel; the heading names the domain there (hidden on screen).
      const printTitle = `<h3 class="report-print-title">Анализ: ${escapeHtml(LL.domainTitleFor(x.domain, reportDomainTitles))}</h3>`;
      pane.innerHTML = `${printTitle}<div class="report-analysis-card">${aiBadge}${html}</div>`;
      tabsBody.appendChild(pane);
    });
    box.innerHTML = ''; box.appendChild(tabsNav); box.appendChild(tabsBody);
    LL.ui.tabs(tabsNav);
    LL.ui.applyTerms(box);
    bindFeedbackVotes(box);
  }

  // Chart data of /run_series points: category labels are the minute buckets in time order; series are
  // ordered by orderSeries and take palette slots in that order (the Confluence PNG of the chart does the
  // same). Series past MAX_VISIBLE_SERIES start hidden; the legend chips show them on demand.
  function buildChartData(points) {
    const seen = new Set();
    const labels = [];
    const map = {};
    (Array.isArray(points) ? points : []).forEach((p) => {
      if (!p) return;
      const t = p.t;
      if (!seen.has(t)) { seen.add(t); labels.push(t); }
      const key = String(p.series || 'series');
      if (!map[key]) map[key] = new Map();
      map[key].set(t, p.value);
    });
    labels.sort((a, b) => {
      const ma = utcMoment(a); const mb = utcMoment(b);
      if (ma === null || mb === null) return (ma === null) - (mb === null);
      return ma - mb;
    });
    const values = {};
    Object.keys(map).forEach((key) => { values[key] = labels.map((t) => (map[key].has(t) ? map[key].get(t) : null)); });
    const palette = (i) => (LL.charts && typeof LL.charts.palette === 'function' ? LL.charts.palette(i) : LL.colorFor(String(i)));
    const datasets = orderSeries(values).map((name, i) => ({
      label: name,
      data: values[name],
      llSeries: i,
      borderColor: palette(i),
      backgroundColor: palette(i),
      pointRadius: 0,
      pointHoverRadius: 3,
      borderWidth: 2,
      spanGaps: true,
      hidden: i >= MAX_VISIBLE_SERIES
    }));
    return { labels: labels, datasets: datasets };
  }
  function clockLabel(label) {
    const ms = utcMoment(label);
    if (ms === null) return String(label || '');
    const d = new Date(ms);
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  }
  async function reportsDrawOne(run, domain, ql, canvas, legendHost, group) {
    const u = new URL('/run_series', location.origin);
    u.searchParams.set('run_name', run);
    u.searchParams.set('domain', domain);
    u.searchParams.set('query_label', ql);
    u.searchParams.set('series_key', 'auto');
    u.searchParams.set('align', 'absolute');
    let data = null;
    try {
      const resp = await fetch(u);
      data = await resp.json();
    } catch (e) {
      data = null;
    }
    const box = canvas.parentNode;
    if (!data || !Array.isArray(data.points) || !data.points.length) {
      if (box) {
        box.classList.add('is-empty');
        box.innerHTML = '<div class="report-chart-empty">Нет данных за время теста</div>';
      }
      if (legendHost) legendHost.remove();
      return null;
    }
    const built = buildChartData(data.points);
    const colors = LL.ui.chartColors();
    const tooltip = LL.charts && typeof LL.charts.htmlTooltip === 'function' ? LL.charts.htmlTooltip({ max: TOOLTIP_ROWS }) : {};
    tooltip.callbacks = Object.assign({}, tooltip.callbacks, {
      title(items) {
        const ms = items && items.length ? utcMoment(items[0].label) : null;
        return ms === null ? '' : LL.formatDateTime(ms);
      }
    });
    const plugins = LL.charts ? [LL.charts.markers, LL.charts.thresholds, LL.charts.crosshair] : [];
    // eslint-disable-next-line no-undef
    const chart = new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: built,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          cmpBg: { color: colors.background },
          tooltip: tooltip,
          llMarkers: { items: reportState.markers.slice(), highlight: null },
          llThresholds: { items: thresholdsForQuery(reportState.thresholds, ql, reportState.queryLabels) }
        },
        scales: {
          x: {
            ticks: {
              color: colors.text,
              maxRotation: 0,
              autoSkipPadding: 16,
              callback(value) { return clockLabel(this.getLabelForValue(value)); }
            },
            grid: { color: colors.grid }
          },
          y: { ticks: { color: colors.text }, grid: { color: colors.grid } }
        }
      },
      plugins: plugins
    });
    const shown = Math.min(built.datasets.length, MAX_VISIBLE_SERIES);
    canvas.setAttribute('aria-label', `График «${ql}»: серий ${built.datasets.length}${built.datasets.length > shown ? `, показаны ${shown} с наибольшим средним` : ''}`);
    reportState.charts.push(chart);
    if (group) group.charts.push(chart);
    if (legendHost && LL.charts && typeof LL.charts.legendChips === 'function') {
      const chips = LL.charts.legendChips(chart, legendHost, { filename: `report-${domain}-${ql}`, maxVisible: MAX_VISIBLE_SERIES });
      if (group) group.chips.push(chips);
    }
    return chart;
  }

  // Charts of one domain tab share hover and zoom once all of them are drawn.
  function wireChartGroup(group) {
    const live = group.charts.filter((chart) => chart && chart.canvas);
    if (!live.length || !LL.charts) return;
    if (typeof LL.charts.syncHover === 'function') group.sync = LL.charts.syncHover(live);
    if (typeof LL.charts.zoom === 'function') group.zoom = LL.charts.zoom(live, { onChange: () => syncZoomButton() });
  }
  function syncZoomButton() {
    const btn = document.querySelector('#rep-domain-tabs .chart-zoom-reset');
    if (!btn) return;
    const group = reportState.groups[reportState.activePane];
    btn.hidden = !(group && group.zoom && group.zoom.range());
  }
  // The domain tab bar sticks to the top; a sentinel just above it tells when it is stuck (for the shadow).
  function observeSticky(bar, sentinel) {
    if (!('IntersectionObserver' in window)) return;
    let observer = null;
    const connect = () => {
      if (observer) observer.disconnect();
      const top = parseFloat(getComputedStyle(bar).top) || 0;
      observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          bar.classList.toggle('is-stuck', !entry.isIntersecting && entry.boundingClientRect.top < top + 1);
        });
      }, { rootMargin: `-${Math.ceil(top) + 1}px 0px 0px 0px`, threshold: 0 });
      observer.observe(sentinel);
    };
    connect();
    const narrow = window.matchMedia ? window.matchMedia('(max-width: 1023px)') : null;
    if (narrow && typeof narrow.addEventListener === 'function') narrow.addEventListener('change', connect);
    else if (narrow && typeof narrow.addListener === 'function') narrow.addListener(connect);
  }

  // ---- print / PDF -------------------------------------------------------------------------------
  // Print shows every tab panel with its charts, opens collapsed sections and draws the charts in the light
  // theme (the print stylesheet forces light tokens too); everything is restored after printing.
  const printState = { active: false, theme: null, opened: [], restylePending: false };
  function restyleReportCharts() {
    if (LL.ui && typeof LL.ui.restyleCharts === 'function') LL.ui.restyleCharts();
    Object.keys(reportState.groups).forEach((id) => {
      (reportState.groups[id].chips || []).forEach((chips) => { if (chips && typeof chips.refresh === 'function') chips.refresh(); });
    });
    reportState.charts.forEach((chart) => { try { if (chart && chart.canvas) chart.resize(); } catch (e) { /* destroyed */ } });
  }
  function enterPrintMode() {
    if (printState.active) return;
    printState.active = true;
    const root = document.documentElement;
    printState.theme = root.getAttribute('data-theme');
    root.classList.add('report-printing');
    root.setAttribute('data-theme', 'light');
    printState.opened = Array.from(document.querySelectorAll('#main details:not([open])'));
    printState.opened.forEach((details) => { details.open = true; });
    highlightCharts(null);
    restyleReportCharts();
  }
  const printMedia = window.matchMedia ? window.matchMedia('print') : null;
  function leavePrintMode() {
    const root = document.documentElement;
    root.classList.remove('report-print-prep');
    if (!printState.active) return;
    printState.active = false;
    root.classList.remove('report-printing');
    if (printState.theme) root.setAttribute('data-theme', printState.theme);
    else root.removeAttribute('data-theme');
    printState.opened.forEach((details) => { details.open = false; });
    printState.opened = [];
    // While print media still applies, the print stylesheet keeps the light tokens: restyle once it is gone.
    if (printMedia && printMedia.matches) printState.restylePending = true;
    else restyleReportCharts();
  }
  function onPrintMediaChange(event) {
    if (event.matches) { enterPrintMode(); return; }
    if (printState.active) { leavePrintMode(); return; }
    if (printState.restylePending) {
      printState.restylePending = false;
      restyleReportCharts();
    }
  }
  async function printReport(btn) {
    if (btn) btn.disabled = true;
    // All tab panels are laid out first so charts of tabs never opened are drawn at their real size.
    document.documentElement.classList.add('report-print-prep');
    try {
      await Promise.all(Object.keys(reportState.paneRenderers).map((id) => Promise.resolve(reportState.paneRenderers[id]()).catch(() => null)));
      enterPrintMode();
      await new Promise((resolve) => (window.requestAnimationFrame ? window.requestAnimationFrame(() => resolve()) : setTimeout(resolve, 16)));
      window.print();
    } finally {
      if (btn) btn.disabled = false;
    }
  }
  function wirePrint() {
    const btn = document.getElementById('printReport');
    if (btn) btn.addEventListener('click', () => { printReport(btn); });
    window.addEventListener('beforeprint', enterPrintMode);
    window.addEventListener('afterprint', leavePrintMode);
    // Browsers without print events (and page.pdf-style captures) still switch the print media query.
    if (printMedia && typeof printMedia.addEventListener === 'function') printMedia.addEventListener('change', onPrintMediaChange);
    else if (printMedia && typeof printMedia.addListener === 'function') printMedia.addListener(onPrintMediaChange);
  }

  async function engineerLoad() {
    try {
      const run = getRunFromPath(); if (!run) return;
      const r = await fetch('/engineer_summary?run_name=' + encodeURIComponent(run));
      if (!r.ok) return;
      const j = await r.json();
      const ed = document.getElementById('engineerEditor');
      const updated = document.getElementById('engineerUpdated');
      const raw = String(j.content_html || '');
      const html = sanitizeOrEscape(raw, NOTE_PURIFY, raw);
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
    if (linkBtn) {
      linkBtn.addEventListener('click', async () => {
        // The dialog steals focus, so remember the selection and restore it before inserting the link.
        const selection = window.getSelection();
        const range = selection && selection.rangeCount ? selection.getRangeAt(0).cloneRange() : null;
        const url = await LL.ui.prompt({
          title: 'Вставить ссылку',
          label: 'Адрес',
          value: 'https://',
          validate: (v) => (/^https?:\/\/\S+$/i.test(v.trim()) ? '' : 'Укажите адрес вида https://…')
        });
        if (!url) return;
        const editorEl = document.getElementById('engineerEditor');
        if (editorEl) editorEl.focus();
        if (range) { const s = window.getSelection(); s.removeAllRanges(); s.addRange(range); }
        editorExec('createLink', url.trim());
      });
    }
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

  // Charts of one domain are drawn on first activation of its tab, not all at once. Returns a promise
  // that settles when every chart of the tab is drawn and the tab's hover and zoom group is wired.
  function renderDomainPane(run, domain, pane, queryLabels, domainIdx) {
    if (pane.dataset.rendered === '1') return pane.$llReady || Promise.resolve();
    pane.dataset.rendered = '1';
    pane.innerHTML = '';
    const printTitle = document.createElement('h3');
    printTitle.className = 'report-print-title';
    printTitle.textContent = `Графики: ${LL.domainTitleFor(domain, reportDomainTitles)}`;
    pane.appendChild(printTitle);
    const hint = document.createElement('p');
    hint.className = 'report-chart-hint';
    hint.textContent = 'Выделите участок графика мышью, чтобы увеличить его на всех графиках вкладки; двойной щелчок возвращает весь тест.';
    pane.appendChild(hint);
    const group = { charts: [], chips: [], sync: null, zoom: null };
    reportState.groups[pane.id] = group;
    const jobs = queryLabels.map((ql, qi) => {
      const wrap = document.createElement('section');
      wrap.className = 'report-chart';
      const title = document.createElement('h4');
      title.className = 'report-chart-title';
      title.textContent = ql;
      const canvasBox = document.createElement('div');
      canvasBox.className = 'report-chart-canvas';
      const cnv = document.createElement('canvas');
      cnv.id = `rep-chart-${domainIdx}-${qi}`;
      cnv.setAttribute('role', 'img');
      cnv.setAttribute('aria-label', `График «${ql}»`);
      canvasBox.appendChild(cnv);
      const legend = document.createElement('div');
      legend.className = 'report-chart-legend';
      wrap.appendChild(title);
      wrap.appendChild(canvasBox);
      wrap.appendChild(legend);
      pane.appendChild(wrap);
      return reportsDrawOne(run, domain, ql, cnv, legend, group).catch(() => null);
    });
    pane.$llReady = Promise.all(jobs).then(() => { wireChartGroup(group); syncZoomButton(); });
    return pane.$llReady;
  }

  async function reportsRenderDomains(schema) {
    const run = getRunFromPath(); if (!run) return;
    const root = document.getElementById('rep-domain-tabs'); if (!root) return;
    root.innerHTML = 'Загрузка…';
    const domainsOrder = ['lt_framework', 'microservices', 'jvm', 'database', 'kafka', 'hard_resources', 'lt_generators'];
    const domains = Object.keys(schema || {}).sort((a, b) => {
      const ia = domainsOrder.indexOf(a); const ib = domainsOrder.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
    // Sticky bar: the domain tabs and, while the open tab is zoomed, «Сбросить масштаб».
    const bar = document.createElement('div'); bar.className = 'report-tabs-bar tabs-sticky';
    const sentinel = document.createElement('div'); sentinel.className = 'tabs-sticky-sentinel'; sentinel.setAttribute('aria-hidden', 'true');
    const tabsNav = document.createElement('div'); tabsNav.className = 'tabs'; tabsNav.setAttribute('aria-label', 'Графики по доменам');
    const reset = document.createElement('button');
    reset.type = 'button';
    reset.className = 'chart-zoom-reset';
    reset.hidden = true;
    reset.title = 'Вернуть весь интервал теста на графиках вкладки (или двойной щелчок по графику)';
    reset.innerHTML = '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg><span>Сбросить масштаб</span>';
    reset.addEventListener('click', () => {
      const group = reportState.groups[reportState.activePane];
      if (group && group.zoom) group.zoom.reset();
    });
    bar.appendChild(tabsNav);
    bar.appendChild(reset);
    const tabsBody = document.createElement('div');
    const idBase = 'rep-dom-tab-';
    const paneRenderers = {};
    domains.forEach((domain, di) => {
      const list = (schema[domain] || []).map((x) => x.query_label);
      const btn = document.createElement('button'); btn.className = 'tab' + (di === 0 ? ' active' : '');
      btn.textContent = `${LL.domainTitleFor(domain, reportDomainTitles)} (${list.length})`; btn.dataset.target = idBase + di; tabsNav.appendChild(btn);
      const pane = document.createElement('div'); pane.id = idBase + di; pane.className = 'tabpanel' + (di === 0 ? ' active' : '');
      pane.innerHTML = '<div class="chart-pending">Графики загрузятся при открытии вкладки…</div>';
      paneRenderers[pane.id] = () => renderDomainPane(run, domain, pane, list, di);
      tabsBody.appendChild(pane);
    });
    reportState.paneRenderers = paneRenderers;
    reportState.queryLabels = domains.reduce((all, domain) => all.concat((schema[domain] || []).map((x) => x.query_label)), []);
    if (!domains.length) { root.innerHTML = '<div class="empty-state">Для этого запуска нет сохранённых метрик.</div>'; return; }
    root.innerHTML = ''; root.appendChild(sentinel); root.appendChild(bar); root.appendChild(tabsBody);
    LL.ui.tabs(tabsNav, {
      onChange: (target) => {
        reportState.activePane = target;
        if (paneRenderers[target]) paneRenderers[target]();
        syncZoomButton();
      }
    });
    observeSticky(bar, sentinel);
  }

  // Pure helpers for tests (tests/test_ui_report_js.py, tests/test_report_markers.py).
  LL.report = LL.report || {};
  LL.report._internals = {
    utcMoment: utcMoment,
    reportMarkers: reportMarkers,
    slaThresholds: slaThresholds,
    thresholdsForQuery: thresholdsForQuery,
    resolveQueryLabel: resolveQueryLabel,
    orderSeries: orderSeries,
    buildChartData: buildChartData,
    chosenSlaStep: chosenSlaStep,
    markerMode: markerMode,
    slaBarModel: slaBarModel,
    heroKpis: heroKpis,
    heroTypeEntries: heroTypeEntries,
    verdictSourceText: verdictSourceText,
    verdictHistoryHtml: verdictHistoryHtml,
    verdictDialogHtml: verdictDialogHtml,
    verdictFormProblem: verdictFormProblem,
    reportState: reportState
  };

  document.addEventListener('DOMContentLoaded', async () => {
    wirePrint();
    await loadReportRef();
    setPageTitle(getRunFromPath());
    wireRenameReport();
    const schema = await reportsLoadSchema(getRunFromPath());
    await reportsLoadLlm();
    await reportsRenderDomains(schema);
    wireEngineerEditor();
    await engineerLoad();
    wireConfluencePublish();
    await loadConfluencePublication();
  });
})();


