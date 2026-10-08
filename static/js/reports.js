// Reports page logic
(function () {
  // Chart background plugin
  const cmpBgPlugin = { id: 'cmpBg', beforeDraw(chart, args, opts) { const { ctx, chartArea } = chart; if (!chartArea) return; ctx.save(); ctx.fillStyle = (opts && opts.color) || '#151515'; ctx.fillRect(chartArea.left, chartArea.top, chartArea.right - chartArea.left, chartArea.bottom - chartArea.top); ctx.restore(); } };
  if (window.Chart && Chart.register) Chart.register(cmpBgPlugin);

  const LL = window.LoadLens;

  let feedbackByKey = {};

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
  function markdownToSafeHtml(md) {
    const source = safeStr(md);
    if (!source) return '';
    let html = (window.marked && typeof marked.parse === 'function')
      ? window.marked.parse(source)
      : escapeHtml(source).replace(/\n/g, '<br>');
    try { if (window.DOMPurify) html = window.DOMPurify.sanitize(html); } catch (e) {}
    return html;
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
    const sections = [
      renderKeyValueTable('', verdictRows),
      renderRationaleBox(verdictRationale),
      peakTableApplies(domain, peak) ? renderKeyValueTable('Пиковая производительность', peakRows) : '',
      renderProblemRecommendationRows(pairFindingsWithActions(findings, actions, domain)),
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
  function confidenceLevelLabel(scores) {
    const s = scores || {};
    const judge = s.judge || {};
    const raw = Number.isFinite(Number(s.final_score)) ? Number(s.final_score) : Number(judge.overall);
    if (!Number.isFinite(raw) || raw <= 0) return '';
    if (raw >= 0.9) return 'очень высокая';
    if (raw >= 0.75) return 'высокая';
    if (raw >= 0.55) return 'умеренная';
    return 'низкая';
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
      `<summary><span>${titleHtml}</span><small class="report-collapsible-subtitle">как выбран и проверен ответ ИИ</small></summary>`,
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
    memory_usage: 'Память, %'
  };
  function formatNumber(value) {
    const n = Number(value);
    if (value === null || value === undefined || value === '' || !Number.isFinite(n)) return '—';
    return Number.isInteger(n) ? String(n) : n.toFixed(2).replace(/\.?0+$/, '');
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
        `<td class="num">${escapeHtml(formatNumber(check && check.actual))}</td>`,
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
    const bits = checks.map((check) => {
      const label = SLA_CHECK_LABELS[safeStr(check && check.name)] || safeStr(check && check.name) || 'критерий';
      const actual = formatNumber(check && check.actual);
      return label + (actual === '—' ? '' : ' ' + actual) + ' — ' + slaStatusWord(check && check.passed);
    });
    const head = 'SLA проверено на ступени ' + (chosen.index || '') + ' (' + stepInterval(chosen) + ')';
    const windowBit = windowText
      ? ', окно ' + windowText + (levelText ? ', RPS окна ' + levelText : '')
      : '';
    const meanBit = differs ? '. RPS в таблице — среднее всего интервала, пороги считаются по этому окну' : '';
    return head + windowBit + meanBit + '. ' + bits.join('; ') + '.';
  }
  function linkSlaToSteps(table, finalRow) {
    const steps = (table.steps || []).map((step) => Object.assign({}, step));
    const checks = slaChecksOf(finalRow);
    const window = slaWindowOf(finalRow);
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
      return [
        `<tr class="${classes.join(' ')}">`,
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
      `<p class="step-note">RPS — уровень ступени. Время отклика — p95 на плато, без хвоста после просадки.${source}</p>`,
      table.slaNote ? `<p class="step-sla">${escapeHtml(table.slaNote)}</p>` : '',
      '</div></details>'
    ].join('');
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
  async function renderLoadSteps(finalRow) {
    const box = stepsHost();
    if (!box) return;
    const embedded = embeddedLoadStepTable(finalRow);
    if (embedded) {
      const linked = linkSlaToSteps(embedded, finalRow);
      box.hidden = false;
      box.innerHTML = renderLoadStepTableHtml(Object.assign({}, embedded, linked));
      return;
    }
    const run = getRunFromPath();
    if (!finalRow || !run) {
      box.hidden = true;
      box.innerHTML = '';
      return;
    }
    try {
      const sla = parseJsonish(finalRow.sla_details) || {};
      const checks = Array.isArray(sla.checks) ? sla.checks : [];
      let pack = await fetchContextPack(run, 'lt_framework');
      if (!pack || !Array.isArray(pack.load_steps) || !pack.load_steps.length) {
        pack = await fetchContextPack(run, 'final');
      }
      const table = stepTableFromPack(pack, checks);
      if (!table) {
        box.hidden = true;
        box.innerHTML = '';
        return;
      }
      const linked = linkSlaToSteps(table, finalRow);
      box.hidden = false;
      box.innerHTML = renderLoadStepTableHtml(Object.assign({}, table, linked));
    } catch (e) {
      box.hidden = true;
      box.innerHTML = '';
    }
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

  function renderHero(finalRow, allRows) {
    const hero = document.getElementById('reportHero');
    if (!hero) return;
    if (!finalRow) {
      hero.innerHTML = '<div class="hero-card hero-empty">Итоговый отчёт для этого запуска ещё не сформирован. Если генерация идёт, обновите страницу позже; статус задачи виден в архиве.</div>';
      return;
    }
    const effective = verdictTone(finalRow.verdict);
    const llmVerdict = safeStr(finalRow.llm_verdict);
    const slaVerdict = safeStr(finalRow.sla_verdict);
    const parsed = parseJsonish(finalRow.parsed) || {};
    const peak = (parsed.peak_performance && typeof parsed.peak_performance === 'object') ? parsed.peak_performance : {};
    const service = safeStr(finalRow.service) || getServiceFromPath();
    const startMs = Number(finalRow.start_ms);
    const endMs = Number(finalRow.end_ms);
    const period = (Number.isFinite(startMs) && Number.isFinite(endMs))
      ? `${LL.formatDateTime(startMs)} — ${LL.formatDateTime(endMs)}`
      : '';
    const durationMin = (Number.isFinite(startMs) && Number.isFinite(endMs)) ? Math.round((endMs - startMs) / 60000) : null;
    const verdictSource = slaVerdict
      ? `Определён по SLA-критериям. Оценка ИИ: ${llmVerdict || '—'}.`
      : 'Определён оценкой ИИ: SLA-критерии не заданы.';
    const maxRps = formatNumber(peak.max_rps);
    const peakMethod = safeStr(peak.method);
    const verdictTermKey = { 'Успешно': 'verdict_success', 'Есть риски': 'verdict_risk', 'Провал': 'verdict_fail' }[effective.text] || 'verdict_na';
    const rpsTermKey = /stable/i.test(peakMethod) ? 'stable_max' : 'peak_max';
    hero.innerHTML = [
      '<div class="hero-card">',
      `<div class="hero-label">${LL.ui.term('sla_first', 'Вердикт по тесту')}</div>`,
      `<div class="hero-verdict-row"><abbr class="term" data-term="${verdictTermKey}" data-tip="${escapeHtml(LL.GLOSSARY[verdictTermKey] || '')}" title="${escapeHtml(LL.GLOSSARY[verdictTermKey] || '')}" tabindex="0" style="text-decoration:none"><span class="report-verdict-pill ${effective.className}">${escapeHtml(effective.text)}</span></abbr></div>`,
      `<div class="hero-sub">${escapeHtml(verdictSource)}</div>`,
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
      `<div class="hero-feedback">${renderVoteGroup('final', 'verdict', '', 'Согласен с вердиктом', 'Не согласен')}</div>`,
      renderHeroMeta([
        { label: 'Сервис', value: service || '—' },
        { label: 'Тип теста', labelHtml: LL.ui.term('step_profile', 'Тип теста'), value: LL.testTypeLabel(finalRow.test_type) },
        { label: 'Период', value: period || '—', note: period ? `${durationMin} мин · ${LL.timeZoneLabel()}` : '' },
        ...(peakTableApplies('final', peak) ? [
          { label: 'Максимальный RPS', labelHtml: LL.ui.term(rpsTermKey, 'Максимальный RPS'), value: maxRps, note: peakMethod ? `метод: ${peakMethod}` : '' },
          { label: 'Время пика', value: safeStr(peak.max_time) || '—' },
          { label: 'Время деградации', value: safeStr(peak.drop_time) || '—' }
        ] : [])
      ]),
      '</div>',
      `<div class="hero-card">${renderSlaChecklist(finalRow)}</div>`
    ].join('');
    bindFeedbackVotes(hero);
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
      renderLoadSteps(null);
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
    renderHero(arr.find((x) => x.domain === 'final') || null, arr);
    renderLoadSteps(arr.find((x) => x.domain === 'final') || null);
    renderForecastLink();
    const domainsOrder = ['final', 'jvm', 'database', 'kafka', 'microservices', 'hard_resources', 'lt_framework', 'application_logs'];
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
      btn.appendChild(document.createTextNode(LL.domainTitle(x.domain)));
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
      pane.innerHTML = `<div class="report-analysis-card">${aiBadge}${html}</div>`;
      tabsBody.appendChild(pane);
    });
    box.innerHTML = ''; box.appendChild(tabsNav); box.appendChild(tabsBody);
    LL.ui.tabs(tabsNav, { onChange: () => scheduleLegendSync() });
    LL.ui.applyTerms(box);
    bindFeedbackVotes(box);
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
    const datasets = Object.keys(map).map((k) => { const color = LL.colorFor(k); return { label: k, data: labels.map((t) => (map[k].has(t) ? map[k].get(t) : null)), borderColor: color, backgroundColor: color, pointRadius: 0, borderWidth: 2, spanGaps: true }; });
    const ctx = document.getElementById(canvasId).getContext('2d');
    const colors = LL.ui.chartColors();
    // eslint-disable-next-line no-undef
    const chart = new Chart(ctx, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        interaction: { mode: 'nearest', intersect: false },
        plugins: { legend: { display: false }, cmpBg: { color: colors.background } },
        scales: {
          x: {
            title: { display: true, text: 'Время', color: colors.text },
            ticks: {
              color: colors.text,
              callback(value) {
                const raw = this.getLabelForValue(value);
                const d = new Date(raw);
                const hh = String(d.getHours()).padStart(2, '0');
                const mm = String(d.getMinutes()).padStart(2, '0');
                return `${hh}:${mm}`;
              }
            },
            grid: { color: colors.grid }
          },
          y: { ticks: { color: colors.text }, grid: { color: colors.grid } }
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
    const theme = LL.ui.chartColors();
    ctx.fillStyle = theme.background;
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(canvas, 0, 0);
    if (rows.length) {
      let y = chartH + pad;
      ctx.fillStyle = theme.text; ctx.font = '16px Montserrat, Arial, sans-serif';
      ctx.fillText('Легенда', pad, y);
      y += titleH;
      ctx.font = '13px Montserrat, Arial, sans-serif';
      rows.forEach((r) => {
        ctx.fillStyle = r.color || '#888';
        ctx.fillRect(pad, y - 12, 14, 14);
        ctx.strokeStyle = theme.grid; ctx.strokeRect(pad, y - 12, 14, 14);
        ctx.fillStyle = theme.text;
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
      const tr = document.createElement('tr'); tr.className = 'cmp-legend-row' + (row.visible ? '' : ' cmp-off');
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

  // Charts of one domain are drawn on first activation of its tab, not all at once.
  function renderDomainPane(run, domain, pane, queryLabels, domainIdx) {
    if (pane.dataset.rendered === '1') return;
    pane.dataset.rendered = '1';
    pane.innerHTML = '';
    queryLabels.forEach((ql, qi) => {
      const wrap = document.createElement('div'); wrap.className = 'cmp-chart-wrap'; wrap.style.marginTop = '12px';
      const canvasBox = document.createElement('div'); canvasBox.className = 'cmp-chart-canvas';
      const title = document.createElement('div'); title.style.padding = '8px 0'; title.style.fontWeight = '600'; title.textContent = ql;
      const cnv = document.createElement('canvas'); cnv.id = `rep-chart-${domainIdx}-${qi}`; cnv.height = 120; canvasBox.appendChild(title); canvasBox.appendChild(cnv);
      const legend = document.createElement('div'); legend.className = 'cmp-legend-panel'; legend.innerHTML = '<h4>Легенда</h4><div class=\"cmp-legend-controls\"><button class=\"hideAll\">Выключить все</button><button class=\"showAll\">Включить все</button><button class=\"downloadPng\">Скачать PNG</button></div><div class=\"table\"></div>';
      wrap.appendChild(canvasBox); wrap.appendChild(legend); pane.appendChild(wrap);
      setTimeout(async () => { await reportsDrawOne(run, domain, ql, cnv.id, legend); }, 0);
    });
  }

  async function reportsRenderDomains(schema) {
    const run = getRunFromPath(); if (!run) return;
    const root = document.getElementById('rep-domain-tabs'); if (!root) return;
    root.innerHTML = 'Загрузка…';
    const domainsOrder = ['lt_framework', 'microservices', 'jvm', 'database', 'kafka', 'hard_resources'];
    const domains = Object.keys(schema || {}).sort((a, b) => {
      const ia = domainsOrder.indexOf(a); const ib = domainsOrder.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
    const tabsNav = document.createElement('div'); tabsNav.className = 'tabs'; tabsNav.setAttribute('aria-label', 'Графики по доменам');
    const tabsBody = document.createElement('div');
    const idBase = 'rep-dom-tab-';
    const paneRenderers = {};
    domains.forEach((domain, di) => {
      const list = (schema[domain] || []).map((x) => x.query_label);
      const btn = document.createElement('button'); btn.className = 'tab' + (di === 0 ? ' active' : '');
      btn.textContent = `${LL.domainTitle(domain)} (${list.length})`; btn.dataset.target = idBase + di; tabsNav.appendChild(btn);
      const pane = document.createElement('div'); pane.id = idBase + di; pane.className = 'tabpanel' + (di === 0 ? ' active' : '');
      pane.innerHTML = '<div class="chart-pending">Графики загрузятся при открытии вкладки…</div>';
      paneRenderers[pane.id] = () => renderDomainPane(run, domain, pane, list, di);
      tabsBody.appendChild(pane);
    });
    if (!domains.length) { root.innerHTML = '<div class="empty-state">Для этого запуска нет сохранённых метрик.</div>'; return; }
    root.innerHTML = ''; root.appendChild(tabsNav); root.appendChild(tabsBody);
    LL.ui.tabs(tabsNav, {
      onChange: (target) => {
        if (paneRenderers[target]) paneRenderers[target]();
        scheduleLegendSync();
      }
    });
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
    attachPanelObservers();
    scheduleLegendSync();
  });
})();


