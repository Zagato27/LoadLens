// SPDX-License-Identifier: Apache-2.0
// Reports page logic
(function () {
  // Chart background plugin: the plot area gets the color passed in options.plugins.cmpBg (ui.restyleCharts
  // keeps it in sync with the theme), else --bg of the current theme; without the token the canvas stays
  // transparent and shows the page.
  function chartBackground() {
    try {
      const css = getComputedStyle(document.documentElement);
      return css.getPropertyValue('--bg').trim();
    } catch (e) { return ''; }
  }
  const cmpBgPlugin = { id: 'cmpBg', beforeDraw(chart, args, opts) { const { ctx, chartArea } = chart; if (!chartArea) return; const color = (opts && opts.color) || chartBackground(); if (!color) return; ctx.save(); ctx.fillStyle = color; ctx.fillRect(chartArea.left, chartArea.top, chartArea.right - chartArea.left, chartArea.bottom - chartArea.top); ctx.restore(); } };
  if (window.Chart && Chart.register) Chart.register(cmpBgPlugin);

  const LL = window.LoadLens;
  // Report-wide chart state: markers and thresholds of the final row, every chart drawn so far and the
  // per-tab chart groups (synchronized hover, drag zoom, legend chips).
  const reportState = { finalRow: null, stepTable: null, markers: [], thresholds: [], queryLabels: [], charts: [], groups: {}, activePane: '', paneRenderers: {}, profile: null, domainVerdicts: {} };
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
        title: target === 'verdict' ? LL.t('report.feedback.why_verdict') : LL.t('report.feedback.why_finding'),
        message: LL.t('report.feedback.comment_hint'),
        label: LL.t('report.feedback.comment_label'),
        confirmText: LL.t('common.action.save'),
        cancelText: LL.t('common.action.cancel'),
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
      if (!resp.ok) throw new Error((data && data.error) || LL.t('report.feedback.save_failed'));
      setVote(domain, target, findingId, (data && data.vote) || vote, (data && data.comment) || comment);
      document.querySelectorAll('.feedback-votes').forEach(paintVoteGroup);
    } catch (e) {
      LL.ui.alert({ title: LL.t('report.feedback.not_saved'), message: e.message || String(e) });
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
    if (prompt != null && completion != null) return LL.t('report.usage.calls_tokens', { calls: usage.calls, prompt: prompt, completion: completion });
    return LL.t('report.usage.calls', { calls: usage.calls });
  }
  function isUnverifiedFinding(item) {
    const verification = item && item.verification;
    return !!(verification && typeof verification === 'object' && verification.status === 'unverified');
  }
  function renderUnverifiedPill(item) {
    if (!isUnverifiedFinding(item)) return '';
    const unmatched = Array.isArray(item.verification.unmatched) ? item.verification.unmatched.map(safeStr).filter(Boolean) : [];
    const missing = unmatched.length ? ` ${LL.t('report.finding.unmatched', { items: unmatched.join(', ') })}` : '';
    const tip = `${(LL.GLOSSARY && LL.GLOSSARY.unverified) || ''}${missing}`.trim();
    return `<abbr class="term" data-term="unverified" data-tip="${escapeHtml(tip)}" title="${escapeHtml(tip)}" tabindex="0" style="text-decoration:none"><span class="pill warn finding-unverified">${escapeHtml(LL.t('report.finding.unverified'))}</span></abbr>`;
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

  // The title is the run name itself (the crumbs and the meta line say what it is); the tab title keeps the words.
  function setPageTitle(run) {
    const title = document.getElementById('pageTitle');
    if (title) {
      title.textContent = run || LL.t('report.page.title');
      title.title = run ? LL.t('report.page.title_run', { run: run }) : '';
    }
    document.title = run ? LL.t('report.page.doc_title_run', { run: run }) : LL.t('report.page.doc_title');
    renderBreadcrumbs(getServiceFromPath(), run);
  }

  function renderBreadcrumbs(service, run) {
    const nav = document.getElementById('reportBreadcrumbs');
    if (!nav) return;
    const sep = '<span class="sep" aria-hidden="true">/</span>';
    const parts = [`<a href="/reports">${escapeHtml(LL.t('report.page.breadcrumb_archive'))}</a>`];
    if (service) parts.push(`${sep}<a href="/reports?service=${encodeURIComponent(service)}">${escapeHtml(service)}</a>`);
    if (run) parts.push(`${sep}<span class="current" aria-current="page">${escapeHtml(run)}</span>`);
    nav.innerHTML = parts.join('');
  }

  function wireRenameReport() {
    const btn = document.getElementById('renameReport');
    const form = document.getElementById('renameReportForm');
    const input = document.getElementById('renameReportInput');
    const save = document.getElementById('renameReportSave');
    const cancel = document.getElementById('renameReportCancel');
    const status = document.getElementById('renameReportStatus');
    // The form takes the place of the title while it is open.
    const title = document.getElementById('pageTitle');
    if (!btn || !form || !input || !save || !cancel) return;

    function openForm() {
      input.value = getRunFromPath() || '';
      form.style.display = 'flex';
      btn.style.display = 'none';
      if (title) title.hidden = true;
      if (status) status.textContent = '';
      input.focus();
      input.select();
    }

    function closeForm() {
      form.style.display = 'none';
      btn.style.display = '';
      if (title) title.hidden = false;
      if (status) status.textContent = '';
    }

    async function submit() {
      const run = getRunFromPath();
      const next = String(input.value || '').trim();
      if (!run) {
        if (status) status.textContent = LL.t('report.page.no_run');
        return;
      }
      if (!next) {
        if (status) status.textContent = LL.t('report.rename.empty');
        return;
      }
      save.disabled = true;
      if (status) status.textContent = LL.t('report.status.saving');
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
          if (status) status.textContent = data.error || LL.t('report.rename.failed');
          save.disabled = false;
          return;
        }
        const url = data.page_url || (reportRef.run_id ? ('/reports/' + encodeURIComponent(reportRef.run_id)) : location.pathname);
        LL.ui.toast(LL.t('report.rename.done', { name: data.run_name }), { tone: 'ok' });
        location.replace(url);
      } catch (e) {
        if (status) status.textContent = LL.t('report.rename.failed');
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
      if (btn) btn.textContent = LL.t('report.confluence.update');
    } else {
      link.style.display = 'none';
      if (btn) btn.textContent = LL.t('report.confluence.add');
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

  // Status next to the publish button; `state` marks a status that clears itself (data-state, not its text).
  function setPublishStatus(text, state) {
    const statusEl = document.getElementById('confluenceStatus');
    if (!statusEl) return;
    statusEl.textContent = text;
    if (state) statusEl.dataset.state = state;
    else delete statusEl.dataset.state;
  }

  async function pollConfluenceJob(jobId) {
    const btn = document.getElementById('publishConfluence');
    for (let i = 0; i < 180; i += 1) {
      const resp = await fetch('/job_status/' + encodeURIComponent(jobId));
      const job = await resp.json();
      const pct = Number(job.progress || 0);
      setPublishStatus((job.message || LL.t('report.confluence.publishing')) + (pct ? ` (${pct}%)` : ''));
      if (job.status === 'done') {
        setConfluenceLink(job.page_url || '');
        setPublishStatus(LL.t('report.confluence.published'), 'published');
        if (btn) btn.disabled = false;
        LL.ui.toast(LL.t('report.confluence.published_toast'), { tone: 'ok' });
        setTimeout(() => {
          const statusEl = document.getElementById('confluenceStatus');
          if (statusEl && statusEl.dataset.state === 'published') setPublishStatus('');
        }, 2500);
        return;
      }
      if (job.status === 'error' || job.status === 'not_found') {
        const headline = job.message || LL.t('report.confluence.failed');
        const text = job.error ? `${headline}: ${job.error}` : headline;
        setPublishStatus(text);
        LL.ui.toast(text, { tone: 'error' });
        if (btn) btn.disabled = false;
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 1500));
    }
    setPublishStatus(LL.t('report.confluence.too_long'));
    if (btn) btn.disabled = false;
  }

  function wireConfluencePublish() {
    const btn = document.getElementById('publishConfluence');
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const run = getRunFromPath();
      if (!run) {
        setPublishStatus(LL.t('report.page.no_run'));
        return;
      }
      btn.disabled = true;
      setPublishStatus(LL.t('report.confluence.publishing'));
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
          setPublishStatus(data.error || data.message || LL.t('report.confluence.start_failed'));
          btn.disabled = false;
          return;
        }
        await pollConfluenceJob(data.job_id);
      } catch (e) {
        setPublishStatus(LL.t('report.confluence.start_failed'));
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
    if (safeStr(system.name)) sysBits.push(`**${LL.t('report.context.system')}:** ${safeStr(system.name)}`);
    if (safeStr(system.domain)) sysBits.push(`**${LL.t('report.context.domain')}:** ${safeStr(system.domain)}`);
    if (safeStr(system.description)) sysBits.push(safeStr(system.description));
    if (safeStr(system.test_goal)) sysBits.push(`**${LL.t('report.context.test_goal')}:** ${safeStr(system.test_goal)}`);
    if (sysBits.length) {
      lines.push('');
      sysBits.forEach((item) => lines.push(`- ${item}`));
    }
    if (safeStr(architecture.style)) {
      lines.push('');
      lines.push(`- **${LL.t('report.context.architecture')}:** ${safeStr(architecture.style)}`);
    }
    const components = Array.isArray(architecture.components) ? architecture.components : [];
    if (components.length) {
      lines.push('');
      lines.push(`#### ${LL.t('report.context.components')}`);
      components.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const name = safeStr(item.name || item.id);
        if (!name) return;
        const bits = [];
        if (safeStr(item.role)) bits.push(`${LL.t('report.context.role')}: ${safeStr(item.role)}`);
        if (safeStr(item.criticality)) bits.push(`criticality: ${safeStr(item.criticality)}`);
        const technologies = Array.isArray(item.technologies) ? item.technologies.map((x) => safeStr(x)).filter(Boolean) : [];
        if (technologies.length) bits.push(`tech: ${technologies.join(', ')}`);
        lines.push(bits.length ? `- **${name}** (${bits.join('; ')})` : `- **${name}**`);
      });
    }
    const deps = Array.isArray(architecture.dependencies) ? architecture.dependencies : [];
    if (deps.length) {
      lines.push('');
      lines.push(`#### ${LL.t('report.context.dependencies')}`);
      deps.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const from = safeStr(item.from);
        const to = safeStr(item.to);
        if (!from && !to) return;
        const kind = safeStr(item.kind);
        const purpose = safeStr(item.purpose);
        const suffix = [kind && `${LL.t('report.context.kind')}: ${kind}`, purpose && purpose].filter(Boolean).join('; ');
        lines.push(`- ${from || 'unknown'} -> ${to || 'unknown'}${suffix ? ` (${suffix})` : ''}`);
      });
    }
    const flows = Array.isArray(loadModel.critical_user_flows) ? loadModel.critical_user_flows : [];
    if (flows.length) {
      lines.push('');
      lines.push(`#### ${LL.t('report.context.critical_flows')}`);
      flows.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const name = safeStr(item.name || item.id);
        if (!name) return;
        const steps = Array.isArray(item.steps) ? item.steps.map((x) => safeStr(x)).filter(Boolean) : [];
        const successSignals = Array.isArray(item.success_signals) ? item.success_signals.map((x) => safeStr(x)).filter(Boolean) : [];
        lines.push(`- **${name}**${steps.length ? `: ${steps.join(' -> ')}` : ''}`);
        if (successSignals.length) lines.push(`  ${LL.t('report.context.success')}: ${successSignals.join(', ')}`);
      });
    }
    const entrypoints = Array.isArray(loadModel.entrypoints) ? loadModel.entrypoints : [];
    if (entrypoints.length) {
      lines.push('');
      lines.push(`#### ${LL.t('report.context.entrypoints')}`);
      entrypoints.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const name = safeStr(item.name || item.id);
        if (!name) return;
        const kind = safeStr(item.kind);
        const priority = safeStr(item.business_priority);
        const suffix = [kind && `${LL.t('report.context.kind')}: ${kind}`, priority && `priority: ${priority}`].filter(Boolean).join('; ');
        lines.push(`- ${name}${suffix ? ` (${suffix})` : ''}`);
      });
    }
    const hotspots = Array.isArray(loadModel.expected_hotspots) ? loadModel.expected_hotspots.map((x) => safeStr(x)).filter(Boolean) : [];
    if (hotspots.length) {
      lines.push('');
      lines.push(`- **${LL.t('report.context.hotspots')}:** ${hotspots.join(', ')}`);
    }
    const pushSimpleList = (title, items) => {
      const clean = Array.isArray(items) ? items.map((x) => safeStr(x)).filter(Boolean) : [];
      if (!clean.length) return;
      lines.push('');
      lines.push(`#### ${title}`);
      clean.forEach((item) => lines.push(`- ${item}`));
    };
    pushSimpleList(LL.t('report.context.known_constraints'), operational.known_constraints);
    pushSimpleList(LL.t('report.context.known_risks'), operational.known_risks);
    pushSimpleList(LL.t('report.context.degradation_rules'), operational.normal_degradation_rules);
    pushSimpleList(LL.t('report.context.analysis_focus'), operational.analysis_focus);
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
      `### ${LL.t('report.invalid.title')}`,
      `- ${LL.t('report.invalid.not_valid')}`,
      `- ${LL.t('report.invalid.hint')}`
    ];
    if (excerpt) {
      lines.push('');
      lines.push('```json');
      lines.push(excerpt);
      lines.push('```');
    }
    return lines.join('\n');
  }
  // ---- manual verdict ----------------------------------------------------------------------------
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
      const auto = LL.verdictLabel(row.auto_verdict || row.sla_verdict || row.llm_verdict);
      return who ? LL.t('report.verdict.source_manual_by', { who: who, auto: auto }) : LL.t('report.verdict.source_manual', { auto: auto });
    }
    const sla = safeStr(row.sla_verdict);
    const llm = safeStr(row.llm_verdict);
    if (sla) {
      // Reports built before the SLA verdict replaced the model's one keep a different model verdict.
      return llm && LL.verdictCode(llm) !== LL.verdictCode(sla)
        ? LL.t('report.verdict.source_sla_ai', { verdict: LL.verdictLabel(llm) })
        : LL.t('report.verdict.source_sla');
    }
    return LL.verdictCode(row.auto_verdict || row.verdict) === 'no_data'
      ? LL.t('report.verdict.source_no_sla_no_data', { verdict: LL.verdictLabel('no_data') })
      : LL.t('report.verdict.source_no_sla');
  }
  function verdictHistoryItem(entry) {
    const e = entry || {};
    const head = [e.changed_at ? LL.formatDateTime(e.changed_at) : '', safeStr(e.author)].filter(Boolean).join(' · ');
    const change = e.verdict
      ? `${LL.verdictLabel(e.previous_verdict)} → ${LL.verdictLabel(e.verdict)}`
      : LL.t('report.verdict.history_reset', { verdict: LL.verdictLabel(e.auto_verdict) });
    const reason = safeStr(e.reason);
    return `<li>${head ? `<span class="dim">${escapeHtml(head)}</span> ` : ''}${escapeHtml(change)}`
      + `${reason ? `<div class="verdict-history-reason">${escapeHtml(reason)}</div>` : ''}</li>`;
  }
  function verdictHistoryHtml(history) {
    const items = Array.isArray(history) ? history : [];
    if (!items.length) return '';
    return `<details class="verdict-history"><summary>${escapeHtml(LL.t('report.verdict.history_title', { count: items.length }))}</summary>`
      + `<ul>${items.map(verdictHistoryItem).join('')}</ul></details>`;
  }
  // Body of the «Изменить вердикт» dialog from GET /runs/<run>/verdict.
  function verdictDialogHtml(state) {
    const s = state || {};
    // Radio values are verdict codes (the API accepts them); the pills show the labels.
    const current = LL.verdictCode(s.verdict);
    const auto = LL.verdictLabel(s.automatic);
    const max = Number(s.reason_max) || 1000;
    const choices = (Array.isArray(s.choices) && s.choices.length ? s.choices : LL.VERDICTS).map((v) => {
      const code = LL.verdictCode(v);
      const same = code === current;
      return `<label class="verdict-choice${same ? ' is-current' : ''}"><input type="radio" name="verdictChoice" value="${escapeHtml(code)}"${same ? ' disabled' : ''}>`
        + `<span class="report-verdict-pill ${LL.verdictReportClass(code)}">${escapeHtml(LL.verdictLabel(code))}</span>${same ? `<span class="dim">${escapeHtml(LL.t('report.verdict.current_mark'))}</span>` : ''}</label>`;
    }).join('');
    const now = s.manual
      ? LL.t('report.verdict.now_manual', { verdict: LL.verdictLabel(current), auto: auto })
      : LL.t('report.verdict.now_auto', { verdict: LL.verdictLabel(current) });
    return [
      '<div class="verdict-dialog">',
      `<p class="muted">${escapeHtml(now)} ${escapeHtml(LL.t('report.verdict.dialog_note'))}</p>`,
      `<fieldset class="verdict-choices"><legend>${escapeHtml(LL.t('report.verdict.new_legend'))}</legend>${choices}</fieldset>`,
      `<label class="verdict-reason"><span>${escapeHtml(LL.t('report.verdict.reason_label'))}</span><textarea name="verdictReason" rows="3" maxlength="${max}" placeholder="${escapeHtml(LL.t('report.verdict.reason_placeholder'))}"></textarea></label>`,
      '<div class="dialog-error" data-verdict-error role="alert"></div>',
      verdictHistoryHtml(s.history),
      '</div>'
    ].join('');
  }
  function verdictFormProblem(verdict, reason, state) {
    if (!verdict) return LL.t('report.verdict.pick');
    if (!reason) return LL.t('report.verdict.reason_required');
    const max = Number(state && state.reason_max) || 1000;
    if (reason.length > max) return LL.t('report.verdict.reason_too_long', { max: max });
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
      LL.ui.toast(LL.t('report.verdict.done_published', { done: doneText, button: LL.t('report.confluence.update') }), { tone: 'info', timeout: 10000 });
    } else {
      LL.ui.toast(doneText, { tone: 'ok' });
    }
  }
  async function openVerdictDialog() {
    let state;
    try {
      state = await verdictRequest('GET');
    } catch (e) {
      LL.ui.toast(LL.t('report.verdict.load_failed', { error: e.message }), { tone: 'error' });
      return;
    }
    let busy = false;
    await LL.ui.dialog({
      title: LL.t('report.verdict.change'),
      bodyHtml: verdictDialogHtml(state),
      // A message about a missing choice or reason goes away as soon as the form changes.
      onOpen: (body) => {
        const error = body.querySelector('[data-verdict-error]');
        body.addEventListener('input', () => { error.textContent = ''; });
        body.addEventListener('change', () => { error.textContent = ''; });
      },
      actions: [
        { label: LL.t('common.action.cancel') },
        {
          label: LL.t('common.action.save'),
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
              applyVerdictState(next, LL.t('report.verdict.changed'));
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
      title: LL.t('report.verdict.reset_title'),
      message: LL.t('report.verdict.reset_message', { verdict: LL.verdictLabel(row.auto_verdict) }),
      confirmText: LL.t('report.verdict.reset_confirm')
    });
    if (!ok) return;
    try {
      applyVerdictState(await verdictRequest('DELETE', {}), LL.t('report.verdict.reset_done'));
    } catch (e) {
      LL.ui.toast(LL.t('report.verdict.reset_failed', { error: e.message }), { tone: 'error' });
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
  // «Проблемы и рекомендации»: two columns without boxes, a problem and its recommendation side by side on one
  // grid row (one column on narrow screens, where each item names its column).
  function renderProblemRecommendationRows(rows) {
    const empty = `<span class="report-empty">${escapeHtml('—')}</span>`;
    const problems = escapeHtml(LL.t('report.findings.problems'));
    const recommendations = escapeHtml(LL.t('report.findings.recommendations'));
    const body = rows.map((row) => [
      `<div class="rep-f rep-f-problem" data-col="${problems}">${row.problemHtml || empty}</div>`,
      `<div class="rep-f rep-f-action" data-col="${recommendations}">${row.recommendationHtml || empty}</div>`
    ].join('')).join('');
    return [
      '<section class="rep-findings">',
      `<div class="report-block-title">${escapeHtml(LL.t('report.findings.title'))}</div>`,
      '<div class="rep-findings-grid">',
      `<div class="rep-findings-col lbl">${problems}</div>`,
      `<div class="rep-findings-col lbl">${recommendations}</div>`,
      body,
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
    const contentHtml = markdownToSafeHtml(text) || `<p class="report-empty">${escapeHtml(LL.t('report.rationale.empty'))}</p>`;
    return [
      '<section class="report-note">',
      `<div class="report-note-title">${escapeHtml(LL.t('report.rationale.title'))}</div>`,
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
          if (evidenceItem.observedValue) bits.push(LL.t('report.evidence.value', { value: evidenceItem.observedValue }));
          if (evidenceItem.threshold) bits.push(LL.t('report.evidence.threshold', { value: evidenceItem.threshold }));
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
      { label: LL.t('report.finding.severity'), value: safeStr(item.severity) },
      { label: LL.t('report.finding.component'), value: safeStr(item.component) }
    ]);
    const evidenceHtml = renderFindingEvidence(item);
    const findingId = normalizeFindingLinkId(item.id || item.finding_id || item.key, '');
    const voteHtml = (domain && findingId) ? renderVoteGroup(domain, 'finding', findingId, LL.t('report.finding.agree'), LL.t('report.finding.disagree')) : '';
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
        recommendationHtml: renderActionEntries(recommendationTexts, LL.t('report.findings.no_recommendation'))
      };
    });
    actionEntries
      .filter((action) => (
        (!action.forFindingIds.length && unusedLegacyActions.has(action.idx))
        || (action.forFindingIds.length && !action.forFindingIds.some((id) => linkedFindingIds.has(id)))
      ))
      .forEach((action) => {
        rows.push({
          problemHtml: `<span class="report-empty">${escapeHtml(LL.t('report.findings.extra_recommendation'))}</span>`,
          recommendationHtml: renderActionEntries([action.item], LL.t('report.findings.no_recommendation'))
        });
      });
    if (!rows.length) {
      rows.push({
        problemHtml: `<span class="report-empty">${escapeHtml(LL.t('report.findings.no_problems'))}</span>`,
        recommendationHtml: renderActionEntries(actionEntries.map((action) => action.item), LL.t('report.findings.no_recommendations'))
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
      `<span>${escapeHtml(LL.t('report.context.title'))}</span>`,
      `<small class="report-collapsible-subtitle">${escapeHtml(LL.t('report.context.subtitle'))}</small>`,
      '</summary>',
      `<div class="report-collapsible-body report-markdown-block">${html}</div>`,
      '</details>'
    ].join('');
    box.style.display = '';
  }
  function renderAnalysisError(report) {
    const error = report && report.analysis_error;
    if (!error || typeof error !== 'object') return '';
    const message = safeStr(error.message) || LL.t('report.analysis_error.default_message');
    const bits = [];
    if (error.provider) bits.push(String(error.provider));
    if (error.max_tokens) bits.push(LL.t('report.analysis_error.limit', { value: error.max_tokens }));
    if (error.output_tokens != null && error.output_tokens !== '') bits.push(LL.t('report.analysis_error.output', { value: error.output_tokens }));
    const meta = bits.length ? `<p class="dim">${escapeHtml(bits.join(' · '))}</p>` : '';
    const excerpt = error.excerpt ? `<pre>${escapeHtml(String(error.excerpt))}</pre>` : '';
    return `<div class="report-note"><div class="report-note-title">${escapeHtml(LL.t('report.analysis_error.title'))}</div><div class="report-note-body"><p>${escapeHtml(message)}</p>${meta}${excerpt}</div></div>`;
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
  const TEST_TYPES = ['step', 'soak', 'spike', 'stress'];
  const METRIC_KINDS = ['rps', 'p95', 'error_rate', 'cpu', 'memory'];
  const RESOURCE_KINDS = ['cpu', 'memory'];
  // Name of a metric kind (trends, limiting resource) in the UI language; '' for an unknown kind.
  function metricTitle(kind) {
    const key = safeStr(kind);
    return METRIC_KINDS.indexOf(key) >= 0 ? LL.t(`report.metric.${key}`) : '';
  }
  function resourceTitle(kind) {
    return RESOURCE_KINDS.indexOf(safeStr(kind)) >= 0 ? metricTitle(kind) : '';
  }
  // Codes stored by the analysis and the dictionary keys of their texts (report.<group>.<code>); the Python
  // tables of AI/test_profile.py, AI/stability.py, AI/spike.py and AI/saturation.py hold the same texts
  // (tests/test_verdicts.py). Reports saved before the codes carry only the text, which is shown as it is.
  const STORED_CODES = {
    typeFocus: ['report.focus.', TEST_TYPES],
    holdNotes: ['report.hold_note.', ['no_steady_level', 'early_drop', 'long_ramp_up']],
    spikeNotes: ['report.spike_note.', ['no_latency_after', 'no_latency_query']],
    saturationNotes: ['report.saturation_note.', ['limit_not_reached', 'no_sla_step']],
    failureModes: ['report.failure_mode.', ['latency+errors', 'latency', 'errors', 'throughput_drop', 'none']]
  };
  // Key segment of a code: 'latency+errors' -> 'latency_errors'.
  function codeKey(code) {
    return String(code).replace(/[^a-z0-9_]+/g, '_');
  }
  // Text of a stored code in the UI language; '' for an unknown code.
  function storedText(group, code) {
    const spec = STORED_CODES[group];
    const value = safeStr(code);
    return spec && value && spec[1].indexOf(value) >= 0 ? LL.t(spec[0] + codeKey(value)) : '';
  }
  // {group: {code: text}} of every stored code (tests compare it with the Python tables).
  function storedTexts() {
    const out = {};
    Object.keys(STORED_CODES).forEach((group) => {
      out[group] = {};
      STORED_CODES[group][1].forEach((code) => { out[group][code] = storedText(group, code); });
    });
    return out;
  }
  // Note of a payload: rendered from note_codes / note_code, else the saved text.
  function storedNote(item, group) {
    const source = item || {};
    const codes = Array.isArray(source.note_codes) ? source.note_codes : (safeStr(source.note_code) ? [source.note_code] : []);
    const rendered = codes.map((code) => storedText(group, code) || safeStr(code)).filter(Boolean).join('; ');
    return rendered || safeStr(source.note);
  }
  function failureModeTitle(sat) {
    return storedText('failureModes', (sat || {}).failure_mode) || safeStr((sat || {}).failure_mode_title);
  }
  // Name of the test type by normalized_test_type; old reports without a known type keep their saved label.
  function storedTypeLabel(profile) {
    const key = safeStr(profile.normalized_test_type).toLowerCase();
    return TEST_TYPES.indexOf(key) >= 0 ? LL.t(`common.test_type.${key}`) : safeStr(profile.label_ru || profile.label);
  }
  function stabilityRows(report) {
    const st = report.stability_under_load || {};
    const profile = report.test_profile || {};
    const rows = [
      { label: LL.t('report.section.test_type'), value: safeStr(storedTypeLabel(profile) || profile.test_type) || 'stability/soak' },
      { label: LL.t('report.section.focus'), value: storedText('typeFocus', safeStr(profile.normalized_test_type).toLowerCase()) || safeStr(st.focus || profile.focus) || LL.t('report.section.focus_default_soak') },
      { label: LL.t('report.section.target_rps'), value: formatNumber(st.target_rps) },
      { label: LL.t('report.section.actual_rps'), value: formatNumber(st.actual_rps) }
    ];
    const hold = (st.hold && typeof st.hold === 'object') ? st.hold : null;
    if (hold) {
      const span = hhmm(hold.start_iso) + '–' + hhmm(hold.end_iso);
      rows.push({ label: LL.t('report.section.hold_window'), value: hold.duration_min != null ? LL.t('report.section.hold_window_minutes', { span: span, minutes: formatNumber(hold.duration_min) }) : span });
      if (hold.min_bucket_rps != null) {
        const bucket = { label: safeStr(hold.min_bucket_label) || LL.t('report.section.interval'), rps: formatNumber(hold.min_bucket_rps), drop: formatNumber(hold.drop_pct) };
        rows.push({ label: LL.t('report.section.min_interval'), value: hold.drop_pct ? LL.t('report.section.min_interval_drop', bucket) : `${bucket.label}: ${bucket.rps} RPS` });
      }
      if (hold.early_drop_iso) rows.push({ label: LL.t('report.section.early_drop'), value: hhmm(hold.early_drop_iso) });
      const holdNote = storedNote(hold, 'holdNotes');
      if (holdNote) rows.push({ label: LL.t('report.section.note'), value: holdNote });
    }
    rows.push({ label: LL.t('report.section.sla_summary'), value: safeStr(st.sla_summary) || '—' });
    return rows;
  }
  function spikeRows(report) {
    const sa = report.spike_analysis || {};
    const profile = report.test_profile || {};
    const rows = [{ label: LL.t('report.section.test_type'), value: safeStr(storedTypeLabel(profile) || profile.test_type) || 'spike' }];
    if (sa.baseline_rps != null) rows.push({ label: LL.t('report.section.baseline'), value: formatNumber(sa.baseline_rps) + ' RPS' });
    if (!sa.detected) {
      rows.push({ label: LL.t('report.section.spike'), value: safeStr(sa.reason) || LL.t('report.section.spike_not_detected') });
    } else {
      if (sa.target_rps != null) {
        const rps = formatNumber(sa.target_rps);
        const target = sa.target_reached === true
          ? LL.t('report.section.target_reached', { rps: rps })
          : (sa.target_reached === false ? LL.t('report.section.target_missed', { rps: rps }) : rps);
        rows.push({ label: LL.t('report.section.spike_target_rps'), value: target });
      }
      (Array.isArray(sa.spikes) ? sa.spikes : []).forEach((spike) => {
        const pre = spike.pre || {};
        const during = spike.during || {};
        const bits = [LL.t('report.section.spike_window', { span: `${hhmm(spike.start_iso)}–${hhmm(spike.end_iso)}`, seconds: formatNumber(spike.duration_sec) })];
        if (pre.p95 != null && during.p95 != null) {
          const over = signed(spike.overshoot_p95_pct, ' %');
          bits.push(`p95 ${formatNumber(pre.p95)} → ${formatNumber(during.p95)}` + (over ? ` (${over})` : ''));
        }
        const errors = signed(spike.overshoot_error_rate_pp, ' ' + LL.t('report.unit.pp'));
        if (errors) bits.push(LL.t('report.section.spike_errors', { value: errors }));
        if (spike.recovered === true) bits.push(LL.t('report.section.spike_recovered', { seconds: formatNumber(spike.recovery_time_sec) }));
        else if (spike.recovered === false) bits.push(LL.t('report.section.spike_not_recovered'));
        const spikeNote = storedNote(spike, 'spikeNotes');
        if (spikeNote) bits.push(spikeNote);
        rows.push({ label: safeStr(spike.label) || LL.t('report.section.spike_fallback'), value: bits.join(', ') });
      });
    }
    rows.push({ label: LL.t('report.section.sla_summary'), value: safeStr(sa.sla_summary) || '—' });
    return rows;
  }
  function resourceText(resource) {
    const title = resourceTitle(resource.kind) || safeStr(resource.kind);
    if (resource.evidence === 'threshold') {
      // The comparison sign is a parameter: dictionary values carry no < or >.
      const params = { title: title, value: formatNumber(resource.value), op: '>', threshold: formatNumber(resource.threshold), series: safeStr(resource.series) };
      return params.series ? LL.t('report.resource.threshold_series', params) : LL.t('report.resource.threshold', params);
    }
    return LL.t('report.resource.growth', { title: title, growth: signed(resource.growth_pct, ' %'), value: formatNumber(resource.value) });
  }
  function saturationRows(report) {
    const sat = report.saturation || {};
    const peak = report.peak_performance || report.peak_perfomance || {};
    const rows = [{ label: LL.t('report.section.saturation_point'), value: formatNumber(sat.saturation_rps != null ? sat.saturation_rps : peak.max_rps) + ' RPS' }];
    if (safeStr(sat.saturation_step_label)) rows.push({ label: LL.t('report.section.saturation_step'), value: safeStr(sat.saturation_step_label) });
    if (safeStr(peak.max_time)) rows.push({ label: LL.t('report.section.peak_time'), value: safeStr(peak.max_time) });
    if (safeStr(sat.degraded_step_label) || sat.first_degraded_rps != null) {
      rows.push({ label: LL.t('report.section.first_degraded_step'), value: safeStr(sat.degraded_step_label) || `≈${formatNumber(sat.first_degraded_rps)} RPS` });
    }
    const checks = Array.isArray(sat.degraded_checks) ? sat.degraded_checks : [];
    if (checks.length) rows.push({ label: LL.t('report.section.broken_checks'), value: checks.map((name) => slaCheckLabel(name) || name).join(', ') });
    rows.push({ label: LL.t('report.section.failure_mode'), value: safeStr(failureModeTitle(sat) || sat.failure_mode) || '—' });
    rows.push({ label: LL.t('report.section.limiting_resource'), value: (sat.limiting_resource && typeof sat.limiting_resource === 'object') ? resourceText(sat.limiting_resource) : LL.t('report.section.not_determined') });
    if (safeStr(peak.drop_time || sat.rps_drop_iso)) rows.push({ label: LL.t('report.section.drop_time'), value: safeStr(peak.drop_time || sat.rps_drop_iso) });
    const saturationNote = storedNote(sat, 'saturationNotes');
    if (saturationNote) rows.push({ label: LL.t('report.section.note'), value: saturationNote });
    return rows;
  }
  function testTypeSection(report) {
    const r = report || {};
    const mode = reportMode(r);
    if (mode === 'stability') return { title: LL.t('report.section.title_stability'), rows: stabilityRows(r) };
    if (mode === 'spike') return { title: LL.t('report.section.title_spike'), rows: spikeRows(r) };
    if (mode === 'stress' && r.saturation && typeof r.saturation === 'object') return { title: LL.t('report.section.title_saturation'), rows: saturationRows(r) };
    return null;
  }
  function renderTrendsTable(report) {
    const st = (report || {}).stability_under_load || {};
    const trends = Array.isArray(st.trends) ? st.trends : [];
    if (!trends.length) return '';
    const rows = trends.map((trend) => [
      `<tr class="${trend.flagged ? 'is-unstable' : ''}">`,
      `<td>${escapeHtml(metricTitle(trend.metric_kind) || safeStr(trend.title) || safeStr(trend.metric_kind))}${trend.flagged ? `<span class="step-flag">${escapeHtml(LL.t('report.trends.risk_flag'))}</span>` : ''}<div class="sla-check-note">${escapeHtml(safeStr(trend.label))}</div></td>`,
      `<td class="num">${escapeHtml(formatNumber(trend.first))}</td>`,
      `<td class="num">${escapeHtml(formatNumber(trend.last))}</td>`,
      `<td class="num">${escapeHtml(signed(trend.change_pct, ' %') || '—')}</td>`,
      `<td class="num">${escapeHtml(formatNumber(trend.slope_per_hour))}</td>`,
      '</tr>'
    ].join('')).join('');
    const note = st.trend_threshold_pct != null
      ? LL.t('report.trends.note_threshold', { pct: formatNumber(st.trend_threshold_pct) })
      : LL.t('report.trends.note');
    const th = (key, cls) => `<th${cls ? ` class="${cls}"` : ''}>${escapeHtml(LL.t(key))}</th>`;
    return [
      '<section>',
      `<div class="report-block-title">${LL.ui.term('trend_drift', LL.t('report.trends.title'))}</div>`,
      '<div class="step-table-wrap"><table class="step-table">',
      `<thead><tr>${th('report.trends.col_metric')}${th('report.trends.col_first', 'num')}${th('report.trends.col_last', 'num')}${th('report.trends.col_change', 'num')}${th('report.trends.col_slope', 'num')}</tr></thead>`,
      `<tbody>${rows}</tbody></table></div>`,
      `<p class="step-note">${escapeHtml(note)}</p>`,
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
    // A verdict code from the API (old labels map as well, see LL.verdictCode).
    const verdict = (report || {}).verdict || '';
    const verdictRationale = safeStr((report || {}).verdict_rationale || (report || {}).verdict_reason || (report || {}).rationale)
      .replace(/^([^\n]+)\n(?=- )/, '$1\n\n');
    const peak = ((report || {}).peak_performance || (report || {}).peak_perfomance || {});
    const findings = (report || {}).findings || [];
    const actions = (report || {}).recommended_actions || (report || {}).actions || [];
    const verdictRows = [
      {
        label: LL.t('report.structured.ai_verdict'),
        html: `<span class="report-verdict-pill ${LL.verdictReportClass(verdict)}">${escapeHtml(LL.verdictLabel(verdict))}</span>`
      }
    ];
    verdictRows._tableClass = 'report-aligned-value-table';
    const peakRows = [
      { label: LL.t('report.structured.max_rps'), html: escapeHtml(safeStr(peak.max_rps) || '—') },
      { label: LL.t('report.section.peak_time'), html: escapeHtml(safeStr(peak.max_time) || '—') },
      { label: LL.t('report.section.drop_time'), html: escapeHtml(safeStr(peak.drop_time) || '—') }
    ];
    peakRows._tableClass = 'report-aligned-value-table';
    const typeSection = renderTestTypeSection(report, domain);
    const sections = [
      renderKeyValueTable('', verdictRows),
      renderRationaleBox(verdictRationale),
      typeSection || (peakTableApplies(domain, peak) ? renderKeyValueTable(LL.t('report.structured.peak_title'), peakRows) : ''),
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
      if (meta.reason === 'partial_scores') reason = LL.t('report.judge.reason_partial');
      else if (meta.reason === 'error' || meta.error) reason = LL.t('report.judge.reason_error');
      else reason = LL.t('report.judge.reason_unparsed');
    }
    return { skipped, unavailable, reason, final };
  }
  // Words of a 0…1 score as the report shows them; '' for no score.
  function scoreLevel(x) {
    const v = Number(x);
    if (!Number.isFinite(v) || v <= 0) return '';
    if (v >= 0.9) return LL.t('report.judge.level.very_high');
    if (v >= 0.75) return LL.t('report.judge.level.high');
    if (v >= 0.55) return LL.t('report.judge.level.moderate');
    return LL.t('report.judge.level.low');
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
      const level = (x) => scoreLevel(x) || LL.t('report.judge.level.none');
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
            ? LL.t('report.judge.intro_skipped')
            : LL.t('report.judge.intro_unavailable', { reason: scoreState.reason });
          if (data === null) return intro;
          // Same thresholds as the level in the section title.
          if (data >= 0.75) return `${intro} ${LL.t('report.judge.data_good')}`;
          if (data >= 0.55) return `${intro} ${LL.t('report.judge.data_fair')}`;
          return `${intro} ${LL.t('report.judge.data_weak')}`;
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
              return LL.t('report.judge.summary.labels_weak_peak');
            }
            return LL.t('report.judge.summary.labels_weak');
          }
          if (numericClaimsTotal > 0 && numericGrounding !== null && numericGrounding < 0.35) {
            return LL.t('report.judge.summary.numbers_weak');
          }
          if (
            labelGrounding !== null
            && labelGrounding < 0.35
            && (numericClaimsTotal === 0 || numericGrounding === null || numericGrounding < 0.55)
          ) {
            return LL.t('report.judge.summary.binding_weak');
          }
          if (peakChecked && peakConsistency !== null && peakConsistency < 0.2 && data >= 0.4) {
            return LL.t('report.judge.summary.peak_mismatch');
          }
          if (judge >= 0.8 && data >= 0.7) return LL.t('report.judge.summary.reliable');
          if (judge >= 0.8 && data < 0.5) return LL.t('report.judge.summary.data_partial');
          if (judge < 0.6 && data >= 0.7) return LL.t('report.judge.summary.data_better');
          if (judge < 0.6 && data < 0.6) return LL.t('report.judge.summary.both_reserved');
          if (Math.abs(judge - data) < 0.12) return LL.t('report.judge.summary.close');
        }
        if (judge !== null && final !== null && final + 0.15 < judge) {
          return LL.t('report.judge.summary.final_lower');
        }
        return LL.t('report.judge.summary.service');
      })();
      const lines = [];
      if (summary) lines.push(summary);
      lines.push('');
      const item = (text) => lines.push(`- ${text}`);
      const heading = (text) => lines.push(`#### ${text}`);
      heading(LL.t('report.judge.main'));
      if (judgeSkipped) {
        item(LL.t('report.judge.judge_skipped'));
        item(LL.t('report.judge.data_check', { pct: pct(dataScore), level: level(dataScore) }));
        if (typeof conf === 'number') item(LL.t('report.judge.confidence', { pct: pct(conf), level: level(conf) }));
      } else if (judgeUnavailable) {
        item(LL.t('report.judge.judge_unavailable'));
        item(LL.t('report.judge.data_check', { pct: pct(dataScore), level: level(dataScore) }));
        item(LL.t('report.judge.final_without_judge', { pct: pct(finalScore), level: level(finalScore) }));
        if (typeof conf === 'number') item(LL.t('report.judge.confidence', { pct: pct(conf), level: level(conf) }));
      } else {
        item(LL.t('report.judge.judge_score', { pct: pct(overall), level: level(overall) }));
        item(LL.t('report.judge.data_check', { pct: pct(dataScore), level: level(dataScore) }));
        item(LL.t('report.judge.final_score', { pct: pct(finalScore), level: level(finalScore) }));
        if (typeof conf === 'number') item(LL.t('report.judge.confidence', { pct: pct(conf), level: level(conf) }));
        lines.push('');
        heading(LL.t('report.judge.criteria_title'));
        item(LL.t('report.judge.factual', { pct: pct(factual) }));
        item(LL.t('report.judge.completeness', { pct: pct(completeness) }));
        item(LL.t('report.judge.specificity', { pct: pct(specificity) }));
      }
      if (!judgeSkipped && !judgeUnavailable && Object.keys(rubric).length) {
        lines.push('');
        heading(LL.t('report.judge.rubric_title'));
        item(LL.t('report.judge.rubric.evidence_grounding', { pct: pct(rubric.evidence_grounding) }));
        item(LL.t('report.judge.rubric.issue_coverage', { pct: pct(rubric.issue_coverage) }));
        item(LL.t('report.judge.rubric.specificity', { pct: pct(rubric.specificity) }));
        item(LL.t('report.judge.rubric.sla_alignment', { pct: pct(rubric.sla_alignment) }));
        item(LL.t('report.judge.rubric.actionability', { pct: pct(rubric.actionability) }));
      }
      if (Object.keys(dataDetails).length) {
        lines.push('');
        heading(LL.t('report.judge.data_title'));
        item(LL.t('report.judge.data.label_grounding', { pct: pct(dataDetails.label_grounding) }));
        if (Number(dataDetails.numeric_claims_total || 0) > 0) {
          item(LL.t('report.judge.data.numeric', {
            pct: pct(dataDetails.numeric_grounding),
            supported: Number(dataDetails.numeric_claims_supported || 0),
            total: Number(dataDetails.numeric_claims_total || 0)
          }));
        } else {
          item(LL.t('report.judge.data.numeric_none'));
        }
        if (dataDetails.peak_checked) {
          item(LL.t('report.judge.data.peak', { pct: pct(dataDetails.peak_consistency) }));
        } else {
          item(LL.t('report.judge.data.peak_none'));
        }
        item(LL.t('report.judge.data.recommendation_grounding', { pct: pct(dataDetails.recommendation_grounding) }));
      }
      if (meta.used_safe_path || meta.context_truncated || Number(meta.truncated_candidates || 0) > 0) {
        lines.push('');
        heading(LL.t('report.judge.notes_title'));
        if (meta.used_safe_path) item(LL.t('report.judge.notes.safe_path'));
        if (meta.context_truncated) item(LL.t('report.judge.notes.context_truncated'));
        if (Number(meta.truncated_candidates || 0) > 0) item(LL.t('report.judge.notes.truncated_candidates', { count: Number(meta.truncated_candidates || 0) }));
      }
      lines.push('');
      lines.push(`_${LL.t('report.judge.footer')}_`);
      return lines.join('\n');
    } catch (e) { return ''; }
  }
  function confidenceLevelLabel(scores) {
    const s = scores || {};
    const judge = s.judge || {};
    const state = answerScoreState(s);
    const raw = state.final !== null ? state.final : Number(judge.overall);
    return scoreLevel(raw);
  }
  function answerSubtitle(scores) {
    const state = answerScoreState(scores);
    if (state.skipped) return LL.t('report.judge.subtitle_skipped');
    if (state.unavailable) return LL.t('report.judge.subtitle_unavailable');
    return LL.t('report.judge.subtitle_default');
  }
  function judgeSectionHtml(scores) {
    const md = judgeDetailsMarkdown(scores);
    if (!safeStr(md)) return '';
    const level = confidenceLevelLabel(scores);
    const titleHtml = level
      ? `${LL.ui.term('confidence', LL.t('report.judge.title_reliability'))}: ${escapeHtml(level)}`
      : LL.ui.term('confidence', LL.t('report.judge.title_score'));
    return [
      '<details class="report-collapsible">',
      `<summary><span>${titleHtml}</span><small class="report-collapsible-subtitle">${escapeHtml(answerSubtitle(scores))}</small></summary>`,
      `<div class="report-collapsible-body report-markdown-block">${markdownToSafeHtml(md)}</div>`,
      '</details>'
    ].join('');
  }

  const SLA_CHECK_NAMES = ['target_rps', 'error_rate', 'p95_latency', 'p99_latency', 'cpu_usage', 'memory_usage', 'recovery'];
  // Visible name of an SLA check (criterion and unit) in the UI language; '' for an unknown name.
  function slaCheckLabel(name) {
    const key = safeStr(name);
    return SLA_CHECK_NAMES.indexOf(key) >= 0 ? LL.t(`report.sla_check.${key}`) : '';
  }
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
  // 3px bar "actual against the threshold" (the .fig-bar of the figures): fill in the check tone, tick at the threshold.
  function slaBarHtml(check) {
    const bar = slaBarModel(check);
    if (!bar) return '';
    const title = bar.higherBetter ? LL.t('report.sla.bar_title_higher', { percent: bar.percent }) : LL.t('report.sla.bar_title', { percent: bar.percent });
    const tone = bar.tone === 'ok' ? '' : ` class="${bar.tone}"`;
    return `<div class="fig-bar${bar.higherBetter ? ' is-higher-better' : ''}" title="${escapeHtml(title)}"><i${tone} style="width:${bar.fill}%"></i><b style="left:${bar.mark}%"></b></div>`;
  }
  // Comparison sign of a threshold: the target RPS is a floor, every other check a ceiling.
  function thresholdSign(name) {
    return safeStr(name) === 'target_rps' ? '≥' : '≤';
  }
  // Query and series a check was computed on («LT: RPS sum by all groups · all»), '' when the report does not say.
  function slaSourceText(check, window, table) {
    const where = checkSource(check, window, table);
    return [where.label, where.series].map(safeStr).filter(Boolean).join(' · ');
  }
  // Where the criteria were checked, for the caption of the SLA table: «ступень 5 · 15:54–15:58», the hold window
  // or the spike windows (clock of the report, as the step intervals); '' when the report does not say.
  function slaScopeText(finalRow) {
    if (!finalRow) return '';
    const parsed = parseJsonish(finalRow.parsed) || {};
    const window = markerSlaWindow(finalRow);
    const mode = markerMode(parsed, window);
    if (mode === 'spike') return LL.t('report.sla.scope_spike');
    const table = embeddedLoadStepTable(finalRow);
    const steps = table ? table.steps.filter(isPlainObject) : [];
    const offset = reportOffsetHours(steps);
    const startMs = utcMoment(window.start || window.start_iso);
    const endMs = utcMoment(window.end || window.end_iso);
    const span = startMs !== null && endMs !== null ? clockAt(startMs, offset) + '–' + clockAt(endMs, offset) : '';
    if (mode === 'hold') return span ? LL.t('report.sla.scope_hold', { span: span }) : LL.t('report.sla.scope_hold_only');
    const chosen = table ? chosenSlaStep(linkSlaToSteps(table, finalRow).steps, window, markerChecks(finalRow)) : null;
    const index = chosen ? safeStr(chosen.index) : '';
    if (index) return span ? LL.t('report.sla.scope_step', { index: index, span: span }) : LL.t('report.sla.scope_step_only', { index: index });
    return span ? LL.t('report.sla.scope_window', { span: span }) : '';
  }
  // SLA criteria without a box: ✓/✗ glyph, criterion with its query and series under it, threshold with the sign,
  // actual and the bar; the summary stored with the report under the table.
  function renderSlaChecklist(finalRow) {
    const slaDetails = parseJsonish(finalRow.sla_details) || {};
    const checks = Array.isArray(slaDetails.checks) ? slaDetails.checks : [];
    if (!checks.length) {
      const link = LL.html(`<a href="/settings">${escapeHtml(LL.t('report.sla.empty_link'))}</a>`);
      return [
        `<div class="lbl rep-sla-cap">${escapeHtml(LL.t('report.sla.title'))}</div>`,
        `<div class="hero-empty">${LL.tHtml('report.sla.empty', { link: link })}</div>`
      ].join('');
    }
    const window = markerSlaWindow(finalRow);
    const source = markerTable(finalRow, null);
    const rows = checks.map((check) => {
      const name = safeStr(check && check.name);
      const plainLabel = slaCheckLabel(name) || name || '—';
      const termKey = name === 'target_rps' ? 'target_rps' : (name === 'error_rate' ? 'error_rate' : (name.startsWith('p95') ? 'p95' : ''));
      const label = termKey ? LL.ui.term(termKey, plainLabel) : escapeHtml(plainLabel);
      const passed = check ? check.passed : null;
      const statusClass = passed === true ? 'ok' : (passed === false ? 'fail' : 'na');
      const statusMark = passed === true ? '✓' : (passed === false ? '✗' : '—');
      const statusTitle = passed === true ? LL.t('report.sla.status_ok') : (passed === false ? LL.t('report.sla.status_fail') : LL.t('report.sla.status_na'));
      // The stored message (report language) stays as the tooltip when the query names the source.
      const note = safeStr(check && check.message);
      const where = isPlainObject(check) ? slaSourceText(check, window, source) : '';
      const sub = where || note;
      const threshold = formatNumber(check && check.threshold);
      return [
        `<tr class="is-${statusClass}">`,
        `<td class="g"><span class="rep-glyph ${statusClass}" role="img" title="${escapeHtml(statusTitle)}" aria-label="${escapeHtml(statusTitle)}">${statusMark}</span></td>`,
        `<td class="name"${where && note ? ` title="${escapeHtml(note)}"` : ''}>${label}${sub ? `<span class="sub m">${escapeHtml(sub)}</span>` : ''}</td>`,
        `<td class="n th">${escapeHtml(threshold === '—' ? threshold : `${thresholdSign(name)} ${threshold}`)}</td>`,
        `<td class="n act">${escapeHtml(formatNumber(check && check.actual))}</td>`,
        `<td class="n bar">${slaBarHtml(check)}</td>`,
        '</tr>'
      ].join('');
    }).join('');
    const summary = safeStr(slaDetails.summary);
    const scope = slaScopeText(finalRow);
    const stepNote = slaStepNote(finalRow);
    const th = (key, cls) => `<th${cls ? ` class="${cls}"` : ''}><span class="visually-hidden">${escapeHtml(LL.t(key))}</span></th>`;
    return [
      `<div class="lbl rep-sla-cap"${stepNote ? ` title="${escapeHtml(stepNote)}"` : ''}>${escapeHtml(LL.t('report.sla.title'))}${scope ? ` · ${escapeHtml(scope)}` : ''}</div>`,
      '<table class="rep-sla-table">',
      `<thead><tr>${th('report.sla.col_status')}${th('report.sla.col_criterion')}${th('report.sla.col_threshold', 'n')}${th('report.sla.col_actual', 'n')}${th('report.sla.col_bar', 'n')}</tr></thead>`,
      `<tbody>${rows}</tbody></table>`,
      summary ? `<div class="hero-sub rep-sla-note">${escapeHtml(summary)}</div>` : ''
    ].join('');
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
    const hint = messageHint(safeStr(check && check.message));
    return {
      label: safeStr(check && check.query_label) || hint.label,
      series: safeStr(check && check.series) || hint.series,
      actual: check ? check.actual : null
    };
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
  // «2026-10-21 14:05 UTC+3»: LL.formatDateTime in the clock of the report instead of the browser's.
  function dateTimeAt(ms, offsetHours) {
    const shifted = new Date(ms + offsetHours * 3600000);
    const pad = (n) => String(n).padStart(2, '0');
    const day = `${shifted.getUTCFullYear()}-${pad(shifted.getUTCMonth() + 1)}-${pad(shifted.getUTCDate())}`;
    return `${day} ${clockAt(ms, offsetHours)} ${utcOffsetText(offsetHours)}`;
  }
  // Offset of the report clock: the step row, the markers and the chart axes all read it.
  function reportClockOffset() {
    const table = reportState.stepTable;
    return reportOffsetHours(table && Array.isArray(table.steps) ? table.steps.filter(isPlainObject) : []);
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
    if (passed === true) return LL.t('report.sla.word_pass');
    if (passed === false) return LL.t('report.sla.word_fail');
    return LL.t('report.sla.word_na');
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
    const params = { index: chosen.index || '', interval: stepInterval(chosen), window: windowText, level: levelText };
    let head = LL.t('report.steps.sla_step', params);
    if (windowText) head = levelText ? LL.t('report.steps.sla_step_window_level', params) : LL.t('report.steps.sla_step_window', params);
    const meanBit = differs ? '. ' + LL.t('report.steps.sla_mean_note') : '';
    return head + meanBit + '. ' + slaChecksText(checks) + '.';
  }
  function slaChecksText(checks) {
    return checks.map((check) => {
      const label = slaCheckLabel(check && check.name) || safeStr(check && check.name) || LL.t('report.steps.check_fallback');
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
      const span = startMs != null && endMs != null ? clockAt(startMs, offset) + '–' + clockAt(endMs, offset) : '';
      const level = Number(window.level);
      const params = { span: span, level: Number.isFinite(level) ? formatNumber(level) : '' };
      let head;
      if (span) head = Number.isFinite(level) ? LL.t('report.steps.sla_hold_span_level', params) : LL.t('report.steps.sla_hold_span', params);
      else head = Number.isFinite(level) ? LL.t('report.steps.sla_hold_level', params) : LL.t('report.steps.sla_hold');
      return { steps: steps, slaNote: checks.length ? head + '. ' + slaChecksText(checks) + '.' : head + '.' };
    }
    if (kind === 'spike') {
      const moment = momentMs(window.start || window.start_iso);
      steps.forEach((step) => { step.selected = false; });
      const covered = steps.find((step) => stepCovers(step, moment));
      if (covered) covered.selected = true;
      const head = LL.t('report.steps.sla_spike');
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
  // The full «SLA проверено на ступени …» sentence (plain text), shown as the tooltip of the SLA caption.
  function slaStepNote(finalRow) {
    const table = embeddedLoadStepTable(finalRow);
    if (!table) return '';
    return safeStr(linkSlaToSteps(table, finalRow).slaNote);
  }
  // One cell of the step row: «5 · 250 RPS» (SLA tag on the step the criteria were checked on) over «15:54–15:58 ·
  // p95 462.67 мс»; a step after a drop is named so and drawn in the fail tone. The cell grows with the step length,
  // so it sits under its step of the profile; hovering or focusing it highlights that interval on the charts.
  function stepCellHtml(step) {
    const classes = ['rep-step'];
    if (step.selected) classes.push('is-sla');
    if (step.after_drop === true) classes.push('is-bad');
    else if (step.stable === false) classes.push('is-unstable');
    const rps = formatNumber(step.rps);
    const rpsText = rps === '—' ? '' : `${rps} RPS`;
    const head = step.after_drop === true
      ? [LL.t('report.steps.flag_after_drop'), rpsText].filter(Boolean).join(' · ')
      : [safeStr(step.index), rpsText].filter(Boolean).join(' · ');
    const p95 = finiteOrNull(step.latency_p95);
    const sub = [stepInterval(step), p95 === null ? '' : LL.t('report.steps.cell_p95', { value: formatNumber(p95) })].filter(Boolean).join(' · ');
    let flag = '';
    if (step.after_drop !== true && step.dip === true) flag = LL.t('report.steps.flag_dip');
    else if (step.after_drop !== true && step.stable === false) flag = LL.t('report.steps.flag_unstable');
    // The interval shown in the cell (plateau when known) is what the charts highlight.
    const range = stepRangeMs(step);
    const rangeAttrs = range[0] !== null && range[1] !== null ? ` data-step-from="${range[0]}" data-step-to="${range[1]}" tabindex="0"` : '';
    const indexAttr = step.index === null || step.index === undefined || step.index === '' ? '' : ` data-step-index="${escapeHtml(String(step.index))}"`;
    const start = utcMoment(step.start_iso);
    const end = utcMoment(step.end_iso);
    const minutes = start !== null && end !== null && end > start ? Math.round((end - start) / 6000) / 10 : 1;
    return [
      `<div class="${classes.join(' ')}" role="listitem"${rangeAttrs}${indexAttr} style="flex-grow:${minutes}">`,
      `<b>${escapeHtml(head || '—')}${step.selected ? '<span class="rep-step-tag">SLA</span>' : ''}</b>`,
      `<span class="rep-step-sub">${escapeHtml(sub)}</span>`,
      flag ? `<span class="rep-step-flag">${escapeHtml(flag)}</span>` : '',
      '</div>'
    ].join('');
  }
  // Row of steps under the load profile («Ход теста») and the note on where the numbers come from.
  function renderLoadStepTableHtml(table) {
    const cells = table.steps.map(stepCellHtml).join('');
    const sourceBits = [];
    if (safeStr(table.latency_query)) sourceBits.push(LL.t('report.steps.source_query', { query: safeStr(table.latency_query) }));
    if (safeStr(table.latency_series)) sourceBits.push(LL.t('report.steps.source_series', { series: safeStr(table.latency_series) }));
    else if (safeStr(table.latency_query)) sourceBits.push(LL.t('report.steps.source_max'));
    const note = sourceBits.length ? LL.t('report.steps.note_source', { source: sourceBits.join(', ') }) : LL.t('report.steps.note');
    return [
      `<div class="rep-steprow" role="list" aria-label="${escapeHtml(LL.t('report.steps.title'))}">${cells}</div>`,
      `<p class="rep-step-note">${escapeHtml(note)}</p>`
    ].join('');
  }
  // «UTC+3», «UTC-5», «UTC+5:30» for an offset in hours.
  function utcOffsetText(hours) {
    const value = Number(hours);
    if (!Number.isFinite(value)) return 'UTC';
    const abs = Math.abs(value);
    const whole = Math.floor(abs);
    const minutes = Math.round((abs - whole) * 60);
    return `UTC${value < 0 ? '-' : '+'}${whole}${minutes ? `:${String(minutes).padStart(2, '0')}` : ''}`;
  }
  // «Ход теста»: heading with the clock of the report, the profile host (LL.charts.stepProfile) and the step row.
  function timelineHtml(table) {
    return [
      '<div class="rep-sec">',
      `<h2>${escapeHtml(LL.t('report.timeline.title'))}</h2>`,
      `<span class="lbl">${escapeHtml(LL.t('report.timeline.caption', { zone: utcOffsetText(reportOffsetHours(table.steps)) }))}</span>`,
      '</div>',
      '<div class="rep-profile" data-step-profile></div>',
      renderLoadStepTableHtml(table)
    ].join('');
  }
  // ---- chart markers, SLA thresholds and series order ------------------------------------------------
  // Mirrors AI/report_markers.py (markers_for, thresholds_for, order_series), which draws the same items
  // on the Confluence PNGs; tests/test_report_markers.py runs both sides on the same rows. Only
  // deterministic data is used: the step table, the SLA window and the test-type payloads, never the
  // LLM text of peak_performance.max_time / drop_time.
  // Check name -> [dictionary key of the threshold label ({value}), tone].
  const THRESHOLD_SPECS = {
    target_rps: ['report.threshold.target_rps', 'accent'],
    p95_latency: ['report.threshold.p95_latency', 'fail'],
    p99_latency: ['report.threshold.p99_latency', 'fail'],
    error_rate: ['report.threshold.error_rate', 'fail'],
    cpu_usage: ['report.threshold.cpu_usage', 'fail'],
    memory_usage: ['report.threshold.memory_usage', 'fail']
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
    const label = chosen && safeStr(chosen.index) ? LL.t('report.marker.sla_step', { index: safeStr(chosen.index) }) : LL.t('report.marker.sla_window');
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
      if (degradationMs !== null) out.push(markerItem('line', degradationMs, null, LL.t('report.marker.degradation', { time: clockAt(degradationMs, offset) }), 'fail', 'degradation'));
    }
    let dropMs = saturation ? utcMoment(saturation.rps_drop_iso) : null;
    if (dropMs === null) {
      const afterDrop = steps.find((step) => step.after_drop === true);
      dropMs = afterDrop ? utcMoment(afterDrop.start_iso) : null;
    }
    if (dropMs !== null && (degradationMs === null || Math.abs(dropMs - degradationMs) > DROP_MERGE_MS)) {
      out.push(markerItem('line', dropMs, null, LL.t('report.marker.rps_drop', { time: clockAt(dropMs, offset) }), 'fail', 'rps_drop'));
    }
    return out;
  }
  function holdMarkers(parsed, window) {
    const stability = isPlainObject(parsed.stability_under_load) ? parsed.stability_under_load : {};
    const hold = isPlainObject(stability.hold) ? stability.hold : {};
    let band = bandItem(hold.start_iso, hold.end_iso, LL.t('report.marker.hold_window'), 'accent', 'hold');
    if (!band && safeStr(window.kind).toLowerCase() === 'hold') {
      band = bandItem(window.start || window.start_iso, window.end || window.end_iso, LL.t('report.marker.hold_window'), 'accent', 'hold');
    }
    return band ? [band] : [];
  }
  function spikeMarkers(parsed, window) {
    const out = [];
    const analysis = isPlainObject(parsed.spike_analysis) ? parsed.spike_analysis : {};
    (Array.isArray(analysis.spikes) ? analysis.spikes : []).forEach((spike, position) => {
      if (!isPlainObject(spike)) return;
      const label = safeStr(spike.label) || LL.t('report.marker.spike_n', { index: safeStr(spike.index) || position + 1 });
      const band = bandItem(spike.start_iso, spike.end_iso, label, 'warn', 'spike');
      if (band) out.push(band);
    });
    if (!out.length && safeStr(window.kind).toLowerCase() === 'spike') {
      const band = bandItem(window.start || window.start_iso, window.end || window.end_iso, safeStr(window.title) || LL.t('report.marker.spike'), 'warn', 'spike');
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
  // Query label and series named in the text of a check (reports saved before checks carried query_label and
  // series): label='…' / series='…', or «запрос «…»» / «серия «…»» of a ru report and «query «…»» / «series «…»»
  // of an en one (AI/report_markers.py parses the same).
  const QUERY_HINT = /(?:запрос|query) «([^»]*)»/;
  const SERIES_HINT = /(?:серия|series) «([^»]*)»/;
  function messageHint(message) {
    const label = (message.match(/label='([^']*)'/) || message.match(QUERY_HINT) || [])[1] || '';
    const series = (message.match(/series='([^']*)'/) || message.match(SERIES_HINT) || [])[1] || '';
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
      out.push({ name: name, query_label: where.label, series: where.series, value: value, label: LL.t(spec[0], { value: formatNumber(value) }), tone: spec[1] });
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
    box = document.createElement('section');
    box.id = 'reportSteps';
    box.className = 'rep-timeline';
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
  // Shades a step of «Ход теста» (null clears it).
  function highlightProfile(stepIndex) {
    const profile = reportState.profile;
    if (profile && typeof profile.highlight === 'function') profile.highlight(stepIndex);
  }
  // Hovering (or focusing) a cell of the step row highlights that step on the load profile and on every chart of
  // the report, in a neutral tone (the accent marks the SLA zone only).
  function bindStepHighlight(box) {
    const cells = Array.from(box.querySelectorAll('[data-step-from]'));
    const clear = () => { highlightCharts(null); highlightProfile(null); };
    cells.forEach((cell) => {
      const range = { t: Number(cell.dataset.stepFrom), to: Number(cell.dataset.stepTo), tone: 'muted' };
      const show = () => { highlightCharts(range); highlightProfile(cell.dataset.stepIndex === undefined ? null : cell.dataset.stepIndex); };
      cell.addEventListener('mouseenter', show);
      cell.addEventListener('mouseleave', clear);
      cell.addEventListener('focus', show);
      cell.addEventListener('blur', clear);
    });
  }
  // Resolves the step table, the chart markers and thresholds (charts read them, so this runs before they are
  // drawn) and draws «Ход теста»: the load profile and the step row. Without steps the block stays hidden.
  async function renderLoadSteps(finalRow) {
    const box = stepsHost();
    const table = await resolveStepTable(finalRow);
    reportState.stepTable = table;
    try {
      reportState.markers = reportMarkers(finalRow, table);
      reportState.thresholds = slaThresholds(finalRow, table);
    } catch (e) {
      reportState.markers = [];
      reportState.thresholds = [];
    }
    if (reportState.profile) {
      reportState.profile.destroy();
      reportState.profile = null;
    }
    if (!box) return table;
    if (!table) {
      box.hidden = true;
      box.innerHTML = '';
      return null;
    }
    const linked = linkSlaToSteps(table, finalRow);
    box.hidden = false;
    box.innerHTML = timelineHtml(Object.assign({}, table, linked));
    const host = box.querySelector('[data-step-profile]');
    if (host && LL.charts && typeof LL.charts.stepProfile === 'function') {
      const target = finiteOrNull(checkNamed(markerChecks(finalRow), 'target_rps').threshold);
      reportState.profile = LL.charts.stepProfile(host, { steps: linked.steps, markers: reportState.markers, target: target, height: 190 });
    }
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
        { label: LL.t('report.hero.min_interval'), value: hold.min_bucket_rps != null ? formatNumber(hold.min_bucket_rps) : '', note: hold.drop_pct ? LL.t('report.hero.drop_from_mean', { pct: formatNumber(hold.drop_pct) }) : '' }
      ];
    }
    if (mode === 'spike') return [];
    if (mode === 'stress') {
      const sat = parsed.saturation || {};
      const resource = (sat.limiting_resource && typeof sat.limiting_resource === 'object') ? resourceText(sat.limiting_resource) : '';
      return [
        { label: LL.t('report.section.failure_mode'), value: failureModeTitle(sat) || '—', note: safeStr(sat.degraded_step_label) ? LL.t('report.hero.from_step', { step: safeStr(sat.degraded_step_label) }) : '' },
        { label: LL.t('report.section.limiting_resource'), labelHtml: LL.ui.term('limiting_resource', LL.t('report.section.limiting_resource')), value: resource || LL.t('report.section.not_determined') }
      ];
    }
    if (!peakTableApplies('final', peak)) return [];
    return [
      { label: LL.t('report.hero.peak_time'), value: safeStr(peak.max_time) || '—' },
      { label: LL.t('report.section.drop_time'), value: safeStr(peak.drop_time) || '—' }
    ];
  }
  function kpiTone(passed) {
    if (passed === true) return 'ok';
    if (passed === false) return 'fail';
    return '';
  }
  // {limit: «порог ≤ 300 мс», state: «нарушен»}: `bound` renders the threshold text, `words` are the passed / failed
  // states; null without a threshold.
  function kpiThresholdParts(check, bound, words) {
    if (!check || finiteOrNull(check.threshold) === null) return null;
    const state = check.passed === true ? words[0] : (check.passed === false ? words[1] : '');
    return { limit: bound(formatNumber(check.threshold)), state: state };
  }
  // «порог ≤ 300 мс · нарушен».
  function kpiThresholdNote(check, bound, words) {
    const parts = kpiThresholdParts(check, bound, words);
    return parts ? `${parts.limit}${parts.state ? ' · ' + parts.state : ''}` : '';
  }
  // Figure extras of a tile: the threshold and its state apart (the state is set in bold) and the bar of the check.
  function withFigure(item, check, parts) {
    if (!item) return item;
    if (parts) Object.assign(item, { limit: parts.limit, state: parts.state });
    item.bar = check ? slaBarModel(check) : null;
    return item;
  }
  // heroKpis(finalRow): up to three number tiles {key, label, term, value, unit, note, tone, limit?, state?, bar}
  // drawn as the figures of the verdict block (limit and state split the note of a threshold, bar is slaBarModel).
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
    const limitWords = [LL.t('report.kpi.state_met'), LL.t('report.kpi.state_broken')];
    const latencyBound = (value) => LL.t('report.kpi.limit_ms', { value: value });
    const latencyNote = kpiThresholdNote(latency, latencyBound, limitWords);
    const latencyTile = latency ? withFigure(tile('latency', latency.name === 'p99_latency' ? 'p99' : 'p95', latency.name === 'p99_latency' ? '' : 'p95', latency.actual, ' ' + LL.t('report.unit.ms'), latencyNote, kpiTone(latency.passed)), latency, kpiThresholdParts(latency, latencyBound, limitWords)) : null;
    const errorBound = (value) => LL.t('report.kpi.limit_pct', { value: value });
    const errorNote = kpiThresholdNote(errors, errorBound, limitWords);
    const errorTile = errors ? withFigure(tile('errors', LL.t('report.kpi.errors'), 'error_rate', errors.actual, ' %', errorNote, kpiTone(errors.passed)), errors, kpiThresholdParts(errors, errorBound, limitWords)) : null;
    const targetBound = (value) => LL.t('report.kpi.goal_rps', { value: value });
    const targetWords = [LL.t('report.kpi.goal_reached'), LL.t('report.kpi.goal_missed')];
    const targetNote = kpiThresholdNote(target, targetBound, targetWords);
    if (mode === 'spike') {
      const sa = parsed.spike_analysis || {};
      const spikes = Array.isArray(sa.spikes) ? sa.spikes : [];
      const worst = spikes.find((item) => item && item.index === sa.worst_spike_index) || spikes[0] || {};
      const notRecovered = spikes.filter((item) => item && item.recovered === false).length;
      const times = spikes.map((item) => Number(item && item.recovery_time_sec)).filter((n) => Number.isFinite(n));
      const recovery = notRecovered
        ? LL.t('report.kpi.not_recovered', { count: notRecovered, total: spikes.length })
        : (times.length ? LL.t('report.kpi.seconds', { value: formatNumber(Math.max.apply(null, times)) }) : '—');
      const recoveryCheck = byName('recovery');
      const recoveryLimit = recoveryCheck && finiteOrNull(recoveryCheck.threshold) !== null ? LL.t('report.kpi.recovery_limit', { value: formatNumber(recoveryCheck.threshold) }) : '';
      const baseline = sa.baseline_rps != null ? LL.t('report.kpi.baseline', { rps: formatNumber(sa.baseline_rps) }) : '';
      return [
        sa.detected
          ? withFigure(tile('spike', LL.t('report.kpi.spike'), 'spike_window', sa.max_spike_rps, ' RPS', [baseline, LL.t('report.kpi.spikes_count', { count: spikes.length }), targetNote].filter(Boolean).join(' · '), kpiTone(target && target.passed)), target, null)
          : textTile('spike', LL.t('report.kpi.spike'), 'spike_window', LL.t('report.section.spike_not_detected'), baseline, ''),
        textTile('overshoot', LL.t('report.kpi.overshoot'), 'overshoot', signed(worst.overshoot_p95_pct, ' %') || '—', LL.t('report.kpi.worst_spike'), kpiTone(latency && latency.passed)),
        withFigure(textTile('recovery', LL.t('report.kpi.recovery'), 'recovery_time', recovery, recoveryLimit, kpiTone(recoveryCheck && recoveryCheck.passed)), recoveryCheck, null)
      ];
    }
    if (mode === 'stability') {
      const st = parsed.stability_under_load || {};
      const hold = isPlainObject(st.hold) ? st.hold : {};
      const span = hold.start_iso ? `${hhmm(hold.start_iso)}–${hhmm(hold.end_iso)}` : '';
      const held = target && target.passed === true ? LL.t('report.kpi.goal_held') : (target && target.passed === false ? LL.t('report.kpi.goal_not_held') : '');
      const goal = finiteOrNull(st.target_rps) !== null ? LL.t('report.kpi.goal_rps', { value: formatNumber(st.target_rps) }) + (held ? ' · ' + held : '') : targetNote;
      return [
        withFigure(tile('rps', LL.t('report.kpi.mean_rps'), 'hold_window', st.actual_rps, ' RPS', [goal, span].filter(Boolean).join(' · '), kpiTone(target && target.passed)), target, null),
        latencyTile,
        errorTile
      ].filter(Boolean);
    }
    if (mode === 'stress') {
      const sat = parsed.saturation || {};
      const point = finiteOrNull(sat.saturation_rps) !== null ? sat.saturation_rps : peak.max_rps;
      return [
        tile('rps', LL.t('report.kpi.saturation_point'), 'saturation_point', point, ' RPS', safeStr(sat.saturation_step_label), ''),
        latencyTile,
        errorTile
      ].filter(Boolean);
    }
    const peakRps = finiteOrNull(peak.max_rps) !== null ? peak.max_rps : (target ? target.actual : null);
    const method = safeStr(peak.method);
    const rpsTerm = /stable/i.test(method) ? 'stable_max' : 'peak_max';
    const rpsTile = finiteOrNull(peakRps) !== null || target
      ? withFigure(tile('rps', LL.t('report.kpi.max_rps'), rpsTerm, peakRps, ' RPS', targetNote || (method ? LL.t('report.kpi.method', { method: method }) : ''), kpiTone(target && target.passed)), target, kpiThresholdParts(target, targetBound, targetWords))
      : null;
    return [rpsTile, latencyTile, errorTile].filter(Boolean);
  }
  // Figures of the verdict block: value with its unit, «label · threshold <b>state</b>» and the bar against the
  // threshold. A value that breaks its threshold takes the fail tone; text values (spike tests) are smaller.
  function renderHeroKpis(tiles) {
    if (!tiles.length) return '';
    const items = tiles.map((item) => {
      const label = item.term ? LL.ui.term(item.term, item.label) : escapeHtml(item.label);
      const unit = item.unit ? `<small>${escapeHtml(safeStr(item.unit))}</small>` : '';
      const tone = item.tone ? ` ${item.tone}` : '';
      let caption = label;
      if (item.limit) caption += ` · ${escapeHtml(item.limit)}${item.state ? ` <b>${escapeHtml(item.state)}</b>` : ''}`;
      else if (item.note) caption += ` · ${escapeHtml(item.note)}`;
      const bar = item.bar;
      const barHtml = bar
        ? `<div class="fig-bar" aria-hidden="true"><i${bar.tone === 'ok' ? '' : ` class="${bar.tone}"`} style="width:${bar.fill}%"></i><b style="left:${bar.mark}%"></b></div>`
        : '';
      return [
        `<div class="fig rep-fig${item.unit ? '' : ' is-text'}" role="listitem" data-kpi="${escapeHtml(item.key)}">`,
        `<div class="v${item.tone === 'fail' ? ' fail' : ''}">${escapeHtml(item.value)}${unit}</div>`,
        `<div class="k${tone}">${caption}</div>`,
        barHtml,
        '</div>'
      ].join('');
    });
    return `<div class="figs rep-figs" role="list" aria-label="${escapeHtml(LL.t('report.kpi.aria'))}">${items.join('')}</div>`;
  }
  // Language a report was built in: report_language of the final row (parsed, else sla_details); reports saved
  // before the field are Russian.
  function reportLanguageOf(finalRow) {
    const parsed = parseJsonish(finalRow && finalRow.parsed) || {};
    const sla = parseJsonish(finalRow && finalRow.sla_details) || {};
    const lang = safeStr(parsed.report_language || sla.report_language).toLowerCase().slice(0, 2);
    return lang === 'en' ? 'en' : 'ru';
  }
  // Texts stored with the report (AI findings, SLA messages and summaries, step labels) stay in its language;
  // a badge says so when that is not the UI language. The language is named in the UI language ("Russian").
  function reportLanguageBadge(finalRow) {
    const lang = reportLanguageOf(finalRow);
    if (lang === LL.locale) return '';
    const name = lang === 'en' ? LL.t('common.lang_name.en') : LL.t('common.lang_name.ru');
    const text = LL.t('report.language_badge', { language: name });
    return `<span class="hero-report-language"><span class="pill na" data-report-language="${lang}" title="${escapeHtml(LL.t('report.language_badge_title'))}">${escapeHtml(text)}</span></span>`;
  }
  // ---- verdict sentence ----------------------------------------------------------------------------
  // One plain-language sentence under the verdict word, built only from sla_details.checks (never from the model
  // text): whether the target RPS was reached on the checked window, which thresholds were breached and by how much,
  // the error rate and what had no data. HTML with the figures in bold, in the UI language like the rest of the
  // verdict block; '' without checks. Keys are literal (tests/test_i18n.py checks every one exists).
  const SUMMARY_NAMES = {
    target_rps: 'report.summary.name.target_rps',
    p95_latency: 'report.summary.name.p95_latency',
    p99_latency: 'report.summary.name.p99_latency',
    error_rate: 'report.summary.name.error_rate',
    cpu_usage: 'report.summary.name.cpu_usage',
    memory_usage: 'report.summary.name.memory_usage',
    recovery: 'report.summary.name.recovery'
  };
  function summaryName(name) {
    const key = SUMMARY_NAMES[safeStr(name)];
    return key ? LL.t(key) : (slaCheckLabel(name) || safeStr(name) || LL.t('report.steps.check_fallback'));
  }
  // Unit of a check value in prose: «450 мс», «0.4 %», «249.3 RPS», «7 мин».
  function checkUnit(name) {
    const key = safeStr(name);
    if (key === 'target_rps') return ' RPS';
    if (key === 'p95_latency' || key === 'p99_latency') return ' ' + LL.t('report.unit.ms');
    if (key === 'recovery') return ' ' + LL.t('report.unit.min');
    if (key === 'error_rate' || key === 'cpu_usage' || key === 'memory_usage') return ' %';
    return '';
  }
  function summaryFigure(value, name) {
    return LL.html(`<b>${escapeHtml(formatNumber(value) + checkUnit(name))}</b>`);
  }
  function summaryNames(checks) {
    return checks.map((check) => summaryName(check.name)).join(', ');
  }
  // Sentence start in upper case, unless it opens with markup or a metric name such as «p95».
  function capitalizeHtml(html) {
    const text = String(html || '');
    return /^\p{Ll}\p{L}/u.test(text) ? text[0].toUpperCase() + text.slice(1) : text;
  }
  function verdictSummaryHtml(finalRow) {
    if (!finalRow) return '';
    const checks = markerChecks(finalRow).filter((check) => safeStr(check.name));
    if (!checks.length) return '';
    const mode = markerMode(parseJsonish(finalRow.parsed) || {}, markerSlaWindow(finalRow));
    const target = checks.find((check) => safeStr(check.name) === 'target_rps') || null;
    const others = checks.filter((check) => check !== target);
    const failed = others.filter((check) => check.passed === false);
    const unknown = others.filter((check) => check.passed !== true && check.passed !== false);
    const met = others.filter((check) => check.passed === true);
    const html = (key, params) => LL.html(LL.tHtml(key, params));
    let head = null;
    if (target && target.passed === true) {
      head = html(mode === 'hold' ? 'report.summary.target_held' : (mode === 'spike' ? 'report.summary.target_spike' : 'report.summary.target_met'));
    } else if (target && target.passed === false) {
      const params = { actual: summaryFigure(target.actual, 'target_rps'), threshold: summaryFigure(target.threshold, 'target_rps') };
      head = html(mode === 'hold' ? 'report.summary.target_not_held' : 'report.summary.target_missed', params);
    } else if (target) {
      head = html('report.summary.target_unknown');
    }
    let breach = null;
    if (failed.length === 1) {
      const check = failed[0];
      const name = safeStr(check.name);
      if (finiteOrNull(check.actual) === null || finiteOrNull(check.threshold) === null) {
        breach = html('report.summary.breach_no_value', { name: summaryName(name) });
      } else {
        const params = { name: summaryName(name), actual: summaryFigure(check.actual, name), threshold: summaryFigure(check.threshold, name) };
        breach = html(name === 'recovery' ? 'report.summary.breach_recovery' : 'report.summary.breach_one', params);
      }
    } else if (failed.length > 1) {
      breach = html('report.summary.breach_many', { names: summaryNames(failed) });
    }
    let text = null;
    if (head && breach) text = html(target.passed === true ? 'report.summary.but' : 'report.summary.and', { head: head, rest: breach });
    else if (head) text = met.length ? html('report.summary.rest_met', { head: head }) : head;
    else if (breach) text = LL.html(capitalizeHtml(breach.html));
    else if (met.length) text = html('report.summary.all_met');
    const errors = met.find((check) => safeStr(check.name) === 'error_rate' && finiteOrNull(check.actual) !== null);
    if (text && errors) text = html('report.summary.with_errors', { text: text, value: summaryFigure(errors.actual, 'error_rate') });
    if (unknown.length) {
      text = text
        ? html('report.summary.with_unknown', { text: text, names: summaryNames(unknown) })
        : html('report.summary.only_unknown', { names: summaryNames(unknown) });
    }
    return text ? `${capitalizeHtml(text.html)}.` : '';
  }

  // ---- report head: meta line, verdict block ----------------------------------------------------------
  const META_SEP = '<span class="sep" aria-hidden="true">·</span>';
  // Mono meta line under the title: test type · service · date and window · duration · report time · AI usage ·
  // report language (when it is not the UI language). heroHtml carries it; renderHero moves it under the title.
  function reportMetaHtml(finalRow, allRows) {
    const parsed = parseJsonish(finalRow.parsed) || {};
    const testType = normalizedTestType(parsed, finalRow.test_type);
    const typeTermKey = LL.GLOSSARY['test_type_' + testType] ? 'test_type_' + testType : 'step_profile';
    const items = [LL.ui.term(typeTermKey, LL.testTypeLabel(finalRow.test_type))];
    const service = safeStr(finalRow.service) || getServiceFromPath();
    if (service) items.push(escapeHtml(service));
    const startMs = Number(finalRow.start_ms);
    const endMs = Number(finalRow.end_ms);
    const zone = LL.timeZoneLabel();
    const zoneShort = escapeHtml(zone.split(' ')[0]);
    if (Number.isFinite(startMs) && Number.isFinite(endMs) && startMs > 0 && endMs >= startMs) {
      const from = LL.formatDateTime(startMs).split(' ');
      const to = LL.formatDateTime(endMs).split(' ');
      const window = from[0] === to[0]
        ? `${escapeHtml(from[0])} <b>${escapeHtml(from[1] + '–' + to[1])}</b>`
        : `<b>${escapeHtml(from.join(' '))} — ${escapeHtml(to.join(' '))}</b>`;
      items.push(`<span title="${escapeHtml(zone)}">${window} ${zoneShort}</span>`);
      items.push(escapeHtml(LL.formatDuration(endMs - startMs)));
    }
    const created = momentMs(finalRow.created_at);
    if (created !== null) {
      const stamp = LL.formatDateTime(created).split(' ');
      const sameDay = Number.isFinite(endMs) && LL.formatDateTime(endMs).split(' ')[0] === stamp[0];
      items.push(escapeHtml(LL.t('report.meta.created', { time: sameDay ? stamp[1] : stamp.join(' ') })));
    }
    const usage = formatUsageLine(collectRunUsage(finalRow, allRows));
    if (usage) items.push(`<span class="hero-usage">${escapeHtml(usage)}</span>`);
    const badge = reportLanguageBadge(finalRow);
    if (badge) items.push(badge);
    return `<div class="rep-meta mono" data-report-meta role="group" aria-label="${escapeHtml(LL.t('report.meta.aria'))}">${items.join(META_SEP)}</div>`;
  }
  // Facts of the test type that are not figures (failure mode, limiting resource, peak and drop times …) in one line.
  function heroFactsHtml(entries) {
    const items = (entries || []).filter((entry) => entry && safeStr(entry.value)).map((entry) => [
      '<span class="rep-fact">',
      `<span class="rep-fact-k">${entry.labelHtml || escapeHtml(entry.label)}</span> `,
      `<span class="rep-fact-v">${escapeHtml(entry.value)}</span>`,
      entry.note ? ` <span class="rep-fact-n">${escapeHtml(entry.note)}</span>` : '',
      '</span>'
    ].join(''));
    return items.length ? `<div class="rep-facts">${items.join(META_SEP)}</div>` : '';
  }
  // «Вердикт верен?» with the votes, then «Изменить вердикт» / «Вернуть автоматический», on one line under the SLA table.
  function verdictLineHtml(finalRow) {
    const manual = manualOf(finalRow);
    const votes = renderVoteGroup('final', 'verdict', '', LL.t('report.feedback.agree_verdict'), LL.t('report.feedback.disagree_verdict'));
    return [
      '<div class="rep-verdict-line" data-min-role="engineer">',
      votes ? `<span class="rep-verdict-q">${escapeHtml(LL.t('report.feedback.verdict_question'))}</span>${votes}` : '',
      '<span class="rep-verdict-actions">',
      `<button type="button" class="btn btn-sm" data-verdict-action="change">${escapeHtml(LL.t('report.verdict.change'))}</button>`,
      manual ? `<button type="button" class="btn btn-sm" data-verdict-action="reset">${escapeHtml(LL.t('report.verdict.reset_button'))}</button>` : '',
      '</span>',
      '</div>'
    ].join('');
  }
  // Markup of the report head of the final row: the meta line, then the verdict block — a rule in the verdict
  // color, the verdict word, the manual tag and reason, one sentence on why, the figures; the SLA criteria with the
  // verdict line on the right. renderHero puts it on the page.
  function heroHtml(finalRow, allRows) {
    if (!finalRow) return `<div class="rep-empty hero-empty">${escapeHtml(LL.t('report.hero.no_final'))}</div>`;
    const parsed = parseJsonish(finalRow.parsed) || {};
    const peak = (parsed.peak_performance && typeof parsed.peak_performance === 'object') ? parsed.peak_performance : {};
    const manual = manualOf(finalRow);
    const code = LL.verdictCode(finalRow.verdict);
    const verdictTermKey = LL.verdictTermKey(code);
    const tip = escapeHtml(LL.GLOSSARY[verdictTermKey] || '');
    const summary = verdictSummaryHtml(finalRow);
    const unverified = (function () {
      const verification = (parsed.verification_summary && typeof parsed.verification_summary === 'object') ? parsed.verification_summary : null;
      const unverifiedCount = verification ? Number(verification.unverified) : 0;
      if (!verification || !(unverifiedCount > 0)) return '';
      const note = verification.revised_by_model
        ? LL.tn('report.hero.unverified', unverifiedCount)
        : LL.tn('report.hero.unverified_no_revision', unverifiedCount);
      return `<div class="hero-sub hero-unverified">${LL.ui.term('unverified', note)}</div>`;
    })();
    return [
      reportMetaHtml(finalRow, allRows),
      `<div class="rep-verdict tone-${LL.verdictClass(code)}" data-verdict="${code}">`,
      '<div class="rep-verdict-main">',
      `<div class="lbl rep-verdict-cap">${LL.ui.term('sla_first', LL.t('report.hero.verdict_title'))}</div>`,
      '<div class="rep-verdict-row">',
      `<abbr class="term rep-verdict-term" data-term="${verdictTermKey}" data-tip="${tip}" title="${tip}" tabindex="0"><span class="rep-verdict-word">${escapeHtml(LL.verdictLabel(code))}</span></abbr>`,
      manual ? `<span class="verdict-manual-tag">${escapeHtml(LL.t('report.verdict.manual_tag'))}</span>` : '',
      '</div>',
      summary ? `<p class="rep-verdict-why">${summary}</p>` : '',
      `<div class="hero-sub rep-verdict-source">${escapeHtml(verdictSourceText(finalRow))}</div>`,
      manual && safeStr(manual.reason) ? `<div class="hero-sub hero-manual-reason">${escapeHtml(LL.t('report.verdict.reason_line', { reason: safeStr(manual.reason) }))}</div>` : '',
      unverified,
      renderHeroKpis(heroKpis(finalRow)),
      heroFactsHtml(heroTypeEntries(parsed, peak)),
      '</div>',
      '<div class="rep-sla">',
      renderSlaChecklist(finalRow),
      verdictLineHtml(finalRow),
      '</div>',
      '</div>'
    ].join('');
  }
  function renderHero(finalRow, allRows) {
    const hero = document.getElementById('reportHero');
    if (!hero) return;
    hero.innerHTML = heroHtml(finalRow, allRows);
    // The meta line sits under the title, in the slot that showed the test time while the report loaded.
    const meta = hero.querySelector('[data-report-meta]');
    const slot = document.getElementById('pageTimeRange');
    if (meta && slot) {
      slot.innerHTML = '';
      slot.appendChild(meta);
    }
    if (!finalRow) return;
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
      box.textContent = LL.t('report.page.no_data');
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
      if (range) range.textContent = LL.t('report.page.time_range', { start: startStr, end: endStr, zone: LL.timeZoneLabel() });
    } catch (e) {}
    const finalRow = arr.find((x) => x.domain === 'final') || null;
    reportState.finalRow = finalRow;
    reportState.rows = arr;
    renderHero(finalRow, arr);
    // Charts read the markers and thresholds, so the step table is resolved (and «Ход теста» drawn) before them.
    await renderLoadSteps(finalRow);
    reportState.finalRow = finalRow;
    renderForecastLink();
    const domainsOrder = ['final', 'jvm', 'database', 'kafka', 'microservices', 'hard_resources', 'lt_framework', 'lt_generators', 'application_logs'];
    arr.sort((a, b) => domainsOrder.indexOf(a.domain) - domainsOrder.indexOf(b.domain));
    const tabsNav = document.createElement('div'); tabsNav.className = 'tabs'; tabsNav.setAttribute('aria-label', LL.t('report.tabs.analysis_aria'));
    const tabsBody = document.createElement('div');
    const idBase = 'rep-llm-tab-';
    arr.forEach((x, idx) => {
      const btn = document.createElement('button'); btn.className = 'tab' + (idx === 0 ? ' active' : '');
      const dot = document.createElement('span');
      const verdictForTab = x.verdict || x.llm_verdict || '';
      reportState.domainVerdicts[x.domain] = verdictForTab;
      dot.className = `tab-dot ${LL.verdictClass(verdictForTab)}`;
      dot.setAttribute('aria-hidden', 'true');
      btn.title = LL.t('report.tabs.verdict_title', { verdict: LL.verdictLabel(verdictForTab) });
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
      const aiBadge = `<div class="ai-badge" title="${escapeHtml(LL.t('report.ai_badge.title'))}">${escapeHtml(LL.t('report.ai_badge.text'))}</div>`;
      // Print shows every tab panel; the heading names the domain there (hidden on screen).
      const printTitle = `<h3 class="report-print-title">${escapeHtml(LL.t('report.tabs.print_analysis', { domain: LL.domainTitleFor(x.domain, reportDomainTitles) }))}</h3>`;
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
  // Time axis tick in the clock of the report, so that the ticks agree with the marker labels and the step row
  // whatever the browser's time zone is.
  function clockLabel(label, offsetHours) {
    const ms = utcMoment(label);
    if (ms === null) return String(label || '');
    return clockAt(ms, offsetHours === undefined ? reportClockOffset() : offsetHours);
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
        box.innerHTML = `<div class="report-chart-empty">${escapeHtml(LL.t('report.charts.empty'))}</div>`;
      }
      if (legendHost) legendHost.remove();
      return null;
    }
    const built = buildChartData(data.points);
    const colors = LL.ui.chartColors();
    const offset = reportClockOffset();
    const tooltip = LL.charts && typeof LL.charts.htmlTooltip === 'function' ? LL.charts.htmlTooltip({ max: TOOLTIP_ROWS }) : {};
    tooltip.callbacks = Object.assign({}, tooltip.callbacks, {
      title(items) {
        const ms = items && items.length ? utcMoment(items[0].label) : null;
        return ms === null ? '' : dateTimeAt(ms, offset);
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
              callback(value) { return clockLabel(this.getLabelForValue(value), offset); }
            },
            // Instrument look: horizontal grid only, the time axis is a hairline (step starts are marker ticks).
            grid: { color: colors.grid, display: false }
          },
          y: { ticks: { color: colors.text }, grid: { color: colors.grid }, border: { display: false } }
        }
      },
      plugins: plugins
    });
    const shown = Math.min(built.datasets.length, MAX_VISIBLE_SERIES);
    const count = built.datasets.length;
    canvas.setAttribute('aria-label', count > shown
      ? LL.t('report.charts.aria_series_shown', { query: ql, count: count, shown: shown })
      : LL.t('report.charts.aria_series', { query: ql, count: count }));
    reportState.charts.push(chart);
    if (group) group.charts.push(chart);
    if (legendHost && LL.charts && typeof LL.charts.legendChips === 'function') {
      const chips = LL.charts.legendChips(chart, legendHost, { filename: `report-${domain}-${ql}`, maxVisible: MAX_VISIBLE_SERIES, separator: '' });
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
      ed.innerHTML = html || `<p class="rep-engineer-empty">${escapeHtml(LL.t('report.engineer.placeholder'))}</p>`;
      updated.textContent = j.created_at ? LL.t('report.engineer.updated', { time: j.created_at.replace('T', ' ').slice(0, 16) }) : '';
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
          title: LL.t('report.engineer.link_title'),
          label: LL.t('report.engineer.link_label'),
          value: 'https://',
          validate: (v) => (/^https?:\/\/\S+$/i.test(v.trim()) ? '' : LL.t('report.engineer.link_invalid'))
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
        if (editBtn) editBtn.textContent = on ? LL.t('report.engineer.finish') : LL.t('report.engineer.edit');
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
        st.textContent = LL.t('report.status.saving');
        try {
          const resp = await fetch('/engineer_summary', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ run_name: run, content_html: html }) });
          const j = await resp.json();
          if (resp.ok) { st.textContent = LL.t('report.status.saved'); setEngineerEditing(false); await engineerLoad(); setTimeout(() => { st.textContent = ''; }, 1500); }
          else { st.textContent = j.error || LL.t('report.status.save_failed'); }
        } catch (e) { st.textContent = LL.t('report.status.save_failed'); }
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
    printTitle.textContent = LL.t('report.charts.print_title', { domain: LL.domainTitleFor(domain, reportDomainTitles) });
    pane.appendChild(printTitle);
    const group = { charts: [], chips: [], sync: null, zoom: null };
    reportState.groups[pane.id] = group;
    const jobs = queryLabels.map((ql, qi) => {
      const wrap = document.createElement('section');
      wrap.className = 'report-chart';
      // Title row: the query in mono, the zoom hint on the right of the first chart of the tab.
      const head = document.createElement('div');
      head.className = 'report-chart-head';
      const title = document.createElement('h4');
      title.className = 'report-chart-title';
      title.textContent = ql;
      head.appendChild(title);
      if (qi === 0) {
        const hint = document.createElement('span');
        hint.className = 'report-chart-hint';
        hint.textContent = LL.t('report.charts.zoom_hint_short');
        hint.title = LL.t('report.charts.zoom_hint');
        head.appendChild(hint);
      }
      const canvasBox = document.createElement('div');
      canvasBox.className = 'report-chart-canvas';
      const cnv = document.createElement('canvas');
      cnv.id = `rep-chart-${domainIdx}-${qi}`;
      cnv.setAttribute('role', 'img');
      cnv.setAttribute('aria-label', LL.t('report.charts.aria', { query: ql }));
      canvasBox.appendChild(cnv);
      const legend = document.createElement('div');
      legend.className = 'report-chart-legend';
      wrap.appendChild(head);
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
    root.innerHTML = escapeHtml(LL.t('report.charts.loading'));
    const domainsOrder = ['lt_framework', 'microservices', 'jvm', 'database', 'kafka', 'hard_resources', 'lt_generators'];
    const domains = Object.keys(schema || {}).sort((a, b) => {
      const ia = domainsOrder.indexOf(a); const ib = domainsOrder.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
    // Sticky bar: the domain tabs and, while the open tab is zoomed, «Сбросить масштаб».
    const bar = document.createElement('div'); bar.className = 'report-tabs-bar tabs-sticky';
    const sentinel = document.createElement('div'); sentinel.className = 'tabs-sticky-sentinel'; sentinel.setAttribute('aria-hidden', 'true');
    const tabsNav = document.createElement('div'); tabsNav.className = 'tabs'; tabsNav.setAttribute('aria-label', LL.t('report.charts.tabs_aria'));
    const reset = document.createElement('button');
    reset.type = 'button';
    reset.className = 'chart-zoom-reset';
    reset.hidden = true;
    reset.title = LL.t('report.charts.reset_title');
    reset.innerHTML = `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg><span>${escapeHtml(LL.t('report.charts.reset'))}</span>`;
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
      // Status square of the domain's analysis (when the report has one), the name, the chart count in mono.
      const verdict = reportState.domainVerdicts[domain];
      if (verdict) {
        const dot = document.createElement('span');
        dot.className = `tab-dot ${LL.verdictClass(verdict)}`;
        dot.setAttribute('aria-hidden', 'true');
        btn.appendChild(dot);
        btn.title = LL.t('report.tabs.verdict_title', { verdict: LL.verdictLabel(verdict) });
      }
      btn.appendChild(document.createTextNode(LL.domainTitleFor(domain, reportDomainTitles)));
      const count = document.createElement('span');
      count.className = 'c';
      count.textContent = String(list.length);
      btn.appendChild(count);
      btn.dataset.target = idBase + di; tabsNav.appendChild(btn);
      const pane = document.createElement('div'); pane.id = idBase + di; pane.className = 'tabpanel' + (di === 0 ? ' active' : '');
      pane.innerHTML = `<div class="chart-pending">${escapeHtml(LL.t('report.charts.pending'))}</div>`;
      paneRenderers[pane.id] = () => renderDomainPane(run, domain, pane, list, di);
      tabsBody.appendChild(pane);
    });
    reportState.paneRenderers = paneRenderers;
    reportState.queryLabels = domains.reduce((all, domain) => all.concat((schema[domain] || []).map((x) => x.query_label)), []);
    if (!domains.length) { root.innerHTML = `<div class="empty-state">${escapeHtml(LL.t('report.charts.no_metrics'))}</div>`; return; }
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
    testTypeSection: testTypeSection,
    // {group: {code: text}} in the current UI language (built on each read).
    get STORED_TEXTS() { return storedTexts(); },
    storedText: storedText,
    reportLanguageOf: reportLanguageOf,
    reportLanguageBadge: reportLanguageBadge,
    heroHtml: heroHtml,
    verdictSummaryHtml: verdictSummaryHtml,
    reportMetaHtml: reportMetaHtml,
    renderHeroKpis: renderHeroKpis,
    slaScopeText: slaScopeText,
    utcOffsetText: utcOffsetText,
    clockLabel: clockLabel,
    dateTimeAt: dateTimeAt,
    timelineHtml: timelineHtml,
    linkSlaToSteps: linkSlaToSteps,
    renderLoadStepTableHtml: renderLoadStepTableHtml,
    renderSlaChecklist: renderSlaChecklist,
    renderStructuredReportHtml: renderStructuredReportHtml,
    renderTrendsTable: renderTrendsTable,
    judgeSectionHtml: judgeSectionHtml,
    systemContextMarkdown: systemContextMarkdown,
    invalidStructuredResponseMarkdown: invalidStructuredResponseMarkdown,
    formatUsageLine: formatUsageLine,
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


