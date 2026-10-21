// SPDX-License-Identifier: Apache-2.0
// Settings → Projects: overview of project areas, card edits, copies, service moves,
// resetting overrides and deletion with or without reports.
(function () {
  const LL = window.LoadLens;
  const ui = LL.ui;
  const esc = ui.escapeHtml;
  const $ = (id) => document.getElementById(id);
  // Labels are read when shown: they follow the UI language.
  const SECTION_LABELS = {
    llm: () => LL.t('projects.section.llm'),
    domain_sources: () => LL.t('projects.section.domain_sources'),
    logs_source: () => LL.t('projects.section.logs_source'),
    default_params: () => LL.t('projects.section.default_params'),
    queries: () => LL.t('projects.section.queries'),
    sla: () => LL.t('projects.section.sla'),
    system_context: () => LL.t('projects.section.system_context'),
    prompts: () => LL.t('projects.section.prompts')
  };
  const STATUS_VIEW = {
    inherited: () => ['na', LL.t('projects.status.inherited')],
    same: () => ['na', LL.t('projects.status.same')],
    own: () => ['warn', LL.t('projects.status.own')]
  };
  let overview = { projects: [], active_project: '', reports_error: null };

  const projectUrl = (id) => '/projects/' + encodeURIComponent(id);
  const sectionLabel = (key) => (Object.prototype.hasOwnProperty.call(SECTION_LABELS, key) ? SECTION_LABELS[key]() : key);
  const domainLabel = (key) => (key === 'overall' ? LL.t('common.domain.final') : LL.domainTitle(key));

  async function api(method, url, body) {
    const init = { method };
    if (body !== undefined) { init.headers = { 'Content-Type': 'application/json' }; init.body = JSON.stringify(body); }
    const resp = await fetch(url, init);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    return data;
  }
  const switchProject = (id) => api('POST', '/project_area', { project_area: id || '' });

  const SLA_FIELD_TITLES = {
    max_performance_query: () => LL.t('projects.sla_field.max_performance_query'),
    mean_latency_query: () => LL.t('projects.sla_field.mean_latency_query'),
    service_cpu_query: () => LL.t('projects.sla_field.service_cpu_query'),
    cpu_query: () => LL.t('projects.sla_field.cpu_query')
  };
  const slaFieldTitle = (field) => (Object.prototype.hasOwnProperty.call(SLA_FIELD_TITLES, field) ? SLA_FIELD_TITLES[field]() : field);
  // Settings links of the checklist: "<section>:<element id>" (activateFromHash in settings.js
  // opens the section, scrolls to the element and highlights it).
  const SLA_FIELD_LINKS = {
    max_performance_query: 'sla:sla-query-target_rps', mean_latency_query: 'sla:sla-peak-mean_latency_query',
    service_cpu_query: 'sla:sla-peak-service_cpu_query', cpu_query: 'sla:sla-query-max_cpu_pct'
  };
  // Status squares of the checklist: a point done is ok, a point to do is a warning (the text says which).
  const CHECK_MARKS = {
    ok: '<span class="sq ok" aria-hidden="true"></span>',
    todo: '<span class="sq warn" aria-hidden="true"></span>'
  };
  function packsChip(packs) {
    if (!Array.isArray(packs) || !packs.length) return '';
    const outdated = packs.filter((p) => p.outdated).length;
    const names = packs.map((p) => p.title).join(', ');
    const text = outdated
      ? LL.t('projects.packs.with_updates', { count: packs.length, outdated })
      : LL.t('projects.packs.count', { count: packs.length });
    return `<span class="pill ${outdated ? 'warn' : 'na'}" title="${esc(names)}">${esc(text)}</span>`;
  }
  function featureItem(key, missing, okText, title, fallbackLink) {
    const fields = Array.isArray(missing) ? missing : [];
    return {
      key,
      ok: !fields.length,
      text: fields.length ? LL.t('projects.readiness.feature_missing', { feature: title, fields: fields.map(slaFieldTitle).join(', ') }) : okText,
      link: (fields.length && SLA_FIELD_LINKS[fields[0]]) || fallbackLink
    };
  }
  // The six readiness points of a project: {key, ok, text, link}.
  function readinessItems(readiness) {
    const r = readiness || {};
    const counts = r.query_counts || {};
    const total = Object.values(counts).reduce((sum, n) => sum + (Number(n) || 0), 0);
    const missing = r.missing_sla || {};
    const hasTarget = r.target_rps !== null && r.target_rps !== undefined;
    return [
      {
        key: 'target_rps',
        ok: hasTarget,
        text: hasTarget ? LL.t('projects.readiness.target_rps', { value: r.target_rps }) : LL.t('projects.readiness.target_rps_missing'),
        link: 'sla:sla-val-target_rps'
      },
      {
        key: 'performance_query',
        ok: !!r.performance_query,
        text: LL.t(r.performance_query ? 'projects.readiness.performance_query_ok' : 'projects.readiness.performance_query_missing'),
        link: 'sla:sla-query-target_rps'
      },
      {
        key: 'system_context',
        ok: !!r.has_system_context,
        text: LL.t(r.has_system_context ? 'projects.readiness.system_context_ok' : 'projects.readiness.system_context_missing'),
        link: 'context:contextForm'
      },
      {
        key: 'queries',
        ok: total > 0,
        text: total > 0 ? LL.t('projects.readiness.queries', { queries: total, domains: Object.keys(counts).length }) : LL.t('projects.readiness.queries_missing'),
        link: 'queries:addFromPackBtn'
      },
      featureItem('forecast', missing.forecast, LL.t('projects.readiness.forecast_ok'), LL.t('projects.readiness.forecast'), 'sla:sla-peak-mean_latency_query'),
      featureItem('resource_plan', missing.resource_plan, LL.t('projects.readiness.resource_plan_ok'), LL.t('projects.readiness.resource_plan'), 'sla:sla-peak-service_cpu_query')
    ];
  }
  function readinessList(p) {
    const items = readinessItems(p.readiness);
    const done = items.filter((item) => item.ok).length;
    const label = LL.t('projects.readiness.progress', { done, total: items.length });
    const rows = items.map((item) => `<li class="readiness-item ${item.ok ? 'is-ok' : 'is-todo'}">
        <a href="#${esc(item.link)}" data-action="goto" data-project="${esc(p.id)}" data-link="${esc(item.link)}">${item.ok ? CHECK_MARKS.ok : CHECK_MARKS.todo}<span>${esc(item.text)}</span></a></li>`).join('');
    // One square per point, the points done first: a progress meter without a bar.
    const squares = items.map((item, i) => `<i${i < done ? ' class="ok"' : ''}></i>`).join('');
    return `<div class="readiness${done === items.length ? ' is-complete' : ''}">
      <div class="readiness-head">
        <span class="readiness-count">${esc(label)}</span>
        <span class="readiness-squares" role="progressbar" aria-label="${esc(label)}" aria-valuemin="0" aria-valuemax="${items.length}" aria-valuenow="${done}">${squares}</span>
        ${packsChip((p.readiness || {}).query_packs)}
      </div>
      <ul class="readiness-list">${rows}</ul>
    </div>`;
  }
  // Opens a settings field of the checklist; another project is selected first (settings edit the active one).
  async function gotoSetting(p, link) {
    if (!link) return;
    if (p.id !== overview.active_project) {
      await switchProject(p.id);
      try { history.replaceState(null, '', `#${link}`); } catch (e) { /* history API unavailable */ }
      location.reload();
      return;
    }
    if (location.hash === `#${link}`) window.dispatchEvent(new HashChangeEvent('hashchange'));
    else location.hash = link;
  }
  function lastReport(p) {
    if (!p.last_report_at) return esc(LL.t('projects.card.last_none'));
    return `${esc(LL.formatDateTime(p.last_report_at))} <span class="pill ${LL.verdictClass(p.last_verdict)}">${esc(p.last_verdict ? LL.verdictLabel(p.last_verdict) : '—')}</span>`;
  }
  function servicesTable(p, attrs) {
    if (!p.services.length) return `<p class="dim">${esc(LL.t('projects.services.empty'))}</p>`;
    const rows = p.services.map((s) => {
      const own = s.own_sections.map((section) => `<button type="button" class="btn btn-sm" title="${esc(LL.t('projects.services.reset_title'))}" ${attrs('reset-service', ` data-service="${esc(s.id)}" data-section="${esc(section)}"`)}>${esc(sectionLabel(section))} ×</button>`).join(' ')
        || `<span class="dim">${esc(LL.t('projects.services.own_none'))}</span>`;
      return `<tr><td><strong>${esc(s.title)}</strong>${s.title !== s.id ? `<div class="dim mono">${esc(s.id)}</div>` : ''}</td>
        <td class="num">${s.reports === null ? '—' : s.reports}</td>
        <td class="nowrap m">${s.last_report_at ? esc(LL.formatDateTime(s.last_report_at)) : '—'}</td>
        <td>${own}</td>
        <td class="actions"><button type="button" class="btn btn-sm" ${attrs('move', ` data-service="${esc(s.id)}"`)}>${esc(LL.t('projects.services.move'))}</button></td></tr>`;
    }).join('');
    const head = `<th>${esc(LL.t('projects.services.col_service'))}</th><th class="num">${esc(LL.t('projects.services.col_reports'))}</th>`
      + `<th>${esc(LL.t('projects.services.col_last_report'))}</th><th>${esc(LL.t('projects.services.col_own'))}</th>`
      + `<th><span class="visually-hidden">${esc(LL.t('projects.services.col_actions'))}</span></th>`;
    return `<div class="table-wrap"><table class="data-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
  }
  function overridesBlock(p, attrs) {
    const rows = p.overrides.map((o) => {
      const [tone, label] = Object.prototype.hasOwnProperty.call(STATUS_VIEW, o.status) ? STATUS_VIEW[o.status]() : ['na', o.status];
      const domains = o.own_domains && o.own_domains.length ? `<div class="dim">${esc(o.own_domains.map(domainLabel).join(', '))}</div>` : '';
      const action = o.status === 'inherited' ? '' : `<button type="button" class="btn btn-sm" ${attrs('reset-area', ` data-section="${esc(o.section)}"`)}>${esc(LL.t('projects.overrides.reset'))}</button>`;
      return `<tr><td>${esc(sectionLabel(o.section))}${domains}</td><td><span class="pill ${tone}">${esc(label)}</span></td><td class="actions">${action}</td></tr>`;
    }).join('');
    const own = p.overrides.filter((o) => o.status === 'own').length;
    const summary = `<span>${esc(LL.t('projects.overrides.title'))}</span><small class="report-collapsible-subtitle">${esc(LL.t('projects.overrides.own_count', { count: own }))}</small>`;
    return `<details class="report-collapsible"><summary>${summary}</summary><div class="report-collapsible-body"><table class="data-table"><tbody>${rows}</tbody></table></div></details>`;
  }
  function projectCard(p) {
    const attrs = (action, extra) => `data-action="${action}" data-project="${esc(p.id)}"${extra || ''}`;
    const active = p.id === overview.active_project;
    return `<article class="card project-card">
      <div class="project-head">
        <div>
          <div class="project-title">${esc(p.title)} ${active ? `<span class="pill ok">${esc(LL.t('projects.card.current'))}</span>` : ''}</div>
          ${p.title !== p.id ? `<div class="dim project-id">${esc(p.id)}</div>` : ''}
          ${p.description ? `<p class="help">${esc(p.description)}</p>` : ''}
        </div>
        <div class="project-actions">
          ${active ? '' : `<button type="button" class="btn btn-sm" ${attrs('select')}>${esc(LL.t('projects.card.select'))}</button>`}
          <button type="button" class="btn btn-sm" ${attrs('edit')}>${esc(LL.t('projects.card.edit'))}</button>
          <button type="button" class="btn btn-sm" ${attrs('copy')}>${esc(LL.t('projects.card.copy'))}</button>
          <button type="button" class="btn btn-sm btn-danger" ${attrs('delete')}>${esc(LL.t('projects.card.delete'))}</button>
        </div>
      </div>
      <div class="dim">${LL.tHtml('projects.card.summary', { services: p.services.length, reports: p.reports === null ? '—' : p.reports, last: LL.html(lastReport(p)) })}</div>
      ${readinessList(p)}
      ${servicesTable(p, attrs)}
      ${overridesBlock(p, attrs)}
    </article>`;
  }
  function renderProjects() {
    const banner = overview.reports_error ? `<div class="report-note">${esc(overview.reports_error)}</div>` : '';
    const cards = overview.projects.length ? overview.projects.map(projectCard).join('') : `<div class="card">${esc(LL.t('projects.list.empty'))}</div>`;
    $('projectsList').innerHTML = banner + cards;
  }
  async function loadProjects() {
    const list = $('projectsList');
    list.innerHTML = ui.skeleton(3);
    try {
      overview = await api('GET', '/projects');
      renderProjects();
    } catch (e) {
      list.innerHTML = `<div class="report-note">${LL.tHtml('projects.list.load_failed', { error: e.message })}</div>`;
    }
  }

  function openProjectForm(mode, project, copyFrom) {
    const editing = mode === 'edit';
    const sources = overview.projects.map((x) => `<option value="${esc(x.id)}"${x.id === copyFrom ? ' selected' : ''}>${esc(x.title)}</option>`).join('');
    const currentTitle = editing && project.title !== project.id ? project.title : '';
    const bodyHtml = [
      editing
        ? `<p class="dim">${LL.tHtml('projects.form.id_value', { id: project.id })}</p>`
        : `<label>${esc(LL.t('projects.form.id'))}<input id="pfId" type="text" maxlength="40" placeholder="${esc(LL.t('projects.form.id_placeholder'))}" /></label>`,
      `<label>${esc(LL.t('projects.form.title'))}<input id="pfTitle" type="text" maxlength="80" value="${esc(currentTitle)}" /></label>`,
      `<label>${esc(LL.t('projects.form.description'))}<textarea id="pfDescription" rows="3" maxlength="500">${esc(editing ? project.description : '')}</textarea></label>`,
      editing ? '' : `<label>${esc(LL.t('projects.form.copy_from'))}<select id="pfCopyFrom"><option value="">${esc(LL.t('projects.form.no_copy'))}</option>${sources}</select></label>`
        + `<p class="dim">${esc(LL.t('projects.form.copy_hint'))}</p>`,
      '<div class="dialog-error" id="pfError"></div>'
    ].join('');
    const submit = async (close, body) => {
      const value = (selector) => body.querySelector(selector).value.trim();
      try {
        if (editing) {
          await api('PATCH', projectUrl(project.id), { title: value('#pfTitle'), description: value('#pfDescription') });
        } else {
          const id = value('#pfId');
          await api('POST', '/projects', { id, title: value('#pfTitle'), description: value('#pfDescription'), copy_from: value('#pfCopyFrom') });
          await switchProject(id);
        }
        close(true);
        location.reload();
      } catch (e) {
        body.querySelector('#pfError').textContent = e.message;
      }
    };
    return ui.dialog({
      title: editing ? LL.t('projects.form.edit_title', { title: project.title }) : LL.t('projects.form.create_title'),
      bodyHtml,
      actions: [{ label: LL.t('common.action.cancel') }, { label: LL.t(editing ? 'common.action.save' : 'projects.form.create'), primary: true, onClick: submit }]
    });
  }
  async function openDeleteDialog(p) {
    const reports = p.reports === null ? LL.t('projects.delete.reports_unknown') : String(p.reports);
    const services = p.services.map((s) => s.id).join(', ') || LL.t('projects.delete.services_none');
    const choice = await ui.dialog({
      title: LL.t('projects.delete.title', { title: p.title }),
      bodyHtml: `<p>${LL.tHtml('projects.delete.summary', { services, reports })}</p>`
        + `<p>${LL.tHtml('projects.delete.settings_only_hint', { all_areas: LL.t('ui.project_area.all') })}</p>`,
      actions: [
        { label: LL.t('common.action.cancel') },
        { label: LL.t('projects.delete.settings_only'), danger: true, value: 'settings' },
        { label: LL.t('projects.delete.with_reports'), danger: true, value: 'data' }
      ]
    });
    if (!choice) return;
    if (choice === 'data') {
      const sure = await ui.confirm({
        title: LL.t('projects.delete.reports_title'),
        message: LL.t('projects.delete.reports_message', { reports, services }),
        confirmText: LL.t('projects.delete.forever'),
        danger: true
      });
      if (!sure) return;
    }
    await api('DELETE', `${projectUrl(p.id)}?with_data=${choice === 'data' ? '1' : '0'}`);
    if (overview.active_project === p.id) await switchProject('');
    location.reload();
  }
  async function openMoveDialog(p, serviceId) {
    const targets = overview.projects.filter((x) => x.id !== p.id);
    if (!targets.length) { ui.toast(LL.t('projects.move.no_target'), { tone: 'warn' }); return; }
    const service = p.services.find((s) => s.id === serviceId);
    const reports = service && service.reports !== null ? service.reports : '—';
    const options = targets.map((x) => `<option value="${esc(x.id)}">${esc(x.title)}</option>`).join('');
    const target = await ui.dialog({
      title: LL.t('projects.move.title', { service: serviceId }),
      bodyHtml: `<label>${esc(LL.t('projects.move.target'))}<select id="moveTarget">${options}</select></label>`
        + `<p class="dim">${LL.tHtml('projects.move.hint', { reports })}</p>`,
      actions: [
        { label: LL.t('common.action.cancel') },
        { label: LL.t('projects.move.confirm'), primary: true, onClick: (close, body) => close(body.querySelector('#moveTarget').value) }
      ]
    });
    if (!target) return;
    await api('POST', `${projectUrl(p.id)}/services/${encodeURIComponent(serviceId)}/move`, { target });
    ui.toast(LL.t('projects.move.done', { service: serviceId }), { tone: 'ok' });
    location.reload();
  }
  async function resetArea(p, section) {
    const ok = await ui.confirm({
      title: LL.t('projects.overrides.reset'),
      message: LL.t('projects.reset_area.message', { section: sectionLabel(section), project: p.title }),
      confirmText: LL.t('projects.reset.confirm'),
      danger: true
    });
    if (!ok) return;
    await api('DELETE', `${projectUrl(p.id)}/overrides/${encodeURIComponent(section)}`);
    location.reload();
  }
  async function resetService(p, serviceId, section) {
    const ok = await ui.confirm({
      title: LL.t('projects.services.reset_title'),
      message: LL.t('projects.reset_service.message', { section: sectionLabel(section), service: serviceId, project: p.title }),
      confirmText: LL.t('projects.reset.confirm'),
      danger: true
    });
    if (!ok) return;
    await api('DELETE', `${projectUrl(p.id)}/services/${encodeURIComponent(serviceId)}/overrides/${encodeURIComponent(section)}`);
    location.reload();
  }

  const ACTIONS = {
    select: async (p) => { await switchProject(p.id); location.reload(); },
    goto: (p, btn) => gotoSetting(p, btn.dataset.link),
    edit: (p) => openProjectForm('edit', p, ''),
    copy: (p) => openProjectForm('create', null, p.id),
    delete: (p) => openDeleteDialog(p),
    move: (p, btn) => openMoveDialog(p, btn.dataset.service),
    'reset-area': (p, btn) => resetArea(p, btn.dataset.section),
    'reset-service': (p, btn) => resetService(p, btn.dataset.service, btn.dataset.section)
  };
  async function onListClick(event) {
    const btn = event.target.closest('[data-action]');
    if (!btn) return;
    const project = overview.projects.find((p) => p.id === btn.dataset.project);
    const handler = ACTIONS[btn.dataset.action];
    if (!project || !handler) return;
    if (btn.tagName === 'A') event.preventDefault();
    try { await handler(project, btn); } catch (e) { ui.toast(e.message, { tone: 'error' }); }
  }

  LL.projects = { _internals: { readinessItems, readinessList, packsChip, servicesTable, overridesBlock, projectCard } };

  document.addEventListener('DOMContentLoaded', () => {
    const list = $('projectsList');
    if (!list) return;
    list.addEventListener('click', onListClick);
    $('projectCreateBtn').addEventListener('click', () => openProjectForm('create', null, ''));
    loadProjects();
  });
})();
