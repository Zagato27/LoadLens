// Settings → «Проекты»: overview of project areas, card edits, copies, service moves,
// resetting overrides and deletion with or without reports.
(function () {
  const LL = window.LoadLens;
  const ui = LL.ui;
  const esc = ui.escapeHtml;
  const $ = (id) => document.getElementById(id);
  const SECTION_LABELS = {
    llm: 'LLM', domain_sources: 'Источники доменов', logs_source: 'Логи приложений',
    default_params: 'Параметры сбора', queries: 'Запросы', sla: 'SLA', system_context: 'Контекст системы', prompts: 'Промпты'
  };
  const STATUS_VIEW = { inherited: ['na', 'общие'], same: ['na', 'копия общих'], own: ['warn', 'свои'] };
  let overview = { projects: [], active_project: '', reports_error: null };

  const projectUrl = (id) => '/projects/' + encodeURIComponent(id);
  const sectionLabel = (key) => SECTION_LABELS[key] || key;
  const domainLabel = (key) => (key === 'overall' ? 'Итог' : LL.domainTitle(key));

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
    max_performance_query: 'макс. RPS', mean_latency_query: 'среднее время',
    service_cpu_query: 'CPU сервисов', cpu_query: 'CPU узлов'
  };
  // Settings links of the checklist: "<section>:<element id>" (activateFromHash in settings.js
  // opens the section, scrolls to the element and highlights it).
  const SLA_FIELD_LINKS = {
    max_performance_query: 'sla:sla-query-target_rps', mean_latency_query: 'sla:sla-peak-mean_latency_query',
    service_cpu_query: 'sla:sla-peak-service_cpu_query', cpu_query: 'sla:sla-query-max_cpu_pct'
  };
  const CHECK_ICONS = {
    ok: '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.5 2.5L16 9.5"/></svg>',
    todo: '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/></svg>'
  };
  function packsChip(packs) {
    if (!Array.isArray(packs) || !packs.length) return '';
    const outdated = packs.filter((p) => p.outdated).length;
    const names = packs.map((p) => p.title).join(', ');
    const text = outdated ? `Пакетов запросов: ${packs.length}, есть обновления: ${outdated}` : `Пакетов запросов: ${packs.length}`;
    return `<span class="pill ${outdated ? 'warn' : 'na'}" title="${esc(names)}">${esc(text)}</span>`;
  }
  function featureItem(key, missing, okText, title, fallbackLink) {
    const fields = Array.isArray(missing) ? missing : [];
    return {
      key,
      ok: !fields.length,
      text: fields.length ? `${title}: не заданы ${fields.map((f) => SLA_FIELD_TITLES[f] || f).join(', ')}` : okText,
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
      { key: 'target_rps', ok: hasTarget, text: hasTarget ? `Целевой RPS: ${r.target_rps}` : 'Целевой RPS не задан', link: 'sla:sla-val-target_rps' },
      { key: 'performance_query', ok: !!r.performance_query, text: r.performance_query ? 'Запрос производительности выбран' : 'Запрос производительности не выбран', link: 'sla:sla-query-target_rps' },
      { key: 'system_context', ok: !!r.has_system_context, text: r.has_system_context ? 'Контекст системы заполнен' : 'Контекст системы пуст', link: 'context:contextForm' },
      { key: 'queries', ok: total > 0, text: total > 0 ? `Запросов: ${total}, доменов: ${Object.keys(counts).length}` : 'Запросов нет', link: 'queries:addFromPackBtn' },
      featureItem('forecast', missing.forecast, 'Прогноз: ссылки SLA заданы', 'Прогноз', 'sla:sla-peak-mean_latency_query'),
      featureItem('resource_plan', missing.resource_plan, 'План ресурсов: ссылки SLA заданы', 'План ресурсов', 'sla:sla-peak-service_cpu_query')
    ];
  }
  function readinessList(p) {
    const items = readinessItems(p.readiness);
    const done = items.filter((item) => item.ok).length;
    const pct = Math.round((done / items.length) * 100);
    const label = `Настроено ${done} из ${items.length}`;
    const rows = items.map((item) => `<li class="readiness-item ${item.ok ? 'is-ok' : 'is-todo'}">
        <a href="#${esc(item.link)}" data-action="goto" data-project="${esc(p.id)}" data-link="${esc(item.link)}">${item.ok ? CHECK_ICONS.ok : CHECK_ICONS.todo}<span>${esc(item.text)}</span></a></li>`).join('');
    return `<div class="readiness${done === items.length ? ' is-complete' : ''}">
      <div class="readiness-head">
        <span class="readiness-count">${esc(label)}</span>
        <span class="readiness-bar" role="progressbar" aria-label="${esc(label)}" aria-valuemin="0" aria-valuemax="${items.length}" aria-valuenow="${done}"><span style="width:${pct}%"></span></span>
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
    if (!p.last_report_at) return 'нет';
    return `${esc(LL.formatDateTime(p.last_report_at))} <span class="pill ${LL.verdictClass(p.last_verdict)}">${esc(p.last_verdict || '—')}</span>`;
  }
  function servicesTable(p, attrs) {
    if (!p.services.length) return '<p class="dim">Сервисов нет.</p>';
    const rows = p.services.map((s) => {
      const own = s.own_sections.map((section) => `<button type="button" class="btn btn-sm" title="Вернуть к настройкам проекта" ${attrs('reset-service', ` data-service="${esc(s.id)}" data-section="${esc(section)}"`)}>${esc(sectionLabel(section))} ×</button>`).join(' ') || '<span class="dim">нет</span>';
      return `<tr><td><strong>${esc(s.title)}</strong>${s.title !== s.id ? `<div class="dim">${esc(s.id)}</div>` : ''}</td>
        <td class="num">${s.reports === null ? '—' : s.reports}</td>
        <td class="nowrap">${s.last_report_at ? esc(LL.formatDateTime(s.last_report_at)) : '—'}</td>
        <td>${own}</td>
        <td class="actions"><button type="button" class="btn btn-sm" ${attrs('move', ` data-service="${esc(s.id)}"`)}>Перенести…</button></td></tr>`;
    }).join('');
    return `<div class="table-wrap"><table class="data-table"><thead><tr><th>Сервис</th><th class="num">Отчётов</th><th>Последний отчёт</th><th>Свои настройки</th><th><span class="visually-hidden">Действия</span></th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }
  function overridesBlock(p, attrs) {
    const rows = p.overrides.map((o) => {
      const [tone, label] = STATUS_VIEW[o.status] || ['na', o.status];
      const domains = o.own_domains && o.own_domains.length ? `<div class="dim">${esc(o.own_domains.map(domainLabel).join(', '))}</div>` : '';
      const action = o.status === 'inherited' ? '' : `<button type="button" class="btn btn-sm" ${attrs('reset-area', ` data-section="${esc(o.section)}"`)}>Вернуть к общим</button>`;
      return `<tr><td>${esc(sectionLabel(o.section))}${domains}</td><td><span class="pill ${tone}">${esc(label)}</span></td><td class="actions">${action}</td></tr>`;
    }).join('');
    const own = p.overrides.filter((o) => o.status === 'own').length;
    return `<details class="report-collapsible"><summary><span>Что переопределено</span><small class="report-collapsible-subtitle">своих разделов: ${own}</small></summary><div class="report-collapsible-body"><table class="data-table"><tbody>${rows}</tbody></table></div></details>`;
  }
  function projectCard(p) {
    const attrs = (action, extra) => `data-action="${action}" data-project="${esc(p.id)}"${extra || ''}`;
    const active = p.id === overview.active_project;
    return `<article class="card project-card">
      <div class="project-head">
        <div>
          <div class="project-title">${esc(p.title)} ${active ? '<span class="pill ok">Текущий</span>' : ''}</div>
          ${p.title !== p.id ? `<div class="dim">${esc(p.id)}</div>` : ''}
          ${p.description ? `<p class="help">${esc(p.description)}</p>` : ''}
        </div>
        <div class="project-actions">
          ${active ? '' : `<button type="button" class="btn btn-sm" ${attrs('select')}>Выбрать</button>`}
          <button type="button" class="btn btn-sm" ${attrs('edit')}>Изменить</button>
          <button type="button" class="btn btn-sm" ${attrs('copy')}>Копировать</button>
          <button type="button" class="btn btn-sm btn-danger" ${attrs('delete')}>Удалить…</button>
        </div>
      </div>
      <div class="dim">Сервисов: ${p.services.length} · Отчётов: ${p.reports === null ? '—' : p.reports} · Последний: ${lastReport(p)}</div>
      ${readinessList(p)}
      ${servicesTable(p, attrs)}
      ${overridesBlock(p, attrs)}
    </article>`;
  }
  function renderProjects() {
    const banner = overview.reports_error ? `<div class="report-note">${esc(overview.reports_error)}</div>` : '';
    const cards = overview.projects.length ? overview.projects.map(projectCard).join('') : '<div class="card">Проектов пока нет. Создайте первый.</div>';
    $('projectsList').innerHTML = banner + cards;
  }
  async function loadProjects() {
    const list = $('projectsList');
    list.innerHTML = ui.skeleton(3);
    try {
      overview = await api('GET', '/projects');
      renderProjects();
    } catch (e) {
      list.innerHTML = `<div class="report-note">Не удалось загрузить проекты: ${esc(e.message)}</div>`;
    }
  }

  function openProjectForm(mode, project, copyFrom) {
    const editing = mode === 'edit';
    const sources = overview.projects.map((x) => `<option value="${esc(x.id)}"${x.id === copyFrom ? ' selected' : ''}>${esc(x.title)}</option>`).join('');
    const currentTitle = editing && project.title !== project.id ? project.title : '';
    const bodyHtml = [
      editing ? `<p class="dim">Идентификатор: ${esc(project.id)}</p>` : '<label>Идентификатор<input id="pfId" type="text" maxlength="40" placeholder="без пробелов, например NSI" /></label>',
      `<label>Отображаемое имя<input id="pfTitle" type="text" maxlength="80" value="${esc(currentTitle)}" /></label>`,
      `<label>Описание<textarea id="pfDescription" rows="3" maxlength="500">${esc(editing ? project.description : '')}</textarea></label>`,
      editing ? '' : `<label>Скопировать настройки из<select id="pfCopyFrom"><option value="">Не копировать</option>${sources}</select></label><p class="dim">Копируются подключения, запросы, SLA, промпты и контекст системы. Сервисы и отчёты не копируются.</p>`,
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
      title: editing ? `Проект «${project.title}»` : 'Новый проект',
      bodyHtml,
      actions: [{ label: 'Отмена' }, { label: editing ? 'Сохранить' : 'Создать', primary: true, onClick: submit }]
    });
  }
  async function openDeleteDialog(p) {
    const reports = p.reports === null ? 'неизвестно, база недоступна' : String(p.reports);
    const services = p.services.map((s) => s.id).join(', ') || 'нет';
    const choice = await ui.dialog({
      title: `Удалить проект «${p.title}»`,
      bodyHtml: `<p>Сервисы: ${esc(services)}. Отчётов: ${esc(reports)}.</p><p>«Только настройки» удаляет проект и настройки его сервисов. Отчёты и метрики остаются и будут видны в «Все области».</p>`,
      actions: [
        { label: 'Отмена' },
        { label: 'Удалить только настройки', danger: true, value: 'settings' },
        { label: 'Удалить вместе с отчётами', danger: true, value: 'data' }
      ]
    });
    if (!choice) return;
    if (choice === 'data') {
      const sure = await ui.confirm({ title: 'Удалить отчёты', message: `Будут безвозвратно удалены отчёты (${reports}) и метрики сервисов: ${services}.`, confirmText: 'Удалить навсегда', danger: true });
      if (!sure) return;
    }
    await api('DELETE', `${projectUrl(p.id)}?with_data=${choice === 'data' ? '1' : '0'}`);
    if (overview.active_project === p.id) await switchProject('');
    location.reload();
  }
  async function openMoveDialog(p, serviceId) {
    const targets = overview.projects.filter((x) => x.id !== p.id);
    if (!targets.length) { ui.toast('Нет другого проекта для переноса', { tone: 'warn' }); return; }
    const service = p.services.find((s) => s.id === serviceId);
    const reports = service && service.reports !== null ? service.reports : '—';
    const options = targets.map((x) => `<option value="${esc(x.id)}">${esc(x.title)}</option>`).join('');
    const target = await ui.dialog({
      title: `Перенести сервис «${serviceId}»`,
      bodyHtml: `<label>В проект<select id="moveTarget">${options}</select></label><p class="dim">Настройки сервиса и его отчёты (${esc(reports)}) перейдут в выбранный проект.</p>`,
      actions: [{ label: 'Отмена' }, { label: 'Перенести', primary: true, onClick: (close, body) => close(body.querySelector('#moveTarget').value) }]
    });
    if (!target) return;
    await api('POST', `${projectUrl(p.id)}/services/${encodeURIComponent(serviceId)}/move`, { target });
    ui.toast(`Сервис «${serviceId}» перенесён`, { tone: 'ok' });
    location.reload();
  }
  async function resetArea(p, section) {
    const ok = await ui.confirm({ title: 'Вернуть к общим', message: `Раздел «${sectionLabel(section)}» проекта «${p.title}» станет общим. Свои значения проекта в этом разделе удалятся.`, confirmText: 'Вернуть', danger: true });
    if (!ok) return;
    await api('DELETE', `${projectUrl(p.id)}/overrides/${encodeURIComponent(section)}`);
    location.reload();
  }
  async function resetService(p, serviceId, section) {
    const ok = await ui.confirm({ title: 'Вернуть к настройкам проекта', message: `Раздел «${sectionLabel(section)}» сервиса «${serviceId}» будет браться из проекта «${p.title}».`, confirmText: 'Вернуть', danger: true });
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

  LL.projects = { _internals: { readinessItems, readinessList } };

  document.addEventListener('DOMContentLoaded', () => {
    const list = $('projectsList');
    if (!list) return;
    list.addEventListener('click', onListClick);
    $('projectCreateBtn').addEventListener('click', () => openProjectForm('create', null, ''));
    loadProjects();
  });
})();
