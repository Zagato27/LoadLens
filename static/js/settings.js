// Settings page logic (moved from inline)
(function () {
  const editors = {};
  const originalData = {};
  let currentArea = '';
  let servicesMeta = {};
  let domainList = [];
  let currentPromptService = '';
  let queriesMap = {};
  let metricsConfigMap = {};
  let currentQueriesService = '';
  let currentMetricsService = '';

  function makeEditor(id) {
    // eslint-disable-next-line no-undef
    const ed = ace.edit(id);
    ed.setTheme('ace/theme/twilight');
    if (id.startsWith('ed_prompt_')) {
      ed.session.setMode(null);
    } else {
      ed.session.setMode('ace/mode/json');
    }
    ed.setShowPrintMargin(false);
    ed.session.setUseWrapMode(true);
    ed.setReadOnly(true);
    editors[id] = ed;
    ed.session.on('change', () => {
      const st = document.getElementById('st_' + id.replace('ed_', ''));
      if (id.startsWith('ed_prompt_')) {
        if (st) st.textContent = '';
      } else {
        try { JSON.parse(ed.getValue() || '{}'); if (st) st.textContent = 'OK'; } catch (e) { if (st) st.textContent = 'Ошибка JSON'; }
      }
    });
    return ed;
  }

  function sectionIdToEditorId(section) {
    return {
      'llm': 'ed_llm',
      'confluence': 'ed_confluence',
      'metrics_source': 'ed_metrics_source',
      'lt_metrics_source': 'ed_lt_metrics_source',
      'metrics_config': 'ed_metrics_config',
      'storage.timescale': 'ed_storage_timescale',
      'default_params': 'ed_default_params',
      'queries': 'ed_queries'
    }[section];
  }

  function setEditorJson(edId, obj) {
    const ed = editors[edId];
    if (ed) ed.setValue(JSON.stringify(obj || {}, null, 2), -1);
  }

  function setEditing(section, on) {
    const edId = sectionIdToEditorId(section);
    const ed = editors[edId]; if (!ed) return;
    ed.setReadOnly(!on);
    const saveBtn = document.querySelector(`button[data-save="${section}"]`);
    const editBtn = document.querySelector(`button[data-edit="${section}"]`);
    if (saveBtn) saveBtn.style.display = on ? '' : 'none';
    if (editBtn) editBtn.textContent = on ? 'Завершить' : 'Редактировать';
  }

  function promptEditorId(domain) { return 'ed_prompt_' + domain; }
  function setPromptEditing(domain, on) {
    const edId = promptEditorId(domain);
    const ed = editors[edId]; if (!ed) return;
    ed.setReadOnly(!on);
    const saveBtn = document.querySelector(`button[data-prompt-save="${domain}"]`);
    const editBtn = document.querySelector(`button[data-prompt-edit="${domain}"]`);
    if (saveBtn) saveBtn.style.display = on ? '' : 'none';
    if (editBtn) editBtn.textContent = on ? 'Завершить' : 'Редактировать';
  }

  async function loadPrompts() {
    try {
      const params = new URLSearchParams();
      if (currentArea) params.append('area', currentArea);
      if (currentPromptService) params.append('service', currentPromptService);
      const r = await fetch(`/prompts${params.toString() ? `?${params.toString()}` : ''}`);
      const j = await r.json();
      if (typeof j.active_service === 'string') {
        currentPromptService = j.active_service;
      }
      const serviceSelect = document.getElementById('promptServiceSelect');
      if (serviceSelect) serviceSelect.value = currentPromptService || '';
      const dom = j.domains || {};
      const setText = (edId, txt) => { const ed = editors[edId]; if (ed) ed.setValue(String(txt || ''), -1); };
      setText('ed_prompt_overall', dom.overall || '');
      setText('ed_prompt_database', dom.database || '');
      setText('ed_prompt_kafka', dom.kafka || '');
      setText('ed_prompt_microservices', dom.microservices || '');
      setText('ed_prompt_jvm', dom.jvm || '');
      setText('ed_prompt_hard_resources', dom.hard_resources || '');
      renderServiceMetaPanel(currentPromptService);
    } catch (e) {}
  }

  async function savePrompt(domain) {
    try {
      const ed = editors[promptEditorId(domain)]; if (!ed) return;
      const st = document.getElementById('st_prompt_' + domain); if (st) st.textContent = 'Сохранение…';
      const body = { area: currentArea, domain, text: ed.getValue() || '' };
      if (currentPromptService) body.service = currentPromptService;
      const resp = await fetch('/prompts', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const j = await resp.json();
      if (resp.ok) { if (st) st.textContent = 'Сохранено'; setPromptEditing(domain, false); setTimeout(() => { if (st) st.textContent = ''; }, 1500); }
      else { if (st) st.textContent = j.error || 'Ошибка'; }
    } catch (e) { const st = document.getElementById('st_prompt_' + domain); if (st) st.textContent = 'Ошибка'; }
  }

  async function loadConfig() {
    try {
      const r = await fetch('/config' + (currentArea ? ('?area=' + encodeURIComponent(currentArea)) : ''));
      const j = await r.json();
      const sel = document.getElementById('areaSelect');
      if (Array.isArray(j.areas)) {
        sel.innerHTML = '';
        j.areas.forEach((a) => {
          const opt = document.createElement('option'); opt.value = a; opt.textContent = a; sel.appendChild(opt);
        });
      }
      if (j.active_area) { currentArea = j.active_area; }
      if (currentArea) { const o = [...document.getElementById('areaSelect').options].find((x) => x.value === currentArea); if (o) o.selected = true; }
      setEditorJson('ed_llm', j.llm);
      setEditorJson('ed_metrics_source', j.metrics_source);
      setEditorJson('ed_lt_metrics_source', j.lt_metrics_source);
      setEditorJson('ed_confluence', j.confluence);
      setEditorJson('ed_storage_timescale', (j.storage || {}).timescale || {});
      setEditorJson('ed_default_params', j.default_params);
      servicesMeta = j.services_meta || {};
      domainList = Array.isArray(j.domain_list) ? j.domain_list : [];
      queriesMap = j.queries_map || {};
      if (!Object.prototype.hasOwnProperty.call(queriesMap, '')) {
        queriesMap[''] = j.queries || {};
      }
      metricsConfigMap = j.metrics_config_map || {};
      if (!Object.prototype.hasOwnProperty.call(metricsConfigMap, '')) {
        metricsConfigMap[''] = j.metrics_config || {};
      }
      if (currentPromptService && !servicesMeta[currentPromptService]) {
        currentPromptService = '';
      }
      if (currentQueriesService && !servicesMeta[currentQueriesService]) {
        currentQueriesService = '';
      }
      if (currentMetricsService && !servicesMeta[currentMetricsService]) {
        currentMetricsService = '';
      }
      populatePromptServiceSelect();
      populateServiceSelect('queriesServiceSelect', currentQueriesService, 'Настройки области (по умолчанию)');
      populateServiceSelect('metricsServiceSelect', currentMetricsService, 'Настройки области (по умолчанию)');
      renderServiceMetaPanel(currentPromptService);
      refreshQueriesEditor();
      refreshMetricsConfigEditor();
    } catch (e) {}
  }

  function populatePromptServiceSelect() {
    const sel = document.getElementById('promptServiceSelect');
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = '<option value="">Промпты области (по умолчанию)</option>';
    Object.keys(servicesMeta || {}).forEach((svcId) => {
      const opt = document.createElement('option');
      opt.value = svcId;
      const meta = servicesMeta[svcId] || {};
      opt.textContent = (meta.title && typeof meta.title === 'string') ? meta.title : svcId;
      sel.appendChild(opt);
    });
    const target = (currentPromptService && servicesMeta[currentPromptService]) ? currentPromptService : '';
    sel.value = target;
    currentPromptService = sel.value || '';
  }

  function populateServiceSelect(selectId, currentValue, placeholderText) {
    const sel = document.getElementById(selectId);
    if (!sel) return;
    const options = [`<option value="">${placeholderText}</option>`];
    Object.keys(servicesMeta || {}).forEach((svcId) => {
      const meta = servicesMeta[svcId] || {};
      const title = (meta.title && typeof meta.title === 'string') ? meta.title : svcId;
      options.push(`<option value="${svcId}">${title}</option>`);
    });
    sel.innerHTML = options.join('');
    const hasCurrent = currentValue && servicesMeta[currentValue];
    sel.value = hasCurrent ? currentValue : '';
    if (selectId === 'queriesServiceSelect') {
      currentQueriesService = hasCurrent ? currentValue : '';
    } else if (selectId === 'metricsServiceSelect') {
      currentMetricsService = hasCurrent ? currentValue : '';
    }
  }

  function renderServiceMetaPanel(serviceId) {
    const panel = document.getElementById('serviceMetaPanel');
    if (!panel) return;
    if (!serviceId) {
      panel.style.display = 'none';
      const titleInput = document.getElementById('serviceTitleInput');
      if (titleInput) titleInput.value = '';
      const list = document.getElementById('domainToggleList');
      if (list) list.innerHTML = '';
      return;
    }
    panel.style.display = 'block';
    const meta = servicesMeta[serviceId] || {};
    const disabled = Array.isArray(meta.disabled_domains) ? meta.disabled_domains : [];
    const titleInput = document.getElementById('serviceTitleInput');
    if (titleInput) titleInput.value = meta.title || '';
    const list = document.getElementById('domainToggleList');
    if (list) {
      list.innerHTML = '';
      (domainList || []).forEach((domain) => {
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.dataset.domain = domain;
        checkbox.checked = disabled.indexOf(domain) < 0;
        label.appendChild(checkbox);
        label.append(domain);
        list.appendChild(label);
      });
      if (!domainList || !domainList.length) {
        const empty = document.createElement('span');
        empty.textContent = 'Домены не найдены для текущей конфигурации.';
        empty.style.color = '#888';
        list.appendChild(empty);
      }
    }
  }

  async function persistServiceMeta(payload, statusEl) {
    if (!currentArea) {
      if (statusEl) statusEl.textContent = 'Сначала выберите область';
      return;
    }
    const body = { area: currentArea, service: payload.service, data: payload.data || {} };
    if (statusEl) statusEl.textContent = 'Сохранение…';
    try {
      const resp = await fetch('/service_meta', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const j = await resp.json();
      if (!resp.ok) {
        if (statusEl) statusEl.textContent = j.error || 'Ошибка';
        return;
      }
      servicesMeta[payload.service] = j.meta || {};
      populatePromptServiceSelect();
      if (statusEl) {
        statusEl.textContent = 'Сохранено';
        setTimeout(() => { statusEl.textContent = ''; }, 1500);
      }
      renderServiceMetaPanel(currentPromptService);
    } catch (e) {
      if (statusEl) statusEl.textContent = 'Ошибка';
    }
  }

  async function handleAddService() {
    if (!currentArea) return;
    const rawId = prompt('Введите идентификатор сервиса (как в metrics_config):') || '';
    const svcId = rawId.trim();
    if (!svcId) return;
    const title = (prompt('Отображаемое имя (необязательно):') || '').trim();
    const status = document.getElementById('serviceManageStatus');
    await persistServiceMeta({ service: svcId, data: { title } }, status);
    currentPromptService = svcId;
    renderServiceMetaPanel(currentPromptService);
    await loadPrompts();
  }

  async function saveServiceMeta() {
    if (!currentPromptService) return;
    const titleInput = document.getElementById('serviceTitleInput');
    const list = document.getElementById('domainToggleList');
    const disabledDomains = [];
    if (list) {
      list.querySelectorAll('input[type="checkbox"][data-domain]').forEach((chk) => {
        if (!chk.checked && chk.dataset.domain) disabledDomains.push(chk.dataset.domain);
      });
    }
    const data = {
      title: titleInput ? titleInput.value : '',
      disabled_domains: disabledDomains
    };
    const status = document.getElementById('serviceMetaStatus');
    await persistServiceMeta({ service: currentPromptService, data }, status);
  }

  function refreshQueriesEditor() {
    const payload = queriesMap[currentQueriesService || ''] || {};
    setEditorJson('ed_queries', payload);
  }

  function refreshMetricsConfigEditor() {
    const payload = metricsConfigMap[currentMetricsService || ''] || {};
    setEditorJson('ed_metrics_config', payload);
  }

  async function saveSection(section) {
    try {
      const edId = sectionIdToEditorId(section);
      const ed = editors[edId]; if (!ed) return;
      const stId = 'st_' + section.replace('.', '_');
      const st = document.getElementById(stId); if (st) st.textContent = 'Сохранение…';
      let data = {};
      try { data = JSON.parse(ed.getValue() || '{}'); }
      catch (e) { if (st) st.textContent = 'Ошибка: некорректный JSON'; return; }
      const areaSections = new Set(['llm', 'metrics_source', 'default_params', 'queries', 'metrics_config']);
      areaSections.add('lt_metrics_source');
      const body = { section, data };
      if (areaSections.has(section) && currentArea) { body.area = currentArea; }
      if (section === 'queries' && currentQueriesService) {
        body.service = currentQueriesService;
      }
      if (section === 'metrics_config' && currentMetricsService) {
        body.service = currentMetricsService;
      }
      const resp = await fetch('/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const j = await resp.json();
      if (resp.ok) {
        if (section === 'queries') {
          queriesMap[currentQueriesService || ''] = data;
          refreshQueriesEditor();
        } else if (section === 'metrics_config') {
          metricsConfigMap[currentMetricsService || ''] = data;
          refreshMetricsConfigEditor();
        }
        if (st) st.textContent = 'Сохранено';
        setEditing(section, false);
        setTimeout(() => { if (st) st.textContent = ''; }, 1500);
      }
      else { if (st) st.textContent = j.error || 'Ошибка'; }
    } catch (e) { const st = document.getElementById('st_' + section.replace('.', '_')); if (st) st.textContent = 'Ошибка'; }
  }

  function wireUI() {
    ['ed_llm', 'ed_confluence', 'ed_metrics_source', 'ed_lt_metrics_source', 'ed_metrics_config', 'ed_default_params', 'ed_queries', 'ed_storage_timescale',
      'ed_prompt_overall', 'ed_prompt_database', 'ed_prompt_kafka', 'ed_prompt_microservices', 'ed_prompt_jvm', 'ed_prompt_hard_resources'
    ].forEach(makeEditor);
    document.querySelectorAll('button[data-save]').forEach((b) => { b.addEventListener('click', () => saveSection(b.getAttribute('data-save'))); });
    document.querySelectorAll('button[data-edit]').forEach((b) => {
      b.addEventListener('click', () => {
        const s = b.getAttribute('data-edit');
        const edId = sectionIdToEditorId(s);
        const ed = editors[edId];
        if (!ed) return;
        const isEditing = ed.getReadOnly ? (ed.getReadOnly() === false) : false;
        if (!isEditing) {
          originalData[s] = ed.getValue();
          setEditing(s, true);
        } else {
          if (Object.prototype.hasOwnProperty.call(originalData, s)) {
            ed.setValue(originalData[s] || '', -1);
          }
          const st = document.getElementById('st_' + s.replace('.', '_'));
          if (st) st.textContent = '';
          setEditing(s, false);
        }
      });
    });
    document.querySelectorAll('button[data-prompt-save]').forEach((b) => { b.addEventListener('click', () => savePrompt(b.getAttribute('data-prompt-save'))); });
    document.querySelectorAll('button[data-prompt-edit]').forEach((b) => {
      b.addEventListener('click', () => {
        const d = b.getAttribute('data-prompt-edit');
        const ed = editors[promptEditorId(d)]; if (!ed) return;
        const isEditing = ed.getReadOnly ? (ed.getReadOnly() === false) : false;
        if (!isEditing) { originalData['prompt:' + d] = ed.getValue(); setPromptEditing(d, true); }
        else {
          if (Object.prototype.hasOwnProperty.call(originalData, 'prompt:' + d)) ed.setValue(originalData['prompt:' + d] || '', -1);
          const st = document.getElementById('st_prompt_' + d); if (st) st.textContent = '';
          setPromptEditing(d, false);
        }
      });
    });
    const sel = document.getElementById('areaSelect');
    if (sel) {
      sel.addEventListener('change', async () => {
        currentArea = sel.value || '';
        currentPromptService = '';
        currentQueriesService = '';
        currentMetricsService = '';
        await loadConfig();
        await loadPrompts();
      });
    }
    const addBtn = document.getElementById('addAreaBtn');
    const areaStatus = document.getElementById('areaStatus');
    if (addBtn) {
      addBtn.addEventListener('click', async () => {
        const name = (prompt('Введите идентификатор новой проектной области (латиницей, без пробелов):') || '').trim();
        if (!name) return;
        areaStatus.textContent = 'Создание…';
        try {
          const resp = await fetch('/areas', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
          const j = await resp.json();
          if (!resp.ok) { areaStatus.textContent = j.error || 'Ошибка'; return; }
          currentArea = name; areaStatus.textContent = 'Создано'; setTimeout(() => areaStatus.textContent = '', 1200);
          await loadConfig(); await loadPrompts();
        } catch (e) { areaStatus.textContent = 'Ошибка'; setTimeout(() => areaStatus.textContent = '', 1200); }
      });
    const promptServiceSelect = document.getElementById('promptServiceSelect');
    if (promptServiceSelect) {
      promptServiceSelect.addEventListener('change', async () => {
        currentPromptService = promptServiceSelect.value || '';
        renderServiceMetaPanel(currentPromptService);
        await loadPrompts();
      });
    }
    const queriesServiceSelect = document.getElementById('queriesServiceSelect');
    if (queriesServiceSelect) {
      queriesServiceSelect.addEventListener('change', () => {
        currentQueriesService = queriesServiceSelect.value || '';
        refreshQueriesEditor();
      });
    }
    const metricsServiceSelect = document.getElementById('metricsServiceSelect');
    if (metricsServiceSelect) {
      metricsServiceSelect.addEventListener('change', () => {
        currentMetricsService = metricsServiceSelect.value || '';
        refreshMetricsConfigEditor();
      });
    }
    const addServiceBtn = document.getElementById('addServiceBtn');
    if (addServiceBtn) addServiceBtn.addEventListener('click', handleAddService);
    const addServiceBtnQueries = document.getElementById('addServiceBtnQueries');
    if (addServiceBtnQueries) addServiceBtnQueries.addEventListener('click', handleAddService);
    const addServiceBtnMetrics = document.getElementById('addServiceBtnMetrics');
    if (addServiceBtnMetrics) addServiceBtnMetrics.addEventListener('click', handleAddService);
    const saveServiceMetaBtn = document.getElementById('saveServiceMetaBtn');
    if (saveServiceMetaBtn) saveServiceMetaBtn.addEventListener('click', saveServiceMeta);
    }
  }

  document.addEventListener('DOMContentLoaded', async () => {
    try { await window.LoadLens.initProjectArea(); } catch (e) {}
    wireUI();
    await loadConfig();
    await loadPrompts();
  });
})();


