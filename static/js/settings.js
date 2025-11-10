// Settings page logic (moved from inline)
(function () {
  const editors = {};
  const originalData = {};
  let currentArea = '';

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
      const r = await fetch('/prompts' + (currentArea ? ('?area=' + encodeURIComponent(currentArea)) : ''));
      const j = await r.json();
      const dom = j.domains || {};
      const setText = (edId, txt) => { const ed = editors[edId]; if (ed) ed.setValue(String(txt || ''), -1); };
      setText('ed_prompt_overall', dom.overall || '');
      setText('ed_prompt_judge', dom.judge || '');
      setText('ed_prompt_critic', dom.critic || '');
      setText('ed_prompt_database', dom.database || '');
      setText('ed_prompt_kafka', dom.kafka || '');
      setText('ed_prompt_microservices', dom.microservices || '');
      setText('ed_prompt_jvm', dom.jvm || '');
      setText('ed_prompt_hard_resources', dom.hard_resources || '');
    } catch (e) {}
  }

  async function savePrompt(domain) {
    try {
      const ed = editors[promptEditorId(domain)]; if (!ed) return;
      const st = document.getElementById('st_prompt_' + domain); if (st) st.textContent = 'Сохранение…';
      const body = { area: currentArea, domain, text: ed.getValue() || '' };
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
      const set = (edId, obj) => { const ed = editors[edId]; if (ed) ed.setValue(JSON.stringify(obj || {}, null, 2), -1); };
      set('ed_llm', j.llm);
      set('ed_metrics_source', j.metrics_source);
      set('ed_lt_metrics_source', j.lt_metrics_source);
      set('ed_confluence', j.confluence);
      set('ed_storage_timescale', (j.storage || {}).timescale || {});
      set('ed_default_params', j.default_params);
      set('ed_queries', j.queries);
      set('ed_metrics_config', j.metrics_config);
    } catch (e) {}
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
      const resp = await fetch('/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const j = await resp.json();
      if (resp.ok) { if (st) st.textContent = 'Сохранено'; setEditing(section, false); setTimeout(() => { if (st) st.textContent = ''; }, 1500); }
      else { if (st) st.textContent = j.error || 'Ошибка'; }
    } catch (e) { const st = document.getElementById('st_' + section.replace('.', '_')); if (st) st.textContent = 'Ошибка'; }
  }

  function wireUI() {
    ['ed_llm', 'ed_confluence', 'ed_metrics_source', 'ed_lt_metrics_source', 'ed_metrics_config', 'ed_default_params', 'ed_queries', 'ed_storage_timescale',
      'ed_prompt_overall', 'ed_prompt_judge', 'ed_prompt_critic', 'ed_prompt_database', 'ed_prompt_kafka', 'ed_prompt_microservices', 'ed_prompt_jvm', 'ed_prompt_hard_resources'
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
      sel.addEventListener('change', async () => { currentArea = sel.value || ''; await loadConfig(); await loadPrompts(); });
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
    }
  }

  document.addEventListener('DOMContentLoaded', async () => {
    try { await window.LoadLens.initProjectArea(); } catch (e) {}
    wireUI();
    await loadConfig();
    await loadPrompts();
  });
})();


