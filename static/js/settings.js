// SPDX-License-Identifier: Apache-2.0
// Settings page: connection forms with checks, SLA form, JSON editors, prompts, services,
// metrics_config, project areas and the first-run wizard.
//
// Scope model: the header selector sets the active project area (cookie). Without an area the
// page edits global defaults; with an area, area-overridable sections (metrics, LLM, SLA, queries,
// prompts, visualizations) are stored as per-area overrides, and services belong to that area.
(function () {
  const LL = window.LoadLens;
  const ui = LL.ui;
  const $ = (id) => document.getElementById(id);
  const esc = ui.escapeHtml;

  // Sections that can be overridden per project area (mirrors AREA_OVERRIDABLE_SECTIONS on the server).
  const AREA_SECTIONS = new Set(['llm', 'domain_sources', 'logs_source', 'default_params', 'queries', 'sla', 'system_context', 'metrics_config']);

  const state = {
    area: '',
    config: null,
    prompts: { area: {}, byService: {} },
    promptDefaults: {},
    // Language of the defaults (= the language of reports) and, per scope ('' = area), the /prompts fields
    // language, sources and override_language.
    promptLanguage: 'ru',
    promptMeta: {},
    servicesMeta: {},
    domainList: [],
    metricsConfigIds: [],
    models: {},
    saved: {},
    editors: {},
    jsonSyncing: false,
    slaScope: '',
    queriesScope: '',
    contextScope: '',
    metricsScope: '',
    promptService: '',
    promptDomain: 'overall',
    promptLoaded: '',
    checkOk: {},
    domainLang: {},
    domainLangPicker: {},
    // Language picked for an empty query row (its text decides once it has one), by domain:index.
    rowLangDraft: {},
    openSources: {},
    sourceChecks: {},
    bindingChecks: {},
    grafanaDatasources: {},
    wizard: false,
    wizardStep: 0,
    dirty: new Map()
  };

  // ---- generic helpers -----------------------------------------------------
  const clone = (v) => JSON.parse(JSON.stringify(v === undefined ? null : v));
  const stable = (v) => JSON.stringify(v === undefined ? null : v);
  function getPath(obj, path) {
    return path.split('.').reduce((acc, key) => (acc && typeof acc === 'object' ? acc[key] : undefined), obj);
  }
  function setPath(obj, path, value) {
    const parts = path.split('.');
    let node = obj;
    parts.slice(0, -1).forEach((key) => {
      if (!node[key] || typeof node[key] !== 'object') node[key] = {};
      node = node[key];
    });
    node[parts[parts.length - 1]] = value;
  }
  function isPlainObject(v) { return !!v && typeof v === 'object' && !Array.isArray(v); }
  function deepMerge(base, override) {
    const out = isPlainObject(base) ? { ...base } : {};
    if (!isPlainObject(override)) return out;
    Object.keys(override).forEach((key) => {
      out[key] = (isPlainObject(out[key]) && isPlainObject(override[key])) ? deepMerge(out[key], override[key]) : override[key];
    });
    return out;
  }
  function setStatus(id, text, tone) {
    const el = $(id);
    if (!el) return;
    el.textContent = text || '';
    el.classList.remove('error', 'ok', 'warn');
    if (tone) el.classList.add(tone);
    if (text && tone === 'ok') setTimeout(() => { if (el.textContent === text) el.textContent = ''; }, 2500);
  }
  function showWarnings(warnings) {
    (warnings || []).forEach((w) => ui.toast(w, { tone: 'warn', timeout: 10000 }));
  }
  // Body of POST /config for a section: adds the area for area-scoped sections and the service when given.
  function configBody(section, data, service) {
    const body = { section, data };
    if (AREA_SECTIONS.has(section) && state.area) body.area = state.area;
    if (service) body.service = service;
    return body;
  }
  async function postConfig(body) {
    const resp = await fetch('/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    return data;
  }
  function requireArea(action) {
    if (state.area) return true;
    ui.toast(LL.t('settings.area.required', { action }), { tone: 'warn' });
    return false;
  }

  // ---- Ace ---------------------------------------------------------------
  function aceTheme() { return ui.theme.current() === 'light' ? 'ace/theme/chrome' : 'ace/theme/twilight'; }
  function makeAce(hostId, mode, minLines) {
    if (typeof ace === 'undefined') return null;
    ace.config.set('basePath', '/static/vendor/ace');
    const ed = ace.edit(hostId);
    ed.setTheme(aceTheme());
    ed.session.setMode(mode || 'ace/mode/json');
    ed.setShowPrintMargin(false);
    ed.session.setUseWrapMode(true);
    ed.setOptions({ minLines: minLines || 12, maxLines: 60, fontSize: 13 });
    ed.renderer.setScrollMargin(6, 6);
    state.editors[hostId] = ed;
    return ed;
  }
  function setJson(ed, obj) {
    if (!ed) return;
    state.jsonSyncing = true;
    ed.setValue(JSON.stringify(obj || {}, null, 2), -1);
    state.jsonSyncing = false;
  }

  // ---- dirty tracking ------------------------------------------------------
  function markDirty(key, label, dirty) {
    if (dirty) state.dirty.set(key, label); else state.dirty.delete(key);
    const bar = $('dirtyBar');
    const list = $('dirtyList');
    if (list) list.textContent = Array.from(state.dirty.values()).join(', ');
    if (bar) bar.classList.toggle('visible', state.dirty.size > 0);
    document.querySelectorAll('#settingsMenu button').forEach((btn) => {
      const section = btn.dataset.section;
      const has = Array.from(state.dirty.keys()).some((k) => k.startsWith(section + ':'));
      let mark = btn.querySelector('.dirty-mark');
      if (has && !mark) { mark = document.createElement('span'); mark.className = 'dirty-mark'; mark.textContent = '●'; mark.title = LL.t('settings.dirty.mark_title'); btn.appendChild(mark); }
      if (!has && mark) mark.remove();
    });
  }
  window.addEventListener('beforeunload', (e) => {
    if (state.dirty.size) { e.preventDefault(); e.returnValue = ''; }
  });

  // ---- project areas -------------------------------------------------------
  // The area comes from the project_area cookie. A cookie can name an area the settings do not have (for example
  // a service id written by a report page whose service is in no area); the area-scoped endpoints answer 404 then.
  function areaKnown() {
    if (!state.area) return true;
    return (LL.projectAreas || []).some((a) => (a && typeof a === 'object' ? String(a.id || '') : String(a || '')) === state.area);
  }
  function renderAreaBar() {
    const label = $('areaScopeLabel');
    const note = $('areaScopeNote');
    const known = areaKnown();
    $('areaBar').classList.toggle('is-unknown', !known);
    if (state.area) {
      const meta = (LL.projectAreas || []).find((a) => a && a.id === state.area);
      label.textContent = LL.t('settings.area.scoped', { title: meta && meta.title ? meta.title : state.area });
      note.textContent = known ? LL.t('settings.area.scoped_note') : LL.t('settings.area.unknown_note', { area: state.area });
    } else {
      label.textContent = LL.t('settings.area.global');
      note.textContent = LL.t('settings.area.global_note');
    }
  }

  // ---- connection cards ----------------------------------------------------
  const SSL_MODES = ['disable', 'allow', 'prefer', 'require', 'verify-ca', 'verify-full'];
  const LLM_PROVIDERS = [
    { v: 'perplexity', l: 'Perplexity' }, { v: 'openai', l: LL.t('settings.llm.provider_openai') }, { v: 'anthropic', l: 'Anthropic' }, { v: 'gigachat', l: 'GigaChat' }
  ];
  const providerFields = (p, extra) => ([
    { path: `${p}.api_base_url`, label: 'API URL', type: 'text', showIf: (m) => m.provider === p },
    { path: `${p}.model`, label: LL.t('settings.field.llm.model.label'), type: 'combo', showIf: (m) => m.provider === p },
    ...(extra || []),
    { path: `${p}.max_concurrent`, label: LL.t('settings.field.llm.max_concurrent.label'), type: 'number', showIf: (m) => m.provider === p },
    { path: `${p}.request_timeout_sec`, label: LL.t('settings.field.llm.request_timeout_sec.label'), type: 'number', showIf: (m) => m.provider === p },
    { path: `${p}.generation.temperature`, label: 'Temperature', type: 'number', step: '0.05', showIf: (m) => m.provider === p },
    { path: `${p}.generation.max_tokens`, label: LL.t('settings.field.llm.max_tokens.label'), type: 'number', hint: LL.t('settings.field.llm.max_tokens.hint'), showIf: (m) => m.provider === p },
    { path: `${p}.context_window_tokens`, label: LL.t('settings.field.llm.context_window_tokens.label'), type: 'number', hint: LL.t('settings.field.llm.context_window_tokens.hint'), showIf: (m) => m.provider === p && (p === 'anthropic' || p === 'gigachat') }
  ]);
  const apiKeyField = (p) => ({ path: `${p}.api_key`, label: LL.t('settings.field.llm.api_key.label'), type: 'password', showIf: (m) => m.provider === p });
  const CONNECTION_CARDS = [
    {
      id: 'storage', section: 'storage.timescale', title: LL.t('settings.card.storage.title'), scope: 'global',
      desc: LL.t('settings.card.storage.desc'),
      fields: [
        { path: 'host', label: LL.t('settings.field.storage.host.label'), type: 'text', placeholder: 'localhost' },
        { path: 'port', label: LL.t('settings.field.storage.port.label'), type: 'number', placeholder: '5432' },
        { path: 'dbname', label: LL.t('settings.field.storage.dbname.label'), type: 'text' },
        { path: 'user', label: LL.t('settings.field.storage.user.label'), type: 'text' },
        { path: 'password', label: LL.t('settings.field.storage.password.label'), type: 'password' },
        { path: 'sslmode', label: 'SSL', type: 'select', options: SSL_MODES },
        { path: 'schema', label: LL.t('settings.field.storage.schema.label'), type: 'text', placeholder: 'public' }
      ]
    },
    {
      id: 'data_sources', section: 'data_sources', title: LL.t('settings.card.data_sources.title'), scope: 'global', kind: 'catalog', testable: false,
      desc: LL.t('settings.card.data_sources.desc'),
      fields: []
    },
    {
      id: 'domain_sources', section: 'domain_sources', title: LL.t('settings.card.domain_sources.title'), scope: 'area', kind: 'bindings', testable: false,
      desc: LL.t('settings.card.domain_sources.desc'),
      fields: []
    },
    {
      id: 'default_params', section: 'default_params', title: LL.t('settings.card.default_params.title'), scope: 'area', testable: false,
      desc: LL.t('settings.card.default_params.desc'),
      fields: [
        { path: 'step', label: LL.t('settings.field.default_params.step.label'), type: 'text', placeholder: '1m', hint: LL.t('settings.field.default_params.step.hint') },
        { path: 'resample_interval', label: LL.t('settings.field.default_params.resample_interval.label'), type: 'text', placeholder: '5T', hint: LL.t('settings.field.default_params.resample_interval.hint') }
      ]
    },
    {
      id: 'logs_source', section: 'logs_source', title: LL.t('settings.card.logs_source.title'), scope: 'area',
      desc: LL.t('settings.card.logs_source.desc'),
      fields: [
        { path: 'enabled', label: LL.t('settings.field.logs_source.enabled.label'), type: 'checkbox' },
        { path: 'opensearch.base_url', label: 'URL OpenSearch Dashboards', type: 'text', placeholder: 'https://opensearch:5601' },
        { path: 'opensearch.index_pattern', label: LL.t('settings.field.logs_source.index_pattern.label'), type: 'text', placeholder: 'app-logs-*' },
        { path: 'opensearch.username_env', label: LL.t('settings.field.logs_source.username.label'), type: 'text' },
        { path: 'opensearch.password_env', label: LL.t('settings.field.logs_source.password.label'), type: 'password' },
        { path: 'opensearch.verify_ssl', label: LL.t('settings.field.logs_source.verify_ssl.label'), type: 'checkbox' }
      ]
    },
    {
      id: 'confluence', section: 'confluence', title: LL.t('settings.card.confluence.title'), scope: 'global',
      desc: LL.t('settings.card.confluence.desc'),
      fields: [
        { path: 'enabled', label: LL.t('settings.field.confluence.enabled.label'), type: 'checkbox' },
        { path: 'base_url', label: 'URL Confluence', type: 'text', placeholder: 'https://confluence.example.ru' },
        { path: 'username', label: LL.t('settings.field.confluence.username.label'), type: 'text' },
        { path: 'password', label: LL.t('settings.field.confluence.password.label'), type: 'password' },
        { path: 'space_key', label: LL.t('settings.field.confluence.space_key.label'), type: 'text' },
        { path: 'parent_page_id', label: LL.t('settings.field.confluence.parent_page_id.label'), type: 'text', hint: LL.t('settings.field.confluence.parent_page_id.hint') },
        { path: 'forecast_parent_page_id', label: LL.t('settings.field.confluence.forecast_parent_page_id.label'), type: 'text', hint: LL.t('settings.field.confluence.forecast_parent_page_id.hint') },
        { path: 'verify_ssl', label: LL.t('settings.field.confluence.verify_ssl.label'), type: 'checkbox' }
      ]
    },
    {
      id: 'confluence_template', section: 'confluence_template', title: LL.t('settings.card.confluence_template.title'), scope: 'global',
      desc: LL.t('settings.card.confluence_template.desc'),
      fields: [
        { path: 'url_basic', label: 'URL Confluence', type: 'text', placeholder: 'https://confluence.example.ru' },
        { path: 'space_conf', label: LL.t('settings.field.confluence_template.space_conf.label'), type: 'text' },
        { path: 'user', label: LL.t('settings.field.confluence_template.user.label'), type: 'text' },
        { path: 'password', label: LL.t('settings.field.confluence_template.password.label'), type: 'password' },
        { path: 'grafana_base_url', label: 'URL Grafana (render)', type: 'text', placeholder: 'http://grafana:3000' },
        { path: 'grafana_login', label: LL.t('settings.field.confluence_template.grafana_login.label'), type: 'text' },
        { path: 'grafana_pass', label: LL.t('settings.field.confluence_template.grafana_pass.label'), type: 'password' },
        { path: 'loki_url', label: 'URL Loki query_range', type: 'text', placeholder: 'http://loki:3100/loki/api/v1/query_range' }
      ]
    },
    {
      id: 'llm', section: 'llm', title: LL.t('settings.card.llm.title'), scope: 'area',
      desc: LL.t('settings.card.llm.desc'),
      fields: [
        { path: 'provider', label: LL.t('settings.field.llm.provider.label'), type: 'select', options: LLM_PROVIDERS },
        { path: 'max_domain_workers', label: LL.t('settings.field.llm.max_domain_workers.label'), type: 'number', hint: LL.t('settings.field.llm.max_domain_workers.hint') },
        { path: 'self_consistency_k', label: LL.t('settings.field.llm.self_consistency_k.label'), type: 'number', hint: LL.t('settings.field.llm.self_consistency_k.hint') },
        ...providerFields('perplexity', [apiKeyField('perplexity')]),
        ...providerFields('openai', [apiKeyField('openai')]),
        ...providerFields('anthropic', [apiKeyField('anthropic')]),
        ...providerFields('gigachat', [
          { path: 'gigachat.cert_file', label: LL.t('settings.field.llm.gigachat_cert_file.label'), type: 'text', hint: LL.t('settings.field.llm.gigachat_cert_file.hint'), showIf: (m) => m.provider === 'gigachat' },
          { path: 'gigachat.key_file', label: LL.t('settings.field.llm.gigachat_key_file.label'), type: 'text', showIf: (m) => m.provider === 'gigachat' },
          { path: 'gigachat.verify', label: LL.t('settings.field.llm.gigachat_verify.label'), type: 'checkbox', showIf: (m) => m.provider === 'gigachat' }
        ])
      ]
    },
    {
      id: 'load_generators', section: 'load_generators', title: LL.t('settings.card.load_generators.title'), scope: 'global', kind: 'kit', testable: false,
      desc: LL.t('settings.card.load_generators.desc'),
      fields: []
    },
    {
      id: 'load_tools', section: 'load_tools', title: LL.t('settings.card.load_tools.title'), scope: 'global', kind: 'kit', testable: false,
      desc: LL.t('settings.card.load_tools.desc'),
      fields: []
    }
  ];

  function sectionData(section) {
    if (section === 'storage.timescale') return (state.config.storage || {}).timescale || {};
    return state.config[section] || {};
  }
  function storeSectionData(section, value) {
    if (section === 'storage.timescale') { state.config.storage = state.config.storage || {}; state.config.storage.timescale = value; return; }
    state.config[section] = value;
  }

  function coerce(field, raw, checked) {
    if (field.type === 'checkbox') return !!checked;
    if (field.type === 'number') { const t = String(raw).trim(); if (!t) return null; const n = Number(t); return Number.isFinite(n) ? n : null; }
    if (field.type === 'list') return String(raw).split(',').map((s) => s.trim()).filter(Boolean);
    if (field.type === 'intlist') return String(raw).split(',').map((s) => s.trim()).filter(Boolean).map(Number).filter((n) => Number.isInteger(n));
    if (field.type === 'mounts') {
      return String(raw).split('\n').map((line) => line.trim()).filter(Boolean).map((line) => {
        const parts = line.split(':');
        const mode = parts.length > 2 ? parts[parts.length - 1].toLowerCase() : 'ro';
        return { source: parts[0] || '', target: parts[1] || '', read_only: mode !== 'rw' };
      });
    }
    if (field.type === 'kv') {
      const out = {};
      String(raw).split('\n').forEach((line) => {
        const text = line.trim();
        const at = text.indexOf('=');
        if (!text || text.startsWith('#') || at <= 0) return;
        out[text.slice(0, at).trim()] = text.slice(at + 1);
      });
      return out;
    }
    return String(raw);
  }
  // Text shown in an input for structured values (lists and NAME=value maps).
  function fieldText(field, value) {
    if (value === undefined || value === null) return '';
    if ((field.type === 'list' || field.type === 'intlist') && Array.isArray(value)) return value.join(', ');
    if (field.type === 'kv' && isPlainObject(value)) return Object.keys(value).map((k) => `${k}=${value[k]}`).join('\n');
    if (field.type === 'mounts' && Array.isArray(value)) {
      return value.map((m) => `${m.source || ''}:${m.target || ''}:${m.read_only === false ? 'rw' : 'ro'}`).join('\n');
    }
    return String(value);
  }

  function modelCombo(id, field, value) {
    const names = Array.isArray(state.llmModels) ? state.llmModels.map((name) => String(name || '').trim()).filter(Boolean) : [];
    const current = value === undefined || value === null ? '' : String(value);
    const picker = document.createElement('div');
    picker.className = 'combobox model-picker';
    const input = document.createElement('input');
    input.type = 'text';
    input.id = id;
    input.autocomplete = 'off';
    input.placeholder = field.placeholder || (names.length ? LL.t('settings.llm.model_placeholder') : '');
    input.value = current;
    picker.appendChild(input);
    if (!names.length) return picker;

    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'model-picker-toggle';
    toggle.setAttribute('aria-label', LL.t('settings.llm.models_toggle_aria'));
    toggle.textContent = '▾';
    const list = document.createElement('div');
    list.className = 'combobox-list';
    list.setAttribute('role', 'listbox');
    let suppressFocus = false;
    const close = () => picker.classList.remove('open');
    const renderOptions = (query) => {
      const needle = String(query || '').trim().toLowerCase();
      const shown = needle ? names.filter((name) => name.toLowerCase().includes(needle)) : names.slice();
      list.replaceChildren();
      if (!shown.length) {
        const empty = document.createElement('div');
        empty.className = 'combobox-empty';
        empty.textContent = LL.t('settings.llm.models_none');
        list.appendChild(empty);
        return;
      }
      shown.forEach((name) => {
        const option = document.createElement('button');
        option.type = 'button';
        option.className = 'combobox-option';
        option.setAttribute('role', 'option');
        option.textContent = name;
        option.addEventListener('mousedown', (event) => {
          event.preventDefault();
          input.value = name;
          input.dispatchEvent(new Event('input', { bubbles: true }));
          close();
        });
        list.appendChild(option);
      });
    };
    const open = (query) => {
      renderOptions(query);
      const spaceBelow = window.innerHeight - input.getBoundingClientRect().bottom;
      picker.classList.toggle('drop-up', spaceBelow < 220);
      picker.classList.add('open');
    };
    toggle.addEventListener('mousedown', (event) => {
      event.preventDefault();
      if (picker.classList.contains('open')) { close(); return; }
      open('');
      suppressFocus = true;
      input.focus();
      suppressFocus = false;
    });
    input.addEventListener('focus', () => { if (!suppressFocus) open(''); });
    input.addEventListener('input', () => open(input.value === current ? '' : input.value));
    input.addEventListener('keydown', (event) => { if (event.key === 'Escape') close(); });
    input.addEventListener('blur', () => setTimeout(close, 150));
    picker.append(toggle, list);
    return picker;
  }

  function renderField(card, field) {
    const wrap = document.createElement('div');
    wrap.dataset.fieldPath = field.path;
    const id = `f-${card.id}-${field.path.replace(/\./g, '-')}`;
    const model = state.models[card.id];
    const value = getPath(model, field.path);
    if (field.type === 'checkbox') {
      wrap.innerHTML = `<label class="checkbox-row"><input type="checkbox" id="${id}" ${value ? 'checked' : ''}/> <span>${esc(field.label)}</span></label>${field.hint ? `<div class="field-hint">${esc(field.hint)}</div>` : ''}`;
    } else if (field.type === 'select') {
      const options = (field.options || []).map((o) => (typeof o === 'string' ? { v: o, l: o } : o));
      const current = value === undefined || value === null ? '' : String(value);
      if (current && !options.some((o) => o.v === current)) options.push({ v: current, l: current });
      wrap.innerHTML = `<label for="${id}">${esc(field.label)}</label><select id="${id}">${options.map((o) => `<option value="${esc(o.v)}" ${o.v === current ? 'selected' : ''}>${esc(o.l)}</option>`).join('')}</select>${field.hint ? `<div class="field-hint">${esc(field.hint)}</div>` : ''}`;
    } else if (field.type === 'combo') {
      const label = document.createElement('label');
      label.htmlFor = id;
      label.textContent = field.label;
      wrap.append(label, modelCombo(id, field, value));
      if (field.hint) {
        const hint = document.createElement('div');
        hint.className = 'field-hint';
        hint.textContent = field.hint;
        wrap.appendChild(hint);
      }
    } else if (field.type === 'password') {
      wrap.innerHTML = `<label for="${id}">${esc(field.label)}</label><div class="password-field"><input type="password" id="${id}" value="${esc(value === undefined || value === null ? '' : value)}" autocomplete="new-password" /><button type="button" class="btn btn-reveal" aria-label="${LL.tHtml('settings.field.reveal_aria')}">${LL.tHtml('settings.field.reveal')}</button></div>${field.hint ? `<div class="field-hint">${esc(field.hint)}</div>` : ''}`;
      const input = wrap.querySelector('input');
      wrap.querySelector('.btn-reveal').addEventListener('click', (e) => {
        const show = input.type === 'password';
        input.type = show ? 'text' : 'password';
        e.currentTarget.textContent = show ? LL.t('settings.field.hide') : LL.t('settings.field.reveal');
      });
    } else if (field.type === 'textarea' || field.type === 'kv' || field.type === 'mounts') {
      wrap.innerHTML = `<label for="${id}">${esc(field.label)}</label><textarea id="${id}" class="mono" rows="${field.rows || 3}" spellcheck="false" placeholder="${esc(field.placeholder || (field.type === 'kv' ? LL.t('settings.field.kv_placeholder') : ''))}"></textarea>${field.hint ? `<div class="field-hint">${esc(field.hint)}</div>` : ''}`;
      wrap.querySelector('textarea').value = fieldText(field, value);
    } else {
      wrap.innerHTML = `<label for="${id}">${esc(field.label)}</label><input type="${field.type === 'number' ? 'number' : 'text'}" id="${id}" ${field.step ? `step="${field.step}"` : ''} value="${esc(fieldText(field, value))}" placeholder="${esc(field.placeholder || '')}" />${field.hint ? `<div class="field-hint">${esc(field.hint)}</div>` : ''}`;
    }
    const control = wrap.querySelector('input, select, textarea');
    control.addEventListener(field.type === 'checkbox' || field.type === 'select' ? 'change' : 'input', () => {
      setPath(state.models[card.id], field.path, coerce(field, control.value, control.checked));
      setJson(state.editors[`ed_${card.id}`], state.models[card.id]);
      updateCardDirty(card);
      updateFieldVisibility(card);
    });
    return wrap;
  }

  function updateFieldVisibility(card) {
    const model = state.models[card.id];
    const root = document.getElementById(`card-${card.id}`);
    if (!root) return;
    (card.fields || []).forEach((field) => {
      const el = root.querySelector(`[data-field-path="${field.path}"]`);
      if (el) el.style.display = (typeof field.showIf === 'function' && !field.showIf(model)) ? 'none' : '';
    });
  }

  function refreshCardFields(card) {
    if (card.kind === 'catalog') { renderCatalogBody(card); return; }
    if (card.kind === 'kit') { renderKitBody(card); return; }
    if (card.kind === 'bindings') { renderBindingsBody(card); return; }
    const grid = document.querySelector(`#card-${card.id} .form-grid`);
    if (!grid) return;
    grid.innerHTML = '';
    card.fields.forEach((field) => grid.appendChild(renderField(card, field)));
    updateFieldVisibility(card);
  }

  function updateCardDirty(card) {
    const dirty = stable(state.models[card.id]) !== state.saved[card.id];
    const btn = document.querySelector(`#card-${card.id} .card-save`);
    if (btn) btn.disabled = !dirty;
    markDirty(`connections:${card.id}`, card.title, dirty);
    if (dirty) state.checkOk[card.id] = false;
    updateWizardNav();
  }

  function renderConnectionCards() {
    const host = $('connectionCards');
    host.innerHTML = '';
    CONNECTION_CARDS.forEach((card) => {
      state.models[card.id] = clone(sectionData(card.section));
      state.saved[card.id] = stable(state.models[card.id]);
      const scopeNote = card.scope === 'area'
        ? (state.area ? LL.t('settings.card.scope_area', { area: state.area }) : LL.t('settings.card.scope_area_global'))
        : LL.t('settings.card.scope_global');
      const el = document.createElement('article');
      el.className = 'conn-card';
      el.id = `card-${card.id}`;
      const bodyClass = card.kind ? 'card-body' : 'form-grid three';
      el.innerHTML = `
        <div class="conn-head"><h3>${esc(card.title)}</h3><span class="conn-actions">${card.id === 'llm' ? `<button type="button" class="btn card-models">${LL.tHtml('settings.llm.models_button')}</button>` : ''}${card.testable === false ? '' : `<button type="button" class="btn card-test">${LL.tHtml('settings.card.test_button')}</button>`}</span></div>
        <p class="help">${esc(card.desc)} <span class="dim">(${esc(scopeNote)})</span></p>
        <div class="${bodyClass}"></div>
        <div class="conn-result" role="status"></div>
        <details class="report-collapsible" style="margin-top:12px">
          <summary><span>${LL.tHtml('settings.json.advanced_params')}</span><small class="report-collapsible-subtitle">${LL.tHtml('settings.json.all_keys', { section: card.section })}</small></summary>
          <div class="report-collapsible-body">
            <div class="editor-wrap"><div class="editor-header"><span class="editor-title">${esc(card.section)}</span><span class="dim json-status"></span></div><div id="ed_${card.id}" class="ace-host"></div></div>
          </div>
        </details>
        <div class="card-actions">
          <button type="button" class="btn btn-primary card-save" disabled>${LL.tHtml('settings.action.save')}</button>
          <span class="status card-status"></span>
        </div>`;
      host.appendChild(el);
      refreshCardFields(card);
      const ed = makeAce(`ed_${card.id}`, 'ace/mode/json', 10);
      setJson(ed, state.models[card.id]);
      if (ed) {
        ed.session.on('change', () => {
          if (state.jsonSyncing) return;
          const status = el.querySelector('.json-status');
          try {
            const parsed = JSON.parse(ed.getValue() || '{}');
            if (!isPlainObject(parsed)) throw new Error(LL.t('settings.json.expected_object'));
            state.models[card.id] = parsed;
            status.textContent = 'OK';
            refreshCardFields(card);
            updateCardDirty(card);
          } catch (e) {
            status.textContent = LL.t('settings.json.error', { error: e.message });
          }
        });
      }
      const testBtn = el.querySelector('.card-test');
      if (testBtn) testBtn.addEventListener('click', () => testCard(card));
      const modelsBtn = el.querySelector('.card-models');
      if (modelsBtn) modelsBtn.addEventListener('click', () => loadModels(card));
      el.querySelector('.card-save').addEventListener('click', () => saveCard(card));
    });
  }

  async function loadModels(card) {
    const el = $(`card-${card.id}`);
    const result = el.querySelector('.conn-result');
    const btn = el.querySelector('.card-models');
    result.className = 'conn-result pending';
    result.textContent = LL.t('settings.llm.models_loading');
    btn.disabled = true;
    try {
      const resp = await fetch('/config/llm_models', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(configBody('llm', state.models[card.id])) });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      state.llmModels = Array.isArray(data.models) ? data.models : [];
      refreshCardFields(card);
      result.className = `conn-result ${data.ok ? 'ok' : 'error'}`;
      result.textContent = data.message || (data.ok ? LL.t('settings.llm.models_count', { n: state.llmModels.length }) : LL.t('settings.llm.models_unsupported'));
    } catch (e) {
      result.className = 'conn-result error';
      result.textContent = LL.t('settings.llm.models_failed', { error: e.message });
    } finally {
      btn.disabled = false;
    }
  }

  async function testCard(card, extra, resultEl) {
    const el = $(`card-${card.id}`);
    const result = resultEl || el.querySelector('.conn-result');
    const btn = el.querySelector('.card-test');
    if (!result) return;
    result.className = 'conn-result pending';
    result.textContent = LL.t('settings.check.pending');
    if (btn) btn.disabled = true;
    try {
      const body = configBody(card.section, state.models[card.id]);
      if (extra) Object.assign(body, extra);
      const resp = await fetch('/config/test_connection', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      result.className = `conn-result ${data.ok ? 'ok' : 'error'}`;
      result.innerHTML = `${esc(data.message)} <span class="dim">${LL.tHtml('settings.check.elapsed', { ms: data.elapsed_ms })}</span>${!data.ok && data.details && data.details.error ? `<pre>${esc(data.details.error)}</pre>` : ''}`;
      if (data.ok && (!card.kind || card.kind === 'catalog' || (extra && extra.domain === 'default'))) state.checkOk[card.id] = true;
      else if (!card.kind) state.checkOk[card.id] = false;
      updateWizardNav();
      return data;
    } catch (e) {
      result.className = 'conn-result error';
      result.textContent = LL.t('settings.check.failed', { error: e.message });
      if (!card.kind) state.checkOk[card.id] = false;
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  async function saveCard(card) {
    const el = $(`card-${card.id}`);
    const status = el.querySelector('.card-status');
    status.textContent = LL.t('settings.status.saving');
    try {
      const data = await postConfig(configBody(card.section, state.models[card.id]));
      state.saved[card.id] = stable(state.models[card.id]);
      storeSectionData(card.section, clone(state.models[card.id]));
      if (card.id === 'data_sources') {
        state.grafanaDatasources = {};
        const bindings = CONNECTION_CARDS.find((item) => item.id === 'domain_sources');
        if (bindings) renderBindingsBody(bindings);
      }
      if (card.id === 'load_tools') {
        // Generators list the tools by their saved templates.
        const generators = CONNECTION_CARDS.find((item) => item.id === 'load_generators');
        if (generators) renderKitBody(generators);
      }
      updateCardDirty(card);
      const savedText = LL.t('settings.status.saved');
      status.textContent = savedText;
      ui.toast(LL.t('settings.status.section_saved', { section: card.title }), { tone: 'ok' });
      showWarnings(data.warnings);
      setTimeout(() => { if (status.textContent === savedText) status.textContent = ''; }, 2000);
      return true;
    } catch (e) {
      status.textContent = LL.t('settings.status.error', { error: e.message });
      ui.toast(`${card.title}: ${e.message}`, { tone: 'error' });
      return false;
    }
  }

  // ---- data source catalog and domain bindings ----------------------------
  const SOURCE_ID_RE = /^[a-z0-9][a-z0-9_-]{0,39}$/;
  const SOURCE_TYPES = {
    grafana_proxy: { label: 'Grafana', baseId: 'grafana', hint: LL.t('settings.source.type.grafana_proxy_hint') },
    prometheus: { label: 'Prometheus', baseId: 'prometheus', hint: LL.t('settings.source.type.prometheus_hint') },
    influxdb: { label: 'InfluxDB', baseId: 'influxdb', hint: LL.t('settings.source.type.influxdb_hint') }
  };
  const DATASOURCE_MODE_LABELS = { promql: 'PromQL', influxql: 'InfluxQL', flux: 'Flux', sql: 'SQL' };
  const BINDING_FIELDS = {
    datasource_name: { label: LL.t('settings.binding.datasource_name.label'), placeholder: LL.t('settings.binding.datasource_name.placeholder') },
    datasource_uid: { label: LL.t('settings.binding.datasource_uid.label'), placeholder: LL.t('settings.binding.datasource_uid.placeholder') },
    database: { label: LL.t('settings.binding.database.label'), placeholder: LL.t('settings.binding.database.placeholder') },
    bucket: { label: LL.t('settings.binding.bucket.label'), placeholder: LL.t('settings.binding.bucket.placeholder') }
  };
  const hasOwn = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);

  function blankSource(type, title) {
    if (type === 'prometheus') return { title, type, prometheus: { url: '' } };
    if (type === 'influxdb') return { title, type, influxdb: { url: '', org: '', token: '' } };
    return { title, type: 'grafana_proxy', grafana: { base_url: '', verify_ssl: false, auth: { method: 'basic', username: '', password: '', token: '' } } };
  }
  function sourceTypeLabel(type) {
    return (SOURCE_TYPES[type] || {}).label || type || LL.t('settings.summary.no_type');
  }
  function sourceUrl(entry) {
    if (entry.type === 'grafana_proxy') return String(getPath(entry, 'grafana.base_url') || '');
    if (entry.type === 'prometheus') return String(getPath(entry, 'prometheus.url') || '');
    if (entry.type === 'influxdb') return String(getPath(entry, 'influxdb.url') || '');
    return '';
  }
  function sourceSummary(id, entry) {
    const parts = [`id ${id}`, sourceUrl(entry) || LL.t('settings.summary.no_address')];
    if (entry.type === 'grafana_proxy') {
      parts.push(getPath(entry, 'grafana.auth.method') === 'bearer' ? LL.t('settings.source.summary_token') : LL.t('settings.source.summary_login', { login: getPath(entry, 'grafana.auth.username') || LL.t('settings.source.summary_login_unset') }));
    }
    if (entry.type === 'influxdb' && getPath(entry, 'influxdb.org')) parts.push(LL.t('settings.source.summary_org', { org: getPath(entry, 'influxdb.org') }));
    return parts.join(' · ');
  }
  function sourceFields(id, type) {
    const path = (suffix) => `${id}.${suffix}`;
    const fields = [{ path: path('title'), label: LL.t('settings.field.source.title.label'), type: 'text', group: 'main' }];
    if (type === 'prometheus') {
      fields.push({ path: path('prometheus.url'), label: 'URL Prometheus', type: 'text', placeholder: 'http://prometheus:9090', group: 'main' });
    } else if (type === 'influxdb') {
      fields.push(
        { path: path('influxdb.url'), label: 'URL InfluxDB', type: 'text', placeholder: 'http://influxdb:8086', group: 'main' },
        { path: path('influxdb.org'), label: LL.t('settings.field.source.influxdb_org.label'), type: 'text', group: 'auth' },
        { path: path('influxdb.token'), label: LL.t('settings.field.source.influxdb_token.label'), type: 'password', group: 'auth' }
      );
    } else if (type === 'grafana_proxy') {
      const bearer = (m) => getPath(m, path('grafana.auth.method')) === 'bearer';
      fields.push(
        { path: path('grafana.base_url'), label: 'URL Grafana', type: 'text', placeholder: 'http://grafana:3000', group: 'main' },
        { path: path('grafana.auth.method'), label: LL.t('settings.field.source.grafana_auth_method.label'), type: 'select', options: [{ v: 'basic', l: LL.t('settings.field.source.grafana_auth_method.basic') }, { v: 'bearer', l: LL.t('settings.field.source.grafana_auth_method.bearer') }], group: 'auth' },
        { path: path('grafana.auth.username'), label: LL.t('settings.field.source.grafana_username.label'), type: 'text', showIf: (m) => !bearer(m), group: 'auth' },
        { path: path('grafana.auth.password'), label: LL.t('settings.field.source.grafana_password.label'), type: 'password', showIf: (m) => !bearer(m), group: 'auth' },
        { path: path('grafana.auth.token'), label: LL.t('settings.field.source.grafana_token.label'), type: 'password', showIf: bearer, group: 'auth' },
        { path: path('grafana.verify_ssl'), label: LL.t('settings.field.source.grafana_verify_ssl.label'), type: 'checkbox', group: 'flag' }
      );
    }
    return fields;
  }
  function rememberCheck(store, key, result) {
    state[store][key] = { className: result.className, html: result.innerHTML };
  }
  function restoreCheck(store, key, result) {
    const saved = state[store][key];
    if (!saved) return;
    result.className = saved.className;
    result.innerHTML = saved.html;
  }
  async function runCheck(card, extra, result, store, key) {
    await testCard(card, extra, result);
    rememberCheck(store, key, result);
  }
  function sourceEditor(card, id, entry) {
    const body = makeEl('div', 'q-item-body');
    body.appendChild(makeEl('p', 'dim', (SOURCE_TYPES[entry.type] || {}).hint || LL.t('settings.source.type_unsupported')));
    const groups = { main: makeEl('div', 'source-form-main'), auth: makeEl('div', 'source-form-auth'), flag: makeEl('div', 'source-form-flags') };
    sourceFields(id, entry.type).forEach((field) => groups[field.group].appendChild(renderField(card, field)));
    Object.values(groups).forEach((group) => { if (group.childElementCount) body.appendChild(group); });
    const actions = makeEl('div', 'q-item-actions');
    actions.appendChild(makeBtn(LL.t('settings.source.delete'), () => deleteSource(card, id), 'btn btn-sm btn-danger'));
    body.appendChild(actions);
    return body;
  }
  function sourceItem(card, id, entry) {
    const open = !!state.openSources[id];
    const item = makeEl('div', 'q-item source-item');
    const head = makeEl('div', 'q-item-head');
    head.setAttribute('role', 'button');
    head.setAttribute('aria-expanded', String(open));
    head.tabIndex = 0;
    const toggle = () => { state.openSources[id] = !open; renderCatalogBody(card); };
    head.addEventListener('click', toggle);
    head.addEventListener('keydown', (event) => {
      if (event.target === head && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); toggle(); }
    });
    const main = makeEl('div', 'q-item-main');
    const title = makeEl('div', 'q-item-title source-title');
    const titleText = makeEl('span', '', String(entry.title || id));
    title.append(titleText, makeEl('span', 'pill na', sourceTypeLabel(entry.type)));
    const meta = makeEl('div', 'q-item-meta', sourceSummary(id, entry));
    main.append(title, meta);
    const result = makeEl('div', 'conn-result');
    result.setAttribute('role', 'status');
    const check = makeBtn(LL.t('settings.action.check'), (event) => {
      event.stopPropagation();
      runCheck(card, { source: id }, result, 'sourceChecks', id);
    }, 'btn btn-sm');
    head.append(main, check, makeEl('span', 'q-item-caret', open ? '▾' : '▸'));
    item.appendChild(head);
    if (open) {
      const editor = sourceEditor(card, id, entry);
      const refreshHead = () => {
        const current = state.models[card.id][id] || {};
        titleText.textContent = String(current.title || id);
        meta.textContent = sourceSummary(id, current);
      };
      editor.addEventListener('input', refreshHead);
      editor.addEventListener('change', refreshHead);
      item.appendChild(editor);
    }
    item.appendChild(result);
    restoreCheck('sourceChecks', id, result);
    return item;
  }
  async function deleteSource(card, id) {
    const ok = await ui.confirm({
      title: LL.t('settings.source.delete'),
      message: LL.t('settings.source.delete_confirm', { id }),
      confirmText: LL.t('settings.action.delete'),
      danger: true
    });
    if (!ok) return;
    delete state.models[card.id][id];
    delete state.openSources[id];
    delete state.sourceChecks[id];
    setJson(state.editors[`ed_${card.id}`], state.models[card.id]);
    renderCatalogBody(card);
    updateCardDirty(card);
  }
  function suggestSourceId(type, used) {
    const base = (SOURCE_TYPES[type] || SOURCE_TYPES.grafana_proxy).baseId;
    if (!used.has(base)) return base;
    let number = 2;
    while (used.has(`${base}-${number}`)) number += 1;
    return `${base}-${number}`;
  }
  function wireAddSourceForm(body, used) {
    const type = body.querySelector('#dsType');
    const title = body.querySelector('#dsTitle');
    const idInput = body.querySelector('#dsId');
    let titleTouched = false;
    let idTouched = false;
    const sync = () => {
      const meta = SOURCE_TYPES[type.value];
      body.querySelector('#dsTypeHint').textContent = meta.hint;
      if (!titleTouched) title.value = meta.label;
      if (!idTouched) idInput.value = suggestSourceId(type.value, used);
    };
    type.addEventListener('change', sync);
    title.addEventListener('input', () => { titleTouched = true; });
    idInput.addEventListener('input', () => { idTouched = true; });
    sync();
    type.focus();
  }
  function readAddSourceForm(body, used) {
    const type = body.querySelector('#dsType').value;
    const title = body.querySelector('#dsTitle').value.trim();
    const id = body.querySelector('#dsId').value.trim();
    if (!title) return { error: LL.t('settings.form.title_required') };
    if (!SOURCE_ID_RE.test(id)) return { error: LL.t('settings.form.id_rule') };
    if (used.has(id)) return { error: LL.t('settings.source.exists', { id }) };
    return { value: { id, type, title } };
  }
  async function openAddSource(card) {
    const model = state.models[card.id];
    const used = new Set(Object.keys(model));
    const options = Object.keys(SOURCE_TYPES).map((type) => `<option value="${type}">${esc(SOURCE_TYPES[type].label)}</option>`).join('');
    const created = await ui.dialog({
      title: LL.t('settings.source.new_title'),
      bodyHtml: [
        `<label>${LL.tHtml('settings.form.type')}<select id="dsType">${options}</select></label>`,
        '<p class="dim" id="dsTypeHint"></p>',
        `<label>${LL.tHtml('settings.form.title')}<input id="dsTitle" type="text" maxlength="80" /></label>`,
        `<label>${LL.tHtml('settings.form.id')}<input id="dsId" type="text" maxlength="40" spellcheck="false" autocomplete="off" /></label>`,
        `<p class="dim">${LL.tHtml('settings.source.id_hint')}</p>`,
        '<div class="dialog-error" id="dsError"></div>'
      ].join(''),
      onOpen: (body) => wireAddSourceForm(body, used),
      actions: [
        { label: LL.t('settings.action.cancel') },
        {
          label: LL.t('settings.action.add'),
          primary: true,
          onClick: (close, body) => {
            const read = readAddSourceForm(body, used);
            if (read.error) { body.querySelector('#dsError').textContent = read.error; return; }
            close(read.value);
          }
        }
      ]
    });
    if (!created) return;
    model[created.id] = blankSource(created.type, created.title);
    state.openSources[created.id] = true;
    setJson(state.editors[`ed_${card.id}`], model);
    renderCatalogBody(card);
    updateCardDirty(card);
  }
  function renderCatalogBody(card) {
    const host = document.querySelector(`#card-${card.id} .card-body`);
    if (!host) return;
    const model = isPlainObject(state.models[card.id]) ? state.models[card.id] : {};
    state.models[card.id] = model;
    const ids = Object.keys(model);
    card.fields = ids.flatMap((id) => sourceFields(id, (model[id] || {}).type));
    host.replaceChildren();
    const list = makeEl('div', 'q-list source-list');
    if (!ids.length) list.appendChild(makeEl('p', 'dim', LL.t('settings.source.empty')));
    ids.forEach((id) => list.appendChild(sourceItem(card, id, isPlainObject(model[id]) ? model[id] : {})));
    const bar = makeEl('div', 'struct-toolbar');
    bar.appendChild(makeBtn(LL.t('settings.source.add'), () => openAddSource(card)));
    host.append(list, bar);
    updateFieldVisibility(card);
  }
  // ---- load generators and load tool templates (Run test page) -----------------------------
  const GENERATOR_TYPES = {
    ssh: { label: 'SSH', baseId: 'gen', hint: LL.t('settings.gen.type.ssh_hint') },
    docker: { label: 'Docker', baseId: 'docker', hint: LL.t('settings.gen.type.docker_hint') },
    kubernetes: { label: 'Kubernetes', baseId: 'k8s', hint: LL.t('settings.gen.type.kubernetes_hint') },
    gitlab: { label: 'GitLab CI', baseId: 'gitlab', hint: LL.t('settings.gen.type.gitlab_hint') },
    jenkins: { label: 'Jenkins', baseId: 'jenkins', hint: LL.t('settings.gen.type.jenkins_hint') },
    manual: { label: LL.t('settings.gen.type.manual'), baseId: 'manual', hint: LL.t('settings.gen.type.manual_hint') }
  };
  const KIT_GROUPS = { main: 'kit-form-grid', auth: 'kit-form-grid', dirs: 'kit-form-grid', codes: 'kit-form-grid', wide: 'kit-form-wide' };
  const CONTAINER_TYPES = new Set(['docker', 'kubernetes']);
  const CI_TYPES = new Set(['gitlab', 'jenkins']);
  function blankGenerator(type, title) {
    const common = { title, type, tools: {}, max_parallel: 0, env: {} };
    if (type === 'manual') return { title, type: 'manual' };
    if (type === 'docker') {
      return { ...common, docker: { url: '', ca_cert: '', client_cert: '', client_key: '', verify: true }, network: '', mounts: [],
        scripts_dir: '/scripts', work_dir: '/tmp', resources: { cpus: 0, memory_mb: 0 }, remove_finished: true };
    }
    if (type === 'kubernetes') {
      return { ...common, kubernetes: { api_url: '', token: '', ca_cert: '', verify: true, in_cluster: false }, namespace: 'loadtest',
        scripts: { config_map: '', pvc: '' }, scripts_dir: '/scripts', work_dir: '/tmp', node_selector: {}, resources: { cpu: '', memory: '' },
        service_account: '', image_pull_secret: '', ttl_seconds: 3600 };
    }
    if (type === 'gitlab') {
      return { title, type, max_parallel: 0, gitlab: { url: '', project: '', ref: '', api_token: '', trigger_token: '', verify: true, inputs: 'auto' },
        loadlens_url: '', tags_metrics: false, param_options: {} };
    }
    if (type === 'jenkins') {
      return { title, type, max_parallel: 0, jenkins: { url: '', job: '', ref: '', username: '', api_token: '', verify: true },
        loadlens_url: '', tags_metrics: false, param_options: {} };
    }
    return { ...common, type: 'ssh', ssh: { host: '', port: 22, username: '', private_key: '', passphrase: '', password: '', host_key: '' },
      work_dir: '/opt/loadlens/runs', scripts_dir: '' };
  }
  function generatorFields(id, type) {
    const p = (suffix) => `${id}.${suffix}`;
    const fields = [{ path: p('title'), label: LL.t('settings.field.gen.title.label'), type: 'text', group: 'main' }];
    const parallel = { path: p('max_parallel'), label: LL.t('settings.field.gen.max_parallel.label'), type: 'number', placeholder: LL.t('settings.field.gen.unlimited.placeholder'), group: 'dirs' };
    const env = { path: p('env'), label: LL.t('settings.field.gen.env.label'), type: 'kv', rows: 2, hint: LL.t('settings.field.gen.env.hint'), group: 'wide' };
    if (type === 'ssh') {
      fields.push(
        { path: p('ssh.host'), label: LL.t('settings.field.gen.ssh_host.label'), type: 'text', placeholder: LL.t('settings.field.gen.ssh_host.placeholder'), group: 'main' },
        { path: p('ssh.port'), label: LL.t('settings.field.gen.ssh_port.label'), type: 'number', placeholder: '22', group: 'main' },
        { path: p('ssh.username'), label: LL.t('settings.field.gen.ssh_username.label'), type: 'text', group: 'auth' },
        { path: p('ssh.passphrase'), label: LL.t('settings.field.gen.ssh_passphrase.label'), type: 'password', group: 'auth' },
        { path: p('ssh.password'), label: LL.t('settings.field.gen.ssh_password.label'), type: 'password', group: 'auth' },
        { path: p('ssh.private_key'), label: LL.t('settings.field.gen.ssh_private_key.label'), type: 'textarea', rows: 4, placeholder: '-----BEGIN OPENSSH PRIVATE KEY-----', hint: LL.t('settings.field.gen.ssh_private_key.hint'), group: 'wide' },
        { path: p('ssh.host_key'), label: LL.t('settings.field.gen.ssh_host_key.label'), type: 'text', placeholder: LL.t('settings.field.gen.ssh_host_key.placeholder'), hint: LL.t('settings.field.gen.ssh_host_key.hint'), group: 'wide' },
        { path: p('work_dir'), label: LL.t('settings.field.gen.work_dir.label'), type: 'text', placeholder: '/opt/loadlens/runs', hint: LL.t('settings.field.gen.work_dir.hint'), group: 'dirs' },
        { path: p('scripts_dir'), label: LL.t('settings.field.gen.scripts_dir.label'), type: 'text', placeholder: '/opt/loadlens/tests', hint: LL.t('settings.field.gen.scripts_dir.hint'), group: 'dirs' },
        parallel, env
      );
    } else if (type === 'docker') {
      fields.push(
        { path: p('docker.url'), label: LL.t('settings.field.gen.docker_url.label'), type: 'text', placeholder: 'https://gen2.example.ru:2376', group: 'main' },
        { path: p('network'), label: LL.t('settings.field.gen.network.label'), type: 'text', placeholder: LL.t('settings.field.gen.network.placeholder'), group: 'main' },
        { path: p('docker.verify'), label: LL.t('settings.field.gen.verify.label'), type: 'checkbox', group: 'auth' },
        { path: p('remove_finished'), label: LL.t('settings.field.gen.remove_finished.label'), type: 'checkbox', group: 'auth' },
        { path: p('docker.ca_cert'), label: LL.t('settings.field.gen.docker_ca_cert.label'), type: 'textarea', rows: 3, placeholder: '-----BEGIN CERTIFICATE-----', group: 'wide' },
        { path: p('docker.client_cert'), label: LL.t('settings.field.gen.docker_client_cert.label'), type: 'textarea', rows: 3, placeholder: '-----BEGIN CERTIFICATE-----', group: 'wide' },
        { path: p('docker.client_key'), label: LL.t('settings.field.gen.docker_client_key.label'), type: 'textarea', rows: 3, placeholder: '-----BEGIN PRIVATE KEY-----', group: 'wide' },
        { path: p('mounts'), label: LL.t('settings.field.gen.mounts.label'), type: 'mounts', rows: 2, placeholder: 'loadlens-scripts:/scripts:ro', hint: LL.t('settings.field.gen.mounts.hint'), group: 'wide' },
        { path: p('scripts_dir'), label: LL.t('settings.field.gen.container_scripts_dir.label'), type: 'text', placeholder: '/scripts', group: 'dirs' },
        { path: p('work_dir'), label: LL.t('settings.field.gen.container_work_dir.label'), type: 'text', placeholder: '/tmp', group: 'dirs' },
        { path: p('resources.cpus'), label: LL.t('settings.field.gen.docker_cpus.label'), type: 'number', step: '0.5', placeholder: LL.t('settings.field.gen.unlimited.placeholder'), group: 'dirs' },
        { path: p('resources.memory_mb'), label: LL.t('settings.field.gen.docker_memory_mb.label'), type: 'number', placeholder: LL.t('settings.field.gen.unlimited.placeholder'), group: 'dirs' },
        parallel, env
      );
    } else if (type === 'kubernetes') {
      const outside = (m) => !getPath(m, p('kubernetes.in_cluster'));
      fields.push(
        { path: p('kubernetes.api_url'), label: LL.t('settings.field.gen.k8s_api_url.label'), type: 'text', placeholder: 'https://k8s.example.ru:6443', showIf: outside, group: 'main' },
        { path: p('namespace'), label: LL.t('settings.field.gen.k8s_namespace.label'), type: 'text', placeholder: 'loadtest', group: 'main' },
        { path: p('kubernetes.in_cluster'), label: LL.t('settings.field.gen.k8s_in_cluster.label'), type: 'checkbox', group: 'auth' },
        { path: p('kubernetes.verify'), label: LL.t('settings.field.gen.verify.label'), type: 'checkbox', showIf: outside, group: 'auth' },
        { path: p('kubernetes.token'), label: LL.t('settings.field.gen.k8s_token.label'), type: 'password', showIf: outside, group: 'auth' },
        { path: p('kubernetes.ca_cert'), label: LL.t('settings.field.gen.k8s_ca_cert.label'), type: 'textarea', rows: 3, placeholder: '-----BEGIN CERTIFICATE-----', showIf: outside, group: 'wide' },
        { path: p('scripts.config_map'), label: LL.t('settings.field.gen.k8s_config_map.label'), type: 'text', placeholder: 'loadtest-scripts', group: 'dirs' },
        { path: p('scripts.pvc'), label: LL.t('settings.field.gen.k8s_pvc.label'), type: 'text', group: 'dirs' },
        { path: p('scripts_dir'), label: LL.t('settings.field.gen.container_scripts_dir.label'), type: 'text', placeholder: '/scripts', group: 'dirs' },
        { path: p('resources.cpu'), label: LL.t('settings.field.gen.k8s_cpu.label'), type: 'text', placeholder: '2', group: 'dirs' },
        { path: p('resources.memory'), label: LL.t('settings.field.gen.k8s_memory.label'), type: 'text', placeholder: '2Gi', group: 'dirs' },
        { path: p('service_account'), label: LL.t('settings.field.gen.k8s_service_account.label'), type: 'text', group: 'dirs' },
        { path: p('image_pull_secret'), label: LL.t('settings.field.gen.k8s_image_pull_secret.label'), type: 'text', group: 'dirs' },
        { path: p('ttl_seconds'), label: LL.t('settings.field.gen.k8s_ttl_seconds.label'), type: 'number', placeholder: '3600', group: 'dirs' },
        { path: p('node_selector'), label: LL.t('settings.field.gen.k8s_node_selector.label'), type: 'kv', rows: 2, placeholder: 'role=loadgen', group: 'wide' },
        parallel, env
      );
    } else if (type === 'gitlab' || type === 'jenkins') {
      const ci = [
        { path: p('loadlens_url'), label: LL.t('settings.field.gen.loadlens_url.label'), type: 'text', placeholder: 'https://loadlens.example.ru', hint: LL.t('settings.field.gen.loadlens_url.hint'), group: 'dirs' },
        { path: p('tags_metrics'), label: LL.t('settings.field.gen.tags_metrics.label'), type: 'checkbox', hint: LL.t('settings.field.gen.tags_metrics.hint'), group: 'wide' },
        { path: p('param_options'), label: LL.t('settings.field.gen.param_options.label'), type: 'kv', rows: 2, placeholder: 'STAND=stage, preprod', hint: LL.t('settings.field.gen.param_options.hint'), group: 'wide' },
        parallel
      ];
      if (type === 'gitlab') {
        fields.push(
          { path: p('gitlab.url'), label: LL.t('settings.field.gen.gitlab_url.label'), type: 'text', placeholder: 'https://gitlab.example.ru', group: 'main' },
          { path: p('gitlab.project'), label: LL.t('settings.field.gen.gitlab_project.label'), type: 'text', placeholder: LL.t('settings.field.gen.gitlab_project.placeholder'), group: 'main' },
          { path: p('gitlab.ref'), label: LL.t('settings.field.gen.gitlab_ref.label'), type: 'text', placeholder: LL.t('settings.field.gen.gitlab_ref.placeholder'), group: 'main' },
          { path: p('gitlab.api_token'), label: LL.t('settings.field.gen.gitlab_api_token.label'), type: 'password', hint: LL.t('settings.field.gen.gitlab_api_token.hint'), group: 'auth' },
          { path: p('gitlab.trigger_token'), label: LL.t('settings.field.gen.gitlab_trigger_token.label'), type: 'password', group: 'auth' },
          { path: p('gitlab.verify'), label: LL.t('settings.field.gen.verify.label'), type: 'checkbox', group: 'auth' },
          { path: p('gitlab.inputs'), label: LL.t('settings.field.gen.gitlab_inputs.label'), type: 'select', options: [{ v: 'auto', l: LL.t('settings.field.gen.gitlab_inputs.auto') }, { v: 'on', l: LL.t('settings.field.gen.gitlab_inputs.on') }, { v: 'off', l: LL.t('settings.field.gen.gitlab_inputs.off') }], group: 'dirs' },
          ...ci
        );
      } else {
        fields.push(
          { path: p('jenkins.url'), label: LL.t('settings.field.gen.jenkins_url.label'), type: 'text', placeholder: 'https://jenkins.example.ru', group: 'main' },
          { path: p('jenkins.job'), label: LL.t('settings.field.gen.jenkins_job.label'), type: 'text', placeholder: 'perf/load-test', group: 'main' },
          { path: p('jenkins.ref'), label: LL.t('settings.field.gen.jenkins_ref.label'), type: 'text', placeholder: LL.t('settings.field.gen.jenkins_ref.placeholder'), group: 'main' },
          { path: p('jenkins.username'), label: LL.t('settings.field.gen.jenkins_username.label'), type: 'text', group: 'auth' },
          { path: p('jenkins.api_token'), label: LL.t('settings.field.gen.jenkins_api_token.label'), type: 'password', group: 'auth' },
          { path: p('jenkins.verify'), label: LL.t('settings.field.gen.verify.label'), type: 'checkbox', group: 'auth' },
          ...ci
        );
      }
    }
    // wizardStep, not step: `step` is the HTML step of number inputs (CPU 0.5).
    fields.forEach((field) => { field.wizardStep = generatorStep(field.path.slice(id.length + 1)); });
    return fields;
  }
  // Wizard step of a generator field (path without the id): the connection fields are what a check vouches for.
  function generatorStep(rel) {
    if (rel === 'title') return 'type';
    if (rel === 'gitlab.inputs') return 'setup';
    if (/^(ssh|docker|kubernetes|gitlab|jenkins)\./.test(rel) || rel === 'namespace') return 'connect';
    if (rel === 'max_parallel' || rel === 'env') return 'finish';
    return 'setup';
  }
  function formatStamp(iso) {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return String(iso || '');
    return LL.fmt.dateTime(date, { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  }
  function generatorStatus(entry) {
    if (!isPlainObject(entry) || !entry.type || entry.type === 'manual') return null;
    const verified = isPlainObject(entry.verified) && entry.verified.at ? entry.verified : null;
    return verified ? { text: LL.t('settings.gen.status_verified', { at: formatStamp(verified.at) }), tone: 'ok' } : { text: LL.t('settings.gen.status_unverified'), tone: 'warn' };
  }
  function hasPinnedKey(entry) {
    return !isPlainObject(entry) || entry.type !== 'ssh' || (isPlainObject(entry.ssh) && !!entry.ssh.host_key);
  }
  // A change of the endpoint makes an earlier check say nothing about the generator.
  function generatorFieldEdited(card, id, rel) {
    const entry = state.models[card.id][id];
    if (generatorStep(rel) !== 'connect' || !isPlainObject(entry) || !entry.verified) return;
    delete entry.verified;
    syncKit(card);
  }
  const VERSION_RE = /v?\d+(?:\.\d+)+/;
  function toolBadge(info, toolId, type) {
    if (!isPlainObject(info)) return null;
    if (Array.isArray(info.tools_found) && info.tools_found.includes(toolId)) {
      const full = isPlainObject(info.versions) && info.versions[toolId] ? String(info.versions[toolId]) : '';
      // `k6 v0.54.0 (commit/…)` and `locust 2.31.8 from /opt/…` become the version number; the full line is the tooltip.
      const short = full ? (full.match(VERSION_RE) || [full.slice(0, 24)])[0] : '';
      const badge = makeEl('span', 'gen-tool-badge ok', short || ({ docker: LL.t('settings.gen.badge.image_present'), kubernetes: LL.t('settings.gen.badge.image_set') }[type] || LL.t('settings.gen.badge.found')));
      if (full) badge.title = full;
      return badge;
    }
    if (Array.isArray(info.tools_missing) && info.tools_missing.includes(toolId)) {
      return makeEl('span', 'gen-tool-badge warn', type === 'docker' ? LL.t('settings.gen.badge.image_missing') : LL.t('settings.gen.badge.not_found'));
    }
    return null;
  }
  function generatorSummary(id, entry) {
    if (entry.type === 'manual') return `id ${id} · ${LL.t('settings.gen.summary.manual')}`;
    const noAddress = LL.t('settings.summary.no_address');
    const tools = isPlainObject(entry.tools) && Object.keys(entry.tools).length ? Object.keys(entry.tools).join(', ') : LL.t('settings.gen.summary.all_tools');
    if (entry.type === 'docker') return `id ${id} · ${getPath(entry, 'docker.url') || noAddress} · ${tools}`;
    if (entry.type === 'kubernetes') {
      const where = getPath(entry, 'kubernetes.in_cluster') ? LL.t('settings.gen.summary.in_cluster') : (getPath(entry, 'kubernetes.api_url') || noAddress);
      return `id ${id} · ${where} · ${entry.namespace || 'default'} · ${tools}`;
    }
    if (entry.type === 'gitlab') return `id ${id} · ${getPath(entry, 'gitlab.url') || noAddress} · ${getPath(entry, 'gitlab.project') || LL.t('settings.gen.summary.no_project')} · ${getPath(entry, 'gitlab.ref') || LL.t('settings.gen.summary.default_ref')}`;
    if (entry.type === 'jenkins') return `id ${id} · ${getPath(entry, 'jenkins.url') || noAddress} · ${getPath(entry, 'jenkins.job') || LL.t('settings.gen.summary.no_job')}${getPath(entry, 'jenkins.ref') ? ` · ${getPath(entry, 'jenkins.ref')}` : ''}`;
    const ssh = isPlainObject(entry.ssh) ? entry.ssh : {};
    const target = ssh.host ? `${ssh.username || '?'}@${ssh.host}${ssh.port && Number(ssh.port) !== 22 ? `:${ssh.port}` : ''}` : noAddress;
    return `id ${id} · ${target} · ${tools} · ${ssh.host_key ? LL.t('settings.gen.summary.host_key_pinned') : LL.t('settings.gen.summary.host_key_unpinned')}`;
  }
  function toolFields(id) {
    const p = (suffix) => `${id}.${suffix}`;
    return [
      { path: p('title'), label: LL.t('settings.field.tool.title.label'), type: 'text', group: 'main' },
      { path: p('executable'), label: LL.t('settings.field.tool.executable.label'), type: 'text', hint: LL.t('settings.field.tool.executable.hint'), group: 'main' },
      { path: p('cwd'), label: LL.t('settings.field.tool.cwd.label'), type: 'text', placeholder: '{script_dir}', group: 'main' },
      { path: p('image'), label: LL.t('settings.field.tool.image.label'), type: 'text', placeholder: 'grafana/k6:0.54.0', group: 'main' },
      { path: p('command'), label: LL.t('settings.field.tool.command.label'), type: 'textarea', rows: 2, hint: LL.t('settings.field.tool.command.hint'), group: 'wide' },
      { path: p('env'), label: LL.t('settings.field.tool.env.label'), type: 'kv', rows: 2, hint: LL.t('settings.field.tool.env.hint'), group: 'wide' },
      { path: p('script_kind'), label: LL.t('settings.field.tool.script_kind.label'), type: 'select', options: [{ v: 'file', l: LL.t('settings.field.tool.script_kind.file') }, { v: 'class', l: LL.t('settings.field.tool.script_kind.class') }, { v: 'none', l: LL.t('settings.field.tool.script_kind.none') }], group: 'codes' },
      { path: p('scripts_glob'), label: LL.t('settings.field.tool.scripts_glob.label'), type: 'list', placeholder: '*.js, *.ts', group: 'codes' },
      { path: p('stop_signal'), label: LL.t('settings.field.tool.stop_signal.label'), type: 'select', options: ['TERM', 'INT', 'HUP', 'QUIT'], group: 'codes' },
      { path: p('success_exit_codes'), label: LL.t('settings.field.tool.success_exit_codes.label'), type: 'intlist', placeholder: '0', group: 'codes' },
      { path: p('failed_checks_exit_codes'), label: LL.t('settings.field.tool.failed_checks_exit_codes.label'), type: 'intlist', placeholder: '99', group: 'codes' },
      { path: p('grace_seconds'), label: LL.t('settings.field.tool.grace_seconds.label'), type: 'number', placeholder: '30', group: 'codes' },
      { path: p('version_command'), label: LL.t('settings.field.tool.version_command.label'), type: 'text', placeholder: '{exe} --version', group: 'wide' },
      { path: p('stop_command'), label: LL.t('settings.field.tool.stop_command.label'), type: 'text', placeholder: '{exe_dir}shutdown.sh {control_port}', hint: LL.t('settings.field.tool.stop_command.hint'), group: 'wide' },
      { path: p('status_command'), label: LL.t('settings.field.tool.status_command.label'), type: 'text', placeholder: '{exe} status --address 127.0.0.1:{control_port}', hint: LL.t('settings.field.tool.status_command.hint'), group: 'wide' },
      { path: p('run_filter_packs'), label: LL.t('settings.field.tool.run_filter_packs.label'), type: 'list', hint: LL.t('settings.field.tool.run_filter_packs.hint'), group: 'wide' },
      { path: p('help'), label: LL.t('settings.field.tool.help.label'), type: 'textarea', rows: 2, group: 'wide' }
    ];
  }
  function toolSummary(id, entry) {
    return `id ${id} · ${String(entry.command || LL.t('settings.tool.no_command'))}`;
  }
  const KITS = {
    load_generators: {
      title: (id, entry) => String(entry.title || id),
      pill: (entry) => (GENERATOR_TYPES[entry.type] || {}).label || entry.type || LL.t('settings.summary.no_type'),
      summary: generatorSummary,
      fields: (id, entry) => generatorFields(id, entry.type),
      hint: (entry) => (GENERATOR_TYPES[entry.type] || {}).hint || LL.t('settings.gen.type_unknown'),
      checkable: (entry) => !!entry.type && entry.type !== 'manual',
      empty: LL.t('settings.gen.empty'),
      addLabel: LL.t('settings.gen.add'),
      deleteLabel: LL.t('settings.gen.delete'),
      extra: generatorToolsEditor,
      add: openGeneratorWizard,
      status: generatorStatus,
      onFieldEdit: generatorFieldEdited,
      open: {},
      checks: {},
      checkData: {}
    },
    load_tools: {
      title: (id, entry) => String(entry.title || id),
      pill: (entry) => ({ file: LL.t('settings.tool.pill.file'), class: LL.t('settings.tool.pill.class'), none: LL.t('settings.tool.pill.none') }[entry.script_kind] || LL.t('settings.tool.pill.file')),
      summary: toolSummary,
      fields: (id) => toolFields(id),
      hint: () => LL.t('settings.tool.hint'),
      checkable: () => false,
      empty: LL.t('settings.tool.empty'),
      addLabel: LL.t('settings.tool.add'),
      deleteLabel: LL.t('settings.tool.delete'),
      add: openAddTool,
      open: {},
      checks: {}
    }
  };
  function kitFor(card) { return KITS[card.id]; }
  function syncKit(card) {
    setJson(state.editors[`ed_${card.id}`], state.models[card.id]);
    updateCardDirty(card);
  }
  function generatorToolsEditor(card, id, entry, opts = {}) {
    if (!entry.type || entry.type === 'manual' || CI_TYPES.has(entry.type)) return null;
    const info = opts.info !== undefined ? opts.info : (KITS.load_generators.checkData[id] || null);
    const rerender = opts.rerender || (() => renderKitBody(card));
    const tools = isPlainObject(state.config.load_tools) ? state.config.load_tools : {};
    const ids = Object.keys(tools);
    const container = CONTAINER_TYPES.has(entry.type);
    const ci = CI_TYPES.has(entry.type);
    const box = makeEl('div', 'gen-tools');
    box.appendChild(makeEl('label', '', LL.t('settings.gen.tools.label')));
    const own = isPlainObject(entry.tools) ? entry.tools : {};
    const all = !Object.keys(own).length;
    const note = container
      ? LL.t('settings.gen.tools.note_container')
      : (ci ? LL.t('settings.gen.tools.note_ci')
        : LL.t('settings.gen.tools.note_ssh'));
    box.appendChild(makeEl('div', 'field-hint', LL.t(all ? 'settings.gen.tools.all' : 'settings.gen.tools.some', { note })));
    const columns = container ? ['image', 'executable'] : (ci ? ['executable'] : ['executable', 'scripts_dir']);
    const placeholders = (tool) => ({
      image: String(tool.image || LL.t('settings.gen.tools.no_image')),
      executable: String(tool.executable || ''),
      scripts_dir: entry.scripts_dir ? LL.t('settings.gen.tools.scripts_at', { dir: entry.scripts_dir }) : LL.t('settings.gen.tools.tool_scripts_dir')
    });
    const labels = {
      image: (name) => LL.t('settings.gen.tools.aria_image', { tool: name }),
      executable: (name) => LL.t('settings.gen.tools.aria_executable', { tool: name }),
      scripts_dir: (name) => LL.t('settings.gen.tools.aria_scripts_dir', { tool: name })
    };
    ids.forEach((toolId) => {
      const tool = tools[toolId] || {};
      const row = makeEl('div', `gen-tool-row cols-${columns.length}`);
      const check = makeEl('input');
      check.type = 'checkbox';
      check.checked = all || hasOwn(own, toolId);
      check.id = `gt-${id}-${toolId}`;
      const label = makeEl('label', 'gen-tool-name', String(tool.title || toolId));
      label.htmlFor = check.id;
      const badge = toolBadge(info, toolId, entry.type);
      if (badge) label.append(' ', badge);
      const inputs = {};
      columns.forEach((column) => {
        const input = makeEl('input', 'mono');
        input.type = 'text';
        input.placeholder = placeholders(tool)[column];
        input.setAttribute('aria-label', labels[column](tool.title || toolId));
        input.value = isPlainObject(own[toolId]) ? String(own[toolId][column] || '') : '';
        inputs[column] = input;
      });
      const write = () => {
        const current = state.models[card.id][id];
        if (!isPlainObject(current.tools) || !Object.keys(current.tools).length) {
          current.tools = {};
          ids.forEach((other) => { current.tools[other] = {}; });
        }
        if (check.checked) {
          const value = {};
          columns.forEach((column) => { if (inputs[column].value.trim()) value[column] = inputs[column].value.trim(); });
          current.tools[toolId] = value;
        } else {
          delete current.tools[toolId];
        }
        const everything = ids.every((other) => hasOwn(current.tools, other) && !Object.keys(current.tools[other]).length);
        if (everything) current.tools = {};
        syncKit(card);
      };
      check.addEventListener('change', () => { write(); rerender(); });
      columns.forEach((column) => inputs[column].addEventListener('input', write));
      row.append(check, label, ...columns.map((column) => inputs[column]));
      box.appendChild(row);
    });
    return box;
  }
  function kitEditor(card, kit, id, entry) {
    const body = makeEl('div', 'q-item-body');
    body.appendChild(makeEl('p', 'dim', kit.hint(entry)));
    const groups = {};
    kit.fields(id, entry).forEach((field) => {
      const group = field.group || 'main';
      if (!groups[group]) groups[group] = makeEl('div', KIT_GROUPS[group] || 'kit-form-grid');
      groups[group].appendChild(renderField(card, field));
    });
    ['main', 'auth', 'wide', 'dirs', 'codes'].forEach((name) => { if (groups[name]) body.appendChild(groups[name]); });
    if (kit.onFieldEdit) {
      const edited = (event) => {
        const wrap = event.target.closest ? event.target.closest('[data-field-path]') : null;
        if (wrap && wrap.dataset.fieldPath.startsWith(`${id}.`)) kit.onFieldEdit(card, id, wrap.dataset.fieldPath.slice(id.length + 1));
      };
      body.addEventListener('input', edited);
      body.addEventListener('change', edited);
    }
    const extra = kit.extra ? kit.extra(card, id, entry) : null;
    if (extra) body.appendChild(extra);
    const actions = makeEl('div', 'q-item-actions');
    actions.appendChild(makeBtn(kit.deleteLabel, () => deleteKitEntry(card, id), 'btn btn-sm btn-danger'));
    body.appendChild(actions);
    return body;
  }
  function hostKeyMessage(entry, details) {
    const ssh = isPlainObject(entry.ssh) ? entry.ssh : {};
    const where = `${ssh.host || '?'}${ssh.port && Number(ssh.port) !== 22 ? `:${ssh.port}` : ''}`;
    return LL.t('settings.gen.host_key_message', { where, key: details.host_key_fingerprint || details.host_key });
  }
  async function checkGenerator(card, kit, id, result) {
    const data = await testCard(card, { source: id }, result);
    const entry = state.models[card.id][id];
    kit.checks[id] = { className: result.className, html: result.innerHTML };
    if (!data || !isPlainObject(entry)) return;
    const details = isPlainObject(data.details) ? data.details : {};
    kit.checkData[id] = details;
    let changed = false;
    if (details.host_key && isPlainObject(entry.ssh) && !entry.ssh.host_key) {
      const trusted = await ui.confirm({ title: LL.t('settings.gen.host_key_title'), message: hostKeyMessage(entry, details), confirmText: LL.t('settings.gen.trust'), cancelText: LL.t('settings.gen.distrust') });
      if (trusted) {
        entry.ssh.host_key = details.host_key;
        changed = true;
      }
    }
    if (data.ok && hasPinnedKey(entry) && details.target) {
      entry.verified = { at: details.checked_at, target: details.target };
      changed = true;
    } else if (entry.verified) {
      delete entry.verified;
      changed = true;
    }
    if (changed) syncKit(card);
    renderKitBody(card);
    if (changed) ui.toast(LL.t('settings.gen.check_recorded'), { tone: 'warn', timeout: 8000 });
  }
  function kitItem(card, kit, id, entry) {
    const open = !!kit.open[id];
    const item = makeEl('div', 'q-item source-item');
    const head = makeEl('div', 'q-item-head');
    head.setAttribute('role', 'button');
    head.setAttribute('aria-expanded', String(open));
    head.tabIndex = 0;
    const toggle = () => { kit.open[id] = !open; renderKitBody(card); };
    head.addEventListener('click', toggle);
    head.addEventListener('keydown', (event) => {
      if (event.target === head && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); toggle(); }
    });
    const main = makeEl('div', 'q-item-main');
    const title = makeEl('div', 'q-item-title source-title');
    const titleText = makeEl('span', '', kit.title(id, entry));
    title.append(titleText, makeEl('span', 'pill na', kit.pill(entry)));
    const statusPill = makeEl('span', 'pill');
    const showStatus = (current) => {
      const status = kit.status ? kit.status(current) : null;
      statusPill.className = `pill ${status ? status.tone : ''}`;
      statusPill.textContent = status ? status.text : '';
      statusPill.style.display = status ? '' : 'none';
    };
    showStatus(entry);
    title.appendChild(statusPill);
    const meta = makeEl('div', 'q-item-meta', kit.summary(id, entry));
    main.append(title, meta);
    const result = makeEl('div', 'conn-result');
    result.setAttribute('role', 'status');
    head.appendChild(main);
    if (kit.checkable(entry)) {
      head.appendChild(makeBtn(LL.t('settings.action.check'), (event) => { event.stopPropagation(); checkGenerator(card, kit, id, result); }, 'btn btn-sm'));
    }
    head.appendChild(makeEl('span', 'q-item-caret', open ? '▾' : '▸'));
    item.appendChild(head);
    if (open) {
      const editor = kitEditor(card, kit, id, entry);
      const refreshHead = () => {
        const current = state.models[card.id][id] || {};
        titleText.textContent = kit.title(id, current);
        meta.textContent = kit.summary(id, current);
        showStatus(current);
      };
      editor.addEventListener('input', refreshHead);
      editor.addEventListener('change', refreshHead);
      item.appendChild(editor);
    }
    item.appendChild(result);
    const saved = kit.checks[id];
    if (saved) { result.className = saved.className; result.innerHTML = saved.html; }
    return item;
  }
  function renderKitBody(card) {
    const kit = kitFor(card);
    const host = document.querySelector(`#card-${card.id} .card-body`);
    if (!host || !kit) return;
    const model = isPlainObject(state.models[card.id]) ? state.models[card.id] : {};
    state.models[card.id] = model;
    const ids = Object.keys(model);
    card.fields = ids.flatMap((id) => kit.fields(id, isPlainObject(model[id]) ? model[id] : {}));
    host.replaceChildren();
    const list = makeEl('div', 'q-list source-list');
    if (!ids.length) list.appendChild(makeEl('p', 'dim', kit.empty));
    ids.forEach((id) => list.appendChild(kitItem(card, kit, id, isPlainObject(model[id]) ? model[id] : {})));
    const bar = makeEl('div', 'struct-toolbar');
    bar.appendChild(makeBtn(kit.addLabel, () => kit.add(card)));
    host.append(list, bar);
    updateFieldVisibility(card);
  }
  async function deleteKitEntry(card, id) {
    const kit = kitFor(card);
    const ok = await ui.confirm({
      title: kit.deleteLabel,
      message: LL.t('settings.kit.delete_confirm', { id }),
      confirmText: LL.t('settings.action.delete'),
      danger: true
    });
    if (!ok) return;
    delete state.models[card.id][id];
    delete kit.open[id];
    delete kit.checks[id];
    syncKit(card);
    renderKitBody(card);
  }
  function suggestKitId(base, used) {
    if (!used.has(base)) return base;
    let number = 2;
    while (used.has(`${base}-${number}`)) number += 1;
    return `${base}-${number}`;
  }
  const WIZARD_STEP_TITLES = {
    type: LL.t('settings.gen.wizard.step_type'),
    connect: LL.t('settings.gen.wizard.step_connect'),
    setup: LL.t('settings.gen.wizard.step_setup'),
    finish: LL.t('settings.gen.wizard.step_finish')
  };
  // New generator, step by step: type and title, connection with a check (SSH: also trust in the host key),
  // tools and directories prefilled from the check, limits and saving. The identifier is assigned by LoadLens.
  async function openGeneratorWizard(card) {
    const kit = kitFor(card);
    const model = state.models[card.id];
    const otherEdits = stable(model) !== state.saved[card.id];
    const w = { step: 0, type: 'ssh', id: '', title: '', titleAuto: false, check: null, view: null, toolsTouched: false, distrust: false, checking: false, saving: false, error: '' };
    const entry = () => model[w.id];
    const stepList = () => (w.type === 'manual' ? ['type', 'finish'] : ['type', 'connect', 'setup', 'finish']);
    const details = () => (w.check && isPlainObject(w.check.details) ? w.check.details : {});
    const connected = () => !!(w.check && details().connected);
    const passed = () => !!(w.check && w.check.ok && hasPinnedKey(entry()));
    const startDraft = (type) => {
      if (w.id) delete model[w.id];
      w.type = type;
      w.id = suggestKitId(GENERATOR_TYPES[type].baseId, new Set(Object.keys(model)));
      // A title the wizard filled in itself (the manual type) is dropped when the type changes again.
      const title = type === 'manual' ? (w.title || LL.t('settings.gen.type.manual')) : (w.titleAuto ? '' : w.title);
      w.titleAuto = type === 'manual' && !w.title;
      model[w.id] = blankGenerator(type, title);
      if (CI_TYPES.has(type)) model[w.id].loadlens_url = window.location.origin;
      w.check = null;
      w.view = null;
      w.toolsTouched = false;
      w.distrust = false;
      syncKit(card);
    };
    startDraft('ssh');
    let body = null;
    let actions = null;
    let close = null;

    const fieldsBlock = (stepName) => {
      const wrap = makeEl('div', 'gen-wizard-fields');
      const groups = {};
      generatorFields(w.id, w.type)
        .filter((field) => field.wizardStep === stepName && field.path !== `${w.id}.ssh.host_key` && field.path !== `${w.id}.title`)
        .forEach((field) => {
          const group = field.group || 'main';
          if (!groups[group]) groups[group] = makeEl('div', KIT_GROUPS[group] || 'kit-form-grid');
          groups[group].appendChild(renderField(card, field));
        });
      ['main', 'auth', 'wide', 'dirs', 'codes'].forEach((name) => { if (groups[name]) wrap.appendChild(groups[name]); });
      return wrap;
    };
    const applyVisibility = () => {
      generatorFields(w.id, w.type).forEach((field) => {
        if (typeof field.showIf !== 'function') return;
        const el = body.querySelector(`[data-field-path="${field.path}"]`);
        if (el) el.style.display = field.showIf(model) ? '' : 'none';
      });
    };
    const focusFirst = () => {
      const first = body.querySelector('input:not([type="checkbox"]), select, textarea');
      if (first) first.focus();
    };

    const renderCheck = () => {
      const host = body.querySelector('.gen-check-host');
      if (!host) return;
      host.replaceChildren();
      const result = makeEl('div', w.view ? w.view.className : 'conn-result');
      result.setAttribute('role', 'status');
      if (w.view) result.innerHTML = w.view.html;
      host.appendChild(result);
      const d = details();
      const current = entry();
      if (w.type === 'ssh' && w.check && isPlainObject(current.ssh)) {
        if (d.host_key && !current.ssh.host_key) {
          const trust = makeEl('div', 'gen-trust');
          trust.appendChild(makeEl('div', '', hostKeyMessage(current, d)));
          if (w.distrust) {
            trust.appendChild(makeEl('div', 'gen-trust-note', LL.t('settings.gen.wizard.distrust_note')));
          }
          const buttons = makeEl('div', 'gen-trust-actions');
          buttons.append(
            makeBtn(LL.t('settings.gen.trust'), () => { current.ssh.host_key = d.host_key; w.distrust = false; syncKit(card); renderCheck(); renderActions(); }, 'btn btn-sm btn-primary'),
            makeBtn(LL.t('settings.gen.distrust'), () => { w.distrust = true; renderCheck(); renderActions(); }, 'btn btn-sm')
          );
          trust.appendChild(buttons);
          host.appendChild(trust);
        } else if (current.ssh.host_key) {
          host.appendChild(makeEl('div', 'gen-pinned', LL.t('settings.gen.wizard.pinned', { key: d.host_key_fingerprint || `${String(current.ssh.host_key).slice(0, 40)}…` })));
        }
      }
      const problems = Array.isArray(d.problems) ? d.problems : [];
      if (w.check && problems.length) {
        const list = makeEl('ul', 'gen-problems');
        problems.forEach((problem) => list.appendChild(makeEl('li', '', problem)));
        host.appendChild(list);
      }
    };
    const runCheck = async () => {
      if (w.checking) return;
      w.checking = true;
      w.distrust = false;
      w.error = '';
      renderActions();
      const host = body.querySelector('.gen-check-host');
      const result = makeEl('div', 'conn-result');
      result.setAttribute('role', 'status');
      if (host) host.replaceChildren(result);
      const data = await testCard(card, { source: w.id }, result);
      w.checking = false;
      w.check = data || null;
      w.view = { className: result.className, html: result.innerHTML };
      if (!body.isConnected) return;
      if (stepList()[w.step] === 'setup') render();
      else { renderCheck(); renderActions(); }
    };
    const checkBar = (label, note) => {
      const bar = makeEl('div', 'gen-check-bar');
      bar.appendChild(makeBtn(label, () => runCheck(), 'btn gen-check-btn'));
      if (note) bar.appendChild(makeEl('span', 'dim', note));
      return bar;
    };
    // Tools found by the check become the generator's list; untouched by the admin only.
    const prefillTools = () => {
      if (w.toolsTouched || !connected() || !['ssh', 'docker'].includes(w.type)) return;
      const found = Array.isArray(details().tools_found) ? details().tools_found : [];
      const all = Object.keys(isPlainObject(state.config.load_tools) ? state.config.load_tools : {});
      entry().tools = !found.length || all.every((toolId) => found.includes(toolId)) ? {} : Object.fromEntries(found.map((toolId) => [toolId, {}]));
      syncKit(card);
    };

    const renderTypeStep = () => {
      const typeLabel = makeEl('label', '', LL.t('settings.form.type'));
      const select = makeEl('select');
      Object.keys(GENERATOR_TYPES).forEach((type) => {
        const option = makeEl('option', '', GENERATOR_TYPES[type].label);
        option.value = type;
        option.selected = type === w.type;
        select.appendChild(option);
      });
      typeLabel.appendChild(select);
      const titleLabel = makeEl('label', '', LL.t('settings.form.title'));
      const title = makeEl('input');
      title.type = 'text';
      title.maxLength = 80;
      title.placeholder = LL.t('settings.gen.wizard.title_placeholder');
      title.value = String(entry().title || '');
      titleLabel.appendChild(title);
      select.addEventListener('change', () => { w.title = title.value.trim(); w.error = ''; startDraft(select.value); render(); });
      title.addEventListener('input', () => { entry().title = title.value; w.title = title.value.trim(); w.titleAuto = false; syncKit(card); });
      body.append(typeLabel, makeEl('p', 'dim', GENERATOR_TYPES[w.type].hint), titleLabel,
        makeEl('p', 'dim', LL.t('settings.gen.wizard.title_note')));
    };
    const renderConnectStep = () => {
      body.appendChild(makeEl('p', 'dim', GENERATOR_TYPES[w.type].hint));
      body.appendChild(fieldsBlock('connect'));
      const note = w.type === 'ssh'
        ? LL.t('settings.gen.wizard.ssh_check_required')
        : LL.t('settings.gen.wizard.check_optional');
      body.appendChild(checkBar(LL.t('settings.gen.wizard.check'), note));
      body.appendChild(makeEl('div', 'gen-check-host'));
      renderCheck();
    };
    const renderCiSetupStep = () => {
      const d = details();
      const panel = makeEl('div', 'gen-check-panel');
      const params = Array.isArray(d.params) ? d.params : [];
      if (connected()) {
        panel.appendChild(makeEl('div', '', `${d.system || ''}${d.ref ? ` · ${LL.t('settings.gen.wizard.ref', { ref: d.ref })}` : ''}`));
        panel.appendChild(makeEl('div', '', params.length
          ? LL.t('settings.gen.wizard.ci_params', { list: params.map((param) => `${param.name}${param.options && param.options.length ? ` ${LL.t('settings.gen.wizard.param_options', { n: param.options.length })}` : ''}`).join(', ') })
          : LL.t('settings.gen.wizard.ci_no_params')));
        if (Array.isArray(d.auto) && d.auto.length) panel.appendChild(makeEl('div', 'dim', LL.t('settings.gen.wizard.ci_auto', { list: d.auto.join(', ') })));
      } else {
        panel.appendChild(makeEl('div', 'warn-text', LL.t('settings.gen.wizard.ci_unchecked')));
      }
      body.appendChild(panel);
      body.appendChild(fieldsBlock('setup'));
      body.appendChild(checkBar(w.check ? LL.t('settings.gen.wizard.recheck') : LL.t('settings.gen.wizard.check'), ''));
      body.appendChild(makeEl('div', 'gen-check-host'));
      renderCheck();
    };
    const renderSetupStep = () => {
      if (CI_TYPES.has(w.type)) { renderCiSetupStep(); return; }
      prefillTools();
      const d = details();
      const panel = makeEl('div', 'gen-check-panel');
      if (connected()) {
        panel.appendChild(makeEl('div', '', LL.t('settings.gen.wizard.system', { system: d.system || LL.t('settings.gen.wizard.system_unknown') })));
        panel.appendChild(makeEl('div', 'dim', LL.t('settings.gen.wizard.setup_note')));
      } else {
        panel.appendChild(makeEl('div', 'warn-text', LL.t('settings.gen.wizard.setup_unchecked')));
      }
      body.appendChild(panel);
      body.appendChild(fieldsBlock('setup'));
      const tools = generatorToolsEditor(card, w.id, entry(), { info: connected() ? d : null, rerender: () => { w.toolsTouched = true; render(); } });
      if (tools) body.appendChild(tools);
      body.appendChild(checkBar(w.check ? LL.t('settings.gen.wizard.recheck') : LL.t('settings.gen.wizard.check'), ''));
      body.appendChild(makeEl('div', 'gen-check-host'));
      renderCheck();
    };
    const renderFinishStep = () => {
      const current = entry();
      if (w.type !== 'manual') body.appendChild(fieldsBlock('finish'));
      const summary = makeEl('div', 'gen-check-panel');
      summary.appendChild(makeEl('div', 'gen-summary-title', `${GENERATOR_TYPES[w.type].label} · ${current.title || w.id}`));
      summary.appendChild(makeEl('div', 'dim', generatorSummary(w.id, current)));
      const problems = Array.isArray(details().problems) ? details().problems : [];
      if (w.type === 'manual') summary.appendChild(makeEl('div', '', LL.t('settings.gen.wizard.manual_no_check')));
      else if (passed()) summary.appendChild(makeEl('div', 'ok-text', LL.t('settings.gen.wizard.passed', { message: w.check.message })));
      else if (connected()) summary.appendChild(makeEl('div', 'warn-text', LL.t('settings.gen.wizard.problems', { problems: problems.join('; ') || w.check.message })));
      else summary.appendChild(makeEl('div', 'warn-text', LL.t('settings.gen.wizard.finish_unchecked')));
      if (otherEdits) summary.appendChild(makeEl('div', 'dim', LL.t('settings.gen.wizard.other_edits')));
      body.appendChild(summary);
    };

    const renderActions = () => {
      actions.replaceChildren();
      const name = stepList()[w.step];
      actions.appendChild(makeBtn(LL.t('settings.action.cancel'), () => close(null), 'btn'));
      if (w.step > 0) actions.appendChild(makeBtn(LL.t('settings.wizard.back'), () => { w.error = ''; w.step -= 1; render(); focusFirst(); }, 'btn'));
      if (name === 'finish') {
        const save = makeBtn(w.saving ? LL.t('settings.status.saving') : LL.t('settings.gen.wizard.save'), () => saveDraft(), 'btn btn-primary');
        save.disabled = w.saving || w.checking;
        actions.appendChild(save);
      } else {
        let label = LL.t('settings.wizard.next');
        let blocked = w.checking;
        if (name === 'connect') {
          if (w.type === 'ssh') blocked = blocked || !connected() || !hasPinnedKey(entry());
          else if (!connected()) label = LL.t('settings.gen.wizard.continue_unchecked');
        }
        const next = makeBtn(label, () => goNext(), 'btn btn-primary');
        next.disabled = blocked;
        if (blocked && !w.checking) next.title = LL.t('settings.gen.wizard.next_blocked');
        actions.appendChild(next);
      }
      body.querySelectorAll('.gen-check-btn').forEach((btn) => { btn.disabled = w.checking; });
    };
    const render = () => {
      const list = stepList();
      if (w.step >= list.length) w.step = list.length - 1;
      const name = list[w.step];
      body.replaceChildren();
      const progress = makeEl('ol', 'gen-wizard-steps');
      const stepTitle = (step) => (step === 'setup' && CI_TYPES.has(w.type) ? LL.t('settings.gen.wizard.step_ci_params') : WIZARD_STEP_TITLES[step]);
      list.forEach((step, index) => progress.appendChild(makeEl('li', index === w.step ? 'active' : (index < w.step ? 'done' : ''), `${index + 1}. ${stepTitle(step)}`)));
      body.appendChild(progress);
      if (name === 'type') renderTypeStep();
      else if (name === 'connect') renderConnectStep();
      else if (name === 'setup') renderSetupStep();
      else renderFinishStep();
      const error = makeEl('div', 'dialog-error', w.error);
      error.setAttribute('role', 'alert');
      body.appendChild(error);
      applyVisibility();
      renderActions();
    };
    const goNext = () => {
      if (stepList()[w.step] === 'type') {
        const title = String(entry().title || '').trim();
        if (!title) { w.error = LL.t('settings.form.title_required'); render(); return; }
        entry().title = title;
        syncKit(card);
      }
      w.error = '';
      w.step += 1;
      render();
      focusFirst();
    };
    // Only the new generator is saved: the saved section plus this entry, other edits stay in the form.
    const saveDraft = async () => {
      const current = entry();
      const d = details();
      if (w.type !== 'manual' && passed() && d.target) current.verified = { at: d.checked_at, target: d.target };
      else delete current.verified;
      let saved = {};
      try { saved = JSON.parse(state.saved[card.id] || '{}'); } catch (e) { saved = {}; }
      const next = { ...(isPlainObject(saved) ? saved : {}), [w.id]: clone(current) };
      w.saving = true;
      w.error = '';
      renderActions();
      try {
        const data = await postConfig(configBody(card.section, next));
        state.saved[card.id] = stable(next);
        storeSectionData(card.section, clone(next));
        if (w.view) kit.checks[w.id] = w.view;
        if (w.check) kit.checkData[w.id] = d;
        showWarnings(data.warnings);
        close('saved');
      } catch (e) {
        w.saving = false;
        w.error = LL.t('settings.gen.wizard.save_failed', { error: e.message });
        render();
      }
    };
    // Editing the connection after a check makes the check say nothing; a new host also needs a new key.
    const edited = (event) => {
      const wrap = event.target.closest ? event.target.closest('[data-field-path]') : null;
      if (!wrap || !wrap.dataset.fieldPath.startsWith(`${w.id}.`)) return;
      const rel = wrap.dataset.fieldPath.slice(w.id.length + 1);
      applyVisibility();
      if (generatorStep(rel) !== 'connect' || !w.check) return;
      w.check = null;
      w.view = { className: 'conn-result pending', html: LL.tHtml('settings.gen.wizard.changed') };
      w.distrust = false;
      const current = entry();
      if (w.type === 'ssh' && ['ssh.host', 'ssh.port'].includes(rel) && isPlainObject(current.ssh) && current.ssh.host_key) {
        current.ssh.host_key = '';
        syncKit(card);
      }
      renderCheck();
      renderActions();
    };

    const outcome = await ui.dialog({
      title: LL.t('settings.gen.wizard.title'),
      wide: true,
      actions: [],
      onOpen: (el, closer) => {
        body = el;
        close = closer;
        actions = el.parentElement.querySelector('.dialog-actions');
        body.classList.add('gen-wizard');
        body.addEventListener('input', edited);
        body.addEventListener('change', edited);
        render();
        focusFirst();
      }
    });
    if (outcome !== 'saved') {
      delete model[w.id];
      syncKit(card);
      return;
    }
    kit.open[w.id] = false;
    renderKitBody(card);
    updateCardDirty(card);
    ui.toast(LL.t('settings.gen.saved', { title: entry().title }), { tone: 'ok' });
  }
  async function openAddTool(card) {
    const model = state.models[card.id];
    const used = new Set(Object.keys(model));
    const options = [`<option value="">${LL.tHtml('settings.tool.blank_template')}</option>`].concat(Object.keys(model).map((id) => `<option value="${esc(id)}">${esc(String((model[id] || {}).title || id))}</option>`)).join('');
    const created = await ui.dialog({
      title: LL.t('settings.tool.new_title'),
      bodyHtml: [
        `<label>${LL.tHtml('settings.tool.copy_from')}<select id="ltFrom">${options}</select></label>`,
        `<label>${LL.tHtml('settings.form.title')}<input id="ltTitle" type="text" maxlength="80" /></label>`,
        `<label>${LL.tHtml('settings.form.id')}<input id="ltId" type="text" maxlength="40" spellcheck="false" autocomplete="off" /></label>`,
        '<div class="dialog-error" id="ltError"></div>'
      ].join(''),
      onOpen: (body) => body.querySelector('#ltFrom').focus(),
      actions: [
        { label: LL.t('settings.action.cancel') },
        {
          label: LL.t('settings.action.add'),
          primary: true,
          onClick: (close, body) => {
            const from = body.querySelector('#ltFrom').value;
            const title = body.querySelector('#ltTitle').value.trim();
            const id = body.querySelector('#ltId').value.trim();
            const error = body.querySelector('#ltError');
            if (!title) { error.textContent = LL.t('settings.form.title_required'); return; }
            if (!SOURCE_ID_RE.test(id)) { error.textContent = LL.t('settings.form.id_rule'); return; }
            if (used.has(id)) { error.textContent = LL.t('settings.tool.exists', { id }); return; }
            close({ id, title, from });
          }
        }
      ]
    });
    if (!created) return;
    const base = created.from && isPlainObject(model[created.from]) ? clone(model[created.from]) : {
      executable: '', command: '{exe} {params} {script}', env: {}, cwd: '{script_dir}', script_kind: 'file',
      scripts_glob: [], success_exit_codes: [0], failed_checks_exit_codes: [], stop_signal: 'TERM', grace_seconds: 30,
      version_command: '', run_filter_packs: [], help: ''
    };
    model[created.id] = { ...base, title: created.title };
    KITS.load_tools.open[created.id] = true;
    syncKit(card);
    renderKitBody(card);
  }

  function savedCatalog() {
    return isPlainObject(state.config.data_sources) ? state.config.data_sources : {};
  }
  function savedSource(id) {
    const entry = savedCatalog()[id];
    return isPlainObject(entry) ? entry : null;
  }
  function bindingRows() {
    return [{ id: 'default', title: LL.t('settings.binding.default_row'), note: LL.t('settings.binding.default_row_note') }]
      .concat(QUERY_DOMAINS.map((id) => ({ id, title: LL.domainTitle(id) })));
  }
  function bindingSummary(binding) {
    if (!isPlainObject(binding) || !binding.source) return LL.t('settings.binding.no_default');
    const entry = savedSource(binding.source);
    const parts = [entry ? String(entry.title || binding.source) : LL.t('settings.binding.source_missing', { source: binding.source })];
    const datasource = binding.datasource_name || binding.datasource_uid;
    if (datasource) parts.push(datasource);
    if (binding.database) parts.push(LL.t('settings.binding.summary_database', { name: binding.database }));
    if (binding.bucket) parts.push(`bucket ${binding.bucket}`);
    return parts.join(' · ');
  }
  function datasourceList(sourceId) {
    if (!state.grafanaDatasources[sourceId]) {
      state.grafanaDatasources[sourceId] = { status: 'loading', items: [], message: '' };
      loadGrafanaDatasources(sourceId);
    }
    return state.grafanaDatasources[sourceId];
  }
  async function loadGrafanaDatasources(sourceId) {
    let next;
    try {
      const resp = await fetch(`/config/grafana_datasources?source=${encodeURIComponent(sourceId)}`);
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      next = data.ok
        ? { status: 'ok', items: Array.isArray(data.datasources) ? data.datasources : [], message: data.message || '' }
        : { status: 'error', items: [], message: data.message || LL.t('settings.binding.ds_list_failed') };
    } catch (e) {
      next = { status: 'error', items: [], message: e.message };
    }
    state.grafanaDatasources[sourceId] = next;
    refreshBindingViews();
  }
  // Every view with a binding editor waits for the list of Grafana datasources. Editors of a query row or of
  // the pack card are drawn again in place: the text typed into the rest of the form stays.
  const liveBindingEditors = new Map();
  function mountBindingEditor(ctx) {
    const editor = makeEl('div', 'q-source-editor');
    editor.dataset.bindingEditor = ctx.id;
    liveBindingEditors.set(ctx.id, { editor, ctx });
    ctx.rerender = () => redrawBindingEditor(ctx.id);
    editor.append(bindingSourceSelect(ctx), bindingParams(ctx));
    return editor;
  }
  function redrawBindingEditor(id) {
    const item = liveBindingEditors.get(id);
    if (!item || !item.editor.isConnected) {
      liveBindingEditors.delete(id);
      return;
    }
    item.editor.replaceChildren(bindingSourceSelect(item.ctx), bindingParams(item.ctx));
  }
  function refreshBindingViews() {
    const card = CONNECTION_CARDS.find((item) => item.id === 'domain_sources');
    if (card) renderBindingsBody(card);
    Array.from(liveBindingEditors.keys()).forEach(redrawBindingEditor);
  }
  function findDatasource(items, binding) {
    const uid = String(binding.datasource_uid || '');
    if (uid) return items.find((ds) => ds.uid === uid) || null;
    const name = String(binding.datasource_name || '');
    return name ? items.find((ds) => ds.name === name) || null : null;
  }
  function datasourceLabel(ds) {
    const parts = [ds.name];
    if (ds.type === 'influxdb') parts.push(DATASOURCE_MODE_LABELS[ds.mode] || ds.mode);
    if (ds.is_default) parts.push(LL.t('settings.binding.ds_default'));
    return parts.join(' · ');
  }
  function commitBindings(card) {
    setJson(state.editors[`ed_${card.id}`], state.models[card.id]);
    renderBindingsBody(card);
    updateCardDirty(card);
  }
  // ---- binding editor ------------------------------------------------------------------------------
  // The domain bindings card, a query row and the pack card edit a binding the same way. ctx:
  //   id, title       ids of the inputs and the name in aria labels
  //   own             the binding is set here (else the inherited one is described by inheritNote)
  //   get()/set(b)    read the binding, replace it (null: inherit)
  //   setField(f, v)  change one text field without re-rendering
  //   rerender()      draw the editor again (a datasource list arrived)
  //   emptyLabel      the first option of the source list; language: list only sources that run it
  function cardBindingCtx(card, row, model) {
    return {
      id: `bf-${row.id}`,
      title: row.title,
      own: row.id === 'default' || hasOwn(model, row.id),
      get: () => (isPlainObject(model[row.id]) ? model[row.id] : {}),
      set: (binding) => {
        if (binding) model[row.id] = binding;
        else delete model[row.id];
        delete state.bindingChecks[row.id];
        commitBindings(card);
      },
      setField: (field, value) => {
        if (!isPlainObject(model[row.id])) model[row.id] = {};
        const text = String(value || '').trim();
        if (text) model[row.id][field] = text;
        else delete model[row.id][field];
        setJson(state.editors[`ed_${card.id}`], model);
        updateCardDirty(card);
      },
      rerender: () => renderBindingsBody(card),
      emptyLabel: row.id === 'default' ? LL.t('settings.binding.pick_source') : LL.t('settings.binding.as_default'),
      inheritNote: row.id === 'default' ? '' : LL.t('settings.binding.as_default_summary', { summary: bindingSummary(model.default) })
    };
  }
  function withDatasource(sourceId, ds) {
    const binding = { source: String(sourceId || '') };
    if (ds) {
      binding.datasource_uid = ds.uid;
      binding.datasource_name = ds.name;
      if (ds.mode === 'influxql' && ds.database) binding.database = ds.database;
      if (ds.mode === 'flux' && ds.bucket) binding.bucket = ds.bucket;
    }
    return binding;
  }
  // Whether a source of this type runs queries in the language (Grafana runs whatever its datasource does).
  function sourceRunsLanguage(type, language) {
    if (!language) return true;
    if (type === 'prometheus') return language === 'promql';
    if (type === 'influxdb') return language === 'flux';
    return true;
  }
  function bindingField(ctx, field, value) {
    const meta = BINDING_FIELDS[field];
    const wrap = makeEl('div', 'binding-field');
    const id = `${ctx.id}-${field}`;
    const label = makeEl('label', '', meta.label);
    label.htmlFor = id;
    const input = makeEl('input');
    input.type = 'text';
    input.id = id;
    input.value = value || '';
    input.placeholder = meta.placeholder;
    input.addEventListener('change', () => ctx.setField(field, input.value));
    wrap.append(label, input);
    return wrap;
  }
  function bindingFieldGrid(ctx, binding, fields) {
    const grid = makeEl('div', 'binding-extra');
    fields.forEach((field) => grid.appendChild(bindingField(ctx, field, binding[field])));
    return grid;
  }
  function datasourceSelect(ctx, binding, items) {
    const select = makeEl('select');
    select.setAttribute('aria-label', LL.t('settings.binding.ds_aria', { domain: ctx.title }));
    const current = findDatasource(items, binding);
    const placeholder = makeEl('option', '', LL.t('settings.binding.ds_pick'));
    placeholder.value = '';
    select.appendChild(placeholder);
    const stored = binding.datasource_name || binding.datasource_uid;
    if (!current && stored) {
      const missing = makeEl('option', '', LL.t('settings.binding.ds_missing', { name: stored }));
      missing.value = '__missing__';
      select.appendChild(missing);
    }
    [['prometheus', 'Prometheus'], ['influxdb', 'InfluxDB']].forEach(([type, label]) => {
      const group = items.filter((ds) => ds.type === type);
      if (!group.length) return;
      const optgroup = document.createElement('optgroup');
      optgroup.label = label;
      group.forEach((ds) => {
        const option = makeEl('option', '', datasourceLabel(ds));
        option.value = ds.uid;
        optgroup.appendChild(option);
      });
      select.appendChild(optgroup);
    });
    select.value = current ? current.uid : (stored ? '__missing__' : '');
    select.addEventListener('change', () => {
      if (select.value === '__missing__') return;
      ctx.set(withDatasource(binding.source, items.find((ds) => ds.uid === select.value) || null));
    });
    return { select, current };
  }
  function grafanaExtraFields(binding, current) {
    const fields = [];
    if (current && current.mode === 'influxql') fields.push('database');
    if (current && current.mode === 'flux') fields.push('bucket');
    ['database', 'bucket'].forEach((field) => { if (binding[field] && !fields.includes(field)) fields.push(field); });
    return fields;
  }
  function grafanaParams(ctx, binding, sourceId, box) {
    const list = datasourceList(sourceId);
    if (list.status === 'loading') {
      box.appendChild(makeEl('div', 'binding-note', LL.t('settings.binding.ds_loading')));
      return;
    }
    if (list.status === 'error') {
      const note = makeEl('div', 'binding-error');
      note.append(makeEl('span', '', LL.t('settings.binding.ds_unavailable', { error: list.message })), makeBtn(LL.t('settings.action.retry'), () => {
        delete state.grafanaDatasources[sourceId];
        ctx.rerender();
      }, 'btn btn-sm'));
      box.append(note, bindingFieldGrid(ctx, binding, ['datasource_name', 'datasource_uid', 'database', 'bucket']));
      return;
    }
    if (!list.items.length) box.appendChild(makeEl('div', 'binding-error', LL.t('settings.binding.ds_none')));
    const { select, current } = datasourceSelect(ctx, binding, list.items);
    box.appendChild(select);
    const extra = grafanaExtraFields(binding, current);
    if (extra.length) box.appendChild(bindingFieldGrid(ctx, binding, extra));
  }
  function bindingParams(ctx) {
    const box = makeEl('div', 'binding-params');
    if (!ctx.own) {
      if (ctx.inheritNote) box.appendChild(makeEl('div', 'binding-note', ctx.inheritNote));
      return box;
    }
    const binding = ctx.get();
    const sourceId = String(binding.source || '');
    const entry = sourceId ? savedSource(sourceId) : null;
    if (!sourceId) box.appendChild(makeEl('div', 'binding-note', LL.t('settings.binding.pick_source')));
    else if (!entry) box.appendChild(makeEl('div', 'binding-error', LL.t('settings.binding.source_not_saved')));
    else if (entry.type === 'prometheus') box.appendChild(makeEl('div', 'binding-note', LL.t('settings.binding.prometheus_direct')));
    else if (entry.type === 'influxdb') box.appendChild(bindingFieldGrid(ctx, binding, ['bucket']));
    else grafanaParams(ctx, binding, sourceId, box);
    return box;
  }
  function bindingSourceSelect(ctx) {
    const current = ctx.own ? String(ctx.get().source || '') : '';
    const select = makeEl('select');
    select.id = `${ctx.id}-source`;
    select.setAttribute('aria-label', LL.t('settings.binding.source_aria', { domain: ctx.title }));
    const first = makeEl('option', '', ctx.emptyLabel);
    first.value = '';
    select.appendChild(first);
    const catalog = savedCatalog();
    Object.keys(catalog).forEach((id) => {
      const entry = isPlainObject(catalog[id]) ? catalog[id] : {};
      if (id !== current && !sourceRunsLanguage(String(entry.type || ''), ctx.language)) return;
      const title = String(entry.title || id);
      const typeLabel = sourceTypeLabel(entry.type);
      const option = makeEl('option', '', title.toLowerCase().includes(typeLabel.toLowerCase()) ? title : `${title} · ${typeLabel}`);
      option.value = id;
      select.appendChild(option);
    });
    if (current && !hasOwn(catalog, current)) {
      const missing = makeEl('option', '', LL.t('settings.binding.source_missing_option', { id: current }));
      missing.value = current;
      select.appendChild(missing);
    }
    select.value = current;
    select.addEventListener('change', () => ctx.set(select.value ? { source: select.value } : null));
    const cell = makeEl('div', 'binding-source');
    cell.appendChild(select);
    return cell;
  }
  // Rows of a domain's queries (the area scope the bindings card edits) that read a source of their own.
  function rowOverrideCount(domain) {
    const queries = isPlainObject(state.config.queries) ? state.config.queries : {};
    const block = isPlainObject(queries[domain]) ? queries[domain] : {};
    const labels = Array.isArray(block.labels) ? block.labels : [];
    const sources = isPlainObject(block.sources) ? block.sources : {};
    return labels.filter((label) => isPlainObject(sources[label]) && sources[label].source).length;
  }
  function bindingRow(card, row, model) {
    const ctx = cardBindingCtx(card, row, model);
    const wrap = makeEl('div', 'binding-row');
    const domain = makeEl('div', 'binding-domain');
    domain.appendChild(makeEl('div', '', row.title));
    if (row.note) domain.appendChild(makeEl('div', 'dim', row.note));
    const overridden = row.id === 'default' ? 0 : rowOverrideCount(row.id);
    if (overridden) domain.appendChild(makeEl('div', 'dim', LL.t('settings.binding.rows_override', { n: overridden })));
    const result = makeEl('div', 'conn-result');
    result.setAttribute('role', 'status');
    const actions = makeEl('div', 'binding-actions');
    actions.appendChild(makeBtn(LL.t('settings.action.check'), () => runCheck(card, { domain: row.id }, result, 'bindingChecks', row.id), 'btn btn-sm'));
    wrap.append(domain, bindingSourceSelect(ctx), bindingParams(ctx), actions, result);
    restoreCheck('bindingChecks', row.id, result);
    return wrap;
  }
  function bindingHeader() {
    const head = makeEl('div', 'binding-row binding-head');
    head.setAttribute('aria-hidden', 'true');
    [LL.t('settings.binding.col_domain'), LL.t('settings.binding.col_source'), LL.t('settings.binding.col_params'), ''].forEach((text) => head.appendChild(makeEl('div', '', text)));
    return head;
  }
  function renderBindingsBody(card) {
    const host = document.querySelector(`#card-${card.id} .card-body`);
    if (!host) return;
    const model = isPlainObject(state.models[card.id]) ? state.models[card.id] : {};
    state.models[card.id] = model;
    host.replaceChildren();
    if (!Object.keys(savedCatalog()).length) {
      host.appendChild(makeEl('p', 'dim', LL.t('settings.binding.need_catalog')));
      return;
    }
    const list = makeEl('div', 'binding-list');
    list.appendChild(bindingHeader());
    bindingRows().forEach((row) => list.appendChild(bindingRow(card, row, model)));
    host.appendChild(list);
  }

  // ---- SLA -----------------------------------------------------------------
  const SLA_FIELDS = [
    { key: 'target_rps', queryKey: 'max_performance_query', domain: 'lt_framework', label: LL.t('settings.sla.field.target_rps'), unit: 'RPS', term: 'target_rps', desc: LL.t('settings.sla.field.target_rps_desc') },
    { key: 'max_error_rate_pct', queryKey: 'error_rate_query', domain: 'lt_framework', label: LL.t('settings.sla.field.max_error_rate_pct'), unit: '%', term: 'error_rate' },
    { key: 'max_p95_ms', queryKey: 'p95_query', domain: 'lt_framework', label: LL.t('settings.sla.field.max_p95_ms'), unit: LL.t('settings.sla.unit_ms'), term: 'p95' },
    { key: 'max_p99_ms', queryKey: 'p99_query', domain: 'lt_framework', label: LL.t('settings.sla.field.max_p99_ms'), unit: LL.t('settings.sla.unit_ms') },
    { key: 'max_cpu_pct', queryKey: 'cpu_query', domain: 'hard_resources', label: LL.t('settings.sla.field.max_cpu_pct'), unit: '%' },
    { key: 'max_memory_pct', queryKey: 'memory_query', domain: 'hard_resources', label: LL.t('settings.sla.field.max_memory_pct'), unit: '%' }
  ];
  // One explanation for every criterion except the target RPS (it used to repeat under each row).
  const SLA_SHARED_DESC = LL.t('settings.sla.shared_desc');
  const SLA_PEAK_FIELDS = [
    { key: 'min_stable_minutes', label: LL.t('settings.sla.field.min_stable_minutes'), type: 'number', step: '0.5' },
    { key: 'target_margin_pct', label: LL.t('settings.sla.field.target_margin_pct'), type: 'number', step: '1', hint: LL.t('settings.sla.field.target_margin_pct_hint') },
    { key: 'step_detection_preset', label: LL.t('settings.sla.field.step_detection_preset'), type: 'select', options: null },
    { key: 'step_detection_enabled', label: LL.t('settings.sla.field.step_detection_enabled'), type: 'checkbox', term: 'step_profile' },
    { key: 'target_rps_allow_peak_fallback', label: LL.t('settings.sla.field.target_rps_allow_peak_fallback'), type: 'checkbox', term: 'peak_max' },
    { key: 'debug_peak_logging', label: LL.t('settings.sla.field.debug_peak_logging'), type: 'checkbox' }
  ];
  const SLA_FORECAST_FIELDS = [
    { key: 'mean_latency_query', label: LL.t('settings.sla.field.mean_latency_query'), type: 'query', domain: 'lt_framework' },
    { key: 'vus_query', label: LL.t('settings.sla.field.vus_query'), type: 'query', domain: 'lt_framework' },
    { key: 'load_model', label: LL.t('settings.sla.field.load_model'), type: 'select', options: null, hint: LL.t('settings.sla.field.load_model_hint') },
    { key: 'service_cpu_query', label: LL.t('settings.sla.field.service_cpu_query'), type: 'query', domain: 'jvm', hint: LL.t('settings.sla.field.service_cpu_query_hint') }
  ];
  const SLA_PRESETS = [{ v: 'strict', l: LL.t('settings.sla.preset.strict') }, { v: 'balanced', l: LL.t('settings.sla.preset.balanced') }, { v: 'lenient', l: LL.t('settings.sla.preset.lenient') }];
  const LATENCY_UNIT_OPTIONS = [{ v: 'ms', l: LL.t('settings.sla.latency_unit.ms') }, { v: 's', l: LL.t('settings.sla.latency_unit.s') }];
  const LOAD_MODEL_OPTIONS = [{ v: 'open', l: LL.t('settings.sla.load_model.open') }, { v: 'closed', l: LL.t('settings.sla.load_model.closed') }];

  function slaModel() { return state.models.sla; }
  function slaSavedKey() { return `sla:${state.slaScope}`; }
  function effectiveSla(scope) {
    const base = isPlainObject(state.config.sla) ? state.config.sla : {};
    if (!scope) return clone(base);
    const override = isPlainObject((state.config.service_sla || {})[scope]) ? state.config.service_sla[scope] : {};
    return clone(deepMerge(base, override));
  }
  function domainBlock(domain, scope) {
    const queries = isPlainObject(state.config.queries) ? state.config.queries : {};
    let block = isPlainObject(queries[domain]) ? queries[domain] : {};
    if (scope) {
      const svcQueries = (state.config.service_queries || {})[scope];
      if (isPlainObject(svcQueries) && isPlainObject(svcQueries[domain])) block = deepMerge(block, svcQueries[domain]);
    }
    return block;
  }
  function domainLabels(domain, scope) {
    const block = domainBlock(domain, scope);
    return Array.isArray(block.labels) ? block.labels.map(String).filter(Boolean) : [];
  }
  // Query the reports take for an empty SLA reference: the first row of the domain whose role fills it.
  function roleRefLabel(slaField, domain, scope) {
    const roles = (packUi.catalog && packUi.catalog.roles) || {};
    const wanted = Object.keys(roles).filter((name) => roles[name].sla_field === slaField);
    if (!wanted.length) return '';
    const block = domainBlock(domain, scope);
    const blockRoles = isPlainObject(block.roles) ? block.roles : {};
    return (Array.isArray(block.labels) ? block.labels : []).find((label) => wanted.includes(blockRoles[label])) || '';
  }
  function queryOptions(labels, current, autoLabel) {
    const options = labels.map((label) => ({ v: label, l: label }));
    // A reference to a query that no longer exists stays visible, marked, until it is changed.
    if (current && !labels.includes(current)) options.push({ v: current, l: LL.t('settings.sla.query_missing_option', { name: current }) });
    const empty = autoLabel ? LL.t('settings.sla.query_auto', { name: autoLabel }) : LL.t('settings.sla.query_unset');
    return [{ v: '', l: empty }].concat(options);
  }
  // Warning under a query selector whose saved reference names a missing query; '' when it is fine.
  function deadRefText(labels, current, autoLabel) {
    if (!current || labels.includes(current)) return '';
    return autoLabel
      ? LL.t('settings.sla.dead_ref_auto', { name: current, auto: autoLabel })
      : LL.t('settings.sla.dead_ref', { name: current });
  }
  function renderSlaForm(options) {
    const skipJson = !!(options && options.skipJson);
    const model = slaModel();
    const rows = $('slaRows');
    rows.innerHTML = '';
    SLA_FIELDS.forEach((field) => {
      const value = model[field.key];
      const enabled = value !== null && value !== undefined && value !== '';
      const currentQuery = String(model[field.queryKey] || '');
      const labels = domainLabels(field.domain, state.slaScope);
      const autoLabel = roleRefLabel(field.queryKey, field.domain, state.slaScope);
      const options = queryOptions(labels, currentQuery, autoLabel);
      const dead = deadRefText(labels, currentQuery, autoLabel);
      const row = document.createElement('div');
      row.className = 'sla-row';
      row.innerHTML = `
        <input type="checkbox" id="sla-on-${field.key}" ${enabled ? 'checked' : ''} aria-label="${LL.tHtml('settings.sla.check_aria', { name: field.label })}" />
        <div><label for="sla-val-${field.key}" class="sla-name" style="margin:0">${field.term ? ui.term(field.term, field.label) : esc(field.label)}</label></div>
        <input type="number" id="sla-val-${field.key}" value="${enabled ? esc(value) : ''}" ${enabled ? '' : 'disabled'} step="any" />
        <span class="dim">${esc(field.unit)}</span>
        <select id="sla-query-${field.key}" aria-label="${LL.tHtml('settings.sla.query_aria', { name: field.label })}">${options.map((o) => `<option value="${esc(o.v)}" ${o.v === currentQuery ? 'selected' : ''}>${esc(o.l)}</option>`).join('')}</select>
        ${field.desc ? `<div class="sla-desc">${esc(field.desc)}</div>` : ''}
        ${dead ? `<div class="sla-desc sla-dead" role="status">${esc(dead)}</div>` : ''}`;
      const check = row.querySelector(`#sla-on-${field.key}`);
      const num = row.querySelector(`#sla-val-${field.key}`);
      const query = row.querySelector(`#sla-query-${field.key}`);
      check.addEventListener('change', () => {
        num.disabled = !check.checked;
        if (!check.checked) { model[field.key] = null; }
        else { const n = Number(num.value); model[field.key] = Number.isFinite(n) && num.value !== '' ? n : null; num.focus(); }
        afterSlaChange();
      });
      num.addEventListener('input', () => {
        const n = Number(num.value);
        model[field.key] = num.value === '' || !Number.isFinite(n) ? null : n;
        afterSlaChange();
      });
      query.addEventListener('change', () => {
        model[field.queryKey] = query.value;
        afterSlaChange();
        renderSlaForm({ skipJson: true });
      });
      rows.appendChild(row);
      if (field.key === 'target_rps') rows.appendChild(makeEl('p', 'sla-common-note', SLA_SHARED_DESC));
    });
    renderSlaUnitRow(model);
    renderSlaFieldGroup($('slaPeakFields'), SLA_PEAK_FIELDS, model);
    renderSlaFieldGroup($('slaForecastFields'), SLA_FORECAST_FIELDS, model);
    ui.applyTerms($('sec-sla'));
    if (!skipJson) setJson(state.editors.ed_sla, model);
    updateSlaDirty();
  }
  function slaFieldOptions(f) {
    if (f.key === 'step_detection_preset') return SLA_PRESETS;
    if (f.key === 'load_model') return LOAD_MODEL_OPTIONS;
    if (f.key === 'latency_unit') return LATENCY_UNIT_OPTIONS;
    return f.options || [];
  }
  // One settings control; selects and numbers go into the grid, checkboxes into the list below it.
  function slaField(f, model) {
    const wrap = document.createElement('div');
    wrap.className = f.type === 'checkbox' ? 'sla-check' : 'sla-field';
    const id = `sla-peak-${f.key}`;
    const value = model[f.key];
    const labelHtml = f.term ? ui.term(f.term, f.label) : esc(f.label);
    let dead = '';
    if (f.type === 'checkbox') {
      wrap.innerHTML = `<label class="checkbox-row"><input type="checkbox" id="${id}" ${value ? 'checked' : ''}/> <span>${labelHtml}</span></label>`;
    } else if (f.type === 'select' || f.type === 'query') {
      const current = value === undefined || value === null ? '' : String(value);
      let options = slaFieldOptions(f);
      if (f.type === 'query') {
        const labels = domainLabels(f.domain, state.slaScope);
        const autoLabel = roleRefLabel(f.key, f.domain, state.slaScope);
        options = queryOptions(labels, current, autoLabel);
        dead = deadRefText(labels, current, autoLabel);
      }
      wrap.innerHTML = `<label for="${id}">${labelHtml}</label><select id="${id}">${options.map((o) => `<option value="${esc(o.v)}" ${o.v === current ? 'selected' : ''}>${esc(o.l)}</option>`).join('')}</select>`
        + (dead ? `<div class="field-hint warn" role="status">${esc(dead)}</div>` : '')
        + (f.hint ? `<div class="field-hint">${esc(f.hint)}</div>` : '');
    } else {
      wrap.innerHTML = `<label for="${id}">${labelHtml}</label><input type="number" id="${id}" step="${f.step || 'any'}" value="${value === undefined || value === null ? '' : esc(value)}" />`
        + (f.hint ? `<div class="field-hint">${esc(f.hint)}</div>` : '');
    }
    const control = wrap.querySelector('input, select');
    control.addEventListener('change', () => {
      if (f.type === 'checkbox') model[f.key] = control.checked;
      else if (f.type === 'number') { const n = Number(control.value); model[f.key] = control.value === '' || !Number.isFinite(n) ? null : n; }
      else model[f.key] = control.value;
      afterSlaChange();
      if (f.type === 'query') renderSlaForm({ skipJson: true });
    });
    return wrap;
  }
  function renderSlaFieldGroup(host, fields, model) {
    host.replaceChildren();
    const grid = makeEl('div', 'form-grid sla-grid');
    const checks = makeEl('div', 'sla-checks');
    fields.forEach((f) => (f.type === 'checkbox' ? checks : grid).appendChild(slaField(f, model)));
    if (grid.childElementCount) host.appendChild(grid);
    if (checks.childElementCount) host.appendChild(checks);
  }
  // Latency units belong to the P95/P99 rows above, so they sit right under the criteria.
  function renderSlaUnitRow(model) {
    const host = $('slaUnitRow');
    host.replaceChildren();
    const f = { key: 'latency_unit', label: LL.t('settings.sla.field.latency_unit'), type: 'select' };
    const field = slaField(f, model);
    field.className = 'sla-field sla-unit-field';
    host.appendChild(field);
  }
  function afterSlaChange() {
    setJson(state.editors.ed_sla, slaModel());
    updateSlaDirty();
  }
  function updateSlaDirty() {
    const dirty = stable(slaModel()) !== state.saved[slaSavedKey()];
    $('slaSaveBtn').disabled = !dirty;
    markDirty('sla:model', state.slaScope ? `SLA (${state.slaScope})` : 'SLA', dirty);
  }
  function loadSlaScope(scope) {
    state.slaScope = scope;
    state.models.sla = effectiveSla(scope);
    state.saved[slaSavedKey()] = stable(state.models.sla);
    renderSlaForm();
  }
  async function saveSla() {
    setStatus('slaStatus', LL.t('settings.status.saving'));
    try {
      const data = await postConfig(configBody('sla', slaModel(), state.slaScope));
      if (state.slaScope) { state.config.service_sla = state.config.service_sla || {}; state.config.service_sla[state.slaScope] = clone(slaModel()); }
      else state.config.sla = clone(slaModel());
      state.saved[slaSavedKey()] = stable(slaModel());
      updateSlaDirty();
      setStatus('slaStatus', LL.t('settings.status.saved'), 'ok');
      ui.toast(LL.t('settings.sla.saved'), { tone: 'ok' });
      showWarnings(data.warnings);
      return true;
    } catch (e) {
      setStatus('slaStatus', LL.t('settings.status.error', { error: e.message }), 'error');
      return false;
    }
  }

  // ---- structured context and queries (JSON stays the stored shape) ------
  const QUERY_DOMAINS = ['jvm', 'database', 'kafka', 'microservices', 'hard_resources', 'lt_framework', 'lt_generators'];
  const LT_QUERY_PRIORITY = ['influxql', 'promql', 'flux'];
  const QUERY_LANG = {
    promql: { query: 'promql_queries', keys: 'label_keys_list', label: 'PromQL' },
    influxql: { query: 'influxql_queries', keys: 'label_tag_keys_list', label: 'InfluxQL' },
    flux: { query: 'flux_queries', keys: 'label_tag_keys_list', label: 'Flux' }
  };
  // Positional arrays of one domain; a row operation is applied to all of them.
  const QUERY_PARALLEL_KEYS = ['labels', 'promql_queries', 'label_keys_list', 'influxql_queries', 'flux_queries', 'label_tag_keys_list'];
  const SLA_QUERY_TITLES = {
    max_performance_query: LL.t('settings.sla.ref.max_performance_query'), error_rate_query: LL.t('settings.sla.ref.error_rate_query'), p95_query: 'P95',
    p99_query: 'P99', cpu_query: 'CPU', memory_query: LL.t('settings.sla.ref.memory_query'),
    mean_latency_query: LL.t('settings.sla.ref.mean_latency_query'), vus_query: LL.t('settings.sla.ref.vus_query'),
    service_cpu_query: LL.t('settings.sla.ref.service_cpu_query')
  };
  let structuredFieldSeq = 0;

  function makeEl(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }
  function makeBtn(text, onClick, className) {
    const btn = makeEl('button', className || 'btn', text);
    btn.type = 'button';
    btn.addEventListener('click', onClick);
    return btn;
  }
  function nextFieldId() {
    structuredFieldSeq += 1;
    return `sf-${structuredFieldSeq}`;
  }
  function splitList(text, separator) {
    return String(text || '').split(separator || ',').map((s) => s.trim()).filter(Boolean);
  }

  function commitStructured(cfgKey, data, editorId, saveBtn, menu, label) {
    state.models[cfgKey] = data;
    setJson(state.editors[editorId], data);
    const dirty = stable(data) !== state.saved[cfgKey];
    if ($(saveBtn)) $(saveBtn).disabled = !dirty;
    markDirty(`${menu}:${cfgKey}`, label, dirty);
    if (cfgKey === 'queries') updateWizardNav();
  }

  function slaLabelsUsing(name) {
    if (!name) return [];
    const sla = Object.assign({}, state.config.sla || {}, state.models.sla || {});
    return Object.keys(sla).filter((key) => key.endsWith('_query') && sla[key] === name);
  }
  function slaUsageText(keys) {
    return keys.map((key) => SLA_QUERY_TITLES[key] || key).join(', ');
  }
  function renameSlaLabel(from, to) {
    if (!from || from === to) return;
    const used = slaLabelsUsing(from);
    if (!used.length) return;
    if (!window.confirm(LL.t('settings.sla.rename_confirm', { from, usage: slaUsageText(used), to }))) return;
    used.forEach((key) => {
      if (state.models.sla) state.models.sla[key] = to;
      if (state.config.sla) state.config.sla[key] = to;
    });
    if (state.editors.ed_sla && state.models.sla) setJson(state.editors.ed_sla, state.models.sla);
    ui.toast(LL.t('settings.sla.rename_done'), { tone: 'warn', timeout: 8000 });
  }

  // ---- system context form -------------------------------------------------
  const LEVEL_OPTIONS = [['', '—'], ['high', LL.t('settings.ctx.level.high')], ['medium', LL.t('settings.ctx.level.medium')], ['low', LL.t('settings.ctx.level.low')]];
  const DEP_KIND_OPTIONS = [['', '—'], ['sync', LL.t('settings.ctx.dep.sync')], ['async', LL.t('settings.ctx.dep.async')]];
  const ARCH_STYLES = [['', '—'], ['microservices', LL.t('settings.ctx.arch.microservices')], ['monolith', LL.t('settings.ctx.arch.monolith')], ['modular-monolith', LL.t('settings.ctx.arch.modular_monolith')], ['event-driven', LL.t('settings.ctx.arch.event_driven')], ['serverless', 'Serverless']];
  const CONTEXT_LIST_FIELDS = new Set(['technologies', 'used_by', 'steps', 'success_signals']);
  const CONTEXT_GROUPS = [
    { key: 'system', title: LL.t('settings.ctx.group.system'), open: true, hint: LL.t('settings.ctx.group.system_hint') },
    {
      key: 'architecture', title: LL.t('settings.ctx.group.architecture'),
      hint: LL.t('settings.ctx.group.architecture_hint'),
      lists: [
        ['components', LL.t('settings.ctx.components.title'), [['name', LL.t('settings.ctx.components.name')], ['role', LL.t('settings.ctx.components.role')], ['criticality', LL.t('settings.ctx.components.criticality'), LEVEL_OPTIONS], ['technologies', LL.t('settings.ctx.components.technologies')]]],
        ['dependencies', LL.t('settings.ctx.dependencies.title'), [['from', LL.t('settings.ctx.dependencies.from')], ['to', LL.t('settings.ctx.dependencies.to')], ['kind', LL.t('settings.ctx.dependencies.kind'), DEP_KIND_OPTIONS], ['purpose', LL.t('settings.ctx.dependencies.purpose')]]],
        ['data_stores', LL.t('settings.ctx.data_stores.title'), [['id', LL.t('settings.ctx.data_stores.id')], ['type', LL.t('settings.ctx.data_stores.type')], ['purpose', LL.t('settings.ctx.data_stores.purpose')], ['used_by', LL.t('settings.ctx.data_stores.used_by')]]]
      ]
    },
    {
      key: 'load_model', title: LL.t('settings.ctx.group.load_model'),
      hint: LL.t('settings.ctx.group.load_model_hint'),
      lists: [
        ['entrypoints', LL.t('settings.ctx.entrypoints.title'), [['name', LL.t('settings.ctx.entrypoints.name')], ['kind', LL.t('settings.ctx.entrypoints.kind')], ['business_priority', LL.t('settings.ctx.entrypoints.business_priority'), LEVEL_OPTIONS]]],
        ['critical_user_flows', LL.t('settings.ctx.critical_user_flows.title'), [['name', LL.t('settings.ctx.critical_user_flows.name')], ['steps', LL.t('settings.ctx.critical_user_flows.steps')], ['success_signals', LL.t('settings.ctx.critical_user_flows.success_signals')]]]
      ],
      lines: [['expected_hotspots', LL.t('settings.ctx.line.expected_hotspots')]]
    },
    {
      key: 'operational_context', title: LL.t('settings.ctx.group.operational_context'),
      hint: LL.t('settings.ctx.group.operational_context_hint'),
      // The summary line of the group shows the part of a title before ' — '.
      lines: [
        ['normal_degradation_rules', LL.t('settings.ctx.line.normal_degradation_rules')],
        ['known_risks', LL.t('settings.ctx.line.known_risks')],
        ['known_constraints', LL.t('settings.ctx.line.known_constraints')],
        ['analysis_focus', LL.t('settings.ctx.line.analysis_focus')]
      ]
    }
  ];

  // A service with its own context replaces the area context as a whole (lists are not merged).
  function serviceContext(scope) {
    const map = isPlainObject(state.config.service_system_context) ? state.config.service_system_context : {};
    return scope && isPlainObject(map[scope]) ? map[scope] : null;
  }
  function effectiveContext(scope) {
    return clone(serviceContext(scope) || state.config.system_context || {});
  }
  function updateContextOverrideNote() {
    const note = $('contextOverrideNote');
    if (!note) return;
    if (!state.contextScope) note.textContent = state.area ? LL.t('settings.ctx.note.area') : LL.t('settings.ctx.note.global');
    else note.textContent = serviceContext(state.contextScope) ? LL.t('settings.ctx.note.service_own') : LL.t('settings.ctx.note.service_inherits');
  }
  function contextSnapshot() {
    return isPlainObject(state.models.system_context) ? state.models.system_context : {};
  }
  function writeContext(mutator) {
    const next = clone(contextSnapshot());
    mutator(next);
    commitStructured('system_context', next, 'ed_system_context', 'contextSaveBtn', 'context', LL.t('settings.menu.context'));
  }
  function setContextPath(group, key, value) {
    writeContext((ctx) => { ctx[group] = Object.assign({}, isPlainObject(ctx[group]) ? ctx[group] : {}, { [key]: value }); });
  }
  function currentContextList(group, key) {
    const section = contextSnapshot()[group];
    const list = isPlainObject(section) && Array.isArray(section[key]) ? section[key] : [];
    return list.map((item) => (isPlainObject(item) ? Object.assign({}, item) : {}));
  }

  function renderContextForm() {
    const host = $('contextForm');
    if (!host) return;
    const ctx = contextSnapshot();
    host.innerHTML = '';
    host.appendChild(contextEnabledToggle(ctx));
    CONTEXT_GROUPS.forEach((group) => host.appendChild(contextGroup(group, ctx)));
    markContextImport(host);
  }

  function contextEnabledToggle(ctx) {
    const label = makeEl('label', 'checkbox-row');
    const input = makeEl('input');
    input.type = 'checkbox';
    input.checked = ctx.enabled !== false;
    input.addEventListener('change', () => writeContext((next) => { next.enabled = input.checked; }));
    label.append(input, makeEl('span', '', ` ${LL.t('settings.ctx.enabled')}`));
    return label;
  }

  function contextGroup(group, ctx) {
    const details = makeEl('details', 'report-collapsible ctx-group');
    const remembered = state.contextOpen && Object.prototype.hasOwnProperty.call(state.contextOpen, group.key);
    details.open = remembered ? state.contextOpen[group.key] : !!group.open;
    details.addEventListener('toggle', () => { state.contextOpen = Object.assign({}, state.contextOpen, { [group.key]: details.open }); });
    const summary = makeEl('summary');
    summary.append(makeEl('span', '', group.title), makeEl('small', 'report-collapsible-subtitle', contextGroupSummary(group, ctx)));
    const body = makeEl('div', 'report-collapsible-body');
    body.appendChild(makeEl('p', 'help', group.hint));
    const section = isPlainObject(ctx[group.key]) ? ctx[group.key] : {};
    if (group.key === 'system') body.appendChild(systemFields(section));
    if (group.key === 'architecture') {
      const style = selectField(LL.t('settings.ctx.arch.style'), section.style, ARCH_STYLES, (v) => setContextPath('architecture', 'style', v));
      style.dataset.ctxPath = 'architecture.style';
      body.appendChild(style);
    }
    (group.lists || []).forEach(([key, title, columns]) => body.appendChild(contextTable(group.key, key, title, columns, section[key])));
    (group.lines || []).forEach(([key, title]) => body.appendChild(contextLines(group.key, key, title, section[key])));
    details.append(summary, body);
    return details;
  }

  function contextGroupSummary(group, ctx) {
    const section = isPlainObject(ctx[group.key]) ? ctx[group.key] : {};
    if (group.key === 'system') return section.name ? String(section.name) : LL.t('settings.ctx.summary.unfilled');
    const count = (key) => (Array.isArray(section[key]) ? section[key].length : 0);
    const parts = (group.lists || []).map(([key, title]) => `${title.toLowerCase()}: ${count(key)}`);
    (group.lines || []).forEach(([key, title]) => { if (count(key)) parts.push(`${title.split(' — ')[0].toLowerCase()}: ${count(key)}`); });
    return parts.join(' · ') || LL.t('settings.ctx.summary.empty');
  }

  function systemFields(section) {
    const box = makeEl('div');
    const grid = makeEl('div', 'form-grid two');
    const field = (label, key, multiline) => {
      const wrap = textField(label, section[key], (v) => setContextPath('system', key, v), multiline);
      wrap.dataset.ctxPath = `system.${key}`;
      return wrap;
    };
    grid.append(field(LL.t('settings.ctx.system.name'), 'name'), field(LL.t('settings.ctx.system.domain'), 'domain'));
    box.append(grid, field(LL.t('settings.ctx.system.description'), 'description', true), field(LL.t('settings.ctx.system.test_goal'), 'test_goal', true));
    return box;
  }

  function textField(label, value, onInput, multiline) {
    const wrap = makeEl('div', 'ctx-field');
    const id = nextFieldId();
    const lab = makeEl('label', '', label);
    lab.htmlFor = id;
    const input = makeEl(multiline ? 'textarea' : 'input');
    input.id = id;
    if (multiline) input.rows = 3; else input.type = 'text';
    input.value = value || '';
    input.addEventListener('input', () => onInput(input.value));
    wrap.append(lab, input);
    return wrap;
  }

  function selectControl(value, options, onChange) {
    const select = makeEl('select');
    const opts = options.slice();
    if (value && !opts.some(([v]) => v === value)) opts.push([value, value]);
    opts.forEach(([v, text]) => {
      const option = makeEl('option', '', text);
      option.value = v;
      option.selected = v === (value || '');
      select.appendChild(option);
    });
    select.addEventListener('change', () => onChange(select.value));
    return select;
  }

  function selectField(label, value, options, onChange) {
    const wrap = makeEl('div', 'ctx-field ctx-field-short');
    const id = nextFieldId();
    const lab = makeEl('label', '', label);
    lab.htmlFor = id;
    const select = selectControl(value, options, onChange);
    select.id = id;
    wrap.append(lab, select);
    return wrap;
  }

  function contextTable(groupKey, key, title, columns, items) {
    const rows = Array.isArray(items) ? items : [];
    const box = makeEl('div', 'struct-block');
    box.dataset.ctxPath = `${groupKey}.${key}`;
    const head = makeEl('div', 'struct-head');
    head.append(makeEl('h4', '', title), makeBtn(LL.t('settings.action.add'), () => {
      setContextPath(groupKey, key, currentContextList(groupKey, key).concat([{}]));
      renderContextForm();
    }));
    box.appendChild(head);
    if (!rows.length) {
      box.appendChild(makeEl('p', 'dim', LL.t('settings.ctx.table.empty')));
      return box;
    }
    const table = makeEl('table', 'data-table ctx-table');
    const headRow = makeEl('tr');
    columns.forEach(([, label]) => headRow.appendChild(makeEl('th', '', label)));
    headRow.appendChild(makeEl('th'));
    const thead = makeEl('thead');
    thead.appendChild(headRow);
    const tbody = makeEl('tbody');
    rows.forEach((item, index) => tbody.appendChild(contextTableRow(groupKey, key, columns, isPlainObject(item) ? item : {}, index)));
    table.append(thead, tbody);
    box.appendChild(table);
    return box;
  }

  function contextTableRow(groupKey, key, columns, item, index) {
    const tr = makeEl('tr');
    const update = (field, value) => {
      const next = currentContextList(groupKey, key);
      if (!next[index]) return;
      next[index][field] = CONTEXT_LIST_FIELDS.has(field) ? splitList(value) : value;
      setContextPath(groupKey, key, next);
    };
    columns.forEach(([field, label, options]) => {
      const td = makeEl('td');
      const raw = item[field];
      if (options) {
        const select = selectControl(raw ? String(raw) : '', options, (v) => update(field, v));
        select.setAttribute('aria-label', label);
        td.appendChild(select);
      } else {
        const input = makeEl('input');
        input.type = 'text';
        input.setAttribute('aria-label', label);
        input.value = Array.isArray(raw) ? raw.join(', ') : (raw || '');
        input.addEventListener('change', () => update(field, input.value));
        td.appendChild(input);
      }
      tr.appendChild(td);
    });
    const actions = makeEl('td', 'ctx-row-actions');
    const remove = makeBtn('×', () => {
      setContextPath(groupKey, key, currentContextList(groupKey, key).filter((_, i) => i !== index));
      renderContextForm();
    });
    remove.setAttribute('aria-label', LL.t('settings.ctx.table.remove_row'));
    actions.appendChild(remove);
    tr.appendChild(actions);
    return tr;
  }

  function contextLines(groupKey, key, title, values) {
    const wrap = makeEl('div', 'ctx-field');
    wrap.dataset.ctxPath = `${groupKey}.${key}`;
    const id = nextFieldId();
    const lab = makeEl('label', '', title);
    lab.htmlFor = id;
    const area = makeEl('textarea');
    area.id = id;
    area.rows = 3;
    area.placeholder = LL.t('settings.ctx.lines_placeholder');
    area.value = (Array.isArray(values) ? values : []).join('\n');
    area.addEventListener('change', () => setContextPath(groupKey, key, splitList(area.value, '\n')));
    wrap.append(lab, area);
    return wrap;
  }

  // ---- system context from a document ---------------------------------------
  const CONTEXT_IMPORT_MAX_BYTES = 10 * 1024 * 1024;
  const CONTEXT_IMPORT_ACTIONS = {
    filled: LL.t('settings.ctx.import.action.filled'),
    added: LL.t('settings.ctx.import.action.added'),
    replaced: LL.t('settings.ctx.import.action.replaced'),
    cleared: LL.t('settings.ctx.import.action.cleared')
  };

  function contextScopeTitle() {
    const areaMeta = (state.config.areas_meta || []).find((item) => item && item.id === state.area);
    const areaTitle = state.area ? ((areaMeta && areaMeta.title) || state.area) : LL.t('settings.ctx.import.scope_global');
    if (!state.contextScope) return state.area ? LL.t('settings.ctx.import.scope_area', { area: areaTitle }) : areaTitle;
    const svc = state.servicesMeta[state.contextScope];
    return LL.t('settings.ctx.import.scope_service', { service: (svc && svc.title) || state.contextScope, area: areaTitle });
  }
  function contextLlmTitle() {
    const llm = isPlainObject(state.config.llm) ? state.config.llm : {};
    const provider = String(llm.provider || '');
    const model = provider && isPlainObject(llm[provider]) ? String(llm[provider].model || '') : '';
    return provider ? (model ? `${provider} / ${model}` : provider) : '';
  }
  function clearContextImport() {
    state.contextImport = null;
    const banner = $('contextImportBanner');
    if (banner) { banner.hidden = true; banner.replaceChildren(); }
  }
  function markContextImport(host) {
    const paths = state.contextImport ? state.contextImport.paths : null;
    host.querySelectorAll('[data-ctx-path]').forEach((el) => {
      const hit = !!(paths && paths.has(el.dataset.ctxPath));
      el.classList.toggle('is-imported', hit);
      if (hit) { const group = el.closest('details'); if (group) group.open = true; }
    });
  }
  function showContextImportBanner(result) {
    const banner = $('contextImportBanner');
    banner.replaceChildren();
    const source = result.source || {};
    let from = LL.t('settings.ctx.import.from_text');
    if (source.format === 'confluence') from = LL.t('settings.ctx.import.from_confluence', { name: source.name });
    else if (source.name) from = LL.t('settings.ctx.import.from_file', { name: source.name });
    const changes = Array.isArray(result.changes) ? result.changes : [];
    const head = makeEl('div', 'ctx-import-head');
    head.appendChild(makeEl('strong', '', changes.length
      ? LL.t('settings.ctx.import.filled', { from })
      : LL.t('settings.ctx.import.nothing', { from })));
    const actions = makeEl('div', 'ctx-import-actions');
    actions.append(
      makeBtn(LL.t('settings.ctx.import.undo'), () => {
        if (state.contextImport) commitStructured('system_context', state.contextImport.previous, 'ed_system_context', 'contextSaveBtn', 'context', LL.t('settings.menu.context'));
        clearContextImport();
        renderContextForm();
      }, 'btn btn-sm'),
      makeBtn(LL.t('settings.ctx.import.hide'), () => { clearContextImport(); renderContextForm(); }, 'btn btn-sm btn-ghost')
    );
    head.appendChild(actions);
    banner.appendChild(head);
    if (changes.length) {
      const list = makeEl('ul', 'ctx-import-changes');
      changes.forEach((change) => {
        const count = change.count ? `: ${change.count}` : '';
        list.appendChild(makeEl('li', '', `${change.title} — ${CONTEXT_IMPORT_ACTIONS[change.action] || change.action}${count}`));
      });
      banner.appendChild(list);
    }
    (result.warnings || []).forEach((warning) => banner.appendChild(makeEl('p', 'field-hint warn', warning)));
    banner.hidden = false;
  }
  function applyContextImport(result) {
    const previous = clone(contextSnapshot());
    commitStructured('system_context', result.context || {}, 'ed_system_context', 'contextSaveBtn', 'context', LL.t('settings.menu.context'));
    state.contextImport = { previous, paths: new Set((result.changes || []).map((change) => change.path)) };
    renderContextForm();
    showContextImportBanner(result);
    $('contextImportBanner').scrollIntoView({ block: 'nearest' });
  }
  async function postContextImport(body, setStatus) {
    const source = body.querySelector('[data-source][aria-pressed="true"]').dataset.source;
    const mode = body.querySelector('input[name="ctxImportMode"]:checked').value;
    const common = { area: state.area || '', service: state.contextScope || '', mode };
    let init;
    if (source === 'file') {
      const file = body.querySelector('.ctx-import-file').files[0];
      if (!file) { setStatus(LL.t('settings.ctx.import.pick_file'), 'error'); return null; }
      if (file.size > CONTEXT_IMPORT_MAX_BYTES) { setStatus(LL.t('settings.ctx.import.file_too_big'), 'error'); return null; }
      const form = new FormData();
      form.append('file', file);
      Object.entries(common).forEach(([key, value]) => form.append(key, value));
      form.append('current', JSON.stringify(contextSnapshot()));
      init = { method: 'POST', body: form };
    } else if (source === 'confluence') {
      const url = body.querySelector('.ctx-import-url').value.trim();
      if (!/^https?:\/\//i.test(url)) { setStatus(LL.t('settings.ctx.import.need_url'), 'error'); return null; }
      init = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...common, confluence_url: url, current: contextSnapshot() }) };
    } else {
      const text = body.querySelector('.ctx-import-text').value;
      if (text.trim().length < 40) { setStatus(LL.t('settings.ctx.import.need_text'), 'error'); return null; }
      init = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...common, text, current: contextSnapshot() }) };
    }
    const started = Date.now();
    setStatus(LL.t('settings.ctx.import.parsing'), '');
    const timer = setInterval(() => setStatus(LL.t('settings.ctx.import.parsing_elapsed', { n: Math.round((Date.now() - started) / 1000) }), ''), 1000);
    try {
      const resp = await fetch('/config/system_context/import', init);
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) { setStatus(data.error || LL.t('settings.ctx.import.http_error', { status: resp.status }), 'error'); return null; }
      return data;
    } catch (e) {
      setStatus(LL.t('settings.ctx.import.no_response'), 'error');
      return null;
    } finally {
      clearInterval(timer);
    }
  }
  async function openContextImport() {
    const llm = contextLlmTitle();
    const confluence = isPlainObject(state.config.confluence) ? state.config.confluence : {};
    const confluenceBase = String(confluence.base_url || '').trim().replace(/\/+$/, '');
    const warning = llm
      ? LL.tHtml('settings.ctx.import.warning', { model: llm })
      : LL.tHtml('settings.ctx.import.no_llm');
    const bodyHtml = `
      <p class="field-hint warn">${warning}</p>
      <p class="dim">${LL.tHtml('settings.ctx.import.target', { scope: contextScopeTitle() })}</p>
      <div class="seg-group ctx-import-source" role="group" aria-label="${LL.tHtml('settings.ctx.import.source_aria')}">
        <button type="button" class="seg-btn" data-source="file" aria-pressed="true">${LL.tHtml('settings.ctx.import.source_file')}</button>
        <button type="button" class="seg-btn" data-source="text" aria-pressed="false">${LL.tHtml('settings.ctx.import.source_text')}</button>
        <button type="button" class="seg-btn" data-source="confluence" aria-pressed="false">Confluence</button>
      </div>
      <div class="ctx-import-pane" data-pane="file">
        <input type="file" class="ctx-import-file" accept=".pdf,.docx,.md,.markdown,.txt" aria-label="${LL.tHtml('settings.ctx.import.file_aria')}" />
        <div class="field-hint">${LL.tHtml('settings.ctx.import.file_hint')}</div>
      </div>
      <div class="ctx-import-pane" data-pane="text" hidden>
        <textarea class="ctx-import-text" rows="10" aria-label="${LL.tHtml('settings.ctx.import.text_aria')}" placeholder="${LL.tHtml('settings.ctx.import.text_placeholder')}"></textarea>
      </div>
      <div class="ctx-import-pane" data-pane="confluence" hidden>
        <input type="url" class="ctx-import-url" aria-label="${LL.tHtml('settings.ctx.import.url_aria')}" placeholder="${esc(confluenceBase ? `${confluenceBase}/pages/viewpage.action?pageId=…` : 'https://confluence.example.ru/pages/viewpage.action?pageId=…')}" />
        <div class="field-hint">${confluenceBase
          ? LL.tHtml('settings.ctx.import.url_hint', { base: confluenceBase })
          : LL.tHtml('settings.ctx.import.url_hint_unset')}</div>
      </div>
      <fieldset class="ctx-import-mode">
        <legend>${LL.tHtml('settings.ctx.import.mode_legend')}</legend>
        <label class="checkbox-row"><input type="radio" name="ctxImportMode" value="fill" checked /> <span>${LL.tHtml('settings.ctx.import.mode_fill')}</span></label>
        <label class="checkbox-row"><input type="radio" name="ctxImportMode" value="replace" /> <span>${LL.tHtml('settings.ctx.import.mode_replace')}</span></label>
      </fieldset>
      <div class="ctx-import-status" role="status" aria-live="polite"></div>`;
    let busy = false;
    await ui.dialog({
      title: LL.t('settings.ctx.import.title'),
      wide: true,
      bodyHtml,
      onOpen: (body) => {
        body.querySelectorAll('[data-source]').forEach((btn) => btn.addEventListener('click', () => {
          body.querySelectorAll('[data-source]').forEach((other) => other.setAttribute('aria-pressed', other === btn ? 'true' : 'false'));
          body.querySelectorAll('[data-pane]').forEach((pane) => { pane.hidden = pane.dataset.pane !== btn.dataset.source; });
        }));
      },
      actions: [
        { label: LL.t('settings.action.cancel') },
        {
          label: LL.t('settings.ctx.import.parse'),
          primary: true,
          onClick: async (close, body) => {
            if (busy || !llm) return;
            busy = true;
            const status = body.querySelector('.ctx-import-status');
            const setStatus = (text, tone) => { status.textContent = text; status.className = `ctx-import-status ${tone || ''}`.trim(); };
            const buttons = Array.from(body.closest('dialog').querySelectorAll('.dialog-actions button'));
            buttons.forEach((btn) => { btn.disabled = true; });
            const result = await postContextImport(body, setStatus);
            buttons.forEach((btn) => { btn.disabled = false; });
            busy = false;
            if (!result) return;
            close(true);
            applyContextImport(result);
          }
        }
      ]
    });
  }

  // ---- queries form ----------------------------------------------------------
  function queriesModel() {
    return isPlainObject(state.models.queries) ? state.models.queries : {};
  }
  function queryBlock(domain) {
    const block = queriesModel()[domain];
    return isPlainObject(block) ? block : {};
  }
  function filledLangs(block) {
    return LT_QUERY_PRIORITY.filter((lang) => (block[QUERY_LANG[lang].query] || []).some((q) => String(q || '').trim()));
  }
  function savedSourceType(domain) {
    const bindings = isPlainObject(state.config.domain_sources) ? state.config.domain_sources : {};
    const catalog = isPlainObject(state.config.data_sources) ? state.config.data_sources : {};
    const own = isPlainObject(bindings[domain]) ? bindings[domain] : null;
    const binding = own || (isPlainObject(bindings.default) ? bindings.default : {});
    const entry = catalog[binding.source];
    return String((entry && entry.type) || '').toLowerCase();
  }
  function activeQueryLang(domain, block) {
    if (state.domainLang[domain]) return state.domainLang[domain];
    const type = savedSourceType(domain);
    if (type === 'prometheus') return 'promql';
    if (type === 'influxdb') return 'flux';
    return filledLangs(block)[0] || LT_QUERY_PRIORITY[0];
  }
  function currentQuerySpec(domain) {
    return QUERY_LANG[activeQueryLang(domain, queryBlock(domain))];
  }
  function blankQueryCell(key) {
    return key.endsWith('_list') ? [] : '';
  }
  function queryRowCount(block) {
    return QUERY_PARALLEL_KEYS.reduce((max, key) => Math.max(max, Array.isArray(block[key]) ? block[key].length : 0), 0);
  }
  // Pads the arrays in use (and the active language) so positions stay aligned.
  function alignQueryBlock(block, spec) {
    const count = queryRowCount(block);
    const required = ['labels', spec.query, spec.keys];
    QUERY_PARALLEL_KEYS.forEach((key) => {
      const list = Array.isArray(block[key]) ? block[key].slice() : [];
      if (!list.length && !required.includes(key)) return;
      while (list.length < count) list.push(blankQueryCell(key));
      block[key] = list;
    });
    return count;
  }
  // ``spec``: the language whose arrays the edit writes (default: the language of new rows).
  function editQueryBlock(domain, mutate, rowSpec) {
    const queries = clone(queriesModel());
    const block = Object.assign({}, isPlainObject(queries[domain]) ? queries[domain] : {});
    const spec = rowSpec || currentQuerySpec(domain);
    const count = alignQueryBlock(block, spec);
    mutate(block, spec, count);
    queries[domain] = block;
    commitStructured('queries', queries, 'ed_queries', 'queriesSaveBtn', 'queries', state.queriesScope ? LL.t('settings.dirty.queries_scoped', { scope: state.queriesScope }) : LL.t('settings.menu.queries'));
  }
  // Languages with a query text in the row, in the order Grafana prefers them (as the server picks).
  function rowLanguages(block, index) {
    return LT_QUERY_PRIORITY.filter((lang) => {
      const values = block[QUERY_LANG[lang].query];
      return Array.isArray(values) && index < values.length && String(values[index] || '').trim();
    });
  }
  function sourceTypeOf(binding) {
    const entry = binding && binding.source ? savedSource(binding.source) : null;
    return String((entry && entry.type) || '').toLowerCase();
  }
  // Language a row runs in: its text decides (the one its source runs when there are several); an empty row
  // takes the language picked for it, else the one its source runs, else the language of new rows.
  function rowLang(domain, block, index, label) {
    const type = sourceTypeOf(rowSource(domain, label) || domainBindingOf(domain));
    const present = rowLanguages(block, index);
    if (present.length) return present.find((lang) => sourceRunsLanguage(type, lang)) || present[0];
    const draft = state.rowLangDraft[`${domain}:${index}`];
    if (draft) return draft;
    // A row with a direct source of its own runs that source's language; otherwise the language of new rows.
    const ownType = sourceTypeOf(rowSource(domain, label));
    if (ownType === 'prometheus') return 'promql';
    if (ownType === 'influxdb') return 'flux';
    return activeQueryLang(domain, block);
  }
  function queryRows(block, domain) {
    const labels = Array.isArray(block.labels) ? block.labels : [];
    const count = queryRowCount(block);
    const rows = [];
    for (let i = 0; i < count; i += 1) {
      const label = String(labels[i] || '');
      const lang = rowLang(domain, block, i, label);
      const spec = QUERY_LANG[lang];
      const queries = Array.isArray(block[spec.query]) ? block[spec.query] : [];
      const keys = Array.isArray(block[spec.keys]) ? block[spec.keys] : [];
      rows.push({
        index: i,
        label,
        lang,
        spec,
        langs: rowLanguages(block, i),
        query: String(queries[i] || ''),
        keys: Array.isArray(keys[i]) ? keys[i].map(String) : []
      });
    }
    return rows;
  }
  // Moves the row's text to another language and clears the other languages of the row. ``current`` holds
  // what the editor fields show ({text, keys}): typing that has not left the field yet is not in the model.
  function setRowLanguage(domain, row, lang, current) {
    const spec = QUERY_LANG[lang];
    state.rowLangDraft[`${domain}:${row.index}`] = lang;
    editQueryBlock(domain, (block) => {
      const texts = Array.isArray(block[row.spec.query]) ? block[row.spec.query] : [];
      const keyLists = Array.isArray(block[row.spec.keys]) ? block[row.spec.keys] : [];
      const text = current ? String(current.text || '') : String(texts[row.index] || '');
      const keys = current ? current.keys.slice() : (Array.isArray(keyLists[row.index]) ? keyLists[row.index].slice() : []);
      Object.values(QUERY_LANG).forEach((other) => {
        if (Array.isArray(block[other.query]) && row.index < block[other.query].length) block[other.query][row.index] = '';
      });
      block[spec.query][row.index] = text;
      block[spec.keys][row.index] = keys;
    }, spec);
    renderQueryList();
  }
  // Languages picked for empty rows follow positions: a removed or moved row would hand its pick to another row.
  function clearRowLangDrafts(domain) {
    Object.keys(state.rowLangDraft).forEach((key) => { if (key.startsWith(`${domain}:`)) delete state.rowLangDraft[key]; });
  }

  function renderQueriesForm() {
    const host = $('queriesForm');
    if (!host) return;
    const domain = QUERY_DOMAINS.includes(state.queryDomain) ? state.queryDomain : 'jvm';
    state.queryDomain = domain;
    host.innerHTML = '';
    host.appendChild(queryDomainTabs(domain));
    host.appendChild(queryLangBar(domain, queryBlock(domain)));
    host.appendChild(domainSourceLine(domain));
    host.appendChild(previewWindowBar());
    const search = makeEl('input');
    search.type = 'search';
    search.placeholder = LL.t('settings.queries.search_placeholder');
    search.value = state.querySearch || '';
    search.addEventListener('input', () => { state.querySearch = search.value; renderQueryList(); });
    const toolbar = makeEl('div', 'struct-toolbar');
    const fromPack = makeBtn(LL.t('settings.queries.add_from_pack'), () => openPackDialog(state.wizard ? '' : domain), 'btn btn-primary');
    fromPack.id = 'addFromPackBtn';  // target of the project readiness checklist (projects.js)
    toolbar.append(search, makeBtn(LL.t('settings.queries.add'), addQueryRow), fromPack);
    if (domain === 'lt_framework' && rpsRoleCount(queryBlock(domain)) >= 2) {
      const sum = makeBtn(LL.t('settings.queries.summed_rps'), () => addSummedRps(sum));
      sum.title = LL.t('settings.queries.summed_rps_title');
      toolbar.appendChild(sum);
    }
    host.appendChild(toolbar);
    const installed = installedPacksLine(domain);
    if (installed) host.appendChild(installed);
    host.appendChild(domainTitleField(domain));
    const list = makeEl('div', 'q-list');
    list.id = 'queryList';
    host.appendChild(list);
    renderQueryList();
  }

  // Which source the domain reads, and how many rows read one of their own.
  function domainSourceLine(domain) {
    const block = queryBlock(domain);
    const labels = Array.isArray(block.labels) ? block.labels : [];
    const own = labels.filter((label) => rowSource(domain, label)).length;
    const parts = [LL.t('settings.queries.domain_source', { summary: bindingSummary(domainBindingOf(domain)) })];
    if (own) parts.push(LL.t('settings.queries.rows_own_source', { n: own }));
    return makeEl('p', 'dim', parts.join(' '));
  }
  function queryDomainTabs(active) {
    const tabs = makeEl('div', 'struct-tabs');
    QUERY_DOMAINS.forEach((name) => {
      const block = queryBlock(name);
      const count = Array.isArray(block.labels) ? block.labels.length : 0;
      tabs.appendChild(makeBtn(`${LL.domainTitle(name)} (${count})`, () => {
        state.queryDomain = name;
        state.queryOpen = null;
        renderQueriesForm();
      }, 'btn' + (name === active ? ' is-active' : '')));
    });
    return tabs;
  }

  // Language of new rows. Every row keeps its own language: a domain may read several sources.
  function queryLangBar(domain, block) {
    const lang = activeQueryLang(domain, block);
    const bar = makeEl('div', 'struct-toolbar');
    const ownSources = Object.values(isPlainObject(block.sources) ? block.sources : {}).some((value) => isPlainObject(value) && value.source);
    const switcher = savedSourceType(domain) === 'grafana_proxy' || filledLangs(block).length > 1 || ownSources || state.domainLangPicker[domain];
    if (!switcher) {
      bar.appendChild(makeEl('span', 'pill na', LL.t('settings.queries.lang', { lang: QUERY_LANG[lang].label })));
      return bar;
    }
    bar.appendChild(makeEl('span', 'dim', LL.t('settings.queries.new_rows_lang')));
    LT_QUERY_PRIORITY.forEach((name) => {
      bar.appendChild(makeBtn(QUERY_LANG[name].label, () => {
        state.domainLang[domain] = name;
        state.domainLangPicker[domain] = true;
        state.queryOpen = null;
        renderQueriesForm();
      }, 'btn' + (name === lang ? ' is-active' : '')));
    });
    return bar;
  }

  function renderQueryList() {
    const list = $('queryList');
    if (!list) return;
    const domain = state.queryDomain;
    const needle = String(state.querySearch || '').trim().toLowerCase();
    const rows = queryRows(queryBlock(domain), domain)
      .filter((row) => !needle || row.label.toLowerCase().includes(needle) || row.query.toLowerCase().includes(needle));
    list.innerHTML = '';
    if (!rows.length) {
      list.appendChild(makeEl('p', 'dim', needle ? LL.t('settings.queries.not_found') : LL.t('settings.queries.empty')));
      return;
    }
    rows.forEach((row) => list.appendChild(queryItem(domain, row.spec, row)));
  }

  function queryItem(domain, spec, row) {
    const item = makeEl('div', 'q-item');
    const open = state.queryOpen === `${domain}:${row.index}`;
    const head = makeEl('div', 'q-item-head');
    head.setAttribute('role', 'button');
    head.tabIndex = 0;
    const toggle = () => { state.queryOpen = open ? null : `${domain}:${row.index}`; renderQueryList(); };
    head.addEventListener('click', toggle);
    head.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
    const main = makeEl('div', 'q-item-main');
    main.appendChild(makeEl('div', 'q-item-title', row.label || LL.t('settings.queries.untitled')));
    const meta = [spec.label];
    if (row.keys.length) meta.push(LL.t('settings.queries.series', { keys: row.keys.join(', ') }));
    if (meta.length) main.appendChild(makeEl('div', 'q-item-meta', meta.join(' · ')));
    if (!open) main.appendChild(makeEl('div', 'q-item-code', (row.query.split('\n').find((line) => line.trim()) || '').trim() || '—'));
    head.appendChild(main);
    const roles = isPlainObject(queryBlock(domain).roles) ? queryBlock(domain).roles : {};
    if (roles[row.label]) {
      const rolePill = makeEl('span', 'pill na', roleTitle(roles[row.label]));
      rolePill.title = LL.t('settings.queries.role_title', { role: roles[row.label] });
      head.appendChild(rolePill);
    }
    const used = slaLabelsUsing(row.label);
    if (used.length) head.appendChild(makeEl('span', 'pill na', `SLA: ${slaUsageText(used)}`));
    const own = rowSource(domain, row.label);
    if (own) {
      const sourcePill = makeEl('span', 'pill warn', LL.t('settings.queries.source_pill', { source: bindingSummary(own) }));
      sourcePill.title = LL.t('settings.queries.source_pill_title');
      head.appendChild(sourcePill);
    }
    if (row.langs.length > 1) head.appendChild(makeEl('span', 'pill error', LL.t('settings.queries.two_langs_pill')));
    head.appendChild(makeEl('span', 'q-item-caret', open ? '▾' : '▸'));
    item.appendChild(head);
    if (open) item.appendChild(queryEditor(domain, spec, row));
    return item;
  }

  function previewStamp(date) {
    const pad = (n) => String(n).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
  }
  function ensurePreviewWindow() {
    if (state.previewStart && state.previewEnd) return;
    const end = new Date();
    end.setSeconds(0, 0);
    state.previewEnd = previewStamp(end);
    state.previewStart = previewStamp(new Date(end.getTime() - 15 * 60000));
  }
  function previewWindowBar() {
    ensurePreviewWindow();
    const bar = makeEl('div', 'struct-toolbar');
    const start = makeEl('input');
    start.type = 'datetime-local';
    start.value = state.previewStart;
    start.setAttribute('aria-label', LL.t('settings.queries.window_start_aria'));
    start.addEventListener('change', () => { state.previewStart = start.value; });
    const end = makeEl('input');
    end.type = 'datetime-local';
    end.value = state.previewEnd;
    end.setAttribute('aria-label', LL.t('settings.queries.window_end_aria'));
    end.addEventListener('change', () => { state.previewEnd = end.value; });
    bar.append(makeEl('span', 'dim', LL.t('settings.queries.window')), start, makeEl('span', 'dim', '—'), end);
    return bar;
  }
  // The source a row reads when it is not the domain binding (from the unsaved model of the section).
  function rowSource(domain, label) {
    const sources = isPlainObject(queryBlock(domain).sources) ? queryBlock(domain).sources : {};
    const own = sources[label];
    return isPlainObject(own) && own.source ? own : null;
  }
  // Binding the domain reads in the open scope: its own entry, else the default one.
  function domainBindingOf(domain) {
    const bindings = isPlainObject(state.config.domain_sources) ? state.config.domain_sources : {};
    return isPlainObject(bindings[domain]) ? bindings[domain] : (isPlainObject(bindings.default) ? bindings.default : null);
  }
  function setRowSource(block, label, binding) {
    const sources = isPlainObject(block.sources) ? Object.assign({}, block.sources) : {};
    if (binding) sources[label] = binding;
    // An explicit null keeps an override of the wider scope from coming back (the scopes are merged).
    else if (hasOwn(sources, label)) sources[label] = null;
    if (Object.keys(sources).length) block.sources = sources;
    else delete block.sources;
  }
  function rowSourceCtx(domain, row, language) {
    const own = rowSource(domain, row.label);
    const edit = (mutate) => editQueryBlock(domain, (block) => mutate(block));
    return {
      id: `rs-${domain}-${row.index}`,
      title: row.label || LL.t('settings.queries.untitled'),
      own: !!own,
      get: () => own || {},
      set: (binding) => { edit((block) => setRowSource(block, row.label, binding)); renderQueryList(); },
      setField: (field, value) => edit((block) => {
        const current = isPlainObject(block.sources) && isPlainObject(block.sources[row.label]) ? Object.assign({}, block.sources[row.label]) : {};
        const text = String(value || '').trim();
        if (text) current[field] = text; else delete current[field];
        setRowSource(block, row.label, current);
      }),
      rerender: () => {},
      emptyLabel: LL.t('settings.queries.source_domain', { summary: bindingSummary(domainBindingOf(domain)) }),
      inheritNote: LL.t('settings.queries.source_domain_note'),
      language
    };
  }
  function rowSourceField(domain, row, language) {
    const ctx = rowSourceCtx(domain, row, language);
    const wrap = makeEl('div', 'ctx-field q-source');
    const label = makeEl('label', '', LL.t('settings.queries.field_source'));
    label.htmlFor = `${ctx.id}-source`;
    wrap.append(label, mountBindingEditor(ctx), makeEl('div', 'dim', LL.t('settings.queries.source_hint')));
    return wrap;
  }
  async function runQueryPreview(domain, queryInput, keysInput, result, label, lang) {
    const query = String(queryInput.value || '').trim();
    result.className = 'conn-result pending';
    result.textContent = LL.t('settings.check.running');
    if (!query) {
      result.className = 'conn-result error';
      result.textContent = LL.t('settings.queries.empty_query');
      return;
    }
    const body = {
      domain: domain,
      lang: lang,
      query: query,
      label_keys: splitList(keysInput.value),
      label: label || '',
      // The source the form shows for the row (null: the domain binding), saved or not.
      binding: rowSource(domain, label)
    };
    if (state.area) body.area = state.area;
    if (state.queriesScope) body.service = state.queriesScope;
    if (state.previewStart) body.start = state.previewStart;
    if (state.previewEnd) body.end = state.previewEnd;
    const resp = await fetch('/config/query_preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await resp.json();
    const message = data.message || data.error || ('HTTP ' + resp.status);
    result.className = 'conn-result ' + (data.ok ? 'ok' : 'error');
    result.textContent = message;
    if (Array.isArray(data.series) && data.series.length) {
      const names = makeEl('div', '', data.series.join(', '));
      result.appendChild(names);
    }
  }
  // Max length of a row description; the server checks the same limit (AI.query_packs.blocks.DESCRIPTION_MAX_CHARS).
  const QUERY_NOTE_MAX = 500;
  // Roles a row of the domain may take: the roles of the domain, then (in a component domain) the host roles
  // (AI.query_packs.role_fits), each group by title. Live-chart roles are for packs only and are listed when set.
  function roleOptions(catalog, domain, current) {
    catalog = catalog || {};
    const roles = isPlainObject(catalog.roles) ? catalog.roles : {};
    const hostDomains = Array.isArray(catalog.host_domains) ? catalog.host_domains : [];
    const own = [];
    const host = [];
    Object.keys(roles).forEach((name) => {
      const role = roles[name] || {};
      const fits = role.domain === domain || (role.domain == null && hostDomains.includes(domain));
      if (name !== current && (!fits || role.live_only)) return;
      (role.domain == null ? host : own).push([name, `${role.title} (${role.unit})`]);
    });
    const byTitle = (a, b) => a[1].localeCompare(b[1]);
    return [['', LL.t('settings.queries.role_none')]].concat(own.sort(byTitle), host.sort(byTitle));
  }
  // First row of the block with this SLA role when several rows share it ('' otherwise): an empty SLA reference
  // takes the first one (AI.query_packs.resolve_sla_refs).
  function sharedRoleLabel(catalog, block, role) {
    const info = ((catalog && catalog.roles) || {})[role];
    if (!role || !info || !info.sla_field) return '';
    const roles = isPlainObject(block && block.roles) ? block.roles : {};
    const same = ((block && block.labels) || []).filter((other) => other && roles[other] === role);
    return same.length > 1 ? same[0] : '';
  }
  function setRowKey(domain, key, label, value) {
    editQueryBlock(domain, (block) => {
      const next = isPlainObject(block[key]) ? Object.assign({}, block[key]) : {};
      if (value) next[label] = value; else delete next[label];
      if (Object.keys(next).length) block[key] = next; else delete block[key];
    });
  }
  function rowRoleField(domain, row) {
    const roles = isPlainObject(queryBlock(domain).roles) ? queryBlock(domain).roles : {};
    const current = row.label ? String(roles[row.label] || '') : '';
    const field = selectField(LL.t('settings.queries.field_role'), current, roleOptions(packUi.catalog, domain, current), (role) => {
      setRowKey(domain, 'roles', row.label, role);
      renderQueryList();
    });
    const select = field.querySelector('select');
    select.disabled = !row.label;
    field.appendChild(makeEl('div', 'dim', row.label ? LL.t('settings.queries.role_hint') : LL.t('settings.queries.name_first')));
    const first = row.label ? sharedRoleLabel(packUi.catalog, queryBlock(domain), current) : '';
    if (first) field.appendChild(makeEl('p', 'pack-note warn', LL.t('settings.queries.role_taken', { label: first })));
    return field;
  }
  // Description of the pack query with this label (the catalog renders the packs with their default variables).
  function packQueryDescription(label) {
    const packs = (packUi.catalog && packUi.catalog.packs) || [];
    for (const pack of packs) {
      const query = (pack.queries || []).find((item) => item.label === label);
      if (query && query.description) return String(query.description);
    }
    return '';
  }
  function rowDescriptionField(domain, row) {
    const notes = isPlainObject(queryBlock(domain).descriptions) ? queryBlock(domain).descriptions : {};
    const field = textField(LL.t('settings.queries.field_description'), row.label ? notes[row.label] || '' : '', () => {}, true);
    field.classList.add('q-note');
    const input = field.querySelector('textarea');
    input.rows = 2;
    input.maxLength = QUERY_NOTE_MAX;
    input.disabled = !row.label;
    const fromPack = row.label ? packQueryDescription(row.label) : '';
    input.placeholder = fromPack || LL.t('settings.queries.description_placeholder');
    input.addEventListener('change', () => setRowKey(domain, 'descriptions', row.label, input.value.trim()));
    const hint = fromPack ? 'settings.queries.description_hint_pack' : 'settings.queries.description_hint';
    field.appendChild(makeEl('div', 'dim', LL.t(hint, { max: QUERY_NOTE_MAX })));
    return field;
  }
  function queryEditor(domain, spec, row) {
    const body = makeEl('div', 'q-item-body');
    const name = textField(LL.t('settings.queries.field_name'), row.label, () => {});
    const nameInput = name.querySelector('input');
    nameInput.addEventListener('change', () => {
      renameSlaLabel(row.label, nameInput.value);
      editQueryBlock(domain, (block) => {
        block.labels[row.index] = nameInput.value;
        // The role, the description and the source belong to the row, so they follow the new name.
        ['roles', 'descriptions', 'sources'].forEach((key) => {
          if (isPlainObject(block[key]) && Object.prototype.hasOwnProperty.call(block[key], row.label)) {
            block[key][nameInput.value] = block[key][row.label];
            delete block[key][row.label];
          }
        });
      });
      renderQueriesForm();
    });
    const query = textField(LL.t('settings.queries.field_query', { lang: spec.label }), row.query, () => {}, true);
    const queryInput = query.querySelector('textarea');
    queryInput.rows = Math.min(14, Math.max(4, row.query.split('\n').length + 1));
    queryInput.addEventListener('change', () => {
      editQueryBlock(domain, (block) => { block[spec.query][row.index] = queryInput.value; }, spec);
    });
    const keys = textField(LL.t('settings.queries.field_keys'), row.keys.join(', '), () => {});
    const keysInput = keys.querySelector('input');
    keysInput.addEventListener('change', () => {
      editQueryBlock(domain, (block) => { block[spec.keys][row.index] = splitList(keysInput.value); }, spec);
    });
    const language = selectField(LL.t('settings.queries.row_lang'), row.lang, LT_QUERY_PRIORITY.map((lang) => [lang, QUERY_LANG[lang].label]), (lang) => {
      if (lang !== row.lang) setRowLanguage(domain, row, lang, { text: queryInput.value, keys: splitList(keysInput.value) });
    });
    body.append(name, language, query, keys, rowRoleField(domain, row), rowDescriptionField(domain, row));
    if (row.langs.length > 1) {
      const mixed = makeEl('p', 'pack-note error', LL.t('settings.queries.two_langs', { langs: row.langs.map((lang) => QUERY_LANG[lang].label).join(', '), lang: spec.label }));
      mixed.appendChild(document.createTextNode(' '));
      mixed.appendChild(makeBtn(LL.t('settings.queries.keep_lang', { lang: spec.label }), () => setRowLanguage(domain, row, row.lang, { text: queryInput.value, keys: splitList(keysInput.value) }), 'btn btn-sm'));
      body.appendChild(mixed);
    }
    if (spec.keys === 'label_tag_keys_list') body.appendChild(makeEl('p', 'help', LL.t('settings.queries.keys_shared')));
    const type = sourceTypeOf(rowSource(domain, row.label) || domainBindingOf(domain));
    if (type && !sourceRunsLanguage(type, row.lang)) body.appendChild(makeEl('p', 'pack-note error', LL.t('settings.queries.source_lang_mismatch', { lang: spec.label })));
    body.appendChild(rowSourceField(domain, row, row.query.trim() ? row.lang : ''));
    const preview = makeEl('div', 'conn-result');
    const check = makeBtn(LL.t('settings.action.check'), async () => {
      check.disabled = true;
      try {
        await runQueryPreview(domain, queryInput, keysInput, preview, row.label, row.lang);
      } catch (e) {
        preview.className = 'conn-result error';
        preview.textContent = e.message || LL.t('settings.check.error');
      } finally {
        check.disabled = false;
      }
    });
    const actions = makeEl('div', 'q-item-actions');
    actions.append(
      check,
      makeBtn(LL.t('settings.queries.move_up'), () => moveQueryRow(domain, row.index, -1)),
      makeBtn(LL.t('settings.queries.move_down'), () => moveQueryRow(domain, row.index, 1)),
      makeBtn(LL.t('settings.action.delete'), () => removeQueryRow(domain, row.index)),
      makeBtn(LL.t('settings.queries.collapse'), () => { state.queryOpen = null; renderQueryList(); })
    );
    body.append(preview, actions);
    return body;
  }

  function addQueryRow() {
    const domain = state.queryDomain;
    let newIndex = 0;
    editQueryBlock(domain, (block, spec, count) => {
      QUERY_PARALLEL_KEYS.forEach((key) => {
        if (Array.isArray(block[key]) && (block[key].length || [spec.query, spec.keys, 'labels'].includes(key))) block[key].push(blankQueryCell(key));
      });
      newIndex = count;
    });
    state.querySearch = '';
    state.queryOpen = `${domain}:${newIndex}`;
    renderQueriesForm();
  }

  function removeQueryRow(domain, index) {
    const label = String((queryBlock(domain).labels || [])[index] || '');
    if (!window.confirm(LL.t('settings.queries.delete_confirm', { name: label || LL.t('settings.queries.untitled_lower') }))) return;
    editQueryBlock(domain, (block) => {
      QUERY_PARALLEL_KEYS.forEach((key) => { if (Array.isArray(block[key]) && index < block[key].length) block[key].splice(index, 1); });
      if (isPlainObject(block.roles)) delete block.roles[label];
      if (isPlainObject(block.descriptions)) delete block.descriptions[label];
      if (isPlainObject(block.sources)) delete block.sources[label];
    });
    clearRowLangDrafts(domain);
    state.queryOpen = null;
    renderQueriesForm();
  }

  function moveQueryRow(domain, index, dir) {
    const target = index + dir;
    if (target < 0 || target >= queryRowCount(queryBlock(domain))) return;
    editQueryBlock(domain, (block) => {
      QUERY_PARALLEL_KEYS.forEach((key) => {
        const list = block[key];
        if (!Array.isArray(list) || Math.max(index, target) >= list.length) return;
        const tmp = list[index];
        list[index] = list[target];
        list[target] = tmp;
      });
    });
    clearRowLangDrafts(domain);
    state.queryOpen = `${domain}:${target}`;
    renderQueryList();
  }

  // ---- query packs -----------------------------------------------------------
  // Ready-made queries for common stacks. The dialog lists packs and bundles for the scope this
  // section edits (area and service), checks a pack against the sources, previews the changes and
  // applies them through /config/query_packs/*; the page then reloads queries and SLA from the server.
  const PACK_COMPAT = {
    ok: { cls: 'ok', text: LL.t('settings.packs.compat.ok') },
    replace_only: { cls: 'warn', text: LL.t('settings.packs.compat.replace_only') },
    incompatible: { cls: 'error', text: LL.t('settings.packs.compat.incompatible') }
  };
  const packUi = { catalog: null, forms: {}, view: { name: 'list', domain: '' }, body: null };

  function rpsRoleCount(block) {
    const roles = isPlainObject(block && block.roles) ? block.roles : {};
    const derived = isPlainObject(block && block.derived) ? block.derived : {};
    return Object.keys(roles).filter((label) => roles[label] === 'lt.rps_total' && !hasOwn(derived, label)).length;
  }
  async function addSummedRps(button) {
    const ok = await ui.confirm({
      title: LL.t('settings.queries.summed_rps'),
      message: LL.t('settings.queries.summed_rps_confirm'),
      confirmText: LL.t('settings.action.add')
    });
    if (!ok) return;
    button.disabled = true;
    try {
      const resp = await postPack('summed_rps', { ...packScope(), use_for_sla: true });
      await refreshQueriesAndSla();
      ui.toast(LL.t('settings.queries.summed_rps_added', { label: resp.label, parts: resp.parts.join(' + ') }), { tone: 'ok', timeout: 10000 });
    } catch (e) {
      ui.toast(LL.t('settings.queries.summed_rps_failed', { error: e.message }), { tone: 'error' });
      button.disabled = false;
    }
  }
  function packScope() {
    const scope = {};
    if (state.area) scope.area = state.area;
    if (state.area && state.queriesScope) scope.service = state.queriesScope;
    return scope;
  }
  async function loadPackCatalog() {
    const url = new URL('/config/query_packs', location.origin);
    Object.entries(packScope()).forEach(([key, value]) => url.searchParams.set(key, value));
    const resp = await fetch(url);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    packUi.catalog = data;
    return data;
  }
  function findPack(id) { return ((packUi.catalog && packUi.catalog.packs) || []).find((p) => p.id === id) || null; }
  function findBundle(id) { return ((packUi.catalog && packUi.catalog.bundles) || []).find((b) => b.id === id) || null; }
  function roleTitle(role) {
    const roles = (packUi.catalog && packUi.catalog.roles) || {};
    return (roles[role] && roles[role].title) || role;
  }
  // Compatibility and installations of the pack in one domain (a host pack is installed into the domain picked).
  function packTarget(pack, domain) {
    const targets = isPlainObject(pack.targets) ? pack.targets : {};
    const target = targets[domain] || targets[pack.domain];
    return target || { compatibility: pack.compatibility, installed: pack.installed, instances: pack.installed ? [pack.installed] : [] };
  }
  function packDomains(pack) {
    return Array.isArray(pack.domains) && pack.domains.length ? pack.domains : [pack.domain];
  }
  // Domain the pack card works in: the one picked in the card, else the tab the dialog came from.
  function packDomain(pack) {
    const domains = packDomains(pack);
    const view = packUi.view || {};
    if (view.domain && domains.includes(view.domain)) return view.domain;
    const back = view.back || {};
    if (back.domain && domains.includes(back.domain)) return back.domain;
    return domains[0];
  }
  function packInstances(pack, domain) {
    const target = packTarget(pack, domain);
    return Array.isArray(target.instances) ? target.instances : (target.installed ? [target.installed] : []);
  }
  // A title for a new installation that no installation of the domain uses yet: «Host», «Host 2», …
  function freeInstanceTitle(base, taken) {
    if (!taken.includes(base)) return base;
    for (let n = 2; n <= 99; n += 1) {
      const name = `${base} ${n}`;
      if (!taken.includes(name)) return name;
    }
    return base;
  }
  function packFormFor(pack, domain, installed) {
    const saved = (installed && installed.variables) || {};
    // An installed pack keeps the user's selection: present rows plus queries added by newer versions.
    const selected = installed && Array.isArray(installed.selected_indexes) ? new Set(installed.selected_indexes) : null;
    const variables = Object.fromEntries(pack.variables.map((v) => [v.name, saved[v.name] !== undefined ? saved[v.name] : v.default]));
    if (!installed && pack.instance_variable && variables[pack.instance_variable] !== undefined) {
      const taken = packInstances(pack, domain).map((item) => item.instance);
      variables[pack.instance_variable] = freeInstanceTitle(variables[pack.instance_variable], taken);
    }
    return {
      variables,
      include: pack.queries.map((_, index) => (selected ? selected.has(index) : true)),
      // A language conflict is solved by another source for the pack as well, so merging stays the default.
      mode: 'merge',
      fillSla: true,
      overwriteSla: true,
      check: null,
      preview: null,
      instance: installed ? String(installed.instance || '') : '',
      // The source the pack rows read: the one the installation reads, else the domain binding (null).
      binding: installed && isPlainObject(installed.binding) && installed.binding.source ? Object.assign({}, installed.binding) : null
    };
  }
  const BINDING_TEXT_FIELDS = ['source', 'datasource_uid', 'datasource_name', 'database', 'bucket'];
  function sameBinding(a, b) {
    return BINDING_TEXT_FIELDS.every((field) => String((a || {})[field] || '').trim() === String((b || {})[field] || '').trim());
  }
  function packOwnSource(form) {
    return isPlainObject(form.binding) && form.binding.source ? form.binding : null;
  }
  function packForm(pack) {
    const domain = packDomain(pack);
    const key = `${pack.id}|${domain}`;
    if (!packUi.forms[key]) {
      const instances = packInstances(pack, domain);
      const wanted = (packUi.view || {}).instance;
      let installed = null;
      if (wanted !== undefined && wanted !== null) installed = instances.find((item) => item.instance === wanted) || null;
      // A multi-instance pack opens on a new installation while there is room for one.
      else if (!(pack.max_instances > 1) || instances.length >= pack.max_instances) installed = instances[0] || null;
      packUi.forms[key] = packFormFor(pack, domain, installed);
    }
    return packUi.forms[key];
  }
  function includedIndexes(form) {
    return form.include.map((on, index) => (on ? index : -1)).filter((index) => index >= 0);
  }
  function packRequest(pack, form, extra) {
    return Object.assign({
      pack_id: pack.id,
      domain: packDomain(pack),
      variables: form.variables,
      include_indexes: includedIndexes(form),
      binding: packOwnSource(form)
    }, packScope(), extra || {});
  }
  async function postPack(path, body) {
    const resp = await fetch(`/config/query_packs/${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    return data;
  }
  function scopeLine() {
    const parts = [state.area ? LL.t('settings.packs.scope_area', { area: state.area }) : LL.t('settings.packs.scope_global')];
    if (state.area && state.queriesScope) parts.push(LL.t('settings.packs.scope_service', { service: state.queriesScope }));
    return makeEl('p', 'help', LL.t('settings.packs.scope_line', { scope: parts.join(', ') }));
  }
  function packPills(pack, domain) {
    const target = packTarget(pack, domain || packDomains(pack)[0]);
    const wrap = makeEl('span', 'pack-pills');
    wrap.appendChild(makeEl('span', 'pill na', pack.language_title));
    const compat = PACK_COMPAT[target.compatibility.status] || PACK_COMPAT.incompatible;
    const pill = makeEl('span', `pill ${compat.cls}`, compat.text);
    if (target.compatibility.reason) pill.title = target.compatibility.reason;
    wrap.appendChild(pill);
    const instances = packInstances(pack, domain || packDomains(pack)[0]);
    if (pack.max_instances > 1 && instances.length) {
      const outdated = instances.some((item) => item.outdated);
      wrap.appendChild(makeEl('span', `pill ${outdated ? 'warn' : 'ok'}`, LL.t('settings.packs.installed_count', { n: instances.length, max: pack.max_instances })));
    } else if (target.installed) {
      const outdated = target.installed.outdated;
      wrap.appendChild(makeEl('span', `pill ${outdated ? 'warn' : 'ok'}`, outdated
        ? LL.t('settings.packs.installed_outdated', { installed: target.installed.version, latest: pack.version })
        : LL.t('settings.packs.installed', { version: target.installed.version })));
    }
    if (pack.verified) {
      const verified = makeEl('span', 'pill done', LL.t('settings.packs.verified'));
      verified.title = LL.t('settings.packs.verified_title');
      wrap.appendChild(verified);
    }
    const detected = detectionPill(pack);
    if (detected) wrap.appendChild(detected);
    return wrap;
  }

  // ---- detection: which packs fit the metric names the bound sources hold ----
  function packDetection(pack) {
    const det = packUi.detection;
    if (!det) return null;
    const item = (det.packs || {})[pack.id];
    if (!item || !(item.found.length + item.missing.length)) return null;
    // A pack whose domain is picked is scored against the best of its domains' sources.
    const domainInfo = (det.domains || {})[item.domain || pack.domain] || {};
    // Another source that holds the metrics counts even when the domain's own source did not answer.
    if (domainInfo.error && !item.alternative) return null;
    return item;
  }
  function packScore(pack) {
    const item = packDetection(pack);
    if (!item) return -1;
    return Math.max(item.score, item.alternative ? item.alternative.score : -1);
  }
  function detectionPill(pack) {
    const item = packDetection(pack);
    if (!item) return null;
    const alt = item.alternative;
    if (alt && alt.score > item.score) {
      const pill = makeEl('span', `pill ${alt.score >= 1 ? 'ok' : 'warn'}`, LL.t('settings.packs.detect.found_in', { source: alt.title }));
      if (alt.missing.length) pill.title = LL.t('settings.packs.detect.missing', { list: alt.missing.join(', ') });
      return pill;
    }
    const total = item.found.length + item.missing.length;
    let pill;
    if (item.score >= 1) pill = makeEl('span', 'pill ok', LL.t('settings.packs.detect.found_all'));
    else if (item.found.length) pill = makeEl('span', 'pill warn', LL.t('settings.packs.detect.found_some', { found: item.found.length, total }));
    else pill = makeEl('span', 'pill na', LL.t('settings.packs.detect.found_none'));
    if (item.missing.length) pill.title = LL.t('settings.packs.detect.missing', { list: item.missing.join(', ') });
    return pill;
  }
  async function runPackDetection() {
    packUi.detecting = true;
    packUi.detectError = '';
    renderPackDialog(false);
    try {
      packUi.detection = await postPack('detect', packScope());
    } catch (e) {
      packUi.detection = null;
      packUi.detectError = e.message;
    } finally {
      packUi.detecting = false;
      renderPackDialog(false);
    }
  }
  function detectionBar() {
    const wrap = makeEl('div');
    const bar = makeEl('div', 'struct-toolbar');
    const label = packUi.detecting ? LL.t('settings.packs.detect.running') : (packUi.detection ? LL.t('settings.packs.detect.again') : LL.t('settings.packs.detect.start'));
    const btn = makeBtn(label, runPackDetection, packUi.detection ? 'btn btn-sm' : 'btn btn-sm btn-primary');
    btn.disabled = !!packUi.detecting;
    bar.append(btn, makeEl('span', 'dim', packUi.detection
      ? LL.t('settings.packs.detect.sorted')
      : LL.t('settings.packs.detect.hint')));
    wrap.appendChild(bar);
    if (packUi.detectError) wrap.appendChild(makeEl('p', 'pack-note error', LL.t('settings.packs.detect.failed', { error: packUi.detectError })));
    if (packUi.detection) {
      Object.entries(packUi.detection.domains || {}).forEach(([domain, info]) => {
        if (info.error) wrap.appendChild(makeEl('p', 'pack-note warn', `${LL.domainTitle(domain)}: ${info.error}`));
      });
      const extra = Object.values(packUi.detection.sources || {}).filter((info) => !info.error).map((info) => info.title);
      if (extra.length) wrap.appendChild(makeEl('p', 'dim', LL.t('settings.packs.detect.extra_sources', { list: extra.join(', ') })));
    }
    return wrap;
  }
  function bundlePill(bundle) {
    const det = packUi.detection && (packUi.detection.bundles || {})[bundle.id];
    if (!det) return null;
    const tone = det.matched === det.total ? 'ok' : (det.score > 0 ? 'warn' : 'na');
    return makeEl('span', `pill ${tone}`, LL.t('settings.packs.bundle_match', { matched: det.matched, total: det.total }));
  }

  async function openPackDialog(domain, packId, instance) {
    if (state.dirty.has('queries:queries') || state.dirty.has('sla:model')) {
      ui.toast(LL.t('settings.packs.unsaved_first'), { tone: 'warn', timeout: 8000 });
      return;
    }
    packUi.forms = {};
    packUi.detection = null;
    packUi.detectError = '';
    // The wizard starts from scratch, so it reads the sources right away to suggest a bundle.
    packUi.autoDetect = !!state.wizard;
    packUi.view = packId
      ? { name: 'pack', packId, domain: domain || '', instance, back: { name: 'list', domain: domain || '' } }
      : { name: 'list', domain: domain || '' };
    await ui.dialog({
      title: LL.t('settings.packs.title'),
      wide: true,
      actions: [{ label: LL.t('settings.action.close') }],
      onOpen: (body) => { packUi.body = body; renderPackDialog(true); }
    });
    packUi.body = null;
  }
  async function renderPackDialog(reload) {
    const body = packUi.body;
    if (!body) return;
    if (reload || !packUi.catalog) {
      body.innerHTML = '';
      body.appendChild(makeEl('p', 'dim', LL.t('settings.packs.catalog_loading')));
      try {
        await loadPackCatalog();
      } catch (e) {
        body.innerHTML = '';
        body.appendChild(makeEl('p', 'binding-error', LL.t('settings.packs.catalog_failed', { error: e.message })));
        return;
      }
      if (packUi.autoDetect && !packUi.detection && !packUi.detecting) {
        packUi.autoDetect = false;
        runPackDetection();
        return;
      }
    }
    body.innerHTML = '';
    const view = packUi.view;
    const pack = view.name === 'pack' ? findPack(view.packId) : null;
    const bundle = view.name === 'bundle' ? findBundle(view.bundleId) : null;
    if (pack) body.appendChild(packCard(pack));
    else if (bundle) body.appendChild(bundleCard(bundle));
    else body.appendChild(packList(view.domain || ''));
  }
  function showPackView(view) {
    packUi.view = view;
    renderPackDialog(false);
  }

  function packList(domain) {
    const host = makeEl('div', 'pack-browser');
    host.appendChild(scopeLine());
    const tabs = makeEl('div', 'struct-tabs');
    [['', LL.t('settings.packs.all_domains')]].concat(QUERY_DOMAINS.map((d) => [d, LL.domainTitle(d)])).forEach(([key, title]) => {
      tabs.appendChild(makeBtn(title, () => showPackView({ name: 'list', domain: key }), 'btn btn-sm' + (key === domain ? ' is-active' : '')));
    });
    host.appendChild(tabs);
    host.appendChild(detectionBar());
    const catalog = packUi.catalog || {};
    if (!domain && (catalog.bundles || []).length) {
      host.appendChild(makeEl('h4', 'pack-group', LL.t('settings.packs.bundles_heading')));
      const bundleScore = (bundle) => ((packUi.detection && (packUi.detection.bundles || {})[bundle.id]) || { score: -1 }).score;
      catalog.bundles.slice().sort((a, b) => bundleScore(b) - bundleScore(a)).forEach((bundle) => {
        const row = makeEl('div', 'pack-row');
        const main = makeEl('div', 'pack-row-main');
        const head = makeEl('div', 'pack-row-title');
        head.appendChild(makeEl('strong', '', bundle.title));
        const pill = bundlePill(bundle);
        if (pill) head.appendChild(pill);
        main.append(head, makeEl('div', 'pack-row-desc', bundle.description), makeEl('div', 'dim', LL.t('settings.packs.bundle_count', { n: bundle.packs.length })));
        row.append(main, makeBtn(LL.t('settings.packs.open'), () => showPackView({ name: 'bundle', bundleId: bundle.id }), 'btn btn-sm'));
        host.appendChild(row);
      });
    }
    const packs = (catalog.packs || []).filter((p) => !domain || packDomains(p).includes(domain));
    Object.entries(catalog.categories || {}).forEach(([category, title]) => {
      const items = packs.filter((p) => p.category === category).sort((a, b) => packScore(b) - packScore(a));
      if (!items.length) return;
      host.appendChild(makeEl('h4', 'pack-group', title));
      items.forEach((pack) => host.appendChild(packRow(pack, { name: 'list', domain })));
    });
    if (!packs.length) host.appendChild(makeEl('p', 'dim', LL.t('settings.packs.none_for_domain')));
    return host;
  }
  function packDomainText(pack) {
    if (!pack.domain_picked) return LL.domainTitle(pack.domain);
    return LL.t('settings.packs.domains_choice', { list: packDomains(pack).map((d) => LL.domainTitle(d)).join(', ') });
  }
  function packRow(pack, back) {
    const domain = back && back.domain && packDomains(pack).includes(back.domain) ? back.domain : packDomains(pack)[0];
    const row = makeEl('div', 'pack-row');
    const main = makeEl('div', 'pack-row-main');
    const head = makeEl('div', 'pack-row-title');
    head.append(makeEl('strong', '', pack.title), packPills(pack, domain));
    main.append(head, makeEl('div', 'dim', `${packDomainText(pack)} · ${LL.t('settings.packs.query_count', { n: pack.queries.length })}`), makeEl('div', 'pack-row-desc', pack.description));
    row.append(main, makeBtn(LL.t('settings.packs.open'), () => showPackView({ name: 'pack', packId: pack.id, domain, back }), 'btn btn-sm'));
    return row;
  }
  function bundleCard(bundle) {
    const host = makeEl('div', 'pack-browser');
    host.appendChild(makeBtn(LL.t('settings.packs.back_to_list'), () => showPackView({ name: 'list', domain: '' }), 'btn btn-sm'));
    host.appendChild(makeEl('h4', 'pack-group', bundle.title));
    host.appendChild(makeEl('p', 'help', bundle.description));
    host.appendChild(scopeLine());
    host.appendChild(makeEl('p', 'help', LL.t('settings.packs.bundle_help')));
    bundle.packs.map(findPack).filter(Boolean).forEach((pack) => host.appendChild(packRow(pack, { name: 'bundle', bundleId: bundle.id })));
    return host;
  }
  function checkboxField(label, checked, onChange) {
    const wrap = makeEl('label', 'pack-check');
    const input = makeEl('input');
    input.type = 'checkbox';
    input.checked = !!checked;
    input.addEventListener('change', () => onChange(input.checked));
    wrap.append(input, document.createTextNode(` ${label}`));
    return wrap;
  }

  // Domain of a pack whose domain is picked at install time (resources of a database or broker host).
  function packDomainPicker(pack, domain) {
    const field = selectField(LL.t('settings.packs.domain_label'), domain, packDomains(pack).map((d) => [d, LL.domainTitle(d)]), (value) => {
      packUi.view = Object.assign({}, packUi.view, { domain: value, instance: undefined });
      renderPackDialog(false);
    });
    field.appendChild(makeEl('div', 'dim', LL.t('settings.packs.domain_pick_hint')));
    return field;
  }
  // Installation of a multi-instance pack: a new one, or one of those in the domain (to update it).
  function packInstancePicker(pack, domain, form) {
    const instances = packInstances(pack, domain);
    const full = instances.length >= pack.max_instances;
    const options = [['__new__', LL.t('settings.packs.instance_new')]].concat(instances.map((item) => [item.instance, LL.t('settings.packs.instance_option', { name: item.instance, version: item.version })]));
    const field = selectField(LL.t('settings.packs.instance_label'), form.instance || '__new__', options, (value) => {
      const installed = value === '__new__' ? null : instances.find((item) => item.instance === value) || null;
      packUi.forms[`${pack.id}|${domain}`] = packFormFor(pack, domain, installed);
      renderPackDialog(false);
    });
    const newOption = field.querySelector('option[value="__new__"]');
    if (newOption && full) newOption.disabled = true;
    field.appendChild(makeEl('div', 'dim', LL.t('settings.packs.instances_count', { n: instances.length, max: pack.max_instances })));
    return field;
  }
  // What applying does with the typed instance title: a new installation or an update of an existing one.
  function packInstanceNote(pack, domain, form) {
    const title = String(form.variables[pack.instance_variable] || '').trim();
    const instances = packInstances(pack, domain);
    const variable = pack.variables.find((v) => v.name === pack.instance_variable);
    const field = variable ? variable.title : pack.instance_variable;
    if (instances.some((item) => item.instance === title)) return { tone: '', text: LL.t('settings.packs.instance_updates', { name: title, field }) };
    if (instances.length >= pack.max_instances) return { tone: 'error', text: LL.t('settings.packs.instances_full', { max: pack.max_instances, field }) };
    return { tone: '', text: LL.t('settings.packs.instance_adds', { name: title, n: instances.length + 1, max: pack.max_instances, field }) };
  }

  // The source the pack rows read: the domain binding, or another source of the catalog that runs the pack language.
  function packSourceField(pack, domain, form) {
    const own = packOwnSource(form);
    const changed = () => { form.check = null; form.preview = null; form.previewError = ''; };
    const ctx = {
      id: `ps-${pack.id}`,
      title: pack.title,
      own: !!own,
      get: () => own || {},
      set: (binding) => { form.binding = binding; changed(); renderPackDialog(false); },
      setField: (field, value) => {
        const next = Object.assign({}, form.binding || {});
        const text = String(value || '').trim();
        if (text) next[field] = text; else delete next[field];
        form.binding = next;
        changed();
      },
      rerender: () => {},
      emptyLabel: LL.t('settings.packs.source_domain', { summary: bindingSummary(domainBindingOf(domain)) }),
      inheritNote: '',
      language: pack.language
    };
    const wrap = makeEl('div', 'ctx-field q-source');
    const label = makeEl('label', '', LL.t('settings.packs.source_label'));
    label.htmlFor = `${ctx.id}-source`;
    wrap.append(label, mountBindingEditor(ctx), makeEl('div', 'dim', LL.t('settings.packs.source_hint')));
    return wrap;
  }

  function packCard(pack) {
    const domain = packDomain(pack);
    const target = packTarget(pack, domain);
    const form = packForm(pack);
    const host = makeEl('div', 'pack-browser');
    const back = packUi.view.back || { name: 'list', domain: pack.domain };
    host.appendChild(makeBtn(back.name === 'bundle' ? LL.t('settings.packs.back_to_bundle') : LL.t('settings.packs.back_to_list'), () => showPackView(back), 'btn btn-sm'));
    const title = makeEl('div', 'pack-row-title');
    title.append(makeEl('h4', 'pack-group', pack.title), packPills(pack, domain));
    host.appendChild(title);
    host.appendChild(makeEl('p', 'help', pack.description));
    host.appendChild(makeEl('p', 'dim', LL.t('settings.packs.domain_metrics', { domain: LL.domainTitle(domain), metrics: pack.requires_metrics.join(', ') || '—' })));
    const pickers = makeEl('div', 'pack-vars');
    if (pack.domain_picked) pickers.appendChild(packDomainPicker(pack, domain));
    if (pack.max_instances > 1) pickers.appendChild(packInstancePicker(pack, domain, form));
    pickers.appendChild(packSourceField(pack, domain, form));
    host.appendChild(pickers);
    const detected = packDetection(pack);
    if (detected) {
      const parts = [];
      if (detected.found.length) parts.push(LL.t('settings.packs.detect.found_list', { list: detected.found.join(', ') }));
      if (detected.missing.length) parts.push(LL.t('settings.packs.detect.missing_list', { list: detected.missing.join(', ') }));
      host.appendChild(makeEl('p', `pack-note ${detected.score >= 1 ? '' : 'warn'}`, LL.t('settings.packs.detect.source_has', { details: parts.join('; ') })));
      const alt = detected.alternative;
      if (alt) {
        const note = makeEl('p', `pack-note ${alt.score >= 1 ? '' : 'warn'}`, LL.t('settings.packs.detect.alternative', { source: alt.title, found: alt.found.length, total: alt.found.length + alt.missing.length }));
        if (!sameBinding(packOwnSource(form), alt.binding)) {
          note.appendChild(document.createTextNode(' '));
          note.appendChild(makeBtn(LL.t('settings.packs.detect.use_source'), () => {
            form.binding = Object.assign({}, alt.binding);
            form.check = null;
            form.preview = null; form.previewError = '';
            renderPackDialog(false);
          }, 'btn btn-sm'));
        }
        host.appendChild(note);
      }
    }
    // With a source of its own the pack does not depend on the domain binding.
    const ownSource = packOwnSource(form);
    if (target.compatibility.reason && !ownSource) {
      host.appendChild(makeEl('p', `pack-note ${target.compatibility.status === 'incompatible' ? 'error' : 'warn'}`, target.compatibility.reason));
    }
    const installed = pack.max_instances > 1
      ? packInstances(pack, domain).find((item) => item.instance === form.instance) || null
      : target.installed;
    if (installed && installed.outdated) {
      host.appendChild(makeEl('p', 'pack-note warn', LL.t('settings.packs.outdated_note', { installed: installed.version, latest: pack.version, count: installed.new_queries })));
    } else if (installed) {
      host.appendChild(makeEl('p', 'help', LL.t('settings.packs.installed_note')));
    }
    host.appendChild(scopeLine());
    const incompatible = !ownSource && target.compatibility.status === 'incompatible';
    const instanceNote = makeEl('p', 'pack-note');
    const renderInstanceNote = () => {
      if (!(pack.max_instances > 1)) return;
      const note = packInstanceNote(pack, domain, form);
      instanceNote.className = `pack-note ${note.tone}`.trim();
      instanceNote.textContent = note.text;
    };

    const invalidate = () => { form.check = null; form.preview = null; form.previewError = ''; renderQueries(); renderPlan(); renderInstanceNote(); };
    if (pack.variables.length) {
      host.appendChild(makeEl('h4', 'pack-group', LL.t('settings.packs.variables')));
      const grid = makeEl('div', 'pack-vars');
      pack.variables.forEach((v) => {
        const onChange = (value) => { form.variables[v.name] = value; invalidate(); };
        const field = v.choices.length
          ? selectField(v.title, form.variables[v.name], v.choices.map((c) => [c, c]), onChange)
          : textField(v.title, form.variables[v.name], onChange);
        if (v.help) field.appendChild(makeEl('div', 'dim', v.help));
        grid.appendChild(field);
      });
      host.appendChild(grid);
    }
    if (pack.max_instances > 1) {
      host.appendChild(instanceNote);
      renderInstanceNote();
    }

    host.appendChild(makeEl('h4', 'pack-group', LL.t('settings.packs.queries_heading')));
    host.appendChild(previewWindowBar());
    const checkBar = makeEl('div', 'struct-toolbar');
    const checkBtn = makeBtn(LL.t('settings.action.check'), async () => {
      if (!includedIndexes(form).length) { ui.toast(LL.t('settings.packs.pick_one'), { tone: 'warn' }); return; }
      checkBtn.disabled = true;
      checkStatus.textContent = LL.t('settings.packs.checking');
      try {
        const indexes = includedIndexes(form);
        const data = await postPack('check', packRequest(pack, form, { start: state.previewStart, end: state.previewEnd }));
        form.check = { byIndex: Object.fromEntries(indexes.map((index, k) => [index, data.results[k]])), data };
        checkStatus.textContent = data.ok ? LL.t('settings.packs.check_ok', { source: data.source }) : LL.t('settings.packs.check_missing', { list: data.missing_required.join(', ') });
      } catch (e) {
        form.check = null;
        checkStatus.textContent = LL.t('settings.packs.check_failed', { error: e.message });
      } finally {
        checkBtn.disabled = incompatible;
        renderQueries();
      }
    }, 'btn');
    checkBtn.disabled = incompatible;
    const keepBtn = makeBtn(LL.t('settings.packs.keep_with_data'), () => {
      if (!form.check) return;
      Object.entries(form.check.byIndex).forEach(([index, result]) => { if (result && !result.ok && !pack.queries[index].required) form.include[index] = false; });
      form.preview = null; form.previewError = '';
      renderQueries();
      renderPlan();
    }, 'btn');
    const checkStatus = makeEl('span', 'dim', LL.t('settings.packs.check_hint'));
    checkBar.append(checkBtn, keepBtn, checkStatus);
    host.appendChild(checkBar);
    const tableWrap = makeEl('div');
    host.appendChild(tableWrap);

    function renderQueries() {
      tableWrap.innerHTML = '';
      keepBtn.style.display = form.check && Object.values(form.check.byIndex).some((r, i) => r && !r.ok) ? '' : 'none';
      const table = makeEl('table', 'data-table pack-queries');
      const head = makeEl('tr');
      ['', LL.t('settings.packs.col_query'), LL.t('settings.packs.col_meaning'), LL.t('settings.packs.col_unit'), LL.t('settings.packs.col_check')].forEach((text) => head.appendChild(makeEl('th', '', text)));
      table.appendChild(head);
      const renderedAll = form.preview && Array.isArray(form.preview.all_queries) ? form.preview.all_queries : null;
      pack.queries.forEach((query, index) => {
        const result = form.check ? form.check.byIndex[index] : null;
        const shown = renderedAll ? renderedAll[index] : null;
        const tr = makeEl('tr');
        const pick = makeEl('td');
        const box = makeEl('input');
        box.type = 'checkbox';
        box.checked = !!form.include[index];
        box.setAttribute('aria-label', LL.t('settings.packs.apply_query_aria', { name: query.label }));
        box.addEventListener('change', () => { form.include[index] = box.checked; form.preview = null; form.previewError = ''; renderPlan(); });
        pick.appendChild(box);
        const main = makeEl('td');
        main.appendChild(makeEl('div', 'q-item-title', (result && result.label) || (shown && shown.label) || query.label));
        const badges = makeEl('div', 'pack-pills');
        if (query.required) badges.appendChild(makeEl('span', 'pill warn', LL.t('settings.packs.required')));
        if (query.sla_field) badges.appendChild(makeEl('span', 'pill na', `SLA: ${SLA_QUERY_TITLES[query.sla_field] || query.sla_field}`));
        if (badges.childNodes.length) main.appendChild(badges);
        if (query.description) main.appendChild(makeEl('div', 'dim', query.description));
        const details = makeEl('details');
        details.appendChild(makeEl('summary', 'dim', shown ? LL.t('settings.packs.query_text') : LL.t('settings.packs.query_text_default')));
        details.appendChild(makeEl('pre', 'pack-code', shown ? shown.query : query.query));
        main.appendChild(details);
        const status = makeEl('td');
        if (result) {
          status.appendChild(makeEl('span', `pill ${result.ok ? 'ok' : (query.required ? 'error' : 'warn')}`, result.ok ? LL.t('settings.packs.series_count', { n: result.series_count }) : LL.t('settings.packs.no_data')));
          status.appendChild(makeEl('div', 'pack-check-msg', result.ok && result.series && result.series.length ? result.series.join(', ') : result.message));
        } else {
          status.appendChild(makeEl('span', 'dim', '—'));
        }
        tr.append(pick, main, makeEl('td', '', query.role_title || '—'), makeEl('td', '', query.unit_title), status);
        table.appendChild(tr);
      });
      tableWrap.appendChild(table);
    }

    host.appendChild(makeEl('h4', 'pack-group', LL.t('settings.packs.apply_heading')));
    const options = makeEl('div', 'pack-vars');
    options.appendChild(selectField(LL.t('settings.packs.mode_label'), form.mode, [
      ['merge', LL.t('settings.packs.mode_merge')],
      ['replace', LL.t('settings.packs.mode_replace')]
    ], (value) => { form.mode = value; form.preview = null; form.previewError = ''; renderPlan(); }));
    if (pack.sla_fields.length) {
      const sla = makeEl('div', 'ctx-field');
      sla.appendChild(checkboxField(LL.t('settings.packs.fill_sla', { fields: pack.sla_fields.map((f) => SLA_QUERY_TITLES[f] || f).join(', ') }), form.fillSla, (on) => { form.fillSla = on; form.preview = null; form.previewError = ''; renderPlan(); }));
      sla.appendChild(checkboxField(LL.t('settings.packs.overwrite_sla'), form.overwriteSla, (on) => { form.overwriteSla = on; form.preview = null; form.previewError = ''; renderPlan(); }));
      sla.appendChild(makeEl('div', 'field-hint', LL.t('settings.packs.dead_refs_note')));
      options.appendChild(sla);
    }
    host.appendChild(options);
    const planBox = makeEl('div', 'pack-plan');
    host.appendChild(planBox);
    const actions = makeEl('div', 'struct-toolbar');
    const previewBtn = makeBtn(LL.t('settings.packs.preview'), async () => {
      if (!includedIndexes(form).length) { ui.toast(LL.t('settings.packs.pick_one'), { tone: 'warn' }); return; }
      previewBtn.disabled = true;
      try {
        form.preview = await postPack('preview', packRequest(pack, form, applyOptions()));
        form.previewError = '';
      } catch (e) {
        form.preview = null;
        form.previewError = e.message;
      } finally {
        previewBtn.disabled = incompatible;
        renderQueries();
        renderPlan();
      }
    }, 'btn');
    previewBtn.disabled = incompatible;
    const applyBtn = makeBtn(LL.t('settings.packs.apply'), () => applyPack(pack, form, applyBtn), 'btn btn-primary');
    actions.append(previewBtn, applyBtn);
    host.appendChild(actions);

    function applyOptions() {
      return { mode: form.mode, fill_sla: !!form.fillSla, overwrite_sla: !!form.overwriteSla };
    }
    form.applyOptions = applyOptions;
    function planList(title, items, tone) {
      if (!items || !items.length) return null;
      const block = makeEl('div', `pack-plan-group ${tone || ''}`);
      block.appendChild(makeEl('strong', '', `${title} (${items.length})`));
      const list = makeEl('ul');
      items.forEach((item) => list.appendChild(makeEl('li', '', item)));
      block.appendChild(list);
      return block;
    }
    function renderPlan() {
      planBox.innerHTML = '';
      applyBtn.disabled = incompatible || !form.preview;
      if (form.previewError) { planBox.appendChild(makeEl('p', 'binding-error', form.previewError)); return; }
      if (!form.preview) {
        planBox.appendChild(makeEl('p', 'dim', LL.t('settings.packs.plan.hint')));
        return;
      }
      const plan = form.preview.plan;
      [
        planList(LL.t('settings.packs.plan.added'), plan.added, 'ok'),
        planList(LL.t('settings.packs.plan.updated'), plan.updated, 'warn'),
        planList(LL.t('settings.packs.plan.removed'), plan.removed, 'error'),
        planList(LL.t('settings.packs.plan.unchanged'), plan.unchanged, '')
      ].filter(Boolean).forEach((node) => planBox.appendChild(node));
      const sla = form.preview.sla || {};
      const changes = Object.entries(sla.changes || {}).map(([fieldName, change]) => {
        const from = change.reason === 'missing' ? LL.t('settings.packs.plan.ref_missing', { name: change.from }) : (change.from || LL.t('settings.sla.query_unset'));
        return `${SLA_QUERY_TITLES[fieldName] || fieldName}: «${from}» → «${change.to}»`;
      });
      const kept = Object.entries(sla.kept || {}).map(([fieldName, value]) => LL.t('settings.packs.plan.ref_kept', { field: SLA_QUERY_TITLES[fieldName] || fieldName, value }));
      [planList(LL.t('settings.packs.plan.sla_refs'), changes, 'ok'), planList(LL.t('settings.packs.plan.sla_refs_kept'), kept, '')].filter(Boolean).forEach((node) => planBox.appendChild(node));
      (form.preview.warnings || []).forEach((warning) => planBox.appendChild(makeEl('p', 'pack-note warn', warning)));
    }
    renderQueries();
    renderPlan();
    return host;
  }

  async function applyPack(pack, form, button) {
    if (!form.preview) return;
    const removed = form.preview.plan.removed || [];
    if (removed.length && !window.confirm(LL.t('settings.packs.remove_confirm', { n: removed.length, list: removed.join(', ') }))) return;
    const scope = packScope();
    button.disabled = true;
    try {
      const resp = await postPack('apply', packRequest(pack, form, form.applyOptions()));
      await refreshQueriesAndSla();
      offerPackUndo(resp, scope, pack.title);
      form.preview = null; form.previewError = '';
      await loadPackCatalog();
      if (pack.max_instances > 1) {
        // The card reopens on the installation just applied.
        delete packUi.forms[`${pack.id}|${packDomain(pack)}`];
        packUi.view = Object.assign({}, packUi.view, { instance: resp.instance || '' });
      }
      renderPackDialog(false);
    } catch (e) {
      ui.toast(LL.t('settings.packs.apply_failed', { error: e.message }), { tone: 'error' });
      button.disabled = false;
    }
  }
  async function refreshQueriesAndSla() {
    const cfgUrl = new URL('/config', location.origin);
    if (state.area) cfgUrl.searchParams.set('area', state.area);
    const resp = await fetch(cfgUrl);
    const cfg = await resp.json();
    if (!resp.ok) throw new Error(cfg.error || `HTTP ${resp.status}`);
    state.config.queries = cfg.queries;
    state.config.sla = cfg.sla;
    const queriesMap = isPlainObject(cfg.queries_map) ? cfg.queries_map : {};
    state.config.service_queries = Object.fromEntries(Object.entries(queriesMap).filter(([sid, value]) => sid && isPlainObject(value) && Object.keys(value).length));
    state.config.service_sla = isPlainObject(cfg.service_sla) ? cfg.service_sla : {};
    if (typeof state.reloadQueries === 'function') state.reloadQueries();
    loadSlaScope(state.slaScope);
    updateWizardNav();
  }
  function offerPackUndo(resp, scope, title) {
    const plan = resp.plan || {};
    const parts = [LL.t('settings.packs.undo.added', { n: (plan.added || []).length }), LL.t('settings.packs.undo.updated', { n: (plan.updated || []).length })];
    if ((plan.removed || []).length) parts.push(LL.t('settings.packs.undo.removed', { n: plan.removed.length }));
    if (resp.sla_changed) parts.push(LL.t('settings.packs.undo.sla'));
    const toast = ui.toast(LL.t('settings.packs.applied', { title, parts: parts.join(', ') }), { tone: 'ok', timeout: 20000 });
    (resp.warnings || []).forEach((warning) => ui.toast(warning, { tone: 'warn', timeout: 12000 }));
    const undo = makeBtn(LL.t('settings.packs.undo.button'), async () => {
      undo.disabled = true;
      try {
        await undoPackApply(resp, scope);
        toast.remove();
        ui.toast(LL.t('settings.packs.undo.done'), { tone: 'ok' });
        if (packUi.body) { await loadPackCatalog(); renderPackDialog(false); }
      } catch (e) {
        ui.toast(LL.t('settings.packs.undo.failed', { error: e.message }), { tone: 'error' });
        undo.disabled = false;
      }
    }, 'btn btn-sm');
    toast.insertBefore(undo, toast.lastChild);
  }
  async function undoPackApply(resp, scope) {
    const previous = resp.previous || {};
    const restore = async (section, value, own) => {
      if (!own && scope.area) {
        const base = `/projects/${encodeURIComponent(scope.area)}`;
        const url = scope.service ? `${base}/services/${encodeURIComponent(scope.service)}/overrides/${section}` : `${base}/overrides/${section}`;
        const r = await fetch(url, { method: 'DELETE' });
        if (!r.ok) {
          let data = {};
          try { data = await r.json(); } catch (e) { /* not JSON */ }
          throw new Error(data.error || `HTTP ${r.status}`);
        }
        return;
      }
      await postConfig(Object.assign({ section, data: value }, scope));
    };
    await restore('queries', previous.queries, previous.own_queries);
    if (resp.sla_changed) await restore('sla', previous.sla, previous.own_sla);
    await refreshQueriesAndSla();
  }
  // Title the reports show for the domain: the block's own title, else titles of its packs (Go, Python…).
  function autoDomainTitle(domain) {
    const entries = Array.isArray(queryBlock(domain).packs) ? queryBlock(domain).packs : [];
    const titles = [];
    entries.forEach((entry) => {
      const meta = findPack(entry.id);
      const title = meta && meta.domain_title ? meta.domain_title : '';
      if (title && !titles.includes(title)) titles.push(title);
    });
    return titles.length ? titles.join(' · ') : LL.domainTitle(domain);
  }
  function domainTitleField(domain) {
    const own = typeof queryBlock(domain).title === 'string' ? queryBlock(domain).title : '';
    const field = textField(LL.t('settings.queries.domain_title'), own, () => {});
    field.classList.add('ctx-field-short');
    const input = field.querySelector('input');
    input.placeholder = autoDomainTitle(domain);
    input.maxLength = 60;
    input.addEventListener('change', () => {
      editQueryBlock(domain, (block) => {
        const value = input.value.trim();
        if (value) block.title = value; else delete block.title;
      });
    });
    field.appendChild(makeEl('div', 'dim', LL.t('settings.queries.domain_title_hint')));
    return field;
  }

  function installedPacksLine(domain) {
    const entries = Array.isArray(queryBlock(domain).packs) ? queryBlock(domain).packs : [];
    if (!entries.length) return null;
    const line = makeEl('div', 'pack-installed');
    // An installation of a multi-instance pack is named by its instance variable (the host title).
    const instanceOf = (entry, meta) => {
      if (!meta || !(meta.max_instances > 1) || !meta.instance_variable) return '';
      const variables = isPlainObject(entry.variables) ? entry.variables : {};
      const variable = (meta.variables || []).find((v) => v.name === meta.instance_variable);
      return String(variables[meta.instance_variable] !== undefined ? variables[meta.instance_variable] : (variable ? variable.default : '')).trim();
    };
    const titles = entries.map((entry) => {
      const meta = findPack(entry.id);
      const instance = instanceOf(entry, meta);
      const title = meta ? meta.title : entry.id;
      return instance ? LL.t('settings.packs.installed_instance_item', { title, name: instance, version: entry.version }) : `${title} v${entry.version}`;
    });
    line.appendChild(makeEl('span', 'dim', LL.t('settings.packs.installed_line', { list: titles.join('; ') })));
    entries.forEach((entry) => {
      const meta = findPack(entry.id);
      if (!meta || !(entry.version < meta.version)) return;
      const instance = instanceOf(entry, meta);
      const text = instance
        ? LL.t('settings.packs.update_instance_to', { title: meta.title, name: instance, version: meta.version })
        : LL.t('settings.packs.update_to', { title: meta.title, version: meta.version });
      line.appendChild(makeBtn(text, () => openPackDialog(domain, entry.id, instance || undefined), 'btn btn-sm'));
    });
    return line;
  }

  // ---- JSON-only sections (system_context, queries, metrics_config) --------
  function bindJsonSection(cfg) {
    const ed = makeAce(cfg.editorId, 'ace/mode/json', 16);
    const load = () => { state.models[cfg.key] = clone(cfg.getData()); state.saved[cfg.key] = stable(state.models[cfg.key]); setJson(ed, state.models[cfg.key]); $(cfg.saveBtn).disabled = true; markDirty(`${cfg.menu}:${cfg.key}`, cfg.label(), false); $(cfg.jsonStatus).textContent = ''; if (typeof cfg.render === 'function') cfg.render(); };
    ed.session.on('change', () => {
      if (state.jsonSyncing) return;
      try {
        const parsed = JSON.parse(ed.getValue() || '{}');
        if (!isPlainObject(parsed)) throw new Error(LL.t('settings.json.expected_object'));
        state.models[cfg.key] = parsed;
        if (typeof cfg.render === 'function') cfg.render();
        $(cfg.jsonStatus).textContent = 'OK';
        const dirty = stable(parsed) !== state.saved[cfg.key];
        $(cfg.saveBtn).disabled = !dirty;
        markDirty(`${cfg.menu}:${cfg.key}`, cfg.label(), dirty);
      } catch (e) {
        $(cfg.jsonStatus).textContent = LL.t('settings.json.error', { error: e.message });
        $(cfg.saveBtn).disabled = true;
      }
    });
    $(cfg.saveBtn).addEventListener('click', async () => {
      setStatus(cfg.status, LL.t('settings.status.saving'));
      try {
        const data = await cfg.save(state.models[cfg.key]);
        state.saved[cfg.key] = stable(state.models[cfg.key]);
        $(cfg.saveBtn).disabled = true;
        markDirty(`${cfg.menu}:${cfg.key}`, cfg.label(), false);
        setStatus(cfg.status, LL.t('settings.status.saved'), 'ok');
        ui.toast(LL.t('settings.status.section_saved', { section: cfg.label() }), { tone: 'ok' });
        showWarnings(data && data.warnings);
      } catch (e) {
        setStatus(cfg.status, LL.t('settings.status.error', { error: e.message }), 'error');
      }
    });
    return { load };
  }

  // ---- prompts -------------------------------------------------------------
  const PROMPT_DOMAINS = ['overall', 'jvm', 'database', 'kafka', 'microservices', 'hard_resources', 'lt_framework', 'lt_generators', 'application_logs'];
  const PLACEHOLDER_RE = /\{[a-z_]+\}/g;
  let promptMarkers = [];

  function promptDefault(domain) { return String(state.promptDefaults[domain] || ''); }
  function promptArea(domain) { return String(state.prompts.area[domain] || promptDefault(domain)); }
  async function fetchPrompts(service) {
    const u = new URL('/prompts', location.origin);
    if (state.area) u.searchParams.set('area', state.area);
    if (service) u.searchParams.set('service', service);
    const resp = await fetch(u);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    state.promptMeta[service || ''] = {
      language: typeof data.language === 'string' && data.language ? data.language : state.promptLanguage,
      sources: isPlainObject(data.sources) ? data.sources : {},
      overrideLanguage: isPlainObject(data.override_language) ? data.override_language : {}
    };
    return isPlainObject(data.domains) ? data.domains : {};
  }
  function promptMeta(service) {
    return (service && state.promptMeta[service]) || state.promptMeta[''] || { language: state.promptLanguage, sources: {}, overrideLanguage: {} };
  }
  // An override written in another language than the reports (plans/i18n-plan.md, 3.4): the answer still follows
  // the report language. `scope` is 'service' or 'area' (the level being edited); reset is offered only when the
  // override belongs to that level. Pure, for tests.
  function promptLanguageNote(meta, domain, scope) {
    const info = meta || {};
    const source = (info.sources || {})[domain];
    const written = (info.overrideLanguage || {})[domain];
    const lang = info.language || 'ru';
    if ((source !== 'area' && source !== 'service') || !written || written === lang) return { show: false, lang };
    return {
      show: true,
      lang,
      text: written === 'en' ? LL.t('settings.prompts.override_lang.en') : LL.t('settings.prompts.override_lang.ru'),
      button: LL.t('settings.prompts.show_default', { lang }),
      canReset: source === scope
    };
  }
  function updatePromptLanguageNote() {
    const box = $('promptLanguageNote');
    if (!box) return;
    const note = promptLanguageNote(promptMeta(state.promptService), state.promptDomain, state.promptService ? 'service' : 'area');
    box.hidden = !state.area || !note.show;
    if (box.hidden) return;
    $('promptLanguageText').textContent = note.text;
    $('promptShowDefaultBtn').textContent = note.button;
    $('promptLanguageResetBtn').hidden = !note.canReset;
  }
  function promptDomainTitle(domain) {
    return domain === 'overall' ? LL.t('settings.prompts.domain_overall') : `${LL.domainTitle(domain)} (${domain})`;
  }
  async function showPromptDefault() {
    const lang = promptMeta(state.promptService).language || state.promptLanguage;
    const text = promptDefault(state.promptDomain);
    await ui.dialog({
      title: LL.t('settings.prompts.default_title', { lang, domain: promptDomainTitle(state.promptDomain) }),
      bodyHtml: '<pre class="diff-view prompt-default-view" style="max-height:360px;padding:10px;white-space:pre-wrap"></pre>',
      wide: true,
      actions: [{ label: LL.t('settings.action.close') }],
      onOpen: (bodyEl) => { const pre = bodyEl.querySelector('pre'); if (pre) pre.textContent = text; }
    });
  }
  // The language of reports changed: defaults and the language fields of /prompts are read again.
  async function reloadPromptsForLanguage() {
    try {
      const resp = await fetch('/prompts/defaults');
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      state.promptDefaults = isPlainObject(data.domains) ? data.domains : {};
      state.promptLanguage = data.language || state.promptLanguage;
      state.prompts.area = await fetchPrompts('');
      state.prompts.byService = {};
      if (!state.dirty.has('prompts:editor')) await loadPromptEditor();
      else updatePromptLanguageNote();
    } catch (e) { /* the old texts stay until the page is reloaded */ }
  }
  async function ensureServicePrompts(service) {
    if (!service || state.prompts.byService[service]) return;
    state.prompts.byService[service] = await fetchPrompts(service);
  }
  function promptEffective(service, domain) {
    const svc = service ? state.prompts.byService[service] : null;
    if (svc && typeof svc[domain] === 'string') return svc[domain];
    return promptArea(domain);
  }
  // Baseline the editor is compared against: area text for a service, file default for the area.
  function promptBaseline(service, domain) { return service ? promptArea(domain) : promptDefault(domain); }
  function refreshPromptMarkers(ed) {
    const Range = ace.require('ace/range').Range;
    promptMarkers.forEach((id) => ed.session.removeMarker(id));
    promptMarkers = [];
    ed.session.getDocument().getAllLines().forEach((line, row) => {
      let m;
      PLACEHOLDER_RE.lastIndex = 0;
      while ((m = PLACEHOLDER_RE.exec(line)) !== null) {
        promptMarkers.push(ed.session.addMarker(new Range(row, m.index, row, m.index + m[0].length), 'placeholder-marker', 'text', false));
      }
    });
  }
  function promptWarnings(text) {
    const expected = new Set((promptDefault(state.promptDomain).match(PLACEHOLDER_RE) || []));
    const present = new Set((text.match(PLACEHOLDER_RE) || []));
    const missing = Array.from(expected).filter((p) => !present.has(p));
    return missing.length ? LL.t('settings.prompts.missing_placeholders', { list: missing.join(', ') }) : '';
  }
  async function loadPromptEditor() {
    const ed = state.editors.ed_prompt;
    try { await ensureServicePrompts(state.promptService); }
    catch (e) { ui.toast(LL.t('settings.prompts.service_load_failed', { error: e.message }), { tone: 'error' }); }
    const text = promptEffective(state.promptService, state.promptDomain);
    state.promptLoaded = text;
    state.jsonSyncing = true;
    ed.setValue(text, -1);
    state.jsonSyncing = false;
    ed.setReadOnly(!state.area);
    refreshPromptMarkers(ed);
    $('promptEditorTitle').textContent = `${LL.domainTitle(state.promptDomain)} (${state.promptDomain})`;
    const overriddenForService = state.promptService && text !== promptArea(state.promptDomain);
    const overriddenForArea = promptArea(state.promptDomain) !== promptDefault(state.promptDomain);
    $('promptOverrideNote').textContent = !state.area
      ? LL.t('settings.prompts.read_only')
      : (overriddenForService ? LL.t('settings.prompts.overridden_service') : (overriddenForArea ? LL.t('settings.prompts.overridden_area') : LL.t('settings.prompts.standard')));
    $('promptWarnings').textContent = promptWarnings(text);
    $('promptDiff').style.display = 'none';
    updatePromptLanguageNote();
    updatePromptDirty();
  }
  function updatePromptDirty() {
    const ed = state.editors.ed_prompt;
    const dirty = ed.getValue() !== state.promptLoaded;
    $('promptSaveBtn').disabled = !dirty || !state.area;
    markDirty('prompts:editor', LL.t('settings.prompts.dirty_label', { domain: state.promptDomain }), dirty);
  }
  async function savePromptText(text) {
    const baseline = promptBaseline(state.promptService, state.promptDomain);
    // Text equal to the baseline removes the override instead of storing a copy.
    const payload = { area: state.area, domain: state.promptDomain, text: text === baseline ? '' : text };
    if (state.promptService) payload.service = state.promptService;
    const resp = await fetch('/prompts', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    if (state.promptService) {
      state.prompts.byService[state.promptService] = state.prompts.byService[state.promptService] || {};
      state.prompts.byService[state.promptService][state.promptDomain] = text;
    } else {
      state.prompts.area[state.promptDomain] = text;
      // Service texts inherit from the area unless overridden: refetch lazily.
      state.prompts.byService = {};
    }
    // Sources and the override language come from the server (a copy of a default is not an override).
    try {
      if (state.promptService) state.prompts.byService[state.promptService] = await fetchPrompts(state.promptService);
      else state.prompts.area = await fetchPrompts('');
    } catch (e) { /* the local copy above stays */ }
  }
  function lineDiff(a, b) {
    const x = a.split('\n'); const y = b.split('\n');
    const n = x.length; const m = y.length;
    const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
    for (let i = n - 1; i >= 0; i -= 1) for (let j = m - 1; j >= 0; j -= 1) dp[i][j] = x[i] === y[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    const out = [];
    let i = 0; let j = 0;
    while (i < n && j < m) {
      if (x[i] === y[j]) { out.push({ type: 'ctx', text: x[i] }); i += 1; j += 1; }
      else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push({ type: 'del', text: x[i] }); i += 1; }
      else { out.push({ type: 'add', text: y[j] }); j += 1; }
    }
    while (i < n) { out.push({ type: 'del', text: x[i] }); i += 1; }
    while (j < m) { out.push({ type: 'add', text: y[j] }); j += 1; }
    return out;
  }
  function togglePromptDiff() {
    const box = $('promptDiff');
    if (box.style.display !== 'none') { box.style.display = 'none'; return; }
    const baseline = promptBaseline(state.promptService, state.promptDomain);
    const current = state.editors.ed_prompt.getValue();
    const diff = lineDiff(baseline, current);
    const changed = diff.filter((d) => d.type !== 'ctx').length;
    box.innerHTML = changed
      ? `<div class="diff-line ctx">${esc(LL.t('settings.prompts.diff_head', { baseline: state.promptService ? LL.t('settings.prompts.diff_baseline_area') : LL.t('settings.prompts.diff_baseline_standard'), changed: LL.tn('settings.prompts.diff_changed', changed) }))}</div>` + diff.map((d) => `<div class="diff-line ${d.type}">${d.type === 'add' ? '+ ' : (d.type === 'del' ? '− ' : '  ')}${esc(d.text)}</div>`).join('')
      : `<div class="diff-line ctx">${LL.tHtml('settings.prompts.diff_same')}</div>`;
    box.style.display = '';
  }
  async function resetPrompt() {
    if (!requireArea(LL.t('settings.prompts.action_reset'))) return;
    const ok = await ui.confirm({ title: LL.t('settings.prompts.reset_title'), message: state.promptService ? LL.t('settings.prompts.reset_service') : LL.t('settings.prompts.reset_area'), confirmText: LL.t('settings.action.reset'), danger: true });
    if (!ok) return;
    try {
      await savePromptText(promptBaseline(state.promptService, state.promptDomain));
      await loadPromptEditor();
      ui.toast(LL.t('settings.prompts.reset_done'), { tone: 'ok' });
    } catch (e) { ui.toast(LL.t('settings.status.error', { error: e.message }), { tone: 'error' }); }
  }
  async function showPromptHistory() {
    if (!requireArea(LL.t('settings.prompts.action_history'))) return;
    let versions = [];
    try {
      const u = new URL('/prompts/history', location.origin);
      u.searchParams.set('domain', state.promptDomain);
      u.searchParams.set('area', state.area);
      if (state.promptService) u.searchParams.set('service', state.promptService);
      const resp = await fetch(u);
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      versions = data.versions || [];
    } catch (e) { ui.toast(LL.t('settings.prompts.history_failed', { error: e.message }), { tone: 'error' }); return; }
    const body = versions.length
      ? `<div class="history-list">${versions.map((v, i) => `<div class="history-item"><span>${esc(LL.formatDateTime(v.saved_at))} · ${LL.tHtml('settings.prompts.history_chars', { n: String(v.text || '').length })}</span><span><button type="button" class="btn btn-sm" data-preview="${i}">${LL.tHtml('settings.prompts.history_show')}</button> <button type="button" class="btn btn-sm btn-primary" data-restore="${i}">${LL.tHtml('settings.prompts.history_restore')}</button></span></div>`).join('')}</div><pre id="historyPreview" class="diff-view" style="display:none;max-height:260px;padding:10px"></pre>`
      : `<p>${LL.tHtml('settings.prompts.history_empty')}</p>`;
    await ui.dialog({
      title: LL.t('settings.prompts.history_title', { domain: LL.domainTitle(state.promptDomain) }),
      bodyHtml: body,
      wide: true,
      actions: [{ label: LL.t('settings.action.close') }],
      onOpen: (bodyEl, close) => {
        bodyEl.querySelectorAll('[data-preview]').forEach((btn) => btn.addEventListener('click', () => { const pre = bodyEl.querySelector('#historyPreview'); pre.textContent = versions[Number(btn.dataset.preview)].text; pre.style.display = ''; }));
        bodyEl.querySelectorAll('[data-restore]').forEach((btn) => btn.addEventListener('click', () => {
          const ed = state.editors.ed_prompt;
          ed.setValue(versions[Number(btn.dataset.restore)].text, -1);
          refreshPromptMarkers(ed);
          updatePromptDirty();
          close(true);
          ui.toast(LL.t('settings.prompts.history_loaded'), { tone: 'info' });
        }));
      }
    });
  }
  function initPrompts() {
    const ed = makeAce('ed_prompt', null, 18);
    ed.session.setMode(null);
    ed.session.on('change', () => {
      if (state.jsonSyncing) return;
      refreshPromptMarkers(ed);
      $('promptWarnings').textContent = promptWarnings(ed.getValue());
      updatePromptDirty();
    });
    const domainSel = $('promptDomainSelect');
    domainSel.innerHTML = PROMPT_DOMAINS.map((d) => `<option value="${d}">${esc(d === 'overall' ? LL.t('settings.prompts.domain_overall') : `${LL.domainTitle(d)} (${d})`)}</option>`).join('');
    domainSel.addEventListener('change', async () => { if (!(await confirmDiscardPrompt())) { domainSel.value = state.promptDomain; return; } state.promptDomain = domainSel.value; await loadPromptEditor(); });
    $('promptServiceSelect').addEventListener('change', async (e) => { if (!(await confirmDiscardPrompt())) { e.target.value = state.promptService; return; } state.promptService = e.target.value; await loadPromptEditor(); });
    $('promptSaveBtn').addEventListener('click', async () => {
      if (!requireArea(LL.t('settings.prompts.action_save'))) return;
      setStatus('promptStatus', LL.t('settings.status.saving'));
      try { await savePromptText(ed.getValue()); await loadPromptEditor(); setStatus('promptStatus', LL.t('settings.status.saved'), 'ok'); ui.toast(LL.t('settings.prompts.saved'), { tone: 'ok' }); }
      catch (e) { setStatus('promptStatus', LL.t('settings.status.error', { error: e.message }), 'error'); }
    });
    $('promptDiffBtn').addEventListener('click', togglePromptDiff);
    $('promptResetBtn').addEventListener('click', resetPrompt);
    $('promptShowDefaultBtn').addEventListener('click', showPromptDefault);
    $('promptLanguageResetBtn').addEventListener('click', resetPrompt);
    $('promptHistoryBtn').addEventListener('click', showPromptHistory);
  }
  async function confirmDiscardPrompt() {
    if (!state.dirty.has('prompts:editor')) return true;
    return ui.confirm({ title: LL.t('settings.prompts.discard_title'), message: LL.t('settings.prompts.discard_message'), confirmText: LL.t('settings.prompts.discard_confirm'), danger: true });
  }

  // ---- services --------------------------------------------------------------
  function allServiceIds() {
    const ids = new Set([...state.metricsConfigIds, ...Object.keys(state.servicesMeta)]);
    return Array.from(ids).sort();
  }
  function fillServiceSelects() {
    const options = allServiceIds().map((id) => `<option value="${esc(id)}">${esc((state.servicesMeta[id] && state.servicesMeta[id].title) || id)}</option>`).join('');
    const areaDefault = LL.t('settings.scope.area_default');
    [['slaServiceSelect', state.slaScope, areaDefault], ['queriesServiceSelect', state.queriesScope, areaDefault], ['contextServiceSelect', state.contextScope, areaDefault], ['promptServiceSelect', state.promptService, areaDefault], ['metricsServiceSelect', state.metricsScope, LL.t('settings.scope.area_whole')]].forEach(([id, current, placeholder]) => {
      const sel = $(id);
      sel.innerHTML = `<option value="">${esc(placeholder)}</option>${options}`;
      sel.value = allServiceIds().includes(current) ? current : '';
    });
  }
  function renderServices() {
    const body = $('servicesBody');
    body.innerHTML = '';
    const ids = allServiceIds();
    if (!state.area) {
      body.innerHTML = ui.emptyRow(4, LL.t('settings.services.need_area'));
      return;
    }
    if (!ids.length) {
      body.innerHTML = ui.emptyRow(4, LL.t('settings.services.empty'));
      return;
    }
    ids.forEach((id) => {
      const meta = state.servicesMeta[id] || {};
      const disabled = Array.isArray(meta.disabled_domains) ? meta.disabled_domains : [];
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${esc(id)}</strong>${state.metricsConfigIds.includes(id) ? '' : `<div class="dim">${LL.tHtml('settings.services.no_metrics_config')}</div>`}</td>
        <td><input type="text" value="${esc(meta.title || '')}" placeholder="${esc(id)}" aria-label="${LL.tHtml('settings.services.title_aria', { id })}" /></td>
        <td><div class="domain-toggle-list">${state.domainList.map((d) => `<label><input type="checkbox" data-domain="${esc(d)}" ${disabled.includes(d) ? '' : 'checked'}/> ${esc(LL.domainTitle(d))}</label>`).join('')}</div></td>
        <td class="actions"><div class="actions-inner" style="display:flex;gap:6px;justify-content:flex-end;flex-wrap:wrap"><button type="button" class="btn btn-sm svc-save">${LL.tHtml('settings.action.save')}</button><button type="button" class="btn btn-sm svc-reset">${LL.tHtml('settings.services.reset')}</button><button type="button" class="btn btn-sm btn-danger svc-delete">${LL.tHtml('settings.services.delete')}</button></div></td>`;
      tr.querySelector('.svc-save').addEventListener('click', async () => {
        const title = tr.querySelector('input[type="text"]').value.trim();
        const disabledDomains = Array.from(tr.querySelectorAll('input[data-domain]')).filter((c) => !c.checked).map((c) => c.dataset.domain);
        await saveServiceMeta(id, { title, disabled_domains: disabledDomains });
      });
      tr.querySelector('.svc-reset').addEventListener('click', () => resetServiceSettings(id));
      tr.querySelector('.svc-delete').addEventListener('click', () => deleteServiceWithData(id));
      body.appendChild(tr);
    });
  }
  async function saveServiceMeta(id, data) {
    if (!requireArea(LL.t('settings.services.action_save'))) return;
    setStatus('servicesStatus', LL.t('settings.status.saving'));
    try {
      const resp = await fetch('/service_meta', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ area: state.area, service: id, data }) });
      const j = await resp.json();
      if (!resp.ok) throw new Error(j.error || `HTTP ${resp.status}`);
      state.servicesMeta[id] = { ...(state.servicesMeta[id] || {}), ...(j.meta || {}) };
      renderServices();
      fillServiceSelects();
      setStatus('servicesStatus', LL.t('settings.status.saved'), 'ok');
      ui.toast(LL.t('settings.services.saved', { id }), { tone: 'ok' });
    } catch (e) {
      setStatus('servicesStatus', LL.t('settings.status.error', { error: e.message }), 'error');
    }
  }
  async function resetServiceSettings(id) {
    const ok = await ui.confirm({ title: LL.t('settings.services.reset_title'), message: LL.t('settings.services.reset_message', { id }), confirmText: LL.t('settings.action.reset'), danger: true });
    if (!ok) return;
    try {
      const resp = await fetch('/service_meta', { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ area: state.area, service: id }) });
      const j = await resp.json();
      if (!resp.ok) throw new Error(j.error || `HTTP ${resp.status}`);
      ui.toast(LL.t('settings.services.reset_done', { id }), { tone: 'ok' });
      location.reload();
    } catch (e) { ui.toast(LL.t('settings.status.error', { error: e.message }), { tone: 'error' }); }
  }
  async function deleteServiceWithData(id) {
    const ok = await ui.confirm({ title: LL.t('settings.services.delete_title'), message: LL.t('settings.services.delete_message', { id }), confirmText: LL.t('settings.action.delete'), danger: true });
    if (!ok) return;
    try {
      const resp = await fetch('/service', { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ area: state.area, service: id }) });
      const j = await resp.json();
      if (!resp.ok) throw new Error(j.error || `HTTP ${resp.status}`);
      ui.toast(LL.t('settings.services.deleted', { id }), { tone: 'ok' });
      location.reload();
    } catch (e) { ui.toast(LL.t('settings.status.error', { error: e.message }), { tone: 'error' }); }
  }
  async function addService() {
    if (!requireArea(LL.t('settings.services.action_add'))) return;
    const known = state.metricsConfigIds.filter((id) => !state.servicesMeta[id]);
    const id = await ui.prompt({
      title: LL.t('settings.services.add'),
      message: known.length ? LL.t('settings.services.add_known', { list: known.join(', ') }) : LL.t('settings.services.add_message', { area: state.area }),
      label: LL.t('settings.form.id'),
      list: known,
      validate: (v) => (!v.trim() ? LL.t('settings.services.id_required') : (/[\s/\\]/.test(v) ? LL.t('settings.services.id_no_spaces') : ''))
    });
    if (!id) return;
    const title = await ui.prompt({ title: LL.t('settings.services.col_title'), message: LL.t('settings.services.title_message'), label: LL.t('settings.services.title_label'), value: '', confirmText: LL.t('settings.action.add') });
    if (title === null) return;
    await saveServiceMeta(id.trim(), { title: title.trim() });
    // The server bootstraps per-service configs on the next GET /config: reload to pick them up.
    location.reload();
  }

  // ---- about ------------------------------------------------------------------------
  // THIRD_PARTY_NOTICES.md (GET /about/third-party) rendered with the vendored marked + DOMPurify, the same
  // allow-list as the report page: formatting, tables and http(s) links only. Without the libraries the text
  // is shown escaped, never as markup.
  const NOTICES_PURIFY = {
    ALLOWED_TAGS: ['p', 'br', 'b', 'strong', 'i', 'em', 'u', 's', 'del', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'blockquote', 'pre', 'code', 'hr', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'a', 'span'],
    ALLOWED_ATTR: ['href', 'title', 'align'],
    ALLOWED_URI_REGEXP: new RegExp('^(?:https?:|mailto:|/(?!/)|#)', 'i'),
  };
  function aboutMarkdown(md) {
    const source = String(md || '');
    if (!source) return '';
    const escaped = () => esc(source).replace(/\n/g, '<br>');
    if (!(window.marked && typeof window.marked.parse === 'function')) return escaped();
    if (!(window.DOMPurify && typeof window.DOMPurify.sanitize === 'function')) return escaped();
    try { return window.DOMPurify.sanitize(window.marked.parse(source), NOTICES_PURIFY); } catch (e) { return escaped(); }
  }
  function initAbout() {
    const details = $('aboutThirdParty');
    const box = $('aboutThirdPartyBody');
    if (!details || !box) return;
    let loaded = false;
    details.addEventListener('toggle', async () => {
      if (!details.open || loaded) return;
      loaded = true;
      try {
        const resp = await fetch(box.dataset.url, { headers: { Accept: 'text/plain, application/json' } });
        if (resp.status === 404) throw new Error(LL.t('about.third_party.missing'));
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        box.innerHTML = aboutMarkdown(await resp.text());
      } catch (error) {
        loaded = false;  // a transient failure: opening the block again retries
        box.innerHTML = `<p class="help">${esc(LL.t('about.third_party.failed', { error: error.message }))}</p>`;
      }
    });
  }

  // ---- menu / sections ----------------------------------------------------------
  function activateSection(key) {
    document.querySelectorAll('#settingsMenu button').forEach((b) => b.classList.toggle('active', b.dataset.section === key));
    document.querySelectorAll('.settings-section').forEach((s) => s.classList.toggle('active', s.id === `sec-${key}`));
    if (!state.wizard) { try { history.replaceState(null, '', `#${key}`); } catch (e) { /* ignore */ } }
  }

  // ---- wizard ---------------------------------------------------------------------
  // The language comes first: switching the interface language reloads the page, and the wizard starts over.
  const WIZARD_STEPS = [
    { key: 'appearance', label: LL.t('settings.wizard.step.language'), section: 'appearance' },
    { key: 'storage', label: LL.t('settings.wizard.step.storage'), section: 'connections', card: 'storage' },
    { key: 'data_sources', label: LL.t('settings.wizard.step.data_sources'), section: 'connections', card: 'data_sources' },
    { key: 'domain_sources', label: LL.t('settings.wizard.step.domain_sources'), section: 'connections', card: 'domain_sources' },
    { key: 'queries', label: LL.t('settings.wizard.step.queries'), section: 'queries' },
    { key: 'llm', label: 'LLM', section: 'connections', card: 'llm' },
    { key: 'sla', label: 'SLA', section: 'sla' },
    { key: 'done', label: LL.t('settings.wizard.step.done'), section: 'wizard-done' }
  ];
  function renderWizard() {
    const step = WIZARD_STEPS[state.wizardStep];
    const stepper = $('wizardStepper');
    // Text steps: the number (a check once done) in mono, the current step underlined.
    stepper.innerHTML = WIZARD_STEPS.map((s, i) => `<span class="step-item ${i === state.wizardStep ? 'active' : ''} ${i < state.wizardStep ? 'done' : ''}"${i === state.wizardStep ? ' aria-current="step"' : ''}><span class="step-num" aria-hidden="true">${i < state.wizardStep ? '✓' : i + 1}</span>${esc(s.label)}</span>`).join('');
    activateSection(step.section);
    CONNECTION_CARDS.forEach((card) => { const el = $(`card-${card.id}`); if (el) el.style.display = (step.card && step.card !== card.id) ? 'none' : ''; });
    $('wizardNav').style.display = step.key === 'done' ? 'none' : '';
    $('wizardPrevBtn').disabled = state.wizardStep === 0;
    updateWizardNav();
  }
  function updateWizardNav() {
    if (!state.wizard) return;
    const step = WIZARD_STEPS[state.wizardStep];
    const next = $('wizardNextBtn');
    const hint = $('wizardHint');
    if (!step || step.key === 'done') return;
    if (step.card) {
      const ok = !!state.checkOk[step.card];
      next.disabled = !ok;
      hint.textContent = ok ? LL.t('settings.wizard.hint.checked') : LL.t('settings.wizard.hint.check_or_skip');
    } else if (step.key === 'queries') {
      const ltLabels = Array.isArray(queryBlock('lt_framework').labels) ? queryBlock('lt_framework').labels.filter(Boolean) : [];
      next.disabled = !ltLabels.length;
      hint.textContent = ltLabels.length
        ? LL.t('settings.wizard.hint.queries_ok')
        : LL.t('settings.wizard.hint.queries_needed');
    } else if (step.key === 'appearance') {
      next.disabled = false;
      hint.textContent = LL.t('settings.wizard.hint.language');
    } else {
      next.disabled = false;
      hint.textContent = LL.t('settings.wizard.hint.sla');
    }
  }
  async function wizardNext(skip) {
    const step = WIZARD_STEPS[state.wizardStep];
    if (!skip) {
      if (step.card) {
        const card = CONNECTION_CARDS.find((c) => c.id === step.card);
        if (state.dirty.has(`connections:${card.id}`) && !(await saveCard(card))) return;
      } else if (step.key === 'appearance' && (state.dirty.has('appearance:language') || state.dirty.has('appearance:accent'))) {
        if (!(await saveAppearanceNow())) return;
      } else if (step.key === 'sla' && state.dirty.has('sla:model') && !(await saveSla())) return;
      else if (step.key === 'queries' && state.dirty.has('queries:queries')) $('queriesSaveBtn').click();
    }
    state.wizardStep = Math.min(WIZARD_STEPS.length - 1, state.wizardStep + 1);
    renderWizard();
  }
  function initWizard() {
    state.wizard = new URLSearchParams(location.search).get('wizard') === '1';
    if (!state.wizard) return;
    document.body.classList.add('wizard');
    $('wizardNextBtn').addEventListener('click', () => wizardNext(false));
    $('wizardSkipBtn').addEventListener('click', () => wizardNext(true));
    $('wizardPrevBtn').addEventListener('click', () => { state.wizardStep = Math.max(0, state.wizardStep - 1); renderWizard(); });
    $('wizardDemoBtn').addEventListener('click', async () => {
      const btn = $('wizardDemoBtn');
      btn.disabled = true;
      setStatus('wizardDemoStatus', LL.t('settings.wizard.demo_creating'));
      try {
        const resp = await fetch('/demo/seed', { method: 'POST' });
        const data = await resp.json();
        if (resp.status === 409 && data.report_url) { setStatus('wizardDemoStatus', LL.t('settings.wizard.demo_exists')); location.href = data.report_url; return; }
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
        setStatus('wizardDemoStatus', LL.t('settings.wizard.demo_ready'), 'ok');
        location.href = data.report_url;
      } catch (e) {
        setStatus('wizardDemoStatus', LL.t('settings.wizard.demo_failed', { error: e.message }), 'error');
        btn.disabled = false;
      }
    });
    renderWizard();
  }

  // ---- init ----------------------------------------------------------------------
  async function loadAll() {
    state.area = await LL.initProjectArea();
    const cfgUrl = new URL('/config', location.origin);
    if (state.area) cfgUrl.searchParams.set('area', state.area);
    const [cfgResp, defaultsResp] = await Promise.all([fetch(cfgUrl), fetch('/prompts/defaults')]);
    const cfg = await cfgResp.json();
    if (!cfgResp.ok) throw new Error(cfg.error || LL.t('settings.load_failed'));
    const defaults = await defaultsResp.json();
    state.config = cfg;
    // Per-service maps come as queries_map (with "" for the area) and service_sla.
    const queriesMap = isPlainObject(cfg.queries_map) ? cfg.queries_map : {};
    state.config.service_queries = Object.fromEntries(Object.entries(queriesMap).filter(([sid, value]) => sid && isPlainObject(value) && Object.keys(value).length));
    state.config.service_sla = isPlainObject(cfg.service_sla) ? cfg.service_sla : {};
    state.domainList = Array.isArray(cfg.domain_list) ? cfg.domain_list : [];
    const metricsServices = isPlainObject(cfg.metrics_config) && isPlainObject(cfg.metrics_config.services) ? cfg.metrics_config.services : {};
    state.metricsConfigIds = Object.keys(metricsServices);
    state.servicesMeta = isPlainObject(cfg.services_meta) ? clone(cfg.services_meta) : {};
    state.promptDefaults = isPlainObject(defaults.domains) ? defaults.domains : {};
    state.promptLanguage = typeof defaults.language === 'string' && defaults.language ? defaults.language : 'ru';
    state.prompts.area = await fetchPrompts('');
    state.prompts.byService = {};
  }

  const ACCENT_PRESETS = [
    { label: LL.t('settings.appearance.preset.violet'), color: '#6c5ce7' },
    { label: LL.t('settings.appearance.preset.blue'), color: '#1d4ed8' },
    { label: LL.t('settings.appearance.preset.sky'), color: '#0284c7' },
    { label: LL.t('settings.appearance.preset.teal'), color: '#0f766e' },
    { label: LL.t('settings.appearance.preset.green'), color: '#15803d' },
    { label: LL.t('settings.appearance.preset.orange'), color: '#c2410c' },
    { label: LL.t('settings.appearance.preset.red'), color: '#b91c1c' },
    { label: LL.t('settings.appearance.preset.pink'), color: '#be185d' },
    { label: LL.t('settings.appearance.preset.graphite'), color: '#64748b' }
  ];
  const DEFAULT_ACCENT = '#6c5ce7';
  const REPORT_LANGUAGES = ['ru', 'en'];
  // Saves the appearance section (accent and language); set by initAppearance, used by the wizard.
  let saveAppearanceNow = async () => true;

  function initAppearance() {
    const stored = String((state.config.appearance || {}).accent || DEFAULT_ACCENT);
    let saved = stored.toLowerCase();
    let current = saved;
    let previewTimer = 0;
    const languageSelect = $('reportLanguage');
    const storedLanguage = String((state.config.appearance || {}).language || 'ru').toLowerCase().slice(0, 2);
    let savedLanguage = REPORT_LANGUAGES.includes(storedLanguage) ? storedLanguage : 'ru';
    languageSelect.value = savedLanguage;
    languageSelect.addEventListener('change', updateDirty);
    saveAppearanceNow = saveAppearance;
    const presets = $('accentPresets');
    ACCENT_PRESETS.forEach((preset) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'accent-swatch';
      button.style.setProperty('--swatch', preset.color);
      button.title = preset.label;
      button.setAttribute('aria-label', preset.label);
      button.dataset.color = preset.color;
      button.addEventListener('click', () => choose(preset.color, false));
      presets.appendChild(button);
    });
    const custom = document.createElement('button');
    custom.type = 'button';
    custom.className = 'accent-swatch is-custom';
    custom.title = LL.t('settings.appearance.custom');
    custom.setAttribute('aria-label', LL.t('settings.appearance.custom'));
    custom.addEventListener('click', () => { $('accentCustom').hidden = false; choose(current, true); });
    presets.appendChild(custom);
    const wheel = $('accentWheel');
    wheel.addEventListener('pointerdown', (event) => { wheel.setPointerCapture(event.pointerId); pickWheel(event); });
    wheel.addEventListener('pointermove', (event) => { if (wheel.hasPointerCapture(event.pointerId)) pickWheel(event); });
    $('accentValue').addEventListener('input', () => choose(hexFromControls(), true));
    $('accentHex').addEventListener('change', () => {
      const text = $('accentHex').value.trim();
      if (/^#?[0-9a-fA-F]{6}$/.test(text)) choose(text.startsWith('#') ? text : `#${text}`, true);
    });
    $('appearanceSave').addEventListener('click', saveAppearance);
    $('appearanceReset').addEventListener('click', () => choose(DEFAULT_ACCENT, false));
    if (!/^#[0-9a-fA-F]{6}$/.test(stored)) {
      const error = $('appearanceError');
      error.hidden = false;
      error.textContent = LL.t('settings.appearance.bad_color', { color: stored });
      current = DEFAULT_ACCENT;
    }
    choose(current, !ACCENT_PRESETS.some((preset) => preset.color === current.toLowerCase()), true);

    function choose(color, customOpen, silent) {
      current = color.toLowerCase();
      $('accentCustom').hidden = !customOpen;
      $('accentHex').value = current;
      placeFromHex(current);
      presets.querySelectorAll('.accent-swatch').forEach((button) => {
        const match = button.dataset.color === current;
        button.setAttribute('aria-pressed', match ? 'true' : 'false');
      });
      custom.setAttribute('aria-pressed', customOpen ? 'true' : 'false');
      if (!silent) schedulePreview();
      updateDirty();
    }
    function updateDirty() {
      const accentDirty = current !== saved;
      const languageDirty = languageSelect.value !== savedLanguage;
      $('appearanceSave').disabled = !(accentDirty || languageDirty);
      markDirty('appearance:accent', LL.t('settings.menu.appearance'), accentDirty);
      markDirty('appearance:language', LL.t('settings.appearance.language_label'), languageDirty);
    }
    function schedulePreview() {
      clearTimeout(previewTimer);
      previewTimer = setTimeout(previewAccent, 120);
    }
    async function previewAccent() {
      const warnings = $('appearanceWarnings');
      try {
        const resp = await fetch('/appearance/palette?accent=' + encodeURIComponent(current.slice(1)));
        const body = await resp.json();
        if (!resp.ok) throw new Error(body.error || `HTTP ${resp.status}`);
        document.getElementById('accentVars').textContent = body.css || '';
        document.querySelectorAll('img.logo').forEach((img) => { img.src = body.logo_url; });
        const icon = document.getElementById('tabIcon');
        if (icon) icon.href = body.logo_url;
        warnings.textContent = (body.warnings || []).join(' ');
      } catch (error) {
        warnings.textContent = error.message;
      }
    }
    async function saveAppearance() {
      setStatus('appearanceStatus', LL.t('settings.status.saving'));
      const language = languageSelect.value;
      const languageChanged = language !== savedLanguage;
      try {
        // The server merges the section: other keys (language_switch) stay as they are.
        await postConfig(configBody('appearance', { accent: current, language }));
        saved = current;
        savedLanguage = language;
        state.config.appearance = Object.assign({}, state.config.appearance || {}, { accent: current, language });
        updateDirty();
        $('appearanceError').hidden = true;
        setStatus('appearanceStatus', LL.t('settings.status.saved'), 'ok');
        ui.toast(LL.t('settings.appearance.saved'), { tone: 'ok' });
        if (languageChanged) reloadPromptsForLanguage();
        return true;
      } catch (error) {
        setStatus('appearanceStatus', error.message, 'error');
        return false;
      }
    }
    function pickWheel(event) {
      const rect = wheel.getBoundingClientRect();
      const dx = event.clientX - (rect.left + rect.width / 2);
      const dy = event.clientY - (rect.top + rect.height / 2);
      const hue = (Math.atan2(dx, -dy) * 180 / Math.PI + 360) % 360;
      const saturation = Math.min(1, Math.hypot(dx, dy) / (rect.width / 2));
      $('accentMarker').style.left = `${rect.width / 2 + Math.sin(hue * Math.PI / 180) * saturation * rect.width / 2}px`;
      $('accentMarker').style.top = `${rect.height / 2 - Math.cos(hue * Math.PI / 180) * saturation * rect.height / 2}px`;
      choose(hsvToHex(hue, saturation, Number($('accentValue').value) / 100), true);
    }
    function placeFromHex(color) {
      const hsv = hexToHsv(color);
      if (!hsv) return;
      $('accentValue').value = String(Math.round(hsv.v * 100));
      const size = wheel.clientWidth || 180;
      const radius = hsv.s * size / 2;
      const angle = hsv.h * Math.PI / 180;
      $('accentMarker').style.left = `${size / 2 + Math.sin(angle) * radius}px`;
      $('accentMarker').style.top = `${size / 2 - Math.cos(angle) * radius}px`;
    }
    function hexFromControls() {
      const hsv = hexToHsv($('accentHex').value) || { h: 0, s: 1, v: 0.7 };
      return hsvToHex(hsv.h, hsv.s, Number($('accentValue').value) / 100);
    }
  }
  function hsvToHex(hue, saturation, value) {
    const chroma = value * saturation;
    const x = chroma * (1 - Math.abs((hue / 60) % 2 - 1));
    const match = value - chroma;
    const sector = Math.floor(hue / 60) % 6;
    const channels = [[chroma, x, 0], [x, chroma, 0], [0, chroma, x], [0, x, chroma], [x, 0, chroma], [chroma, 0, x]][sector];
    const hex = channels.map((channel) => Math.round((channel + match) * 255).toString(16).padStart(2, '0')).join('');
    return `#${hex}`;
  }
  function hexToHsv(color) {
    const match = String(color || '').trim().match(/^#?([0-9a-fA-F]{6})$/);
    if (!match) return null;
    const value = parseInt(match[1], 16);
    const red = ((value >> 16) & 255) / 255;
    const green = ((value >> 8) & 255) / 255;
    const blue = (value & 255) / 255;
    const max = Math.max(red, green, blue);
    const min = Math.min(red, green, blue);
    const delta = max - min;
    let hue = 0;
    if (delta) {
      if (max === red) hue = ((green - blue) / delta) % 6;
      else if (max === green) hue = (blue - red) / delta + 2;
      else hue = (red - green) / delta + 4;
      hue = (hue * 60 + 360) % 360;
    }
    return { h: hue, s: max ? delta / max : 0, v: max };
  }

  // ---- pure parts for tests (tests/test_i18n_settings.py) ------------------------------------------
  // Every user-visible string of the static tables and field schemas, for the language checks.
  function schemaTexts() {
    const out = [];
    const push = (value) => { if (typeof value === 'string' && value) out.push(value); };
    const fieldTexts = (field) => {
      push(field.label);
      push(field.hint);
      push(field.placeholder);
      (field.options || []).forEach((o) => push(typeof o === 'string' ? o : o.l));
    };
    CONNECTION_CARDS.forEach((card) => { push(card.title); push(card.desc); (card.fields || []).forEach(fieldTexts); });
    Object.keys(SOURCE_TYPES).forEach((type) => { push(SOURCE_TYPES[type].label); push(SOURCE_TYPES[type].hint); sourceFields('x', type).forEach(fieldTexts); });
    Object.values(BINDING_FIELDS).forEach((field) => { push(field.label); push(field.placeholder); });
    Object.keys(GENERATOR_TYPES).forEach((type) => { push(GENERATOR_TYPES[type].label); push(GENERATOR_TYPES[type].hint); generatorFields('x', type).forEach(fieldTexts); });
    toolFields('x').forEach(fieldTexts);
    Object.values(KITS).forEach((kit) => { push(kit.empty); push(kit.addLabel); push(kit.deleteLabel); push(kit.hint({})); push(kit.pill({})); });
    Object.values(WIZARD_STEP_TITLES).forEach(push);
    [SLA_FIELDS, SLA_PEAK_FIELDS, SLA_FORECAST_FIELDS].forEach((list) => list.forEach((f) => { push(f.label); push(f.desc); push(f.hint); push(f.unit); }));
    push(SLA_SHARED_DESC);
    [SLA_PRESETS, LATENCY_UNIT_OPTIONS, LOAD_MODEL_OPTIONS, LLM_PROVIDERS].forEach((list) => list.forEach((o) => push(o.l)));
    Object.values(SLA_QUERY_TITLES).forEach(push);
    [LEVEL_OPTIONS, DEP_KIND_OPTIONS, ARCH_STYLES].forEach((list) => list.forEach(([, text]) => push(text)));
    CONTEXT_GROUPS.forEach((group) => {
      push(group.title);
      push(group.hint);
      (group.lists || []).forEach(([, title, columns]) => { push(title); columns.forEach(([, label]) => push(label)); });
      (group.lines || []).forEach(([, title]) => push(title));
    });
    Object.values(CONTEXT_IMPORT_ACTIONS).forEach(push);
    Object.values(PACK_COMPAT).forEach((compat) => push(compat.text));
    WIZARD_STEPS.forEach((step) => push(step.label));
    ACCENT_PRESETS.forEach((preset) => push(preset.label));
    return out;
  }
  LL.settings = {
    _internals: {
      schemaTexts, sourceSummary, generatorSummary, generatorStatus, toolSummary, toolBadge, hostKeyMessage,
      queryOptions, deadRefText, contextGroupSummary, slaUsageText, CONTEXT_GROUPS, promptLanguageNote, WIZARD_STEPS,
      aboutMarkdown, roleOptions, sharedRoleLabel, QUERY_NOTE_MAX
    }
  };

  document.addEventListener('DOMContentLoaded', async () => {
    try {
      const servicesBody = $('servicesBody');
      if (servicesBody) servicesBody.innerHTML = ui.skeletonRows(4, 3);
      await loadAll();
    } catch (e) {
      ui.toast(e.message, { tone: 'error', timeout: 0 });
      return;
    }
    ui.theme.onChange(() => Object.values(state.editors).forEach((ed) => ed.setTheme(aceTheme())));

    renderAreaBar();

    renderConnectionCards();

    makeAce('ed_sla', 'ace/mode/json', 10);
    state.editors.ed_sla.session.on('change', () => {
      if (state.jsonSyncing) return;
      try { const parsed = JSON.parse(state.editors.ed_sla.getValue() || '{}'); if (!isPlainObject(parsed)) throw new Error(LL.t('settings.json.expected_object')); state.models.sla = parsed; $('slaJsonStatus').textContent = 'OK'; renderSlaForm({ skipJson: true }); }
      catch (e) { $('slaJsonStatus').textContent = LL.t('settings.json.error', { error: e.message }); }
    });
    $('slaSaveBtn').addEventListener('click', saveSla);
    $('slaServiceSelect').addEventListener('change', (e) => loadSlaScope(e.target.value));

    const contextSection = bindJsonSection({
      key: 'system_context', menu: 'context', editorId: 'ed_system_context', saveBtn: 'contextSaveBtn', status: 'contextStatus', jsonStatus: 'contextJsonStatus',
      label: () => (state.contextScope ? LL.t('settings.dirty.context_scoped', { scope: state.contextScope }) : LL.t('settings.menu.context')),
      getData: () => effectiveContext(state.contextScope),
      save: async (data) => {
        const r = await postConfig(configBody('system_context', data, state.contextScope));
        clearContextImport();
        renderContextForm();
        if (state.contextScope) {
          state.config.service_system_context = state.config.service_system_context || {};
          // A context equal to the area's is not stored: the service follows the area again.
          if (r && r.inherited) delete state.config.service_system_context[state.contextScope];
          else state.config.service_system_context[state.contextScope] = clone(data);
        } else {
          state.config.system_context = clone(data);
        }
        updateContextOverrideNote();
        return r;
      },
      render: renderContextForm
    });
    $('contextServiceSelect').addEventListener('change', (e) => {
      state.contextScope = e.target.value;
      clearContextImport();
      contextSection.load();
      updateContextOverrideNote();
    });
    state.reloadContext = () => contextSection.load();
    $('contextImportBtn').addEventListener('click', () => openContextImport());
    const queriesSection = bindJsonSection({
      key: 'queries', menu: 'queries', editorId: 'ed_queries', saveBtn: 'queriesSaveBtn', status: 'queriesStatus', jsonStatus: 'queriesJsonStatus',
      label: () => (state.queriesScope ? LL.t('settings.dirty.queries_scoped', { scope: state.queriesScope }) : LL.t('settings.menu.queries')),
      getData: () => {
        const base = isPlainObject(state.config.queries) ? state.config.queries : {};
        if (!state.queriesScope) return base;
        const override = (state.config.service_queries || {})[state.queriesScope];
        return deepMerge(base, isPlainObject(override) ? override : {});
      },
      save: async (data) => {
        const r = await postConfig(configBody('queries', data, state.queriesScope));
        if (state.queriesScope) { state.config.service_queries = state.config.service_queries || {}; state.config.service_queries[state.queriesScope] = clone(data); }
        else state.config.queries = clone(data);
        return r;
      },
      render: renderQueriesForm
    });
    $('queriesServiceSelect').addEventListener('change', (e) => { state.queriesScope = e.target.value; queriesSection.load(); });
    state.reloadQueries = () => queriesSection.load();
    // Role titles and installed pack names in the queries list; the dialog reloads the catalog itself. An area the
    // settings do not know has no catalog (404): the list keeps role ids and the area bar says what to do.
    if (areaKnown()) loadPackCatalog().then(() => { renderQueriesForm(); renderSlaForm({ skipJson: true }); }).catch(() => { /* the list falls back to role ids */ });

    const metricsSection = bindJsonSection({
      key: 'metrics_config', menu: 'metrics', editorId: 'ed_metrics_config', saveBtn: 'metricsSaveBtn', status: 'metricsStatus', jsonStatus: 'metricsJsonStatus',
      label: () => (state.metricsScope ? LL.t('settings.dirty.metrics_scoped', { scope: state.metricsScope }) : LL.t('settings.menu.metrics')),
      getData: () => {
        const map = isPlainObject(state.config.metrics_config_map) ? state.config.metrics_config_map : {};
        return map[state.metricsScope || ''] || (state.metricsScope ? {} : (state.config.metrics_config || {}));
      },
      save: async (data) => {
        if (!state.area) throw new Error(LL.t('settings.area.pick'));
        const r = await postConfig(configBody('metrics_config', data, state.metricsScope));
        state.config.metrics_config_map = state.config.metrics_config_map || {};
        state.config.metrics_config_map[state.metricsScope || ''] = clone(data);
        return r;
      }
    });
    $('metricsServiceSelect').addEventListener('change', (e) => { state.metricsScope = e.target.value; metricsSection.load(); });

    initAppearance();
    initAbout();
    initPrompts();
    fillServiceSelects();
    renderServices();
    loadSlaScope('');
    contextSection.load();
    updateContextOverrideNote();
    queriesSection.load();
    metricsSection.load();
    await loadPromptEditor();

    document.querySelectorAll('#settingsMenu button').forEach((btn) => btn.addEventListener('click', () => activateSection(btn.dataset.section)));
    $('addServiceBtn').addEventListener('click', addService);
    $('dirtySaveAll').addEventListener('click', async () => {
      for (const card of CONNECTION_CARDS) if (state.dirty.has(`connections:${card.id}`)) await saveCard(card);
      if (state.dirty.has('sla:model')) await saveSla();
      if (state.dirty.has('context:system_context')) $('contextSaveBtn').click();
      if (state.dirty.has('queries:queries')) $('queriesSaveBtn').click();
      if (state.dirty.has('metrics:metrics_config')) $('metricsSaveBtn').click();
      if (state.dirty.has('prompts:editor')) $('promptSaveBtn').click();
      if (state.dirty.has('appearance:accent') || state.dirty.has('appearance:language')) $('appearanceSave').click();
    });

    initWizard();
    // "#<section>:<element id>" (links of the project readiness checklist) also scrolls to the element and
    // highlights its row for 2 s; elements rendered after the section loads are waited for up to 3 s.
    const revealTarget = (id, attempt) => {
      const target = document.getElementById(id);
      if (!target) {
        if ((attempt || 0) < 30) setTimeout(() => revealTarget(id, (attempt || 0) + 1), 100);
        return;
      }
      for (let node = target.parentElement; node; node = node.parentElement) if (node.tagName === 'DETAILS') node.open = true;
      const box = target.closest('.sla-row, .sla-field, .sla-check') || target;
      const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      box.scrollIntoView({ block: 'center', behavior: reduced ? 'auto' : 'smooth' });
      if (!target.disabled && target.matches('input, select, textarea, button')) target.focus({ preventScroll: true });
      document.querySelectorAll('.is-highlighted').forEach((node) => node.classList.remove('is-highlighted'));
      void box.offsetWidth;  // restart the highlight animation on a repeated click
      box.classList.add('is-highlighted');
      clearTimeout(box.highlightTimer);
      box.highlightTimer = setTimeout(() => box.classList.remove('is-highlighted'), 2000);
    };
    const activateFromHash = () => {
      if (state.wizard) return;
      let raw = (location.hash || '').replace('#', '');
      try { raw = decodeURIComponent(raw); } catch (e) { /* malformed escape: use as is */ }
      const [hash, targetId] = raw.split(':');
      if (hash && document.getElementById(`sec-${hash}`)) {
        activateSection(hash);
        if (targetId) revealTarget(targetId, 0);
      }
    };
    activateFromHash();
    window.addEventListener('hashchange', activateFromHash);
  });
})();
