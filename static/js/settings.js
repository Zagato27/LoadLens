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
    servicesMeta: {},
    domainList: [],
    metricsConfigIds: [],
    models: {},
    saved: {},
    editors: {},
    jsonSyncing: false,
    slaScope: '',
    queriesScope: '',
    metricsScope: '',
    promptService: '',
    promptDomain: 'overall',
    promptLoaded: '',
    checkOk: {},
    domainLang: {},
    domainLangPicker: {},
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
    ui.toast(`${action}: сначала выберите область в шапке страницы`, { tone: 'warn' });
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
      if (has && !mark) { mark = document.createElement('span'); mark.className = 'dirty-mark'; mark.textContent = '●'; mark.title = 'Есть несохранённые изменения'; btn.appendChild(mark); }
      if (!has && mark) mark.remove();
    });
  }
  window.addEventListener('beforeunload', (e) => {
    if (state.dirty.size) { e.preventDefault(); e.returnValue = ''; }
  });

  // ---- project areas -------------------------------------------------------
  function renderAreaBar() {
    const label = $('areaScopeLabel');
    const note = $('areaScopeNote');
    if (state.area) {
      const meta = (LL.projectAreas || []).find((a) => a && a.id === state.area);
      label.textContent = `Область: ${meta && meta.title ? meta.title : state.area}`;
      note.textContent = 'Каталог источников данных общий. Привязка доменов, LLM, SLA, запросы, промпты и визуализации сохраняются как переопределения этой области. Хранилище и Confluence общие.';
    } else {
      label.textContent = 'Глобальные настройки';
      note.textContent = 'Выберите область в шапке, чтобы задать переопределения для неё; без области редактируются значения по умолчанию для всех областей.';
    }
  }

  // ---- connection cards ----------------------------------------------------
  const SSL_MODES = ['disable', 'allow', 'prefer', 'require', 'verify-ca', 'verify-full'];
  const LLM_PROVIDERS = [
    { v: 'perplexity', l: 'Perplexity' }, { v: 'openai', l: 'OpenAI-совместимый API' }, { v: 'anthropic', l: 'Anthropic' }, { v: 'gigachat', l: 'GigaChat' }
  ];
  const providerFields = (p, extra) => ([
    { path: `${p}.api_base_url`, label: 'API URL', type: 'text', showIf: (m) => m.provider === p },
    { path: `${p}.model`, label: 'Модель', type: 'combo', showIf: (m) => m.provider === p },
    ...(extra || []),
    { path: `${p}.max_concurrent`, label: 'Параллельных запросов', type: 'number', showIf: (m) => m.provider === p },
    { path: `${p}.request_timeout_sec`, label: 'Таймаут запроса, с', type: 'number', showIf: (m) => m.provider === p },
    { path: `${p}.generation.temperature`, label: 'Temperature', type: 'number', step: '0.05', showIf: (m) => m.provider === p },
    { path: `${p}.generation.max_tokens`, label: 'Макс. токенов ответа', type: 'number', hint: 'Резерв под ответ: ввод урезается под остаток окна.', showIf: (m) => m.provider === p },
    { path: `${p}.context_window_tokens`, label: 'Контекстное окно, токенов', type: 'number', hint: 'Ввод и ответ вместе не должны превышать окно модели.', showIf: (m) => m.provider === p && (p === 'anthropic' || p === 'gigachat') }
  ]);
  const apiKeyField = (p) => ({ path: `${p}.api_key`, label: 'API-ключ', type: 'password', showIf: (m) => m.provider === p });
  const CONNECTION_CARDS = [
    {
      id: 'storage', section: 'storage.timescale', title: 'Хранилище (TimescaleDB)', scope: 'global',
      desc: 'База, в которой хранятся метрики, отчёты и статусы задач. Нужна для работы всех страниц.',
      fields: [
        { path: 'host', label: 'Хост', type: 'text', placeholder: 'localhost' },
        { path: 'port', label: 'Порт', type: 'number', placeholder: '5432' },
        { path: 'dbname', label: 'База данных', type: 'text' },
        { path: 'user', label: 'Пользователь', type: 'text' },
        { path: 'password', label: 'Пароль', type: 'password' },
        { path: 'sslmode', label: 'SSL', type: 'select', options: SSL_MODES },
        { path: 'schema', label: 'Схема', type: 'text', placeholder: 'public' }
      ]
    },
    {
      id: 'data_sources', section: 'data_sources', title: 'Источники данных', scope: 'global', kind: 'catalog', testable: false,
      desc: 'Подключения к Grafana, Prometheus и InfluxDB: адрес и учётная запись. Одно подключение можно назначить нескольким доменам.',
      fields: []
    },
    {
      id: 'domain_sources', section: 'domain_sources', title: 'Источники доменов', scope: 'area', kind: 'bindings', testable: false,
      desc: 'Откуда каждый домен берёт метрики. Для Grafana выберите датасорс — список приходит из самой Grafana. Своя привязка области заменяет глобальную целиком.',
      fields: []
    },
    {
      id: 'default_params', section: 'default_params', title: 'Параметры выборки', scope: 'area', testable: false,
      desc: 'Шаг запроса к источникам метрик и интервал агрегации временных рядов перед анализом.',
      fields: [
        { path: 'step', label: 'Шаг выборки', type: 'text', placeholder: '1m', hint: 'Гранулярность запроса: 30s, 1m, 5m' },
        { path: 'resample_interval', label: 'Интервал агрегации', type: 'text', placeholder: '5T', hint: 'Pandas offset: 5T = 5 минут, 1H = час' }
      ]
    },
    {
      id: 'logs_source', section: 'logs_source', title: 'Логи приложений (OpenSearch)', scope: 'area',
      desc: 'Необязательный домен: агрегированные ERROR-логи за окно теста попадают в анализ ИИ.',
      fields: [
        { path: 'enabled', label: 'Включить домен «Логи приложений»', type: 'checkbox' },
        { path: 'opensearch.base_url', label: 'URL OpenSearch Dashboards', type: 'text', placeholder: 'https://opensearch:5601' },
        { path: 'opensearch.index_pattern', label: 'Шаблон индекса', type: 'text', placeholder: 'app-logs-*' },
        { path: 'opensearch.username_env', label: 'Логин', type: 'text' },
        { path: 'opensearch.password_env', label: 'Пароль', type: 'password' },
        { path: 'opensearch.verify_ssl', label: 'Проверять TLS-сертификат', type: 'checkbox' }
      ]
    },
    {
      id: 'confluence', section: 'confluence', title: 'Confluence: публикация веб-отчёта', scope: 'global',
      desc: 'Публикация готового отчёта страницей в Confluence кнопкой на странице отчёта (вкладки анализа и графики).',
      fields: [
        { path: 'enabled', label: 'Разрешить публикацию', type: 'checkbox' },
        { path: 'base_url', label: 'URL Confluence', type: 'text', placeholder: 'https://confluence.example.ru' },
        { path: 'username', label: 'Логин', type: 'text' },
        { path: 'password', label: 'Пароль', type: 'password' },
        { path: 'space_key', label: 'Ключ пространства', type: 'text' },
        { path: 'parent_page_id', label: 'ID родительской страницы', type: 'text', hint: 'Отчёты создаются дочерними страницами' },
        { path: 'forecast_parent_page_id', label: 'ID родительской страницы прогнозов', type: 'text', hint: 'Отчёты прогноза мощностей создаются дочерними страницами; пусто — та же страница, что для отчётов' },
        { path: 'verify_ssl', label: 'Проверять TLS-сертификат', type: 'checkbox' }
      ]
    },
    {
      id: 'confluence_template', section: 'confluence_template', title: 'Confluence по шаблону, Grafana render, Loki', scope: 'global',
      desc: 'Классический отчёт при создании: копия страницы-шаблона, снимки панелей Grafana и выгрузка логов Loki. Идентификаторы шаблона задаются в разделе «Визуализации».',
      fields: [
        { path: 'url_basic', label: 'URL Confluence', type: 'text', placeholder: 'https://confluence.example.ru' },
        { path: 'space_conf', label: 'Ключ пространства', type: 'text' },
        { path: 'user', label: 'Логин Confluence', type: 'text' },
        { path: 'password', label: 'Пароль Confluence', type: 'password' },
        { path: 'grafana_base_url', label: 'URL Grafana (render)', type: 'text', placeholder: 'http://grafana:3000' },
        { path: 'grafana_login', label: 'Логин Grafana', type: 'text' },
        { path: 'grafana_pass', label: 'Пароль Grafana', type: 'password' },
        { path: 'loki_url', label: 'URL Loki query_range', type: 'text', placeholder: 'http://loki:3100/loki/api/v1/query_range' }
      ]
    },
    {
      id: 'llm', section: 'llm', title: 'LLM-провайдер (анализ ИИ)', scope: 'area',
      desc: 'Модель, которая пишет доменные и итоговый отчёты. Проверка отправляет короткий запрос и тратит немного токенов.',
      fields: [
        { path: 'provider', label: 'Провайдер', type: 'select', options: LLM_PROVIDERS },
        { path: 'max_domain_workers', label: 'Доменов параллельно', type: 'number', hint: 'Сколько доменов анализировать одновременно' },
        { path: 'self_consistency_k', label: 'Кандидатов на домен', type: 'number', hint: 'Число вариантов ответа, из которых судья выбирает лучший' },
        ...providerFields('perplexity', [apiKeyField('perplexity')]),
        ...providerFields('openai', [apiKeyField('openai')]),
        ...providerFields('anthropic', [apiKeyField('anthropic')]),
        ...providerFields('gigachat', [
          { path: 'gigachat.cert_file', label: 'Файл сертификата (mTLS)', type: 'text', hint: 'Путь на сервере LoadLens', showIf: (m) => m.provider === 'gigachat' },
          { path: 'gigachat.key_file', label: 'Файл ключа (mTLS)', type: 'text', showIf: (m) => m.provider === 'gigachat' },
          { path: 'gigachat.verify', label: 'Проверять TLS-сертификат API', type: 'checkbox', showIf: (m) => m.provider === 'gigachat' }
        ])
      ]
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
    return String(raw);
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
    input.placeholder = field.placeholder || (names.length ? 'Выберите или введите модель' : '');
    input.value = current;
    picker.appendChild(input);
    if (!names.length) return picker;

    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'model-picker-toggle';
    toggle.setAttribute('aria-label', 'Показать список моделей');
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
        empty.textContent = 'Нет подходящих моделей';
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
      wrap.innerHTML = `<label for="${id}">${esc(field.label)}</label><div class="password-field"><input type="password" id="${id}" value="${esc(value === undefined || value === null ? '' : value)}" autocomplete="new-password" /><button type="button" class="btn btn-reveal" aria-label="Показать">показать</button></div>${field.hint ? `<div class="field-hint">${esc(field.hint)}</div>` : ''}`;
      const input = wrap.querySelector('input');
      wrap.querySelector('.btn-reveal').addEventListener('click', (e) => {
        const show = input.type === 'password';
        input.type = show ? 'text' : 'password';
        e.currentTarget.textContent = show ? 'скрыть' : 'показать';
      });
    } else {
      wrap.innerHTML = `<label for="${id}">${esc(field.label)}</label><input type="${field.type === 'number' ? 'number' : 'text'}" id="${id}" ${field.step ? `step="${field.step}"` : ''} value="${esc(value === undefined || value === null ? '' : value)}" placeholder="${esc(field.placeholder || '')}" />${field.hint ? `<div class="field-hint">${esc(field.hint)}</div>` : ''}`;
    }
    const control = wrap.querySelector('input, select');
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
        ? (state.area ? `переопределение для области «${state.area}»` : 'глобальные значения по умолчанию')
        : 'общие для всех областей';
      const el = document.createElement('article');
      el.className = 'conn-card';
      el.id = `card-${card.id}`;
      const bodyClass = card.kind ? 'card-body' : 'form-grid three';
      el.innerHTML = `
        <div class="conn-head"><h3>${esc(card.title)}</h3><span class="conn-actions">${card.id === 'llm' ? '<button type="button" class="btn card-models">Получить список моделей</button>' : ''}${card.testable === false ? '' : '<button type="button" class="btn card-test">Проверить соединение</button>'}</span></div>
        <p class="help">${esc(card.desc)} <span class="dim">(${esc(scopeNote)})</span></p>
        <div class="${bodyClass}"></div>
        <div class="conn-result" role="status"></div>
        <details class="report-collapsible" style="margin-top:12px">
          <summary><span>Расширенные параметры (JSON)</span><small class="report-collapsible-subtitle">все ключи раздела ${esc(card.section)}</small></summary>
          <div class="report-collapsible-body">
            <div class="editor-wrap"><div class="editor-header"><span class="editor-title">${esc(card.section)}</span><span class="dim json-status"></span></div><div id="ed_${card.id}" class="ace-host"></div></div>
          </div>
        </details>
        <div class="card-actions">
          <button type="button" class="btn btn-primary card-save" disabled>Сохранить</button>
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
            if (!isPlainObject(parsed)) throw new Error('ожидается объект');
            state.models[card.id] = parsed;
            status.textContent = 'OK';
            refreshCardFields(card);
            updateCardDirty(card);
          } catch (e) {
            status.textContent = `Ошибка JSON: ${e.message}`;
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
    result.textContent = 'Запрашиваем список моделей…';
    btn.disabled = true;
    try {
      const resp = await fetch('/config/llm_models', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(configBody('llm', state.models[card.id])) });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      state.llmModels = Array.isArray(data.models) ? data.models : [];
      refreshCardFields(card);
      result.className = `conn-result ${data.ok ? 'ok' : 'error'}`;
      result.textContent = data.message || (data.ok ? `Доступно моделей: ${state.llmModels.length}` : 'Провайдер не отдаёт список моделей');
    } catch (e) {
      result.className = 'conn-result error';
      result.textContent = `Не удалось получить список моделей: ${e.message}`;
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
    result.textContent = 'Проверяем…';
    if (btn) btn.disabled = true;
    try {
      const body = configBody(card.section, state.models[card.id]);
      if (extra) Object.assign(body, extra);
      const resp = await fetch('/config/test_connection', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      result.className = `conn-result ${data.ok ? 'ok' : 'error'}`;
      result.innerHTML = `${esc(data.message)} <span class="dim">(${data.elapsed_ms} мс)</span>${!data.ok && data.details && data.details.error ? `<pre>${esc(data.details.error)}</pre>` : ''}`;
      if (data.ok && (!card.kind || card.kind === 'catalog' || (extra && extra.domain === 'default'))) state.checkOk[card.id] = true;
      else if (!card.kind) state.checkOk[card.id] = false;
      updateWizardNav();
    } catch (e) {
      result.className = 'conn-result error';
      result.textContent = `Не удалось выполнить проверку: ${e.message}`;
      if (!card.kind) state.checkOk[card.id] = false;
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  async function saveCard(card) {
    const el = $(`card-${card.id}`);
    const status = el.querySelector('.card-status');
    status.textContent = 'Сохранение…';
    try {
      const data = await postConfig(configBody(card.section, state.models[card.id]));
      state.saved[card.id] = stable(state.models[card.id]);
      storeSectionData(card.section, clone(state.models[card.id]));
      if (card.id === 'data_sources') {
        state.grafanaDatasources = {};
        const bindings = CONNECTION_CARDS.find((item) => item.id === 'domain_sources');
        if (bindings) renderBindingsBody(bindings);
      }
      updateCardDirty(card);
      status.textContent = 'Сохранено';
      ui.toast(`${card.title}: сохранено`, { tone: 'ok' });
      showWarnings(data.warnings);
      setTimeout(() => { if (status.textContent === 'Сохранено') status.textContent = ''; }, 2000);
      return true;
    } catch (e) {
      status.textContent = `Ошибка: ${e.message}`;
      ui.toast(`${card.title}: ${e.message}`, { tone: 'error' });
      return false;
    }
  }

  // ---- data source catalog and domain bindings ----------------------------
  const SOURCE_ID_RE = /^[a-z0-9][a-z0-9_-]{0,39}$/;
  const SOURCE_TYPES = {
    grafana_proxy: { label: 'Grafana', baseId: 'grafana', hint: 'Запросы идут через Grafana: PromQL, InfluxQL и Flux по её датасорсам. Датасорс выбирается у каждого домена.' },
    prometheus: { label: 'Prometheus', baseId: 'prometheus', hint: 'PromQL-запросы напрямую в Prometheus или совместимый API.' },
    influxdb: { label: 'InfluxDB', baseId: 'influxdb', hint: 'Flux-запросы напрямую в InfluxDB 2.x. Bucket задаётся у домена.' }
  };
  const DATASOURCE_MODE_LABELS = { promql: 'PromQL', influxql: 'InfluxQL', flux: 'Flux', sql: 'SQL' };
  const BINDING_FIELDS = {
    datasource_name: { label: 'Имя датасорса', placeholder: 'как в Grafana' },
    datasource_uid: { label: 'UID датасорса', placeholder: 'если имя не уникально' },
    database: { label: 'База InfluxQL', placeholder: 'например, k6' },
    bucket: { label: 'Bucket для Flux', placeholder: 'подставляется вместо {bucket}' }
  };
  const hasOwn = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);

  function blankSource(type, title) {
    if (type === 'prometheus') return { title, type, prometheus: { url: '' } };
    if (type === 'influxdb') return { title, type, influxdb: { url: '', org: '', token: '' } };
    return { title, type: 'grafana_proxy', grafana: { base_url: '', verify_ssl: false, auth: { method: 'basic', username: '', password: '', token: '' } } };
  }
  function sourceTypeLabel(type) {
    return (SOURCE_TYPES[type] || {}).label || type || 'тип не задан';
  }
  function sourceUrl(entry) {
    if (entry.type === 'grafana_proxy') return String(getPath(entry, 'grafana.base_url') || '');
    if (entry.type === 'prometheus') return String(getPath(entry, 'prometheus.url') || '');
    if (entry.type === 'influxdb') return String(getPath(entry, 'influxdb.url') || '');
    return '';
  }
  function sourceSummary(id, entry) {
    const parts = [`id ${id}`, sourceUrl(entry) || 'адрес не указан'];
    if (entry.type === 'grafana_proxy') {
      parts.push(getPath(entry, 'grafana.auth.method') === 'bearer' ? 'вход по API-токену' : `логин ${getPath(entry, 'grafana.auth.username') || 'не указан'}`);
    }
    if (entry.type === 'influxdb' && getPath(entry, 'influxdb.org')) parts.push(`организация ${getPath(entry, 'influxdb.org')}`);
    return parts.join(' · ');
  }
  function sourceFields(id, type) {
    const path = (suffix) => `${id}.${suffix}`;
    const fields = [{ path: path('title'), label: 'Название', type: 'text', group: 'main' }];
    if (type === 'prometheus') {
      fields.push({ path: path('prometheus.url'), label: 'URL Prometheus', type: 'text', placeholder: 'http://prometheus:9090', group: 'main' });
    } else if (type === 'influxdb') {
      fields.push(
        { path: path('influxdb.url'), label: 'URL InfluxDB', type: 'text', placeholder: 'http://influxdb:8086', group: 'main' },
        { path: path('influxdb.org'), label: 'Организация', type: 'text', group: 'auth' },
        { path: path('influxdb.token'), label: 'Токен', type: 'password', group: 'auth' }
      );
    } else if (type === 'grafana_proxy') {
      const bearer = (m) => getPath(m, path('grafana.auth.method')) === 'bearer';
      fields.push(
        { path: path('grafana.base_url'), label: 'URL Grafana', type: 'text', placeholder: 'http://grafana:3000', group: 'main' },
        { path: path('grafana.auth.method'), label: 'Авторизация', type: 'select', options: [{ v: 'basic', l: 'Логин и пароль' }, { v: 'bearer', l: 'API-токен' }], group: 'auth' },
        { path: path('grafana.auth.username'), label: 'Логин', type: 'text', showIf: (m) => !bearer(m), group: 'auth' },
        { path: path('grafana.auth.password'), label: 'Пароль', type: 'password', showIf: (m) => !bearer(m), group: 'auth' },
        { path: path('grafana.auth.token'), label: 'API-токен', type: 'password', showIf: bearer, group: 'auth' },
        { path: path('grafana.verify_ssl'), label: 'Проверять TLS-сертификат Grafana', type: 'checkbox', group: 'flag' }
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
    body.appendChild(makeEl('p', 'dim', (SOURCE_TYPES[entry.type] || {}).hint || 'Этот тип источника не поддерживается: удалите его и добавьте заново.'));
    const groups = { main: makeEl('div', 'source-form-main'), auth: makeEl('div', 'source-form-auth'), flag: makeEl('div', 'source-form-flags') };
    sourceFields(id, entry.type).forEach((field) => groups[field.group].appendChild(renderField(card, field)));
    Object.values(groups).forEach((group) => { if (group.childElementCount) body.appendChild(group); });
    const actions = makeEl('div', 'q-item-actions');
    actions.appendChild(makeBtn('Удалить источник', () => deleteSource(card, id), 'btn btn-sm btn-danger'));
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
    const check = makeBtn('Проверить', (event) => {
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
      title: 'Удалить источник',
      message: `Источник «${id}» пропадёт из каталога после сохранения. Если на него ссылаются домены, сохранение не пройдёт и покажет, где он используется.`,
      confirmText: 'Удалить',
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
    if (!title) return { error: 'Укажите название' };
    if (!SOURCE_ID_RE.test(id)) return { error: 'Идентификатор: латиница в нижнем регистре, цифры, дефис и подчёркивание, до 40 символов' };
    if (used.has(id)) return { error: `Источник «${id}» уже есть` };
    return { value: { id, type, title } };
  }
  async function openAddSource(card) {
    const model = state.models[card.id];
    const used = new Set(Object.keys(model));
    const options = Object.keys(SOURCE_TYPES).map((type) => `<option value="${type}">${esc(SOURCE_TYPES[type].label)}</option>`).join('');
    const created = await ui.dialog({
      title: 'Новый источник данных',
      bodyHtml: [
        `<label>Тип<select id="dsType">${options}</select></label>`,
        '<p class="dim" id="dsTypeHint"></p>',
        '<label>Название<input id="dsTitle" type="text" maxlength="80" /></label>',
        '<label>Идентификатор<input id="dsId" type="text" maxlength="40" spellcheck="false" autocomplete="off" /></label>',
        '<p class="dim">По идентификатору домены ссылаются на источник, после сохранения его не изменить.</p>',
        '<div class="dialog-error" id="dsError"></div>'
      ].join(''),
      onOpen: (body) => wireAddSourceForm(body, used),
      actions: [
        { label: 'Отмена' },
        {
          label: 'Добавить',
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
    if (!ids.length) list.appendChild(makeEl('p', 'dim', 'Источников пока нет. Добавьте подключение к Grafana, Prometheus или InfluxDB.'));
    ids.forEach((id) => list.appendChild(sourceItem(card, id, isPlainObject(model[id]) ? model[id] : {})));
    const bar = makeEl('div', 'struct-toolbar');
    bar.appendChild(makeBtn('Добавить источник', () => openAddSource(card)));
    host.append(list, bar);
    updateFieldVisibility(card);
  }
  function savedCatalog() {
    return isPlainObject(state.config.data_sources) ? state.config.data_sources : {};
  }
  function savedSource(id) {
    const entry = savedCatalog()[id];
    return isPlainObject(entry) ? entry : null;
  }
  function bindingRows() {
    return [{ id: 'default', title: 'По умолчанию', note: 'для доменов без своей строки' }]
      .concat(QUERY_DOMAINS.map((id) => ({ id, title: LL.domainTitle(id) })));
  }
  function bindingSummary(binding) {
    if (!isPlainObject(binding) || !binding.source) return 'источник по умолчанию не выбран';
    const entry = savedSource(binding.source);
    const parts = [entry ? String(entry.title || binding.source) : `${binding.source} (нет в каталоге)`];
    const datasource = binding.datasource_name || binding.datasource_uid;
    if (datasource) parts.push(datasource);
    if (binding.database) parts.push(`база ${binding.database}`);
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
        : { status: 'error', items: [], message: data.message || 'Grafana не вернула список датасорсов' };
    } catch (e) {
      next = { status: 'error', items: [], message: e.message };
    }
    state.grafanaDatasources[sourceId] = next;
    const card = CONNECTION_CARDS.find((item) => item.id === 'domain_sources');
    if (card) renderBindingsBody(card);
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
    if (ds.is_default) parts.push('по умолчанию в Grafana');
    return parts.join(' · ');
  }
  function commitBindings(card) {
    setJson(state.editors[`ed_${card.id}`], state.models[card.id]);
    renderBindingsBody(card);
    updateCardDirty(card);
  }
  function setBindingSource(card, domain, sourceId) {
    const model = state.models[card.id];
    if (sourceId) model[domain] = { source: sourceId };
    else delete model[domain];
    delete state.bindingChecks[domain];
    commitBindings(card);
  }
  function chooseDatasource(card, domain, ds) {
    const model = state.models[card.id];
    const binding = { source: String((model[domain] || {}).source || '') };
    if (ds) {
      binding.datasource_uid = ds.uid;
      binding.datasource_name = ds.name;
      if (ds.mode === 'influxql' && ds.database) binding.database = ds.database;
      if (ds.mode === 'flux' && ds.bucket) binding.bucket = ds.bucket;
    }
    model[domain] = binding;
    delete state.bindingChecks[domain];
    commitBindings(card);
  }
  function setBindingField(card, domain, field, value) {
    const model = state.models[card.id];
    if (!isPlainObject(model[domain])) model[domain] = {};
    const text = String(value || '').trim();
    if (text) model[domain][field] = text;
    else delete model[domain][field];
    setJson(state.editors[`ed_${card.id}`], model);
    updateCardDirty(card);
  }
  function bindingField(card, domain, field, value) {
    const meta = BINDING_FIELDS[field];
    const wrap = makeEl('div', 'binding-field');
    const id = `bf-${domain}-${field}`;
    const label = makeEl('label', '', meta.label);
    label.htmlFor = id;
    const input = makeEl('input');
    input.type = 'text';
    input.id = id;
    input.value = value || '';
    input.placeholder = meta.placeholder;
    input.addEventListener('change', () => setBindingField(card, domain, field, input.value));
    wrap.append(label, input);
    return wrap;
  }
  function bindingFieldGrid(card, domain, binding, fields) {
    const grid = makeEl('div', 'binding-extra');
    fields.forEach((field) => grid.appendChild(bindingField(card, domain, field, binding[field])));
    return grid;
  }
  function datasourceSelect(card, row, binding, items) {
    const select = makeEl('select');
    select.setAttribute('aria-label', `Датасорс Grafana: ${row.title}`);
    const current = findDatasource(items, binding);
    const placeholder = makeEl('option', '', 'Выберите датасорс');
    placeholder.value = '';
    select.appendChild(placeholder);
    const stored = binding.datasource_name || binding.datasource_uid;
    if (!current && stored) {
      const missing = makeEl('option', '', `${stored} — нет в Grafana`);
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
      chooseDatasource(card, row.id, items.find((ds) => ds.uid === select.value) || null);
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
  function grafanaParams(card, row, binding, sourceId, box) {
    const list = datasourceList(sourceId);
    if (list.status === 'loading') {
      box.appendChild(makeEl('div', 'binding-note', 'Загружаем датасорсы из Grafana…'));
      return;
    }
    if (list.status === 'error') {
      const note = makeEl('div', 'binding-error');
      note.append(makeEl('span', '', `Список датасорсов недоступен: ${list.message}`), makeBtn('Повторить', () => {
        delete state.grafanaDatasources[sourceId];
        renderBindingsBody(card);
      }, 'btn btn-sm'));
      box.append(note, bindingFieldGrid(card, row.id, binding, ['datasource_name', 'datasource_uid', 'database', 'bucket']));
      return;
    }
    if (!list.items.length) box.appendChild(makeEl('div', 'binding-error', 'В Grafana нет датасорсов Prometheus или InfluxDB'));
    const { select, current } = datasourceSelect(card, row, binding, list.items);
    box.appendChild(select);
    const extra = grafanaExtraFields(binding, current);
    if (extra.length) box.appendChild(bindingFieldGrid(card, row.id, binding, extra));
  }
  function bindingParams(card, row, model) {
    const box = makeEl('div', 'binding-params');
    if (row.id !== 'default' && !hasOwn(model, row.id)) {
      box.appendChild(makeEl('div', 'binding-note', `Как по умолчанию: ${bindingSummary(model.default)}`));
      return box;
    }
    const binding = isPlainObject(model[row.id]) ? model[row.id] : {};
    const sourceId = String(binding.source || '');
    const entry = sourceId ? savedSource(sourceId) : null;
    if (!sourceId) box.appendChild(makeEl('div', 'binding-note', 'Выберите источник'));
    else if (!entry) box.appendChild(makeEl('div', 'binding-error', 'Источника нет в сохранённом каталоге: сохраните каталог или выберите другой'));
    else if (entry.type === 'prometheus') box.appendChild(makeEl('div', 'binding-note', 'Дополнительных параметров нет: запросы идут прямо в Prometheus'));
    else if (entry.type === 'influxdb') box.appendChild(bindingFieldGrid(card, row.id, binding, ['bucket']));
    else grafanaParams(card, row, binding, sourceId, box);
    return box;
  }
  function bindingSourceSelect(card, row, model) {
    const own = row.id === 'default' || hasOwn(model, row.id);
    const current = own && isPlainObject(model[row.id]) ? String(model[row.id].source || '') : '';
    const select = makeEl('select');
    select.setAttribute('aria-label', `Источник: ${row.title}`);
    const first = makeEl('option', '', row.id === 'default' ? 'Выберите источник' : 'Как по умолчанию');
    first.value = '';
    select.appendChild(first);
    const catalog = savedCatalog();
    Object.keys(catalog).forEach((id) => {
      const entry = isPlainObject(catalog[id]) ? catalog[id] : {};
      const title = String(entry.title || id);
      const typeLabel = sourceTypeLabel(entry.type);
      const option = makeEl('option', '', title.toLowerCase().includes(typeLabel.toLowerCase()) ? title : `${title} · ${typeLabel}`);
      option.value = id;
      select.appendChild(option);
    });
    if (current && !hasOwn(catalog, current)) {
      const missing = makeEl('option', '', `${current} — нет в каталоге`);
      missing.value = current;
      select.appendChild(missing);
    }
    select.value = current;
    select.addEventListener('change', () => setBindingSource(card, row.id, select.value));
    const cell = makeEl('div', 'binding-source');
    cell.appendChild(select);
    return cell;
  }
  function bindingRow(card, row, model) {
    const wrap = makeEl('div', 'binding-row');
    const domain = makeEl('div', 'binding-domain');
    domain.appendChild(makeEl('div', '', row.title));
    if (row.note) domain.appendChild(makeEl('div', 'dim', row.note));
    const result = makeEl('div', 'conn-result');
    result.setAttribute('role', 'status');
    const actions = makeEl('div', 'binding-actions');
    actions.appendChild(makeBtn('Проверить', () => runCheck(card, { domain: row.id }, result, 'bindingChecks', row.id), 'btn btn-sm'));
    wrap.append(domain, bindingSourceSelect(card, row, model), bindingParams(card, row, model), actions, result);
    restoreCheck('bindingChecks', row.id, result);
    return wrap;
  }
  function bindingHeader() {
    const head = makeEl('div', 'binding-row binding-head');
    head.setAttribute('aria-hidden', 'true');
    ['Домен', 'Источник', 'Датасорс и параметры', ''].forEach((text) => head.appendChild(makeEl('div', '', text)));
    return head;
  }
  function renderBindingsBody(card) {
    const host = document.querySelector(`#card-${card.id} .card-body`);
    if (!host) return;
    const model = isPlainObject(state.models[card.id]) ? state.models[card.id] : {};
    state.models[card.id] = model;
    host.replaceChildren();
    if (!Object.keys(savedCatalog()).length) {
      host.appendChild(makeEl('p', 'dim', 'Сначала добавьте и сохраните источник в карточке «Источники данных».'));
      return;
    }
    const list = makeEl('div', 'binding-list');
    list.appendChild(bindingHeader());
    bindingRows().forEach((row) => list.appendChild(bindingRow(card, row, model)));
    host.appendChild(list);
  }

  // ---- SLA -----------------------------------------------------------------
  const SLA_FIELDS = [
    { key: 'target_rps', queryKey: 'max_performance_query', domain: 'lt_framework', label: 'Целевой RPS', unit: 'RPS', term: 'target_rps', desc: 'Главный критерий: достигнут на стабильной ступени выбранного запроса.' },
    { key: 'max_error_rate_pct', queryKey: 'error_rate_query', domain: 'lt_framework', label: 'Доля ошибок', unit: '%', term: 'error_rate', desc: '95-й перцентиль участка stable_max, до просадки RPS. Хвост после плато не входит. Для soak — всё окно. Единицы запроса.' },
    { key: 'max_p95_ms', queryKey: 'p95_query', domain: 'lt_framework', label: 'P95 задержки', unit: 'мс', term: 'p95', desc: '95-й перцентиль участка stable_max, до просадки RPS. Хвост после плато не входит. Для soak — всё окно. Единицы запроса.' },
    { key: 'max_p99_ms', queryKey: 'p99_query', domain: 'lt_framework', label: 'P99 задержки', unit: 'мс', desc: '95-й перцентиль участка stable_max, до просадки RPS. Хвост после плато не входит. Для soak — всё окно. Единицы запроса.' },
    { key: 'max_cpu_pct', queryKey: 'cpu_query', domain: 'hard_resources', label: 'CPU узлов', unit: '%', desc: '95-й перцентиль участка stable_max, до просадки RPS. Хвост после плато не входит. Для soak — всё окно. Единицы запроса.' },
    { key: 'max_memory_pct', queryKey: 'memory_query', domain: 'hard_resources', label: 'Память узлов', unit: '%', desc: '95-й перцентиль участка stable_max, до просадки RPS. Хвост после плато не входит. Для soak — всё окно. Единицы запроса.' }
  ];
  const SLA_PRESETS = [{ v: 'strict', l: 'strict — строгий' }, { v: 'balanced', l: 'balanced — сбалансированный' }, { v: 'lenient', l: 'lenient — мягкий' }];
  const LATENCY_UNIT_OPTIONS = [{ v: 'ms', l: 'миллисекунды' }, { v: 's', l: 'секунды' }];
  const LOAD_MODEL_OPTIONS = [{ v: 'open', l: 'открытая — задаётся RPS (arrival-rate)' }, { v: 'closed', l: 'закрытая — задаются VU' }];

  function slaModel() { return state.models.sla; }
  function slaSavedKey() { return `sla:${state.slaScope}`; }
  function effectiveSla(scope) {
    const base = isPlainObject(state.config.sla) ? state.config.sla : {};
    if (!scope) return clone(base);
    const override = isPlainObject((state.config.service_sla || {})[scope]) ? state.config.service_sla[scope] : {};
    return clone(deepMerge(base, override));
  }
  function domainLabels(domain, scope) {
    const queries = isPlainObject(state.config.queries) ? state.config.queries : {};
    let block = isPlainObject(queries[domain]) ? queries[domain] : {};
    if (scope) {
      const svcQueries = (state.config.service_queries || {})[scope];
      if (isPlainObject(svcQueries) && isPlainObject(svcQueries[domain])) block = deepMerge(block, svcQueries[domain]);
    }
    return Array.isArray(block.labels) ? block.labels.map(String).filter(Boolean) : [];
  }
  function queryOptions(labels, current) {
    const values = labels.slice();
    if (current && !values.includes(current)) values.push(current);
    return [{ v: '', l: 'не задано' }].concat(values.map((label) => ({ v: label, l: label })));
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
      const options = queryOptions(domainLabels(field.domain, state.slaScope), currentQuery);
      const row = document.createElement('div');
      row.className = 'sla-row';
      row.innerHTML = `
        <input type="checkbox" id="sla-on-${field.key}" ${enabled ? 'checked' : ''} aria-label="Проверять: ${esc(field.label)}" />
        <div><label for="sla-val-${field.key}" class="sla-name" style="margin:0">${field.term ? ui.term(field.term, field.label) : esc(field.label)}</label></div>
        <input type="number" id="sla-val-${field.key}" value="${enabled ? esc(value) : ''}" ${enabled ? '' : 'disabled'} step="any" />
        <span class="dim">${esc(field.unit)}</span>
        <select id="sla-query-${field.key}" aria-label="Запрос для критерия: ${esc(field.label)}">${options.map((o) => `<option value="${esc(o.v)}" ${o.v === currentQuery ? 'selected' : ''}>${esc(o.l)}</option>`).join('')}</select>
        <div class="sla-desc">${esc(field.desc)}</div>`;
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
      });
      rows.appendChild(row);
    });
    const peak = $('slaPeakFields');
    peak.innerHTML = '';
    const fields = [
      { key: 'min_stable_minutes', label: 'Мин. длительность стабильной ступени, мин', type: 'number', step: '0.5' },
      { key: 'step_detection_preset', label: 'Чувствительность детектора ступеней', type: 'select', options: SLA_PRESETS },
      { key: 'target_rps_allow_peak_fallback', label: 'Если стабильной ступени нет — использовать пиковый RPS', type: 'checkbox', term: 'peak_max' },
      { key: 'step_detection_enabled', label: 'Использовать детектор ступеней (step-профиль)', type: 'checkbox', term: 'step_profile' },
      { key: 'debug_peak_logging', label: 'Подробные логи расчёта RPS (для диагностики)', type: 'checkbox' },
      { key: 'mean_latency_query', label: 'Запрос среднего времени отклика (прогноз мощностей)', type: 'query', domain: 'lt_framework' },
      { key: 'vus_query', label: 'Запрос числа VU (прогноз мощностей)', type: 'query', domain: 'lt_framework' },
      { key: 'latency_unit', label: 'Единицы времени отклика в запросах p95 и среднего', type: 'select', options: LATENCY_UNIT_OPTIONS },
      { key: 'load_model', label: 'Модель нагрузки (прогноз мощностей)', type: 'select', options: LOAD_MODEL_OPTIONS, hint: 'Открытая: параллелизм = RPS × среднее время отклика. Закрытая: параллелизм = число VU.' },
      { key: 'service_cpu_query', label: 'Запрос CPU экземпляров сервисов (прогноз мощностей)', type: 'query', domain: 'jvm', hint: 'Доля CPU одного экземпляра от 0 до 1 по (сервис, instance), например process_cpu_usage by (application, instance). По нему считается, сколько подов нужно для цели.' }
    ];
    fields.forEach((f) => {
      const wrap = document.createElement('div');
      const id = `sla-peak-${f.key}`;
      const value = model[f.key];
      const labelHtml = f.term ? ui.term(f.term, f.label) : esc(f.label);
      if (f.type === 'checkbox') {
        wrap.innerHTML = `<label class="checkbox-row"><input type="checkbox" id="${id}" ${value ? 'checked' : ''}/> <span>${labelHtml}</span></label>`;
      } else if (f.type === 'select' || f.type === 'query') {
        const current = value === undefined || value === null ? '' : String(value);
        const options = f.type === 'query'
          ? queryOptions(domainLabels(f.domain, state.slaScope), current)
          : f.options;
        wrap.innerHTML = `<label for="${id}">${labelHtml}</label><select id="${id}">${options.map((o) => `<option value="${esc(o.v)}" ${o.v === current ? 'selected' : ''}>${esc(o.l)}</option>`).join('')}</select>${f.hint ? `<div class="field-hint">${esc(f.hint)}</div>` : ''}`;
      } else {
        wrap.innerHTML = `<label for="${id}">${labelHtml}</label><input type="number" id="${id}" step="${f.step || 'any'}" value="${value === undefined || value === null ? '' : esc(value)}" />`;
      }
      const control = wrap.querySelector('input, select');
      control.addEventListener('change', () => {
        if (f.type === 'checkbox') model[f.key] = control.checked;
        else if (f.type === 'number') { const n = Number(control.value); model[f.key] = control.value === '' || !Number.isFinite(n) ? null : n; }
        else model[f.key] = control.value;
        afterSlaChange();
      });
      peak.appendChild(wrap);
    });
    ui.applyTerms($('sec-sla'));
    if (!skipJson) setJson(state.editors.ed_sla, model);
    updateSlaDirty();
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
    setStatus('slaStatus', 'Сохранение…');
    try {
      const data = await postConfig(configBody('sla', slaModel(), state.slaScope));
      if (state.slaScope) { state.config.service_sla = state.config.service_sla || {}; state.config.service_sla[state.slaScope] = clone(slaModel()); }
      else state.config.sla = clone(slaModel());
      state.saved[slaSavedKey()] = stable(slaModel());
      updateSlaDirty();
      setStatus('slaStatus', 'Сохранено', 'ok');
      ui.toast('SLA-критерии сохранены', { tone: 'ok' });
      showWarnings(data.warnings);
      return true;
    } catch (e) {
      setStatus('slaStatus', `Ошибка: ${e.message}`, 'error');
      return false;
    }
  }

  // ---- structured context and queries (JSON stays the stored shape) ------
  const QUERY_DOMAINS = ['jvm', 'database', 'kafka', 'microservices', 'hard_resources', 'lt_framework'];
  const LT_QUERY_PRIORITY = ['influxql', 'promql', 'flux'];
  const QUERY_LANG = {
    promql: { query: 'promql_queries', keys: 'label_keys_list', label: 'PromQL' },
    influxql: { query: 'influxql_queries', keys: 'label_tag_keys_list', label: 'InfluxQL' },
    flux: { query: 'flux_queries', keys: 'label_tag_keys_list', label: 'Flux' }
  };
  // Positional arrays of one domain; a row operation is applied to all of them.
  const QUERY_PARALLEL_KEYS = ['labels', 'promql_queries', 'label_keys_list', 'influxql_queries', 'flux_queries', 'label_tag_keys_list'];
  const SLA_QUERY_TITLES = {
    max_performance_query: 'макс. RPS', error_rate_query: 'ошибки', p95_query: 'P95',
    p99_query: 'P99', cpu_query: 'CPU', memory_query: 'память',
    mean_latency_query: 'среднее время (прогноз)', vus_query: 'VU (прогноз)',
    service_cpu_query: 'CPU сервисов (прогноз)'
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
    if (!window.confirm(`«${from}» используется в SLA (${slaUsageText(used)}). Обновить ссылки на «${to}»?`)) return;
    used.forEach((key) => {
      if (state.models.sla) state.models.sla[key] = to;
      if (state.config.sla) state.config.sla[key] = to;
    });
    if (state.editors.ed_sla && state.models.sla) setJson(state.editors.ed_sla, state.models.sla);
    ui.toast('Ссылки в SLA обновлены в форме. Сохраните раздел SLA.', { tone: 'warn', timeout: 8000 });
  }

  // ---- system context form -------------------------------------------------
  const LEVEL_OPTIONS = [['', '—'], ['high', 'высокая'], ['medium', 'средняя'], ['low', 'низкая']];
  const DEP_KIND_OPTIONS = [['', '—'], ['sync', 'синхронная'], ['async', 'асинхронная']];
  const ARCH_STYLES = [['', '—'], ['microservices', 'Микросервисы'], ['monolith', 'Монолит'], ['modular-monolith', 'Модульный монолит'], ['event-driven', 'Событийная'], ['serverless', 'Serverless']];
  const CONTEXT_LIST_FIELDS = new Set(['technologies', 'used_by', 'steps', 'success_signals']);
  const CONTEXT_GROUPS = [
    { key: 'system', title: 'О системе', open: true, hint: 'Передаётся ИИ вместе с метриками каждого домена.' },
    {
      key: 'architecture', title: 'Архитектура',
      hint: 'Компоненты и зависимости помогают ИИ связывать метрики разных доменов.',
      lists: [
        ['components', 'Компоненты', [['name', 'Имя'], ['role', 'Роль'], ['criticality', 'Критичность', LEVEL_OPTIONS], ['technologies', 'Технологии, через запятую']]],
        ['dependencies', 'Зависимости', [['from', 'Откуда'], ['to', 'Куда'], ['kind', 'Тип', DEP_KIND_OPTIONS], ['purpose', 'Назначение']]],
        ['data_stores', 'Хранилища', [['id', 'Имя'], ['type', 'Тип: PostgreSQL, Redis…'], ['purpose', 'Назначение'], ['used_by', 'Кем используется, через запятую']]]
      ]
    },
    {
      key: 'load_model', title: 'Модель нагрузки',
      hint: 'Показывает ИИ, какие точки входа и сценарии важнее для бизнеса.',
      lists: [
        ['entrypoints', 'Точки входа', [['name', 'Имя'], ['kind', 'Тип: http, kafka…'], ['business_priority', 'Приоритет', LEVEL_OPTIONS]]],
        ['critical_user_flows', 'Критичные сценарии', [['name', 'Сценарий'], ['steps', 'Шаги, через запятую'], ['success_signals', 'Признаки успеха, через запятую']]]
      ],
      lines: [['expected_hotspots', 'Ожидаемые узкие места']]
    },
    {
      key: 'operational_context', title: 'Эксплуатация',
      hint: 'Попадает в промпт как правила заказчика: что считать нормой и на чём сосредоточиться.',
      lines: [
        ['normal_degradation_rules', 'Нормальная деградация — не считать проблемой'],
        ['known_risks', 'Известные риски — проверить по данным'],
        ['known_constraints', 'Ограничения'],
        ['analysis_focus', 'Фокус анализа']
      ]
    }
  ];

  function contextSnapshot() {
    return isPlainObject(state.models.system_context) ? state.models.system_context : {};
  }
  function writeContext(mutator) {
    const next = clone(contextSnapshot());
    mutator(next);
    commitStructured('system_context', next, 'ed_system_context', 'contextSaveBtn', 'context', 'Контекст системы');
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
  }

  function contextEnabledToggle(ctx) {
    const label = makeEl('label', 'checkbox-row');
    const input = makeEl('input');
    input.type = 'checkbox';
    input.checked = ctx.enabled !== false;
    input.addEventListener('change', () => writeContext((next) => { next.enabled = input.checked; }));
    label.append(input, makeEl('span', '', ' Передавать контекст в анализ ИИ'));
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
      body.appendChild(selectField('Стиль архитектуры', section.style, ARCH_STYLES, (v) => setContextPath('architecture', 'style', v)));
    }
    (group.lists || []).forEach(([key, title, columns]) => body.appendChild(contextTable(group.key, key, title, columns, section[key])));
    (group.lines || []).forEach(([key, title]) => body.appendChild(contextLines(group.key, key, title, section[key])));
    details.append(summary, body);
    return details;
  }

  function contextGroupSummary(group, ctx) {
    const section = isPlainObject(ctx[group.key]) ? ctx[group.key] : {};
    if (group.key === 'system') return section.name ? String(section.name) : 'не заполнено';
    const count = (key) => (Array.isArray(section[key]) ? section[key].length : 0);
    const parts = (group.lists || []).map(([key, title]) => `${title.toLowerCase()}: ${count(key)}`);
    (group.lines || []).forEach(([key, title]) => { if (count(key)) parts.push(`${title.split(' — ')[0].toLowerCase()}: ${count(key)}`); });
    return parts.join(' · ') || 'пусто';
  }

  function systemFields(section) {
    const box = makeEl('div');
    const grid = makeEl('div', 'form-grid two');
    grid.append(
      textField('Название', section.name, (v) => setContextPath('system', 'name', v)),
      textField('Домен', section.domain, (v) => setContextPath('system', 'domain', v))
    );
    box.append(
      grid,
      textField('Описание', section.description, (v) => setContextPath('system', 'description', v), true),
      textField('Цель теста', section.test_goal, (v) => setContextPath('system', 'test_goal', v), true)
    );
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
    const head = makeEl('div', 'struct-head');
    head.append(makeEl('h4', '', title), makeBtn('Добавить', () => {
      setContextPath(groupKey, key, currentContextList(groupKey, key).concat([{}]));
      renderContextForm();
    }));
    box.appendChild(head);
    if (!rows.length) {
      box.appendChild(makeEl('p', 'dim', 'Пока пусто.'));
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
    remove.setAttribute('aria-label', 'Удалить строку');
    actions.appendChild(remove);
    tr.appendChild(actions);
    return tr;
  }

  function contextLines(groupKey, key, title, values) {
    const wrap = makeEl('div', 'ctx-field');
    const id = nextFieldId();
    const lab = makeEl('label', '', title);
    lab.htmlFor = id;
    const area = makeEl('textarea');
    area.id = id;
    area.rows = 3;
    area.placeholder = 'По одному на строку';
    area.value = (Array.isArray(values) ? values : []).join('\n');
    area.addEventListener('change', () => setContextPath(groupKey, key, splitList(area.value, '\n')));
    wrap.append(lab, area);
    return wrap;
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
  function editQueryBlock(domain, mutate) {
    const queries = clone(queriesModel());
    const block = Object.assign({}, isPlainObject(queries[domain]) ? queries[domain] : {});
    const spec = currentQuerySpec(domain);
    const count = alignQueryBlock(block, spec);
    mutate(block, spec, count);
    queries[domain] = block;
    commitStructured('queries', queries, 'ed_queries', 'queriesSaveBtn', 'queries', state.queriesScope ? `Запросы (${state.queriesScope})` : 'Запросы');
  }
  function queryRows(block, spec) {
    const labels = Array.isArray(block.labels) ? block.labels : [];
    const queries = Array.isArray(block[spec.query]) ? block[spec.query] : [];
    const keys = Array.isArray(block[spec.keys]) ? block[spec.keys] : [];
    const count = Math.max(labels.length, queries.length, keys.length);
    const rows = [];
    for (let i = 0; i < count; i += 1) {
      rows.push({ index: i, label: String(labels[i] || ''), query: String(queries[i] || ''), keys: Array.isArray(keys[i]) ? keys[i].map(String) : [] });
    }
    return rows;
  }

  function renderQueriesForm() {
    const host = $('queriesForm');
    if (!host) return;
    const domain = QUERY_DOMAINS.includes(state.queryDomain) ? state.queryDomain : 'jvm';
    state.queryDomain = domain;
    host.innerHTML = '';
    host.appendChild(queryDomainTabs(domain));
    host.appendChild(queryLangBar(domain, queryBlock(domain)));
    host.appendChild(previewWindowBar());
    const search = makeEl('input');
    search.type = 'search';
    search.placeholder = 'Поиск по названию или запросу';
    search.value = state.querySearch || '';
    search.addEventListener('input', () => { state.querySearch = search.value; renderQueryList(); });
    const toolbar = makeEl('div', 'struct-toolbar');
    toolbar.append(search, makeBtn('Добавить запрос', addQueryRow));
    host.appendChild(toolbar);
    const list = makeEl('div', 'q-list');
    list.id = 'queryList';
    host.appendChild(list);
    renderQueryList();
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

  function queryLangBar(domain, block) {
    const lang = activeQueryLang(domain, block);
    const bar = makeEl('div', 'struct-toolbar');
    const switcher = savedSourceType(domain) === 'grafana_proxy' || filledLangs(block).length > 1 || state.domainLangPicker[domain];
    if (!switcher) {
      bar.appendChild(makeEl('span', 'pill na', `Язык: ${QUERY_LANG[lang].label}`));
      return bar;
    }
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
    const spec = currentQuerySpec(domain);
    const needle = String(state.querySearch || '').trim().toLowerCase();
    const rows = queryRows(queryBlock(domain), spec)
      .filter((row) => !needle || row.label.toLowerCase().includes(needle) || row.query.toLowerCase().includes(needle));
    list.innerHTML = '';
    if (!rows.length) {
      list.appendChild(makeEl('p', 'dim', needle ? 'Ничего не найдено.' : 'Запросов пока нет.'));
      return;
    }
    rows.forEach((row) => list.appendChild(queryItem(domain, spec, row)));
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
    main.appendChild(makeEl('div', 'q-item-title', row.label || 'Без названия'));
    const meta = [spec.label];
    if (row.keys.length) meta.push(`серии: ${row.keys.join(', ')}`);
    if (meta.length) main.appendChild(makeEl('div', 'q-item-meta', meta.join(' · ')));
    if (!open) main.appendChild(makeEl('div', 'q-item-code', (row.query.split('\n').find((line) => line.trim()) || '').trim() || '—'));
    head.appendChild(main);
    const used = slaLabelsUsing(row.label);
    if (used.length) head.appendChild(makeEl('span', 'pill na', `SLA: ${slaUsageText(used)}`));
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
    start.setAttribute('aria-label', 'Начало окна проверки');
    start.addEventListener('change', () => { state.previewStart = start.value; });
    const end = makeEl('input');
    end.type = 'datetime-local';
    end.value = state.previewEnd;
    end.setAttribute('aria-label', 'Конец окна проверки');
    end.addEventListener('change', () => { state.previewEnd = end.value; });
    bar.append(makeEl('span', 'dim', 'Окно проверки'), start, makeEl('span', 'dim', '—'), end);
    return bar;
  }
  function queryLangName(domain) {
    return activeQueryLang(domain, queryBlock(domain));
  }
  async function runQueryPreview(domain, queryInput, keysInput, result) {
    const query = String(queryInput.value || '').trim();
    result.className = 'conn-result pending';
    result.textContent = 'Проверка…';
    if (!query) {
      result.className = 'conn-result error';
      result.textContent = 'Запрос пустой';
      return;
    }
    const body = {
      domain: domain,
      lang: queryLangName(domain),
      query: query,
      label_keys: splitList(keysInput.value)
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
  function queryEditor(domain, spec, row) {
    const body = makeEl('div', 'q-item-body');
    const name = textField('Название', row.label, () => {});
    const nameInput = name.querySelector('input');
    nameInput.addEventListener('change', () => {
      renameSlaLabel(row.label, nameInput.value);
      editQueryBlock(domain, (block) => { block.labels[row.index] = nameInput.value; });
      renderQueriesForm();
    });
    const query = textField(`Запрос (${spec.label})`, row.query, () => {}, true);
    const queryInput = query.querySelector('textarea');
    queryInput.rows = Math.min(14, Math.max(4, row.query.split('\n').length + 1));
    queryInput.addEventListener('change', () => {
      editQueryBlock(domain, (block, active) => { block[active.query][row.index] = queryInput.value; });
    });
    const keys = textField('Ключи подписи серии, через запятую', row.keys.join(', '), () => {});
    const keysInput = keys.querySelector('input');
    keysInput.addEventListener('change', () => {
      editQueryBlock(domain, (block, active) => { block[active.keys][row.index] = splitList(keysInput.value); });
    });
    body.append(name, query, keys);
    if (spec.keys === 'label_tag_keys_list') body.appendChild(makeEl('p', 'help', 'Ключи подписи общие для InfluxQL и Flux.'));
    const preview = makeEl('div', 'conn-result');
    const check = makeBtn('Проверить', async () => {
      check.disabled = true;
      try {
        await runQueryPreview(domain, queryInput, keysInput, preview);
      } catch (e) {
        preview.className = 'conn-result error';
        preview.textContent = e.message || 'Ошибка проверки';
      } finally {
        check.disabled = false;
      }
    });
    const actions = makeEl('div', 'q-item-actions');
    actions.append(
      check,
      makeBtn('Выше', () => moveQueryRow(domain, row.index, -1)),
      makeBtn('Ниже', () => moveQueryRow(domain, row.index, 1)),
      makeBtn('Удалить', () => removeQueryRow(domain, row.index)),
      makeBtn('Свернуть', () => { state.queryOpen = null; renderQueryList(); })
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
    if (!window.confirm(`Удалить запрос «${label || 'без названия'}»?`)) return;
    editQueryBlock(domain, (block) => {
      QUERY_PARALLEL_KEYS.forEach((key) => { if (Array.isArray(block[key]) && index < block[key].length) block[key].splice(index, 1); });
    });
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
    state.queryOpen = `${domain}:${target}`;
    renderQueryList();
  }

  // ---- JSON-only sections (system_context, queries, metrics_config) --------
  function bindJsonSection(cfg) {
    const ed = makeAce(cfg.editorId, 'ace/mode/json', 16);
    const load = () => { state.models[cfg.key] = clone(cfg.getData()); state.saved[cfg.key] = stable(state.models[cfg.key]); setJson(ed, state.models[cfg.key]); $(cfg.saveBtn).disabled = true; markDirty(`${cfg.menu}:${cfg.key}`, cfg.label(), false); $(cfg.jsonStatus).textContent = ''; if (typeof cfg.render === 'function') cfg.render(); };
    ed.session.on('change', () => {
      if (state.jsonSyncing) return;
      try {
        const parsed = JSON.parse(ed.getValue() || '{}');
        if (!isPlainObject(parsed)) throw new Error('ожидается объект');
        state.models[cfg.key] = parsed;
        if (typeof cfg.render === 'function') cfg.render();
        $(cfg.jsonStatus).textContent = 'OK';
        const dirty = stable(parsed) !== state.saved[cfg.key];
        $(cfg.saveBtn).disabled = !dirty;
        markDirty(`${cfg.menu}:${cfg.key}`, cfg.label(), dirty);
      } catch (e) {
        $(cfg.jsonStatus).textContent = `Ошибка JSON: ${e.message}`;
        $(cfg.saveBtn).disabled = true;
      }
    });
    $(cfg.saveBtn).addEventListener('click', async () => {
      setStatus(cfg.status, 'Сохранение…');
      try {
        const data = await cfg.save(state.models[cfg.key]);
        state.saved[cfg.key] = stable(state.models[cfg.key]);
        $(cfg.saveBtn).disabled = true;
        markDirty(`${cfg.menu}:${cfg.key}`, cfg.label(), false);
        setStatus(cfg.status, 'Сохранено', 'ok');
        ui.toast(`${cfg.label()}: сохранено`, { tone: 'ok' });
        showWarnings(data && data.warnings);
      } catch (e) {
        setStatus(cfg.status, `Ошибка: ${e.message}`, 'error');
      }
    });
    return { load };
  }

  // ---- prompts -------------------------------------------------------------
  const PROMPT_DOMAINS = ['overall', 'jvm', 'database', 'kafka', 'microservices', 'hard_resources', 'lt_framework', 'application_logs'];
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
    return isPlainObject(data.domains) ? data.domains : {};
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
    return missing.length ? `В тексте отсутствуют плейсхолдеры из стандартного промпта: ${missing.join(', ')} — соответствующие данные не попадут в запрос к модели.` : '';
  }
  async function loadPromptEditor() {
    const ed = state.editors.ed_prompt;
    try { await ensureServicePrompts(state.promptService); }
    catch (e) { ui.toast(`Не удалось загрузить промпты сервиса: ${e.message}`, { tone: 'error' }); }
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
      ? 'Только просмотр: выберите область, чтобы редактировать'
      : (overriddenForService ? 'Переопределён для сервиса' : (overriddenForArea ? 'Переопределён для области' : 'Стандартный текст'));
    $('promptWarnings').textContent = promptWarnings(text);
    $('promptDiff').style.display = 'none';
    updatePromptDirty();
  }
  function updatePromptDirty() {
    const ed = state.editors.ed_prompt;
    const dirty = ed.getValue() !== state.promptLoaded;
    $('promptSaveBtn').disabled = !dirty || !state.area;
    markDirty('prompts:editor', `Промпт ${state.promptDomain}`, dirty);
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
      ? `<div class="diff-line ctx">— ${state.promptService ? 'текст области' : 'стандартный текст'} / + текущий (${changed} строк отличается)</div>` + diff.map((d) => `<div class="diff-line ${d.type}">${d.type === 'add' ? '+ ' : (d.type === 'del' ? '− ' : '  ')}${esc(d.text)}</div>`).join('')
      : '<div class="diff-line ctx">Текст совпадает с базовым.</div>';
    box.style.display = '';
  }
  async function resetPrompt() {
    if (!requireArea('Сброс промпта')) return;
    const ok = await ui.confirm({ title: 'Сбросить промпт', message: state.promptService ? 'Переопределение для сервиса будет удалено, будет использоваться текст области.' : 'Переопределение области будет удалено, вернётся стандартный текст из файла.', confirmText: 'Сбросить', danger: true });
    if (!ok) return;
    try {
      await savePromptText(promptBaseline(state.promptService, state.promptDomain));
      await loadPromptEditor();
      ui.toast('Промпт сброшен', { tone: 'ok' });
    } catch (e) { ui.toast(`Ошибка: ${e.message}`, { tone: 'error' }); }
  }
  async function showPromptHistory() {
    if (!requireArea('История промптов')) return;
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
    } catch (e) { ui.toast(`Не удалось загрузить историю: ${e.message}`, { tone: 'error' }); return; }
    const body = versions.length
      ? `<div class="history-list">${versions.map((v, i) => `<div class="history-item"><span>${esc(LL.formatDateTime(v.saved_at))} · ${esc(String(v.text || '').length)} симв.</span><span><button type="button" class="btn btn-sm" data-preview="${i}">Показать</button> <button type="button" class="btn btn-sm btn-primary" data-restore="${i}">Восстановить</button></span></div>`).join('')}</div><pre id="historyPreview" class="diff-view" style="display:none;max-height:260px;padding:10px"></pre>`
      : '<p>История пуста: сохранённых ранее версий для этого домена нет.</p>';
    await ui.dialog({
      title: `История промпта: ${LL.domainTitle(state.promptDomain)}`,
      bodyHtml: body,
      wide: true,
      actions: [{ label: 'Закрыть' }],
      onOpen: (bodyEl, close) => {
        bodyEl.querySelectorAll('[data-preview]').forEach((btn) => btn.addEventListener('click', () => { const pre = bodyEl.querySelector('#historyPreview'); pre.textContent = versions[Number(btn.dataset.preview)].text; pre.style.display = ''; }));
        bodyEl.querySelectorAll('[data-restore]').forEach((btn) => btn.addEventListener('click', () => {
          const ed = state.editors.ed_prompt;
          ed.setValue(versions[Number(btn.dataset.restore)].text, -1);
          refreshPromptMarkers(ed);
          updatePromptDirty();
          close(true);
          ui.toast('Версия загружена в редактор — нажмите «Сохранить», чтобы применить', { tone: 'info' });
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
    domainSel.innerHTML = PROMPT_DOMAINS.map((d) => `<option value="${d}">${esc(d === 'overall' ? 'Итоговый (overall)' : `${LL.domainTitle(d)} (${d})`)}</option>`).join('');
    domainSel.addEventListener('change', async () => { if (!(await confirmDiscardPrompt())) { domainSel.value = state.promptDomain; return; } state.promptDomain = domainSel.value; await loadPromptEditor(); });
    $('promptServiceSelect').addEventListener('change', async (e) => { if (!(await confirmDiscardPrompt())) { e.target.value = state.promptService; return; } state.promptService = e.target.value; await loadPromptEditor(); });
    $('promptSaveBtn').addEventListener('click', async () => {
      if (!requireArea('Сохранение промпта')) return;
      setStatus('promptStatus', 'Сохранение…');
      try { await savePromptText(ed.getValue()); await loadPromptEditor(); setStatus('promptStatus', 'Сохранено', 'ok'); ui.toast('Промпт сохранён', { tone: 'ok' }); }
      catch (e) { setStatus('promptStatus', `Ошибка: ${e.message}`, 'error'); }
    });
    $('promptDiffBtn').addEventListener('click', togglePromptDiff);
    $('promptResetBtn').addEventListener('click', resetPrompt);
    $('promptHistoryBtn').addEventListener('click', showPromptHistory);
  }
  async function confirmDiscardPrompt() {
    if (!state.dirty.has('prompts:editor')) return true;
    return ui.confirm({ title: 'Несохранённые изменения', message: 'Текущий промпт изменён, но не сохранён. Переключиться и потерять изменения?', confirmText: 'Переключиться', danger: true });
  }

  // ---- services --------------------------------------------------------------
  function allServiceIds() {
    const ids = new Set([...state.metricsConfigIds, ...Object.keys(state.servicesMeta)]);
    return Array.from(ids).sort();
  }
  function fillServiceSelects() {
    const options = allServiceIds().map((id) => `<option value="${esc(id)}">${esc((state.servicesMeta[id] && state.servicesMeta[id].title) || id)}</option>`).join('');
    [['slaServiceSelect', state.slaScope, 'Область (по умолчанию)'], ['queriesServiceSelect', state.queriesScope, 'Область (по умолчанию)'], ['promptServiceSelect', state.promptService, 'Область (по умолчанию)'], ['metricsServiceSelect', state.metricsScope, 'Область целиком']].forEach(([id, current, placeholder]) => {
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
      body.innerHTML = '<tr class="empty-row"><td colspan="4">Сервисы принадлежат областям: выберите область в шапке страницы или создайте новую.</td></tr>';
      return;
    }
    if (!ids.length) {
      body.innerHTML = '<tr class="empty-row"><td colspan="4">В этой области сервисов пока нет. Добавьте первый — идентификатор используется в имени отчёта и как ключ metrics_config.</td></tr>';
      return;
    }
    ids.forEach((id) => {
      const meta = state.servicesMeta[id] || {};
      const disabled = Array.isArray(meta.disabled_domains) ? meta.disabled_domains : [];
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${esc(id)}</strong>${state.metricsConfigIds.includes(id) ? '' : '<div class="dim">нет в metrics_config: снимки панелей недоступны</div>'}</td>
        <td><input type="text" value="${esc(meta.title || '')}" placeholder="${esc(id)}" aria-label="Отображаемое имя ${esc(id)}" /></td>
        <td><div class="domain-toggle-list">${state.domainList.map((d) => `<label><input type="checkbox" data-domain="${esc(d)}" ${disabled.includes(d) ? '' : 'checked'}/> ${esc(LL.domainTitle(d))}</label>`).join('')}</div></td>
        <td class="actions"><div class="actions-inner" style="display:flex;gap:6px;justify-content:flex-end;flex-wrap:wrap"><button type="button" class="btn btn-sm svc-save">Сохранить</button><button type="button" class="btn btn-sm svc-reset">Сбросить настройки</button><button type="button" class="btn btn-sm btn-danger svc-delete">Удалить с данными</button></div></td>`;
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
    if (!requireArea('Сохранение сервиса')) return;
    setStatus('servicesStatus', 'Сохранение…');
    try {
      const resp = await fetch('/service_meta', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ area: state.area, service: id, data }) });
      const j = await resp.json();
      if (!resp.ok) throw new Error(j.error || `HTTP ${resp.status}`);
      state.servicesMeta[id] = { ...(state.servicesMeta[id] || {}), ...(j.meta || {}) };
      renderServices();
      fillServiceSelects();
      setStatus('servicesStatus', 'Сохранено', 'ok');
      ui.toast(`Сервис «${id}» сохранён`, { tone: 'ok' });
    } catch (e) {
      setStatus('servicesStatus', `Ошибка: ${e.message}`, 'error');
    }
  }
  async function resetServiceSettings(id) {
    const ok = await ui.confirm({ title: 'Сбросить настройки сервиса', message: `Пользовательские настройки сервиса «${id}» (имя, домены, промпты, запросы, SLA, визуализации) будут удалены. Отчёты и метрики останутся.`, confirmText: 'Сбросить', danger: true });
    if (!ok) return;
    try {
      const resp = await fetch('/service_meta', { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ area: state.area, service: id }) });
      const j = await resp.json();
      if (!resp.ok) throw new Error(j.error || `HTTP ${resp.status}`);
      ui.toast(`Настройки сервиса «${id}» сброшены`, { tone: 'ok' });
      location.reload();
    } catch (e) { ui.toast(`Ошибка: ${e.message}`, { tone: 'error' }); }
  }
  async function deleteServiceWithData(id) {
    const ok = await ui.confirm({ title: 'Удалить сервис', message: `Сервис «${id}» будет удалён вместе с настройками, метриками и отчётами. Это действие необратимо.`, confirmText: 'Удалить', danger: true });
    if (!ok) return;
    try {
      const resp = await fetch('/service', { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ area: state.area, service: id }) });
      const j = await resp.json();
      if (!resp.ok) throw new Error(j.error || `HTTP ${resp.status}`);
      ui.toast(`Сервис «${id}» удалён`, { tone: 'ok' });
      location.reload();
    } catch (e) { ui.toast(`Ошибка: ${e.message}`, { tone: 'error' }); }
  }
  async function addService() {
    if (!requireArea('Добавление сервиса')) return;
    const known = state.metricsConfigIds.filter((id) => !state.servicesMeta[id]);
    const id = await ui.prompt({
      title: 'Добавить сервис',
      message: known.length ? `Сервисы из metrics_config без настроек: ${known.join(', ')}. Можно ввести и другой идентификатор — конфигурация визуализаций для него будет создана из шаблона.` : `Идентификатор сервиса в области «${state.area}»: он используется в имени отчёта и как ключ metrics_config.`,
      label: 'Идентификатор',
      list: known,
      validate: (v) => (!v.trim() ? 'Введите идентификатор' : (/[\s/\\]/.test(v) ? 'Без пробелов и слэшей' : ''))
    });
    if (!id) return;
    const title = await ui.prompt({ title: 'Отображаемое имя', message: 'Как показывать сервис в интерфейсе (необязательно).', label: 'Имя', value: '', confirmText: 'Добавить' });
    if (title === null) return;
    await saveServiceMeta(id.trim(), { title: title.trim() });
    // The server bootstraps per-service configs on the next GET /config: reload to pick them up.
    location.reload();
  }

  // ---- menu / sections ----------------------------------------------------------
  function activateSection(key) {
    document.querySelectorAll('#settingsMenu button').forEach((b) => b.classList.toggle('active', b.dataset.section === key));
    document.querySelectorAll('.settings-section').forEach((s) => s.classList.toggle('active', s.id === `sec-${key}`));
    if (!state.wizard) { try { history.replaceState(null, '', `#${key}`); } catch (e) { /* ignore */ } }
  }

  // ---- wizard ---------------------------------------------------------------------
  const WIZARD_STEPS = [
    { key: 'storage', label: 'Хранилище', section: 'connections', card: 'storage' },
    { key: 'data_sources', label: 'Источники данных', section: 'connections', card: 'data_sources' },
    { key: 'domain_sources', label: 'Привязка доменов', section: 'connections', card: 'domain_sources' },
    { key: 'llm', label: 'LLM', section: 'connections', card: 'llm' },
    { key: 'sla', label: 'SLA', section: 'sla' },
    { key: 'done', label: 'Готово', section: 'wizard-done' }
  ];
  function renderWizard() {
    const step = WIZARD_STEPS[state.wizardStep];
    const stepper = $('wizardStepper');
    stepper.innerHTML = WIZARD_STEPS.map((s, i) => `<span class="step-item ${i === state.wizardStep ? 'active' : ''} ${i < state.wizardStep ? 'done' : ''}">${i < state.wizardStep ? '✓' : i + 1}. ${esc(s.label)}</span>`).join('');
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
      hint.textContent = ok ? 'Соединение проверено — можно продолжать.' : 'Проверьте соединение, чтобы перейти дальше, или пропустите шаг.';
    } else {
      next.disabled = false;
      hint.textContent = 'Задайте пороги (можно позже) и нажмите «Далее».';
    }
  }
  async function wizardNext(skip) {
    const step = WIZARD_STEPS[state.wizardStep];
    if (!skip) {
      if (step.card) {
        const card = CONNECTION_CARDS.find((c) => c.id === step.card);
        if (state.dirty.has(`connections:${card.id}`) && !(await saveCard(card))) return;
      } else if (step.key === 'sla' && state.dirty.has('sla:model') && !(await saveSla())) return;
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
      setStatus('wizardDemoStatus', 'Создаём демонстрационный прогон…');
      try {
        const resp = await fetch('/demo/seed', { method: 'POST' });
        const data = await resp.json();
        if (resp.status === 409 && data.report_url) { setStatus('wizardDemoStatus', 'Демо-прогон уже существует.'); location.href = data.report_url; return; }
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
        setStatus('wizardDemoStatus', 'Готово, открываем отчёт…', 'ok');
        location.href = data.report_url;
      } catch (e) {
        setStatus('wizardDemoStatus', `Не удалось создать демо: ${e.message}`, 'error');
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
    if (!cfgResp.ok) throw new Error(cfg.error || 'Не удалось загрузить конфигурацию');
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
    state.prompts.area = await fetchPrompts('');
    state.prompts.byService = {};
  }

  const ACCENT_PRESETS = [
    { label: 'Фиолетовый', color: '#6200ee' },
    { label: 'Синий', color: '#1d4ed8' },
    { label: 'Голубой', color: '#0284c7' },
    { label: 'Бирюзовый', color: '#0f766e' },
    { label: 'Зелёный', color: '#15803d' },
    { label: 'Оранжевый', color: '#c2410c' },
    { label: 'Красный', color: '#b91c1c' },
    { label: 'Розовый', color: '#be185d' },
    { label: 'Графитовый', color: '#64748b' }
  ];
  const DEFAULT_ACCENT = '#6200ee';

  function initAppearance() {
    const stored = String((state.config.appearance || {}).accent || DEFAULT_ACCENT);
    let saved = stored.toLowerCase();
    let current = saved;
    let previewTimer = 0;
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
    custom.title = 'Свой цвет';
    custom.setAttribute('aria-label', 'Свой цвет');
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
      error.textContent = `Сохранённый цвет «${stored}» не в формате #rrggbb — на страницах пока фиолетовый`;
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
      const dirty = current !== saved;
      $('appearanceSave').disabled = !dirty;
      markDirty('appearance:accent', 'Оформление', dirty);
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
      setStatus('appearanceStatus', 'Сохранение…');
      try {
        await postConfig(configBody('appearance', { accent: current }));
        saved = current;
        state.config.appearance = { accent: current };
        $('appearanceSave').disabled = true;
        markDirty('appearance:accent', 'Оформление', false);
        $('appearanceError').hidden = true;
        setStatus('appearanceStatus', 'Сохранено', 'ok');
        ui.toast('Оформление сохранено', { tone: 'ok' });
      } catch (error) {
        setStatus('appearanceStatus', error.message, 'error');
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

  document.addEventListener('DOMContentLoaded', async () => {
    try {
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
      try { const parsed = JSON.parse(state.editors.ed_sla.getValue() || '{}'); if (!isPlainObject(parsed)) throw new Error('ожидается объект'); state.models.sla = parsed; $('slaJsonStatus').textContent = 'OK'; renderSlaForm({ skipJson: true }); }
      catch (e) { $('slaJsonStatus').textContent = `Ошибка JSON: ${e.message}`; }
    });
    $('slaSaveBtn').addEventListener('click', saveSla);
    $('slaServiceSelect').addEventListener('change', (e) => loadSlaScope(e.target.value));

    const contextSection = bindJsonSection({
      key: 'system_context', menu: 'context', editorId: 'ed_system_context', saveBtn: 'contextSaveBtn', status: 'contextStatus', jsonStatus: 'contextJsonStatus',
      label: () => 'Контекст системы',
      getData: () => state.config.system_context || {},
      save: async (data) => { const r = await postConfig(configBody('system_context', data)); state.config.system_context = clone(data); return r; },
      render: renderContextForm
    });
    const queriesSection = bindJsonSection({
      key: 'queries', menu: 'queries', editorId: 'ed_queries', saveBtn: 'queriesSaveBtn', status: 'queriesStatus', jsonStatus: 'queriesJsonStatus',
      label: () => (state.queriesScope ? `Запросы (${state.queriesScope})` : 'Запросы'),
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

    const metricsSection = bindJsonSection({
      key: 'metrics_config', menu: 'metrics', editorId: 'ed_metrics_config', saveBtn: 'metricsSaveBtn', status: 'metricsStatus', jsonStatus: 'metricsJsonStatus',
      label: () => (state.metricsScope ? `Визуализации (${state.metricsScope})` : 'Визуализации'),
      getData: () => {
        const map = isPlainObject(state.config.metrics_config_map) ? state.config.metrics_config_map : {};
        return map[state.metricsScope || ''] || (state.metricsScope ? {} : (state.config.metrics_config || {}));
      },
      save: async (data) => {
        if (!state.area) throw new Error('Выберите область в шапке страницы');
        const r = await postConfig(configBody('metrics_config', data, state.metricsScope));
        state.config.metrics_config_map = state.config.metrics_config_map || {};
        state.config.metrics_config_map[state.metricsScope || ''] = clone(data);
        return r;
      }
    });
    $('metricsServiceSelect').addEventListener('change', (e) => { state.metricsScope = e.target.value; metricsSection.load(); });

    initAppearance();
    initPrompts();
    fillServiceSelects();
    renderServices();
    loadSlaScope('');
    contextSection.load();
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
      if (state.dirty.has('appearance:accent')) $('appearanceSave').click();
    });

    initWizard();
    const activateFromHash = () => {
      if (state.wizard) return;
      const hash = (location.hash || '').replace('#', '');
      if (hash && document.getElementById(`sec-${hash}`)) activateSection(hash);
    };
    activateFromHash();
    window.addEventListener('hashchange', activateFromHash);
  });
})();
