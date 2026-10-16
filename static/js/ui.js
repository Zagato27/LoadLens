// Shared UI components: preferences, toasts, dialogs, combobox, tabs, theme, glossary tooltips.
(function () {
  const LL = window.LoadLens = window.LoadLens || {};
  const ui = LL.ui = LL.ui || {};

  function escapeHtml(value) {
    return String(value === undefined || value === null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  ui.escapeHtml = escapeHtml;

  // ---- preferences -------------------------------------------------------
  // localStorage under "loadlens.<key>", JSON-encoded. "theme" stays a plain string because the
  // pre-paint script in base.html reads it raw. Known keys: theme, recentServices.
  const PREFS_PREFIX = 'loadlens.';
  const RAW_PREFS = new Set(['theme']);
  const prefListeners = new Map();
  function decodePref(key, raw, fallback) {
    if (raw === null || raw === undefined) return fallback;
    if (RAW_PREFS.has(key)) return raw;
    try { return JSON.parse(raw); } catch (e) { return fallback; }
  }
  function notifyPref(key, value) {
    (prefListeners.get(key) || []).slice().forEach((cb) => {
      try { cb(value, key); } catch (e) { /* listener error must not break others */ }
    });
  }
  LL.prefs = {
    get(key, fallback) {
      try { return decodePref(key, localStorage.getItem(PREFS_PREFIX + key), fallback); } catch (e) { return fallback; }
    },
    // null or undefined removes the key.
    set(key, value) {
      try {
        if (value === undefined || value === null) localStorage.removeItem(PREFS_PREFIX + key);
        else localStorage.setItem(PREFS_PREFIX + key, RAW_PREFS.has(key) ? String(value) : JSON.stringify(value));
      } catch (e) { /* storage unavailable or full */ }
      notifyPref(key, value === undefined ? null : value);
    },
    // cb(value, key) runs on set() in this tab and on changes made in other tabs; returns an unsubscribe function.
    onChange(key, cb) {
      if (typeof cb !== 'function') return () => {};
      if (!prefListeners.has(key)) prefListeners.set(key, []);
      prefListeners.get(key).push(cb);
      return () => {
        const list = prefListeners.get(key) || [];
        const idx = list.indexOf(cb);
        if (idx >= 0) list.splice(idx, 1);
      };
    }
  };
  try {
    window.addEventListener('storage', (event) => {
      if (!event || typeof event.key !== 'string' || !event.key.startsWith(PREFS_PREFIX)) return;
      const key = event.key.slice(PREFS_PREFIX.length);
      notifyPref(key, decodePref(key, event.newValue, null));
    });
  } catch (e) { /* no window events */ }
  // The interface density setting was removed; drop the value it may have left behind.
  try { localStorage.removeItem(`${PREFS_PREFIX}density`); } catch (e) { /* storage unavailable */ }

  // ---- theme -------------------------------------------------------------
  // The stored preference (mode) is 'system', 'light' or 'dark'; <html data-theme> always carries the
  // resolved theme. The pre-paint scripts (base.html, login.html, recovery.html, storage_unavailable.html)
  // resolve 'system' the same way before the first paint; here it keeps following the OS setting.
  const THEME_MODES = ['system', 'light', 'dark'];
  const THEME_LABELS = { system: 'Как в системе', light: 'Светлая', dark: 'Тёмная' };
  const themeListeners = [];
  let lightQuery = null;
  try {
    lightQuery = typeof window.matchMedia === 'function' ? window.matchMedia('(prefers-color-scheme: light)') : null;
  } catch (e) { lightQuery = null; }
  function documentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  }
  function resolveTheme(mode) {
    if (mode === 'system') return lightQuery && lightQuery.matches ? 'light' : 'dark';
    return mode === 'light' ? 'light' : 'dark';
  }
  const storedTheme = LL.prefs.get('theme', null);
  // Without a stored choice the page keeps the theme it was rendered with (dark by default).
  let themeMode = THEME_MODES.includes(storedTheme) ? storedTheme : documentTheme();
  // Listeners get the resolved theme; `always` notifies even when it did not change (explicit set()).
  function applyTheme(always) {
    const before = documentTheme();
    const next = resolveTheme(themeMode);
    if (document.documentElement.getAttribute('data-theme') !== next) document.documentElement.setAttribute('data-theme', next);
    syncViewMenu();
    if (!always && before === next) return;
    themeListeners.slice().forEach((cb) => { try { cb(next); } catch (e) { /* listener error must not break others */ } });
  }
  ui.theme = {
    // Resolved theme of the document: 'light' or 'dark' (charts and editors depend on it).
    current: documentTheme,
    // Stored preference: 'system', 'light' or 'dark'.
    mode() { return themeMode; },
    set(name) {
      themeMode = THEME_MODES.includes(name) ? name : 'dark';
      LL.prefs.set('theme', themeMode);
      applyTheme(true);
    },
    toggle() { ui.theme.set(documentTheme() === 'light' ? 'dark' : 'light'); },
    // cb(resolvedTheme) runs on set() and whenever the resolved theme changes (OS switch while the mode
    // is 'system', a choice made in another tab). Returns an unsubscribe function.
    onChange(cb) {
      if (typeof cb !== 'function') return () => {};
      themeListeners.push(cb);
      return () => {
        const idx = themeListeners.indexOf(cb);
        if (idx >= 0) themeListeners.splice(idx, 1);
      };
    }
  };
  // A choice made in another tab (storage event) applies here too; set() in this tab is already applied.
  LL.prefs.onChange('theme', (value) => {
    const mode = THEME_MODES.includes(value) ? value : 'dark';
    if (mode === themeMode) return;
    themeMode = mode;
    applyTheme(false);
  });
  if (lightQuery) {
    const onSystemTheme = () => { if (themeMode === 'system') applyTheme(false); };
    try {
      if (typeof lightQuery.addEventListener === 'function') lightQuery.addEventListener('change', onSystemTheme);
      else if (typeof lightQuery.addListener === 'function') lightQuery.addListener(onSystemTheme);
    } catch (e) { /* no media query events */ }
  }
  if (themeMode === 'system') applyTheme(false);

  // ---- theme menu (sidebar foot) --------------------------------------------
  function syncViewMenu() {
    const menu = document.getElementById('viewMenu');
    if (!menu) return;
    menu.querySelectorAll('[data-view-pref="theme"]').forEach((item) => {
      item.setAttribute('aria-checked', item.dataset.value === themeMode ? 'true' : 'false');
    });
    const value = document.getElementById('viewMenuValue');
    if (value) value.textContent = THEME_LABELS[themeMode] || '';
  }
  function wireViewMenu() {
    const menu = document.getElementById('viewMenu');
    if (!menu) return;
    const summary = menu.querySelector('summary');
    const items = () => Array.from(menu.querySelectorAll('[data-view-pref]'));
    syncViewMenu();
    // Opening moves focus to the checked theme item, as in the menu button pattern.
    menu.addEventListener('toggle', () => {
      if (!menu.open) return;
      const checked = items().find((item) => item.getAttribute('aria-checked') === 'true') || items()[0];
      if (checked) checked.focus();
    });
    // A choice applies the theme and closes the menu, returning focus to its button.
    menu.addEventListener('click', (event) => {
      const item = event.target.closest('[data-view-pref="theme"]');
      if (!item || !menu.contains(item)) return;
      ui.theme.set(item.dataset.value);
      menu.open = false;
      if (summary) summary.focus();
    });
    menu.addEventListener('keydown', (event) => {
      if (!menu.open) return;
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        menu.open = false;
        if (summary) summary.focus();
        return;
      }
      const list = items();
      const idx = list.indexOf(document.activeElement);
      let next = null;
      if (event.key === 'ArrowDown') next = list[(idx + 1) % list.length];
      else if (event.key === 'ArrowUp') next = list[idx <= 0 ? list.length - 1 : idx - 1];
      else if (event.key === 'Home') next = list[0];
      else if (event.key === 'End') next = list[list.length - 1];
      if (next) { event.preventDefault(); next.focus(); }
    });
    // Tabbing out of the open menu closes it.
    menu.addEventListener('focusout', (event) => {
      if (menu.open && event.relatedTarget && !menu.contains(event.relatedTarget)) menu.open = false;
    });
  }

  // Chart.js colors follow the active theme.
  ui.chartColors = function chartColors() {
    const css = getComputedStyle(document.documentElement);
    return {
      grid: css.getPropertyValue('--chart-grid').trim() || 'rgba(255, 255, 255, 0.06)',
      text: css.getPropertyValue('--chart-text').trim() || '#8b93a5',
      background: css.getPropertyValue('--bg-chart').trim() || '#12151c',
      accent: css.getPropertyValue('--accent').trim() || '#6c5ce7',
      accentSoft: css.getPropertyValue('--accent-soft').trim() || 'rgba(108, 92, 231, 0.16)',
      statusOk: css.getPropertyValue('--status-ok').trim() || '#22c55e',
      statusWarn: css.getPropertyValue('--status-warn').trim() || '#f59e0b',
      statusFail: css.getPropertyValue('--status-fail').trim() || '#ef4444',
      statusNa: css.getPropertyValue('--status-na').trim() || '#94a3b8',
      surface: css.getPropertyValue('--bg-elev').trim() || '#151821',
      border: css.getPropertyValue('--border').trim() || 'rgba(255, 255, 255, 0.09)',
      ink: css.getPropertyValue('--text').trim() || '#e8eaf0',
      inkMuted: css.getPropertyValue('--text-muted').trim() || '#a3aab8'
    };
  };
  function applyChartDefaults() {
    if (!window.Chart || !window.Chart.defaults) return;
    const colors = ui.chartColors();
    const defaults = window.Chart.defaults;
    defaults.color = colors.text;
    defaults.borderColor = colors.grid;
    if (defaults.font) defaults.font.family = "'Inter', 'Segoe UI', system-ui, Arial, sans-serif";
    const tooltip = defaults.plugins && defaults.plugins.tooltip;
    if (tooltip) {
      tooltip.backgroundColor = colors.surface;
      tooltip.borderColor = colors.border;
      tooltip.borderWidth = 1;
      tooltip.titleColor = colors.ink;
      tooltip.bodyColor = colors.ink;
      tooltip.footerColor = colors.inkMuted;
      tooltip.padding = 10;
      tooltip.cornerRadius = 8;
      tooltip.boxPadding = 4;
      tooltip.usePointStyle = true;
    }
  }
  // Chart.js is loaded synchronously in <head> on chart pages, so defaults can be set right away.
  applyChartDefaults();
  ui.restyleCharts = function restyleCharts() {
    if (!window.Chart) return;
    applyChartDefaults();
    const colors = ui.chartColors();
    Object.values(window.Chart.instances || {}).forEach((chart) => {
      try {
        if (LL.charts && typeof LL.charts.recolor === 'function') LL.charts.recolor(chart);
        const scales = (chart.options && chart.options.scales) || {};
        Object.values(scales).forEach((scale) => {
          if (scale.ticks) scale.ticks.color = colors.text;
          if (scale.grid) scale.grid.color = colors.grid;
          if (scale.title) scale.title.color = colors.text;
        });
        const plugins = (chart.options && chart.options.plugins) || {};
        if (plugins.cmpBg) plugins.cmpBg.color = colors.background;
        if (plugins.customBackground) plugins.customBackground.color = colors.background;
        if (plugins.legend && plugins.legend.labels) plugins.legend.labels.color = colors.text;
        chart.update('none');
      } catch (e) { /* chart may be destroyed */ }
    });
  };

  // ---- toasts ------------------------------------------------------------
  function toastHost() {
    let host = document.querySelector('.toast-host');
    if (!host) {
      host = document.createElement('div');
      host.className = 'toast-host';
      host.setAttribute('role', 'status');
      host.setAttribute('aria-live', 'polite');
      document.body.appendChild(host);
    }
    return host;
  }
  ui.toast = function toast(message, options) {
    const opts = options || {};
    const el = document.createElement('div');
    el.className = `toast ${opts.tone || 'info'}`;
    const text = document.createElement('div');
    text.textContent = String(message || '');
    const close = document.createElement('button');
    close.className = 'toast-close';
    close.type = 'button';
    close.setAttribute('aria-label', 'Закрыть уведомление');
    close.textContent = '×';
    close.addEventListener('click', () => el.remove());
    el.appendChild(text);
    el.appendChild(close);
    toastHost().appendChild(el);
    const timeout = typeof opts.timeout === 'number' ? opts.timeout : (opts.tone === 'error' ? 8000 : 4000);
    if (timeout > 0) setTimeout(() => el.remove(), timeout);
    return el;
  };

  // ---- dialogs -----------------------------------------------------------
  function buildDialog() {
    const dialog = document.createElement('dialog');
    dialog.className = 'll-dialog';
    dialog.innerHTML = [
      '<form method="dialog">',
      '<div class="dialog-title"></div>',
      '<div class="dialog-body"></div>',
      '<div class="dialog-actions"></div>',
      '</form>'
    ].join('');
    document.body.appendChild(dialog);
    return dialog;
  }
  function openDialog(config) {
    return new Promise((resolve) => {
      const dialog = buildDialog();
      const title = dialog.querySelector('.dialog-title');
      const body = dialog.querySelector('.dialog-body');
      const actions = dialog.querySelector('.dialog-actions');
      title.textContent = config.title || '';
      body.innerHTML = '';
      if (config.message) {
        const p = document.createElement('div');
        p.textContent = config.message;
        body.appendChild(p);
      }
      if (config.bodyHtml) {
        const wrap = document.createElement('div');
        wrap.innerHTML = config.bodyHtml;
        body.appendChild(wrap);
      }
      let input = null;
      let error = null;
      if (config.input) {
        const label = document.createElement('label');
        label.textContent = config.input.label || '';
        input = document.createElement(config.input.multiline ? 'textarea' : 'input');
        if (!config.input.multiline) input.type = 'text';
        input.value = config.input.value || '';
        if (config.input.placeholder) input.placeholder = config.input.placeholder;
        if (config.input.list) {
          const listId = `ll-dl-${Date.now()}`;
          const dl = document.createElement('datalist');
          dl.id = listId;
          config.input.list.forEach((v) => { const o = document.createElement('option'); o.value = v; dl.appendChild(o); });
          input.setAttribute('list', listId);
          body.appendChild(dl);
        }
        label.appendChild(input);
        body.appendChild(label);
        error = document.createElement('div');
        error.className = 'dialog-error';
        body.appendChild(error);
      }
      const cancel = document.createElement('button');
      cancel.type = 'button';
      cancel.className = 'btn';
      cancel.textContent = config.cancelText || 'Отмена';
      const ok = document.createElement('button');
      ok.type = 'button';
      ok.className = `btn ${config.danger ? 'btn-danger' : 'btn-primary'}`;
      ok.textContent = config.confirmText || 'ОК';
      actions.appendChild(cancel);
      actions.appendChild(ok);
      let settled = false;
      const finish = (value) => {
        if (settled) return;
        settled = true;
        try { dialog.close(); } catch (e) { /* already closed */ }
        dialog.remove();
        resolve(value);
      };
      cancel.addEventListener('click', () => finish(config.input ? null : false));
      ok.addEventListener('click', () => {
        if (input) {
          const value = String(input.value || '');
          const problem = typeof config.input.validate === 'function' ? config.input.validate(value) : '';
          if (problem) { error.textContent = problem; input.focus(); return; }
          finish(value);
          return;
        }
        finish(true);
      });
      dialog.addEventListener('cancel', (e) => { e.preventDefault(); finish(config.input ? null : false); });
      dialog.querySelector('form').addEventListener('submit', (e) => { e.preventDefault(); ok.click(); });
      if (typeof dialog.showModal === 'function') dialog.showModal(); else dialog.setAttribute('open', '');
      (input || ok).focus();
      if (input && input.select) input.select();
    });
  }
  ui.confirm = function confirm(config) {
    const cfg = typeof config === 'string' ? { message: config } : (config || {});
    return openDialog({ title: cfg.title || 'Подтверждение', message: cfg.message, confirmText: cfg.confirmText || 'Подтвердить', cancelText: cfg.cancelText, danger: !!cfg.danger });
  };
  ui.prompt = function prompt(config) {
    const cfg = config || {};
    return openDialog({
      title: cfg.title || '',
      message: cfg.message,
      bodyHtml: cfg.bodyHtml,
      confirmText: cfg.confirmText || 'Сохранить',
      cancelText: cfg.cancelText,
      input: { label: cfg.label || '', value: cfg.value || '', placeholder: cfg.placeholder || '', validate: cfg.validate, list: cfg.list, multiline: !!cfg.multiline }
    });
  };
  ui.alert = function alert(config) {
    const cfg = typeof config === 'string' ? { message: config } : (config || {});
    return openDialog({ title: cfg.title || 'Сообщение', message: cfg.message, bodyHtml: cfg.bodyHtml, confirmText: 'Закрыть', cancelText: '' }).then(() => true);
  };
  // Generic dialog with custom body and actions: actions = [{label, primary, danger, onClick(close, bodyEl)}].
  ui.dialog = function dialog(config) {
    const cfg = config || {};
    return new Promise((resolve) => {
      const el = buildDialog();
      el.classList.toggle('wide', !!cfg.wide);
      el.querySelector('.dialog-title').textContent = cfg.title || '';
      const body = el.querySelector('.dialog-body');
      body.innerHTML = cfg.bodyHtml || '';
      const actions = el.querySelector('.dialog-actions');
      let settled = false;
      const close = (value) => {
        if (settled) return;
        settled = true;
        try { el.close(); } catch (e) { /* already closed */ }
        el.remove();
        resolve(value);
      };
      (cfg.actions || [{ label: 'Закрыть' }]).forEach((action) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `btn ${action.primary ? 'btn-primary' : ''} ${action.danger ? 'btn-danger' : ''}`.trim();
        btn.textContent = action.label;
        btn.addEventListener('click', () => {
          if (typeof action.onClick === 'function') action.onClick(close, body);
          else close(action.value !== undefined ? action.value : null);
        });
        actions.appendChild(btn);
      });
      el.addEventListener('cancel', (e) => { e.preventDefault(); close(null); });
      el.querySelector('form').addEventListener('submit', (e) => e.preventDefault());
      if (typeof el.showModal === 'function') el.showModal(); else el.setAttribute('open', '');
      if (typeof cfg.onOpen === 'function') cfg.onOpen(body, close);
    });
  };

  // ---- combobox ----------------------------------------------------------
  // host: element that receives the widget; options.fetchOptions(query) -> Promise<[{value, label, meta}]>
  ui.combobox = function combobox(host, options) {
    const opts = options || {};
    const listId = `cb-list-${Math.random().toString(36).slice(2, 8)}`;
    host.classList.add('combobox');
    host.innerHTML = `
      <input type="text" class="combobox-input" role="combobox" aria-autocomplete="list" aria-expanded="false" aria-controls="${listId}" placeholder="${escapeHtml(opts.placeholder || '')}" autocomplete="off" />
      <button type="button" class="combobox-clear" aria-label="Очистить" style="display:none">×</button>
      <div class="combobox-list" id="${listId}" role="listbox"></div>`;
    const input = host.querySelector('.combobox-input');
    const clearBtn = host.querySelector('.combobox-clear');
    const list = host.querySelector('.combobox-list');
    let current = null;
    let items = [];
    let activeIdx = -1;
    let debounceTimer = null;
    let requestSeq = 0;

    function setOpen(open) {
      host.classList.toggle('open', open);
      input.setAttribute('aria-expanded', open ? 'true' : 'false');
      if (!open) { activeIdx = -1; input.removeAttribute('aria-activedescendant'); }
    }
    function renderList() {
      list.innerHTML = '';
      if (!items.length) {
        const empty = document.createElement('div');
        empty.className = 'combobox-empty';
        empty.textContent = opts.emptyText || 'Ничего не найдено';
        list.appendChild(empty);
        return;
      }
      items.forEach((item, idx) => {
        const el = document.createElement('div');
        el.className = 'combobox-option';
        el.id = `${listId}-opt-${idx}`;
        el.setAttribute('role', 'option');
        el.setAttribute('aria-selected', idx === activeIdx ? 'true' : 'false');
        el.innerHTML = `<span class="opt-label">${escapeHtml(item.label || item.value)}</span>${item.meta ? `<span class="opt-meta">${escapeHtml(item.meta)}</span>` : ''}`;
        el.addEventListener('mousedown', (e) => { e.preventDefault(); choose(item); });
        list.appendChild(el);
      });
    }
    function highlight(idx) {
      activeIdx = idx;
      list.querySelectorAll('.combobox-option').forEach((el, i) => el.setAttribute('aria-selected', i === idx ? 'true' : 'false'));
      const active = list.querySelectorAll('.combobox-option')[idx];
      if (active) { input.setAttribute('aria-activedescendant', active.id); active.scrollIntoView({ block: 'nearest' }); }
    }
    async function search(query) {
      const seq = ++requestSeq;
      try {
        const result = await opts.fetchOptions(query);
        if (seq !== requestSeq) return;
        items = Array.isArray(result) ? result : [];
        activeIdx = items.length ? 0 : -1;
        renderList();
        setOpen(true);
      } catch (e) {
        if (seq !== requestSeq) return;
        items = [];
        renderList();
        list.querySelector('.combobox-empty').textContent = `Ошибка загрузки: ${e.message}`;
        setOpen(true);
      }
    }
    function choose(item) {
      current = item;
      input.value = item.label || item.value;
      clearBtn.style.display = '';
      setOpen(false);
      if (typeof opts.onSelect === 'function') opts.onSelect(item);
    }
    function clear(silent) {
      current = null;
      input.value = '';
      clearBtn.style.display = 'none';
      setOpen(false);
      if (!silent && typeof opts.onSelect === 'function') opts.onSelect(null);
    }
    input.addEventListener('input', () => {
      current = null;
      clearBtn.style.display = input.value ? '' : 'none';
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => search(input.value.trim()), opts.debounceMs || 250);
    });
    input.addEventListener('focus', () => { if (!host.classList.contains('open')) search(input.value.trim()); });
    input.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown') { e.preventDefault(); if (!host.classList.contains('open')) { search(input.value.trim()); return; } highlight(Math.min(items.length - 1, activeIdx + 1)); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); highlight(Math.max(0, activeIdx - 1)); }
      else if (e.key === 'Enter') { if (host.classList.contains('open') && items[activeIdx]) { e.preventDefault(); choose(items[activeIdx]); } }
      else if (e.key === 'Escape') { setOpen(false); }
    });
    input.addEventListener('blur', () => setTimeout(() => setOpen(false), 120));
    clearBtn.addEventListener('click', () => { clear(false); input.focus(); });
    return {
      getValue: () => (current ? current.value : ''),
      getItem: () => current,
      setItem(item) { if (item) choose(item); else clear(true); },
      setValue(value, label, meta) { if (value) choose({ value, label: label || value, meta }); else clear(true); },
      clear: () => clear(true),
      focus: () => input.focus(),
      input
    };
  };

  // ---- tabs --------------------------------------------------------------
  // nav: container with buttons [data-target=panelId]; panels toggle .active; keyboard arrows supported.
  ui.tabs = function tabs(nav, options) {
    const opts = options || {};
    nav.setAttribute('role', 'tablist');
    const buttons = Array.from(nav.querySelectorAll('[data-target]'));
    function panelOf(btn) { return document.getElementById(btn.dataset.target); }
    function activate(btn, focus) {
      buttons.forEach((b) => {
        const on = b === btn;
        b.classList.toggle('active', on);
        b.setAttribute('aria-selected', on ? 'true' : 'false');
        b.setAttribute('tabindex', on ? '0' : '-1');
        const panel = panelOf(b);
        if (panel) { panel.classList.toggle('active', on); panel.hidden = !on; }
      });
      if (focus) btn.focus();
      if (typeof opts.onChange === 'function') opts.onChange(btn.dataset.target, btn);
    }
    buttons.forEach((btn, idx) => {
      btn.setAttribute('role', 'tab');
      btn.setAttribute('type', 'button');
      const panel = panelOf(btn);
      if (panel) {
        panel.setAttribute('role', 'tabpanel');
        if (!btn.id) btn.id = `${btn.dataset.target}-tab`;
        panel.setAttribute('aria-labelledby', btn.id);
      }
      btn.addEventListener('click', () => activate(btn, false));
      btn.addEventListener('keydown', (e) => {
        let next = null;
        if (e.key === 'ArrowRight') next = buttons[(idx + 1) % buttons.length];
        else if (e.key === 'ArrowLeft') next = buttons[(idx - 1 + buttons.length) % buttons.length];
        else if (e.key === 'Home') next = buttons[0];
        else if (e.key === 'End') next = buttons[buttons.length - 1];
        if (next) { e.preventDefault(); activate(next, true); }
      });
    });
    const initial = buttons.find((b) => b.classList.contains('active')) || buttons[0];
    if (initial) activate(initial, false);
    return { activate: (target) => { const btn = buttons.find((b) => b.dataset.target === target); if (btn) activate(btn, false); } };
  };

  // ---- glossary ----------------------------------------------------------
  LL.GLOSSARY = LL.GLOSSARY || {
    stable_max: 'Максимальная нагрузка (RPS), которую система держала стабильно не менее заданного времени (по умолчанию 5 минут) без деградации. Кратковременные пики не учитываются.',
    peak_max: 'Абсолютный максимум метрики за окно теста. Используется как запасной вариант, когда стабильная ступень не найдена.',
    verdict_success: 'Все заданные SLA-критерии соблюдены (или, если SLA не заданы, ИИ не нашёл существенных проблем).',
    verdict_risk: 'Целевой RPS достигнут, но нарушены вторичные пороги (задержки, ошибки, ресурсы), либо ИИ отметил локальные деградации.',
    verdict_fail: 'Целевой RPS не достигнут или нарушены критичные пороги SLA.',
    verdict_na: 'Метрик недостаточно для оценки: не собраны данные lt_framework или не заданы критерии.',
    sla_first: 'Если для сервиса заданы SLA-критерии, итоговый вердикт определяется ими; оценка ИИ показывается рядом как дополнение.',
    confidence: 'Служебная оценка качества ответа ИИ: сочетание оценки судьи (второй модели) и эвристической проверки чисел и названий метрик по данным.',
    judge: 'Отдельный запрос к модели, который оценивает кандидаты ответов по рубрике: опора на данные, покрытие проблем, конкретика, учёт SLA, полезность рекомендаций.',
    p95: '95-й перцентиль: значение, ниже которого лежат 95 % измерений. Показывает «хвост» задержек без влияния единичных выбросов.',
    step_profile: 'Ступенчатый тест: нагрузка растёт ступенями; анализ ищет последнюю ступень, на которой система была стабильна.',
    resample: 'Интервал агрегации временных рядов перед анализом (например, 5 минут): метрики усредняются внутри интервала.',
    target_rps: 'Целевая пропускная способность. В ступенчатом и стресс-тесте — уровень, который нужно достичь на стабильной ступени; в soak — уровень, который нужно удерживать весь тест; в spike — уровень всплеска.',
    error_rate: 'Доля запросов с ошибками в процентах за окно теста.',
    trend_better: 'Изменение метрики между прогонами оценивается по её смыслу: рост RPS — лучше, рост задержек и ошибок — хуже.',
    unverified: 'Числа в находке не нашлись в собранных метриках. Такая находка остаётся в отчёте, но не используется как основание вердикта.',
    test_type_step: 'Ступенчатый тест: нагрузка растёт ступенями; анализ ищет последнюю ступень, на которой система была стабильна.',
    test_type_soak: 'Долговременный тест: заданная нагрузка держится долго; анализ проверяет удержание нагрузки и ищет дрейф задержек, ошибок и ресурсов.',
    test_type_spike: 'Тест всплесками: нагрузка резко растёт и возвращается к фону; анализ оценивает рост задержек и ошибок во время всплеска и время восстановления.',
    test_type_stress: 'Стресс-тест: нагрузка выходит за проектную мощность; анализ ищет точку насыщения, характер деградации и ограничивающий ресурс.',
    hold_window: 'Окно удержания: часть soak-теста без разгона и финального спада, где RPS держался на рабочем уровне. SLA soak-теста проверяется на этом окне.',
    trend_drift: 'Изменение метрики от начала к концу окна удержания при неизменной нагрузке. Рост задержек, ошибок или памяти больше порога отмечается как риск.',
    spike_window: 'Всплеск: участок, где RPS поднялся выше фона в 1,5 раза и держался не меньше минуты.',
    overshoot: 'Рост p95 во время всплеска относительно фона прямо перед ним.',
    recovery_time: 'Время от конца всплеска до момента, когда p95 вернулся к уровню фона (с допуском 10 %).',
    saturation_point: 'Точка насыщения: последняя ступень стресс-теста, на которой выполнялись пороги задержек и ошибок.',
    limiting_resource: 'Ресурс узлов (CPU или память), который превысил порог SLA или сильнее всего вырос к ступени деградации.'
  };
  ui.term = function term(key, text) {
    const tip = LL.GLOSSARY[key];
    const label = text || key;
    if (!tip) return escapeHtml(label);
    return `<abbr class="term" data-term="${escapeHtml(key)}" data-tip="${escapeHtml(tip)}" title="${escapeHtml(tip)}" tabindex="0">${escapeHtml(label)}</abbr>`;
  };
  ui.applyTerms = function applyTerms(root) {
    (root || document).querySelectorAll('abbr.term[data-term]:not([data-tip])').forEach((el) => {
      const tip = LL.GLOSSARY[el.dataset.term];
      if (tip) { el.setAttribute('data-tip', tip); el.title = tip; if (!el.hasAttribute('tabindex')) el.tabIndex = 0; }
    });
  };

  // ---- skeletons ---------------------------------------------------------
  ui.skeleton = function skeleton(lines, widths) {
    const n = Math.max(1, lines || 3);
    const ws = widths || ['w-60', '', 'w-40'];
    let html = '';
    for (let i = 0; i < n; i += 1) html += `<div class="skeleton skeleton-line ${ws[i % ws.length]}"></div>`;
    return html;
  };
  // Table body placeholders while data loads: `rows` rows of `cols` shimmering cells.
  ui.skeletonRows = function skeletonRows(cols, rows = 5) {
    const nCols = Math.max(1, Math.floor(Number(cols)) || 1);
    const nRows = Math.max(1, Math.floor(Number(rows)) || 1);
    const widths = ['w-60', '', 'w-40'];
    let html = '';
    for (let r = 0; r < nRows; r += 1) {
      html += '<tr class="skeleton-row" aria-hidden="true">';
      for (let c = 0; c < nCols; c += 1) html += `<td><div class="skeleton skeleton-line ${widths[(r + c) % widths.length]}"></div></td>`;
      html += '</tr>';
    }
    return html;
  };

  // ---- empty states ------------------------------------------------------
  // <svg> referencing a symbol of the sprite in base.html (#i-<name>); '' for an empty name.
  ui.icon = function icon(name, className) {
    const id = String(name || '').replace(/[^a-z0-9-]/gi, '');
    if (!id) return '';
    const cls = String(className || '').replace(/[^\w\- ]/g, '').trim();
    return `<svg class="icon${cls ? ` ${cls}` : ''}" aria-hidden="true"><use href="#i-${id}"/></svg>`;
  };
  // One table row spanning the table; `text` is escaped, `actionHtml` is trusted markup (a link or button).
  // With `icon` (a sprite name) the row becomes a centered block: icon, text, action.
  ui.emptyRow = function emptyRow(colspan, text, options) {
    const { actionHtml, icon } = options || {};
    const span = Math.max(1, Math.floor(Number(colspan)) || 1);
    if (icon) {
      const action = actionHtml ? `<span class="empty-row-action">${actionHtml}</span>` : '';
      return `<tr class="empty-row"><td colspan="${span}"><div class="empty-row-body">${ui.icon(icon, 'empty-icon')}<span class="empty-row-text">${escapeHtml(text)}</span>${action}</div></td></tr>`;
    }
    const action = actionHtml ? ` ${actionHtml}` : '';
    return `<tr class="empty-row"><td colspan="${span}">${escapeHtml(text)}${action}</td></tr>`;
  };
  // Block empty state: { icon, title, text, actionHtml } -> <div class="empty-state">; title and text are
  // escaped, actionHtml is trusted markup.
  ui.emptyState = function emptyState(options) {
    const { icon, title, text, actionHtml } = options || {};
    return [
      '<div class="empty-state">',
      ui.icon(icon, 'empty-icon'),
      title ? `<h3>${escapeHtml(title)}</h3>` : '',
      text ? `<p class="empty-state-text">${escapeHtml(text)}</p>` : '',
      actionHtml ? `<div class="empty-state-action">${actionHtml}</div>` : '',
      '</div>'
    ].join('');
  };

  // ---- responsive tables -------------------------------------------------
  // Below 720px a table.data-table that is a direct child of .table-wrap (unless .no-cards) shows its
  // rows as cards; every cell shows its column title from data-label (td::before). labelTable copies the
  // titles from the last <thead> row and marks special cells: data-cell="primary" (card title, first
  // regular cell), "actions" (last cell with buttons or menus, full width) and "check" (a lone checkbox,
  // kept compact). Cells spanning several columns (empty and loading rows) get no label.
  const CARD_TABLES = '.table-wrap > table.data-table:not(.no-cards)';
  function setAttr(el, name, value) {
    if (value) { if (el.getAttribute(name) !== value) el.setAttribute(name, value); }
    else if (el.hasAttribute(name)) el.removeAttribute(name);
  }
  function headerLabels(table) {
    const head = table.tHead;
    const row = head && head.rows && head.rows.length ? head.rows[head.rows.length - 1] : null;
    if (!row) return null;
    const labels = [];
    Array.from(row.cells).forEach((th) => {
      const text = String(th.getAttribute('data-label') || th.textContent || '').replace(/\s+/g, ' ').trim();
      const span = Math.max(1, Number(th.colSpan) || 1);
      for (let i = 0; i < span; i += 1) labels.push(text);
    });
    return labels;
  }
  function isCheckCell(td) {
    const inputs = td.querySelectorAll('input');
    if (inputs.length !== 1) return false;
    const type = String(inputs[0].type || '').toLowerCase();
    return (type === 'checkbox' || type === 'radio') && !String(td.textContent || '').trim();
  }
  function isActionsCell(td) {
    return td.classList.contains('actions') || !!td.querySelector('button, a.btn, details.menu');
  }
  ui.labelTable = function labelTable(table) {
    if (!table || !table.tBodies) return;
    const labels = headerLabels(table);
    if (!labels) return;
    Array.from(table.tBodies).forEach((tbody) => {
      Array.from(tbody.rows).forEach((tr) => {
        const special = tr.classList.contains('empty-row') || tr.classList.contains('skeleton-row');
        const cells = Array.from(tr.cells);
        let col = 0;
        let primaryDone = false;
        let hasCheck = false;
        cells.forEach((td, i) => {
          const span = Math.max(1, Number(td.colSpan) || 1);
          const plain = !special && span === 1;
          let kind = '';
          if (plain && isCheckCell(td)) kind = 'check';
          else if (plain && i > 0 && i === cells.length - 1 && isActionsCell(td)) kind = 'actions';
          else if (plain && !primaryDone) { kind = 'primary'; primaryDone = true; }
          if (kind === 'check') hasCheck = true;
          setAttr(td, 'data-cell', kind);
          setAttr(td, 'data-label', plain && kind !== 'check' && kind !== 'actions' ? (labels[col] || '') : '');
          col += span;
        });
        setAttr(tr, 'data-has-check', hasCheck ? '1' : '');
      });
    });
  };
  function tablesIn(node, found) {
    if (!node || node.nodeType !== 1) return;
    if (node.matches('table.data-table')) found.add(node);
    else if (node.getElementsByTagName('table').length) node.querySelectorAll('table.data-table').forEach((t) => found.add(t));
  }
  // Labels the tables already on the page and every row that page scripts render later. Only child list
  // changes are observed, so setting the attributes does not re-trigger the observer.
  function watchCardTables() {
    document.querySelectorAll(CARD_TABLES).forEach((table) => ui.labelTable(table));
    if (typeof MutationObserver !== 'function' || !document.body) return;
    const observer = new MutationObserver((records) => {
      const found = new Set();
      records.forEach((rec) => {
        const target = rec.target;
        if (!target || target.nodeType !== 1) return;
        const own = target.closest('table.data-table');
        if (own) found.add(own);
        // Added subtrees may hold their own tables, also inside a cell of another table.
        rec.addedNodes.forEach((node) => tablesIn(node, found));
      });
      found.forEach((table) => { if (table.matches(CARD_TABLES)) ui.labelTable(table); });
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }

  // ---- header wiring -----------------------------------------------------
  document.addEventListener('DOMContentLoaded', () => {
    wireViewMenu();
    // On narrow screens the sidebar is an off-canvas drawer toggled from the top bar.
    const navToggle = document.getElementById('navToggle');
    const sidebar = document.getElementById('appSidebar');
    if (navToggle && sidebar) {
      const setOpen = (open) => {
        document.body.classList.toggle('nav-open', open);
        navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
        navToggle.setAttribute('aria-label', open ? 'Закрыть меню' : 'Открыть меню');
      };
      navToggle.addEventListener('click', () => setOpen(!document.body.classList.contains('nav-open')));
      const navClose = document.getElementById('navClose');
      if (navClose) navClose.addEventListener('click', () => setOpen(false));
      const backdrop = document.getElementById('navBackdrop');
      if (backdrop) backdrop.addEventListener('click', () => setOpen(false));
      document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && document.body.classList.contains('nav-open')) setOpen(false); });
    }
    // The sidebar user and view menus close on an outside click or Escape.
    const closeUserMenus = (except) => document.querySelectorAll('details.user-menu[open], details.view-menu[open]').forEach((m) => { if (!except || !m.contains(except)) m.removeAttribute('open'); });
    document.addEventListener('click', (event) => closeUserMenus(event.target));
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeUserMenus(null); });
    applyChartDefaults();
    ui.theme.onChange(() => ui.restyleCharts());
    ui.applyTerms(document);
    watchCardTables();
    if (typeof LL.initProjectArea === 'function') LL.initProjectArea();
  });
})();
