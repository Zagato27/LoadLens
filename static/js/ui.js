// Shared UI components: toasts, dialogs, combobox, tabs, theme, glossary tooltips.
(function () {
  const LL = window.LoadLens = window.LoadLens || {};
  const ui = LL.ui = LL.ui || {};
  const THEME_KEY = 'loadlens.theme';

  function escapeHtml(value) {
    return String(value === undefined || value === null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  ui.escapeHtml = escapeHtml;

  // ---- theme -------------------------------------------------------------
  const themeListeners = [];
  ui.theme = {
    current() {
      return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
    },
    set(name) {
      const next = name === 'light' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* storage unavailable */ }
      syncThemeButton();
      themeListeners.forEach((cb) => { try { cb(next); } catch (e) { /* listener error must not break others */ } });
    },
    toggle() { ui.theme.set(ui.theme.current() === 'light' ? 'dark' : 'light'); },
    onChange(cb) { if (typeof cb === 'function') themeListeners.push(cb); }
  };
  function syncThemeButton() {
    const btn = document.getElementById('themeToggle');
    if (!btn) return;
    const light = ui.theme.current() === 'light';
    btn.textContent = light ? '☾' : '☀';
    btn.setAttribute('aria-label', light ? 'Включить тёмную тему' : 'Включить светлую тему');
    btn.title = btn.getAttribute('aria-label');
  }

  // Chart.js colors follow the active theme.
  ui.chartColors = function chartColors() {
    const css = getComputedStyle(document.documentElement);
    return {
      grid: css.getPropertyValue('--chart-grid').trim() || '#2f2f2f',
      text: css.getPropertyValue('--chart-text').trim() || '#bbb',
      background: css.getPropertyValue('--bg-chart').trim() || '#151515',
      accent: css.getPropertyValue('--accent').trim() || '#6200ee',
      accentSoft: css.getPropertyValue('--accent-soft').trim() || 'rgba(98, 0, 238, 0.16)'
    };
  };
  function applyChartDefaults() {
    if (!window.Chart) return;
    const colors = ui.chartColors();
    window.Chart.defaults.color = colors.text;
    window.Chart.defaults.borderColor = colors.grid;
    window.Chart.defaults.font.family = "'Montserrat', Arial, sans-serif";
  }
  ui.restyleCharts = function restyleCharts() {
    if (!window.Chart) return;
    applyChartDefaults();
    const colors = ui.chartColors();
    Object.values(window.Chart.instances || {}).forEach((chart) => {
      try {
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
    target_rps: 'Целевая пропускная способность. Главный критерий SLA: если достигнута на стабильной ступени, тест считается успешным.',
    error_rate: 'Доля запросов с ошибками в процентах за окно теста.',
    trend_better: 'Изменение метрики между прогонами оценивается по её смыслу: рост RPS — лучше, рост задержек и ошибок — хуже.',
    unverified: 'Числа в находке не нашлись в собранных метриках. Такая находка остаётся в отчёте, но не используется как основание вердикта.'
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

  // ---- header wiring -----------------------------------------------------
  document.addEventListener('DOMContentLoaded', () => {
    syncThemeButton();
    const themeBtn = document.getElementById('themeToggle');
    if (themeBtn) themeBtn.addEventListener('click', () => ui.theme.toggle());
    const navToggle = document.getElementById('navToggle');
    const nav = document.getElementById('appNav');
    if (navToggle && nav) {
      navToggle.addEventListener('click', () => {
        const open = nav.classList.toggle('open');
        navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
        navToggle.setAttribute('aria-label', open ? 'Закрыть меню' : 'Открыть меню');
      });
    }
    // The header user menu closes on an outside click or Escape.
    const closeUserMenus = (except) => document.querySelectorAll('details.user-menu[open]').forEach((m) => { if (!except || !m.contains(except)) m.removeAttribute('open'); });
    document.addEventListener('click', (event) => closeUserMenus(event.target));
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeUserMenus(null); });
    applyChartDefaults();
    ui.theme.onChange(() => ui.restyleCharts());
    ui.applyTerms(document);
    if (typeof LL.initProjectArea === 'function') LL.initProjectArea();
  });
})();
