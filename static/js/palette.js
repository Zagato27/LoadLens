// SPDX-License-Identifier: Apache-2.0
// Command palette: Ctrl+K / Cmd+K or the «Search» button of the top bar (#paletteOpen).
// Sources: pages from the top bar navigation (Jinja already filtered it by role), reports from
// GET /runs, project areas (LL.projectAreas) and actions. The «/» key focuses the page's own search
// field (marked with data-slash-focus, e.g. the archive) unless focus is already in a text field.
(function () {
  const LL = window.LoadLens = window.LoadLens || {};
  const palette = LL.palette = LL.palette || {};
  const RUNS_LIMIT = 10;
  const RECENT_RUNS_LIMIT = 5;
  const DEBOUNCE_MS = 250;
  const NON_TEXT_INPUTS = ['checkbox', 'radio', 'button', 'submit', 'reset', 'range', 'color', 'file', 'image'];

  // Extra words a page is found by (the label is always searched): Russian and English, so a page is found by
  // its name in either language whatever the UI language is. Not shown, so not translated.
  const PAGE_KEYWORDS = {
    '/': 'главная обзор дэшборд дашборд dashboard home overview',
    '/launch': 'запуск теста запустить нагрузку генератор launch run test',
    '/reports': 'архив отчёты список история archive reports history',
    '/compare': 'сравнение сравнить baseline compare',
    '/forecasting': 'прогнозирование прогноз мощности usl forecasting capacity',
    '/settings': 'настройки конфигурация источники сервисы settings configuration',
    '/account': 'профиль пароль токены api account profile password tokens',
  };

  // ---- pure helpers (LL.palette._internals) ------------------------------------------------

  function normalize(text) {
    return String(text === null || text === undefined ? '' : text).toLowerCase().replace(/ё/g, 'е').replace(/\s+/g, ' ').trim();
  }

  // Rank of an item for a query, lower is better: 0 the label starts with the query, 1 a word of the label
  // starts with its first word, 2 the label contains it, 3 only keywords or meta match; -1 no match.
  // Every word of the query must occur in the label, keywords or meta.
  function matchRank(item, query) {
    const q = normalize(query);
    if (!q) return 0;
    const label = normalize(item && item.label);
    const haystack = `${label} ${normalize(item && item.keywords)} ${normalize(item && item.meta)}`;
    const words = q.split(' ');
    if (!words.every((w) => haystack.includes(w))) return -1;
    if (label.startsWith(q)) return 0;
    if (label.split(/[\s\-_·./]+/).some((part) => part.startsWith(words[0]))) return 1;
    if (label.includes(words[0])) return 2;
    return 3;
  }

  // Matching items, best rank first; equal ranks keep their original order.
  function filterItems(items, query) {
    return (Array.isArray(items) ? items : [])
      .map((item, idx) => ({ item, idx, rank: matchRank(item, query) }))
      .filter((entry) => entry.rank >= 0)
      .sort((a, b) => a.rank - b.rank || a.idx - b.idx)
      .map((entry) => entry.item);
  }

  // links: [{href, label, icon}] read from the top bar. /new becomes the «Новый отчёт» action instead.
  function pageItems(links) {
    const seen = new Set();
    const out = [];
    (Array.isArray(links) ? links : []).forEach((link) => {
      const href = String((link && link.href) || '');
      const label = String((link && link.label) || '').trim();
      if (!href || !label || href === '/new' || seen.has(href)) return;
      seen.add(href);
      out.push({ kind: 'page', key: `page:${href}`, label, href, icon: (link && link.icon) || '', keywords: PAGE_KEYWORDS[href] || '' });
    });
    return out;
  }

  // Theme of the page for the actions: the resolved theme and the stored mode of LL.ui.theme ('system' without a
  // stored choice). Without ui.js the mode is unknown ('') and the page counts as dark.
  function themeState() {
    const theme = LL.ui && LL.ui.theme ? LL.ui.theme : null;
    return {
      theme: theme && typeof theme.current === 'function' ? theme.current() : 'dark',
      themeMode: theme && typeof theme.mode === 'function' ? theme.mode() : '',
    };
  }

  // canCreate: the top bar offers /new (engineers and admins); theme: the current one; themeMode: the stored mode
  // (themeState); languageSwitch: the top bar menu offers the interface language (off with
  // appearance.language_switch=false); locale: the current language.
  function actionItems(options) {
    const opts = options || {};
    const out = [];
    if (opts.canCreate) {
      out.push({ kind: 'action', key: 'action:new', label: LL.t('palette.action.new_report'), href: '/new', icon: 'file-plus', keywords: 'создать анализ new report' });
    }
    const light = opts.theme === 'light';
    out.push({
      kind: 'action', key: 'action:theme', action: 'theme',
      label: light ? LL.t('palette.action.theme_dark') : LL.t('palette.action.theme_light'),
      icon: light ? 'moon' : 'sun',
      keywords: 'сменить тему оформление тёмная светлая theme dark light',
    });
    // «Theme as in the system» is offered whenever the mode is not system already: after an explicit light or dark
    // choice (made here, in the view menu or in another tab) and when the mode is unknown.
    if (opts.themeMode !== 'system') {
      out.push({
        kind: 'action', key: 'action:theme-system', action: 'theme-system',
        label: LL.t('palette.action.theme_system'), icon: 'sliders',
        keywords: 'тема системная авто оформление theme system auto',
      });
    }
    if (opts.languageSwitch) {
      // The label is in the language it switches to («Switch to English» in ru, «Переключить на русский» in en).
      out.push({
        kind: 'action', key: 'action:lang', action: 'lang', lang: opts.locale === 'en' ? 'ru' : 'en',
        label: LL.t('palette.action.switch_language'), icon: 'globe',
        keywords: 'язык интерфейса русский английский language interface english russian',
      });
    }
    return out;
  }

  // areas: LL.projectAreas (strings or {id, title}); «All areas» first, the active one marked.
  // Empty when there is nothing to switch to.
  function areaItems(areas, active) {
    const current = String(active || '');
    const list = [{ id: '', title: LL.t('ui.project_area.all') }];
    (Array.isArray(areas) ? areas : []).forEach((a) => {
      const id = a && typeof a === 'object' ? String(a.id || '') : String(a || '');
      if (!id || list.some((x) => x.id === id)) return;
      list.push({ id, title: (a && typeof a === 'object' && a.title) ? String(a.title) : id });
    });
    if (current && !list.some((x) => x.id === current)) list.push({ id: current, title: current });
    if (list.length < 2) return [];
    return list.map((x) => ({
      kind: 'area', key: `area:${x.id}`, area: x.id, label: x.title, icon: 'columns',
      current: x.id === current, meta: x.id === current ? LL.t('palette.area.current') : '', keywords: `область проект area project ${x.id}`,
    }));
  }

  function runItems(runs, now) {
    return (Array.isArray(runs) ? runs : []).filter((r) => r && r.run_name).map((r) => {
      const when = r.end_time || r.start_time || r.report_created_at;
      const meta = [r.service, when ? LL.relativeTime(when, now) : '', r.verdict ? LL.verdictLabel(r.verdict) : ''].filter(Boolean).join(' · ');
      return { kind: 'run', key: `run:${r.run_name}`, label: r.run_name, meta, href: LL.reportUrl(r), verdict: r.verdict || '' };
    });
  }

  // runs: {status: 'off'|'loading'|'ready'|'error', items}. Groups without items or a note are dropped.
  function buildGroups(sources, query) {
    const src = sources || {};
    const runs = src.runs || { status: 'off', items: [] };
    const searching = !!normalize(query);
    const groups = [];
    const add = (id, title, items, note) => { if (items.length || note) groups.push({ id, title, items, note: note || '' }); };
    add('pages', LL.t('palette.group.pages'), filterItems(src.pages, query));
    add('actions', LL.t('palette.group.actions'), filterItems(src.actions, query));
    let runsNote = '';
    if (runs.status === 'loading') runsNote = searching ? LL.t('palette.runs.searching') : LL.t('palette.runs.loading');
    else if (runs.status === 'error') runsNote = LL.t('palette.runs.error');
    else if (runs.status === 'ready' && searching && !(runs.items || []).length) runsNote = LL.t('palette.runs.none');
    add('runs', searching ? LL.t('palette.group.runs') : LL.t('palette.group.recent_runs'), runs.status === 'ready' ? (runs.items || []) : [], runsNote);
    add('areas', LL.t('palette.group.areas'), filterItems(src.areas, query));
    return groups;
  }

  // Arrow keys stop at the ends of the list, as in LL.ui.combobox.
  function moveIndex(index, delta, count) {
    if (!count) return -1;
    return Math.min(count - 1, Math.max(0, (index < 0 ? -1 : index) + delta));
  }

  function isOpenShortcut(event) {
    if (!event || !(event.ctrlKey || event.metaKey) || event.altKey || event.shiftKey) return false;
    const key = String(event.key || '').toLowerCase();
    // event.code keeps Ctrl+K working on the Russian layout, where event.key is «л».
    return event.code === 'KeyK' || key === 'k' || key === 'л';
  }

  function isTextTarget(el) {
    if (!el || el.nodeType !== 1) return false;
    if (el.isContentEditable) return true;
    const tag = String(el.tagName || '').toUpperCase();
    if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
    if (tag !== 'INPUT') return false;
    const type = String(el.type || (el.getAttribute && el.getAttribute('type')) || 'text').toLowerCase();
    return !NON_TEXT_INPUTS.includes(type);
  }

  function isSlashShortcut(event) {
    return !!event && event.key === '/' && !event.ctrlKey && !event.metaKey && !event.altKey && !isTextTarget(event.target);
  }

  palette._internals = {
    normalize, matchRank, filterItems, pageItems, themeState, actionItems, areaItems, runItems, buildGroups, moveIndex,
    isOpenShortcut, isTextTarget, isSlashShortcut, RUNS_LIMIT, RECENT_RUNS_LIMIT, DEBOUNCE_MS,
  };

  // ---- dialog ------------------------------------------------------------------------------

  const esc = (value) => (LL.ui && LL.ui.escapeHtml ? LL.ui.escapeHtml(value) : String(value));
  let dialog = null;
  let input = null;
  let listEl = null;
  let statusEl = null;
  let flat = [];
  let activeKey = '';
  let runsState = { status: 'off', items: [] };
  let runsSeq = 0;
  let debounceTimer = null;
  let lastFocus = null;
  let switching = false;

  function isOpen() {
    return !!(dialog && dialog.open);
  }

  function navLinks() {
    const links = [];
    document.querySelectorAll('#appNav a.nav-btn[href]').forEach((a) => {
      const use = a.querySelector('use');
      const ref = use ? (use.getAttribute('href') || use.getAttribute('xlink:href') || '') : '';
      const label = a.querySelector('.nav-label');
      links.push({ href: a.getAttribute('href'), label: (label || a).textContent.trim(), icon: ref.replace(/^#i-/, '') });
    });
    document.querySelectorAll('#appSidebar .user-menu a.menu-link[href]').forEach((a) => {
      links.push({ href: a.getAttribute('href'), label: a.textContent.trim(), icon: '' });
    });
    return links;
  }

  // While a password change is enforced the top bar has no links and data calls are refused.
  function dataAllowed() {
    return !!document.querySelector('#appNav a.nav-btn[href]');
  }

  function sources() {
    const links = navLinks();
    return {
      pages: pageItems(links),
      actions: actionItems(Object.assign(themeState(), {
        canCreate: links.some((l) => l.href === '/new'),
        languageSwitch: !!(LL.ui && LL.ui.locale && document.querySelector('#viewMenu [data-view-pref="lang"]')),
        locale: LL.ui && LL.ui.locale ? LL.ui.locale.current() : LL.locale,
      })),
      areas: dataAllowed() ? areaItems(LL.projectAreas, LL.activeProjectArea) : [],
      runs: runsState,
    };
  }

  function iconHtml(item) {
    if (item.kind === 'run') return `<span class="palette-dot ${esc(LL.verdictClass ? LL.verdictClass(item.verdict) : 'na')}" aria-hidden="true"></span>`;
    if (item.icon) return `<svg class="icon" aria-hidden="true"><use href="#i-${esc(item.icon)}"/></svg>`;
    return '<span class="icon" aria-hidden="true"></span>';
  }

  function render() {
    if (!dialog) return;
    const query = input.value;
    const groups = buildGroups(sources(), query);
    flat = [];
    let html = '';
    groups.forEach((group) => {
      html += `<div class="palette-group" role="group" aria-labelledby="palette-g-${group.id}"><div class="palette-group-title" id="palette-g-${group.id}">${esc(group.title)}</div>`;
      group.items.forEach((item) => {
        const n = flat.length;
        flat.push(item);
        html += `<div class="palette-option${item.current ? ' current' : ''}" role="option" id="palette-opt-${n}" data-index="${n}" aria-selected="false">${iconHtml(item)}<span class="palette-label">${esc(item.label)}</span>${item.meta ? `<span class="palette-meta">${esc(item.meta)}</span>` : ''}</div>`;
      });
      if (group.note) html += `<div class="palette-note" aria-hidden="true">${esc(group.note)}</div>`;
      html += '</div>';
    });
    if (!groups.length) html = `<div class="palette-empty">${esc(LL.t('palette.empty'))}</div>`;
    listEl.innerHTML = html;
    let idx = flat.findIndex((item) => item.key === activeKey);
    if (idx < 0) idx = flat.length ? 0 : -1;
    highlight(idx);
    if (!switching) statusEl.textContent = flat.length ? LL.t('palette.found', { n: flat.length }) : LL.t('palette.empty');
  }

  function highlight(idx) {
    const options = listEl.querySelectorAll('.palette-option');
    options.forEach((el, i) => el.setAttribute('aria-selected', i === idx ? 'true' : 'false'));
    activeKey = idx >= 0 && flat[idx] ? flat[idx].key : '';
    const active = options[idx];
    if (active) {
      input.setAttribute('aria-activedescendant', active.id);
      active.scrollIntoView({ block: 'nearest' });
    } else {
      input.removeAttribute('aria-activedescendant');
    }
  }

  function activeIndex() {
    return flat.findIndex((item) => item.key === activeKey);
  }

  function requestRuns(query, immediate) {
    clearTimeout(debounceTimer);
    runsSeq += 1;
    if (!dataAllowed()) { runsState = { status: 'off', items: [] }; return; }
    runsState = { status: 'loading', items: [] };
    const run = () => fetchRuns(query);
    if (immediate) run();
    else debounceTimer = setTimeout(run, DEBOUNCE_MS);
  }

  async function fetchRuns(query) {
    const seq = ++runsSeq;
    const params = new URLSearchParams();
    if (query) params.set('q', query);
    params.set('limit', String(query ? RUNS_LIMIT : RECENT_RUNS_LIMIT));
    let next;
    try {
      const resp = await fetch(`/runs?${params.toString()}`);
      const rows = await resp.json();
      if (!resp.ok || !Array.isArray(rows)) throw new Error((rows && rows.error) || `HTTP ${resp.status}`);
      next = { status: 'ready', items: runItems(rows, Date.now()) };
    } catch (e) {
      next = { status: 'error', items: [] };
    }
    // Drop answers to an older query or for a palette that was closed meanwhile.
    if (seq !== runsSeq || !isOpen()) return;
    runsState = next;
    render();
  }

  async function loadAreas() {
    if (!dataAllowed() || typeof LL.initProjectArea !== 'function') return;
    try { await LL.initProjectArea(); } catch (e) { /* areas stay hidden */ }
    if (isOpen()) render();
  }

  async function switchArea(item) {
    if (item.current) { close(); return; }
    switching = true;
    input.disabled = true;
    statusEl.textContent = LL.t('palette.switching_area', { name: item.label });
    listEl.setAttribute('aria-busy', 'true');
    try {
      await fetch('/project_area', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_area: item.area || '' }),
      });
    } finally {
      location.reload();
    }
  }

  function activate(item, newTab) {
    if (!item || switching) return;
    if (item.kind === 'area') { switchArea(item); return; }
    if (item.action === 'theme') {
      if (LL.ui && LL.ui.theme) LL.ui.theme.toggle();
      close();
      return;
    }
    if (item.action === 'theme-system') {
      if (LL.ui && LL.ui.theme) LL.ui.theme.set('system');
      close();
      return;
    }
    if (item.action === 'lang') {
      close();
      if (LL.ui && LL.ui.locale) LL.ui.locale.set(item.lang);
      return;
    }
    if (!item.href) return;
    if (newTab) { window.open(item.href, '_blank', 'noopener'); return; }
    close(false);
    location.href = item.href;
  }

  function onKeydown(e) {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      highlight(moveIndex(activeIndex(), e.key === 'ArrowDown' ? 1 : -1, flat.length));
    } else if (e.key === 'PageDown' || e.key === 'PageUp') {
      e.preventDefault();
      highlight(moveIndex(activeIndex(), e.key === 'PageDown' ? 5 : -5, flat.length));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      activate(flat[activeIndex()], e.ctrlKey || e.metaKey);
    }
  }

  function build() {
    dialog = document.createElement('dialog');
    dialog.className = 'll-dialog palette-dialog';
    dialog.setAttribute('aria-label', LL.t('palette.dialog_label'));
    dialog.innerHTML = [
      '<div class="palette-search">',
      '<svg class="icon" aria-hidden="true"><use href="#i-search"/></svg>',
      '<input type="text" class="palette-input" role="combobox" aria-expanded="true" aria-controls="paletteList" aria-autocomplete="list"',
      ` aria-label="${esc(LL.t('palette.search_label'))}" placeholder="${esc(LL.t('palette.search_placeholder'))}" autocomplete="off" spellcheck="false" />`,
      '<kbd class="palette-kbd" aria-hidden="true">Esc</kbd>',
      '</div>',
      `<div class="palette-list" id="paletteList" role="listbox" aria-label="${esc(LL.t('palette.results'))}"></div>`,
      `<div class="palette-foot" aria-hidden="true"><span><kbd class="palette-kbd">↑</kbd><kbd class="palette-kbd">↓</kbd> ${esc(LL.t('palette.foot.select'))}</span>`,
      `<span><kbd class="palette-kbd">Enter</kbd> ${esc(LL.t('palette.foot.open'))}</span><span><kbd class="palette-kbd">Ctrl</kbd>+<kbd class="palette-kbd">Enter</kbd> ${esc(LL.t('palette.foot.new_tab'))}</span></div>`,
      '<div class="visually-hidden" role="status" aria-live="polite"></div>',
    ].join('');
    document.body.appendChild(dialog);
    input = dialog.querySelector('.palette-input');
    listEl = dialog.querySelector('.palette-list');
    statusEl = dialog.querySelector('[role="status"]');
    input.addEventListener('input', () => {
      activeKey = '';
      requestRuns(input.value.trim(), false);
      render();
    });
    input.addEventListener('keydown', onKeydown);
    // Clicks keep focus in the input; the option under the pointer becomes the active one.
    listEl.addEventListener('mousedown', (e) => e.preventDefault());
    listEl.addEventListener('mousemove', (e) => {
      const option = e.target.closest ? e.target.closest('.palette-option') : null;
      if (option && option.getAttribute('aria-selected') !== 'true') highlight(Number(option.dataset.index));
    });
    listEl.addEventListener('click', (e) => {
      const option = e.target.closest ? e.target.closest('.palette-option') : null;
      if (option) activate(flat[Number(option.dataset.index)], e.ctrlKey || e.metaKey);
    });
    dialog.addEventListener('cancel', (e) => { e.preventDefault(); close(); });
    // A click on the backdrop lands on the dialog element itself.
    dialog.addEventListener('click', (e) => { if (e.target === dialog) close(); });
  }

  function open() {
    if (isOpen()) { input.focus(); input.select(); return; }
    // Another modal (confirm, prompt) is up: do not stack the palette on top of it.
    if (document.querySelector('dialog[open]')) return;
    if (!dialog) build();
    lastFocus = document.activeElement;
    // On phones the palette is opened from the drawer; the drawer should not stay open behind it.
    if (document.body.classList.contains('nav-open')) {
      const navClose = document.getElementById('navClose');
      if (navClose) navClose.click();
    }
    switching = false;
    input.disabled = false;
    input.value = '';
    activeKey = '';
    listEl.removeAttribute('aria-busy');
    requestRuns('', true);
    render();
    if (typeof dialog.showModal === 'function') dialog.showModal(); else dialog.setAttribute('open', '');
    input.focus();
    loadAreas();
  }

  function close(restoreFocus) {
    clearTimeout(debounceTimer);
    runsSeq += 1;
    if (dialog && dialog.open) {
      try { dialog.close(); } catch (e) { dialog.removeAttribute('open'); }
    }
    if (restoreFocus !== false && lastFocus && typeof lastFocus.focus === 'function' && document.contains(lastFocus)) lastFocus.focus();
    lastFocus = null;
  }

  palette.open = open;
  palette.close = close;
  palette.isOpen = isOpen;

  document.addEventListener('keydown', (e) => {
    if (isOpenShortcut(e)) {
      e.preventDefault();
      if (isOpen()) close(); else open();
      return;
    }
    if (isSlashShortcut(e) && !document.querySelector('dialog[open]')) {
      // A page with tabs has a search field per tab: the visible one takes the focus.
      const field = Array.from(document.querySelectorAll('[data-slash-focus]')).find((el) => !el.disabled && el.getClientRects().length);
      if (field) {
        e.preventDefault();
        field.focus();
        if (typeof field.select === 'function') field.select();
      }
    }
  });

  document.addEventListener('DOMContentLoaded', () => {
    const trigger = document.getElementById('paletteOpen');
    if (!trigger) return;
    trigger.setAttribute('aria-haspopup', 'dialog');
    trigger.title = LL.t('palette.trigger_title');
    trigger.addEventListener('click', open);
    const platform = String((navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || '');
    if (/mac|iphone|ipad/i.test(platform)) {
      trigger.setAttribute('aria-keyshortcuts', 'Meta+K');
      const hint = trigger.querySelector('.kbd-hint');
      if (hint) hint.textContent = '⌘K';
    }
  });
})();
