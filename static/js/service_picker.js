// SPDX-License-Identifier: Apache-2.0
// Searchable service picker of the new report and launch forms (templates/index.html, launch.html).
// Options come from GET /services and are filtered locally. Recently used services (LL.prefs
// "recentServices": ids, newest first, up to five) are listed first with the meta «recent». The chosen
// id is mirrored into a hidden input (with a "change" event), so the form code reads it as before.
(function () {
  const LL = window.LoadLens = window.LoadLens || {};
  const RECENT_KEY = 'recentServices';
  const RECENT_MAX = 5;

  function normalizeServices(list) {
    const seen = new Set();
    return (Array.isArray(list) ? list : []).map((svc) => {
      if (typeof svc === 'string') return { id: svc, title: svc };
      const id = String((svc && svc.id) || '');
      const title = svc && typeof svc.title === 'string' && svc.title.trim() ? svc.title : id;
      return { id, title };
    }).filter((svc) => {
      if (!svc.id || seen.has(svc.id)) return false;
      seen.add(svc.id);
      return true;
    });
  }

  // Combobox options: recent services first (most recent on top), then the others in server order.
  // The query matches the id or the title, case-insensitively.
  function serviceOptions(services, recent, query) {
    const q = String(query || '').trim().toLowerCase();
    const byId = new Map(services.map((svc) => [svc.id, svc]));
    const matches = (svc) => !q || svc.id.toLowerCase().includes(q) || svc.title.toLowerCase().includes(q);
    const recentIds = (Array.isArray(recent) ? recent : []).map(String).filter((id, i, all) => byId.has(id) && all.indexOf(id) === i);
    const first = recentIds.map((id) => byId.get(id)).filter(matches);
    const firstIds = new Set(first.map((svc) => svc.id));
    const rest = services.filter((svc) => !firstIds.has(svc.id) && matches(svc));
    const option = (svc, isRecent) => ({
      value: svc.id,
      label: svc.title,
      meta: [isRecent ? LL.t('picker.recent') : '', svc.title !== svc.id ? svc.id : ''].filter(Boolean).join(' · ')
    });
    return first.map((svc) => option(svc, true)).concat(rest.map((svc) => option(svc, false)));
  }

  // New recent list after using ``id``: moved to the front, duplicates dropped, at most five.
  function withRecent(recent, id) {
    const value = String(id || '');
    const list = (Array.isArray(recent) ? recent : []).map(String).filter((x) => x && x !== value);
    return (value ? [value].concat(list) : list).slice(0, RECENT_MAX);
  }

  // A service whose id or title equals the typed text (case-insensitively), or null.
  function exactMatch(services, text) {
    const q = String(text || '').trim().toLowerCase();
    if (!q) return null;
    return services.find((svc) => svc.id.toLowerCase() === q) || services.find((svc) => svc.title.toLowerCase() === q) || null;
  }

  function remember(id) {
    if (!id || !LL.prefs) return;
    LL.prefs.set(RECENT_KEY, withRecent(LL.prefs.get(RECENT_KEY, []), id));
  }

  // options: {host, input (hidden), inputId, label, describedBy, selected}
  // Returns {ready: Promise<services>, services(), has(id), getValue(), setValue(id, {silent}), setInvalid(bool), focus()}.
  // ready rejects when /services fails; with no services the picker stays disabled.
  function servicePicker(options) {
    const opts = options || {};
    const hidden = opts.input;
    let services = [];
    let silent = false;
    let fromUser = true;
    const cb = LL.ui.combobox(opts.host, {
      placeholder: LL.t('picker.loading'),
      emptyText: LL.t('picker.not_found'),
      debounceMs: 80,
      fetchOptions: async (query) => {
        const item = cb.getItem();
        // Focusing a filled picker lists every service rather than only the chosen one.
        const q = item && query === item.label ? '' : query;
        return serviceOptions(services, LL.prefs ? LL.prefs.get(RECENT_KEY, []) : [], q);
      },
      onSelect: (item) => {
        const value = item ? item.value : '';
        if (item && fromUser) remember(value);
        setHidden(value);
      }
    });
    const input = cb.input;
    if (opts.inputId) input.id = opts.inputId;
    input.setAttribute('aria-label', opts.label || LL.t('picker.label'));
    if (opts.describedBy) input.setAttribute('aria-describedby', opts.describedBy);
    input.disabled = true;

    function setHidden(value) {
      if (!hidden || hidden.value === value) return;
      hidden.value = value;
      if (!silent) hidden.dispatchEvent(new Event('change', { bubbles: true }));
    }
    function setValue(id, setOpts) {
      const svc = services.find((s) => s.id === id);
      const o = setOpts || {};
      silent = !!o.silent;
      fromUser = !!o.remember;
      try {
        if (svc) cb.setValue(svc.id, svc.title, '');
        else { cb.clear(); setHidden(''); }
      } finally {
        silent = false;
        fromUser = true;
      }
      return !!svc;
    }
    // Typing drops the choice until an option is picked; an exact name is accepted when leaving the field.
    input.addEventListener('input', () => setHidden(''));
    input.addEventListener('focus', () => { try { input.select(); } catch (e) { /* not selectable */ } });
    input.addEventListener('blur', () => {
      if (hidden && hidden.value) return;
      const svc = exactMatch(services, input.value);
      if (svc) setValue(svc.id, { remember: true });
    });
    // Like the old <select>: Enter with the list closed accepts an exact name instead of submitting the form.
    input.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' || opts.host.classList.contains('open')) return;
      e.preventDefault();
      if (hidden && hidden.value) return;
      const svc = exactMatch(services, input.value);
      if (svc) setValue(svc.id, { remember: true });
    });

    const ready = (async () => {
      let payload = {};
      try {
        const resp = await fetch('/services');
        payload = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(payload.error || payload.message || `HTTP ${resp.status}`);
      } catch (e) {
        input.placeholder = LL.t('picker.load_failed');
        throw e;
      }
      services = normalizeServices(payload && payload.services);
      if (!services.length) {
        input.placeholder = LL.t('picker.empty');
        return services;
      }
      input.disabled = false;
      input.placeholder = opts.placeholder || LL.t('picker.placeholder');
      const wanted = (opts.selected && services.find((s) => s.id === opts.selected)) || (services.length === 1 ? services[0] : null);
      if (wanted) setValue(wanted.id, { silent: true });
      return services;
    })();

    return {
      ready,
      combobox: cb,
      input,
      services: () => services.slice(),
      has: (id) => services.some((s) => s.id === id),
      getValue: () => (hidden ? hidden.value : cb.getValue()),
      setValue,
      setInvalid(invalid) { input.setAttribute('aria-invalid', invalid ? 'true' : 'false'); },
      focus: () => input.focus()
    };
  }

  servicePicker.remember = remember;
  servicePicker._internals = { normalizeServices, serviceOptions, withRecent, exactMatch, RECENT_KEY, RECENT_MAX };
  LL.servicePicker = servicePicker;
})();
