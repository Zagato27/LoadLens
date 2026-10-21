// SPDX-License-Identifier: Apache-2.0
// Interface language switch (plans/i18n-plan.md, section 5.2). The server reads the cookie loadlens.lang (only ru and
// en are accepted; Accept-Language is never read) and renders the page in that language. A copy goes to
// localStorage like the theme, so other tabs learn about the change (ui.js asks them to reload).
// Loaded by base.html (the "View" menu of the top bar and the command palette call LL.ui.locale.set) and by the standalone
// pages (login, recovery, storage outage), where links with data-set-locale switch the language.
(function () {
  const LL = window.LoadLens = window.LoadLens || {};
  const ui = LL.ui = LL.ui || {};
  const COOKIE = 'loadlens.lang';
  const STORAGE_KEY = 'loadlens.lang';
  const SUPPORTED = ['ru', 'en'];
  const MAX_AGE = 31536000; // one year, in seconds
  let switching = null;

  // 'ru' / 'en' for a supported code ('EN', 'en-GB' and 'en_US' included), else null.
  function normalize(value) {
    const match = /^([a-z]{2})(?:[-_][a-z0-9]+)*$/i.exec(String(value === undefined || value === null ? '' : value).trim());
    const code = match ? match[1].toLowerCase() : '';
    return SUPPORTED.indexOf(code) >= 0 ? code : null;
  }
  // Language the page was rendered in (i18n.js sets LL.locale; standalone pages only have <html lang>).
  function current() {
    const root = document.documentElement;
    return normalize(typeof LL.locale === 'string' ? LL.locale : '') || normalize(root ? root.getAttribute('lang') : '') || 'ru';
  }
  function cookieText(code) {
    const secure = window.location && window.location.protocol === 'https:' ? '; Secure' : '';
    return `${COOKIE}=${code}; Path=/; Max-Age=${MAX_AGE}; SameSite=Lax${secure}`;
  }
  // Writes the cookie and the localStorage copy, then reloads the page (options.reload === false skips it).
  function set(code, options) {
    const lang = normalize(code);
    if (!lang) return false;
    switching = lang;
    try { document.cookie = cookieText(lang); } catch (e) { /* cookies blocked: the page stays as it is */ }
    if (LL.prefs && typeof LL.prefs.set === 'function') LL.prefs.set('lang', lang);
    else {
      try { window.localStorage.setItem(STORAGE_KEY, lang); } catch (e) { /* storage unavailable */ }
    }
    if (!options || options.reload !== false) {
      try { window.location.reload(); } catch (e) { /* no navigation available */ }
    }
    return true;
  }

  ui.locale = {
    COOKIE,
    SUPPORTED: SUPPORTED.slice(),
    normalize,
    current,
    cookieText,
    set,
    // The code this tab is switching to (between set() and the reload), else null.
    switching() { return switching; },
  };

  // Links and buttons with data-set-locale="<code>" (the RU · EN links of the standalone pages).
  function onClick(event) {
    const start = event && event.target;
    const target = start && typeof start.closest === 'function' ? start.closest('[data-set-locale]') : null;
    if (!target) return;
    if (typeof event.preventDefault === 'function') event.preventDefault();
    const lang = normalize(target.getAttribute('data-set-locale'));
    if (lang && lang !== current()) set(lang);
  }
  try { document.addEventListener('click', onClick); } catch (e) { /* no document events */ }
})();
