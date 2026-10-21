// SPDX-License-Identifier: Apache-2.0
// UI messages and locale-aware formatting; loaded first, before common.js (plans/i18n-plan.md, section 2.3).
// The page renders window.LoadLens.i18n = {locale, intl, messages} inline (loadlens_app/i18n.py, i18n_script)
// with the namespaces it declares; en messages already fall back to ru key by key.
//   LL.t(key, params)       plain text for textContent: {name} placeholders replaced by params
//   LL.tn(key, n, params)   plural form by Intl.PluralRules: key.one/.few/.many (ru), key.one/.other (en); {n} = n
//   LL.tHtml(key, params)   HTML: the message and params escaped, except params wrapped in LL.html(raw)
//   LL.fmt.number(x, opts), LL.fmt.dateTime(d, opts), LL.fmt.date(d, opts): Intl formats of LL.intl
// A missing key returns the key itself and warns once; LoadLens.i18n.strict = true throws instead (tests).
(function () {
  'use strict';

  const LL = window.LoadLens = window.LoadLens || {};
  const INTL = { ru: 'ru-RU', en: 'en-GB' };
  const hasOwn = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);
  const initial = LL.i18n && typeof LL.i18n === 'object' ? LL.i18n : null;
  const locale = initial && hasOwn(INTL, initial.locale) ? initial.locale : 'ru';
  if (!initial) LL.i18n = { locale, intl: INTL[locale], messages: {} };
  LL.locale = locale;
  LL.intl = initial && typeof initial.intl === 'string' && initial.intl ? initial.intl : INTL[locale];

  const PLACEHOLDER = /\{([A-Za-z_][A-Za-z0-9_]*)\}/g;
  const warned = new Set();

  function store() {
    const data = LL.i18n;
    return data && typeof data === 'object' ? data : {};
  }

  function lookup(key) {
    const messages = store().messages;
    if (!messages || typeof messages !== 'object' || !hasOwn(messages, key)) return null;
    const text = messages[key];
    return typeof text === 'string' ? text : null;
  }

  function missing(key, result) {
    const message = `i18n: no message "${key}" for locale ${locale}`;
    if (store().strict) throw new Error(message);
    if (!warned.has(key)) {
      warned.add(key);
      try { console.warn(message); } catch (e) { /* no console */ }
    }
    return result;
  }

  function escapeHtml(value) {
    return String(value === undefined || value === null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // Trusted markup for LL.tHtml parameters (links, <code>): passed into the message without escaping.
  function SafeHtml(raw) { this.html = String(raw === undefined || raw === null ? '' : raw); }
  SafeHtml.prototype.toString = function () { return this.html; };

  // undefined leaves the placeholder as it is (same as a missing parameter on the server), null is empty.
  function interpolate(text, params, convert) {
    if (!params || typeof params !== 'object' || text.indexOf('{') < 0) return text;
    return text.replace(PLACEHOLDER, (match, name) => (hasOwn(params, name) && params[name] !== undefined ? convert(params[name]) : match));
  }

  const plainParam = (value) => (value === null ? '' : String(value));
  const htmlParam = (value) => (value instanceof SafeHtml ? value.html : escapeHtml(plainParam(value)));

  // {n} of a plural message: integers as digits, fractions with the decimal separator of the locale.
  function numberText(n) {
    if (typeof n !== 'number' || !Number.isFinite(n) || Number.isInteger(n)) return String(n);
    return locale === 'ru' ? String(n).replace('.', ',') : String(n);
  }

  let pluralRules = null;
  // CLDR category of n; ru fractions (Intl "other") take the few form, as Russian grammar wants for 1.5.
  function pluralCategory(n) {
    const number = Math.abs(Number(n));
    if (!Number.isFinite(number)) return locale === 'en' ? 'other' : 'many';
    let category = 'other';
    try {
      if (!pluralRules) pluralRules = new Intl.PluralRules(locale);
      category = pluralRules.select(number);
    } catch (e) { /* Intl without plural rules: the other/many forms below */ }
    if (locale === 'ru' && category === 'other') category = 'few';
    return category;
  }

  LL.t = function t(key, params) {
    const name = String(key);
    const text = lookup(name);
    if (text === null) return missing(name, name);
    return interpolate(text, params, plainParam);
  };

  function pluralText(key, n) {
    const base = String(key);
    const category = pluralCategory(n);
    const tried = [category, 'other', 'many'].filter((suffix, i, list) => list.indexOf(suffix) === i);
    for (const suffix of tried) {
      const text = lookup(`${base}.${suffix}`);
      if (text !== null) return text;
    }
    return missing(`${base}.${category}`, null);
  }

  LL.tn = function tn(key, n, params) {
    const text = pluralText(key, n);
    const all = Object.assign({ n: numberText(n) }, params || {});
    return text === null ? String(key) : interpolate(text, all, plainParam);
  };

  LL.html = (raw) => (raw instanceof SafeHtml ? raw : new SafeHtml(raw));

  LL.tHtml = function tHtml(key, params) {
    const name = String(key);
    const text = lookup(name);
    if (text === null) return escapeHtml(missing(name, name));
    return interpolate(escapeHtml(text), params, htmlParam);
  };

  LL.pluralCategory = pluralCategory;

  // ---- numbers and dates ---------------------------------------------------------------------------
  const formats = new Map();
  function formatter(kind, opts) {
    const cacheKey = `${kind}|${opts ? JSON.stringify(opts) : ''}`;
    let f = formats.get(cacheKey);
    if (!f) {
      f = kind === 'number' ? new Intl.NumberFormat(LL.intl, opts || undefined) : new Intl.DateTimeFormat(LL.intl, opts);
      formats.set(cacheKey, f);
    }
    return f;
  }

  function toDate(value) {
    if (value === null || value === undefined || value === '') return null;
    const date = Object.prototype.toString.call(value) === '[object Date]' ? value : new Date(typeof value === 'number' ? value : String(value));
    return Number.isNaN(date.getTime()) ? null : date;
  }

  const DATE_TIME = { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' };
  const DATE_ONLY = { year: 'numeric', month: '2-digit', day: '2-digit' };

  LL.fmt = LL.fmt || {};
  // Number in the UI locale (ru: «1 234,5», en: «1,234.5»); '—' for no number.
  LL.fmt.number = function number(x, opts) {
    const value = typeof x === 'number' ? x : (x === null || x === undefined || x === '' ? NaN : Number(x));
    if (!Number.isFinite(value)) return '—';
    return formatter('number', opts).format(value);
  };
  // Date and time (default dd.mm.yyyy, hh:mm in ru; dd/mm/yyyy, hh:mm in en); '—' for an invalid date.
  LL.fmt.dateTime = function dateTime(d, opts) {
    const date = toDate(d);
    return date ? formatter('dateTime', opts || DATE_TIME).format(date) : '—';
  };
  LL.fmt.date = function dateOnly(d, opts) {
    const date = toDate(d);
    return date ? formatter('dateTime', opts || DATE_ONLY).format(date) : '—';
  };
})();
