// SPDX-License-Identifier: Apache-2.0
// Recovery page: check and save the storage connection while the database is down.
// Standalone on purpose: common.js expects the regular layout and a working database.
(function () {
  'use strict';

  // static/js/i18n.js is loaded first and holds the recovery.* messages of the page.
  const LL = window.LoadLens;
  const POLL_MS = 15000;
  const FIELDS = ['host', 'port', 'dbname', 'schema', 'user', 'password', 'sslmode', 'connect_timeout_sec'];
  const form = document.getElementById('storageForm');
  const result = document.getElementById('formResult');
  const checkBtn = document.getElementById('checkBtn');
  const saveBtn = document.getElementById('saveBtn');
  let formFilled = false;
  let busy = false;
  let pollTimer = null;

  function csrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute('content') || '' : '';
  }

  async function api(method, url, body) {
    const headers = { Accept: 'application/json' };
    if (method !== 'GET') {
      headers['Content-Type'] = 'application/json';
      headers['X-CSRF-Token'] = csrfToken();
    }
    const response = await fetch(url, {
      method,
      headers,
      credentials: 'same-origin',
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    let data = {};
    try { data = await response.json(); } catch (e) { data = {}; }
    // The recovery session ended (storage is back or the time ran out): go where the server says.
    if (response.status === 401 && data && data.login_url) {
      window.location.href = data.login_url;
      return null;
    }
    if (response.status === 401) {
      window.location.href = '/login';
      return null;
    }
    return { status: response.status, ok: response.ok, data: data || {} };
  }

  function setText(id, text) {
    const node = document.getElementById(id);
    if (node) node.textContent = text || '';
  }

  function showState(storage) {
    const box = document.getElementById('storageState');
    const details = document.getElementById('storageDetails');
    if (!storage) return;
    if (storage.available) {
      box.classList.remove('banner-fail');
      box.classList.add('banner-ok');
      setText('storageTitle', LL.t('recovery.state.available'));
      setText('storageHint', LL.t('recovery.state.available_hint'));
      details.hidden = true;
      return;
    }
    box.classList.remove('banner-ok');
    box.classList.add('banner-fail');
    setText('storageTitle', storage.available === false ? storage.title : LL.t('recovery.state.not_checked'));
    setText('storageHint', storage.hint || '');
    setText('storageAddress', storage.address || '');
    setText('storageMessage', storage.message || '');
    details.hidden = !(storage.address || storage.message);
  }

  function fillForm(values) {
    FIELDS.forEach((name) => {
      const input = form.elements[name];
      if (!input) return;
      const value = values[name];
      input.value = value === undefined || value === null ? '' : String(value);
    });
    if (!form.elements.sslmode.value) form.elements.sslmode.value = 'prefer';
  }

  function formValues() {
    const out = {};
    FIELDS.forEach((name) => {
      const input = form.elements[name];
      if (!input) return;
      const raw = input.value;
      out[name] = name === 'password' ? raw : raw.trim();
    });
    return out;
  }

  function showResult(kind, text) {
    result.hidden = false;
    result.className = 'recovery-result ' + kind;
    result.textContent = text;
  }

  function setBusy(value) {
    busy = value;
    checkBtn.disabled = value;
    saveBtn.disabled = value;
  }

  async function loadStatus() {
    let reply;
    try {
      reply = await api('GET', '/recovery/status');
    } catch (e) {
      showResult('fail', LL.t('recovery.error.no_server_retry'));
      return;
    }
    if (!reply) return;
    if (!reply.ok) {
      showResult('fail', reply.data.error || LL.t('recovery.error.status_failed'));
      return;
    }
    const data = reply.data;
    showState(data.storage);
    document.getElementById('dsnNote').hidden = !data.dsn_configured;
    if (!formFilled) {
      fillForm(data.form || {});
      formFilled = true;
    }
  }

  async function check() {
    if (busy) return;
    setBusy(true);
    showResult('pending', LL.t('recovery.check.pending'));
    try {
      const reply = await api('POST', '/recovery/storage/check', formValues());
      if (!reply) return;
      if (!reply.ok) {
        showResult('fail', reply.data.error || LL.t('recovery.check.failed'));
      } else {
        const message = reply.data.message || '';
        const text = reply.data.elapsed_ms
          ? LL.t('recovery.check.result_elapsed', { message, ms: reply.data.elapsed_ms })
          : message;
        showResult(reply.data.ok ? 'ok' : 'fail', text);
      }
    } catch (e) {
      showResult('fail', LL.t('recovery.error.no_server'));
    } finally {
      setBusy(false);
    }
  }

  async function save(event) {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    showResult('pending', LL.t('recovery.save.pending'));
    let redirecting = false;
    try {
      const reply = await api('POST', '/recovery/storage/save', formValues());
      if (!reply) return;
      if (!reply.ok) {
        showResult('fail', reply.data.error || LL.t('recovery.save.failed'));
        return;
      }
      const data = reply.data;
      if (data.available && data.redirect) {
        redirecting = true;
        showState({ available: true });
        showResult('ok', LL.t('recovery.save.done'));
        setTimeout(() => { window.location.href = data.redirect; }, 1200);
        return;
      }
      showState(data.storage);
      const reason = data.storage && data.storage.title
        ? data.storage.title.toLowerCase()
        : LL.t('recovery.save.no_answer');
      showResult('warn', LL.t('recovery.save.still_down', { reason }));
    } catch (e) {
      showResult('fail', LL.t('recovery.error.no_server'));
    } finally {
      if (!redirecting) setBusy(false);
    }
  }

  function schedulePoll() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => {
      if (!busy && !document.hidden) loadStatus();
    }, POLL_MS);
  }

  checkBtn.addEventListener('click', check);
  form.addEventListener('submit', save);
  loadStatus();
  schedulePoll();
})();
