// SPDX-License-Identifier: Apache-2.0
// Account page: password change and personal API tokens.
// UI text comes from the `account` namespace (templates/account.html declares it) and common.role.* / common.action.*.
(function () {
  const LL = window.LoadLens;
  const ui = LL.ui;
  const esc = ui.escapeHtml;
  const $ = (id) => document.getElementById(id);

  async function api(method, url, body) {
    const init = { method };
    if (body !== undefined) { init.headers = { 'Content-Type': 'application/json' }; init.body = JSON.stringify(body); }
    const resp = await fetch(url, init);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    return data;
  }

  // ---- password ------------------------------------------------------------------
  // The check before the request: '' when the new password may be sent, else the message to show.
  function passwordProblem(next, repeat, minLength) {
    if (next.length < minLength) return LL.t('account.password.too_short', { n: minLength });
    if (next !== repeat) return LL.t('account.password.mismatch');
    return '';
  }

  function initPasswordForm() {
    const form = $('passwordForm');
    if (!form) return;
    const minLength = Number(form.dataset.minLength) || 10;
    const forced = form.dataset.forced === '1';
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const error = $('passwordError');
      error.textContent = '';
      const current = $('currentPassword').value;
      const next = $('newPassword').value;
      const problem = passwordProblem(next, $('confirmPassword').value, minLength);
      if (problem) { error.textContent = problem; return; }
      const button = form.querySelector('button[type="submit"]');
      button.disabled = true;
      try {
        await api('POST', '/auth/password', { current_password: current, new_password: next });
        form.reset();
        ui.toast(LL.t('account.password.changed'), { tone: 'ok' });
        if (forced) window.location.assign('/');
      } catch (e) {
        error.textContent = e.message;
      } finally {
        button.disabled = false;
      }
    });
  }

  // ---- API tokens ------------------------------------------------------------------
  const ROLE_KEYS = ['viewer', 'engineer', 'admin'];
  const roleLabel = (key) => (ROLE_KEYS.includes(key) ? LL.t(`common.role.${key}`) : key);
  const fmt = (value) => (value ? LL.formatDateTime(value) : '—');

  function tokenRow(token) {
    const expired = token.expires_at && new Date(token.expires_at) <= new Date();
    const [tone, label] = token.revoked_at
      ? ['na', LL.tHtml('account.tokens.status.revoked')]
      : (expired ? ['warn', LL.tHtml('account.tokens.status.expired')] : ['ok', LL.tHtml('account.tokens.status.active')]);
    const action = token.active ? `<button type="button" class="btn btn-sm btn-danger" data-revoke="${token.id}" data-name="${esc(token.name)}">${LL.tHtml('account.tokens.revoke')}</button>` : '';
    return `<tr>
      <td>${esc(token.name)}</td>
      <td><code>${esc(token.prefix)}…</code></td>
      <td>${esc(roleLabel(token.role))}</td>
      <td class="nowrap m">${esc(fmt(token.created_at))}</td>
      <td class="nowrap m">${esc(fmt(token.expires_at))}</td>
      <td class="nowrap m">${esc(fmt(token.last_used_at))}</td>
      <td><span class="pill ${tone}">${label}</span></td>
      <td class="actions">${action}</td>
    </tr>`;
  }

  async function loadTokens() {
    const body = $('tokensBody');
    // Skeleton rows only before the first load; a reload after an action keeps the old rows.
    if (!body.querySelector('tr:not(.empty-row):not(.skeleton-row)')) body.innerHTML = ui.skeletonRows(8, 2);
    try {
      const data = await api('GET', '/auth/tokens');
      body.innerHTML = data.tokens.length
        ? data.tokens.map(tokenRow).join('')
        : ui.emptyRow(8, LL.t('account.tokens.empty'));
    } catch (e) {
      body.innerHTML = ui.emptyRow(8, LL.t('account.tokens.load_failed', { error: e.message }));
    }
  }

  function showNewToken(token) {
    return ui.dialog({
      title: LL.t('account.tokens.created.title'),
      bodyHtml: `<p>${LL.tHtml('account.tokens.created.text')}</p><code class="token-secret" id="newTokenValue">${esc(token)}</code>`,
      actions: [
        {
          label: LL.t('account.tokens.created.copy'),
          primary: true,
          onClick: async (close, body) => {
            try {
              await navigator.clipboard.writeText(token);
              ui.toast(LL.t('account.tokens.created.copied'), { tone: 'ok' });
            } catch (e) {
              const range = document.createRange();
              range.selectNodeContents(body.querySelector('#newTokenValue'));
              const selection = window.getSelection();
              selection.removeAllRanges();
              selection.addRange(range);
              ui.toast(LL.t('account.tokens.created.selected'), { tone: 'info' });
            }
          }
        },
        { label: LL.t('common.action.close') }
      ]
    });
  }

  function initTokens() {
    const form = $('tokenForm');
    if (!form) return;
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const button = form.querySelector('button[type="submit"]');
      button.disabled = true;
      try {
        const data = await api('POST', '/auth/tokens', {
          name: $('tokenName').value.trim(),
          role: $('tokenRole').value,
          expires_in_days: Number($('tokenDays').value)
        });
        form.querySelector('#tokenName').value = '';
        await loadTokens();
        await showNewToken(data.token);
      } catch (e) {
        ui.toast(e.message, { tone: 'error' });
      } finally {
        button.disabled = false;
      }
    });
    $('tokensBody').addEventListener('click', async (event) => {
      const btn = event.target.closest('[data-revoke]');
      if (!btn) return;
      const ok = await ui.confirm({
        title: LL.t('account.tokens.revoke_confirm.title'),
        message: LL.t('account.tokens.revoke_confirm.message', { name: btn.dataset.name }),
        confirmText: LL.t('account.tokens.revoke'),
        danger: true
      });
      if (!ok) return;
      try {
        await api('DELETE', `/auth/tokens/${encodeURIComponent(btn.dataset.revoke)}`);
        await loadTokens();
      } catch (e) {
        ui.toast(e.message, { tone: 'error' });
      }
    });
    loadTokens();
  }

  document.addEventListener('DOMContentLoaded', () => {
    initPasswordForm();
    initTokens();
  });

  // Builders for tests (tests/test_i18n_account.py).
  LL.account = { _internals: { passwordProblem, roleLabel, tokenRow, showNewToken, loadTokens } };
})();
