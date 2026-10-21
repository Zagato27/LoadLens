// SPDX-License-Identifier: Apache-2.0
// Settings → «Пользователи»: accounts, roles, API tokens of everybody and the audit log.
// UI text comes from the `users` namespace (the settings page declares it) and common.role.* / common.action.*.
(function () {
  const LL = window.LoadLens;
  const ui = LL.ui;
  const esc = ui.escapeHtml;
  const $ = (id) => document.getElementById(id);
  const fmt = (value) => (value ? LL.formatDateTime(value) : '—');
  let state = { users: [], roles: [], currentUserId: null };
  // Skeleton rows only while a table has no data rows yet; a reload after an action keeps the old rows.
  function showLoading(body, cols) {
    if (!body.querySelector('tr:not(.empty-row):not(.skeleton-row)')) body.innerHTML = ui.skeletonRows(cols, 3);
  }

  // Audit action codes -> labels in the UI language; built on first use, when the dictionary is in place.
  let actionLabels = null;
  function actionLabel(action) {
    if (!actionLabels) {
      actionLabels = {
        login: LL.t('users.audit_action.login'), 'login.failed': LL.t('users.audit_action.login_failed'), logout: LL.t('users.audit_action.logout'),
        'user.create': LL.t('users.audit_action.user_create'), 'user.update': LL.t('users.audit_action.user_update'),
        'user.delete': LL.t('users.audit_action.user_delete'), 'user.password_reset': LL.t('users.audit_action.user_password_reset'),
        'user.unlock': LL.t('users.audit_action.user_unlock'), 'user.provision': LL.t('users.audit_action.user_provision'),
        'password.change': LL.t('users.audit_action.password_change'), 'token.create': LL.t('users.audit_action.token_create'),
        'token.revoke': LL.t('users.audit_action.token_revoke'), 'config.update': LL.t('users.audit_action.config_update'),
        'config.test_connection': LL.t('users.audit_action.config_test_connection'),
        'config.query_preview': LL.t('users.audit_action.config_query_preview'), 'report.create': LL.t('users.audit_action.report_create'),
        'run.delete': LL.t('users.audit_action.run_delete'), 'run.rename': LL.t('users.audit_action.run_rename'),
        'confluence.publish': LL.t('users.audit_action.confluence_publish'),
        'engineer_summary.save': LL.t('users.audit_action.engineer_summary_save'),
        'forecast.explain': LL.t('users.audit_action.forecast_explain'), 'forecast.publish': LL.t('users.audit_action.forecast_publish'),
        'demo.seed': LL.t('users.audit_action.demo_seed'), 'prompts.save': LL.t('users.audit_action.prompts_save'),
        'service.delete': LL.t('users.audit_action.service_delete'), 'service.meta_update': LL.t('users.audit_action.service_meta_update'),
        'service.meta_delete': LL.t('users.audit_action.service_meta_delete'), 'project.create': LL.t('users.audit_action.project_create'),
        'project.update': LL.t('users.audit_action.project_update'), 'project.delete': LL.t('users.audit_action.project_delete'),
        'project.move_service': LL.t('users.audit_action.project_move_service'),
        'project.override_delete': LL.t('users.audit_action.project_override_delete')
      };
    }
    return actionLabels[action] || action;
  }

  async function api(method, url, body) {
    const init = { method };
    if (body !== undefined) { init.headers = { 'Content-Type': 'application/json' }; init.body = JSON.stringify(body); }
    const resp = await fetch(url, init);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    return data;
  }
  // Role names of the UI language; a role the dictionary does not know keeps the label the server sent.
  const ROLE_KEYS = ['viewer', 'engineer', 'admin'];
  const roleLabel = (key) => (ROLE_KEYS.includes(key) ? LL.t(`common.role.${key}`) : ((state.roles.find((r) => r.key === key) || {}).label || key));
  const roleOptions = (selected) => state.roles.map((r) => `<option value="${esc(r.key)}"${r.key === selected ? ' selected' : ''}>${esc(roleLabel(r.key))}</option>`).join('');

  function randomPassword() {
    const alphabet = 'abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789';
    const bytes = new Uint32Array(16);
    crypto.getRandomValues(bytes);
    return Array.from(bytes, (n) => alphabet[n % alphabet.length]).join('');
  }

  // ---- users ----------------------------------------------------------------------
  function statusPill(user) {
    if (!user.is_active) return `<span class="pill na">${LL.tHtml('users.status.disabled')}</span>`;
    if (user.locked) return `<span class="pill warn">${LL.tHtml('users.status.locked')}</span>`;
    if (user.must_change_password) return `<span class="pill warn">${LL.tHtml('users.status.must_change')}</span>`;
    return `<span class="pill ok">${LL.tHtml('users.status.active')}</span>`;
  }
  function userRow(user) {
    const self = user.id === state.currentUserId;
    const local = user.provider === 'local';
    const act = (action, label, extra) => `<button type="button" class="btn btn-sm${extra || ''}" data-action="${action}" data-id="${user.id}">${label}</button>`;
    const actions = [
      act('edit', LL.tHtml('users.action.edit')),
      local ? act('password', LL.tHtml('users.action.password')) : '',
      user.locked ? act('unlock', LL.tHtml('users.action.unlock')) : '',
      self ? '' : act('toggle', LL.tHtml(user.is_active ? 'users.action.disable' : 'users.action.enable')),
      self ? '' : act('delete', LL.tHtml('users.action.delete'), ' btn-danger')
    ].join(' ');
    return `<tr>
      <td><strong>${esc(user.username)}</strong>${self ? ` <span class="pill ok">${LL.tHtml('users.self')}</span>` : ''}${user.email ? `<div class="dim">${esc(user.email)}</div>` : ''}</td>
      <td>${esc(user.display_name || '—')}</td>
      <td>${esc(roleLabel(user.role))}</td>
      <td>${esc(local ? LL.t('users.provider_local') : user.provider)}</td>
      <td>${statusPill(user)}</td>
      <td class="nowrap m">${esc(fmt(user.last_login_at))}</td>
      <td class="actions">${actions}</td>
    </tr>`;
  }
  async function loadUsers() {
    const body = $('usersBody');
    showLoading(body, 7);
    try {
      const data = await api('GET', '/auth/users');
      state = { users: data.users, roles: data.roles, currentUserId: data.current_user_id };
      body.innerHTML = state.users.length ? state.users.map(userRow).join('') : ui.emptyRow(7, LL.t('users.empty'));
      $('usersStatus').textContent = '';
    } catch (e) {
      body.innerHTML = ui.emptyRow(7, LL.t('users.load_failed', { error: e.message }));
    }
  }

  function createDialog() {
    const generated = randomPassword();
    const bodyHtml = [
      `<label>${LL.tHtml('users.form.username')}<input id="ufUsername" type="text" maxlength="64" autocomplete="off" spellcheck="false" /></label>`,
      `<label>${LL.tHtml('users.form.name')}<input id="ufName" type="text" maxlength="120" autocomplete="off" /></label>`,
      `<label>Email <span class="optional">${LL.tHtml('users.form.optional')}</span><input id="ufEmail" type="text" maxlength="254" autocomplete="off" /></label>`,
      `<label>${LL.tHtml('users.form.role')}<select id="ufRole">${roleOptions('viewer')}</select></label>`,
      `<label>${LL.tHtml('users.form.temp_password')}<input id="ufPassword" type="text" autocomplete="off" spellcheck="false" value="${esc(generated)}" /></label>`,
      `<label class="checkbox-row"><input id="ufMustChange" type="checkbox" checked /> ${LL.tHtml('users.form.must_change_first')}</label>`,
      `<p class="dim">${LL.tHtml('users.form.handover')}</p>`,
      '<div class="dialog-error" id="ufError"></div>'
    ].join('');
    return ui.dialog({
      title: LL.t('users.create.title'),
      bodyHtml,
      actions: [{ label: LL.t('common.action.cancel') }, {
        label: LL.t('users.create.submit'),
        primary: true,
        onClick: async (close, body) => {
          const value = (selector) => body.querySelector(selector).value.trim();
          try {
            await api('POST', '/auth/users', {
              username: value('#ufUsername'),
              display_name: value('#ufName'),
              email: value('#ufEmail'),
              role: body.querySelector('#ufRole').value,
              password: body.querySelector('#ufPassword').value,
              must_change_password: body.querySelector('#ufMustChange').checked
            });
            close(true);
          } catch (e) {
            body.querySelector('#ufError').textContent = e.message;
          }
        }
      }]
    });
  }

  function editDialog(user) {
    const bodyHtml = [
      `<p class="dim">${LL.tHtml('users.edit.username', { username: user.username })}</p>`,
      `<label>${LL.tHtml('users.form.name')}<input id="ufName" type="text" maxlength="120" value="${esc(user.display_name)}" /></label>`,
      `<label>Email<input id="ufEmail" type="text" maxlength="254" value="${esc(user.email)}" /></label>`,
      `<label>${LL.tHtml('users.form.role')}<select id="ufRole">${roleOptions(user.role)}</select></label>`,
      `<p class="dim">${LL.tHtml('users.edit.role_note')}</p>`,
      '<div class="dialog-error" id="ufError"></div>'
    ].join('');
    return ui.dialog({
      title: LL.t('users.edit.title'),
      bodyHtml,
      actions: [{ label: LL.t('common.action.cancel') }, {
        label: LL.t('common.action.save'),
        primary: true,
        onClick: async (close, body) => {
          try {
            await api('PATCH', `/auth/users/${user.id}`, {
              display_name: body.querySelector('#ufName').value.trim(),
              email: body.querySelector('#ufEmail').value.trim(),
              role: body.querySelector('#ufRole').value
            });
            close(true);
          } catch (e) {
            body.querySelector('#ufError').textContent = e.message;
          }
        }
      }]
    });
  }

  function passwordDialog(user) {
    const bodyHtml = [
      `<p>${LL.tHtml('users.password.intro', { username: LL.html(`<strong>${esc(user.username)}</strong>`) })}</p>`,
      `<label>${LL.tHtml('users.password.field')}<input id="ufPassword" type="text" autocomplete="off" spellcheck="false" value="${esc(randomPassword())}" /></label>`,
      `<label class="checkbox-row"><input id="ufMustChange" type="checkbox" checked /> ${LL.tHtml('users.password.must_change_next')}</label>`,
      '<div class="dialog-error" id="ufError"></div>'
    ].join('');
    return ui.dialog({
      title: LL.t('users.password.title'),
      bodyHtml,
      actions: [{ label: LL.t('common.action.cancel') }, {
        label: LL.t('users.password.submit'),
        primary: true,
        onClick: async (close, body) => {
          try {
            await api('POST', `/auth/users/${user.id}/password`, {
              new_password: body.querySelector('#ufPassword').value,
              must_change: body.querySelector('#ufMustChange').checked
            });
            close(true);
          } catch (e) {
            body.querySelector('#ufError').textContent = e.message;
          }
        }
      }]
    });
  }

  const ACTIONS = {
    edit: async (user) => { if (await editDialog(user)) ui.toast(LL.t('users.edit.done'), { tone: 'ok' }); },
    password: async (user) => { if (await passwordDialog(user)) ui.toast(LL.t('users.password.done'), { tone: 'ok' }); },
    unlock: async (user) => { await api('POST', `/auth/users/${user.id}/unlock`); ui.toast(LL.t('users.unlock.done'), { tone: 'ok' }); },
    toggle: async (user) => {
      if (user.is_active) {
        const ok = await ui.confirm({ title: LL.t('users.disable.title'), message: LL.t('users.disable.message', { username: user.username }), confirmText: LL.t('users.action.disable'), danger: true });
        if (!ok) return;
      }
      await api('PATCH', `/auth/users/${user.id}`, { is_active: !user.is_active });
    },
    delete: async (user) => {
      const ok = await ui.confirm({ title: LL.t('users.delete.title'), message: LL.t('users.delete.message', { username: user.username }), confirmText: LL.t('common.action.delete'), danger: true });
      if (!ok) return;
      await api('DELETE', `/auth/users/${user.id}`);
    }
  };

  async function onUsersClick(event) {
    const btn = event.target.closest('[data-action]');
    if (!btn) return;
    const user = state.users.find((u) => String(u.id) === btn.dataset.id);
    const handler = ACTIONS[btn.dataset.action];
    if (!user || !handler) return;
    try {
      await handler(user);
    } catch (e) {
      ui.toast(e.message, { tone: 'error' });
    }
    await loadUsers();
  }

  // ---- tokens of all users ------------------------------------------------------------
  function tokenRow(token) {
    const expired = token.expires_at && new Date(token.expires_at) <= new Date();
    const [tone, label] = token.revoked_at
      ? ['na', LL.tHtml('users.tokens.status.revoked')]
      : (expired ? ['warn', LL.tHtml('users.tokens.status.expired')] : ['ok', LL.tHtml('users.tokens.status.active')]);
    const action = token.active ? `<button type="button" class="btn btn-sm btn-danger" data-revoke="${token.id}" data-name="${esc(token.name)}">${LL.tHtml('users.tokens.revoke')}</button>` : '';
    return `<tr><td>${esc(token.name)}</td><td>${esc(token.owner)}</td><td><code>${esc(token.prefix)}…</code></td><td>${esc(roleLabel(token.role))}</td>
      <td class="nowrap m">${esc(fmt(token.expires_at))}</td><td class="nowrap m">${esc(fmt(token.last_used_at))}</td><td><span class="pill ${tone}">${label}</span></td><td class="actions">${action}</td></tr>`;
  }
  async function loadAllTokens() {
    const body = $('allTokensBody');
    showLoading(body, 8);
    try {
      const data = await api('GET', '/auth/tokens?all=1');
      body.innerHTML = data.tokens.length ? data.tokens.map(tokenRow).join('') : ui.emptyRow(8, LL.t('users.tokens.empty'));
    } catch (e) {
      body.innerHTML = ui.emptyRow(8, LL.t('users.tokens.load_failed', { error: e.message }));
    }
  }
  async function onTokensClick(event) {
    const btn = event.target.closest('[data-revoke]');
    if (!btn) return;
    const ok = await ui.confirm({ title: LL.t('users.tokens.revoke_confirm.title'), message: LL.t('users.tokens.revoke_confirm.message', { name: btn.dataset.name }), confirmText: LL.t('users.tokens.revoke'), danger: true });
    if (!ok) return;
    try {
      await api('DELETE', `/auth/tokens/${encodeURIComponent(btn.dataset.revoke)}`);
    } catch (e) {
      ui.toast(e.message, { tone: 'error' });
    }
    await loadAllTokens();
  }

  // ---- audit -------------------------------------------------------------------------
  function auditRow(event) {
    const details = Object.entries(event.details || {}).map(([k, v]) => `${k}: ${v}`).join(', ');
    const label = actionLabel(event.action);
    const failed = event.success ? '' : ` <span class="pill fail">${LL.tHtml('users.audit.failed')}</span>`;
    return `<tr><td class="nowrap m">${esc(LL.formatDateTime(event.created_at))}</td><td>${esc(event.actor || '—')}${event.via && event.via !== 'session' ? ` <span class="dim">(${esc(event.via)})</span>` : ''}</td>
      <td>${esc(label)}${failed}</td><td>${esc(event.target || '')}</td><td class="dim">${esc(details)}</td><td class="nowrap dim m">${esc(event.ip || '')}</td></tr>`;
  }
  async function loadAudit() {
    const body = $('auditBody');
    $('auditStatus').textContent = LL.t('users.audit.loading');
    showLoading(body, 6);
    try {
      const data = await api('GET', '/auth/audit?limit=100');
      body.innerHTML = data.events.length ? data.events.map(auditRow).join('') : ui.emptyRow(6, LL.t('users.audit.empty'));
      $('auditStatus').textContent = LL.t('users.audit.latest', { n: data.events.length });
    } catch (e) {
      body.innerHTML = '';
      $('auditStatus').textContent = LL.t('users.audit.error', { error: e.message });
    }
  }

  document.addEventListener('DOMContentLoaded', async () => {
    if (!$('usersBody')) return;
    $('usersBody').addEventListener('click', onUsersClick);
    $('userCreateBtn').addEventListener('click', async () => {
      if (await createDialog()) { ui.toast(LL.t('users.create.done'), { tone: 'ok' }); await loadUsers(); }
    });
    $('allTokensBody').addEventListener('click', onTokensClick);
    $('tokensBox').addEventListener('toggle', () => { if ($('tokensBox').open) loadAllTokens(); });
    $('auditBox').addEventListener('toggle', () => { if ($('auditBox').open) loadAudit(); });
    $('auditRefreshBtn').addEventListener('click', loadAudit);
    await loadUsers();
  });

  // Builders for tests (tests/test_i18n_account.py); setState replaces the loaded users/roles.
  LL.users = {
    _internals: {
      statusPill, userRow, tokenRow, auditRow, actionLabel, roleLabel, roleOptions, createDialog, editDialog, passwordDialog,
      ACTIONS, loadUsers, loadAllTokens, loadAudit, onTokensClick, setState: (next) => { state = next; }
    }
  };
})();
