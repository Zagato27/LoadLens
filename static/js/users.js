// Settings → «Пользователи»: accounts, roles, API tokens of everybody and the audit log.
(function () {
  const LL = window.LoadLens;
  const ui = LL.ui;
  const esc = ui.escapeHtml;
  const $ = (id) => document.getElementById(id);
  const fmt = (value) => (value ? LL.formatDateTime(value) : '—');
  let state = { users: [], roles: [], currentUserId: null };

  const ACTION_LABELS = {
    login: 'Вход', 'login.failed': 'Неудачный вход', logout: 'Выход',
    'user.create': 'Пользователь создан', 'user.update': 'Пользователь изменён', 'user.delete': 'Пользователь удалён',
    'user.password_reset': 'Пароль сброшен', 'user.unlock': 'Пользователь разблокирован', 'user.provision': 'Пользователь создан провайдером',
    'password.change': 'Смена пароля', 'token.create': 'Токен создан', 'token.revoke': 'Токен отозван',
    'config.update': 'Изменение настроек', 'config.test_connection': 'Проверка соединения', 'config.query_preview': 'Предпросмотр запроса',
    'report.create': 'Запуск отчёта', 'run.delete': 'Удаление прогона', 'run.rename': 'Переименование прогона',
    'confluence.publish': 'Публикация в Confluence', 'engineer_summary.save': 'Заметка инженера',
    'forecast.explain': 'Объяснение прогноза', 'forecast.publish': 'Публикация прогноза', 'demo.seed': 'Демо-данные',
    'prompts.save': 'Сохранение промпта', 'service.delete': 'Удаление сервиса', 'service.meta_update': 'Изменение сервиса', 'service.meta_delete': 'Сброс сервиса',
    'project.create': 'Проект создан', 'project.update': 'Проект изменён', 'project.delete': 'Проект удалён',
    'project.move_service': 'Сервис перенесён', 'project.override_delete': 'Сброс переопределения'
  };

  async function api(method, url, body) {
    const init = { method };
    if (body !== undefined) { init.headers = { 'Content-Type': 'application/json' }; init.body = JSON.stringify(body); }
    const resp = await fetch(url, init);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    return data;
  }
  const roleLabel = (key) => (state.roles.find((r) => r.key === key) || {}).label || key;
  const roleOptions = (selected) => state.roles.map((r) => `<option value="${esc(r.key)}"${r.key === selected ? ' selected' : ''}>${esc(r.label)}</option>`).join('');

  function randomPassword() {
    const alphabet = 'abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789';
    const bytes = new Uint32Array(16);
    crypto.getRandomValues(bytes);
    return Array.from(bytes, (n) => alphabet[n % alphabet.length]).join('');
  }

  // ---- users ----------------------------------------------------------------------
  function statusPill(user) {
    if (!user.is_active) return '<span class="pill na">отключён</span>';
    if (user.locked) return '<span class="pill warn">заблокирован</span>';
    if (user.must_change_password) return '<span class="pill warn">ждёт смены пароля</span>';
    return '<span class="pill ok">активен</span>';
  }
  function userRow(user) {
    const self = user.id === state.currentUserId;
    const local = user.provider === 'local';
    const act = (action, label, extra) => `<button type="button" class="btn btn-sm${extra || ''}" data-action="${action}" data-id="${user.id}">${label}</button>`;
    const actions = [
      act('edit', 'Изменить'),
      local ? act('password', 'Пароль…') : '',
      user.locked ? act('unlock', 'Разблокировать') : '',
      self ? '' : act('toggle', user.is_active ? 'Отключить' : 'Включить'),
      self ? '' : act('delete', 'Удалить…', ' btn-danger')
    ].join(' ');
    return `<tr>
      <td><strong>${esc(user.username)}</strong>${self ? ' <span class="pill ok">это вы</span>' : ''}${user.email ? `<div class="dim">${esc(user.email)}</div>` : ''}</td>
      <td>${esc(user.display_name || '—')}</td>
      <td>${esc(roleLabel(user.role))}</td>
      <td>${esc(local ? 'локальная' : user.provider)}</td>
      <td>${statusPill(user)}</td>
      <td class="nowrap">${esc(fmt(user.last_login_at))}</td>
      <td class="actions">${actions}</td>
    </tr>`;
  }
  async function loadUsers() {
    const body = $('usersBody');
    try {
      const data = await api('GET', '/auth/users');
      state = { users: data.users, roles: data.roles, currentUserId: data.current_user_id };
      body.innerHTML = state.users.length ? state.users.map(userRow).join('') : '<tr class="empty-row"><td colspan="7">Пользователей нет</td></tr>';
      $('usersStatus').textContent = '';
    } catch (e) {
      body.innerHTML = `<tr class="empty-row"><td colspan="7">Не удалось загрузить пользователей: ${esc(e.message)}</td></tr>`;
    }
  }

  function createDialog() {
    const generated = randomPassword();
    const bodyHtml = [
      '<label>Логин<input id="ufUsername" type="text" maxlength="64" autocomplete="off" spellcheck="false" /></label>',
      '<label>Имя<input id="ufName" type="text" maxlength="120" autocomplete="off" /></label>',
      '<label>Email <span class="optional">(необязательно)</span><input id="ufEmail" type="text" maxlength="254" autocomplete="off" /></label>',
      `<label>Роль<select id="ufRole">${roleOptions('viewer')}</select></label>`,
      `<label>Временный пароль<input id="ufPassword" type="text" autocomplete="off" spellcheck="false" value="${esc(generated)}" /></label>`,
      '<label class="checkbox-row"><input id="ufMustChange" type="checkbox" checked /> Потребовать смену пароля при первом входе</label>',
      '<p class="dim">Передайте пароль пользователю безопасным способом: после закрытия окна он не отображается.</p>',
      '<div class="dialog-error" id="ufError"></div>'
    ].join('');
    return ui.dialog({
      title: 'Новый пользователь',
      bodyHtml,
      actions: [{ label: 'Отмена' }, {
        label: 'Создать',
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
      `<p class="dim">Логин: ${esc(user.username)}</p>`,
      `<label>Имя<input id="ufName" type="text" maxlength="120" value="${esc(user.display_name)}" /></label>`,
      `<label>Email<input id="ufEmail" type="text" maxlength="254" value="${esc(user.email)}" /></label>`,
      `<label>Роль<select id="ufRole">${roleOptions(user.role)}</select></label>`,
      '<p class="dim">Смена роли завершает все открытые сессии пользователя.</p>',
      '<div class="dialog-error" id="ufError"></div>'
    ].join('');
    return ui.dialog({
      title: 'Пользователь',
      bodyHtml,
      actions: [{ label: 'Отмена' }, {
        label: 'Сохранить',
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
      `<p>Новый пароль для <strong>${esc(user.username)}</strong>. Все его сессии будут завершены.</p>`,
      `<label>Пароль<input id="ufPassword" type="text" autocomplete="off" spellcheck="false" value="${esc(randomPassword())}" /></label>`,
      '<label class="checkbox-row"><input id="ufMustChange" type="checkbox" checked /> Потребовать смену пароля при следующем входе</label>',
      '<div class="dialog-error" id="ufError"></div>'
    ].join('');
    return ui.dialog({
      title: 'Сброс пароля',
      bodyHtml,
      actions: [{ label: 'Отмена' }, {
        label: 'Установить пароль',
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
    edit: async (user) => { if (await editDialog(user)) ui.toast('Сохранено', { tone: 'ok' }); },
    password: async (user) => { if (await passwordDialog(user)) ui.toast('Пароль изменён', { tone: 'ok' }); },
    unlock: async (user) => { await api('POST', `/auth/users/${user.id}/unlock`); ui.toast('Пользователь разблокирован', { tone: 'ok' }); },
    toggle: async (user) => {
      if (user.is_active) {
        const ok = await ui.confirm({ title: 'Отключить пользователя', message: `«${user.username}» не сможет войти, его сессии и API-токены перестанут работать.`, confirmText: 'Отключить', danger: true });
        if (!ok) return;
      }
      await api('PATCH', `/auth/users/${user.id}`, { is_active: !user.is_active });
    },
    delete: async (user) => {
      const ok = await ui.confirm({ title: 'Удалить пользователя', message: `Учётная запись «${user.username}» и её API-токены будут удалены безвозвратно. Если нужно лишь запретить вход, используйте «Отключить».`, confirmText: 'Удалить', danger: true });
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
    const [tone, label] = token.revoked_at ? ['na', 'отозван'] : (expired ? ['warn', 'истёк'] : ['ok', 'действует']);
    const action = token.active ? `<button type="button" class="btn btn-sm btn-danger" data-revoke="${token.id}" data-name="${esc(token.name)}">Отозвать</button>` : '';
    return `<tr><td>${esc(token.name)}</td><td>${esc(token.owner)}</td><td><code>${esc(token.prefix)}…</code></td><td>${esc(roleLabel(token.role))}</td>
      <td class="nowrap">${esc(fmt(token.expires_at))}</td><td class="nowrap">${esc(fmt(token.last_used_at))}</td><td><span class="pill ${tone}">${label}</span></td><td class="actions">${action}</td></tr>`;
  }
  async function loadAllTokens() {
    const body = $('allTokensBody');
    try {
      const data = await api('GET', '/auth/tokens?all=1');
      body.innerHTML = data.tokens.length ? data.tokens.map(tokenRow).join('') : '<tr class="empty-row"><td colspan="8">Токенов нет</td></tr>';
    } catch (e) {
      body.innerHTML = `<tr class="empty-row"><td colspan="8">${esc(e.message)}</td></tr>`;
    }
  }
  async function onTokensClick(event) {
    const btn = event.target.closest('[data-revoke]');
    if (!btn) return;
    const ok = await ui.confirm({ title: 'Отозвать токен', message: `Токен «${btn.dataset.name}» перестанет работать сразу.`, confirmText: 'Отозвать', danger: true });
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
    const label = ACTION_LABELS[event.action] || event.action;
    const failed = event.success ? '' : ' <span class="pill fail">отказ</span>';
    return `<tr><td class="nowrap">${esc(LL.formatDateTime(event.created_at))}</td><td>${esc(event.actor || '—')}${event.via && event.via !== 'session' ? ` <span class="dim">(${esc(event.via)})</span>` : ''}</td>
      <td>${esc(label)}${failed}</td><td>${esc(event.target || '')}</td><td class="dim">${esc(details)}</td><td class="nowrap dim">${esc(event.ip || '')}</td></tr>`;
  }
  async function loadAudit() {
    const body = $('auditBody');
    $('auditStatus').textContent = 'Загрузка…';
    try {
      const data = await api('GET', '/auth/audit?limit=100');
      body.innerHTML = data.events.length ? data.events.map(auditRow).join('') : '<tr class="empty-row"><td colspan="6">Записей нет</td></tr>';
      $('auditStatus').textContent = `Последние ${data.events.length}`;
    } catch (e) {
      body.innerHTML = '';
      $('auditStatus').textContent = `Ошибка: ${e.message}`;
    }
  }

  document.addEventListener('DOMContentLoaded', async () => {
    if (!$('usersBody')) return;
    $('usersBody').addEventListener('click', onUsersClick);
    $('userCreateBtn').addEventListener('click', async () => {
      if (await createDialog()) { ui.toast('Пользователь создан', { tone: 'ok' }); await loadUsers(); }
    });
    $('allTokensBody').addEventListener('click', onTokensClick);
    $('tokensBox').addEventListener('toggle', () => { if ($('tokensBox').open) loadAllTokens(); });
    $('auditBox').addEventListener('toggle', () => { if ($('auditBox').open) loadAudit(); });
    $('auditRefreshBtn').addEventListener('click', loadAudit);
    await loadUsers();
  });
})();
