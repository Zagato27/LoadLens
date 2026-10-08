// Account page: password change and personal API tokens.
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
      if (next.length < minLength) { error.textContent = `Пароль должен содержать не менее ${minLength} символов`; return; }
      if (next !== $('confirmPassword').value) { error.textContent = 'Новый пароль и повтор не совпадают'; return; }
      const button = form.querySelector('button[type="submit"]');
      button.disabled = true;
      try {
        await api('POST', '/auth/password', { current_password: current, new_password: next });
        form.reset();
        ui.toast('Пароль изменён', { tone: 'ok' });
        if (forced) window.location.assign('/');
      } catch (e) {
        error.textContent = e.message;
      } finally {
        button.disabled = false;
      }
    });
  }

  // ---- API tokens ------------------------------------------------------------------
  const ROLE_LABELS = { viewer: 'Наблюдатель', engineer: 'Инженер', admin: 'Администратор' };
  const fmt = (value) => (value ? LL.formatDateTime(value) : '—');

  function tokenRow(token) {
    const expired = token.expires_at && new Date(token.expires_at) <= new Date();
    const [tone, label] = token.revoked_at ? ['na', 'отозван'] : (expired ? ['warn', 'истёк'] : ['ok', 'действует']);
    const action = token.active ? `<button type="button" class="btn btn-sm btn-danger" data-revoke="${token.id}" data-name="${esc(token.name)}">Отозвать</button>` : '';
    return `<tr>
      <td>${esc(token.name)}</td>
      <td><code>${esc(token.prefix)}…</code></td>
      <td>${esc(ROLE_LABELS[token.role] || token.role)}</td>
      <td class="nowrap">${esc(fmt(token.created_at))}</td>
      <td class="nowrap">${esc(fmt(token.expires_at))}</td>
      <td class="nowrap">${esc(fmt(token.last_used_at))}</td>
      <td><span class="pill ${tone}">${label}</span></td>
      <td class="actions">${action}</td>
    </tr>`;
  }

  async function loadTokens() {
    const body = $('tokensBody');
    try {
      const data = await api('GET', '/auth/tokens');
      body.innerHTML = data.tokens.length
        ? data.tokens.map(tokenRow).join('')
        : '<tr class="empty-row"><td colspan="8">Токенов пока нет</td></tr>';
    } catch (e) {
      body.innerHTML = `<tr class="empty-row"><td colspan="8">Не удалось загрузить токены: ${esc(e.message)}</td></tr>`;
    }
  }

  function showNewToken(token) {
    return ui.dialog({
      title: 'Токен создан',
      bodyHtml: `<p>Скопируйте токен сейчас: позже его показать нельзя.</p><code class="token-secret" id="newTokenValue">${esc(token)}</code>`,
      actions: [
        {
          label: 'Копировать',
          primary: true,
          onClick: async (close, body) => {
            try {
              await navigator.clipboard.writeText(token);
              ui.toast('Токен скопирован', { tone: 'ok' });
            } catch (e) {
              const range = document.createRange();
              range.selectNodeContents(body.querySelector('#newTokenValue'));
              const selection = window.getSelection();
              selection.removeAllRanges();
              selection.addRange(range);
              ui.toast('Выделено — нажмите Ctrl+C', { tone: 'info' });
            }
          }
        },
        { label: 'Закрыть' }
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
      const ok = await ui.confirm({ title: 'Отозвать токен', message: `Токен «${btn.dataset.name}» перестанет работать сразу.`, confirmText: 'Отозвать', danger: true });
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
})();
