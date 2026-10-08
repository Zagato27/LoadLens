// Common utilities shared across pages
window.LoadLens = window.LoadLens || {};

// ---- authentication glue -------------------------------------------------------------
// The server renders the CSRF token and the user's role into <meta> tags. fetch() is wrapped once
// here so that every page sends the token on state-changing requests and reacts to an expired
// session or a forced password change without each call site knowing about it.
(function installAuthFetch() {
  const LL = window.LoadLens;
  const meta = (name) => {
    const el = document.querySelector(`meta[name="${name}"]`);
    return el ? (el.getAttribute('content') || '') : '';
  };
  const RANK = { viewer: 1, engineer: 2, admin: 3 };
  LL.role = meta('loadlens-role') || 'admin';
  LL.csrfToken = () => meta('csrf-token');
  // LL.can('engineer') is true for engineers and admins. The server enforces the same rules;
  // this only hides controls that would be refused.
  LL.can = (role) => (RANK[LL.role] || 0) >= (RANK[role] || 0);

  const UNSAFE = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);
  const nativeFetch = window.fetch.bind(window);
  let navigating = false;
  let staleShown = false;

  function onAuthFailure(response, url) {
    if (response.status === 401 && !navigating && url.pathname !== '/login' && url.pathname !== '/auth/me') {
      navigating = true;
      window.location.assign(`/login?next=${encodeURIComponent(window.location.pathname + window.location.search)}`);
      return;
    }
    if (response.status !== 403) return;
    response.clone().json().then((data) => {
      if (data && data.code === 'password_change_required' && !navigating && window.location.pathname !== '/account') {
        navigating = true;
        window.location.assign('/account?force=1');
      } else if (data && data.code === 'csrf_failed' && !staleShown && LL.ui && LL.ui.toast) {
        staleShown = true;
        LL.ui.toast(data.error, { tone: 'warn', timeout: 0 });
      }
    }).catch(() => { /* not JSON */ });
  }

  window.fetch = async function loadLensFetch(input, init) {
    const options = init ? { ...init } : {};
    const method = String(options.method || (input && input.method) || 'GET').toUpperCase();
    const url = new URL(typeof input === 'string' ? input : (input.url || String(input)), window.location.href);
    const sameOrigin = url.origin === window.location.origin;
    if (sameOrigin && UNSAFE.has(method)) {
      const token = LL.csrfToken();
      if (token) {
        const headers = new Headers(options.headers || (input && input.headers) || undefined);
        if (!headers.has('X-CSRF-Token')) headers.set('X-CSRF-Token', token);
        options.headers = headers;
      }
    }
    const response = await nativeFetch(input, options);
    if (sameOrigin) onAuthFailure(response, url);
    return response;
  };
})();

// Project area selector in the header. The chosen area is stored in a cookie that
// every data endpoint reads, so a change reloads the page. Idempotent: the same
// promise is returned to every caller, and it resolves to the active area id.
window.LoadLens.activeProjectArea = '';
window.LoadLens.initProjectArea = function initProjectArea() {
  const LL = window.LoadLens;
  if (LL._projectAreaReady) return LL._projectAreaReady;
  LL._projectAreaReady = (async () => {
    const sel = document.getElementById('projectAreaSelect');
    try {
      const [areasResp, curResp] = await Promise.all([fetch('/project_areas'), fetch('/current_project_area')]);
      const areas = await areasResp.json();
      const cur = await curResp.json();
      const current = (cur && cur.project_area) || '';
      LL.activeProjectArea = current;
      LL.projectAreas = Array.isArray(areas) ? areas : [];
      if (!sel) return current;
      sel.innerHTML = '';
      const all = document.createElement('option');
      all.value = '';
      all.textContent = 'Все области';
      sel.appendChild(all);
      LL.projectAreas.forEach((a) => {
        const id = (a && typeof a === 'object') ? String(a.id || '') : String(a || '');
        if (!id) return;
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = (a && typeof a === 'object' && a.title) ? a.title : id;
        sel.appendChild(opt);
      });
      if (current && !Array.from(sel.options).some((o) => o.value === current)) {
        const opt = document.createElement('option');
        opt.value = current;
        opt.textContent = current;
        sel.appendChild(opt);
      }
      sel.value = current;
      sel.addEventListener('change', async () => {
        try {
          await fetch('/project_area', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ project_area: sel.value || '' })
          });
        } finally {
          location.reload();
        }
      });
      return current;
    } catch (e) {
      return LL.activeProjectArea;
    }
  })();
  return LL._projectAreaReady;
};

// Random color generator for charts (legacy; prefer colorFor for stable colors)
window.LoadLens.randColor = function randColor(alpha = 0.7) {
  const r = Math.floor(100 + Math.random() * 155);
  const g = Math.floor(100 + Math.random() * 155);
  const b = Math.floor(100 + Math.random() * 155);
  return `rgba(${r},${g},${b},${alpha})`;
};

// Deterministic color for a series name: same name -> same color on every chart and reload.
window.LoadLens.colorFor = function colorFor(name, alpha = 0.85) {
  const text = String(name || '');
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
  }
  // Golden-angle spread gives well-separated hues for neighbouring hashes.
  const hue = Math.abs(hash * 137.508) % 360;
  const saturation = 60 + (Math.abs(hash >> 8) % 20);
  const lightness = 55 + (Math.abs(hash >> 16) % 12);
  return `hsla(${hue.toFixed(0)}, ${saturation}%, ${lightness}%, ${alpha})`;
};

window.LoadLens.DOMAIN_TITLES = {
  final: 'Итог',
  jvm: 'JVM',
  database: 'База данных',
  kafka: 'Kafka',
  microservices: 'Микросервисы',
  hard_resources: 'Ресурсы узлов',
  lt_framework: 'Нагрузочный инструмент',
  application_logs: 'Логи приложений',
};
window.LoadLens.domainTitle = function domainTitle(key) {
  return window.LoadLens.DOMAIN_TITLES[key] || String(key || '');
};

window.LoadLens.TEST_TYPE_LABELS = {
  step: 'Ступенчатый (max perf)',
  soak: 'Долговременный (soak)',
  spike: 'Всплески (spike)',
  stress: 'Стресс',
};
window.LoadLens.testTypeLabel = function testTypeLabel(key) {
  const k = String(key || '').trim().toLowerCase();
  if (!k) return '—';
  return window.LoadLens.TEST_TYPE_LABELS[k] || k;
};

window.LoadLens.VERDICTS = ['Успешно', 'Есть риски', 'Провал', 'Недостаточно данных'];
window.LoadLens.verdictClass = function verdictClass(verdict) {
  const t = String(verdict || '').toLowerCase();
  if (t.includes('усп')) return 'ok';
  if (t.includes('риск')) return 'warn';
  if (t.includes('провал')) return 'fail';
  return 'na';
};

window.LoadLens.formatDateTime = function formatDateTime(value) {
  const d = value instanceof Date ? value : new Date(value);
  if (!Number.isFinite(d.getTime())) return '—';
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
};

// Browser time zone as "UTC+03:00 (Europe/Moscow)" so users know which clock the dates use.
window.LoadLens.timeZoneLabel = function timeZoneLabel() {
  const offsetMin = -new Date().getTimezoneOffset();
  const sign = offsetMin >= 0 ? '+' : '-';
  const abs = Math.abs(offsetMin);
  const hh = String(Math.floor(abs / 60)).padStart(2, '0');
  const mm = String(abs % 60).padStart(2, '0');
  let zone = '';
  try { zone = Intl.DateTimeFormat().resolvedOptions().timeZone || ''; } catch (e) { zone = ''; }
  return `UTC${sign}${hh}:${mm}${zone ? ` (${zone})` : ''}`;
};

// Local datetime-local input value -> ISO 8601 with explicit offset (e.g. 2025-01-15T10:00:00+03:00).
window.LoadLens.localInputToIso = function localInputToIso(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  const d = new Date(raw);
  if (!Number.isFinite(d.getTime())) return '';
  const pad = (n) => String(n).padStart(2, '0');
  const offsetMin = -d.getTimezoneOffset();
  const sign = offsetMin >= 0 ? '+' : '-';
  const abs = Math.abs(offsetMin);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:00`
    + `${sign}${pad(Math.floor(abs / 60))}:${pad(abs % 60)}`;
};
