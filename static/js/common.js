// SPDX-License-Identifier: Apache-2.0
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

  // One banner per page when a data call reports that the storage database is down.
  let outageShown = false;
  function showStorageOutage() {
    const main = document.getElementById('main');
    if (outageShown || !main || document.getElementById('storageDownBanner')) return;
    outageShown = true;
    const box = document.createElement('div');
    box.className = 'banner banner-fail';
    box.id = 'storageDownBanner';
    box.setAttribute('role', 'status');
    const inner = document.createElement('div');
    const title = document.createElement('div');
    title.className = 'banner-title';
    title.textContent = LL.t('ui.storage_down.title');
    inner.appendChild(title);
    if (LL.can('admin')) {
      // The link stands where the {action} placeholder is in the sentence of the UI language.
      const link = document.createElement('a');
      link.href = '/recovery';
      link.textContent = LL.t('ui.storage_down.fix_link');
      const marker = '\u0000';
      LL.t('ui.storage_down.text', { action: marker }).split(marker).forEach((part, i) => {
        if (i) inner.appendChild(link);
        if (part) inner.appendChild(document.createTextNode(part));
      });
    } else {
      inner.appendChild(document.createTextNode(LL.t('ui.storage_down.text', { action: LL.t('ui.storage_down.ask_admin') })));
    }
    box.appendChild(inner);
    main.prepend(box);
  }

  function onAuthFailure(response, url) {
    if (response.status === 503) {
      response.clone().json().then((data) => {
        if (data && data.code === 'storage_unavailable') showStorageOutage();
      }).catch(() => { /* not JSON */ });
      return;
    }
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

// Hides finished background jobs from the shared job list (they stay reachable by link).
// Resolves to the server answer, or null after an error that was already shown to the user.
window.LoadLens.dismissJobs = async function dismissJobs(jobIds) {
  const LL = window.LoadLens;
  const toast = (text, tone) => { if (LL.ui && LL.ui.toast) LL.ui.toast(text, { tone }); };
  const ids = Array.from(new Set((jobIds || []).filter(Boolean)));
  if (!ids.length) return null;
  try {
    const resp = await fetch('/jobs/dismiss', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_ids: ids }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      toast(data.error || LL.t('ui.jobs.dismiss_failed', { status: resp.status }), 'error');
      return null;
    }
    const hidden = Array.isArray(data.dismissed) ? data.dismissed.length : 0;
    const running = Array.isArray(data.skipped_running) ? data.skipped_running.length : 0;
    let text = LL.t('ui.jobs.already_hidden');
    if (hidden && running) text = LL.t('ui.jobs.dismissed_some_running', { hidden, running });
    else if (hidden) text = LL.t('ui.jobs.dismissed', { n: hidden });
    else if (running) text = LL.t('ui.jobs.still_running', { n: running });
    toast(text, running ? 'warn' : 'ok');
    return data;
  } catch (e) {
    toast(LL.t('ui.jobs.no_answer'), 'error');
    return null;
  }
};

// Builds the "Hide" button of a finished job row; ``onDone`` reloads the list.
window.LoadLens.jobDismissButton = function jobDismissButton(job, onDone) {
  const LL = window.LoadLens;
  if (job.status === 'running' || !LL.can('engineer')) return null;
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'btn btn-sm btn-ghost job-dismiss';
  btn.textContent = LL.t('ui.jobs.hide');
  btn.title = LL.t('ui.jobs.hide_title');
  btn.setAttribute('aria-label', LL.t('ui.jobs.hide_aria', { name: job.run_name || '' }).trim());
  btn.addEventListener('click', async () => {
    btn.disabled = true;
    const result = await LL.dismissJobs([job.job_id]);
    if (result) onDone();
    else btn.disabled = false;
  });
  return btn;
};

// Badge with the number of load tests going on now at «Run a test» in the menu, for engineers and
// admins: polled every 30 s. The dashboard reads the runs itself, claims the badge and passes the number.
(function installActiveRunsBadge() {
  const LL = window.LoadLens;
  const POLL_MS = 30000;
  let timer = 0;
  let claimed = false;
  function show(total) {
    const badge = document.getElementById('navLaunchBadge');
    if (!badge) return;
    const n = Math.max(0, Math.floor(Number(total) || 0));
    badge.hidden = n === 0;
    badge.textContent = n > 99 ? '99+' : String(n);
    badge.title = n ? LL.t('ui.active_runs.title', { n }) : '';
  }
  async function poll() {
    clearTimeout(timer);
    if (claimed) return;
    try {
      const resp = await fetch('/load_tests/active?limit=0');
      if (resp.ok) show((await resp.json()).total);
    } catch (e) {
      /* the badge is auxiliary */
    }
    if (!claimed) timer = setTimeout(poll, POLL_MS);
  }
  function claim() {
    claimed = true;
    clearTimeout(timer);
  }
  LL.activeRuns = { show, claim };
  document.addEventListener('DOMContentLoaded', () => {
    if (!document.getElementById('navLaunchBadge') || !LL.can('engineer')) return;
    // After the other DOMContentLoaded listeners: the dashboard claims the badge before the first poll.
    timer = setTimeout(poll, 0);
  });
})();

// «Changed manually: ivanov, 2026-10-17 15:20. Reason: …» for a manual verdict ({verdict, reason, author,
// changed_at}); '' without one.
window.LoadLens.manualVerdictTitle = function manualVerdictTitle(manual) {
  if (!manual || !manual.verdict) return '';
  const LL = window.LoadLens;
  const who = [String(manual.author || '').trim(), manual.changed_at ? LL.formatDateTime(manual.changed_at) : ''].filter(Boolean).join(', ');
  const reason = String(manual.reason || '').trim();
  const head = who ? LL.t('ui.manual_verdict.changed_by', { who }) : LL.t('ui.manual_verdict.changed');
  return reason ? `${head} ${LL.t('ui.manual_verdict.reason', { reason })}` : head;
};

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
      all.textContent = LL.t('ui.project_area.all');
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

// Tables whose values follow the UI language: every value is read from the dictionary when it is accessed,
// so the keys, Object.keys() and JSON stay the same while the text is LL.t of the key.
window.LoadLens.messageTable = function messageTable(keys, text) {
  const out = {};
  (keys || []).forEach((key) => {
    Object.defineProperty(out, key, { enumerable: true, get: () => text(key) });
  });
  return out;
};

window.LoadLens.DOMAIN_TITLES = window.LoadLens.messageTable(
  ['final', 'jvm', 'database', 'kafka', 'microservices', 'hard_resources', 'lt_framework', 'lt_generators', 'application_logs'],
  (key) => window.LoadLens.t(`common.domain.${key}`),
);
window.LoadLens.domainTitle = function domainTitle(key) {
  return window.LoadLens.DOMAIN_TITLES[key] || String(key || '');
};
// Title a report stored for the domain (e.g. «Go runtime» in jvm), else the default one.
window.LoadLens.domainTitleFor = function domainTitleFor(key, titles) {
  const own = titles && typeof titles === 'object' ? titles[key] : '';
  return typeof own === 'string' && own.trim() ? own.trim() : window.LoadLens.domainTitle(key);
};
// domain_titles of the final row of /llm_reports (parsed may arrive as an object or a JSON string).
window.LoadLens.reportDomainTitles = function reportDomainTitles(rows) {
  const final = (Array.isArray(rows) ? rows : []).find((row) => row && row.domain === 'final');
  let parsed = final ? final.parsed : null;
  if (typeof parsed === 'string') { try { parsed = JSON.parse(parsed); } catch (e) { parsed = null; } }
  const titles = parsed && typeof parsed === 'object' ? parsed.domain_titles : null;
  return titles && typeof titles === 'object' ? titles : {};
};

window.LoadLens.TEST_TYPE_LABELS = window.LoadLens.messageTable(
  ['step', 'soak', 'spike', 'stress'],
  (key) => window.LoadLens.t(`common.test_type.${key}`),
);
window.LoadLens.testTypeLabel = function testTypeLabel(key) {
  const k = String(key || '').trim().toLowerCase();
  if (!k) return '—';
  return window.LoadLens.TEST_TYPE_LABELS[k] || k;
};

// Verdicts: the API sends codes (pass | risk | fail | no_data); reports saved before the codes may still
// carry the Russian labels. The one JS copy of AI/verdicts.py (tests/test_verdicts.py keeps them equal):
// codes and old labels map exactly, other text is no_data (the heuristic for free text lives in Python).
(function verdictHelpers(LL) {
  const CODES = ['pass', 'risk', 'fail', 'no_data'];
  // The Russian labels stored as the verdict before the codes (AI/verdicts.py LEGACY): parsed into codes in
  // every UI language. Shown labels come from the dictionary (common.verdict.<code>).
  const LEGACY_LABELS = { pass: 'Успешно', risk: 'Есть риски', fail: 'Провал', no_data: 'Недостаточно данных' };
  const label = (code) => LL.t(`common.verdict.${code}`);
  // Pill tones of app.css, the report pill classes and the glossary terms of each code.
  const TONES = { pass: 'ok', risk: 'warn', fail: 'fail', no_data: 'na' };
  const REPORT_CLASSES = { pass: 'report-verdict-success', risk: 'report-verdict-risk', fail: 'report-verdict-fail', no_data: 'report-verdict-na' };
  const TERMS = { pass: 'verdict_success', risk: 'verdict_risk', fail: 'verdict_fail', no_data: 'verdict_na' };
  // Lowercase with the letter ё folded into е and runs of spaces collapsed.
  const norm = (value) => String(value).toLowerCase().replace(/ё/g, 'е').split(/\s+/).filter(Boolean).join(' ');
  const EXACT = {};
  CODES.forEach((code) => { EXACT[norm(code)] = code; });
  CODES.forEach((code) => { EXACT[norm(LEGACY_LABELS[code])] = code; });

  LL.VERDICTS = CODES.slice();
  // {code: label} in the UI language.
  LL.VERDICT_LABELS = LL.messageTable(CODES, label);
  // Code of a code or an old label (case, ё and spaces do not matter); ``fallback`` (default 'no_data')
  // for anything else, e.g. null to tell an unknown value from no_data.
  LL.verdictCode = function verdictCode(value, fallback) {
    const missing = fallback === undefined ? 'no_data' : fallback;
    if (value === null || value === undefined) return missing;
    return Object.prototype.hasOwnProperty.call(EXACT, norm(value)) ? EXACT[norm(value)] : missing;
  };
  LL.verdictLabel = function verdictLabel(value) { return label(LL.verdictCode(value)); };
  // Pill tone (ok / warn / fail / na) of the status classes.
  LL.verdictClass = function verdictClass(value) { return TONES[LL.verdictCode(value)]; };
  LL.verdictReportClass = function verdictReportClass(value) { return REPORT_CLASSES[LL.verdictCode(value)]; };
  LL.verdictTermKey = function verdictTermKey(value) { return TERMS[LL.verdictCode(value)]; };
})(window.LoadLens);

window.LoadLens.formatDateTime = function formatDateTime(value) {
  const d = value instanceof Date ? value : new Date(value);
  if (!Number.isFinite(d.getTime())) return '—';
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
};

// Web report link of a run: by run_id when known, else by run name, else the archive.
window.LoadLens.reportUrl = function reportUrl(run) {
  const id = run && run.run_id ? String(run.run_id) : '';
  if (id) return `/reports/${encodeURIComponent(id)}`;
  const name = run && run.run_name ? String(run.run_name) : '';
  return name ? `/reports/${encodeURIComponent(name)}` : '/reports';
};

// "5 min ago" style date for lists (common.time.*); older than a week (or in the future) falls back to
// formatDateTime. Up to a minute in the future still reads "just now" to absorb clock skew between server and browser.
window.LoadLens.relativeTime = function relativeTime(value, now = Date.now()) {
  const LL = window.LoadLens;
  const d = value instanceof Date ? value : new Date(value);
  const time = d.getTime();
  const nowMs = now instanceof Date ? now.getTime() : Number(now);
  if (value === null || value === undefined || value === '' || !Number.isFinite(time) || !Number.isFinite(nowMs)) return '—';
  const diff = nowMs - time;
  const MIN = 60000; const HOUR = 60 * MIN; const DAY = 24 * HOUR;
  if (diff < -MIN) return LL.formatDateTime(d);
  if (diff < MIN) return LL.t('common.time.just_now');
  if (diff < HOUR) return LL.t('common.time.minutes_ago', { n: Math.floor(diff / MIN) });
  if (diff < DAY) return LL.t('common.time.hours_ago', { n: Math.floor(diff / HOUR) });
  const midnight = (ms) => { const x = new Date(ms); x.setHours(0, 0, 0, 0); return x.getTime(); };
  const days = Math.round((midnight(nowMs) - midnight(time)) / DAY);
  const pad = (n) => String(n).padStart(2, '0');
  if (days <= 1) return LL.t('common.time.yesterday_at', { time: `${pad(d.getHours())}:${pad(d.getMinutes())}` });
  if (days < 7) return LL.tn('common.time.days_ago', days);
  return LL.formatDateTime(d);
};

// Human duration (common.time.*): ru «45 с», «59 мин», «1 ч 05 мин», «2 д 3 ч»; '—' for invalid or negative input.
window.LoadLens.formatDuration = function formatDuration(ms) {
  const LL = window.LoadLens;
  if (ms === null || ms === undefined || ms === '' || typeof ms === 'boolean') return '—';
  const value = Number(ms);
  if (!Number.isFinite(value) || value < 0) return '—';
  const totalSec = Math.floor(value / 1000);
  if (totalSec < 60) return LL.t('common.time.seconds', { n: totalSec });
  const totalMin = Math.floor(totalSec / 60);
  if (totalMin < 60) return LL.t('common.time.minutes', { n: totalMin });
  const totalHours = Math.floor(totalMin / 60);
  if (totalHours < 24) return LL.t('common.time.hours_minutes', { h: totalHours, m: String(totalMin % 60).padStart(2, '0') });
  return LL.t('common.time.days_hours', { d: Math.floor(totalHours / 24), h: totalHours % 24 });
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
