// Archive page logic
(function () {
  let currentPage = 1;
  const pageSize = 20;
  let currentSort = 'end_time';
  let currentDir = 'desc';
  let currentQ = '';

  function fmtDate(iso) {
    try {
      const d = new Date(iso);
      const yyyy = d.getFullYear();
      const MM = String(d.getMonth() + 1).padStart(2, '0');
      const dd = String(d.getDate()).padStart(2, '0');
      const hh = String(d.getHours()).padStart(2, '0');
      const mm = String(d.getMinutes()).padStart(2, '0');
      return `${yyyy}-${MM}-${dd} ${hh}:${mm}`;
    } catch (e) {
      return iso || '';
    }
  }

  function fmtDuration(startIso, endIso) {
    try {
      const s = new Date(startIso).getTime();
      const e = new Date(endIso).getTime();
      const ms = Math.max(0, e - s);
      const m = Math.floor(ms / 60000);
      const h = Math.floor(m / 60);
      const mm = String(m % 60).padStart(2, '0');
      return h ? `${h}ч ${mm}м` : `${m}м`;
    } catch (e) {
      return '—';
    }
  }

  function verdictClass(v) {
    try {
      const t = String(v || '').toLowerCase();
      if (!t) return 'na';
      if (t.includes('усп')) return 'ok';
      if (t.includes('риск')) return 'warn';
      if (t.includes('провал')) return 'fail';
      if (t.includes('недостат')) return 'na';
      return 'na';
    } catch (e) {
      return 'na';
    }
  }

  async function loadPage() {
    const offset = (currentPage - 1) * pageSize;
    const u = new URL('/runs', location.origin);
    if (currentQ) u.searchParams.set('q', currentQ);
    u.searchParams.set('offset', String(offset));
    u.searchParams.set('limit', String(pageSize));
    u.searchParams.set('sort', currentSort);
    u.searchParams.set('dir', currentDir);
    const resp = await fetch(u);
    const rows = await resp.json();
    const tbody = document.getElementById('tbody');
    if (!tbody) return;
    tbody.innerHTML = '';
    (rows || []).forEach((r) => {
      const tr = document.createElement('tr');
      tr.className = 'data-row';
      const dt = fmtDate(r.end_time || r.start_time);
      const created = fmtDate(r.report_created_at || '');
      const dur = fmtDuration(r.start_time, r.end_time);
      const v = (r.verdict || 'Недостаточно данных');
      const vc = verdictClass(v);
      const verdictHtml = `<span class=\"pill ${vc}\">${v}</span>`;
      const repUrl = '/reports/' + encodeURIComponent(r.service || '') + '/' + encodeURIComponent(r.run_name);
      const testType = (r.test_type || '');
      tr.innerHTML = `<td>${r.run_name}</td><td>${r.service || ''}</td><td>${testType}</td><td>${dt}</td><td>${created}</td><td>${verdictHtml}</td><td>${dur}</td><td class=\"actions\"><button class=\"btn act-view\">Открыть</button><button class=\"btn act-compare\">Сравнить</button><button class=\"btn act-copy\">Скопировать ссылку</button><button class=\"btn act-delete\" style=\"border-color:#6b2e2e;\">Удалить</button></td>`;
      tr.addEventListener('click', (e) => {
        if (!(e.target && (e.target.classList.contains('btn')))) {
          location.href = repUrl;
        }
      });
      tr.querySelector('.act-view').addEventListener('click', (e) => { e.stopPropagation(); location.href = repUrl; });
      tr.querySelector('.act-compare').addEventListener('click', (e) => { e.stopPropagation(); location.href = '/compare?run_a=' + encodeURIComponent(r.run_name); });
      tr.querySelector('.act-copy').addEventListener('click', async (e) => {
        e.stopPropagation();
        try { await navigator.clipboard.writeText(location.origin + repUrl); } catch (err) {}
      });
      const delBtn = tr.querySelector('.act-delete');
      if (delBtn) {
        delBtn.addEventListener('click', async (e) => {
          e.stopPropagation();
          const ok = confirm(`Удалить отчёт '${r.run_name}'? Это действие необратимо.`);
          if (!ok) return;
          try {
            const resp = await fetch('/runs/' + encodeURIComponent(r.run_name), { method: 'DELETE' });
            const j = await resp.json();
            if (resp.ok) { await loadPage(); }
            else { alert(j.error || 'Ошибка удаления'); }
          } catch (err) { alert('Ошибка удаления'); }
        });
      }
      tbody.appendChild(tr);
    });
    const pageInfo = document.getElementById('pageInfo');
    if (pageInfo) pageInfo.textContent = `Стр. ${currentPage}`;
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    if (prevBtn) prevBtn.disabled = currentPage <= 1;
    if (nextBtn) nextBtn.disabled = (rows || []).length < pageSize;
  }

  function wireControls() {
    const searchBtn = document.getElementById('searchBtn');
    const clearBtn = document.getElementById('clearBtn');
    const searchInput = document.getElementById('searchInput');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    if (searchBtn) searchBtn.addEventListener('click', () => { currentQ = String(searchInput.value || '').trim(); currentPage = 1; loadPage(); });
    if (clearBtn) clearBtn.addEventListener('click', () => { searchInput.value = ''; currentQ = ''; currentPage = 1; loadPage(); });
    if (searchInput) searchInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') { currentQ = String(e.target.value || '').trim(); currentPage = 1; loadPage(); } });
    if (prevBtn) prevBtn.addEventListener('click', () => { if (currentPage > 1) { currentPage--; loadPage(); } });
    if (nextBtn) nextBtn.addEventListener('click', () => { currentPage++; loadPage(); });
    document.querySelectorAll('th[data-sort]').forEach((th) => {
      th.addEventListener('click', () => {
        let by = th.getAttribute('data-sort');
        if (by === 'duration') by = 'end_time';
        if (currentSort === by) { currentDir = (currentDir === 'asc') ? 'desc' : 'asc'; }
        else { currentSort = by; currentDir = by === 'end_time' ? 'desc' : 'asc'; }
        loadPage();
      });
    });
  }

  document.addEventListener('DOMContentLoaded', async () => {
    try { await window.LoadLens.initProjectArea(); } catch (e) {}
    wireControls();
    await loadPage();
  });
})();


