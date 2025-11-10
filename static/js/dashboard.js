// Dashboard page logic
(function () {
  function fmtPeriod(startMs, endMs) {
    try {
      if (!startMs && !endMs) return '—';
      const s = startMs ? new Date(parseInt(startMs, 10)) : null;
      const e = endMs ? new Date(parseInt(endMs, 10)) : null;
      const pad = (n) => String(n).padStart(2, '0');
      const fmt = (d) =>
        `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
      if (s && e) return `${fmt(s)} — ${fmt(e)}`;
      if (s) return fmt(s);
      if (e) return fmt(e);
      return '—';
    } catch (e) {
      return '—';
    }
  }

  function verdictClass(v) {
    const t = (v || '').toLowerCase();
    if (t.includes('усп')) return 'ok';
    if (t.includes('риск')) return 'warn';
    if (t.includes('провал')) return 'fail';
    return 'na';
  }
  function verdictLabel(v) {
    return v || 'Недостаточно данных';
  }

  async function loadDashboard() {
    try {
      const resp = await fetch('/dashboard_data');
      const data = await resp.json();
      const last = data.last_run || null;
      if (last) {
        const sumRun = document.getElementById('sumRun');
        const sumService = document.getElementById('sumService');
        const sumPeriod = document.getElementById('sumPeriod');
        const vbox = document.getElementById('sumVerdict');
        if (sumRun) sumRun.textContent = last.run_name || '—';
        if (sumService) sumService.textContent = last.service || '—';
        if (sumPeriod) sumPeriod.textContent = fmtPeriod(last.start_ms, last.end_ms);
        const v = verdictLabel(last.verdict);
        const cls = verdictClass(v);
        if (vbox) vbox.innerHTML = `<span class=\"pill ${cls}\">${v}</span>`;
        try {
          const link = document.getElementById('lastReportLink');
          if (link && last.run_name) {
            link.href = '/reports/' + encodeURIComponent(last.run_name);
            link.style.display = 'block';
          }
        } catch (e) {}
      }

      const vc = data.verdict_counts || {};
      const labels = ['Успешно', 'Есть риски', 'Провал', 'Недостаточно данных'];
      const values = labels.map((k) => Number(vc[k] || 0));
      const colors = ['#2ecc71', '#f1c40f', '#e74c3c', '#7f8c8d'];
      const ctx = document.getElementById('statusChart').getContext('2d');
      // eslint-disable-next-line no-undef
      new Chart(ctx, {
        type: 'doughnut',
        data: { labels, datasets: [{ data: values, backgroundColor: colors, borderColor: '#222', borderWidth: 1 }] },
        options: { responsive: true, plugins: { legend: { position: 'bottom', labels: { color: '#ddd' } } } }
      });
    } catch (e) {
      // noop
    }
  }

  document.addEventListener('DOMContentLoaded', async () => {
    try {
      await window.LoadLens.initProjectArea();
    } catch (e) {}
    await loadDashboard();
  });
})();


