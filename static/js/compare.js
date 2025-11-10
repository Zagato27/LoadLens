// Compare page logic
(function () {
  let chart;
  let legendSortBy = 'avg'; // 'name' | 'avg'
  let legendSortDir = 'desc'; // 'asc' | 'desc'
  let lastQueryLabel = '';

  // Background plugin for chart area
  const backgroundPlugin = {
    id: 'customBackground',
    beforeDraw(c, args, opts) {
      const { ctx, chartArea } = c;
      if (!chartArea) return;
      ctx.save();
      ctx.fillStyle = (opts && opts.color) || '#151515';
      ctx.fillRect(chartArea.left, chartArea.top, chartArea.right - chartArea.left, chartArea.bottom - chartArea.top);
      ctx.restore();
    }
  };
  if (window.Chart && Chart.register) Chart.register(backgroundPlugin);

  function randColor(alpha = 0.7) {
    if (window.LoadLens && typeof window.LoadLens.randColor === 'function') {
      return window.LoadLens.randColor(alpha);
    }
    const r = Math.floor(100 + Math.random() * 155);
    const g = Math.floor(100 + Math.random() * 155);
    const b = Math.floor(100 + Math.random() * 155);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  async function loadRuns() {
    try {
      const runs = await (await fetch('/runs')).json();
      const runAOpts = document.getElementById('runAOptions');
      const runBOpts = document.getElementById('runBOptions');
      if (!runAOpts || !runBOpts) return;
      runAOpts.innerHTML = '';
      runBOpts.innerHTML = '';
      runs.forEach((r) => {
        const d1 = document.createElement('div');
        d1.className = 'custom-option';
        d1.textContent = r.run_name;
        d1.dataset.value = r.run_name;
        runAOpts.appendChild(d1);
        const d2 = document.createElement('div');
        d2.className = 'custom-option';
        d2.textContent = r.run_name;
        d2.dataset.value = r.run_name;
        runBOpts.appendChild(d2);
      });
      if (runs.length >= 2) {
        const a = document.getElementById('runASelect');
        const b = document.getElementById('runBSelect');
        if (a) a.textContent = runs[0].run_name;
        if (b) b.textContent = runs[1].run_name;
      }
    } catch (e) {
      // noop
    }
  }

  async function loadSchema() {
    const root = document.getElementById('accRoot');
    if (!root) return;
    root.innerHTML = 'Загрузка…';
    const schema = await (await fetch('/domains_schema')).json();
    root.innerHTML = '';
    Object.keys(schema).forEach((domain) => {
      const item = document.createElement('div'); item.className = 'acc-item';
      const header = document.createElement('div'); header.className = 'acc-header'; header.innerHTML = `<div>${domain}</div><div>▼</div>`;
      const content = document.createElement('div'); content.className = 'acc-content'; content.dataset.domain = domain;
      // summary table container
      const tableWrap = document.createElement('div'); tableWrap.className = 'cmp-summary-wrap'; content.appendChild(tableWrap);
      // metrics wrap
      const metricsWrap = document.createElement('div'); metricsWrap.className = 'metrics-wrap'; content.appendChild(metricsWrap);
      // pills for charts
      const pillsWrap = document.createElement('div'); pillsWrap.className = 'charts-pills'; content.appendChild(pillsWrap);
      schema[domain].forEach((ql) => {
        const pill = document.createElement('span'); pill.className = 'pill'; pill.textContent = ql.query_label;
        pill.addEventListener('click', () => renderOverlay(domain, ql.query_label));
        pillsWrap.appendChild(pill);
        // metric series summary card
        const mItem = document.createElement('div'); mItem.className = 'metric-item';
        const mHeader = document.createElement('div'); mHeader.className = 'metric-header'; mHeader.innerHTML = `<div>${ql.query_label}</div><div>▼</div>`;
        const mContent = document.createElement('div'); mContent.className = 'metric-content'; mContent.textContent = '—';
        mHeader.addEventListener('click', async () => {
          const opening = mContent.style.display !== 'block';
          mContent.style.display = opening ? 'block' : 'none';
          if (opening) await loadMetricSeriesSummary(domain, ql.query_label, mContent);
        });
        mItem.appendChild(mHeader); mItem.appendChild(mContent); metricsWrap.appendChild(mItem);
      });
      header.addEventListener('click', async () => {
        const opening = content.style.display !== 'block';
        content.style.display = opening ? 'block' : 'none';
        if (opening) {
          await loadDomainSummary(domain, tableWrap);
        }
      });
      item.appendChild(header); item.appendChild(content); root.appendChild(item);
    });
    if (!Object.keys(schema || {}).length) { root.textContent = 'Нет данных'; }
  }

  async function loadMetricSeriesSummary(domain, queryLabel, container) {
    try {
      container.textContent = 'Загрузка…';
      const runA = document.getElementById('runASelect').textContent;
      const runB = document.getElementById('runBSelect').textContent;
      const u = new URL('/compare_metric_summary', location.origin);
      u.searchParams.set('run_a', runA);
      u.searchParams.set('run_b', runB);
      u.searchParams.set('domain', domain);
      u.searchParams.set('query_label', queryLabel);
      const resp = await fetch(u);
      const json = await resp.json();
      if (!json || !Array.isArray(json.rows) || !json.rows.length) {
        container.textContent = 'Нет данных';
        return;
      }
      const rows = json.rows;
      const tbl = document.createElement('table'); tbl.className = 'metric-table';
      tbl.innerHTML = '<thead><tr><th data-sort=\"series\">Серия</th><th data-sort=\"a\">Тест A (P95)</th><th data-sort=\"b\">Тест B (P95)</th><th data-sort=\"trend\">Тенденция</th></tr></thead>';
      const tb = document.createElement('tbody');
      const fmt = (v) => (typeof v === 'number' && isFinite(v)) ? v.toFixed(2) : '—';
      rows.forEach((r) => {
        const tr = document.createElement('tr');
        let trendStr = '—'; let cls = '';
        if (typeof r.trend_pct === 'number' && isFinite(r.trend_pct)) {
          const sign = r.trend_pct >= 0 ? 1 : -1;
          const abs = Math.abs(r.trend_pct).toFixed(1);
          const arrow = (sign > 0) ? '↑' : '↓';
          cls = (sign > 0) ? 'up' : 'down';
          trendStr = `${arrow} ${abs}%`;
        }
        tr.innerHTML = `<td>${r.series}</td><td>${fmt(r.p95_a)}</td><td>${fmt(r.p95_b)}</td><td class=\"metric-trend ${cls}\">${trendStr}</td>`;
        tr.dataset.series = r.series || '';
        tr.dataset.a = (typeof r.p95_a === 'number' && isFinite(r.p95_a)) ? String(r.p95_a) : '';
        tr.dataset.b = (typeof r.p95_b === 'number' && isFinite(r.p95_b)) ? String(r.p95_b) : '';
        tr.dataset.trend = (typeof r.trend_pct === 'number' && isFinite(r.trend_pct)) ? String(r.trend_pct) : '';
        tb.appendChild(tr);
      });
      tbl.appendChild(tb);
      container.innerHTML = '';
      container.appendChild(tbl);
      const theadEl = tbl.querySelector('thead');
      tbl.dataset.sortBy = tbl.dataset.sortBy || 'series';
      tbl.dataset.sortDir = tbl.dataset.sortDir || 'asc';
      function sortRows(by, dir) {
        const rowsArr = Array.from(tb.querySelectorAll('tr'));
        function val(tr) {
          if (by === 'series') return (tr.dataset.series || '').toLowerCase();
          const num = parseFloat(tr.dataset[by] || '');
          return Number.isFinite(num) ? num : null;
        }
        rowsArr.sort((r1, r2) => {
          const v1 = val(r1); const v2 = val(r2);
          if (by === 'series') {
            const cmp = v1.localeCompare(v2);
            return dir === 'asc' ? cmp : -cmp;
          } else {
            if (v1 == null && v2 == null) return 0;
            if (v1 == null) return 1;
            if (v2 == null) return -1;
            return dir === 'asc' ? (v1 - v2) : (v2 - v1);
          }
        });
        tb.innerHTML = '';
        rowsArr.forEach((tr) => tb.appendChild(tr));
      }
      theadEl.addEventListener('click', (e) => {
        const th = e.target.closest('[data-sort]');
        if (!th) return;
        const by = th.getAttribute('data-sort');
        let dir = 'asc';
        if (tbl.dataset.sortBy === by) {
          dir = (tbl.dataset.sortDir === 'asc') ? 'desc' : 'asc';
        }
        tbl.dataset.sortBy = by;
        tbl.dataset.sortDir = dir;
        sortRows(by, dir);
      });
    } catch (e) {
      container.textContent = 'Ошибка загрузки';
    }
  }

  async function renderOverlay(domain, queryLabel) {
    lastQueryLabel = queryLabel || '';
    const runA = document.getElementById('runASelect').textContent;
    const runB = document.getElementById('runBSelect').textContent;
    const align = 'offset';
    // heuristic seriesKey
    let seriesKey = 'application';
    if (domain === 'kafka') seriesKey = 'client_id';
    if (domain === 'database') seriesKey = 'service';
    if (domain === 'hard_resources') seriesKey = 'node';
    const u = new URL('/compare_series', location.origin);
    u.searchParams.set('run_a', runA);
    u.searchParams.set('run_b', runB);
    u.searchParams.set('domain', domain);
    u.searchParams.set('query_label', queryLabel);
    u.searchParams.set('series_key', seriesKey);
    u.searchParams.set('align', align);
    const resp = await fetch(u);
    const data = await resp.json();
    if (!data || !data.points || !data.points.length) {
      try { document.getElementById('legend').innerHTML = 'Нет данных для выбранной метрики'; } catch (e) {}
      if (chart) { try { chart.destroy(); } catch (e) {} }
      return;
    }
    const keyT = 't_offset_sec';
    const xMap = new Map();
    const seriesMap = new Map();
    data.points.forEach((p) => {
      const x = p[keyT];
      xMap.set(x, true);
      const k = `${p.run_name} / ${p.series}`;
      if (!seriesMap.has(k)) seriesMap.set(k, new Map());
      seriesMap.get(k).set(x, p.value);
    });
    const labels = Array.from(xMap.keys()).sort((a, b) => (parseInt(a, 10) || 0) - (parseInt(b, 10) || 0));
    const datasets = [];
    seriesMap.forEach((m, name) => {
      const color = randColor();
      const arr = labels.map((x) => (m.has(x) ? m.get(x) : null));
      const isRunB = (typeof name === 'string') && name.startsWith((runB || '') + ' /');
      const ds = { label: name, data: arr, borderColor: color, backgroundColor: color, pointRadius: 0, borderWidth: 2, spanGaps: true };
      if (isRunB) ds.borderDash = [6, 4];
      datasets.push(ds);
    });
    const ctx = document.getElementById('chart').getContext('2d');
    if (chart) chart.destroy();
    // eslint-disable-next-line no-undef
    chart = new Chart(ctx, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        interaction: { mode: 'nearest', intersect: false },
        scales: {
          x: {
            title: { display: true, text: 'Время от старта (чч:мм)', color: '#ccc' },
            ticks: {
              color: '#bbb',
              callback(value) {
                const raw = this.getLabelForValue(value);
                const sec = parseInt(raw, 10) || 0;
                const h = Math.floor(sec / 3600);
                const m = Math.floor((sec % 3600) / 60);
                const hh = String(h).padStart(2, '0');
                const mm = String(m).padStart(2, '0');
                return `${hh}:${mm}`;
              }
            },
            grid: { color: '#2f2f2f', drawBorder: true, borderColor: '#444' }
          },
          y: {
            title: { display: true, text: queryLabel, color: '#ccc' },
            ticks: { color: '#bbb' },
            grid: { color: '#2f2f2f', drawBorder: true, borderColor: '#444' }
          }
        },
        plugins: {
          legend: { display: false },
          customBackground: { color: '#151515' }
        }
      }
    });
    syncLegendHeight();
    renderLegendTable(chart);
  }

  async function loadDomainSummary(domain, container) {
    try {
      container.innerHTML = 'Загрузка сводки…';
      const runA = document.getElementById('runASelect').textContent;
      const runB = document.getElementById('runBSelect').textContent;
      const u = new URL('/compare_summary', location.origin);
      u.searchParams.set('run_a', runA);
      u.searchParams.set('run_b', runB);
      u.searchParams.set('domain', domain);
      const resp = await fetch(u);
      const arr = await resp.json();
      const rows = Array.isArray(arr) ? arr : [];
      const tbl = document.createElement('table'); tbl.className = 'cmp-summary';
      tbl.innerHTML = '<thead><tr><th data-sort=\"ql\">Метрика</th><th data-sort=\"a\">Тест A (P95)</th><th data-sort=\"b\">Тест B (P95)</th><th data-sort=\"trend\">Тенденция</th></tr></thead>';
      const tb = document.createElement('tbody');
      rows.forEach((r) => {
        const tr = document.createElement('tr');
        const a = (typeof r.p95_a === 'number' && isFinite(r.p95_a)) ? r.p95_a : null;
        const b = (typeof r.p95_b === 'number' && isFinite(r.p95_b)) ? r.p95_b : null;
        const fmt = (v) => (typeof v === 'number' && isFinite(v)) ? v.toFixed(2) : '—';
        let trendStr = '—'; let cls = '';
        if (typeof r.trend_pct === 'number' && isFinite(r.trend_pct)) {
          const sign = r.trend_pct >= 0 ? 1 : -1;
          cls = (sign > 0) ? 'up' : 'down';
          const abs = Math.abs(r.trend_pct).toFixed(1);
          const arrow = (sign > 0) ? '↑' : '↓';
          trendStr = `${arrow} ${abs}%`;
        }
        tr.innerHTML = `<td>${r.query_label}</td><td>${fmt(a)}</td><td>${fmt(b)}</td><td class=\"trend ${cls}\">${trendStr}</td>`;
        tr.dataset.ql = r.query_label || '';
        tr.dataset.a = (a != null && isFinite(a)) ? String(a) : '';
        tr.dataset.b = (b != null && isFinite(b)) ? String(b) : '';
        tr.dataset.trend = (typeof r.trend_pct === 'number' && isFinite(r.trend_pct)) ? String(r.trend_pct) : '';
        tb.appendChild(tr);
      });
      tbl.appendChild(tb);
      container.innerHTML = '';
      container.appendChild(tbl);
      const theadEl = tbl.querySelector('thead');
      tbl.dataset.sortBy = tbl.dataset.sortBy || 'ql';
      tbl.dataset.sortDir = tbl.dataset.sortDir || 'asc';
      function sortRows(by, dir) {
        const rowsArr = Array.from(tb.querySelectorAll('tr'));
        function val(tr) {
          if (by === 'ql') return (tr.dataset.ql || '').toLowerCase();
          const num = parseFloat(tr.dataset[by] || '');
          return Number.isFinite(num) ? num : null;
        }
        rowsArr.sort((r1, r2) => {
          const v1 = val(r1); const v2 = val(r2);
          if (by === 'ql') {
            const cmp = v1.localeCompare(v2);
            return dir === 'asc' ? cmp : -cmp;
          } else {
            if (v1 == null && v2 == null) return 0;
            if (v1 == null) return 1;
            if (v2 == null) return -1;
            return dir === 'asc' ? (v1 - v2) : (v2 - v1);
          }
        });
        tb.innerHTML = '';
        rowsArr.forEach((tr) => tb.appendChild(tr));
      }
      theadEl.addEventListener('click', (e) => {
        const th = e.target.closest('[data-sort]');
        if (!th) return;
        const by = th.getAttribute('data-sort');
        let dir = 'asc';
        if (tbl.dataset.sortBy === by) {
          dir = (tbl.dataset.sortDir === 'asc') ? 'desc' : 'asc';
        }
        tbl.dataset.sortBy = by;
        tbl.dataset.sortDir = dir;
        sortRows(by, dir);
      });
    } catch (e) {
      container.innerHTML = 'Не удалось загрузить сводку';
    }
  }

  function renderLegendTable(c) {
    const box = document.getElementById('legend');
    if (!box || !c) return;
    box.innerHTML = '';
    const tbl = document.createElement('table');
    tbl.className = 'legend-table';
    const thead = document.createElement('thead');
    thead.innerHTML = '<tr><th data-sort=\"name\" style=\"cursor:pointer\">Серия</th><th>Цвет</th><th data-sort=\"avg\" style=\"cursor:pointer\">Среднее</th><th>Вкл</th></tr>';
    tbl.appendChild(thead);
    const tbody = document.createElement('tbody');
    const rows = c.data.datasets.map((ds, i) => {
      const vals = (ds.data || []).filter((v) => v != null && !isNaN(v));
      const avg = vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length) : 0;
      return { idx: i, label: ds.label, color: ds.borderColor, avg, visible: c.isDatasetVisible(i) };
    });
    rows.sort((a, b) => {
      if (legendSortBy === 'name') {
        return legendSortDir === 'asc' ? a.label.localeCompare(b.label) : b.label.localeCompare(a.label);
      }
      return legendSortDir === 'asc' ? (a.avg - b.avg) : (b.avg - a.avg);
    });
    rows.forEach((row) => {
      const tr = document.createElement('tr');
      tr.className = 'legend-row' + (row.visible ? '' : ' hidden');
      const tdName = document.createElement('td'); tdName.textContent = row.label;
      const tdColor = document.createElement('td'); const sw = document.createElement('span'); sw.className = 'legend-color'; sw.style.background = row.color; tdColor.appendChild(sw);
      const tdAvg = document.createElement('td'); tdAvg.textContent = (isFinite(row.avg) ? row.avg.toFixed(2) : '—');
      const tdToggle = document.createElement('td'); tdToggle.textContent = row.visible ? '✓' : '✕';
      tr.appendChild(tdName); tr.appendChild(tdColor); tr.appendChild(tdAvg); tr.appendChild(tdToggle);
      tr.addEventListener('click', () => {
        const vis = c.isDatasetVisible(row.idx);
        c.setDatasetVisibility(row.idx, !vis);
        c.update();
        tr.classList.toggle('hidden', vis);
        tdToggle.textContent = vis ? '✕' : '✓';
      });
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody);
    box.appendChild(tbl);
    thead.addEventListener('click', (e) => {
      const th = e.target.closest('[data-sort]');
      if (!th) return;
      const by = th.getAttribute('data-sort');
      if (legendSortBy === by) {
        legendSortDir = (legendSortDir === 'asc') ? 'desc' : 'asc';
      } else {
        legendSortBy = by;
        legendSortDir = (by === 'avg') ? 'desc' : 'asc';
      }
      renderLegendTable(c);
    });
    const hideBtn = document.getElementById('legendHideAll');
    const showBtn = document.getElementById('legendShowAll');
    if (hideBtn) hideBtn.onclick = () => {
      for (let i = 0; i < c.data.datasets.length; i++) c.setDatasetVisibility(i, false);
      c.update(); renderLegendTable(c);
    };
    if (showBtn) showBtn.onclick = () => {
      for (let i = 0; i < c.data.datasets.length; i++) c.setDatasetVisibility(i, true);
      c.update(); renderLegendTable(c);
    };
  }

  function syncLegendHeight() {
    try {
      const canvas = document.getElementById('chart');
      const legend = document.querySelector('.legend-panel');
      if (!canvas || !legend) return;
      const doSync = () => {
        legend.style.height = 'auto';
        const rect = canvas.getBoundingClientRect();
        const attrH = (canvas.height || parseInt(canvas.getAttribute('height') || '0', 10)) || 0;
        // Если панель скрыта и rect.height = 0, используем атрибут или опорную высоту
        let h = Math.max(Math.floor((rect && rect.height) || 0), attrH);
        if (!h || h < 30) {
          // опорная высота — первая видимая легенда или дефолт
          const visibleCanvas = Array.from(document.querySelectorAll('.chart-canvas canvas')).find(c => c && c.offsetParent !== null);
          if (visibleCanvas) {
            const r2 = visibleCanvas.getBoundingClientRect();
            const a2 = (visibleCanvas.height || parseInt(visibleCanvas.getAttribute('height') || '0', 10)) || 0;
            h = Math.max(Math.floor((r2 && r2.height) || 0), a2, h);
          }
        }
        h = Math.max(160, h || 0);
        legend.style.height = h + 'px';
      };
      doSync();
      if (window.requestAnimationFrame) {
        requestAnimationFrame(() => doSync());
        requestAnimationFrame(() => requestAnimationFrame(() => doSync()));
      }
      setTimeout(doSync, 60);
      setTimeout(doSync, 120);
      setTimeout(doSync, 250);
    } catch (e) {}
  }
  window.addEventListener('resize', syncLegendHeight);

  function bindSelectors() {
    const aSel = document.getElementById('runASelect');
    const bSel = document.getElementById('runBSelect');
    const aWrap = document.getElementById('runAWrapper');
    const bWrap = document.getElementById('runBWrapper');
    const aOpts = document.getElementById('runAOptions');
    const bOpts = document.getElementById('runBOptions');
    if (aSel) aSel.addEventListener('click', (e) => { e.stopPropagation(); if (aWrap) aWrap.classList.toggle('open'); });
    if (bSel) bSel.addEventListener('click', (e) => { e.stopPropagation(); if (bWrap) bWrap.classList.toggle('open'); });
    if (aOpts) aOpts.addEventListener('click', (e) => {
      const opt = e.target.closest('.custom-option'); if (!opt) return;
      const v = opt.dataset.value || opt.textContent;
      if (v && aSel) aSel.textContent = v;
      if (aWrap) aWrap.classList.remove('open');
    });
    if (bOpts) bOpts.addEventListener('click', (e) => {
      const opt = e.target.closest('.custom-option'); if (!opt) return;
      const v = opt.dataset.value || opt.textContent;
      if (v && bSel) bSel.textContent = v;
      if (bWrap) bWrap.classList.remove('open');
    });
    window.addEventListener('click', (e) => {
      if (aWrap && !aWrap.contains(e.target)) aWrap.classList.remove('open');
      if (bWrap && !bWrap.contains(e.target)) bWrap.classList.remove('open');
    });
  }

  function wireCompareButton() {
    const btn = document.getElementById('compareBtn');
    if (!btn) return;
    btn.addEventListener('click', async () => {
      try { btn.classList.add('active'); } catch (e) {}
      await loadSchema();
      try {
        document.querySelectorAll('.acc-content').forEach(async (content) => {
          if (content && content.style.display === 'block') {
            const domain = content.dataset ? content.dataset.domain : null;
            const wrap = content.querySelector('.cmp-summary-wrap');
            if (domain && wrap) { await loadDomainSummary(domain, wrap); }
            try {
              content.querySelectorAll('.metric-item').forEach(async (m) => {
                const body = m.querySelector('.metric-content');
                const title = m.querySelector('.metric-header');
                if (body && body.style.display === 'block' && title) {
                  const ql = title.firstChild ? title.firstChild.textContent : '';
                  if (ql) { await loadMetricSeriesSummary(domain, ql, body); }
                }
              });
            } catch (e) {}
          }
        });
      } catch (e) {}
      setTimeout(() => { try { btn.classList.remove('active'); } catch (e) {} }, 300);
    });
  }

  function wireDownload() {
    const btn = document.getElementById('legendDownloadPng');
    if (!btn) return;
    btn.addEventListener('click', () => {
      if (!chart) return;
      // Create downloadable image with legend rows
      const canvas = chart.canvas;
      const chartW = canvas.width;
      const chartH = canvas.height;
      const rows = (chart.data?.datasets || []).map((ds, i) => {
        const vals = (ds.data || []).filter((v) => v != null && !isNaN(v));
        const avg = vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length) : 0;
        return { idx: i, label: ds.label, color: ds.borderColor, avg, visible: chart.isDatasetVisible(i) };
      }).filter((r) => r.visible);
      const pad = 16; const rowH = 24; const titleH = 22; const hdrH = rows.length ? (titleH + 10) : 0;
      const legendH = rows.length ? (hdrH + rows.length * rowH + pad) : 0;
      const outH = chartH + (legendH ? (legendH + pad) : 0);
      const out = document.createElement('canvas');
      out.width = chartW; out.height = outH;
      const ctx = out.getContext('2d');
      ctx.fillStyle = '#0f0f0f';
      ctx.fillRect(0, 0, out.width, out.height);
      ctx.drawImage(canvas, 0, 0);
      if (rows.length) {
        let y = chartH + pad;
        ctx.fillStyle = '#ddd'; ctx.font = '16px Montserrat, Arial, sans-serif';
        ctx.fillText('Легенда', pad, y);
        y += titleH;
        ctx.font = '13px Montserrat, Arial, sans-serif';
        rows.forEach((r) => {
          ctx.fillStyle = r.color || '#888';
          ctx.fillRect(pad, y - 12, 14, 14);
          ctx.strokeStyle = '#444'; ctx.strokeRect(pad, y - 12, 14, 14);
          ctx.fillStyle = '#ddd';
          const label = String(r.label || '');
          ctx.fillText(label, pad + 20, y);
          const avgStr = Number.isFinite(r.avg) ? r.avg.toFixed(2) : '—';
          const right = out.width - pad;
          const text = `Среднее: ${avgStr}`;
          const tw = ctx.measureText(text).width;
          ctx.fillText(text, right - tw, y);
          y += rowH;
        });
      }
      const url = out.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = url; a.download = `compare-${(lastQueryLabel || 'chart')}.png`;
      document.body.appendChild(a); a.click(); a.remove();
    });
  }

  document.addEventListener('DOMContentLoaded', async () => {
    try { await window.LoadLens.initProjectArea(); } catch (e) {}
    await loadRuns();
    await loadSchema();
    // Prefill from query string
    try {
      const p = new URLSearchParams(location.search);
      const ra = p.get('run_a'); const rb = p.get('run_b');
      if (ra) { const a = document.getElementById('runASelect'); if (a) a.textContent = ra; }
      if (rb) { const b = document.getElementById('runBSelect'); if (b) b.textContent = rb; }
    } catch (e) {}
    bindSelectors();
    wireCompareButton();
    wireDownload();
  });
})();


