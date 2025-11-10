// Reports page logic
(function () {
  // Chart background plugin
  const cmpBgPlugin = { id: 'cmpBg', beforeDraw(chart, args, opts) { const { ctx, chartArea } = chart; if (!chartArea) return; ctx.save(); ctx.fillStyle = (opts && opts.color) || '#151515'; ctx.fillRect(chartArea.left, chartArea.top, chartArea.right - chartArea.left, chartArea.bottom - chartArea.top); ctx.restore(); } };
  if (window.Chart && Chart.register) Chart.register(cmpBgPlugin);

  function randColor(alpha = 0.7) {
    if (window.LoadLens && typeof window.LoadLens.randColor === 'function') {
      return window.LoadLens.randColor(alpha);
    }
    const r = Math.floor(100 + Math.random() * 155);
    const g = Math.floor(100 + Math.random() * 155);
    const b = Math.floor(100 + Math.random() * 155);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  function getRunFromPath() {
    try {
      const parts = location.pathname.split('/').filter(Boolean);
      const idx = parts.indexOf('reports');
      if (idx >= 0) {
        if (parts[idx + 2]) return decodeURIComponent(parts[idx + 2]);
        if (parts[idx + 1]) return decodeURIComponent(parts[idx + 1]);
      }
    } catch (e) {}
    return null;
  }

  async function reportsLoadSchema() {
    try {
      const r = await fetch('/domains_schema');
      return await r.json();
    } catch (e) {
      return {};
    }
  }

  function safeStr(x) { return (x === undefined || x === null) ? '' : String(x).trim(); }
  function pct(x) { try { if (x === undefined || x === null) return '—'; const v = Number(x); if (!isFinite(v)) return '—'; return `${Math.round(v * 100)}%`; } catch (e) { return '—'; } }
  function renderLlmParsedToMarkdown(report, scores) {
    const standardizeVerdict = (vRaw) => {
      const v = (safeStr(vRaw) || '').toLowerCase();
      if (!v) return 'Недостаточно данных';
      const ok = ['ok', 'okay', 'успех', 'успешно', 'success', 'passed', 'green'];
      const warn = ['warn', 'warning', 'есть риски', 'риски', 'risk', 'risks', 'degrad', 'degraded', 'предупреждение'];
      const crit = ['critical', 'критично', 'fail', 'failed', 'ошибка', 'error', 'red', 'провал'];
      const na = ['insufficient', 'нет данных', 'недостаточно', 'no data', 'unknown', 'n/a'];
      if (ok.some(x => v.includes(x))) return 'Успешно';
      if (warn.some(x => v.includes(x))) return 'Есть риски';
      if (crit.some(x => v.includes(x))) return 'Провал';
      if (na.some(x => v.includes(x))) return 'Недостаточно данных';
      return 'Недостаточно данных';
    };
    const lines = [];
    const verdict = standardizeVerdict((report || {}).verdict || '');
    let judgeOverall = undefined;
    try { if (scores && scores.judge && typeof scores.judge.overall === 'number') { judgeOverall = scores.judge.overall; } } catch (e) {}
    const confStr = (typeof judgeOverall === 'number' && isFinite(judgeOverall)) ? `${Math.round(judgeOverall * 100)}%` : '—';
    lines.push('### Итог LLM');
    lines.push(`- Вердикт: ${verdict}`);
    lines.push(`- Доверие: ${confStr}`);
    lines.push('');
    const peak = (report || {}).peak_performance || (report || {}).peak_perfomance;
    if (peak && typeof peak === 'object') {
      const max_rps = safeStr(peak.max_rps), max_time = safeStr(peak.max_time), drop_time = safeStr(peak.drop_time), method = safeStr(peak.method);
      if (max_rps || max_time || drop_time || method) {
        lines.push('#### Пиковая производительность');
        if (max_rps) lines.push(`- Максимальный RPS: ${max_rps}`);
        if (max_time) lines.push(`- Время пика: ${max_time}`);
        if (drop_time) lines.push(`- Время деградации: ${drop_time}`);
        if (method) lines.push(`- Метод оценки: ${method}`);
        lines.push('');
      }
    }
    const findings = (report || {}).findings || [];
    lines.push('#### Ключевые находки');
    if (!findings.length) {
      lines.push('- Нет существенных находок');
    } else {
      findings.forEach((f) => {
        if (f && typeof f === 'object') {
          const summary = safeStr(f.summary);
          const sev = safeStr(f.severity), comp = safeStr(f.component), ev = safeStr(f.evidence);
          const meta = []; if (sev) meta.push(`severity: ${sev}`); if (comp) meta.push(`component: ${comp}`); if (ev) meta.push(`evidence: ${ev}`);
          const metaStr = meta.join('; ');
          lines.push(metaStr ? `- ${summary} (${metaStr})` : `- ${summary}`);
        } else {
          lines.push(`- ${safeStr(f)}`);
        }
      });
    }
    const actions = (report || {}).recommended_actions || (report || {}).actions || [];
    lines.push('');
    lines.push('#### Рекомендации');
    if (!actions.length) {
      lines.push('- Нет рекомендаций');
    } else {
      actions.forEach((a) => { const s = safeStr(a); if (s) lines.push(`- ${s}`); });
    }
    const affected = (report || {}).affected_components || [];
    if (affected && affected.length) {
      lines.push('');
      lines.push('#### Затронутые компоненты');
      lines.push(affected.map((x) => `\`${safeStr(x)}\``).join(', '));
    }
    return lines.join('\n');
  }
  function judgeMarkdown(scores) {
    try {
      const s = scores || {};
      const j = (s.judge) || {};
      const overall = j.overall, factual = j.factual, completeness = j.completeness, specificity = j.specificity;
      const dataScore = s.data_score, finalScore = s.final_score, conf = s.confidence;
      const lines = ['', '#### Доверие (судья)'];
      lines.push(`- Итог: ${pct(overall)}`);
      lines.push(`- Согласованность (factual): ${pct(factual)}`);
      lines.push(`- Полнота (completeness): ${pct(completeness)}`);
      lines.push(`- Конкретика (specificity): ${pct(specificity)}`);
      lines.push(`- По данным: ${pct(dataScore)}`);
      lines.push(`- Агрегат: ${pct(finalScore)}`);
      if (typeof conf === 'number') lines.push(`- Доверие модели: ${pct(conf)}`);
      lines.push('');
      lines.push('_Пояснения: итог = 0.6·factual + 0.3·completeness + 0.2·specificity._');
      return '\n' + lines.join('\n');
    } catch (e) { return ''; }
  }

  async function reportsLoadLlm() {
    const run = getRunFromPath(); if (!run) return;
    const box = document.getElementById('rep-llm-tabs'); if (!box) return;
    box.textContent = 'Загрузка…';
    const resp = await fetch('/llm_reports?run_name=' + encodeURIComponent(run));
    let arr = await resp.json();
    try {
      const title = document.getElementById('pageTitle');
      if (title) title.textContent = `Отчет по тесту ${run}`;
      const list = (Array.isArray(arr) ? arr : []);
      const starts = list.map((x) => parseInt(x.start_ms, 10)).filter((v) => Number.isFinite(v));
      const ends = list.map((x) => parseInt(x.end_ms, 10)).filter((v) => Number.isFinite(v));
      let startStr = '—', endStr = '—';
      if (starts.length) {
        const ms = Math.min.apply(null, starts);
        const d = new Date(ms);
        const yyyy = d.getFullYear();
        const MM = String(d.getMonth() + 1).padStart(2, '0');
        const dd = String(d.getDate()).padStart(2, '0');
        const hh = String(d.getHours()).padStart(2, '0');
        const mm = String(d.getMinutes()).padStart(2, '0');
        startStr = `${yyyy}-${MM}-${dd} ${hh}:${mm}`;
      }
      if (ends.length) {
        const me = Math.max.apply(null, ends);
        const de = new Date(me);
        const yyyy = de.getFullYear();
        const MM = String(de.getMonth() + 1).padStart(2, '0');
        const dd = String(de.getDate()).padStart(2, '0');
        const hh = String(de.getHours()).padStart(2, '0');
        const mm = String(de.getMinutes()).padStart(2, '0');
        endStr = `${yyyy}-${MM}-${dd} ${hh}:${mm}`;
      }
      const range = document.getElementById('pageTimeRange');
      if (range) range.textContent = `Время теста: ${startStr} - ${endStr}`;
    } catch (e) {}
    if (!Array.isArray(arr)) { box.textContent = 'Нет данных'; return; }
    arr = arr.filter((x) => (x && x.domain && x.domain !== 'engineer'));
    const domainsOrder = ['final', 'jvm', 'database', 'kafka', 'microservices', 'hard_resources', 'lt_framework'];
    arr.sort((a, b) => domainsOrder.indexOf(a.domain) - domainsOrder.indexOf(b.domain));
    const tabsNav = document.createElement('div'); tabsNav.className = 'app-nav-inner';
    const tabsBody = document.createElement('div');
    const idBase = 'rep-llm-tab-';
    arr.forEach((x, idx) => {
      const btn = document.createElement('button'); btn.className = 'nav-btn' + (idx === 0 ? ' active' : ''); btn.textContent = (x.domain === 'final' ? 'Итог' : x.domain);
      btn.dataset.target = idBase + idx;
      tabsNav.appendChild(btn);
      const pane = document.createElement('div'); pane.id = idBase + idx; pane.className = 'panel' + (idx === 0 ? ' active' : '');
      let md = '';
      let sc = x ? x.scores : null; if (sc && typeof sc === 'string') { try { sc = JSON.parse(sc); } catch (e) {} }
      try {
        let parsed = x ? x.parsed : null;
        if (parsed && typeof parsed === 'string') { try { parsed = JSON.parse(parsed); } catch (e) {} }
        if (parsed && typeof parsed === 'object') {
          md = renderLlmParsedToMarkdown(parsed, sc);
        } else {
          const raw = String(x && x.text ? x.text : '');
          let fallbackParsed = null;
          if (raw.trim().startsWith('{')) { try { fallbackParsed = JSON.parse(raw); } catch (e) {} }
          if (!fallbackParsed && raw.includes('\"verdict\"')) {
            try { const start = raw.indexOf('{'); const end = raw.lastIndexOf('}'); if (start >= 0 && end > start) { fallbackParsed = JSON.parse(raw.slice(start, end + 1)); } } catch (e) {}
          }
          md = (fallbackParsed && typeof fallbackParsed === 'object') ? renderLlmParsedToMarkdown(fallbackParsed, sc) : raw;
        }
      } catch (e) { md = String(x && x.text ? x.text : ''); }
      try { md += judgeMarkdown(sc); } catch (e) {}
      let html = (window.marked && typeof marked.parse === 'function') ? window.marked.parse(md) : md;
      try { if (window.DOMPurify) html = window.DOMPurify.sanitize(html); } catch (e) {}
      pane.innerHTML = `<div style=\"background:#1e1e1e; border:1px solid #333; border-radius:6px; padding:12px;\">${html}</div>`;
      tabsBody.appendChild(pane);
    });
    box.innerHTML = ''; box.appendChild(tabsNav); box.appendChild(tabsBody);
    tabsNav.querySelectorAll('.nav-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        tabsNav.querySelectorAll('.nav-btn').forEach((b) => b.classList.remove('active'));
        tabsBody.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
        btn.classList.add('active');
        const t = btn.dataset.target; const pane = tabsBody.querySelector('#' + t); if (pane) pane.classList.add('active');
        scheduleLegendSync();
      });
    });
  }

  async function reportsDrawOne(run, domain, ql, canvasId, legendRoot) {
    const u = new URL('/run_series', location.origin);
    u.searchParams.set('run_name', run);
    u.searchParams.set('domain', domain);
    u.searchParams.set('query_label', ql);
    u.searchParams.set('series_key', 'auto');
    u.searchParams.set('align', 'absolute');
    const resp = await fetch(u);
    const data = await resp.json();
    if (!data || !data.points || !data.points.length) {
      try { const tbl = legendRoot.querySelector('.table'); if (tbl) tbl.innerHTML = 'Нет данных'; } catch (e) {}
      return;
    }
    const labels = []; const map = {};
    data.points.forEach((p) => {
      const t = p.t;
      if (labels.indexOf(t) < 0) labels.push(t);
      const k = p.series; if (!map[k]) map[k] = new Map();
      map[k].set(t, p.value);
    });
    labels.sort((a, b) => new Date(a) - new Date(b));
    const datasets = Object.keys(map).map((k) => { const color = randColor(); return { label: k, data: labels.map((t) => (map[k].has(t) ? map[k].get(t) : null)), borderColor: color, backgroundColor: color, pointRadius: 0, borderWidth: 2, spanGaps: true }; });
    const ctx = document.getElementById(canvasId).getContext('2d');
    // eslint-disable-next-line no-undef
    const chart = new Chart(ctx, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        interaction: { mode: 'nearest', intersect: false },
        plugins: { legend: { display: false }, cmpBg: { color: '#151515' } },
        scales: {
          x: {
            title: { display: true, text: 'Время', color: '#ccc' },
            ticks: {
              color: '#bbb',
              callback(value) {
                const raw = this.getLabelForValue(value);
                const d = new Date(raw);
                const hh = String(d.getHours()).padStart(2, '0');
                const mm = String(d.getMinutes()).padStart(2, '0');
                return `${hh}:${mm}`;
              }
            },
            grid: { color: '#2f2f2f', drawBorder: true, borderColor: '#444' }
          },
          y: { ticks: { color: '#bbb' }, grid: { color: '#2f2f2f', drawBorder: true, borderColor: '#444' } }
        }
      }
    });
    try { legendRoot.dataset.domain = domain; legendRoot.dataset.queryLabel = ql; } catch (e) {}
    buildLegendFor(chart, legendRoot);
    try {
      const canvasEl = document.getElementById(canvasId);
      legendRoot.style.height = 'auto';
      const rect = canvasEl.getBoundingClientRect();
      const attrH = (canvasEl.height || parseInt(canvasEl.getAttribute('height') || '0', 10)) || 0;
      const h = Math.max(160, Math.floor((rect && rect.height) || attrH || 0));
      legendRoot.style.height = h + 'px';
      if (window.requestAnimationFrame) requestAnimationFrame(() => {
        const rect2 = canvasEl.getBoundingClientRect();
        const attrH2 = (canvasEl.height || parseInt(canvasEl.getAttribute('height') || '0', 10)) || 0;
        const h2 = Math.max(160, Math.floor((rect2 && rect2.height) || attrH2 || 0));
        legendRoot.style.height = h2 + 'px';
      });
    } catch (e) {}
    scheduleLegendSync();
  }

  function downloadChartWithLegend(currentChart, root) {
    if (!currentChart) return null;
    const canvas = currentChart.canvas;
    const chartW = canvas.width;
    const chartH = canvas.height;
    const rows = (currentChart.data?.datasets || []).map((ds, i) => {
      const vals = (ds.data || []).filter((v) => v != null && !isNaN(v));
      const avg = vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length) : 0;
      return { idx: i, label: ds.label, color: ds.borderColor, avg, visible: currentChart.isDatasetVisible(i) };
    }).filter((r) => r.visible);
    const pad = 16, rowH = 24, titleH = 22, hdrH = rows.length ? (titleH + 10) : 0;
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
    const a = document.createElement('a');
    a.href = out.toDataURL('image/png');
    const nameParts = [];
    try { const d = root?.dataset?.domain; if (d) nameParts.push(d); const q = root?.dataset?.queryLabel; if (q) nameParts.push(q); } catch (e) {}
    a.download = `report-${(nameParts.join('-') || 'chart')}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    return true;
  }
  let repLegendSortBy = 'avg'; let repLegendSortDir = 'desc';
  function buildLegendFor(chart, root) {
    const box = root.querySelector('.table'); if (!box) return; box.innerHTML = '';
    const tbl = document.createElement('table'); tbl.className = 'cmp-legend-table';
    const thead = document.createElement('thead'); thead.innerHTML = '<tr><th data-sort=\"name\" style=\"cursor:pointer\">Серия</th><th>Цвет</th><th data-sort=\"avg\" style=\"cursor:pointer\">Среднее</th><th>Вкл</th></tr>';
    tbl.appendChild(thead); const tbody = document.createElement('tbody');
    let rows = (chart?.data?.datasets || []).map((ds, i) => {
      const vals = (ds.data || []).filter((v) => v != null && !isNaN(v));
      const avg = vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length) : 0;
      return { idx: i, label: ds.label, color: ds.borderColor, avg: avg, visible: chart.isDatasetVisible(i) };
    });
    rows.sort((a, b) => repLegendSortBy === 'name' ? (repLegendSortDir === 'asc' ? a.label.localeCompare(b.label) : b.label.localeCompare(a.label)) : (repLegendSortDir === 'asc' ? (a.avg - b.avg) : (b.avg - a.avg)));
    rows.forEach((row) => {
      const tr = document.createElement('tr'); tr.className = 'cmp-legend-row' + (row.visible ? '' : ' hidden');
      const tdName = document.createElement('td'); tdName.textContent = row.label;
      const tdColor = document.createElement('td'); const sw = document.createElement('span'); sw.className = 'cmp-legend-color'; sw.style.background = row.color; tdColor.appendChild(sw);
      const tdAvg = document.createElement('td'); tdAvg.textContent = isFinite(row.avg) ? row.avg.toFixed(2) : '—';
      const tdToggle = document.createElement('td'); tdToggle.textContent = row.visible ? '✓' : '✕';
      tr.appendChild(tdName); tr.appendChild(tdColor); tr.appendChild(tdAvg); tr.appendChild(tdToggle);
      tr.addEventListener('click', () => {
        const vis = chart.isDatasetVisible(row.idx);
        chart.setDatasetVisibility(row.idx, !vis);
        chart.update();
        buildLegendFor(chart, root);
      });
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody); box.appendChild(tbl);
    thead.addEventListener('click', (e) => {
      const th = e.target.closest('[data-sort]'); if (!th) return;
      const by = th.getAttribute('data-sort');
      if (repLegendSortBy === by) { repLegendSortDir = (repLegendSortDir === 'asc') ? 'desc' : 'asc'; }
      else { repLegendSortBy = by; repLegendSortDir = (by === 'avg') ? 'desc' : 'asc'; }
      buildLegendFor(chart, root);
    });
    const hideBtn = root.querySelector('.hideAll'); const showBtn = root.querySelector('.showAll');
    if (hideBtn) hideBtn.onclick = () => { for (let i = 0; i < chart.data.datasets.length; i++) { chart.setDatasetVisibility(i, false); } chart.update(); buildLegendFor(chart, root); };
    if (showBtn) showBtn.onclick = () => { for (let i = 0; i < chart.data.datasets.length; i++) { chart.setDatasetVisibility(i, true); } chart.update(); buildLegendFor(chart, root); };
    const dlBtn = root.querySelector('.downloadPng'); if (dlBtn) dlBtn.onclick = () => { downloadChartWithLegend(chart, root); };
  }

  function syncLegends() {
    try {
      const doSync = () => {
        // Определяем эталонную высоту по первому видимому канвасу
        let referenceH = 0;
        document.querySelectorAll('.cmp-chart-wrap').forEach((w) => {
          if (referenceH > 0) return;
          // считаем видимым, если элемент участвует в раскладке
          if (!w || w.offsetParent === null) return;
          const canvas = w.querySelector('canvas');
          if (!canvas) return;
          const rect = canvas.getBoundingClientRect();
          const attrH = (canvas.height || parseInt(canvas.getAttribute('height') || '0', 10)) || 0;
          const h = Math.max(0, Math.floor((rect && rect.height) || 0), attrH);
          if (h > 0) referenceH = h;
        });
        // Фолбэк по умолчанию
        if (referenceH <= 0) referenceH = 160;

        document.querySelectorAll('.cmp-chart-wrap').forEach((w) => {
          const canvas = w.querySelector('canvas');
          const legend = w.querySelector('.cmp-legend-panel');
          if (!canvas || !legend) return;
          legend.style.height = 'auto';
          const rect = canvas.getBoundingClientRect();
          const attrH = (canvas.height || parseInt(canvas.getAttribute('height') || '0', 10)) || 0;
          let h = Math.max(Math.floor((rect && rect.height) || 0), attrH, referenceH);
          h = Math.max(160, h);
          legend.style.height = h + 'px';
        });
      };
      doSync();
      if (window.requestAnimationFrame) {
        requestAnimationFrame(() => doSync());
        // Дополнительная попытка после финального ресайза графиков
        requestAnimationFrame(() => requestAnimationFrame(() => doSync()));
      }
      // Фолбэк таймером для случаев скрытых табов
      setTimeout(doSync, 60);
      setTimeout(doSync, 120);
      setTimeout(doSync, 250);
    } catch (e) {}
  }
  window.addEventListener('resize', syncLegends);

  // Планировщик повторного пересчёта (надёжнее при скрытых табах и ленивой отрисовке)
  function scheduleLegendSync() {
    try {
      try {
        if (window.Chart && typeof Chart.getChart === 'function') {
          document.querySelectorAll('.cmp-chart-wrap canvas').forEach((c) => {
            const inst = Chart.getChart(c);
            if (inst && typeof inst.resize === 'function') {
              try { inst.resize(); } catch (e) {}
            }
          });
        }
      } catch (e) {}
      syncLegends();
      if (window.requestAnimationFrame) {
        requestAnimationFrame(syncLegends);
        requestAnimationFrame(() => requestAnimationFrame(syncLegends));
      }
      setTimeout(syncLegends, 0);
      setTimeout(syncLegends, 60);
      setTimeout(syncLegends, 120);
      setTimeout(syncLegends, 250);
    } catch (e) {}
  }

  async function engineerLoad() {
    try {
      const run = getRunFromPath(); if (!run) return;
      const r = await fetch('/engineer_summary?run_name=' + encodeURIComponent(run));
      if (!r.ok) return;
      const j = await r.json();
      const ed = document.getElementById('engineerEditor');
      const updated = document.getElementById('engineerUpdated');
      let html = String(j.content_html || '');
      try { if (window.DOMPurify) html = window.DOMPurify.sanitize(html); } catch (e) {}
      ed.innerHTML = html || '<p style=\"color:#888\">Добавьте итоговый комментарий…</p>';
      updated.textContent = j.created_at ? ('Обновлено: ' + j.created_at.replace('T', ' ').slice(0, 16)) : '';
    } catch (e) {}
  }
  function editorExec(cmd, val) {
    try {
      if (cmd === 'h3' || cmd === 'h4') { document.execCommand('formatBlock', false, cmd.toUpperCase()); return; }
      document.execCommand(cmd, false, val || null);
    } catch (e) {}
  }
  function wireEngineerEditor() {
    const tb = document.getElementById('engineerToolbar');
    if (tb) {
      tb.addEventListener('click', (e) => {
        const btn = e.target.closest('button'); if (!btn) return;
        const cmd = btn.getAttribute('data-cmd'); if (cmd) editorExec(cmd);
      });
    }
    const linkBtn = document.getElementById('cmdLink');
    if (linkBtn) { linkBtn.addEventListener('click', () => { const url = prompt('URL ссылки:', 'https://'); if (url) { editorExec('createLink', url); } }); }
    const saveBtn = document.getElementById('saveEngineer');
    const editBtn = document.getElementById('editEngineer');
    const editor = document.getElementById('engineerEditor');
    function setEngineerEditing(on) {
      try {
        if (editor) editor.setAttribute('contenteditable', on ? 'true' : 'false');
        if (tb) tb.style.display = on ? '' : 'none';
        if (saveBtn) saveBtn.style.display = on ? '' : 'none';
        if (editBtn) editBtn.textContent = on ? 'Завершить' : 'Редактировать';
      } catch (e) {}
    }
    if (editBtn) {
      editBtn.addEventListener('click', () => {
        const isOn = editor && editor.getAttribute('contenteditable') === 'true';
        setEngineerEditing(!isOn);
      });
    }
    setEngineerEditing(false);
    if (saveBtn) {
      saveBtn.addEventListener('click', async () => {
        const run = getRunFromPath(); if (!run) return;
        const html = document.getElementById('engineerEditor').innerHTML;
        const st = document.getElementById('engineerStatus');
        st.textContent = 'Сохранение…';
        try {
          const resp = await fetch('/engineer_summary', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ run_name: run, content_html: html }) });
          const j = await resp.json();
          if (resp.ok) { st.textContent = 'Сохранено'; setEngineerEditing(false); await engineerLoad(); setTimeout(() => { st.textContent = ''; }, 1500); }
          else { st.textContent = j.error || 'Ошибка сохранения'; }
        } catch (e) { st.textContent = 'Ошибка сохранения'; }
      });
    }
  }

  async function reportsRenderDomains(schema) {
    const run = getRunFromPath(); if (!run) return;
    const root = document.getElementById('rep-domain-tabs'); if (!root) return;
    root.innerHTML = 'Загрузка…';
    const domains = Object.keys(schema || {});
    const tabsNav = document.createElement('div'); tabsNav.className = 'app-nav-inner';
    const tabsBody = document.createElement('div');
    const idBase = 'rep-dom-tab-';
    domains.forEach((domain, di) => {
      const btn = document.createElement('button'); btn.className = 'nav-btn' + (di === 0 ? ' active' : ''); btn.textContent = domain; btn.dataset.target = idBase + di; tabsNav.appendChild(btn);
      const pane = document.createElement('div'); pane.id = idBase + di; pane.className = 'panel' + (di === 0 ? ' active' : '');
      const list = (schema[domain] || []).map((x) => x.query_label);
      list.forEach((ql, qi) => {
        const wrap = document.createElement('div'); wrap.className = 'cmp-chart-wrap'; wrap.style.marginTop = '12px';
        const canvasBox = document.createElement('div'); canvasBox.className = 'cmp-chart-canvas';
        const title = document.createElement('div'); title.style.padding = '8px 0'; title.style.fontWeight = '600'; title.textContent = ql;
        const cnv = document.createElement('canvas'); cnv.id = `rep-chart-${di}-${qi}`; cnv.height = 120; canvasBox.appendChild(title); canvasBox.appendChild(cnv);
        const legend = document.createElement('div'); legend.className = 'cmp-legend-panel'; legend.innerHTML = '<h4>Легенда</h4><div class=\"cmp-legend-controls\"><button class=\"hideAll\">Выключить все</button><button class=\"showAll\">Включить все</button><button class=\"downloadPng\">Скачать PNG</button></div><div class=\"table\"></div>';
        wrap.appendChild(canvasBox); wrap.appendChild(legend); pane.appendChild(wrap);
        setTimeout(async () => { await reportsDrawOne(run, domain, ql, cnv.id, legend); }, 0);
      });
      tabsBody.appendChild(pane);
    });
    root.innerHTML = ''; root.appendChild(tabsNav); root.appendChild(tabsBody);
    tabsNav.querySelectorAll('.nav-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        tabsNav.querySelectorAll('.nav-btn').forEach((b) => b.classList.remove('active'));
        tabsBody.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
        btn.classList.add('active');
        const t = btn.dataset.target; const pane = tabsBody.querySelector('#' + t); if (pane) pane.classList.add('active');
        scheduleLegendSync();
      });
    });
    if (!domains.length) { root.textContent = 'Нет данных'; }
  }

  // Наблюдатели за переключением классов панелей (пересчёт при показе)
  function attachPanelObservers() {
    try {
      const targets = ['rep-llm-tabs', 'rep-domain-tabs'];
      targets.forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;
        const mo = new MutationObserver(() => { scheduleLegendSync(); });
        mo.observe(el, { attributes: true, subtree: true, attributeFilter: ['class'] });
      });
    } catch (e) {}
  }

  document.addEventListener('DOMContentLoaded', async () => {
    try { await window.LoadLens.initProjectArea(); } catch (e) {}
    const schema = await reportsLoadSchema();
    await reportsLoadLlm();
    await reportsRenderDomains(schema);
    wireEngineerEditor();
    await engineerLoad();
    attachPanelObservers();
    scheduleLegendSync();
  });
})();


