(function () {
  const LL = window.LoadLens;

  function hintText(missingSeries, missingSettings) {
    const total = missingSeries + missingSettings;
    if (!total) return '';
    const parts = [];
    if (missingSeries) parts.push(`${missingSeries} — нет нужных рядов в данных (добавьте запросы в «Нагрузочный инструмент» и перегенерируйте отчёт)`);
    if (missingSettings) parts.push(`${missingSettings} — не заданы настройки прогноза (SLA-критерии → Определение максимальной производительности)`);
    return `Ещё ${total} отчётов тестов на максимальную производительность пока без прогноза: ${parts.join(', ')}.`;
  }

  function showHint(text) {
    const hint = document.getElementById('forecastHint');
    if (!hint) return;
    hint.hidden = !text;
    hint.textContent = text;
  }

  function renderRows(reports) {
    const body = document.getElementById('forecastRows');
    body.innerHTML = '';
    if (!reports.length) {
      const row = document.createElement('tr');
      row.className = 'empty-row';
      const cell = document.createElement('td');
      cell.colSpan = 6;
      cell.textContent = 'Отчётов, по которым возможен прогноз, пока нет';
      row.appendChild(cell);
      body.appendChild(row);
      return;
    }
    reports.forEach((report) => {
      const row = document.createElement('tr');
      row.className = 'data-row';
      const name = document.createElement('td');
      name.className = 'run-name';
      name.textContent = report.run_name || '—';
      name.title = report.run_name || '';
      row.appendChild(name);
      const service = document.createElement('td');
      service.textContent = report.service || '—';
      row.appendChild(service);
      const type = document.createElement('td');
      type.className = 'muted';
      type.textContent = LL.testTypeLabel(report.test_type);
      row.appendChild(type);
      const created = document.createElement('td');
      created.className = 'muted nowrap';
      created.textContent = report.created_at ? LL.formatDateTime(report.created_at) : '—';
      row.appendChild(created);
      const verdictCell = document.createElement('td');
      const pill = document.createElement('span');
      pill.className = `pill ${LL.verdictClass(report.verdict)}`;
      pill.textContent = report.verdict || 'Недостаточно данных';
      verdictCell.appendChild(pill);
      row.appendChild(verdictCell);
      const model = document.createElement('td');
      model.textContent = report.load_model === 'closed' ? 'закрытая' : 'открытая';
      row.appendChild(model);
      row.addEventListener('click', () => {
        location.href = '/forecasting/' + encodeURIComponent(report.run_id);
      });
      body.appendChild(row);
    });
  }

  document.addEventListener('DOMContentLoaded', async () => {
    const body = document.getElementById('forecastRows');
    body.innerHTML = `<tr><td colspan="6">${LL.ui.skeleton(3)}</td></tr>`;
    try {
      const resp = await fetch('/forecast_reports');
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || 'Не удалось загрузить список');
      renderRows(Array.isArray(data.reports) ? data.reports : []);
      showHint(hintText(Number(data.missing_series) || 0, Number(data.missing_settings) || 0));
    } catch (error) {
      body.innerHTML = '';
      showHint('');
      document.getElementById('forecastHint').hidden = false;
      document.getElementById('forecastHint').innerHTML = `<div class="report-note-title">Список недоступен</div><p>${LL.ui.escapeHtml(error.message)}</p>`;
    }
  });
})();
