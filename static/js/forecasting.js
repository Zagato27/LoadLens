// SPDX-License-Identifier: Apache-2.0
(function () {
  const LL = window.LoadLens;
  // Columns of the reports table in forecasting.html.
  const COLUMNS = 6;

  // Why the other step and stress tests have no forecast: missing series (queries of the load testing tool
  // domain) and missing forecast settings; '' when every such test has one.
  function hintText(missingSeries, missingSettings) {
    const total = missingSeries + missingSettings;
    if (!total) return '';
    const parts = [];
    if (missingSeries) parts.push(LL.t('forecast.list.hint_missing_series', { count: missingSeries, domain: LL.domainTitle('lt_framework') }));
    if (missingSettings) parts.push(LL.t('forecast.list.hint_missing_settings', { count: missingSettings }));
    return LL.tn('forecast.list.hint', total, { reasons: parts.join(', ') });
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
      body.innerHTML = LL.ui.emptyRow(COLUMNS, LL.t('forecast.list.empty'), {
        icon: 'trending',
        actionHtml: `<a class="btn btn-primary" href="/new" data-min-role="engineer">${LL.ui.escapeHtml(LL.t('forecast.list.new_report'))}</a>`
      });
      return;
    }
    reports.forEach((report) => {
      const row = document.createElement('tr');
      row.className = 'data-row';
      const name = document.createElement('td');
      name.className = 'run-name';
      name.title = report.run_name || '';
      // A real link: keyboard focus, middle click and «open in new tab» work; the rest of the row opens it too.
      const link = document.createElement('a');
      link.className = 'run-link';
      link.href = '/forecasting/' + encodeURIComponent(report.run_id);
      link.textContent = report.run_name || '—';
      name.appendChild(link);
      row.appendChild(name);
      const service = document.createElement('td');
      service.textContent = report.service || '—';
      row.appendChild(service);
      const type = document.createElement('td');
      type.className = 'muted';
      type.textContent = LL.testTypeLabel(report.test_type);
      row.appendChild(type);
      const created = document.createElement('td');
      created.className = 'm muted nowrap';
      created.textContent = report.created_at ? LL.formatDateTime(report.created_at) : '—';
      row.appendChild(created);
      const verdictCell = document.createElement('td');
      const pill = document.createElement('span');
      pill.className = `pill ${LL.verdictClass(report.verdict)}`;
      pill.textContent = LL.verdictLabel(report.verdict);
      verdictCell.appendChild(pill);
      row.appendChild(verdictCell);
      const model = document.createElement('td');
      model.textContent = LL.t(report.load_model === 'closed' ? 'forecast.list.model_closed' : 'forecast.list.model_open');
      row.appendChild(model);
      row.addEventListener('click', (event) => {
        if (event.target.closest && event.target.closest('a')) return;
        location.href = '/forecasting/' + encodeURIComponent(report.run_id);
      });
      body.appendChild(row);
    });
  }

  LL.forecasting = { _internals: { hintText, renderRows } };

  document.addEventListener('DOMContentLoaded', async () => {
    const body = document.getElementById('forecastRows');
    body.innerHTML = LL.ui.skeletonRows(COLUMNS, 4);
    try {
      const resp = await fetch('/forecast_reports');
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || LL.t('forecast.list.load_failed'));
      renderRows(Array.isArray(data.reports) ? data.reports : []);
      showHint(hintText(Number(data.missing_series) || 0, Number(data.missing_settings) || 0));
    } catch (error) {
      body.innerHTML = '';
      showHint('');
      document.getElementById('forecastHint').hidden = false;
      document.getElementById('forecastHint').innerHTML = `<div class="report-note-title">${LL.ui.escapeHtml(LL.t('forecast.list.unavailable'))}</div><p>${LL.ui.escapeHtml(error.message)}</p>`;
    }
  });
})();
