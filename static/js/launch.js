// Launch page: the form of a load test run (service, instances, report options) and the list of runs.
(function () {
  const LL = window.LoadLens;
  const ui = LL.ui;
  const $ = (id) => document.getElementById(id);
  const esc = ui.escapeHtml;
  const DRAFT_KEY = 'loadlens.launchDraft';
  const RUNS_POLL_MS = 5000;
  // From this many instances on, adding one collapses the others to their one-line summary.
  const COLLAPSE_FROM = 3;
  const CHEVRON = '<svg class="instance-chevron" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>';

  const state = { tools: {}, toolOrder: [], generators: {}, generatorOrder: [], limits: {}, scripts: {}, ciRefs: {}, seq: 0, runsTimer: null, runsSignature: '', presets: [], picker: null };
  const CI_KIND_TITLES = { gitlab: 'пайплайн GitLab', jenkins: 'сборка Jenkins' };

  const STATUS_TONE = {
    starting: 'running', running: 'running', stopping: 'warn', draining: 'running', reporting: 'running',
    done: 'done', failed: 'fail', cancelled: 'na'
  };

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function setError(text) {
    $('formError').textContent = text || '';
  }

  async function fetchJson(url, options) {
    const resp = await fetch(url, options);
    let data = {};
    try { data = await resp.json(); } catch (e) { data = {}; }
    if (!resp.ok) throw new Error(data.message || data.error || `HTTP ${resp.status}`);
    return data;
  }

  // ---- runs list -------------------------------------------------------------------------------

  function runRow(run) {
    const row = el('div', 'run-row');
    const main = el('div');
    const name = el('a', 'name', run.run_name);
    name.href = `/launch/${encodeURIComponent(run.run_id)}`;
    const tone = STATUS_TONE[run.status] || 'na';
    const pill = el('span', `pill ${tone}`, run.status_title);
    pill.style.marginLeft = '8px';
    main.append(name, pill);
    const tools = Array.from(new Set(run.instances.map((i) => i.tool_title || 'вручную'))).join(', ');
    const when = run.started_at ? LL.formatDateTime(run.started_at) : LL.formatDateTime(run.created_at);
    main.appendChild(el('div', 'meta', `${run.service} · ${run.instances.length} инст. (${tools}) · ${when} · ${run.owner || ''}${run.message ? ' · ' + run.message : ''}`));
    const link = el('a', 'btn btn-sm', run.active ? 'Открыть' : 'Подробнее');
    link.href = name.href;
    row.append(main, link);
    return row;
  }

  async function loadRuns() {
    const host = $('runsList');
    try {
      const data = await fetchJson('/load_tests/runs?limit=10');
      const runs = Array.isArray(data.runs) ? data.runs : [];
      // Redraw only when something changed, so a click on a row link is not lost to the refresh.
      const signature = JSON.stringify(runs);
      if (signature !== state.runsSignature) {
        state.runsSignature = signature;
        host.replaceChildren();
        if (!runs.length) host.appendChild(el('p', 'dim', 'Прогонов ещё не было.'));
        runs.forEach((run) => host.appendChild(runRow(run)));
      }
      const active = runs.filter((r) => r.active).length;
      $('runsNote').textContent = active ? `идёт: ${active}` : '';
      clearTimeout(state.runsTimer);
      if (active) state.runsTimer = setTimeout(loadRuns, RUNS_POLL_MS);
    } catch (e) {
      host.replaceChildren(el('p', 'dim', `Список прогонов не загрузился: ${e.message}`));
    }
  }

  // ---- CI instances: the team's pipeline with its own parameters -------------------------------

  // Values with the password parameters emptied: drafts, presets and repeats never keep them.
  function blankSecrets(values, secret) {
    const out = {};
    const hidden = new Set(secret || []);
    Object.keys(values || {}).forEach((name) => { out[name] = hidden.has(name) || values[name] === '***' ? '' : values[name]; });
    return out;
  }

  // «A=1\nB=2» from {A: '1', B: '2'}.
  function extraText(extra) {
    return Object.keys(extra || {}).map((name) => `${name}=${extra[name]}`).join('\n');
  }

  // The form part of a CI instance of a finished run (the run view masks the passwords).
  function ciSpecFromView(ci) {
    if (!ci) return {};
    return { ci_ref: ci.ref || '', ci_params: blankSecrets(ci.params, ci.secret), ci_extra: extraText(ci.extra), ci_secret: ci.secret || [] };
  }

  // The value a field shows: the one typed before, else the pipeline's default (never a password default).
  function ciFieldValue(def, values) {
    if (values && Object.prototype.hasOwnProperty.call(values, def.name)) return values[def.name];
    if (def.secret) return '';
    if (def.type === 'boolean') return def.default === true || def.default === 'true';
    return def.default === undefined || def.default === null ? '' : def.default;
  }

  // ---- instances -------------------------------------------------------------------------------

  function generatorOptions(selected) {
    return state.generatorOrder.map((id) => {
      const g = state.generators[id];
      const busy = g.busy_runs && g.busy_runs.length ? ` — идёт: ${g.busy_runs.join(', ')}` : '';
      const host = g.host ? ` (${g.host})` : '';
      const unchecked = g.verified === false ? ' · не проверен' : '';
      return `<option value="${esc(id)}"${id === selected ? ' selected' : ''}>${esc(g.title)}${esc(host)} · ${esc(g.type_title)}${esc(unchecked)}${esc(busy)}</option>`;
    }).join('');
  }

  function toolOptions(generatorId, selected) {
    const g = state.generators[generatorId];
    const allowed = g ? (g.type === 'manual' ? state.toolOrder : g.tools) : [];
    const head = g && g.type === 'manual' ? '<option value="">Не указан</option>' : '';
    return head + allowed.filter((id) => state.tools[id]).map((id) => `<option value="${esc(id)}"${id === selected ? ' selected' : ''}>${esc(state.tools[id].title)}</option>`).join('');
  }

  function scriptsKey(card) {
    return `${card.querySelector('.f-generator').value}|${card.querySelector('.f-tool').value}`;
  }

  function fillDatalist(card) {
    const list = card.querySelector('datalist');
    const scripts = state.scripts[scriptsKey(card)] || [];
    list.innerHTML = scripts.map((s) => `<option value="${esc(s)}"></option>`).join('');
  }

  async function loadScripts(card, quiet) {
    const generator = card.querySelector('.f-generator').value;
    const tool = card.querySelector('.f-tool').value;
    const note = card.querySelector('.f-scripts-note');
    const g = state.generators[generator] || {};
    if (!generator || !tool || g.type === 'manual' || g.lists_scripts === false) return;
    note.textContent = 'Ищем скрипты на генераторе…';
    try {
      const data = await fetchJson(`/load_tests/scripts?generator=${encodeURIComponent(generator)}&tool=${encodeURIComponent(tool)}`);
      state.scripts[`${generator}|${tool}`] = data.scripts || [];
      fillDatalist(card);
      note.textContent = data.message || (data.scripts && data.scripts.length
        ? `Найдено: ${data.scripts.length} в ${data.scripts_dir}`
        : `В ${data.scripts_dir || 'каталоге'} ничего не нашлось по маскам ${(state.tools[tool].scripts_glob || []).join(', ')}`);
    } catch (e) {
      note.textContent = quiet ? '' : `Список скриптов недоступен: ${e.message}`;
    }
  }

  function ciGenerator(card) {
    const g = state.generators[card.querySelector('.f-generator').value];
    return g && g.ci ? g : null;
  }

  function ciControl(def, value, id) {
    if (def.type === 'boolean') {
      const box = el('input');
      box.type = 'checkbox';
      box.id = id;
      box.checked = value === true || value === 'true';
      return box;
    }
    if (def.type === 'choice') {
      const select = el('select');
      select.id = id;
      select.multiple = !!def.multiple;
      const chosen = new Set(String(value === undefined || value === null ? '' : value).split(',').map((v) => v.trim()));
      if (!def.required && !def.multiple && !def.options.includes(String(value))) select.appendChild(el('option', '', ''));
      def.options.forEach((option) => {
        const node = el('option', '', option);
        node.value = option;
        node.selected = def.multiple ? chosen.has(option) : String(value) === option;
        select.appendChild(node);
      });
      return select;
    }
    if (def.type === 'text' || def.type === 'array') {
      const area = el('textarea', 'mono');
      area.id = id;
      area.rows = def.type === 'array' ? 1 : 2;
      area.value = String(value === undefined || value === null ? '' : value);
      if (def.type === 'array') area.placeholder = 'значения через запятую или список JSON';
      return area;
    }
    const input = el('input', def.type === 'password' ? '' : 'mono');
    input.type = def.type === 'password' ? 'password' : (def.type === 'number' ? 'number' : 'text');
    if (def.type === 'number') input.step = 'any';
    input.id = id;
    input.value = String(value === undefined || value === null ? '' : value);
    if (def.type === 'password') {
      input.autocomplete = 'new-password';
      input.placeholder = 'пусто — секрет из пайплайна';
    }
    return input;
  }

  function renderCiParams(card) {
    const host = card.querySelector('.f-ci-params');
    const data = card.ciState.defs;
    host.replaceChildren();
    if (!data || !Array.isArray(data.params) || !data.params.length) return;
    data.params.forEach((def, i) => {
      const id = `${card.querySelector('.f-name').id}-ci-${i}`;
      const field = el('div', def.type === 'text' ? 'ci-field wide-field' : 'ci-field');
      field.dataset.ciName = def.name;
      const control = ciControl(def, ciFieldValue(def, card.ciState.values), id);
      control.classList.add('f-ci-value');
      control.dataset.ciName = def.name;
      control.dataset.ciType = def.type;
      if (def.type === 'boolean') {
        const line = el('label', 'checkbox-line');
        line.htmlFor = id;
        line.append(control, el('span', 'mono', def.name));
        field.appendChild(line);
      } else {
        const label = el('label', 'mono', `${def.name}${def.required ? ' *' : ''}`);
        label.htmlFor = id;
        field.append(label, control);
      }
      const hint = [def.description, def.secret ? 'секрет: не хранится в прогоне и пресетах' : '', def.note].filter(Boolean).join(' · ');
      if (hint) field.appendChild(el('div', 'field-hint', hint));
      host.appendChild(field);
    });
  }

  // {name: value} of the rendered parameter fields; checkboxes give true/false, multi-selects «a,b».
  function readCiValues(card) {
    const values = {};
    card.querySelectorAll('.f-ci-value').forEach((control) => {
      const name = control.dataset.ciName;
      if (control.type === 'checkbox') values[name] = control.checked;
      else if (control.multiple) values[name] = Array.from(control.selectedOptions).map((o) => o.value).join(',');
      else values[name] = control.value;
    });
    return values;
  }

  function ciSecretNames(card) {
    const data = card.ciState.defs;
    return data && Array.isArray(data.params) ? data.params.filter((p) => p.secret).map((p) => p.name) : [];
  }

  async function loadRefs(card) {
    const g = ciGenerator(card);
    if (!g || !g.ci.reads_params) return;
    const list = card.querySelector('.f-ci-refs');
    if (!state.ciRefs[g.id]) {
      try {
        state.ciRefs[g.id] = await fetchJson(`/load_tests/ci_refs?generator=${encodeURIComponent(g.id)}`);
      } catch (e) {
        state.ciRefs[g.id] = { refs: [], default: g.ci.ref || '' };
      }
    }
    const refs = state.ciRefs[g.id];
    list.innerHTML = (refs.refs || []).map((ref) => `<option value="${esc(ref)}"></option>`).join('');
    const input = card.querySelector('.f-ci-ref');
    input.placeholder = refs.default ? `по умолчанию: ${refs.default}` : 'по умолчанию';
    // A plain Jenkins job has no branches; a multibranch one lists them.
    if (g.ci.kind === 'jenkins') card.querySelector('.f-ci-ref-box').style.display = (refs.refs || []).length ? '' : 'none';
  }

  async function loadCi(card, force) {
    const g = ciGenerator(card);
    if (!g) return;
    const ref = card.querySelector('.f-ci-ref').value.trim();
    const key = `${g.id}|${ref}`;
    if (!force && card.ciState.key === key) return;
    // Another branch keeps the typed values; another pipeline starts from its own defaults.
    const previous = card.ciState.key === null ? null : String(card.ciState.key).split('|')[0];
    if (previous === g.id) Object.assign(card.ciState.values, readCiValues(card));
    else if (previous !== null) card.ciState.values = {};
    card.ciState.key = key;
    const note = card.querySelector('.f-ci-note');
    if (!g.ci.reads_params) {
      card.ciState.defs = null;
      renderCiParams(card);
      note.textContent = 'Без API-токена GitLab LoadLens не видит параметры пайплайна: задайте значения дополнительными переменными.';
      return;
    }
    note.textContent = 'Читаем параметры пайплайна…';
    try {
      const data = await fetchJson(`/load_tests/ci_params?generator=${encodeURIComponent(g.id)}&ref=${encodeURIComponent(ref)}`);
      if (card.ciState.key !== key) return;
      card.ciState.defs = data;
      renderCiParams(card);
      const parts = [];
      if (data.ref) parts.push(`Ветка ${data.ref}.`);
      parts.push(data.params && data.params.length ? `Параметров: ${data.params.length}.` : 'Пайплайн не объявляет параметров.');
      if (data.auto && data.auto.length) parts.push(`LoadLens заполнит сам: ${data.auto.join(', ')}.`);
      (data.notes || []).forEach((n) => parts.push(n));
      note.textContent = parts.join(' ');
    } catch (e) {
      if (card.ciState.key !== key) return;
      card.ciState.defs = null;
      renderCiParams(card);
      note.textContent = `Параметры пайплайна не прочитаны: ${e.message}`;
    }
  }

  function syncCard(card) {
    const generatorId = card.querySelector('.f-generator').value;
    const g = state.generators[generatorId];
    const manual = !!(g && g.type === 'manual');
    const ci = !!(g && g.ci);
    card.querySelectorAll('.ci-only').forEach((node) => { node.style.display = ci ? '' : 'none'; });
    card.querySelectorAll('.tool-only').forEach((node) => { node.style.display = ci || manual ? 'none' : ''; });
    // The start delay applies to every generator but the manual one.
    card.querySelectorAll('.ssh-only:not(.tool-only)').forEach((node) => { node.style.display = manual ? 'none' : ''; });
    card.querySelector('.tool-field').style.display = ci ? 'none' : '';
    card.querySelector('.f-ci-extra-box').style.display = ci && g.ci.extra_vars ? '' : 'none';
    if (ci && g.ci.kind === 'jenkins') card.querySelector('.f-ci-ref-box').style.display = 'none';
    if (ci) card.querySelector('.f-ci-ref').placeholder = 'по умолчанию';
    if (ci) {
      card.querySelector('.tool-help').textContent = `Запускается ${CI_KIND_TITLES[g.ci.kind] || 'пайплайн'} как есть: инструмент, скрипт и их настройки задаёт пайплайн, здесь — ветка и его параметры.`
        + (g.ci.tags_metrics ? '' : ' Метрики пайплайна не помечены тегом прогона: отчёт читает все метрики окна.');
      const where = card.querySelector('.f-where');
      const unchecked = g.verified === false ? ' · подключение не проверено в настройках: запуск может не пройти' : '';
      where.textContent = `${g.type_title}${g.host ? ` · ${g.host}` : ''}${unchecked}`;
      where.classList.toggle('warn-text', !!unchecked);
      loadRefs(card);
      loadCi(card, false);
      return;
    }
    const toolSelect = card.querySelector('.f-tool');
    const prev = toolSelect.value || card.dataset.tool || '';
    toolSelect.innerHTML = toolOptions(generatorId, prev);
    if (!toolSelect.value && toolSelect.options.length && !manual) toolSelect.selectedIndex = 0;
    const tool = state.tools[toolSelect.value];
    card.querySelectorAll('.ssh-only').forEach((node) => { node.style.display = manual ? 'none' : ''; });
    const help = card.querySelector('.tool-help');
    help.textContent = manual
      ? 'Ручной генератор: LoadLens ничего не запускает. Запустите инструмент сами и остановите прогон кнопкой, когда тест закончится.'
      : (tool ? tool.help : '');
    const scriptLabel = card.querySelector('.f-script-label');
    scriptLabel.textContent = tool && tool.script_kind === 'class' ? 'Класс симуляции' : 'Скрипт';
    const lists = !!(g && g.lists_scripts);
    card.querySelector('.f-refresh').style.display = lists ? '' : 'none';
    card.querySelector('.f-script').placeholder = tool && tool.script_kind === 'class'
      ? 'computerdatabase.BasicSimulation'
      : (g && g.relative_scripts ? 'путь в репозитории задачи CI' : 'путь от каталога скриптов генератора или абсолютный');
    const where = card.querySelector('.f-where');
    const unchecked = g && g.verified === false ? ' · подключение не проверено в настройках: запуск может не пройти' : '';
    where.textContent = g && !manual ? `${g.type_title}${g.host ? ` · ${g.host}` : ''}${tool && tool.image && (g.type === 'docker' || g.type === 'kubernetes') ? ` · образ по умолчанию ${tool.image}` : ''}${unchecked}` : '';
    where.classList.toggle('warn-text', !!unchecked);
    if (!manual && lists && !state.scripts[scriptsKey(card)]) loadScripts(card, true);
    else {
      fillDatalist(card);
      if (!lists && !manual) card.querySelector('.f-scripts-note').textContent = g && g.relative_scripts
        ? 'Скрипт берётся из репозитория задачи CI.'
        : 'Список скриптов генератора недоступен: укажите путь.';
    }
  }

  // Summary line of a collapsed instance: «Инстанс 1 · rest · Gen 1 · k6» (empty parts skipped).
  function instanceSummary(index, parts) {
    const p = parts || {};
    return [`Инстанс ${index + 1}`, p.name, p.generator, p.tool].map((x) => String(x || '').trim()).filter(Boolean).join(' · ');
  }

  function summaryParts(card) {
    const generatorId = card.querySelector('.f-generator').value;
    const g = state.generators[generatorId];
    const toolId = card.querySelector('.f-tool').value;
    const tool = state.tools[toolId];
    if (g && g.ci) {
      const ref = card.querySelector('.f-ci-ref').value.trim();
      return { name: card.querySelector('.f-name').value, generator: g.title, tool: ref ? `ветка ${ref}` : (CI_KIND_TITLES[g.ci.kind] || '') };
    }
    return {
      name: card.querySelector('.f-name').value,
      generator: g ? g.title : generatorId,
      tool: tool ? tool.title : (g && g.type === 'manual' ? 'вручную' : toolId)
    };
  }

  function updateSummary(card, index) {
    const cards = Array.from(document.querySelectorAll('.instance-card'));
    const i = Number.isInteger(index) ? index : cards.indexOf(card);
    const text = instanceSummary(Math.max(0, i), summaryParts(card));
    const cut = text.indexOf(' · ');
    card.querySelector('.instance-title').textContent = cut >= 0 ? text.slice(0, cut) : text;
    card.querySelector('.instance-meta').textContent = cut >= 0 ? text.slice(cut) : '';
    card.querySelector('.instance-head').title = text;
  }

  // Indexes of the cards to collapse after a card was added: all but the newest from COLLAPSE_FROM cards on.
  function cardsToCollapse(count, from) {
    const threshold = from || COLLAPSE_FROM;
    return count >= threshold ? Array.from({ length: count - 1 }, (_, i) => i) : [];
  }

  function collapseAllButLast() {
    const cards = Array.from(document.querySelectorAll('.instance-card'));
    cardsToCollapse(cards.length).forEach((i) => { cards[i].querySelector('.instance-details').open = false; });
  }

  function renumber() {
    const cards = Array.from(document.querySelectorAll('.instance-card'));
    cards.forEach((card, i) => {
      updateSummary(card, i);
      card.querySelector('.f-remove').disabled = cards.length === 1;
    });
    const max = state.limits.max_instances || 20;
    $('addInstance').disabled = cards.length >= max;
    $('instancesNote').textContent = cards.length >= max ? `Не больше ${max} инстансов` : '';
  }

  function addInstance(values) {
    const v = values || {};
    const id = `inst${state.seq += 1}`;
    const card = el('div', 'instance-card');
    card.dataset.tool = v.tool || '';
    card.innerHTML = `
      <details class="instance-details" open>
      <summary class="instance-head">${CHEVRON}<span class="instance-title"></span><span class="instance-meta"></span></summary>
      <div class="instance-grid">
        <div><label for="${id}-name">Имя</label><input type="text" id="${id}-name" class="f-name" maxlength="60" placeholder="rest, kafka…" /></div>
        <div><label for="${id}-gen">Генератор</label><select id="${id}-gen" class="f-generator">${generatorOptions(v.generator)}</select></div>
        <div class="tool-field"><label for="${id}-tool">Инструмент</label><select id="${id}-tool" class="f-tool"></select></div>
        <div class="ssh-only"><label for="${id}-delay">Задержка старта, с</label><input type="number" id="${id}-delay" class="f-delay" min="0" step="1" value="0" /></div>
        <div class="wide field-hint f-where"></div>
        <div class="wide tool-help"></div>
        <div class="wide ci-only f-ci-ref-box">
          <label for="${id}-ref">Ветка</label>
          <div class="script-row"><input type="text" id="${id}-ref" class="f-ci-ref mono" list="${id}-refs" autocomplete="off" spellcheck="false" />
          <button type="button" class="btn btn-sm f-ci-reload">Обновить параметры</button></div>
          <datalist id="${id}-refs" class="f-ci-refs"></datalist>
        </div>
        <div class="wide ci-only field-hint f-ci-note"></div>
        <div class="wide ci-only ci-params f-ci-params"></div>
        <div class="wide ci-only f-ci-extra-box">
          <label for="${id}-extra">Дополнительные переменные пайплайна</label>
          <textarea id="${id}-extra" class="f-ci-extra mono" rows="2" spellcheck="false" placeholder="ИМЯ=значение, по одной в строке"></textarea>
          <div class="field-hint">Уходят в GitLab переменными пайплайна и видны в нём участникам проекта: не передавайте так секреты.</div>
        </div>
        <div class="wide ssh-only tool-only">
          <label for="${id}-script" class="f-script-label">Скрипт</label>
          <div class="script-row"><input type="text" id="${id}-script" class="f-script mono" list="${id}-scripts" autocomplete="off" spellcheck="false" />
          <button type="button" class="btn btn-sm f-refresh">Обновить список</button></div>
          <datalist id="${id}-scripts"></datalist>
          <div class="field-hint f-scripts-note"></div>
        </div>
        <div class="wide ssh-only tool-only">
          <div class="label-row"><label for="${id}-params">Параметры запуска</label>
          <button type="button" class="btn btn-sm btn-ghost f-hints" title="Флаги и переменные вывода метрик под источник домена «Нагрузка» сервиса">Вывод метрик</button></div>
          <textarea id="${id}-params" class="f-params mono" rows="2" spellcheck="false" placeholder="--vus 50 --duration 10m -e BASE_URL=https://stage.example"></textarea>
          <div class="field-hint">Как в командной строке: кавычки группируют, переменные и подстановки оболочки не раскрываются.</div>
          <div class="field-hint f-hints-note"></div>
        </div>
        <div class="wide ssh-only tool-only">
          <label for="${id}-env">Переменные окружения</label>
          <textarea id="${id}-env" class="f-env mono" rows="2" spellcheck="false" placeholder="ИМЯ=значение, по одной в строке"></textarea>
          <div class="field-hint">Для токенов и паролей: значения не показываются в интерфейсе и журнале и стираются после завершения инстанса.</div>
        </div>
      </div>
      </details>
      <button type="button" class="btn btn-sm btn-ghost f-remove instance-remove">Убрать</button>
      <div class="instance-preview" hidden></div>`;
    card.querySelector('.f-name').value = v.name || '';
    card.querySelector('.f-script').value = v.script || '';
    card.querySelector('.f-params').value = v.params || '';
    card.querySelector('.f-env').value = typeof v.env === 'string' ? v.env : '';
    card.querySelector('.f-delay').value = v.start_delay_sec || 0;
    card.querySelector('.f-ci-ref').value = v.ci_ref || '';
    card.querySelector('.f-ci-extra').value = typeof v.ci_extra === 'string' ? v.ci_extra : extraText(v.ci_extra);
    card.ciState = { values: { ...(v.ci_params || {}) }, defs: null, key: null };
    card.querySelector('.f-ci-ref').addEventListener('change', () => { loadCi(card, false); updateSummary(card); });
    card.querySelector('.f-ci-reload').addEventListener('click', () => loadCi(card, true));
    card.querySelector('.f-hints').addEventListener('click', () => applyHints(card));
    card.querySelector('.f-generator').addEventListener('change', () => { syncCard(card); updateSummary(card); });
    card.querySelector('.f-tool').addEventListener('change', () => { card.dataset.tool = card.querySelector('.f-tool').value; syncCard(card); updateSummary(card); });
    card.querySelector('.f-name').addEventListener('input', () => updateSummary(card));
    card.querySelector('.f-refresh').addEventListener('click', () => loadScripts(card, false));
    card.querySelector('.f-remove').addEventListener('click', () => { card.remove(); renumber(); markPreviewStale(); });
    $('instances').appendChild(card);
    syncCard(card);
    renumber();
    return card;
  }

  async function applyHints(card) {
    const note = card.querySelector('.f-hints-note');
    const service = $('service').value;
    const tool = card.querySelector('.f-tool').value;
    if (!service || !tool) { note.textContent = 'Выберите сервис и инструмент.'; return; }
    note.textContent = 'Смотрим источник домена «Нагрузка»…';
    try {
      const hint = await fetchJson(`/load_tests/hints?service=${encodeURIComponent(service)}&tool=${encodeURIComponent(tool)}&project_area=${encodeURIComponent(LL.activeProjectArea || '')}`);
      const params = card.querySelector('.f-params');
      if (hint.params && !params.value.includes(hint.params)) params.value = `${params.value.trim()} ${hint.params}`.trim();
      const envBox = card.querySelector('.f-env');
      const present = new Set(envBox.value.split('\n').map((line) => line.split('=')[0].trim()).filter(Boolean));
      const added = Object.keys(hint.env || {}).filter((name) => !present.has(name)).map((name) => `${name}=${hint.env[name]}`);
      if (added.length) envBox.value = `${envBox.value.trim()}\n${added.join('\n')}`.trim();
      note.textContent = (hint.notes || []).join(' ') || (hint.params || added.length ? 'Добавлено.' : 'Добавлять нечего.');
    } catch (e) {
      note.textContent = `Подсказка недоступна: ${e.message}`;
    }
  }

  function applyForm(spec) {
    if (!spec) return;
    // Silent: a "change" here would reload the presets and drop the one being applied.
    if (spec.service && state.picker && state.picker.has(spec.service)) state.picker.setValue(spec.service, { silent: true });
    $('testType').value = spec.test_type || '';
    $('maxDuration').value = spec.max_duration_min || '';
    $('autoReport').checked = spec.auto_report !== false;
    $('useLlm').checked = spec.use_llm !== false;
    const instances = Array.isArray(spec.instances) && spec.instances.length ? spec.instances : [{}];
    $('instances').replaceChildren();
    clearPreview();
    instances.forEach((inst) => addInstance(inst.generator && state.generators[inst.generator] ? inst : { ...inst, generator: state.generatorOrder[0] }));
    collapseAllButLast();
  }

  function runToSpec(run) {
    return {
      service: run.service,
      test_type: run.test_type,
      max_duration_min: run.options && run.options.max_seconds ? Math.round(run.options.max_seconds / 60) : null,
      auto_report: !run.options || run.options.auto_report !== false,
      use_llm: !run.options || run.options.use_llm !== false,
      instances: run.instances.map((inst) => ({
        name: inst.name,
        generator: inst.generator,
        tool: inst.tool,
        script: inst.script,
        params: inst.params,
        env: (inst.env_keys || []).map((name) => `${name}=`).join('\n'),
        start_delay_sec: inst.start_delay_sec,
        ...ciSpecFromView(inst.ci)
      }))
    };
  }

  async function repeatFrom(runId) {
    try {
      const run = await fetchJson(`/load_tests/runs/${encodeURIComponent(runId)}`);
      applyForm(runToSpec(run));
      const banner = $('repeatBanner');
      const envNames = run.instances.flatMap((inst) => (inst.env_keys || []).concat(inst.ci ? inst.ci.secret || [] : []));
      banner.querySelector('.banner-title').textContent = `Повтор прогона «${run.run_name}»`;
      banner.querySelector('.banner-text').textContent = envNames.length
        ? `Значения переменных окружения и секретных параметров не сохраняются: введите заново ${Array.from(new Set(envNames)).join(', ')}.`
        : 'Инстансы и параметры взяты из прошлого прогона; при необходимости измените их и запустите.';
      banner.style.display = '';
      await loadPresets();
      return true;
    } catch (e) {
      setError(`Прогон для повтора не загрузился: ${e.message}`);
      return false;
    }
  }

  async function loadPresets() {
    const select = $('presetSelect');
    const service = $('service').value;
    if (!service) {
      state.presets = [];
      select.innerHTML = '<option value="">Сначала выберите сервис</option>';
      return;
    }
    try {
      const data = await fetchJson(`/load_tests/presets?service=${encodeURIComponent(service)}`);
      state.presets = Array.isArray(data.presets) ? data.presets : [];
    } catch (e) {
      state.presets = [];
    }
    select.innerHTML = state.presets.length
      ? '<option value="">Выберите пресет…</option>' + state.presets.map((p) => `<option value="${esc(p.preset_id)}">${esc(p.name)}${p.owner ? ` · ${esc(p.owner)}` : ''}</option>`).join('')
      : '<option value="">Пресетов нет</option>';
  }

  function applyPreset() {
    const preset = state.presets.find((p) => p.preset_id === $('presetSelect').value);
    if (!preset) return;
    applyForm({ ...preset.spec, service: preset.service });
    ui.toast(`Пресет «${preset.name}» подставлен`, { tone: 'ok' });
  }

  async function savePreset() {
    const form = readForm();
    if (!form.service) { setError('Выберите сервис'); return; }
    const current = state.presets.find((p) => p.preset_id === $('presetSelect').value);
    const name = await ui.prompt({ title: 'Сохранить как пресет', label: 'Название', value: current ? current.name : '', confirmText: 'Сохранить' });
    if (!name) return;
    try {
      const saved = await fetchJson('/load_tests/presets', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, service: form.service, project_area: form.project_area, spec: form })
      });
      await loadPresets();
      $('presetSelect').value = saved.preset_id;
      ui.toast(`Пресет «${saved.name}» сохранён; значения переменных окружения в нём не хранятся`, { tone: 'ok' });
    } catch (e) {
      ui.toast(`Пресет не сохранён: ${e.message}`, { tone: 'error' });
    }
  }

  async function deletePreset() {
    const preset = state.presets.find((p) => p.preset_id === $('presetSelect').value);
    if (!preset) return;
    const ok = await ui.confirm({ title: 'Удалить пресет', message: `Пресет «${preset.name}» исчезнет для всех инженеров сервиса.`, confirmText: 'Удалить', danger: true });
    if (!ok) return;
    try {
      await fetchJson(`/load_tests/presets/${encodeURIComponent(preset.preset_id)}`, { method: 'DELETE' });
      await loadPresets();
    } catch (e) {
      ui.toast(`Пресет не удалён: ${e.message}`, { tone: 'error' });
    }
  }

  // ---- form ------------------------------------------------------------------------------------

  function readForm() {
    const instances = Array.from(document.querySelectorAll('.instance-card')).map((card) => {
      const common = {
        name: card.querySelector('.f-name').value.trim(),
        generator: card.querySelector('.f-generator').value,
        start_delay_sec: Number(card.querySelector('.f-delay').value || 0)
      };
      const g = ciGenerator(card);
      if (g) {
        return {
          ...common,
          ci_ref: card.querySelector('.f-ci-ref').value.trim(),
          ci_params: card.ciState.defs ? readCiValues(card) : {},
          ci_extra: g.ci.extra_vars ? card.querySelector('.f-ci-extra').value : '',
          ci_secret: ciSecretNames(card)
        };
      }
      return {
        ...common,
        tool: card.querySelector('.f-tool').value,
        script: card.querySelector('.f-script').value.trim(),
        params: card.querySelector('.f-params').value,
        env: card.querySelector('.f-env').value
      };
    });
    const maxDuration = $('maxDuration').value;
    return {
      service: $('service').value,
      project_area: LL.activeProjectArea || '',
      test_type: $('testType').value,
      run_name: $('runName').value.trim(),
      max_duration_min: maxDuration ? Number(maxDuration) : null,
      auto_report: $('autoReport').checked,
      use_llm: $('useLlm').checked,
      instances
    };
  }

  function saveDraft(payload) {
    // Environment values may hold secrets: the draft keeps everything else.
    const draft = { ...payload, run_name: '', instances: payload.instances.map((i) => ({ ...i, env: '', ci_params: blankSecrets(i.ci_params, i.ci_secret) })) };
    try { localStorage.setItem(DRAFT_KEY, JSON.stringify(draft)); } catch (e) { /* storage full or disabled */ }
  }

  function loadDraft() {
    try { return JSON.parse(localStorage.getItem(DRAFT_KEY) || 'null'); } catch (e) { return null; }
  }

  // Commands of the preview grouped by card position ({index: command}); entries without a valid index are left over.
  function commandsByIndex(commands, cardCount) {
    const byIndex = {};
    const leftovers = [];
    (Array.isArray(commands) ? commands : []).forEach((cmd) => {
      const i = cmd && Number.isInteger(cmd.index) ? cmd.index : -1;
      if (i >= 0 && i < cardCount && !byIndex[i]) byIndex[i] = cmd;
      else if (cmd) leftovers.push(cmd);
    });
    return { byIndex, leftovers };
  }

  function commandNodes(cmd, withTitle) {
    const nodes = [];
    if (withTitle) nodes.push(el('div', '', `${cmd.name} · ${cmd.generator_title}`));
    if (cmd.manual) {
      nodes.push(el('p', 'dim', 'Ручной генератор: LoadLens отметит только время начала и окончания.'));
      return nodes;
    }
    nodes.push(el('pre', '', cmd.command));
    if (cmd.image) nodes.push(el('div', 'field-hint', `Образ: ${cmd.image}`));
    if (cmd.cwd) nodes.push(el('div', 'field-hint', `Рабочий каталог: ${cmd.cwd}`));
    return nodes;
  }

  function clearPreview() {
    document.querySelectorAll('.instance-preview').forEach((node) => { node.replaceChildren(); node.hidden = true; node.classList.remove('is-stale'); });
    $('previewCard').style.display = 'none';
  }

  // After an edit the shown commands no longer match the form until «Проверить команды» runs again.
  function markPreviewStale() {
    document.querySelectorAll('.instance-preview:not([hidden]):not(.is-stale)').forEach((node) => {
      node.classList.add('is-stale');
      const label = node.querySelector('.preview-label');
      if (label) label.textContent = 'Команда до изменений: проверьте заново';
    });
  }

  function renderPreview(data) {
    const cards = Array.from(document.querySelectorAll('.instance-card'));
    const { byIndex, leftovers } = commandsByIndex(data.commands, cards.length);
    cards.forEach((card, i) => {
      const host = card.querySelector('.instance-preview');
      host.replaceChildren();
      host.classList.remove('is-stale');
      const cmd = byIndex[i];
      host.hidden = !cmd;
      if (!cmd) return;
      host.appendChild(el('div', 'preview-label', 'Команда'));
      commandNodes(cmd, false).forEach((node) => host.appendChild(node));
    });
    const body = $('previewBody');
    body.replaceChildren();
    body.appendChild(el('p', 'dim', `Прогон «${data.run_name}», тег ${data.run_tag}. Команды инстансов показаны в их карточках.`));
    leftovers.forEach((cmd) => commandNodes(cmd, true).forEach((node) => body.appendChild(node)));
    if (data.warnings && data.warnings.length) {
      const list = el('ul', 'warn-list');
      data.warnings.forEach((w) => list.appendChild(el('li', '', w)));
      body.appendChild(list);
    }
    $('previewCard').style.display = '';
  }

  // «Инстанс 2: …» from the server: open that card and bring it into view.
  function revealInstanceError(message) {
    const match = /^Инстанс (\d+):/.exec(String(message || ''));
    const card = match ? document.querySelectorAll('.instance-card')[Number(match[1]) - 1] : null;
    if (!card) return;
    card.querySelector('.instance-details').open = true;
    card.scrollIntoView({ block: 'nearest' });
  }

  async function preview() {
    setError('');
    const btn = $('previewBtn');
    btn.disabled = true;
    try {
      const data = await fetchJson('/load_tests/preview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(readForm()) });
      renderPreview(data);
    } catch (e) {
      setError(e.message);
      clearPreview();
      revealInstanceError(e.message);
    } finally {
      btn.disabled = false;
    }
  }

  async function launch(event) {
    event.preventDefault();
    setError('');
    const payload = readForm();
    if (!payload.service) { setError('Выберите сервис'); return; }
    const btn = $('launchBtn');
    btn.disabled = true;
    try {
      const data = await fetchJson('/load_tests', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      saveDraft(payload);
      if (LL.servicePicker) LL.servicePicker.remember(payload.service);
      (data.warnings || []).forEach((w) => ui.toast(w, { tone: 'warn', timeout: 10000 }));
      window.location.href = data.url;
    } catch (e) {
      setError(e.message);
      ui.toast(e.message, { tone: 'error' });
      revealInstanceError(e.message);
      btn.disabled = false;
    }
  }

  async function loadServices(selected) {
    if (!LL.servicePicker) return;
    state.picker = LL.servicePicker({ host: $('serviceCombo'), input: $('service'), inputId: 'serviceInput', describedBy: 'serviceHint', selected });
    const hint = $('serviceHint');
    try {
      const services = await state.picker.ready;
      if (!services.length) {
        hint.textContent = 'Нет настроенных сервисов: администратор добавляет их в настройках (раздел «Сервисы»).';
        hint.classList.add('warn');
      }
    } catch (e) {
      hint.textContent = `Сервисы не загрузились: ${e.message}`;
      hint.classList.add('error');
    }
  }

  LL.launch = { _internals: { instanceSummary, cardsToCollapse, commandsByIndex, blankSecrets, extraText, ciSpecFromView, ciFieldValue } };

  async function init() {
    loadRuns();
    let options;
    try {
      options = await fetchJson('/load_tests/options');
    } catch (e) {
      setError(`Настройки запуска не загрузились: ${e.message}`);
      return;
    }
    (options.tools || []).forEach((t) => { state.tools[t.id] = t; state.toolOrder.push(t.id); });
    (options.generators || []).forEach((g) => { state.generators[g.id] = g; state.generatorOrder.push(g.id); });
    state.limits = options.limits || {};
    const max = state.limits.max_duration_min || 1440;
    $('maxDuration').max = String(max);
    $('maxDuration').placeholder = String(max);
    $('maxDurationHint').textContent = `Не больше ${max} мин. Генератор остановит инструмент по истечении этого времени, даже если LoadLens недоступен.`;
    if (!state.generatorOrder.length) {
      $('noGenerators').style.display = '';
      $('launchBtn').disabled = true;
      $('previewBtn').disabled = true;
    }
    const draft = loadDraft();
    const repeat = new URLSearchParams(window.location.search).get('from');
    await loadServices(draft && draft.service);
    if (state.generatorOrder.length) {
      if (!(repeat && await repeatFrom(repeat))) applyForm(draft || { instances: [{}] });
    }
    await loadPresets();
    $('service').addEventListener('change', loadPresets);
    $('presetSelect').addEventListener('change', applyPreset);
    $('presetSave').addEventListener('click', savePreset);
    $('presetDelete').addEventListener('click', deletePreset);
    $('addInstance').addEventListener('click', () => {
      const cards = document.querySelectorAll('.instance-card');
      const last = cards[cards.length - 1];
      const card = addInstance(last ? { generator: last.querySelector('.f-generator').value, tool: last.querySelector('.f-tool').value, ci_ref: last.querySelector('.f-ci-ref').value } : {});
      collapseAllButLast();
      markPreviewStale();
      card.querySelector('.f-name').focus();
    });
    $('previewBtn').addEventListener('click', preview);
    $('launchForm').addEventListener('submit', launch);
    $('launchForm').addEventListener('input', markPreviewStale);
    $('launchForm').addEventListener('change', markPreviewStale);
  }

  document.addEventListener('DOMContentLoaded', init);
})();
