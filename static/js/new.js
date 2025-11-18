// New report page logic (templates/index.html)
(function () {
  // Independent timers for progress branches
  const progressTimers = { confluence: null, web: null };
  let activeAreaCache = '';

  function resetServiceSelection() {
    const serviceSelect = document.getElementById('serviceSelect');
    if (serviceSelect) {
      serviceSelect.textContent = 'Выберите сервис';
      delete serviceSelect.dataset.serviceId;
      delete serviceSelect.dataset.serviceTitle;
    }
  }

  function renderServicePlaceholder(message) {
    const serviceOptions = document.getElementById('serviceOptions');
    if (!serviceOptions) return;
    serviceOptions.innerHTML = '';
    const optionDiv = document.createElement('div');
    optionDiv.classList.add('custom-option');
    optionDiv.classList.add('disabled');
    optionDiv.textContent = message;
    serviceOptions.appendChild(optionDiv);
  }

  // Load available services into custom select options
  async function loadServices(areaOverride) {
    try {
      const serviceOptions = document.getElementById('serviceOptions');
      if (!serviceOptions) return;
      let effectiveArea = typeof areaOverride === 'string' ? areaOverride : '';
      if (!effectiveArea) {
        effectiveArea = (window.LoadLens && window.LoadLens.activeProjectArea) || '';
      }
      if (!effectiveArea) {
        try {
          const curResp = await fetch('/current_project_area');
          const cur = await curResp.json();
          effectiveArea = (cur && cur.project_area) || '';
          if (window.LoadLens) {
            window.LoadLens.activeProjectArea = effectiveArea;
          }
        } catch (err) {
          effectiveArea = '';
        }
      }
      activeAreaCache = effectiveArea;
      resetServiceSelection();
      if (!effectiveArea) {
        renderServicePlaceholder('Выберите область в шапке страницы');
        return;
      }
      const response = await fetch(`/services?area=${encodeURIComponent(effectiveArea)}`);
      const payload = await response.json();
      const services = Array.isArray(payload?.services) ? payload.services : (Array.isArray(payload) ? payload : []);
      if (!services.length) {
        renderServicePlaceholder('Нет сервисов для выбранной области');
        return;
      }
      serviceOptions.innerHTML = '';
      services.forEach((svc) => {
        const id = typeof svc === 'string' ? svc : (svc.id || '');
        if (!id) return;
        const title = typeof svc === 'string' ? svc : ((svc && svc.title) || id);
        const optionDiv = document.createElement('div');
        optionDiv.classList.add('custom-option');
        optionDiv.textContent = title;
        optionDiv.dataset.value = id;
        optionDiv.dataset.title = title;
        serviceOptions.appendChild(optionDiv);
      });
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Ошибка загрузки сервисов:', error);
      renderServicePlaceholder('Не удалось загрузить сервисы');
    }
  }

  // Step-by-step visibility sync
  function syncSteps() {
    try {
      const targetVal = (document.getElementById('target_mode') && document.getElementById('target_mode').value) || '';
      const anyTarget = !!targetVal;
      const startEl = document.getElementById('start');
      const endEl = document.getElementById('end');
      const serviceLabel = document.getElementById('serviceSelect');

      const stepStart = document.getElementById('stepStart');
      const stepEnd = document.getElementById('stepEnd');
      const stepService = document.getElementById('stepService');
      const stepLLM = document.getElementById('stepLLM');
      const stepTestType = document.getElementById('stepTestType');
      const runNameBlock = document.getElementById('runNameContainer');
      const createBtn = document.getElementById('createBtn');
      const runNameInput = document.getElementById('run_name');

      if (stepStart) stepStart.style.display = anyTarget ? 'block' : 'none';
      const hasStart = !!(startEl && startEl.value);
      if (stepEnd) stepEnd.style.display = anyTarget && hasStart ? 'block' : 'none';
      const hasEnd = !!(endEl && endEl.value);
      if (stepService) stepService.style.display = anyTarget && hasStart && hasEnd ? 'block' : 'none';
      const serviceChosen = !!(serviceLabel && serviceLabel.dataset && serviceLabel.dataset.serviceId);
      if (stepTestType) stepTestType.style.display = anyTarget && hasStart && hasEnd && serviceChosen ? 'flex' : 'none';
      const testTypeVal = (document.getElementById('test_type')?.value || '').trim();
      const tail = anyTarget && hasStart && hasEnd && serviceChosen && !!testTypeVal;
      if (stepLLM) stepLLM.style.display = tail ? 'flex' : 'none';
      if (runNameBlock) runNameBlock.style.display = tail ? 'block' : 'none';
      const hasName = !!(runNameInput && (runNameInput.value || '').trim());
      if (createBtn) createBtn.style.display = tail && hasName ? 'inline-block' : 'none';
    } catch (e) {
      // noop
    }
  }

  function selectTarget(val) {
    const inp = document.getElementById('target_mode');
    if (inp) inp.value = val || '';
    const b1 = document.getElementById('target_btn_conf');
    const b2 = document.getElementById('target_btn_web');
    if (b1) b1.classList.toggle('active', val === 'confluence');
    if (b2) b2.classList.toggle('active', val === 'web');
    syncSteps();
  }

  function selectLlm(val) {
    const inp = document.getElementById('use_llm_mode');
    if (inp) inp.value = val || 'no';
    const y = document.getElementById('use_llm_btn_yes');
    const n = document.getElementById('use_llm_btn_no');
    if (y) y.classList.toggle('active', val === 'yes');
    if (n) n.classList.toggle('active', val === 'no');
  }

  function selectTestType(val) {
    const inp = document.getElementById('test_type');
    if (inp) inp.value = val || '';
    ['tt_step', 'tt_soak', 'tt_spike', 'tt_stress'].forEach(id => {
      const btn = document.getElementById(id);
      if (btn) btn.classList.toggle('active', btn.getAttribute('data-value') === val);
    });
    try {
      syncSteps();
    } catch (e) {
      // noop
    }
  }

  async function startProgressBranch(scope, jobId) {
    const isConf = scope === 'confluence';
    const bar = document.getElementById(isConf ? 'progressConfluenceBarFill' : 'progressWebBarFill');
    const text = document.getElementById(isConf ? 'progressConfluenceText' : 'progressWebText');
    const link = document.getElementById(isConf ? 'confluenceLink' : 'webLink');

    if (link) {
      link.style.display = 'none';
      link.removeAttribute('href');
    }

    if (progressTimers[scope]) clearInterval(progressTimers[scope]);
    progressTimers[scope] = setInterval(async () => {
      try {
        const r = await fetch(`/job_status/${jobId}`);
        if (!r.ok) return;
        const j = await r.json();
        const p = Math.max(0, Math.min(100, j.progress || 0));
        if (bar) bar.style.width = `${p}%`;
        const msg = j.message ? ` — ${j.message}` : '';
        if (text) text.textContent = `Прогресс${isConf ? ' (Confluence)' : ' (LoadLens)'}: ${p}%${msg}`;
        if (j.report_url && link && (!link.href || link.href !== j.report_url)) {
          link.href = j.report_url;
          link.style.display = 'inline-block';
        }
        if (j.status === 'done') {
          clearInterval(progressTimers[scope]);
          if (j.report_url && link) {
            link.href = j.report_url;
            link.style.display = 'inline-block';
          }
        } else if (j.status === 'error') {
          clearInterval(progressTimers[scope]);
          if (text) text.textContent = `Ошибка: ${j.error || 'unknown'}`;
        }
      } catch (e) {
        // ignore transient errors
      }
    }, 1000);
  }

  async function createReport() {
    const responseMessage = document.getElementById('responseMessage');
    if (responseMessage) responseMessage.innerText = '';
    const start = document.getElementById('start')?.value;
    const end = document.getElementById('end')?.value;
    const areaValue = activeAreaCache || (window.LoadLens && window.LoadLens.activeProjectArea) || '';
    if (!areaValue) {
      if (responseMessage) responseMessage.innerText = 'Пожалуйста, выберите проектную область в шапке страницы.';
      return;
    }
    const serviceSelectEl = document.getElementById('serviceSelect');
    const selectedService = serviceSelectEl?.dataset?.serviceId || '';
    if (!selectedService) {
      if (responseMessage) responseMessage.innerText = 'Пожалуйста, выберите сервис.';
      return;
    }
    const serviceTitle = serviceSelectEl?.dataset?.serviceTitle || serviceSelectEl?.textContent || selectedService;

    const target = (document.getElementById('target_mode')?.value || '').trim();
    const toConfluence = target === 'confluence';
    const toWeb = target === 'web' || target === 'confluence';
    if (!toWeb && !toConfluence) {
      if (responseMessage) responseMessage.innerText = 'Выберите хотя бы одно направление публикации.';
      return;
    }

    const common = {
      start,
      end,
      service: selectedService,
      service_title: serviceTitle,
      project_area: areaValue,
      area: areaValue,
      test_type: (document.getElementById('test_type')?.value || '').trim(),
      use_llm: ((document.getElementById('use_llm_mode')?.value || 'yes') === 'yes'),
      run_name: (document.getElementById('run_name')?.value || '').trim()
    };

    const progressBox = document.getElementById('progressContainer');
    if (progressBox) progressBox.style.display = 'block';

    // Helper to start job
    const startJob = async (scope, payload) => {
      const scopeText = scope === 'confluence' ? 'progressConfluenceText' : 'progressWebText';
      try {
        const resp = await fetch('/create_report', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const result = await resp.json();
        if (result.status === 'accepted' && result.job_id) {
          if (responseMessage) responseMessage.innerText = result.message || 'Задача принята.';
          await startProgressBranch(scope, result.job_id);
        } else if (result.status === 'success') {
          const link = scope === 'confluence' ? document.getElementById('confluenceLink') : document.getElementById('webLink');
          if (result.report_url && link) {
            link.href = result.report_url;
            link.style.display = 'inline-block';
          }
        } else {
          const textEl = document.getElementById(scopeText);
          if (textEl) textEl.textContent = result.message || 'Ошибка запуска задачи';
        }
      } catch (error) {
        const textEl = document.getElementById(scopeText);
        if (textEl) textEl.textContent = `Ошибка создания отчёта: ${error.message}`;
      }
    };

    if (toConfluence) {
      const branch = document.getElementById('progressBranchConfluence');
      if (branch) branch.style.display = 'block';
      // Конфлюенс выполняет ИИ-анализ и сохраняет ИИ-результаты в БД
      await startJob('confluence', { ...common, web_only: false, save_to_db: true });
    }
    if (toWeb) {
      const branch = document.getElementById('progressBranchWeb');
      if (branch) branch.style.display = 'block';
      // Если выбрана Конфлюенс, веб-ветка пишет только метрики (без ИИ), иначе — как выбрано пользователем
      const useLlmForWeb = toConfluence ? false : common.use_llm;
      await startJob('web', { ...common, use_llm: useLlmForWeb, web_only: true, save_to_db: true });
    }
  }

  // Wire up events on DOM ready
  document.addEventListener('DOMContentLoaded', async function () {
    try { await window.LoadLens.initProjectArea(); } catch (e) {}
    await loadServices();

    // Custom select open/close
    const serviceSelect = document.getElementById('serviceSelect');
    const serviceWrapper = document.getElementById('serviceWrapper');
    const serviceOptions = document.getElementById('serviceOptions');
    if (serviceSelect) {
      serviceSelect.addEventListener('click', function (e) {
        e.stopPropagation();
        if (serviceWrapper) serviceWrapper.classList.toggle('open');
      });
    }
    window.addEventListener('click', function (e) {
      if (serviceWrapper && !serviceWrapper.contains(e.target)) {
        serviceWrapper.classList.remove('open');
      }
    });
    if (serviceOptions) {
      serviceOptions.addEventListener('click', function (e) {
        const opt = e.target.closest('.custom-option');
        if (!opt) return;
        const selectedService = opt.dataset.value || opt.textContent || '';
        if (!selectedService) return;
        const selectedTitle = opt.dataset.title || opt.textContent || selectedService;
        const sel = document.getElementById('serviceSelect');
        if (sel) {
          sel.textContent = selectedTitle;
          sel.dataset.serviceId = selectedService;
          sel.dataset.serviceTitle = selectedTitle;
        }
        if (serviceWrapper) serviceWrapper.classList.remove('open');
        try { syncSteps(); } catch (err) {}
      });
    }

    // Toggle targets
    const b1 = document.getElementById('target_btn_conf');
    const b2 = document.getElementById('target_btn_web');
    if (b1) b1.addEventListener('click', () => selectTarget('confluence'));
    if (b2) b2.addEventListener('click', () => selectTarget('web'));

    // Inputs
    ['start', 'end'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('change', syncSteps);
    });
    const rn = document.getElementById('run_name');
    if (rn) rn.addEventListener('input', syncSteps);

    // LLM toggle
    const ly = document.getElementById('use_llm_btn_yes');
    const ln = document.getElementById('use_llm_btn_no');
    if (ly) ly.addEventListener('click', () => selectLlm('yes'));
    if (ln) ln.addEventListener('click', () => selectLlm('no'));

    // Test type
    const tStep = document.getElementById('tt_step'); if (tStep) tStep.addEventListener('click', () => selectTestType('step'));
    const tSoak = document.getElementById('tt_soak'); if (tSoak) tSoak.addEventListener('click', () => selectTestType('soak'));
    const tSpike = document.getElementById('tt_spike'); if (tSpike) tSpike.addEventListener('click', () => selectTestType('spike'));
    const tStress = document.getElementById('tt_stress'); if (tStress) tStress.addEventListener('click', () => selectTestType('stress'));

    // Create button
    const createBtn = document.getElementById('createBtn');
    if (createBtn) createBtn.addEventListener('click', createReport);

    // Initial state
    syncSteps();
  });
})();


