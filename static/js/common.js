// Common utilities shared across pages
// Namespace-safe attachment
window.LoadLens = window.LoadLens || {};

// Initialize project area selector in header
window.LoadLens.initProjectArea = async function initProjectArea() {
  try {
    const sel = document.getElementById('projectAreaSelect');
    if (!sel) return;
    const areasResp = await fetch('/project_areas');
    const areas = await areasResp.json();
    const options = (areas || []).map((a) => {
      if (a && typeof a === 'object' && 'id' in a) {
        const title = (a.title && typeof a.title === 'string') ? a.title : a.id;
        return `<option value="${a.id}">${title}</option>`;
      }
      return `<option value="${a}">${a}</option>`;
    }).join('');
    sel.innerHTML =
      '<option value="">Все области</option>' +
      options;
    const curResp = await fetch('/current_project_area');
    const cur = await curResp.json();
    const val = (cur && cur.project_area) || '';
    sel.value = val || '';
    window.LoadLens.activeProjectArea = val || '';
    sel.addEventListener('change', async () => {
      try {
        window.LoadLens.activeProjectArea = sel.value || '';
        await fetch('/project_area', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ service: sel.value })
        });
        location.reload();
      } catch (e) {
        // noop
      }
    });
  } catch (e) {
    // noop
  }
};

// Random color generator for charts
window.LoadLens.randColor = function randColor(alpha = 0.7) {
  const r = Math.floor(100 + Math.random() * 155);
  const g = Math.floor(100 + Math.random() * 155);
  const b = Math.floor(100 + Math.random() * 155);
  return `rgba(${r},${g},${b},${alpha})`;
};


