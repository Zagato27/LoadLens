// Common utilities shared across pages
// Namespace-safe attachment
window.LoadLens = window.LoadLens || {};

// Initialize project area selector in header
window.LoadLens.initProjectArea = async function initProjectArea() {
  try {
    const sel = document.getElementById('projectAreaSelect');
    if (!sel) return;
    const servicesResp = await fetch('/services');
    const services = await servicesResp.json();
    sel.innerHTML =
      '<option value="">Все области</option>' +
      (services || []).map(s => `<option value="${s}">${s}</option>`).join('');
    const curResp = await fetch('/current_project_area');
    const cur = await curResp.json();
    const val = (cur && cur.project_area) || '';
    sel.value = val || '';
    sel.addEventListener('change', async () => {
      try {
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


