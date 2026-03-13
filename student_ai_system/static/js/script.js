/* ============================================================
   Student AI System v2 — Advanced JavaScript
   ============================================================ */

// ── Theme ────────────────────────────────────────────────────
const themeToggle = document.getElementById('themeToggle');
const savedTheme = localStorage.getItem('theme') || 'dark';
document.documentElement.setAttribute('data-theme', savedTheme);
updateThemeIcon(savedTheme);

themeToggle?.addEventListener('click', () => {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  updateThemeIcon(next);
});

function updateThemeIcon(theme) {
  const icon = themeToggle?.querySelector('i');
  if (icon) icon.className = theme === 'dark' ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
}

// ── Flash auto-dismiss ───────────────────────────────────────
document.querySelectorAll('.flash-container .alert').forEach(a => {
  setTimeout(() => { a.classList.remove('show'); setTimeout(() => a.remove(), 300); }, 4500);
});

// ── Navbar scroll ────────────────────────────────────────────
const navbar = document.querySelector('.glass-nav');
window.addEventListener('scroll', () => {
  if (navbar) navbar.style.background = window.scrollY > 40 ? 'rgba(13,15,26,0.98)' : '';
}, { passive: true });

// ── Intersection Observer for cards ─────────────────────────
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.style.opacity = '1';
      e.target.style.transform = 'translateY(0)';
      observer.unobserve(e.target);
    }
  });
}, { threshold: 0.08 });

document.querySelectorAll('.animate-card').forEach(c => {
  c.style.opacity = '0';
  c.style.transform = 'translateY(22px)';
  c.style.transition = 'opacity .55s ease, transform .55s ease';
  observer.observe(c);
});

// ── Counter animation ────────────────────────────────────────
function animateCounter(el, target, duration = 1200) {
  const start = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const val = Math.round(target * eased);
    el.textContent = val + (el.dataset.suffix || '');
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

document.querySelectorAll('.stat-number, .stat-card-value, .kpi-value').forEach(el => {
  const raw = el.textContent.trim();
  const num = parseFloat(raw.replace(/[^0-9.]/g, ''));
  if (!isNaN(num) && num > 0 && num < 100000) {
    const suffix = raw.replace(/[0-9.]/g, '').trim();
    el.dataset.suffix = suffix;
    animateCounter(el, num);
  }
});

// ── Progress bar animation ───────────────────────────────────
document.querySelectorAll('.progress-bar').forEach(bar => {
  const w = bar.style.width;
  bar.style.width = '0%';
  bar.style.transition = 'width .9s cubic-bezier(0.4,0,0.2,1)';
  requestAnimationFrame(() => requestAnimationFrame(() => bar.style.width = w));
});

// ── Leaderboard bar animation ────────────────────────────────
document.querySelectorAll('.lb-bar').forEach(bar => {
  const w = bar.style.width;
  bar.style.width = '0%';
  bar.style.transition = 'width 1.2s cubic-bezier(0.4,0,0.2,1)';
  setTimeout(() => requestAnimationFrame(() => bar.style.width = w), 300);
});

// ── Tooltips ─────────────────────────────────────────────────
if (typeof bootstrap !== 'undefined') {
  document.querySelectorAll('[title]').forEach(el => {
    new bootstrap.Tooltip(el, { placement: 'top', trigger: 'hover' });
  });
}

// ── Live realtime prediction tick (index page demo) ──────────
let predCount = parseInt(document.getElementById('statPredictions')?.textContent) || 0;
const predEl = document.getElementById('statPredictions');
if (predEl) {
  setInterval(() => {
    // Cosmetic live counter ticks
  }, 15000);
}

// ── Chart.js global defaults (dark theme friendly) ──────────
if (typeof Chart !== 'undefined') {
  Chart.defaults.color = '#a8b2c1';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
  Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
}

console.log('%c🎓 Student AI v2 Ready', 'color:#6c5ce7;font-size:15px;font-weight:bold');
console.log('%cFeatures: SHAP Explainability | What-If Simulator | Cohort Compare | RBAC | Audit Log | API Keys', 'color:#00a8ff;font-size:11px');
