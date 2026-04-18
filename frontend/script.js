/**
 * CardioSense — Heart Disease Prediction System
 * Frontend Logic (4-page SPA)
 */

const API_BASE = window.location.origin;
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ═══════════════════════════════════════════════════════
//  NAVIGATION
// ═══════════════════════════════════════════════════════
function navigateTo(pageId) {
  $$('.page').forEach(p => p.classList.remove('active'));
  $$('.nav-link').forEach(l => l.classList.remove('active'));

  const page = $(`#${pageId}`);
  if (page) {
    page.classList.add('active');
    // Re-trigger animation
    page.style.animation = 'none';
    page.offsetHeight; // reflow
    page.style.animation = '';
  }

  const link = $(`.nav-link[data-page="${pageId}"]`);
  if (link) link.classList.add('active');

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Handle nav link clicks
$$('.nav-link').forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    navigateTo(link.dataset.page);
  });
});

// ═══════════════════════════════════════════════════════
//  HEALTH CHECK
// ═══════════════════════════════════════════════════════
const statusDot = $('#statusDot');
const statusText = $('#statusText');

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (data.status === 'ok') {
      statusDot.classList.remove('offline');
      const parts = [];
      if (data.binary_loaded) parts.push('Binary');
      if (data.multi_loaded) parts.push('Multiclass');
      statusText.textContent = parts.length
        ? `Online · ${parts.join(' + ')} loaded`
        : 'Online · No model trained';
    } else {
      throw new Error('unhealthy');
    }
  } catch {
    statusDot.classList.add('offline');
    statusText.textContent = 'Offline';
  }
}
checkHealth();
setInterval(checkHealth, 30000);

// ═══════════════════════════════════════════════════════
//  TOAST
// ═══════════════════════════════════════════════════════
const toastEl = $('#toast');
function showToast(msg, duration = 5000) {
  toastEl.textContent = msg;
  toastEl.classList.add('visible');
  setTimeout(() => toastEl.classList.remove('visible'), duration);
}

// ═══════════════════════════════════════════════════════
//  SLIDER LIVE VALUES
// ═══════════════════════════════════════════════════════
const sliders = {
  age: 'ageVal',
  trestbps: 'trestbpsVal',
  chol: 'cholVal',
  thalach: 'thalachVal',
  oldpeak: 'oldpeakVal',
  ca: 'caVal',
};

Object.entries(sliders).forEach(([id, valId]) => {
  const slider = $(`#${id}`);
  const valEl = $(`#${valId}`);
  if (slider && valEl) {
    slider.addEventListener('input', () => {
      valEl.textContent = slider.value;
    });
  }
});

// ═══════════════════════════════════════════════════════
//  FORM RESET
// ═══════════════════════════════════════════════════════
function resetForm() {
  const form = $('#predictForm');
  form.reset();
  // Reset slider display values
  Object.entries(sliders).forEach(([id, valId]) => {
    const slider = $(`#${id}`);
    const valEl = $(`#${valId}`);
    if (slider && valEl) valEl.textContent = slider.value;
  });
}

// ═══════════════════════════════════════════════════════
//  FORM SUBMISSION
// ═══════════════════════════════════════════════════════
const form = $('#predictForm');
const btn = $('#btnPredict');
const loadingStep = $('#loadingStep');

const loadingSteps = [
  'Running XGBoost…',
  'Running DNN…',
  'Running Random Forest…',
  'SVC Meta-Learner…',
  'Generating Results…',
];

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  // Collect & validate
  const fd = new FormData(form);
  const payload = {};
  for (const [k, v] of fd.entries()) {
    payload[k] = parseFloat(v);
    if (isNaN(payload[k])) {
      showToast(`Invalid value for ${k}`);
      return;
    }
  }

  // Check all selects are filled
  const selects = form.querySelectorAll('select[required]');
  for (const sel of selects) {
    if (!sel.value) {
      showToast(`Please select a value for ${sel.previousElementSibling.textContent.trim()}`);
      sel.focus();
      return;
    }
  }

  // Start loading animation
  btn.classList.add('loading');
  btn.disabled = true;

  // Cycle through loading steps
  let stepIdx = 0;
  const stepInterval = setInterval(() => {
    stepIdx = (stepIdx + 1) % loadingSteps.length;
    loadingStep.textContent = loadingSteps[stepIdx];
  }, 600);

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    if (data.error) throw new Error(data.error);

    // Save to localStorage
    localStorage.setItem('lastPrediction', JSON.stringify(data));
    localStorage.setItem('lastPatient', JSON.stringify(payload));

    // Render results
    renderResults(data, payload);

    // Navigate to results
    navigateTo('results');

  } catch (err) {
    showToast(`Prediction failed: ${err.message}`);
    console.error(err);
  } finally {
    clearInterval(stepInterval);
    loadingStep.textContent = 'Analysing…';
    btn.classList.remove('loading');
    btn.disabled = false;
  }
});

// ═══════════════════════════════════════════════════════
//  RENDER RESULTS
// ═══════════════════════════════════════════════════════
const severityColors = ['#1D9E75', '#06B6D4', '#BA7517', '#f97316', '#E24B4A'];
const modelColors = {
  xgboost: '#3B82F6',
  dnn: '#8B5CF6',
  random_forest: '#1D9E75',
};
const modelNames = {
  xgboost: 'XGBoost',
  dnn: 'DNN (MLP)',
  random_forest: 'Random Forest',
};

const featureLabels = {
  age: 'Age', sex: 'Sex', cp: 'Chest Pain', trestbps: 'Resting BP',
  chol: 'Cholesterol', fbs: 'Fasting BS', restecg: 'Resting ECG',
  thalach: 'Max HR', exang: 'Exercise Angina', oldpeak: 'ST Depression',
  slope: 'Slope', ca: 'Vessels (CA)', thal: 'Thalassemia',
};

function renderResults(data, patientData) {
  $('#resultsPlaceholder').style.display = 'none';
  $('#resultsContent').style.display = 'block';

  const b = data.binary;
  const s = data.severity;
  const isPositive = b && b.prediction === 1;
  const riskPct = b ? b.probabilities['Disease'] : 0;

  // 1. Risk Gauge
  renderGauge(riskPct);

  // 2. Verdict Banner
  const banner = $('#verdictBanner');
  banner.className = `verdict-banner ${isPositive ? 'positive' : 'negative'}`;
  $('#verdictIcon').textContent = isPositive ? '⚠️' : '✅';
  $('#verdictText').textContent = b ? b.label : 'Unknown';
  $('#verdictText').style.color = isPositive ? 'var(--accent-red)' : 'var(--accent-green)';
  $('#verdictConfidence').textContent = b ? `Confidence: ${b.confidence}%` : '';

  // 3. Model Contribution Bars
  const contribDiv = $('#modelContribBars');
  contribDiv.innerHTML = '';
  if (b && b.model_contributions) {
    for (const [name, info] of Object.entries(b.model_contributions)) {
      const color = modelColors[name] || '#6366f1';
      const displayName = modelNames[name] || name;
      contribDiv.innerHTML += `
        <div class="model-bar-row">
          <span class="model-bar-name">${displayName}</span>
          <div class="model-bar-bg">
            <div class="model-bar-fill" style="width:0%; background:${color}" data-width="${info.confidence}%"></div>
          </div>
          <span class="model-bar-pct" style="color:${color}">${info.confidence}%</span>
        </div>`;
    }
  }

  // Ensemble decision
  $('#ensembleDecision').innerHTML = `
    <strong>Ensemble Decision:</strong> ${b ? b.label : 'N/A'} 
    <span style="color:var(--text-muted)">· SVC Meta-Learner · ${b ? b.confidence : 0}% confidence</span>`;

  // 4. Feature Contribution Chart
  const featDiv = $('#featureChart');
  featDiv.innerHTML = '';
  if (data.feature_importance && data.feature_importance.length) {
    const maxImp = data.feature_importance[0].importance;
    data.feature_importance.forEach(fi => {
      const pct = ((fi.importance / maxImp) * 100).toFixed(1);
      const isRisk = ['ca', 'oldpeak', 'thal', 'cp', 'exang'].includes(fi.feature);
      const color = isRisk
        ? 'linear-gradient(90deg, #E24B4A, #f97316)'
        : 'linear-gradient(90deg, #1D9E75, #06B6D4)';
      featDiv.innerHTML += `
        <div class="feature-bar-row">
          <span class="feature-bar-name">${featureLabels[fi.feature] || fi.feature}</span>
          <div class="feature-bar-bg">
            <div class="feature-bar-fill" style="width:0%; background:${color}" data-width="${pct}%"></div>
          </div>
          <span class="feature-bar-val">${(fi.importance * 100).toFixed(1)}%</span>
        </div>`;
    });
  }

  // 5. Severity Assessment
  const sevDiv = $('#severityContent');
  sevDiv.innerHTML = '';
  if (s) {
    const sevLabel = s.label;
    sevDiv.innerHTML += `<div style="font-size:1.3rem; font-weight:800; color:${severityColors[s.prediction] || '#fff'}; margin-bottom:0.75rem;">${sevLabel}</div>`;
    Object.entries(s.probabilities).forEach(([label, pct], i) => {
      const color = severityColors[i] || '#6366f1';
      sevDiv.innerHTML += `
        <div class="severity-bar-row">
          <span class="severity-bar-label">${label}</span>
          <div class="severity-bar-bg">
            <div class="severity-bar-fill" style="width:0%; background:${color}" data-width="${pct}%"></div>
          </div>
          <span class="severity-bar-pct" style="color:${color}">${pct}%</span>
        </div>`;
    });
  } else {
    sevDiv.innerHTML = '<p style="color:var(--text-muted)">Severity model not loaded.</p>';
  }

  // 6. Patient Summary
  const sumDiv = $('#patientSummary');
  const selectLabels = {
    sex: { 0: 'Female', 1: 'Male' },
    cp: { 0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal', 3: 'Asymptomatic' },
    fbs: { 0: 'No', 1: 'Yes' },
    restecg: { 0: 'Normal', 1: 'ST-T Abnormality', 2: 'LV Hypertrophy' },
    exang: { 0: 'No', 1: 'Yes' },
    slope: { 0: 'Upsloping', 1: 'Flat', 2: 'Downsloping' },
    thal: { 3: 'Normal', 6: 'Fixed Defect', 7: 'Reversible Defect' },
  };

  let tableHtml = '<table class="summary-table">';
  for (const [key, val] of Object.entries(patientData)) {
    const label = featureLabels[key] || key;
    let displayVal = val;
    if (selectLabels[key] && selectLabels[key][val] !== undefined) {
      displayVal = selectLabels[key][val];
    }
    tableHtml += `<tr><td>${label}</td><td>${displayVal}</td></tr>`;
  }
  tableHtml += '</table>';
  sumDiv.innerHTML = tableHtml;

  // 7. Clinical Interpretation
  const clinDiv = $('#clinicalContent');
  const riskLevel = b ? b.risk_level : 'low';
  const interpretations = {
    low: {
      title: '✅ Low Risk',
      text: 'Your risk indicators are within normal range. Maintain a healthy lifestyle with regular exercise, balanced diet, and periodic cardiac check-ups.',
      class: 'low',
    },
    moderate: {
      title: '⚠️ Moderate Risk',
      text: 'Some risk factors are present. We recommend consulting your cardiologist for further evaluation, including stress tests and detailed blood work.',
      class: 'moderate',
    },
    high: {
      title: '🚨 High Risk',
      text: 'Multiple high-risk indicators have been detected. Please seek immediate medical consultation. An electrocardiogram (ECG) and coronary angiography may be recommended.',
      class: 'high',
    },
  };
  const interp = interpretations[riskLevel] || interpretations.low;
  clinDiv.innerHTML = `
    <div class="clinical-box ${interp.class}">
      <div class="clinical-title">${interp.title}</div>
      <p>${interp.text}</p>
      <p style="margin-top:0.75rem; opacity:0.7; font-size:0.78rem;">⚕️ <em>This is an AI-assisted assessment. Always consult a qualified healthcare professional for medical decisions.</em></p>
    </div>`;

  // Animate bars after DOM paint
  requestAnimationFrame(() => {
    setTimeout(() => {
      $$('[data-width]').forEach(bar => {
        bar.style.width = bar.dataset.width;
      });
    }, 100);
  });
}

// ═══════════════════════════════════════════════════════
//  SVG RISK GAUGE
// ═══════════════════════════════════════════════════════
function renderGauge(riskPct) {
  const gaugeArc = $('#gaugeArc');
  const gaugeNeedle = $('#gaugeNeedle');
  const gaugeValue = $('#gaugeValue');

  const totalLength = 251.3; // approximation of arc length
  const fillLength = (riskPct / 100) * totalLength;

  // Animate
  gaugeArc.style.transition = 'stroke-dasharray 1.2s cubic-bezier(0.4,0,0.2,1)';
  gaugeArc.setAttribute('stroke-dasharray', `${fillLength} ${totalLength}`);

  // Needle rotation: -90 (left) to +90 (right) mapped to 0-100%
  const angle = -90 + (riskPct / 100) * 180;
  gaugeNeedle.style.transition = 'transform 1.2s cubic-bezier(0.4,0,0.2,1)';
  gaugeNeedle.setAttribute('transform', `rotate(${angle} 100 100)`);

  // Value text
  gaugeValue.textContent = `${Math.round(riskPct)}%`;

  // Color
  if (riskPct < 40) {
    gaugeValue.style.color = 'var(--accent-green)';
  } else if (riskPct < 65) {
    gaugeValue.style.color = 'var(--accent-amber)';
  } else {
    gaugeValue.style.color = 'var(--accent-red)';
  }
}

// ═══════════════════════════════════════════════════════
//  DOWNLOAD REPORT (Print-based PDF)
// ═══════════════════════════════════════════════════════
function downloadReport() {
  window.print();
}

// ═══════════════════════════════════════════════════════
//  LOAD MODEL INFO (confusion matrix)
// ═══════════════════════════════════════════════════════
async function loadModelInfo() {
  try {
    const res = await fetch(`${API_BASE}/api/model-info`);
    const data = await res.json();

    // Confusion matrix
    if (data.binary && data.binary.metrics && data.binary.metrics.confusion_matrix) {
      const cm = data.binary.metrics.confusion_matrix;
      const card = $('#confusionCard');
      card.style.display = 'block';
      const grid = $('#confusionMatrix');
      if (cm.length >= 2) {
        grid.innerHTML = `
          <div class="confusion-cell cell-tn">${cm[0][0]}<span class="cell-label">True Neg</span></div>
          <div class="confusion-cell cell-fp">${cm[0][1]}<span class="cell-label">False Pos</span></div>
          <div class="confusion-cell cell-fn">${cm[1][0]}<span class="cell-label">False Neg</span></div>
          <div class="confusion-cell cell-tp">${cm[1][1]}<span class="cell-label">True Pos</span></div>`;
      }
    }
  } catch {
    // Model info not available
  }
}
loadModelInfo();

// ═══════════════════════════════════════════════════════
//  RESTORE LAST PREDICTION FROM localStorage
// ═══════════════════════════════════════════════════════
(function restoreLastPrediction() {
  const saved = localStorage.getItem('lastPrediction');
  const patient = localStorage.getItem('lastPatient');
  if (saved && patient) {
    try {
      renderResults(JSON.parse(saved), JSON.parse(patient));
    } catch {} // ignore parse errors
  }
})();
