/* ── Config ──────────────────────────────────────────────────────────────── */
const API_BASE = 'http://localhost:8000';

const ALL_ACTIVITIES = [
  'anti-bacterial', 'anti-cancer', 'anti-fungal', 'anti-parasitic',
  'anti-viral', 'cell-cell-communication', 'drug-delivery',
  'immunological', 'inhibitor', 'metabolic', 'other-functional',
  'signal-peptide', 'toxic',
];

const COMPONENT_LABELS = {
  ngram_bleu: 'N-gram BLEU',
  charge: 'Charge',
  hydrophobicity: 'Hydrophobicity',
  functional_group: 'Functional Group',
  property_distribution: 'Property Distribution',
  structural: 'Structural',
  blosum: 'BLOSUM62',
};

/* ── State ───────────────────────────────────────────────────────────────── */
let activeActivities = new Set();
let customMode = false;   // true = free-text prompt mode

/* ── Init ────────────────────────────────────────────────────────────────── */
function init() {
  buildActivityGrid();
  updatePreview();
  checkHealth();
}

function buildActivityGrid() {
  const grid = document.getElementById('activityGrid');
  grid.innerHTML = '';
  ALL_ACTIVITIES.forEach(act => {
    const chip = document.createElement('label');
    chip.className = 'activity-chip';
    chip.dataset.act = act;
    chip.innerHTML = `<input type="checkbox" value="${act}" onchange="toggleActivity('${act}', this.checked)" />${act}`;
    grid.appendChild(chip);
  });
}

function toggleActivity(act, checked) {
  if (checked) activeActivities.add(act);
  else activeActivities.delete(act);
  const chip = document.querySelector(`.activity-chip[data-act="${act}"]`);
  if (chip) chip.classList.toggle('active', checked);
  updatePreview();
}

/* ── Custom prompt mode toggle ───────────────────────────────────────────── */
function toggleCustomMode() {
  customMode = !customMode;
  const preview   = document.getElementById('promptPreview');
  const hint      = document.getElementById('customHint');
  const form      = document.getElementById('structuredForm');
  const controls  = document.getElementById('customControls');
  const badge     = document.getElementById('modeBadge');
  const btnLabel  = document.getElementById('customBtnLabel');

  if (customMode) {
    preview.readOnly = false;
    preview.style.color = 'var(--text-primary)';
    preview.placeholder = 'Write your peptide specification here...\n\nExample:\nGenerate one antimicrobial peptide meeting ALL of:\n- Length: between 15 and 20 amino acids\n- Net charge: between +2 and +5\n- Hydrophobic fraction: between 35% and 55%\n- Alphabet: standard amino acids only\n\nOutput: one line, uppercase letters only.\nSequence:';
    preview.value = '';
    hint.style.display = 'block';
    // Keep the form visible so the user can still set ranges for validation
    form.style.display = 'block';
    controls.style.display = 'none';
    badge.textContent = 'CUSTOM';
    badge.style.background = 'rgba(59,130,246,0.2)';
    badge.style.color = '#60a5fa';
    badge.style.borderColor = '#3b82f6';
    btnLabel.textContent = '← Back to Form';
  } else {
    preview.readOnly = true;
    preview.style.color = '';
    preview.placeholder = 'Fill in the fields below — prompt preview will appear here';
    hint.style.display = 'none';
    form.style.display = 'block';
    controls.style.display = 'none';
    badge.textContent = 'STRUCTURED';
    badge.style.background = '';
    badge.style.color = '';
    badge.style.borderColor = '';
    btnLabel.textContent = 'Custom Prompt';
    updatePreview();
  }
}

/* ── Prompt preview (mirrors prompt_builder.py logic) ────────────────────── */
function buildPromptText() {
  const lenMin    = parseInt(document.getElementById('inputLenMin').value)  || 15;
  const lenMax    = parseInt(document.getElementById('inputLenMax').value)  || 20;
  const chargeMin = parseInt(document.getElementById('inputChargeMin').value);
  const chargeMax = parseInt(document.getElementById('inputChargeMax').value);
  const hydroMin  = parseInt(document.getElementById('inputHydroMin').value) || 35;
  const hydroMax  = parseInt(document.getElementById('inputHydroMax').value) || 55;
  const acts      = [...activeActivities];

  const actLabels = {
    'anti-bacterial': 'antimicrobial (AMP)', 'anti-fungal': 'antifungal (AMP)',
    'anti-viral': 'antiviral (AMP)', 'anti-cancer': 'anticancer (AMP)',
    'drug-delivery': 'cell-penetrating (CPP)', 'signal-peptide': 'signal peptide',
    'immunological': 'immunological / epitope',
  };
  const actStr = acts.length
    ? acts.map(a => actLabels[a] || a).join(', ')
    : 'general';

  const cLoStr = isNaN(chargeMin) ? '?' : (chargeMin >= 0 ? `+${chargeMin}` : `${chargeMin}`);
  const cHiStr = isNaN(chargeMax) ? '?' : (chargeMax >= 0 ? `+${chargeMax}` : `${chargeMax}`);

  return [
    `Generate one ${actStr} peptide meeting ALL of:`,
    `- Length: between ${lenMin} and ${lenMax} amino acids`,
    `- Net charge: between ${cLoStr} and ${cHiStr}`,
    `- Hydrophobic fraction: between ${hydroMin}% and ${hydroMax}%`,
    `- Alphabet: 20 standard amino acids only`,
    ``,
    `Output: one line, uppercase letters only, nothing else.`,
    `Sequence:`,
  ].join('\n');
}

function updatePreview() {
  if (customMode) return;
  document.getElementById('promptPreview').value = buildPromptText();
}

/* ── Quick-start presets ─────────────────────────────────────────────────── */
function applyPreset(name) {
  if (customMode) toggleCustomMode();   // switch back to structured form
  clearActivities();

  if (name === 'antimicrobial') {
    document.getElementById('inputLenMin').value    = 15;
    document.getElementById('inputLenMax').value    = 20;
    document.getElementById('inputChargeMin').value = 2;
    document.getElementById('inputChargeMax').value = 5;
    document.getElementById('inputHydroMin').value  = 35;
    document.getElementById('inputHydroMax').value  = 55;
    setActivity('anti-bacterial', true);
  } else if (name === 'cpp') {
    document.getElementById('inputLenMin').value    = 12;
    document.getElementById('inputLenMax').value    = 16;
    document.getElementById('inputChargeMin').value = 4;
    document.getElementById('inputChargeMax').value = 8;
    document.getElementById('inputHydroMin').value  = 30;
    document.getElementById('inputHydroMax').value  = 50;
    setActivity('drug-delivery', true);
  }
  updatePreview();
}

function setActivity(act, on) {
  const chip = document.querySelector(`.activity-chip[data-act="${act}"]`);
  if (!chip) return;
  const cb = chip.querySelector('input');
  cb.checked = on;
  if (on) activeActivities.add(act);
  else activeActivities.delete(act);
  chip.classList.toggle('active', on);
}

function clearActivities() {
  activeActivities.clear();
  document.querySelectorAll('.activity-chip').forEach(chip => {
    chip.classList.remove('active');
    chip.querySelector('input').checked = false;
  });
}

function clearForm() {
  if (customMode) toggleCustomMode();
  document.getElementById('inputLenMin').value    = 15;
  document.getElementById('inputLenMax').value    = 20;
  document.getElementById('inputChargeMin').value = 2;
  document.getElementById('inputChargeMax').value = 5;
  document.getElementById('inputHydroMin').value  = 35;
  document.getElementById('inputHydroMax').value  = 55;
  document.getElementById('inputRetries').value   = 6;
  document.getElementById('inputRef').value       = '';
  clearActivities();
  resetResults();
  updatePreview();
}

/* ── Health check ────────────────────────────────────────────────────────── */
async function checkHealth() {
  try {
    const r = await fetch(`${API_BASE}/health`);
    const data = await r.json();
    setStatus(data.status === 'ok');
  } catch {
    setStatus(false);
  }
}

function setStatus(ok) {
  const dot  = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  dot.className  = 'status-dot' + (ok ? '' : ' error');
  text.textContent = ok ? 'Ready' : 'Error';
}

/* ── Generation ──────────────────────────────────────────────────────────── */
let eventSource = null;

async function generate() {
  if (eventSource) { eventSource.close(); eventSource = null; }

  let body;

  if (customMode) {
    const promptText = document.getElementById('promptPreview').value.trim();
    if (!promptText) {
      alert('Please write a prompt in the text area first.');
      return;
    }
    const lenMin    = parseInt(document.getElementById('inputLenMin').value);
    const lenMax    = parseInt(document.getElementById('inputLenMax').value);
    const chargeMin = parseFloat(document.getElementById('inputChargeMin').value);
    const chargeMax = parseFloat(document.getElementById('inputChargeMax').value);
    const hydroMin  = parseFloat(document.getElementById('inputHydroMin').value);
    const hydroMax  = parseFloat(document.getElementById('inputHydroMax').value);
    const retries   = parseInt(document.getElementById('inputRetries').value) || 6;
    const ref       = document.getElementById('inputRef').value.trim() || null;
    body = {
      prompt_override: promptText,
      activities: [...activeActivities],
      max_retries: retries,
      threshold: 0.35,
      length_min: isNaN(lenMin) ? 15 : lenMin,
      length_max: isNaN(lenMax) ? 20 : lenMax,
      charge_min: isNaN(chargeMin) ? 0 : chargeMin,
      charge_max: isNaN(chargeMax) ? 5 : chargeMax,
      hydro_min: isNaN(hydroMin) ? undefined : hydroMin,
      hydro_max: isNaN(hydroMax) ? undefined : hydroMax,
      reference: ref,
    };
  } else {
    const lenMin    = parseInt(document.getElementById('inputLenMin').value);
    const lenMax    = parseInt(document.getElementById('inputLenMax').value);
    const chargeMin = parseFloat(document.getElementById('inputChargeMin').value);
    const chargeMax = parseFloat(document.getElementById('inputChargeMax').value);
    const hydroMin  = parseFloat(document.getElementById('inputHydroMin').value);
    const hydroMax  = parseFloat(document.getElementById('inputHydroMax').value);
    const retries   = parseInt(document.getElementById('inputRetries').value);
    const ref       = document.getElementById('inputRef').value.trim() || null;

    if (isNaN(lenMin) || isNaN(lenMax) || lenMin < 2 || lenMax < lenMin) {
      alert('Please enter a valid length range (min ≥ 2, max ≥ min).');
      return;
    }
    if (isNaN(chargeMin) || isNaN(chargeMax) || chargeMax < chargeMin) {
      alert('Please enter a valid charge range (max ≥ min).');
      return;
    }

    body = {
      length_min: lenMin,
      length_max: lenMax,
      charge_min: chargeMin,
      charge_max: chargeMax,
      hydro_min: isNaN(hydroMin) ? undefined : hydroMin,
      hydro_max: isNaN(hydroMax) ? undefined : hydroMax,
      activities: [...activeActivities],
      reference: ref,
      max_retries: retries,
      threshold: 0.35,
    };
  }

  resetResults();
  setGenerating(true);
  clearTrace();

  try {
    const response = await fetch(`${API_BASE}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const raw = line.slice(6).trim();
          if (!raw || raw === '[DONE]') continue;
          try { handleEvent(JSON.parse(raw)); } catch { /* skip malformed */ }
        }
      }
    }
  } catch (err) {
    addTraceRow({ n: '!', sequence: '', score: null, passed: false, issues: [String(err)] });
    setStatus(false);
  } finally {
    setGenerating(false);
  }
}

function handleEvent(evt) {
  if (evt.type === 'attempt') {
    addTraceRow(evt);
  } else if (evt.type === 'final') {
    renderResult(evt.result);
  }
}

/* ── Results rendering ───────────────────────────────────────────────────── */
function resetResults() {
  document.getElementById('resultSequence').textContent = 'ASSISTANT';
  document.getElementById('resultSequence').className   = 'result-sequence';
  document.getElementById('bleuValue').textContent  = '0.0000';
  document.getElementById('rbScoreValue').textContent = '0.0000';
  document.getElementById('iterValue').textContent  = '–';
  document.getElementById('timeValue').textContent  = '–';
  document.getElementById('metricBleu').className   = 'metric-badge metric-bleu';
  document.getElementById('metricRulebook').className = 'metric-badge metric-rulebook';
  document.getElementById('refNote').textContent = '';
  document.getElementById('componentCard').style.display = 'none';
}

function clearTrace() {
  document.getElementById('traceList').innerHTML =
    '<div class="trace-empty">Generating…</div>';
}

function addTraceRow(evt) {
  const list = document.getElementById('traceList');
  const empty = list.querySelector('.trace-empty');
  if (empty) empty.remove();

  const seqPreview = evt.sequence
    ? evt.sequence.slice(0, 20) + (evt.sequence.length > 20 ? '…' : '')
    : '—';
  const bleuStr = evt.score != null ? evt.score.toFixed(4) : '–';
  const rbStr   = evt.rulebook_score != null ? evt.rulebook_score.toFixed(2) : '–';
  const statusWord = evt.passed ? 'PASS' : (evt.sequence ? 'FAIL' : 'ERR');

  const row = document.createElement('div');
  row.className = `trace-row ${evt.passed ? 'pass' : (evt.sequence ? 'fail' : 'error')}`;
  row.innerHTML =
    `<span class="trace-status">${evt.passed ? '✓' : '✗'}</span>` +
    `Attempt ${evt.n}: ${seqPreview} | BLEU: ${bleuStr} | RB: ${rbStr} | ${statusWord}`;

  list.appendChild(row);
}

function renderResult(result) {
  const seq   = result.sequence || '';
  const score = result.score;
  const rb    = result.rulebook_score;

  const seqEl = document.getElementById('resultSequence');
  seqEl.textContent = seq || 'No sequence generated';
  const displayScore = score != null ? score : rb;
  if (displayScore != null) {
    seqEl.className = 'result-sequence ' + scoreClass(displayScore);
  }

  if (score != null) {
    document.getElementById('bleuValue').textContent = score.toFixed(4);
    document.getElementById('metricBleu').className =
      'metric-badge metric-bleu ' + scoreClass(score);
  }

  if (rb != null) {
    document.getElementById('rbScoreValue').textContent = rb.toFixed(4);
    document.getElementById('metricRulebook').className =
      'metric-badge metric-rulebook ' + scoreClass(rb);
  }

  const refNote = document.getElementById('refNote');
  if (result.reference_used && result.reference_used.startsWith('default:')) {
    const preset = result.reference_used.split(':')[1];
    refNote.textContent = `* PeptideBLEU scored vs canonical ${preset.toUpperCase()} reference`;
  } else if (result.reference_used === 'user') {
    refNote.textContent = '* PeptideBLEU scored vs your reference sequence';
  } else {
    refNote.textContent = '* No reference — PeptideBLEU not computed';
  }

  document.getElementById('iterValue').textContent = result.iterations ?? '–';
  document.getElementById('timeValue').textContent = result.time_seconds != null
    ? result.time_seconds.toFixed(1) + 's' : '–';

  if (result.components && Object.keys(result.components).length > 0) {
    renderComponents(result.components);
  }
}

function renderComponents(comp) {
  const card = document.getElementById('componentCard');
  const container = document.getElementById('componentScores');
  container.innerHTML = '';

  const order = ['ngram_bleu','charge','hydrophobicity','functional_group',
                  'property_distribution','structural','blosum'];

  order.forEach(key => {
    const val = comp[key] ?? 0;
    const label = COMPONENT_LABELS[key] || key;
    const pct = Math.round(val * 100);

    const row = document.createElement('div');
    row.className = 'comp-row';
    row.innerHTML = `
      <span class="comp-label">${label}</span>
      <div class="comp-bar-bg">
        <div class="comp-bar-fill" style="width:${pct}%"></div>
      </div>
      <span class="comp-val">${val.toFixed(4)}</span>
    `;
    container.appendChild(row);
  });

  card.style.display = 'block';
}

function scoreClass(s) {
  if (s >= 0.35) return 'score-good';
  if (s >= 0.25) return 'score-ok';
  return 'score-bad';
}

function setGenerating(on) {
  const btn = document.getElementById('btnGenerate');
  if (on) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Generating…';
  } else {
    btn.disabled = false;
    btn.innerHTML = '✨ GENERATE';
  }
}

/* ── Boot ────────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', init);
