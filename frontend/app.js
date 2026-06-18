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

/* ── Init ────────────────────────────────────────────────────────────────── */
let activeActivities = new Set();

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

/* ── Prompt preview (mirrors prompt_builder.py) ──────────────────────────── */
function buildPromptText() {
  const length   = document.getElementById('inputLength').value || '?';
  const charge   = parseFloat(document.getElementById('inputCharge').value) || 0;
  const hydro    = document.getElementById('inputHydro').value || '?';
  const acts     = [...activeActivities];

  const actLines = ALL_ACTIVITIES.map(a =>
    acts.includes(a) ? `- IS ${a}` : `- is NOT ${a}`
  ).join('\n');

  const chargeStr = charge >= 0 ? `+${charge}` : `${charge}`;

  let lines = [
    'Design a peptide with the following properties:',
    '',
    'BIOLOGICAL ACTIVITIES:',
    actLines,
    '',
    'PHYSICOCHEMICAL REQUIREMENTS:',
    `- Exact length: ${length} amino acids`,
    `- Net charge: ${chargeStr}`,
    `- Average hydrophobicity (Kyte-Doolittle): ${hydro}`,
    '',
    'STRICT RULES:',
    '1. Use ONLY standard single-letter amino acid codes: ACDEFGHIKLMNPQRSTVWY',
    '2. No modifications, numbers, spaces, or non-standard characters',
    '3. Output ONLY the sequence — nothing else',
  ];

  const ref = document.getElementById('inputRef').value.trim();
  if (ref) lines.push('', `Reference sequence available for scoring: ${ref}`);

  return lines.join('\n');
}

function updatePreview() {
  document.getElementById('promptPreview').value = buildPromptText();
}

/* ── Quick-start presets ──────────────────────────────────────────────────── */
function applyPreset(name) {
  clearActivities();
  if (name === 'antimicrobial') {
    document.getElementById('inputLength').value = 20;
    document.getElementById('inputCharge').value = 3;
    document.getElementById('inputHydro').value  = 0.5;
    setActivity('anti-bacterial', true);
    setActivity('anti-fungal', true);
  } else if (name === 'cpp') {
    document.getElementById('inputLength').value = 15;
    document.getElementById('inputCharge').value = 5;
    document.getElementById('inputHydro').value  = 0.3;
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
  document.getElementById('inputLength').value  = 12;
  document.getElementById('inputCharge').value  = 5;
  document.getElementById('inputHydro').value   = 0.5;
  document.getElementById('inputRetries').value = 6;
  document.getElementById('inputRef').value     = '';
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
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  const length   = parseInt(document.getElementById('inputLength').value);
  const charge   = parseFloat(document.getElementById('inputCharge').value);
  const hydro    = parseFloat(document.getElementById('inputHydro').value);
  const retries  = parseInt(document.getElementById('inputRetries').value);
  const ref      = document.getElementById('inputRef').value.trim() || null;

  if (isNaN(length) || length < 2) {
    alert('Please enter a valid length (min 2).');
    return;
  }

  const body = {
    length,
    charge,
    hydrophobicity: hydro,
    activities: [...activeActivities],
    reference: ref,
    max_retries: retries,
    threshold: 0.35,
  };

  resetResults();
  setGenerating(true);
  clearTrace();

  try {
    // POST the request and get SSE back
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

      // Process SSE lines
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const raw = line.slice(6).trim();
          if (!raw || raw === '[DONE]') continue;
          try {
            const evt = JSON.parse(raw);
            handleEvent(evt);
          } catch { /* skip malformed */ }
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
  const statusIcon = evt.passed ? '✓' : '✗';
  const statusWord = evt.passed ? 'PASS' : (evt.sequence ? 'FAIL' : 'ERR');

  const row = document.createElement('div');
  row.className = `trace-row ${evt.passed ? 'pass' : 'fail'}`;
  row.innerHTML =
    `<span class="trace-status">${statusIcon}</span>` +
    `Attempt ${evt.n}: ${seqPreview} | BLEU: ${bleuStr} | RB: ${rbStr} | ${statusWord}`;

  list.appendChild(row);
}

function renderResult(result) {
  const seq   = result.sequence || '';
  const score = result.score;
  const rb    = result.rulebook_score;

  // Sequence display
  const seqEl = document.getElementById('resultSequence');
  seqEl.textContent = seq || 'No sequence generated';
  const displayScore = score != null ? score : rb;
  if (displayScore != null) {
    seqEl.className = 'result-sequence ' + scoreClass(displayScore);
  }

  // PeptideBLEU badge
  if (score != null) {
    document.getElementById('bleuValue').textContent = score.toFixed(4);
    document.getElementById('metricBleu').className =
      'metric-badge metric-bleu ' + scoreClass(score);
  }

  // Rulebook fitness badge
  if (rb != null) {
    document.getElementById('rbScoreValue').textContent = rb.toFixed(4);
    document.getElementById('metricRulebook').className =
      'metric-badge metric-rulebook ' + scoreClass(rb);
  }

  // Reference note
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
