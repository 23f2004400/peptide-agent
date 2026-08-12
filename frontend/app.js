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
let currentPdb = null;    // raw PDB text for the last successful structure prediction
let structureViewer = null;

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
  document.getElementById('metricPlddt').style.display = 'none';
  document.getElementById('refNote').textContent = '';
  document.getElementById('componentCard').style.display = 'none';
  document.getElementById('dsspCard').style.display = 'none';
  document.getElementById('structureCard').style.display = 'none';
  currentPdb = null;
  structureViewer = null;
  document.getElementById('structureViewer').innerHTML = '';
  document.getElementById('graphCard').style.display = 'none';
  // ── Structural graph analysis UI (temporarily disabled) ──
  // document.getElementById('protein-graph-container').style.display = 'none';
}

function clearTrace() {
  document.getElementById('traceList').innerHTML =
    '<div class="trace-empty">Generating…</div>';
}

/* Generate/edit mode badges — additive alongside the existing PASS/FAIL/ERR trace rows */
const MODE_BADGES = {
  generate:            { label: 'GENERATE',          color: '#64748b' },
  llm_edit:            { label: null /* + weakest */, color: '#3b82f6' },
  deterministic:       { label: 'EDIT det',          color: '#00e5c3' },
  edit_explored:       { label: null /* + weakest */, color: '#fbbf24' },
  edit_stuck:          { label: 'edit_stuck',        color: '#64748b' },
  motif_inject:        { label: 'MOTIF INJECT',      color: '#10b981' },
  escape_positional:   { label: 'ESCAPE 1: positions', color: '#f97316' },
  escape_forced:       { label: 'ESCAPE 2: forced',  color: '#f97316' },
  escape_exact:        { label: 'ESCAPE 3: exact',   color: '#f97316' },
  generate_degenerate: { label: 'generate_degen',    color: '#ef4444' },
};

function modeBadgeHtml(mode, weakest) {
  if (!mode) return '';
  const cfg = MODE_BADGES[mode];
  if (!cfg) return '';
  const prefix = mode === 'edit_explored' ? 'explored' : 'EDIT';
  const label = cfg.label || `${prefix} ${weakest || ''}`.trim();
  return `<span class="mode-badge" style="color:${cfg.color};border-color:${cfg.color}">${label}</span>`;
}

function deltaHtml(delta) {
  if (delta == null) return '';
  if (delta > 0) return `<span class="delta-badge delta-up">+${delta.toFixed(4)} ↑</span>`;
  if (delta < 0) return `<span class="delta-badge delta-down">${delta.toFixed(4)} ↓</span>`;
  return `<span class="delta-badge delta-same">±0.0000 →</span>`;
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
    modeBadgeHtml(evt.mode, evt.weakest) +
    `Attempt ${evt.n}: ${seqPreview} | BLEU: ${bleuStr} | RB: ${rbStr} | ${statusWord}` +
    (evt.sequence ? deltaHtml(evt.delta_score) : '');

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

  renderPlddt(result.plddt_score, result.plddt_confidence);
  renderStructure(result.plddt_pdb);
  renderGraphFeatures(result.graph_features);
  renderDssp(result.secondary_structure);
  renderNovelty(result.blast_similarity);

  if (result.components && Object.keys(result.components).length > 0) {
    renderComponents(result.components, result.plddt_score, result.secondary_structure?.helix_pct);
  }
}

/* pLDDT (ESMFold) structural confidence — additive, alongside PeptideBLEU/rulebook */
const PLDDT_COLORS = {
  very_high: { color: '#00e5c3', label: 'Very High' },
  high:      { color: '#3b82f6', label: 'Confident' },
  low:       { color: '#fbbf24', label: 'Low' },
  very_low:  { color: '#ef4444', label: 'Very Low' },
  unknown:   { color: '#64748b', label: 'N/A' },
};

function renderPlddt(score, confidence) {
  const badge = document.getElementById('metricPlddt');
  const valueEl = document.getElementById('plddtValue');
  if (score == null) {
    badge.style.display = 'none';
    return;
  }
  const style = PLDDT_COLORS[confidence] || PLDDT_COLORS.unknown;
  valueEl.textContent = `${score.toFixed(1)}  ${style.label}`;
  valueEl.style.color = style.color;
  badge.style.display = '';
}

/* BLAST novelty assessment — additive, alongside pLDDT/DSSP */
const NOVELTY_COLORS = {
  novel:          { color: '#00e5c3', label: 'Novel' },
  low_similarity: { color: '#34d399', label: 'Mostly Novel' },
  similar:        { color: '#fbbf24', label: 'Similar' },
  known:          { color: '#ef4444', label: 'Known' },
  unknown:        { color: '#64748b', label: 'N/A' },
};

function renderNovelty(blast) {
  const badge = document.getElementById('metricNovelty');
  const valueEl = document.getElementById('noveltyValue');
  const card = document.getElementById('noveltyCard');
  const barsEl = document.getElementById('noveltyBars');
  const interpEl = document.getElementById('noveltyInterp');

  if (!blast) {
    badge.style.display = 'none';
    card.style.display = 'none';
    return;
  }

  const style = NOVELTY_COLORS[blast.novelty_label] || NOVELTY_COLORS.unknown;

  if (!blast.blast_available) {
    badge.style.display = 'none';
    barsEl.innerHTML = '';
    interpEl.textContent = 'BLAST not installed (sudo apt-get install ncbi-blast+)';
    card.style.display = 'block';
    return;
  }

  if (!blast.db_exists) {
    badge.style.display = 'none';
    barsEl.innerHTML = '';
    interpEl.textContent = 'Building database... (first run only)';
    card.style.display = 'block';
    return;
  }

  valueEl.textContent = style.label;
  valueEl.style.color = style.color;
  badge.style.display = '';

  barsEl.innerHTML = '';
  if (blast.similarity_score != null) {
    const row = document.createElement('div');
    row.className = 'comp-row';
    row.innerHTML = `
      <span class="comp-label">Similarity to Database</span>
      <div class="comp-bar-bg">
        <div class="comp-bar-fill" style="width:${Math.round(blast.similarity_score * 100)}%;background:${style.color}"></div>
      </div>
      <span class="comp-val">${blast.similarity_score.toFixed(2)}</span>
    `;
    barsEl.appendChild(row);
  }

  interpEl.textContent = blast.interpretation || '';
  card.style.display = 'block';
}

/* 3D structure (ESMFold PDB) — embedded 3Dmol.js viewer + raw-file download */
function renderStructure(pdbText) {
  const card = document.getElementById('structureCard');
  currentPdb = pdbText || null;

  if (!pdbText) {
    card.style.display = 'none';
    return;
  }

  const container = document.getElementById('structureViewer');
  container.innerHTML = '';

  if (typeof $3Dmol === 'undefined') {
    container.innerHTML = '<div class="trace-empty">3Dmol.js failed to load (check network/CDN access)</div>';
    card.style.display = 'block';
    return;
  }

  structureViewer = $3Dmol.createViewer(container, { backgroundColor: '#000000' });
  structureViewer.addModel(pdbText, 'pdb');
  // Color by pLDDT (stored in the B-factor column by ESMFold), same bands as the badge.
  structureViewer.setStyle({}, {
    cartoon: {
      colorfunc: (atom) => {
        const b = atom.b;
        if (b >= 90) return '#00e5c3';
        if (b >= 70) return '#3b82f6';
        if (b >= 50) return '#fbbf24';
        return '#ef4444';
      },
    },
  });
  structureViewer.zoomTo();
  structureViewer.render();

  card.style.display = 'block';
}

function downloadPdb() {
  if (!currentPdb) return;
  const blob = new Blob([currentPdb], { type: 'chemical/x-pdb' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const seq = document.getElementById('resultSequence').textContent.trim();
  a.download = `${seq || 'structure'}.pdb`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* Structural graph analysis (H-bonds/ionic/pi-pi interactions) — additive,
   derived from the same ESMFold PDB structure as the pLDDT badge/viewer. */
function graphScoreStyle(score) {
  if (score >= 3.0) return { color: '#00e5c3', label: 'Highly Structured' };
  if (score >= 1.5) return { color: '#3b82f6', label: 'Moderately Structured' };
  if (score >= 0.5) return { color: '#fbbf24', label: 'Loosely Structured' };
  return { color: '#64748b', label: 'Flexible / Disordered' };
}

function renderGraphFeatures(graph) {
  const card = document.getElementById('graphCard');
  if (!graph || graph.structure_score == null) {
    card.style.display = 'none';
    // document.getElementById('protein-graph-container').style.display = 'none';
    return;
  }

  const style = graphScoreStyle(graph.structure_score);
  const bar = document.getElementById('graphScoreBar');
  const val = document.getElementById('graphScoreVal');
  bar.style.width = `${Math.min(100, graph.structure_score / 5 * 100)}%`;
  bar.style.background = style.color;
  val.textContent = `${graph.structure_score.toFixed(2)}`;
  val.style.color = style.color;

  const counts = document.getElementById('graphCounts');
  counts.innerHTML = `
    <span>H-bonds: <span class="count-val">${graph.n_hbonds ?? '–'}</span></span>
    <span>Ionic: <span class="count-val">${graph.n_ionic ?? '–'}</span></span>
    <span>Pi-Pi Stacking: <span class="count-val">${graph.n_pi_pi ?? '–'}</span></span>
  `;

  document.getElementById('graphInterp').textContent = graph.interpretation || '';
  card.style.display = 'block';

  // ── Structural graph analysis UI (temporarily disabled) ──
  // if (graph.edges && graph.nodes) {
  //   renderProteinGraph(graph);
  // } else {
  //   document.getElementById('protein-graph-container').style.display = 'none';
  // }
}

/* Secondary structure (DSSP) — additive, derived from the same ESMFold PDB
   structure as pLDDT. Fails gracefully (message, not a crash) whenever the
   mkdssp/dssp binary isn't installed on the server, since that's expected
   on plenty of machines and never blocks generation. */
const SS_BAR_COLORS = {
  helix_pct: '#00e5c3',
  sheet_pct: '#3b82f6',
  turn_pct: '#fbbf24',
  coil_pct: '#64748b',
};
const SS_BAR_LABELS = {
  helix_pct: 'Alpha Helix',
  sheet_pct: 'Beta Sheet',
  turn_pct: 'Turn',
  coil_pct: 'Coil',
};

function renderDssp(ss) {
  const card = document.getElementById('dsspCard');
  if (!ss) {
    card.style.display = 'none';
    return;
  }

  const ssStringEl = document.getElementById('dsspSsString');
  const barsEl = document.getElementById('dsspBars');
  const simEl = document.getElementById('dsspSimilarity');
  const interpEl = document.getElementById('dsspInterp');
  ssStringEl.innerHTML = '';
  barsEl.innerHTML = '';
  simEl.innerHTML = '';
  interpEl.textContent = '';

  if (!ss.dssp_available) {
    interpEl.textContent = 'DSSP binary not installed on server. Install with: sudo apt-get install dssp';
    card.style.display = 'block';
    return;
  }

  if (ss.error || ss.helix_pct == null) {
    interpEl.textContent = 'Secondary structure: unavailable';
    card.style.display = 'block';
    return;
  }

  ssStringEl.textContent = ss.ss_string || '';

  ['helix_pct', 'sheet_pct', 'turn_pct', 'coil_pct'].forEach(key => {
    const val = ss[key] ?? 0;
    const row = document.createElement('div');
    row.className = 'comp-row';
    row.innerHTML = `
      <span class="comp-label">${SS_BAR_LABELS[key]}</span>
      <div class="comp-bar-bg">
        <div class="comp-bar-fill" style="width:${Math.round(val)}%;background:${SS_BAR_COLORS[key]}"></div>
      </div>
      <span class="comp-val">${val.toFixed(1)}%</span>
    `;
    barsEl.appendChild(row);
  });

  if (ss.ss_similarity) {
    const sim = ss.ss_similarity;
    const simRow = document.createElement('div');
    simRow.className = 'comp-row';
    simRow.innerHTML = `
      <span class="comp-label">SS Similarity vs Ref</span>
      <div class="comp-bar-bg">
        <div class="comp-bar-fill" style="width:${Math.round((sim.overall_ss_similarity ?? 0) * 100)}%"></div>
      </div>
      <span class="comp-val">${(sim.overall_ss_similarity ?? 0).toFixed(3)}</span>
    `;
    simEl.appendChild(simRow);
  }

  interpEl.textContent = ss.interpretation || '';
  card.style.display = 'block';
}

/* 2D residue-interaction graph (D3 force layout) — raw edges/nodes come from
   backend/graph_features.py's pdb_to_graph_features(), same PDB structure as
   the summary metrics above. Purely additive: renderGraphFeatures() already
   hides the whole card (and this container) whenever graph_features is null. */
const RESIDUE_COLORS = {
  K: '#3b82f6', R: '#3b82f6', H: '#60a5fa',
  D: '#ef4444', E: '#f87171',
  L: '#00e5c3', I: '#00e5c3', V: '#00e5c3',
  F: '#00e5c3', W: '#00e5c3', M: '#00e5c3',
  A: '#34d399', P: '#34d399',
  S: '#fbbf24', T: '#fbbf24', N: '#fbbf24',
  Q: '#fbbf24', Y: '#f59e0b', C: '#d97706',
  G: '#9ca3af',
};

const EDGE_COLORS = {
  hbond: '#fbbf24',
  ionic: '#3b82f6',
  pi_pi: '#a855f7',
  hydro: '#34d399',
  ca_dist: 'rgba(255,255,255,0.05)',
};

function renderProteinGraph(graphData) {
  if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
    document.getElementById('protein-graph-container').style.display = 'none';
    return;
  }
  if (typeof d3 === 'undefined') {
    // CDN unreachable — fail gracefully, summary metrics above still show.
    document.getElementById('protein-graph-container').style.display = 'none';
    return;
  }

  const container = document.getElementById('protein-graph-container');
  const svg = d3.select('#protein-graph');
  container.style.display = 'block';
  svg.selectAll('*').remove();

  const width = svg.node().getBoundingClientRect().width || 600;
  const height = 320;
  svg.attr('viewBox', `0 0 ${width} ${height}`);

  const nodes = graphData.nodes.map(n => ({ ...n }));

  // Backbone (residue i -> i+1) — pure sequence adjacency, synthesized here
  // rather than sourced from the API. Without this, any residue with zero
  // interaction edges has nothing anchoring it, so forceCenter pulls it to
  // the exact center where it overlaps other orphaned nodes — the "missing
  // residues" artifact. As a real force-link every node is now anchored to
  // its sequence neighbors, and the chain is traceable N- to C-terminus.
  const backbone = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    backbone.push({ source: i, target: i + 1, weight: 0.05, type: 'backbone' });
  }

  // ca_dist omitted — too dense (near-complete graph), clutters the render.
  const edges = [
    ...(graphData.edges.hbonds || []),
    ...(graphData.edges.ionic || []),
    ...(graphData.edges.pi_pi || []),
    ...(graphData.edges.hydro || []),
  ].filter(e => e.weight > 0.005).map(e => ({ ...e }));

  const simulation = d3.forceSimulation(nodes)
    .force('backbone', d3.forceLink(backbone)
      .id(d => d.id)
      .distance(30)
      .strength(0.6))
    .force('interactions', d3.forceLink(edges)
      .id(d => d.id)
      .distance(d => 40 / (d.weight + 0.01))
      .strength(0.3))
    .force('charge', d3.forceManyBody().strength(-80))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide(18))
    .stop();

  // forceCenter only pulls the *average* position toward center — it does
  // not stop individual nodes drifting outside the visible canvas, and with
  // 15-20 residues + repulsion there's not enough room in a 320px-tall box
  // for that not to happen. Hard-clamp every node back inside the viewBox
  // (minus a margin for the node radius/label) after every tick so nothing
  // ends up rendered off-screen with only its edge-stub visible.
  const margin = 20;
  for (let i = 0; i < 200; i++) {
    simulation.tick();
    for (const n of nodes) {
      n.x = Math.max(margin, Math.min(width - margin, n.x));
      n.y = Math.max(margin, Math.min(height - margin, n.y));
    }
  }

  // Backbone drawn first (underneath) so the colored interaction edges sit
  // on top of it.
  svg.append('g').attr('class', 'backbone')
    .selectAll('line')
    .data(backbone)
    .join('line')
    .attr('x1', d => d.source.x)
    .attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x)
    .attr('y2', d => d.target.y)
    .attr('stroke', 'rgba(255,255,255,0.25)')
    .attr('stroke-width', 1.5);

  svg.append('g').attr('class', 'edges')
    .selectAll('line')
    .data(edges)
    .join('line')
    .attr('x1', d => d.source.x)
    .attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x)
    .attr('y2', d => d.target.y)
    .attr('stroke', d => EDGE_COLORS[d.type] || '#ffffff')
    .attr('stroke-width', d => Math.min(d.weight * 20 + 0.5, 2.5))
    .attr('stroke-opacity', 0.6);

  const nodeGroup = svg.append('g').attr('class', 'nodes')
    .selectAll('g')
    .data(nodes)
    .join('g')
    .attr('transform', d => `translate(${d.x},${d.y})`);

  nodeGroup.append('circle')
    .attr('r', 10)
    .attr('fill', d => RESIDUE_COLORS[d.residue] || '#64748b')
    .attr('fill-opacity', 0.85)
    .attr('stroke', 'rgba(255,255,255,0.2)')
    .attr('stroke-width', 1);

  nodeGroup.append('text')
    .attr('class', 'node-label')
    .text(d => d.residue);

  nodeGroup.append('title')
    .text(d => `${d.residue} (pos ${d.id + 1})`);
}

function renderComponents(comp, plddtScore, helixPct) {
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

  // pLDDT is on a 0-100 scale (not 0-1 like the other components) — shown
  // separately since it's a structural metric, not a PeptideBLEU component.
  if (plddtScore != null) {
    const row = document.createElement('div');
    row.className = 'comp-row';
    row.innerHTML = `
      <span class="comp-label">pLDDT (ESMFold)</span>
      <div class="comp-bar-bg">
        <div class="comp-bar-fill" style="width:${Math.round(plddtScore)}%"></div>
      </div>
      <span class="comp-val">${plddtScore.toFixed(1)}</span>
    `;
    container.appendChild(row);
  }

  // Helix % (DSSP) — also 0-100 scale, shown right after pLDDT for the
  // same reason (structural metric, not a PeptideBLEU component).
  if (helixPct != null) {
    const row = document.createElement('div');
    row.className = 'comp-row';
    row.innerHTML = `
      <span class="comp-label">Helix % (DSSP)</span>
      <div class="comp-bar-bg">
        <div class="comp-bar-fill" style="width:${Math.round(helixPct)}%;background:#00e5c3"></div>
      </div>
      <span class="comp-val">${helixPct.toFixed(1)}%</span>
    `;
    container.appendChild(row);
  }

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
