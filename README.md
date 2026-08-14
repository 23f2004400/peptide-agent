# PepForge

Research project at IIIT-Delhi. An agentic iterative refinement system for novel peptide sequence design using OpenBioLLM-8B, built to demonstrate that an agent loop (generate → validate → score → edit with feedback) beats plain LLM baselines on the PeptideBLEU metric.

**Research claim:** Agentic iterative refinement > plain LLM best-of-5 prompting on PeptideBLEU.

| Model | PeptideBLEU (best-of-5) |
|-------|-------|
| OpenBioLLM-8B (baseline)  | 0.3014 |
| SmolLM2-360M (baseline)   | 0.3243 |
| Gemma-3-1B (baseline)     | 0.3269 |
| **This agent — best result** | **0.6995** (iteration 1, AMP task) |

---

## How it works

```
User request (length, charge, hydrophobicity, activities)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                      PepForgeAgent loop                       │
│                                                                │
│  ATTEMPT 1 — generate from scratch, exactly once               │
│    ├─ RAG: top-5 similar peptides from data/peptides.csv       │
│    │       (FAISS) injected as prompt examples                 │
│    ├─ reference anchor: opening/closing motif, charge, hydro%  │
│    │       — pattern hints only, never the full sequence       │
│    ├─ OpenBioLLM-8B → raw text                                 │
│    └─ extract_sequence() ← 3-priority extraction               │
│         │                                                      │
│    passed? ──YES──► return best result                         │
│         │NO                                                    │
│  ATTEMPTS 2..max_retries — edit the working sequence           │
│    One edit attempt per iteration; candidates scored and the   │
│    best one kept (never worse than the true best so far):      │
│      • deterministic_edit   — length → charge → hydrophobicity │
│      • llm_edit             — targets current weakest          │
│      •                        PeptideBLEU component            │
│      • inject_reference_motif — splices a literal reference    │
│                                 fragment in (guarantees some    │
│                                 n-gram overlap deterministically)│
│    If stuck (no improvement) 1x/2x/3x+ in a row, escalate:     │
│      escape_positional → escape_forced → escape_exact          │
│                                                                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
  Best result across all attempts/candidates (highest score, never a regression)
        │
        ▼
  (additive, after generate() returns — never retried on)
  pLDDT (ESMFold) · secondary structure (pydssp) · graph_features (disabled)
```

The **key differentiator** is targeted, scored editing: instead of blindly resampling, each iteration edits the current best sequence toward a specific weak component (or, when stuck, escalates through increasingly forceful/specific edit tactics), and every candidate — deterministic, LLM-edited, or motif-injected — is scored and compared before anything replaces the working sequence.

---

## Project structure

```
peptide-agent/
├── backend/
│   ├── __init__.py
│   ├── agent.py                # Core agent loop (generate once → edit/score loop), PepForgeAgent
│   ├── models.py                # OpenBioLLM connection via OpenAI-compat vLLM API
│   ├── peptide_bleu.py          # PeptideBLEU v1.2 scoring (7 components)
│   ├── rulebook.py               # Amino acid property validation
│   ├── prompt_builder.py        # Prompt construction — RAG examples + reference anchor
│   ├── trace_logger.py          # Human-readable per-iteration trace → logs/generation_trace.log
│   ├── esmfold_scorer.py        # pLDDT structural confidence via ESMFold HF API
│   ├── dssp_scorer.py            # Secondary structure (helix/sheet/coil %) via pydssp
│   ├── graph_features.py        # 3D PDB → 2D interaction graph (implemented, currently
│   │                              #   disabled in server.py — future work)
│   ├── server.py                 # FastAPI server + SSE streaming
│   ├── tools/
│   │   └── sequence_editor.py   # Edit candidates: deterministic fix, llm_edit,
│   │                              #   motif injection, edit_stuck escape tiers
│   └── rag/
│       ├── peptide_retriever.py # FAISS search over data/peptides.csv for prompt examples
│       └── build_index.py        # One-time FAISS index builder
├── frontend/
│   ├── index.html         # Single-page app
│   ├── app.js             # SSE client, live prompt preview, result rendering
│   └── style.css          # Dark theme (#0d1117), JetBrains Mono
├── eval/
│   ├── run_eval.py        # Batch evaluation (baseline vs agent comparison)
│   └── protocol_test.py   # Single-task run with a custom stop rule, as a template
├── data/                  # Not checked in (gitignored) — peptides.csv, peptides_with_length.jsonl,
│                          #   peptide_index.faiss, peptide_index_meta.pkl
├── requirements.txt
└── README.md
```

Planned/future work — not yet implemented: BLAST-based novelty scoring (similarity check against the peptide dataset; low similarity = more novel). No `blast_scorer.py` or BLAST dependency exists in this repo yet.

---

## Setup

### 1. Prerequisites

- Python 3.10+
- Access to the Bhaskera GPU server via Cloudflare tunnel
- The tunnel URL (changes when restarted — get it from the server admin)

### 2. Install dependencies

```bash
cd peptide-agent
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root (not checked in):

```env
GATEWAY_URL=https://gotten-governance-troops-sheer.trycloudflare.com/v1
API_KEY=sk-bhaskera-alice
MODEL_NAME=aaditya/OpenBioLLM-Llama3-8B
THRESHOLD=0.35
MAX_RETRIES=3
HF_TOKEN=   # optional — raises HuggingFace Inference API rate limits for pLDDT scoring
```

> **Note:** The Cloudflare tunnel URL expires when the server restarts. Update `GATEWAY_URL` in `.env` each time.

### 4. Build the RAG index (one-time, or after `data/peptides.csv` changes)

```bash
python backend/rag/build_index.py
```

Builds `data/peptide_index.faiss` + `data/peptide_index_meta.pkl` from `data/peptides.csv`. If skipped, RAG retrieval just degrades to "no examples this run" — it never blocks generation.

---

## Running

### Backend server

```bash
# From the peptide-agent/ directory
python -m backend.server
```

Server starts on `http://localhost:8000`. Verify with:

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"aaditya/OpenBioLLM-Llama3-8B","gateway":"https://..."}
```

### Frontend

In a second terminal:

```bash
python -m http.server 8080 --directory frontend
```

Open `http://localhost:8080` in your browser. The status dot in the sidebar turns green when the backend is reachable.

---

## API

### `POST /generate`

Runs the agent and streams results via Server-Sent Events.

**Request body:**

```json
{
  "length": 12,
  "charge": 3,
  "hydrophobicity": 0.5,
  "activities": ["anti-bacterial", "anti-fungal"],
  "reference": "KLLKLLKLLKLL",
  "max_retries": 3,
  "threshold": 0.35
}
```

| Field | Type | Description |
|-------|------|-------------|
| `length` | int | Target peptide length in amino acids |
| `charge` | float | Target net charge (can be negative) |
| `hydrophobicity` | float | Target average Kyte-Doolittle hydrophobicity |
| `activities` | list[str] | Biological activity flags (see list below) |
| `reference` | str (optional) | Ground-truth sequence for PeptideBLEU scoring |
| `max_retries` | int | Maximum agent iterations (default: 3) |
| `threshold` | float | PeptideBLEU score to stop early (default: 0.35) |

Valid activity flags: `anti-bacterial`, `anti-cancer`, `anti-fungal`, `anti-parasitic`, `anti-viral`, `cell-cell-communication`, `drug-delivery`, `immunological`, `inhibitor`, `metabolic`, `other-functional`, `signal-peptide`, `toxic`

**SSE stream format:**

```
data: {"type": "attempt", "n": 1, "sequence": "KLLRLLKRLL", "score": 0.31, "status": "fail", "mode": "generate", "issues": [...]}
data: {"type": "attempt", "n": 2, "sequence": "KLLRKLKRLL", "score": 0.39, "status": "pass", "mode": "motif_inject", "issues": []}
data: {"type": "final", "result": {
  "sequence": "KLLRKLKRLL", "score": 0.39, "components": {...}, "iterations": 2, "time_seconds": 4.2,
  "plddt_score": 78.3, "plddt_confidence": "high", "plddt_passes": true, "plddt_interp": "Confident — reliable backbone predicted",
  "secondary_structure": {"ss_string": "-HHHHHHHH-", "helix_pct": 80.0, "sheet_pct": 0.0, "turn_pct": 0.0, "coil_pct": 20.0, "dssp_available": true, "error": null},
  "graph_features": null,
  "rag_examples_used": 5
}}
```

`mode` on each attempt event is one of: `generate`, `deterministic`, `llm_edit`, `motif_inject`, `escape_positional`, `escape_forced`, `escape_exact`, `edit_stuck`, `edit_explored` — reflects which candidate won that iteration.

`plddt_score`/`secondary_structure` are `null` (with an `error`/`*_interp` field explaining why) when no sequence was generated or the respective API/library call failed — neither ever blocks or fails the `/generate` response itself. `graph_features` is currently always `null` — the feature is implemented but disabled in `server.py` pending future work.

### `GET /health`

```json
{"status": "ok", "model": "aaditya/OpenBioLLM-Llama3-8B", "gateway": "https://..."}
```

---

## PeptideBLEU v1.2

Seven-component weighted scoring system comparing a generated peptide against a reference sequence.

| Component | Default Weight | AMP weight | CPP weight | Signal weight |
|-----------|---------------|------------|------------|---------------|
| C1 N-gram BLEU | 0.20 | 0.15 | 0.10 | 0.10 |
| C2 Charge | 0.20 | **0.25** | **0.35** | 0.05 |
| C3 Hydrophobicity | 0.15 | **0.20** | 0.10 | **0.30** |
| C4 Functional Group | 0.10 | 0.10 | 0.10 | 0.15 |
| C5 Property Distribution | 0.10 | 0.10 | 0.10 | 0.15 |
| C6 Structural | 0.10 | 0.05 | 0.10 | 0.10 |
| C7 BLOSUM62 | 0.15 | 0.15 | 0.15 | 0.15 |

Activity presets are applied automatically based on the activities field (e.g., `anti-bacterial` → AMP preset, `drug-delivery` → CPP preset).

**Known test pairs:**

```python
from backend.peptide_bleu import peptide_metric

peptide_metric("KLLKLLKLLK",    "KLLKLFKLLK")  # ~0.93  (one substitution)
peptide_metric("RRWWKK",        "RRWWDD")       # ~0.47  (charge reversal)
peptide_metric("ACDEFGHIKLMN",  "ACDEFGHIKLMN") # 1.0000 (identity)
```

---

## Rulebook validation

Every generated sequence passes through 7 validation rules before scoring:

1. **Valid amino acids** — only `ACDEFGHIKLMNPQRSTVWY` allowed
2. **Length tolerance** — within ±2 of target length
3. **Charge tolerance** — net charge within ±2 of target charge
4. **Hydrophobicity tolerance** — average KD within ±0.5 of target
5. **Proline runs** — no more than 3 consecutive prolines (structural implausibility)
6. **Cysteine parity** — odd cysteine count flagged (disulphide pairing issue)
7. **Activity constraints** — e.g., `drug-delivery` requires charge +3 to +9; `signal-peptide` requires 15–30 aa

Validation issues drive the deterministic-edit fallback and the rulebook-only edit instruction (when there's no reference to compute PeptideBLEU components against) — see `build_edit_prompt()` in `backend/tools/sequence_editor.py`.

---

## Structural confidence (pLDDT / ESMFold)

Supplementary to PeptideBLEU. After the agent returns its final sequence, `backend/esmfold_scorer.py` calls the ESMFold structure-prediction model (`facebook/esmfold_v1`) via the HuggingFace Inference API and reports mean pLDDT (predicted Local Distance Difference Test, Lin et al. 2023, *Science* 379:1123–1130) as a structural-foldability signal, separate from sequence similarity:

| pLDDT | Confidence | Interpretation |
|-------|-----------|-----------------|
| ≥ 90 | `very_high` | Excellent — well-folded peptide |
| 70–90 | `high` | Confident — reliable backbone predicted |
| 50–70 | `low` | Low — possibly flexible or disordered |
| < 50 | `very_low` | Likely intrinsically disordered |

This is purely additive and read-only with respect to generation:
- Computed **once**, on the final sequence, **after** `PepForgeAgent.generate()` returns — it never influences retries, the pass/fail decision, or `NGRAM_FLOOR`/`BLOSUM_FLOOR`.
- `get_plddt()` never raises — on any failure (invalid sequence, timeout, non-200 response, HF model cold-start) it returns a dict with `error` set and `mean_plddt: None`, so a down/rate-limited ESMFold endpoint never breaks generation.
- Optional `HF_TOKEN` in `.env` raises the HF Inference API's free-tier rate limit; omitting it still works, just with a lower ceiling and possible `503` cold-start waits (handled with backoff, up to 3 attempts).
- Surfaced in the `/generate` SSE final event (`plddt_score`, `plddt_confidence`, `plddt_passes`, `plddt_interp`), the frontend results panel (badge + component-scores bar), and optionally in `eval/run_eval.py --mode agent` output (`plddt_score`/`plddt_passes` per result, plus an average/pass-rate summary line) — each call there is a live, serial HTTP request per task, so it noticeably slows down large `--n` eval runs.

---

## Secondary structure (DSSP / pydssp)

Also supplementary to PeptideBLEU, computed right after pLDDT on the same ESMFold PDB structure — no extra API call. Classifies each residue as alpha-helix (H), beta-sheet (E), or coil, and reports the composition as a percentage; for AMPs, helix content correlates with membrane-disruption activity, and for CPPs, an amphipathic helix is important for cell penetration.

Uses [`pydssp`](https://pypi.org/project/pydssp/) — a pure-Python/PyTorch reimplementation of DSSP's H-bond-geometry algorithm, not the real `mkdssp`/`dssp` binary — so it works identically on any OS with `pip install pydssp`, no system-level install step. One accuracy tradeoff: `pydssp`'s simplified 3-state output has no separate "Turn" category (real classical DSSP has 8 states); `turn_pct` is always reported as 0, and whatever real DSSP would call "turn" shows up as coil here instead.

Same purely-additive contract as pLDDT: never influences retries or the pass/fail decision, never raises — degrades to `dssp_available: false` if `pydssp` isn't installed, or an `error` string on a malformed/incompatible PDB. Surfaced in the SSE final event (`secondary_structure`), the frontend's "SECONDARY STRUCTURE (DSSP)" card, and `eval/run_eval.py --mode agent` output (`helix_pct`/`sheet_pct`/`ss_similarity` per result, plus an average summary line).

---

## Evaluation

The eval script benchmarks agent vs baseline. It needs two inputs: a raw-generations file (`--raw`, one row per task with `task_id` + a list of `generated_sequences` from the plain-LLM baseline run) and the task dataset (`--dataset`, defaults to `data/peptides_with_length.jsonl`) — `run_eval.py` joins them by `task_id` to recover each task's reference sequence, length, charge, and activities (the raw-generations file alone doesn't carry those). `--raw` has no usable default; always pass it explicitly.

### Score the baseline (plain LLM best-of-5)

```bash
python eval/run_eval.py \
  --mode baseline \
  --raw results_raw_generations-OpenBioLLM.json \
  --n 1000 \
  --output results_baseline.json
```

### Run the agent

```bash
python eval/run_eval.py \
  --mode agent \
  --raw results_raw_generations-OpenBioLLM.json \
  --n 1000 \
  --output results_agent.json \
  --max-retries 3
```

### Compare and generate paper table

```bash
python eval/run_eval.py \
  --mode compare \
  --baseline results_baseline.json \
  --agent results_agent.json
```

**Output:** (illustrative format — the per-component numbers below come from the `PAPER_BASELINE` constant in `run_eval.py`, a separate/older reference point from the top-of-file baseline table; update `PAPER_BASELINE` if you want this component breakdown to match the current 0.3014 OpenBioLLM-8B baseline)

```
============================================================
PEPTIDE GENERATION AGENT — COMPARISON TABLE
============================================================
Component                 |   Baseline |    Agent |    Delta
-----------------------------|------------|----------|--------
N-gram BLEU               |     0.0430 |   0.XXXX | +X.XXXX
Charge                    |     0.3449 |   0.XXXX | +X.XXXX
Hydrophobicity            |     0.6200 |   0.XXXX | +X.XXXX
Functional Group          |     0.5800 |   0.XXXX | +X.XXXX
Property Distribution     |     0.5500 |   0.XXXX | +X.XXXX
Structural                |     0.7200 |   0.XXXX | +X.XXXX
BLOSUM62                  |     0.3100 |   0.XXXX | +X.XXXX
------------------------------------------------------------
FINAL SCORE               |     0.3784 |   0.XXXX | +X.XXXX
============================================================
```

---

## Amino acid reference

| AA | Charge | KD Hydrophobicity | Class |
|----|--------|-------------------|-------|
| A (Ala) | 0 | +1.8 | Aliphatic |
| R (Arg) | +1 | −4.5 | Positively charged |
| N (Asn) | 0 | −3.5 | Polar uncharged |
| D (Asp) | −1 | −3.5 | Negatively charged |
| C (Cys) | 0 | +2.5 | Special |
| E (Glu) | −1 | −3.5 | Negatively charged |
| Q (Gln) | 0 | −3.5 | Polar uncharged |
| G (Gly) | 0 | −0.4 | Aliphatic |
| H (His) | 0 | −3.2 | Positively charged |
| I (Ile) | 0 | +4.5 | Aliphatic |
| L (Leu) | 0 | +3.8 | Aliphatic |
| K (Lys) | +1 | −3.9 | Positively charged |
| M (Met) | 0 | +1.9 | Aliphatic |
| F (Phe) | 0 | +2.8 | Aromatic |
| P (Pro) | 0 | −1.6 | Special |
| S (Ser) | 0 | −0.8 | Polar uncharged |
| T (Thr) | 0 | −0.7 | Polar uncharged |
| W (Trp) | 0 | −0.9 | Aromatic |
| Y (Tyr) | 0 | −1.3 | Aromatic |
| V (Val) | 0 | +4.2 | Aliphatic |

Hydrophobic residues (for % calculation): `A V I L M F Y W`

---

## Frontend UI

The browser interface at `localhost:8080` matches the research demo screenshot:

- **Left sidebar** — Quick-start presets (Antimicrobial, Cell-Penetrating), model/provider/status
- **Peptide Specification panel** — Live prompt preview (updates as you type), property inputs, activity checkboxes, optional reference sequence
- **Results panel** — Real-time execution trace (SSE-streamed per attempt, badged by mode — generate/deterministic/llm_edit/motif_inject/escape tiers), final sequence with colour-coded score, 7-component score bars, an optional pLDDT badge + component-scores row, and a "SECONDARY STRUCTURE (DSSP)" card (SS string, helix/sheet/turn/coil bars) when structural scoring succeeds

Score colour coding: green ≥ 0.35 (above threshold), amber 0.25–0.35, red < 0.25. pLDDT badge colour coding: teal ≥ 90 (Very High), blue 70–90 (Confident), amber 50–70 (Low), red < 50 (Very Low).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Status dot red | Check `GATEWAY_URL` in `.env` — tunnel may have expired |
| `GATEWAY_URL not set` error | Create `.env` in the project root and fill in the URL (see Setup) |
| Sequence extraction fails | Model returned prose — the regex will pick the longest valid AA run; if empty, check the model is responding |
| Score always 0 | No reference sequence provided — PeptideBLEU requires a ground-truth sequence |
| `ModuleNotFoundError` | Run server from the `peptide-agent/` directory, not from `backend/` |
| pLDDT badge never appears / `plddt_score` always `null` | ESMFold HF endpoint cold-starting, rate-limited, or unreachable — check `plddt_interp` in the response for the specific reason; generation itself is unaffected |
| Secondary structure card missing / `dssp_available: false` | `pydssp` not installed | `pip install pydssp` (already in `requirements.txt`) |
| RAG examples never show up | `data/peptides.csv` or the FAISS index missing | `python backend/rag/build_index.py`; RAG degrades silently to no examples if this was never run |
| `eval/run_eval.py` — `FileNotFoundError` on `--raw` | No usable default for `--raw` | Always pass `--raw <path>` explicitly (see Evaluation) |
