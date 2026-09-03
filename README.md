# PepForge — Agentic Peptide Generation System

Research project at IIIT-Delhi. PepForge is an agentic iterative-refinement system for peptide sequence design: instead of asking a LLM for a peptide once (or best-of-N), it generates once, then repeatedly **validates, scores, and edits** the sequence toward its weakest property, the same generate → validate → score → edit loop a human designer would run by hand. The research claim this whole project exists to test is that this loop beats plain-LLM prompting on the same base model, measured by **PeptideBLEU**, a 7-component reference-similarity metric purpose-built for peptides (n-gram overlap, charge, hydrophobicity, functional groups, property distribution, structural plausibility, BLOSUM62 substitution similarity).


---

## Architecture

```
Task specification (length, charge, hydrophobicity, activities)
        │
        ▼
RAG retrieval — FAISS over data/peptides.csv (105K peptides), top-5
similar examples injected into the prompt (attempt 1 only)
        │
        ▼
Prompt builder — reference anchor (opening/closing motif, charge,
hydro% hints — never the full reference sequence)
        │
        ▼
LLM generation — via an OpenAI-compatible endpoint (Bhaskera GPU
tunnel or any compatible gateway)
        │
        ▼
Sequence extraction — 3-priority parser with a case-origin filter,
rejects English prose masquerading as a sequence
        │
        ▼
PeptideBLEU scoring (needs a reference) + Rulebook validation
(reference-free, always available)
        │
        ▼
Iterative editing — up to max_retries iterations, one edit
candidate per iteration (deterministic fix, LLM edit targeting the
weakest component, or literal reference-motif injection), each
scored and compared before it can replace the working sequence;
escalating escape tactics kick in if the loop gets stuck
        │
        ▼
Best result across all attempts (never a regression from the true
best found so far)
        │
        ▼
Post-hoc, strictly additive — never retried on:
pLDDT (ESMFold) · secondary structure (pydssp) · BLAST novelty
```

**Reasoning models** (e.g. DeepSeek-R1-Distill-Llama-8B) are supported alongside standard instruction-tuned models: when `IS_REASONING_MODEL=true` (or `MODEL_NAME` contains `deepseek-r1`/`r1-distill`/`qwq`), the agent skips the assistant-primer trick used to stop non-reasoning models from prefacing answers with "Explanation:" (a primer would make it structurally impossible for the model to emit a `<think>` block first), and raises the per-call token budget so the model has room to finish reasoning before the sequence.

---

## Key components

### `backend/`

| File | Purpose |
|------|---------|
| `agent.py` | Core agent loop — `PepForgeAgent.generate()`: one fresh generation, then up to `max_retries` scored edit iterations. |
| `models.py` | LLM connection via an OpenAI-compatible endpoint; empty-response retries; reasoning-model detection (`is_reasoning_model()`). |
| `peptide_bleu.py` | PeptideBLEU metric — 7 weighted components, per-activity weight presets. |
| `rulebook.py` | Reference-free physicochemical validation (length, charge, hydrophobicity, proline runs, cysteine parity, activity constraints). |
| `prompt_builder.py` | Builds generation/edit prompts — RAG examples, reference anchor, weakest-component edit instructions. |
| `tools/sequence_editor.py` | Edit candidate generators: deterministic fix, LLM edit, motif injection, and the `edit_stuck` escalation tiers. |
| `rag/peptide_retriever.py`, `rag/build_index.py` | FAISS similarity search over the local peptide dataset; one-time index builder. |
| `trace_logger.py` | Appends a full human-readable trace of every iteration to `logs/generation_trace.log`. |
| `esmfold_scorer.py` | Post-hoc pLDDT structural-confidence scoring via the ESMFold HF Inference API. |
| `dssp_scorer.py` | Post-hoc secondary-structure (helix/sheet/coil %) via `pydssp`. |
| `blast_scorer.py`, `blast_db_builder.py` | Post-hoc novelty scoring against the peptide dataset via local BLAST+; one-time database builder. |
| `graph_features.py` | 3D PDB → 2D interaction graph features — implemented but currently disabled in `server.py` (future work). |
| `score_seq.py` | Ad-hoc manual scoring scratch script (fill in sequences by hand, not part of the main pipeline). |
| `server.py` | FastAPI app — `POST /generate` (SSE stream) and `GET /health`. |

### `eval/`

| File | Purpose |
|------|---------|
| `cluster_dataset.py` | CD-HIT clustering of the peptide dataset (leak-proofing step 1). |
| `split_pools.py` | Splits clusters into working pool A / held-out pool B (step 2). |
| `sample_test_cases.py` | Stratified per-class test-case sampling from pool A (step 3). |
| `run_three_arm_eval.py` | Runs zero-shot / best-of-N / agent arms per task, scored against pool B; checkpointed to JSONL so a killed run resumes (step 4). |
| `compare_three_arms.py` | Prints the paper-ready comparison table across all models found in a results directory (step 5). |
| `run_eval.py` | Older two-arm (baseline vs. agent) batch evaluator, joined against a raw-generations file by `task_id`. |
| `generate_baseline.py`, `compare_all_models.py` | Alternative, simpler per-model-folder baseline generator + cross-model comparison flow. |

### `frontend/`

`index.html` / `app.js` / `style.css` — single-page app: live prompt preview, SSE-streamed execution trace per attempt, final sequence with colour-coded score, 7-component score bars, optional pLDDT/DSSP panels.

---

## PeptideBLEU metric

Seven components, weighted and summed; weights are chosen per-activity via a preset (`ACTIVITY_WEIGHTS` in `peptide_bleu.py`).

| Component | Default | AMP | CPP | Signal | Immunological |
|-----------|---------|-----|-----|--------|---------------|
| N-gram BLEU | 0.20 | 0.15 | 0.10 | 0.10 | 0.15 |
| Charge | 0.20 | 0.25 | **0.35** | 0.05 | 0.20 |
| Hydrophobicity | 0.15 | 0.20 | 0.10 | **0.30** | 0.15 |
| Functional Group | 0.10 | 0.10 | 0.10 | 0.15 | 0.15 |
| Property Distribution | 0.10 | 0.10 | 0.10 | 0.15 | 0.10 |
| Structural | 0.10 | 0.05 | 0.10 | 0.10 | 0.10 |
| BLOSUM62 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 |

`anti-bacterial`/`anti-fungal`/`anti-viral`/`anti-cancer` → AMP preset · `drug-delivery` → CPP preset · `signal-peptide` → Signal preset · `immunological` → Immunological preset · everything else → Default.

```python
from backend.peptide_bleu import peptide_metric

peptide_metric("ACDEFGHIKLMN", "ACDEFGHIKLMN")  # 1.0000 (identity)
peptide_metric("KLLKLLKLLK",   "KLLKLFKLLK")    # ~0.93  (one substitution)
```

---

## Evaluation results

Three-arm evaluation (`eval/run_three_arm_eval.py`): zero-shot (single call) vs. best-of-4 (independent calls, best scored against held-out pool B) vs. agent (full PepForge loop). All four models on identical 130 test cases, scored against pool B — sequences the model never saw.


| Model | Arm | Valid% | Score | RB-pass% | N-gram | Time |
|-------|-----|--------|-------|----------|--------|------|
| OpenBioLLM-70B AWQ | Zero-shot | 100% | 0.5920 | 24.6% | 0.0129 | 2.2s |
| | Best-of-4 | 100% | 0.6554 | 34.6% | 0.0341 | 3.3s |
| | **Agent** | 98.5% | **0.6917** | **51.5%** | 0.1630 | 11.6s |
| BioMistral-7B | Zero-shot | 100% | 0.6280 | 23.8% | 0.0083 | 1.0s |
| | Best-of-4 | 100% | 0.6758 | 30.0% | 0.0478 | 1.6s |
| | **Agent** | 99.2% | **0.6329** | **39.2%** | 0.0567 | 11.8s |
| DeepSeek-R1-Distill-Llama-8B | Zero-shot | 100% | 0.6277 | 41.5% | 0.0048 | 1.9s |
| | Best-of-4 | 100% | 0.6595 | 60.0% | 0.0078 | 2.0s |
| | **Agent** | 89.2% | **0.7131** | 50.0% | 0.2144 | 78.5s |
| Qwen2.5-7B-Instruct | Zero-shot | 100% | 0.6224 | 49.2% | 0.0113 | 1.0s |
| | Best-of-4 | 100% | 0.6753 | 57.7% | 0.0490 | 1.4s |
| | **Agent** | 100% | 0.6695 | **62.3%** | 0.0899 | 4.6s |

**Findings:**
- The agent arm beats *both* baselines outright for **2 of 4 models** — OpenBioLLM-70B AWQ (0.6917 vs. 0.6554/0.5920) and DeepSeek-R1-8B (0.7131 vs. 0.6595/0.6277). 
- For the other two, the agent arm underperforms its own best-of-4 on mean PeptideBLEU: BioMistral-7B drops sharply (0.6329 vs. 0.6758 best-of-4 — barely above its own zero-shot), and Qwen2.5-7B dips slightly (0.6695 vs. 0.6753). These are honest negative results, not cherry-picked — the agent loop does not uniformly help across every base model, and BioMistral-7B in particular looks like biomedical-QA fine-tuning not transferring well to iterative sequence editing.
- DeepSeek-R1-8B has the highest agent-arm score of all four (0.7131) and edges out the ~9x-larger OpenBioLLM-70B (0.6917) — suggestive that reasoning-before-answering matters more than raw parameter count here, though this is one run on one metric, not a controlled ablation.
- Qwen2.5-7B has the best rulebook-pass rate (62.3%) and perfect valid-sequence extraction (100%) of all four on the agent arm, despite its middling raw score.
- DeepSeek-R1-8B's agent arm is drastically slower per task (78.5s avg) than the other three (1.4s–11.8s), consistent with a reasoning model spending most of its budget on the `<think>` block before ever reaching the sequence.


---

## Setup and installation

### Prerequisites

- Python 3.10+
- An OpenAI-compatible LLM gateway (e.g. the Bhaskera GPU server via Cloudflare tunnel)
- Optional, only if you need these specific features:
  - `pydssp` (secondary structure) — installed via `requirements.txt`
  - NCBI BLAST+ (`makeblastdb`, `blastp`) on `PATH` — for novelty scoring (`blast_scorer.py`)
  - `cd-hit` on `PATH` (or inside WSL on Windows) — only needed for the leak-proof eval data-prep step (`eval/cluster_dataset.py`)

### Install

```bash
cd peptide-agent
python -m venv venv
# Windows: venv\Scripts\activate   |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### Configure environment

Create a `.env` file in the project root (not checked in):

```env
GATEWAY_URL=https://xxxx.trycloudflare.com/v1
API_KEY=sk-bhaskera-alice
MODEL_NAME=aaditya/OpenBioLLM-Llama3-8B
IS_REASONING_MODEL=false
```

> The Cloudflare tunnel URL changes every time the GPU server restarts — update `GATEWAY_URL` each time, and restart `python -m backend.server` afterward (it only reads `.env` at process start).

---

## Environment variables

Only variables actually read by the code — see [Setup](#setup-and-installation) above for a working `.env`.

| Variable | Read by | Description | Example |
|----------|---------|-------------|---------|
| `GATEWAY_URL` | `backend/models.py` | OpenAI-compatible generation endpoint | `https://xxxx.trycloudflare.com/v1` |
| `API_KEY` | `backend/models.py` | API key for the gateway above | `sk-bhaskera-alice` |
| `MODEL_NAME` | `backend/models.py` | Model ID sent in each request | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| `IS_REASONING_MODEL` | `backend/models.py`, `backend/agent.py` | Forces reasoning-model handling (skips the assistant primer, raises the token budget). Auto-detected from `MODEL_NAME` if unset — only needed to force it for a model whose name doesn't match `deepseek-r1`/`r1-distill`/`qwq` | `true` |
| `PEPTIDE_DATASET_PATH` | `backend/rag/peptide_retriever.py`, `backend/blast_scorer.py` | Overrides the default `data/peptides.csv` path | `data/peptides.csv` |
| `BLAST_DB_PATH` | `backend/blast_scorer.py` | Overrides the default BLAST database path | `data/blast_db/known_peptides_db` |

`threshold` and `max_retries` are **not** `.env` variables — they're per-request fields on `POST /generate` (defaults `0.35` / `6`, see [API](#api) below).

---

## One-time setup

Build the FAISS RAG index (needed once, or whenever `data/peptides.csv` changes — degrades silently to "no RAG examples" if skipped):

```bash
python -m backend.rag.build_index
```

Build the local BLAST database (only needed for novelty scoring; requires `makeblastdb` on `PATH`):

```bash
python backend/blast_db_builder.py
```

---

## Running the agent

```bash
# Terminal 1 — from peptide-agent/, not backend/
python -m backend.server
```

```bash
# Terminal 2
python -m http.server 8080 --directory frontend
```

Open `http://localhost:8080`. Verify the backend directly with:

```bash
curl http://localhost:8000/health
```

## API

### `POST /generate`

Streams results via Server-Sent Events. Key request fields: `length`/`length_min`/`length_max`, `charge`/`charge_min`/`charge_max`, `hydro_min`/`hydro_max`, `activities` (list), `reference` (optional — auto-selected per activity if omitted), `max_retries` (default 6), `threshold` (default 0.35).

```
data: {"type": "attempt", "n": 1, "sequence": "...", "score": 0.31, "status": "fail", "mode": "generate", ...}
data: {"type": "final", "result": {"sequence": "...", "score": 0.39, "components": {...}, "iterations": 2, ...}}
```

### `GET /health`

```json
{"status": "ok", "model": "...", "gateway": "https://..."}
```

---

## Running evaluation

The current, leak-proof pipeline (five stages, in order):

```bash
# 1. Cluster the dataset (needs cd-hit on PATH)
python eval/cluster_dataset.py --dataset data/peptides.csv --output eval/clusters --identity 0.9

# 2. Split clusters into working pool A / held-out pool B
python eval/split_pools.py --clusters eval/clusters --output eval/pools --seed 42 --split 0.7

# 3. Sample stratified test cases from pool A
python eval/sample_test_cases.py --pool eval/pools/pool_a_working.json --output eval/test_cases.json --n-per-class 10 --seed 42

# 4. Run the three-arm evaluation for a model (set GATEWAY_URL/MODEL_NAME in .env first)
python eval/run_three_arm_eval.py \
    --test-cases eval/test_cases.json \
    --pool-b eval/pools/pool_b_heldout.json \
    --output eval/results \
    --n-bestofn 4 \
    --model-name ModelName

# 5. Print the comparison table across every model in eval/results/
python eval/compare_three_arms.py --results eval/results
```

`run_three_arm_eval.py` checkpoints every `(task_id, arm)` result to `eval/results/checkpoint_<model>.jsonl` as it goes, so a killed/interrupted run resumes from where it left off on the next invocation with the same `--model-name`.

An older, simpler two-arm flow (`eval/run_eval.py`, baseline vs. agent, joined against a raw-generations file by `task_id`) is still present — see comments at the top of that file for its `--raw`/`--mode` usage.

---

## Amino acid reference

| AA | Charge | KD Hydrophobicity | Class |
|----|--------|--------------------|-------|
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

## Adding a new model

No repo-side config files are needed — model selection is entirely `.env`-driven:

1. Point `GATEWAY_URL` at an OpenAI-compatible endpoint serving the model.
2. Set `MODEL_NAME` to the model's ID as the gateway expects it.
3. If it's a reasoning model that emits `<think>...</think>` before its answer, set `IS_REASONING_MODEL=true` (or make sure `MODEL_NAME` contains `deepseek-r1`, `r1-distill`, or `qwq` — that auto-detects it).
4. Restart `python -m backend.server` (it only reads `.env` at process start).
5. To add it to the evaluation table, run `eval/run_three_arm_eval.py --model-name <Name>` and then `eval/compare_three_arms.py`.

---

## File structure

```
peptide-agent/
├── backend/
│   ├── agent.py                  # Core agent loop (PepForgeAgent)
│   ├── models.py                 # LLM connection + reasoning-model detection
│   ├── peptide_bleu.py           # PeptideBLEU metric
│   ├── rulebook.py               # Reference-free validation
│   ├── prompt_builder.py         # Prompt construction
│   ├── trace_logger.py           # logs/generation_trace.log writer
│   ├── esmfold_scorer.py         # pLDDT (post-hoc)
│   ├── dssp_scorer.py            # Secondary structure (post-hoc)
│   ├── blast_scorer.py           # Novelty scoring (post-hoc)
│   ├── blast_db_builder.py       # One-time BLAST DB builder
│   ├── graph_features.py         # PDB → graph features (implemented, disabled)
│   ├── score_seq.py              # Manual ad-hoc scoring scratch script
│   ├── server.py                 # FastAPI + SSE
│   ├── tools/
│   │   └── sequence_editor.py    # Edit candidates + escape tiers
│   └── rag/
│       ├── peptide_retriever.py  # FAISS retrieval
│       └── build_index.py        # One-time index builder
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── eval/
│   ├── cluster_dataset.py        # Leak-proofing step 1 (CD-HIT)
│   ├── split_pools.py            # Leak-proofing step 2 (pool A/B)
│   ├── sample_test_cases.py      # Leak-proofing step 3 (stratified sampling)
│   ├── run_three_arm_eval.py     # Zero-shot / best-of-N / agent evaluation
│   ├── compare_three_arms.py     # Cross-model comparison table
│   ├── run_eval.py               # Older two-arm baseline/agent evaluator
│   ├── generate_baseline.py      # Alternative per-model baseline generator
│   ├── compare_all_models.py     # Alternative per-model-folder comparison
│   └── results/                  # Gitignored — checkpoint_*.jsonl, results_*.json
├── data/                         # Gitignored — peptides.csv, FAISS index, BLAST DB
├── requirements.txt
└── README.md
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Status dot red / `GATEWAY_URL not set` | Check/create `.env`, restart the server after any edit |
| Sequence extraction fails for a reasoning model | Check `MODEL_NAME`/`IS_REASONING_MODEL` — without reasoning-mode enabled, the token budget is too small for a `<think>` block to complete |
| Score always 0 | No reference sequence resolved for this activity — PeptideBLEU needs one |
| `ModuleNotFoundError` | Run the server from `peptide-agent/`, not from `backend/` |
| RAG examples never show up | `data/peptides.csv` or the FAISS index missing — run `python -m backend.rag.build_index` |
| Novelty/BLAST fields always null | `makeblastdb`/`blastp` not on `PATH`, or `python backend/blast_db_builder.py` was never run |
| `eval/cluster_dataset.py` fails to find cd-hit | Install `cd-hit` (`apt-get install cd-hit` / `conda install -c bioconda cd-hit`), or run it inside WSL on Windows |

---

## Acknowledgements

IIIT-Delhi · Dr. Bapi Chatterjee · Tanya Sheemar
