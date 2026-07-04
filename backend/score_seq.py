# score_sequences.py
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   1. Fill in the sequences below for each model
#   2. Run: python score_sequences.py
#   3. Get comparison table
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os

# Add backend folder to path so metric.py and rulebook.py can be found
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from peptide_bleu import peptide_metric, score_components
from rulebook     import validate_sequence

# =============================================================================
# FILL IN YOUR SEQUENCES HERE
# For each task: put the reference sequence and what each model generated
# =============================================================================

TASKS = [
    {
        "task_id":      1,
        # YLGYLE is the dataset ground-truth (6 AA, charge -1) but causes N-gram
        # collapse for all 15-AA outputs. Use the agent's auto-selected AMP
        # reference (Dermaseptin-S1, 20 AA, charge +4) for a fair comparison.
        "ref_sequence": "ALWKTMLKKLGTMALHAGKA",
        "task_constraints": {          # used for rulebook validation
            "length_min": 15, "length_max": 20,
            "charge_min": 2,  "charge_max": 5,
            "hydro_min":  35, "hydro_max":  55,
        },
        "our_agent":    "ALWKIQLQLQILRIRALRKQ",
        "claude":       "KLLFKKILKFLKKIG",
        "chatgpt":      "KRLAVNGGKWLTFQRSI",
        "gemini":       "FLFKKILKGVAKHFRA",
    },
    # ── Add more tasks here ──
    # {
    #     "task_id":      3,
    #     "ref_sequence": "LTVEPWL",
    #     "our_agent":    "XXXXX",
    #     "claude":       "XXXXX",
    #     "chatgpt":      "XXXXX",
    #     "gemini":       "XXXXX",
    # },
]

MODELS = ["our_agent", "claude", "chatgpt", "gemini"]

# =============================================================================
# SCORING — no need to change anything below
# =============================================================================

CHARGE_AA = {'K': 1, 'R': 1, 'D': -1, 'E': -1}
HYDRO_AA  = set('LIVFWMAYC')
VALID_AA  = set('ACDEFGHIKLMNPQRSTVWY')

def is_valid(seq):
    return bool(seq) and all(c in VALID_AA for c in seq.upper())

def actual_charge(seq):
    return sum(CHARGE_AA.get(aa, 0) for aa in seq.upper())

def actual_hydro_pct(seq):
    return round(100 * sum(1 for aa in seq.upper() if aa in HYDRO_AA) / len(seq), 1)

def score(ref, pred):
    if not is_valid(pred) or pred.startswith("["):
        return None
    return peptide_metric(ref.upper(), pred.upper())

def comps(ref, pred):
    if not is_valid(pred) or pred.startswith("["):
        return {}
    return score_components(ref.upper(), pred.upper())

def rulebook(pred, task={}):
    if not is_valid(pred) or pred.startswith("["):
        return {"passed": False, "issues": ["No valid sequence"]}
    result = validate_sequence(pred.upper(), task)
    return {
        "passed": result.get("valid", False),
        "issues": result.get("issues", [])
    }
    
# ── Run scoring ───────────────────────────────────────────────────────────────
all_scores = {m: [] for m in MODELS}
all_passed = {m: [] for m in MODELS}
all_comps  = {m: [] for m in MODELS}

print("\n" + "="*90)
print("PEPTIDE SEQUENCE SCORING — PeptideBLEU + Rulebook")
print("="*90)

COMP_NAMES = [
    "ngram_bleu", "charge", "hydrophobicity",
    "functional_group", "property_distribution", "structural", "blosum"
]
COMP_SHORT = ["N-gram", "Charge", "Hydro", "FuncGrp", "PropDist", "Struct", "BLOSUM62"]

for task in TASKS:
    ref = task["ref_sequence"]
    print(f"\nTask {task['task_id']}  |  Reference: {ref}")
    print("-"*90)
    print(f"{'Model':<12} {'Sequence':<25} {'Score':>7} {'RB':>5} "
          f"{'Charge':>7} {'Hydro%':>7} {'Length':>7}")
    print("-"*90)

    task_constraints = task.get("task_constraints", {})
    for model in MODELS:
        pred = task.get(model, "[ paste here ]")
        s    = score(ref, pred)
        rb   = rulebook(pred, task_constraints)
        c    = comps(ref, pred)

        s_str  = f"{s:.4f}" if s is not None else "  N/A "
        rb_str = "PASS" if rb["passed"] else "FAIL"
        ch_str = str(actual_charge(pred))    if is_valid(pred) and not pred.startswith("[") else "—"
        hy_str = str(actual_hydro_pct(pred)) if is_valid(pred) and not pred.startswith("[") else "—"
        ln_str = str(len(pred))              if is_valid(pred) and not pred.startswith("[") else "—"

        marker = " *" if model == "our_agent" else ""
        print(f"{model+marker:<12} {pred[:24]:<25} {s_str:>7} {rb_str:>5} "
              f"{ch_str:>7} {hy_str:>7} {ln_str:>7}")

        all_scores[model].append(s if s is not None else 0.0)
        all_passed[model].append(1 if rb["passed"] else 0)
        if c:
            all_comps[model].append(c)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print("SUMMARY — Average across all tasks")
print("="*70)
print(f"{'Model':<14} {'Avg Score':>10} {'Pass Rate':>10} {'Tasks':>7}")
print("-"*70)

for model in MODELS:
    scores = all_scores[model]
    passed = all_passed[model]
    if not scores:
        continue
    avg  = sum(scores) / len(scores)
    pct  = 100 * sum(passed) / len(passed)
    n    = len(scores)
    marker = " *" if model == "our_agent" else ""
    print(f"{model+marker:<14} {avg:>10.4f} {pct:>9.1f}% {n:>7}")

print("="*70)

# ── Component breakdown ───────────────────────────────────────────────────────
print("\n\n" + "="*90)
print("COMPONENT BREAKDOWN — Averages per model")
print("="*90)
print(f"{'Component':<22}", end="")
for m in MODELS:
    print(f"{m:>14}", end="")
print()
print("-"*90)

for i, (cname, cshort) in enumerate(zip(COMP_NAMES, COMP_SHORT)):
    print(f"{cshort:<22}", end="")
    for model in MODELS:
        vals = [c.get(cname, 0) for c in all_comps[model] if c]
        avg  = sum(vals) / len(vals) if vals else 0.0
        print(f"{avg:>14.4f}", end="")
    print()

print("="*90)
print("\n* = Our Agent")
