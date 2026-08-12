"""
Batch evaluation: compare plain LLM baseline vs PepForgeAgent.

Usage:
  python run_eval.py --mode baseline --n 1000 --output results_baseline.json
  python run_eval.py --mode agent    --n 1000 --output results_agent.json
  python run_eval.py --mode compare  --baseline results_baseline.json --agent results_agent.json
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from backend.peptide_bleu import peptide_metric, score_components, batch_peptide_metric
from backend.rulebook import validate_sequence

COMPONENT_KEYS = [
    'ngram_bleu', 'charge', 'hydrophobicity', 'functional_group',
    'property_distribution', 'structural', 'blosum',
]

COMPONENT_LABELS = {
    'ngram_bleu': 'N-gram BLEU',
    'charge': 'Charge',
    'hydrophobicity': 'Hydrophobicity',
    'functional_group': 'Functional Group',
    'property_distribution': 'Property Distribution',
    'structural': 'Structural',
    'blosum': 'BLOSUM62',
}

# Baseline scores from LLM-Evaluation_4.md for comparison
PAPER_BASELINE = {
    'ngram_bleu': 0.0430,
    'charge': 0.3449,
    'hydrophobicity': 0.6200,
    'functional_group': 0.5800,
    'property_distribution': 0.5500,
    'structural': 0.7200,
    'blosum': 0.3100,
    'final': 0.3784,
}


def _load_raw_generations(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _load_dataset(path: str) -> dict[str, dict]:
    """Load peptides_with_length.jsonl keyed by task_id or sequence."""
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get('task_id') or row.get('id') or row.get('sequence', '')[:20]
            data[key] = row
    return data


def _default_dataset_path() -> str:
    # eval/peptides_with_length.jsonl (new location, kept in sync with
    # data/peptides_with_length.jsonl) takes priority; falls back to the
    # original data/ location if the eval/ copy isn't present.
    eval_path = Path(__file__).parent / "peptides_with_length.jsonl"
    if eval_path.exists():
        return str(eval_path)
    return str(Path(__file__).parent.parent / "data" / "peptides_with_length.jsonl")

_ACTIVITY_PHRASES = [
    ('anti-bacterial', 'anti-bacterial'),
    ('anti-cancer', 'anti-cancer'),
    ('anti-fungal', 'anti-fungal'),
    ('anti-parasitic', 'anti-parasitic'),
    ('anti-viral', 'anti-viral'),
    ('cell-cell communication', 'cell-cell-communication'),
    ('drug delivery', 'drug-delivery'),
    ('immunological', 'immunological'),
    ('inhibitor', 'inhibitor'),
    ('metabolic', 'metabolic'),
    ('bioactive', 'other-functional'),
    ('signal peptide', 'signal-peptide'),
]


def _parse_activities_from_prompt(prompt: str) -> list[str]:
    """
    Recover the positive activity labels from a peptides_with_length.jsonl
    `prompt` string (e.g. "...not anti-bacterial, ... an inhibitor, ...").
    Parsed clause-by-clause (split on ", ") rather than by a fixed
    look-behind window: a fixed window over the raw string can pick up a
    "not " that actually negates the *previous* clause (e.g. in "not
    anti-bacterial, anti-cancer", "anti-cancer" is a positive clause of its
    own, but a naive N-character look-behind from its start still reaches
    back into the preceding "not anti-bacterial, " and misreads it as
    negated) — scoping the negation check to each clause avoids that.
    """
    if not prompt:
        return []
    body = prompt.split(':', 1)[-1]
    body = body.split('. The peptide length', 1)[0]
    activities: list[str] = []
    for clause in body.split(','):
        low = clause.strip().lower()
        negated = low.startswith('not ')
        if negated:
            low = low[4:]
        if negated:
            continue
        for needle, label in _ACTIVITY_PHRASES:
            if needle in low:
                activities.append(label)
    if re.search(r'(?<!non-)\btoxic\b', prompt.lower()):
        activities.append('toxic')
    return list(dict.fromkeys(activities))


def _merge_task_record(raw_rec: dict, dataset: dict[str, dict]) -> dict | None:
    """
    Join a raw-generations record (task_id + generated_sequences only) with
    its matching peptides_with_length.jsonl row to recover reference/length/
    charge/hydrophobicity/activities. Returns None if no matching row exists.
    """
    ds_row = dataset.get(raw_rec.get('task_id'))
    if not ds_row:
        return None
    merged = dict(raw_rec)
    merged['reference'] = ds_row.get('sequence', '')
    merged['length'] = ds_row.get('length')
    merged['ref_net_charge'] = ds_row.get('ref_net_charge', 0)
    merged['ref_hydrophobic_pct'] = ds_row.get('ref_hydrophobic_pct', 0.0)
    merged['activities'] = _parse_activities_from_prompt(ds_row.get('prompt', ''))
    return merged


def _best_of_n_score(sequences: list[str], reference: str, activity: str | None = None) -> tuple[float, str]:
    """Return (best_score, best_seq) from a list of candidate sequences."""
    best_score = 0.0
    best_seq = sequences[0] if sequences else ''
    for seq in sequences:
        s = peptide_metric(reference, seq, activity=activity)
        if s > best_score:
            best_score = s
            best_seq = seq
    return best_score, best_seq


def run_baseline(
    raw_gen_path: str,
    dataset_path: str | None,
    n: int,
    output_path: str,
) -> dict:
    """Score the existing generated sequences using best-of-5 PeptideBLEU."""
    dataset_path = dataset_path or _default_dataset_path()
    print(f"\n[BASELINE] Loading {raw_gen_path} ...")
    raw_records = _load_raw_generations(raw_gen_path)[:n]
    print(f"[BASELINE] Loading dataset {dataset_path} ...")
    dataset = _load_dataset(dataset_path)
    records = [m for m in (_merge_task_record(r, dataset) for r in raw_records) if m]
    print(f"[BASELINE] Evaluating {len(records)} tasks "
          f"({len(raw_records) - len(records)} skipped — no matching dataset row) ...")

    all_scores = []
    comp_sums = {k: 0.0 for k in COMPONENT_KEYS}
    results = []

    for i, rec in enumerate(records):
        seqs = rec.get('generated_sequences', [])
        ref = rec.get('reference', '') or ''
        if not ref or not seqs:
            continue

        activity = rec.get('activity') or (rec.get('activities') or [None])[0]
        score, best_seq = _best_of_n_score(seqs, ref, activity)
        comps = score_components(ref, best_seq)

        all_scores.append(score)
        for k in COMPONENT_KEYS:
            comp_sums[k] += comps.get(k, 0.0)

        results.append({
            'task_id': rec.get('task_id', i),
            'reference': ref,
            'best_sequence': best_seq,
            'score': score,
            'components': comps,
            'n_candidates': len(seqs),
        })

        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(records)} done, running avg={mean(all_scores):.4f}")

    n_valid = max(len(all_scores), 1)
    summary = {
        'mode': 'baseline',
        'n_evaluated': n_valid,
        'mean_score': round(mean(all_scores), 4),
        'component_means': {k: round(v / n_valid, 4) for k, v in comp_sums.items()},
        'results': results,
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary)
    print(f"\n[BASELINE] Written to {output_path}")
    return summary


def run_agent(
    raw_gen_path: str,
    n: int,
    output_path: str,
    max_retries: int = 3,
    threshold: float = 0.35,
    dataset_path: str | None = None,
) -> dict:
    """Run the PepForgeAgent on a sample of tasks."""
    from backend.agent import PepForgeAgent
    from backend.esmfold_scorer import get_plddt
    from backend.dssp_scorer import get_secondary_structure
    from backend.blast_scorer import assess_novelty

    dataset_path = dataset_path or _default_dataset_path()
    print(f"\n[AGENT] Loading tasks from {raw_gen_path} ...")
    raw_records = _load_raw_generations(raw_gen_path)[:n]
    print(f"[AGENT] Loading dataset {dataset_path} ...")
    dataset = _load_dataset(dataset_path)
    records = [m for m in (_merge_task_record(r, dataset) for r in raw_records) if m]
    print(f"[AGENT] Running agent on {len(records)} tasks (max_retries={max_retries}) "
          f"({len(raw_records) - len(records)} skipped — no matching dataset row) ...")

    agent = PepForgeAgent(threshold=threshold, max_retries=max_retries)
    all_scores = []
    comp_sums = {k: 0.0 for k in COMPONENT_KEYS}
    results = []

    for i, rec in enumerate(records):
        ref = rec.get('reference', '') or ''
        if not ref:
            continue

        task = {
            'length': rec.get('length', 15),
            'ref_net_charge': rec.get('ref_net_charge', 0),
            'ref_hydrophobic_pct': rec.get('ref_hydrophobic_pct', 0.0),
            'activities': rec.get('activities', []),
            'reference': ref,
            'max_retries': max_retries,
            'threshold': threshold,
        }

        t0 = time.time()
        try:
            result = agent.generate(task)
        except Exception as exc:
            print(f"  [WARN] Task {i} failed: {exc}")
            continue

        score = result.score or 0.0
        comps = result.components or {}

        all_scores.append(score)
        for k in COMPONENT_KEYS:
            comp_sums[k] += comps.get(k, 0.0)

        # pLDDT scoring — additive only, computed once on the final sequence,
        # never affects generation/retry logic. Never raises.
        plddt = get_plddt(result.sequence) if result.sequence else {}

        # DSSP secondary structure — runs on the same PDB pLDDT already
        # fetched above (no new API call). Never raises; degrades to
        # dssp_available: False if the mkdssp/dssp binary isn't installed.
        dssp = get_secondary_structure(
            pdb_string=plddt.get('pdb'),
            reference_pdb=None,
            activities=task.get('activities', []),
        ) if plddt.get('pdb') else {}

        # BLAST novelty assessment — additive only, runs once on the final
        # best sequence. Never raises; degrades to blast_available: False
        # if the blastp/makeblastdb binaries aren't installed.
        blast = assess_novelty(result.sequence) if result.sequence else {}

        results.append({
            'task_id': rec.get('task_id', i),
            'reference': ref,
            'best_sequence': result.sequence,
            'score': score,
            'components': comps,
            'iterations': result.iterations,
            'time_seconds': result.time_seconds,
            'trace': result.trace,
            'plddt_score': plddt.get('mean_plddt'),
            'plddt_passes': plddt.get('passes', False),
            'helix_pct': dssp.get('helix_pct'),
            'sheet_pct': dssp.get('sheet_pct'),
            'ss_similarity': (dssp.get('ss_similarity') or {}).get('overall_ss_similarity'),
            'blast_similarity': blast.get('similarity_score'),
            'novelty_label': blast.get('novelty_label'),
        })

        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(records)} done, running avg={mean(all_scores):.4f}")

    n_valid = max(len(all_scores), 1)
    summary = {
        'mode': 'agent',
        'n_evaluated': n_valid,
        'mean_score': round(mean(all_scores), 4),
        'component_means': {k: round(v / n_valid, 4) for k, v in comp_sums.items()},
        'results': results,
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary)

    plddt_col = [r.get('plddt_score') for r in results if r.get('plddt_score') is not None]
    if plddt_col:
        passes = sum(1 for r in results if r.get('plddt_passes'))
        print(f"  Avg pLDDT:           {sum(plddt_col)/len(plddt_col):.1f}")
        print(f"  pLDDT pass (>=70):   {passes}/{len(results)}")

    ss_data = [r for r in results if r.get('helix_pct') is not None]
    if ss_data:
        avg_helix = sum(r['helix_pct'] for r in ss_data) / len(ss_data)
        print(f"  Avg Helix %:         {avg_helix:.1f}%")
        sim_data = [r['ss_similarity'] for r in ss_data if r.get('ss_similarity') is not None]
        if sim_data:
            print(f"  Avg SS Similarity:   {sum(sim_data)/len(sim_data):.3f}")

    blast_data = [r for r in results if r.get('blast_similarity') is not None]
    if blast_data:
        avg_sim = sum(r['blast_similarity'] for r in blast_data) / len(blast_data)
        novel_count = sum(1 for r in blast_data
                           if r.get('novelty_label') in ('novel', 'low_similarity'))
        print(f"  Avg BLAST Similarity: {avg_sim:.4f}")
        print(f"  Novel peptides:       {novel_count}/{len(blast_data)} "
              f"({100*novel_count/len(blast_data):.1f}%)")

    print(f"\n[AGENT] Written to {output_path}")
    return summary


def compare(baseline_path: str, agent_path: str):
    """Print a paper-ready comparison table."""
    with open(baseline_path) as f:
        base = json.load(f)
    with open(agent_path) as f:
        agent = json.load(f)

    base_comps = base.get('component_means', PAPER_BASELINE)
    agent_comps = agent.get('component_means', {})
    base_final = base.get('mean_score', PAPER_BASELINE['final'])
    agent_final = agent.get('mean_score', 0.0)

    print("\n" + "=" * 60)
    print("PEPTIDE GENERATION AGENT — COMPARISON TABLE")
    print("=" * 60)
    header = f"{'Component':<25} | {'Baseline':>10} | {'Agent':>8} | {'Delta':>8}"
    print(header)
    print("-" * 60)

    for k in COMPONENT_KEYS:
        label = COMPONENT_LABELS[k]
        b_val = base_comps.get(k, PAPER_BASELINE.get(k, 0.0))
        a_val = agent_comps.get(k, 0.0)
        delta = a_val - b_val
        sign = '+' if delta >= 0 else ''
        print(f"{label:<25} | {b_val:>10.4f} | {a_val:>8.4f} | {sign}{delta:>7.4f}")

    print("-" * 60)
    delta_final = agent_final - base_final
    sign = '+' if delta_final >= 0 else ''
    print(f"{'FINAL SCORE':<25} | {base_final:>10.4f} | {agent_final:>8.4f} | {sign}{delta_final:>7.4f}")
    print("=" * 60)

    improvement_pct = (delta_final / max(base_final, 1e-9)) * 100
    print(f"\nAbsolute improvement : {delta_final:+.4f}")
    print(f"Relative improvement : {improvement_pct:+.1f}%")
    print(f"Agent beats baseline : {'YES ✓' if delta_final > 0 else 'NO ✗'}")
    print()


def _print_summary(summary: dict):
    print("\n  Average Score:", summary['mean_score'])
    print("  --- Component Averages ---")
    for k, v in summary['component_means'].items():
        print(f"  {COMPONENT_LABELS.get(k, k):25s}: {v:.4f}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="PeptideBLEU batch evaluation")
    p.add_argument('--mode', choices=['baseline', 'agent', 'compare'], required=True)
    p.add_argument('--raw', default='results_raw_generations.json',
                   help='Path to results_raw_generations.json')
    p.add_argument('--dataset', default=None, help='Path to peptides_with_length.jsonl')
    p.add_argument('--n', type=int, default=1000, help='Number of tasks to evaluate')
    p.add_argument('--output', default=None, help='Output JSON path')
    p.add_argument('--baseline', default='results_baseline.json', help='Baseline results JSON')
    p.add_argument('--agent', default='results_agent.json', help='Agent results JSON')
    p.add_argument('--max-retries', type=int, default=3)
    p.add_argument('--threshold', type=float, default=0.35)
    args = p.parse_args()

    if args.mode == 'baseline':
        # Default output alongside --raw's own folder (e.g.
        # eval/BioMistral-7B/results_raw_generations.json ->
        # eval/BioMistral-7B/results_baseline.json) rather than the cwd.
        out = args.output or str(Path(args.raw).parent / 'results_baseline.json')
        run_baseline(args.raw, args.dataset, args.n, out)
    elif args.mode == 'agent':
        out = args.output or str(Path(args.raw).parent / 'results_agent.json')
        run_agent(args.raw, args.n, out, args.max_retries, args.threshold, args.dataset)
    elif args.mode == 'compare':
        compare(args.baseline, args.agent)


if __name__ == '__main__':
    main()
