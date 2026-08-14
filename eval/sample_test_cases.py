"""
Sample a fixed number of test cases per class from pool A.
These test cases are used for all three evaluation arms.

Usage:
  python eval/sample_test_cases.py \
      --pool eval/pools/pool_a_working.json \
      --dataset data/peptides_with_length.jsonl \
      --output eval/test_cases.json \
      --n-per-class 10 \
      --seed 42
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path


def _default_dataset_path() -> str:
    # Mirrors run_eval.py's fallback: an eval/ copy (if kept in sync) takes
    # priority, otherwise the canonical data/ location.
    eval_path = Path(__file__).parent / "peptides_with_length.jsonl"
    if eval_path.exists():
        return str(eval_path)
    return str(Path(__file__).parent.parent / "data" / "peptides_with_length.jsonl")


ACTIVITY_COLS = [
    "anti-bacterial", "anti-cancer", "anti-fungal", "anti-parasitic",
    "anti-viral", "cell-cell-communication", "drug-delivery",
    "immunological", "inhibitor", "metabolic", "other-functional",
    "signal-peptide", "toxic",
]


def _load_dataset(path: str) -> dict[str, dict]:
    """Load peptides_with_length.jsonl keyed by sequence."""
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                seq = row.get('sequence', '')
                if seq:
                    data[seq] = row
    return data


def sample_test_cases(
    pool_path: str,
    dataset_path: str,
    output_path: str,
    n_per_class: int = 10,
    seed: int = 42,
) -> None:
    """
    Sample n_per_class test cases from pool A.
    Matches pool sequences to peptides_with_length.jsonl for task specs.
    """
    random.seed(seed)

    with open(pool_path) as f:
        pool_a = json.load(f)

    print(f"[SAMPLE] Loading dataset {dataset_path}...")
    dataset = _load_dataset(dataset_path)
    print(f"[SAMPLE] Dataset: {len(dataset)} peptides with task specs")

    test_cases = []
    stats: dict[str, int] = {}

    for cls in ACTIVITY_COLS:
        class_seqs = pool_a.get(cls, [])
        if not class_seqs:
            print(f"  {cls:<25}: no sequences in pool A — skipping")
            stats[cls] = 0
            continue

        n = min(n_per_class, len(class_seqs))
        sampled = random.sample(class_seqs, n)

        matched = []
        for item in sampled:
            seq = item['sequence']
            task_row = dataset.get(seq)
            if task_row:
                matched.append({
                    'seq_id': item['seq_id'],
                    'class': cls,
                    'cluster': item.get('cluster', ''),
                    'task_id': task_row.get('task_id', ''),
                    'sequence': seq,
                    'length': task_row.get('length', len(seq)),
                    'ref_net_charge': task_row.get('ref_net_charge', 0),
                    'ref_hydrophobic_pct': task_row.get('ref_hydrophobic_pct', 0),
                    'prompt': task_row.get('prompt', ''),
                    'activities': [cls],
                })
            else:
                matched.append({
                    'seq_id': item['seq_id'],
                    'class': cls,
                    'sequence': seq,
                    'length': len(seq),
                    'activities': [cls],
                })

        test_cases.extend(matched)
        stats[cls] = len(matched)
        print(f"  {cls:<25}: {len(matched)} test cases (from {len(class_seqs)} pool A seqs)")

    output = {
        'seed': seed,
        'n_per_class': n_per_class,
        'total': len(test_cases),
        'stats': stats,
        'test_cases': test_cases,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[SAMPLE] Total test cases: {len(test_cases)}")
    print(f"[SAMPLE] Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample test cases from pool A")
    parser.add_argument('--pool', default='eval/pools/pool_a_working.json')
    parser.add_argument('--dataset', default=_default_dataset_path())
    parser.add_argument('--output', default='eval/test_cases.json')
    parser.add_argument('--n-per-class', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    sample_test_cases(args.pool, args.dataset, args.output, args.n_per_class, args.seed)


if __name__ == '__main__':
    main()
