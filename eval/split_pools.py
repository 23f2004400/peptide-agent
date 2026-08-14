"""
Split clustered peptides into Working pool A (70%) and Held-out pool B (30%).
Split is done at CLUSTER level to prevent sequence leakage.
Uses a fixed random seed for reproducibility.

Usage:
  python eval/split_pools.py \
      --clusters eval/clusters/ \
      --output eval/pools/ \
      --seed 42 \
      --split 0.7
"""

from __future__ import annotations
import argparse
import json
import random
import os
from pathlib import Path


def split_pools(
    clusters_dir: str,
    output_dir: str,
    seed: int = 42,
    train_ratio: float = 0.7,
) -> None:
    """
    Split clusters into pool A (working) and pool B (held-out).
    Split at cluster level — all members of a cluster go to the same pool.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    pool_a: dict[str, list] = {}
    pool_b: dict[str, list] = {}

    cluster_files = list(Path(clusters_dir).glob('*_clusters.json'))
    if not cluster_files:
        print(f"[SPLIT] No cluster files found in {clusters_dir}")
        print("Run eval/cluster_dataset.py first.")
        return

    print(f"[SPLIT] Processing {len(cluster_files)} class cluster files...")
    print(f"[SPLIT] Seed={seed}, Train ratio={train_ratio}")

    for cluster_file in sorted(cluster_files):
        with open(cluster_file) as f:
            data = json.load(f)

        cls = data['class']
        clusters = data['clusters']          # {rep -> [member_ids]}
        seq_map = data['sequences']           # {seq_id -> sequence}
        cluster_reps = list(clusters.keys())

        random.shuffle(cluster_reps)

        n_train = max(1, int(len(cluster_reps) * train_ratio))
        train_reps = cluster_reps[:n_train]
        test_reps = cluster_reps[n_train:]

        pool_a_seqs = []
        for rep in train_reps:
            for member_id in clusters[rep]:
                if member_id in seq_map:
                    pool_a_seqs.append({
                        'seq_id': member_id,
                        'sequence': seq_map[member_id],
                        'class': cls,
                        'cluster': rep,
                    })

        pool_b_seqs = []
        for rep in test_reps:
            for member_id in clusters[rep]:
                if member_id in seq_map:
                    pool_b_seqs.append({
                        'seq_id': member_id,
                        'sequence': seq_map[member_id],
                        'class': cls,
                        'cluster': rep,
                    })

        pool_a[cls] = pool_a_seqs
        pool_b[cls] = pool_b_seqs

        print(
            f"  {cls:<25}: "
            f"A={len(pool_a_seqs)} seqs ({len(train_reps)} clusters)  "
            f"B={len(pool_b_seqs)} seqs ({len(test_reps)} clusters)"
        )

    pool_a_path = os.path.join(output_dir, 'pool_a_working.json')
    pool_b_path = os.path.join(output_dir, 'pool_b_heldout.json')

    with open(pool_a_path, 'w') as f:
        json.dump(pool_a, f, indent=2)
    with open(pool_b_path, 'w') as f:
        json.dump(pool_b, f, indent=2)

    total_a = sum(len(v) for v in pool_a.values())
    total_b = sum(len(v) for v in pool_b.values())
    denom = max(total_a + total_b, 1)

    print(f"\n[SPLIT] Pool A (working):   {total_a} sequences")
    print(f"[SPLIT] Pool B (held-out):  {total_b} sequences")
    print(f"[SPLIT] Actual split ratio: {total_a/denom:.1%} / {total_b/denom:.1%}")
    print(f"[SPLIT] Saved: {pool_a_path}")
    print(f"[SPLIT] Saved: {pool_b_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split clusters into working pool A and held-out pool B"
    )
    parser.add_argument('--clusters', default='eval/clusters')
    parser.add_argument('--output', default='eval/pools')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', type=float, default=0.7,
                         help='Fraction for working pool A (default: 0.7)')
    args = parser.parse_args()
    split_pools(args.clusters, args.output, args.seed, args.split)


if __name__ == '__main__':
    main()
