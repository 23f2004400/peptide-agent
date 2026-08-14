"""
CD-HIT clustering of peptide dataset per activity class.
Clusters peptides at 90% sequence identity to remove near-duplicates.
Saves cluster representatives and assignments for pool splitting.

Usage:
  python eval/cluster_dataset.py \
      --dataset data/peptides.csv \
      --output eval/clusters/ \
      --identity 0.9

Requirements:
  sudo apt-get install cd-hit  (or conda install -c bioconda cd-hit)

"""

from __future__ import annotations
import argparse
import os
import subprocess
import shutil
import tempfile
import csv
import json
from pathlib import Path
from collections import defaultdict

ACTIVITY_COLS = [
    "anti-bacterial", "anti-cancer", "anti-fungal", "anti-parasitic",
    "anti-viral", "cell-cell-communication", "drug-delivery",
    "immunological", "inhibitor", "metabolic", "other-functional",
    "signal-peptide", "toxic",
]

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

CDHIT_TIMEOUT_SECONDS = 600

# Sentinel returned by _check_cdhit() when cd-hit isn't on the native PATH
# but is available inside WSL — run_cdhit() checks for this exact value to
# decide whether to bridge through `wsl -e`.
_WSL_SENTINEL = 'wsl'


def _check_cdhit() -> str | None:
    """
    Look for a native cd-hit/cdhit binary on PATH first; if neither is
    found, check whether WSL has one (common on Windows dev machines where
    cd-hit was installed via apt inside WSL rather than natively). Returns
    a native path, the _WSL_SENTINEL, or None if cd-hit isn't available
    anywhere.
    """
    for name in ('cd-hit', 'cdhit'):
        path = shutil.which(name)
        if path:
            return path

    if shutil.which('wsl'):
        try:
            result = subprocess.run(
                ['wsl', '-e', 'which', 'cd-hit'],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                return _WSL_SENTINEL
        except Exception:
            pass

    return None


def _win_to_wsl_path(path: str) -> str:
    """Convert a native Windows path (as produced by tempfile on this OS)
    into the /mnt/<drive>/... form WSL expects, e.g.
    'C:\\Users\\x\\tmp\\a.fasta' -> '/mnt/c/Users/x/tmp/a.fasta'."""
    p = Path(path).resolve()
    drive = p.drive.rstrip(':').lower()
    rest = str(p)[len(p.drive):].replace('\\', '/').lstrip('/')
    return f"/mnt/{drive}/{rest}"


def _write_fasta(sequences: list[tuple], fasta_path: str) -> None:
    """sequences = [(id, seq), ...]"""
    with open(fasta_path, 'w') as f:
        for seq_id, seq in sequences:
            f.write(f">{seq_id}\n{seq}\n")


def _parse_cdhit_clusters(clstr_path: str) -> dict[str, str]:
    """Parse CD-HIT .clstr file. Returns {seq_id -> cluster_representative_id}."""
    assignments: dict[str, str] = {}
    current_rep: str | None = None

    with open(clstr_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>Cluster'):
                current_rep = None
                continue
            if '>' not in line:
                continue
            seq_id = line.split('>')[1].split('...')[0].strip()
            is_rep = line.endswith('*')
            if is_rep:
                current_rep = seq_id
            if current_rep:
                assignments[seq_id] = current_rep

    return assignments


def run_cdhit(
    sequences: list[tuple],
    identity: float = 0.9,
    cdhit_bin: str = 'cd-hit',
) -> dict[str, str]:
    """Run CD-HIT on sequences. Returns {seq_id -> cluster_rep_id}."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_in = os.path.join(tmpdir, 'input.fasta')
        fasta_out = os.path.join(tmpdir, 'output')

        _write_fasta(sequences, fasta_in)

        if identity >= 0.9:
            word_size = 5
        elif identity >= 0.8:
            word_size = 4
        else:
            word_size = 3

        if cdhit_bin == _WSL_SENTINEL:
            # tmpdir is a native Windows path (tempfile always creates one on
            # this OS) — WSL can't resolve it directly, so translate to the
            # /mnt/<drive>/... form it expects.
            cdhit_args = [
                'wsl', '-e', 'cd-hit',
                '-i', _win_to_wsl_path(fasta_in),
                '-o', _win_to_wsl_path(fasta_out),
                '-c', str(identity), '-n', str(word_size),
                '-M', '16000', '-T', '4', '-d', '0',
            ]
        else:
            cdhit_args = [
                cdhit_bin,
                '-i', fasta_in,
                '-o', fasta_out,
                '-c', str(identity),
                '-n', str(word_size),
                '-M', '16000',
                '-T', '4',
                '-d', '0',
            ]

        try:
            result = subprocess.run(
                cdhit_args, capture_output=True, text=True,
                timeout=CDHIT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"CD-HIT timed out after {CDHIT_TIMEOUT_SECONDS}s "
                f"({'via WSL bridge' if cdhit_bin == _WSL_SENTINEL else 'native'})"
            )
        if result.returncode != 0:
            raise RuntimeError(f"CD-HIT failed: {result.stderr[:500]}")

        assignments = _parse_cdhit_clusters(fasta_out + '.clstr')

    return assignments


def cluster_dataset(dataset_path: str, output_dir: str, identity: float = 0.9) -> None:
    """
    Cluster peptide dataset per activity class.
    Saves cluster assignments to <output_dir>/<class_name>_clusters.json
    """
    cdhit_bin = _check_cdhit()
    if not cdhit_bin:
        print("[CLUSTER] CD-HIT not found.")
        print("Install: sudo apt-get install cd-hit")
        print("         or: conda install -c bioconda cd-hit")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[CLUSTER] Loading dataset from {dataset_path}...")
    peptides_by_class: dict[str, list[tuple]] = defaultdict(list)

    with open(dataset_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            seq = row.get('sequence', '').strip().upper()
            if not seq or not all(c in VALID_AA for c in seq):
                continue
            for col in ACTIVITY_COLS:
                if str(row.get(col, '0')).strip() == '1':
                    peptides_by_class[col].append((f"{col}_{i}", seq))

    print("[CLUSTER] Found peptides per class:")
    for cls, peps in peptides_by_class.items():
        print(f"  {cls:<25}: {len(peps)} peptides")

    all_results = {}
    for cls, sequences in peptides_by_class.items():
        print(f"\n[CLUSTER] Clustering {cls} ({len(sequences)} seqs)...")
        try:
            assignments = run_cdhit(sequences, identity, cdhit_bin)
            clusters: dict[str, list[str]] = defaultdict(list)
            for seq_id, rep_id in assignments.items():
                clusters[rep_id].append(seq_id)

            class_result = {
                'class': cls,
                'n_sequences': len(sequences),
                'n_clusters': len(clusters),
                'identity': identity,
                'clusters': dict(clusters),
                'seq_to_cluster': assignments,
                'sequences': {seq_id: seq for seq_id, seq in sequences},
            }
            all_results[cls] = class_result

            out_path = os.path.join(output_dir, f"{cls}_clusters.json")
            with open(out_path, 'w') as f:
                json.dump(class_result, f, indent=2)

            print(f"  -> {len(clusters)} clusters from {len(sequences)} sequences")
            print(f"  -> Saved to {out_path}")

        except Exception as e:
            print(f"  [WARN] Clustering failed for {cls}: {e}")

    summary = {
        cls: {'n_sequences': v['n_sequences'], 'n_clusters': v['n_clusters']}
        for cls, v in all_results.items()
    }
    summary_path = os.path.join(output_dir, 'cluster_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[CLUSTER] Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="CD-HIT clustering per activity class")
    parser.add_argument('--dataset', default='data/peptides.csv')
    parser.add_argument('--output', default='eval/clusters')
    parser.add_argument('--identity', type=float, default=0.9,
                         help='Sequence identity threshold (default: 0.9)')
    args = parser.parse_args()
    cluster_dataset(args.dataset, args.output, args.identity)


if __name__ == '__main__':
    main()
