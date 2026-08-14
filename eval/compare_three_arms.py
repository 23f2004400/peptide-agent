"""
Compare results across all models and three arms.
Generates the final paper-ready comparison table.

Usage:
  python eval/compare_three_arms.py --results eval/results/
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

COMPONENT_KEYS = [
    'ngram_bleu', 'charge', 'hydrophobicity', 'functional_group',
    'property_distribution', 'structural', 'blosum',
]

COMPONENT_SHORT = {
    'ngram_bleu': 'N-gram',
    'charge': 'Charge',
    'hydrophobicity': 'Hydro',
    'functional_group': 'FuncGrp',
    'property_distribution': 'PropDist',
    'structural': 'Struct',
    'blosum': 'BLOSUM62',
}


def print_comparison_table(results_dir: str) -> None:
    result_files = list(Path(results_dir).glob('results_*.json'))

    if not result_files:
        print(f"No result files found in {results_dir}")
        return

    print(f"\n{'='*120}")
    print("THREE-ARM COMPARISON TABLE")
    print("Scored against held-out pool B")
    print(f"{'='*120}")

    comp_header = "  ".join(f"{COMPONENT_SHORT[k]:>8}" for k in COMPONENT_KEYS)
    print(f"{'Model':<20} {'Arm':<12} {'Valid%':>7} {comp_header}  {'Score':>8} {'RB%':>6}")
    print(f"{'-'*120}")

    for result_file in sorted(result_files):
        with open(result_file) as f:
            data = json.load(f)

        model = data.get('model', result_file.stem)

        for arm in ['zero_shot', 'best_of_n', 'agent']:
            summary = data.get('summaries', {}).get(arm)
            if not summary:
                continue

            valid_pct = summary.get('valid_pct', 0)
            score = summary.get('mean_score', 0)
            rb_pct = summary.get('rb_pass_pct', 0)
            comps = summary.get('component_means', {})

            comp_vals = "  ".join(f"{comps.get(k, 0):>8.4f}" for k in COMPONENT_KEYS)
            marker = " *" if arm == 'agent' else "  "
            print(f"{model:<20} {arm+marker:<12} {valid_pct:>6.1f}%  "
                  f"{comp_vals}  {score:>8.4f} {rb_pct:>5.1f}%")

        print(f"{'-'*120}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='eval/results')
    args = parser.parse_args()
    print_comparison_table(args.results)


if __name__ == '__main__':
    main()
