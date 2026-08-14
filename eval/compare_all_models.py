"""
Compare all 6 models (baseline vs agent) in one table.
Reads result JSON files from each model subfolder.

Usage:
  python eval/compare_all_models.py

Expects these files to exist:
  eval/OpenBioLLM-8B/results_baseline.json
  eval/OpenBioLLM-8B/results_agent.json
  eval/BioMistral-7B/results_baseline.json
  eval/BioMistral-7B/results_agent.json
  ... etc
"""

from __future__ import annotations
import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent

# Matched pairs as per Dr. Bapi's model selection
MODEL_PAIRS = [
    {
        "biomedical": "OpenBioLLM-8B",
        "general":    "Llama-3.1-8B",
        "pair_label": "Pair 1 (Llama-3 base)",
    },
    {
        "biomedical": "BioMistral-7B",
        "general":    "Mistral-7B",
        "pair_label": "Pair 2 (Mistral base)",
    },
    {
        "biomedical": "Qwen2.5-7B",
        "general":    "DeepSeek-R1-8B",
        "pair_label": "Pair 3 (Reasoning comparison)",
    },
]

ALL_MODELS = [
    "OpenBioLLM-8B",
    "Llama-3.1-8B",
    "BioMistral-7B",
    "Mistral-7B",
    "Qwen2.5-7B",
    "DeepSeek-R1-8B",
]

COMPONENT_KEYS = [
    'ngram_bleu', 'charge', 'hydrophobicity',
    'functional_group', 'property_distribution',
    'structural', 'blosum',
]

COMPONENT_SHORT = {
    'ngram_bleu':            'N-gram',
    'charge':                'Charge',
    'hydrophobicity':        'Hydro',
    'functional_group':      'FuncGrp',
    'property_distribution': 'PropDist',
    'structural':            'Struct',
    'blosum':                'BLOSUM62',
}


def _load(model: str, mode: str) -> dict | None:
    path = EVAL_DIR / model / f"results_{mode}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _valid_pct(results: list) -> float:
    """Fraction of tasks with score > 0 (extraction succeeded)."""
    if not results:
        return 0.0
    valid = sum(1 for r in results if (r.get('score') or 0) > 0)
    return round(100 * valid / len(results), 1)


def print_main_table():
    """Print full comparison table — all models, baseline vs agent."""

    print("\n" + "=" * 110)
    print("PEPTIDE GENERATION AGENT — FULL MODEL COMPARISON")
    print("Does biomedical tuning help? + Does the agent help?")
    print("=" * 110)

    comp_headers = "  ".join(f"{COMPONENT_SHORT[k]:>8}" for k in COMPONENT_KEYS)
    print(f"{'Model':<22} {'Mode':<10} {'Valid%':>7} "
          f"{comp_headers}  {'PeptideBLEU':>12}")
    print("-" * 110)

    for model in ALL_MODELS:
        for mode in ['baseline', 'agent']:
            data = _load(model, mode)
            if data is None:
                print(f"{model:<22} {mode:<10}  {'—':>6}  "
                      + "  ".join(f"{'—':>8}" for _ in COMPONENT_KEYS)
                      + f"  {'—':>12}")
                continue

            results = data.get('results', [])
            valid_pct = _valid_pct(results)
            comps     = data.get('component_means', {})
            score     = data.get('mean_score', 0.0)

            comp_vals = "  ".join(
                f"{comps.get(k, 0.0):>8.4f}" for k in COMPONENT_KEYS
            )
            marker = " *" if mode == 'agent' else "  "
            print(f"{model:<22} {mode+marker:<10} {valid_pct:>6.1f}%  "
                  f"{comp_vals}  {score:>12.4f}")

        print("-" * 110)


def print_matched_pairs():
    """Print matched pair analysis — does biomedical tuning help?"""

    print("\n" + "=" * 80)
    print("MATCHED PAIR ANALYSIS — Does biomedical tuning help?")
    print("=" * 80)

    for pair in MODEL_PAIRS:
        bio_model = pair["biomedical"]
        gen_model = pair["general"]
        label     = pair["pair_label"]

        print(f"\n{label}")
        print(f"  Biomedical: {bio_model}")
        print(f"  General:    {gen_model}")
        print()

        for mode in ['baseline', 'agent']:
            bio_data = _load(bio_model, mode)
            gen_data = _load(gen_model, mode)

            bio_score = bio_data.get('mean_score', 0.0) if bio_data else None
            gen_score = gen_data.get('mean_score', 0.0) if gen_data else None

            if bio_score is not None and gen_score is not None:
                delta  = bio_score - gen_score
                winner = bio_model if delta > 0 else gen_model
                print(f"  {mode:<10}: {bio_model} {bio_score:.4f} vs "
                      f"{gen_model} {gen_score:.4f}  "
                      f"delta={delta:+.4f}  winner={winner}")
            else:
                print(f"  {mode:<10}: data missing")

    print()


def print_agent_improvement():
    """Print how much the agent improves each model."""

    print("\n" + "=" * 60)
    print("AGENT IMPROVEMENT — plain LLM -> + Agent")
    print("=" * 60)
    print(f"{'Model':<22} {'Baseline':>10} {'Agent':>8} {'Delta':>8} {'%Gain':>8}")
    print("-" * 60)

    for model in ALL_MODELS:
        base_data  = _load(model, 'baseline')
        agent_data = _load(model, 'agent')

        base_score  = base_data.get('mean_score',  0.0) if base_data  else None
        agent_score = agent_data.get('mean_score', 0.0) if agent_data else None

        if base_score is not None and agent_score is not None:
            delta = agent_score - base_score
            pct   = (delta / max(base_score, 1e-9)) * 100
            sign  = '+' if delta >= 0 else ''
            print(f"{model:<22} {base_score:>10.4f} {agent_score:>8.4f} "
                  f"{sign}{delta:>7.4f} {sign}{pct:>6.1f}%")
        else:
            print(f"{model:<22} {'—':>10} {'—':>8} {'—':>8} {'—':>8}")

    print("=" * 60)


def main():
    print_main_table()
    print_matched_pairs()
    print_agent_improvement()


if __name__ == '__main__':
    main()
