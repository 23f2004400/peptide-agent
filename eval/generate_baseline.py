"""
Generate raw sequences from a plain LLM (no agent) for baseline evaluation.
Produces results_raw_generations.json in the same format as the existing
per-model result files (task_id + generated_sequences).

Usage:
  # Set .env with correct GATEWAY_URL and MODEL_NAME first, then:

  python eval/generate_baseline.py \
      --model-folder eval/BioMistral-7B \
      --n 100 --k 5

  # Saves to eval/BioMistral-7B/results_raw_generations.json
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from backend import models


def _default_dataset_path() -> Path:
    # Mirrors run_eval.py's fallback: an eval/ copy (if kept in sync) takes
    # priority, otherwise the canonical data/ location.
    eval_path = Path(__file__).parent / "peptides_with_length.jsonl"
    if eval_path.exists():
        return eval_path
    return Path(__file__).parent.parent / "data" / "peptides_with_length.jsonl"


DATASET_PATH = _default_dataset_path()


def _generate_k_sequences(prompt: str, k: int = 5) -> list[str]:
    """
    Generate k sequences from the plain LLM — no agent, no RAG, no editing,
    no assistant primer (deliberately unlike agent.py's calls, since this is
    meant to represent an un-augmented baseline). Reuses backend.models.generate()
    rather than a fresh OpenAI client so this inherits the same empty-response
    retry handling (EMPTY_RETRY_ATTEMPTS in models.py) already proven
    necessary for this deployment's chat_template bug — a bare client.chat.
    completions.create() call would silently undercount valid generations
    whenever the gateway returns 0 tokens.
    """
    full_prompt = (
        f"{prompt}\n\n"
        f"Generate one valid peptide sequence meeting the above properties.\n"
        f"Output: one line, uppercase letters only, nothing else.\n"
        f"Sequence:"
    )

    sequences = []
    for attempt in range(k):
        try:
            content = models.generate(full_prompt, max_tokens=64, temperature=0.8)
            sequences.append(content.strip())
        except Exception as e:
            print(f"  [WARN] attempt {attempt+1} failed: {e}")
            sequences.append("")
        time.sleep(0.3)  # be nice to the server

    return sequences


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline sequences from plain LLM"
    )
    parser.add_argument(
        '--model-folder', required=True,
        help='Path to model subfolder e.g. eval/BioMistral-7B'
    )
    parser.add_argument(
        '--n', type=int, default=100,
        help='Number of tasks to evaluate (default: 100)'
    )
    parser.add_argument(
        '--k', type=int, default=5,
        help='Sequences per task — best-of-k (default: 5)'
    )
    parser.add_argument(
        '--dataset', default=str(DATASET_PATH),
        help='Path to peptides_with_length.jsonl'
    )
    args = parser.parse_args()

    folder = Path(args.model_folder)
    folder.mkdir(parents=True, exist_ok=True)
    output_path = folder / "results_raw_generations.json"

    tasks = []
    with open(args.dataset) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    tasks = tasks[:args.n]

    model_name = os.environ.get("MODEL_NAME", "unknown")
    gateway    = os.environ.get("GATEWAY_URL", "not set")

    print(f"\n{'='*60}")
    print(f"BASELINE GENERATION")
    print(f"{'='*60}")
    print(f"Model:    {model_name}")
    print(f"Gateway:  {gateway}")
    print(f"Tasks:    {len(tasks)}")
    print(f"k:        {args.k} sequences per task")
    print(f"Output:   {output_path}")
    print(f"{'='*60}\n")

    results = []
    for i, task in enumerate(tasks):
        prompt = task.get('prompt', '')
        if not prompt:
            continue

        seqs = _generate_k_sequences(prompt, k=args.k)
        results.append({
            "task_id":             task.get('task_id', i),
            "generated_sequences": seqs,
        })

        if (i + 1) % 10 == 0:
            valid = sum(1 for r in results
                        for s in r['generated_sequences']
                        if s and len(s) >= 4)
            total = sum(len(r['generated_sequences']) for r in results)
            print(f"  {i+1:4d}/{len(tasks)} done  "
                  f"(valid sequences: {valid}/{total})")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Saved {len(results)} task results to {output_path}")


if __name__ == '__main__':
    main()
