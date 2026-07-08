"""
Ad-hoc test: run the agent against a supervisor-specified generation task,
applying a custom iteration/stop rule that differs from PeptideAgent's own:

    1. Generate a candidate sequence.
    2. Compute its PeptideBLEU score.
    3. Track the best score seen so far.
    4. If two consecutive iterations fail to improve the best score by
       at least 0.02, terminate and return the best sequence.
    5. Cap total iterations at 10 regardless.

PeptideAgent.generate() has its own early-stop logic (threshold + N-gram/
BLOSUM floor), so this script runs the agent with an unreachable threshold
to force it through all 10 iterations, then replays the supervisor's exact
stop rule as post-processing over the per-iteration trace. No changes to
backend/agent.py or backend/prompt_builder.py.

Usage (from project root):
    python eval/protocol_test.py
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from backend.agent import PeptideAgent

TASK = {
    'length': 25,
    'length_min': 20, 'length_max': 30,
    'charge': 3,
    'charge_min': 2, 'charge_max': 5,
    'hydro_min': 40, 'hydro_max': 49,
    'activities': ['anti-bacterial'],
    'reference': '',
    'max_retries': 10,
    'threshold': 1.1,  
}

MIN_IMPROVEMENT = 0.02
MAX_STALL = 2


def apply_stop_rule(trace: list[dict]) -> tuple[str, int, float]:
    best_score, best_seq, best_iter = -1.0, '', 0
    stall = 0
    for entry in trace:
        s = entry.get('score')
        if s is None:
            continue
        if s > best_score + MIN_IMPROVEMENT:
            best_score, best_seq, best_iter = s, entry['sequence'], entry['n']
            stall = 0
        else:
            stall += 1
        if stall >= MAX_STALL:
            break
    return best_seq, best_iter, best_score


def main():
    agent = PeptideAgent(threshold=TASK['threshold'], max_retries=TASK['max_retries'])
    result = agent.generate(TASK)

    print("Per-iteration trace:")
    for entry in result.trace:
        score_str = f"{entry['score']:.4f}" if entry.get('score') is not None else "N/A"
        print(f"  [{entry['n']}] seq={entry['sequence']!r:30s} score={score_str}")

    best_seq, best_iter, best_score = apply_stop_rule(result.trace)

    print()
    print(f"{best_seq}\titeration={best_iter}\tpeptide_bleu={best_score:.4f}")


if __name__ == '__main__':
    main()
