"""
Three-arm evaluation pipeline.
Arm 1: Zero-shot  — plain LLM, single call
Arm 2: Best-of-N  — plain LLM, N calls, keep best (N = avg agent iterations)
Arm 3: Agent      — full PepForge pipeline

All arms scored against held-out pool B.

Usage:
  python eval/run_three_arm_eval.py \
      --test-cases eval/test_cases.json \
      --pool-b eval/pools/pool_b_heldout.json \
      --output eval/results/ \
      --n-bestofn 4 \
      --model-name OpenBioLLM-8B
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
# override=True: the .env file on disk must always win over whatever's
# already in this shell's environment. Without it, a GATEWAY_URL set once
# in a terminal session (e.g. an earlier ad-hoc `$env:GATEWAY_URL = ...`)
# silently outlives every future edit to .env — every new process launched
# from that same terminal keeps reusing the stale value no matter how many
# times .env is updated, which is exactly what caused a run to keep hitting
# an already-dead Cloudflare tunnel after the URL had already been rotated.
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from backend import models
from backend.peptide_bleu import peptide_metric, score_components
from backend.rulebook import validate_sequence, rulebook_score

COMPONENT_KEYS = [
    'ngram_bleu', 'charge', 'hydrophobicity', 'functional_group',
    'property_distribution', 'structural', 'blosum',
]

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# == SCORING AGAINST POOL B ===================================================

def score_against_pool_b(
    generated_seq: str,
    pool_b_class: list[dict],
    activity: str | None = None,
) -> tuple[float, str]:
    """
    Score generated sequence against every sequence in pool B for this class.
    Returns (best_score, best_matching_reference) — best-match scoring, since
    pool B has no single "correct" reference per generated sequence.
    """
    if not generated_seq or not pool_b_class:
        return 0.0, ''

    best_score = 0.0
    best_ref = ''

    for item in pool_b_class:
        ref = item.get('sequence', '')
        if not ref:
            continue
        try:
            score = peptide_metric(ref, generated_seq, activity=activity)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_ref = ref

    return round(best_score, 4), best_ref


# == SEQUENCE EXTRACTION ======================================================

def _extract_sequence(raw: str, expected_length: int | None = None) -> str:
    """Extract an AA sequence from a raw model response — best-effort, no
    dependency on backend.agent.extract_sequence() (kept standalone since the
    zero-shot/best-of-N arms deliberately don't use any agent machinery)."""
    if not raw:
        return ''
    raw = raw.strip().upper()
    for line in raw.split('\n'):
        cleaned = ''.join(c for c in line.strip() if c in VALID_AA)
        if len(cleaned) >= 4:
            if expected_length and len(cleaned) >= expected_length:
                return cleaned[:expected_length]
            if not expected_length and 4 <= len(cleaned) <= 60:
                return cleaned
    import re
    candidates = re.findall(r'[ACDEFGHIKLMNPQRSTVWY]{4,}', raw)
    if not candidates:
        return ''
    if expected_length:
        candidates.sort(key=lambda c: abs(len(c) - expected_length))
    return candidates[0]


def _build_prompt(task: dict) -> str:
    """Build a plain, un-augmented generation prompt from a task spec — no
    RAG, no reference anchor, no assistant primer (see agent.py for those;
    deliberately excluded here since arms 1/2 represent the un-augmented
    baseline this whole eval exists to beat)."""
    prompt = task.get('prompt', '')
    if prompt:
        return (
            f"{prompt}\n\n"
            f"Generate one valid peptide sequence.\n"
            f"Output: one line, uppercase letters only.\n"
            f"Sequence:"
        )
    cls = task.get('class', 'bioactive')
    length = task.get('length', 15)
    charge = task.get('ref_net_charge', 0)
    hydro = task.get('ref_hydrophobic_pct', 40)
    return (
        f"Generate one {cls} peptide of exactly {length} amino acids.\n"
        f"Net charge: {charge:+d}\n"
        f"Hydrophobic fraction: ~{hydro:.0f}%\n"
        f"Output: one line, uppercase letters only.\n"
        f"Sequence:"
    )


# == TIMEOUT GUARD =============================================================
def _call_with_timeout(fn, args=(), timeout=90):
    """Run fn(*args) on a daemon thread; return (value, timed_out).
    On timeout, the thread is abandoned (not killed — Python can't do that)
    but doesn't block the caller or process exit."""
    box = {'value': None, 'error': None, 'done': False}

    def _run():
        try:
            box['value'] = fn(*args)
        except Exception as e:
            box['error'] = e
        finally:
            box['done'] = True

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if not box['done']:
        return None, True
    if box['error'] is not None:
        raise box['error']
    return box['value'], False


# == THREE ARMS ================================================================
# Arms 1/2 call backend.models.generate() directly (not a raw OpenAI client)
# so they inherit its EMPTY_RETRY_ATTEMPTS handling for this deployment's
# chat_template bug — see CLAUDE.md and eval/generate_baseline.py, which
# uses the same pattern for the same reason. No assistant_primer is passed,
# since that's an agent-specific mitigation these baseline arms must not use.

def _one_zero_shot_call(prompt: str, task: dict) -> tuple[str, str]:
    raw = models.generate(prompt, max_tokens=64, temperature=0.8)
    return raw, _extract_sequence(raw, task.get('length'))


def run_zero_shot(task: dict, call_timeout: float = 90) -> dict:
    """Arm 1: Single LLM call, no loop."""
    prompt = _build_prompt(task)
    t0 = time.time()
    timed_out = False
    try:
        result, timed_out = _call_with_timeout(
            _one_zero_shot_call, args=(prompt, task), timeout=call_timeout,
        )
        raw, seq = result if result else ('', '')
    except Exception as e:
        raw, seq = str(e), ''

    return {
        'arm': 'zero_shot',
        'sequence': seq,
        'raw': raw,
        'time_s': round(time.time() - t0, 2),
        'n_calls': 1,
        'timed_out': timed_out,
    }


def _one_best_of_n_call(prompt: str, task: dict) -> str:
    raw = models.generate(prompt, max_tokens=64, temperature=0.9)
    return _extract_sequence(raw, task.get('length'))


def run_best_of_n(task: dict, n: int = 4, call_timeout: float = 90) -> dict:
    """Arm 2: N independent LLM calls; the caller scores all candidates
    against pool B and keeps the best (scoring needs pool B, which this
    function doesn't have — see the per-candidate scoring loop in
    run_evaluation() below).

    The N calls fire concurrently rather than sequentially — they're
    independent (different temperature-sampled completions of the same
    prompt, no shared state), and vLLM's continuous batching means a single
    replica handles several in-flight requests far more efficiently than
    one full generation at a time followed by the next. See CLAUDE.md /
    this session's notes: the Bhaskera server here runs num_replicas=1, so
    request-level parallelism (not more replicas) is the available lever.

    Each call is individually timeout-guarded (daemon threads, joined with
    call_timeout) — worst case (every call hangs) this takes up to
    n * call_timeout, not forever; typically all n finish together well
    within one call_timeout window since they run concurrently.
    """
    prompt = _build_prompt(task)
    t0 = time.time()
    results = [''] * n
    threads: list[threading.Thread] = []

    def _worker(idx: int) -> None:
        try:
            results[idx] = _one_best_of_n_call(prompt, task)
        except Exception:
            results[idx] = ''

    for i in range(n):
        th = threading.Thread(target=_worker, args=(i,), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join(call_timeout)

    n_timed_out = sum(1 for th in threads if th.is_alive())
    sequences = [seq for seq in results if seq]

    return {
        'arm': 'best_of_n',
        'all_seqs': sequences,
        'n_calls': n,
        'n_timed_out': n_timed_out,
        'time_s': round(time.time() - t0, 2),
    }


def _one_agent_call(agent, agent_task: dict):
    return agent.generate(agent_task)


def run_agent_arm(task: dict, max_retries: int = 6, call_timeout: float = 300) -> dict:
    """Arm 3: Full PepForge agent pipeline. `reference` is deliberately left
    blank — the agent must not see the held-out answer, only auto-pick a
    reference the same way it would for a real user request (see
    agent.py's _pick_reference()).

    call_timeout is generous (5 min default) relative to the other two arms
    since this is a whole multi-iteration loop (up to max_retries LLM
    calls), not a single request."""
    from backend.agent import PepForgeAgent

    agent = PepForgeAgent(threshold=0.35, max_retries=max_retries)
    t0 = time.time()

    agent_task = {
        'length': task.get('length', 15),
        'ref_net_charge': task.get('ref_net_charge', 0),
        'ref_hydrophobic_pct': task.get('ref_hydrophobic_pct', 40),
        'activities': task.get('activities', []),
        'reference': '',
        'max_retries': max_retries,
        'threshold': 0.35,
    }

    seq, iterations, timed_out = '', 0, False
    try:
        result, timed_out = _call_with_timeout(
            _one_agent_call, args=(agent, agent_task), timeout=call_timeout,
        )
        if result is not None:
            seq = result.sequence or ''
            iterations = result.iterations or 1
    except Exception:
        seq, iterations = '', 0

    return {
        'arm': 'agent',
        'sequence': seq,
        'iterations': iterations,
        'time_s': round(time.time() - t0, 2),
        'timed_out': timed_out,
    }


# == CHECKPOINTING =============================================================
# A hung/killed run previously lost all progress, since results were only
# ever written once at the very end. Every (task, arm) result is now
# appended to a JSONL checkpoint file as soon as it's computed, and a run
# restarted with the same --model-name skips whatever's already there
# instead of recomputing (and re-billing gateway calls for) it.

def _checkpoint_path(output_dir: str, model_name: str) -> str:
    return os.path.join(output_dir, f"checkpoint_{model_name}.jsonl")


def _load_checkpoint(path: str, arms: list[str]) -> tuple[dict[str, list[dict]], set[tuple]]:
    arm_results: dict[str, list[dict]] = {arm: [] for arm in arms}
    done: set[tuple] = set()
    if not os.path.exists(path):
        return arm_results, done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a truncated last line from a killed process
            arm = rec.get('arm')
            if arm in arm_results:
                arm_results[arm].append(rec)
                done.add((rec.get('task_id'), arm))
    return arm_results, done


def _write_summary(output_dir: str, model_name: str, n_bestofn: int, n_tasks: int,
                    arms: list[str], arm_results: dict[str, list[dict]]) -> dict:
    """(Re)build the results_<model>.json summary from whatever's in
    arm_results so far — called after every task, not just at the end, so a
    killed run always leaves a valid, up-to-date summary on disk."""
    summaries = {}
    for arm in arms:
        results = arm_results[arm]
        scores = [r['score'] for r in results if r['score'] > 0]
        valid_pct = 100 * len(scores) / max(len(results), 1)
        rb_pass = 100 * sum(1 for r in results if r['rb_score'] >= 0.8) / max(len(results), 1)
        avg_time = mean(r['time_s'] for r in results) if results else 0
        avg_score = mean(scores) if scores else 0.0

        comp_means = {}
        for k in COMPONENT_KEYS:
            vals = [r['components'].get(k, 0) for r in results if r['components']]
            comp_means[k] = round(mean(vals), 4) if vals else 0.0

        summaries[arm] = {
            'mean_score': round(avg_score, 4),
            'valid_pct': round(valid_pct, 1),
            'rb_pass_pct': round(rb_pass, 1),
            'avg_time_s': round(avg_time, 2),
            'component_means': comp_means,
            'results': results,
        }

    output = {
        'model': model_name,
        'n_bestofn': n_bestofn,
        'n_tasks': n_tasks,
        'n_tasks_completed': max((len(arm_results[a]) for a in arms), default=0),
        'summaries': summaries,
    }
    out_path = os.path.join(output_dir, f"results_{model_name}.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    return output


# == MAIN EVALUATION ============================================================

def run_evaluation(
    test_cases_path: str,
    pool_b_path: str,
    output_dir: str,
    n_bestofn: int = 4,
    model_name: str = 'model',
    arms: list[str] | None = None,
    call_timeout: float = 90,
    agent_timeout: float = 300,
) -> None:
    """Run all three arms on test cases, score against pool B."""
    if arms is None:
        arms = ['zero_shot', 'best_of_n', 'agent']

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(test_cases_path) as f:
        test_data = json.load(f)
    test_cases = test_data['test_cases']

    with open(pool_b_path) as f:
        pool_b = json.load(f)

    ckpt_path = _checkpoint_path(output_dir, model_name)
    arm_results, done = _load_checkpoint(ckpt_path, arms)
    all_iterations = [
        r['n_calls'] for r in arm_results.get('agent', [])
        if r.get('n_calls')
    ]

    print(f"\n{'='*70}")
    print("THREE-ARM EVALUATION")
    print(f"{'='*70}")
    print(f"Model:       {os.environ.get('MODEL_NAME', 'unknown')}")
    print(f"Test cases:  {len(test_cases)}")
    print(f"Arms:        {', '.join(arms)}")
    print(f"Best-of-N:   N={n_bestofn}")
    print(f"Call timeout: {call_timeout}s (agent arm: {agent_timeout}s)")
    if done:
        n_resumed = len(done) // max(len(arms), 1)
        print(f"Resuming from checkpoint: {ckpt_path} (~{n_resumed} tasks already done)")
    print(f"{'='*70}\n")

    with open(ckpt_path, 'a') as ckpt_f:
        for i, task in enumerate(test_cases):
            cls = task.get('class', 'unknown')
            pool_b_c = pool_b.get(cls, [])
            activity = (task.get('activities') or [cls])[0]
            task_id = task.get('task_id', i)

            if all((task_id, arm) in done for arm in arms):
                continue  # every requested arm already checkpointed for this task

            print(f"Task {i+1:3d}/{len(test_cases)} [{cls}] len={task.get('length', '?')} ...")

            for arm in arms:
                if (task_id, arm) in done:
                    print(f"  [{arm:<12}] (skipped — already in checkpoint)")
                    continue

                if arm == 'zero_shot':
                    gen = run_zero_shot(task, call_timeout=call_timeout)
                    seq = gen.get('sequence', '')
                    score, best_ref = score_against_pool_b(seq, pool_b_c, activity)

                elif arm == 'best_of_n':
                    gen = run_best_of_n(task, n=n_bestofn, call_timeout=call_timeout)
                    # Score every candidate against pool B, keep the best —
                    # this is what "best-of-N" means when there's no single
                    # ground-truth reference to score against during generation.
                    best_score, best_seq, best_ref = 0.0, '', ''
                    for cand in gen.get('all_seqs', []):
                        cand_score, cand_ref = score_against_pool_b(cand, pool_b_c, activity)
                        if cand_score > best_score:
                            best_score, best_seq, best_ref = cand_score, cand, cand_ref
                    seq, score = best_seq, best_score

                elif arm == 'agent':
                    gen = run_agent_arm(task, call_timeout=agent_timeout)
                    if gen.get('iterations'):
                        all_iterations.append(gen['iterations'])
                    seq = gen.get('sequence', '')
                    score, best_ref = score_against_pool_b(seq, pool_b_c, activity)

                else:
                    continue

                comps = score_components(best_ref, seq) if (seq and best_ref) else {}
                validation = validate_sequence(seq, task) if seq else {}
                rb_score = rulebook_score(validation) if validation else 0.0

                result = {
                    'task_id': task_id,
                    'class': cls,
                    'arm': arm,
                    'sequence': seq,
                    'score': score,
                    'components': comps,
                    'rb_score': rb_score,
                    'best_ref': best_ref,
                    'time_s': gen.get('time_s', 0),
                    'n_calls': gen.get('n_calls', gen.get('iterations', 1)),
                    'timed_out': gen.get('timed_out', False),
                }
                arm_results[arm].append(result)
                done.add((task_id, arm))

                ckpt_f.write(json.dumps(result) + '\n')
                ckpt_f.flush()

                status = 'TIMEOUT' if result['timed_out'] else ('PASS' if rb_score >= 0.8 else 'FAIL')
                print(f"  [{arm:<12}] {seq[:20]:<20} score={score:.4f} {status}")

            # Rewrite the summary after every task, not just at the end, so
            # a killed/interrupted run always leaves a valid, current result
            # file on disk instead of nothing.
            _write_summary(output_dir, model_name, n_bestofn, len(test_cases), arms, arm_results)

    final = _write_summary(output_dir, model_name, n_bestofn, len(test_cases), arms, arm_results)

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY — {model_name}")
    print(f"{'='*70}")
    print(f"{'Arm':<15} {'Avg Score':>10} {'Valid%':>8} {'RB Pass%':>10} {'Avg Time':>10}")
    print(f"{'-'*55}")
    for arm in arms:
        s = final['summaries'][arm]
        print(f"{arm:<15} {s['mean_score']:>10.4f} {s['valid_pct']:>7.1f}% {s['rb_pass_pct']:>9.1f}% {s['avg_time_s']:>9.1f}s")
    print(f"{'='*70}")

    if all_iterations:
        avg_iters = mean(all_iterations)
        print(f"\nAgent avg iterations: {avg_iters:.1f}")
        print(f"Recommended N for Best-of-N: {round(avg_iters)}")

    out_path = os.path.join(output_dir, f"results_{model_name}.json")
    print(f"\nResults saved to {out_path}")
    print(f"Checkpoint at {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Three-arm evaluation: zero-shot, best-of-N, agent"
    )
    parser.add_argument('--test-cases', default='eval/test_cases.json')
    parser.add_argument('--pool-b', default='eval/pools/pool_b_heldout.json')
    parser.add_argument('--output', default='eval/results')
    parser.add_argument('--n-bestofn', type=int, default=4)
    parser.add_argument('--model-name', default='model')
    parser.add_argument('--arms', nargs='+', default=['zero_shot', 'best_of_n', 'agent'],
                         choices=['zero_shot', 'best_of_n', 'agent'])
    parser.add_argument('--call-timeout', type=float, default=90,
                         help='Per-call wall-clock ceiling in seconds for zero_shot/best_of_n (default: 90)')
    parser.add_argument('--agent-timeout', type=float, default=300,
                         help='Wall-clock ceiling in seconds for one full agent-arm run (default: 300)')
    args = parser.parse_args()
    run_evaluation(
        args.test_cases, args.pool_b, args.output,
        args.n_bestofn, args.model_name, args.arms,
        args.call_timeout, args.agent_timeout,
    )


if __name__ == '__main__':
    main()
