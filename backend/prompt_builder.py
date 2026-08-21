"""
Dynamic prompt builder with feedback injection for iterative refinement.
"""

from __future__ import annotations

import os

# Models observed reciting the standard prompt's own constraint text back
# ("Alphabet: The alphabet must include...") instead of generating a
# sequence — a different manifestation of the same chat_template-less
# deployment issue documented for OpenBioLLM's prose/explanation failures,
# but triggered by prompt length/structure rather than fixed by the
# assistant primer alone. ECHO_PRONE_MODEL=true in .env force-enables the
# short-prompt path for a model not in this built-in list.
_ECHO_PRONE_MODELS = ("biomistral", "meditron")


def _is_echo_prone_model() -> bool:
    if os.environ.get("ECHO_PRONE_MODEL", "").strip().lower() == "true":
        return True
    model_name = os.environ.get("MODEL_NAME", "").lower()
    return any(name in model_name for name in _ECHO_PRONE_MODELS)


_ACTIVITY_LABELS = {
    "anti-bacterial": "antimicrobial (AMP)",
    "anti-fungal": "antifungal (AMP)",
    "anti-viral": "antiviral (AMP)",
    "anti-cancer": "anticancer (AMP)",
    "drug-delivery": "cell-penetrating (CPP) for drug delivery",
    "signal-peptide": "signal peptide",
    "immunological": "immunological / epitope",
    "inhibitor": "enzyme inhibitor",
    "metabolic": "metabolic regulator",
    "cell-cell-communication": "cell-cell communication",
    "other-functional": "functional",
    "toxic": "toxic / venom-derived",
}

BASE_SYSTEM = (
    "You are a peptide sequence generator. "
    "Your entire response must be ONE LINE containing only the amino acid sequence "
    "in uppercase single-letter codes (A C D E F G H I K L M N P Q R S T V W Y). "
    "No explanation, no labels, no punctuation — just the letters."
)


def _resolve_charge(task: dict) -> float:
    """Support both 'charge' (frontend) and 'ref_net_charge' (eval dataset) keys."""
    if 'charge' in task:
        return float(task['charge'])
    return float(task.get('ref_net_charge', 0))


def _resolve_hydro_pct(task: dict) -> int | None:
    """
    Return target hydrophobicity as a percentage (0-100), or None if not specified.
    """
    if 'ref_hydrophobic_pct' in task:
        return int(task['ref_hydrophobic_pct'])
    if 'hydrophobicity' in task:
        kd = float(task['hydrophobicity'])
        return max(10, min(80, int(30 + kd * 20)))
    return None


def _get_ranges(task: dict) -> tuple[int, int, int, int, int, int]:
    """
    Return (len_lo, len_hi, charge_lo, charge_hi, hydro_lo, hydro_hi).
    Prefers explicit range params; falls back to computing ±tolerance from single values.
    """
    length = task.get('length', 12)

    # Length range
    len_lo = int(task['length_min']) if 'length_min' in task else max(1, length - 2)
    len_hi = int(task['length_max']) if 'length_max' in task else length + 2

    # Charge range
    if 'charge_min' in task and 'charge_max' in task:
        charge_lo = int(task['charge_min'])
        charge_hi = int(task['charge_max'])
    else:
        c = int(round(_resolve_charge(task)))
        charge_lo = c - 1
        charge_hi = c + 1

    # Hydrophobic % range
    if 'hydro_min' in task and 'hydro_max' in task:
        hydro_lo = int(task['hydro_min'])
        hydro_hi = int(task['hydro_max'])
    else:
        pct = _resolve_hydro_pct(task) or 40
        hydro_lo = max(0, pct - 15)
        hydro_hi = min(100, pct + 15)

    return len_lo, len_hi, charge_lo, charge_hi, hydro_lo, hydro_hi


_ANCHOR_CHARGE_AA = {'K': 1, 'R': 1, 'D': -1, 'E': -1}
_ANCHOR_HYDRO_AA = set('LIVFWMAYC')


def _initial_prompt(
    task: dict,
    feedback_history: list[dict] | None = None,
    rag_examples: list[dict] | None = None,
    reference: str = '',
) -> str:
    length = task.get('length', 12)
    activities = task.get('activities', [])
    act_labels = [_ACTIVITY_LABELS.get(a, a) for a in activities]
    act_str = ", ".join(act_labels) if act_labels else "general"

    len_lo, len_hi, charge_lo, charge_hi, hydro_lo, hydro_hi = _get_ranges(task)

    charge_lo_str = f"{charge_lo:+d}"
    charge_hi_str = f"{charge_hi:+d}"

    if _is_echo_prone_model():
        # Short, direct prompt — minimizes text available to echo back.
        # Deliberately skips RAG examples and the reference anchor: this
        # model class was observed reciting the full prompt's own
        # constraint lines ("Alphabet: The alphabet must include...")
        # instead of generating, and the full prompt gives far more text
        # to echo from than this stripped-down version does.
        return (
            f"Task: generate one {act_str} peptide sequence.\n\n"
            f"Requirements:\n"
            f"- Length: {len_lo} to {len_hi} amino acids\n"
            f"- Net charge: {charge_lo_str} to {charge_hi_str}\n"
            f"- Hydrophobic residues: {hydro_lo}% to {hydro_hi}%\n"
            f"- Use only standard amino acids: ACDEFGHIKLMNPQRSTVWY\n\n"
            f"Output the peptide sequence only. No explanation. No bullet points. No words.\n\n"
            f"Sequence:"
        )

    # Build terse inline correction hints from the most recent attempt that had issues.
    # Bracket notes like "[use fewer K/R]" steer residue choice without triggering
    # OpenBioLLM's comprehension/prose mode (which a full retry block always triggers).
    charge_note = ""
    hydro_note = ""
    if feedback_history:
        for fb in reversed(feedback_history):
            if fb.get('seq') and fb.get('fix_hints'):
                for hint in fb['fix_hints']:
                    h = hint.lower()
                    if 'charge' in h and not charge_note:
                        charge_note = (" [replace K/R with A/L/V]"
                                       if ('replace' in h)
                                       else " [add K for charge]")
                    if 'hydrophob' in h and not hydro_note:
                        hydro_note = (" [add more L/F/I/V/M]"
                                      if 'add l' in h
                                      else " [use fewer L/I/V/F/W/M]")
                break

    rag_section = ""
    if rag_examples:
        rag_lines = [
            "\nREAL PEPTIDES FROM DATABASE (use as structural inspiration):",
            "These are validated peptides with similar properties to your target.",
            "Do NOT copy them directly. Use them to guide residue choices.\n",
        ]
        for i, ex in enumerate(rag_examples, 1):
            acts = ', '.join(ex.get('activities', [])) or 'none'
            rag_lines.append(
                f"  Example {i}: {ex['sequence']}"
                f"  [len={ex['length']}, "
                f"charge={ex['charge']:+d}, "
                f"hydro={ex['hydro_pct']:.0f}%, "
                f"activity={acts}]"
            )
        rag_section = "\n".join(rag_lines) + "\n"

    # Reference anchor — surfaces the actual target's motifs/charge/hydro so
    # generation has something concrete to converge toward instead of
    # producing a functionally-valid but sequentially unrelated sequence
    # (the main driver of n-gram BLEU sitting near 0 in iteration 1). Only a
    # real sequence, not an auto-picked default's absence, triggers this —
    # callers that want the terse fallback prompt simply omit `reference`.
    anchor_section = ""
    ref_upper = reference.upper() if reference else ""
    if ref_upper and len(ref_upper) >= 5:
        ref_charge = sum(_ANCHOR_CHARGE_AA.get(aa, 0) for aa in ref_upper)
        ref_hydro = round(
            100 * sum(1 for aa in ref_upper if aa in _ANCHOR_HYDRO_AA) / len(ref_upper), 1
        )
        anchor_section = (
            f"\nREFERENCE SEQUENCE (use as structural inspiration):\n"
            f"{ref_upper}\n"
            f"Key patterns to approximate in your generated sequence:\n"
            f"- Opening motif (first 4 residues): {ref_upper[:4]}\n"
            f"- Closing motif (last 3 residues): {ref_upper[-3:]}\n"
            f"- Net charge: {ref_charge:+d}\n"
            f"- Hydrophobic fraction: {ref_hydro:.0f}%\n"
            f"Generate a NOVEL sequence that shares similar motifs "
            f"but is not identical to the reference.\n"
        )

        # The reference anchor above only ever shows a same-length-ish motif
        # (opening 4 / closing 3 residues) — if the reference itself is
        # shorter than the task's minimum length, the model tends to just
        # reproduce the reference's own length instead of the target range
        # (observed: 30-40 AA task, 23 AA reference -> 23 AA output). Call
        # this out explicitly since the length constraint at the top of the
        # prompt alone wasn't enough to override the reference's pull.
        if len(ref_upper) < len_lo:
            anchor_section += (
                f"\nCRITICAL LENGTH WARNING: The reference sequence above is "
                f"only {len(ref_upper)} amino acids long. Your generated "
                f"sequence MUST be {len_lo}-{len_hi} amino acids long — "
                f"significantly longer than the reference. Use the "
                f"reference as a MOTIF PATTERN only; extend it with "
                f"additional residues to reach the required length. Do NOT "
                f"generate a sequence as short as the reference "
                f"({len(ref_upper)} AA).\n"
            )

    return (
        f"Generate one {act_str} peptide meeting ALL of:\n"
        f"- Length: between {len_lo} and {len_hi} amino acids\n"
        f"- Net charge: between {charge_lo_str} and {charge_hi_str}{charge_note}\n"
        f"- Hydrophobic residues: between {hydro_lo}% and {hydro_hi}%{hydro_note}\n"
        f"- Alphabet: 20 standard amino acids only\n"
        f"{rag_section}"
        f"{anchor_section}"
        f"\n"
        f"Output: one line, uppercase letters only, nothing else.\n"
        f"Sequence:"
    )


def build_prompt(
    task: dict,
    feedback_history: list[dict] | None = None,
    rag_examples: list[dict] | None = None,
    reference: str = '',
) -> str:
    return _initial_prompt(task, feedback_history, rag_examples, reference)


def build_feedback_entry(seq: str, score, validation: dict, task: dict,
                          comp: dict | None = None, reference: str = '') -> dict:
    """Build a feedback dict with plain-English per-axis failure summaries."""
    issues: list[str] = list(validation.get('issues', []))
    fix_hints: list[str] = []

    len_lo, len_hi, charge_lo, charge_hi, hydro_lo, hydro_hi = _get_ranges(task)

    actual_charge    = int(round(float(validation.get('net_charge', 0))))
    actual_hydro_pct = float(validation.get('hydro_percent', 0.0))
    actual_len       = validation.get('length', len(seq))

    if not (charge_lo <= actual_charge <= charge_hi):
        direction = "add K residues" if actual_charge < charge_lo else "replace some K/R with A, L, V, or I"
        fix_hints.append(
            f"Charge was {actual_charge:+d}, needs {charge_lo:+d} to {charge_hi:+d} — {direction}"
        )

    if not (hydro_lo <= actual_hydro_pct <= hydro_hi):
        if actual_hydro_pct < hydro_lo:
            direction = "add L, I, V, F, W, M, or A residues"
        else:
            direction = "replace some L/I/V/F/W/M/A with K, R, S, N, or Q"
        fix_hints.append(
            f"Hydrophobic fraction was {actual_hydro_pct:.0f}%, "
            f"needs {hydro_lo}% to {hydro_hi}% — {direction}"
        )

    if not (len_lo <= actual_len <= len_hi):
        if actual_len < len_lo:
            fix_hints.append(
                f"Length was {actual_len}, needs {len_lo} to {len_hi} — add more residues"
            )
        else:
            fix_hints.append(
                f"Length was {actual_len}, needs {len_lo} to {len_hi} — remove some residues"
            )

    if reference and comp:
        if comp.get('ngram_bleu', 1.0) < 0.5:
            fix_hints.append(
                f"Try incorporating a fragment similar to '{reference[:4]}' seen in related peptides"
            )
        if comp.get('blosum', 1.0) < 0.5:
            fix_hints.append(
                f"Use residues chemically similar to the reference motif '{reference[:6]}'"
            )

    return {
        'seq': seq,
        'score': score,
        'issues': issues,
        'fix_hints': fix_hints,
    }


def get_system_prompt() -> str:
    return BASE_SYSTEM
