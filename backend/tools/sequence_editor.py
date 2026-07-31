"""
Sequence-editing tool: deterministic property fixes + LLM-guided targeted edits.

Used by PepForgeAgent.generate() to *edit* the current best sequence each
iteration instead of regenerating from scratch. Pure functions only — no
dependency on models.py or extract_sequence, so these are independently
testable without a live model/gateway.
"""

from __future__ import annotations

import random

from ..rulebook import validate_sequence

VALID_AA  = set("ACDEFGHIKLMNPQRSTVWY")
CHARGE_AA = {'K': 1, 'R': 1, 'D': -1, 'E': -1}
HYDRO_AA  = set('LIVFWMAYC')


# == DETERMINISTIC FIXES ======================================================

def fix_charge(sequence: str, target_charge: int, tolerance: int = 1) -> str:
    """Deterministically fix net charge by swapping residues."""
    seq  = list(sequence.upper())
    curr = sum(CHARGE_AA.get(aa, 0) for aa in seq)
    diff = curr - target_charge

    if abs(diff) <= tolerance:
        return sequence

    if diff > 0:
        for i, aa in enumerate(seq):
            if diff <= tolerance:
                break
            if aa in ('K', 'R'):
                seq[i] = 'L'
                diff -= 1
    else:
        for i, aa in enumerate(seq):
            if diff >= -tolerance:
                break
            if aa in ('D', 'E'):
                seq[i] = 'K'
                diff += 1
        if diff < -tolerance:
            for i, aa in enumerate(seq):
                if diff >= -tolerance:
                    break
                if aa not in ('K', 'R', 'D', 'E'):
                    seq[i] = 'K'
                    diff += 1

    return ''.join(seq)


def fix_length(sequence: str, target_length: int) -> str:
    """Trim or pad sequence to target length."""
    seq = sequence.upper()
    if len(seq) == target_length:
        return seq
    if len(seq) > target_length:
        return seq[:target_length]
    return seq + 'L' * (target_length - len(seq))


def fix_hydrophobicity(sequence: str, target_min: float, target_max: float) -> str:
    """Adjust hydrophobic fraction by targeted swaps."""
    seq    = list(sequence.upper())
    n      = len(seq)
    actual = 100 * sum(1 for aa in seq if aa in HYDRO_AA) / n

    if target_min <= actual <= target_max:
        return sequence

    if actual < target_min:
        for i, aa in enumerate(seq):
            actual = 100 * sum(1 for a in seq if a in HYDRO_AA) / n
            if actual >= target_min:
                break
            if aa not in HYDRO_AA and aa not in ('K', 'R', 'D', 'E'):
                seq[i] = 'L'
    else:
        polar = ['S', 'T', 'N', 'Q']
        pi    = 0
        for i, aa in enumerate(seq):
            actual = 100 * sum(1 for a in seq if a in HYDRO_AA) / n
            if actual <= target_max:
                break
            if aa in HYDRO_AA and aa not in ('W', 'F'):
                seq[i] = polar[pi % len(polar)]
                pi += 1

    return ''.join(seq)


# Conservative, chemically-similar substitutions used as the last-resort
# guaranteed-change fallback below — swapping to one of these preserves the
# residue's general character (size/charge/polarity) rather than a random pick.
_CONSERVATIVE_SUBS = {
    'L': 'I', 'I': 'V', 'V': 'L',
    'K': 'R', 'R': 'K',
    'D': 'E', 'E': 'D',
    'S': 'T', 'T': 'S',
    'F': 'Y', 'Y': 'W', 'W': 'F',
    'A': 'G', 'G': 'A',
    'N': 'Q', 'Q': 'N',
    'M': 'L', 'H': 'N', 'C': 'A', 'P': 'A',
}


def deterministic_edit(sequence: str, task: dict) -> str | None:
    """
    Apply length -> charge -> hydrophobicity fixes in sequence.

    If the sequence is already within every tolerance (nothing to fix), force
    one conservative substitution at the midpoint residue instead of
    returning None — this guarantees a genuinely different, valid candidate
    every time, so the caller never has to retry the LLM on its own stuck
    output just to get *some* alternative to try.

    Returns None only if the input itself isn't a valid AA sequence.
    """
    if not sequence or not all(c in VALID_AA for c in sequence.upper()):
        return None

    target_length = task.get('length', len(sequence))
    charge_min    = task.get('charge_min', task.get('charge', 3) - 1)
    charge_max    = task.get('charge_max', task.get('charge', 3) + 2)
    target_charge = (charge_min + charge_max) // 2
    hydro_min     = task.get('hydro_min', 35)
    hydro_max     = task.get('hydro_max', 55)

    edited = sequence
    edited = fix_length(edited, target_length)
    edited = fix_charge(edited, target_charge, tolerance=1)
    edited = fix_hydrophobicity(edited, hydro_min, hydro_max)

    if edited == sequence:
        seq = list(sequence.upper())
        mid = len(seq) // 2
        seq[mid] = _CONSERVATIVE_SUBS.get(seq[mid], 'A' if seq[mid] != 'A' else 'G')
        edited = ''.join(seq)

    if not all(c in VALID_AA for c in edited):
        return None
    return edited


def inject_reference_motif(sequence: str, reference: str, target_length: int) -> str:
    """
    Forcibly splice a 4-char motif from `reference` into `sequence` at a fixed
    offset. Deterministic, no LLM involved — a guaranteed way to get some
    literal subsequence overlap with the reference when ngram_bleu is stuck
    at 0 and the LLM edit isn't reliably following a "reuse this motif"
    instruction. Just another candidate for the caller to score alongside the
    others — not assumed to be an improvement (splicing in 4 residues can
    just as easily wreck charge/hydrophobicity), so it only wins if it's
    actually competitive on the real metric.
    """
    if not reference or len(reference) < 4:
        return sequence

    ref = ''.join(c for c in reference.upper() if c in VALID_AA)
    if len(ref) < 4:
        return sequence

    seq = list(sequence.upper())

    best_motif = None
    for i in range(len(ref) - 3):
        motif = ref[i:i + 4]
        if all(c in VALID_AA for c in motif):
            best_motif = motif
            break

    if not best_motif:
        return sequence

    # Insert at position 4 (keeps first 4 chars intact) — preserves the
    # opening motif/charge pattern while adding reference similarity.
    insert_pos = min(4, max(0, len(seq) - 4))
    for j, aa in enumerate(best_motif):
        if insert_pos + j < len(seq):
            seq[insert_pos + j] = aa

    result = ''.join(seq)

    if len(result) > target_length:
        result = result[:target_length]
    elif len(result) < target_length:
        result = result + 'L' * (target_length - len(result))

    return result


# == LLM-GUIDED EDIT ==========================================================

_COMPONENT_INSTRUCTIONS_TEMPLATE = {
    "ngram_bleu": (
        "Make 3-4 substitutions using residues common in {activities} peptides "
        "(K, R, L, I, F, W are common in AMPs). "
        "Focus on positions 1-5 and last 3 positions."
    ),
    "functional_group": (
        "Improve amino acid diversity. "
        "Avoid runs of more than 3 same-type residues. "
        "Mix charged (K,R), hydrophobic (L,I,V,F,W), polar (S,T,N,Q) groups."
    ),
    "property_distribution": (
        "Redistribute properties along the sequence. "
        "Place charged residues at termini and hydrophobic in the center "
        "(typical AMP amphipathic helix pattern)."
    ),
    "structural": (
        "Fix structural issues: remove internal prolines (P) that break helices. "
        "Ensure cysteines are paired or replace unpaired ones with A."
    ),
    "blosum": (
        "Make evolutionarily conservative substitutions: "
        "L<->I<->V, K<->R, D<->E, S<->T, F<->Y. "
        "Avoid radical changes like charged<->hydrophobic."
    ),
}


def build_edit_prompt(
    sequence: str,
    score: float | None,
    components: dict,
    task: dict,
    iteration: int,
    rotation_offset: int = 0,
    reference: str = '',
    override_instruction: str | None = None,
) -> tuple[str, str, str]:
    """
    Build (system, user, weakest) prompt text for a targeted LLM edit of `sequence`.
    `weakest` is the PeptideBLEU component name targeted (or "rulebook" when
    there's no reference), for the caller to log/display.

    When `components` has PeptideBLEU scores, targets the component at
    `rotation_offset` positions from the weakest (0 = weakest, 1 = 2nd
    weakest, ...) — lets the caller rotate through different improvement
    angles across attempts instead of always hammering the single weakest
    component. When there's no reference (components empty), falls back to
    instructions built from rulebook validation issues instead.
    """
    actual_charge = sum(CHARGE_AA.get(aa, 0) for aa in sequence)
    actual_hydro  = round(
        100 * sum(1 for aa in sequence if aa in HYDRO_AA) / max(len(sequence), 1), 1
    )

    charge_min = int(task.get('charge_min', task.get('charge', 3) - 1))
    charge_max = int(task.get('charge_max', task.get('charge', 3) + 2))
    hydro_min  = task.get('hydro_min', 35)
    hydro_max  = task.get('hydro_max', 55)
    length     = task.get('length', len(sequence))
    activities = task.get('activities', [])

    if components:
        sorted_comps = sorted(components.items(), key=lambda kv: kv[1])
        target_idx = rotation_offset % len(sorted_comps)
        weakest_key, weakest_val = sorted_comps[target_idx]
    else:
        weakest_key = ""
        weakest_val = None

    if override_instruction is not None:
        # Escape-mechanism callers (see build_exact_swap_instruction) supply
        # an exact, pre-built instruction — skip weakest-component selection
        # entirely rather than have it compete with/dilute the override.
        instruction = override_instruction
        weakest_key = "escape_exact"
        weakest_val = None
    elif weakest_key == "charge":
        # Cap the requested count at 3 so this instruction never asks for more
        # changes than the "Change ONLY 2-4 residues" hard constraint below
        # allows — a large charge gap (e.g. needing +5) previously produced
        # a contradictory "add 5 ... change only 2-4" prompt, and the model
        # would follow the higher number, overshooting into other properties.
        if actual_charge < charge_min:
            changes = min(charge_min - actual_charge, 3)
            instruction = (
                f"Current charge {actual_charge:+d}, target {charge_min:+d} to {charge_max:+d}. "
                f"Replace exactly {changes} neutral residues (A, L, V, G, S, N) with K or R."
            )
        else:
            changes = min(actual_charge - charge_max, 3)
            instruction = (
                f"Current charge {actual_charge:+d}, target {charge_min:+d} to {charge_max:+d}. "
                f"Replace exactly {changes} K/R residues with A or L."
            )
    elif weakest_key == "hydrophobicity":
        instruction = (
            f"Hydrophobic fraction {actual_hydro}%, target {hydro_min}-{hydro_max}%. "
            + ("Add L/I/V/F/W residues by replacing polar AA."
               if actual_hydro < hydro_min
               else "Replace some L/I/V/F with S/T/N/Q.")
        )
    elif weakest_key in _COMPONENT_INSTRUCTIONS_TEMPLATE:
        instruction = _COMPONENT_INSTRUCTIONS_TEMPLATE[weakest_key].format(
            activities=', '.join(activities) or 'antimicrobial'
        )
    else:
        # No reference / no components — fall back to rulebook validation issues.
        validation = validate_sequence(sequence, task)
        issues = validation.get('issues', [])
        instruction = (
            "; ".join(issues) if issues
            else "Make 2-3 targeted substitutions to improve sequence quality."
        )
        weakest_key = "rulebook"

    # N-gram BLEU stuck near zero means the edit has no literal subsequence
    # overlap with the reference at all — the generic ngram_bleu instruction
    # above (common AMP residues) doesn't fix that, since "common" residues
    # aren't necessarily the reference's own residues. Surface a short literal
    # fragment of the actual reference as concrete motif material, regardless
    # of which component the rotation is currently targeting this iteration
    # (mirrors the fragment-hint pattern already used in prompt_builder.py's
    # build_feedback_entry() for the same failure mode).
    if override_instruction is None and reference and components.get("ngram_bleu", 1.0) < 0.05:
        motif = reference[:6]
        instruction += (
            f' Your sequence shares no common subsequence with the reference '
            f'peptide — work this literal fragment in somewhere: "{motif}".'
        )

    system = (
        "You are a peptide sequence editor. "
        "You receive a peptide and make MINIMAL targeted substitutions "
        "to improve a specific property. "
        "Output ONLY the modified sequence - uppercase letters, "
        "no spaces, no explanation, nothing else. One line only."
    )

    score_str = f"{score:.4f}" if score is not None else "N/A"
    weakest_line = (
        f"WEAKEST COMPONENT: {weakest_key.upper().replace('_', ' ')} (score={weakest_val:.3f})\n"
        if weakest_val is not None
        else f"TARGETING: {weakest_key.upper()}\n"
    )

    # Escape 3 asks for exactly one change — echoing "2-4 residues" here would
    # reproduce the same contradictory-instruction bug already fixed once for
    # the charge branch (the model follows whichever number appears larger).
    change_count_line = (
        "- Change ONLY 1 residue — the exact one named above, keep everything else identical\n"
        if override_instruction is not None
        else "- Change ONLY 2-4 residues, keep the rest identical\n"
    )

    user = (
        f"Peptide to edit (best so far after iteration {iteration}, PeptideBLEU={score_str}):\n"
        f"{sequence}\n\n"
        f"{weakest_line}"
        f"EDIT INSTRUCTION: {instruction}\n\n"
        f"HARD CONSTRAINTS:\n"
        f"- Keep length exactly {length} amino acids\n"
        f"- Net charge: {charge_min:+d} to {charge_max:+d} (currently {actual_charge:+d})\n"
        f"- Hydrophobic {hydro_min}-{hydro_max}% (currently {actual_hydro}%)\n"
        f"- Only ACDEFGHIKLMNPQRSTVWY\n"
        f"{change_count_line}\n"
        f"Output the edited sequence:"
    )

    return system, user, weakest_key


# == edit_stuck ESCAPE MECHANISM ==============================================
# Three escalating tactics tried in order across consecutive edit_stuck
# iterations — each one more forceful/specific than the last, none of them
# falling back to a from-scratch fresh generation. All three are just more
# candidates for the caller to score and compare like any other; none of
# them assume they're an improvement.

def build_positional_edit_prompt(
    sequence: str,
    components: dict,
    task: dict,
    excluded_positions: set[int] | None = None,
) -> tuple[str, str]:
    """
    Escape 1 (first stuck iteration): target different positions than the
    ones already changed by prior successful edits, instead of repeating the
    same weakest-component instruction that just produced a no-op.
    """
    excluded = excluded_positions or set()
    available = [i for i in range(len(sequence)) if i not in excluded]
    if not available:
        # Every position has been touched at some point — exclusion has
        # nothing left to bite on, so target the whole sequence again rather
        # than produce an instruction with no positions in it.
        available = list(range(len(sequence)))
    target_pos = available[:4] if len(available) >= 4 else available

    weakest_key = min(components, key=lambda k: components[k]) if components else "rulebook"
    label = weakest_key.upper().replace('_', ' ')

    system = (
        "You are a peptide sequence editor. "
        "You receive a peptide and make MINIMAL targeted substitutions "
        "to improve a specific property. "
        "Output ONLY the modified sequence - uppercase letters, "
        "no spaces, no explanation, nothing else. One line only."
    )
    user = (
        f"Peptide to edit:\n{sequence}\n\n"
        f"TARGETING: {label}\n"
        f"Change ONLY these specific positions (1-indexed): "
        f"{[p + 1 for p in target_pos]}. Do not touch any other position.\n"
        f"Replace each with a residue that improves {label.lower()}.\n"
        f"HARD CONSTRAINTS:\n"
        f"- Keep length exactly {len(sequence)} amino acids\n"
        f"- Only ACDEFGHIKLMNPQRSTVWY\n\n"
        f"Output the complete edited sequence:"
    )
    return system, user


def forced_position_swap(sequence: str, components: dict, task: dict) -> str:
    """
    Escape 2 (second consecutive stuck iteration): no LLM call, pure code.
    Guaranteed to produce a different sequence via a BLOSUM-conservative
    change targeted at whichever component is currently weakest.
    """
    seq = list(sequence)
    n = len(seq)
    weakest = min(components, key=lambda k: components[k]) if components else ""

    if weakest in ('ngram_bleu', 'blosum') and n >= 2:
        pos_a = min(6, n - 1)
        pos_b = min(12, n - 1)
        if pos_a == pos_b:
            pos_b = max(0, pos_a - 1)
        if seq[pos_a] == seq[pos_b]:
            # Swapping two identical residues is a no-op — this escape's
            # entire point is guaranteeing a real change, so fall back to a
            # conservative substitution instead.
            seq[pos_a] = _CONSERVATIVE_SUBS.get(seq[pos_a], 'L')
        else:
            seq[pos_a], seq[pos_b] = seq[pos_b], seq[pos_a]
    elif weakest == 'charge':
        for i, aa in enumerate(seq):
            if aa not in ('K', 'R', 'D', 'E'):
                seq[i] = 'K'
                break
        else:
            seq[min(7, n - 1)] = _CONSERVATIVE_SUBS.get(seq[min(7, n - 1)], 'L')
    elif weakest == 'hydrophobicity':
        for i, aa in enumerate(seq):
            if aa not in HYDRO_AA and aa not in ('K', 'R', 'D', 'E'):
                seq[i] = 'L'
                break
        else:
            seq[min(7, n - 1)] = _CONSERVATIVE_SUBS.get(seq[min(7, n - 1)], 'L')
    else:
        pos = min(7, n - 1)
        seq[pos] = _CONSERVATIVE_SUBS.get(seq[pos], 'L')

    return ''.join(seq)


def build_exact_swap_instruction(sequence: str, components: dict) -> str:
    """
    Escape 3 (third+ consecutive stuck iteration): name one exact position
    and one exact substitution, too specific for the model to reproduce the
    input unchanged. Passed as `override_instruction` to build_edit_prompt().
    """
    n = len(sequence)
    pos = n // 2 if n < 6 else random.randint(2, n - 3)
    current = sequence[pos]
    new_aa = _CONSERVATIVE_SUBS.get(current, 'L')

    return (
        f"Make EXACTLY ONE change: replace position {pos + 1} "
        f"(amino acid {current}) with {new_aa}. "
        f"Keep ALL other {n - 1} residues IDENTICAL. "
        f"Output the complete {n}-residue sequence."
    )
