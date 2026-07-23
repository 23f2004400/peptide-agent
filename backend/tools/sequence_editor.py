"""
Sequence-editing tool: deterministic property fixes + LLM-guided targeted edits.

Used by PepForgeAgent.generate() to *edit* the current best sequence each
iteration instead of regenerating from scratch. Pure functions only — no
dependency on models.py or extract_sequence, so these are independently
testable without a live model/gateway.
"""

from __future__ import annotations

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


def deterministic_edit(sequence: str, task: dict) -> str | None:
    """
    Apply length -> charge -> hydrophobicity fixes in sequence.
    Returns None if the result is unchanged or invalid, so the caller can
    skip adding a redundant/bad candidate.
    """
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

    if edited == sequence or not all(c in VALID_AA for c in edited):
        return None
    return edited


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


# Prompt-text creativity nudge, cycled per rotation attempt. Not real sampling
# temperature — models.generate() doesn't expose one, and backend/models.py is
# off-limits — this is the fallback the spec itself anticipates: a phrase in
# the system prompt instead of an actual temperature parameter.
_CREATIVITY_HINTS = [
    "Be conservative — change the minimum number of residues necessary.",
    "Try a moderately different substitution than an obvious one.",
    "Be more exploratory — consider a bolder set of substitutions.",
]


def build_edit_prompt(
    sequence: str,
    score: float | None,
    components: dict,
    task: dict,
    iteration: int,
    rotation_offset: int = 0,
    creativity_idx: int = 0,
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

    if weakest_key == "charge":
        instruction = (
            f"Current charge {actual_charge:+d}, target {charge_min:+d} to {charge_max:+d}. "
            + (f"Add {charge_min - actual_charge} K or R residues by replacing neutral AA."
               if actual_charge < charge_min
               else f"Replace {actual_charge - charge_max} K/R with A or L.")
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

    creativity_hint = _CREATIVITY_HINTS[creativity_idx % len(_CREATIVITY_HINTS)]
    system = (
        "You are a peptide sequence editor. "
        "You receive a peptide and make MINIMAL targeted substitutions "
        "to improve a specific property. "
        f"{creativity_hint} "
        "Output ONLY the modified sequence - uppercase letters, "
        "no spaces, no explanation, nothing else. One line only."
    )

    score_str = f"{score:.4f}" if score is not None else "N/A"
    weakest_line = (
        f"WEAKEST COMPONENT: {weakest_key.upper().replace('_', ' ')} (score={weakest_val:.3f})\n"
        if weakest_val is not None
        else f"TARGETING: {weakest_key.upper()}\n"
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
        f"- Change ONLY 2-4 residues, keep the rest identical\n\n"
        f"Output the edited sequence:"
    )

    return system, user, weakest_key
