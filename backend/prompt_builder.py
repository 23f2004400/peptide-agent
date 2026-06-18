"""
Dynamic prompt builder with feedback injection for iterative refinement.
"""

from __future__ import annotations

# Amino acid charge cheat sheet injected into every prompt so the model can
# reason about charge without knowing biochemistry conventions.
_CHARGE_REF = (
    "Charge reference: K=+1, R=+1, H=+1, D=-1, E=-1; all others = 0"
)

# Hydrophobicity bucket guide (Kyte-Doolittle ranges, simplified)
_HYDRO_REF = (
    "Hydrophobicity guide: "
    "very hydrophobic (>2): I,V,L,F,C,M,A,W,Y  |  "
    "neutral (~0): G,T,S,P  |  "
    "hydrophilic (<-1): H,Q,N,D,E,K,R"
)

# Human-readable activity names for the prompt
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
    "You are a computational biochemist. "
    "When asked to design a peptide, respond with ONLY the amino acid sequence "
    "in uppercase single-letter codes — nothing else."
)


def build_prompt(task: dict, feedback_history: list[dict] | None = None) -> str:
    """
    Build a compact, completion-primed prompt for OpenBioLLM.

    Improvements over v1:
    - Shorter: ~50% fewer tokens → less likely to trigger chatty responses
    - No listing of all inactive activities (the long "is NOT …" list)
    - Hydrophobicity expressed as % hydrophobic residues (more intuitive for LLM)
    - "Sequence:" at end primes the model to continue with the sequence
    - Feedback gives specific amino-acid substitution instructions
    """
    length     = task.get('length', 12)
    charge     = task.get('charge', 0)
    hydro      = task.get('hydrophobicity', 0.0)
    activities = task.get('activities', [])

    # Estimate hydrophobic residue % from target Kyte-Doolittle avg
    # KD avg 0 ≈ 30%, 0.5 ≈ 40%, 1.0 ≈ 50%, -1.0 ≈ 20%
    hydro_pct = max(10, min(80, int(30 + hydro * 20)))

    act_labels = [_ACTIVITY_LABELS.get(a, a) for a in activities]
    act_str = ", ".join(act_labels) if act_labels else "general"

    charge_sign = f"+{int(charge)}" if charge >= 0 else str(int(charge))

    lines = [
        f"Design a {act_str} peptide:",
        "",
        f"  Length : exactly {length} amino acids",
        f"  Charge : {charge_sign}  ({_CHARGE_REF})",
        f"  Hydrophobicity : ~{hydro_pct}% hydrophobic residues  ({_HYDRO_REF})",
        "",
        "Rules: uppercase single-letter codes only (A C D E F G H I K L M N P Q R S T V W Y).",
        "Output the sequence on a single line with no labels, spaces, or dashes.",
        "",
    ]

    # Add a worked example for the first attempt to show the model the expected format
    if not feedback_history:
        if "drug-delivery" in activities or "cell-cell-communication" in activities:
            example = "RKKRKLLLKRKLLLK"[:length] or "KLLKLLLKLWKK"
        elif any(a in activities for a in ("anti-bacterial","anti-fungal","anti-viral","anti-cancer")):
            example = "GIGKFLKKAKKFGKAF"[:length] or "KLLKLLKLWKK"
        else:
            example = "RKKRQLLKKLWKK"[:length] or "KLLKLLLKLWKK"
        # Pad/trim example to target length using canonical AA pattern
        if len(example) < length:
            filler = "KLKLKLKLKLKLKLKLKL"
            example = (example + filler)[:length]
        lines.append(f"Example format (do NOT copy exactly): {example}")
        lines.append("")

    if feedback_history:
        lines.append("PREVIOUS ATTEMPTS — do NOT repeat these sequences:")
        lines.append("")
        for i, fb in enumerate(feedback_history, 1):
            seq = fb.get('seq') or '(no sequence)'
            score_str = f"{fb['score']:.4f}" if fb.get('score') is not None else "N/A"
            lines.append(f"  Attempt {i}: {seq}  (score {score_str})")
            if fb.get('issues'):
                for issue in fb['issues']:
                    lines.append(f"    PROBLEM: {issue}")
            if fb.get('fix_hints'):
                for hint in fb['fix_hints']:
                    lines.append(f"    FIX:     {hint}")
        lines.append("")
        lines.append("Generate a NEW sequence that fixes ALL problems above.")
        lines.append("")

    lines.append("Sequence:")
    return "\n".join(lines)


def _charge_hints(actual: float, target: float) -> list[str]:
    """Return actionable amino-acid substitution hints to correct charge."""
    diff = target - actual
    hints = []
    if diff > 1:
        n = int(round(diff))
        hints.append(
            f"Add ~{n} positively charged residues: replace some neutral AA with K (lysine) or R (arginine)"
        )
        hints.append("Remove any D (aspartate) or E (glutamate) if present — they subtract charge")
    elif diff < -1:
        n = int(round(-diff))
        hints.append(
            f"Reduce positive charge by {n}: replace some K/R with neutral AA (A, L, V, G, S)"
        )
        hints.append("Or add D (aspartate) / E (glutamate) to lower net charge")
    return hints


def _hydro_hints(actual: float, target: float) -> list[str]:
    """Return actionable hydrophobicity correction hints."""
    diff = target - actual
    hints = []
    if diff > 0.3:
        hints.append(
            "Hydrophobicity too low: replace some K/R/D/E/N/Q with hydrophobic residues L, I, V, F, W, or Y"
        )
    elif diff < -0.3:
        hints.append(
            "Hydrophobicity too high: replace some L/I/V/F/W with polar residues K, R, S, N, Q, or E"
        )
    return hints


def build_feedback_entry(seq: str, score, validation: dict, task: dict) -> dict:
    """
    Build a rich feedback dict with fix hints for the next prompt iteration.
    Call this instead of building the dict inline in agent.py.
    """
    issues = list(validation.get('issues', []))
    fix_hints: list[str] = []

    # Charge hints
    actual_c = validation.get('net_charge', 0)
    target_c = task.get('charge')
    if target_c is not None and abs(actual_c - target_c) > 1:
        fix_hints.extend(_charge_hints(actual_c, target_c))

    # Hydrophobicity hints
    actual_h = validation.get('hydrophobicity', 0.0)
    target_h = task.get('hydrophobicity')
    if target_h is not None and abs(actual_h - target_h) > 0.3:
        fix_hints.extend(_hydro_hints(actual_h, target_h))

    # Length hint
    actual_len = validation.get('length', len(seq))
    target_len = task.get('length')
    if target_len is not None and abs(actual_len - target_len) > 2:
        if actual_len < target_len:
            fix_hints.append(f"Too short ({actual_len} AA): add {target_len - actual_len} more residues")
        else:
            fix_hints.append(f"Too long ({actual_len} AA): remove {actual_len - target_len} residues")

    return {
        'seq': seq,
        'score': score,
        'issues': issues,
        'fix_hints': fix_hints,
    }


def get_system_prompt() -> str:
    return BASE_SYSTEM
