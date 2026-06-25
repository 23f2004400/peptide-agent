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
    "You are a peptide sequence generator. "
    "Your entire response must be ONE LINE containing only the amino acid sequence "
    "in uppercase single-letter codes (A C D E F G H I K L M N P Q R S T V W Y). "
    "No explanation, no labels, no punctuation — just the letters."
)


# Multiple candidate examples per preset so retries rotate through different ones.
# Each example has a different amino acid composition — model won't always copy the same one.
_AMP_EXAMPLES = [
    "GIGKFLKKAKKFGKAF",   # magainin-like, charge +5
    "KLNKSGLKSFLVATS",   # charge +3, ~40% hydrophobic
    "KFLKLFVKASHLLVS",   # charge +3, amphipathic
    "KLLKLLKLWKKLSAL",   # charge +4, strong AMP
]
_CPP_EXAMPLES = [
    "RKKRKLLLKRKLLLK",
    "GRKKRRQRRRPQKLK",
    "KALAKALAKALAKAL",
]
_SIGNAL_EXAMPLES = [
    "MSVPTQVLGLLLLWL",
    "MKFLILLFNILCLFP",
]
_IMMUNO_EXAMPLES = [
    "SIINFEKLFGILGFV",
    "GILGFVFTLKLFLKL",
]


def _get_example(activities: list[str], length: int, attempt: int = 0) -> str:
    if "drug-delivery" in activities or "cell-cell-communication" in activities:
        pool = _CPP_EXAMPLES
    elif any(a in activities for a in ("anti-bacterial", "anti-fungal", "anti-viral", "anti-cancer")):
        pool = _AMP_EXAMPLES
    elif "signal-peptide" in activities:
        pool = _SIGNAL_EXAMPLES
    elif "immunological" in activities:
        pool = _IMMUNO_EXAMPLES
    else:
        pool = _AMP_EXAMPLES
    base = pool[attempt % len(pool)]
    return (base + "KLKLKLKLKLKLKLKL")[:length]


def build_prompt(task: dict, feedback_history: list[dict] | None = None) -> str:
    length     = task.get('length', 12)
    charge     = task.get('charge', 0)
    hydro      = task.get('hydrophobicity', 0.0)
    activities = task.get('activities', [])

    hydro_pct   = max(10, min(80, int(30 + hydro * 20)))
    act_labels  = [_ACTIVITY_LABELS.get(a, a) for a in activities]
    act_str     = ", ".join(act_labels) if act_labels else "general"
    charge_sign = f"+{int(charge)}" if charge >= 0 else str(int(charge))
    attempt     = len(feedback_history) if feedback_history else 0
    example     = _get_example(activities, length, attempt)

    if not feedback_history:
        return (
            f"Generate a {length}-residue {act_str} peptide sequence.\n"
            f"Net charge {charge_sign} (K=+1, R=+1, H=+1, D=-1, E=-1). "
            f"About {hydro_pct}% hydrophobic residues (L,I,V,F,W,M,A,Y,C are hydrophobic).\n"
            f"Output only uppercase amino acid letters on one line.\n"
            f"Example: {example}\n"
            f"Sequence:"
        )

    # Retry — minimal, direct; avoid commentary language that triggers explanation mode
    prev     = feedback_history[-1]
    prev_seq = prev.get('seq') or ''
    issues   = prev.get('issues', [])
    hints    = prev.get('fix_hints', [])

    lines = [
        f"{length}-residue {act_str} peptide, charge {charge_sign}, "
        f"~{hydro_pct}% hydrophobic (K=+1,R=+1,D=-1,E=-1). Output: {example}",
    ]

    if prev_seq and len(prev_seq) >= 5 and prev_seq == prev_seq.upper():
        lines.append(f"Avoid: {prev_seq}")
        for hint in hints[:2]:
            lines.append(f"Instead: {hint}")

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
