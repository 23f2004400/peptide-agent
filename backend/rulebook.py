"""
Amino acid property validation rulebook.
"""

from __future__ import annotations

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")

CHARGE_MAP = {
    'A': 0, 'R': +1, 'N': 0, 'D': -1, 'C': 0, 'E': -1, 'Q': 0, 'G': 0,
    'H': 0, 'I': 0, 'L': 0, 'K': +1, 'M': 0, 'F': 0, 'P': 0, 'S': 0,
    'T': 0, 'W': 0, 'Y': 0, 'V': 0,
}

KD_HYDRO = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'E': -3.5,
    'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
    'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
    'Y': -1.3, 'V': 4.2,
}

HYDROPHOBIC_AAS = set("AVILMFYW")

# Activity-specific length and charge constraints
ACTIVITY_CONSTRAINTS = {
    'anti-bacterial':       {'min_len': 10, 'max_len': 50},
    'anti-fungal':          {'min_len': 10, 'max_len': 50},
    'anti-viral':           {'min_len': 10, 'max_len': 50},
    'anti-cancer':          {'min_len': 10, 'max_len': 50},
    'drug-delivery':        {'min_len': 10, 'max_len': 50, 'min_charge': 3, 'max_charge': 9},
    'signal-peptide':       {'min_len': 15, 'max_len': 30},
    'cell-cell-communication': {'min_len': 2, 'max_len': 9},
}

CHARGE_CLASS_THRESHOLDS = [
    (-99, -3, 'strongly negative'),
    (-3, -1, 'weakly negative'),
    (-1, 1, 'neutral'),
    (1, 3, 'weakly positive'),
    (3, 99, 'strongly positive'),
]

LENGTH_CLASS_THRESHOLDS = [
    (0, 10, 'very short'),
    (10, 20, 'short'),
    (20, 40, 'medium'),
    (40, 80, 'long'),
    (80, 9999, 'very long'),
]


def _charge_class(net_charge: float) -> str:
    for lo, hi, label in CHARGE_CLASS_THRESHOLDS:
        if lo <= net_charge < hi:
            return label
    return 'strongly positive'


def _length_class(length: int) -> str:
    for lo, hi, label in LENGTH_CLASS_THRESHOLDS:
        if lo <= length < hi:
            return label
    return 'very long'


def validate_sequence(seq: str, task: dict) -> dict:
    """
    Validate a peptide sequence against task constraints.

    Supports both range-based keys (length_min/max, charge_min/max, hydro_min/max)
    and legacy single-value keys (length, charge, hydrophobicity, ref_net_charge,
    ref_hydrophobic_pct).  Range keys take priority when present.
    """
    issues: list[str] = []
    seq = seq.upper().strip()

    # Rule 1: valid amino acids
    invalid = [c for c in seq if c not in VALID_AAS]
    if invalid:
        issues.append(f"Invalid amino acids: {set(invalid)}")

    seq_clean = ''.join(c for c in seq if c in VALID_AAS)
    n = len(seq_clean)

    net_charge = sum(CHARGE_MAP.get(aa, 0) for aa in seq_clean)
    hydro_vals = [KD_HYDRO.get(aa, 0) for aa in seq_clean]
    avg_hydro = sum(hydro_vals) / max(n, 1)
    hydro_pct = round(sum(1 for aa in seq_clean if aa in HYDROPHOBIC_AAS) / max(n, 1) * 100, 1)

    # Rule 2: length check
    if 'length_min' in task and 'length_max' in task:
        len_lo = int(task['length_min'])
        len_hi = int(task['length_max'])
        if not (len_lo <= n <= len_hi):
            issues.append(f"Length {n} outside range {len_lo}–{len_hi}")
    elif task.get('length') is not None:
        target_len = task['length']
        if abs(n - target_len) > 2:
            issues.append(f"Length {n} is outside tolerance (target {target_len} ± 2)")

    # Rule 3: net charge check
    if 'charge_min' in task and 'charge_max' in task:
        c_lo = float(task['charge_min'])
        c_hi = float(task['charge_max'])
        if not (c_lo <= net_charge <= c_hi):
            if net_charge < c_lo:
                issues.append(
                    f"Net charge {net_charge:+.0f} too low "
                    f"(target {c_lo:+.0f} to {c_hi:+.0f}): add K or R residues."
                )
            else:
                issues.append(
                    f"Net charge {net_charge:+.0f} too high "
                    f"(target {c_lo:+.0f} to {c_hi:+.0f}): add D/E or replace K/R."
                )
    else:
        target_charge = task.get('charge') if 'charge' in task else task.get('ref_net_charge')
        if target_charge is not None:
            target_charge = float(target_charge)
            if abs(net_charge - target_charge) >= 2:
                diff = int(net_charge - target_charge)
                if diff > 0:
                    issues.append(
                        f"Net charge {net_charge:+.0f} too high "
                        f"(target {target_charge:+.0f} ± 1): "
                        f"replace {diff} K/R with neutral AA or add D/E."
                    )
                else:
                    issues.append(
                        f"Net charge {net_charge:+.0f} too low "
                        f"(target {target_charge:+.0f} ± 1): "
                        f"replace {abs(diff)} neutral AA with K or R."
                    )

    # Rule 4: hydrophobicity check
    if 'hydro_min' in task and 'hydro_max' in task:
        h_lo = float(task['hydro_min'])
        h_hi = float(task['hydro_max'])
        if not (h_lo <= hydro_pct <= h_hi):
            direction = ('add L/I/V/F/W/M/A' if hydro_pct < h_lo
                         else 'replace L/I/V/F/W with S/T/N/Q/G')
            issues.append(
                f"Hydrophobicity {hydro_pct:.1f}% outside range "
                f"{h_lo:.0f}%–{h_hi:.0f}%: {direction}."
            )
    elif task.get('ref_hydrophobic_pct') is not None:
        target_hydro_pct = float(task['ref_hydrophobic_pct'])
        if abs(hydro_pct - target_hydro_pct) > 15:
            issues.append(
                f"Hydrophobicity {hydro_pct:.1f}% outside target "
                f"{target_hydro_pct:.1f}% ± 15%: "
                f"{'add L/I/V/F/W/M/A' if hydro_pct < target_hydro_pct else 'replace L/I/V/F/W with S/T/N/Q/G'}."
            )
    else:
        target_hydro = task.get('hydrophobicity')
        if target_hydro is not None:
            if abs(avg_hydro - target_hydro) > 0.5:
                issues.append(
                    f"Avg hydrophobicity {avg_hydro:.3f} is outside tolerance "
                    f"(target {target_hydro:.3f} ± 0.5)"
                )

    # Rule 5: no more than 3 consecutive prolines
    import re
    if re.search(r'P{4,}', seq_clean):
        issues.append("More than 3 consecutive prolines (structural implausibility)")

    # Rule 6: odd cysteine count
    cys_count = seq_clean.count('C')
    if cys_count % 2 != 0 and cys_count > 0:
        issues.append(f"Odd cysteine count ({cys_count}) — potential disulphide pairing issue")

    # Rule 7: activity-specific length/charge constraints
    activities = task.get('activities', [])
    for act in activities:
        constraints = ACTIVITY_CONSTRAINTS.get(act)
        if not constraints:
            continue
        min_l = constraints.get('min_len', 0)
        max_l = constraints.get('max_len', 9999)
        if not (min_l <= n <= max_l):
            issues.append(
                f"Activity '{act}' requires length {min_l}–{max_l} aa (got {n})"
            )
        if 'min_charge' in constraints and net_charge < constraints['min_charge']:
            issues.append(
                f"Activity '{act}' requires charge ≥ {constraints['min_charge']} "
                f"(got {net_charge:+.0f})"
            )
        if 'max_charge' in constraints and net_charge > constraints['max_charge']:
            issues.append(
                f"Activity '{act}' requires charge ≤ {constraints['max_charge']} "
                f"(got {net_charge:+.0f})"
            )

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'net_charge': net_charge,
        'hydrophobicity': round(avg_hydro, 3),
        'hydro_percent': round(hydro_pct, 1),
        'length': n,
        'charge_class': _charge_class(net_charge),
        'length_class': _length_class(n),
    }


def rulebook_score(validation: dict) -> float:
    """
    Convert rulebook validation result to a 0-1 fitness score.
    Each issue deducts 0.2 from 1.0 (floor at 0.0).
    """
    n_issues = len(validation.get('issues', []))
    return round(max(0.0, 1.0 - n_issues * 0.2), 4)
