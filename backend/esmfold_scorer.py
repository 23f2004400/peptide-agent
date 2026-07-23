"""
Computes pLDDT structural confidence via the ESM Metagenomic Atlas API
(Meta/FAIR's free, no-auth ESMFold endpoint — api.esmatlas.com).

Note: HuggingFace's Inference API no longer serves facebook/esmfold_v1
(confirmed empty inferenceProviderMapping as of 2026-07) — this endpoint
is the working replacement, same underlying ESMFold model.

Safe: always returns a dict, never raises, never affects agent logic.
Reference: Lin et al. (2023) Science 379:1123-1130
"""

from __future__ import annotations
import time
import requests

ESMFOLD_API = "https://api.esmatlas.com/foldSequence/v1/pdb/"
VALID_AA    = set("ACDEFGHIKLMNPQRSTVWY")


def _parse_plddt_from_pdb(pdb_string: str) -> list:
    """Extract per-residue pLDDT from ESMFold PDB b-factor column."""
    scores, seen = [], set()
    for line in pdb_string.split('\n'):
        if not line.startswith('ATOM'):
            continue
        try:
            res_num   = int(line[22:26].strip())
            atom_name = line[12:16].strip()
            b_factor  = float(line[60:66].strip())
            if atom_name == 'CA' and res_num not in seen:
                scores.append(b_factor)
                seen.add(res_num)
        except (ValueError, IndexError):
            continue
    if scores and max(scores) <= 1.0:
        scores = [s * 100 for s in scores]
    return scores


def get_plddt(sequence: str) -> dict:
    """
    Get ESMFold pLDDT for a peptide sequence.
    ALWAYS returns a dict. NEVER raises. NEVER affects agent logic.

    Returns:
        mean_plddt:      float 0-100 or None
        per_residue:     list of floats or []
        confidence:      "very_high"|"high"|"low"|"very_low"|"unknown"
        passes:          bool (True if mean >= 70)
        interpretation:  human-readable string
        error:           None or error description string
        pdb:             raw PDB-format structure text, or None
    """
    if not sequence:
        return _empty("Empty sequence")
    seq = sequence.upper().strip()
    if not all(c in VALID_AA for c in seq):
        return _empty("Invalid amino acid characters")
    if len(seq) > 400:
        return _empty("Sequence too long (>400 AA for this API)")

    for attempt in range(3):
        try:
            resp = requests.post(
                ESMFOLD_API,
                data=seq,
                timeout=60,
            )
            if resp.status_code in (429, 503, 504):
                # 504 (gateway timeout) is common on this free endpoint under
                # load — transient, same backoff-and-retry as 429/503.
                time.sleep(min(5 * (attempt + 1), 30))
                continue
            if resp.status_code == 200:
                scores = _parse_plddt_from_pdb(resp.text)
                if scores:
                    return _build_result(scores, resp.text)
                return _empty("Could not parse pLDDT from PDB response")
            return _empty(f"HTTP {resp.status_code}: {resp.text[:80]}")
        except requests.exceptions.Timeout:
            if attempt == 2:
                return _empty("Timeout after 3 attempts")
            time.sleep(5)
        except Exception as e:
            return _empty(str(e))

    return _empty("All retry attempts failed")


def _build_result(scores: list, pdb: str | None = None) -> dict:
    mean = sum(scores) / len(scores)
    if mean >= 90:
        conf, interp = "very_high", "Excellent — well-folded peptide"
    elif mean >= 70:
        conf, interp = "high",      "Confident — reliable backbone predicted"
    elif mean >= 50:
        conf, interp = "low",       "Low — possibly flexible or disordered"
    else:
        conf, interp = "very_low",  "Very low — likely intrinsically disordered"
    return {
        "mean_plddt":     round(mean, 2),
        "per_residue":    [round(s, 2) for s in scores],
        "confidence":     conf,
        "passes":         mean >= 70,
        "interpretation": interp,
        "error":          None,
        "pdb":            pdb,
    }


def _empty(error: str) -> dict:
    return {
        "mean_plddt":     None,
        "per_residue":    [],
        "confidence":     "unknown",
        "passes":         False,
        "interpretation": error,
        "error":          error,
        "pdb":            None,
    }


if __name__ == "__main__":
    for seq in ["KLLKLLKLLK", "GIGKFLHSAKKFGKAFVGEIMNS", "GGGGGGGG"]:
        r = get_plddt(seq)
        val = f"{r['mean_plddt']:.1f} ({r['confidence']})" \
              if r['mean_plddt'] else f"FAILED: {r['error']}"
        print(f"{seq:30s}  pLDDT: {val}")
