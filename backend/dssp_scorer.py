"""
Secondary structure prediction using pydssp on ESMFold PDB output.

pydssp is a pure-Python/PyTorch reimplementation of DSSP's H-bond-geometry
algorithm — no external binary needed, works identically on Windows/Linux/
Mac via `pip install pydssp`. This replaces an earlier version of this
module that shelled out to the real mkdssp/dssp CLI (see git history):
that approach required bridging a WSL-installed binary from a native
Windows Python process, and while the bridge itself was made to work
(subprocess stdin/path-escaping fixes), mkdssp itself hung indefinitely on
this machine for reasons outside this codebase (likely a stalled first-run
dependency fetch inherent to modern mkdssp/libcifpp builds). pydssp sidesteps
all of that entirely.

Important limitation vs. real DSSP: pydssp's 'c3' output is a simplified
3-state scheme — helix (H) / sheet (E) / coil (-) — with no separate Turn
category. Classic DSSP's T/S/G/I/B substates all collapse into pydssp's
H/E/coil. `turn_pct` is kept in this module's output shape for frontend/eval
compatibility, but is always 0 — whatever real DSSP would call "turn" here
just reports as coil.

For AMPs: high helix % correlates with antimicrobial/membrane activity.
For CPPs: amphipathic helix is critical for cell penetration.

Pipeline:
  ESMFold PDB string (already computed for pLDDT)
      -> pydssp.read_pdbtext() -> per-residue backbone (N,CA,C,O) coords
      -> pydssp.assign(coord, out_type='c3') -> per-residue H/E/- codes
      -> compute helix/sheet/coil percentages
      -> compare against reference structure (if available)

SAFE: Always returns a dict. Never raises. Never blocks generation. Degrades
to dssp_available: False if pydssp isn't installed, or error-populated if
assignment fails (e.g. malformed/too-short PDB) — never affects generation.
"""

from __future__ import annotations

# SS class groupings — see module docstring re: no Turn state from pydssp.
SS_HELIX = {'H'}
SS_SHEET = {'E'}
SS_TURN: set[str] = set()          # pydssp's c3 mode has no turn state
SS_COIL = {'-', 'C', ' '}          # '-' is pydssp's actual coil code


def _check_pydssp() -> bool:
    """Check if pydssp is importable."""
    try:
        import pydssp  # noqa: F401
        return True
    except ImportError:
        return False


def _run_pydssp(pdb_string: str) -> list:
    """
    Run pydssp on a PDB string. Returns a list of per-residue SS codes
    ('H'/'E'/'-'), or an empty list on failure (missing package, malformed
    PDB — e.g. non-standard backbone atom order/gaps that pydssp's own
    assertions reject — or any other error).
    """
    try:
        import pydssp
        coord = pydssp.read_pdbtext(pdb_string)
        ss_array = pydssp.assign(coord, out_type='c3')
        return list(ss_array)
    except Exception:
        return []


def _compute_ss_percentages(ss_codes: list) -> dict:
    """Compute % of residues in each SS class."""
    n = max(len(ss_codes), 1)
    n_helix = sum(1 for s in ss_codes if s in SS_HELIX)
    n_sheet = sum(1 for s in ss_codes if s in SS_SHEET)
    n_turn = sum(1 for s in ss_codes if s in SS_TURN)
    n_coil = sum(1 for s in ss_codes if s in SS_COIL)

    return {
        "helix_pct": round(100 * n_helix / n, 1),
        "sheet_pct": round(100 * n_sheet / n, 1),
        "turn_pct": round(100 * n_turn / n, 1),
        "coil_pct": round(100 * n_coil / n, 1),
        "n_helix": n_helix,
        "n_sheet": n_sheet,
        "n_turn": n_turn,
        "n_coil": n_coil,
        "n_residues": n,
        "ss_string": ''.join(str(s) for s in ss_codes),
    }


def _compare_ss(gen_pcts: dict, ref_pcts: dict) -> dict:
    """
    Compare secondary structure percentages between generated and reference
    peptide. Returns similarity scores (0-1) per SS class.
    """
    def sim(a, b):
        return round(1.0 - abs(a - b) / 100.0, 3)

    helix_sim = sim(gen_pcts["helix_pct"], ref_pcts["helix_pct"])
    sheet_sim = sim(gen_pcts["sheet_pct"], ref_pcts["sheet_pct"])
    coil_sim = sim(gen_pcts["coil_pct"], ref_pcts["coil_pct"])

    overall = round(helix_sim * 0.5 + sheet_sim * 0.25 + coil_sim * 0.25, 3)

    return {
        "helix_similarity": helix_sim,
        "sheet_similarity": sheet_sim,
        "coil_similarity": coil_sim,
        "overall_ss_similarity": overall,
    }


def _interpret_ss(pcts: dict, activities: list | None = None) -> str:
    """Human-readable interpretation of secondary structure."""
    activities = activities or []
    helix = pcts["helix_pct"]
    sheet = pcts["sheet_pct"]
    coil = pcts["coil_pct"]

    parts = []

    if helix >= 60:
        parts.append(f"highly helical ({helix:.0f}% alpha-helix)")
    elif helix >= 30:
        parts.append(f"moderately helical ({helix:.0f}% alpha-helix)")
    else:
        parts.append(f"low helix content ({helix:.0f}%)")

    if sheet >= 20:
        parts.append(f"{sheet:.0f}% beta-sheet")

    if coil >= 50:
        parts.append("largely unstructured")

    is_amp = any(a in activities for a in ('anti-bacterial', 'anti-fungal', 'anti-viral'))
    is_cpp = 'drug-delivery' in activities

    if is_amp:
        if helix >= 40:
            parts.append("good helix content for AMP membrane activity")
        else:
            parts.append("low helix — may reduce AMP membrane disruption")

    if is_cpp:
        if helix >= 30:
            parts.append("amphipathic helix present — favorable for CPP")
        else:
            parts.append("low helix — may reduce cell penetration")

    return "; ".join(parts) if parts else "mixed secondary structure"


def get_secondary_structure(
    pdb_string: str,
    reference_pdb: str | None = None,
    activities: list | None = None,
) -> dict:
    """
    Compute secondary structure from ESMFold PDB output using pydssp.

    SAFE: Always returns a dict. Never raises. Never blocks generation.

    Args:
        pdb_string:    PDB string from ESMFold for generated peptide
        reference_pdb: PDB string from ESMFold for reference peptide
                       (optional — for comparison)
        activities:    list of activity flags for interpretation

    Returns a dict with ss_string/helix_pct/sheet_pct/turn_pct/coil_pct/
    n_helix/n_sheet/interpretation/ss_similarity/dssp_available/error.
    """
    if not pdb_string:
        return _empty("No PDB string provided")

    if not _check_pydssp():
        return _empty("pydssp not installed. Run: pip install pydssp")

    try:
        ss_codes = _run_pydssp(pdb_string)
        if not ss_codes:
            return _empty("pydssp could not assign secondary structure (malformed/incompatible PDB)")

        pcts = _compute_ss_percentages(ss_codes)

        result = {
            **pcts,
            "interpretation": _interpret_ss(pcts, activities),
            "dssp_available": True,
            "error": None,
        }

        if reference_pdb:
            ref_codes = _run_pydssp(reference_pdb)
            if ref_codes:
                ref_pcts = _compute_ss_percentages(ref_codes)
                result["reference_ss"] = ref_pcts
                result["ss_similarity"] = _compare_ss(pcts, ref_pcts)
            else:
                result["reference_ss"] = None
                result["ss_similarity"] = None
        else:
            result["reference_ss"] = None
            result["ss_similarity"] = None

        return result

    except Exception as e:
        return _empty(str(e))


def _empty(error: str) -> dict:
    return {
        "ss_string": None,
        "helix_pct": None,
        "sheet_pct": None,
        "turn_pct": None,
        "coil_pct": None,
        "n_helix": None,
        "n_sheet": None,
        "n_turn": None,
        "n_coil": None,
        "n_residues": None,
        "interpretation": error,
        "dssp_available": _check_pydssp(),
        "reference_ss": None,
        "ss_similarity": None,
        "error": error,
    }


if __name__ == "__main__":
    print(f"pydssp available: {_check_pydssp()}")
    if _check_pydssp():
        import pydssp
        print(f"pydssp module: {pydssp.__file__}")
    print("Run get_secondary_structure(pdb_string) with ESMFold output.")
