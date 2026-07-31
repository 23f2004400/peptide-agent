"""
Converts ESMFold PDB output to a protein graph and extracts interpretable
structural quality metrics (H-bonds, ionic/pi-pi interactions, etc.).

Supplementary to pLDDT, strictly additive: computed once on the final
sequence's PDB structure (already fetched by esmfold_scorer.get_plddt()),
never affects generation/retry logic. Always returns a dict; never raises.
"""

from __future__ import annotations
import numpy as np


# == PDB PARSING ==============================================================

def parsePdbLine(line: str) -> dict:
    return {
        'atom':                   line[0:6].strip(),
        'atom_serial':            line[6:11].strip(),
        'atom_name':              line[12:16].strip(),
        'alternate_location':     line[16].strip(),
        'residue_name':           line[17:20].strip(),
        'chain_identifier':       line[21].strip(),
        'residue_sequence_number': int(line[22:26].strip()),
        'x_coordinate':           float(line[30:38].strip()),
        'y_coordinate':           float(line[38:46].strip()),
        'z_coordinate':           float(line[46:54].strip()),
        'occupancy':              line[54:60].strip(),
        'temperature':            line[60:66].strip(),
        'element_symbol':         line[76:78].strip() if len(line) > 76 else '',
    }


class PdbResidue:
    def __init__(self):
        self.residue     = None
        self.residue_num = None
        self.atoms       = {}

    def addLine(self, line: str):
        d = parsePdbLine(line)
        if self.residue is None:
            self.residue     = d['residue_name']
            self.residue_num = d['residue_sequence_number']
        self.atoms[d['atom_name']] = np.array([
            d['x_coordinate'], d['y_coordinate'], d['z_coordinate']
        ])

    def coords(self, name: str) -> np.ndarray:
        return self.atoms[name]

    def has_atom(self, name: str) -> bool:
        return name in self.atoms


def pdb2Residues(pdb: str) -> list:
    residues     = []
    curr_res_num = -1
    res          = None

    for line in pdb.split('\n'):
        if not line.startswith('ATOM'):
            continue
        try:
            num = int(line[22:26].strip())
            if curr_res_num == -1:
                res          = PdbResidue()
                curr_res_num = num
                res.addLine(line)
            elif num == curr_res_num:
                res.addLine(line)
            else:
                if res is not None:
                    residues.append(res)
                res          = PdbResidue()
                curr_res_num = num
                res.addLine(line)
        except (ValueError, IndexError):
            continue

    if res is not None:
        residues.append(res)
    return residues


# == INTERACTION FINDERS ======================================================

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def findIonicInteraction(residues: list, threshold: float = 8.0) -> list:
    """Ionic bonds between positively and negatively charged residues."""
    positives, negatives = [], []
    for i, r in enumerate(residues):
        if r.residue == 'ARG' and r.has_atom('NH1'):
            positives.append((i, [r.coords('NH1'), r.coords('NH2'),
                                   r.coords('NE')]))
        elif r.residue == 'LYS' and r.has_atom('NZ'):
            positives.append((i, [r.coords('NZ')]))
        elif r.residue == 'HIS' and r.has_atom('ND1'):
            positives.append((i, [r.coords('ND1'), r.coords('NE2')]))
        elif r.residue == 'ASP' and r.has_atom('OD1'):
            negatives.append((i, [r.coords('OD1'), r.coords('OD2')]))
        elif r.residue == 'GLU' and r.has_atom('OE1'):
            negatives.append((i, [r.coords('OE1'), r.coords('OE2')]))

    edges = set()
    for pi, pcoords in positives:
        for ni, ncoords in negatives:
            for pc in pcoords:
                for nc in ncoords:
                    d = _dist(pc, nc)
                    if d <= threshold and d > 0:
                        edges.add((pi, ni, round(1 / d ** 2, 6)))
                        break
    return list(edges)


def findHbondingInteraction(residues: list, threshold: float = 3.5) -> list:
    """
    Find backbone H-bonds only (N...O donor-acceptor pairs).
    Uses a stricter 3.5A threshold instead of 4.0A, restricted to backbone
    N/O atoms only (not side-chain N/O like NZ, ND1, OD1, OE1, etc — the
    previous version counted every atom name starting with 'N' or 'O', wildly
    overcounting: e.g. 124 "H-bonds" for an 18-residue peptide). Also skips
    adjacent residues (i, i+1) since their backbone N/O are covalently close
    and not a genuine hydrogen bond, and caps the result at a physically
    plausible 2 bonds per residue.
    """
    backbone_n = []  # backbone N atoms only
    backbone_o = []  # backbone O atoms only

    for i, r in enumerate(residues):
        # Backbone N (not side-chain N like NZ, NH1, ND1, etc)
        if r.has_atom('N'):
            backbone_n.append((i, r.coords('N')))
        # Backbone O (not side-chain O like OD1, OE1, etc)
        if r.has_atom('O'):
            backbone_o.append((i, r.coords('O')))

    edges = set()
    for ni, nc in backbone_n:
        for oi, oc in backbone_o:
            if ni != oi and abs(ni - oi) > 1:  # skip adjacent residues
                d = _dist(nc, oc)
                if d <= threshold and d > 0:
                    edges.add((ni, oi, round(1 / d ** 2, 6)))

    # Sanity cap — physically implausible to exceed ~2 H-bonds per residue.
    max_hbonds = 2 * len(residues)
    if len(edges) > max_hbonds:
        edges = set(sorted(edges, key=lambda e: -e[2])[:max_hbonds])

    return list(edges)


def findAtomicDistanceInteraction(residues: list, threshold: float = 8.0) -> list:
    """C-alpha contacts — backbone proximity."""
    edges = set()
    for i in range(len(residues)):
        if not residues[i].has_atom('CA'):
            continue
        for j in range(i + 1, len(residues)):
            if not residues[j].has_atom('CA'):
                continue
            d = _dist(residues[i].coords('CA'), residues[j].coords('CA'))
            if d <= threshold and d > 0:
                edges.add((i, j, round(1 / d ** 2, 6)))
    return list(edges)


def findHydroInteraction(residues: list, threshold: float = 5.0) -> list:
    """Hydrophobic and polar clustering interactions."""
    POLAR     = {'SER', 'THR', 'CYS', 'TYR', 'ASN', 'GLN'}
    NONPOLAR  = {'GLY', 'ALA', 'VAL', 'LEU', 'ILE',
                 'MET', 'PHE', 'TRP', 'PRO'}
    polars    = []
    nonpolars = []

    for i, r in enumerate(residues):
        if not r.has_atom('CA'):
            continue
        if r.residue in POLAR:
            polars.append((i, r.coords('CA')))
        elif r.residue in NONPOLAR:
            nonpolars.append((i, r.coords('CA')))

    edges = set()
    for group in (polars, nonpolars):
        for ii in range(len(group)):
            for jj in range(ii + 1, len(group)):
                d = _dist(group[ii][1], group[jj][1])
                if d <= threshold and d > 0:
                    edges.add((group[ii][0], group[jj][0],
                               round(1 / d ** 2, 6)))
    return list(edges)


def findPiPiInteractions(residues: list,
                          dist_threshold: float = 7.0,
                          angle_threshold: float = 30.0) -> list:
    """Pi-pi stacking between aromatic residues."""
    RING_ATOMS = {
        'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
        'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
        'TRP': ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    }
    rings = []
    for i, r in enumerate(residues):
        if r.residue not in RING_ATOMS:
            continue
        coords = [r.coords(a) for a in RING_ATOMS[r.residue]
                  if r.has_atom(a)]
        if len(coords) < 3:
            continue
        coords  = np.array(coords)
        centroid = coords.mean(axis=0)
        v1       = coords[1] - coords[0]
        v2       = coords[2] - coords[0]
        normal   = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len > 0:
            normal = normal / norm_len
            rings.append((i, centroid, normal))

    edges = set()
    for ii in range(len(rings)):
        ri, c1, n1 = rings[ii]
        for jj in range(ii + 1, len(rings)):
            rj, c2, n2 = rings[jj]
            d = float(np.linalg.norm(c1 - c2))
            if d <= dist_threshold and d > 0:
                angle = np.degrees(np.arccos(
                    np.clip(np.dot(n1, n2), -1.0, 1.0)
                ))
                if angle <= angle_threshold or abs(angle - 90) <= angle_threshold:
                    edges.add((ri, rj, round(1 / d ** 2, 6)))
    return list(edges)


# == GRAPH FEATURE EXTRACTION =================================================

_THREE_TO_ONE = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
    'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
    'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
    'TRP':'W','TYR':'Y',
}


def pdb_to_graph_features(pdb_string: str) -> dict:
    """
    Convert ESMFold PDB string to graph and extract structural metrics.

    Returns interpretable structural quality metrics including:
    - Interaction counts (H-bonds, ionic, pi-pi, hydrophobic)
    - Density scores (per residue)
    - Interaction strengths (sum of 1/d^2 weights)
    - Composite structure score
    - Human-readable interpretation

    SAFE: Never raises. Returns dict with error key if parsing fails.
    """
    try:
        if not pdb_string:
            return _empty_graph("Empty PDB string")

        residues = pdb2Residues(pdb_string)
        if not residues:
            return _empty_graph("No residues parsed from PDB")

        n = len(residues)

        # Compute all interaction types
        ionic   = findIonicInteraction(residues)
        pi_pi   = findPiPiInteractions(residues)
        ca_dist = findAtomicDistanceInteraction(residues)
        hbonds  = findHbondingInteraction(residues)
        hydro   = findHydroInteraction(residues)

        total_edges = len(ionic) + len(pi_pi) + len(ca_dist) + \
                      len(hbonds) + len(hydro)

        # Density scores (interactions per residue)
        hbond_density   = round(len(hbonds)  / max(n, 1), 3)
        ionic_density   = round(len(ionic)   / max(n, 1), 3)
        contact_density = round(len(ca_dist) / max(n, 1), 3)
        hydro_density   = round(len(hydro)   / max(n, 1), 3)

        # Interaction strengths (weighted by 1/distance^2)
        ionic_strength  = round(sum(e[2] for e in ionic),   3)
        hbond_strength  = round(sum(e[2] for e in hbonds),  3)
        hydro_strength  = round(sum(e[2] for e in hydro),   3)
        ca_strength     = round(sum(e[2] for e in ca_dist), 3)

        # Composite structure score
        # Weighted: H-bonds most important, then ionic, pi-pi, hydrophobic
        structure_score = round(
            (len(hbonds) * 2.0 +
             len(ionic)  * 3.0 +
             len(pi_pi)  * 2.0 +
             len(hydro)  * 1.0) / max(n, 1),
            3
        )

        # Node sequence (for display)
        sequence_from_pdb = ''.join(
            _THREE_TO_ONE.get(r.residue, 'X') for r in residues
        )

        result = {
            # Counts
            "n_residues":       n,
            "n_hbonds":         len(hbonds),
            "n_ionic":          len(ionic),
            "n_pi_pi":          len(pi_pi),
            "n_ca_contacts":    len(ca_dist),
            "n_hydro":          len(hydro),
            "total_edges":      total_edges,

            # Density (per residue)
            "hbond_density":    hbond_density,
            "ionic_density":    ionic_density,
            "contact_density":  contact_density,
            "hydro_density":    hydro_density,

            # Strength (weighted)
            "ionic_strength":   ionic_strength,
            "hbond_strength":   hbond_strength,
            "hydro_strength":   hydro_strength,
            "ca_strength":      ca_strength,

            # Composite
            "structure_score":  structure_score,
            "sequence_from_pdb": sequence_from_pdb,

            # Interpretation
            "interpretation":   _interpret(
                n, len(hbonds), len(ionic), len(pi_pi), structure_score
            ),
            "error":            None,
        }
        return result

    except Exception as e:
        return _empty_graph(str(e))


def _interpret(n: int, n_hbonds: int, n_ionic: int,
               n_pi: int, score: float) -> str:
    """Human-readable structural interpretation."""
    parts = []

    if score >= 3.0:
        parts.append("highly structured peptide")
    elif score >= 1.5:
        parts.append("moderately structured peptide")
    else:
        parts.append("loosely structured / flexible peptide")

    if n_hbonds > n * 1.5:
        parts.append(f"{n_hbonds} H-bonds (strong backbone stability)")
    elif n_hbonds > n * 0.8:
        parts.append(f"{n_hbonds} H-bonds (moderate stability)")
    else:
        parts.append(f"{n_hbonds} H-bonds (limited backbone stability)")

    if n_ionic > 2:
        parts.append(
            f"{n_ionic} ionic interactions (good for membrane activity)"
        )
    if n_pi > 0:
        parts.append(f"{n_pi} aromatic stacking interactions")

    return "; ".join(parts)


def _empty_graph(error: str) -> dict:
    return {
        "n_residues":       None,
        "n_hbonds":         None,
        "n_ionic":          None,
        "n_pi_pi":          None,
        "n_ca_contacts":    None,
        "n_hydro":          None,
        "total_edges":      None,
        "hbond_density":    None,
        "ionic_density":    None,
        "contact_density":  None,
        "hydro_density":    None,
        "ionic_strength":   None,
        "hbond_strength":   None,
        "hydro_strength":   None,
        "ca_strength":      None,
        "structure_score":  None,
        "sequence_from_pdb": None,
        "interpretation":   error,
        "error":            error,
    }
