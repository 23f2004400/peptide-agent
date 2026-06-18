"""
PeptideBLEU v1.2 — 7-component weighted scoring system.
"""

from __future__ import annotations
import math
from collections import Counter
from typing import Optional

# ── Amino acid property tables ──────────────────────────────────────────────

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

# Functional group classifications
POLAR_UNCHARGED = set("STNQ")
NEGATIVELY_CHARGED = set("DE")
POSITIVELY_CHARGED = set("RKH")
AROMATIC = set("FYW")
ALIPHATIC = set("AVILMG")
SPECIAL = set("CP")

# ── BLOSUM62 matrix (selected pairs; full symmetric matrix) ──────────────────

BLOSUM62_RAW = {
    ('A','A'):4,  ('A','R'):-1, ('A','N'):-2, ('A','D'):-2, ('A','C'):0,
    ('A','Q'):-1, ('A','E'):-1, ('A','G'):0,  ('A','H'):-2, ('A','I'):-1,
    ('A','L'):-1, ('A','K'):-1, ('A','M'):-1, ('A','F'):-2, ('A','P'):-1,
    ('A','S'):1,  ('A','T'):0,  ('A','W'):-3, ('A','Y'):-2, ('A','V'):0,

    ('R','R'):5,  ('R','N'):-1, ('R','D'):-2, ('R','C'):-3, ('R','Q'):1,
    ('R','E'):0,  ('R','G'):-2, ('R','H'):0,  ('R','I'):-3, ('R','L'):-2,
    ('R','K'):2,  ('R','M'):-1, ('R','F'):-3, ('R','P'):-2, ('R','S'):-1,
    ('R','T'):-1, ('R','W'):-3, ('R','Y'):-2, ('R','V'):-3,

    ('N','N'):6,  ('N','D'):1,  ('N','C'):-3, ('N','Q'):0,  ('N','E'):0,
    ('N','G'):0,  ('N','H'):1,  ('N','I'):-3, ('N','L'):-3, ('N','K'):0,
    ('N','M'):-2, ('N','F'):-3, ('N','P'):-2, ('N','S'):1,  ('N','T'):0,
    ('N','W'):-4, ('N','Y'):-2, ('N','V'):-3,

    ('D','D'):6,  ('D','C'):-3, ('D','Q'):0,  ('D','E'):2,  ('D','G'):-1,
    ('D','H'):-1, ('D','I'):-3, ('D','L'):-4, ('D','K'):-1, ('D','M'):-3,
    ('D','F'):-3, ('D','P'):-1, ('D','S'):0,  ('D','T'):-1, ('D','W'):-4,
    ('D','Y'):-3, ('D','V'):-3,

    ('C','C'):9,  ('C','Q'):-3, ('C','E'):-4, ('C','G'):-3, ('C','H'):-3,
    ('C','I'):-1, ('C','L'):-1, ('C','K'):-3, ('C','M'):-1, ('C','F'):-2,
    ('C','P'):-3, ('C','S'):-1, ('C','T'):-1, ('C','W'):-2, ('C','Y'):-2,
    ('C','V'):-1,

    ('Q','Q'):5,  ('Q','E'):2,  ('Q','G'):-2, ('Q','H'):0,  ('Q','I'):-3,
    ('Q','L'):-2, ('Q','K'):1,  ('Q','M'):0,  ('Q','F'):-3, ('Q','P'):-1,
    ('Q','S'):0,  ('Q','T'):-1, ('Q','W'):-2, ('Q','Y'):-1, ('Q','V'):-2,

    ('E','E'):5,  ('E','G'):-2, ('E','H'):0,  ('E','I'):-3, ('E','L'):-3,
    ('E','K'):1,  ('E','M'):-2, ('E','F'):-3, ('E','P'):-1, ('E','S'):0,
    ('E','T'):-1, ('E','W'):-3, ('E','Y'):-2, ('E','V'):-2,

    ('G','G'):6,  ('G','H'):-2, ('G','I'):-4, ('G','L'):-4, ('G','K'):-2,
    ('G','M'):-3, ('G','F'):-3, ('G','P'):-2, ('G','S'):0,  ('G','T'):-2,
    ('G','W'):-2, ('G','Y'):-3, ('G','V'):-3,

    ('H','H'):8,  ('H','I'):-3, ('H','L'):-3, ('H','K'):-1, ('H','M'):-2,
    ('H','F'):-1, ('H','P'):-2, ('H','S'):-1, ('H','T'):-2, ('H','W'):-2,
    ('H','Y'):2,  ('H','V'):-3,

    ('I','I'):4,  ('I','L'):2,  ('I','K'):-1, ('I','M'):1,  ('I','F'):0,
    ('I','P'):-3, ('I','S'):-2, ('I','T'):-1, ('I','W'):-3, ('I','Y'):-1,
    ('I','V'):3,

    ('L','L'):4,  ('L','K'):-2, ('L','M'):2,  ('L','F'):0,  ('L','P'):-3,
    ('L','S'):-2, ('L','T'):-1, ('L','W'):-2, ('L','Y'):-1, ('L','V'):1,

    ('K','K'):5,  ('K','M'):-1, ('K','F'):-3, ('K','P'):-1, ('K','S'):0,
    ('K','T'):-1, ('K','W'):-3, ('K','Y'):-2, ('K','V'):-2,

    ('M','M'):5,  ('M','F'):0,  ('M','P'):-2, ('M','S'):-1, ('M','T'):-1,
    ('M','W'):-1, ('M','Y'):-1, ('M','V'):1,

    ('F','F'):6,  ('F','P'):-4, ('F','S'):-2, ('F','T'):-2, ('F','W'):1,
    ('F','Y'):3,  ('F','V'):-1,

    ('P','P'):7,  ('P','S'):-1, ('P','T'):-1, ('P','W'):-4, ('P','Y'):-3,
    ('P','V'):-2,

    ('S','S'):4,  ('S','T'):1,  ('S','W'):-3, ('S','Y'):-2, ('S','V'):-2,

    ('T','T'):5,  ('T','W'):-2, ('T','Y'):-2, ('T','V'):0,

    ('W','W'):11, ('W','Y'):2,  ('W','V'):-3,

    ('Y','Y'):7,  ('Y','V'):-1,

    ('V','V'):4,
}

def _blosum62(a: str, b: str) -> int:
    pair = (a, b)
    if pair in BLOSUM62_RAW:
        return BLOSUM62_RAW[pair]
    pair = (b, a)
    if pair in BLOSUM62_RAW:
        return BLOSUM62_RAW[pair]
    return -4  # default penalty for unknown pair

# ── Activity preset weights ──────────────────────────────────────────────────

ACTIVITY_WEIGHTS = {
    'amp': dict(ngram=0.15, charge=0.25, hydro=0.20, fg=0.10, dist=0.10, struct=0.05, blosum=0.15),
    'cpp': dict(ngram=0.10, charge=0.35, hydro=0.10, fg=0.10, dist=0.10, struct=0.10, blosum=0.15),
    'signal': dict(ngram=0.10, charge=0.05, hydro=0.30, fg=0.15, dist=0.15, struct=0.10, blosum=0.15),
    'immunological': dict(ngram=0.15, charge=0.20, hydro=0.15, fg=0.15, dist=0.10, struct=0.10, blosum=0.15),
    'default': dict(ngram=0.20, charge=0.20, hydro=0.15, fg=0.10, dist=0.10, struct=0.10, blosum=0.15),
}

# ── Helper: sanitize to valid AAs ────────────────────────────────────────────

def _clean(seq: str) -> str:
    return ''.join(c for c in seq.upper() if c in VALID_AAS)

# ── C1: N-gram BLEU ──────────────────────────────────────────────────────────

def _ngram_bleu(ref: str, pred: str, max_n: int = 4) -> float:
    if not ref or not pred:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(ref[i:i+n] for i in range(len(ref) - n + 1))
        pred_ngrams = Counter(pred[i:i+n] for i in range(len(pred) - n + 1))
        if not pred_ngrams:
            scores.append(0.0)
            continue
        clipped = sum(min(cnt, ref_ngrams[ng]) for ng, cnt in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        scores.append(clipped / total if total > 0 else 0.0)

    if not any(s > 0 for s in scores):
        return 0.0

    geo_mean = math.exp(sum(math.log(s) if s > 0 else -1e9 for s in scores) / len(scores))

    # brevity penalty
    bp = 1.0 if len(pred) >= len(ref) else math.exp(1 - len(ref) / len(pred))
    return bp * geo_mean

# ── C2: Charge similarity ────────────────────────────────────────────────────

def _charge_score(ref: str, pred: str) -> float:
    ref_charge = sum(CHARGE_MAP.get(aa, 0) for aa in ref)
    pred_charge = sum(CHARGE_MAP.get(aa, 0) for aa in pred)
    diff = abs(ref_charge - pred_charge)
    return max(0.0, 1.0 - diff / max(abs(ref_charge) + 1, 1))

# ── C3: Hydrophobicity similarity ────────────────────────────────────────────

def _hydro_score(ref: str, pred: str) -> float:
    def avg_kd(seq):
        vals = [KD_HYDRO.get(aa, 0) for aa in seq]
        return sum(vals) / len(vals) if vals else 0.0

    ref_h = avg_kd(ref)
    pred_h = avg_kd(pred)
    diff = abs(ref_h - pred_h)
    scale = max(abs(ref_h), 1.0)
    return max(0.0, 1.0 - diff / scale)

# ── C4: Functional group composition ────────────────────────────────────────

_FG_GROUPS = [POLAR_UNCHARGED, NEGATIVELY_CHARGED, POSITIVELY_CHARGED,
              AROMATIC, ALIPHATIC, SPECIAL]

def _functional_group_score(ref: str, pred: str) -> float:
    def profile(seq):
        n = len(seq)
        if n == 0:
            return [0.0] * len(_FG_GROUPS)
        return [sum(1 for aa in seq if aa in g) / n for g in _FG_GROUPS]

    rp = profile(ref)
    pp = profile(pred)
    diff = sum(abs(r - p) for r, p in zip(rp, pp))
    return max(0.0, 1.0 - diff)

# ── C5: Property distribution ────────────────────────────────────────────────

def _property_distribution_score(ref: str, pred: str) -> float:
    """Compare positional distribution of hydrophobic vs hydrophilic patches."""
    def encode(seq):
        return [1 if aa in HYDROPHOBIC_AAS else 0 for aa in seq]

    def window_avg(enc, w=3):
        out = []
        for i in range(len(enc)):
            chunk = enc[max(0, i-w//2): i+w//2+1]
            out.append(sum(chunk) / len(chunk))
        return out

    r_enc = encode(ref)
    p_enc = encode(pred)

    if not r_enc or not p_enc:
        return 0.0

    # Interpolate to same length for comparison
    target_len = max(len(r_enc), len(p_enc))
    def interp(enc, length):
        if len(enc) == length:
            return enc
        result = []
        for i in range(length):
            idx = i * (len(enc) - 1) / max(length - 1, 1)
            lo, hi = int(idx), min(int(idx) + 1, len(enc) - 1)
            result.append(enc[lo] * (1 - (idx - lo)) + enc[hi] * (idx - lo))
        return result

    r_i = interp(window_avg(r_enc), target_len)
    p_i = interp(window_avg(p_enc), target_len)

    diff = sum(abs(rv - pv) for rv, pv in zip(r_i, p_i)) / target_len
    return max(0.0, 1.0 - diff)

# ── C6: Structural features ──────────────────────────────────────────────────

def _structural_score(ref: str, pred: str) -> float:
    """Compare proline%, glycine%, cysteine% as proxy structural features."""
    def struct_profile(seq):
        n = len(seq)
        if n == 0:
            return (0.0, 0.0, 0.0)
        return (
            seq.count('P') / n,
            seq.count('G') / n,
            seq.count('C') / n,
        )

    rp = struct_profile(ref)
    pp = struct_profile(pred)
    diff = sum(abs(r - p) for r, p in zip(rp, pp))
    return max(0.0, 1.0 - diff * 3)

# ── C7: BLOSUM62 n-gram similarity ──────────────────────────────────────────

def _blosum_sim(a: str, b: str) -> float:
    """Normalised BLOSUM62 similarity between two amino acids."""
    raw = _blosum62(a, b)
    denom = max(_blosum62(a, a), _blosum62(b, b))
    if denom <= 0:
        return 0.0
    return max(0.0, raw / denom)

def _blosum_ngram_score(ref: str, pred: str) -> float:
    if not ref or not pred:
        return 0.0

    def ngram_sim(n: int) -> float:
        r_grams = [ref[i:i+n] for i in range(len(ref) - n + 1)]
        p_grams = [pred[i:i+n] for i in range(len(pred) - n + 1)]
        if not r_grams or not p_grams:
            return 0.0
        total = 0.0
        for pg in p_grams:
            best = max(
                sum(_blosum_sim(pg[k], rg[k]) for k in range(n)) / n
                for rg in r_grams
            )
            total += best
        return total / len(p_grams)

    s1 = ngram_sim(1)
    s2 = ngram_sim(2) if len(ref) >= 2 and len(pred) >= 2 else s1
    s3 = ngram_sim(3) if len(ref) >= 3 and len(pred) >= 3 else s2
    avg = (s1 + s2 + s3) / 3

    lp = min(1.0, len(pred) / max(len(ref), 1))
    return avg * lp

# ── Main scoring function ────────────────────────────────────────────────────

def score_components(ref: str, pred: str) -> dict:
    ref = _clean(ref)
    pred = _clean(pred)
    return {
        'ngram_bleu': _ngram_bleu(ref, pred),
        'charge': _charge_score(ref, pred),
        'hydrophobicity': _hydro_score(ref, pred),
        'functional_group': _functional_group_score(ref, pred),
        'property_distribution': _property_distribution_score(ref, pred),
        'structural': _structural_score(ref, pred),
        'blosum': _blosum_ngram_score(ref, pred),
    }


def peptide_metric(
    ref: str,
    pred: str,
    activity: Optional[str] = None,
    weights: Optional[dict] = None,
    verbose: bool = False,
) -> float:
    comp = score_components(ref, pred)

    if weights is None:
        w = ACTIVITY_WEIGHTS.get(activity, ACTIVITY_WEIGHTS['default'])
    else:
        w = weights

    score = (
        w['ngram']  * comp['ngram_bleu'] +
        w['charge'] * comp['charge'] +
        w['hydro']  * comp['hydrophobicity'] +
        w['fg']     * comp['functional_group'] +
        w['dist']   * comp['property_distribution'] +
        w['struct'] * comp['structural'] +
        w['blosum'] * comp['blosum']
    )

    if verbose:
        print(f"Ref:  {ref}")
        print(f"Pred: {pred}")
        for k, v in comp.items():
            print(f"  {k:25s}: {v:.4f}  (w={w.get(k[:5], '?')})")
        print(f"  {'TOTAL':25s}: {score:.4f}")

    return round(score, 4)


def batch_peptide_metric(refs: list[str], preds: list[str], activity: str = None) -> dict:
    if len(refs) != len(preds):
        raise ValueError("refs and preds must have equal length")
    scores = [peptide_metric(r, p, activity=activity) for r, p in zip(refs, preds)]
    comp_sums = {}
    for r, p in zip(refs, preds):
        for k, v in score_components(r, p).items():
            comp_sums[k] = comp_sums.get(k, 0.0) + v
    n = max(len(scores), 1)
    return {
        'scores': scores,
        'mean': round(sum(scores) / n, 4),
        'components': {k: round(v / n, 4) for k, v in comp_sums.items()},
    }


def classify_sequence(seq: str) -> dict:
    seq = _clean(seq)
    charge = sum(CHARGE_MAP.get(aa, 0) for aa in seq)
    hydro = sum(KD_HYDRO.get(aa, 0) for aa in seq) / max(len(seq), 1)
    hydro_pct = sum(1 for aa in seq if aa in HYDROPHOBIC_AAS) / max(len(seq), 1)
    aro_pct = sum(1 for aa in seq if aa in AROMATIC) / max(len(seq), 1)

    activity_guess = []
    if 10 <= len(seq) <= 50 and charge >= 2 and hydro_pct >= 0.3:
        activity_guess.append('amp')
    if 10 <= len(seq) <= 50 and 3 <= charge <= 9:
        activity_guess.append('cpp')
    if 15 <= len(seq) <= 30:
        activity_guess.append('signal')

    return {
        'length': len(seq),
        'net_charge': charge,
        'avg_hydrophobicity': round(hydro, 3),
        'hydrophobic_percent': round(hydro_pct * 100, 1),
        'aromatic_percent': round(aro_pct * 100, 1),
        'predicted_activities': activity_guess,
    }


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        ("KLLKLLKLLK", "KLLKLFKLLK", None, 0.9306),
        ("RRWWKK", "RRWWDD", None, 0.4748),
        ("ACDEFGHIKLMN", "ACDEFGHIKLMN", None, 1.0000),
    ]
    print("PeptideBLEU v1.2 self-test\n" + "-" * 50)
    for ref, pred, act, expected in tests:
        got = peptide_metric(ref, pred, activity=act)
        status = "PASS" if abs(got - expected) < 0.05 else "WARN"
        print(f"[{status}] peptide_metric({ref!r}, {pred!r}) = {got:.4f}  (expected ~{expected})")
