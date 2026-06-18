"""
Core peptide generation agent loop: generate → validate → score → retry.
"""

from __future__ import annotations
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from . import models
from .rulebook import validate_sequence, rulebook_score
from .peptide_bleu import peptide_metric, score_components, ACTIVITY_WEIGHTS
from .prompt_builder import build_prompt, get_system_prompt, build_feedback_entry

logger = logging.getLogger(__name__)

# Map frontend activity names to PeptideBLEU preset names
ACTIVITY_PRESET_MAP = {
    'anti-bacterial': 'amp',
    'anti-fungal': 'amp',
    'anti-viral': 'amp',
    'anti-cancer': 'amp',
    'drug-delivery': 'cpp',
    'signal-peptide': 'signal',
    'immunological': 'immunological',
}

# Valid single-letter amino acid set
VALID_AAS_SET = set("ACDEFGHIKLMNPQRSTVWY")

# String used in the STRICT RULES section of the prompt. Any contiguous
# substring of this that the model echoes back would be a false positive.
CANONICAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

VALID_AA_RE = re.compile(r'[ACDEFGHIKLMNPQRSTVWY]{5,}')

# ── Reference peptide pools ──────────────────────────────────────────────────
# Multiple validated references per preset. Stored as (seq, charge, avg_kd_hydro).
# The agent picks the one with charge and length closest to the task target.

_CPP_REFS = [
    ("RQIKIWFQNRRMKWKK", +7, -0.9),    # Penetratin (16 AA) — moderate hydro
    ("KLLKLLLKLWKK",     +4, +1.1),    # TP10 variant (12 AA) — amphipathic
    ("RKKRRQRRR",        +8, -4.3),    # HIV-TAT (9 AA) — highly cationic
    ("GRKKRRQRRRPQ",    +8, -2.2),    # HIV-TAT ext (12 AA)
    ("RQRRNRRTRRNRRRVR", +12, -4.0),  # Protamine-like (17 AA)
    ("KALAKALAKALA",    +3, +0.8),    # Amphipathic helical (12 AA)
]
_AMP_REFS = [
    ("GIGKFLKKAKKFGKAFVKILKK", +6, +0.3),  # Magainin-2 (22 AA)
    ("KLLKLLKLWKK",            +5, +0.9),  # Short AMP (11 AA)
    ("FLGALFKALSHLL",          +2, +1.6),  # Helical AMP (13 AA)
]
_SIGNAL_REFS = [
    ("MSVPTQVLGLLLLWLTDARC", 0, +1.2),   # IgK signal (20 AA)
    ("MKFLILLFNILCLFPVFAHP", 0, +1.5),   # Generic signal (20 AA)
]
_IMMUNO_REFS = [
    ("SIINFEKL", -1, +0.5),              # OVA257-264 MHC-I epitope (8 AA)
    ("GILGFVFTL", 0, +1.8),              # Influenza epitope (9 AA)
]

_PRESET_REF_POOLS = {
    'cpp':          _CPP_REFS,
    'amp':          _AMP_REFS,
    'signal':       _SIGNAL_REFS,
    'immunological': _IMMUNO_REFS,
}


def _pick_reference(preset: str, target_length: int, target_charge: float) -> str:
    """
    Pick the reference whose (length, charge) is closest to the target.
    Falls back to the first entry if the preset is unknown.
    """
    pool = _PRESET_REF_POOLS.get(preset)
    if not pool:
        return ""
    # Score each reference by Euclidean distance in (length, charge) space.
    def dist(entry):
        seq, charge, _ = entry
        return (len(seq) - target_length) ** 2 + (charge - target_charge) ** 2
    return min(pool, key=dist)[0]


@dataclass
class AttemptLog:
    n: int
    sequence: str
    score: Optional[float]
    issues: list[str]
    passed: bool
    rulebook: dict = field(default_factory=dict)
    components: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    sequence: str
    score: float            # PeptideBLEU (vs reference)
    rulebook_score: float   # Physicochemical fitness (no reference needed)
    components: dict
    rulebook: dict
    iterations: int
    trace: list[dict]
    time_seconds: float
    reference_used: str = ''


class PeptideAgent:
    def __init__(
        self,
        gateway_url: str | None = None,
        api_key: str | None = None,
        threshold: float = 0.35,
        max_retries: int = 6,
    ):
        import os
        if gateway_url:
            os.environ['GATEWAY_URL'] = gateway_url
        if api_key:
            os.environ['API_KEY'] = api_key
        self.threshold = threshold
        self.max_retries = max_retries

    def _resolve_activity_preset(self, activities: list[str]) -> Optional[str]:
        for act in activities:
            preset = ACTIVITY_PRESET_MAP.get(act)
            if preset:
                return preset
        return None

    def extract_sequence(self, raw: str, target_length: int | None = None) -> str:
        """
        Extract the best valid AA sequence from raw LLM output.

        Priority order:
        1. A line that consists ENTIRELY of valid AA chars (the ideal model output)
        2. Dash-separated AA sequences like K-L-L-K-L (model formatting artefact)
        3. Longest contiguous AA match from the original text
        """
        if not raw or not raw.strip():
            return ""

        upper = raw.strip().upper()

        # Priority 1: a line that IS the sequence (only AA chars, nothing else)
        for line in upper.split('\n'):
            line = line.strip().strip('"\'`*_-| ')
            if len(line) >= 5 and all(c in VALID_AAS_SET for c in line):
                if line not in CANONICAL_AA_ALPHABET:
                    logger.debug("extract: pure-line match %r", line)
                    return line

        # Priority 2: collapse dash-separated notation (A-C-D-E → ACDE)
        dash_collapsed = re.sub(
            r'(?<=[ACDEFGHIKLMNPQRSTVWY])-(?=[ACDEFGHIKLMNPQRSTVWY])',
            '', upper
        )

        # Priority 3: regex search over both original and dash-collapsed text
        candidates = []
        for text in (upper, dash_collapsed):
            for m in VALID_AA_RE.findall(text):
                # Filter: discard any match that is a substring of the canonical
                # alphabet string (these are the model echoing the instruction).
                if m in CANONICAL_AA_ALPHABET:
                    continue
                candidates.append(m)

        if not candidates:
            return ""

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique = [c for c in candidates if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]

        logger.debug("extract: %d candidate(s): %s", len(unique), unique)

        if len(unique) == 1:
            return unique[0]

        # Prefer the candidate closest to target length
        if target_length:
            return min(unique, key=lambda m: abs(len(m) - target_length))
        return max(unique, key=len)

    def generate(self, task: dict, on_attempt=None) -> "AgentResult":
        """
        Run the agentic loop and return the best result.

        task keys: length, charge, hydrophobicity, activities, reference (optional),
                   max_retries (optional), threshold (optional)
        on_attempt: optional callback(AttemptLog) for SSE streaming
        """
        t0 = time.time()
        max_retries = task.get('max_retries', self.max_retries)
        threshold   = task.get('threshold', self.threshold)
        activities  = task.get('activities', [])
        reference   = task.get('reference', '') or ''
        target_length = task.get('length', 12)
        target_charge = task.get('charge', 0)
        activity_preset = self._resolve_activity_preset(activities)

        # Choose reference: user-supplied > activity-specific best-fit > none
        if reference:
            effective_reference = reference
            reference_used = 'user'
        elif activity_preset:
            effective_reference = _pick_reference(
                activity_preset, target_length, target_charge
            )
            reference_used = f'default:{activity_preset}'
            logger.info(
                "No reference supplied; using %r (preset=%s)",
                effective_reference, activity_preset,
            )
        else:
            effective_reference = ''
            reference_used = 'none'

        feedback_history: list[dict] = []
        best_result: Optional[AgentResult] = None
        trace: list[dict] = []

        for attempt_num in range(1, max_retries + 1):
            prompt = build_prompt(task, feedback_history)
            system = get_system_prompt()

            logger.debug("[A%d] prompt (first 200 chars): %s", attempt_num, prompt[:200])

            try:
                raw = models.generate(prompt, system=system)
            except Exception as exc:
                logger.warning("[A%d] LLM call raised: %s", attempt_num, exc)
                log = AttemptLog(
                    n=attempt_num, sequence='', score=None,
                    issues=[f"LLM error: {exc}"], passed=False,
                )
                trace.append(_log_to_dict(log))
                if on_attempt:
                    on_attempt(log)
                feedback_history.append({'seq': '', 'score': None, 'issues': log.issues, 'fix_hints': []})
                continue

            logger.debug("[A%d] raw output (%d chars): %r", attempt_num, len(raw), raw[:120])

            if not raw:
                logger.warning("[A%d] empty response from model (all retries exhausted)", attempt_num)
                log = AttemptLog(
                    n=attempt_num, sequence='', score=None,
                    issues=["Empty response from model — server chat_template bug"], passed=False,
                )
                trace.append(_log_to_dict(log))
                if on_attempt:
                    on_attempt(log)
                feedback_history.append({'seq': '', 'score': None, 'issues': log.issues, 'fix_hints': []})
                continue

            seq = self.extract_sequence(raw, target_length)
            if not seq:
                logger.warning("[A%d] extraction failed. Raw: %r", attempt_num, raw[:80])
                log = AttemptLog(
                    n=attempt_num, sequence='', score=None,
                    issues=["Could not extract valid AA sequence from LLM output"],
                    passed=False,
                )
                trace.append(_log_to_dict(log))
                if on_attempt:
                    on_attempt(log)
                feedback_history.append({'seq': raw[:50], 'score': None, 'issues': log.issues, 'fix_hints': []})
                continue

            logger.debug("[A%d] extracted: %r (len=%d)", attempt_num, seq, len(seq))

            validation = validate_sequence(seq, task)
            rb_score = rulebook_score(validation)

            if effective_reference:
                comp  = score_components(effective_reference, seq)
                score = peptide_metric(effective_reference, seq, activity=activity_preset)
            else:
                comp  = {}
                score = None

            issues = validation['issues']
            passed = validation['valid'] and (score is None or score >= threshold)

            logger.info(
                "[A%d] seq=%s len=%d charge=%+d hydro=%.2f rb=%.2f bleu=%s passed=%s",
                attempt_num, seq, validation['length'], validation['net_charge'],
                validation['hydrophobicity'], rb_score,
                f"{score:.4f}" if score is not None else "N/A", passed,
            )
            if issues:
                logger.debug("[A%d] issues: %s", attempt_num, "; ".join(issues))

            log = AttemptLog(
                n=attempt_num, sequence=seq, score=score,
                issues=issues, passed=passed,
                rulebook=validation, components=comp,
            )
            trace.append(_log_to_dict(log))
            if on_attempt:
                on_attempt(log)

            # Track best result: highest PeptideBLEU first, tie-break on rulebook score
            cur_bleu = score if score is not None else -1.0
            best_bleu = best_result.score if best_result is not None else -1.0
            if (best_result is None
                    or cur_bleu > best_bleu
                    or (cur_bleu == best_bleu and rb_score > best_result.rulebook_score)):
                best_result = AgentResult(
                    sequence=seq,
                    score=score if score is not None else 0.0,
                    rulebook_score=rb_score,
                    components=comp,
                    rulebook=validation,
                    iterations=attempt_num,
                    trace=trace[:],
                    time_seconds=round(time.time() - t0, 2),
                    reference_used=reference_used,
                )

            if passed:
                logger.info("[A%d] threshold met — stopping early", attempt_num)
                break

            # Build rich feedback for the next attempt
            fb = build_feedback_entry(seq, score, validation, task)
            # Add physics feedback even when rulebook passes but score is low
            if validation['valid'] and score is not None and score < threshold:
                net_c  = validation['net_charge']
                avg_h  = validation['hydrophobicity']
                tgt_c  = task.get('charge')
                tgt_h  = task.get('hydrophobicity')
                if tgt_c is not None and abs(net_c - tgt_c) > 1:
                    # fix_hints already set by build_feedback_entry
                    pass
                if tgt_h is not None and abs(avg_h - tgt_h) > 0.3:
                    pass
            feedback_history.append(fb)

        elapsed = round(time.time() - t0, 2)

        if best_result is None:
            last = trace[-1] if trace else {}
            last_val = last.get('rulebook', {})
            best_result = AgentResult(
                sequence=last.get('sequence', ''),
                score=last.get('score') or 0.0,
                rulebook_score=rulebook_score(last_val) if last_val else 0.0,
                components=last.get('components', {}),
                rulebook=last_val,
                iterations=len(trace),
                trace=trace,
                time_seconds=elapsed,
                reference_used=reference_used,
            )

        best_result.time_seconds = elapsed
        best_result.trace = trace
        return best_result


def _log_to_dict(log: AttemptLog) -> dict:
    return {
        'n': log.n,
        'sequence': log.sequence,
        'score': log.score,
        'issues': log.issues,
        'passed': log.passed,
        'rulebook': log.rulebook,
        'components': log.components,
    }
