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
from .prompt_builder import build_prompt, get_system_prompt, build_feedback_entry, _resolve_charge
from .trace_logger import TraceLogger

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

VALID_AA_RE = re.compile(r'[ACDEFGHIKLMNPQRSTVWY]{4,}')

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
    ("ALWKTMLKKLGTMALHAGKA",   +4, +0.2),  # Dermaseptin-S1 1-20 (20 AA, charge +4, 50% hydrophobic)
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

        raw_stripped = raw.strip()
        upper = raw_stripped.upper()

        # Strip model prefixes that appear before the sequence
        _PREFIXES = (
            "SEQUENCE:", "ANSWER:", "ASSISTANT:", "OUTPUT:",
            "THE SEQUENCE IS", "GENERATED SEQUENCE:",
        )
        for pfx in _PREFIXES:
            if upper.startswith(pfx):
                cut = len(pfx)
                upper = upper[cut:].strip()
                raw_stripped = raw_stripped[cut:].strip()
                break

        # Priority 1: a line that IS the sequence (only AA chars, nothing else)
        # Also accepts lines where stripping punctuation/spaces yields a clean sequence.
        for line in upper.split('\n'):
            line = line.strip().strip('"\'`*_-| ')
            if not line:
                continue
            if len(line) >= 5 and all(c in VALID_AAS_SET for c in line):
                if line not in CANONICAL_AA_ALPHABET:
                    logger.debug("extract: pure-line match %r", line)
                    if target_length and len(line) > target_length:
                        line = line[:target_length]
                    return line
            # Also accept if stripping non-AA chars from the EDGES only yields a clean
            # sequence — handles trailing punctuation like "KLLKLL." or "KLLKLL,".
            # We strip edges only (not middle) to avoid extracting from English text.
            edge_stripped = re.sub(
                r'^[^ACDEFGHIKLMNPQRSTVWY]+|[^ACDEFGHIKLMNPQRSTVWY]+$', '', line
            )
            if (len(edge_stripped) >= 5
                    and edge_stripped not in CANONICAL_AA_ALPHABET
                    and all(c in VALID_AAS_SET for c in edge_stripped)):
                logger.debug("extract: edge-stripped match %r -> %r", line, edge_stripped)
                if target_length and len(edge_stripped) > target_length:
                    edge_stripped = edge_stripped[:target_length]
                return edge_stripped

        # Priority 2: collapse dash-separated notation (A-C-D-E → ACDE)
        dash_collapsed = re.sub(
            r'(?<=[ACDEFGHIKLMNPQRSTVWY])-(?=[ACDEFGHIKLMNPQRSTVWY])',
            '', upper
        )

        # Priority 3: regex search — skip matches whose original span was
        # lowercase (those are English words, not peptide sequences).
        candidates: list[str] = []
        seen: set[str] = set()

        for m in VALID_AA_RE.finditer(upper):
            match_str = m.group()
            if match_str in CANONICAL_AA_ALPHABET or match_str in seen:
                continue
            # If every char in the original span was already uppercase the model
            # wrote a real sequence; if any char is lowercase it is an English word
            # that happens to use only AA characters (e.g. "answer" → ANSWER).
            if not raw_stripped[m.start():m.end()].isupper():
                logger.debug("extract: skipping lowercase-origin %r", match_str)
                continue
            seen.add(match_str)
            candidates.append(match_str)

        # Dash-collapsed candidates: position mapping is ambiguous so skip the
        # origin check, but still deduplicate.
        if dash_collapsed != upper:
            for m in VALID_AA_RE.finditer(dash_collapsed):
                match_str = m.group()
                if match_str in CANONICAL_AA_ALPHABET or match_str in seen:
                    continue
                seen.add(match_str)
                candidates.append(match_str)

        if not candidates:
            # Primer-prefix fallback: scan from pos 0 collecting uppercase valid-AA chars
            # until the first non-AA or lowercase char. Catches cases like "KFLKThe..." → "KFLKT".
            prefix: list[str] = []
            for ch in raw_stripped:
                if ch.isupper() and ch in VALID_AAS_SET:
                    prefix.append(ch)
                else:
                    break
            if len(prefix) >= 4:
                result = ''.join(prefix)
                if target_length and len(result) > target_length:
                    result = result[:target_length]
                logger.debug("extract: primer-prefix fallback %r", result)
                return result
            return ""

        logger.debug("extract: %d candidate(s): %s", len(candidates), candidates)

        if len(candidates) == 1:
            best = candidates[0]
        elif target_length is not None:
            best = min(candidates, key=lambda m: abs(len(m) - target_length))
        else:
            best = max(candidates, key=len)

        # Trim to target length if model overshot (e.g. repetitive continuation)
        if target_length and len(best) > target_length:
            best = best[:target_length]

        return best

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
        target_charge = _resolve_charge(task)
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

        # Primer forces the model to continue with AA letters instead of prose.
        # The assistant turn already starts with these chars, so the model
        # cannot begin with "Explanation:" or other preambles.
        _PRIMERS = {'amp': 'KFLK', 'cpp': 'RKK', 'signal': 'MSV', 'immunological': 'GIL'}
        assistant_primer = _PRIMERS.get(activity_preset or '', 'KL')

        feedback_history: list[dict] = []
        best_passing_result: Optional[AgentResult] = None   # rulebook-valid, by score
        best_overall_result: Optional[AgentResult] = None   # any sequence, by score
        trace: list[dict] = []
        trace_log = TraceLogger(task)

        for attempt_num in range(1, max_retries + 1):
            # First attempt: use custom prompt if provided, else build from task.
            if attempt_num == 1 and task.get('prompt_override'):
                prompt = task['prompt_override']
            else:
                prompt = build_prompt(task, feedback_history)
            system = get_system_prompt()
            trace_log.log_prompt(attempt_num, prompt)

            logger.debug("[A%d] prompt (first 200 chars): %s", attempt_num, prompt[:200])

            try:
                raw = models.generate(
                    prompt, system=system,
                    assistant_primer=assistant_primer,
                    max_tokens=target_length + 5,
                )
            except Exception as exc:
                logger.warning("[A%d] LLM call raised: %s", attempt_num, exc)
                trace_log.log_llm_error(attempt_num, exc)
                log = AttemptLog(
                    n=attempt_num, sequence='', score=None,
                    issues=[f"LLM error: {exc}"], passed=False,
                )
                trace.append(_log_to_dict(log))
                if on_attempt:
                    on_attempt(log)
                feedback_history.append({'seq': '', 'score': None, 'issues': log.issues, 'fix_hints': []})
                continue

            trace_log.log_raw_response(attempt_num, raw)
            logger.debug("[A%d] raw output (%d chars): %r", attempt_num, len(raw), raw[:120])

            if not raw:
                logger.warning("[A%d] empty response from model (all retries exhausted)", attempt_num)
                trace_log.log_empty_response(attempt_num)
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
            # Trim to target length if model overshot (stop=["\n"] doesn't bound length)
            if seq and len(seq) > target_length:
                seq = seq[:target_length]
            if not seq:
                logger.warning("[A%d] extraction failed. Raw: %r", attempt_num, raw[:80])
                trace_log.log_extraction_failure(attempt_num, raw)
                log = AttemptLog(
                    n=attempt_num, sequence='', score=None,
                    issues=["Could not extract valid AA sequence from LLM output"],
                    passed=False,
                )
                trace.append(_log_to_dict(log))
                if on_attempt:
                    on_attempt(log)
                feedback_history.append({'seq': '', 'score': None, 'issues': log.issues, 'fix_hints': []})
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

            # Track best results: prefer a rulebook-valid sequence (best_passing_result)
            # but also keep best by score regardless of validity (best_overall_result).
            cur_bleu = score if score is not None else -1.0
            cur_result = AgentResult(
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
            overall_bleu = best_overall_result.score if best_overall_result is not None else -1.0
            if (best_overall_result is None
                    or cur_bleu > overall_bleu
                    or (cur_bleu == overall_bleu and rb_score > best_overall_result.rulebook_score)):
                best_overall_result = cur_result

            if validation['valid']:
                passing_bleu = best_passing_result.score if best_passing_result is not None else -1.0
                if (best_passing_result is None
                        or cur_bleu > passing_bleu
                        or (cur_bleu == passing_bleu and rb_score > best_passing_result.rulebook_score)):
                    best_passing_result = cur_result

            if passed:
                logger.info("[A%d] threshold met — stopping early", attempt_num)
                trace_log.log_iteration(
                    attempt_num=attempt_num, task=task, seq=seq,
                    validation=validation, rb_score=rb_score, comp=comp,
                    score=score, passed=passed, fb={},
                    effective_reference=effective_reference,
                    reference_used=reference_used, threshold=threshold,
                    next_prompt="",
                )
                break

            fb = build_feedback_entry(seq, score, validation, task)
            feedback_history.append(fb)

            # Build next iteration's prompt now (while feedback is fresh) so we
            # can log the exact prompt the model will see — this is read-only,
            # no side effects; the loop rebuilds it identically at the top.
            next_prompt = (
                build_prompt(task, feedback_history)
                if attempt_num < max_retries
                else ""
            )
            trace_log.log_iteration(
                attempt_num=attempt_num, task=task, seq=seq,
                validation=validation, rb_score=rb_score, comp=comp,
                score=score, passed=passed, fb=fb,
                effective_reference=effective_reference,
                reference_used=reference_used, threshold=threshold,
                next_prompt=next_prompt,
            )

        elapsed = round(time.time() - t0, 2)

        # Prefer rulebook-valid result; fall back to best-by-score overall.
        best_result = best_passing_result or best_overall_result
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

        trace_log.log_final_summary(
            best_sequence=best_result.sequence,
            best_score=best_result.score if best_result.score != 0.0 else None,
            best_rb_score=best_result.rulebook_score,
            best_iteration=best_result.iterations,
            total_iterations=len(trace),
            elapsed=elapsed,
        )

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
