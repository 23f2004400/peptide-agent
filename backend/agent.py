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
from .prompt_builder import build_prompt, get_system_prompt, _resolve_charge
from .trace_logger import TraceLogger
from .tools.sequence_editor import deterministic_edit, build_edit_prompt

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

# Minimum acceptable N-gram/BLOSUM62 scores for an attempt to be considered
# "passed" — prevents the loop from stopping early on aggregate score alone
# while residue ordering/motif similarity to the reference is still poor.
NGRAM_FLOOR = 0.35
BLOSUM_FLOOR = 0.35

# Extra same-prompt tries when the model's completion degenerates into the
# assistant primer plus noise (garbage/foreign-script continuation) instead of
# a real sequence — catches "non-empty but useless" completions that the
# EMPTY_RETRY_ATTEMPTS check in models.py can't see (it only retries on 0 tokens).
INTERNAL_SHORT_RETRY_ATTEMPTS = 2

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
    mode: str = ""            # "generate" | "deterministic" | "llm_edit" | "edit_no_improve" | "generate_degenerate"
    weakest: str = ""         # PeptideBLEU component (or "rulebook") the edit targeted
    delta_score: float = 0.0  # winner_score - best_score before this step


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


class PepForgeAgent:
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

        Generation happens exactly once (iteration 1). Every iteration from
        then on — including immediately after that first generation — edits
        the current best-known sequence instead of generating a fresh one,
        via backend.tools.sequence_editor (deterministic fixes + an
        LLM-guided targeted edit). An edit only replaces the current best if
        it scores strictly higher, so the trajectory is monotonically
        non-decreasing by construction.

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
        _PRIMERS = {'amp': 'ALWK', 'cpp': 'RKK', 'signal': 'MSV', 'immunological': 'GIL'}
        assistant_primer = _PRIMERS.get(activity_preset or '', 'KL')

        degenerate_floor = max(5, target_length // 2)

        def score_candidate(seq: str):
            """Return (validation, rb_score, comp, score) for a candidate sequence."""
            validation = validate_sequence(seq, task)
            rb_score = rulebook_score(validation)
            if effective_reference:
                comp = score_components(effective_reference, seq)
                score = peptide_metric(effective_reference, seq, activity=activity_preset)
            else:
                comp = {}
                score = None
            return validation, rb_score, comp, score

        def is_passed(score, comp, validation) -> bool:
            ref_floor_ok = (
                not effective_reference
                or (comp.get('ngram_bleu', 1.0) >= NGRAM_FLOOR
                    and comp.get('blosum', 1.0) >= BLOSUM_FLOOR)
            )
            return validation['valid'] and (score is None or score >= threshold) and ref_floor_ok

        best_passing_result: Optional[AgentResult] = None   # rulebook-valid, by score
        best_overall_result: Optional[AgentResult] = None   # any sequence, by score
        trace: list[dict] = []
        trace_log = TraceLogger(task)

        def track_best(seq, score, rb_score, comp, validation, attempt_num) -> None:
            nonlocal best_overall_result, best_passing_result
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

        generated_once = False
        best_sequence = ''
        best_score: Optional[float] = None
        best_rb_score = 0.0
        best_components: dict = {}
        best_validation: dict = {}

        for attempt_num in range(1, max_retries + 1):
            # ================================================================
            # STEP A — generate from scratch. Runs exactly once, on the first
            # loop pass that reaches here. Internal-retry + clean-prompt
            # fallback logic is unchanged from before this feature.
            # ================================================================
            if not generated_once:
                generated_once = True

                if task.get('prompt_override'):
                    prompt = task['prompt_override']
                else:
                    prompt = build_prompt(task, [])
                system = get_system_prompt()
                trace_log.log_prompt(attempt_num, prompt)

                logger.debug("[A%d] prompt (first 200 chars): %s", attempt_num, prompt[:200])

                raw, seq = '', ''
                try:
                    for internal_try in range(1 + INTERNAL_SHORT_RETRY_ATTEMPTS):
                        raw = models.generate(
                            prompt, system=system,
                            assistant_primer=assistant_primer,
                            max_tokens=target_length + 5,
                        )
                        if not raw:
                            continue
                        seq = self.extract_sequence(raw, target_length)
                        if seq and len(seq) > target_length:
                            seq = seq[:target_length]
                        if seq and len(seq) >= degenerate_floor:
                            break
                        logger.debug(
                            "[A%d] internal retry %d/%d: degenerate extraction (len=%d, raw=%r)",
                            attempt_num, internal_try + 1, INTERNAL_SHORT_RETRY_ATTEMPTS,
                            len(seq or ''), raw[:60],
                        )
                    else:
                        # All hint-augmented tries were degenerate — fall back to the
                        # plain, hint-free prompt shape that reliably works better
                        # (hint text appears to be a destabilizing factor for this
                        # chat_template-less deployment; see CLAUDE.md known issues).
                        clean_prompt = build_prompt(task, [])
                        fallback_raw = models.generate(
                            clean_prompt, system=system,
                            assistant_primer=assistant_primer,
                            max_tokens=target_length + 5,
                        )
                        if fallback_raw:
                            fallback_seq = self.extract_sequence(fallback_raw, target_length)
                            if fallback_seq and len(fallback_seq) > target_length:
                                fallback_seq = fallback_seq[:target_length]
                            logger.debug(
                                "[A%d] clean-prompt fallback: len=%d, raw=%r",
                                attempt_num, len(fallback_seq or ''), fallback_raw[:60],
                            )
                            if fallback_seq and len(fallback_seq) >= degenerate_floor:
                                raw, seq = fallback_raw, fallback_seq
                except Exception as exc:
                    logger.warning("[A%d] LLM call raised: %s", attempt_num, exc)
                    trace_log.log_llm_error(attempt_num, exc)
                    log = AttemptLog(
                        n=attempt_num, sequence='', score=None,
                        issues=[f"LLM error: {exc}"], passed=False,
                        mode='generate_degenerate',
                    )
                    trace.append(_log_to_dict(log))
                    if on_attempt:
                        on_attempt(log)
                    continue

                trace_log.log_raw_response(attempt_num, raw)
                logger.debug("[A%d] raw output (%d chars): %r", attempt_num, len(raw), raw[:120])

                if not raw:
                    logger.warning("[A%d] empty response from model (all retries exhausted)", attempt_num)
                    trace_log.log_empty_response(attempt_num)
                    log = AttemptLog(
                        n=attempt_num, sequence='', score=None,
                        issues=["Empty response from model — server chat_template bug"],
                        passed=False, mode='generate_degenerate',
                    )
                    trace.append(_log_to_dict(log))
                    if on_attempt:
                        on_attempt(log)
                    continue

                if not seq:
                    logger.warning("[A%d] extraction failed. Raw: %r", attempt_num, raw[:80])
                    trace_log.log_extraction_failure(attempt_num, raw)
                    log = AttemptLog(
                        n=attempt_num, sequence='', score=None,
                        issues=["Could not extract valid AA sequence from LLM output"],
                        passed=False, mode='generate_degenerate',
                    )
                    trace.append(_log_to_dict(log))
                    if on_attempt:
                        on_attempt(log)
                    continue

                logger.debug("[A%d] extracted: %r (len=%d)", attempt_num, seq, len(seq))

                validation, rb_score, comp, score = score_candidate(seq)
                passed = is_passed(score, comp, validation)

                logger.info(
                    "[A%d] GENERATE seq=%s len=%d charge=%+d hydro=%.2f rb=%.2f bleu=%s passed=%s",
                    attempt_num, seq, validation['length'], validation['net_charge'],
                    validation['hydrophobicity'], rb_score,
                    f"{score:.4f}" if score is not None else "N/A", passed,
                )

                # No trace/SSE entry here on the success path — Step B (which
                # always runs right after, including this same iteration) emits
                # the iteration's one and only logged entry. Scoring and
                # best-tracking still happen silently so Step B has a seed and
                # the generate result remains eligible as the final answer.
                best_sequence, best_score = seq, score
                best_rb_score, best_components, best_validation = rb_score, comp, validation
                track_best(best_sequence, best_score, best_rb_score, best_components, best_validation, attempt_num)

                if passed:
                    # Terminal case: threshold already met before any edit — Step B
                    # never runs this iteration, so this is the only chance to log it.
                    logger.info("[A%d] threshold met — stopping early (generate step)", attempt_num)
                    log = AttemptLog(
                        n=attempt_num, sequence=seq, score=score,
                        issues=validation['issues'], passed=passed,
                        rulebook=validation, components=comp,
                        mode='generate', delta_score=round(score or rb_score, 4),
                    )
                    trace.append(_log_to_dict(log))
                    if on_attempt:
                        on_attempt(log)
                    trace_log.log_iteration(
                        attempt_num=attempt_num, task=task, seq=seq,
                        validation=validation, rb_score=rb_score, comp=comp,
                        score=score, passed=passed, fb={},
                        effective_reference=effective_reference,
                        reference_used=reference_used, threshold=threshold,
                        next_prompt="", mode='generate', weakest='',
                    )
                    break

            # ================================================================
            # STEP B — edit the current best. Runs every iteration, including
            # immediately after Step A on the very first pass. If Step A
            # never produced a usable sequence there's nothing to edit yet.
            # ================================================================
            if not best_sequence:
                continue

            candidates: list[tuple] = [
                ('original', best_sequence, best_score, best_rb_score, best_components, best_validation)
            ]

            det_seq = deterministic_edit(best_sequence, task)
            if det_seq:
                det_val, det_rb, det_comp, det_score = score_candidate(det_seq)
                candidates.append(('deterministic', det_seq, det_score, det_rb, det_comp, det_val))

            # Widened to cover the whole edit-candidate construction (prompt build,
            # model call, extraction, scoring) — not just the model call — so any
            # exception here (e.g. a bad task value) degrades to "no llm_edit
            # candidate this iteration" instead of killing the entire run.
            #
            # Rotates through up to 3 different target components (weakest, 2nd
            # weakest, 3rd weakest) with a matching creativity-level hint, instead
            # of always re-asking about the same weakest component — once nothing
            # more can be squeezed from that one angle, every later iteration was
            # coming back identical. Each rotation still gets its own
            # internal-retry-on-prose loop (this model reliably answers with
            # commentary instead of a sequence for the longer edit prompt).
            weakest = ''
            edit_raw = ''
            llm_seq = ''
            try:
                for rotation in range(3):
                    edit_system, edit_user, weakest = build_edit_prompt(
                        best_sequence, best_score, best_components, task, attempt_num,
                        rotation_offset=rotation, creativity_idx=rotation,
                    )
                    # Cache the actual edit prompt now — trace_log.log_iteration()
                    # below otherwise falls back to whatever was last cached, which
                    # would mislabel this block with a stale prompt.
                    trace_log.log_prompt(attempt_num, edit_user)

                    # Anchor the primer on the sequence actually being edited, not
                    # the activity's generation primer — priming with e.g. "ALWK"
                    # here made the model prepend that primer to (a corrupted copy
                    # of) the original sequence instead of a genuine minimal edit.
                    edit_primer = best_sequence[:4]

                    llm_seq = ''
                    for internal_try in range(1 + INTERNAL_SHORT_RETRY_ATTEMPTS):
                        edit_raw = models.generate(
                            edit_user, system=edit_system,
                            assistant_primer=edit_primer,
                            max_tokens=target_length + 5,
                        )
                        if not edit_raw:
                            continue
                        llm_seq = self.extract_sequence(edit_raw, target_length)
                        if llm_seq and len(llm_seq) > target_length:
                            llm_seq = llm_seq[:target_length]
                        if llm_seq and len(llm_seq) >= degenerate_floor and llm_seq != best_sequence:
                            break
                        logger.debug(
                            "[A%d] edit rotation %d, internal retry %d/%d: degenerate/prose response (len=%d, raw=%r)",
                            attempt_num, rotation, internal_try + 1, INTERNAL_SHORT_RETRY_ATTEMPTS,
                            len(llm_seq or ''), edit_raw[:60],
                        )
                        llm_seq = ''

                    if llm_seq:
                        break  # this rotation produced a genuinely different candidate
                    logger.debug(
                        "[A%d] edit rotation %d/3 (targeting %s) found nothing different — trying next angle",
                        attempt_num, rotation, weakest,
                    )
                trace_log.log_raw_response(attempt_num, edit_raw)

                if llm_seq:
                    llm_val, llm_rb, llm_comp, llm_score = score_candidate(llm_seq)
                    candidates.append(('llm_edit', llm_seq, llm_score, llm_rb, llm_comp, llm_val))
            except Exception as exc:
                logger.warning("[A%d] llm_edit candidate failed: %s", attempt_num, exc)
                trace_log.log_raw_response(attempt_num, edit_raw)

            def cand_key(c):
                _, _, s, rb, _, _ = c
                return (s if s is not None else -1.0, rb)

            w_mode, w_seq, w_score, w_rb, w_comp, w_val = max(candidates, key=cand_key)

            # No reference: judge improvement by rulebook_score instead of
            # PeptideBLEU (score stays None for every candidate otherwise,
            # so the comparison would never fire).
            improved = (
                ((w_score if w_score is not None else -1.0)
                 > (best_score if best_score is not None else -1.0))
                if effective_reference else (w_rb > best_rb_score)
            )
            if not improved:
                # Honest labeling instead of a blanket "no improvement": did we
                # actually find a different sequence to try this iteration
                # (deterministic and/or a rotation of llm_edit), just not one that
                # beat the current best — or did every angle come back empty?
                # best_sequence/best_score are NOT updated to the explored
                # candidate either way — the monotonic-best guarantee holds.
                w_mode = 'edit_explored' if len(candidates) > 1 else 'edit_stuck'
                w_seq, w_score, w_rb, w_comp, w_val = (
                    best_sequence, best_score, best_rb_score, best_components, best_validation
                )

            delta = (w_score or 0.0) - (best_score or 0.0)
            best_sequence, best_score = w_seq, w_score
            best_rb_score, best_components, best_validation = w_rb, w_comp, w_val

            passed = is_passed(w_score, w_comp, w_val)
            label_weakest = (
                'det' if w_mode == 'deterministic'
                else weakest if w_mode in ('llm_edit', 'edit_explored', 'edit_stuck')
                else ''
            )

            logger.info(
                "[A%d] EDIT mode=%s seq=%s bleu=%s rb=%.2f delta=%+.4f passed=%s",
                attempt_num, w_mode, best_sequence,
                f"{w_score:.4f}" if w_score is not None else "N/A", w_rb, delta, passed,
            )

            log = AttemptLog(
                n=attempt_num, sequence=best_sequence, score=best_score,
                issues=w_val.get('issues', []), passed=passed,
                rulebook=w_val, components=w_comp,
                mode=w_mode, weakest=label_weakest, delta_score=round(delta, 4),
            )
            trace.append(_log_to_dict(log))
            if on_attempt:
                on_attempt(log)

            track_best(best_sequence, best_score, best_rb_score, best_components, best_validation, attempt_num)

            trace_log.log_iteration(
                attempt_num=attempt_num, task=task, seq=best_sequence,
                validation=w_val, rb_score=best_rb_score, comp=w_comp,
                score=best_score, passed=passed, fb={},
                effective_reference=effective_reference,
                reference_used=reference_used, threshold=threshold,
                next_prompt="", mode=w_mode, weakest=weakest,
            )

            if passed:
                logger.info("[A%d] threshold met — stopping early (edit step)", attempt_num)
                break

        elapsed = round(time.time() - t0, 2)

        # Prefer rulebook-valid result; fall back to best-by-score overall.
        best_result = best_passing_result or best_overall_result
        if best_result is None:
            best_result = AgentResult(
                sequence='', score=0.0, rulebook_score=0.0,
                components={}, rulebook={}, iterations=len(trace),
                trace=trace, time_seconds=elapsed, reference_used=reference_used,
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
        'mode': log.mode,
        'weakest': log.weakest,
        'delta_score': log.delta_score,
    }
