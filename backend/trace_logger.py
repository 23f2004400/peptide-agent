"""
Iteration-level trace logger for the peptide generation agent.
Writes human-readable blocks to logs/generation_trace.log.
One TraceLogger instance per agent.generate() call.
"""

from __future__ import annotations

import traceback as _traceback
from datetime import datetime
from pathlib import Path

_LOG_DIR  = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "generation_trace.log"
_SEP      = "=" * 60

# Amino acid residue weights (average Da) — display only, not used in scoring
_AA_WEIGHTS: dict[str, float] = {
    'A':  89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
    'E': 147.1, 'Q': 146.2, 'G':  75.1, 'H': 155.2, 'I': 131.2,
    'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
    'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1,
}
_AROMATIC = frozenset('FWYH')
_EXTRACT_RE_SRC = r'[ACDEFGHIKLMNPQRSTVWY]{5,}'


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _charge_sign(c: float | int) -> str:
    n = int(round(float(c)))
    return f"+{n}" if n >= 0 else str(n)


def _mol_weight(seq: str) -> float:
    """Approximate average MW of the peptide chain (Da)."""
    if not seq:
        return 0.0
    total = sum(_AA_WEIGHTS.get(aa, 111.1) for aa in seq)
    return round(total - (len(seq) - 1) * 18.02, 1)


def _aromaticity(seq: str) -> float:
    """Percentage of aromatic residues (F/W/Y/H)."""
    if not seq:
        return 0.0
    return round(sum(1 for aa in seq if aa in _AROMATIC) / len(seq) * 100, 1)


class TraceLogger:
    """
    Append-mode human-readable trace writer.
    One instance per agent.generate() call.
    """

    def __init__(self, task: dict) -> None:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._task            = task
        self._run_ts          = _now()
        self._pending_prompt: str = ""
        self._pending_raw:    str = ""
        # (attempt_num, score_or_None) for every scored iteration
        self._score_history: list[tuple[int, float | None]] = []

    # ── Internal ──────────────────────────────────────────────────────────────

    def _append(self, text: str) -> None:
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(_LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).error(
                "TraceLogger._append failed: %s", exc, exc_info=True
            )

    # ── Caching (called before the LLM call) ──────────────────────────────────

    def log_prompt(self, attempt_num: int, prompt: str) -> None:
        """Cache prompt so failure blocks can include it without extra args."""
        self._pending_prompt = prompt

    def log_raw_response(self, attempt_num: int, raw: str) -> None:
        """Cache raw LLM response so failure blocks can include it."""
        self._pending_raw = raw

    # ── Failure blocks ────────────────────────────────────────────────────────

    def log_llm_error(self, attempt_num: int, exc: Exception) -> None:
        tb = _traceback.format_exc()
        lines = [
            _SEP,
            f"[{_now()}]  Iteration {attempt_num} — LLM ERROR",
            _SEP,
            "",
            "PROMPT SENT TO MODEL",
            self._pending_prompt or "(none cached)",
            "",
            f"EXCEPTION: {type(exc).__name__}: {exc}",
            "",
            "FULL TRACEBACK:",
            tb.strip(),
            "",
            _SEP, "",
        ]
        self._append("\n".join(lines))

    def log_empty_response(self, attempt_num: int) -> None:
        lines = [
            _SEP,
            f"[{_now()}]  Iteration {attempt_num} — EMPTY RESPONSE",
            _SEP,
            "",
            "Cause: Model returned 0 tokens after all internal retries (vLLM chat_template bug).",
            "",
            "PROMPT SENT TO MODEL",
            self._pending_prompt or "(none cached)",
            "",
            _SEP, "",
        ]
        self._append("\n".join(lines))

    def log_extraction_failure(self, attempt_num: int, raw: str) -> None:
        preview = raw[:400].replace("\n", "\\n")
        lines = [
            _SEP,
            f"[{_now()}]  Iteration {attempt_num} — EXTRACTION FAILURE",
            _SEP,
            "",
            "Cause: No valid amino-acid sequence (5+ chars, ACDEFGHIKLMNPQRSTVWY only) found.",
            f"Regex used: {_EXTRACT_RE_SRC}",
            "",
            "PROMPT SENT TO MODEL",
            self._pending_prompt or "(none cached)",
            "",
            f"RAW MODEL RESPONSE ({len(raw)} chars):",
            preview,
            "",
            _SEP, "",
        ]
        self._append("\n".join(lines))

    # ── Full iteration block ──────────────────────────────────────────────────

    def log_iteration(
        self,
        *,
        attempt_num: int,
        task: dict,
        seq: str,
        validation: dict,
        rb_score: float,
        comp: dict,
        score: float | None,
        passed: bool,
        fb: dict,
        effective_reference: str,
        reference_used: str,
        threshold: float,
        next_prompt: str = "",
        mode: str = "",
        weakest: str = "",
        ngram_floor: float = 0.35,
        blosum_floor: float = 0.35,
    ) -> None:
        ts     = _now()
        prompt = self._pending_prompt
        raw    = self._pending_raw

        # Task targets — support both frontend keys and eval-dataset keys
        tgt_len    = task.get("length", "?")
        tgt_charge = task.get("charge") if "charge" in task else task.get("ref_net_charge", 0)
        tgt_hydro  = (
            task.get("hydrophobicity", 0.0)
            if "hydrophobicity" in task
            else task.get("ref_hydrophobic_pct", 0.0)
        )
        activities = task.get("activities", [])

        # Actual measured properties
        act_len   = validation.get("length", len(seq))
        act_charge = validation.get("net_charge", 0)
        act_hydro  = validation.get("hydrophobicity", 0.0)
        hydro_pct  = validation.get("hydro_percent", 0.0)
        chg_class  = validation.get("charge_class", "")
        len_class  = validation.get("length_class", "")
        mol_wt     = _mol_weight(seq)
        arom_pct   = _aromaticity(seq)

        # Diffs
        diff_len    = (act_len - tgt_len) if isinstance(tgt_len, int) else None
        diff_charge = int(round(float(act_charge) - float(tgt_charge)))
        diff_hydro  = round(float(act_hydro) - float(tgt_hydro), 4)

        def cs(key: str) -> str:
            return f"{comp[key]:.4f}" if key in comp else "N/A"

        issues   = validation.get("issues", [])
        n_issues = len(issues)

        # Score delta vs previous scored iteration
        prev_entry = next(
            ((n, s) for n, s in reversed(self._score_history) if s is not None),
            None,
        )
        self._score_history.append((attempt_num, score))

        # Pass/fail reason
        if passed:
            reason = (
                f"rulebook valid=True, score={score:.4f} >= threshold={threshold:.2f}"
                if score is not None
                else "rulebook valid=True (no reference score required)"
            )
        else:
            parts = []
            if not validation.get("valid"):
                parts.append(f"rulebook valid=False ({n_issues} issue(s))")
            if score is not None and score < threshold:
                parts.append(f"score={score:.4f} < threshold={threshold:.2f}")
            if effective_reference:
                cur_ngram = comp.get("ngram_bleu", 1.0)
                cur_blosum = comp.get("blosum", 1.0)
                if cur_ngram < ngram_floor:
                    parts.append(f"ngram_bleu={cur_ngram:.4f} < floor={ngram_floor:.2f}")
                if cur_blosum < blosum_floor:
                    parts.append(f"blosum={cur_blosum:.4f} < floor={blosum_floor:.2f}")
            reason = "; ".join(parts) or "unknown"

        fb_issues     = fb.get("issues", [])
        fb_fix_hints  = fb.get("fix_hints", [])

        lines: list[str] = [
            _SEP,
            f"[{ts}]  ITERATION {attempt_num}",
            _SEP,
            "",
            "INPUT SPECIFICATION",
            f"  Length:         {tgt_len}",
            f"  Charge:         {_charge_sign(tgt_charge)}",
            f"  Hydrophobicity: {tgt_hydro:.2f}",
            f"  Activity Flags: {activities}",
            f"  Reference:      {effective_reference or '(none)'}  [{reference_used}]",
            f"  Mode:           {mode or '(unspecified)'}" + (f"  (targeting: {weakest})" if weakest else ""),
            "",
            "PROMPT SENT TO MODEL",
            prompt or "(none cached)",
            "",
            f"RAW MODEL RESPONSE ({len(raw)} chars):",
            (raw[:500] if raw else "(empty)"),
            "",
            f"EXTRACTED SEQUENCE: {seq}",
            "",
            "ACTUAL PROPERTIES OF GENERATED SEQUENCE",
            f"  Length:         {act_len}",
            f"  Charge:         {_charge_sign(act_charge)}",
            f"  Hydrophobicity: {act_hydro:.3f}  (hydrophobic%: {hydro_pct:.1f}%)",
            f"  Mol. Weight:    {mol_wt:.1f} Da",
            f"  Aromaticity:    {arom_pct:.1f}%",
            f"  Charge Class:   {chg_class}",
            f"  Length Class:   {len_class}",
            "",
            "DIFFERENCES (Target vs Actual)",
            (
                f"  Length:         target={tgt_len}, actual={act_len}, diff={diff_len:+d}"
                if diff_len is not None
                else f"  Length:         target={tgt_len}, actual={act_len}, diff=?"
            ),
            f"  Charge:         target={_charge_sign(tgt_charge)}, actual={_charge_sign(act_charge)}, diff={diff_charge:+d}",
            f"  Hydrophobicity: target={tgt_hydro:.2f}, actual={act_hydro:.3f}, diff={diff_hydro:+.4f}",
            "",
            "COMPONENT SCORES",
            f"  N-gram BLEU:         {cs('ngram_bleu')}",
            f"  Charge Similarity:   {cs('charge')}",
            f"  Hydrophobicity Sim:  {cs('hydrophobicity')}",
            f"  Functional Group:    {cs('functional_group')}",
            f"  Property Dist.:      {cs('property_distribution')}",
            f"  Structural:          {cs('structural')}",
            f"  BLOSUM62:            {cs('blosum')}",
            "",
        ]

        if score is not None:
            lines.append(
                f"TOTAL SCORE (PeptideBLEU): {score:.4f}"
                f"  [ref: {effective_reference or 'none'}, source: {reference_used}]"
            )
            if prev_entry is not None:
                prev_n, prev_s = prev_entry
                delta = score - prev_s
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                lines.append(
                    f"  Score Delta vs Iteration {prev_n}: "
                    f"{prev_s:.4f} → {score:.4f}  {arrow} {delta:+.4f}"
                )
        else:
            lines.append("TOTAL SCORE (PeptideBLEU): N/A  (no reference available)")

        lines += [
            f"RULEBOOK FITNESS: {rb_score:.4f}",
            "",
            f"OVERALL RESULT: {'PASS ✓' if passed else 'FAIL ✗'}",
            f"  Reason: {reason}",
            "",
        ]

        if n_issues > 0:
            lines.append(f"RULEBOOK ISSUES ({n_issues}):")
            for i, issue in enumerate(issues, 1):
                lines.append(f"  [{i}] {issue}")
        else:
            lines.append("RULEBOOK ISSUES (0): none")

        lines += ["", "FEEDBACK GENERATED FOR NEXT ITERATION"]

        if fb_issues:
            lines.append("  Issues:")
            for issue in fb_issues:
                lines.append(f"    - {issue}")
        else:
            lines.append("  Issues: none")

        if fb_fix_hints:
            lines.append("  Fix Hints:")
            for hint in fb_fix_hints:
                lines.append(f"    - {hint}")
        else:
            lines.append("  Fix Hints: none")

        lines += [""]

        if next_prompt:
            lines += [
                "EXACT RETRY PROMPT FOR NEXT ITERATION",
                next_prompt,
            ]
        else:
            lines.append(
                "EXACT RETRY PROMPT FOR NEXT ITERATION: "
                "(last iteration or early-pass — no retry)"
            )

        lines += ["", _SEP, ""]
        self._append("\n".join(lines))

    # ── Final summary ──────────────────────────────────────────────────────────

    def log_final_summary(
        self,
        best_sequence: str,
        best_score: float | None,
        best_rb_score: float,
        best_iteration: int,
        total_iterations: int,
        elapsed: float,
    ) -> None:
        ts = _now()
        history = self._score_history

        lines = [
            _SEP,
            f"[{ts}]  FINAL SUMMARY",
            _SEP,
            "",
            f"Best Sequence:    {best_sequence or '(none generated)'}",
        ]

        if best_score is not None:
            lines.append(f"Best Score:       {best_score:.4f}")
        else:
            lines.append("Best Score:       N/A")

        lines += [
            f"Best Rulebook:    {best_rb_score:.4f}",
            f"Best Iteration:   {best_iteration}",
            f"Total Iterations: {total_iterations}",
            f"Time Elapsed:     {elapsed:.2f}s",
            "",
            "SCORE PROGRESSION",
        ]

        for n, s in history:
            if s is not None:
                lines.append(f"  Iteration {n}: {s:.4f}")
            else:
                lines.append(f"  Iteration {n}: N/A  (LLM error / empty / extraction failure)")

        # Delta table between consecutive scored iterations
        scored = [(n, s) for n, s in history if s is not None]
        if len(scored) >= 2:
            lines += ["", "SCORE DELTAS"]
            for i in range(1, len(scored)):
                prev_n, prev_s = scored[i - 1]
                cur_n,  cur_s  = scored[i]
                delta = cur_s - prev_s
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                lines.append(
                    f"  Iteration {prev_n} → {cur_n}: "
                    f"{prev_s:.4f} → {cur_s:.4f}  {arrow} {delta:+.4f}"
                )

            # Largest single improvement
            deltas = [
                (scored[i][0], scored[i][1] - scored[i - 1][1])
                for i in range(1, len(scored))
            ]
            best_gain = max(deltas, key=lambda x: x[1])
            worst_drop = min(deltas, key=lambda x: x[1])

            lines += [
                "",
                "TOP IMPROVEMENTS OBSERVED",
                f"  Largest gain:  Iteration {best_gain[0]}  ({best_gain[1]:+.4f})",
            ]
            if worst_drop[1] < 0:
                lines.append(
                    f"  Largest drop:  Iteration {worst_drop[0]}  ({worst_drop[1]:+.4f})"
                )

        lines += ["", _SEP, "", ""]
        self._append("\n".join(lines))
