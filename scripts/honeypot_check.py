#!/usr/bin/env python3
"""
Stage 3 — Honeypot detection for subtly impossible candidate profiles.

The dataset contains ~80 honeypot candidates (per submission_spec.md §7) with
internal inconsistencies such as:
  - years_of_experience that doesn't match career_history durations or dates
  - "expert" skills with 0 months of use
  - duration_months that contradict start_date/end_date
  - overlapping or impossible date ranges
  - bulk skill lists with 0 endorsements AND 0 duration (stuffing pattern)

Returns is_likely_honeypot (bool), human-readable reason(s), and a
honeypot_demotion_multiplier applied in rank.py (heavy demotion, not just a flag).

Design principle: prefer high-precision structural checks over guessing.
A single strong inconsistency OR two independent medium ones → honeypot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

# Reference "today" for open-ended roles and future-date checks.
# Must match the hackathon evaluation date (see submission environment).
REFERENCE_DATE = date(2026, 6, 17)

# ---------------------------------------------------------------------------
# Tunable thresholds — each is documented so you can defend it in interview.
# ---------------------------------------------------------------------------

# years_of_experience vs sum(career_history.duration_months):
# Flag if absolute gap exceeds this many years AND the gap is large relative
# to the smaller of the two values (catches "13.7y experience, 11 months history").
YOE_DURATION_GAP_YEARS = 2.5
YOE_DURATION_RELATIVE_GAP = 0.45  # gap must be ≥45% of min(yoe, sum_years)

# years_of_experience vs calendar span from earliest start → latest end:
YOE_SPAN_GAP_YEARS = 3.0

# For one job: |duration_months - months_from_dates| above this → inconsistent.
DURATION_DATE_TOLERANCE_MONTHS = 9

# Single job duration exceeds total years_of_experience by this many years.
JOB_LONGER_THAN_YOE_YEARS = 1.0

# Overlapping jobs: flag if two stints overlap by more than this many months.
# Small overlaps (1–3mo) can happen during job transitions; large ones don't.
OVERLAP_TOLERANCE_MONTHS = 4

# Expert proficiency with ≤ this many duration_months counts as suspicious.
EXPERT_NEAR_ZERO_MONTHS = 1
# Need at least this many expert+near-zero skills to flag (spec cites "10 skills").
EXPERT_NEAR_ZERO_COUNT = 3
# Stronger variant: mass expert stuffing.
MASS_EXPERT_NEAR_ZERO_COUNT = 5

# Skill stuffing: skills with 0 endorsements AND 0 duration_months.
SKILL_STUFFING_MIN_COUNT = 8
SKILL_STUFFING_MIN_TOTAL_SKILLS = 12
# Softer ratio-based check when list is long.
SKILL_STUFFING_RATIO = 0.70  # ≥70% of skills are 0/0

# Demotion multipliers applied in rank.py when honeypot signals fire.
# is_likely_honeypot=True gets HONEYPOT_DEMOTION_STRONG (near tier-0 relevance).
# Single medium-only suspicion without crossing honeypot bar gets lighter demotion.
HONEYPOT_DEMOTION_STRONG = 0.05
HONEYPOT_DEMOTION_SUSPICIOUS = 0.35
HONEYPOT_DEMOTION_NONE = 1.0


class SignalStrength(str, Enum):
    """How much weight one triggered check carries toward honeypot classification."""

    STRONG = "strong"    # one alone is enough for is_likely_honeypot
    MEDIUM = "medium"    # need two mediums, or one medium + corroboration
    WEAK = "weak"        # logged but does not classify alone


@dataclass
class HoneypotSignal:
    """One triggered inconsistency check."""

    code: str
    strength: SignalStrength
    reason: str


@dataclass
class HoneypotResult:
    """Full honeypot assessment for one candidate."""

    is_likely_honeypot: bool
    reasons: list[str] = field(default_factory=list)
    signals: list[HoneypotSignal] = field(default_factory=list)
    honeypot_demotion_multiplier: float = HONEYPOT_DEMOTION_NONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_likely_honeypot": self.is_likely_honeypot,
            "honeypot_reasons": self.reasons,
            "honeypot_signal_codes": [s.code for s in self.signals],
            "honeypot_demotion_multiplier": self.honeypot_demotion_multiplier,
        }


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _months_between(start: date, end: date) -> int:
    """Whole-month difference (not calendar-perfect, but consistent across checks)."""
    return (end.year - start.year) * 12 + (end.month - start.month)


def _job_end_date(entry: dict) -> date:
    """Use explicit end_date, or REFERENCE_DATE for current roles."""
    parsed = _parse_date(entry.get("end_date"))
    return parsed if parsed else REFERENCE_DATE


def check_yoe_vs_duration_sum(candidate: dict) -> HoneypotSignal | None:
    """
    Compare profile.years_of_experience to sum(career_history.duration_months).

    Real profiles can have small gaps (unlisted early jobs, rounding). Honeypots
    show huge gaps — e.g. 13.7 years claimed but only 11 months of history total.
    """
    yoe = float(candidate.get("profile", {}).get("years_of_experience", 0))
    history = candidate.get("career_history", [])
    sum_months = sum(int(e.get("duration_months", 0)) for e in history)
    if sum_months <= 0 or yoe <= 0:
        return None

    sum_years = sum_months / 12.0
    gap = abs(yoe - sum_years)
    smaller = min(yoe, sum_years)
    if gap >= YOE_DURATION_GAP_YEARS and gap >= smaller * YOE_DURATION_RELATIVE_GAP:
        return HoneypotSignal(
            code="yoe_vs_duration_sum",
            strength=SignalStrength.STRONG,
            reason=(
                f"years_of_experience ({yoe:.1f}y) inconsistent with sum of "
                f"career_history durations ({sum_years:.1f}y / {sum_months}mo)"
            ),
        )
    return None


def check_yoe_vs_career_span(candidate: dict) -> HoneypotSignal | None:
    """
    Compare years_of_experience to calendar span from earliest start → latest end.

    Catches profiles where listed jobs don't cover the claimed total tenure.
    """
    yoe = float(candidate.get("profile", {}).get("years_of_experience", 0))
    history = candidate.get("career_history", [])
    intervals: list[tuple[date, date]] = []
    for entry in history:
        start = _parse_date(entry.get("start_date"))
        if not start:
            continue
        end = _job_end_date(entry)
        intervals.append((start, end))

    if not intervals or yoe <= 0:
        return None

    earliest = min(s for s, _ in intervals)
    latest = max(e for _, e in intervals)
    span_years = _months_between(earliest, latest) / 12.0
    gap = abs(yoe - span_years)

    if gap >= YOE_SPAN_GAP_YEARS and yoe > span_years + 1.0:
        return HoneypotSignal(
            code="yoe_vs_career_span",
            strength=SignalStrength.MEDIUM,
            reason=(
                f"years_of_experience ({yoe:.1f}y) exceeds calendar span of "
                f"listed jobs ({span_years:.1f}y from {earliest} to {latest})"
            ),
        )
    return None


def check_duration_vs_dates(candidate: dict) -> HoneypotSignal | None:
    """
    For each job, compare duration_months to months implied by start/end dates.

    Example honeypot CAND_0007353: current role started 2023-09-10 but claims
    duration_months=166 (~14y) — dates imply ~33 months.
    """
    worst_gap = 0
    worst_detail = ""

    for entry in candidate.get("career_history", []):
        start = _parse_date(entry.get("start_date"))
        if not start:
            continue
        end = _job_end_date(entry)
        claimed = int(entry.get("duration_months", 0))
        if claimed <= 0:
            continue

        from_dates = max(0, _months_between(start, end))
        gap = abs(claimed - from_dates)
        if gap > worst_gap:
            worst_gap = gap
            worst_detail = (
                f"role '{entry.get('title', '')}' at {entry.get('company', '')}: "
                f"duration_months={claimed} but dates {start}→{end} imply ~{from_dates}mo"
            )

    if worst_gap >= DURATION_DATE_TOLERANCE_MONTHS:
        return HoneypotSignal(
            code="duration_vs_dates",
            strength=SignalStrength.STRONG,
            reason=worst_detail,
        )
    return None


def check_job_longer_than_total_yoe(candidate: dict) -> HoneypotSignal | None:
    """One stint longer than entire claimed career — structurally impossible."""
    yoe = float(candidate.get("profile", {}).get("years_of_experience", 0))
    if yoe <= 0:
        return None

    for entry in candidate.get("career_history", []):
        months = int(entry.get("duration_months", 0))
        if months <= 0:
            continue
        job_years = months / 12.0
        if job_years > yoe + JOB_LONGER_THAN_YOE_YEARS:
            return HoneypotSignal(
                code="job_longer_than_yoe",
                strength=SignalStrength.STRONG,
                reason=(
                    f"single role duration ({job_years:.1f}y) exceeds total "
                    f"years_of_experience ({yoe:.1f}y)"
                ),
            )
    return None


def check_impossible_dates(candidate: dict) -> list[HoneypotSignal]:
    """end before start, or start date in the future."""
    signals: list[HoneypotSignal] = []

    for entry in candidate.get("career_history", []):
        start = _parse_date(entry.get("start_date"))
        end = _parse_date(entry.get("end_date"))

        if start and end and end < start:
            signals.append(
                HoneypotSignal(
                    code="end_before_start",
                    strength=SignalStrength.STRONG,
                    reason=(
                        f"role '{entry.get('title', '')}' has end_date ({end}) "
                        f"before start_date ({start})"
                    ),
                )
            )
        if start and start > REFERENCE_DATE:
            signals.append(
                HoneypotSignal(
                    code="future_start_date",
                    strength=SignalStrength.STRONG,
                    reason=f"role '{entry.get('title', '')}' has future start_date ({start})",
                )
            )
    return signals


def check_overlapping_jobs(candidate: dict) -> HoneypotSignal | None:
    """
    Large overlapping full-time stints are implausible.

    Sorted by start_date; flag if previous job's end is more than
    OVERLAP_TOLERANCE_MONTHS after next job's start.
    """
    intervals: list[tuple[date, date, dict]] = []
    for entry in candidate.get("career_history", []):
        start = _parse_date(entry.get("start_date"))
        if not start:
            continue
        end = _job_end_date(entry)
        intervals.append((start, end, entry))

    intervals.sort(key=lambda x: x[0])
    max_overlap = 0
    overlap_detail = ""

    for i in range(len(intervals) - 1):
        _, end_a, entry_a = intervals[i]
        start_b, _, entry_b = intervals[i + 1]
        if end_a > start_b:
            overlap = _months_between(start_b, end_a)
            if overlap > max_overlap:
                max_overlap = overlap
                overlap_detail = (
                    f"roles '{entry_a.get('title')}' and '{entry_b.get('title')}' "
                    f"overlap by ~{overlap} months"
                )

    if max_overlap > OVERLAP_TOLERANCE_MONTHS:
        return HoneypotSignal(
            code="overlapping_jobs",
            strength=SignalStrength.STRONG,
            reason=overlap_detail,
        )
    return None


def check_expert_near_zero_duration(candidate: dict) -> HoneypotSignal | None:
    """
    'Expert' proficiency with 0–1 months of use — spec's classic honeypot pattern.

    Three or more such skills → medium (corroborating signal).
    Five or more → strong (mass stuffing).
    """
    skills = candidate.get("skills", [])
    expert_near_zero = [
        s["name"]
        for s in skills
        if s.get("proficiency") == "expert"
        and int(s.get("duration_months", 0)) <= EXPERT_NEAR_ZERO_MONTHS
    ]

    if len(expert_near_zero) >= MASS_EXPERT_NEAR_ZERO_COUNT:
        return HoneypotSignal(
            code="mass_expert_zero_duration",
            strength=SignalStrength.STRONG,
            reason=(
                f"{len(expert_near_zero)} 'expert' skills with ≤{EXPERT_NEAR_ZERO_MONTHS}mo "
                f"duration: {', '.join(expert_near_zero[:5])}"
                + ("..." if len(expert_near_zero) > 5 else "")
            ),
        )

    if len(expert_near_zero) >= EXPERT_NEAR_ZERO_COUNT:
        return HoneypotSignal(
            code="expert_zero_duration",
            strength=SignalStrength.MEDIUM,
            reason=(
                f"{len(expert_near_zero)} 'expert' skills with ≤{EXPERT_NEAR_ZERO_MONTHS}mo "
                f"duration: {', '.join(expert_near_zero[:4])}"
            ),
        )
    return None


def check_skill_stuffing(candidate: dict) -> HoneypotSignal | None:
    """
    Bulk skills with 0 endorsements AND 0 duration_months — keyword stuffing artifact.

    Requires both a high count AND either a long skill list or high ratio,
    so a junior with a few unendorsed skills isn't flagged.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return None

    empty_skills = [
        s["name"]
        for s in skills
        if int(s.get("endorsements", 0)) == 0 and int(s.get("duration_months", 0)) == 0
    ]
    ratio = len(empty_skills) / len(skills)

    count_ok = len(empty_skills) >= SKILL_STUFFING_MIN_COUNT and len(skills) >= SKILL_STUFFING_MIN_TOTAL_SKILLS
    ratio_ok = len(skills) >= 10 and ratio >= SKILL_STUFFING_RATIO

    if count_ok or ratio_ok:
        return HoneypotSignal(
            code="skill_stuffing",
            strength=SignalStrength.MEDIUM,
            reason=(
                f"{len(empty_skills)}/{len(skills)} skills have 0 endorsements and "
                f"0 duration_months ({ratio:.0%} of skill list)"
            ),
        )
    return None


def _classify_signals(signals: list[HoneypotSignal]) -> tuple[bool, float]:
    """
    Decide is_likely_honeypot and demotion multiplier from triggered signals.

    Rules (documented for interview):
      - Any STRONG signal alone        → honeypot, demotion ×0.05
      - Two or more MEDIUM signals     → honeypot, demotion ×0.05
      - Exactly one MEDIUM signal      → suspicious, demotion ×0.35 (not full honeypot)
      - Only WEAK signals              → no demotion
    """
    if not signals:
        return False, HONEYPOT_DEMOTION_NONE

    strong = [s for s in signals if s.strength == SignalStrength.STRONG]
    medium = [s for s in signals if s.strength == SignalStrength.MEDIUM]

    if strong:
        return True, HONEYPOT_DEMOTION_STRONG
    if len(medium) >= 2:
        return True, HONEYPOT_DEMOTION_STRONG
    if len(medium) == 1:
        return False, HONEYPOT_DEMOTION_SUSPICIOUS
    return False, HONEYPOT_DEMOTION_NONE


def check_honeypot(candidate: dict) -> HoneypotResult:
    """
    Run all honeypot checks on one candidate.

    Returns HoneypotResult with:
      - is_likely_honeypot: bool
      - reasons: list of human-readable strings (for reasoning column / debugging)
      - honeypot_demotion_multiplier: multiply into final score in rank.py
    """
    signals: list[HoneypotSignal] = []

    # Collect optional single-result checks.
    optional_checks = [
        check_yoe_vs_duration_sum,
        check_yoe_vs_career_span,
        check_duration_vs_dates,
        check_job_longer_than_total_yoe,
        check_overlapping_jobs,
        check_expert_near_zero_duration,
        check_skill_stuffing,
    ]
    for fn in optional_checks:
        hit = fn(candidate)
        if hit:
            signals.append(hit)

    # Checks that may return multiple signals.
    signals.extend(check_impossible_dates(candidate))

    is_honeypot, demotion = _classify_signals(signals)
    reasons = [s.reason for s in signals]

    return HoneypotResult(
        is_likely_honeypot=is_honeypot,
        reasons=reasons,
        signals=signals,
        honeypot_demotion_multiplier=demotion,
    )


# =============================================================================
# Dataset audit when run directly
# =============================================================================
if __name__ == "__main__":
    import argparse
    import json
    import sys
    from collections import Counter
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Audit honeypot detection on candidates.jsonl")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "candidates.jsonl",
    )
    parser.add_argument("--show", type=int, default=5, help="Examples of detected honeypots to print")
    args = parser.parse_args()

    if not args.candidates.exists():
        print(f"File not found: {args.candidates}", file=sys.stderr)
        sys.exit(1)

    total = 0
    honeypot_count = 0
    suspicious_count = 0
    code_counts: Counter = Counter()
    examples: list[tuple[str, HoneypotResult]] = []

    print(f"Auditing honeypot detection: {args.candidates}")
    print(f"Reference date: {REFERENCE_DATE}\n")

    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            total += 1
            result = check_honeypot(cand)

            for sig in result.signals:
                code_counts[sig.code] += 1

            if result.is_likely_honeypot:
                honeypot_count += 1
                if len(examples) < args.show:
                    examples.append((cand["candidate_id"], result))
            elif result.honeypot_demotion_multiplier < 1.0:
                suspicious_count += 1

    print(f"Total candidates:     {total:,}")
    print(f"Likely honeypots:     {honeypot_count:,}  ({100 * honeypot_count / total:.2f}%)")
    print(f"Suspicious (1 medium): {suspicious_count:,}  ({100 * suspicious_count / total:.2f}%)")
    print(f"Expected ~80 honeypots per submission_spec.md §7\n")

    print("Signal code frequencies:")
    for code, count in code_counts.most_common():
        print(f"  {code}: {count}")

    print(f"\n--- Example likely honeypots (up to {args.show}) ---")
    for cid, result in examples:
        print(f"\n{cid}  demotion={result.honeypot_demotion_multiplier}")
        for r in result.reasons:
            print(f"  • {r}")
