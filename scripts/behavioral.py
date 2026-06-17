#!/usr/bin/env python3
"""
Stage 5 — Behavioral signal modifier for Redrob candidate ranking.

Turns platform engagement signals into a single availability/engagement MULTIPLIER
in roughly [0.5, 1.0]. This multiplies the combined qualification score in rank.py.

Design principles (from docs/redrob_signals_doc.md and job_description.md):
  - A perfect-on-paper candidate who hasn't logged in for months and ignores
    recruiters is less hireable — down-weight them modestly.
  - Behavioral signals must NOT dominate qualification fit. A narrow multiplier
    band (0.5–1.0) ensures a great ML engineer at 0.5× still beats a trap
    profile at 1.0×.
  - notice_period_days is handled in features.py (logistics), not here.

Signals used (per hackathon spec):
  - last_active_date recency
  - recruiter_response_rate
  - open_to_work_flag
  - interview_completion_rate
  - applications_submitted_30d
  - github_activity_score
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

# Reference date for last_active recency — must match EDA / honeypot_check.
REFERENCE_DATE = date(2026, 6, 17)

# ---------------------------------------------------------------------------
# Multiplier output range
# ---------------------------------------------------------------------------
# Final multiplier = BEHAVIORAL_FLOOR + (1 - BEHAVIORAL_FLOOR) * weighted_avg
# With floor 0.5 and weighted_avg in [0, 1], output is always [0.5, 1.0].
BEHAVIORAL_FLOOR = 0.50

# Component weights — must sum to 1.0. Documented for interview defense.
# open_to_work and last_active are strongest "actually available" signals.
# github is smallest: JD lists open-source as nice-to-have, not required.
BEHAVIORAL_WEIGHTS: dict[str, float] = {
    "open_to_work": 0.25,
    "last_active_recency": 0.25,
    "recruiter_response_rate": 0.20,
    "interview_completion_rate": 0.15,
    "applications_submitted_30d": 0.10,
    "github_activity": 0.05,
}


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def score_last_active_recency(last_active_date: str | None) -> tuple[float, int | None]:
    """
    Score how recently the candidate logged into Redrob.

    EDA on 100K candidates (Stage 1): most activity is 31–180 days ago;
    nobody was active in the last 7 days. Buckets are wide so small date
    differences don't swing rankings wildly.

    Returns (score 0–1, days_since_active or None if unparseable).
    """
    days = None
    parsed = _parse_date(last_active_date)
    if parsed:
        days = (REFERENCE_DATE - parsed).days

    if days is None:
        return 0.55, None  # neutral if missing
    if days <= 30:
        return 1.00, days
    if days <= 90:
        return 0.85, days
    if days <= 180:
        return 0.70, days
    if days <= 365:
        return 0.55, days
    return 0.40, days


def score_recruiter_response_rate(rate: float | None) -> float:
    """
    Direct use of recruiter_response_rate with a soft floor.

    EDA: distribution is fairly flat between 11–70%; we don't want a 15%
    responder to be crushed — floor keeps worst case at 0.30 component score.
    """
    if rate is None:
        return 0.55
    rate = max(0.0, min(1.0, float(rate)))
    return max(0.30, rate)


def score_open_to_work(open_flag: bool | None) -> float:
    """
    open_to_work_flag: explicit availability marker.

    False → 0.60 (not zero): 65% of the dataset has open_to_work=False per EDA,
    and some are still hireable — this is a nudge, not a hard filter.
    True  → 1.00
    """
    if open_flag is True:
        return 1.00
    return 0.60


def score_interview_completion_rate(rate: float | None) -> float:
    """
    Fraction of scheduled interviews attended — reliability signal.

    Missing → 0.55 (neutral). Very low completion suggests flakiness.
    """
    if rate is None:
        return 0.55
    rate = max(0.0, min(1.0, float(rate)))
    # Soft floor: even 0% completion shouldn't zero out the component entirely.
    return max(0.25, rate)


def score_applications_submitted_30d(count: int | None) -> float:
    """
    Recent job-search activity on the platform.

    Interpretation:
      0 apps  → 0.50  passive / not actively applying (common, not bad)
      1–3     → 0.80  healthy active search
      4–10    → 1.00  clearly in-market (JD wants "active on platform")
      11+     → 0.85  very high volume — slight discount (spray-and-pray risk)
    """
    if count is None:
        return 0.55
    count = max(0, int(count))
    if count == 0:
        return 0.50
    if count <= 3:
        return 0.80
    if count <= 10:
        return 1.00
    return 0.85


def score_github_activity(github_score: float | None) -> float:
    """
  github_activity_score: -1 = no GitHub linked, 0–100 = activity level.

    EDA: 64.6% have no GitHub (-1). JD lists open-source as nice-to-have,
    so missing GitHub is neutral (0.55), not a penalty. Active GitHub is a
    small boost — weight is only 5% of the behavioral blend.
    """
    if github_score is None:
        return 0.55
    score = float(github_score)
    if score < 0:
        return 0.55  # no GitHub linked — neutral
    if score <= 30:
        return 0.65
    if score <= 70:
        return 0.85
    return 1.00


def compute_behavioral_multiplier(candidate: dict) -> dict[str, Any]:
    """
    Compute the behavioral engagement multiplier for one candidate.

    Returns dict with:
      behavioral_multiplier  — float in [BEHAVIORAL_FLOOR, 1.0]
      behavioral_components  — per-signal sub-scores for reasoning/debug
      days_since_last_active — for reasoning strings in Stage 7
    """
    signals = candidate.get("redrob_signals", {})

    components = {
        "open_to_work": score_open_to_work(signals.get("open_to_work_flag")),
        "last_active_recency": score_last_active_recency(signals.get("last_active_date"))[0],
        "recruiter_response_rate": score_recruiter_response_rate(signals.get("recruiter_response_rate")),
        "interview_completion_rate": score_interview_completion_rate(
            signals.get("interview_completion_rate")
        ),
        "applications_submitted_30d": score_applications_submitted_30d(
            signals.get("applications_submitted_30d")
        ),
        "github_activity": score_github_activity(signals.get("github_activity_score")),
    }

    # Weighted average of components → [0, 1]
    weighted_avg = sum(components[k] * BEHAVIORAL_WEIGHTS[k] for k in BEHAVIORAL_WEIGHTS)

    # Map to [BEHAVIORAL_FLOOR, 1.0]
    multiplier = BEHAVIORAL_FLOOR + (1.0 - BEHAVIORAL_FLOOR) * weighted_avg

    _, days_since = score_last_active_recency(signals.get("last_active_date"))

    return {
        "behavioral_multiplier": round(multiplier, 4),
        "behavioral_weighted_avg": round(weighted_avg, 4),
        "behavioral_components": {k: round(v, 4) for k, v in components.items()},
        "days_since_last_active": days_since,
        "open_to_work_flag": signals.get("open_to_work_flag"),
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "github_activity_score": signals.get("github_activity_score"),
        "applications_submitted_30d": signals.get("applications_submitted_30d"),
        "interview_completion_rate": signals.get("interview_completion_rate"),
    }


# =============================================================================
# Distribution audit when run directly
# =============================================================================
if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Audit behavioral multiplier distribution")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "candidates.jsonl",
    )
    args = parser.parse_args()

    if not args.candidates.exists():
        print(f"File not found: {args.candidates}", file=sys.stderr)
        sys.exit(1)

    multipliers: list[float] = []
    buckets = {"0.50-0.59": 0, "0.60-0.69": 0, "0.70-0.79": 0, "0.80-0.89": 0, "0.90-1.00": 0}

    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            m = compute_behavioral_multiplier(cand)["behavioral_multiplier"]
            multipliers.append(m)

    for m in multipliers:
        if m < 0.60:
            buckets["0.50-0.59"] += 1
        elif m < 0.70:
            buckets["0.60-0.69"] += 1
        elif m < 0.80:
            buckets["0.70-0.79"] += 1
        elif m < 0.90:
            buckets["0.80-0.89"] += 1
        else:
            buckets["0.90-1.00"] += 1

    n = len(multipliers)
    print(f"Behavioral multiplier audit ({n:,} candidates)")
    print(f"Reference date: {REFERENCE_DATE}")
    print(f"Output range: [{BEHAVIORAL_FLOOR}, 1.0] via weighted blend\n")
    print("Weights:")
    for k, w in BEHAVIORAL_WEIGHTS.items():
        print(f"  {k}: {w:.0%}")
    print(f"\nMin: {min(multipliers):.4f}  Max: {max(multipliers):.4f}  Mean: {sum(multipliers)/n:.4f}")
    print("\nDistribution:")
    for label, count in buckets.items():
        print(f"  {label}: {count:>6}  ({100*count/n:5.1f}%)")

    # Show spread: a great fit at 0.5× still beats a trap at 1.0× if base scores differ enough
    print(
        "\nDominance check: if qualification score is 0.80 vs 0.05 (trap),"
    )
    print(f"  0.80 × {min(multipliers):.2f} = {0.80 * min(multipliers):.4f}")
    print(f"  0.05 × {max(multipliers):.2f} = {0.05 * max(multipliers):.4f}")
    print("  → qualification fit still decides ordering between fit and trap profiles.")
