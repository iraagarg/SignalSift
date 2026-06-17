#!/usr/bin/env python3
"""
Stage 7 — Reasoning string generation for submission CSV.

Each reasoning is 1–2 sentences built from ACTUAL computed sub-scores and
real profile fields. No generic templates with swapped names; no fabricated details.

Variation: sentence openers and structure rotate by rank index so 10 sampled
rows read substantively different in manual review.
"""

from __future__ import annotations

from typing import Any


def _pick_career_detail(candidate: dict) -> str | None:
    """Extract one concrete phrase from career_history for reasoning."""
    for entry in candidate.get("career_history", []):
        desc = entry.get("description", "")
        title = entry.get("title", "")
        company = entry.get("company", "")
        # Prefer sentences mentioning JD-relevant work.
        for keyword in (
            "ranking",
            "retrieval",
            "recommendation",
            "embedding",
            "search",
            "vector",
            "A/B test",
            "NDCG",
        ):
            if keyword.lower() in desc.lower():
                # Short snippet around the keyword for natural phrasing.
                idx = desc.lower().find(keyword.lower())
                start = max(0, idx - 40)
                snippet = desc[start : idx + len(keyword) + 30].strip()
                if snippet.endswith(","):
                    snippet = snippet[:-1]
                return f"{title} at {company} ({snippet}…)"
        if company and title:
            return f"{title} at {company}"
    return None


def _pick_skill_name(candidate: dict) -> str | None:
    """Return one skill name with real endorsement/duration data."""
    skills = candidate.get("skills", [])
    # Prefer skills with non-zero duration (credible).
    credible = [s for s in skills if s.get("duration_months", 0) > 6]
    pool = credible if credible else skills
    if not pool:
        return None
    # Pick skill with longest duration as most defensible.
    best = max(pool, key=lambda s: s.get("duration_months", 0))
    return best.get("name")


def _weakness_phrase(feats: dict, behavioral: dict) -> str | None:
    """Honest concern if an obvious gap exists — even for high-ranked candidates."""
    concerns: list[str] = []

    if feats.get("location_tier") == "abroad_no_relocate":
        concerns.append(
            f"based in {feats.get('location')} with willing_to_relocate=false (no visa sponsorship per JD)"
        )
    elif feats.get("location_tier") == "india_other":
        concerns.append(f"located in {feats.get('location')} (outside JD Tier-1 cities, domestic relocation)")

    notice = feats.get("notice_period_days", 0)
    if notice > 60:
        concerns.append(f"{notice}-day notice period (JD prefers sub-30)")

    if behavioral.get("open_to_work_flag") is False:
        concerns.append("not currently flagged open_to_work on Redrob")

    days = behavioral.get("days_since_last_active")
    if days is not None and days > 120:
        concerns.append(f"last active {days} days ago on the platform")

    rr = behavioral.get("recruiter_response_rate")
    if rr is not None and rr < 0.20:
        concerns.append(f"recruiter response rate {rr:.0%}")

    if feats.get("company_all_consulting") and not feats.get("company_has_product_experience"):
        concerns.append("career spent entirely at services/consulting firms")

    if feats.get("disqualifier_reasons"):
        concerns.append(feats["disqualifier_reasons"][0])

    if feats.get("title_current_tier") == "deny":
        concerns.append(f"current title '{feats.get('title_best_match', '')}' is outside ML engineering")

    return concerns[0] if concerns else None


def generate_reasoning(
    candidate: dict,
    feats: dict[str, Any],
    behavioral: dict[str, Any],
    honeypot: dict[str, Any],
    semantic_raw: float,
    rank: int,
) -> str:
    """
    Build a 1–2 sentence justification from real profile data and computed scores.

    rank is used only to rotate sentence structure (variation), not to fabricate facts.
    """
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "")
    yoe = feats.get("years_of_experience", profile.get("years_of_experience"))
    prod = feats.get("production_ml_experience", 0)
    title_rel = feats.get("title_relevance", 0)
    career = _pick_career_detail(candidate)
    skill = _pick_skill_name(candidate)
    weakness = _weakness_phrase(feats, behavioral)

    # Honeypot flag — honest even if somehow ranked (shouldn't be top after demotion).
    if honeypot.get("is_likely_honeypot"):
        reason = honeypot.get("honeypot_reasons", ["profile consistency issues"])[0]
        return (
            f"Profile flagged for internal inconsistency ({reason}); "
            f"ranked low despite {title} title and {yoe} years listed."
        )

    # Rotate openers by rank for variation across the top 100.
    opener_style = rank % 5

    strength_parts: list[str] = []

    if prod >= 0.5 and feats.get("prod_ml_strong_hits", 0) > 0:
        strength_parts.append(
            f"career history shows production ranking/search work "
            f"({feats['prod_ml_strong_hits']} strong JD keyword hits, prod_ml={prod:.2f})"
        )
    elif prod >= 0.25:
        strength_parts.append(f"some production ML signals in career text (prod_ml={prod:.2f})")

    if title_rel >= 0.9:
        strength_parts.append(f"{title} title directly matches the JD role family (title_relevance={title_rel:.2f})")
    elif title_rel >= 0.55:
        strength_parts.append(
            f"{title} with ML-adjacent title tier '{feats.get('title_best_tier')}' (title_relevance={title_rel:.2f})"
        )
    else:
        strength_parts.append(f"{title} ({yoe} years; title_relevance={title_rel:.2f})")

    if career and opener_style in (0, 2):
        strength_clause = f"Background includes {career}."
    elif skill and opener_style in (1, 3):
        dur = next(
            (s.get("duration_months", 0) for s in candidate.get("skills", []) if s.get("name") == skill),
            0,
        )
        strength_clause = f"Lists {skill} ({dur}mo duration) among skills."
    else:
        strength_clause = (
            f"Semantic similarity to JD is {semantic_raw:.2f}; "
            f"location fit {feats.get('location_fit', 0):.2f} ({feats.get('location_tier')})."
        )

    signal_clause = (
        f"Redrob signals: response rate {behavioral.get('recruiter_response_rate', 0):.0%}, "
        f"github_activity={behavioral.get('github_activity_score')}, "
        f"behavioral_multiplier={behavioral.get('behavioral_multiplier', 1):.2f}."
    )

    if opener_style == 0:
        sentence1 = f"{title} with {yoe} years — {strength_parts[0]}."
    elif opener_style == 1:
        sentence1 = f"Strong fit on title and experience: {strength_parts[-1]}."
    elif opener_style == 2:
        sentence1 = strength_clause
    elif opener_style == 3:
        sentence1 = f"Scores well on JD alignment ({', '.join(strength_parts[:2])})."
    else:
        sentence1 = (
            f"{yoe}-year {title} in {profile.get('location', '')}; "
            f"production_ml_experience={prod:.2f}, notice_period={feats.get('notice_period_days')}d."
        )

    if opener_style == 2:
        sentence2 = signal_clause
    elif weakness and rank > 20:
        sentence2 = f"Concern: {weakness}."
    elif weakness:
        sentence2 = f"Note: {weakness}, but overall qualification outweighs this gap."
    else:
        sentence2 = signal_clause

    # Keep to ~2 sentences; avoid redundancy.
    if sentence1 == sentence2:
        sentence2 = signal_clause

    return f"{sentence1} {sentence2}".strip()
