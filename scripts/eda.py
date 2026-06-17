#!/usr/bin/env python3
"""
Stage 1 — Exploratory Data Analysis for Redrob hackathon candidates.

Streams data/candidates.jsonl line-by-line (never loads all 100K into memory).
Prints distributions and keyword-stuffing trap examples to stdout for review.

Usage:
    python scripts/eda.py
    python scripts/eda.py --candidates data/candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# AI-sounding skill keywords — used to detect the "keyword stuffing" trap
# described in docs/job_description.md. A high count of these on a profile
# whose title is HR/Marketing/etc. is a red flag, NOT a positive signal.
# ---------------------------------------------------------------------------
AI_SKILL_KEYWORDS = {
    "machine learning",
    "deep learning",
    "nlp",
    "natural language processing",
    "llm",
    "large language model",
    "langchain",
    "openai",
    "rag",
    "retrieval augmented generation",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "faiss",
    "elasticsearch",
    "opensearch",
    "embeddings",
    "sentence-transformers",
    "vector database",
    "vector search",
    "recommendation system",
    "ranking",
    "xgboost",
    "pytorch",
    "tensorflow",
    "transformers",
    "hugging face",
    "fine-tuning",
    "fine-tuning llms",
    "lora",
    "qlora",
    "peft",
    "bert",
    "gpt",
    "neural network",
    "computer vision",
    "generative ai",
    "prompt engineering",
    "semantic search",
    "hybrid search",
    "learning to rank",
    "ndcg",
    "information retrieval",
}

# Titles that are clearly NOT ML/AI engineering roles per the JD.
# We match case-insensitively against current_title.
IRRELEVANT_TITLE_KEYWORDS = [
    "hr manager",
    "human resources",
    "marketing manager",
    "marketing ",
    "sales executive",
    "sales manager",
    "content writer",
    "content manager",
    "accountant",
    "accounting",
    "civil engineer",
    "mechanical engineer",
    "graphic designer",
    "project manager",
    "operations manager",
    "business analyst",  # often non-ML in this dataset
    "recruiter",
    "copywriter",
    "social media",
    "customer success",
    "office manager",
    "administrative",
    "legal counsel",
    "financial analyst",
    "brand manager",
    "seo specialist",
    "ui designer",
    "ux designer",
    "product designer",
    "video editor",
    "photographer",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="EDA for Redrob candidate dataset")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=repo_root / "data" / "candidates.jsonl",
        help="Path to candidates.jsonl",
    )
    return parser.parse_args()


def is_irrelevant_title(title: str) -> bool:
    """Return True if current_title looks like a non-ML role."""
    lower = title.lower().strip()
    return any(kw in lower for kw in IRRELEVANT_TITLE_KEYWORDS)


def count_ai_skills(skills: list[dict]) -> tuple[int, list[str]]:
    """
    Count skills whose name contains an AI keyword.
    Returns (count, list of matched skill names) for display.
    """
    matched = []
    for skill in skills:
        name_lower = skill.get("name", "").lower()
        for kw in AI_SKILL_KEYWORDS:
            if kw in name_lower:
                matched.append(skill["name"])
                break
    return len(matched), matched


def days_since(iso_date_str: str, reference: date) -> int | None:
    """Days between an ISO date string and a reference date."""
    try:
        d = datetime.strptime(iso_date_str, "%Y-%m-%d").date()
        return (reference - d).days
    except (ValueError, TypeError):
        return None


def print_histogram(counter: Counter, title: str, top_n: int = 30) -> None:
    """Print a simple text histogram for a Counter."""
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)
    total = sum(counter.values())
    for key, count in counter.most_common(top_n):
        pct = 100.0 * count / total if total else 0
        bar = "#" * int(pct / 2)  # each # ≈ 2%
        print(f"  {str(key):<45} {count:>6}  ({pct:5.1f}%)  {bar}")
    if len(counter) > top_n:
        print(f"  ... and {len(counter) - top_n} more unique values")


def print_numeric_buckets(values: list[float], title: str, buckets: list[tuple[str, float, float]]) -> None:
    """
    Bucket numeric values and print counts.
    buckets: list of (label, min_inclusive, max_exclusive) — last bucket uses inf.
    """
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)
    counts = Counter()
    for v in values:
        placed = False
        for label, lo, hi in buckets:
            if lo <= v < hi:
                counts[label] += 1
                placed = True
                break
        if not placed:
            counts["(unbucketed)"] += 1
    total = len(values)
    for label, _, _ in buckets:
        c = counts[label]
        pct = 100.0 * c / total if total else 0
        print(f"  {label:<30} {c:>6}  ({pct:5.1f}%)")


def main() -> int:
    args = parse_args()
    candidates_path = args.candidates

    if not candidates_path.exists():
        print(f"ERROR: file not found: {candidates_path}", file=sys.stderr)
        return 1

    # --- accumulators (only store aggregates + a few examples, not all profiles) ---
    title_counter: Counter = Counter()
    country_counter: Counter = Counter()
    location_counter: Counter = Counter()
    years_values: list[float] = []

    # redrob_signals distributions
    last_active_days: list[int] = []
    response_rates: list[float] = []
    github_scores: list[float] = []
    open_to_work_true = 0
    open_to_work_false = 0
    notice_period_values: list[int] = []

    # keyword-stuffing trap candidates: irrelevant title + many AI skills
    stuffing_examples: list[dict] = []

    total = 0
    parse_errors = 0
    reference_date = date(2026, 6, 17)  # hackathon reference date from user_info

    print(f"Streaming: {candidates_path}")
    print(f"Reference date for recency calculations: {reference_date}")

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            total += 1
            profile = cand.get("profile", {})
            signals = cand.get("redrob_signals", {})
            skills = cand.get("skills", [])

            # --- profile fields ---
            title_counter[profile.get("current_title", "(missing)")] += 1
            country_counter[profile.get("country", "(missing)")] += 1
            location_counter[profile.get("location", "(missing)")] += 1
            yoe = profile.get("years_of_experience")
            if yoe is not None:
                years_values.append(float(yoe))

            # --- redrob_signals ---
            days = days_since(signals.get("last_active_date", ""), reference_date)
            if days is not None:
                last_active_days.append(days)

            rr = signals.get("recruiter_response_rate")
            if rr is not None:
                response_rates.append(float(rr))

            gh = signals.get("github_activity_score")
            if gh is not None:
                github_scores.append(float(gh))

            if signals.get("open_to_work_flag") is True:
                open_to_work_true += 1
            else:
                open_to_work_false += 1

            np_days = signals.get("notice_period_days")
            if np_days is not None:
                notice_period_values.append(int(np_days))

            # --- keyword stuffing detection ---
            current_title = profile.get("current_title", "")
            ai_count, ai_names = count_ai_skills(skills)
            if is_irrelevant_title(current_title) and ai_count >= 5:
                stuffing_examples.append(
                    {
                        "candidate_id": cand.get("candidate_id"),
                        "current_title": current_title,
                        "years_of_experience": profile.get("years_of_experience"),
                        "ai_skill_count": ai_count,
                        "ai_skills_sample": ai_names[:8],
                        "total_skills": len(skills),
                        "country": profile.get("country"),
                        "location": profile.get("location"),
                    }
                )

    # =========================================================================
    # PRINT FINDINGS
    # =========================================================================
    print(f"\n{'#' * 70}")
    print(f"DATASET SUMMARY")
    print(f"{'#' * 70}")
    print(f"  Total candidates parsed: {total:,}")
    if parse_errors:
        print(f"  JSON parse errors:       {parse_errors}")

    print_histogram(title_counter, "TOP 30 current_title VALUES")

    print_histogram(country_counter, "TOP 30 country VALUES", top_n=30)

    print_histogram(location_counter, "TOP 30 location VALUES", top_n=30)

    # years_of_experience distribution
    if years_values:
        print_numeric_buckets(
            years_values,
            "years_of_experience DISTRIBUTION (JD wants 5–9, soft range)",
            [
                ("0–2 years", 0, 3),
                ("3–4 years", 3, 5),
                ("5–6 years (JD sweet spot low)", 5, 7),
                ("7–9 years (JD sweet spot high)", 7, 10),
                ("10–14 years", 10, 15),
                ("15+ years", 15, 999),
            ],
        )
        avg_yoe = sum(years_values) / len(years_values)
        print(f"\n  Mean years_of_experience: {avg_yoe:.2f}")
        print(f"  Min: {min(years_values):.1f}  Max: {max(years_values):.1f}")

    # --- last_active_date recency ---
    if last_active_days:
        print_numeric_buckets(
            [float(d) for d in last_active_days],
            "last_active_date RECENCY (days since last login)",
            [
                ("0–7 days (very active)", 0, 8),
                ("8–30 days", 8, 31),
                ("31–90 days", 31, 91),
                ("91–180 days", 91, 181),
                ("181–365 days (6–12 months)", 181, 366),
                ("365+ days (inactive 1yr+)", 366, 99999),
            ],
        )

    # --- recruiter_response_rate ---
    if response_rates:
        print_numeric_buckets(
            response_rates,
            "recruiter_response_rate DISTRIBUTION",
            [
                ("0–10% (very low)", 0.0, 0.11),
                ("11–30%", 0.11, 0.31),
                ("31–50%", 0.31, 0.51),
                ("51–70%", 0.51, 0.71),
                ("71–90%", 0.71, 0.91),
                ("91–100%", 0.91, 1.01),
            ],
        )

    # --- github_activity_score ---
    if github_scores:
        no_github = sum(1 for g in github_scores if g < 0)
        has_github = len(github_scores) - no_github
        print(f"\n{'=' * 70}")
        print("github_activity_score SUMMARY")
        print("=" * 70)
        print(f"  No GitHub linked (score = -1): {no_github:,}  ({100*no_github/len(github_scores):.1f}%)")
        print(f"  Has GitHub linked:               {has_github:,}  ({100*has_github/len(github_scores):.1f}%)")
        active_github = [g for g in github_scores if g >= 0]
        if active_github:
            print_numeric_buckets(
                active_github,
                "github_activity_score (linked profiles only)",
                [
                    ("0–10 (minimal)", 0, 11),
                    ("11–30", 11, 31),
                    ("31–50", 31, 51),
                    ("51–70", 51, 71),
                    ("71–90", 71, 91),
                    ("91–100 (very active)", 91, 101),
                ],
            )

    # --- open_to_work_flag ---
    print(f"\n{'=' * 70}")
    print("open_to_work_flag DISTRIBUTION")
    print("=" * 70)
    otw_total = open_to_work_true + open_to_work_false
    print(f"  open_to_work = True:  {open_to_work_true:>6}  ({100*open_to_work_true/otw_total:.1f}%)")
    print(f"  open_to_work = False: {open_to_work_false:>6}  ({100*open_to_work_false/otw_total:.1f}%)")

    # --- notice_period_days (logistics signal from JD) ---
    if notice_period_values:
        print_numeric_buckets(
            [float(n) for n in notice_period_values],
            "notice_period_days DISTRIBUTION (JD prefers <30 days)",
            [
                ("0 days (immediate)", 0, 1),
                ("1–15 days", 1, 16),
                ("16–30 days (JD preferred max)", 16, 31),
                ("31–60 days", 31, 61),
                ("61–90 days", 61, 91),
                ("90+ days", 91, 999),
            ],
        )

    # --- KEYWORD STUFFING TRAP ---
    print(f"\n{'#' * 70}")
    print("KEYWORD-STUFFING TRAP — irrelevant titles with 5+ AI-sounding skills")
    print(f"{'#' * 70}")
    print(
        f"  Candidates matching trap pattern: {len(stuffing_examples):,} "
        f"({100*len(stuffing_examples)/total:.1f}% of dataset)"
    )
    print(
        "\n  This confirms the JD warning: many profiles list AI keywords in skills\n"
        "  while holding titles like HR Manager, Accountant, Civil Engineer, etc.\n"
        "  A naive keyword ranker would surface these — we must NOT."
    )

    # Show 8 diverse examples (sort by AI skill count descending)
    stuffing_examples.sort(key=lambda x: x["ai_skill_count"], reverse=True)
    print(f"\n  --- Top 8 examples (most AI skills, irrelevant title) ---")
    for i, ex in enumerate(stuffing_examples[:8], start=1):
        print(f"\n  Example {i}: {ex['candidate_id']}")
        print(f"    Title:      {ex['current_title']}")
        print(f"    Experience: {ex['years_of_experience']} years")
        print(f"    Location:   {ex['location']}, {ex['country']}")
        print(f"    AI skills:  {ex['ai_skill_count']} / {ex['total_skills']} total skills")
        print(f"    Sample AI skill names: {', '.join(ex['ai_skills_sample'])}")

    # --- JD-relevant title count (quick sanity check) ---
    relevant_title_keywords = [
        "ml engineer",
        "machine learning engineer",
        "ai engineer",
        "data scientist",
        "applied scientist",
        "research scientist",
        "search engineer",
        "retrieval",
        "ranking engineer",
        "nlp engineer",
        "software engineer",  # might be relevant if career history supports it
    ]
    relevant_count = sum(
        1
        for title, _ in title_counter.items()
        if any(kw in title.lower() for kw in relevant_title_keywords)
    )
    relevant_cand_count = sum(
        count
        for title, count in title_counter.items()
        if any(kw in title.lower() for kw in relevant_title_keywords)
    )
    print(f"\n{'#' * 70}")
    print("JD-RELEVANT TITLE SANITY CHECK")
    print(f"{'#' * 70}")
    print(f"  Unique title strings matching ML/AI/Search patterns: {relevant_count}")
    print(f"  Candidates with those titles: {relevant_cand_count:,} ({100*relevant_cand_count/total:.1f}%)")
    print(
        "\n  Note: even among these titles, many won't have real production ML experience.\n"
        "  Title alone is necessary but not sufficient — career_history text matters."
    )

    print(f"\n{'#' * 70}")
    print("EDA COMPLETE")
    print(f"{'#' * 70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
