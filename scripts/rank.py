#!/usr/bin/env python3
"""
Stage 6 — Final candidate ranker (single entry point).

Combines rule-based features, semantic similarity, behavioral modifier, and
honeypot demotion into one score; outputs top-100 CSV.

Usage (ranking step — must complete in ≤5 min, CPU, no network):
    python scripts/rank.py --candidates data/candidates.jsonl --out submission.csv

Requires precomputed embeddings (Stage 4, offline, no time budget):
    python scripts/embeddings.py precompute \\
        --candidates data/candidates.jsonl --out data/embeddings
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

# Allow imports when run as `python scripts/rank.py`
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from behavioral import compute_behavioral_multiplier
from embeddings import build_id_to_row_index, cosine_similarity_to_job, load_embeddings
from features import compute_features
from honeypot_check import check_honeypot
from reasoning import generate_reasoning

# =============================================================================
# SCORING WEIGHTS — explicit and documented for interview defense.
# =============================================================================
# These weights apply to the QUALIFICATION sub-score only (before multipliers).
# They sum to 1.0. Behavioral and honeypot penalties apply afterward.

QUALIFICATION_WEIGHTS: dict[str, float] = {
    # 32% — Primary anti-keyword-stuffing gate. HR Manager with 13 AI skills
    # still scores ~0.05 here; no weight redistribution can rescue that.
    "title_relevance": 0.32,
    # 28% — JD core requirement: production ranking/search/retrieval in career
    # text (not skill tags). Independent of title so "Software Engineer" with
    # real ranking-system history can still score high.
    "production_ml_experience": 0.28,
    # 12% — Supplementary semantic layer. Stage 4 test showed embeddings ALONE
    # rank Project Managers highest — kept deliberately low so rules dominate.
    "semantic_similarity": 0.12,
    # 10% — JD soft band 5–9 years.
    "experience_fit": 0.10,
    # 10% — Pune/Noida/Tier-1 India preferred; abroad without relocate penalized.
    "location_fit": 0.10,
    # 8% — Notice period logistics (also computed in features.py).
    "notice_period_fit": 0.08,
}

# Semantic cosine similarity on MiniLM vectors typically falls in ~0.35–0.70
# for this dataset. Linear rescale to [0, 1] so it is comparable to other features.
SEMANTIC_SIM_FLOOR = 0.35
SEMANTIC_SIM_CEIL = 0.70

# When current_title is deny-tier (HR, Civil Engineer, etc.), crush the final score.
# Career-history MAX title can still be deny; semantic similarity must not rescue
# keyword-stuffed summaries that embed well against the JD (the dataset trap).
DENY_CURRENT_TITLE_MULTIPLIER = 0.10
# Cap production_ml from career text for deny-tier currents — generic "production"
# language in unrelated roles should not partially compensate.
DENY_CURRENT_TITLE_PROD_ML_CAP = 0.10

DEFAULT_EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"

def normalize_semantic_similarity(raw_cosine: float) -> float:
    """Map raw cosine similarity to [0, 1] using dataset-observed range."""
    if SEMANTIC_SIM_CEIL <= SEMANTIC_SIM_FLOOR:
        return raw_cosine
    norm = (raw_cosine - SEMANTIC_SIM_FLOOR) / (SEMANTIC_SIM_CEIL - SEMANTIC_SIM_FLOOR)
    return float(max(0.0, min(1.0, norm)))


def compute_final_score(
    feats: dict,
    behavioral: dict,
    honeypot_demotion: float,
    semantic_norm: float,
) -> float:
    """
    Combine all signals into one float score.

    Pipeline:
      1. Weighted sum of qualification features + small education bonus
      2. × company_type_penalty (consulting-only careers)
      3. × disqualifier_penalty (JD hard disqualifiers)
      4. × honeypot_demotion_multiplier (impossible profiles)
      5. × behavioral_multiplier (availability nudge, narrow 0.73–0.98 band)
    """
    components = {
        "title_relevance": feats["title_relevance"],
        "production_ml_experience": feats["production_ml_experience"],
        "semantic_similarity": semantic_norm,
        "experience_fit": feats["experience_fit"],
        "location_fit": feats["location_fit"],
        "notice_period_fit": feats["notice_period_fit"],
    }

    # Deny-tier CURRENT title (features.py title_current_tier == "deny"):
    # Block semantic + inflated career-text signals — same rule for app.py and rank.py.
    if feats.get("title_current_tier") == "deny":
        components["semantic_similarity"] = 0.0
        components["production_ml_experience"] = min(
            components["production_ml_experience"],
            DENY_CURRENT_TITLE_PROD_ML_CAP,
        )

    base = sum(components[k] * QUALIFICATION_WEIGHTS[k] for k in QUALIFICATION_WEIGHTS)
    base += feats.get("education_tier_bonus", 0.0)

    score = base
    score *= feats.get("company_type_penalty", 1.0)
    score *= feats.get("disqualifier_penalty", 1.0)
    score *= honeypot_demotion
    score *= behavioral.get("behavioral_multiplier", 1.0)

    if feats.get("title_current_tier") == "deny":
        score *= DENY_CURRENT_TITLE_MULTIPLIER

    return float(score)


def score_one_candidate(
    candidate: dict,
    semantic_raw: float,
) -> dict:
    """Score a single candidate dict; used by rank.py and the Streamlit demo."""
    semantic_norm = normalize_semantic_similarity(semantic_raw)
    feats = compute_features(candidate)
    behavioral = compute_behavioral_multiplier(candidate)
    honeypot = check_honeypot(candidate).to_dict()

    final = compute_final_score(
        feats,
        behavioral,
        honeypot["honeypot_demotion_multiplier"],
        semantic_norm,
    )

    return {
        "candidate_id": candidate["candidate_id"],
        "score": final,
        "candidate": candidate,
        "feats": feats,
        "behavioral": behavioral,
        "honeypot": honeypot,
        "semantic_raw": semantic_raw,
        "semantic_norm": semantic_norm,
    }


def rank_candidate_list(
    candidates: list[dict],
    semantic_raw_by_id: dict[str, float],
    top_n: int = 100,
) -> list[dict]:
    """Score a list of candidates and return top_n sorted results."""
    scored = [
        score_one_candidate(c, semantic_raw_by_id.get(c["candidate_id"], 0.0)) for c in candidates
    ]
    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    return scored[: min(top_n, len(scored))]


def stream_candidates(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def rank_candidates(
    candidates_path: Path,
    embeddings_dir: Path,
    top_n: int = 100,
) -> list[dict]:
    """
    Score all candidates and return top_n dicts ready for CSV export.
    """
    print("Loading precomputed embeddings (no model inference)...")
    t_load = time.perf_counter()
    candidate_ids, cand_embeddings, job_embedding, emb_meta = load_embeddings(embeddings_dir)
    id_to_row = build_id_to_row_index(candidate_ids)
    print(
        f"  Loaded {emb_meta['num_candidates']:,} embeddings "
        f"({emb_meta['embedding_dim']}d) in {time.perf_counter() - t_load:.1f}s"
    )

    scored: list[dict] = []
    missing_emb = 0

    print(f"Scoring candidates from {candidates_path} ...")
    t_score = time.perf_counter()

    for i, candidate in enumerate(stream_candidates(candidates_path), start=1):
        cid = candidate["candidate_id"]

        row = id_to_row.get(cid)
        if row is None:
            missing_emb += 1
            semantic_raw = 0.0
        else:
            semantic_raw = cosine_similarity_to_job(cand_embeddings[row], job_embedding)

        scored.append(score_one_candidate(candidate, semantic_raw))

        if i % 25000 == 0:
            print(f"  {i:,} scored ...")

    print(f"  Scored {len(scored):,} in {time.perf_counter() - t_score:.1f}s")
    if missing_emb:
        print(f"  WARNING: {missing_emb} candidates missing from embedding index")

    # Sort: score descending, tie-break candidate_id ascending (per submission spec).
    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    return scored[:top_n]


def write_submission_csv(rows: list[dict], out_path: Path) -> None:
    """Write submission CSV with rank, monotonically non-increasing scores, reasoning."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        prev_display = float("inf")
        prev_cid = ""

        for rank, row in enumerate(rows, start=1):
            # Monotonically non-increasing raw display score.
            display = min(row["score"], prev_display)

            # Round to 4 decimals for CSV; fix tie-break violations caused by
            # clamping (equal rounded scores require candidate_id ascending).
            rounded = round(display, 4)
            if rank > 1:
                prev_rounded = round(prev_display, 4)
                if rounded == prev_rounded and row["candidate_id"] < prev_cid:
                    display -= 0.0001
                    rounded = round(display, 4)

            reasoning = generate_reasoning(
                candidate=row["candidate"],
                feats=row["feats"],
                behavioral=row["behavioral"],
                honeypot=row["honeypot"],
                semantic_raw=row["semantic_raw"],
                rank=rank,
            )

            writer.writerow([row["candidate_id"], rank, f"{rounded:.4f}", reasoning])
            prev_display = display
            prev_cid = row["candidate_id"]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Rank candidates and write submission CSV")
    parser.add_argument("--candidates", type=Path, default=repo / "data" / "candidates.jsonl")
    parser.add_argument("--out", type=Path, default=repo / "submission.csv")
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--top", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.candidates.exists():
        print(f"ERROR: candidates file not found: {args.candidates}", file=sys.stderr)
        return 1
    if not args.embeddings.exists():
        print(
            f"ERROR: embeddings not found at {args.embeddings}\n"
            "Run precompute first:\n"
            "  python scripts/embeddings.py precompute "
            f"--candidates {args.candidates} --out {args.embeddings}",
            file=sys.stderr,
        )
        return 1

    tracemalloc.start()
    t0 = time.perf_counter()

    print("=" * 60)
    print("Redrob ranker — qualification weights:")
    for k, w in QUALIFICATION_WEIGHTS.items():
        print(f"  {k}: {w:.0%}")
    print("=" * 60)

    top_rows = rank_candidates(args.candidates, args.embeddings, top_n=args.top)
    write_submission_csv(top_rows, args.out)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)

    print(f"\nWrote {args.out} ({len(top_rows)} rows)")
    print(f"Top 5:")
    for i, row in enumerate(top_rows[:5], start=1):
        title = row["candidate"]["profile"]["current_title"]
        print(f"  {i}. {row['candidate_id']}  score={row['score']:.4f}  {title}")

    honeypots_in_top = sum(1 for r in top_rows if r["honeypot"]["is_likely_honeypot"])
    print(f"\nHoneypots in top {args.top}: {honeypots_in_top} (must be <10 per spec)")

    print(f"\nTiming: {elapsed:.1f}s wall-clock")
    print(f"Peak memory (tracemalloc): {peak_mb:.1f} MB")

    if elapsed > 300:
        print("WARNING: exceeded 5-minute ranking budget!", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
