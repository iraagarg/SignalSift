#!/usr/bin/env python3
"""
Stage 2 — Rule-based feature engineering for Redrob candidate ranking.

Each function returns a float sub-score in roughly [0, 1] (or a penalty multiplier)
with explicit, documented rules so every decision can be defended in an interview.

The main entry point is compute_features(candidate) -> dict[str, float | bool | str].

Design philosophy (from docs/job_description.md):
  - title_relevance is the primary guard against keyword-stuffing traps.
  - production_ml_experience reads career_history *text*, not skill tags.
  - Generic engineering titles get a medium-low prior; career evidence can lift them.
  - Behavioral signals are NOT computed here (Stage 5).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

# =============================================================================
# TITLE RELEVANCE TIERS
# =============================================================================
# We score current_title AND every career_history title, then take the MAX.
# Rationale: someone currently titled "Software Engineer" may have been "ML Engineer"
# for years; we don't want to miss that. But a Marketing Manager who stuffed AI
# skills still scores near zero because neither title nor history will match high tiers.

# Score: 0.95 — titles that directly match the JD role family.
TITLE_TIER_HIGH: list[str] = [
    "ml engineer",
    "machine learning engineer",
    "ai engineer",
    "data scientist",
    "applied scientist",
    "research scientist",  # may trigger research-only disqualifier separately
    "search engineer",
    "retrieval engineer",
    "ranking engineer",
    "recommendation systems engineer",
    "recommendation engineer",
    "nlp engineer",
    "information retrieval",
    "applied ml",
    "applied machine learning",
    "ai research engineer",
    "staff ml engineer",
    "senior ml engineer",
    "principal ml engineer",
]

# Score: 0.60 — data/ML-adjacent roles that often do production ML work.
TITLE_TIER_MEDIUM: list[str] = [
    "data engineer",
    "senior data engineer",
    "analytics engineer",
    "mlops engineer",
    "machine learning scientist",
    "research engineer",  # product research eng, not pure academic
]

# Score: 0.40 — generic engineering titles (USER DECISION #1).
# Medium-low PRIOR only: production_ml_experience can lift the combined score later.
# We do NOT denylist these — the JD explicitly values "title doesn't say AI but
# career history proves ranking/search work."
TITLE_TIER_MEDIUM_LOW: list[str] = [
    "software engineer",
    "senior software engineer",
    "backend engineer",
    "full stack",
    "fullstack",
    "frontend engineer",
    "cloud engineer",
    "devops engineer",
    "platform engineer",
    "sre",
    "site reliability",
    "java developer",
    ".net developer",
    "mobile developer",
    "staff engineer",
    "principal engineer",
    "tech lead",
    "engineering manager",
]

# Score: 0.28 — business/analysis roles (USER DECISION #2).
# Low-medium prior: JD wants production code writers, not slide-deck analysts.
# Strong career_history evidence of shipped ranking/search systems can still win.
TITLE_TIER_LOW_MEDIUM: list[str] = [
    "business analyst",
    "data analyst",
    "product analyst",
    "financial analyst",
]

# Score: 0.12 — light technical overlap but not ML engineering.
TITLE_TIER_LOW: list[str] = [
    "qa engineer",
    "test engineer",
    "quality assurance",
    "scrum master",
    "product manager",
    "technical writer",
]

# Score: 0.05 — explicit non-fits per JD "explicitly NOT wanted" and EDA findings.
# REGARDLESS of how many AI keywords appear in the skills section.
TITLE_TIER_DENY: list[str] = [
    "hr manager",
    "human resources",
    "recruiter",
    "talent acquisition",
    "marketing manager",
    "marketing ",
    "brand manager",
    "sales executive",
    "sales manager",
    "account manager",
    "content writer",
    "content manager",
    "copywriter",
    "social media",
    "accountant",
    "accounting",
    "civil engineer",
    "mechanical engineer",
    "electrical engineer",
    "graphic designer",
    "ui designer",
    "ux designer",
    "product designer",
    "video editor",
    "photographer",
    "project manager",
    "operations manager",
    "office manager",
    "customer support",
    "customer success",
    "administrative",
    "legal counsel",
    "seo specialist",
]

# Numeric score assigned to each tier (used in docstrings / debugging output).
TIER_SCORES = {
    "high": 0.95,
    "medium": 0.60,
    "medium_low": 0.40,
    "low_medium": 0.28,
    "low": 0.12,
    "deny": 0.05,
    "unknown": 0.20,  # unmatched title — cautious default, not automatic deny
}

# =============================================================================
# CONSULTING / SERVICES COMPANIES (soft disqualifier per JD)
# =============================================================================
# Normalized lowercase substrings. If EVERY career entry is at one of these
# (or industry == "IT Services") AND no product-company stint exists → penalty.
CONSULTING_COMPANY_PATTERNS: list[str] = [
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "mindtree",
    "hcl",
    "tech mahindra",
    "mphasis",
    "lti",
    "larsen & toubro infotech",
    "persistent systems",
    "cyient",
    "zensar",
    "hexaware",
    "birlasoft",
    "ntt data",
    "deloitte",
    "pwc",
    "ey ",
    "ernst & young",
    "kpmg",
    "ibm global",
    "genpact",
]

# Industries that strongly suggest services/consulting work (not product).
SERVICES_INDUSTRIES: set[str] = {
    "it services",
    "consulting",
    "staffing & recruiting",
    "outsourcing",
}

# =============================================================================
# LOCATION (USER DECISION #3)
# =============================================================================
# Tier-1 Indian cities named in the JD (Pune/Noida preferred; also Hyderabad,
# Mumbai, Delhi NCR). Matched as substrings in profile.location (case-insensitive).
INDIA_TIER1_LOCATION_PATTERNS: list[str] = [
    "pune",
    "noida",
    "hyderabad",
    "mumbai",
    "delhi",
    "gurgaon",
    "gurugram",
    "ncr",
    "faridabad",
    "ghaziabad",
    "greater noida",
]

LOCATION_SCORE_TIER1_INDIA = 0.95   # JD preferred cities
LOCATION_SCORE_OTHER_INDIA = 0.65   # domestic relocation, NOT a visa issue
LOCATION_SCORE_ABROAD_RELOCATE = 0.40  # medium-low: no visa sponsorship
LOCATION_SCORE_ABROAD_NO_RELOCATE = 0.15  # low: effectively out of scope

# =============================================================================
# CAREER-HISTORY TEXT PATTERNS for production_ml_experience
# =============================================================================
# We read descriptions, not skill tags. Grouped by strength.

# Strong signals: directly matches JD core work (ranking / search / retrieval).
PROD_ML_STRONG_PATTERNS: list[str] = [
    r"\branking\b",
    r"learning.to.rank",
    r"\bltr\b",
    r"recommendation system",
    r"recommender system",
    r"information retrieval",
    r"\bretrieval\b",
    r"semantic search",
    r"hybrid search",
    r"vector search",
    r"vector database",
    r"\bembeddings?\b",
    r"sentence.transformer",
    r"\bndcg\b",
    r"\bmrr\b",
    r"\bmap\b",  # careful: also "roadmap" — we require word boundary context below
    r"offline.online",
    r"a/b test",
    r"ab test",
    r"re.ranking",
    r"relevance label",
]

# Medium signals: production ML deployment language (supports but doesn't prove search/ranking).
PROD_ML_MEDIUM_PATTERNS: list[str] = [
    r"shipped",
    r"deployed",
    r"production",
    r"real users",
    r"production load",
    r"serving",
    r"inference",
    r"model serving",
    r"feature pipeline",
    r"ml platform",
    r"fine.tun",
    r"\bpytorch\b",
    r"\btensorflow\b",
]

# Weak signals: generic ML mention in career text (small weight).
PROD_ML_WEAK_PATTERNS: list[str] = [
    r"machine learning",
    r"\bnlp\b",
    r"natural language",
    r"deep learning",
]

# Patterns that suggest NON-search ML domains (used in CV/robotics disqualifier).
NON_NLP_DOMAIN_PATTERNS: list[str] = [
    r"computer vision",
    r"\bcv\b",
    r"image classification",
    r"object detection",
    r"speech recognition",
    r"\basr\b",
    r"robotics",
    r"autonomous vehicle",
    r"lid\b",
]

NLP_IR_PATTERNS: list[str] = [
    r"\bnlp\b",
    r"natural language",
    r"information retrieval",
    r"\bretrieval\b",
    r"semantic search",
    r"embedding",
    r"language model",
    r"\bllm",
    r"text ranking",
    r"search quality",
]

# Research-only environment indicators (disqualifier).
RESEARCH_ENV_PATTERNS: list[str] = [
    r"\bphd\b",
    r"postdoc",
    r"research lab",
    r"university",
    r"academic",
    r"published paper",
    r"peer.review",
]

PRODUCTION_DEPLOY_PATTERNS: list[str] = [
    r"production",
    r"deployed",
    r"shipped",
    r"real users",
    r"serving",
    r"a/b test",
]

# LangChain / OpenAI-wrapper-only recent AI (disqualifier).
LANGCHAIN_WRAPPER_PATTERNS: list[str] = [
    r"langchain",
    r"openai api",
    r"chatgpt api",
    r"gpt-4 api",
    r"prompt chain",
    r"llm wrapper",
]

PRE_LLM_ML_PATTERNS: list[str] = [
    r"pytorch",
    r"tensorflow",
    r"scikit",
    r"xgboost",
    r"feature engineering",
    r"model training",
    r"production ml",
    r"ranking model",
    r"recommendation",
    r"embeddings?",
]

# Senior non-coding role titles (disqualifier: no hands-on code 18+ months).
SENIOR_NON_CODING_TITLES: list[str] = [
    "architect",
    "architecture",
    "tech lead",
    "technical lead",
    "engineering manager",
    "director of engineering",
    "vp engineering",
    "principal architect",
    "staff architect",
    "chief technology",
]

HANDS_ON_CODE_PATTERNS: list[str] = [
    r"\bpython\b",
    r"\bcode\b",
    r"coding",
    r"implemented",
    r"wrote",
    r"built",
    r"developed",
    r"refactor",
    r"pull request",
    r"\bpr\b",
    r"github",
]

# =============================================================================
# EXPERIENCE BAND (JD: 5–9 years, soft range)
# =============================================================================
EXP_IDEAL_MIN = 5.0
EXP_IDEAL_MAX = 9.0
# Gaussian-style taper: full score inside band, smooth decay outside.
# At 0 years → ~0.15; at 12 years → ~0.75; never hard-zero (JD says "not hard req").
EXP_TAPER_SIGMA = 2.5  # controls how fast score drops outside 5–9 band


def _normalize(text: str) -> str:
    return text.lower().strip()


def _title_matches_any(title: str, patterns: list[str]) -> bool:
    lower = _normalize(title)
    return any(p in lower for p in patterns)


def score_single_title(title: str) -> tuple[float, str]:
    """
    Map one job title string to a (score, tier_name) pair.
    First matching tier wins (checked highest relevance first).
    """
    if _title_matches_any(title, TITLE_TIER_HIGH):
        return TIER_SCORES["high"], "high"
    if _title_matches_any(title, TITLE_TIER_MEDIUM):
        return TIER_SCORES["medium"], "medium"
    if _title_matches_any(title, TITLE_TIER_MEDIUM_LOW):
        return TIER_SCORES["medium_low"], "medium_low"
    if _title_matches_any(title, TITLE_TIER_LOW_MEDIUM):
        return TIER_SCORES["low_medium"], "low_medium"
    if _title_matches_any(title, TITLE_TIER_LOW):
        return TIER_SCORES["low"], "low"
    if _title_matches_any(title, TITLE_TIER_DENY):
        return TIER_SCORES["deny"], "deny"
    return TIER_SCORES["unknown"], "unknown"


def compute_title_relevance(candidate: dict) -> dict[str, Any]:
    """
    Title relevance: the primary anti-keyword-stuffing feature.

    Rules:
      1. Score current_title and every career_history.title independently.
      2. Take the MAX score across all titles (best-case career signal).
      3. Deny-tier titles (HR, Marketing, etc.) score 0.05 regardless of skills.
      4. Generic eng titles score 0.40 (medium-low prior) per user decision #1.
      5. Business Analyst scores 0.28 (low-medium prior) per user decision #2.

    Returns dict with score plus debug fields for reasoning generation later.
    """
    profile = candidate.get("profile", {})
    current = profile.get("current_title", "")
    history = candidate.get("career_history", [])

    scored: list[tuple[str, float, str]] = []
    cur_score, cur_tier = score_single_title(current)
    scored.append((current, cur_score, cur_tier))

    for entry in history:
        t = entry.get("title", "")
        s, tier = score_single_title(t)
        scored.append((t, s, tier))

    best_title, best_score, best_tier = max(scored, key=lambda x: x[1])

    return {
        "title_relevance": best_score,
        "title_best_match": best_title,
        "title_best_tier": best_tier,
        "title_current_score": cur_score,
        "title_current_tier": cur_tier,
    }


def _text_matches_any(text: str, patterns: list[str]) -> bool:
    lower = _normalize(text)
    for pat in patterns:
        if re.search(pat, lower):
            return True
    return False


def _count_pattern_hits(text: str, patterns: list[str]) -> int:
    lower = _normalize(text)
    return sum(1 for pat in patterns if re.search(pat, lower))


def compute_production_ml_experience(candidate: dict) -> dict[str, Any]:
    """
    Production ML experience from career_history descriptions (NOT skill tags).

    Scoring logic:
      - Concatenate all career_history description texts.
      - Strong patterns (ranking, retrieval, embeddings, NDCG, etc.) → 0.25 each,
        capped at 0.75 from strong hits alone.
      - Medium patterns (shipped, deployed, production) → 0.08 each, cap 0.24.
      - Weak patterns (generic "machine learning") → 0.04 each, cap 0.12.
      - Bonus if strong hits appear in 2+ separate job entries (breadth): +0.10.
      - Final clamped to [0, 1].

    This is intentionally independent of title_relevance so a "Software Engineer"
    with real ranking-system career text can score high here (user decision #1).
    """
    history = candidate.get("career_history", [])
    if not history:
        return {"production_ml_experience": 0.0, "prod_ml_strong_hits": 0, "prod_ml_entries_with_strong": 0}

    all_text = " ".join(entry.get("description", "") for entry in history)
    strong_hits = _count_pattern_hits(all_text, PROD_ML_STRONG_PATTERNS)
    medium_hits = _count_pattern_hits(all_text, PROD_ML_MEDIUM_PATTERNS)
    weak_hits = _count_pattern_hits(all_text, PROD_ML_WEAK_PATTERNS)

    strong_component = min(0.75, strong_hits * 0.25)
    medium_component = min(0.24, medium_hits * 0.08)
    weak_component = min(0.12, weak_hits * 0.04)

    entries_with_strong = sum(
        1 for entry in history if _text_matches_any(entry.get("description", ""), PROD_ML_STRONG_PATTERNS)
    )
    breadth_bonus = 0.10 if entries_with_strong >= 2 else 0.0

    raw = strong_component + medium_component + weak_component + breadth_bonus
    score = min(1.0, raw)

    return {
        "production_ml_experience": round(score, 4),
        "prod_ml_strong_hits": strong_hits,
        "prod_ml_medium_hits": medium_hits,
        "prod_ml_entries_with_strong": entries_with_strong,
    }


def _is_consulting_company(company: str) -> bool:
    lower = _normalize(company)
    return any(pat in lower for pat in CONSULTING_COMPANY_PATTERNS)


def _is_product_company_entry(entry: dict) -> bool:
    """
    Heuristic: a stint counts as "product company" if industry is NOT pure
    services/consulting. This is imperfect but matches JD intent: TCS-only
    careers are penalized; someone who did 3 years at a product startup then
    joined Infosys is fine.
    """
    industry = _normalize(entry.get("industry", ""))
    if industry in SERVICES_INDUSTRIES:
        return False
    company = entry.get("company", "")
    if _is_consulting_company(company):
        return False
    return True


def compute_company_type_penalty(candidate: dict) -> dict[str, Any]:
    """
    Soft penalty per JD: pure consulting-only career with zero product experience.

    Returns company_type_penalty in [0, 1] where 1.0 = no penalty and
    0.35 = heavy penalty (multiply into final score later).

    Rule:
      - If ALL entries are consulting (by company name OR IT Services industry)
        AND none qualify as product → penalty = 0.35
      - If majority consulting but at least one product stint → penalty = 0.75
      - Otherwise → penalty = 1.0 (no reduction)
    """
    history = candidate.get("career_history", [])
    if not history:
        return {"company_type_penalty": 1.0, "company_all_consulting": False}

    consulting_flags = []
    product_flags = []
    for entry in history:
        is_consult = _is_consulting_company(entry.get("company", "")) or (
            _normalize(entry.get("industry", "")) in SERVICES_INDUSTRIES
        )
        consulting_flags.append(is_consult)
        product_flags.append(_is_product_company_entry(entry))

    all_consulting = all(consulting_flags)
    has_product = any(product_flags)

    if all_consulting and not has_product:
        multiplier = 0.35
    elif sum(consulting_flags) > len(consulting_flags) / 2 and has_product:
        multiplier = 0.75
    else:
        multiplier = 1.0

    return {
        "company_type_penalty": multiplier,
        "company_all_consulting": all_consulting,
        "company_has_product_experience": has_product,
    }


def compute_experience_fit(candidate: dict) -> dict[str, Any]:
    """
    Smooth experience fit for JD band 5–9 years (not a hard cutoff).

    Inside [5, 9]: score = 1.0
    Outside: Gaussian decay centered on midpoint 7.0:
      score = exp(-((years - 7)^2) / (2 * sigma^2))
    Clamped to minimum 0.15 so very junior/senior aren't zeroed out entirely.
    """
    years = float(candidate.get("profile", {}).get("years_of_experience", 0))
    center = (EXP_IDEAL_MIN + EXP_IDEAL_MAX) / 2.0  # 7.0

    if EXP_IDEAL_MIN <= years <= EXP_IDEAL_MAX:
        score = 1.0
    else:
        score = math.exp(-((years - center) ** 2) / (2 * EXP_TAPER_SIGMA**2))

    score = max(0.15, min(1.0, score))
    return {"experience_fit": round(score, 4), "years_of_experience": years}


def compute_location_fit(candidate: dict) -> dict[str, Any]:
    """
    Location scoring per user decision #3 and JD logistics.

    Tier-1 India (Noida/Pune/Hyderabad/Mumbai/Delhi NCR) → 0.95
    Other India (Tier-2/3 domestic relocation)           → 0.65
    Outside India + willing_to_relocate=true             → 0.40
    Outside India + willing_to_relocate=false            → 0.15 (no visa sponsorship)
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    country = _normalize(profile.get("country", ""))
    location = _normalize(profile.get("location", ""))
    willing = bool(signals.get("willing_to_relocate", False))

    if country in ("india", "in"):
        if any(pat in location for pat in INDIA_TIER1_LOCATION_PATTERNS):
            score = LOCATION_SCORE_TIER1_INDIA
            tier = "india_tier1"
        else:
            score = LOCATION_SCORE_OTHER_INDIA
            tier = "india_other"
    else:
        if willing:
            score = LOCATION_SCORE_ABROAD_RELOCATE
            tier = "abroad_relocate"
        else:
            score = LOCATION_SCORE_ABROAD_NO_RELOCATE
            tier = "abroad_no_relocate"

    return {
        "location_fit": score,
        "location_tier": tier,
        "country": profile.get("country", ""),
        "location": profile.get("location", ""),
        "willing_to_relocate": willing,
    }


def compute_notice_period_fit(candidate: dict) -> dict[str, Any]:
    """
    Notice period fit per JD: sub-30 days strongly preferred.

    Piecewise linear taper:
      0 days   → 1.00
      30 days  → 0.85  (JD can buy out up to 30)
      60 days  → 0.65
      90 days  → 0.50
      180 days → 0.35  (floor)
    """
    days = int(candidate.get("redrob_signals", {}).get("notice_period_days", 90))

    if days <= 30:
        # Linear from 1.0 at 0d to 0.85 at 30d
        score = 1.0 - (days / 30.0) * 0.15
    elif days <= 60:
        score = 0.85 - ((days - 30) / 30.0) * 0.20
    elif days <= 90:
        score = 0.65 - ((days - 60) / 30.0) * 0.15
    else:
        # 90→180 maps 0.50→0.35
        score = 0.50 - min(1.0, (days - 90) / 90.0) * 0.15

    score = max(0.35, min(1.0, score))
    return {"notice_period_fit": round(score, 4), "notice_period_days": days}


def compute_education_tier_bonus(candidate: dict) -> dict[str, Any]:
    """
    Small education bonus — deliberately minor so it doesn't dominate ranking.

    tier_1 → +0.05, tier_2 → +0.03, else → 0.0
    Returned as education_tier_bonus (additive, not multiplicative).
    """
    education = candidate.get("education", [])
    best_bonus = 0.0
    best_tier = "none"
    tier_values = {"tier_1": 0.05, "tier_2": 0.03}
    for edu in education:
        tier = edu.get("tier", "unknown")
        bonus = tier_values.get(tier, 0.0)
        if bonus > best_bonus:
            best_bonus = bonus
            best_tier = tier
    return {"education_tier_bonus": best_bonus, "education_best_tier": best_tier}


def compute_disqualifier_flags(candidate: dict) -> dict[str, Any]:
    """
    JD hard disqualifiers expressed as independent boolean flags.
    Each active flag applies a multiplicative penalty to the final score.

    Penalty multipliers (intentionally harsh — these are "hard filters" in spirit):
      pure_research_only          → ×0.15
      langchain_wrapper_only      → ×0.20
      senior_no_hands_on_code       → ×0.25
      cv_speech_robotics_no_nlp   → ×0.20

    Combined: multiply all active flag penalties together.
    Also returns disqualifier_penalty as the combined multiplier.
    """
    history = candidate.get("career_history", [])
    all_desc = " ".join(entry.get("description", "") for entry in history)
    all_titles = " ".join(entry.get("title", "") for entry in history)

    flags: dict[str, bool] = {
        "pure_research_only": False,
        "langchain_wrapper_only": False,
        "senior_no_hands_on_code": False,
        "cv_speech_robotics_no_nlp": False,
    }
    penalties: dict[str, float] = {
        "pure_research_only": 0.15,
        "langchain_wrapper_only": 0.20,
        "senior_no_hands_on_code": 0.25,
        "cv_speech_robotics_no_nlp": 0.20,
    }
    reasons: list[str] = []

    # --- Flag 1: pure research, no production deployment ---
    has_research = _text_matches_any(all_desc + " " + all_titles, RESEARCH_ENV_PATTERNS)
    has_production = _text_matches_any(all_desc, PRODUCTION_DEPLOY_PATTERNS)
    # Also treat "Research Scientist" at academic institution with no ship language
    research_titles = sum(1 for e in history if "research" in _normalize(e.get("title", "")))
    if has_research and not has_production and research_titles >= len(history) // 2:
        flags["pure_research_only"] = True
        reasons.append("career appears research-only without production deployment")

    # --- Flag 2: recent LangChain/OpenAI-wrapper-only AI (<12 months), no pre-LLM ML ---
    recent_months = 0
    recent_wrapper_only = True
    older_ml = _text_matches_any(all_desc, PRE_LLM_ML_PATTERNS)
    for entry in history:
        if entry.get("is_current") or (entry.get("duration_months", 0) <= 12):
            desc = entry.get("description", "")
            if _text_matches_any(desc, LANGCHAIN_WRAPPER_PATTERNS):
                recent_months += entry.get("duration_months", 0)
            elif _text_matches_any(desc, PRE_LLM_ML_PATTERNS):
                recent_wrapper_only = False
    if recent_months > 0 and recent_months <= 12 and recent_wrapper_only and not older_ml:
        flags["langchain_wrapper_only"] = True
        reasons.append("AI experience appears to be recent LangChain/OpenAI-wrapper only")

    # --- Flag 3: senior title, no hands-on code in last ~18 months ---
    recent_entries = [e for e in history if e.get("is_current") or e.get("duration_months", 0) <= 18]
    if recent_entries:
        senior_recent = any(
            _title_matches_any(e.get("title", ""), SENIOR_NON_CODING_TITLES) for e in recent_entries
        )
        any_code = any(_text_matches_any(e.get("description", ""), HANDS_ON_CODE_PATTERNS) for e in recent_entries)
        if senior_recent and not any_code:
            flags["senior_no_hands_on_code"] = True
            reasons.append("senior architecture/lead role in last 18mo without hands-on code signals")

    # --- Flag 4: CV/speech/robotics background without NLP/IR exposure ---
    non_nlp_hits = _count_pattern_hits(all_desc, NON_NLP_DOMAIN_PATTERNS)
    nlp_hits = _count_pattern_hits(all_desc, NLP_IR_PATTERNS)
    if non_nlp_hits >= 2 and nlp_hits == 0:
        flags["cv_speech_robotics_no_nlp"] = True
        reasons.append("CV/speech/robotics focus without NLP/IR career evidence")

    combined = 1.0
    for flag_name, active in flags.items():
        if active:
            combined *= penalties[flag_name]

    return {
        **{f"disqualifier_{k}": v for k, v in flags.items()},
        "disqualifier_penalty": round(combined, 4),
        "disqualifier_reasons": reasons,
    }


def compute_features(candidate: dict) -> dict[str, Any]:
    """
    Compute all rule-based sub-scores for one candidate.

    Returns a flat dict merging all sub-score components. Downstream rank.py
    will combine these with explicit weights; this module only extracts signals.
    """
    result: dict[str, Any] = {"candidate_id": candidate.get("candidate_id")}
    result.update(compute_title_relevance(candidate))
    result.update(compute_production_ml_experience(candidate))
    result.update(compute_company_type_penalty(candidate))
    result.update(compute_experience_fit(candidate))
    result.update(compute_location_fit(candidate))
    result.update(compute_notice_period_fit(candidate))
    result.update(compute_education_tier_bonus(candidate))
    result.update(compute_disqualifier_flags(candidate))
    return result


def load_job_description(path: Path | None = None) -> str:
    """Load JD text for optional semantic similarity stage (Stage 4)."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "docs" / "job_description.md"
    return path.read_text(encoding="utf-8")


# =============================================================================
# Quick sanity demo when run directly (uses sample_candidates.json)
# =============================================================================
if __name__ == "__main__":
    import json
    import sys

    repo = Path(__file__).resolve().parent.parent
    sample_path = repo / "data" / "sample_candidates.json"

    if not sample_path.exists():
        print("sample_candidates.json not found", file=sys.stderr)
        sys.exit(1)

    candidates = json.loads(sample_path.read_text(encoding="utf-8"))

    # Pick a few illustrative profiles if present
    showcase_ids = {"CAND_0000001", "CAND_0004989"}  # backend eng + bad HR from sample submission
    for cand in candidates:
        cid = cand.get("candidate_id", "")
        title = cand.get("profile", {}).get("current_title", "")
        # Print: one good ML title, one trap title, one generic eng
        is_showcase = cid in showcase_ids
        is_ml = "recommendation" in title.lower() or "ml engineer" in title.lower()
        is_trap = "hr manager" in title.lower() or "marketing" in title.lower()
        if is_showcase or is_ml or is_trap:
            feats = compute_features(cand)
            print(f"\n{'─' * 60}")
            print(f"{cid} | {title}")
            for key in sorted(feats.keys()):
                if key != "candidate_id":
                    print(f"  {key}: {feats[key]}")
