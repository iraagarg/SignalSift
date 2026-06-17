#!/usr/bin/env python3
"""
Stage 10 — Streamlit demo for Redrob candidate ranker.

Runs the same scoring pipeline as scripts/rank.py on a small sample (≤100).
For the full 100K pool, use precomputed embeddings + rank.py offline.

Deploy to Streamlit Community Cloud with:
  streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make scripts/ importable from repo root.
REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from embeddings import (  # noqa: E402
    _get_model,
    build_candidate_text,
    cosine_similarity_to_job,
    encode_texts,
    load_job_description_text,
)
from rank import rank_candidate_list, write_submission_csv  # noqa: E402
from reasoning import generate_reasoning  # noqa: E402

SAMPLE_PATH = REPO_ROOT / "data" / "sample_candidates.json"
MAX_DEMO_CANDIDATES = 100


@st.cache_resource(show_spinner="Loading sentence-transformers model (one-time, CPU)…")
def load_embedding_model():
    """Cache the MiniLM model across Streamlit reruns."""
    return _get_model()


@st.cache_data(show_spinner="Embedding job description…")
def embed_job_description():
    model = load_embedding_model()
    jd_text = load_job_description_text()
    return encode_texts(model, [jd_text], show_progress_bar=False)[0]


def embed_candidates(candidates: list[dict]) -> dict[str, float]:
    """Embed a small candidate list and return candidate_id → cosine similarity."""
    model = load_embedding_model()
    job_vec = embed_job_description()
    texts = [build_candidate_text(c) for c in candidates]
    vecs = encode_texts(model, texts, show_progress_bar=False)
    return {
        c["candidate_id"]: cosine_similarity_to_job(vecs[i], job_vec)
        for i, c in enumerate(candidates)
    }


def parse_upload(uploaded_file) -> list[dict]:
    """Parse uploaded JSON array or JSONL file into candidate dicts."""
    raw = uploaded_file.read().decode("utf-8")
    if uploaded_file.name.endswith(".jsonl"):
        candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data = json.loads(raw)
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict) and "candidate_id" in data:
            candidates = [data]
        else:
            raise ValueError("JSON must be an array of candidates or JSONL lines.")
    return candidates


def rows_to_display_table(ranked: list[dict]) -> pd.DataFrame:
    """Build a human-readable results table for the UI."""
    rows = []
    for rank, row in enumerate(ranked, start=1):
        profile = row["candidate"]["profile"]
        rows.append(
            {
                "rank": rank,
                "candidate_id": row["candidate_id"],
                "score": round(row["score"], 4),
                "current_title": profile.get("current_title"),
                "years_of_experience": profile.get("years_of_experience"),
                "location": profile.get("location"),
                "title_relevance": row["feats"].get("title_relevance"),
                "production_ml": row["feats"].get("production_ml_experience"),
                "honeypot": row["honeypot"].get("is_likely_honeypot"),
                "reasoning": generate_reasoning(
                    row["candidate"],
                    row["feats"],
                    row["behavioral"],
                    row["honeypot"],
                    row["semantic_raw"],
                    rank,
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
    st.title("Redrob Intelligent Candidate Ranker")
    st.markdown(
        "Rule-based + semantic ranking for the **Senior AI Engineer** role. "
        "Uses the same pipeline as `scripts/rank.py` on a small sample."
    )

    source = st.radio(
        "Candidate source",
        ["Built-in sample (50 candidates)", "Upload JSON / JSONL (max 100)"],
    )

    candidates: list[dict] = []
    if source.startswith("Built-in"):
        if SAMPLE_PATH.exists():
            candidates = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
            st.info(f"Loaded {len(candidates)} candidates from `data/sample_candidates.json`.")
        else:
            st.error("Sample file not found.")
            return
    else:
        uploaded = st.file_uploader("Upload candidates", type=["json", "jsonl"])
        if uploaded is None:
            st.warning("Upload a file to continue.")
            return
        try:
            candidates = parse_upload(uploaded)
        except (json.JSONDecodeError, ValueError) as exc:
            st.error(f"Could not parse file: {exc}")
            return

    if len(candidates) > MAX_DEMO_CANDIDATES:
        st.error(f"Demo supports at most {MAX_DEMO_CANDIDATES} candidates.")
        return

    top_n = st.slider("Top N to rank", min_value=5, max_value=min(100, len(candidates)), value=min(10, len(candidates)))

    if st.button("Run ranking", type="primary"):
        with st.spinner("Embedding candidates and computing scores…"):
            semantic_map = embed_candidates(candidates)
            ranked = rank_candidate_list(candidates, semantic_map, top_n=top_n)

        st.success(f"Ranked top {len(ranked)} of {len(candidates)} candidates.")
        df = rows_to_display_table(ranked)
        st.dataframe(df, use_container_width=True)

        # CSV download matching submission format.
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            write_submission_csv(ranked, Path(tmp.name))
            csv_content = Path(tmp.name).read_text(encoding="utf-8")

        st.download_button(
            "Download ranked CSV",
            data=csv_content,
            file_name="ranked_sample.csv",
            mime="text/csv",
        )

        with st.expander("Architecture reminder"):
            st.markdown(
                """
                **Scoring layers (same as production ranker):**
                1. `title_relevance` (32%) — anti-keyword-stuffing gate
                2. `production_ml_experience` (28%) — career text, not skill tags
                3. `semantic_similarity` (12%) — MiniLM embeddings vs JD
                4. Experience / location / notice period fits
                5. × consulting penalty × disqualifiers × honeypot demotion
                6. × behavioral engagement multiplier

                Full 100K ranking uses **precomputed** embeddings (`data/embeddings/`)
                and completes in ~90s CPU-only with no network.
                """
            )


if __name__ == "__main__":
    main()
