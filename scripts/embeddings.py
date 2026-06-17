#!/usr/bin/env python3
"""
Stage 4 — Semantic similarity via local sentence-transformers embeddings.

This module has TWO distinct phases (documented per submission_spec.md §3):

  1. PRECOMPUTE (offline, no time budget limit):
       python scripts/embeddings.py precompute \\
           --candidates data/candidates.jsonl \\
           --out data/embeddings

     Downloads the model once, embeds all 100K candidates + the job description,
     and writes artifacts to disk. This step may take 30–90 minutes on CPU.

  2. RANKING (≤5 min budget in rank.py):
       rank.py only *loads* the precomputed .npy files and computes dot products
       (cosine similarity on L2-normalized vectors). No model inference at rank time.

Model: sentence-transformers/all-MiniLM-L6-v2
  - 384 dimensions, ~80MB, runs on CPU without GPU
  - Fully offline after first model download during precompute
  - No OpenAI / hosted API calls

Text embedded per candidate:
  profile.summary + all career_history[].description (space-joined)
  Skills are intentionally excluded — they are the keyword-stuffing trap.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

# Model identifier on HuggingFace Hub. Downloaded once during precompute.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Batch size for encoding during precompute. Tuned for ≤16GB RAM on CPU.
# Each batch holds ~256 texts × ~2KB avg ≈ negligible vs model weights (~80MB).
DEFAULT_BATCH_SIZE = 256

# Output filenames inside --out directory.
CANDIDATE_EMBEDDINGS_FILE = "candidate_embeddings.npy"
CANDIDATE_IDS_FILE = "candidate_ids.json"
JOB_EMBEDDING_FILE = "job_embedding.npy"
METADATA_FILE = "metadata.json"


def build_candidate_text(candidate: dict) -> str:
    """
    Build the text we embed for semantic similarity.

    Uses summary + career descriptions only (NOT skills).
    Rationale: the JD warns that skill tags are keyword-stuffed; career text
  describes what the person actually did.
    """
    profile = candidate.get("profile", {})
    parts: list[str] = []

    summary = profile.get("summary", "").strip()
    if summary:
        parts.append(summary)

    for entry in candidate.get("career_history", []):
        desc = entry.get("description", "").strip()
        if desc:
            parts.append(desc)

    # Fallback so empty profiles still produce a vector (will score low similarity).
    if not parts:
        title = profile.get("current_title", "")
        headline = profile.get("headline", "")
        parts.append(f"{title}. {headline}".strip())

    return " ".join(parts)


def load_job_description_text(jd_path: Path | None = None) -> str:
    """Load full job description markdown for embedding."""
    if jd_path is None:
        jd_path = Path(__file__).resolve().parent.parent / "docs" / "job_description.md"
    return jd_path.read_text(encoding="utf-8").strip()


def _get_model(model_name: str = DEFAULT_MODEL):
    """
    Lazy-import SentenceTransformer so `python scripts/embeddings.py --help`
    works even if sentence-transformers isn't installed yet.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embedding precompute.\n"
            "Install with: pip install sentence-transformers"
        ) from exc

    # device='cpu' is explicit per hackathon compute constraints.
    return SentenceTransformer(model_name, device="cpu")


def encode_texts(
    model,
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """
    Encode a list of strings → (N, 384) float32 array, L2-normalized.

    normalize_embeddings=True makes cosine similarity a simple dot product,
    which is fast at ranking time.

    show_progress_bar defaults to False; the precompute loop prints its own
    progress every 5,000 candidates to avoid flooding the terminal.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def stream_candidates(candidates_path: Path) -> Iterator[dict]:
    """Yield candidates from JSONL one at a time (memory-safe for 100K rows)."""
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_candidates(candidates_path: Path) -> int:
    """Count non-empty lines in JSONL (for pre-allocating embedding matrix)."""
    count = 0
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def precompute_embeddings(
    candidates_path: Path,
    out_dir: Path,
    jd_path: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """
    Precompute and cache all candidate + job embeddings to disk.

    Writes:
      out_dir/candidate_embeddings.npy  — float32 array shape (N, 384)
      out_dir/candidate_ids.json        — list of candidate_id strings, row order
      out_dir/job_embedding.npy         — float32 array shape (384,)
      out_dir/metadata.json             — model name, dims, timestamps

    Returns metadata dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name} (CPU)")
    t0 = time.perf_counter()
    model = _get_model(model_name)
    model_load_s = time.perf_counter() - t0
    print(f"Model loaded in {model_load_s:.1f}s")

    # --- Job description embedding (single vector) ---
    jd_text = load_job_description_text(jd_path)
    print(f"Embedding job description ({len(jd_text):,} chars)...")
    job_embedding = encode_texts(model, [jd_text], batch_size=1)[0]
    np.save(out_dir / JOB_EMBEDDING_FILE, job_embedding)

    # --- Candidate embeddings (streamed in batches) ---
    n_candidates = count_candidates(candidates_path)
    print(f"Counting candidates: {n_candidates:,}")

    dim = job_embedding.shape[0]
    # Pre-allocate full matrix so we can write row indices directly.
    all_embeddings = np.zeros((n_candidates, dim), dtype=np.float32)
    all_ids: list[str] = []

    batch_texts: list[str] = []
    batch_ids: list[str] = []
    row_idx = 0

    print(f"Embedding candidates from {candidates_path} ...")
    encode_start = time.perf_counter()

    for candidate in stream_candidates(candidates_path):
        cid = candidate["candidate_id"]
        text = build_candidate_text(candidate)

        batch_texts.append(text)
        batch_ids.append(cid)

        if len(batch_texts) >= batch_size:
            vecs = encode_texts(model, batch_texts, batch_size=batch_size)
            end = row_idx + len(batch_texts)
            all_embeddings[row_idx:end] = vecs
            all_ids.extend(batch_ids)
            row_idx = end
            batch_texts.clear()
            batch_ids.clear()
            if row_idx % 5000 == 0:
                elapsed = time.perf_counter() - encode_start
                rate = row_idx / elapsed if elapsed > 0 else 0
                print(f"  {row_idx:,}/{n_candidates:,} ({rate:.0f} candidates/s)")

    # Final partial batch
    if batch_texts:
        vecs = encode_texts(model, batch_texts, batch_size=batch_size)
        end = row_idx + len(batch_texts)
        all_embeddings[row_idx:end] = vecs
        all_ids.extend(batch_ids)
        row_idx = end

    if row_idx != n_candidates:
        raise RuntimeError(f"Expected {n_candidates} candidates, processed {row_idx}")

    encode_s = time.perf_counter() - encode_start
    print(f"Encoded {row_idx:,} candidates in {encode_s:.1f}s ({row_idx/encode_s:.0f}/s)")

    # Persist artifacts
    np.save(out_dir / CANDIDATE_EMBEDDINGS_FILE, all_embeddings)
    with open(out_dir / CANDIDATE_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_ids, f)

    metadata = {
        "model_name": model_name,
        "embedding_dim": int(dim),
        "num_candidates": n_candidates,
        "candidate_text_fields": ["profile.summary", "career_history[].description"],
        "job_description_path": str(jd_path or "docs/job_description.md"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_load_seconds": round(model_load_s, 2),
        "encode_seconds": round(encode_s, 2),
        "batch_size": batch_size,
        "normalized": True,
        "files": {
            "candidate_embeddings": CANDIDATE_EMBEDDINGS_FILE,
            "candidate_ids": CANDIDATE_IDS_FILE,
            "job_embedding": JOB_EMBEDDING_FILE,
        },
    }
    with open(out_dir / METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    emb_mb = all_embeddings.nbytes / (1024 * 1024)
    print(f"\nSaved to {out_dir}/")
    print(f"  {CANDIDATE_EMBEDDINGS_FILE}  ({emb_mb:.1f} MB, shape {all_embeddings.shape})")
    print(f"  {CANDIDATE_IDS_FILE}")
    print(f"  {JOB_EMBEDDING_FILE}")
    print(f"  {METADATA_FILE}")
    print("\nPrecompute complete. Ranking step should only load these files.")

    return metadata


def load_embeddings(
    emb_dir: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """
    Load precomputed embedding artifacts for rank.py.

    Returns:
      candidate_ids:  list[str] length N
      candidate_embeddings: np.ndarray shape (N, 384) float32, L2-normalized rows
      job_embedding: np.ndarray shape (384,) float32, L2-normalized
      metadata: dict from metadata.json

    At ranking time this is a few hundred MB of disk read — no model inference.
    """
    emb_dir = Path(emb_dir)
    for fname in (CANDIDATE_EMBEDDINGS_FILE, CANDIDATE_IDS_FILE, JOB_EMBEDDING_FILE, METADATA_FILE):
        if not (emb_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing embedding artifact: {emb_dir / fname}\n"
                f"Run precompute first:\n"
                f"  python scripts/embeddings.py precompute "
                f"--candidates data/candidates.jsonl --out {emb_dir}"
            )

    with open(emb_dir / CANDIDATE_IDS_FILE, "r", encoding="utf-8") as f:
        candidate_ids: list[str] = json.load(f)

    candidate_embeddings = np.load(emb_dir / CANDIDATE_EMBEDDINGS_FILE)
    job_embedding = np.load(emb_dir / JOB_EMBEDDING_FILE)
    with open(emb_dir / METADATA_FILE, "r", encoding="utf-8") as f:
        metadata: dict = json.load(f)

    return candidate_ids, candidate_embeddings, job_embedding, metadata


def build_id_to_row_index(candidate_ids: list[str]) -> dict[str, int]:
    """Map candidate_id → row index for O(1) embedding lookup during ranking."""
    return {cid: i for i, cid in enumerate(candidate_ids)}


def cosine_similarity_to_job(
    candidate_embedding: np.ndarray,
    job_embedding: np.ndarray,
) -> float:
    """
    Cosine similarity between pre-normalized vectors (= dot product).

    Returns float in roughly [-1, 1]; typically 0.2–0.7 for relevant profiles.
    """
    return float(np.dot(candidate_embedding, job_embedding))


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Embed candidates and job description with sentence-transformers"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser(
        "precompute",
        help="Precompute embeddings for all candidates (offline, no time budget)",
    )
    pre.add_argument(
        "--candidates",
        type=Path,
        default=repo / "data" / "candidates.jsonl",
        help="Path to candidates.jsonl",
    )
    pre.add_argument(
        "--out",
        type=Path,
        default=repo / "data" / "embeddings",
        help="Output directory for .npy artifacts",
    )
    pre.add_argument(
        "--job-description",
        type=Path,
        default=repo / "docs" / "job_description.md",
        help="Job description text to embed",
    )
    pre.add_argument("--model", type=str, default=DEFAULT_MODEL)
    pre.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    test = sub.add_parser("test", help="Quick test on sample_candidates.json (no full precompute)")
    test.add_argument(
        "--sample",
        type=Path,
        default=repo / "data" / "sample_candidates.json",
    )
    test.add_argument("--model", type=str, default=DEFAULT_MODEL)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "precompute":
        if not args.candidates.exists():
            print(f"ERROR: candidates file not found: {args.candidates}", file=sys.stderr)
            return 1
        precompute_embeddings(
            candidates_path=args.candidates,
            out_dir=args.out,
            jd_path=args.job_description,
            model_name=args.model,
            batch_size=args.batch_size,
        )
        return 0

    if args.command == "test":
        if not args.sample.exists():
            print(f"ERROR: sample file not found: {args.sample}", file=sys.stderr)
            return 1
        model = _get_model(args.model)
        jd_text = load_job_description_text()
        job_vec = encode_texts(model, [jd_text])[0]

        candidates = json.loads(args.sample.read_text(encoding="utf-8"))
        texts = [build_candidate_text(c) for c in candidates]
        vecs = encode_texts(model, texts)

        scored = []
        for cand, vec in zip(candidates, vecs):
            sim = cosine_similarity_to_job(vec, job_vec)
            scored.append((sim, cand["candidate_id"], cand["profile"]["current_title"]))

        scored.sort(reverse=True)
        print(f"\nTop 10 by semantic similarity to JD ({args.model}):\n")
        for sim, cid, title in scored[:10]:
            print(f"  {sim:.4f}  {cid}  {title}")
        print(f"\nBottom 5:\n")
        for sim, cid, title in scored[-5:]:
            print(f"  {sim:.4f}  {cid}  {title}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
