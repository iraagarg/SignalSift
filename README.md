# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

Rank the top 100 candidates from `data/candidates.jsonl` for the Senior AI Engineer role using a layered, rule-first pipeline designed to avoid the dataset's keyword-stuffing trap.

## Quick start

```bash
# 1. Create environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Precompute embeddings (offline, ~20 min CPU, needs network once for model download)
python scripts/embeddings.py precompute \
    --candidates data/candidates.jsonl \
    --out data/embeddings

# 3. Rank and write submission CSV (≤5 min, CPU, no network)
python scripts/rank.py \
    --candidates data/candidates.jsonl \
    --out submission.csv

# 4. Validate format
python validate_submission.py submission.csv
```

**Single reproduce command** (after `pip install`):

```bash
python scripts/embeddings.py precompute --candidates data/candidates.jsonl --out data/embeddings && python scripts/rank.py --candidates data/candidates.jsonl --out submission.csv
```

## Architecture

```
candidates.jsonl
       │
       ├─► features.py        title, career-text ML, location, notice, disqualifiers
       ├─► honeypot_check.py   structural profile consistency → demotion multiplier
       ├─► behavioral.py       Redrob engagement signals → 0.5–1.0 multiplier
       └─► embeddings/       precomputed MiniLM vectors (summary + career text)
                │
                ▼
           rank.py  →  weighted qualification score
                       × company_type_penalty
                       × disqualifier_penalty
                       × honeypot_demotion
                       × behavioral_multiplier
                │
                ▼
         submission.csv (top 100 + reasoning)
```

### Why each layer exists

| Layer | Purpose |
|---|---|
| **title_relevance** (32%) | Primary defense against HR/Marketing profiles with stuffed AI skills |
| **production_ml_experience** (28%) | Reads career_history for ranking/search/retrieval evidence — not skill tags |
| **semantic_similarity** (12%) | Supplementary JD match; kept low because embeddings alone rank wrong titles highly |
| **honeypot_check** | Demotes ~55 impossible profiles (YoE/duration/date inconsistencies) |
| **behavioral** | Nudges for platform availability without overriding qualification fit |
| **embeddings precompute** | Satisfies 5-min/CPU/no-network constraint at ranking time |

### Compute profile (100K ranking step)

- **Wall-clock:** ~164 seconds
- **Peak RAM:** ~2.2 GB
- **CPU only, no network** during `rank.py` (loads `.npy` artifacts only)

Precompute is separate and took ~20 minutes on CPU for 100K candidates.

## Project layout

```
scripts/
  eda.py              Stage 1 — exploratory data analysis
  features.py         Stage 2 — rule-based sub-scores
  honeypot_check.py   Stage 3 — impossible profile detection
  embeddings.py       Stage 4 — MiniLM precompute + load helpers
  behavioral.py       Stage 5 — engagement multiplier
  rank.py             Stage 6 — final ranker entry point
  reasoning.py        Stage 7 — CSV reasoning strings
app.py                Streamlit demo for ≤100 candidates
data/
  candidates.jsonl    Full dataset (100K)
  sample_candidates.json  50-candidate demo sample
  embeddings/         Precomputed vectors (generated, gitignored)
```

## Streamlit demo

```bash
streamlit run app.py
```

Uses built-in `data/sample_candidates.json` or an uploaded JSON/JSONL (max 100 candidates). Embeds on the fly for the sample; same scoring code as production.

## Design notes

- Generic engineering titles (Software Engineer, Backend) get a **medium-low title prior** but can rank highly if `production_ml_experience` finds real ranking/search work in career text.
- **Skills are never used for positive scoring** — only career descriptions and titles.
- Honeypot rate in top 100: **0** on local test run (spec limit is 10%).
