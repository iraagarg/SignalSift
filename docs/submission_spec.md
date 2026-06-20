# Submission Specification — Redrob Hackathon v4

Submissions that don't match this spec are auto-rejected by the validator, no scoring.

## 1. What you're submitting

A CSV ranking the **top 100 candidates** from candidates.jsonl for the released job description. Rank 1 = best fit, rank 100 = 100th best fit. Do not rank candidates 101+.

## 2. File format

- Filename: your participant ID + `.csv` (e.g. `team_xxx.csv`)
- Encoding: UTF-8
- Required columns, in this exact order: `candidate_id,rank,score,reasoning`

| Column | Type | Required | Description |
|---|---|---|---|
| candidate_id | string | Yes | CAND_XXXXXXX from candidates.jsonl |
| rank | int 1-100 | Yes | Each integer 1-100 exactly once |
| score | float | Yes | Monotonically non-increasing as rank increases |
| reasoning | string | Strongly recommended | 1-2 sentence justification, used in manual review |

## 3. Rules

- Exactly 100 data rows + 1 header row.
- Each rank 1-100 exactly once. Each candidate_id exactly once. Every candidate_id must exist in candidates.jsonl.
- score non-increasing with rank (rank 1 score >= rank 2 score >= ... >= rank 100 score). Ties allowed.
- Tie-break: secondary model signal, or candidate_id ascending.

### Compute constraints (the ranking step itself must satisfy ALL of these)

| Constraint | Limit |
|---|---|
| Total runtime | ≤ 5 minutes wall-clock |
| Memory | ≤ 16 GB RAM |
| Compute | CPU only — no GPU |
| Network | OFF — no external API calls (no OpenAI, Anthropic, Cohere, Gemini, or any hosted LLM/embedding service) during ranking |
| Disk | ≤ 5 GB intermediate state |

Pre-computation (e.g. generating embeddings ahead of time) is allowed and can take as long as needed, and does NOT count against the 5-minute budget — but it must be clearly documented/scripted, and the actual ranking step that produces the CSV from the candidate file must complete within the budget above using only pre-computed artifacts.

At Stage 3, top-N submissions must reproduce their ranking step inside a sandboxed Docker container enforcing these exact limits. Cannot reproduce within limits → disqualified, regardless of score.

### Three-submission cap

Max 3 submissions total. Last valid submission is final. No live leaderboard, no per-submission feedback.

### Reasoning column — graded criteria (Stage 4 samples 10 random rows)

- **Specific facts**: references real profile facts (years, title, named skills, signal values)
- **JD connection**: connects to specific JD requirements, not generic praise
- **Honest concerns**: acknowledges obvious gaps/concerns where present
- **No hallucination**: every claim must correspond to something actually in the candidate's profile
- **Variation**: 10 sampled reasonings must be substantively different, not templated
- **Rank consistency**: tone must match rank (a rank-5 candidate shouldn't have critical reasoning; a rank-95 candidate shouldn't have glowing reasoning)

Penalized: empty reasoning, all-identical strings, templated reasoning with just the name swapped, hallucinated skills/employers, reasoning that contradicts the rank.

## 4. Scoring

| Metric | Weight | Measures |
|---|---|---|
| NDCG@10 | 0.50 | Quality of top-10 picks |
| NDCG@50 | 0.30 | Quality of top-50 picks |
| MAP | 0.15 | Precision across all relevance levels |
| P@10 | 0.05 | Fraction of top-10 that are "relevant" (tier 3+) |

Final composite = 0.50×NDCG@10 + 0.30×NDCG@50 + 0.15×MAP + 0.05×P@10. Computed once, after submissions close, against hidden ground truth.

Tiebreaks: higher P@5 wins → higher P@10 wins → earlier submission timestamp wins.

## 5. Evaluation pipeline stages

1. **Format validation** — auto-validator. Any spec violation in Section 3 → rejected.
2. **Scoring** — composite computed once on full hidden ground truth.
3. **Code reproduction + honeypot check** — top-N: full repo requested, ranking step reproduced in sandbox (5min/16GB/CPU/no network). Honeypot rate computed. Eliminated if: can't reproduce within limits, honeypot rate >10% in top 100, missing/fabricated repo.
4. **Manual review** — reasoning quality (6 checks above), methodology coherence, git history authenticity (real iteration vs single dump), code quality. Eliminated if: failed reasoning checks, flat git history with no iteration, codebase is entirely LLM API calls.
5. **Defend-your-work interview** — top finalists, 30-min call. Eliminated if: can't explain architecture, contradicts submitted code, clearly didn't build it.

AI tool use (Claude, GPT-4, etc.) is explicitly allowed and expected. The pipeline is designed so AI-assisted work with real human engineering succeeds, while mostly-LLM-output submissions with minimal human engineering fail at Stages 3-5.

## 6. Common rejections to avoid

99 or 101 rows instead of 100. Ranks starting at 0. Duplicate candidate_ids. Candidate_id typos not in the dataset. All scores identical (no differentiation). Scores increasing with rank. Submitted as .xlsx/.json instead of .csv.

## 7. Honeypot warning

~80 honeypot candidates have subtly impossible profiles (e.g., 8 years experience at a company founded 3 years ago; "expert" proficiency in 10 skills with 0 duration_months used). These are forced to relevance tier 0 in ground truth. >10% honeypot rate in your top 100 → disqualified at Stage 3. Detectable through careful profile inspection — a good ranking system should naturally avoid them without special-casing.

## 8. Leaderboard policy

Hidden during the competition. No feedback until final results. Validate locally via your own methodology, not by burning submissions.

## 10. Full submission package (all required)

1. **The CSV** (Sections 2-3)
2. **Portal metadata**: team name, primary contact (name/email/phone), GitHub repo URL (reachable; private OK if access can be granted at Stage 3), sandbox/demo link, AI tools declared (multi-select, honest, not penalized), compute environment summary, team member list, methodology summary (optional, ≤200 words, recommended).
3. **Code repository** must include:
   - README.md with setup instructions and the exact single command to reproduce the submission CSV from candidates.jsonl
   - Full source code (no hidden/manual steps)
   - Pre-computed artifacts (embeddings/indexes/weights) or the script that produces them
   - requirements.txt / pyproject.toml with exact dependency versions
   - submission_metadata.yaml at repo root mirroring portal metadata
4. **AI tools declaration** — transparency only, not penalized. Declaration must match what your code/interview actually shows.
5. **Sandbox/demo link** — a hosted environment (HuggingFace Spaces, Streamlit Cloud, Replit, Colab notebook, public Docker image, or Binder) that: accepts a small sample (≤100 candidates), runs the ranking system end-to-end producing a ranked CSV, completes within the compute budget. Does NOT need to handle the full 100K pool. Missing/non-working sandbox → flagged at Stage 1.
