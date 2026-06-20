# Job Description: Senior AI Engineer — Founding Team

**Company:** Redrob AI (Series A AI-native talent intelligence platform)
**Location:** Pune/Noida, India (Hybrid) | Open to relocation candidates from Tier-1 Indian cities
**Employment Type:** Full-time
**Experience Required:** 5–9 years (range, not a hard requirement)

## What you'd actually be doing

Own the intelligence layer of Redrob's product — the ranking, retrieval, and matching systems that decide what recruiters see when they search for candidates and what candidates see when they search for roles. Ship a v2 ranking system involving embeddings, hybrid retrieval, possibly LLM-based re-ranking. Set up evaluation infrastructure (offline benchmarks, online A/B testing).

## Disqualifiers (hard filters)

- Career spent entirely in pure research environments (academic labs, research-only roles) with no production deployment.
- "AI experience" consisting primarily of recent (<12 months) projects using LangChain to call OpenAI, with no substantial pre-LLM-era ML production experience.
- Senior engineers who haven't written production code in the last 18 months because they moved into pure "architecture"/"tech lead" roles.

## Things candidates absolutely need

- Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users — handled embedding drift, index refresh, retrieval-quality regression in production.
- Production experience with vector databases / hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or similar).
- Strong Python, real code quality.
- Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation).

## Nice-to-haves (not disqualifying if absent)

- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost-based or neural)
- Prior HR-tech / recruiting tech / marketplace product exposure
- Distributed systems / large-scale inference optimization background
- Open-source contributions in AI/ML

## Explicitly NOT wanted

- **Title-chasers**: career trajectory optimizing Senior→Staff→Principal by switching companies every ~1.5 years.
- **Framework enthusiasts**: GitHub full of LangChain tutorials / "How I used [hot framework] to build [demo]" blog posts, with no systems thinking.
- **Pure consulting-only careers** (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) with zero product-company experience. (Currently at one of these but with prior product-company experience is fine.)
- **CV / speech / robotics specialists** without significant NLP/IR exposure.
- **5+ years entirely on closed-source proprietary systems** with zero external validation (no papers, talks, open-source).

## Location, comp, logistics

- Pune/Noida preferred, flexible. Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome.
- Outside India: case-by-case, **no work visa sponsorship**.
- Notice period: sub-30-day strongly preferred (can buy out up to 30 days); 30+ day candidates still in scope but bar is higher.

## The "ideal candidate" profile (read between the lines)

- 6–8 years total experience, 4–5 of which in applied ML/AI roles at **product companies** (not pure services).
- Has shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale.
- Strong, defensible opinions on retrieval (hybrid vs dense), evaluation (offline vs online), and LLM integration (fine-tune vs prompt).
- Located in or willing to relocate to Noida or Pune.
- Active on the Redrob platform / clear signal of being in the job market.

They explicitly expect this to be a **narrow profile** — maybe only ~10 great matches in the 100K pool, and that's fine. They'd rather see 10 great matches than 1000 maybes.

## Note for hackathon participants (read carefully — this defines the grading philosophy)

> The "right answer" to this JD is **not** "find candidates whose skills section contains the most AI keywords." That's a trap deliberately built into the dataset.
>
> The right answer involves reasoning about the **gap between what the JD says and what the JD means**. A candidate may never use the words "RAG" or "Pinecone" but if their career history shows they built a recommendation system at a product company, they're a fit. A candidate who lists every AI keyword as a skill but whose title is "Marketing Manager" is not a fit, no matter how perfect their skill list looks.
>
> The ranking system should also weigh behavioral signals — a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available. Down-weight them appropriately.
