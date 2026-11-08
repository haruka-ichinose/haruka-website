# Self-Hosting Migration Plan: Azure → Raspberry Pi 5

## Context

Azure AI Search is the expensive piece of the mecha-haruka RAG chatbot. This plan
replaces it with a self-hosted RAG stack on an 8GB Raspberry Pi 5 — the FastAPI
backend and a local embedding model — reached from the Vercel-hosted site over an
authenticated Cloudflare Tunnel. The marketing site, journal, and contact API stay
on Vercel (free tier). PostgreSQL + pgvector runs in a Docker container on the Pi,
colocated with the RAG service (Neon's free tier kept as a fallback if the 8GB
board's RAM tightens). Azure OpenAI stays only for chat generation (a local LLM on
an 8GB Pi is not practical; revisit later).

This is deliberately a **learning project**. Each phase lists what you'll learn,
the work involved, and a checkpoint to verify before moving on. The centerpiece
is **implementing HNSW from scratch** and eventually serving real queries with it.

## Decisions

1. **Hosting shape**: the site (frontend + journal/contact API) stays on Vercel;
   the mecha-haruka RAG service (backend + embedder + pgvector, optional local LLM
   later) runs on the Pi, reached over an authenticated Cloudflare Tunnel with
   Vercel proxying `/chat` to it. Tradeoff: chat availability depends on home
   power/internet, but the rest of the site stays up on Vercel if the Pi is down.
2. **Database**: PostgreSQL + pgvector in a Docker container **on the Pi**,
   colocated with the RAG service, with **Neon (free tier) kept as a fallback**
   if RAM tightens. Running it on the Pi has sysadmin learning value and keeps
   hybrid-search queries on localhost rather than round-tripping over the
   internet. Watch the RAM: an 8GB Pi has limited headroom once the app processes
   and the embedding model land, and pgvector's HNSW index *builds* spike well
   above the steady-state RSS in the table below — if it gets tight, evict
   Postgres to Neon (the `chunks` data is derived, so it's same DDL + a
   `DATABASE_URL` swap + a re-run of the build). Either way the Phase 1 learning
   goals (vector column type, distance operators, HNSW index limits) hold, and
   the from-scratch HNSW work in Phase 6 is pure Python/numpy that reads vectors
   out of Postgres regardless of where it's hosted.
3. **Embeddings**: local `BAAI/bge-small-en-v1.5` (384 dims) via
   sentence-transformers, replacing `text-embedding-3-large` (3072 dims — which
   exceeds pgvector's 2000-dim HNSW index limit anyway). Full corpus re-embed
   required; the corpus is small, so this takes minutes.
4. **HNSW scope**: build in pure Python inside `llm/` with tests and benchmarks
   (recall vs brute force, vs pgvector's HNSW), then wire it into the serving
   path for real behind a `VECTOR_BACKEND=hnsw_scratch|pgvector` config flag.
5. **Generation**: stays hosted on Azure AI Foundry (`DeepSeek-V4-Flash`, swapped
   from `gpt-4o` for cost). Only retrieval and embeddings move.

## Current architecture (what's being replaced)

- **Online path**: `backend/app/services/rag_service.py` embeds the query via
  `backend/app/core/adapters/azure_openai.py`, then calls
  `backend/app/core/adapters/ai_search.py:hybrid_search()` — Azure AI Search
  hybrid retrieval (keyword + vector + semantic reranker). Clients are wired in
  app state via `backend/app/core/factory.py`; config in
  `backend/app/core/config.py`.
- **Offline pipeline** (`llm/mecha_llm/`): `build.py` orchestrates ingest
  (`sources/`) → `chunking.py` → `embeddings.py` (Azure OpenAI, batched) →
  `index.py` (Azure AI Search index + upsert). Shared dataclasses in
  `documents.py` (`SourceDoc`, `Chunk`).
- **Eval harness**: `llm/mecha_llm/eval/` (RAGAS, DeepEval) — reused here to
  compare retrieval quality before and after migration.

## RAM budget (8GB Pi 5)

Postgres starts on the Pi (see Phase 1); Neon is the pressure valve if memory
gets tight once the app tier lands in Phase 2+. Watch `free -h` and
`docker stats` as services are added.

| Service                        | Approx. RSS |
| ------------------------------ | ----------- |
| FastAPI backend                | ~150 MB     |
| bge-small embedding model      | 300–400 MB  |
| Next.js production server      | 300–500 MB  |
| cloudflared                    | ~50 MB      |
| Postgres + pgvector (Pi phase) | 100–200 MB steady; capped at 1 GB (`mem_limit`) — HNSW builds spike |
| **Total**                      | **< 2.5 GB**|

---

## Phase 0 — Pi preparation

**Goal**: a hardened, containerized base to build on.

**What you'll learn**: headless Linux server setup, SSH hardening, Docker on ARM64.

- Raspberry Pi OS Lite (64-bit), static LAN IP or DHCP reservation.
- SSH key-only auth, disable password login; `ufw` allowing only SSH from LAN.
- Install Docker + Docker Compose plugin. (Compose chosen over k3s for a single
  8GB node — you already know k8s from the Azure migration; k3s is a valid
  alternative if you want to keep exercising it, at the cost of ~500 MB overhead.)

**Checkpoint**: `docker run --rm hello-world` works over SSH; reboot survives.

## Phase 1 — PostgreSQL + pgvector (on the Pi; Neon as pressure valve) ✅ DONE (2026-07-19)

**Goal**: a vector-capable Postgres reachable from both the laptop and the Pi.

**What you'll learn**: running Postgres in Docker (volumes, persistence,
memory caps), the `vector` column type, distance operators (`<=>` cosine,
`<->` L2, `<#>` negative inner product), why normalized embeddings make
cosine equivalent to inner product, and pgvector's 2000-dim HNSW index limit.

**Decision (2026-07-12)**: start with Postgres on the Pi rather than Neon.
The chunks data is derived (rebuilt by the ingestion pipeline), so moving to
Neon later is a `DATABASE_URL` swap + re-run of ingestion — not a real
migration. Running it on the Pi adds sysadmin learning; if RAM gets tight
once the embedding model and app containers land (Phase 2+), evict to Neon
then. Watch `free -h` / `docker stats` at each phase.

- Run `pgvector/pgvector` (arm64) via Docker Compose on the Pi: named volume
  for `/var/lib/postgresql/data`, `mem_limit: 1g`, conservative
  `maintenance_work_mem` (HNSW index builds spike well above steady-state).
  Expose 5432 to the LAN only (ufw already default-denies; allow laptop access).
- Enable the `vector` extension (`CREATE EXTENSION vector;`).
- Create the `chunks` table mirroring the Azure index schema
  (`llm/mecha_llm/documents.py`): `id`, `doc_id`, `chunk_index`, `content`,
  `title`, `source_type`, `url`, `repo_url`, `published_on`,
  `embedding vector(384)`, plus a generated `tsvector` column over
  `title + content` with a GIN index for the keyword leg of hybrid search.
- Add the pgvector HNSW index (`USING hnsw (embedding vector_cosine_ops)`).
- `DATABASE_URL` points at the Pi (LAN address from the laptop,
  `localhost`/service name on the Pi). If/when evicting to Neon: create the
  project, run the same DDL, swap `DATABASE_URL`, re-run ingestion.

**Checkpoint** ✅: `SELECT '[1,2,3]'::vector(3) <=> '[3,2,1]'::vector(3);` returns
a distance from `psql` on the laptop, pointed at the Pi; data survives
`docker compose down && up` and a Pi reboot. Verified 2026-07-19 — cosine
distance query ran from the Mac over LAN; `chunks` survived container
recreate and a full Pi reboot.

## Phase 2 — Local embeddings

**Goal**: replace Azure OpenAI embeddings with a local model.

**What you'll learn**: embedding dimensions and normalization, quality/size
tradeoffs (MTEB), why bge models want a query prefix
(`"Represent this sentence for searching relevant passages: "`), ONNX runtime
for faster ARM inference.

- New `llm/mecha_llm/embeddings_local.py` alongside the existing Azure module;
  a config field (`embedding_provider`) selects between them, and
  `embedding_dimensions` drops to 384.
- Same `embed_texts` / `embed_chunks` interface so `build.py` doesn't change.

**Checkpoint**: embedding the same sentence twice yields identical normalized
384-dim vectors; a similar pair scores closer than a dissimilar pair.

## Phase 3 — Rebuild pipeline → pgvector

**Goal**: `mecha_llm.build` writes to the Pi's Postgres instead of Azure AI Search.

**What you'll learn**: idempotent SQL upserts, asyncpg, schema-as-code.

- New `llm/mecha_llm/pg_store.py` taking over `index.py`'s role: ensure schema,
  `INSERT ... ON CONFLICT (id) DO UPDATE` keyed on chunk id.
- Run the build from the laptop against the Pi over LAN (`DATABASE_URL`
  pointing at the Pi).

**Checkpoint**: row count in `chunks` matches the chunk count logged by the
build; spot-check a journal chunk's content and metadata in `psql`.

## Phase 4 — Backend retrieval swap

**Goal**: the chatbot retrieves from pgvector instead of Azure AI Search.

**What you'll learn**: hybrid search without a managed reranker — vector
similarity + Postgres full-text (`websearch_to_tsquery`) fused with
**Reciprocal Rank Fusion (RRF)**; what Azure's semantic reranker was doing and
what is lost without it.

- New adapter `backend/app/core/adapters/pgvector_search.py` exposing the same
  chunk-dict shape as `ai_search.hybrid_search()` so `rag_service.py` barely
  changes; query embedding comes from the local model instead of
  `azure_openai.embed_query`.
- Rewire `main.py` app state + `factory.py`; update `config.py`
  (`database_url`, `chat_configured` no longer gates on Azure Search creds).

**Checkpoint**: run backend locally against the Pi's Postgres over LAN; `/chat`
answers with correct citations. Run the `llm/mecha_llm/eval/` RAGAS suite and compare
retrieval metrics against the Azure baseline (record the baseline **before**
tearing anything down).

## Phase 5 — RAG service on the Pi; site stays on Vercel

**Goal**: the mecha-haruka RAG pipeline (embed → retrieve → generate) runs on
the Pi and is reachable from the Vercel-hosted site over an authenticated HTTPS
tunnel. The marketing site, journal, and contact API stay on Vercel.

**Why this shape (revised from "whole site onto the Pi")**: the retrieval path
needs the local embedding model (torch / sentence-transformers) and a live
Postgres connection — neither belongs on Vercel serverless (cold starts, bundle
size limits, no persistent DB connection to a home network). Since the embedder,
pgvector, and (optionally, later) a local LLM all live on the Pi, the whole RAG
service belongs there; Vercel just proxies `/chat` to it. Bonus: if the Pi or
home ISP is down, the site stays up on Vercel and only chat degrades — clean
failure isolation the all-on-Pi design didn't have.

**What you'll learn**: arm64 Docker builds, exposing a single home service via
Cloudflare Tunnel (HTTP ingress, no open router ports), service-to-service auth
(Cloudflare Access service token / shared bearer), and proxying a streaming SSE
response through a serverless function.

**Topology**:
```
Browser → Vercel (static site + journal/contact API)
        → Vercel /chat proxy ── HTTPS (Cloudflare Tunnel) ──▶ Pi FastAPI RAG service
                                                              (embed → hybrid_search → generate → stream)
```

- **Pi side**: Dockerfile for the FastAPI backend (arm64) — the Phase 4 code
  unchanged, just running here. Compose stack: `backend`, `cloudflared`, and the
  existing `postgres` (pgvector) container; `DATABASE_URL` points at the local
  container over the compose network (Neon remains the fallback). No `frontend`
  container — the site stays on Vercel.
- **Cloudflare Tunnel**: one public hostname (e.g. `api.<domain>`) routed to the
  local backend container. HTTP only — raw Postgres is never exposed.
- **Auth**: the Pi's `/chat` requires a shared secret (bearer token) or a
  Cloudflare Access service token that only the Vercel proxy holds; otherwise the
  endpoint is an open LLM/DB on the public internet.
- **Vercel side**: a thin `/chat` route that forwards the request to the Pi
  hostname with the auth token and streams the SSE response back to the browser
  (one origin for the client, a place to inject auth, optional Azure-generation
  fallback when the Pi is unreachable). Everything non-chat stays on Vercel
  untouched; the frontend `NEXT_PUBLIC_CHAT_ENABLED` gate is unchanged.
- **Secrets** via an env file on the Pi; restart policies; a simple `deploy.sh`
  (git pull → build → `docker compose up -d`).

**DNS**: the site's DNS stays on Vercel. Only a single `api.` subdomain routes
through Cloudflare to the Pi — no domain-wide cutover here.

**Checkpoint**: with `NEXT_PUBLIC_CHAT_ENABLED=true`, chat works end-to-end from
the production Vercel site, streaming tokens sourced from the Pi; a direct call
to the Pi `/chat` without the auth token is rejected; pulling the Pi's Ethernet
drops chat while the rest of the site keeps serving on Vercel, and restoring it
recovers chat without intervention.

## Phase 6 — HNSW from scratch (the centerpiece)

**Goal**: understand, implement, validate, and finally serve with your own
approximate-nearest-neighbor index.

**6a. Study.** Malkov & Yashunin (2016), and the lineage: skip lists → NSW →
HNSW. Understand `M`, `ef_construction`, `ef_search`, layer assignment
(`floor(-ln(U) * mL)`), greedy `search_layer`, and the neighbor-selection
heuristic (why plain "closest M" degrades the graph).

**6b. Implement.** `llm/mecha_llm/hnsw/` in pure Python + numpy distances:
`graph.py` (structure, insert), `search.py` (layered greedy search). No FAISS,
no shortcuts.

**6c. Test.** Recall@k against brute-force exact search on the real corpus;
edge cases: empty index, k > n, duplicate vectors, single-layer graphs.

**6d. Benchmark.** Recall/latency curves vs exact scan and vs pgvector's HNSW;
a parameter-tuning table (`M`, `ef_construction`, `ef_search`). This is where
the paper's tradeoffs become tangible.

**6e. Serve for real.** The backend loads all vectors from Postgres at startup
into the scratch HNSW and serves ANN queries from it — Postgres stays the
source of truth, rebuild-on-boot is the honest persistence story for a small
corpus. `VECTOR_BACKEND=hnsw_scratch|pgvector` selects the backend; pgvector
remains the fallback.

**Checkpoint**: scratch HNSW serves production chat with recall@5 ≥ 0.95 vs
exact search, and the eval suite shows no quality regression vs pgvector.

## Phase 7 — Cutover & decommission

**Goal**: stop paying.

- Keep the Vercel-hosted site; no domain-wide DNS cutover (only the `api.`
  subdomain routes to the Pi, set up in Phase 5). The Vercel project stays.
- Re-run the eval suite one final time against production; archive results
  next to the Azure-baseline run.
- Decommission Azure AI Search; remove Azure OpenAI *embedding* deployment
  (keep the chat deployment); delete unused Azure resources from the earlier
  AKS migration if still running.
- Observability: the OpenTelemetry → Azure Monitor export can stay (free tier)
  or move to self-hosted Grafana/Prometheus later — out of scope here.

---

## Cost before / after

| Item                 | Before                    | After                     |
| -------------------- | ------------------------- | ------------------------- |
| Azure AI Search      | ~basic tier $$            | $0                        |
| Embeddings           | per-token (small)         | $0 (local)                |
| Chat generation      | per-token                 | per-token (unchanged)     |
| Hosting (app tier)   | Vercel free tier          | $0 + Pi power (~2–5 W)    |
| Database             | —                         | $0 (pgvector on the Pi)   |
| Cloudflare Tunnel    | —                         | $0 (free plan)            |

## Suggested order of work

Phases 0→5 are the migration (each independently verifiable; the site keeps
running on Vercel until Phase 5's cutover). Phase 6 can start any time after
Phase 3 (it only needs vectors in Postgres) and is the deep-learning arc.
Phase 7 only after the eval comparison is recorded.
