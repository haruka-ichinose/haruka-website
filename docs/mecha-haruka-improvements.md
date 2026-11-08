# mecha-haruka — Making the Chatbot Actually Helpful

> Extends `docs/mecha-haruka-rag-chatbot.md` (the original design). Written 2026-08-10,
> after the chatbot went public on Vercel. **Code for Phases 1–5 implemented 2026-11-08**
> (uncommitted). Remaining: review `llm/content/profile.md` TODOs, rebuild the Pi index,
> run the evals, redeploy (Verification steps 2–8).

## Context

mecha-haruka is live and technically working, but it refuses or deflects most questions.
Three root causes, all fixable:

1. **The corpus is too narrow.** ~115 rows in `chunks`, of which only 13 are non-journal
   (6 projects, 4 experience, 3 route docs). There is nothing about Haruka as a person —
   skills, stack, availability, career story, how to get in touch — and nothing about the
   repositories beyond a bare URL. Any question outside "what did you write in the journal"
   retrieves noise.
2. **The system prompt is a refusal machine.** `backend/app/services/rag_service.py:24-39`
   says "Answer ONLY using the provided context" and "Politely decline questions that are
   off-topic, **personal**…". Since almost everything a visitor asks is personal-professional,
   the model declines correctly-but-uselessly.
3. **Its one escape hatch is broken.** The prompt tells the model to "suggest a relevant page
   (e.g. /experience)", but `frontend/components/chat/chat-message.tsx:52` renders answers as
   `whitespace-pre-wrap` plain text — so `/experience` in prose is dead text, not a link. The
   "navigate users to the right page" goal is undelivered.

**Outcome:** a bot that answers professional-profile, project, repo and journal questions from
real grounded context; when it can't answer, it says what it *does* know, offers concrete
follow-up questions, and links the right page as a clickable route.

Scope decisions: add professional profile + light personal/character + career narrative +
site/bot FAQ content; pull repo context from **public GitHub READMEs**; keep the bot
**strictly grounded** (guide, never guess).

---

## Phase 1 — Widen the corpus

### 1a. New hand-authored profile content

Create `llm/content/profile.md` — a single markdown file, `##` sections. The existing chunker
(`llm/mecha_llm/chunking.py`) already splits on headings and prepends the heading to every chunk,
so one section ≈ one well-labelled chunk. Sections to write:

| Section | Covers |
|---|---|
| `## Skills and tech stack` | Languages, frameworks, cloud, AI/ML tooling, what you're strongest in |
| `## About Haruka` | Short bio, location/timezone, work eligibility, languages spoken |
| `## How Haruka works` | Working style, how you approach problems, what you're currently learning |
| `## Background and how I got into tech` | The narrative — physics → CS → AI engineering |
| `## Career direction` | What kind of work you want next, strengths, interests |
| `## Contact` | Email / LinkedIn / GitHub and the contact form on the site |
| `## About this site` | Stack (Next.js 16, FastAPI, Vercel, Pi-hosted RAG), why it exists |
| `## About Mecha Haruka` | What the bot is, what it can answer, what it won't, that it's an AI persona |

New loader `llm/mecha_llm/sources/profile.py` — mirror `sources/content_json.py`: read the file,
split on `## ` into one `SourceDoc` per section, `source_type="profile"`, `url="/"` (or
`/experience` for the career sections), `doc_id=f"profile-{slug}"`. Reuse the `_slugify` idiom
from `content_json.py`. Path configurable via a new `profile_md_path` setting in
`llm/mecha_llm/config.py` (default `"content/profile.md"`).

### 1b. Public GitHub READMEs

New `llm/mecha_llm/sources/github_repos.py`, modelled directly on
`llm/mecha_llm/sources/journals.py` (same `aiohttp` + `_headers()` + `_get_json` pattern,
reuse `settings.github_token`):

- **Repo list**: union of every `links[].url` with `type == "github"` in
  `frontend/lib/content.json` (parse with the existing `_first_github_link` logic, generalised
  to all links) plus a new `extra_repos: list[str]` config field. This auto-syncs as projects
  are added and still allows repos not showcased on the site.
- For each `owner/name`: `GET /repos/{owner}/{name}` for description / topics / primary language,
  and `GET /repos/{owner}/{name}/readme` (base64, same decode as `_get_content`). Skip 404s with
  a warning — the pipeline must not fail on a renamed or private repo.
- Build one `SourceDoc` per repo: title `f"{name} (GitHub repository)"`, text =
  `description + "Language: … · Topics: …" + README body`, `source_type="repo"`,
  `repo_url=html_url`, `url="/"` when the repo is linked from content.json (so the citation chip
  navigates to the Projects page) else `html_url`.
- Cap README size (~8k chars) before chunking so a monster README can't dominate retrieval.

### 1c. Schema + type plumbing for the two new source types

- `llm/mecha_llm/sql/002_source_types.sql` (new): drop and re-add the `source_type` CHECK to
  include `'profile'` and `'repo'`. Needed because `001_chunks.sql` is `CREATE TABLE IF NOT
  EXISTS` — editing it in place will not migrate the existing Pi table.
- `llm/mecha_llm/pg_store.py:23` — replace the single-file `_SCHEMA_SQL` read with all
  `sql/*.sql` applied in sorted filename order, so future migrations are just new files.
- `llm/mecha_llm/documents.py` — add `"profile"`, `"repo"` to `SOURCE_TYPES`.
- `backend/app/models/chat.py` — add both to the `Citation.source_type` `Literal`.

### 1d. Wire into the build

`llm/mecha_llm/build.py:32-51` — add `profile_docs()` and `fetch_repo_docs()` to
`gather_source_docs()` alongside the existing three, with the same `logger.info` count lines.
Add a `--skip-repos` flag next to `--skip-journals` (both need `GITHUB_TOKEN`).

**Files:** `llm/content/profile.md`, `llm/mecha_llm/sources/profile.py`,
`llm/mecha_llm/sources/github_repos.py`, `llm/mecha_llm/sql/002_source_types.sql`,
`llm/mecha_llm/{build,config,documents,pg_store}.py`, `backend/app/models/chat.py`.

---

## Phase 2 — Retrieval tuning

Small, cheap changes in `backend/app/core/adapters/pgvector_search.py` and config:

1. **`RAG_TOP_K` 5 → 8.** Config default in `backend/app/core/config.py`; also bump
   `DEFAULT_TOP_K` in `llm/mecha_llm/eval/rag.py` so eval matches production.
2. **Cap chunks per document at 2.** Today a single long journal entry can occupy all slots.
   Add `c.doc_id` to the `_HYBRID_SQL` `SELECT` list, then post-filter in Python after fusion:
   keep chunks in score order, skipping any whose `doc_id` already has 2 entries, until `top_k`.
   ~6 lines, no SQL restructuring. This is what buys source diversity from the wider corpus.
3. **Cap citations at 4** in `RagService._citations` (already deduped by `url`) so the chips stay
   scannable and context precision isn't dragged down by unused sources.

---

## Phase 3 — Rewrite the system prompt

Replace `SYSTEM_PROMPT` in `backend/app/services/rag_service.py:24-39`. Keep the grounding rule
(faithfulness is currently 0.88 — protect it) but change the failure behaviour and remove the
"decline personal questions" rule that now fights the new content. New prompt covers:

- **Persona + scope**: AI persona of Haruka (not Haruka); knows Haruka's projects, work and
  education history, dev journal, public repositories, skills, background, and this site.
- **Grounding**: answer from the provided context only; never invent projects, employers, dates,
  repos, or links. If context is partial, answer the part that's supported and name the gap.
- **Never dead-end** (the key change): do not reply with a bare "I don't have that information".
  Always (a) state the closest thing the context *does* cover, (b) offer 2–3 specific example
  questions the visitor could ask instead, and (c) link the relevant page.
- **Link format**: write site routes as markdown links — `[Experience](/experience)`,
  `[dev journal](/journal)` — because the UI renders markdown (Phase 4).
- **Decline only** genuinely harmful, abusive, or clearly unrelated requests, and anything
  personal beyond what the context contains. Professional and light-personal questions are
  in scope.
- **Tone**: concise, warm, professional; 1–3 short paragraphs; a closing suggested question when
  the answer is thin.

Also broaden `REFUSAL` (`rag_service.py:42`) to list what the bot *can* discuss.

---

## Phase 4 — Frontend: make guidance clickable

1. **Render assistant messages as markdown.** `frontend/components/chat/chat-message.tsx:52` —
   keep the user bubble as plain text; render `message.content` for assistant messages with
   `react-markdown` + `remark-gfm` (**already dependencies**, used by
   `frontend/app/journal/[id]/page.tsx` — copy that usage). Supply a custom `a` component:
   `href.startsWith("/")` → `next/link` (client-side nav, matching the citation-chip logic in
   `chat-message.tsx:11`), otherwise `<a target="_blank" rel="noopener noreferrer">`. Add
   `prose`-ish Tailwind classes for `p`/`ul`/`li`/`strong` spacing inside the small bubble.
2. **Broaden the starter prompts.** `frontend/components/chat/chat-widget.tsx:10-14` — replace
   the 3 hard-coded suggestions with ~6 spanning the new capabilities, e.g. "What's Haruka's tech
   stack?", "Tell me about Haruka's AI engineering work", "Which projects have public repos?",
   "How was this chatbot built?", "How do I get in touch?", "What has Haruka written about
   Next.js and FastAPI?".
3. **Add a one-line capability hint** above the suggestions on the empty state ("Ask about
   Haruka's projects, experience, skills, repos, or dev journal") so users see the scope before
   typing.

---

## Phase 5 — Refresh evaluation

- **Fix prompt drift.** `llm/mecha_llm/eval/rag.py:27-40` holds a stale copy of the system prompt
  (missing the citation bullet) and formats context without the `[n] title (url)` header that
  `RagService._build_messages` uses. Make both byte-identical to the backend and add a
  cross-reference comment in each file — the two uv projects can't import each other.
- **Extend the golden set.** `llm/mecha_llm/eval/dataset.py` — add ~8 items covering the new
  surface: skills/stack, "tell me about Haruka", career direction, contact, "what can you help
  with", a repo-README detail, plus two deliberately unanswerable questions whose `reference` is
  the *guiding* behaviour ("says it doesn't know, points at /journal, suggests what to ask").
- **Assert on `expected_url`.** It is declared in `dataset.py` and never checked. Add `urls` to
  `RagResult` in `eval/rag.py` and report a "route hit rate" (fraction of items whose
  `expected_url` appears in the retrieved urls) alongside the RAGAS scores.
- **Re-baseline.** The latest run in `llm/eval_results.json` (2026-08-07) is
  faithfulness 0.880 ✓ / answer_relevancy 0.602 ✗ / context_precision 0.519 ✗ /
  context_recall 0.667 ✓ — two metrics already fail their gates. Re-run after Phases 1–3 and
  record the new numbers; the wider corpus + guiding prompt should lift answer_relevancy and
  recall, and precision needs re-judging at `top_k=8`. Adjust thresholds in
  `eval/ragas_eval.py` only after seeing the numbers.

---

## Verification

1. **Ingestion (offline, no DB):** `cd llm && uv run python -m mecha_llm.build --dry-run` —
   log lines should show profile docs and repo docs alongside journals/content/routes, and a
   total chunk count meaningfully above 115.
2. **Index rebuild:** `cd llm && uv run python -m mecha_llm.build` against the Pi's
   `DATABASE_URL`, then
   `psql "$DATABASE_URL" -c "SELECT source_type, count(*) FROM chunks GROUP BY 1 ORDER BY 2 DESC;"`
   — expect `profile` and `repo` rows present and journal no longer ~90% of the table.
3. **Retrieval sanity:** query for "tech stack" and "how do I contact Haruka" — a `profile`
   chunk should rank top; "apple thinning VR" should return the repo README chunk with
   `repo_url = https://github.com/uc-vision/apple-thinning`.
4. **Backend local:** `cd backend && uv run uvicorn app.main:app --reload`, then
   `curl -N -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"message":"What is Haruka'\''s tech stack?"}'`
   — a grounded streamed answer, not a refusal. Repeat with a deliberately unanswerable question
   ("What did Haruka do in 2015?") and confirm the reply names what it *does* know, suggests
   follow-ups, and includes a markdown route link.
5. **Frontend:** `cd frontend && bun run dev` — open the widget; the new suggestion chips appear;
   an answer containing `[Experience](/experience)` renders as a clickable link that navigates
   client-side without closing the widget; citation chips still render; check light + dark.
6. **Guardrails unchanged:** `cd llm && uv run python -m mecha_llm.eval.redteam` still passes —
   the looser prompt must not open a jailbreak.
7. **Eval:** `cd llm && uv run python -m mecha_llm.eval.ragas_eval` (must run from `llm/` — it
   writes a relative `eval_results.json`) and `uv run pytest mecha_llm/eval/deepeval_tests.py`.
   Compare against the 2026-08-07 baseline.
8. **Deployed:** redeploy the Pi backend + Vercel frontend; confirm the same two questions work
   against the public URL and SSE still streams unbuffered through the Cloudflare Tunnel proxy.

---

## Out of scope (noted, not done here)

- History/rate limiting: unbounded chat history is still forwarded to Azure OpenAI with no cap
  and there is no rate limit. Worth a follow-up now that the bot is publicly reachable.
- `from torch import Value` — a stray unused import in `llm/mecha_llm/embeddings_provider.py:13`.
- The dead Azure AI Search path (`llm/mecha_llm/index.py`, `Chunk.to_search_document`).
