# Smallweb Browser — Codebase Journal

A living log of how this codebase was built, what decisions were made, and what tools were used.
Used by codemap sync to understand context when diagram edits are made.

---

## Build History

### Initial Creation (Jan 30, 2026)
- **Author**: xi + claude (sonnet)
- **Method**: iterative pair programming via Discord bot
- **No scaffolding tools used** — all code written directly, no frameworks, no generators

### Core Files

| File | How it was built | Notes |
|------|-----------------|-------|
| `smallweb.py` | Hand-written Python, iterative | Core WebGraph class, crawler, ranking. ~1400 lines. Built over multiple sessions. |
| `server.py` | Hand-written aiohttp | REST API, ~750 lines. Added endpoints incrementally as features were built. |
| `search.py` | Hand-written | Three search backends (fuzzy, FTS5, semantic). Added semantic search later. |
| `taste.py` | Hand-written | Sentence-transformers + logistic regression for personalized ranking. |
| `scoring_config.py` | Hand-written | Tunable scoring parameters with presets. Added after initial ranking was too rigid. |
| `index.html` | Claude Code (Opus 4.5), front end web dev skill | Single-page app, no framework, no build step. Minimal monospace UI. |

### Key Design Decisions
- **No frameworks** — pure Python stdlib + aiohttp for async. No Django, no Flask for the main app.
- **SQLite for search** — FTS5 for full-text, in-memory for fuzzy. No external search engine.
- **Sentence-transformers** — `all-MiniLM-L6-v2` for embeddings. Local, no API calls.
- **JSON for storage** — graphs saved as portable JSON files with optional sidecar files (.taste.json, .config.json).
- **No authentication** — local-first tool, runs on localhost.

### Architecture Notes
- The crawler is async (aiohttp) with domain diversity priority to avoid hammering single domains
- PageRank uses personalized teleport back to seed URLs (not uniform random)
- Quality scoring penalizes tracker scripts, ads, and rewards IndieWeb signals (webmentions, microformats)
- The "smallweb score" is a bell curve that peaks at 3-8 inbound domains — not too obscure, not too mainstream

---

## Changelog

### 2026-01-30
- Initial implementation: WebGraph, crawler, basic ranking
- Server with REST API
- Frontend SPA

### 2026-01-31
- Added taste model (sentence-transformers + logistic regression)
- Added scoring config with presets
- Added co-citation similarity
- Frontend: similarity explorer, fork graphs

### 2026-02-01
- Added semantic search backend
- Search module refactored into modular classes (FuzzySearch, FTS5Search, SemanticSearch)
- Auto graph selection for search queries

### 2026-02-02
- Codemap v2 created — visual diagram editor mapped to this codebase
- 4 views: flow (opus), spec (sonnet), data flow (handwritten), granular (haiku)

### 2026-02-03
- Codemap: added structured logging, Discord notifications, cross-session context
- Codemap: created seed prompt for persistent knowledge
