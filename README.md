# smallweb

A personal web discovery engine. You give it seed URLs of sites you care about, it crawls outward N hops, builds a local link graph, and ranks discoveries using personalized PageRank multiplied by quality, smallweb-ness, and taste.

Google indexes the whole web and ranks globally. **Smallweb indexes YOUR web and ranks locally.**

Graphs are portable JSON files — fork them, merge them, share them. Like git for web discovery.

## Quick Start

```bash
pip install aiohttp beautifulsoup4

# Crawl from seed URLs (1 hop out, max 200 pages)
python smallweb.py crawl "https://wiki.xxiivv.com,https://100r.co,https://solar.lowtechmagazine.com" \
  --hops 1 --max-pages 200 --name "my-web-corner"

# See what you discovered
python smallweb.py discover graph.json --top 20

# Export as a browsable HTML page
python smallweb.py html graph.json --output discoveries.html
```

## How It Works (and Why)

Smallweb combines seven distinct components. Each one encodes a specific belief about what makes a web page worth discovering. Together they produce a compound signal that no single metric could.

### 1. Domain-Diverse BFS Crawling

**What:** The crawler follows links outward from your seeds using breadth-first search, but re-sorts the queue after each batch so that least-crawled domains get priority.

**Why breadth over depth?** Following one site deep gives you that site's worldview. Spreading across domains finds the *connections between* worldviews. The interesting stuff lives at the intersections.

**How it works:**
- Async fetching via aiohttp (10 concurrent requests)
- After each batch, the queue re-sorts by `(domain_count, depth)` — domains you've barely touched get processed first
- Adaptive domain caps: seed domains get 2.5x the base cap (you chose them), hub domains linking to 5+ diverse sites get 1.5x, everything else gets the base cap (default 20)
- Three layers of URL filtering: skip big tech/CDN domains, block 100+ platform domains from link-following (GitHub, Twitter, Medium, Wikipedia...), and regex patterns for spam (git paths, download bait, tracking params)

The result is a graph that's broad and diverse rather than deep and narrow. You see more of the web's topology, not just one corner of it.

### 2. Personalized PageRank

**What:** Standard PageRank finds what the whole graph thinks is important. Personalized PageRank finds what's important *from your corner of the web*.

**Why personalized?** Because the random walker keeps coming home to your seeds. Instead of teleporting to a random page when it gets bored, it teleports back to one of your seeds. This biases the ranking toward pages that are structurally close to what you already care about — discoveries that are "near" your seeds in link-space.

**How it works:**
- Iterative PageRank with damping factor 0.95 (high — follows links deep)
- Teleport vector is non-uniform: weighted toward seed pages
- Dangling nodes (pages with no outlinks) redistribute rank via the teleport vector
- Raw PageRank spans ~4 orders of magnitude, so it's converted to percentile rank (0-1) before combining with other signals
- Cached by `(damping, iterations, personalized)` key — tunable without re-crawling

### 3. Quality Scoring (Marginalia-Inspired)

**What:** An additive penalty/bonus system that measures how "clean" a page's HTML is. Inspired by [Marginalia Search](https://search.marginalia.nu/)'s DocumentValuator.

**Why penalize trackers and scripts?** Not because they're morally bad, but because they correlate with content optimized for extraction rather than expression. A page with 14 tracking scripts, cookie consent popups, and affiliate links was built to capture attention. A page with clean HTML and real content was built to share ideas. The quality score is a proxy for editorial intent.

**How it works:**

Base score is derived from text-to-HTML ratio — the core signal.

Penalties:
- External scripts: -0.08 each
- Known trackers (Google Analytics, Hotjar, Facebook Pixel): -0.15 each
- Ad-tech (DoubleClick, Criteo, Taboola): -0.2 each
- Cookie consent banners (OneTrust, Didomi): -0.1 each (capped at -0.3)
- Affiliate links: -0.05 each (capped at -0.25)
- AI-generated content patterns ("Ultimate Guide", "Key Takeaways"): -0.15 for 1-2 matches, -0.5 for 3+
- High link density (>0.3 links/word): -0.2; link farm (>0.5): -0.4
- Adblock circumvention: -0.6

Bonuses (reward participation in the open web's plumbing):
- Webmention support: +0.1
- IndieAuth: +0.1
- RSS/Atom feed: +0.05
- Microformats (h-card, h-entry): +0.05

Each page gets a detailed quality breakdown stored alongside its score, so you can see exactly why a page scored high or low.

### 4. The Smallweb Bell Curve

**What:** A per-domain "small-web-ness" score that peaks when a site is linked by a few people, not zero and not thousands.

**Why a bell curve?** Zero inbound links means nobody found it — possibly dead or irrelevant. Sixty-plus inbound links means it's mainstream. The interesting zone is "a few people who care linked here" — that's editorial curation, not algorithmic popularity. The sweet spot is 3-8 inbound domains.

**How it works:**

Popularity component (inbound domain count → score):
| Inbound Domains | Score | Why |
|----------------|-------|-----|
| 0 | 0.4 | Nobody found it |
| 1-2 | 0.7 | Getting noticed |
| 3-8 | 1.0 | Sweet spot — curated, not mainstream |
| 9-15 | 0.6 | Getting popular |
| 16-30 | 0.3 | Mainstream |
| 31-60 | 0.15 | Big site |
| 61+ | 0.05 | Platform-scale |

Outlink profile component:
- `ecosystem_fraction`: what fraction of outlinks point to other domains in the graph? Sites linking into the discovered ecosystem score higher than sites linking to random external stuff
- `non_platform_fraction`: what fraction avoid linking to platforms?

Combined: `popularity × (0.5 + 0.5 × outlink_score)`. Known platforms (GitHub, Twitter, etc.) are capped at 0.15 regardless.

### 5. Anchor Texts as First-Class Data

**What:** The crawler stores the visible text of every link pointing to a page — what other sites call it.

**Why?** Self-description is marketing, but how *others* describe you is reputation. It's the difference between your LinkedIn bio and what your coworkers actually say about you. A page's title says "About Us". The anchor texts from sites linking to it say "incredible writeup on local-first software" or "great resource for pixel art". Anchor texts are the web's word-of-mouth.

**How it works:**
- During crawling, every `<a>` tag's visible text is extracted and stored on the *target* node (max 20 per node)
- Anchor texts are searchable via all three search backends
- In the frontend, they appear as pills on each discovery card
- FTS5 weights anchor texts at 8x (nearly as much as titles at 10x)

### 6. Co-Citation Similarity

**What:** Two domains are "similar" if they're linked FROM the same sources. Not because they have similar content — because the same people think they're worth linking to.

**Why is this the killer feature?** Because it reveals editorial communities invisible to keyword search. If three independent niche blogs all link to the same obscure site, those bloggers are functioning as a distributed curation network. They're performing editorial judgment that no algorithm could replicate. Co-citation surfaces these hidden networks.

**How it works:**
- Each domain gets a binary inbound-link vector (which domains in the graph link to it)
- Cosine similarity: `|inbound(A) ∩ inbound(B)| / sqrt(|inbound(A)| × |inbound(B)|)`
- Single-target mode: "find sites similar to X"
- All-pairs mode: "find the most similar pair of sites in the whole graph"
- Minimum shared sources threshold (default 2) to filter noise

### 7. The Fork/Promote Loop

**What:** Crawl → discover → promote the best discoveries to seeds → re-crawl from expanded seeds → discover deeper.

**Why iterative?** Because the best seeds for round 2 are the discoveries from round 1. The graph teaches you where to look next. Each iteration pushes further into the long tail of the web, following the editorial networks you've uncovered.

**How it works:**
```bash
# Start with seeds you know
python smallweb.py crawl "https://seed1.com,https://seed2.com" --name round1

# See what bubbles up
python smallweb.py discover round1.json --top 10

# Promote top 5 discoveries to seeds, add a new one, re-crawl
python smallweb.py fork round1.json --name round2 \
  --promote-top 5 --add-seeds "https://newdiscovery.com" \
  --recrawl --hops 2 --max-pages 500
```

Forked graphs track provenance: `forked_from`, `forked_at`, `seeds_promoted`, `seeds_added`. You can trace the evolution of your discovery graph over time.

### How They Multiply Together

The final discovery score:

```
score = PageRank^exp × Quality^exp × Smallweb^exp × taste_factor × fetched_boost
```

Each component filters a different dimension:
- **PageRank** finds structural importance (what the link graph thinks matters)
- **Quality** filters noise (removes tracker-heavy, ad-laden, AI-slop pages)
- **Smallweb score** keeps it niche (penalizes mainstream, rewards the curated middle)
- **Taste** personalizes it (learns from your thumbs up/down)
- **Co-citation** reveals the hidden network (editorial communities, not keywords)

A page needs to pass ALL these filters to rank high. It needs to be structurally important in your subgraph, cleanly built, linked by a few discerning people, and match your taste. That conjunction is what makes the discoveries feel hand-picked rather than algorithmically generated.

The exponents are tunable per-graph via the scoring config system, with named presets:
- **default** — balanced across all signals
- **indie purist** — smallweb^1.5, quality^1.3, doubled IndieWeb bonuses
- **quality focused** — quality^1.5, pagerank^0.7
- **broad discovery** — pagerank^1.3, quality^0.7, platform cap raised

---

## Taste Model

Train a personal classifier from thumbs up/down on discoveries:

- Uses sentence-transformers (`all-MiniLM-L6-v2`) to embed each page's title + description + anchors
- Logistic regression learns your taste from labeled examples
- Needs at least 3 positive + 3 negative labels to train
- Taste score is multiplied into the final ranking formula

Label pages via the web UI (thumbs up/down buttons on discovery cards) or the API.

## Search

Three pluggable search backends:

| Backend | Best For | How It Works |
|---------|----------|-------------|
| **Fuzzy** | Quick scans | In-memory substring + token matching, bonuses for title/anchor hits |
| **FTS5** | Keyword search | SQLite FTS5 with porter stemming, BM25 ranking, weighted fields (title 10x, anchors 8x, description 5x) |
| **Semantic** | Conceptual queries | Sentence-transformer embeddings, cosine similarity. Finds related pages without keyword overlap |

```bash
# Search across graphs (auto-selects best graph)
python search.py search "malleable software" --top 10

# Specify backend
python search.py fts "local-first" --graph xi-malleable-v3 --top 20

# Semantic search
python search.py semantic "tools for thinking" --top 10

# List all graphs
python search.py graphs
```

## Commands

```bash
# Crawl from seeds
python smallweb.py crawl <urls_or_file> [--hops N] [--max-pages N] [--domain-cap N] [--output FILE]

# Show PageRank of all pages
python smallweb.py rank <graph.json> [--top N] [--damping F] [--iterations N]

# Show top discoveries (non-seed pages ranked by combined score)
python smallweb.py discover <graph.json> [--top N] [--damping F] [--iterations N]

# Fork a graph with new params, seed promotion, optional re-crawl
python smallweb.py fork <graph.json> [--name NAME] [--author AUTHOR] \
  [--damping F] [--iterations N] \
  [--promote-top N] [--add-seeds URL1,URL2] \
  [--recrawl] [--hops N] [--max-pages N] [--domain-cap N]

# Merge two graphs together
python smallweb.py merge <graph1.json> <graph2.json> [--output merged.json]

# Show graph stats
python smallweb.py info <graph.json>

# Serve interactive web UI
python smallweb.py serve <graph.json> [--port 8080]

# Export static HTML page
python smallweb.py html <graph.json> [--output page.html]
```

## Web Server & API

`server.py` runs the web interface and REST API.

```bash
pip install aiohttp beautifulsoup4 sentence-transformers scikit-learn
python server.py --port 8420
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/graphs` | List all saved graphs |
| `GET` | `/api/graphs/{id}` | Get graph JSON |
| `GET` | `/api/graphs/{id}/html` | Get rendered HTML |
| `GET` | `/api/graphs/{id}/discoveries` | Ranked discoveries (`?top=N&personalized=true&sort=overall`) |
| `GET` | `/api/graphs/{id}/similar` | Co-cited similar domains (`?target=DOMAIN&top=N`) |
| `GET` | `/api/graphs/{id}/similarities` | Top similar pairs (`?min_shared=N&top=N`) |
| `GET` | `/api/graphs/{id}/config` | Get scoring config |
| `PUT` | `/api/graphs/{id}/config` | Update scoring config |
| `GET` | `/api/graphs/{id}/config/presets` | List available presets |
| `POST` | `/api/graphs/{id}/config/preset` | Apply a named preset |
| `POST` | `/api/graphs/{id}/taste/label` | Add taste label (thumbs up/down) |
| `POST` | `/api/graphs/{id}/taste/train` | Train taste model |
| `POST` | `/api/crawl` | Start a new crawl |
| `GET` | `/api/crawl/{id}` | Check crawl status |
| `POST` | `/api/graphs/{id}/fork` | Fork a graph |

### Frontend

**Homepage** (`index.html`) — graph cards, crawl form, fork modal. No framework, no build step.

**Graph detail page** (tabbed):
- **Discoveries** — ranked cards with quality bars, smallweb indicators, anchor text pills, score breakdown popups, taste buttons, algorithm tuning sliders with live preview
- **Similarity** — co-citation explorer, search for similar domains, auto-computed top pairs
- **Seeds & Domains** — seed list, clickable domain tag cloud

## Graph Format

A single portable JSON file:

```json
{
  "metadata": {
    "name": "my-web-corner",
    "created_at": "2026-01-30T01:57:24",
    "version": "0.1.0",
    "domain_cap": 20,
    "damping": 0.95
  },
  "seeds": ["https://wiki.xxiivv.com", "https://100r.co"],
  "nodes": {
    "https://wiki.xxiivv.com": {
      "title": "XXIIVV",
      "description": "...",
      "crawled_at": "2026-01-30T01:57:25",
      "depth": 0,
      "domain": "wiki.xxiivv.com",
      "quality": 0.92,
      "quality_breakdown": { "base": 0.85, "penalties": {...}, "bonuses": {...} },
      "anchor_texts": ["hundred rabbits wiki", "xxiivv — personal wiki"]
    }
  },
  "edges": {
    "https://wiki.xxiivv.com": ["https://100r.co", "https://merveilles.town"]
  }
}
```

Sidecar files per graph:
- `{id}.taste.json` — taste labels (positive/negative URL lists)
- `{id}.taste.pkl` — trained taste model + cached embeddings
- `{id}.config.json` — scoring config overrides (only non-default values)
- `{id}.search_embeddings.npz` — cached semantic search vectors

## MCP Integration

An MCP server (`mcp_server.py`) exposes 8 tools for Claude CLI integration, enabling autonomous research loops:

1. Seed discovery (find niche blogs via web search)
2. Initial crawl (2 hops, 200-300 pages)
3. Co-citation analysis (find hidden gems)
4. Fork and go deeper (promote discoveries as seeds)
5. Synthesize results

## Seeds

Seeds can be:
- **Comma-separated URLs**: `"https://site1.com,https://site2.com"`
- **A text file** with one URL per line
- **An existing graph.json** — uses its seeds as starting points

## Dependencies

- Python 3.8+
- `aiohttp` — async HTTP client for crawling and serving
- `beautifulsoup4` — HTML parsing
- `sentence-transformers` — embeddings for taste model and semantic search (optional)
- `scikit-learn` — logistic regression for taste classifier (optional)
- `numpy` — vector operations (optional, comes with sentence-transformers)

## License

MIT
