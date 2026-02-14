# smallweb explorer

Find interesting websites that Google buries.

Give it a few sites you like, it crawls outward, scores pages by quality + smallweb-ness + your taste, and surfaces gems you'd never find through search. Not a search engine — an **explorer**. For when you don't know what you're looking for yet.

```bash
pip install aiohttp beautifulsoup4

# Start from sites you love
python smallweb.py crawl "https://wiki.xxiivv.com,https://100r.co,https://solar.lowtechmagazine.com" \
  --hops 2 --max-pages 300 --name "my-corner"

# See what it found
python smallweb.py discover my-corner.json --top 20

# Serve the web UI
python server.py --port 8420
```

## How It Works

Seven components, each encoding a specific belief about what makes a page worth discovering. They multiply together — a page needs to pass ALL of them to rank high.

### 1. Domain-Diverse BFS Crawling

Breadth-first from your seeds, re-sorting the queue so least-crawled domains get priority. Following one site deep gives you that site's worldview. Spreading across domains finds the connections *between* worldviews.

- Async fetching (10 concurrent), adaptive domain caps (seeds 2.5x, hubs 1.5x)
- Three layers of URL filtering: skip big tech/CDN, block 100+ platform domains, regex for spam

### 2. Personalized PageRank

Standard PageRank finds what the *whole graph* thinks is important. Personalized PageRank finds what's important *from your corner* — the random walker keeps coming home to your seeds.

- Damping 0.95 (follows links deep), seed-biased teleport vector
- Percentile normalization (0-1) before combining with other signals

### 3. Quality Scoring

Additive penalty/bonus system measuring how "clean" a page's HTML is. Inspired by [Marginalia Search](https://search.marginalia.nu/).

A page with 14 tracking scripts and cookie popups was built to capture attention. A page with clean HTML was built to share ideas. Quality is a proxy for editorial intent.

- Penalties: trackers (-0.15), ad-tech (-0.2), AI content patterns (-0.15 to -0.5), link farms (-0.4)
- Bonuses: Webmention (+0.1), IndieAuth (+0.1), RSS/Atom (+0.05), Microformats (+0.05)

### 4. Smallweb Bell Curve

A per-domain score that peaks when a site is linked by a few people — not zero and not thousands.

| Inbound Domains | Score | Why |
|----------------|-------|-----|
| 0 | 0.4 | Nobody found it |
| 3-8 | 1.0 | Sweet spot — curated, not mainstream |
| 16-30 | 0.3 | Mainstream |
| 61+ | 0.05 | Platform-scale |

The interesting zone is "a few people who care linked here" — editorial curation, not algorithmic popularity.

### 5. Anchor Texts

What other sites call a page, not what it calls itself. Self-description is marketing. How others describe you is reputation. Stored as first-class data, searchable, displayed as pills on discovery cards.

### 6. Co-Citation Similarity

Two domains are "similar" if they're linked FROM the same sources — the same people think they're worth linking to. Reveals editorial communities invisible to keyword search.

### 7. Fork/Promote Loop

Crawl → discover → promote the best discoveries to seeds → re-crawl from expanded seeds → discover deeper. The graph teaches you where to look next.

```bash
python smallweb.py fork my-corner.json --name round2 \
  --promote-top 5 --add-seeds "https://newdiscovery.com" \
  --recrawl --hops 2 --max-pages 500
```

### The Formula

```
score = PageRank^exp × Quality^exp × Smallweb^exp × taste_factor
```

Tunable via named presets (indie purist, quality focused, broad discovery) or per-graph config.

## Commands

```bash
python smallweb.py crawl <urls>     # Crawl from seeds
python smallweb.py discover <graph> # Ranked discoveries
python smallweb.py rank <graph>     # Raw PageRank
python smallweb.py fork <graph>     # Fork + optionally re-crawl
python smallweb.py merge <a> <b>    # Merge two graphs
python smallweb.py info <graph>     # Graph stats
python smallweb.py serve <graph>    # Serve single-graph web UI
python smallweb.py html <graph>     # Export static HTML
```

## Web UI

`server.py` runs the full web interface — graph dashboard, crawl launcher, discovery browser with interactive scoring.

```bash
python server.py --port 8420
```

**Dashboard**: graph cards, start crawls, fork graphs.

**Discovery view**: ranked cards with stacked score bars (shows exactly how much PageRank vs quality vs smallweb contributed), "why is this here?" one-liners, anchor text pills, taste training (thumbs up/down), force-directed graph visualization, audit trace (follow the path from any discovery back to your seeds).

### API

| Method | Endpoint | What |
|--------|----------|------|
| `GET` | `/api/graphs` | List all graphs |
| `GET` | `/api/graphs/{id}/discoveries` | Ranked discoveries |
| `GET` | `/api/graphs/{id}/similar?target=DOMAIN` | Co-cited similar domains |
| `GET` | `/api/graphs/{id}/domain-graph` | Domain-level link graph |
| `GET` | `/api/graphs/{id}/audit?url=URL` | Trace path from URL back to seeds |
| `POST` | `/api/crawl` | Start a crawl |
| `POST` | `/api/graphs/{id}/fork` | Fork a graph |
| `POST` | `/api/graphs/{id}/taste/label` | Label for taste training |
| `POST` | `/api/graphs/{id}/taste/train` | Train taste model |

## Search

Three pluggable backends:

| Backend | For | How |
|---------|-----|-----|
| **Fuzzy** | Quick scans | In-memory substring + token matching |
| **FTS5** | Keywords | SQLite FTS5, porter stemming, BM25, weighted fields |
| **Semantic** | Concepts | Sentence-transformer embeddings, cosine similarity |

```bash
python search.py search "malleable software" --top 10
python search.py semantic "tools for thinking" --top 10
```

## Graph Format

Portable JSON. Fork them, merge them, share them.

```json
{
  "metadata": { "name": "my-corner", "created_at": "...", "domain_cap": 20, "damping": 0.95 },
  "seeds": ["https://wiki.xxiivv.com", "https://100r.co"],
  "nodes": {
    "https://wiki.xxiivv.com": {
      "title": "XXIIVV", "description": "...", "domain": "wiki.xxiivv.com",
      "quality": 0.92, "anchor_texts": ["hundred rabbits wiki"]
    }
  },
  "edges": { "https://wiki.xxiivv.com": ["https://100r.co", "https://merveilles.town"] }
}
```

## Dependencies

**Required**: Python 3.8+, `aiohttp`, `beautifulsoup4`

**Optional** (for taste model + semantic search): `sentence-transformers`, `scikit-learn`, `numpy`

## License

MIT
