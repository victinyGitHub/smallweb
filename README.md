# smallweb

A small web discovery engine. Crawl niche corners of the web from seed URLs, build a local link graph, rank pages by local PageRank, and discover related sites.

Google indexes the whole web and ranks globally. **smallweb indexes YOUR web and ranks locally.**

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

## How It Works

1. **Seed** — start with URLs you care about (blogs, personal sites, community pages)
2. **Crawl** — follow links N hops out, building a local link graph
3. **Rank** — run PageRank on just your subgraph, not the whole internet
4. **Discover** — see what bubbles up that you didn't already know about

### Crawler

The crawler is async (aiohttp) with concurrent fetching and two diversity mechanisms:

- **Domain cap** (`--domain-cap 20`) — no single domain can consume more than N pages of your crawl budget. Prevents Wikipedia from eating everything.
- **Domain diversity priority** — the BFS queue re-sorts each round by `(domain_count, depth)`, pulling from least-crawled domains first. Encourages broad exploration over deep indexing.

The queue works as a breadth-first search with a diversity twist: after each batch of concurrent fetches, the queue is re-prioritized so domains you've barely touched get processed before domains you've already hit the cap on.

### PageRank

Local PageRank runs on your subgraph only, not the whole internet. Parameters:

- **Damping** (default `0.95`) — probability of following a link vs jumping to a random page. Higher values (like 0.95) give more weight to link structure; lower values (like 0.5) spread rank more evenly. Think of it as "how much does a link endorsement matter?"
- **Iterations** (default `50`) — number of power iteration rounds. 50 is way more than enough for convergence on small graphs.

Dangling nodes (pages with no outlinks) redistribute their rank uniformly across all pages, matching the original PageRank paper.

## Commands

```bash
# Crawl from seeds
python smallweb.py crawl <urls_or_file> [--hops N] [--max-pages N] [--domain-cap N] [--output FILE]

# Show PageRank of all pages
python smallweb.py rank <graph.json> [--top N] [--damping F] [--iterations N]

# Show top discoveries (non-seed pages ranked by PageRank)
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

## PageRank Tuning

You can experiment with different PageRank parameters without re-crawling:

```bash
# Default: damping=0.95
python smallweb.py discover graph.json --top 10

# Lower damping spreads rank more evenly (less link-dependent)
python smallweb.py discover graph.json --top 10 --damping 0.5

# Higher damping amplifies link authority
python smallweb.py discover graph.json --top 10 --damping 0.99

# Compare rankings side by side
python smallweb.py rank graph.json --top 10 --damping 0.85
python smallweb.py rank graph.json --top 10 --damping 0.95
```

## Seeds

Seeds can be:
- **Comma-separated URLs**: `"https://site1.com,https://site2.com"`
- **A text file** with one URL per line
- **An existing graph.json** — uses its seeds as starting points

## Forking

Forking creates a new graph from an existing one. You can change PageRank parameters, promote discoveries to seeds, add new seeds, and optionally re-crawl.

### Quick param change (no re-crawl)

```bash
# Fork with different damping — instant, no network requests
python smallweb.py fork graph.json --name "high-damping" --damping 0.99
```

### Promote discoveries to seeds

```bash
# Take the top 5 discovered pages and make them seeds in the fork
python smallweb.py fork graph.json --name "expanded" --promote-top 5
```

This is the core loop: crawl → discover → promote the best discoveries → re-crawl from expanded seeds → discover deeper.

### Add custom seeds + re-crawl

```bash
# Add new seeds and re-crawl the whole thing
python smallweb.py fork graph.json --name "with-new-seeds" \
  --add-seeds "https://newsite1.com,https://newsite2.com" \
  --promote-top 3 \
  --recrawl --hops 2 --max-pages 500 --domain-cap 20
```

### Provenance tracking

Forked graphs record where they came from:

```json
{
  "metadata": {
    "name": "expanded",
    "forked_from": "my-web-corner",
    "forked_at": "2026-01-30T12:00:00",
    "forked_by": "xi",
    "seeds_promoted": ["https://discovered-site.com", "https://another.com"]
  }
}
```

## Merging

```bash
# Combine two discovery graphs
python smallweb.py merge my-graph.json friend-graph.json --output combined.json
```

Merging unions the nodes, edges, and seeds from both graphs. The result records both parent graphs in its metadata.

## Graph Format

A single portable JSON file:

```json
{
  "metadata": {
    "name": "my-web-corner",
    "created_at": "2026-01-30T01:57:24",
    "version": "0.1.0",
    "domain_cap": 20
  },
  "seeds": ["https://wiki.xxiivv.com", "https://100r.co"],
  "nodes": {
    "https://wiki.xxiivv.com": {
      "title": "XXIIVV",
      "description": "...",
      "crawled_at": "2026-01-30T01:57:25",
      "depth": 0,
      "domain": "wiki.xxiivv.com"
    }
  },
  "edges": {
    "https://wiki.xxiivv.com": ["https://100r.co", "https://merveilles.town"]
  }
}
```

## Web Server & Frontend

`server.py` is an optional API server for hosting smallweb as a web app. `index.html` is the frontend with graph browsing and fork controls.

### Running

```bash
pip install aiohttp beautifulsoup4
python server.py --port 8420
# Open http://localhost:8420 in your browser
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/graphs` | List all saved graphs |
| `GET` | `/api/graphs/{id}` | Get graph JSON |
| `GET` | `/api/graphs/{id}/html` | Get rendered HTML |
| `GET` | `/api/graphs/{id}/discoveries` | Get discoveries (supports `?damping=F&iterations=N`) |
| `POST` | `/api/crawl` | Start a new crawl |
| `GET` | `/api/crawl/{id}` | Check crawl status |
| `POST` | `/api/graphs/{id}/fork` | Fork a graph |

### Fork API

```bash
curl -X POST http://localhost:8420/api/graphs/my-graph/fork \
  -H "Content-Type: application/json" \
  -d '{
    "name": "expanded",
    "promote_top": 5,
    "add_seeds": ["https://newsite.com"],
    "damping": 0.95,
    "iterations": 50,
    "recrawl": true,
    "hops": 2,
    "max_pages": 500,
    "domain_cap": 20
  }'
```

When `recrawl` is `false` (default), the fork is instant — just a copy with new params. When `true`, it spawns an async crawl from all seeds (including promoted ones).

### Frontend

`index.html` provides:
- **Graph cards** — browse all saved graphs with stats (nodes, edges, seeds, domains)
- **Crawl form** — start new crawls from the browser with seed URLs, hops, max pages, domain cap
- **Fork modal** — fork any graph with: new name, damping/iterations, promote top N discoveries to seeds, add custom seeds, optional re-crawl with full param control

## Philosophy

- **Your web, your rank** — you choose the seeds, you control what gets discovered
- **Transparent** — the ranking algorithm is ~50 lines of PageRank, not a black box
- **Portable** — everything is a JSON file you can move, share, version control
- **No tracking** — runs locally, no analytics, no profiles, no ads
- **Forkable** — like git repos, discovery graphs can be forked, modified, merged
- **Iterative** — crawl → discover → promote → re-crawl. each cycle finds deeper corners

## Dependencies

- Python 3.8+
- `aiohttp` — async HTTP client for crawling
- `beautifulsoup4` — HTML parsing

## License

MIT
