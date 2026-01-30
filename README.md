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

The crawler has two built-in diversity mechanisms:
- **Domain cap** (`--domain-cap 20`) — no single domain can consume more than N pages of your crawl budget. Prevents Wikipedia from eating everything.
- **Domain diversity priority** — the BFS queue re-sorts each round to pull from domains we've crawled least. Encourages broad exploration over deep indexing.

## Commands

```bash
# Crawl from seeds
smallweb crawl <urls_or_file> [--hops N] [--max-pages N] [--domain-cap N] [--output graph.json]

# Show PageRank of all pages
smallweb rank <graph.json> [--top N]

# Show top discoveries (non-seed pages ranked by PageRank)
smallweb discover <graph.json> [--top N]

# Fork a graph (copy with provenance tracking)
smallweb fork <graph.json> [--output forked.json] [--name "my-fork"] [--author "me"]

# Merge two graphs together
smallweb merge <graph1.json> <graph2.json> [--output merged.json]

# Show graph stats
smallweb info <graph.json>

# Serve interactive web UI
smallweb serve <graph.json> [--port 8080]

# Export static HTML page
smallweb html <graph.json> [--output page.html]
```

## Seeds

Seeds can be:
- **Comma-separated URLs**: `"https://site1.com,https://site2.com"`
- **A text file** with one URL per line
- **An existing graph.json** — uses its seeds as starting points

## Forking & Merging

Graphs track provenance:

```bash
# Fork someone's discovery graph
smallweb fork friend-graph.json --output my-fork.json --author xi

# Add your own seeds and crawl deeper
# (edit my-fork.json to add seeds, then re-crawl)
smallweb crawl my-fork.json --hops 2

# Merge your discoveries with theirs
smallweb merge my-graph.json friend-graph.json --output combined.json
```

Forked graphs record where they came from:
```json
{
  "metadata": {
    "name": "my-fork",
    "forked_from": "friend-graph",
    "forked_at": "2026-01-30T12:00:00",
    "forked_by": "xi"
  }
}
```

## Graph Format

The graph is a single JSON file:

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

## Web Server (Optional)

`server.py` is an optional API server for hosting smallweb on a website. It lets users spawn crawls from a browser form and stores results persistently.

```bash
pip install aiohttp beautifulsoup4
python server.py --port 8420
```

Endpoints:
- `GET /api/graphs` — list all saved graphs
- `GET /api/graphs/{id}` — get graph JSON
- `GET /api/graphs/{id}/html` — get rendered HTML
- `GET /api/graphs/{id}/discoveries` — get discoveries
- `POST /api/crawl` — start a new crawl
- `GET /api/crawl/{id}` — check crawl status

## Philosophy

- **Your web, your rank** — you choose the seeds, you control what gets discovered
- **Transparent** — the ranking algorithm is 50 lines of PageRank, not a black box
- **Portable** — everything is a JSON file you can move, share, version control
- **No tracking** — runs locally, no analytics, no profiles, no ads
- **Forkable** — like git repos, discovery graphs can be forked, modified, merged

## Dependencies

- Python 3.8+
- `aiohttp` — async HTTP client for crawling
- `beautifulsoup4` — HTML parsing

## License

MIT
