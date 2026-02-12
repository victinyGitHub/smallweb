#!/usr/bin/env python3
"""
smallweb search - modular, extensible search across web graphs.

Three search backends, pluggable architecture. grug brain.

Usage:
    python search.py fuzzy  "malleable software" --graph xi-malleable-v3 --top 10
    python search.py fts    "local first" --graph auto --top 20
    python search.py semantic "tools for thought" --graph xi-malleable-v3
    python search.py search "indie web" --method auto --top 15
    python search.py graphs  # list available graphs
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Config ───────────────────────────────────────────────────────────

GRAPHS_DIR = Path("/var/www/arg/mains.in.net/smallweb/graphs")

# ── Graph loading (shared, cached) ──────────────────────────────────

_graph_cache: Dict[str, Tuple[float, "WebGraph"]] = {}

def _ensure_smallweb_path():
    """Add smallweb dir to path so we can import WebGraph."""
    sw_dir = str(Path(__file__).parent)
    if sw_dir not in sys.path:
        sys.path.insert(0, sw_dir)

def load_graph(graph_id: str) -> "WebGraph":
    """Load a graph by ID with mtime-based caching."""
    _ensure_smallweb_path()
    from smallweb import WebGraph

    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"graph not found: {graph_id}")

    mtime = path.stat().st_mtime
    if graph_id in _graph_cache:
        cached_mtime, cached_graph = _graph_cache[graph_id]
        if cached_mtime == mtime:
            return cached_graph

    graph = WebGraph.load(str(path))
    _graph_cache[graph_id] = (mtime, graph)
    return graph


def list_graphs() -> List[dict]:
    """List all available graphs with metadata."""
    graphs = []
    for f in sorted(GRAPHS_DIR.glob("*.json")):
        if f.name.endswith(".taste.json") or f.name.endswith(".config.json"):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            meta = data.get("metadata", {})
            graphs.append({
                "id": f.stem,
                "name": meta.get("name", f.stem),
                "description": meta.get("description", ""),
                "nodes": len(data.get("nodes", {})),
                "edges": sum(len(v) for v in data.get("edges", {}).values()),
                "seeds": len(data.get("seeds", [])),
                "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
            })
        except (json.JSONDecodeError, IOError):
            continue
    return graphs


# ── Search Result ────────────────────────────────────────────────────

class SearchResult:
    """A single search result. Uniform across all backends."""

    __slots__ = ("url", "title", "description", "domain", "score", "method")

    def __init__(self, url: str, title: str = "", description: str = "",
                 domain: str = "", score: float = 0.0, method: str = ""):
        self.url = url
        self.title = title
        self.description = description
        self.domain = domain
        self.score = score
        self.method = method

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "domain": self.domain,
            "score": round(self.score, 4),
            "method": self.method,
        }

    def __repr__(self):
        t = self.title[:50] if self.title else self.url[:50]
        return f"<{self.method}:{self.score:.3f}> {t}"


# ── Search Backend ABC ──────────────────────────────────────────────

class SearchBackend(ABC):
    """Base class for search backends. Extend this to add new methods."""

    name: str = "base"

    @abstractmethod
    def search(self, query: str, graph: "WebGraph", top_n: int = 10) -> List[SearchResult]:
        """Search a graph, return ranked results."""
        ...

    def _node_to_result(self, url: str, node: dict, score: float) -> SearchResult:
        """Helper: convert a graph node to a SearchResult."""
        title = node.get("title", "")
        desc = node.get("description", "")
        anchors = node.get("anchor_texts") or []
        # Use first non-URL anchor as fallback title
        if not title and anchors:
            for a in anchors:
                if a.strip() and not a.startswith("http"):
                    title = a.strip()
                    break
        # Use anchors as fallback description
        if not desc and anchors:
            clean = [a for a in anchors if a.strip() and not a.startswith("http")]
            if clean:
                desc = " · ".join(clean[:3])
        return SearchResult(
            url=url,
            title=title,
            description=desc,
            domain=node.get("domain", ""),
            score=score,
            method=self.name,
        )


# ── Backend 1: Fuzzy ────────────────────────────────────────────────

class FuzzySearch(SearchBackend):
    """
    In-memory substring + token matching on title, description, domain,
    and anchor texts (what other sites call this page).
    Scores by: exact match > word boundary match > substring match.
    """

    name = "fuzzy"

    def search(self, query: str, graph, top_n: int = 10) -> List[SearchResult]:
        q_lower = query.lower()
        q_tokens = set(q_lower.split())
        results = []

        for url, node in graph.nodes.items():
            title = (node.get("title") or "").lower()
            desc = (node.get("description") or "").lower()
            domain = (node.get("domain") or "").lower()
            # Anchor texts: what other sites call this page — rich signal
            anchors = " ".join(node.get("anchor_texts") or []).lower()
            haystack = f"{title} {desc} {domain} {anchors}"

            # Score: exact phrase > all tokens present > some tokens > substring
            score = 0.0

            if q_lower in haystack:
                # Exact phrase match
                score = 1.0
                # Bonus if it's in the title or anchors
                if q_lower in title:
                    score += 0.5
                elif q_lower in anchors:
                    score += 0.3
            else:
                # Token matching
                matched = sum(1 for t in q_tokens if t in haystack)
                if matched > 0:
                    score = 0.3 * (matched / len(q_tokens))
                    # Bonus for title matches
                    title_matched = sum(1 for t in q_tokens if t in title)
                    if title_matched:
                        score += 0.2 * (title_matched / len(q_tokens))
                    # Bonus for anchor matches
                    anchor_matched = sum(1 for t in q_tokens if t in anchors)
                    if anchor_matched:
                        score += 0.15 * (anchor_matched / len(q_tokens))

            if score > 0:
                results.append(self._node_to_result(url, node, score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n]


# ── Backend 2: FTS5 ─────────────────────────────────────────────────

class FTS5Search(SearchBackend):
    """
    SQLite FTS5 full-text search. Builds ephemeral index in /tmp on first
    query per graph. Rebuilds if graph mtime changes.

    Supports: phrase queries, boolean (AND/OR/NOT), prefix matching (term*).
    """

    name = "fts5"

    # Cache: graph_id -> (mtime, db_path)
    _index_cache: Dict[str, Tuple[float, str]] = {}

    def _get_index(self, graph_id: str, graph) -> str:
        """Get or build FTS5 index for a graph. Returns db path."""
        path = GRAPHS_DIR / f"{graph_id}.json"
        mtime = path.stat().st_mtime

        if graph_id in self._index_cache:
            cached_mtime, db_path = self._index_cache[graph_id]
            if cached_mtime == mtime and os.path.exists(db_path):
                return db_path

        db_path = f"/tmp/smallweb_fts_{graph_id}.db"
        self._build_index(graph, db_path)
        self._index_cache[graph_id] = (mtime, db_path)
        return db_path

    def _build_index(self, graph, db_path: str):
        """Build FTS5 index from graph nodes."""
        if os.path.exists(db_path):
            os.remove(db_path)

        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS pages USING fts5(
                url UNINDEXED,
                title,
                description,
                domain,
                anchors,
                tokenize='porter unicode61'
            )
        """)

        rows = []
        for url, node in graph.nodes.items():
            # Anchor texts: deduplicate and join. These are what other
            # sites call this page — often the best text signal we have.
            anchor_list = node.get("anchor_texts") or []
            anchors_text = " | ".join(dict.fromkeys(anchor_list))  # dedup, preserve order
            rows.append((
                url,
                node.get("title", "") or "",
                node.get("description", "") or "",
                node.get("domain", "") or "",
                anchors_text,
            ))

        conn.executemany("INSERT INTO pages (url, title, description, domain, anchors) VALUES (?, ?, ?, ?, ?)", rows)
        conn.commit()
        conn.close()

    def search(self, query: str, graph, top_n: int = 10,
               graph_id: str = "") -> List[SearchResult]:
        if not graph_id:
            # Need graph_id for caching — fall back to hash
            graph_id = str(id(graph))

        db_path = self._get_index(graph_id, graph)
        conn = sqlite3.connect(db_path)

        # FTS5 query — if user is using operators (AND/OR/NOT/*/"), pass through.
        # Otherwise, join tokens with OR so multi-word queries find
        # pages matching any of the terms, then BM25 ranks by relevance.
        fts_query = query
        if not any(op in query for op in ['"', 'AND', 'OR', 'NOT', '*']):
            tokens = query.split()
            if len(tokens) == 1:
                fts_query = tokens[0]
            else:
                # Try: exact phrase first, fall back to OR
                fts_query = f'"{query}" OR ' + " OR ".join(tokens)

        try:
            rows = conn.execute("""
                SELECT url, title, description, domain, anchors,
                       bm25(pages, 0, 10.0, 5.0, 2.0, 8.0) as score
                FROM pages
                WHERE pages MATCH ?
                ORDER BY score
                LIMIT ?
            """, (fts_query, top_n)).fetchall()
        except sqlite3.OperationalError:
            # If FTS5 query syntax fails, try simple prefix match
            tokens = query.split()
            fts_query = " OR ".join(f"{t}*" for t in tokens)
            try:
                rows = conn.execute("""
                    SELECT url, title, description, domain, anchors,
                           bm25(pages, 0, 10.0, 5.0, 2.0, 8.0) as score
                    FROM pages
                    WHERE pages MATCH ?
                    ORDER BY score
                    LIMIT ?
                """, (fts_query, top_n)).fetchall()
            except sqlite3.OperationalError:
                conn.close()
                return []

        conn.close()

        results = []
        for url, title, desc, domain, anchors, score in rows:
            # bm25 returns negative scores (lower = better match)
            # If no title, use first anchor text as a stand-in
            display_title = title or (anchors.split(" | ")[0] if anchors else "")
            results.append(SearchResult(
                url=url, title=display_title, description=desc or anchors[:200],
                domain=domain, score=-score, method=self.name,
            ))

        return results


# ── Backend 3: Semantic ─────────────────────────────────────────────

class SemanticSearch(SearchBackend):
    """
    Sentence-transformer cosine similarity search.
    Uses all-MiniLM-L6-v2 (same model as taste.py).
    Embeds all nodes on first use, caches to disk.

    Best for: conceptual queries, finding related content that doesn't
    share exact keywords.
    """

    name = "semantic"

    # Cache: graph_id -> (mtime, embeddings_dict)
    _embed_cache: Dict[str, Tuple[float, dict]] = {}

    def _get_embeddings(self, graph_id: str, graph) -> Tuple[list, "np.ndarray"]:
        """Get or build embeddings for all nodes. Returns (urls, vectors)."""
        import numpy as np

        path = GRAPHS_DIR / f"{graph_id}.json"
        mtime = path.stat().st_mtime
        cache_path = GRAPHS_DIR / f"{graph_id}.search_embeddings.npz"

        # Check memory cache
        if graph_id in self._embed_cache:
            cached_mtime, cached_data = self._embed_cache[graph_id]
            if cached_mtime == mtime:
                return cached_data["urls"], cached_data["vectors"]

        # Check disk cache
        if cache_path.exists() and cache_path.stat().st_mtime >= mtime:
            data = np.load(str(cache_path), allow_pickle=True)
            urls = data["urls"].tolist()
            vectors = data["vectors"]
            self._embed_cache[graph_id] = (mtime, {"urls": urls, "vectors": vectors})
            return urls, vectors

        # Build embeddings
        _ensure_smallweb_path()
        from taste import _get_embed_model

        model = _get_embed_model()
        urls = []
        texts = []

        for url, node in graph.nodes.items():
            urls.append(url)
            parts = []
            title = node.get("title", "")
            if title:
                parts.append(title)
            desc = node.get("description", "")
            if desc:
                parts.append(desc[:200])
            # Anchor texts — often the BEST signal for nodes with no title/desc
            anchor_list = node.get("anchor_texts") or []
            # Deduplicate, skip raw URLs, take first 3
            seen = set()
            for a in anchor_list:
                a_clean = a.strip()
                if a_clean and not a_clean.startswith("http") and a_clean.lower() not in seen:
                    seen.add(a_clean.lower())
                    parts.append(a_clean)
                    if len(seen) >= 3:
                        break
            domain = node.get("domain", "")
            if domain:
                parts.append(domain)
            texts.append(" | ".join(parts) if parts else url)

        # Batch embed — this is the slow part on first run
        vectors = model.encode(texts, show_progress_bar=True, batch_size=128)

        # Save to disk
        np.savez_compressed(str(cache_path), urls=np.array(urls), vectors=vectors)
        self._embed_cache[graph_id] = (mtime, {"urls": urls, "vectors": vectors})

        return urls, vectors

    def search(self, query: str, graph, top_n: int = 10,
               graph_id: str = "") -> List[SearchResult]:
        import numpy as np
        _ensure_smallweb_path()
        from taste import _get_embed_model

        if not graph_id:
            graph_id = str(id(graph))

        urls, vectors = self._get_embeddings(graph_id, graph)
        if len(urls) == 0:
            return []

        # Embed query
        model = _get_embed_model()
        q_vec = model.encode([query], show_progress_bar=False)[0]

        # Cosine similarity
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1  # avoid div by zero
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        similarities = vectors @ q_vec / (norms * q_norm)

        # Top N
        top_indices = np.argsort(similarities)[::-1][:top_n]

        results = []
        for idx in top_indices:
            url = urls[idx]
            node = graph.nodes.get(url, {})
            score = float(similarities[idx])
            if score <= 0:
                break
            results.append(self._node_to_result(url, node, score))

        return results


# ── Backend Registry ─────────────────────────────────────────────────

BACKENDS: Dict[str, SearchBackend] = {
    "fuzzy": FuzzySearch(),
    "fts": FTS5Search(),
    "fts5": FTS5Search(),  # alias
    "semantic": SemanticSearch(),
}

def register_backend(name: str, backend: SearchBackend):
    """Register a custom search backend. Extensibility point."""
    BACKENDS[name] = backend


# ── Graph Selection ──────────────────────────────────────────────────

# Keywords → graph mapping for auto-selection
GRAPH_HINTS = {
    "xi-malleable-v3": ["malleable", "local-first", "spatial", "indie", "tool", "software", "web", "demoscene", "creative"],
    "fork-meg-curius-experiment": ["curius", "bookmark", "article", "read", "blog", "essay"],
    "idealists-v2": ["idealist", "philosophy", "idea"],
    "fork-idealists-v2": ["idealist", "philosophy", "idea"],
    "small-web-test": ["small web", "indie web", "personal"],
}

def select_graphs(query: str, explicit: str = "", mode: str = "auto") -> List[str]:
    """
    Pick which graph(s) to search.

    Modes:
        auto     - pick best graph from query keywords
        explicit - use the one the user specified
        multi    - search all graphs, deduplicate
    """
    available = [f.stem for f in GRAPHS_DIR.glob("*.json")
                 if not f.name.endswith(".taste.json")
                 and not f.name.endswith(".config.json")]

    if mode == "explicit" and explicit:
        # Fuzzy match the graph name
        explicit_lower = explicit.lower()
        for g in available:
            if explicit_lower in g.lower() or g.lower() in explicit_lower:
                return [g]
        # Try exact
        if explicit in available:
            return [explicit]
        raise ValueError(f"graph not found: {explicit}. available: {', '.join(available)}")

    if mode == "multi":
        return available

    # Auto mode: score each graph by keyword overlap
    q_lower = query.lower()
    scores = {}
    for graph_id, hints in GRAPH_HINTS.items():
        if graph_id not in available:
            continue
        score = sum(1 for h in hints if h in q_lower)
        if score > 0:
            scores[graph_id] = score

    if scores:
        best = max(scores, key=scores.get)
        return [best]

    # Default: pick the biggest graph with xi's taste
    default_order = ["xi-malleable-v3", "fork-meg-curius-experiment", "idealists-v2"]
    for g in default_order:
        if g in available:
            return [g]

    return available[:1] if available else []


# ── Unified Search ───────────────────────────────────────────────────

def search(query: str, method: str = "auto", graph: str = "auto",
           graph_mode: str = "auto", top_n: int = 10) -> List[dict]:
    """
    Main entry point. Search across graphs with any method.

    Args:
        query:      search query
        method:     "fuzzy", "fts", "semantic", or "auto" (tries fts then fuzzy)
        graph:      graph ID or "auto"
        graph_mode: "auto", "explicit", "multi"
        top_n:      max results
    """
    # Pick graph(s)
    if graph != "auto":
        graph_mode = "explicit"
    graph_ids = select_graphs(query, explicit=graph, mode=graph_mode)

    if not graph_ids:
        return []

    # Pick method
    if method == "auto":
        # FTS5 first (fast + good ranking), fall back to fuzzy
        method = "fts"

    backend = BACKENDS.get(method)
    if not backend:
        raise ValueError(f"unknown method: {method}. available: {', '.join(BACKENDS.keys())}")

    # Search across selected graphs
    all_results = []
    seen_urls = set()

    for gid in graph_ids:
        try:
            g = load_graph(gid)
        except FileNotFoundError:
            continue

        # Pass graph_id for backends that need it (FTS5, semantic)
        kwargs = {"query": query, "graph": g, "top_n": top_n}
        if hasattr(backend.search, '__code__') and 'graph_id' in backend.search.__code__.co_varnames:
            kwargs["graph_id"] = gid

        results = backend.search(**kwargs)

        for r in results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                # Tag which graph the result came from
                r_dict = r.to_dict()
                r_dict["graph"] = gid
                all_results.append((r.score, r_dict))

    # Sort by score descending, take top_n
    all_results.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in all_results[:top_n]]


# ── CLI ──────────────────────────────────────────────────────────────

def _format_results(results: List[dict], verbose: bool = False) -> str:
    """Pretty-print search results."""
    if not results:
        return "no results found."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")[:60] or r["url"][:60]
        score = r.get("score", 0)
        domain = r.get("domain", "")
        method = r.get("method", "")
        graph = r.get("graph", "")

        line = f"{i:2}. [{score:.3f}] {title}"
        if domain:
            line += f"  ({domain})"
        lines.append(line)

        if verbose:
            if r.get("description"):
                lines.append(f"    {r['description'][:100]}")
            lines.append(f"    {r['url']}")
            lines.append(f"    method={method} graph={graph}")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="smallweb search — modular search across web graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="search method")

    # Shared args
    def add_common(p):
        p.add_argument("query", help="search query")
        p.add_argument("--graph", "-g", default="auto", help="graph ID or 'auto'")
        p.add_argument("--top", "-n", type=int, default=10, help="max results")
        p.add_argument("--json", action="store_true", help="output as JSON")
        p.add_argument("--verbose", "-v", action="store_true", help="show descriptions + URLs")

    # Per-method subcommands
    for method_name in ["fuzzy", "fts", "semantic"]:
        p = sub.add_parser(method_name, help=f"{method_name} search")
        add_common(p)

    # Unified search (auto-picks method)
    p = sub.add_parser("search", aliases=["s"], help="unified search (auto-picks method)")
    add_common(p)
    p.add_argument("--method", "-m", default="auto",
                   choices=["auto", "fuzzy", "fts", "semantic"],
                   help="search method (default: auto)")
    p.add_argument("--multi", action="store_true", help="search all graphs")

    # List graphs
    sub.add_parser("graphs", help="list available graphs")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "graphs":
        graphs = list_graphs()
        for g in graphs:
            print(f"  {g['id']:40s} {g['nodes']:>6d} nodes  {g['size_mb']:>5.1f}MB  {g['description'][:40]}")
        return

    # Determine method from subcommand
    if args.command in ("search", "s"):
        method = args.method
        graph_mode = "multi" if args.multi else "auto"
    else:
        method = args.command
        graph_mode = "auto"

    if args.graph != "auto":
        graph_mode = "explicit"

    t0 = time.time()
    results = search(
        query=args.query,
        method=method,
        graph=args.graph,
        graph_mode=graph_mode,
        top_n=args.top,
    )
    elapsed = time.time() - t0

    if args.json:
        print(json.dumps({"results": results, "elapsed_ms": round(elapsed * 1000, 1)}, indent=2))
    else:
        print(_format_results(results, verbose=args.verbose))
        print(f"\n({len(results)} results in {elapsed*1000:.0f}ms)")


if __name__ == "__main__":
    main()
