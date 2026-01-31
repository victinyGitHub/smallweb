#!/usr/bin/env python3
"""
taste.py - Neural taste classifier for smallweb discovery

Learns what "good small web content" looks like from user-labeled examples.
Uses sentence-transformers to embed page metadata (title + description + domain),
then trains a logistic regression classifier to predict taste scores.

Training data is stored per-graph as a JSON file of labeled URLs:
  graphs/my-graph.taste.json = {"positive": [...urls], "negative": [...urls]}

The model (embeddings + classifier) is cached per-graph and rebuilt on label changes.

Usage:
    from taste import TasteModel

    taste = TasteModel(graph)
    taste.add_positive("https://example.com")
    taste.add_negative("https://spam.com")
    taste.train()

    scores = taste.score_all()  # {url: 0.0-1.0}
    score = taste.score_url("https://other.com")  # single URL
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# Lazy-load heavy deps
_model = None

def _get_embed_model():
    """Lazy-load the sentence-transformer model (first call takes ~1s)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


class TasteModel:
    """
    A per-graph taste classifier that learns from user feedback.

    Embeds pages by their title + description + domain, trains a tiny
    logistic regression to separate "good" from "bad" small web content.
    """

    def __init__(self, graph=None, graph_path: str = ""):
        """
        Args:
            graph: A WebGraph instance (optional, can set later)
            graph_path: Path to the graph JSON file (used to find .taste.json)
        """
        self.graph = graph
        self.graph_path = graph_path
        self.positive: List[str] = []  # URLs the user liked
        self.negative: List[str] = []  # URLs the user rejected
        self.classifier = None
        self._embeddings_cache: Dict[str, np.ndarray] = {}

        # Load existing labels if they exist
        if graph_path:
            self._load_labels()

    @property
    def labels_path(self) -> str:
        """Path to the taste labels file for this graph."""
        if self.graph_path:
            return self.graph_path.replace(".json", ".taste.json")
        return ""

    @property
    def model_path(self) -> str:
        """Path to the cached trained model."""
        if self.graph_path:
            return self.graph_path.replace(".json", ".taste.pkl")
        return ""

    @property
    def has_labels(self) -> bool:
        """Whether we have enough labels to train."""
        return len(self.positive) >= 3 and len(self.negative) >= 3

    @property
    def is_trained(self) -> bool:
        """Whether the classifier has been trained."""
        return self.classifier is not None

    def _load_labels(self):
        """Load existing taste labels from disk."""
        if not self.labels_path or not os.path.exists(self.labels_path):
            return
        try:
            with open(self.labels_path) as f:
                data = json.load(f)
            self.positive = data.get("positive", [])
            self.negative = data.get("negative", [])
        except (json.JSONDecodeError, IOError):
            pass

        # Also try to load cached model
        if self.model_path and os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    cached = pickle.load(f)
                self.classifier = cached.get("classifier")
                self._embeddings_cache = cached.get("embeddings", {})
            except Exception:
                pass

    def _save_labels(self):
        """Save taste labels to disk."""
        if not self.labels_path:
            return
        data = {
            "positive": self.positive,
            "negative": self.negative,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self.labels_path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_model(self):
        """Cache the trained model to disk."""
        if not self.model_path or not self.classifier:
            return
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "classifier": self.classifier,
                "embeddings": self._embeddings_cache,
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_positive": len(self.positive),
                "n_negative": len(self.negative),
            }, f)

    def add_positive(self, url: str):
        """Label a URL as good small web content."""
        url = url.strip()
        if url not in self.positive:
            self.positive.append(url)
        # Remove from negative if it was there
        if url in self.negative:
            self.negative.remove(url)
        self._save_labels()

    def add_negative(self, url: str):
        """Label a URL as not good small web content."""
        url = url.strip()
        if url not in self.negative:
            self.negative.append(url)
        # Remove from positive if it was there
        if url in self.positive:
            self.positive.remove(url)
        self._save_labels()

    def remove_label(self, url: str):
        """Remove all labels for a URL."""
        url = url.strip()
        if url in self.positive:
            self.positive.remove(url)
        if url in self.negative:
            self.negative.remove(url)
        self._save_labels()

    def _page_text(self, url: str) -> str:
        """
        Build the text representation of a page for embedding.
        Uses title + description + domain as the input text.
        """
        if not self.graph:
            return url

        node = self.graph.nodes.get(url, {})
        parts = []

        title = node.get("title", "")
        if title:
            parts.append(title)

        desc = node.get("description", "")
        if desc:
            parts.append(desc)

        domain = node.get("domain", "")
        if domain:
            parts.append(f"[{domain}]")

        # Include anchor texts if available (how others describe this page)
        anchors = node.get("anchor_texts", [])
        if anchors:
            parts.append("linked as: " + ", ".join(anchors[:5]))

        return " | ".join(parts) if parts else url

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings. Returns (n, 384) array."""
        model = _get_embed_model()
        return model.encode(texts, show_progress_bar=False, batch_size=64)

    def _embed_urls(self, urls: List[str]) -> np.ndarray:
        """
        Embed URLs, using cache where possible.
        Returns (n, 384) array aligned with input urls.
        """
        # Split into cached and uncached
        uncached_urls = []
        uncached_indices = []

        for i, url in enumerate(urls):
            if url not in self._embeddings_cache:
                uncached_urls.append(url)
                uncached_indices.append(i)

        # Embed uncached
        if uncached_urls:
            texts = [self._page_text(url) for url in uncached_urls]
            vecs = self._embed(texts)
            for url, vec in zip(uncached_urls, vecs):
                self._embeddings_cache[url] = vec

        # Assemble result
        return np.array([self._embeddings_cache[url] for url in urls])

    def train(self) -> dict:
        """
        Train the taste classifier on current labels.

        Returns dict with training stats:
            n_positive, n_negative, accuracy, trained
        """
        if not self.has_labels:
            return {
                "trained": False,
                "error": f"need at least 3 positive and 3 negative examples "
                         f"(have {len(self.positive)}+ / {len(self.negative)}-)",
                "n_positive": len(self.positive),
                "n_negative": len(self.negative),
            }

        if not self.graph:
            return {"trained": False, "error": "no graph loaded"}

        # Filter to URLs that exist in our graph
        pos_urls = [u for u in self.positive if u in self.graph.nodes]
        neg_urls = [u for u in self.negative if u in self.graph.nodes]

        if len(pos_urls) < 2 or len(neg_urls) < 2:
            return {
                "trained": False,
                "error": f"not enough labeled URLs found in graph "
                         f"({len(pos_urls)}+ / {len(neg_urls)}-)",
                "n_positive": len(pos_urls),
                "n_negative": len(neg_urls),
            }

        # Embed all labeled pages
        all_urls = pos_urls + neg_urls
        X = self._embed_urls(all_urls)
        y = np.array([1] * len(pos_urls) + [0] * len(neg_urls))

        # Train logistic regression
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            C=1.0,             # regularization (1.0 = moderate)
            max_iter=1000,
            class_weight="balanced",  # handle imbalanced classes
        )
        clf.fit(X, y)

        # Training accuracy (on training set — small dataset so no split)
        accuracy = clf.score(X, y)

        self.classifier = clf
        self._save_model()

        return {
            "trained": True,
            "n_positive": len(pos_urls),
            "n_negative": len(neg_urls),
            "accuracy": round(accuracy, 3),
        }

    def score_url(self, url: str) -> float:
        """
        Score a single URL. Returns 0.0-1.0 taste score.
        Higher = more likely to be "good small web content".
        Returns 0.5 (neutral) if no model is trained.
        """
        if not self.is_trained:
            return 0.5

        vec = self._embed_urls([url])
        prob = self.classifier.predict_proba(vec)[0]
        # prob is [P(negative), P(positive)]
        return round(float(prob[1]), 3)

    def score_all(self, urls: List[str] = None) -> Dict[str, float]:
        """
        Score all URLs in the graph (or a provided list).
        Returns {url: taste_score} dict.
        """
        if not self.is_trained:
            return {}

        if urls is None:
            urls = list(self.graph.nodes.keys()) if self.graph else []

        if not urls:
            return {}

        # Batch embed and score
        vecs = self._embed_urls(urls)
        probs = self.classifier.predict_proba(vecs)

        return {url: round(float(prob[1]), 3) for url, prob in zip(urls, probs)}

    def stats(self) -> dict:
        """Return current taste model stats."""
        return {
            "n_positive": len(self.positive),
            "n_negative": len(self.negative),
            "is_trained": self.is_trained,
            "has_labels": self.has_labels,
            "labels_path": self.labels_path,
            "model_path": self.model_path,
        }
