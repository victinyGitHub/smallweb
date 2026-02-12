"""
Scoring config for the smallweb discovery engine.

Grug-brained: it's just a dict. Load it, save it, merge it.
Every hardcoded number from the scoring system lives here as a default.
Per-graph configs only store what differs from defaults.

Usage:
    cfg = load_config("path/to/graph.json")     # loads .config.json if it exists
    cfg["formula"]["quality_exp"]                # read a value
    save_config("path/to/graph.json", cfg)       # saves only non-default values
    cfg = with_overrides(cfg, {"formula": {"quality_exp": 1.5}})  # temp overrides
"""

import copy
import json
from pathlib import Path


# ── The One True Defaults Dict ──────────────────────────────────────────
# Every tunable number in the scoring system. If you add a new signal,
# add its default here. That's it. That's the whole system.

DEFAULTS = {
    # How the 4 signals combine into a final score.
    # Multiplicative with exponents: pagerank^exp × quality^exp × smallweb^exp
    # Higher exponent = that signal matters MORE (amplifies differences)
    # Lower exponent = that signal matters LESS (compresses differences)
    # 0 = signal completely ignored, 1 = linear (current behavior)
    "formula": {
        "pagerank_exp": 1.0,
        "quality_exp": 1.0,
        "smallweb_exp": 1.0,
        "taste_weight": 0.5,       # in: taste_base + taste_weight × taste_score
        "taste_base": 0.5,         # baseline multiplier when taste is active
        "fetched_boost": 0.2,      # multiplier for pages we haven't crawled yet
    },

    # Quality penalties: how much to subtract per occurrence of each signal.
    # Set to 0 to disable a penalty entirely.
    "quality_penalties": {
        "external_scripts": 0.08,       # per external <script src="...">
        "inline_scripts": 0.02,         # per inline <script>
        "dynamic_injection": 0.1,       # per createElement() call in scripts
        "trackers": 0.15,               # per known tracker domain
        "adtech": 0.2,                  # per ad network script
        "cookie_consent": 0.1,          # per consent signal found
        "cookie_consent_max": 0.3,      # cap on total cookie consent penalty
        "affiliate_links": 0.05,        # per affiliate link
        "affiliate_links_max": 0.25,    # cap on total affiliate penalty
        "ai_content_high": 0.5,         # penalty when ≥3 AI heading patterns
        "ai_content_low": 0.15,         # penalty when 1-2 AI heading patterns
        "ai_content_threshold": 3,      # how many patterns = "high"
        "link_farm": 0.4,               # penalty when link density > link_farm_threshold
        "link_farm_threshold": 0.5,     # links-per-word ratio for link farm
        "link_density": 0.2,            # penalty when density > link_density_threshold
        "link_density_threshold": 0.3,  # links-per-word ratio for high density
        "adblock_key": 0.6,             # penalty for adblock circumvention key
    },

    # Quality bonuses: how much to add when we detect each signal.
    "quality_bonuses": {
        "webmention": 0.1,
        "indieauth": 0.1,
        "rss_feed": 0.05,
        "microformats": 0.05,
    },

    # Smallweb scoring: how "indie" a domain is.
    "smallweb": {
        # Bell curve: [min_inbound, max_inbound, score]
        # null max = infinity. Sweet spot at 3-8 inbound domains.
        "popularity_curve": [
            [0, 0, 0.4],
            [1, 2, 0.7],
            [3, 8, 1.0],
            [9, 15, 0.6],
            [16, 30, 0.3],
            [31, 60, 0.15],
            [61, None, 0.05],
        ],
        "ecosystem_weight": 0.7,      # weight for "links to sites in our graph"
        "platform_weight": 0.3,       # weight for "avoids linking to platforms"
        "outlink_moderation": 0.5,    # in: pop × (mod + mod × outlink)
        "platform_cap": 0.15,         # max smallweb score for known platforms
    },

    # Named presets: partial overrides that deep-merge onto defaults.
    # Each preset only specifies what it changes.
    "presets": {
        "indie_purist": {
            "formula": {"smallweb_exp": 1.5, "quality_exp": 1.3},
            "quality_bonuses": {"webmention": 0.2, "indieauth": 0.2, "rss_feed": 0.1, "microformats": 0.1},
        },
        "quality_focused": {
            "formula": {"quality_exp": 1.5, "pagerank_exp": 0.7},
        },
        "broad_discovery": {
            "formula": {"pagerank_exp": 1.3, "quality_exp": 0.7, "smallweb_exp": 0.5},
            "smallweb": {"platform_cap": 0.4},
        },
    },
}


# ── Helpers ─────────────────────────────────────────────────────────────

def _deep_merge(base: dict, overlay: dict) -> dict:
    """Merge overlay into base recursively. Returns new dict."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _diff_from_defaults(cfg: dict, defaults: dict = None) -> dict:
    """Return only the values that differ from defaults. For minimal saves."""
    if defaults is None:
        defaults = DEFAULTS
    diff = {}
    for key, value in cfg.items():
        if key.startswith("_"):
            continue
        if key not in defaults:
            diff[key] = value
        elif isinstance(value, dict) and isinstance(defaults.get(key), dict):
            sub = _diff_from_defaults(value, defaults[key])
            if sub:
                diff[key] = sub
        elif value != defaults.get(key):
            diff[key] = value
    return diff


def _config_path(graph_path: str) -> Path:
    """Get the .config.json path for a graph file."""
    return Path(str(graph_path).replace(".json", ".config.json"))


# ── Public API ──────────────────────────────────────────────────────────

def load_config(graph_path: str = "") -> dict:
    """
    Load scoring config for a graph. Returns full config (defaults + overrides).
    If no .config.json exists, returns pure defaults.
    """
    cfg = copy.deepcopy(DEFAULTS)
    if graph_path:
        p = _config_path(graph_path)
        if p.exists():
            try:
                with open(p) as f:
                    saved = json.load(f)
                cfg = _deep_merge(cfg, saved)
            except (json.JSONDecodeError, OSError):
                pass  # corrupt file, use defaults
    return cfg


def save_config(graph_path: str, cfg: dict):
    """
    Save config, storing only values that differ from defaults.
    This keeps config files small and readable.
    """
    diff = _diff_from_defaults(cfg)
    if diff:
        p = _config_path(graph_path)
        with open(p, "w") as f:
            json.dump(diff, f, indent=2)
    else:
        # Everything is default, remove config file if it exists
        p = _config_path(graph_path)
        if p.exists():
            p.unlink()


def with_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply temporary overrides to a config. Returns new dict."""
    return _deep_merge(cfg, overrides)


def apply_preset(cfg: dict, preset_name: str) -> dict:
    """Apply a named preset to a config. Returns new dict."""
    presets = cfg.get("presets", DEFAULTS.get("presets", {}))
    preset = presets.get(preset_name, {})
    if not preset:
        return cfg
    return _deep_merge(cfg, preset)


def popularity_score(cfg: dict, n_inbound: int) -> float:
    """Look up the bell curve score for an inbound link count."""
    curve = cfg.get("smallweb", {}).get("popularity_curve", DEFAULTS["smallweb"]["popularity_curve"])
    for lo, hi, score in curve:
        if hi is None:
            if n_inbound >= lo:
                return score
        elif lo <= n_inbound <= hi:
            return score
    return 0.05  # fallback
