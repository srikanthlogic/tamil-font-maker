"""Project config.toml access.

Top-level keys (`family`, `psname`) configure build. Per-cell QA-loop
overrides live in `[cells.<gid>]` sections (the SKILL.md contract):

- `threshold` (int): fixed binarization cutoff at ingest — replaces the
  mode default (digital 190 / paper otsu-clamped / extract unclamped otsu)
- `despeckle` (int): minimum ink-component area in px — extract-mode
  cleanup `min_area` and trace `turdsize`
- `smooth` (float): trace `alphamax` (potrace corner smoothing, default 0.8)

Unknown keys or wrong types raise ConfigError: a typo'd override must fail
loudly, not silently do nothing.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

CELL_KEYS = {"threshold": int, "despeckle": int, "smooth": (int, float)}


class ConfigError(ValueError):
    pass


def load(project: Path) -> dict:
    cfg = Path(project) / "config.toml"
    if cfg.exists():
        return tomllib.loads(cfg.read_text())
    return {}


def cell_params(config: dict, gid: str) -> dict:
    """Validated `[cells.<gid>]` overrides for one gid ({} when absent)."""
    raw = config.get("cells", {}).get(gid)
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"[cells.{gid}] must be a table")
    out = {}
    for key, val in raw.items():
        want = CELL_KEYS.get(key)
        if want is None:
            raise ConfigError(
                f"[cells.{gid}]: unknown key {key!r} "
                f"(allowed: {', '.join(sorted(CELL_KEYS))})")
        if not isinstance(val, want) or isinstance(val, bool):
            raise ConfigError(f"[cells.{gid}]: {key} must be "
                              f"{'int' if want is int else 'number'}")
        out[key] = val
    return out
