"""Read-only bridge from C-Sync to the AI Trend Agent checkout it lives in.

C-Sync shows a recorded run: the snapshot at SNAPSHOT_PATH (default
01_data/demo_snapshot.json, written by 02_src/demo_snapshot.py --capture) and
the signals file that snapshot was captured from. Scores come from the stored
run -- no agent is re-run -- so every page agrees with every other.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# c_sync/ sits inside the repo, so the backend is its parent. CSYNC_BACKEND
# (or the older SKILLRADAR_BACKEND) points it at another checkout.
REPO = Path(__file__).resolve().parents[1]
BACKEND = Path(os.environ.get("CSYNC_BACKEND") or os.environ.get("SKILLRADAR_BACKEND") or REPO
               ).expanduser().resolve()
SRC = BACKEND / "02_src"


def backend_ready() -> bool:
    return (SRC / "schemas.py").is_file() and (SRC / "agents" / "evaluation.py").is_file()


def _imports() -> None:
    if not backend_ready():
        raise FileNotFoundError(f"AI Trend Agent backend not found at {BACKEND}")
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def import_backend() -> None:
    """Put the backend's 02_src on sys.path, for a page that imports from it."""
    _imports()


def snapshot_path() -> Path:
    p = Path(os.environ.get("SNAPSHOT_PATH", "01_data/demo_snapshot.json"))
    return p if p.is_absolute() else BACKEND / p


def load_recorded_run() -> tuple[dict, list]:
    """The saved run and the signals it was captured from -- no network."""
    _imports()
    from clustering import load_signals
    from demo_snapshot import load_snapshot

    snapshot = load_snapshot(str(snapshot_path()))
    source = snapshot.get("source_signals") or "01_data/signals.json"
    signals_file = Path(source) if Path(source).is_absolute() else BACKEND / source
    signals = load_signals(str(signals_file)) if signals_file.is_file() else []
    return snapshot, signals


def tier_order() -> tuple[list[str], dict[str, str]]:
    """Tier order and labels from the pipeline, so C-Sync cannot disagree
    with it about what counts as most urgent."""
    _imports()
    from demo_snapshot import TIER_LABEL, TIER_ORDER
    return list(TIER_ORDER), dict(TIER_LABEL)


def stored_scores(record: dict) -> tuple[int, int | None, float | None]:
    """(maturity, relevance, total) for a stored recommendation.

    The snapshot keeps total_score and confidence. Maturity is evaluation.py's
    own pure band function applied to the stored confidence; relevance is
    solved from total = 0.5 * maturity + 0.5 * relevance. Nothing is re-run.
    """
    _imports()
    from agents.evaluation import _maturity_score

    maturity = _maturity_score(record.get("confidence"))
    total = record.get("total_score")
    if not isinstance(total, (int, float)):
        return maturity, None, None
    relevance = max(1, min(5, round(2 * total - maturity)))
    return maturity, relevance, float(total)
