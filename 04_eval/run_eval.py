#!/usr/bin/env python3
"""Run the frozen, label-driven evaluation for AI-Trend-Agent.

Scoring is deliberately standard-library-only.  This runner never edits the
dataset; malformed entries are errors rather than silently changing a metric's
denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "02_src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schemas import RawSignal  # noqa: E402
from clustering import cluster_signals  # noqa: E402
from agents.verification import VerificationAgent  # noqa: E402
from agents.curriculum import CurriculumAgent  # noqa: E402
from agents.evaluation import EvaluationAgent  # noqa: E402
from agents.recommendation import RecommendationAgent  # noqa: E402

DEFAULT_DATASET = ROOT / "04_eval" / "data" / "test_signals_graded.json"
KNOWN_FABRICATED = "OpenTelemetry ships GenAI semantic conventions v1.0 GA with stable agent-graph and guardrail spans"
KNOWN_GENUINE = "OpenTelemetry GenAI conventions define invoke_agent, chat and execute_tool spans for agent runs"
ACTION_ORDER = ["watch", "update_existing_material", "add_optional_content", "add_new_lesson", "investigate_larger_change"]
WEIGHTS = {"clustering": .20, "verification": .25, "curriculum": .15, "evaluation": .20, "recommendation": .20}


def fail(message: str) -> None:
    raise SystemExit(f"eval error: {message}")


def read_dataset(path: Path) -> tuple[list[RawSignal], list[dict[str, Any]], str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read dataset {path}: {exc}")
    if not isinstance(raw, list) or not raw:
        fail("dataset must be a non-empty JSON array")
    required = {"title", "source", "source_tier", "summary", "url", "published"}
    signals: list[RawSignal] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            fail(f"dataset entry {index} must be an object")
        missing = required - item.keys()
        if missing:
            fail(f"dataset entry {index} missing {sorted(missing)}")
        unknown = set(item) - required - {"gold"}
        if unknown:
            fail(f"dataset entry {index} has unsupported fields {sorted(unknown)}")
        if item["source_tier"] not in ("primary", "secondary"):
            fail(f"dataset entry {index} has invalid source_tier")
        if not all(isinstance(item[name], str) for name in required):
            fail(f"dataset entry {index} RawSignal values must all be strings")
        if "gold" in item and not isinstance(item["gold"], dict):
            fail(f"dataset entry {index}.gold must be an object")
        signals.append(RawSignal(**{name: item[name] for name in required}))
    # Hash line-ending-normalised bytes: core.autocrlf checks this file out as
    # CRLF on Windows, and the same labels must not get a different identity
    # (compare.py refuses runs whose dataset hashes differ).
    digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return signals, raw, digest


class RecordingClient:
    """Injects temperature=0 and records API usage without changing agents."""
    def __init__(self, model: str):
        from openai import OpenAI
        self._inner = OpenAI()
        self._model = model
        self.tokens = 0
        self.observed_models: set[str] = set()
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        kwargs["temperature"] = 0
        reply = self._inner.chat.completions.create(**kwargs)
        self.observed_models.add(str(getattr(reply, "model", kwargs.get("model", self._model))))
        usage = getattr(reply, "usage", None)
        total = getattr(usage, "total_tokens", 0) if usage else 0
        self.tokens += int(total or 0)
        return reply


def gold(entry: dict[str, Any], name: str, default: Any = None) -> Any:
    return entry.get("gold", {}).get(name, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def metric(value: Any, requirement: str | None = None) -> dict[str, Any]:
    return {"value": value, "requirement": requirement} if value is None else {"value": value}


def clustering_metrics(clusters, entries: list[dict[str, Any]]) -> dict[str, Any]:
    # The finding is an explicit, versioned label in docs/live_run_findings.md;
    # it is not inferred from titles or agent output.
    known_present = {e["title"] for e in entries}
    documented = {KNOWN_FABRICATED, KNOWN_GENUINE} <= known_present
    contaminated = 0
    if documented:
        for cluster in clusters:
            titles = {s.title for s in cluster.signals}
            if KNOWN_FABRICATED in titles and KNOWN_GENUINE in titles:
                contaminated += 1
    else:
        for cluster in clusters:
            flags = [gold(e, "is_fabricated") for e in entries if e["title"] in {s.title for s in cluster.signals}]
            if True in flags and False in flags:
                contaminated += 1
    contamination = contaminated / len(clusters) if (documented or any(gold(e, "is_fabricated") is not None for e in entries)) else None
    expected = 7 if documented else (len({gold(e, "event_id") for e in entries if gold(e, "event_id") is not None}) or None)
    count_delta = len(clusters) - expected if expected is not None else None
    # Expected count is a diagnostic, not a compensating quality signal: a
    # contaminated merge can still happen to produce the documented count.
    parts = [v for v in ((1 - contamination) * 100 if contamination is not None else None,) if v is not None]
    return {"status": "implemented", "metrics": {
        "contamination_rate": metric(contamination, "gold.is_fabricated for every signal (the current fixture uses the documented finding labels)"),
        "cluster_count": {"value": len(clusters)},
        "expected_cluster_count": metric(expected, "gold.event_id for every signal, or a documented expected count"),
        "cluster_count_delta": metric(count_delta),
    }, "score": mean(parts)}


def verification_metrics(clusters, trends, entries):
    # Gold labels are per SIGNAL; the verifier runs per CLUSTER. Each signal is
    # scored against the trend of the cluster that contains it, matched by
    # title. (This used to zip(trends, entries) positionally: 11 clusters vs 12
    # signals, so after the first merged cluster every trend was compared with
    # the NEXT signal's label. All results before 2026-09-21 were misaligned.)
    by_title = {s.title: t.confidence for c, t in zip(clusters, trends) for s in c.signals}
    scored = [(by_title[e["title"]], e) for e in entries if e["title"] in by_title]
    truth = [(conf, gold(e, "is_genuine")) for conf, e in scored if gold(e, "is_genuine") in (True, False)]
    genuine, fabricated = [c for c, label in truth if label], [c for c, label in truth if not label]
    confidences = [(conf, gold(e, "confidence")) for conf, e in scored if isinstance(gold(e, "confidence"), (int, float))]
    gap = mean(genuine) - mean(fabricated) if genuine and fabricated else None
    mae = mean([abs(c - float(target)) for c, target in confidences])
    # No VerifiedTrend field says whether an old claim was recognized as stale.
    return {"status": "implemented", "metrics": {
        "truth_confidence_gap": metric(gap, "gold.is_genuine=true/false on every truth-scored signal"),
        "confidence_mae": metric(mae, "gold.confidence (0.0–1.0) on every confidence-scored signal"),
        "stale_flag_rate": metric(None, "gold.stale_presented_as_new plus a verification output field/contract that records a stale flag; VerifiedTrend has neither"),
    }, "score": mean([x for x in ((gap * 100) if gap is not None else None, (1 - mae) * 100 if mae is not None else None) if x is not None])}


def score_repeat(signals, entries, model: str) -> tuple[dict[str, Any], int, set[str]]:
    # Keep the pipeline's documented offline fallback usable when a .env has a
    # key but this checkout has not installed the optional OpenAI package.
    try:
        client = RecordingClient(model) if os.environ.get("OPENAI_API_KEY") else None
    except ModuleNotFoundError:
        client = None
    clusters = cluster_signals(signals)
    verifier = VerificationAgent(client=client, model=model)
    curriculum = CurriculumAgent(client=client, model=model)
    evaluator = EvaluationAgent(client=client, model=model)
    recommender = RecommendationAgent(client=client, model=model)
    trends, matches, evaluations, recommendations = [], [], [], []
    for cluster in clusters:
        trend = verifier.run(cluster)
        match = curriculum.run(trend)
        evaluation = evaluator.run(trend, match)
        recommendation = recommender.run(evaluation, curriculum_checked=True)
        trends.append(trend); matches.append(match); evaluations.append(evaluation); recommendations.append(recommendation)
    # Labels are signal-level but agents operate on clusters.  Metrics that
    # need labels remain null until each event has an unambiguous gold record.
    unavailable = "gold labels must be event-level and map each expected event to its signals"
    curriculum = {"status": "implemented", "metrics": {
        "precision_at_3": metric(None, unavailable + "; CurriculumAgent exposes one selected match, not its top-3 candidates"),
        "recall": metric(None, unavailable + " plus gold.acceptable_citations"),
        "correct_abstention_rate": metric(None, unavailable + " plus gold.no_match_expected"),
    }, "score": None}
    evaluation_block = {"status": "implemented", "metrics": {
        "spearman_rank_correlation": metric(None, unavailable + " plus gold.rank"),
        "maturity_mae": metric(None, unavailable + " plus gold.maturity (1–5)"),
        "relevance_mae": metric(None, unavailable + " plus gold.relevance (1–5)"),
    }, "score": None}
    recommendation_block = {"status": "implemented", "metrics": {
        "exact_tier_accuracy": metric(None, unavailable + " plus gold.action_tier"),
        "off_by_one_tier_accuracy": metric(None, unavailable + " plus gold.action_tier and an agreed ordered-tier policy"),
        "over_recommendation_rate": metric(None, unavailable + " plus gold.action_tier and an agreed ordered-tier policy"),
    }, "score": None}
    blocks = {"clustering": clustering_metrics(clusters, entries), "verification": verification_metrics(clusters, trends, entries),
              "curriculum": curriculum, "evaluation": evaluation_block, "recommendation": recommendation_block}
    active = {name: block["score"] for name, block in blocks.items() if block["score"] is not None}
    composite = sum(WEIGHTS[n] * v for n, v in active.items()) / sum(WEIGHTS[n] for n in active) if active else None
    return {"layers": blocks, "composite_score": composite, "pipeline": {"clusters": len(clusters), "recommendations": len(recommendations)}}, (client.tokens if client else 0), (client.observed_models if client else set())


def aggregate(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(values):
        values = [v for v in values if v is not None]
        return {"mean": mean(values), "std": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None}
    layers = {}
    for name in WEIGHTS:
        first = repeats[0]["layers"][name]
        metrics = {key: stats([r["layers"][name]["metrics"][key]["value"] for r in repeats]) for key in first["metrics"]}
        layers[name] = {"status": first["status"], "metrics": metrics, "score": stats([r["layers"][name]["score"] for r in repeats])}
    return {"layers": layers, "composite_score": stats([r["composite_score"] for r in repeats])}


def git_info():
    def get(*args): return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    return {"commit": get("git", "rev-parse", "HEAD"), "dirty": bool(get("git", "status", "--porcelain"))}


def print_table(result):
    print("\nFrozen evaluation baseline")
    print("layer             score (mean ± std)")
    print("-" * 46)
    for name, block in result["aggregate"]["layers"].items():
        score = block["score"]
        text = "null" if score["mean"] is None else f"{score['mean']:.2f} ± {score['std']:.2f}"
        print(f"{name:<17} {text}")
    c = result["aggregate"]["composite_score"]
    print("-" * 46)
    print("composite         " + ("null" if c["mean"] is None else f"{c['mean']:.2f} ± {c['std']:.2f}"))
    print(f"runs: {len(result['runs'])}; tokens: " + ", ".join(str(r['total_tokens']) for r in result['runs']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = ap.parse_args()
    if args.repeats < 1: fail("--repeats must be at least 1")
    args.dataset = args.dataset.resolve()
    signals, entries, digest = read_dataset(args.dataset)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    runs = []
    observed_models: set[str] = set()
    for n in range(args.repeats):
        started = time.perf_counter()
        scored, tokens, observed = score_repeat(signals, entries, model)
        elapsed = time.perf_counter() - started
        scored.update({"repeat": n + 1, "total_tokens": tokens, "wall_clock_seconds": elapsed})
        runs.append(scored); observed_models.update(observed)
    result = {"format_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "code": git_info(),
              "dataset": {"path": str(args.dataset.relative_to(ROOT)), "sha256": digest, "item_count": len(signals)},
              "model": {"requested_identifier": model, "observed_identifiers": sorted(observed_models), "temperature": 0,
                        "version_pinned": bool(__import__("re").search(r"(?:20\d{2}[-_]\d{2}[-_]\d{2}|v\d+(?:\.\d+)+)$", model)),
                        "note": "Set OPENAI_MODEL to a provider version-pinned identifier before freezing a live baseline."},
              "repeats": args.repeats, "runs": runs, "aggregate": aggregate(runs)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print_table(result)


if __name__ == "__main__":
    main()
