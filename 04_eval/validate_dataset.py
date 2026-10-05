#!/usr/bin/env python3
"""Validate human-supplied gold labels without modifying the dataset."""
from __future__ import annotations

import json
import sys
from pathlib import Path


DATASET = Path(__file__).resolve().parent / "data" / "test_signals_graded.json"
FIELDS = ("is_genuine", "confidence", "stale_presented_as_new", "rank", "maturity", "relevance", "action_tier")
# The four tiers recommendation._select_tier actually emits. schemas.ActionTier
# also declares "investigate_larger_change", but the agent never returns it, so
# it is not a valid gold value here (see 04_eval/GOLD_LABELS.md).
TIERS = {"watch", "update_existing_material", "add_optional_content", "add_new_lesson"}


def invalid(field: str, value: object) -> bool:
    if field in {"is_genuine", "stale_presented_as_new"}:
        return not isinstance(value, bool)
    if field == "confidence":
        return isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1
    if field == "rank":
        return isinstance(value, bool) or not isinstance(value, int) or value < 1
    if field in {"maturity", "relevance"}:
        return isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5
    return not isinstance(value, str) or value not in TIERS


def main() -> int:
    try:
        signals = json.loads(DATASET.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"invalid dataset: {exc}", file=sys.stderr)
        return 1
    if not isinstance(signals, list):
        print("invalid dataset: top level must be an array", file=sys.stderr)
        return 1

    incomplete: list[tuple[int, list[str]]] = []
    problems: list[str] = []
    ranks: list[tuple[int, int]] = []
    labelled = 0
    for index, signal in enumerate(signals, start=1):
        signal_id = signal.get("id", index) if isinstance(signal, dict) else index
        gold = signal.get("gold") if isinstance(signal, dict) else None
        if not isinstance(gold, dict):
            incomplete.append((signal_id, list(FIELDS)))
            problems.append(f"signal {signal_id}: missing gold object")
            continue
        extra = set(gold) - set(FIELDS)
        missing = set(FIELDS) - set(gold)
        if extra or missing:
            problems.append(f"signal {signal_id}: gold keys must be exactly {', '.join(FIELDS)}")
        nulls = [field for field in FIELDS if gold.get(field) is None]
        if nulls:
            incomplete.append((signal_id, nulls))
        else:
            labelled += 1
        for field in FIELDS:
            value = gold.get(field)
            if value is not None and invalid(field, value):
                problems.append(f"signal {signal_id}: invalid {field}={value!r}")
        if gold.get("rank") is not None and not invalid("rank", gold["rank"]):
            ranks.append((signal_id, gold["rank"]))

    for signal_id, fields in incomplete:
        print(f"signal {signal_id}: null gold fields: {', '.join(fields)}")
    for problem in problems:
        print(problem)
    seen: dict[int, list[int]] = {}
    for signal_id, rank in ranks:
        seen.setdefault(rank, []).append(signal_id)
    for rank, ids in sorted(seen.items()):
        if len(ids) > 1:
            print(f"duplicate rank {rank}: signals {', '.join(map(str, ids))}")
            problems.append("duplicate rank")
    if ranks:
        expected = set(range(1, len(ranks) + 1))
        actual = set(seen)
        if actual != expected:
            print(f"non-contiguous ranks: found {sorted(actual)}, expected 1 through {len(ranks)}")
            problems.append("non-contiguous ranks")
    print(f"{labelled}/{len(signals)} signals labelled.")
    return 0 if not incomplete and not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
