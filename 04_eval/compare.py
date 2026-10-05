#!/usr/bin/env python3
"""Compare two frozen evaluation result files."""
from __future__ import annotations
import json, sys
from pathlib import Path

LAYERS = ["clustering", "verification", "curriculum", "evaluation", "recommendation"]
def load(path):
    try: return json.loads(Path(path).read_text())
    except Exception as exc: raise SystemExit(f"compare error: cannot read {path}: {exc}")
def main():
    if len(sys.argv) != 3: raise SystemExit("usage: python 04_eval/compare.py <baseline.json> <after.json>")
    before, after = load(sys.argv[1]), load(sys.argv[2])
    for key in ("sha256",):
        if before["dataset"].get(key) != after["dataset"].get(key): raise SystemExit("compare error: dataset sha256 differs; refusing meaningless comparison")
    if before["model"].get("requested_identifier") != after["model"].get("requested_identifier"): raise SystemExit("compare error: model identifier differs; refusing comparison")
    print("layer             baseline     after        delta       relative       flag")
    print("-" * 82)
    for layer in LAYERS + ["composite"]:
        a = before["aggregate"]["composite_score"] if layer == "composite" else before["aggregate"]["layers"][layer]["score"]
        b = after["aggregate"]["composite_score"] if layer == "composite" else after["aggregate"]["layers"][layer]["score"]
        if a["mean"] is None or b["mean"] is None:
            print(f"{layer:<17} null         null         null        null           N/A"); continue
        delta = b["mean"] - a["mean"]
        relative = "n/a" if a["mean"] == 0 else f"{delta / a['mean'] * 100:+.1f}%"
        flag = "NOISE: delta < baseline std" if abs(delta) < a["std"] else ""
        print(f"{layer:<17} {a['mean']:>7.2f} ±{a['std']:<5.2f} {b['mean']:>7.2f} ±{b['std']:<5.2f} {delta:>+7.2f} pts  {relative:<13} {flag}")
if __name__ == "__main__": main()
