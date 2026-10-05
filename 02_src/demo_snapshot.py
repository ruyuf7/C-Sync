"""
Demo snapshot -- capture one good run, replay it forever.
==========================================================
The pipeline is non-deterministic in two places, and only one of them is
fixed by the tool cache:

  1. TOOL RESULTS. Cached already (TOOL_CACHE_ONLY=1 in agents/tools.py).
     Same query, same answer.

  2. THE MODEL'S CHOICES. Not cached, and not cacheable. The agent decides
     which queries to run and when to stop. Measured on the same 10 signals
     minutes apart: five of ten trends changed tier. "Organizing Context in a
     Multi-Agent Harness" went from add_new_lesson (no match found) to
     update_existing_material (matched Week 4 slide 17) between runs.

So freezing the tools is not enough. To demo reliably you have to freeze the
OUTPUT, which is what this does.

    # once, when the output looks good:
    python 02_src/demo_snapshot.py --capture --limit 30

    # every time after, including in front of graders:
    python 02_src/demo_snapshot.py --replay

Replay makes ZERO API calls and needs no key or network. It is the same data
your FastAPI endpoint would serve, so the UI can be built against it too.

This is not cheating -- it is the difference between demoing a system and
demoing a coin flip. Say plainly in the presentation that the snapshot is a
recorded run, and that live mode is one flag away.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

DEFAULT_SNAPSHOT = "01_data/demo_snapshot.json"


# ---------------------------------------------------------------------------
# TRACES
# Both agents already record their own reasoning -- CurriculumAgent into a
# CurriculumTrace, VerificationAgent into a VerificationTrace -- and both
# run() methods already accept one. capture() simply never passed them, so
# every trace was built and then dropped on the floor. These turn a live
# trace into the plain dict the snapshot stores.
#
# Nothing here changes what the pipeline decides. It records what it did.
# Kept as free functions, duck-typed on the trace objects, so importing this
# module still costs nothing (no openai import at module load).
# ---------------------------------------------------------------------------

def verification_trace_dict(trace, trend=None) -> dict:
    """
    VerificationTrace (+ the VerifiedTrend it produced) -> plain dict.

    `steps` are the tool calls only (n, tool, arguments, result_summary) --
    kept exactly as before so older pages and snapshots still work.
    `reasoning` is the verifier's whole loop from VerifiedTrend.reasoning,
    INCLUDING steps with no tool call (the model's own thoughts, "names no
    verifiable repository", "model concluded it has enough evidence").
    `mode` says which loop gathered the evidence: the model driving the tools,
    or the scripted loop that runs when no model is available. Both feed the
    same deterministic scorer. trend is optional: old callers pass only trace.
    """
    reasoning = getattr(trend, "reasoning", None) or []
    return {
        "steps": [{"n": s.n, "tool": s.tool, "arguments": s.arguments,
                   "result_summary": s.result_summary} for s in trace.steps],
        "stopped_early": trace.stopped_early,
        "mode": getattr(trend, "mode", "") or "",
        "reasoning": [{"iteration": r.iteration, "thought": r.thought,
                       "tool": r.tool, "tool_args": dict(r.tool_args or {}),
                       "observation": r.observation} for r in reasoning],
    }


def curriculum_trace_dict(trace, searched: bool, skipped_reason: str = "") -> dict:
    """
    CurriculumTrace -> plain dict. Its Step is (n, query, filters, result_summary).

    THREE outcomes have to stay distinguishable downstream, because two of
    them look identical in the conclusion alone:

      searched=True,  search_failed=False -> it looked; reason says what it found
      searched=True,  search_failed=True  -> it TRIED and could not run. NOT a
                                             finding of "no match" (see below)
      searched=False                      -> the confidence gate skipped it; no
                                             search was ever attempted

    Note there is no separate `search_failed_reason` field on the trace: when
    search_failed is true the agents overwrite `reason` with the failure
    reason, so that one key carries both meanings. Read it together with the
    flag, never alone.
    """
    if not searched:
        return {"searched": False, "skipped_reason": skipped_reason,
                "steps": [], "reason": "", "stopped_early": False,
                "search_failed": False}
    return {
        "searched": True,
        "skipped_reason": "",
        "steps": [{"n": s.n, "query": s.query, "filters": s.filters,
                   "result_summary": s.result_summary} for s in trace.steps],
        "reason": trace.reason,
        "stopped_early": trace.stopped_early,
        "search_failed": trace.search_failed,
    }


# ---------------------------------------------------------------------------
# TRACE VIEW -- the plain-language reading of a stored trace
# What a curriculum lead sees in the Streamlit card. Built here, in plain
# Python, so it is tested in test_chain (the UI file is not importable there).
# Reads a snapshot dict with .get() throughout: a recommendation captured
# before traces existed returns None, and the UI renders nothing for it.
# ---------------------------------------------------------------------------

SEARCH_FAILED_BANNER = ("The curriculum search FAILED -- this is NOT a finding "
                        "of 'no match'. No search result was produced, so the "
                        "recommendation cannot claim the topic is uncovered.")


def verification_mode_text(mode: str) -> str:
    """Plain words for VerifiedTrend.mode. Empty for snapshots without it."""
    m = (mode or "").lower()
    if not m:
        return ""
    if "llm loop failed" in m:
        return ("The model call failed, so the same checks ran from a fixed "
                "script instead -- scored exactly the same way.")
    if m.startswith("deterministic"):
        return ("No model was used: the checks ran from a fixed script. The "
                "score comes from what the checks found, as always.")
    if m.startswith("agentic"):
        return ("The model chose which checks to run. The score was computed "
                "from what those checks found, not by the model.")
    return mode


def trace_view(rec: dict) -> dict | None:
    """A stored recommendation -> what the trace panel shows, or None."""
    trace = rec.get("trace") if isinstance(rec, dict) else None
    if not isinstance(trace, dict):
        return None
    c = trace.get("curriculum") or {}
    v = trace.get("verification") or {}
    failed = c.get("search_failed") is True

    # the full reasoning when captured; older traces only have tool steps
    if v.get("reasoning"):
        v_steps = [{"text": (f"Checked {r['tool']}" if r.get("tool") else r.get("thought", "")),
                    "detail": r.get("observation", ""),
                    "why": r.get("thought", "") if r.get("tool") else ""}
                   for r in v["reasoning"]]
    else:
        v_steps = [{"text": f"Checked {s.get('tool', '')}",
                    "detail": s.get("result_summary", ""), "why": ""}
                   for s in v.get("steps") or []]

    c_steps = []
    for s in c.get("steps") or []:
        filters = ", ".join(f"{k} {val}" for k, val in (s.get("filters") or {}).items()
                            if val not in (None, ""))
        c_steps.append({"text": f"Searched for \"{s.get('query', '')}\""
                                + (f" ({filters})" if filters else ""),
                        "detail": s.get("result_summary", ""), "why": ""})

    if failed:
        c_empty = "The search failed before it issued any query."
    elif c.get("searched") is False:
        c_empty = "Skipped: " + (c.get("skipped_reason") or "it did not clear the confidence gate.")
    else:
        c_empty = "No searches were issued."

    n = len(v_steps) + len(c_steps)
    return {
        "label": ("Agent trace -- why the curriculum search failed" if failed
                  else f"Agent trace ({n} step{'' if n == 1 else 's'})"),
        "search_failed": failed,
        "failure_banner": SEARCH_FAILED_BANNER if failed else None,
        "failure_reason": c.get("reason", "") if failed else "",
        "verification_mode": verification_mode_text(v.get("mode", "")),
        "verification_steps": v_steps,
        "verification_empty": "No verification steps were recorded.",
        "curriculum_steps": c_steps,
        "curriculum_empty": c_empty,
        "stopped_early": bool(v.get("stopped_early") or c.get("stopped_early")),
        "conclusion": (c.get("reason", "") if c.get("searched") and not failed else ""),
    }


# ---------------------------------------------------------------------------
# CAPTURE
# ---------------------------------------------------------------------------

def capture(signals_path: str, snapshot_path: str, limit: int,
            offset: int = 0) -> dict:
    from clustering import cluster_signals, load_signals
    from agents.verification import VerificationAgent, VerificationTrace
    from agents.curriculum import (CurriculumAgent, CurriculumTrace,
                                   search_curriculum_checked)
    from agents.evaluation import EvaluationAgent
    from agents.recommendation import RecommendationAgent, collapse_duplicates

    signals = load_signals(signals_path)
    clusters = cluster_signals(signals)
    clusters.sort(key=lambda c: -len(c.signals))
    batch = clusters[offset:offset + limit]

    verifier, curriculum = VerificationAgent(), CurriculumAgent()
    evaluator, recommender = EvaluationAgent(), RecommendationAgent()

    recs = []
    # Keyed by id(): collapse_duplicates returns the SAME Recommendation
    # objects it was given (it mutates the survivor's plan and drops the
    # rest), so identity survives the merge where a title might not.
    traces: dict[int, dict] = {}

    for i, c in enumerate(batch, start=1):
        print(f"  [{i}/{len(batch)}] {c.representative_title[:60]}")

        vtrace = VerificationTrace()
        trend = verifier.run(c, vtrace)

        # `searched` is the confidence GATE. `checked` additionally goes false
        # when a search was attempted and failed -- keep them separate, or a
        # failure gets recorded as "never tried".
        searched = trend.confidence >= 0.4
        checked = False
        ctrace = CurriculumTrace()
        match = None
        if searched:
            match, checked = search_curriculum_checked(curriculum, trend, ctrace)
            if not checked:
                print(f"      ! curriculum search failed: {ctrace.reason[:80]}")

        ev = evaluator.run(trend, match)
        rec = recommender.run(ev, curriculum_checked=checked)
        recs.append(rec)

        traces[id(rec)] = {
            "verification": verification_trace_dict(vtrace, trend),
            "curriculum": curriculum_trace_dict(
                ctrace, searched=searched,
                skipped_reason=(
                    f"verification confidence {trend.confidence} is below 0.4, "
                    f"so the curriculum was not searched")),
        }

    before = len(recs)
    recs = collapse_duplicates(recs)

    tiers: dict[str, int] = {}
    for r in recs:
        tiers[r.recommended_action] = tiers.get(r.recommended_action, 0) + 1

    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_signals": signals_path,
        "signals_in_file": len(signals),
        "clusters_total": len(clusters),
        "clusters_processed": len(batch),
        "duplicates_collapsed": before - len(recs),
        "tier_counts": tiers,
        # to_dict() is still the whole contract; "trace" rides alongside it
        # rather than inside Recommendation, so schemas.py is untouched.
        "recommendations": [
            ({**r.to_dict(), "trace": traces[id(r)]} if id(r) in traces
             else r.to_dict())
            for r in recs
        ],
    }

    out = Path(snapshot_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    return snapshot


# ---------------------------------------------------------------------------
# REPLAY
# ---------------------------------------------------------------------------

def load_snapshot(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"no snapshot at {path} -- run with --capture first")
    return json.loads(p.read_text(encoding="utf-8"))


# order by how much a curriculum lead should care, not alphabetically
TIER_ORDER = ["update_existing_material", "add_new_lesson",
              "add_optional_content", "investigate_larger_change", "watch"]

TIER_LABEL = {
    "update_existing_material": "UPDATE EXISTING MATERIAL",
    "add_new_lesson": "ADD NEW LESSON",
    "add_optional_content": "ADD OPTIONAL CONTENT",
    "investigate_larger_change": "INVESTIGATE LARGER CHANGE",
    "watch": "WATCH",
}


def show(snapshot: dict, actionable_only: bool = False,
         show_evidence: bool = False) -> None:
    recs = snapshot["recommendations"]

    print(f"\ncaptured {snapshot['captured_at'][:16].replace('T', ' ')} UTC")
    print(f"{snapshot['signals_in_file']} signals -> "
          f"{snapshot['clusters_total']} clusters -> "
          f"{snapshot['clusters_processed']} evaluated -> {len(recs)} recommendations")
    if snapshot.get("duplicates_collapsed"):
        print(f"({snapshot['duplicates_collapsed']} duplicate-citation "
              f"recommendation(s) merged)")

    counts = snapshot.get("tier_counts", {})
    summary = "  ".join(f"{TIER_LABEL.get(t, t)}: {counts[t]}"
                        for t in TIER_ORDER if t in counts)
    print(summary)

    by_tier: dict[str, list] = {}
    for r in recs:
        by_tier.setdefault(r["recommended_action"], []).append(r)

    for tier in TIER_ORDER:
        group = by_tier.get(tier)
        if not group:
            continue
        if actionable_only and tier == "watch":
            print(f"\n... {len(group)} watch item(s) hidden (--all to show)")
            continue

        print(f"\n{'=' * 74}\n{TIER_LABEL[tier]}  ({len(group)})\n{'=' * 74}")
        for r in sorted(group, key=lambda x: -(x.get("total_score") or 0)):
            print(f"\n  {r['trend'][:68]}")
            print(f"    score {r.get('total_score')}/5   confidence {r['confidence']}")
            if r.get("match"):
                print(f"    cite  {r['match'].get('citation', '')}")
            for step in r["action_plan"]:
                print(f"      - {step}")
            if show_evidence and r.get("evidence"):
                print("    evidence:")
                for e in r["evidence"][:4]:
                    note = f" -- {e['note'][:70]}" if e.get("note") else ""
                    print(f"      [{e.get('tier', '?')}] {e.get('source', '?')}{note}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Capture or replay a demo run")
    ap.add_argument("--capture", action="store_true", help="run the pipeline and save")
    ap.add_argument("--replay", action="store_true", help="show a saved run, no API calls")
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--signals", default="01_data/signals.json")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--all", action="store_true",
                    help="on replay, include watch items (hidden by default)")
    ap.add_argument("--evidence", action="store_true",
                    help="on replay, print the evidence trail for each item")
    args = ap.parse_args()

    if args.capture:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY not set -- capture needs it. "
                             "Replay does not.")

        print(f"capturing {args.limit} cluster(s) from offset {args.offset} ...")
        snap = capture(args.signals, args.snapshot, args.limit, args.offset)
        print(f"\nsaved -> {args.snapshot}")
        print(f"{len(snap['recommendations'])} recommendation(s): "
              + ", ".join(f"{k}={v}" for k, v in snap["tier_counts"].items()))
        print("\nReplay it with:  python 02_src/demo_snapshot.py --replay")
        return

    if args.replay or not args.capture:
        show(load_snapshot(args.snapshot),
             actionable_only=not args.all, show_evidence=args.evidence)


if __name__ == "__main__":
    main()
