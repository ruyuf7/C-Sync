#!/usr/bin/env python3
"""Build a gold-label dataset skeleton for run_eval.py.

Picks signals that DISCRIMINATE -- ones where a broken agent and a working
one give different answers -- and writes them out with empty gold blocks for
a human to fill.

    python 04_eval/make_dataset.py --signals 01_data/signals.json \\
        --out 04_eval/data/test_signals_graded.json

Then open the file and fill every `gold` value by hand. Leave anything you
are unsure of as null: run_eval.py skips nulls and reports what is missing,
whereas a guessed label silently corrupts the score it feeds.

WHY NOT LABEL EVERYTHING
60 signals at two minutes each is two hours and tells you little more than a
well-chosen 15. Most signals are routine version bumps that every version of
the agent handles identically -- they cost labelling time and measure nothing.
The cases below were each chosen because we watched the pipeline get one of
them wrong at some point.

FABRICATED SIGNALS
Two are included that do NOT come from your feed. They are invented, and that
is deliberate: without a claim you KNOW is false, truth_confidence_gap cannot
be computed at all. Verification scoring a real release highly proves nothing
on its own -- you need to see it score a fake one low.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_src"))


# The gold fields run_eval.py reads. Every one starts null: a human fills it.
GOLD_SKELETON = {
    "is_genuine": None,              # bool  -- is the central claim actually true?
    "confidence": None,              # float -- what SHOULD verification score it?
    "stale_presented_as_new": None,  # bool  -- old event dressed up as new?
    "rank": None,                    # int   -- 1 = most worth acting on, unique+contiguous
    "maturity": None,                # int 1-5
    "relevance": None,               # int 1-5
    "action_tier": None,             # watch | update_existing_material |
                                     # add_optional_content | add_new_lesson
}


# Signals that are NOT in the feed. Needed so verification has something
# false to score. Keep the wording close to a genuine signal -- a fabricated
# claim that reads obviously fake tests nothing.
FABRICATED = [
    {
        "title": "AgentTrace Protocol 2.0 becomes mandatory for all MCP server implementations",
        "source": "tech_blog",
        "source_tier": "secondary",
        "summary": ("A blog post states that AgentTrace Protocol 2.0 is now required for "
                    "every MCP server. No specification, no repository, and no other "
                    "outlet carries the claim."),
        "url": "",
        "published": "2026-09-05T00:00:00Z",
        "_why": "No such protocol exists. Verification must score this LOW. If it "
                "scores high, the agent is reasoning from plausibility rather than "
                "checking.",
    },
    {
        "title": "VelocityAgent claims 12x faster tool calling than LangChain",
        "source": "tweet",
        "source_tier": "secondary",
        "summary": ("A single tweet thread claims VelocityAgent outperforms LangChain on "
                    "tool dispatch by 12x. No benchmarks, no repository linked."),
        "url": "",
        "published": "2026-09-05T00:00:00Z",
        "_why": "We tested this live: the agent searched GitHub twice, found only an "
                "unrelated 0-star namesake, and scored 0.2. That behaviour is what "
                "this entry protects.",
    },
]


# Patterns that pick out the discriminating cases from a real signals file.
# Each carries the reason it earns a labelling slot.
WANTED = [
    (r"langchain==1\.4\.0|langchain\.mcp|MCP in LangChain",
     "A real feature release that SHOULD match curriculum material. We saw it "
     "match Week 4 slide 54 in one run and find nothing in the next -- this is "
     "the retrieval-drift case."),

    (r"LangGraph v0\.1|LangGraph Cloud",
     "Our strongest observed match: Week 3 lab cell 34, replacing "
     "create_conversational_retrieval_agent with create_react_agent. If a change "
     "breaks this, the demo case is gone."),

    (r"openai-python: v3\.\d+\.0$|openai-python: v3\.\d+\.\d+$",
     "A routine version bump. Must be watch, not add_new_lesson. Four of these "
     "each produced 'create a new lesson: Introduction to the OpenAI Python "
     "Library' before the version-bump gate existed."),

    (r"journalism|Ukraine|Frontline Defenders|teen development|1Password",
     "Verified, genuinely uncovered, and completely out of scope. Nine of ten "
     "trends in one batch became new-lesson recommendations including these."),

    (r"Organizing Context|Multi-Agent Harness",
     "In-domain and uncovered -- a legitimate add_new_lesson. The control that "
     "proves the in-domain gate did not simply disable the tier."),

    (r"An Alien Mind",
     "Matched the AI Ethics deck at 4.5/5. A non-obvious match the agent found "
     "on its own; worth protecting."),

    (r"langchain-anthropic|langchain-core",
     "Scored 0.0 confidence in one run, which should be unreachable. Included so "
     "the silent-fallback problem shows up in the numbers."),
]


REQUIRED = ("title", "source", "source_tier", "summary", "url", "published")


def pick(signals: list[dict], per_pattern: int = 2) -> list[dict]:
    chosen, seen = [], set()
    for pattern, why in WANTED:
        rx = re.compile(pattern, re.IGNORECASE)
        hits = 0
        for s in signals:
            if hits >= per_pattern:
                break
            if s["title"] in seen or not rx.search(s["title"]):
                continue
            seen.add(s["title"])
            entry = {k: s.get(k, "") for k in REQUIRED}
            entry["_why"] = why
            chosen.append(entry)
            hits += 1
        if hits == 0:
            print(f"  ! nothing matched /{pattern}/ -- that case will be unlabelled")
    return chosen


def main():
    ap = argparse.ArgumentParser(description="Build a gold-label dataset skeleton")
    ap.add_argument("--signals", default="01_data/signals.json")
    ap.add_argument("--out", default="04_eval/data/test_signals_graded.json")
    ap.add_argument("--per-pattern", type=int, default=2,
                    help="how many signals to take per discriminating case")
    ap.add_argument("--keep-why", action="store_true",
                    help="keep the _why notes. run_eval.py REJECTS unknown fields, "
                         "so use this only for a human-readable working copy")
    args = ap.parse_args()

    signals = json.loads(Path(args.signals).read_text(encoding="utf-8"))
    print(f"read {len(signals)} signals from {args.signals}")

    entries = pick(signals, args.per_pattern) + [dict(f) for f in FABRICATED]

    for e in entries:
        e["gold"] = dict(GOLD_SKELETON)
        if not args.keep_why:
            e.pop("_why", None)

    out = Path(args.out)
    out.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")

    print(f"\nwrote {len(entries)} entries -> {args.out}")
    print(f"  {len(entries) - len(FABRICATED)} from the feed, "
          f"{len(FABRICATED)} fabricated (needed for truth_confidence_gap)")
    print("\nNEXT: open the file and fill every gold value by hand.")
    print("  * Leave anything uncertain as null -- a guess corrupts the metric.")
    print("  * rank must be unique and contiguous (1, 2, 3, ...) across all entries.")
    print("  * action_tier has only four valid values; investigate_larger_change")
    print("    is declared in schemas.py but _select_tier never returns it.")
    print(f"\nThen: python 04_eval/run_eval.py --dataset {args.out} "
          f"--out 04_eval/results/baseline.json --repeats 3")


if __name__ == "__main__":
    main()
