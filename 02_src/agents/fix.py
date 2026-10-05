"""
Fix Agent
=========
Drafts the concrete fix behind an "update existing material" recommendation:
the course cell (or slide) as it is now, and the same cell rewritten for the
change the release made. It is a DRAFT for the instructor: nothing is applied,
and the instructor approves or rejects it on the Decision page.

Grounding, because a fix that invents an API is worse than no fix:
  * the model sees the FULL cell from the course index (the snapshot keeps only
    an excerpt) and the release evidence: the signal's own release notes plus
    the verification notes;
  * it may change only what that evidence supports, must keep the lab's
    student placeholders (`# YOUR CODE HERE`) intact, and must list anything
    the evidence does not settle under `needs_verification`;
  * code then flags every name in the draft that appears in neither the cell
    nor the evidence (`new_names`), so the instructor sees exactly which parts
    the model supplied from its own memory.

No key -> MODE_OFFLINE, and nothing is drafted. A model failure -> MODE_ERROR,
never a made-up draft.

Usage:
    from agents.fix import FixAgent, full_cell, release_evidence
    draft = FixAgent().draft(record, full_cell(record["match"]), release_evidence(record, signals))
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_DB = str(Path(__file__).resolve().parents[2] / "vectorstore")

MODE_MODEL, MODE_OFFLINE, MODE_ERROR = "model", "offline", "error"


# ---------------------------------------------------------------------------
# INPUTS -- the full course cell and the release evidence
# ---------------------------------------------------------------------------

def full_cell(match: dict | None, db_path: str = DEFAULT_DB) -> str | None:
    """The whole slide or lab cell the recommendation cites, from the course
    index; None when the index or the cell is not available."""
    if not match or not match.get("source_file"):
        return None
    try:
        import curriculum_ingest
        col = curriculum_ingest.get_collection(db_path)
        got = col.get(where={"$and": [{"source_file": match["source_file"]},
                                      {"slide_number": match.get("slide_number", 0)}]},
                      include=["documents"])
    except Exception:
        return None
    docs = [d for d in (got.get("documents") or []) if d]
    return "\n".join(docs) if docs else None


def release_evidence(record: dict, signals: list) -> list[str]:
    """What the run recorded about the release: the signal's own text (release
    notes or post summary), then the verification notes."""
    trend = record.get("trend") or ""
    lines = []
    for s in signals or []:
        title = getattr(s, "title", None) if not isinstance(s, dict) else s.get("title")
        summary = getattr(s, "summary", None) if not isinstance(s, dict) else s.get("summary")
        if title == trend and summary:
            lines.append(f"Release notes / post: {summary}")
    if record.get("verification_note"):
        lines.append(f"Verification: {record['verification_note']}")
    for ev in record.get("evidence") or []:
        if ev.get("note"):
            lines.append(f"Evidence: {ev['note']}")
    return lines


# ---------------------------------------------------------------------------
# THE NEW-NAME CHECK -- what the draft uses that nothing we showed it contains
# ---------------------------------------------------------------------------

_NAME = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b"   # dotted paths
                   r"|\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"                     # snake_case
                   r"|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b")                # CamelCase


def names(text: str) -> set[str]:
    found = set()
    for n in _NAME.findall(text or ""):
        found.add(n)
        found.update(n.split("."))          # a dotted path also vouches for its parts
    return found


def new_names(before: str, after: str, evidence: list[str]) -> list[str]:
    known = names(before) | names("\n".join(evidence))
    return sorted(n for n in names(after) - known if "." not in n)   # a dotted path's parts are checked one by one


# ---------------------------------------------------------------------------
# AGENT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You draft the fix for one course cell that a verified library release affects.
An instructor reviews your draft; nothing is applied without their approval.

Rules:
1. Change only what the release evidence supports. If the evidence says something is
   deprecated but does not name its replacement, do NOT invent one: keep the working code,
   add a short comment explaining the deprecation, and put "which API replaces X" under
   needs_verification.
2. This is a teaching notebook. Keep every student placeholder (such as `# YOUR CODE HERE`)
   exactly where it is, and keep the cell's structure and comments unless they are wrong.
3. Never claim a version, API or behaviour the evidence does not state.
4. Keep the fix small: the fewest lines that address the change.

Reply with JSON only:
{"can_fix": true|false,
 "after": "the full rewritten cell (empty if can_fix is false)",
 "changes": ["one short line per change you made"],
 "needs_verification": ["anything the instructor must check before applying"],
 "reason": "one sentence: why this fix, or why no fix can be drafted"}"""


@dataclass
class FixDraft:
    mode: str                                        # MODE_MODEL / MODE_OFFLINE / MODE_ERROR
    kind: str = "code"                               # "code" for a lab cell, "text" for a slide
    before: str = ""
    after: str = ""
    can_fix: bool = False
    changes: list[str] = field(default_factory=list)
    needs_verification: list[str] = field(default_factory=list)
    new_names: list[str] = field(default_factory=list)
    reason: str = ""
    from_excerpt: bool = False                       # True when the full cell was unavailable


class FixAgent:
    """Drafts, never applies, the fix for one recommendation's course cell."""

    def __init__(self, client=None, model: str = MODEL):
        self._client = client
        self.model = model

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None
        self._client = OpenAI()
        return self._client

    def available(self) -> bool:
        return self._get_client() is not None

    def draft(self, record: dict, cell: str | None, evidence: list[str]) -> FixDraft:
        match = record.get("match") or {}
        kind = "code" if match.get("content_type") == "lab" else "text"
        before = cell or match.get("matched_text") or ""
        base = dict(kind=kind, before=before, from_excerpt=cell is None)
        client = self._get_client()
        if client is None:
            return FixDraft(mode=MODE_OFFLINE, reason="No OpenAI key is set, so no fix can be drafted.", **base)
        if not before.strip():
            return FixDraft(mode=MODE_MODEL, reason="The recommendation cites no course content to fix.", **base)
        user = (f"TREND: {record.get('trend')}\n"
                f"RECOMMENDED ACTION: {record.get('recommended_action')}\n"
                f"ACTION PLAN:\n" + "\n".join(f"- {s}" for s in record.get("action_plan") or []) + "\n\n"
                f"RELEASE EVIDENCE:\n" + "\n".join(f"- {e}" for e in evidence) + "\n\n"
                f"COURSE {'LAB CELL' if kind == 'code' else 'SLIDE'} ({match.get('citation')}):\n{before}")
        try:
            resp = client.chat.completions.create(
                model=self.model, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])
            data = json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:
            return FixDraft(mode=MODE_ERROR, reason=f"The model call failed ({type(e).__name__}); no fix was drafted.", **base)
        can_fix = bool(data.get("can_fix")) and bool(str(data.get("after") or "").strip())
        after = str(data.get("after") or "") if can_fix else ""
        return FixDraft(
            mode=MODE_MODEL, can_fix=can_fix, after=after,
            changes=[str(c) for c in data.get("changes") or []][:8],
            needs_verification=[str(c) for c in data.get("needs_verification") or []][:8],
            new_names=new_names(before, after, evidence) if can_fix else [],
            reason=str(data.get("reason") or ""), **base)
