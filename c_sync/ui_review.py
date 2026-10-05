"""Human in the loop: the instructor's decision on each recommendation.

C-sync only recommends. This panel is where a person approves, asks for
changes, or rejects, with an optional note and, when one was drafted, the
proposed fix. Decisions are kept in a small JSON file (REVIEWS_PATH, default
01_data/reviews.json, gitignored) keyed by the trend, so they survive a
restart and show on the Dashboard. Nothing in the course changes here.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from ui_adapter import BACKEND
from ui_components import e, pill, section

DECISIONS = {  # status shown afterwards, colour, icon
    "approved": ("Approved", "green", ":material/check_circle:"),
    "changes": ("Needs changes", "amber", ":material/edit_note:"),
    "rejected": ("Rejected", "rose", ":material/cancel:"),
}
ACTIONS = {"approved": "Approve", "changes": "Request changes", "rejected": "Reject"}   # what the buttons say
PENDING = ("Awaiting instructor review", "violet")


def reviews_path() -> Path:
    return Path(os.environ.get("REVIEWS_PATH") or BACKEND / "01_data" / "reviews.json")


def load_reviews() -> dict:
    try:
        data = json.loads(reviews_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_review(trend: str, decision: str, note: str = "", fix: dict | None = None) -> dict:
    """Record one decision (replacing any earlier one for the trend)."""
    if decision not in DECISIONS:
        raise ValueError(f"unknown decision {decision!r}")
    entry = {"decision": decision, "note": (note or "").strip(),
             "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if fix:
        entry["fix"] = {k: fix.get(k) for k in ("before", "after", "changes", "needs_verification", "new_names")}
    data = load_reviews()
    data[trend] = entry
    path = reviews_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return entry


def status_pill(trend: str, reviews: dict | None = None) -> str:
    entry = (reviews if reviews is not None else load_reviews()).get(trend)
    if not entry or entry.get("decision") not in DECISIONS:
        return pill(PENDING[0], PENDING[1])
    label, tone, _ = DECISIONS[entry["decision"]]
    return pill(label, tone)


def _record(trend: str, decision: str, index: int) -> None:
    note = st.session_state.get(f"review_note_{index}", "")
    draft = st.session_state.get(f"fix_draft_{index}")
    attach = st.session_state.get(f"review_attach_fix_{index}", False)
    save_review(trend, decision, note, draft if (attach and draft and draft.get("can_fix")) else None)
    st.session_state[f"review_saved_{index}"] = DECISIONS[decision][0]


def review_panel(record: dict, index: int) -> None:
    trend = record.get("trend") or "Untitled"
    section("Your decision", "C-sync only recommends. Nothing in the course changes until an instructor approves.",
            "HUMAN IN THE LOOP")
    entry = load_reviews().get(trend)
    if entry and entry.get("decision") in DECISIONS:
        label, tone, _ = DECISIONS[entry["decision"]]
        when = entry.get("at", "").replace("T", " ").replace("+00:00", " UTC")
        note = f' · “{e(entry["note"])}”' if entry.get("note") else ""
        fixed = " · with the drafted fix" if entry.get("fix") else ""
        st.html(f'<div class="sr-reveal {"green" if tone == "green" else "amber"}">{e(label)} by an instructor'
                f'<div style="font-size:.9rem;font-weight:500;color:#c9d7e9;margin-top:6px">{e(when)}{note}{fixed}</div></div>')
        st.caption("You can change this decision below.")
    else:
        st.html(f'<div class="sr-reveal">{e(PENDING[0])}<div style="font-size:.9rem;font-weight:500;color:#c9d7e9;margin-top:6px">'
                f'Review the plan and the proposed fix, then decide.</div></div>')

    st.text_area("Note for the team (optional)", key=f"review_note_{index}",
                 placeholder="e.g. Approved; update cell 36 before the next cohort.")
    draft = st.session_state.get(f"fix_draft_{index}")
    if draft and draft.get("can_fix"):
        st.checkbox("Attach the drafted fix to this decision", value=True, key=f"review_attach_fix_{index}")
    with st.container(horizontal=True):
        for key, (_, _, icon) in DECISIONS.items():
            st.button(ACTIONS[key], key=f"review_{key}_{index}", icon=icon, type="primary" if key == "approved" else "secondary",
                      on_click=_record, args=(trend, key, index))
    saved = st.session_state.pop(f"review_saved_{index}", None)
    if saved:
        st.toast(f"Saved: {saved}", icon=":material/task_alt:")
