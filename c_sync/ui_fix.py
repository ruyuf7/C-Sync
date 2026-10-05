"""The proposed fix on the Decision page: the course cell now, and a drafted fix.

Besides the Ask page, the only C-Sync file allowed to run an agent: it runs
the FixAgent, only when OPENAI_API_KEY is set, and only when the instructor
presses "Draft the fix". The draft is shown, never applied; the instructor
decides in the review panel below it.
"""

from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from ui_adapter import BACKEND, import_backend
from ui_components import e, empty_state, section

FIXABLE = {"update_existing_material"}


@st.cache_resource(show_spinner=False)
def _agent():
    import_backend()
    try:
        from dotenv import load_dotenv
        load_dotenv(BACKEND / ".env")   # never overrides a key already set
    except ImportError:
        pass
    from agents.fix import FixAgent
    return FixAgent()


def _draft(record: dict, index: int, signals: list) -> None:
    from agents.fix import full_cell, release_evidence
    cell = full_cell(record.get("match"), str(BACKEND / "vectorstore"))
    draft = _agent().draft(record, cell, release_evidence(record, signals))
    st.session_state[f"fix_draft_{index}"] = asdict(draft)


def fix_panel(record: dict, index: int, signals: list) -> None:
    section("The proposed fix", "The change itself, drafted from the full course cell and the release evidence. "
            "Nothing is applied until an instructor approves it.", "THE SOLUTION")
    match = record.get("match")
    if record.get("recommended_action") not in FIXABLE or not match:
        empty_state("No fix to draft", "Fix drafts are offered for “Update existing material” recommendations, "
                    "which point at a specific slide or lab cell.")
        return
    agent = _agent()
    key = f"fix_draft_{index}"
    if not agent.available():
        st.info("No OpenAI key is set, so no fix can be drafted. Set OPENAI_API_KEY and restart C-sync.",
                icon=":material/key_off:")
        return
    if key not in st.session_state:
        if not st.button("Draft the fix", key=f"fix_btn_{index}", icon=":material/auto_fix_high:", type="primary"):
            return
        with st.spinner("Reading the full cell and the release notes..."):
            _draft(record, index, signals)
    draft = st.session_state[key]
    lang = "python" if draft["kind"] == "code" else None
    if draft["mode"] == "error":
        st.warning(draft["reason"], icon=":material/warning:")
        st.button("Try again", key=f"fix_retry_{index}", icon=":material/refresh:",
                  on_click=lambda: st.session_state.pop(key, None))
        return
    st.html('<div style="display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 8px">'
            '<span class="sr-pill amber">Draft · not applied</span>'
            f'<span class="sr-pill violet">{e(match.get("citation") or "")}</span></div>')
    if not draft["can_fix"]:
        st.info(draft["reason"] or "No fix could be drafted from the evidence.", icon=":material/info:")
        st.code(draft["before"], language=lang)
    else:
        now, proposed = st.columns(2, gap="medium")
        with now:
            st.caption("NOW, IN THE COURSE")
            st.code(draft["before"], language=lang)
        with proposed:
            st.caption("PROPOSED FIX (DRAFT)")
            st.code(draft["after"], language=lang)
        if draft["changes"]:
            st.markdown("**What changes**\n" + "\n".join(f"- {c}" for c in draft["changes"]))
    checks = list(draft["needs_verification"])
    if draft["new_names"]:
        checks.append("Not in the lab cell or the release evidence, so check them in the library docs: "
                      + ", ".join(f"`{n}`" for n in draft["new_names"]))
    if checks:
        st.warning("**Check before approving**\n" + "\n".join(f"- {c}" for c in checks), icon=":material/rule:")
    if draft["from_excerpt"]:
        st.caption("The full cell wasn't in the course index, so this was drafted from the saved excerpt.")
    st.button("Draft again", key=f"fix_redo_{index}", icon=":material/refresh:",
              on_click=lambda: st.session_state.pop(key, None))
