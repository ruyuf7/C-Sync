"""The Ask page: the Instructor Companion over one recorded recommendation.

The ONLY C-Sync page that may call a model, and only when OPENAI_API_KEY is
set. Without a key it restates the recorded facts. It explains the recorded
decision; it never changes it -- a human approves.
"""

from __future__ import annotations

import streamlit as st

from ui_adapter import BACKEND, import_backend
from ui_components import action_label, e, empty_state, page_intro, pill
from ui_pages import choose_trend


SUGGESTED = (
    "Why did it get this recommendation?",
    "What did verification actually check?",
    "Which course material is affected, and how?",
)

_OUTCOME_PILL = {
    "searched": ("Curriculum searched", "green"),
    "search_failed": ("Curriculum search FAILED", "rose"),
    "skipped": ("Curriculum search skipped", "amber"),
    "no_trace": ("No curriculum trace", "amber"),
}


@st.cache_resource(show_spinner=False)
def _agent():
    import_backend()
    try:
        from dotenv import load_dotenv
        load_dotenv(BACKEND / ".env")   # never overrides a key already set
    except ImportError:
        pass
    from agents.companion import CompanionAgent
    return CompanionAgent()


def _show_calls(calls: list[dict]) -> None:
    if calls:
        with st.expander(f"Looked up {len(calls)} thing(s)", icon=":material/manage_search:"):
            for c in calls:
                st.html(f'<div style="margin-bottom:8px"><b>[{e(c.get("label", ""))}] {e(c["tool"])}</b> '
                        f'<code>{e(c["arguments"])}</code><div style="color:#a7b5cc">{e(c["summary"])}</div></div>')


def _show_sources(sources: list, problems: list) -> None:
    """What each cited label points at, and anything the citation check caught."""
    if problems:
        st.warning("Citation check: " + "; ".join(problems), icon=":material/rule:")
    if sources:
        with st.expander(f"Sources cited ({len(sources)})", icon=":material/format_quote:"):
            for label, text in sources:
                st.html(f'<div style="margin-bottom:8px"><b>[{e(label)}]</b> '
                        f'<span style="color:#a7b5cc">{e(text)}</span></div>')


def ask(records: list[dict]) -> None:
    page_intro("COMPANION", "Ask about a recommendation",
               "Questions about one recorded trend, answered from its saved evidence and agent trace. "
               "The companion explains the decision; it never changes it. A human approves.")
    if not records:
        empty_state("Nothing to ask about", "The recorded run has no recommendations.")
        return
    from agents.companion import MODE_ERROR, curriculum_state
    agent = _agent()
    selected, record = choose_trend(records, "ask")

    state, outcome = curriculum_state(record)
    label, tone = _OUTCOME_PILL[state]
    st.html(f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 10px">'
            f'{pill(action_label(record.get("recommended_action", "")), "violet")}{pill(label, tone)}</div>')
    if state == "search_failed":
        st.error(outcome, icon=":material/error:")

    if not agent.available():
        st.info("No OpenAI key is set, so the companion cannot answer free-form questions. "
                "Set OPENAI_API_KEY and restart C-sync. Below is what the run recorded.",
                icon=":material/key_off:")
        from agents.companion import offline_reply
        st.markdown(offline_reply(record).text)
        return

    key = f"ask_history_{selected}"
    history = st.session_state.setdefault(key, [])
    for turn in history:
        with st.chat_message("user"):
            st.markdown(turn["q"])
        with st.chat_message("assistant"):
            st.markdown(turn["a"])
            _show_sources(turn["sources"], turn["problems"])
            _show_calls(turn["calls"])

    question = None
    if not history:
        with st.container(horizontal=True):
            for n, text in enumerate(SUGGESTED):
                if st.button(text, key=f"ask_suggest_{n}"):
                    question = text
    typed = st.chat_input("Ask about this recommendation")
    question = typed or question
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Reading the recorded trace..."):
            reply = agent.answer(record, question, [(t["q"], t["a"]) for t in history])
        if reply.mode == MODE_ERROR:
            st.warning(reply.text, icon=":material/warning:")
        else:
            st.markdown(reply.text)
            _show_sources(reply.sources, reply.citation_problems)
        _show_calls(reply.tool_calls)
    if reply.mode != MODE_ERROR:
        history.append({"q": question, "a": reply.text, "calls": reply.tool_calls,
                        "sources": reply.sources, "problems": reply.citation_problems})
