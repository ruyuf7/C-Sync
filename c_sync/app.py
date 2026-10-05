"""C-Sync — a story-led Streamlit interface over the AI Trend Agent's recorded run."""

from __future__ import annotations

import streamlit as st

from ui_adapter import BACKEND, backend_ready, load_recorded_run
from ui_ask import ask
from ui_components import STAGES, brand, inject_css, scroll_to_top
from ui_pages import (
    dashboard, decision, evaluation, gap, go, home, how_it_works, radar, trend_story,
)


st.set_page_config(
    page_title="C-sync",
    page_icon=":material/sync_alt:",
    layout="wide",
    initial_sidebar_state="auto",   # open on desktop, collapsed on phones
)

# Left panel: every page. Top bar: only the five pipeline stages.
PAGES = [
    ("Home", "home"), ("Dashboard", "dashboard"), ("Radar", "radar"),
    ("Trend story", "auto_stories"), ("The gap", "difference"),
    ("Evaluation", "analytics"), ("Decision", "tips_and_updates"), ("Ask", "forum"),
    ("How it works", "help_outline"),
]
STAGE_OF = {"Radar": "Discover", "Trend story": "Verify",
            "The gap": "Compare", "Evaluation": "Evaluate", "Decision": "Decide"}
STAGE_PAGE = {"Discover": "Radar", "Verify": "Trend story", "Compare": "The gap",
              "Evaluate": "Evaluation", "Decide": "Decision"}


@st.cache_data(ttl=300, max_entries=2, show_spinner=False)
def recorded_data():
    return load_recorded_run()


def side_panel() -> None:
    with st.sidebar:
        brand()
        for page, icon in PAGES:
            st.button(page, icon=f":material/{icon}:", key=f"nav_{page}", width="stretch",
                      type="primary" if st.session_state["page"] == page else "tertiary",
                      on_click=go, args=(page,))


def stage_bar() -> None:
    active = STAGE_OF.get(st.session_state["page"])
    # keyed so the CSS can keep it one scrollable row on phones instead of five stacked buttons
    with st.container(key="stagebar"):
        cols = st.columns(len(STAGES), gap="small")
        for i, (col, (name, _)) in enumerate(zip(cols, STAGES), 1):
            with col:
                st.button(f"0{i}  {name}", key=f"stage_{name}", width="stretch",
                          type="primary" if name == active else "secondary",
                          on_click=go, args=(STAGE_PAGE[name],))


def main() -> None:
    st.session_state.setdefault("page", "Home")
    st.session_state.setdefault("selected_trend", 0)
    query_trend = st.query_params.get("trend")
    if query_trend is not None:
        try:
            st.session_state["selected_trend"] = int(query_trend)
            st.session_state["page"] = "Trend story"
        except (TypeError, ValueError):
            pass
        st.query_params.clear()
    inject_css()
    side_panel()
    stage_bar()

    if not backend_ready():
        st.error(f"AI Trend Agent backend not found at {BACKEND}. Set CSYNC_BACKEND to the checkout and restart the app.")
        return
    try:
        snapshot, signals = recorded_data()
    except (ImportError, FileNotFoundError, OSError, ValueError, SystemExit):
        st.error("The recorded run could not be loaded. Check the backend path and saved snapshot files.")
        return
    records = snapshot.get("recommendations") or []
    routes = {
        "Home": lambda: home(snapshot, records, signals),
        "Dashboard": lambda: dashboard(records),
        "Radar": lambda: radar(records, signals),
        "Trend story": lambda: trend_story(records, signals),
        "The gap": lambda: gap(records),
        "Evaluation": lambda: evaluation(records, signals),
        "Decision": lambda: decision(records, signals),
        "Ask": lambda: ask(records),
        "How it works": lambda: how_it_works(snapshot),
    }
    routes.get(st.session_state["page"], routes["Home"])()
    if st.session_state.get("scroll_top", 0) != st.session_state.get("scrolled_to", 0):
        st.session_state["scrolled_to"] = st.session_state["scroll_top"]
        scroll_to_top(st.session_state["scroll_top"])


if __name__ == "__main__":
    main()
