"""Story-led pages for C-Sync. Every value shown comes from the recorded run."""

from __future__ import annotations

from urllib.parse import urlparse

import re

import streamlit as st

from ui_adapter import (
    stored_scores, tier_order,
)
from ui_components import (
    action_label, action_tone, dashboard_card, e, empty_state, page_intro, pill,
    score_ring, section, squares_funnel, stat, trend_card,
)
from ui_fix import fix_panel
from ui_review import load_reviews, review_panel, status_pill
from ui_visuals import radar_bridge, radar_map


def go(page: str, selected: int | None = None) -> None:
    if st.session_state.get("page") != page:
        st.session_state["scroll_top"] = st.session_state.get("scroll_top", 0) + 1
    st.session_state["page"] = page
    if selected is not None:
        st.session_state["selected_trend"] = selected


def selected_index(records: list[dict]) -> int:
    value = st.session_state.get("selected_trend", 0)
    return value if isinstance(value, int) and 0 <= value < len(records) else 0


def choose_trend(records: list[dict], page: str) -> tuple[int, dict]:
    key = f"trend_selector_{page}"
    st.session_state[key] = selected_index(records)

    def changed() -> None:
        st.session_state["selected_trend"] = st.session_state[key]

    index = st.selectbox("Choose a recorded trend", range(len(records)),
                         format_func=lambda i: records[i].get("trend", "Untitled"),
                         key=key, on_change=changed)
    return index, records[index]


def source_link(value: str) -> None:
    parsed = urlparse(value or "")
    if parsed.scheme in {"https", "http"} and parsed.netloc:
        st.link_button("Open source", value, icon=":material/open_in_new:")


_REPO = r"([\w.-]+/[\w.-]+)"


def _repo_of(item: dict) -> str | None:
    """owner/name this evidence item is about, if it names one."""
    note, url = item.get("note") or "", item.get("url") or ""
    for pattern, text in ((r"github\.com/" + _REPO, url), (r"^(?:github_lookup|verify_release)\('" + _REPO, note),
                          (r"^" + _REPO + r":", note)):
        m = re.search(pattern, text)
        if m:
            return m.group(1).removesuffix(".git").lower()
    return None


def _stars_by_repo(items: list[dict]) -> dict[str, int]:
    """Star counts exactly as github_lookup recorded them -- nothing fetched."""
    found = {}
    for item in items:
        m = re.search(r"matched " + _REPO + r" \(([\d,]+) stars", item.get("note") or "")
        if m:
            found[m.group(1).lower()] = int(m.group(2).replace(",", ""))
    return found


def _evidence_view(item: dict, stars: dict[str, int]) -> tuple[str, str, list[tuple[str, str]]]:
    """(title, card state, badges) for one evidence item."""
    note = item.get("note") or ""
    tool = re.match(r"(github_lookup|verify_release)\(", note)
    badges, state = [], ""
    repo = _repo_of(item)
    if (item.get("source") == "github") and repo in stars:
        badges.append((f"⭐ {stars[repo]:,} stars", "stars"))
    if tool and tool.group(1) == "verify_release":
        title = f"Release check · {repo or 'unknown repository'}"
        if "CONFIRMED" in note:
            state = "ok"
            badges.append(("✅ Release confirmed", "ok"))
        elif "NOT confirm" in note:
            state = "bad"
            badges.append(("❌ Not confirmed", "bad"))
        else:
            state = "warn"
            badges.append(("⚠️ Check could not run", "warn"))
    elif tool:
        title = f"Repository lookup · {repo or 'unknown repository'}"
        if " matched " not in note:
            state = "warn"
            badges.append(("⚠️ No matching repository", "warn"))
    else:
        title = note.split("  (")[0] or item.get("source") or "Unnamed source"
    return title, state, badges


def evidence_cards(record: dict) -> None:
    items = record.get("evidence") or []
    if not items:
        empty_state("No verification evidence saved", "The recorded run does not include source notes for this trend.")
        return
    stars = _stars_by_repo(items)
    cards = []
    for index, item in enumerate(items):
        title, state, badges = _evidence_view(item, stars)
        tier = str(item.get("tier") or "tier unavailable").upper()
        chips = "".join(f'<span class="cs-ev-badge {tone}">{e(text)}</span>' for text, tone in badges)
        url = item.get("url") or ""
        link = (f'<div style="margin-top:6px"><a href="{e(url)}" target="_blank" rel="noopener">Open source ↗</a></div>'
                if url.startswith(("http://", "https://")) else "")
        cards.append(
            f'<details class="{state}" style="animation-delay:{0.04 * index:.2f}s">'
            f'<summary><span class="cs-ev-num">SOURCE {index + 1:02d} · {e(tier)} · {e(item.get("source") or "")}</span>'
            f'<span class="cs-ev-title" title="{e(title)}">{e(title)}</span>{chips}</summary>'
            f'<div class="cs-ev-body">{e(item.get("note") or "No source note recorded.")}{link}</div></details>')
    st.html(f'<div class="cs-ev">{"".join(cards)}</div>')


def match_details(match: dict | None) -> None:
    if not match:
        empty_state("No saved curriculum match", "A missing match alone does not prove the course has a gap.")
        return
    sim_pill = (pill(f"Similarity {match['similarity']:.3f}", "violet")
                if isinstance(match.get("similarity"), (int, float)) else "")
    st.html(f'<div class="sr-glass"><div class="sr-kicker">MATCHED COURSE MATERIAL</div><div class="sr-card-title">{e(match.get("citation") or "Citation unavailable")}</div><div class="sr-card-copy">{e(match.get("topic") or "Topic unavailable")}</div><div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:16px">{pill(str(match.get("content_type") or "Unknown"),"cyan")}{pill("Exact identifier: " + str(match["exact_match"]),"green") if match.get("exact_match") else ""}{sim_pill}</div></div>')
    with st.expander("Read the cited course content", icon=":material/description:"):
        st.caption(match.get("source_file") or "Source file unavailable")
        st.code(match.get("matched_text") or "No excerpt saved.", language=None)


def maturity_scores(records: list[dict]) -> list[int | None]:
    """Maturity per record from the STORED confidence (evaluation.py's band
    function) -- the same number the Dashboard page implies; nothing is re-run."""
    return [stored_scores(r)[0] for r in records]


def home(snapshot: dict, records: list[dict], signals: list) -> None:
    st.html('<div class="sr-eyebrow">CURRICULUM INTELLIGENCE</div><h1 class="sr-hero-title">C-<span class="sr-gradient">sync</span></h1>'
            '<p class="sr-hero-copy">C-sync watches the technology landscape, checks what is real, compares it with the course, and recommends what should change. A human approves.</p>')
    actionable = sum(1 for r in records if r.get("recommended_action") != "watch")
    section("From noise to curriculum.", "Each square is a stage of the run; its size is how much survives.", "THE BIG PICTURE")
    squares_funnel([
        (snapshot.get("signals_in_file", len(signals)), "Signals"),
        (snapshot.get("clusters_total", 0), "Trend clusters"),
        (snapshot.get("clusters_processed", 0), "Assessed"),
        (len(records), "Recommendations"),
        (actionable, "Actionable"),
    ])
    with st.container(horizontal=True):
        st.button("Open the dashboard", type="primary", icon=":material/dashboard:",
                  on_click=go, args=("Dashboard",))
        st.button("Explore the radar", icon=":material/radar:", on_click=go, args=("Radar",))
    if records:
        featured = next((r for r in records if r.get("recommended_action") == "add_new_lesson"), records[0])
        index = records.index(featured)
        section("A decision you can trace.", "Start with one trend and follow it to its recommendation.", "FEATURED")
        trend_card(featured)
        st.button("Follow this trend's story", icon=":material/arrow_forward:",
                  key="featured_story", on_click=go, args=("Trend story", index))


def dashboard(records: list[dict]) -> None:
    page_intro("ALL RECOMMENDATIONS", "Dashboard", "Every recommendation in the run, most urgent first.")
    if not records:
        empty_state("No recommendations", "The recorded run has no recommendations.")
        return
    order, labels = tier_order()
    present = [t for t in order if any(r.get("recommended_action") == t for r in records)]
    f1, f2, f3 = st.columns([2.2, 1.2, 1])
    with f1:
        chosen = st.multiselect("Action", present, default=present, format_func=action_label)
    with f2:
        kind = st.segmented_control("Material", ["All", "Lab", "Slides"], default="All")
    with f3:
        actionable_only = st.toggle("Actionable only", value=False)
    rank = {t: i for i, t in enumerate(order)}
    rows = [(i, r) for i, r in enumerate(records)
            if r.get("recommended_action") in chosen
            and not (actionable_only and r.get("recommended_action") == "watch")
            and (kind in (None, "All") or ((r.get("match") or {}).get("content_type") == ("lab" if kind == "Lab" else "slides")))]
    rows.sort(key=lambda p: (rank.get(p[1].get("recommended_action"), len(order)), -(p[1].get("total_score") or 0)))
    st.caption(f"{len(rows)} of {len(records)} recommendations")
    if not rows:
        empty_state("Nothing matches", "Change the filters above.")
        return
    reviews = load_reviews()
    for offset in range(0, len(rows), 2):
        cols = st.columns(2, gap="small")
        for col, (index, record) in zip(cols, rows[offset:offset + 2]):
            with col:
                dashboard_card(record, status_pill(record.get("trend") or "", reviews))
                with st.container(horizontal=True):
                    st.button("Story", key=f"dash_story_{index}", icon=":material/arrow_forward:",
                              on_click=go, args=("Trend story", index))
                    st.button("Decision", key=f"dash_dec_{index}", icon=":material/tips_and_updates:",
                              on_click=go, args=("Decision", index))
                with st.expander("Full plan"):
                    for step in record.get("action_plan") or []:
                        st.markdown(f"- {step}")


def radar(records: list[dict], signals: list) -> None:
    page_intro("01 / DISCOVER", "Technology radar", "Every light is an assessed trend. Select one to follow its story.")
    if not records:
        empty_state("The radar is quiet", "No trends are saved in the current recorded run.")
        return
    radar_map(list(enumerate(records)), maturity_scores(records))
    # One hidden button per light: radar_bridge makes a click on a light press
    # its button, so the story opens in this session instead of reloading the app.
    with st.container(key="radar_nav"):
        for index in range(len(records)):
            st.button(f"Open trend {index}", key=f"radar_go_{index}",
                      on_click=go, args=("Trend story", index))
    radar_bridge()
    st.caption("Node size = maturity · Glow = confidence · Color = action")
    st.button("Next: 02 Verify", type="primary", icon=":material/arrow_forward:",
              on_click=go, args=("Trend story",))


def trend_story(records: list[dict], signals: list) -> None:
    page_intro("02 / VERIFY", "Why is this trending?", "Follow one recorded trend from its original signal to the evidence used for verification.")
    if not records:
        empty_state("No trend selected", "The recorded run has no trends to inspect.")
        return
    selected, record = choose_trend(records, "story")
    left, right = st.columns([1.45, 0.55], gap="large")
    with left:
        st.html(f'<div class="sr-glass"><div class="sr-kicker">TECHNOLOGY TREND</div><div class="sr-card-title" style="font-size:clamp(1.7rem,3vw,2.7rem)">{e(record.get("trend"))}</div><p style="color:#c2d0e1;font-size:1.05rem">{e(record.get("verification_note") or "No verification note recorded.")}</p></div>')
    with right:
        confidence = record.get("confidence")
        stat(f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "—", "Recorded confidence", "VERIFICATION")
        st.caption(f"{len(record.get('evidence') or [])} saved verification evidence item(s)")
    section("The signal trail", "Original monitoring signals are shown by published date when that date was recorded.", "WHAT WE SAW")
    originals = sorted((s for s in signals if s.title == record.get("trend")), key=lambda s: s.published or "")
    if originals:
        items = "".join(f'<div class="sr-timeline-item"><div class="sr-kicker">{e(s.published or "DATE UNAVAILABLE")} · {e(s.source_tier)}</div><div class="sr-card-title">{e(s.source)}</div><div class="sr-card-copy" style="display:block;min-height:0">{e((s.summary or "No summary recorded.")[:280])}</div></div>' for s in originals)
        st.html(f'<div class="sr-timeline">{items}</div>')
        for signal in originals:
            source_link(signal.url)
    else:
        empty_state("Original signal unavailable", "The saved signal file has no exact title match for this recorded trend.")
    section("The verification evidence", "What the Verification Agent checked. Open a source for its full note.", "WHY WE TRUST IT")
    evidence_cards(record)
    section("Should this affect the curriculum?", "Next, compare this trend with the saved curriculum match.", "NEXT STEP")
    st.button("Check the curriculum", type="primary", icon=":material/arrow_forward:",
              on_click=go, args=("The gap", selected))


def gap(records: list[dict]) -> None:
    page_intro("03 / COMPARE", "Is this trend already taught?", "Place the recorded trend beside the course material the Curriculum Agent found.")
    if not records:
        empty_state("No trend to compare", "The recorded run has no trend data.")
        return
    selected, record = choose_trend(records, "gap")
    match = record.get("match")
    # One grid row, so both cards stretch to the taller one's height.
    trend_card_html = (f'<div class="sr-glass sr-compare"><div class="sr-compare-label">TECHNOLOGY TREND</div>'
                       f'<div class="sr-compare-title">{e(record.get("trend"))}</div>'
                       f'<div class="sr-compare-text">{e(record.get("verification_note") or "No verification note recorded.")}</div></div>')
    if match:
        course_html = (f'<div class="sr-glass sr-compare"><div class="sr-compare-label">CURRENT COURSE MATERIAL</div>'
                       f'<div class="sr-compare-title">{e(match.get("citation") or "Citation unavailable")}</div>'
                       f'<div class="sr-compare-text">{e((match.get("matched_text") or "No excerpt saved.")[:260])}</div></div>')
    else:
        course_html = ('<div class="sr-glass sr-compare"><div class="sr-compare-label">CURRENT COURSE MATERIAL</div>'
                       '<div class="sr-compare-title">No match saved</div><div class="sr-compare-text">The recorded run '
                       'contains no cited slide or notebook cell for this trend.</div></div>')
    st.html(f'<div class="cs-compare-row">{trend_card_html}{course_html}</div>')
    action = record.get("recommended_action", "")
    if action == "add_new_lesson" and not match:
        reveal, tone, detail = "Curriculum gap identified", "", "The saved Recommendation Agent proposed a new lesson after the curriculum check."
    elif match:
        reveal, tone, detail = "Related material found", "green", "A saved course citation links this trend to existing material. The recorded action explains what should change."
    else:
        reveal, tone, detail = "Coverage not established", "amber", "No saved match exists, but the recorded agent did not recommend a new lesson. This is not presented as a confirmed gap."
    st.html(f'<div class="sr-reveal {tone}">{e(reveal)}</div>')
    st.caption(detail)
    if match:
        section("The specific course reference", "Inspect the citation used by the backend.", "MATCH DETAIL")
        match_details(match)
    st.button("See the evaluation", type="primary", icon=":material/arrow_forward:",
              on_click=go, args=("Evaluation", selected))


def evaluation(records: list[dict], signals: list) -> None:
    page_intro("04 / EVALUATE", "How important is this?", "Maturity says how established the trend is; relevance says how directly it hits what we teach.")
    if not records:
        empty_state("Nothing to evaluate", "No trends are available in the recorded run.")
        return
    selected, record = choose_trend(records, "evaluation")
    maturity, relevance, total = stored_scores(record)
    shown = f"cs_scores_{selected}"
    if not st.session_state.get(shown):
        st.button("Reveal the scores", type="primary", icon=":material/visibility:",
                  on_click=lambda: st.session_state.update({shown: True}))
    else:
        cols = st.columns(3, gap="small")
        for n, (col, title, value, color) in enumerate(zip(cols, ("MATURITY", "RELEVANCE", "OVERALL"),
                                                           (maturity, relevance, total),
                                                           ("#a78bfa", "#67e8f9", "#34d399"))):
            with col:
                if value is None:
                    empty_state(title.title(), "Not recorded.")
                else:
                    score_ring(title, value, color, delay=0.3 * n)
        st.caption("Overall is the stored total score. Maturity is evaluation.py's band for the stored confidence; "
                   "relevance is solved from total = ½ maturity + ½ relevance.")
    section("Evidence used", "The score is grounded in verification and curriculum evidence.", "TRACEABILITY")
    match_details(record.get("match"))
    with st.expander("Verification sources", icon=":material/fact_check:"):
        evidence_cards(record)
    st.button("See the recommended action", type="primary", icon=":material/arrow_forward:",
              on_click=go, args=("Decision", selected))


def decision(records: list[dict], signals: list) -> None:
    page_intro("05 / DECIDE", "What should we do?", "A specific curriculum action, with its reason and supporting evidence close at hand.")
    if not records:
        empty_state("No decision recorded", "The saved run has no recommendations.")
        return
    selected, record = choose_trend(records, "decision")
    action = record.get("recommended_action", "")
    tone = "green" if action == "add_new_lesson" else "amber" if action == "watch" else ""
    match = record.get("match")
    area = match.get("citation") if match else "No cited course area"
    ev_pill = pill(f"{len(record.get('evidence') or [])} evidence item(s)", "cyan")
    st.html(f'<div class="sr-decision {tone}"><div class="sr-kicker">RECOMMENDATION · {e(record.get("trend"))}</div><div class="sr-decision-title">{e(action_label(action))}</div><div class="sr-decision-copy">{e(record.get("verification_note") or "No verification note recorded.")}</div><div style="margin-top:25px;display:flex;gap:8px;flex-wrap:wrap">{status_pill(record.get("trend") or "")}{ev_pill}{pill(area,"violet") if match else pill("No cited course area","amber")}</div></div>')
    section("The action plan", "Steps returned by the saved Recommendation Agent run.", "WHAT CHANGES")
    for number, step in enumerate(record.get("action_plan") or [], 1):
        st.html(f'<div class="sr-glass" style="margin-bottom:12px;display:flex;align-items:flex-start;gap:18px"><span class="sr-pill violet">{number:02d}</span><div style="color:#e6edf9;font-size:1.04rem;line-height:1.55">{e(step)}</div></div>')
    fix_panel(record, selected, signals)
    review_panel(record, selected)
    chain = [
        ("Trend", record.get("trend") or "Unavailable"),
        ("Evidence", f"{len(record.get('evidence') or [])} saved item(s)"),
        ("Curriculum", area),
        ("Evaluation", f"Saved total {record.get('total_score')}/5" if record.get("total_score") is not None else "Total unavailable"),
        ("Decision", action_label(action)),
    ]
    flow_arrow = '<div class="sr-flow-arrow">→</div>'
    cells = "".join(f'<div class="sr-flow-item sr-glass"><div class="sr-kicker">{e(title)}</div><div class="sr-card-title" style="font-size:1rem">{e(text)}</div></div>{flow_arrow if i<4 else ""}' for i,(title,text) in enumerate(chain))
    with st.expander("Why this decision? · Evidence chain", icon=":material/account_tree:"):
        st.html(f'<div class="sr-flow">{cells}</div>')
    with st.expander("Inspect verification evidence", icon=":material/fact_check:"):
        evidence_cards(record)


def how_it_works(snapshot: dict) -> None:
    page_intro("THE METHOD", "How does C-sync work?", "Five simple steps turn technology noise into curriculum action you can explain.")
    stages = [
        ("01", "Listen", "We watch trusted technology sources for new developments.", "sensors"),
        ("02", "Verify", "We check the evidence before treating a signal as meaningful.", "verified"),
        ("03", "Compare", "We look for related slides and labs in the curriculum.", "difference"),
        ("04", "Evaluate", "We measure how established the trend is and how strongly it connects to current material.", "analytics"),
        ("05", "Recommend", "We suggest a specific course action and keep the evidence attached.", "tips_and_updates"),
    ]
    for number, title, copy, icon in stages:
        st.html(f'<div class="sr-glass" style="margin-bottom:14px;display:flex;align-items:center;gap:25px"><div style="font-size:2rem;font-weight:800;color:#a78bfa;min-width:65px">{number}</div><div><div class="sr-card-title" style="margin:0 0 5px;font-size:1.45rem">{e(title)}</div><div style="color:#b7c6da">{e(copy)}</div></div></div>')
    section("Every recommendation has a trail.", "Follow the source, the course match, the agent evaluation, and the final action.", "EXPLAINABLE BY DESIGN")
    st.html(f'<div class="sr-glass selected" style="text-align:center"><div class="sr-metric-number">{e(snapshot.get("signals_in_file", "—"))} → {e(snapshot.get("clusters_total", "—"))} → {e(len(snapshot.get("recommendations") or []))}</div><div class="sr-metric-label">Signals → clusters → recommendations in this run</div></div>')
    st.button("Explore the radar", type="primary", icon=":material/radar:", on_click=go, args=("Radar",))
