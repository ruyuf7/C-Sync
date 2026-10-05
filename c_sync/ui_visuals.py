"""Interactive radar built from the actual recorded trend nodes."""

from __future__ import annotations

import math

import streamlit as st

from ui_components import action_label, e


ACTION_COLORS = {
    "watch": "#fbbf24",
    "update_existing_material": "#a78bfa",
    "add_optional_content": "#67e8f9",
    "add_new_lesson": "#34d399",
    "investigate_larger_change": "#f0abfc",
}


def radar_bridge() -> None:
    """Make a click on a radar light press that light's hidden st.button
    (key radar_go_<index>) instead of following its ?trend= link, which would
    reload the whole app and drop the session. The link stays as the fallback
    when script can't run. Same hidden-iframe trick as scroll_to_top; the
    listener lives on the parent document and replaces any earlier copy."""
    st.iframe("<!-- cs-radar-bridge --><script>"
              "const w=window.parent,d=w.document;"
              "if(w.__csRadar)d.removeEventListener('click',w.__csRadar,true);"
              "w.__csRadar=function(ev){const a=ev.target.closest&&ev.target.closest('a.sr-data-node');if(!a)return;"
              "const m=(a.getAttribute('href')||'').match(/trend=(\\d+)/);if(!m)return;"
              "const b=d.querySelector('.st-key-radar_go_'+m[1]+' button');if(!b)return;"
              "ev.preventDefault();b.click();};"
              "d.addEventListener('click',w.__csRadar,true);</script>", height=1)


def radar_map(visible: list[tuple[int, dict]], maturity: list[int | None]) -> None:
    """Render clickable trend nodes; the query parameter opens their story."""
    nodes = []
    total = max(len(visible), 1)
    for order, (index, record) in enumerate(visible):
        angle = 2 * math.pi * order / total - math.pi / 2
        radius = 39
        x = 50 + radius * math.cos(angle)
        y = 50 + radius * math.sin(angle)
        confidence = record.get("confidence")
        score = maturity[index] if index < len(maturity) else None
        diameter = 14 + 4 * score if isinstance(score, int) else 23
        glow = 12 + 24 * max(0, min(1, confidence if isinstance(confidence, (int, float)) else 0))
        color = ACTION_COLORS.get(record.get("recommended_action"), "#a78bfa")
        title = f"{record.get('trend', 'Untitled')} · {confidence:.0%} confidence · maturity {score}/5 · {action_label(record.get('recommended_action', ''))}" if isinstance(confidence, (int, float)) else record.get("trend", "Untitled")
        ping = 6 * order / total          # the beam reaches this node after `ping` seconds
        nodes.append(
            f'<a class="sr-data-node" href="?trend={index}" title="{e(title)}" '
            f'aria-label="Open trend story: {e(record.get("trend", "Untitled"))}" '
            f'style="left:{x:.2f}%;top:{y:.2f}%;width:{diameter:.1f}px;height:{diameter:.1f}px;'
            f'--node-color:{color};--node-glow:{glow:.1f}px;animation-delay:{ping:.2f}s"><span>{e(record.get("trend", "Untitled"))}</span></a>'
        )
    legend = "".join(f'<div><i style="background:{color}"></i>{e(action_label(action))}</div>'
                     for action, color in ACTION_COLORS.items()
                     if any(r.get("recommended_action") == action for _, r in visible))
    st.html(f'<div class="sr-data-radar"><div class="sr-radar-stage"><div class="sr-data-grid"></div><div class="sr-sweep"><div class="sr-sweep-beam"></div></div><div class="sr-data-core">C-SYNC</div>{"".join(nodes)}</div><div class="sr-data-caption">SELECT A LIGHT TO OPEN ITS STORY</div></div><div class="sr-radar-legend">{legend}</div>')
