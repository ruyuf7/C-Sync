"""C-Sync presentation components. All backend strings are HTML escaped."""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path

import streamlit as st


CSS = """
<style>
:root{--bg:#080c17;--panel:#111827;--line:#26344a;--text:#f0f4ff;--muted:#a7b5cc;--violet:#a78bfa;--cyan:#67e8f9;--green:#34d399;--amber:#fbbf24;--rose:#f0abfc;--ease-out:cubic-bezier(0.23, 1, 0.32, 1);--ease-in-out:cubic-bezier(0.77, 0, 0.175, 1);--dur-press:160ms;--dur-ui:200ms}
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"]{background:#080c17;color:var(--text)}
[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 8% 4%,#282047 0,transparent 31%),radial-gradient(circle at 92% 8%,#103047 0,transparent 27%),#080c17}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1240px;padding-top:3.4rem;padding-bottom:2.5rem}
h1,h2,h3{letter-spacing:-.04em} p{line-height:1.6}
[data-testid="stButton"] button{border-radius:10px;min-height:36px;font-weight:650;transition:transform var(--dur-press) var(--ease-out),border-color 150ms ease}
@media (hover:hover) and (pointer:fine){[data-testid="stButton"] button:hover{border-color:#7c8fab}}
[data-testid="stButton"] button:active{transform:scale(0.97)}
[data-testid="stButton"] button[kind="primary"]{background:linear-gradient(105deg,#7c3aed,#315bdc);border:1px solid #a78bfa;color:white}
[data-testid="stExpander"]{border:1px solid #2a3950;border-radius:14px;background:#111827aa}
[data-testid="stSelectbox"]>div>div,[data-testid="stTextInput"]>div>div{border-radius:12px}
.sr-brand{font-weight:800;letter-spacing:-.06em;font-size:1.35rem;color:#f6f4ff}
.sr-brand-mark{display:inline-grid;place-items:center;width:34px;height:34px;margin-right:9px;border-radius:10px;background:linear-gradient(145deg,#974cf4,#36b4e8);color:white;box-shadow:0 0 22px #8b5cf64c}
.sr-topline{display:flex;justify-content:space-between;align-items:center;padding:2px 0 8px;border-bottom:1px solid #ffffff18;margin-bottom:8px}
.sr-data-tag{border:1px solid #67e8f955;border-radius:100px;color:#a9eaf4;padding:6px 12px;font-size:.74rem;letter-spacing:.08em;font-weight:750}
.sr-eyebrow{font-size:.78rem;letter-spacing:.16em;text-transform:uppercase;font-weight:800;color:#92a6c6;margin-bottom:12px}
.sr-hero-title{font-size:clamp(2.6rem,4.6vw,4.4rem);line-height:1;letter-spacing:-.06em;margin:6px 0 8px;color:#fff;font-weight:800}
.sr-gradient{background:linear-gradient(110deg,#fff 5%,#bea7ff 51%,#6de4fb 100%);background-clip:text;-webkit-background-clip:text;color:transparent}
.sr-hero-copy{font-size:1rem;color:#bfcbdd;max-width:640px;line-height:1.6;margin:0}
.sr-page-title{font-size:clamp(1.7rem,2.6vw,2.6rem);line-height:1.08;letter-spacing:-.05em;font-weight:800;color:#fff;margin:4px 0 6px}
.sr-lead{font-size:.96rem;color:#b8c5d9;max-width:780px;margin-bottom:14px}
.sr-section{margin-top:24px;margin-bottom:10px}
.sr-section h2{font-size:clamp(1.2rem,1.8vw,1.6rem);margin:2px 0 4px;color:#fff}
.sr-section p{color:#aab9ce;margin:0}
.sr-glass{background:linear-gradient(150deg,#1d2739e8,#101827dd);border:1px solid #ffffff20;border-radius:16px;padding:16px 18px;box-shadow:0 12px 36px #0003}
.sr-glass.selected{border-color:#9e80ee;box-shadow:0 0 0 1px #8767e188,0 15px 45px #7c3aed2b}
.sr-kicker{font-size:.75rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#90a3bf}
.sr-card-title{font-size:1.05rem;font-weight:750;color:#fff;line-height:1.35;margin:6px 0 6px;overflow-wrap:anywhere}
.sr-card-copy{font-size:.9rem;color:#aebed2;line-height:1.5;min-height:0;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.sr-pill{display:inline-flex;align-items:center;gap:6px;border-radius:100px;padding:6px 10px;font-size:.72rem;letter-spacing:.05em;font-weight:780;text-transform:uppercase;border:1px solid #ffffff29;background:#ffffff0b;color:#d8e3f4}
.sr-pill.violet{color:#d3c0ff;border-color:#a78bfa66;background:#a78bfa1c}.sr-pill.cyan{color:#a0edfa;border-color:#67e8f966;background:#67e8f91c}.sr-pill.green{color:#83e9bc;border-color:#34d39966;background:#34d3991c}.sr-pill.amber{color:#ffd988;border-color:#fbbf2466;background:#fbbf241c}.sr-pill.rose{color:#fac8f9;border-color:#f0abfc66;background:#f0abfc1c}
.sr-metric-number{font-size:clamp(1.6rem,2.6vw,2.4rem);font-weight:800;letter-spacing:-.06em;color:#fff;line-height:1;margin:8px 0}
.sr-metric-label{font-size:.88rem;color:#a9bad0}
.sr-pipeline{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:6px 0 16px}
.sr-step{position:relative;background:#111b2bdc;border:1px solid #ffffff18;border-radius:12px;padding:9px 12px;min-height:0}
.sr-step.active{background:linear-gradient(135deg,#392568,#153a64);border-color:#a78bfa99;box-shadow:0 0 28px #8b5cf638}
.sr-step-num{font-size:.69rem;letter-spacing:.14em;color:#87a4ca;font-weight:800}.sr-step-name{font-size:.92rem;font-weight:800;color:white;margin-top:5px}.sr-step-desc{font-size:.73rem;color:#aebed2;margin-top:3px}
.sr-radar{height:410px;position:relative;display:grid;place-items:center;overflow:hidden;border-radius:28px;background:radial-gradient(circle,#271c55 0,#121b37 36%,#0c1325 75%);border:1px solid #9481d34d;box-shadow:inset 0 0 60px #6a4cc62b,0 25px 70px #0008}
.sr-radar::before{content:"";position:absolute;width:350px;height:350px;border:1px solid #8c84d158;border-radius:50%;box-shadow:0 0 0 55px #815dcb0b,0 0 0 110px #815dcb08}
.sr-radar::after{content:"";position:absolute;width:1px;height:380px;background:linear-gradient(transparent,#a78bfa44,transparent);transform:rotate(46deg)}
.sr-radar-sweep{position:absolute;width:360px;height:360px;border-radius:50%;background:conic-gradient(from 80deg,transparent 0deg,transparent 285deg,#7c3aed25 340deg,#67e8f955 359deg);animation:sr-sweep 12s linear infinite}
.sr-radar-core{z-index:2;width:132px;height:132px;border-radius:50%;display:grid;place-items:center;text-align:center;background:radial-gradient(circle at 40% 30%,#9c74ff,#4c2b92 68%,#271a57);border:2px solid #c4b5fd88;box-shadow:0 0 45px #855cf470;color:white;font-size:1.05rem;font-weight:800;line-height:1.2}
.sr-dot{position:absolute;width:10px;height:10px;border-radius:50%;background:#71e8fa;box-shadow:0 0 13px #71e8fa,0 0 27px #71e8fa;z-index:2}.sr-dot.d2{background:#c39aff;box-shadow:0 0 15px #b58aff,0 0 25px #b58aff}.sr-dot.d3{background:#66e0af;box-shadow:0 0 15px #66e0af}
.sr-radar-caption{position:absolute;left:20px;bottom:16px;z-index:3;color:#aebed2;font-size:.75rem;letter-spacing:.11em;text-transform:uppercase}
.sr-data-radar{height:480px;position:relative;overflow:hidden;border-radius:26px;border:1px solid #8f7bcc55;background:radial-gradient(circle at center,#2b2055 0,#15243b 25%,#0d1526 66%);box-shadow:inset 0 0 80px #7254a420,0 22px 60px #0006}
.sr-data-grid{position:absolute;inset:8%;border-radius:50%;border:1px solid #a3a5da48;box-shadow:0 0 0 75px #8c84d10a,0 0 0 150px #8c84d108;pointer-events:none}
.sr-data-grid::before,.sr-data-grid::after{content:"";position:absolute;left:50%;top:0;bottom:0;border-left:1px solid #7f96b126}.sr-data-grid::after{transform:rotate(90deg)}
.sr-data-core{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:110px;height:110px;border-radius:50%;display:grid;place-items:center;text-align:center;font-size:.9rem;line-height:1.25;font-weight:800;letter-spacing:.06em;color:white;background:radial-gradient(circle at 35% 20%,#9d75f5,#49317c);border:2px solid #d4bfff9c;box-shadow:0 0 45px #9061ed6e;z-index:3}
.sr-data-node{position:absolute;display:block;transform:translate(-50%,-50%);border-radius:50%;border:2px solid #f6f8ff;z-index:4;background:var(--node-color);box-shadow:0 0 0 8px color-mix(in srgb,var(--node-color) 16%,transparent),0 0 var(--node-glow) var(--node-color);transition:transform .2s,box-shadow .2s}
.sr-data-node:hover,.sr-data-node:focus{transform:translate(-50%,-50%) scale(1.45);box-shadow:0 0 0 13px color-mix(in srgb,var(--node-color) 23%,transparent),0 0 35px var(--node-color);outline:none}
.sr-data-node span{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
.sr-data-caption{position:absolute;left:23px;bottom:17px;color:#b6c8df;font-size:.72rem;font-weight:800;letter-spacing:.14em;z-index:4}
.sr-radar-legend{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0;color:#adbed3;font-size:.78rem}.sr-radar-legend div{display:flex;align-items:center;gap:7px}.sr-radar-legend i{display:inline-block;width:9px;height:9px;border-radius:50%;box-shadow:0 0 10px currentColor}
.sr-flow{display:flex;align-items:stretch;gap:12px;flex-wrap:wrap}.sr-flow-item{flex:1;min-width:170px}.sr-flow-arrow{display:flex;align-items:center;color:#8b78c3;font-size:1.65rem}
.sr-timeline{border-left:2px solid #7564af;margin:12px 0 12px 15px;padding-left:24px}.sr-timeline-item{position:relative;margin:0 0 20px;padding:13px 18px;background:#151e30;border:1px solid #ffffff17;border-radius:14px}.sr-timeline-item::before{content:"";position:absolute;left:-32px;top:22px;width:12px;height:12px;border-radius:50%;background:#67e8f9;box-shadow:0 0 13px #67e8f9}
.sr-ring-wrap{display:flex;justify-content:center;padding:18px 0}.sr-ring{--pct:0%;--ring-color:#a78bfa;width:150px;height:150px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--ring-color) var(--pct),#29354d 0);box-shadow:0 0 40px color-mix(in srgb,var(--ring-color) 20%,transparent)}.sr-ring-inner{width:124px;height:124px;border-radius:50%;background:#101827;display:flex;flex-direction:column;justify-content:center;align-items:center}.sr-ring-value{font-size:1.7rem;line-height:1.05;font-weight:800;letter-spacing:-.04em;color:white}.sr-ring-denom{font-size:.72rem;color:#b5c4d8}.sr-ring-label{margin-top:4px;font-size:.6rem;font-weight:800;letter-spacing:.1em;color:#c5d2e5;text-align:center;max-width:96px}
.sr-decision{padding:24px;border-radius:20px;background:linear-gradient(135deg,#2b1e52,#172648 52%,#101f36);border:1px solid #a78bfa88;box-shadow:0 25px 70px #0008,0 0 60px #7c3aed20}.sr-decision.green{border-color:#34d3998a;background:linear-gradient(135deg,#173e3b,#172648 60%,#0f202b)}.sr-decision.amber{border-color:#fbbf248a;background:linear-gradient(135deg,#4a351d,#252945 60%,#111a2b)}.sr-decision-title{font-size:clamp(1.6rem,2.8vw,2.6rem);font-weight:800;line-height:1.08;letter-spacing:-.06em;color:white;margin:17px 0}.sr-decision-copy{color:#c9d7e9;font-size:1.08rem;max-width:800px}
.sr-compare{min-height:270px;min-width:0;overflow:hidden}.cs-compare-row{display:grid;grid-template-columns:1fr 1fr;gap:28px;align-items:stretch}@media(max-width:850px){.cs-compare-row{grid-template-columns:1fr}}.sr-compare-label{font-size:.73rem;letter-spacing:.15em;font-weight:800;color:#94a8c7}.sr-compare-title{font-size:clamp(1.3rem,2vw,2rem);font-weight:800;color:#fff;line-height:1.25;margin:20px 0;overflow-wrap:anywhere}.sr-compare-text{font-size:.95rem;color:#bbcadb;line-height:1.55;overflow-wrap:anywhere}
.sr-reveal{border:1px solid #ffffff28;border-radius:14px;padding:14px 18px;margin:12px 0;background:linear-gradient(110deg,#30235f,#162746);font-size:1.2rem;font-weight:800;letter-spacing:-.03em;color:#fff}.sr-reveal.green{background:linear-gradient(110deg,#14433c,#173048);border-color:#34d39966}.sr-reveal.amber{background:linear-gradient(110deg,#4b351a,#242a47);border-color:#fbbf2466}
.sr-empty{padding:32px;border:1px dashed #60718e;border-radius:19px;color:#b7c5d9;background:#141d2bb8}
@keyframes sr-sweep{to{transform:rotate(360deg)}}
[data-testid="stSidebar"]{background:#0b1120;border-right:1px solid #ffffff14}
[data-testid="stSidebar"] [data-testid="stButton"] button{justify-content:flex-start;min-height:34px;width:100%;padding-left:14px}[data-testid="stSidebar"] [data-testid="stButton"] button>div{justify-content:flex-start;width:100%}
.cs-side-brand{font-weight:800;font-size:1.15rem;letter-spacing:-.04em;color:#f6f4ff;margin:2px 0 12px}
/* fade in: whatever Streamlit (re)mounts eases in. fade out: content that is
   about to be replaced (a page change) dims while the new page renders */
@keyframes cs-in{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]{animation:cs-in var(--dur-ui) var(--ease-out) both}
[data-stale="true"],.stale-element{opacity:.15!important;transition:opacity .22s ease!important}
.cs-funnel{display:inline-block;max-width:100%}.cs-squares{display:flex;align-items:flex-start;gap:14px;flex-wrap:wrap;padding:6px 0 2px}.cs-slot{height:var(--slot);display:flex;align-items:center;justify-content:center}
.cs-sq{display:flex;flex-direction:column;align-items:center;gap:6px;opacity:0;animation:cs-in .5s ease forwards;animation-delay:calc(var(--i)*170ms)}
.cs-box{display:grid;place-items:center;border-radius:14px;color:#fff;font-weight:800;letter-spacing:-.04em;
  background:linear-gradient(145deg,color-mix(in srgb,var(--c) 72%,#080c17),color-mix(in srgb,var(--c) 40%,#080c17));
  border:1px solid color-mix(in srgb,var(--c) 70%,#fff);box-shadow:0 0 18px color-mix(in srgb,var(--c) 24%,transparent)}
.cs-sq-label{font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:#a9bad0;font-weight:800;text-align:center;max-width:130px}
.cs-arrow{color:#7d8fb0;font-size:1.3rem;height:var(--slot);display:flex;align-items:center}
.cs-dash{background:linear-gradient(150deg,#1d2739e8,#101827dd);border:1px solid #ffffff1c;border-radius:14px;padding:12px 14px;margin-bottom:6px}
.cs-dash .top{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}
.cs-dash .ttl{font-weight:750;color:#fff;font-size:.98rem;line-height:1.3;overflow-wrap:anywhere}
.cs-dash .score{font-weight:800;color:#fff;font-size:1.2rem;white-space:nowrap}.cs-dash .score small{color:#9fb0c8;font-size:.72rem;font-weight:600}
.cs-dash .cite{font-family:ui-monospace,Consolas,monospace;font-size:.78rem;color:#b8c9df;margin-top:6px;overflow-wrap:anywhere}
@media(max-width:850px){.sr-pipeline{display:flex;overflow-x:auto;padding-bottom:7px}.sr-step{min-width:140px;min-height:82px}.sr-radar{height:330px}.sr-data-radar{height:360px}.sr-flow-arrow{display:none}.sr-hero-title{font-size:3.7rem}.sr-decision{padding:26px}}
@media(prefers-reduced-motion:reduce){.sr-radar-sweep{animation:none}.sr-glass,[data-testid="stButton"] button{transition:none}[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"],.cs-sq{animation:none;opacity:1}[data-stale="true"],.stale-element{transition:none!important}[data-testid="stButton"] button:active{transform:none}}
.sr-radar-stage{position:absolute;top:0;bottom:0;left:50%;aspect-ratio:1;transform:translateX(-50%)}.sr-sweep{position:absolute;inset:5%;pointer-events:none;z-index:2}.sr-sweep-beam{position:absolute;inset:0;border-radius:50%;background:conic-gradient(from 0deg,transparent 0deg 290deg,#67e8f914 320deg,#67e8f955 356deg,#a5f3fc 360deg);animation:sr-spin 6s linear infinite}.sr-sweep-beam::after{content:"";position:absolute;left:50%;top:0;height:50%;border-left:2px solid #a5f3fcd9;box-shadow:0 0 14px #67e8f9}@keyframes sr-spin{to{transform:rotate(360deg)}}.sr-data-node::before{content:"";position:absolute;inset:-3px;border-radius:50%;border:2px solid var(--node-color);opacity:0;pointer-events:none;animation:sr-ping 6s linear infinite;animation-delay:inherit}@keyframes sr-ping{0%{opacity:.95;transform:scale(1)}14%{opacity:0;transform:scale(2.4)}100%{opacity:0;transform:scale(2.4)}}.cs-ev{display:flex;flex-direction:column;gap:8px}.cs-ev details{background:#111b2bdc;border:1px solid #ffffff18;border-left:3px solid var(--ev,#5b6b86);border-radius:12px;opacity:0;animation:cs-rise var(--dur-ui) var(--ease-out) forwards}.cs-ev details.ok{--ev:#34d399;background:linear-gradient(90deg,#34d3991c,#111b2bdc 55%);border-color:#34d39955}.cs-ev details.bad{--ev:#f87171}.cs-ev details.warn{--ev:#fbbf24}.cs-ev summary{cursor:pointer;list-style:none;display:flex;align-items:center;gap:10px;padding:10px 14px;flex-wrap:wrap}.cs-ev summary::-webkit-details-marker{display:none}.cs-ev summary::after{content:"▸";margin-left:auto;color:#87a4ca;transition:transform .2s}.cs-ev details[open] summary::after{transform:rotate(90deg)}.cs-ev-num{font-size:.66rem;letter-spacing:.14em;color:#87a4ca;font-weight:800;white-space:nowrap}.cs-ev-title{color:#f0f4ff;font-weight:650;font-size:.92rem;flex:1 1 220px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.cs-ev-badge{font-size:.72rem;font-weight:800;border-radius:100px;padding:3px 9px;border:1px solid;white-space:nowrap}.cs-ev-badge.stars{color:#fcd34d;border-color:#fbbf2455;background:#fbbf2412}.cs-ev-badge.ok{color:#6ee7b7;border-color:#34d39966;background:#34d3991a}.cs-ev-badge.bad{color:#fca5a5;border-color:#f8717166;background:#f871711a}.cs-ev-badge.warn{color:#fcd34d;border-color:#fbbf2466;background:#fbbf241a}.cs-ev-body{padding:0 14px 12px;color:#c2d0e1;font-size:.88rem;line-height:1.5;word-break:break-word}.cs-ev-body a{color:#67e8f9;font-weight:650}.cs-reveal{opacity:0;animation:cs-rise .6s ease forwards}@keyframes cs-rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}@media (prefers-reduced-motion:reduce){.sr-sweep-beam,.sr-data-node,.sr-data-node::before{animation:none!important}.cs-ev details,.cs-reveal{animation:none!important;opacity:1!important}}[data-testid="stElementContainer"]:has(iframe[srcdoc*="cs-scroll-top"]),[data-testid="stElementContainer"]:has(iframe[srcdoc*="cs-radar-bridge"]){position:absolute!important;height:0!important;overflow:hidden;margin:0!important}h1.sr-page-title,h1.sr-hero-title{padding:0}.sr-data-node::after{content:"";position:absolute;inset:-9px;border-radius:50%}.st-key-radar_nav{display:none!important}.cs-ev summary:focus-visible{outline:2px solid #a78bfa;outline-offset:2px;border-radius:10px}.cs-ev-body a:focus-visible{outline:2px solid #67e8f9;outline-offset:2px;border-radius:4px}.sr-data-node:focus-visible{outline:2px solid #f6f8ff;outline-offset:4px}[data-testid="stButton"] button:focus-visible{outline:2px solid #c4b5fd;outline-offset:2px}@media(max-width:640px){.st-key-stagebar [data-testid="stHorizontalBlock"]{flex-wrap:nowrap!important;overflow-x:auto;padding-bottom:4px}.st-key-stagebar [data-testid="stColumn"]{flex:0 0 auto!important;width:auto!important;min-width:118px!important}}</style>
"""


# The last rules of CSS: page titles are real <h1>s, so Streamlit's heading
# padding is dropped; every radar light gets an invisible tap area at least
# 36px wide; the hidden buttons radar lights press are hidden (radar_bridge);
# on phones the five stage buttons stay one sideways-scrolling row. Focus:
# evidence cards, their links, radar lights and every button get a solid
# outline (Streamlit's own ring is a 50%-alpha primary glow, under 3:1).
# Radar pings are a scaling ring on ::before (transform + opacity only) rather
# than an animated filter on all 29 lights. primaryColor (.streamlit/config.toml)
# is #7C3AED so white text on Streamlit's violet chips reaches 5.7:1.
# Never put a "<" inside CSS, even in a /* comment */: st.html's sanitizer
# then drops the whole <style> block and every page loses its styling.


def inject_css() -> None:
    st.html(CSS)


def e(value: object) -> str:
    return escape(str(value) if value is not None else "", quote=True)


_MARK = Path(__file__).resolve().parents[1] / "03_assets" / "logo" / "c-sync-mark.svg"


def brand() -> None:
    # st.html strips inline <svg>, so the mark goes in as an image data URI
    try:
        data = base64.b64encode(_MARK.read_bytes()).decode()
        mark = (f'<img src="data:image/svg+xml;base64,{data}" alt="" width="38" height="38" '
                f'style="margin-right:8px;flex:none">')
    except OSError:
        mark = '<span class="sr-brand-mark">✦</span>'
    st.html(f'<div class="sr-topline"><div class="sr-brand" style="display:flex;align-items:center">'
            f'{mark}C-<span style="color:#a78bfa">sync</span></div></div>')


STAGES = [
    ("Discover", "Signals appear"),
    ("Verify", "Is it real?"),
    ("Compare", "Is it taught?"),
    ("Evaluate", "How important?"),
    ("Decide", "What should change?"),
]


def pipeline(active: str | None) -> None:
    steps = "".join(
        f'<div class="sr-step {"active" if name == active else ""}"><div class="sr-step-num">0{i}</div><div class="sr-step-name">{name}</div><div class="sr-step-desc">{description}</div></div>'
        for i, (name, description) in enumerate(STAGES, 1)
    )
    st.html(f'<div class="sr-pipeline">{steps}</div>')


def page_intro(kicker: str, title: str, subtitle: str) -> None:
    st.html(f'<div class="sr-eyebrow">{e(kicker)}</div><h1 class="sr-page-title">{e(title)}</h1><p class="sr-lead">{e(subtitle)}</p>')


def section(title: str, subtitle: str = "", kicker: str = "THE STORY") -> None:
    st.html(f'<div class="sr-section"><div class="sr-eyebrow">{e(kicker)}</div><h2>{e(title)}</h2><p>{e(subtitle)}</p></div>')


def pill(text: str, tone: str = "violet") -> str:
    return f'<span class="sr-pill {tone}">{e(text)}</span>'


def stat(value: object, label: str, note: str = "") -> None:
    st.html(f'<div class="sr-glass"><div class="sr-kicker">{e(note)}</div><div class="sr-metric-number">{e(value)}</div><div class="sr-metric-label">{e(label)}</div></div>')


def trend_card(record: dict, selected: bool = False, maturity: int | None = None) -> None:
    action = action_label(record.get("recommended_action", ""))
    confidence = record.get("confidence")
    pct = f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "—"
    evidence = len(record.get("evidence") or [])
    note = record.get("verification_note") or "Verification note unavailable."
    maturity_badge = pill(f"Maturity {maturity}/5", "green") if maturity is not None else ""
    st.html(f'<div class="sr-glass {"selected" if selected else ""}"><div class="sr-kicker">TREND · {e(pct)} CONFIDENCE</div><div class="sr-card-title">{e(record.get("trend", "Untitled"))}</div><div class="sr-card-copy">{e(note)}</div><div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:18px">{pill(action,action_tone(record.get("recommended_action", "")))}{pill(f"{evidence} evidence item(s)","cyan")}{maturity_badge}</div></div>')


def action_label(action: str) -> str:
    return {"watch":"Keep watching", "update_existing_material":"Update existing material", "add_optional_content":"Add optional content", "add_new_lesson":"Create a new lesson", "investigate_larger_change":"Investigate larger change"}.get(action, action.replace("_", " ").title())


def action_tone(action: str) -> str:
    return {"watch":"amber", "update_existing_material":"violet", "add_optional_content":"cyan", "add_new_lesson":"green", "investigate_larger_change":"rose"}.get(action, "violet")


def score_ring(label: str, value: int | float, color: str, delay: float | None = None) -> None:
    # The arc is a display-only fraction of the stored score.
    percent = max(0.0, min(100.0, float(value) / 5 * 100))
    reveal = f' cs-reveal" style="animation-delay:{delay:.2f}s' if delay is not None else ""
    st.html(f'<div class="sr-ring-wrap{reveal}"><div class="sr-ring" style="--pct:{percent:.1f}%;--ring-color:{color}"><div class="sr-ring-inner"><div class="sr-ring-value">{e(f"{value:g}")}</div><div class="sr-ring-denom">out of 5</div><div class="sr-ring-label">{e(label)}</div></div></div></div>')


def empty_state(title: str, detail: str) -> None:
    st.html(f'<div class="sr-empty"><div class="sr-card-title">{e(title)}</div><div class="sr-card-copy">{e(detail)}</div></div>')


def squares_funnel(stages: list[tuple[int, str]]) -> None:
    """From noise to curriculum: one square per stage, its AREA proportional to
    the count, shrinking left to right. Counts come from the recorded run."""
    counts = [max(int(n or 0), 0) for n, _ in stages]
    top = max(counts + [1])
    colors = ["#64748b", "#6d7fd6", "#8b5cf6", "#a78bfa", "#34d399"]
    cells, sides = [], []
    for i, ((n, label), color) in enumerate(zip(stages, colors)):
        side = 46 + 150 * (max(int(n or 0), 0) / top) ** 0.5
        sides.append(side)
        font = max(0.9, side / 70)
        cells.append(f'<div class="cs-sq" style="--i:{i}"><div class="cs-slot"><div class="cs-box" style="--c:{color};'
                     f'width:{side:.0f}px;height:{side:.0f}px;font-size:{font:.2f}rem">{e(n)}</div></div>'
                     f'<div class="cs-sq-label">{e(label)}</div></div>')
    arrow = '<div class="cs-arrow" aria-hidden="true">→</div>'
    summary = ", ".join(f"{n} {label}" for n, label in stages)
    st.html(f'<div class="cs-funnel" style="--slot:{max(sides + [46]):.0f}px"><div class="cs-squares" role="img" '
            f'aria-label="From noise to curriculum: {e(summary)}">{arrow.join(cells)}</div></div>')


def dashboard_card(record: dict, status: str = "") -> None:
    """Compact dashboard row: tier, score, confidence and citation. The action
    plan stays behind the "Full plan" expander under the card. Everything
    comes from the stored record."""
    action = record.get("recommended_action", "")
    match = record.get("match") or {}
    conf = record.get("confidence")
    score = record.get("total_score")
    cite = match.get("citation") or "No curriculum match"
    pills = status + pill(action_label(action), action_tone(action))
    if isinstance(conf, (int, float)):
        pills += pill(f"{conf:.2f} confidence", "cyan")
    if match.get("content_type"):
        pills += pill(match["content_type"], "rose")
    shown = score if score is not None else "—"
    st.html(f'<div class="cs-dash"><div class="top"><div class="ttl">{e(record.get("trend", "Untitled"))}</div>'
            f'<div class="score">{e(shown)}<small> / 5</small></div></div>'
            f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px">{pills}</div>'
            f'<div class="cite">📎 {e(cite)}</div></div>')


def scroll_to_top(nonce: int) -> None:
    """Streamlit keeps the scroll position across reruns, so a page change
    would open the new page wherever the old one was scrolled. Jump back to
    the top. On a phone the sidebar covers the page, so close it too (the
    custom nav buttons don't, unlike st.navigation). st.html runs no script,
    so this is a same-origin iframe, hidden by the :has(...cs-scroll-top)
    rule in CSS; the nonce makes each jump a new element so the script runs again."""
    targets = '["[data-testid=stMain]","[data-testid=stAppViewContainer]","section.main"]'
    st.iframe(f"<!-- cs-scroll-top {nonce} --><script>"
              f"const w=window.parent,d=w.document;const up=()=>{{for(const s of {targets}){{"
              f"const el=d.querySelector(s);if(el)el.scrollTo({{top:0,behavior:'instant'}});}}"
              f"w.scrollTo({{top:0,behavior:'instant'}});}};"
              f"up();[80,250,600].forEach(t=>setTimeout(up,t));"
              f"if(w.innerWidth<768){{const sb=d.querySelector('[data-testid=stSidebar][aria-expanded=true]');"
              f"const b=sb&&sb.querySelector('[data-testid=stSidebarCollapseButton] button');"
              # Streamlit saves the sidebar state in localStorage as if the user
              # chose it; put the saved value back so a narrow window that is
              # widened later still opens the sidebar.
              f"if(b){{const k='stSidebarCollapsed-',s=w.localStorage,was=s.getItem(k);b.click();"
              f"const back=()=>{{try{{was===null?s.removeItem(k):s.setItem(k,was);}}catch(e){{}}}};"
              f"[60,300].forEach(t=>setTimeout(back,t));}}}}"
              f"</script>", height=1)
