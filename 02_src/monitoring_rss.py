"""
T4 -- RSS monitoring
=====================
Pulls recent posts from the official blogs of the tools our curriculum teaches.

Why RSS: an official blog post is the tool's own words, so it is a PRIMARY
source -- same tier as a GitHub release. And it needs no API key, no auth, no
rate limit. This is the cheapest primary source we have.

RSS is just XML that a site publishes listing its recent posts, already split
into title / link / date / summary. No HTML scraping needed.

Output: list[RawSignal] -- same contract as monitoring_github.py, so both feed
into cluster_signals() identically.

SETUP:
    pip install feedparser

Usage:
    python monitoring_rss.py
    python monitoring_rss.py --days 60
    python monitoring_rss.py --check          # verify every feed URL still works
"""

import argparse
from datetime import datetime, timedelta, timezone
from time import mktime

import feedparser

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from schemas import RawSignal


# Official blogs for tools the curriculum teaches.
# Feed URLs move. Run --check before the demo, not during it.
FEEDS = {
    "langchain_blog": "https://blog.langchain.com/rss.xml",
    "openai_blog": "https://openai.com/blog/rss.xml",
    "huggingface_blog": "https://huggingface.co/blog/feed.xml",
}

# Feeds we treat as secondary -- community discussion, not announcements.
# Useful precisely BECAUSE it is noisy: it gives the verification agent
# something to filter, which makes the filtering visible in the demo.
SECONDARY_FEEDS = {
    "hackernews_langchain": "https://hnrss.org/newest?q=langchain",
    "hackernews_agents": "https://hnrss.org/newest?q=AI+agents",
    "hackernews_langgraph": "https://hnrss.org/newest?q=langgraph",
}


def _published_iso(entry) -> str:
    """
    Feeds disagree on date fields (published/updated/created) and formats.
    feedparser normalises them into *_parsed structs -- use those.
    """
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = getattr(entry, attr, None)
        if parsed:
            return datetime.fromtimestamp(mktime(parsed), tz=timezone.utc).isoformat()
    return ""


def _clean(text: str, limit: int = 600) -> str:
    """Feed summaries often contain HTML. Strip tags, collapse whitespace."""
    if not text:
        return ""
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def fetch_feed(name: str, url: str, days: int = 30,
               tier: str = "primary", limit: int = 15) -> list[RawSignal]:
    """Parse one feed into RawSignals, filtered to the last N days."""
    feed = feedparser.parse(url)

    # feedparser does not raise -- it sets .bozo on a malformed/unreachable feed
    if getattr(feed, "bozo", False) and not feed.entries:
        print(f"  ! {name}: could not parse ({getattr(feed, 'bozo_exception', '')})")
        return []

    if not feed.entries:
        print(f"  ! {name}: feed parsed but returned no entries -- URL may have moved")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    signals: list[RawSignal] = []

    for entry in feed.entries[:limit]:
        published = _published_iso(entry)

        # keep undated entries rather than dropping them -- some feeds omit
        # dates entirely, and a missing date is not evidence of being old
        if published:
            when = datetime.fromisoformat(published)
            if when < cutoff:
                continue

        title = getattr(entry, "title", "").strip()
        if not title:
            continue

        summary = _clean(getattr(entry, "summary", "") or
                         getattr(entry, "description", ""))

        signals.append(RawSignal(
            title=title,
            source=name,
            source_tier=tier,
            summary=summary,
            url=getattr(entry, "link", ""),
            published=published,
        ))

    return signals


def fetch_all(days: int = 30, include_secondary: bool = False) -> list[RawSignal]:
    out: list[RawSignal] = []

    for name, url in FEEDS.items():
        found = fetch_feed(name, url, days=days, tier="primary")
        print(f"  {name}: {len(found)} posts in the last {days} days")
        out.extend(found)

    if include_secondary:
        for name, url in SECONDARY_FEEDS.items():
            found = fetch_feed(name, url, days=days, tier="secondary")
            print(f"  {name}: {len(found)} posts (secondary)")
            out.extend(found)

    out.sort(key=lambda s: s.published, reverse=True)
    return out


def check_feeds() -> None:
    """Verify every feed URL still resolves. Run this before the demo."""
    print("checking feeds ...\n")
    for name, url in {**FEEDS, **SECONDARY_FEEDS}.items():
        feed = feedparser.parse(url)
        n = len(feed.entries)
        status = "OK  " if n else "DEAD"
        print(f"  [{status}] {name:20} {n:3} entries   {url}")
    print("\nA DEAD feed usually means the URL moved. Check the blog's homepage")
    print("for a <link type='application/rss+xml'> tag, or try /rss, /feed, /atom.xml")


def main():
    ap = argparse.ArgumentParser(description="Fetch blog RSS posts as RawSignals")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--secondary", action="store_true",
                    help="include secondary-tier community feeds")
    ap.add_argument("--check", action="store_true",
                    help="verify feed URLs and exit")
    args = ap.parse_args()

    if args.check:
        check_feeds()
        return

    print("fetching RSS feeds ...")
    signals = fetch_all(days=args.days, include_secondary=args.secondary)

    print(f"\n{len(signals)} signals total\n")
    for s in signals:
        date = s.published[:10] if s.published else "no date"
        print(f"[{date}] ({s.source_tier}) {s.title}")
        print(f"   {s.summary[:110]}...")
        print(f"   {s.url}\n")


if __name__ == "__main__":
    main()
