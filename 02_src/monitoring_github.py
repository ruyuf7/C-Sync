"""
T3 -- GitHub monitoring
========================
Fetches recent releases from the repos our curriculum actually teaches.

Why releases and not commits/issues: a release is the tool's own announcement
of a change, which makes it a PRIMARY source. Commits are too noisy (hundreds
per week, most irrelevant) and issues are user reports, not announcements.

Output: list[RawSignal] -- the shared contract from schemas.py. This feeds
straight into cluster_signals().

SETUP -- you need a token:
    Unauthenticated: 60 requests/hour, shared across everyone on your IP.
    You WILL hit this. (It happened while writing this file.)
    With a token: 5,000/hour.

    1. github.com -> Settings -> Developer settings
       -> Personal access tokens -> Tokens (classic) -> Generate new
    2. No scopes needed -- public repo data only. Leave every box unchecked.
    3. Put it in .env at the repo root:   GITHUB_TOKEN=ghp_xxxx
    4. Confirm .env is in .gitignore BEFORE you commit anything.

Usage:
    python monitoring_github.py                 # default repo list
    python monitoring_github.py --days 60       # wider window
    python monitoring_github.py --repo langchain-ai/langchain
"""

import argparse
import os
import re
import time
from datetime import datetime, timedelta, timezone

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from schemas import RawSignal


# Repos our curriculum teaches. Add one whenever a new week introduces a tool.
# Keep this list SHORT -- every repo is an API call, and a trend from a tool we
# don't teach can only ever produce a "watch", which is noise in the demo.
WATCHED_REPOS = [
    "langchain-ai/langchain",
    "openai/openai-python",
    "chroma-core/chroma",
    "fastapi/fastapi",
    "langchain-ai/langgraph",
    "langchain-ai/langsmith-sdk",
]

API = "https://api.github.com"
TIMEOUT = 15
PRE_RELEASE_TAG = re.compile(r"(?:a|b|rc)\d+$", re.IGNORECASE)


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get(url: str, params: dict | None = None) -> list | dict | None:
    """
    One GET with the errors that actually happen handled explicitly.
    Returns None on failure rather than raising -- one dead repo should not
    kill the whole monitoring run.
    """
    try:
        r = requests.get(url, headers=_headers(), params=params, timeout=TIMEOUT)
    except requests.RequestException as e:
        print(f"  ! network error: {e}")
        return None

    if r.status_code == 403 and "rate limit" in r.text.lower():
        reset = r.headers.get("X-RateLimit-Reset")
        when = ""
        if reset:
            secs = int(reset) - int(time.time())
            when = f" resets in ~{max(0, secs)//60} min"
        auth = "unauthenticated (60/hr)" if not os.environ.get("GITHUB_TOKEN") \
               else "authenticated (5000/hr)"
        print(f"  ! RATE LIMITED [{auth}]{when} -- set GITHUB_TOKEN in .env")
        return None

    if r.status_code == 404:
        print(f"  ! not found: {url} (typo in the repo name?)")
        return None

    if not r.ok:
        print(f"  ! HTTP {r.status_code}: {r.text[:120]}")
        return None

    return r.json()


def _summarise(body: str | None, limit: int = 600) -> str:
    """
    Release notes are long markdown. Keep the head -- breaking changes and
    deprecations are announced at the top, not buried in the changelog.
    """
    if not body:
        return ""
    text = " ".join(body.split())          # collapse whitespace/newlines
    return text[:limit] + ("..." if len(text) > limit else "")


def fetch_releases(repo: str, days: int = 30, limit: int = 10) -> list[RawSignal]:
    """Recent published releases for one repo, newest first."""
    data = _get(f"{API}/repos/{repo}/releases", {"per_page": limit})
    if not data:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    signals: list[RawSignal] = []

    for rel in data:
        # drafts and prereleases are not what students will install
        tag = rel.get("tag_name") or ""
        if (rel.get("draft") or rel.get("prerelease")
            or PRE_RELEASE_TAG.search(tag)):
            continue

        published = rel.get("published_at")
        if not published:
            continue
        when = datetime.fromisoformat(published.replace("Z", "+00:00"))
        if when < cutoff:
            continue

        # release "name" is often empty -- fall back to the tag
        label = rel.get("name") or rel.get("tag_name") or "release"

        signals.append(RawSignal(
            title=f"{repo}: {label}",
            source="github",
            source_tier="primary",       # the project's own announcement
            summary=_summarise(rel.get("body")),
            url=rel.get("html_url", ""),
            published=published,
        ))

    return signals


def fetch_all(repos: list[str] | None = None, days: int = 30) -> list[RawSignal]:
    """Run every watched repo. One failure does not stop the others."""
    repos = repos or WATCHED_REPOS
    if not os.environ.get("GITHUB_TOKEN"):
        print("  (no GITHUB_TOKEN -- limited to 60 requests/hour)")

    out: list[RawSignal] = []
    for repo in repos:
        found = fetch_releases(repo, days=days)
        print(f"  {repo}: {len(found)} releases in the last {days} days")
        out.extend(found)

    out.sort(key=lambda s: s.published, reverse=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="Fetch GitHub releases as RawSignals")
    ap.add_argument("--days", type=int, default=30, help="how far back to look")
    ap.add_argument("--repo", action="append", help="override the repo list (repeatable)")
    args = ap.parse_args()

    print("fetching GitHub releases ...")
    signals = fetch_all(args.repo, days=args.days)

    print(f"\n{len(signals)} signals total\n")
    for s in signals:
        print(f"[{s.published[:10]}] {s.title}")
        print(f"   {s.summary[:110]}...")
        print(f"   {s.url}\n")


if __name__ == "__main__":
    main()
