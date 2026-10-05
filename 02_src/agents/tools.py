"""
Tools the agents can call.
===========================
This is the boundary between "the model reasons" and "the model acts".

An agent does not fetch anything itself. It emits a tool call -- a small JSON
object naming a tool and its arguments -- and this module executes it and
hands back the result. The model then decides what to do next: answer, or
call another tool.

That loop is what makes the system agentic. We do not write the order of
calls; the model chooses at runtime based on what it has found so far.

Tools, matching the questions the pipeline asks:

    github_lookup     -> "does this project exist?"    (VerificationAgent)
    verify_release    -> "did it ship what was claimed?" (VerificationAgent)
    search_curriculum -> "do we teach this?"           (CurriculumAgent)

Every tool here:
  * returns a JSON-serialisable dict, never raises into the agent loop
  * reports failure as data ({"error": ...}) so the model can react to it
  * is safe to call repeatedly -- the agent WILL call them more than once
  * caches successful GitHub responses on disk, so re-runs of the same
    clusters do not re-spend the rate limit (see RESPONSE CACHE below)
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

# scripts run from the repo root, so 02_src is not automatically importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import curriculum_ingest


DEFAULT_DB = "./vectorstore"
GITHUB_API = "https://api.github.com"
TIMEOUT = 15


# ---------------------------------------------------------------------------
# RESPONSE CACHE
#
# The verification loop now makes up to TWO GitHub calls per signal
# (github_lookup + verify_release), so the demo's rate-limit exposure doubled.
# A small on-disk cache fixes that: successful responses are stored keyed by
# (tool, args), so re-runs of the same clusters hit disk instead of GitHub.
#
#   TOOL_CACHE_DIR   where to store entries (default 01_data/.tool_cache)
#   TOOL_CACHE_TTL   seconds before an entry is considered stale (default: never)
#   TOOL_CACHE_ONLY  =1 -> never touch the network; serve only from cache, and
#                    report a miss as data. This is how you run the demo fully
#                    offline once the cache is warm.
#
# Only successful results are cached -- transient errors (rate limits, network
# blips) must never be frozen into the cache.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _cache_dir() -> Path:
    return Path(os.environ.get("TOOL_CACHE_DIR", _REPO_ROOT / "01_data" / ".tool_cache"))


def _cache_only() -> bool:
    return os.environ.get("TOOL_CACHE_ONLY", "").lower() in ("1", "true", "yes")


def _cache_key(name: str, key_args: dict) -> str:
    # max_results does not change WHICH repo/release matches, so leave it out of
    # the key: an agentic call asking for 5 and a scripted one asking for 3
    # share the same cache entry.
    payload = name + ":" + json.dumps(key_args, sort_keys=True).lower()
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cache_get(key: str) -> dict | None:
    path = _cache_dir() / f"{key}.json"
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ttl = os.environ.get("TOOL_CACHE_TTL")
    if ttl and (time.time() - entry.get("cached_at", 0)) > float(ttl):
        return None
    value = entry.get("value")
    if isinstance(value, dict):
        return {**value, "_cache": "hit"}
    return value


def _cache_put(key: str, value: dict) -> None:
    d = _cache_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{key}.json").write_text(
            json.dumps({"cached_at": time.time(), "value": value}),
            encoding="utf-8")
    except OSError:
        pass  # a broken cache must never break verification


def _with_cache(name: str, key_args: dict, compute) -> dict:
    """Serve `name(key_args)` from cache, computing (and caching) on a miss."""
    key = _cache_key(name, key_args)
    hit = _cache_get(key)
    if hit is not None:
        return hit
    if _cache_only():
        return {**key_args, "error": "cache-only mode (TOOL_CACHE_ONLY): "
                "no cached response for this call", "_cache": "miss"}
    result = compute()
    if isinstance(result, dict) and "error" not in result:
        _cache_put(key, result)
    return result


# ---------------------------------------------------------------------------
# TOOL 1 -- github_lookup
#
# The verification question is usually "does this thing actually exist, and
# did the project really say this?" GitHub answers both: a repo that does not
# exist is a strong signal a claim is fabricated, and release notes are the
# project's own words.
# ---------------------------------------------------------------------------

def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def github_lookup(query: str, max_results: int = 3) -> dict:
    """
    Search GitHub repositories by name or keyword.

    Returns repo name, description, star count and last push date. Stars and
    recency matter for verification: a project with 40k stars pushed last week
    is established; one with 3 stars and no commits in two years is not
    evidence of a real trend, whatever the announcement claimed.

    Successful responses are cached (see the RESPONSE CACHE section).
    """
    return _with_cache("github_lookup", {"query": query},
                       lambda: _github_lookup_uncached(query, max_results))


def _github_lookup_uncached(query: str, max_results: int = 3) -> dict:
    try:
        r = requests.get(f"{GITHUB_API}/search/repositories",
                         headers=_headers(),
                         params={"q": query, "sort": "stars", "per_page": max_results},
                         timeout=TIMEOUT)
    except requests.RequestException as e:
        return {"error": f"network error: {e}", "query": query}

    if r.status_code == 403 and "rate limit" in r.text.lower():
        reset = r.headers.get("X-RateLimit-Reset")
        mins = ""
        if reset:
            mins = f", resets in ~{max(0, int(reset) - int(time.time())) // 60} min"
        return {"error": f"github rate limit reached{mins}. Set GITHUB_TOKEN in .env.",
                "query": query}

    if not r.ok:
        return {"error": f"github returned HTTP {r.status_code}", "query": query}

    items = r.json().get("items", [])
    if not items:
        # a real finding, not a failure -- "no such project" is evidence
        return {"query": query, "found": 0, "results": [],
                "note": "No matching repository. Treat claims about this as unverified."}

    return {
        "query": query,
        "found": len(items),
        "results": [{
            "full_name": it["full_name"],
            "description": (it.get("description") or "")[:200],
            "stars": it.get("stargazers_count", 0),
            "last_push": it.get("pushed_at", ""),
            "url": it.get("html_url", ""),
        } for it in items],
    }


# ---------------------------------------------------------------------------
# TOOL 2 -- verify_release
#
# github_lookup answers "does this project exist?". It cannot answer "did it
# actually ship the thing the signal claims?" -- a 90k-star repo proves
# popularity, never that v1.0.0 was released or that a feature landed. This
# tool checks the project's OWN release record: GET /repos/{owner}/{repo}/
# releases (and the by-tag endpoint for a specific version). A confirmed
# release is the difference between "the repo is real" and "the claim is real".
# ---------------------------------------------------------------------------

def _candidate_tags(version: str) -> list[str]:
    """Tag spellings to try for a claimed version, e.g. v1.0.0 <-> 1.0.0."""
    v = version.strip()
    bare = v[1:] if v[:1].lower() == "v" else v
    out = []
    for cand in (v, bare, f"v{bare}"):
        if cand and cand not in out:
            out.append(cand)
    return out


def _rate_limited(r) -> dict | None:
    if r.status_code == 403 and "rate limit" in r.text.lower():
        reset = r.headers.get("X-RateLimit-Reset")
        mins = ""
        if reset:
            mins = f", resets in ~{max(0, int(reset) - int(time.time())) // 60} min"
        return {"error": f"github rate limit reached{mins}. Set GITHUB_TOKEN in .env."}
    return None


def _release_author(release: dict) -> str:
    """Only expose the account supplied by the release API; never infer it."""
    author = release.get("author")
    login = author.get("login") if isinstance(author, dict) else None
    return login if isinstance(login, str) else ""


def verify_release(repo: str, version: str = "", max_results: int = 8) -> dict:
    """
    Confirm a project's release record via the GitHub Releases API.

    With a ``version`` ("v1.0.0", "1.0.0"), confirms that SPECIFIC release
    exists -- this is what actually verifies a version claim. Without one,
    lists recent releases so the caller can see whether the project ships at
    all. ``release_found`` is True only when the claimed version was matched
    (or, with no version asked, when any release exists).

    Successful responses are cached (see the RESPONSE CACHE section).
    """
    # normalise the key so 'v1.0.0' and '1.0.0' (and REPO casing) share one
    # entry -- the model and the deterministic path spell versions differently.
    key_args = {"repo": repo.lower(),
                "version": version[1:] if version[:1].lower() == "v" else version}
    result = _with_cache("verify_release", key_args,
                         lambda: _verify_release_uncached(repo, version, max_results))
    # Old disk entries remain usable offline. Missing author means unknown,
    # not permission to infer a publisher or fetch while TOOL_CACHE_ONLY is set.
    if isinstance(result, dict):
        matched = result.get("matched_release")
        if isinstance(matched, dict):
            result = {**result, "matched_release": {"author": "", **matched}}
        if isinstance(result.get("releases"), list):
            result = {**result, "releases": [
                {"author": "", **release} if isinstance(release, dict) else release
                for release in result["releases"]
            ]}
    return result


def _verify_release_uncached(repo: str, version: str = "", max_results: int = 8) -> dict:
    if "/" not in repo:
        return {"error": "repo must be 'owner/name'", "repo": repo}

    if version:
        for tag in _candidate_tags(version):
            try:
                r = requests.get(f"{GITHUB_API}/repos/{repo}/releases/tags/{tag}",
                                 headers=_headers(), timeout=TIMEOUT)
            except requests.RequestException as e:
                return {"error": f"network error: {e}", "repo": repo, "version": version}
            limited = _rate_limited(r)
            if limited:
                return {**limited, "repo": repo, "version": version}
            if r.status_code == 200:
                it = r.json()
                return {"repo": repo, "version": version, "release_found": True,
                        "matched_release": {
                            "tag": it.get("tag_name", ""),
                            "author": _release_author(it),
                            "name": (it.get("name") or "")[:120],
                            "published_at": it.get("published_at", ""),
                            "url": it.get("html_url", ""),
                            "prerelease": it.get("prerelease", False),
                        }}
            # 404 -> that spelling is not a release; try the next candidate

    # no version asked, or the claimed version was not a tagged release: list
    # recent releases for context (and, for a version query, as evidence the
    # specific claim is NOT among them).
    try:
        r = requests.get(f"{GITHUB_API}/repos/{repo}/releases",
                         headers=_headers(), params={"per_page": max_results},
                         timeout=TIMEOUT)
    except requests.RequestException as e:
        return {"error": f"network error: {e}", "repo": repo, "version": version}
    limited = _rate_limited(r)
    if limited:
        return {**limited, "repo": repo, "version": version}
    if r.status_code == 404:
        return {"repo": repo, "version": version, "release_found": False,
                "matched_release": None, "releases": [],
                "note": "repository has no releases (or does not exist)"}
    if not r.ok:
        return {"error": f"github returned HTTP {r.status_code}", "repo": repo}

    releases = [{
        "tag": it.get("tag_name", ""),
        "author": _release_author(it),
        "name": (it.get("name") or "")[:120],
        "published_at": it.get("published_at", ""),
        "url": it.get("html_url", ""),
        "prerelease": it.get("prerelease", False),
    } for it in r.json()[:max_results]]

    note = ("claimed version not found among the latest releases"
            if version else "latest releases listed; no specific version was claimed")
    return {"repo": repo, "version": version,
            "release_found": (False if version else bool(releases)),
            "matched_release": None, "releases": releases[:5], "note": note}


# ---------------------------------------------------------------------------
# TOOL 3 -- search_curriculum
#
# Thin wrapper over the RAG store built in curriculum_ingest.py. The agent
# passes a natural-language question; hybrid search handles both meaning and
# exact identifiers.
# ---------------------------------------------------------------------------

def search_curriculum(question: str, week: int | None = None,
                      max_results: int = 3,
                      db_path: str = DEFAULT_DB) -> dict:
    """
    Search the ingested course slides and lab notebooks.

    Each hit carries a citation ("Week 2 / RAG Introduction / slide 34") and
    an is_reliable flag. The flag exists because similarity alone is not
    trustworthy here: measured on our own decks, a slide literally containing
    "FAISS" scored 0.303 while an unrelated slide scored 0.308. An exact
    identifier match is reliable regardless of the number.
    """
    try:
        hits = curriculum_ingest.query(db_path, question, k=max_results,
                                        week=week)
    except Exception as e:
        return {"error": f"curriculum search failed: {e}", "question": question}

    if not hits:
        return {"question": question, "found": 0, "results": [],
                "note": "Nothing in the curriculum matches. The trend may be "
                        "real but outside what we teach."}

    results = []
    for h in hits:
        exact = h.get("exact_match")
        sim = h.get("similarity")
        results.append({
            "citation": h["citation"],
            "week": h.get("week"),
            # topic / source_file / slide_number are required to rebuild a
            # CurriculumMatch downstream. Without them the citation renders as
            # "Week 3 /  / slide 0". content_type carries the lab-vs-slides
            # distinction, which drives is_lab and the recommendation tier.
            "topic": h.get("topic", ""),
            "source_file": h.get("source_file", ""),
            "slide_number": h.get("slide_number", 0),
            "content_type": h.get("content_type", "slides"),
            "text": h["text"][:400],
            "similarity": sim,
            "exact_match": exact,
            "is_reliable": exact is not None or (sim or 0) >= RELEVANCE_FLOOR,
        })

    return {"question": question, "found": len(results), "results": results}


# keep in sync with schemas.RELEVANCE_FLOOR -- imported lazily to avoid a
# circular import if schemas ever needs anything from here
try:
    from schemas import RELEVANCE_FLOOR
except ImportError:
    RELEVANCE_FLOOR = 0.48


# ---------------------------------------------------------------------------
# TOOL SCHEMAS
#
# This is what the model actually sees. The `description` fields are not
# documentation -- they are the prompt. The model decides whether to call a
# tool based entirely on this text, so it should say WHEN to use the tool,
# not just what it does.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "github_lookup",
            "description": (
                "Search GitHub repositories to check whether a project or tool "
                "actually exists and how established it is. Use this when a claim "
                "mentions a specific library, framework or repository and you need "
                "to confirm it is real before trusting the claim. Returns star "
                "counts and last push date, which indicate whether a project is "
                "widely adopted or abandoned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Repository name or keywords, e.g. 'langchain' "
                                       "or 'RapidAgent framework'.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "How many repositories to return. Default 3.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_release",
            "description": (
                "Confirm that a repository actually shipped a SPECIFIC claimed "
                "release, using its GitHub release record. Use this AFTER "
                "github_lookup confirms the repo exists, and ONLY when a signal "
                "names a specific version (e.g. 'v1.0.0'): a repo existing does "
                "not prove it shipped what was claimed. If no version is claimed, "
                "do not call this -- there is nothing to confirm."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "The repository as 'owner/name', e.g. "
                                       "'langchain-ai/langgraph'.",
                    },
                    "version": {
                        "type": "string",
                        "description": "The specific claimed version/tag to confirm, "
                                       "e.g. 'v1.0.0' or '1.0.0'. Required.",
                    },
                },
                "required": ["repo", "version"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_curriculum",
            "description": (
                "Search our course slides and lab notebooks to find out whether we "
                "already teach something. Use this to decide if a technology trend "
                "affects our curriculum. Returns specific slides or notebook cells "
                "with citations. Prefer specific technical terms in the question "
                "(library names, class names, API methods) over general phrasing, "
                "because exact identifier matches are more reliable than topical "
                "similarity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What to look for, e.g. 'AgentExecutor' or "
                                       "'how do we teach chunking'.",
                    },
                    "week": {
                        "type": "integer",
                        "description": "Restrict to one week's material. Omit to search all.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return. Default 3.",
                    },
                },
                "required": ["question"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# DISPATCH
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "github_lookup": github_lookup,
    "verify_release": verify_release,
    "search_curriculum": search_curriculum,
}


def call_tool(name: str, arguments: dict | str) -> dict:
    """
    Execute a tool call from the model and return the result.

    Never raises. A tool that blows up returns {"error": ...} so the model can
    read the failure and decide what to do -- try a different query, or give
    up and report low confidence. An exception here would kill the agent loop
    instead, which loses that reasoning.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError as e:
            return {"error": f"could not parse arguments: {e}"}

    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"unknown tool '{name}'. "
                         f"Available: {', '.join(TOOL_FUNCTIONS)}"}

    try:
        return func(**arguments)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"{name} failed: {type(e).__name__}: {e}"}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Try a tool by hand")
    ap.add_argument("tool", choices=list(TOOL_FUNCTIONS))
    ap.add_argument("query")
    ap.add_argument("--week", type=int)
    args = ap.parse_args()

    if args.tool == "github_lookup":
        out = call_tool("github_lookup", {"query": args.query})
    else:
        out = call_tool("search_curriculum", {
            "question": args.query, "week": args.week})

    print(json.dumps(out, indent=2, ensure_ascii=False))