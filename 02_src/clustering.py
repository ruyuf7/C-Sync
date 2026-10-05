"""
T5 -- Clustering
=================
Groups signals that report the SAME underlying event.

Two jobs, and the second is the one people miss:

  1. Deduplication. A GitHub release and the matching blog post are one event,
     not two. Without this the agent verifies twice and recommends twice.

  2. It is where verification confidence comes from. The VerificationAgent
     scores `independent_source_count >= 2` as 0.9 and a lone source as 0.75.
     That count only exists because clustering merged the signals.

Usage:
    # save signals once, then iterate without re-fetching
    python clustering.py --fetch --save 01_data/signals.json

    python clustering.py --signals 01_data/signals.json
    python clustering.py --signals 01_data/signals.json --threshold 0.55
    python clustering.py --signals 01_data/signals.json --strip-prefix
    python clustering.py --signals 01_data/signals.json --check
"""

import argparse
import difflib
import json
import re
from dataclasses import asdict
from pathlib import Path

from schemas import RawSignal, TrendCluster


# Default from the original prototype. It was a GUESS, never measured against
# real data -- exactly like RELEVANCE_FLOOR was before we calibrated it on
# real slides. Expect to change it.
# Raised from 0.4 after measuring on 63 real signals: at 0.4, unrelated blog
# posts merged on connective words alone and produced FALSE 0.9-confidence
# clusters. The identifier pass now does the real work; this is only a
# fallback for signals with no identifiers at all.
DEFAULT_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# NORMALISATION
# Every GitHub title starts with the same repo prefix:
#     "langchain-ai/langchain: langchain-core==1.6.2"
#     "langchain-ai/langchain: langchain-anthropic==1.7.1"
# That is 24 identical characters before the part that actually differs.
# difflib compares raw strings, so the shared prefix inflates similarity and
# unrelated releases look like duplicates.
# ---------------------------------------------------------------------------

VERSION = re.compile(r"\bv?\d+\.\d+(\.\d+)?([ab]|rc)?\d*\b")


def normalise(title: str, strip_prefix: bool = False) -> str:
    """Reduce a title to the part that carries meaning."""
    text = title
    if strip_prefix and ": " in text:
        text = text.split(": ", 1)[1]     # drop "owner/repo: "
    if strip_prefix:
        text = VERSION.sub("", text)      # version numbers are not meaning
    return text.lower().strip()


# ---------------------------------------------------------------------------
# CLUSTERING
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# IDENTIFIER EXTRACTION
#
# Measured on 63 real signals: title similarity produced FALSE multi-source
# clusters -- four unrelated blog posts merged because long titles from the
# same publisher share connective words ("with", "and", "for"). Those clusters
# then scored 0.9 confidence downstream, which is worse than missing a merge.
#
# Meanwhile "langchain.mcp" was the ONLY identifier shared across sources in
# all 63 signals. It found the GitHub release + blog post pair that string
# matching missed, and merged nothing else. Same lesson as the FAISS case in
# the curriculum RAG: rare technical tokens carry signal that fuzzy matching
# destroys.
#
# A dotted identifier must be captured as its FULL path, not just its leading
# segment. The GenAI spans expose attributes like "gen_ai.request.model" and
# "gen_ai.usage.input_tokens"; splitting those into a bare "gen_ai" turns a
# namespace ROOT into a matchable identifier, and a namespace is a topic, not
# an event. A genuine signal describing gen_ai.* attributes and a fabricated
# one merely name-dropping the gen_ai.* namespace then share "gen_ai" and
# merge -- a claim collapsed into its topic. Keeping the whole path
# ("gen_ai.request.model" != "gen_ai") requires the specific claim to match,
# so the leading segment of the first pattern now allows underscores and the
# dotted tail repeats to consume every segment.
# ---------------------------------------------------------------------------

IDENTIFIER = re.compile(
    r"\b[a-z][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+\b"  # gen_ai.request.model, langchain.mcp
    r"|\b[a-z]+[-_][a-z]+(?:[-_][a-z]+)*\b"       # langchain-anthropic, create_agent
    r"|\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b"           # AgentExecutor, LangGraph
    r"|\b[A-Z]{3,}\b"                             # FAISS, GRPO
    r"|@[a-zA-Z_]+"                               # @tool
)

# too common in this domain to identify anything
STOP_IDENTIFIERS = {
    "AI", "API", "LLM", "LLMS", "GPT", "SDK", "CLI", "URL", "JSON", "HTTP",
    "PDF", "CPU", "GPU", "OSS", "EHR", "NEW", "AND", "THE", "FOR", "WITH",
    "CHATGPT", "OPENAI",
}


def extract_identifiers(signal: RawSignal) -> set[str]:
    """Rare technical tokens from the title and summary."""
    text = f"{signal.title} {signal.summary}"
    found = set()
    for token in IDENTIFIER.findall(text):
        if token.upper() in STOP_IDENTIFIERS or len(token) < 4:
            continue
        found.add(token.lower())
    return found


# An identifier only identifies something if it is RARE. "openai-python"
# appears in every openai release and "langchain-ai" in every langchain one,
# so matching on them merges an entire repo into one cluster. Same idea as
# inverse document frequency: a token in most documents carries no signal.
# Measured on 63 real signals: at 0.15 the ceiling was 9, and "langchain-ai"
# appears in exactly 9 of them -- it squeaked through and merged every
# LangChain release into one cluster. 0.08 excludes repo-owner names while
# keeping genuinely rare tokens like "langchain.mcp" (2 signals).
MAX_DOC_FREQUENCY = 0.08


def identifier_frequencies(signals: list[RawSignal]) -> list[tuple[int, str]]:
    """How often each identifier appears. Use this to pick MAX_DOC_FREQUENCY."""
    counts: dict[str, int] = {}
    for s in signals:
        for token in extract_identifiers(s):
            counts[token] = counts.get(token, 0) + 1
    return sorted(((n, t) for t, n in counts.items()), reverse=True)


def rare_identifiers(signals: list[RawSignal],
                     max_freq: float = MAX_DOC_FREQUENCY) -> dict[int, set[str]]:
    """
    Identifiers per signal, with the over-common ones removed.
    Returns {index in signals: set of rare identifiers}.
    """
    per_signal = [extract_identifiers(s) for s in signals]

    counts: dict[str, int] = {}
    for idents in per_signal:
        for token in idents:
            counts[token] = counts.get(token, 0) + 1

    ceiling = max(2, int(len(signals) * max_freq))
    return {i: {t for t in idents if counts[t] <= ceiling}
            for i, idents in enumerate(per_signal)}


# ---------------------------------------------------------------------------
# CLUSTERING
# ---------------------------------------------------------------------------

def cluster_signals(signals: list[RawSignal],
                    threshold: float = DEFAULT_THRESHOLD,
                    strip_prefix: bool = False,
                    identifiers: bool = True,
                    min_shared: int = 1,
                    max_freq: float = MAX_DOC_FREQUENCY) -> list[TrendCluster]:
    """
    Two-pass clustering.

    Pass 1 -- shared rare identifiers. Two signals mentioning `langchain.mcp`
              are almost certainly the same event, even when their titles look
              nothing alike. This is what catches the cross-source pairs.

    Pass 2 -- title similarity, but ONLY for signals that pass 1 could not
              place. Use a high threshold here: at 0.4 unrelated blog posts
              merge on connective words alone.

    identifiers=False falls back to the old title-only behaviour, for
    comparison.
    """
    if not identifiers:
        return _cluster_by_title(signals, threshold, strip_prefix)

    clusters: list[TrendCluster] = []
    cluster_idents: list[set[str]] = []
    rare = rare_identifiers(signals, max_freq)

    for i, signal in enumerate(signals):
        sig_idents = rare[i]
        placed = False

        # PASS 1: does this share enough rare identifiers with a cluster?
        if sig_idents:
            for cluster, idents in zip(clusters, cluster_idents):
                if len(sig_idents & idents) >= min_shared:
                    cluster.signals.append(signal)
                    # do NOT expand the cluster's identifier set. If it grows
                    # as signals join, the cluster becomes a magnet that
                    # absorbs anything sharing any accumulated token.
                    placed = True
                    break

        # PASS 2: no identifier match -- fall back to title similarity.
        # NOT for GitHub releases: a release is its own event by definition,
        # and v3.7.0 vs v3.8.0 are ~95% identical as strings, so similarity
        # would collapse an entire release history into one cluster.
        if not placed and signal.source != "github":
            key = normalise(signal.title, strip_prefix)
            for cluster, idents in zip(clusters, cluster_idents):
                # skip clusters formed by identifier match; joining them on
                # loose string overlap is how false merges happen
                if idents:
                    continue
                rep = normalise(cluster.representative_title, strip_prefix)
                if difflib.SequenceMatcher(None, key, rep).ratio() >= threshold:
                    cluster.signals.append(signal)
                    placed = True
                    break

        if not placed:
            clusters.append(TrendCluster(representative_title=signal.title,
                                          signals=[signal]))
            cluster_idents.append(set(sig_idents))

    return clusters


def _cluster_by_title(signals: list[RawSignal], threshold: float,
                      strip_prefix: bool) -> list[TrendCluster]:
    """Original title-only clustering. Kept for before/after comparison."""
    clusters: list[TrendCluster] = []

    for signal in signals:
        key = normalise(signal.title, strip_prefix)
        placed = False

        for cluster in clusters:
            rep = normalise(cluster.representative_title, strip_prefix)
            if difflib.SequenceMatcher(None, key, rep).ratio() >= threshold:
                cluster.signals.append(signal)
                placed = True
                break

        if not placed:
            clusters.append(TrendCluster(representative_title=signal.title,
                                          signals=[signal]))

    return clusters


# ---------------------------------------------------------------------------
# INSPECTION -- look at the clusters before touching the threshold
# ---------------------------------------------------------------------------

_RARE: dict[int, set[str]] = {}


def report(clusters: list[TrendCluster], show_all: bool = False) -> None:
    total = sum(len(c.signals) for c in clusters)
    merged = [c for c in clusters if len(c.signals) > 1]

    print(f"\n{total} signals -> {len(clusters)} clusters "
          f"({len(merged)} contain more than one signal)\n")

    if not merged:
        print("  nothing merged at all -- threshold is too high\n")

    for c in sorted(merged, key=lambda c: -len(c.signals)):
        multi = len(c.source_tiers) > 1 or c.independent_source_count > 1
        flag = "  [multi-source -> confidence 0.9]" if c.independent_source_count > 1 else ""
        shared = set.intersection(*[_RARE.get(id(s), set()) for s in c.signals]) if len(c.signals) > 1 else set()
        why = f"  via {sorted(shared)[:3]}" if shared else "  via title similarity"
        print(f"  ({len(c.signals)} signals, {c.independent_source_count} independent sources){flag}{why}")
        for s in c.signals:
            print(f"      [{s.source}] {s.title[:78]}")
        print()

    if show_all:
        singles = [c for c in clusters if len(c.signals) == 1]
        print(f"  --- {len(singles)} single-signal clusters ---")
        for c in singles:
            print(f"      [{c.signals[0].source}] {c.representative_title[:78]}")


def check(clusters: list[TrendCluster]) -> None:
    """
    The specific cases from the task brief. These are checked by eye normally;
    this just surfaces them so you do not have to scroll.
    """
    def cluster_of(fragment: str):
        frag = fragment.lower()
        for c in clusters:
            for s in c.signals:
                if frag in s.title.lower():
                    return c
        return None

    print("\n=== CHECKS ===\n")

    # 1. the headline test: does the cross-source MCP pair merge?
    gh = cluster_of("langchain==1.4.0")
    rss = cluster_of("MCP in LangChain")
    if gh and rss:
        same = gh is rss
        print(f"  MCP cross-source pair merged : {'YES' if same else 'NO'}"
              f"{'' if same else '   <-- the pair we most need to catch'}")
    else:
        print("  MCP pair               : one or both signals not in this data")

    # 2. different LangChain packages must NOT merge
    pkgs = ["langchain-core", "langchain-anthropic", "langchain-fireworks"]
    found = {p: cluster_of(p) for p in pkgs}
    present = [p for p, c in found.items() if c]
    if len(present) > 1:
        ids = {id(found[p]) for p in present}
        ok = len(ids) == len(present)
        print(f"  LangChain packages separate  : {'YES' if ok else 'NO'}"
              f"{'' if ok else '   <-- over-merging on the shared repo prefix'}")

    # 3. sequential OpenAI versions must not collapse into one
    openai_clusters = {id(c) for c in clusters
                       for s in c.signals if "openai-python" in s.title.lower()}
    n_openai = sum(1 for c in clusters for s in c.signals if "openai-python" in s.title.lower())
    if n_openai:
        print(f"  OpenAI releases              : {n_openai} signals in {len(openai_clusters)} clusters"
              f"{'   <-- all merged into one' if len(openai_clusters) == 1 and n_openai > 2 else ''}")

    # 4. sanity on the overall count
    total = sum(len(c.signals) for c in clusters)
    n = len(clusters)
    if n == total:
        verdict = "nothing merged -- threshold too high"
    elif n < total * 0.4:
        verdict = "heavy merging -- check for false merges"
    else:
        verdict = "plausible"
    print(f"  Cluster count                : {n} from {total} signals ({verdict})")
    print()


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_signals(path: str) -> list[RawSignal]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [RawSignal(**d) for d in data]


def fetch_signals(days: int = 30, secondary: bool = False) -> list[RawSignal]:
    """
    Live fetch. Prefer working from a saved file while tuning.

    days -- how far back to look. 30 is a sensible default cadence, but a
    wider window surfaces the rarer events a month rarely contains, such as a
    genuine breaking change.
    """
    from monitoring_github import fetch_all as fetch_github
    from monitoring_rss import fetch_all as fetch_rss
    signals = []
    try:
        signals += fetch_github(days=days)
    except Exception as e:
        print(f"  ! github fetch failed: {e}")
    try:
        signals += fetch_rss(days=days, include_secondary=secondary)
    except Exception as e:
        print(f"  ! rss fetch failed: {e}")
    return signals


def main():
    ap = argparse.ArgumentParser(description="Cluster monitoring signals")
    ap.add_argument("--signals", help="path to a saved signals JSON file")
    ap.add_argument("--fetch", action="store_true", help="fetch live instead of loading")
    ap.add_argument("--days", type=int, default=30,
                    help="how far back to fetch. 90 surfaces rarer events that a "
                         "30-day window usually misses")
    ap.add_argument("--secondary", action="store_true",
                    help="include secondary-tier community feeds (noisier, but gives "
                         "verification something to actually be sceptical about)")
    ap.add_argument("--save", help="write fetched signals to this path")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--strip-prefix", action="store_true",
                    help="drop the 'owner/repo:' prefix and version numbers before comparing")
    ap.add_argument("--no-identifiers", action="store_true",
                    help="disable the identifier pass (old title-only behaviour)")
    ap.add_argument("--max-freq", type=float, default=MAX_DOC_FREQUENCY,
                    help="ignore identifiers appearing in more than this fraction of signals")
    ap.add_argument("--frequencies", action="store_true",
                    help="list identifier frequencies and exit (use to pick --max-freq)")
    ap.add_argument("--min-shared", type=int, default=1,
                    help="identifiers two signals must share to merge")
    ap.add_argument("--check", action="store_true", help="run the task-brief checks")
    ap.add_argument("--all", action="store_true", help="also list single-signal clusters")
    args = ap.parse_args()

    if args.fetch or not args.signals:
        signals = fetch_signals(days=args.days, secondary=args.secondary)
        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save).write_text(
                json.dumps([asdict(s) for s in signals], indent=2), encoding="utf-8")
            print(f"saved {len(signals)} signals -> {args.save}")
    else:
        signals = load_signals(args.signals)
        print(f"loaded {len(signals)} signals from {args.signals}")

    if not signals:
        print("no signals to cluster")
        return

    if args.frequencies:
        total = len(signals)
        print(f"\nidentifier frequencies across {total} signals")
        print(f"current ceiling: {max(2, int(total * args.max_freq))} "
              f"(--max-freq {args.max_freq})\n")
        for n, token in identifier_frequencies(signals)[:30]:
            mark = "  <- EXCLUDED" if n > max(2, int(total * args.max_freq)) else ""
            print(f"  {n:3}  ({n/total:5.1%})  {token}{mark}")
        return

    global _RARE
    _RARE = {id(s): idents for s, idents in
             zip(signals, rare_identifiers(signals, args.max_freq).values())}

    clusters = cluster_signals(signals, args.threshold, args.strip_prefix,
                                identifiers=not args.no_identifiers,
                                min_shared=args.min_shared,
                                max_freq=args.max_freq)
    print(f"threshold={args.threshold}  identifiers={not args.no_identifiers}"
          f"  min_shared={args.min_shared}  max_freq={args.max_freq}")
    report(clusters, show_all=args.all)
    if args.check:
        check(clusters)


if __name__ == "__main__":
    main()
