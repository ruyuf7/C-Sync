"""
VerificationAgent -- is this trend REAL?
========================================
First agent in the chain. Consumes a ``TrendCluster`` and returns a
``VerifiedTrend``: a confidence score, a note, an evidence trail, and the
reasoning loop that produced it.

Division of labour -- AGENTIC DECISIONS, DETERMINISTIC SCORE:

  * The LLM drives the loop. It reads the cluster, decides which tool to call
    (``github_lookup`` to check a repo exists, ``verify_release`` to confirm a
    claimed version actually shipped), judges whether the observation settles
    the question, and decides when to stop. We do not script the tool order --
    the model chooses at runtime from what it has seen. That is the T6
    requirement, and it is a real tool-calling loop, not a pipeline.

  * Python computes the score. The model never outputs a number. Confidence is
    a pure function of FACTS WE ACTUALLY CHECKED -- did a matching repository
    exist, was the specific version confirmed, how many independent sources
    were corroborated -- with hard bands enforced in code. So the score cannot
    be inflated by a persuasive model, and it does not drift with the model.

Three states are tracked separately and never conflated:
    repo_exists     -- a matching repository was found (github_lookup)
    claim_verified  -- the SPECIFIC claim was confirmed (verify_release). A
                       popular repo is not this: 90k stars proves adoption,
                       never that v1.0.0 shipped.
    verified sources-- independent sources actually corroborated. A link no
                       tool can open is recorded, unverified, and does not count.

Confidence bands, enforced deterministically after the loop:
    verified independent sources < 2  ->  confidence <= 0.75
    named repository not found         ->  confidence <= 0.30

DEMO MODE: the score is identical whether the loop was driven by the model or by
the deterministic fallback, because both call the same tools and the same
scorer. Each run prints which mode gathered the evidence.

Usage:
    from agents.verification import VerificationAgent
    trend = VerificationAgent().run(cluster)

    # offline / testing: inject a fake OpenAI client and/or a fake tool runner
    VerificationAgent(client=fake, tool_runner=fake_dispatch).run(cluster)

    python agents/verification.py --signals 01_data/signals.json --show-reasoning
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from schemas import Evidence, ReasoningStep, TrendCluster, VerifiedTrend
from agents import tools


DEFAULT_MODEL = "gpt-4o-mini"
MAX_TOOL_ROUNDS = 6        # safety cap; the model decides when to stop before this

# Hard confidence ceilings, enforced in code regardless of how evidence was got.
SINGLE_SOURCE_CEILING = 0.75   # fewer than 2 verified independent sources
MISSING_REPO_CEILING = 0.30    # a named repository could not be found

# The agent gets both verification tools; curriculum search is a different job.
_VERIFY_TOOLS = [s for s in tools.TOOL_SCHEMAS
                 if s["function"]["name"] in ("github_lookup", "verify_release")]

MODE_AGENTIC = "agentic (LLM-driven tool loop)"
MODE_DETERMINISTIC = "deterministic (no LLM; scripted tool loop)"


SYSTEM_PROMPT = """\
You are a verification agent for an AI-curriculum trend monitor. You are given a
cluster of monitoring signals that all appear to describe ONE development. Your
job is to CHECK whether it is real by calling tools -- you drive the
investigation.

Tools:
- github_lookup(query): does a named repository exist and how established is it?
- verify_release(repo, version): did that repo actually ship a claimed version?

How to work:
- Call ONE tool at a time. Read its result, then decide the next step from what
  you actually observed -- do not plan several calls in advance.
- A repository EXISTING is not the same as its CLAIM being true. When a signal
  names a specific version, confirm it with verify_release -- do not treat stars
  or existence as proof the release happened.
- Call verify_release ONLY when a specific version is claimed, and pass that
  version. If no version is named, do not call it -- there is nothing to confirm.
- If an observation leaves you unsure, call another tool. When you have checked
  what can be checked, STOP by replying with no tool call.
- Do NOT output a confidence score or a verdict. The system computes the score
  from what your checks actually found. Before each tool call, briefly say -- in
  that message -- what the previous observation told you and why you are calling
  this tool now.
"""


class VerificationAgent:
    """Decide whether a trend's claims are supported by sufficient evidence."""

    def __init__(self, client=None, model: str = DEFAULT_MODEL,
                 tool_runner=None, max_tool_rounds: int = MAX_TOOL_ROUNDS):
        self._client = client
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        # injectable dispatch so the loop is testable without the network
        self._run_tool = tool_runner or tools.call_tool

    # -- public API --------------------------------------------------------

    def run(self, cluster: TrendCluster) -> VerifiedTrend:
        """Verify one trend cluster and return its verification result."""
        client = self._get_client()
        if client is None:
            facts, mode = self._deterministic_loop(cluster), MODE_DETERMINISTIC
        else:
            try:
                facts, mode = self._agentic_loop(client, cluster), MODE_AGENTIC
            except Exception as e:
                facts = self._deterministic_loop(cluster)
                mode = MODE_DETERMINISTIC + f" [LLM loop failed: {type(e).__name__}]"

        confidence = _score(cluster, facts)
        note = _build_note(cluster, facts, confidence)

        return VerifiedTrend(
            cluster=cluster,
            confidence=confidence,
            verification_note=note,
            evidence=facts.evidence,
            verified_source_count=facts.verified_source_count,
            repo_exists=facts.repo_exists,
            claim_verified=facts.claim_verified,
            reasoning=facts.reasoning,
            mode=mode,
        )

    # -- AGENTIC loop: the model decides which tools to call and when ------

    def _agentic_loop(self, client, cluster: TrendCluster) -> "Facts":
        acc = _Accumulator(cluster)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _describe(cluster)},
        ]

        for _ in range(self.max_tool_rounds):
            resp = client.chat.completions.create(
                model=self.model, messages=messages,
                tools=_VERIFY_TOOLS, tool_choice="auto", temperature=0,
                # one tool per round, so each round's reasoning is written AFTER
                # seeing the previous observation -- the loop is observation-
                # driven, not a pre-planned batch with one reused thought.
                parallel_tool_calls=False)
            msg = resp.choices[0].message
            thought = (msg.content or "").strip()

            if not msg.tool_calls:
                if thought:
                    acc.note_thought(thought, "model concluded it has enough evidence")
                break

            messages.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name,
                                 "arguments": tc.function.arguments},
                } for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                args = _parse_args(tc.function.arguments)
                # nothing to confirm without a version; don't spend a call, and
                # tell the model so its next thought reflects the correction
                if (tc.function.name == "verify_release"
                        and not str(args.get("version", "")).strip()):
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": json.dumps({"error":
                                         "verify_release needs a specific claimed "
                                         "version; if none is claimed, do not call it"})})
                    continue
                result = self._run_tool(tc.function.name, args)
                acc.record(tc.function.name, args, result, thought)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result)})

        return acc.facts()

    # -- DETERMINISTIC loop: same tools, scripted, for the no-key fallback -

    def _deterministic_loop(self, cluster: TrendCluster) -> "Facts":
        acc = _Accumulator(cluster)
        # GitHub signals carry the authoritative 'owner/repo' and the version
        # claim, so check them first; a non-github sibling naming the same
        # project is then already confirmed and need not be looked up again.
        order = sorted(enumerate(cluster.signals),
                       key=lambda t: (t[1].source != "github", t[0]))

        for i, signal in order:
            repo = _claim_query(signal)
            if not repo:
                acc.note_thought(
                    f"Signal {i + 1} ({signal.source}) names no verifiable "
                    f"repository.", "recorded as an unchecked source")
                continue

            if _already_confirmed(repo, acc.confirmed_repos):
                acc.note_thought(
                    f"Signal {i + 1} names '{repo}', already confirmed by another "
                    f"signal in this cluster.", "not re-checking")
                full = repo
            else:
                result = self._run_tool("github_lookup", {"query": repo})
                acc.record("github_lookup", {"query": repo}, result,
                           f"Signal {i + 1} names '{repo}'; confirm it exists.")
                match = _match_result(repo, result)
                full = match["full_name"] if match else None

            version = _claim_version(signal)
            if signal.source == "github" and version and full:
                acc.record("verify_release", {"repo": full, "version": version},
                           self._run_tool("verify_release",
                                          {"repo": full, "version": version}),
                           f"Signal {i + 1} claims {version}; confirm the release shipped.")
        return acc.facts()

    # -- client acquisition ------------------------------------------------

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None
        self._client = OpenAI()
        return self._client


# ---------------------------------------------------------------------------
# FACT ACCUMULATION
#
# Both loops feed observations here. Facts are derived from the RAW tool
# results (via _match_result / matched_release), never from the model's claims
# about them -- the model chooses what to look up, the data decides what is true.
# ---------------------------------------------------------------------------

@dataclass
class Facts:
    evidence: list[Evidence]
    reasoning: list[ReasoningStep]
    verified_source_count: int
    repo_exists: bool
    repo_missing: bool
    claim_verified: bool


class _Accumulator:
    def __init__(self, cluster: TrendCluster):
        self.cluster = cluster
        self.source_ev = [
            Evidence(source=s.source, tier=s.source_tier, url=s.url,
                     note=s.title, kind="source", verified=False)
            for s in cluster.signals
        ]
        self.tool_ev: list[Evidence] = []
        self.reasoning: list[ReasoningStep] = []
        self.confirmed_repos: set[str] = set()      # full_names found to exist
        self.confirmed_versions: set[str] = set()   # "owner/repo@version" confirmed
        self.looked_up = False                       # any github_lookup attempted?
        self._it = 0

    def note_thought(self, thought: str, observation: str) -> None:
        self._it += 1
        self.reasoning.append(ReasoningStep(
            iteration=self._it, thought=thought, observation=observation))

    def record(self, tool: str, args: dict, result: dict, thought: str) -> None:
        self._it += 1
        if tool == "github_lookup":
            query = str(args.get("query", ""))
            match = _match_result(query, result)
            observation = _describe_lookup(query, result, match)
            self.looked_up = True
            if match:
                self.confirmed_repos.add(match["full_name"].lower())
            url = (match or {}).get("url", "")
        elif tool == "verify_release":
            repo = str(args.get("repo", ""))
            version = str(args.get("version", ""))
            matched = result.get("matched_release") if isinstance(result, dict) else None
            observation = _describe_release(repo, version, result, matched)
            if matched and version:
                self.confirmed_versions.add(f"{repo.lower()}@{_norm_version(version)}")
            url = (matched or {}).get("url", "")
        else:
            observation = f"{tool}({args}) -> {str(result)[:120]}"
            url = ""

        self.reasoning.append(ReasoningStep(
            iteration=self._it, thought=thought or f"call {tool}",
            tool=tool, tool_args=args, observation=observation))
        self.tool_ev.append(Evidence(source="github", tier="tool", kind="tool",
                                     url=url, note=observation, verified=False))

    def facts(self) -> Facts:
        # a source is verified only if the repository IT names was found; a
        # discussion link is not verified because some other repo it mentions
        # exists, and a tool result is never a source.
        for i, s in enumerate(self.cluster.signals):
            repo = _claim_query(s)
            if s.source == "github" and repo and repo in self.confirmed_repos:
                self.source_ev[i].verified = True
                self.source_ev[i].note += "  (repository existence confirmed)"

        repo_exists = bool(self.confirmed_repos)
        return Facts(
            evidence=self.source_ev + self.tool_ev,
            reasoning=self.reasoning,
            verified_source_count=len({e.source for e in self.source_ev if e.verified}),
            repo_exists=repo_exists,
            repo_missing=self.looked_up and not repo_exists,
            claim_verified=bool(self.confirmed_versions),
        )


# ---------------------------------------------------------------------------
# SCORING -- pure functions, no API key, straightforward to test
# ---------------------------------------------------------------------------

def _score(cluster: TrendCluster, facts: Facts) -> float:
    """Deterministic confidence from what was actually verified, then bands."""
    return _enforce_bands(_base_confidence(cluster, facts), facts)


def _base_confidence(cluster: TrendCluster, facts: Facts) -> float:
    n = facts.verified_source_count
    has_primary = "primary" in cluster.source_tiers

    if facts.repo_missing:
        return 0.15                     # named project not found -> likely fabricated
    if n >= 2 and facts.claim_verified:
        return 0.95                     # corroborated AND the claim itself confirmed
    if n >= 2:
        return 0.80                     # multiple checked sources, claim unconfirmed
    if n == 1 and facts.claim_verified:
        return 0.75                     # single source, but the claim IS confirmed
    if n == 1:
        return 0.60                     # repo exists, claim not confirmed
    if facts.repo_exists or has_primary:
        return 0.50                     # something real but nothing we could check
    return 0.40                         # a single unchecked secondary source


def _enforce_bands(confidence: float, facts: Facts) -> float:
    """Hard ceilings. Advisory prompt text does not bind; THIS does."""
    if facts.repo_missing:
        confidence = min(confidence, MISSING_REPO_CEILING)
    if facts.verified_source_count < 2:
        confidence = min(confidence, SINGLE_SOURCE_CEILING)
    return round(max(0.0, min(1.0, confidence)), 2)


def _build_note(cluster: TrendCluster, facts: Facts, confidence: float) -> str:
    """Built FROM the facts, so it can never describe a different score."""
    v = facts.verified_source_count
    parts = [f"{v} of {cluster.independent_source_count} independent source(s) checked"]
    if facts.repo_missing:
        parts.append("named repository not found -- claim treated as unverified")
    elif facts.repo_exists:
        parts.append("repository exists; claim " +
                     ("CONFIRMED via release record" if facts.claim_verified
                      else "NOT independently verified"))
    if v < 2 and not facts.repo_missing:
        parts.append(f"single-source ceiling {SINGLE_SOURCE_CEILING}")
    return f"confidence {confidence:.2f}: " + "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# REPOSITORY / VERSION MATCHING
#
# github_lookup sorts by stars and returns the top hits. Taking results[0] as
# confirmation is the bug that let an unrelated high-star repo "confirm" a claim
# just because it shared a token. A result only confirms the claim if it is the
# repository the signal actually named.
# ---------------------------------------------------------------------------

# A product-like identifier: hyphen/dot/underscore ids (langchain-core,
# openai.beta), CamelCase including trailing capitals (LangGraph, NeuroForgeX),
# or a lowercase-then-capital form (vLLM). Plain words have none of these.
_PRODUCT_TOKEN = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+\b"
    r"|\b[A-Z][a-z0-9]*(?:[A-Z][a-z0-9]*){1,}\b"
    r"|\b[a-z]+[A-Z][A-Za-z0-9]*\b")

_VERSION = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")


def _claim_query(signal) -> str | None:
    """The repository the signal claims to be about, if it names one."""
    if signal.source == "github" and ": " in signal.title:
        candidate = signal.title.split(": ", 1)[0].strip()
        if "/" in candidate:
            return candidate.lower()
    # Otherwise look only at the TITLE for a product-like identifier. Reading
    # the summary too is how "NeuroForgeX" would accidentally get verified via
    # an unrelated "PyTorch" it happens to mention.
    for token in _PRODUCT_TOKEN.findall(signal.title):
        if len(token) >= 4 and any(c.isalpha() for c in token):
            return token.lower()
    return None


def _claim_version(signal) -> str:
    """A version the signal claims, e.g. 'v1.0.0'. Empty if none."""
    m = _VERSION.search(signal.title) or _VERSION.search(signal.summary or "")
    return m.group(0) if m else ""


def _norm_version(version: str) -> str:
    v = version.strip().lower()
    return v[1:] if v[:1] == "v" else v


def _repo_matches(query: str, full_name: str) -> bool:
    """
    Does a returned repository actually correspond to the query, rather than
    merely rank first? "owner/repo" must match exactly; a bare name must equal
    the repository's own name, not just overlap tokens with it.
    """
    q = query.strip().lower()
    fn = full_name.strip().lower()
    if not q or not fn:
        return False
    if "/" in q:
        return q == fn
    return fn.split("/")[-1] == q


def _already_confirmed(query: str, confirmed_repos: set[str]) -> bool:
    """True if the query names a repository a sibling signal already confirmed."""
    return any(_repo_matches(query, fn) for fn in confirmed_repos)


def _match_result(query: str, raw: dict) -> dict | None:
    """The first returned repo that genuinely matches the query, or None."""
    if not isinstance(raw, dict) or raw.get("error"):
        return None
    for r in raw.get("results") or []:
        if _repo_matches(query, r.get("full_name", "")):
            return r
    return None


def _describe_lookup(query: str, raw: dict, match: dict | None) -> str:
    if isinstance(raw, dict) and raw.get("error"):
        return f"github_lookup('{query}') failed: {raw['error']} -- not evidence"
    if match:
        last_push = (match.get("last_push") or "")[:10]
        return (f"github_lookup('{query}') matched {match['full_name']} "
                f"({match.get('stars', 0)} stars, pushed {last_push or 'unknown'})")
    results = (raw or {}).get("results") or []
    if results:
        names = ", ".join(r.get("full_name", "?") for r in results[:3])
        return (f"github_lookup('{query}') found no repository named '{query}' "
                f"(top hits: {names}) -- treated as unverified")
    return f"github_lookup('{query}') found no matching repository -- treated as unverified"


def _describe_release(repo: str, version: str, raw: dict, matched: dict | None) -> str:
    if isinstance(raw, dict) and raw.get("error"):
        return f"verify_release('{repo}', '{version}') failed: {raw['error']} -- not evidence"
    if matched:
        return (f"verify_release('{repo}', '{version}') CONFIRMED release "
                f"{matched.get('tag', version)} (published "
                f"{(matched.get('published_at') or '')[:10] or 'unknown'})")
    if version:
        return (f"verify_release('{repo}', '{version}') could NOT confirm that "
                f"release -- claim not verified")
    return f"verify_release('{repo}') listed releases but no specific version was claimed"


def _parse_args(arguments) -> dict:
    if isinstance(arguments, dict):
        return arguments
    try:
        return json.loads(arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# PROMPT INPUT
#
# RECONSTRUCTED 2026-09-21: lines 501-571 of the original were never recovered.
# _describe() and main() below are rebuilt from how the rest of this file uses
# them. Nothing above this block was changed.
# ---------------------------------------------------------------------------

def _describe(cluster: TrendCluster) -> str:
    """
    The first user message of the agentic loop: the facts the model needs to
    choose its tool calls, and nothing else.

    Shows each signal's title verbatim, because GitHub titles look like
    'owner/repo: tag' and that string is what _repo_matches() compares against;
    the URL, so a non-GitHub signal can be traced to a repository; and source
    and tier. Carries no score and no instructions -- SYSTEM_PROMPT owns those,
    and the score is computed from the tool results, never from this text.
    """
    lines = [f"TREND: {cluster.representative_title}", "", "SIGNALS:"]
    for i, s in enumerate(cluster.signals, 1):
        lines.append(f"{i}. [{s.source} / {s.source_tier}] {s.title}")
        if getattr(s, "url", ""):
            lines.append(f"   url: {s.url}")
        version = _claim_version(s)
        if version:
            lines.append(f"   claimed version: {version}")
        if getattr(s, "summary", ""):
            lines.append(f"   {s.summary[:400]}")
    lines.append("")
    lines.append(f"{len(cluster.signals)} signal(s) from "
                 f"{cluster.independent_source_count} independent source(s): "
                 f"{', '.join(sorted(cluster.source_tiers))}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    from clustering import cluster_signals, load_signals

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    ap = argparse.ArgumentParser(description="Verify clustered trends")
    ap.add_argument("--signals", default="01_data/signals.json")
    ap.add_argument("--index", type=int, help="verify only cluster N")
    ap.add_argument("--limit", type=int, default=3,
                    help="how many clusters to verify")
    ap.add_argument("--show-reasoning", action="store_true",
                    help="print every thought, tool call and observation")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("! OPENAI_API_KEY not set -- the deterministic loop will run the "
              "same tools and the same scorer.\n")

    clusters = cluster_signals(load_signals(args.signals))
    clusters.sort(key=lambda c: -len(c.signals))   # multi-signal clusters first
    chosen = ([clusters[args.index]] if args.index is not None
              else clusters[:args.limit])

    agent = VerificationAgent()
    for c in chosen:
        result = agent.run(c)

        print(f"\n{'=' * 70}\n{c.representative_title[:68]}")
        print(f"{len(c.signals)} signal(s), "
              f"{c.independent_source_count} independent source(s)")
        print(f"  mode       : {result.mode}")

        if args.show_reasoning and result.reasoning:
            print("\n  reasoning:")
            for step in result.reasoning:
                print(f"  [{step.iteration}] {step.thought}")
                if step.tool:
                    print(f"      {step.tool}({step.tool_args})")
                print(f"      -> {step.observation}")

        print(f"\n  confidence : {result.confidence}")
        print(f"  note       : {result.verification_note}")
        print(f"  evidence   : {len(result.evidence)} item(s)")
        for e in result.evidence[:6]:
            mark = "verified" if e.verified else "unchecked"
            print(f"      [{e.tier} / {mark}] {e.source} - {e.note}")


if __name__ == "__main__":
    main()
