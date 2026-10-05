"""
Shared data contracts for the AI Trend Agent pipeline.
=======================================================
Every stage imports its types from HERE. Do not copy these definitions into
your own module -- a copy silently goes stale when someone adds a field, and
the resulting bug is very hard to trace.

    from schemas import RawSignal, TrendCluster, VerifiedTrend

TEAM RULE: nobody changes a field name or type in this file without telling
the team first. Everything downstream breaks silently otherwise.

Pipeline flow -- each stage's output is the next stage's input:

    fetch_raw_signals()          -> list[RawSignal]
    cluster_signals(signals)     -> list[TrendCluster]
    VerificationAgent.run(c)     -> VerifiedTrend
    CurriculumAgent.run(trend)   -> (citation, CurriculumMatch)
    EvaluationAgent.run(...)     -> EvaluationResult
    RecommendationAgent.run(c)   -> Recommendation
"""

from dataclasses import dataclass, field, asdict
from typing import Literal


# ---------------------------------------------------------------------------
# CONTROLLED VOCABULARIES
# Use these constants instead of typing string literals. A typo in
# "secondary" is a bug the type checker cannot catch, but a typo in
# SourceTier is an immediate NameError.
# ---------------------------------------------------------------------------

SourceTier = Literal["primary", "secondary"]
# primary   = the tool's own words: official repo release, official blog, the paper
# secondary = someone reporting on it: news article, forum post, tweet

ActionTier = Literal[
    "watch",                      # plausible but unverified, or no curriculum impact
    "update_existing_material",   # a specific slide/lab now teaches something outdated
    "add_optional_content",       # relevant but supplementary
    "add_new_lesson",             # verified, durable, not covered anywhere
    "investigate_larger_change",  # paradigm shift affecting multiple modules
]


# ---------------------------------------------------------------------------
# STAGE 1 OUTPUT -- monitoring
# ---------------------------------------------------------------------------

@dataclass
class RawSignal:
    """One item pulled from a monitored source, before any processing."""
    title: str
    source: str              # "github" | "official_blog" | "hackernews" | ...
    source_tier: SourceTier
    summary: str
    url: str = ""
    published: str = ""      # ISO date if the source provides one


# ---------------------------------------------------------------------------
# STAGE 2 OUTPUT -- clustering
# ---------------------------------------------------------------------------

@dataclass
class TrendCluster:
    """Several signals judged to be reporting the same underlying event."""
    representative_title: str
    signals: list[RawSignal] = field(default_factory=list)

    @property
    def source_tiers(self) -> set[str]:
        return {s.source_tier for s in self.signals}

    @property
    def independent_source_count(self) -> int:
        """Distinct sources, not signal count -- three articles from one
        outlet is one independent source, not three."""
        return len({s.source for s in self.signals})


# ---------------------------------------------------------------------------
# AGENT 1 OUTPUT -- verification
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """One source consulted while verifying a claim. Powers the UI trail."""
    source: str
    # "tool" marks a verifier tool result (github_lookup / verify_release),
    # which is evidence ABOUT sources, never a source itself. Widened here
    # only -- RawSignal keeps the two-value SourceTier.
    tier: SourceTier | Literal["tool"]
    url: str = ""
    note: str = ""           # what this source actually confirmed or failed to
    kind: Literal["source", "tool"] = "source"
    verified: bool = False   # a tool actually confirmed what this source names


@dataclass
class ReasoningStep:
    """One step of the verifier's loop: the thought, the tool call (if any),
    and what the RAW tool result showed. The audit trail, not the score."""
    iteration: int
    thought: str
    observation: str
    tool: str = ""
    tool_args: dict = field(default_factory=dict)


@dataclass
class VerifiedTrend:
    cluster: TrendCluster
    confidence: float                    # 0.0 - 1.0
    verification_note: str
    evidence: list[Evidence] = field(default_factory=list)
    status: Literal["verified", "contradicted", "unverified", "needs_clarification"] = "unverified"
    # Facts the deterministic scorer used (all defaulted, so older call sites
    # and snapshots stay valid). Confidence is computed FROM these in code.
    verified_source_count: int = 0
    repo_exists: bool = False
    claim_verified: bool = False
    reasoning: list[ReasoningStep] = field(default_factory=list)
    mode: str = ""           # which loop gathered the evidence (agentic / deterministic)


# ---------------------------------------------------------------------------
# AGENT 2 OUTPUT -- curriculum RAG
# Mirrors what curriculum_ingest.query() returns for a single hit.
# ---------------------------------------------------------------------------

@dataclass
class CurriculumMatch:
    """A specific slide the trend was matched against."""
    week: int | None
    topic: str                  # "RAG Introduction"
    source_file: str            # "RAG Introduction.pdf"
    slide_number: int
    matched_text: str
    similarity: float | None    # None when found by literal identifier match
    exact_match: str | None = None   # the identifier found, e.g. "FAISS"
    content_type: str = "slides"     # "slides" | "lab"

    @property
    def citation(self) -> str:
        wk = f"Week {self.week}" if self.week is not None else "Uncategorised"
        if self.content_type == "lab":
            return f"{wk} / Lab: {self.topic} / cell {self.slide_number}"
        return f"{wk} / {self.topic} / slide {self.slide_number}"

    @property
    def is_lab(self) -> bool:
        """
        Labs matter more than slides. A concept slide stays true across
        versions; a notebook cell that CALLS a deprecated API stops running.
        The recommendation tier should weight this.
        """
        return self.content_type == "lab"

    @property
    def is_reliable(self) -> bool:
        """
        An exact identifier match is trustworthy regardless of embedding
        distance. Measured on real decks: a slide literally containing
        "FAISS" scored 0.303 -- indistinguishable from unrelated noise --
        yet was the correct answer. Gate on this, never on similarity alone.
        """
        return self.exact_match is not None or (self.similarity or 0) >= RELEVANCE_FLOOR


# Calibrated against real WeCloudData decks with all-MiniLM-L6-v2:
#   genuine match  ("chunking strategy")      -> 0.65
#   genuine match  ("LangChain tools")        -> 0.66
#   pure noise     ("kubernetes autoscaling") -> 0.31
# MiniLM compresses scores into a narrow band, so unrelated text lands near
# 0.3 rather than 0. Re-measure if the embedding model changes.
RELEVANCE_FLOOR = 0.48


# ---------------------------------------------------------------------------
# AGENT 3 OUTPUT -- evaluation
# ---------------------------------------------------------------------------

# maturity  -> "is this real?"          (from VerificationAgent confidence)
# relevance -> "does it matter to us?"  (from CurriculumAgent match)
MATURITY_WEIGHT = 0.5
RELEVANCE_WEIGHT = 0.5


@dataclass
class EvaluationResult:
    trend: VerifiedTrend
    match: CurriculumMatch | None
    maturity_score: int          # 1-5
    relevance_score: int         # 1-5
    total_score: float
    rationale: str = ""          # why these scores, in plain language


# ---------------------------------------------------------------------------
# AGENT 4 OUTPUT -- recommendation
# This is what FastAPI serves and the UI renders. Freeze this shape early:
# the frontend cannot be built in parallel until it is settled.
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    trend: str
    confidence: float
    verification_note: str
    evidence: list[Evidence]
    recommended_action: ActionTier
    action_plan: list[str]
    match: CurriculumMatch | None = None
    total_score: float | None = None

    def to_dict(self) -> dict:
        """JSON-serialisable form for the API response."""
        d = asdict(self)
        if self.match is not None:
            d["match"]["citation"] = self.match.citation
        return d
