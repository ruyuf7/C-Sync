# Testing and Running Guide

Run every command from the repository root. Use Python 3.10 or newer and install
`requirements.txt` first:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env` only for live agent runs. The commands below were
checked against the current argument parsers.

## Pre-merge check

The intended pre-merge command is:

```powershell
python 02_src/agents/test_chain.py
```

It makes zero API calls (it blanks `OPENAI_API_KEY` at import) and currently
reports `243 passed, 0 failed, 0 errored, 0 skipped`. Run the verifier tests too:

```powershell
python 02_src/tests/test_verification.py
```

which report `14/14 passed`.

The former `test_tool_use.py` diagnostic was also removed. It answered whether
verification actually calls its tools instead of guessing from tone. Its trace
showed that the verification agent did call the configured tools, so the finding
is preserved here even though the one-off file is gone.

## Offline pipeline checks

These commands need no API key unless noted.

### Curriculum ingestion and search

```powershell
python 02_src/curriculum_ingest.py --curriculum 01_data/curriculum --db ./vectorstore
python 02_src/curriculum_ingest.py --db ./vectorstore --query "FAISS" --type lab
python 02_src/curriculum_ingest.py --db ./vectorstore --query "chunking" --week 2
```

The verified ingestion run read weeks 2 through 4, produced `2197 slides -> 1911
chunks`, and reported `lab: 1341, slides: 856`. The FAISS query returned three
exact lab matches, including Week 2 labs `Build a Research Paper Assistant using
RAG` and `Building a Simple RAG System from Scratch`. The chunking query returned
Week 2 RAG Introduction slides 20, 19, and 21 with scores `0.667`, `0.643`, and
`0.54`.

Ingestion is incremental. Adding a new week and rerunning adds records; it does
not overwrite existing records. Delete `./vectorstore` and re-ingest when the
ingestion logic changes because upsert does not remove stale chunks.

Check that the returned citation has the expected week, content type, title, and
slide or cell number. For lab results, inspect the cited notebook cell rather
than trusting the title alone.

### Monitoring and clustering

```powershell
python 02_src/monitoring_rss.py --check
python 02_src/monitoring_github.py --days 30
python 02_src/clustering.py --signals 01_data/signals.json --check
python 02_src/clustering.py --signals 01_data/signals.json --frequencies
```

The RSS check reported all three feeds `[OK]`: LangChain (100 entries), OpenAI
(1192), and Hugging Face (861). GitHub monitoring found `17 signals total`:
7 LangChain, 10 OpenAI, and none for Chroma or FastAPI in the last 30 days.
Clustering loaded 45 signals into 41 clusters and reported all checks as `YES`,
including the MCP cross-source pair and separate LangChain packages. The
frequency report correctly excluded high-frequency identifiers such as
`openai-python` (10), `langsmith` (9), `langchain-ai` (6), and `langchain` (5)
under the ceiling of 3.

Check that network commands report current source counts and URLs, and that the
cluster check still identifies the cross-source pair. Network results naturally
change over time.

### Snapshot replay

```powershell
python 02_src/demo_snapshot.py --replay
```

Replay makes ZERO API calls and needs no key or network. The checked snapshot
rendered `45 signals -> 41 clusters -> 15 evaluated -> 13 recommendations`,
with `5` update-existing-material, `1` add-new-lesson, `1` add-optional-content,
and `6` watch recommendations. Use this to see a complete result without
spending money.

`--capture` calls the live chain and overwrites the snapshot by default. When
experimenting, use a different path, for example:

```powershell
python 02_src/demo_snapshot.py --capture --snapshot 01_data/experiment.json --limit 2
```

## Agent commands

Every agent run costs API calls, roughly 5-8 per trend across the chain. Use
`--limit` while developing. `--offset` processes clusters in batches without
re-paying for earlier clusters.

### Verification agent

Purpose: decide whether a trend is real and collect source evidence by using the
configured verification tools.

```powershell
python 02_src/agents/verification.py --signals 01_data/signals.json --verbose
```

This requires `OPENAI_API_KEY` and was not run during verification because it
costs money. Good output includes a verified trend, a confidence value, source
evidence, and a verbose tool-call trace. Check that evidence comes from the
source records and that the trace shows actual tool calls, not just prose.

### Curriculum agent

Purpose: search the indexed curriculum and identify an exact or similar match.

```powershell
python 02_src/agents/curriculum.py --signals 01_data/signals.json --verbose --limit 2
```

This requires `OPENAI_API_KEY` and was not run. Good output names a curriculum
citation such as a week, slide or lab, and cell where applicable. Check the
citation, content type, and whether the match is genuinely related. A no-match
result is meaningful only when the search was actually performed.

### Evaluation agent

Purpose: combine verification and curriculum evidence into deterministic scores
and an impact assessment.

```powershell
python 02_src/agents/evaluation.py --signals 01_data/signals.json --limit 3
```

This requires `OPENAI_API_KEY` and was not run. Good output contains the trend,
its maturity and relevance scores, and the resulting evaluation. Check that the
numbers are in the documented score ranges and that they agree with the evidence,
not with unsupported model confidence.

### Recommendation agent

Purpose: choose an action tier and write a concrete plan. It also orchestrates
the verification, curriculum, evaluation, and recommendation stages.

```powershell
python 02_src/agents/recommendation.py --signals 01_data/signals.json --limit 10
python 02_src/agents/recommendation.py --signals 01_data/signals.json --offset 10 --limit 10
```

These commands require `OPENAI_API_KEY` and were not run. Good output contains
an allowed tier, evidence or a curriculum citation when applicable, and a short
plan that agrees with the selected tier. Check that plans do not contradict the
Python-selected tier and that `--offset` changes the processed batch.

The pipeline is NON-DETERMINISTIC. The same signals can produce different tiers
between runs because each agent chooses its own tool queries. We measured a
trend matching `Week 3 / cell 17` in one run and no match in the next. This is
expected LLM behavior, not a bug; do not treat a single live run as ground truth.

## Design decisions

Python computes every score; the model only writes prose. When the model was
free to output confidence, it returned `0.85` for both a single-source trend and
a two-source trend. Scores are therefore reproducible and testable without a
key.

`CurriculumMatch.is_reliable` gates on an exact-identifier match or similarity,
never similarity alone. On the project decks, a slide literally containing
`FAISS` scored `0.303` while an unrelated slide scored `0.31`; ranking by
similarity alone gets that exactly backwards.

`curriculum_checked` distinguishes "searched and found nothing" from "never
searched". Without it, the system can recommend a new lesson for material that
may already be taught.

A version bump with no curriculum match is `watch`, not `add_new_lesson`. Four
openai-python releases produced a new-lesson recommendation for a library taught
in several labs, which showed why release noise needs this rule.

An in-domain gate is required before `add_new_lesson`. One batch produced new
lessons for nine of ten trends, including journalism in Ukraine and the
Navier-Stokes Millennium Prize Problem. Domain terms are checked against the
title only because vendor names appear in nearly every post from that vendor.

Plans are validated against the chosen tier. A plan arguing for a new lesson is
rejected when Python chose `watch`, preventing prompt-injected prose from
surviving as a recommendation.

Lab notebooks are ingested in solution form where available. Some labs are
provided by WeCloudData only as student versions and are ingested as-is, so
`studentVersion` in a citation is expected.

## Known repository findings

`requirements.txt` covers every third-party import found under `02_src/**/*.py`:
`chromadb`, `python-pptx`, `pdfplumber`, `feedparser`, `requests`, `pydantic`,
`openai`, and `python-dotenv`.

The editor diagnostics on `02_src/agents/curriculum.py` and
`02_src/curriculum_ingest.py` are unresolved-environment imports: `openai`,
`chromadb`, and `pptx`. They are not code changes made by this cleanup. Install
`requirements.txt` and select that environment in VS Code to resolve them.

The empty `02_src/pipeline.py` stub was deleted. The maintained orchestration is
already in `agents/recommendation.py`; deleting the unused stub avoids a second,
misleading entry point rather than duplicating orchestration.
