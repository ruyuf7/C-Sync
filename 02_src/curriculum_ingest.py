"""
Curriculum Ingestion -> Chroma (RAG source for the CurriculumAgent)
====================================================================
Handles BOTH .pptx and .pdf slide decks, since WeCloudData gives us a mix.
Do NOT convert PPTX to PDF first -- that loses speaker notes and table
structure, and adds a step that can only degrade quality.

Expected layout (matches the bootcamp's required repo structure):

    01_data/curriculum/
      week_01/
        Intro_to_Agents.pptx
        Python_and_APIs.pdf
      week_02/
        RAG_Introduction.pptx
      week_03/          <- empty until materials are released
      ...

The WEEK comes from the folder name, not the filename. That is deliberate:
filenames get renamed ("RAG_intro_FINAL_v2.pptx") but folders don't. The
topic still comes from the filename, which is fine -- descriptive, not
structural.

Design goal: incremental. Drop a new week's slides in, re-run, the store
grows. Existing chunks are upserted by stable ID, never duplicated.

Usage:
    python curriculum_ingest.py --curriculum 01_data/curriculum --db ./vectorstore
    python curriculum_ingest.py --db ./vectorstore --query "LangChain agents"
    python curriculum_ingest.py --db ./vectorstore --query "chunking" --week 2

Embeddings: uses Chroma's built-in local model by default (no API key, free).
Swap to OpenAI embeddings for better quality -- see EMBEDDING NOTE at bottom.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import chromadb
from pptx import Presentation
import pdfplumber


# ---------------------------------------------------------------------------
# SCHEMA -- one chunk of curriculum text plus where it came from.
# The metadata is what makes recommendations citable:
#   "Week 2, RAG Introduction, slide 25"  not  "the curriculum mentions RAG"
# ---------------------------------------------------------------------------

@dataclass
class CurriculumChunk:
    text: str
    week: int | None      # 2                    (from the week_02/ folder)
    topic: str            # "RAG Introduction"   (from the filename)
    source_file: str      # "RAG_Introduction.pptx"
    slide_number: int     # 25   (cell number for notebooks)
    chunk_index: int = 0  # >0 if one slide was split into several chunks
    content_type: str = "slides"   # "slides" | "lab"
    extraction_provenance: str = "native"  # "native" | "ocr" | "hybrid"

    @property
    def chunk_id(self) -> str:
        """Stable ID so re-running ingestion updates rather than duplicates."""
        raw = f"{self.week}:{self.source_file}:{self.slide_number}:{self.chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @property
    def citation(self) -> str:
        wk = f"Week {self.week}" if self.week is not None else "Uncategorised"
        if self.content_type == "lab":
            return f"{wk} / Lab: {self.topic} / cell {self.slide_number}"
        return f"{wk} / {self.topic} / slide {self.slide_number}"


# ---------------------------------------------------------------------------
# PATH PARSING
# ---------------------------------------------------------------------------

WEEK_DIR = re.compile(r"^week[_\-]?(\d+)$", re.IGNORECASE)


def week_from_path(path: Path, root: Path) -> int | None:
    """
    Walk up from the file toward the curriculum root looking for week_NN.
    Returns None for files dropped outside a week folder -- they still get
    ingested, just without a week for filtering.
    """
    for parent in path.relative_to(root).parents:
        if parent.name:
            m = WEEK_DIR.match(parent.name)
            if m:
                return int(m.group(1))
    return None


def topic_from_filename(path: Path) -> str:
    """RAG_Introduction_Part_1.pptx -> 'RAG Introduction Part 1'"""
    return re.sub(r"[_\-]+", " ", path.stem).strip()


# ---------------------------------------------------------------------------
# EXTRACTION -- one function per format, both return the same shape.
# ---------------------------------------------------------------------------

def extract_pptx(path: Path, week: int | None) -> list[CurriculumChunk]:
    """Pull text from every shape on every slide, including speaker notes."""
    prs = Presentation(str(path))
    topic = topic_from_filename(path)
    chunks: list[CurriculumChunk] = []

    for i, slide in enumerate(prs.slides, start=1):
        parts: list[str] = []

        for shape in slide.shapes:
            # tables hold real content in course decks -- don't skip them
            if shape.has_table:
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        parts.append(" | ".join(cells))
            if shape.has_text_frame and shape.text_frame.text.strip():
                parts.append(shape.text_frame.text.strip())

        # speaker notes often explain the concept better than the slide itself
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"[Speaker notes] {notes}")

        text = "\n".join(parts).strip()
        if text:
            chunks.append(CurriculumChunk(
                text=text, week=week, topic=topic,
                source_file=path.name, slide_number=i))

    return chunks


def extract_pdf(path: Path, week: int | None) -> list[CurriculumChunk]:
    """Preserve native text and add local OCR for sparse or visual PDF pages."""
    topic = topic_from_filename(path)
    chunks: list[CurriculumChunk] = []

    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            native_text = _normalize_extracted_text(page.extract_text() or "")
            ocr_text = ""
            if OCR_ENABLED and _should_ocr_pdf_page(page, native_text):
                ocr_text = _ocr_pdf_page(
                    page, regions=_embedded_visual_regions(page))
            text, provenance = _merge_extracted_text_with_provenance(
                native_text, ocr_text)
            if text:
                chunks.append(CurriculumChunk(
                    text=text, week=week, topic=topic,
                    source_file=path.name, slide_number=i,
                    extraction_provenance=provenance))

    return chunks


def _tesseract_executable() -> str:
    """Return an available offline Tesseract executable or explain the blocker."""
    candidates = [
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError(
        "OCR requested for a sparse PDF page, but Tesseract was not found. "
        "Install an offline Tesseract executable, or disable OCR explicitly "
        "with --no-ocr (or CURRICULUM_OCR=0) -- the index will then lack "
        "text that only exists inside slide images."
    )


# Explicit opt-out for machines without Tesseract. Default stays ON, and a
# missing Tesseract still raises: silently skipping OCR would build an index
# missing image-only slide text with nothing to say so.
OCR_ENABLED = os.environ.get("CURRICULUM_OCR", "1").strip().lower() not in {"0", "false", "off", "no"}
OCR_NATIVE_SPARSE_CHARS = 200
OCR_IMAGE_MIN_AREA_RATIO = 0.03
OCR_IMAGE_MAX_AREA_RATIO = 0.92
OCR_VECTOR_MARK_THRESHOLD = 24
OCR_RESOLUTION = 300


def _box_iou(left: tuple[float, float, float, float],
             right: tuple[float, float, float, float]) -> float:
    """Intersection-over-union used to collapse duplicate embedded images."""
    x0, top, x1, bottom = left
    rx0, rtop, rx1, rbottom = right
    intersection = max(0.0, min(x1, rx1) - max(x0, rx0)) * max(
        0.0, min(bottom, rbottom) - max(top, rtop))
    union = ((x1 - x0) * (bottom - top)
             + (rx1 - rx0) * (rbottom - rtop) - intersection)
    return intersection / union if union > 0 else 0.0


def _embedded_visual_regions(page) -> list[tuple[float, float, float, float]]:
    """Return meaningful image boxes, excluding tiny logos and page backgrounds."""
    page_area = float(page.width * page.height)
    if page_area <= 0:
        return []
    regions: list[tuple[float, float, float, float]] = []
    for image in getattr(page, "images", []):
        box = (
            max(0.0, float(image.get("x0", 0.0))),
            max(0.0, float(image.get("top", 0.0))),
            min(float(page.width), float(image.get("x1", 0.0))),
            min(float(page.height), float(image.get("bottom", 0.0))),
        )
        width, height = box[2] - box[0], box[3] - box[1]
        ratio = (width * height) / page_area
        if (width <= 0 or height <= 0
                or not OCR_IMAGE_MIN_AREA_RATIO <= ratio <= OCR_IMAGE_MAX_AREA_RATIO):
            continue
        if any(_box_iou(box, existing) >= 0.85 for existing in regions):
            continue
        regions.append(box)
    return regions


def _should_ocr_pdf_page(page, native_text: str) -> bool:
    """Trigger OCR for sparse text, embedded visuals, or vector-heavy diagrams."""
    if len(native_text.strip()) < OCR_NATIVE_SPARSE_CHARS:
        return True
    if _embedded_visual_regions(page):
        return True
    vector_marks = sum(len(getattr(page, name, []))
                       for name in ("lines", "rects", "curves"))
    return vector_marks >= OCR_VECTOR_MARK_THRESHOLD


def _run_tesseract(image_path: Path, page_segmentation_mode: int) -> str:
    """Run local Tesseract with explicit UTF-8 decoding."""
    executable = _tesseract_executable()
    completed = subprocess.run(
        [executable, str(image_path), "stdout", "-l", "eng", "--oem", "1",
         "--psm", str(page_segmentation_mode),
         "-c", "preserve_interword_spaces=1"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False, timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Tesseract OCR failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()[:300]}"
        )
    return completed.stdout.strip()


def _ocr_pdf_page(page,
                  regions: list[tuple[float, float, float, float]] | None = None
                  ) -> str:
    """OCR meaningful image regions or, when none exist, the rendered page."""
    regions = _embedded_visual_regions(page) if regions is None else regions
    with tempfile.TemporaryDirectory(prefix="curriculum_ocr_") as temp_dir:
        outputs: list[str] = []
        targets = regions or [None]
        for index, box in enumerate(targets):
            image_path = Path(temp_dir) / f"page_{index}.png"
            target = page.crop(box, strict=False) if box is not None else page
            target.to_image(resolution=OCR_RESOLUTION).original.save(
                str(image_path), format="PNG")
            if box is not None:
                # Diagram labels vary between sparse and aligned layouts. Keep
                # both local passes, then remove their overlapping lines.
                aligned = _run_tesseract(image_path, 6)
                sparse = _run_tesseract(image_path, 11)
                outputs.append(_merge_extracted_text(aligned, sparse))
            else:
                outputs.append(_run_tesseract(image_path, 6))
        return _normalize_extracted_text("\n".join(outputs))


MOJIBAKE_REPLACEMENTS = {
    "â€”": "—", "â€“": "–", "â€œ": "“", "â€": "”",
    "â€ک": "”", "â€˜": "‘", "â€™": "’", "â€¢": "•",
    "â†’": "→", "â†گ": "→", "â†": "←", "â‰¥": "≥",
    "â‰¤": "≤", "Â°": "°", "آ°": "°", "Â·": "·",
    "Â ": " ", "â€‹": "", "ï»¿": "",
}


def _normalize_extracted_text(text: str) -> str:
    """Clean encoding/spacing artifacts without paraphrasing extracted text."""
    normalized = unicodedata.normalize("NFC", text or "")
    for broken, repaired in MOJIBAKE_REPLACEMENTS.items():
        normalized = normalized.replace(broken, repaired)
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
    # Limit purely decorative runs without touching ordinary identifiers.
    normalized = re.sub(r"([=_~*#•·—–-])\1{3,}", r"\1\1\1", normalized)
    lines = [re.sub(r"[\t\v\f\u00a0 ]+", " ", line).strip()
             for line in normalized.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [re.sub(r"\s+([,;:)\]}])", r"\1", line) for line in lines]
    lines = [re.sub(r"([([{])\s+", r"\1", line) for line in lines]
    compact: list[str] = []
    for line in lines:
        if line or (compact and compact[-1]):
            compact.append(line)
    while compact and not compact[-1]:
        compact.pop()
    return "\n".join(compact).strip()


def _dedupe_key(line: str) -> str:
    key = unicodedata.normalize("NFKC", line).casefold()
    key = key.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'",
                                      "—": "-", "–": "-"}))
    return re.sub(r"\s+", " ", key).strip()


def _is_duplicate_ocr_line(key: str, native_keys: list[str],
                           native_blob: str) -> bool:
    if not key:
        return True
    if key in native_keys:
        return True
    if len(key) >= 16 and key in native_blob:
        return True
    return any(len(key) >= 20 and SequenceMatcher(None, key, native).ratio() >= 0.96
               for native in native_keys)


def _merge_extracted_text_with_provenance(native_text: str,
                                           ocr_text: str) -> tuple[str, str]:
    """Merge both sources, removing only duplicate native/OCR lines."""
    native = _normalize_extracted_text(native_text)
    ocr = _normalize_extracted_text(ocr_text)
    native_lines = [line for line in native.splitlines() if line]
    native_keys = [_dedupe_key(line) for line in native_lines]
    native_blob = " ".join(native_keys)
    merged = list(native_lines)
    seen = set(native_keys)
    for line in ocr.splitlines():
        key = _dedupe_key(line)
        if key in seen or _is_duplicate_ocr_line(key, native_keys, native_blob):
            continue
        merged.append(line)
        seen.add(key)
    if native and ocr:
        provenance = "hybrid"
    elif ocr:
        provenance = "ocr"
    else:
        provenance = "native"
    return "\n".join(merged).strip(), provenance


def _merge_extracted_text(native_text: str, ocr_text: str) -> str:
    """Backward-compatible text-only wrapper around the Hybrid OCR merger."""
    return _merge_extracted_text_with_provenance(native_text, ocr_text)[0]


def extract_ipynb(path: Path, week: int | None) -> list[CurriculumChunk]:
    """
    Pull cells from a Jupyter/Colab notebook. Each cell becomes one chunk,
    so a citation reads "cell 14" -- as precise as a slide number.

    Labs matter more than slides for this project: a concept slide stays true
    across versions, but a cell that calls AgentExecutor BREAKS when LangChain
    deprecates it. That is a concrete, checkable recommendation.

    Outputs are deliberately ignored -- tracebacks, base64 images and printed
    dataframes are noise that would swamp the actual teaching content.
    """
    topic = topic_from_filename(path)
    chunks: list[CurriculumChunk] = []

    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  ! could not parse notebook {path.name}: {e}")
        return []

    for i, cell in enumerate(nb.get("cells", []), start=1):
        source = cell.get("source", "")
        # nbformat stores source as a list of lines, but some tools write a string
        text = ("".join(source) if isinstance(source, list) else str(source)).strip()
        if not text:
            continue

        kind = cell.get("cell_type")
        if kind == "code":
            # tag it so retrieval can tell prose from executable code
            text = f"[Code cell]\n{text}"
        elif kind != "markdown":
            continue   # raw cells are usually config, not content

        chunks.append(CurriculumChunk(
            text=text, week=week, topic=topic,
            source_file=path.name, slide_number=i,
            content_type="lab"))

    return chunks


def extract_all(curriculum_root: Path) -> list[CurriculumChunk]:
    """
    Recursively walk the curriculum tree, routing each file to the right
    extractor and tagging it with the week from its folder.
    """
    chunks: list[CurriculumChunk] = []
    skipped: list[str] = []
    image_only: list[str] = []

    for path in sorted(curriculum_root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if ".ipynb_checkpoints" in path.parts:
            continue   # Jupyter autosaves - duplicates of the real notebook

        week = week_from_path(path, curriculum_root)
        suffix = path.suffix.lower()

        if suffix == ".pptx":
            found = extract_pptx(path, week)
        elif suffix == ".pdf":
            found = extract_pdf(path, week)
        elif suffix == ".ipynb":
            found = extract_ipynb(path, week)
        else:
            skipped.append(path.name)
            continue

        if not found:
            # no extractable text at all -- almost certainly a scan or an
            # all-image deck. This is the failure mode worth catching early.
            image_only.append(str(path.relative_to(curriculum_root)))
        chunks.extend(found)

    if skipped:
        print(f"  ! skipped unsupported files: {', '.join(skipped)}")
    if image_only:
        print(f"  !! NO TEXT EXTRACTED from: {', '.join(image_only)}")
        print("     Likely image-only slides. Text RAG cannot see them.")

    return chunks


# ---------------------------------------------------------------------------
# CHUNKING -- split long slides so retrieval stays precise.
# A whole dense slide as one chunk buries the relevant sentence.
# ---------------------------------------------------------------------------

MIN_CHARS = 50


def drop_thin_chunks(chunks: list[CurriculumChunk],
                     min_chars: int = MIN_CHARS) -> tuple[list[CurriculumChunk], int]:
    """
    Remove section dividers and title-only slides ("Vector DB | RAG",
    "01 Preamble", "Questions?").

    These are actively harmful, not just useless: a title slide is almost
    entirely topic keywords, so it scores high on any query about that topic
    and crowds out the substantive slide underneath. Measured on real course
    decks, ~19% of pages were under this threshold.
    """
    kept = [c for c in chunks if len(c.text) >= min_chars]
    return kept, len(chunks) - len(kept)


def split_long_chunks(chunks: list[CurriculumChunk],
                      max_chars: int = 1200,
                      overlap: int = 150) -> list[CurriculumChunk]:
    """~1200 chars is roughly 300 tokens. Overlap avoids cutting mid-concept."""
    out: list[CurriculumChunk] = []

    for chunk in chunks:
        if len(chunk.text) <= max_chars:
            out.append(chunk)
            continue

        start, idx = 0, 0
        while start < len(chunk.text):
            out.append(CurriculumChunk(
                text=chunk.text[start:start + max_chars],
                week=chunk.week, topic=chunk.topic,
                source_file=chunk.source_file,
                slide_number=chunk.slide_number, chunk_index=idx,
                content_type=chunk.content_type,
                extraction_provenance=chunk.extraction_provenance))
            start += max_chars - overlap
            idx += 1

    return out


# ---------------------------------------------------------------------------
# VECTOR STORE
# ---------------------------------------------------------------------------

COLLECTION = "wecloud_curriculum"


def get_collection(db_path: str, embedding_function=None):
    """
    embedding_function=None uses Chroma's built-in local model (free, no key).
    Pass the SAME function to ingest and query, or results are meaningless.
    """
    client = chromadb.PersistentClient(path=db_path)
    kwargs = {"name": COLLECTION, "metadata": {"hnsw:space": "cosine"}}
    if embedding_function is not None:
        kwargs["embedding_function"] = embedding_function
    return client.get_or_create_collection(**kwargs)


def ingest(curriculum_root: str, db_path: str, embedding_function=None) -> int:
    root = Path(curriculum_root)
    if not root.is_dir():
        raise SystemExit(f"curriculum folder not found: {curriculum_root}")

    print(f"reading {curriculum_root} ...")
    raw = extract_all(root)

    # coverage per week -- tells you at a glance what is actually indexed
    by_week: dict[str, int] = {}
    for c in raw:
        key = f"week_{c.week:02d}" if c.week is not None else "uncategorised"
        by_week[key] = by_week.get(key, 0) + 1
    for wk in sorted(by_week):
        print(f"    {wk}: {by_week[wk]} items")
    by_type: dict[str, int] = {}
    for c in raw:
        by_type[c.content_type] = by_type.get(c.content_type, 0) + 1
    if len(by_type) > 1:
        print("    (" + ", ".join(f"{k}: {v}" for k, v in sorted(by_type.items())) + ")")

    kept, dropped = drop_thin_chunks(raw)
    if dropped:
        print(f"  dropped {dropped} title/divider slides under {MIN_CHARS} chars")

    chunks = split_long_chunks(kept)
    print(f"  {len(raw)} slides -> {len(chunks)} chunks")

    if not chunks:
        print("  nothing to ingest.")
        return 0

    collection = get_collection(db_path, embedding_function)

    # upsert (not add) so re-running is safe and incremental
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{
            "week": c.week if c.week is not None else -1,   # Chroma rejects None
            "topic": c.topic,
            "source_file": c.source_file,
            "slide_number": c.slide_number,
            "content_type": c.content_type,
            "extraction_provenance": c.extraction_provenance,
        } for c in chunks],
    )

    print(f"  stored -> {db_path} ({collection.count()} chunks total)")
    return len(chunks)


# ---------------------------------------------------------------------------
# HYBRID SEARCH
# Embeddings blur rare identifiers. Measured on real decks: a query for
# "FAISS" returned nothing, even though 3 slides contain the literal word --
# MiniLM has no useful representation for a rare acronym, so the similarity
# score is close to noise.
#
# GitHub release notes are full of exactly these tokens (AgentExecutor,
# create_agent, @tool, FAISS), so this is the common case for us, not an
# edge case. Fix: run the semantic search, then a literal substring search
# for any rare identifiers in the query, and merge.
# ---------------------------------------------------------------------------

# words that look like identifiers, not prose:
#   FAISS, RAG, PEFT        -> all-caps acronyms (2+ chars)
#   AgentExecutor, LangChain -> CamelCase
#   create_agent, text_splitter -> snake_case
#   @tool, .from_documents  -> punctuation-prefixed
IDENTIFIER = re.compile(
    r"\b[A-Z]{2,}\b"                    # FAISS, RAG, LLM
    r"|\b[a-z]+_[a-z_]+\b"              # create_agent
    r"|\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b" # AgentExecutor, LangChain
    r"|@[a-zA-Z_]+"                     # @tool
)

# acronyms too common in this domain to be useful as exact-match signals --
# they appear on dozens of slides and would swamp the semantic results
STOP_IDENTIFIERS = {"AI", "ML", "API", "LLM", "LLMS", "GPU", "CPU", "PDF", "JSON",
                    "HTTP", "URL", "OK", "ID", "UI", "OS", "IT", "US"}


def extract_identifiers(question: str) -> list[str]:
    """Pull out rare technical tokens worth searching for literally."""
    found = IDENTIFIER.findall(question)
    return [t for t in dict.fromkeys(found) if t.upper() not in STOP_IDENTIFIERS]


def _format_citation(wk, meta) -> str:
    head = f"Week {wk}" if wk else "Uncategorised"
    if meta.get("content_type") == "lab":
        return f"{head} / Lab: {meta['topic']} / cell {meta['slide_number']}"
    return f"{head} / {meta['topic']} / slide {meta['slide_number']}"


def _row_to_hit(doc, meta, similarity: float, exact_match: str | None) -> dict:
    wk = meta["week"] if meta["week"] != -1 else None
    return {
        "text": doc,
        "week": wk,
        "topic": meta["topic"],
        "source_file": meta["source_file"],
        "slide_number": meta["slide_number"],
        "content_type": meta.get("content_type", "slides"),
        "extraction_provenance": meta.get("extraction_provenance", "native"),
        "similarity": similarity,
        "exact_match": exact_match,     # the identifier found, or None
        "citation": _format_citation(wk, meta),
    }


LEXICAL_TOKEN = re.compile(r"[a-z0-9]+")
LEXICAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "with", "this", "these", "those", "using", "used", "change", "changes",
}


def _lexical_tokens(text: str) -> list[str]:
    """Normalize prose for deterministic, case-insensitive lexical matching."""
    return [token for token in LEXICAL_TOKEN.findall(text.casefold())
            if len(token) >= 3 and not token.isdigit()
            and token not in LEXICAL_STOPWORDS]


def _lexical_rank(question: str, documents: list[str]) -> dict[int, tuple[float, str | None]]:
    """Return document-index lexical scores and any literal identifier match."""
    query_tokens = set(_lexical_tokens(question))
    if not query_tokens:
        return {}
    tokenized = [_lexical_tokens(doc or "") for doc in documents]
    doc_frequency: dict[str, int] = {}
    for tokens in tokenized:
        for token in set(tokens):
            doc_frequency[token] = doc_frequency.get(token, 0) + 1
    total_docs = max(1, len(tokenized))
    identifiers = [token.casefold() for token in extract_identifiers(question)]
    scored: dict[int, tuple[float, str | None]] = {}
    for index, (doc, tokens) in enumerate(zip(documents, tokenized)):
        present = set(tokens) & query_tokens
        if not present:
            continue
        score = 0.0
        for token in present:
            # Length and inverse document frequency favor distinctive terms
            # without relying on source names, page numbers, or benchmark IDs.
            idf = 1.0 + math.log((total_docs + 1) / (doc_frequency[token] + 1))
            score += (1.0 + min(len(token), 20) / 10.0) * idf
        exact = next((identifier for identifier in identifiers
                      if identifier in (doc or "").casefold()), None)
        if exact:
            score += 2.0
        scored[index] = (score, exact)
    return scored


def query(db_path: str, question: str, k: int = 3,
          week: int | None = None, content_type: str | None = None,
          embedding_function=None, hybrid: bool = True) -> list[dict]:
    """
    This is what CurriculumAgent calls. Returns chunks WITH citations.

    week=N restricts the search to one week's material.
    content_type="lab" restricts to notebooks, "slides" to decks. Useful
    because a trend that breaks LAB CODE is more urgent than one that dates
    a concept slide -- the code literally stops running.
    hybrid=False disables the literal identifier pass (semantic only).

    Hits carry an "exact_match" field: the identifier literally found in the
    slide, or None for purely semantic matches. Exact matches are ranked
    first -- if a slide literally contains "FAISS", it is relevant to a FAISS
    question regardless of what the embedding distance says. Callers should
    treat exact_match hits as clearing RELEVANCE_FLOOR automatically, since
    embedding similarity is not meaningful for these.
    """
    collection = get_collection(db_path, embedding_function)
    total = collection.count()
    if total == 0:
        return []

    filters = []
    if week is not None:
        filters.append({"week": week})
    if content_type is not None:
        filters.append({"content_type": content_type})
    # Chroma needs $and for multiple conditions, a bare dict for one
    where = None
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    # --- pass 1: semantic candidate pool
    # Keep more candidates than the public result limit so lexical evidence
    # can promote a relevant page that semantic ranking initially missed.
    semantic_limit = min(total, max(k * 10, 50))
    kwargs = {"query_texts": [question], "n_results": semantic_limit}
    if where:
        kwargs["where"] = where
    res = collection.query(**kwargs)

    semantic_rows: list[tuple[float, dict, str]] = []
    if res["documents"] and res["documents"][0]:
        semantic_rows = [
            (round(1 - dist, 3), meta, doc)
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    # Fetch the filtered corpus once for deterministic lexical scoring. This
    # replaces the old identifier-only pass while retaining exact_match.
    corpus = collection.get(where=where, include=["documents", "metadatas"])
    corpus_docs = corpus.get("documents", []) or []
    corpus_metas = corpus.get("metadatas", []) or []
    lexical = _lexical_rank(question, corpus_docs) if hybrid else {}
    lexical_rows = sorted(
        lexical.items(), key=lambda item: (-item[1][0],
                                           corpus_metas[item[0]].get("source_file", ""),
                                           corpus_metas[item[0]].get("slide_number", 0))
    )

    # Reciprocal Rank Fusion combines semantic and lexical rank without
    # comparing their incompatible score scales. The stable tie-breakers make
    # repeated calls deterministic.
    rrf_constant = 60.0
    fused: dict[tuple, dict] = {}
    semantic_keys: dict[tuple, tuple[float, dict, str]] = {}
    for rank, (similarity, meta, doc) in enumerate(semantic_rows, start=1):
        key = (meta["source_file"], meta["slide_number"])
        semantic_keys[key] = (similarity, meta, doc)
        fused.setdefault(key, {"semantic_rank": None, "lexical_rank": None,
                               "lexical_exact": None})["semantic_rank"] = rank

    for rank, (index, (_, exact)) in enumerate(lexical_rows, start=1):
        meta = corpus_metas[index]
        key = (meta["source_file"], meta["slide_number"])
        fused.setdefault(key, {"semantic_rank": None, "lexical_rank": None,
                               "lexical_exact": None})["lexical_rank"] = rank
        fused[key]["lexical_exact"] = exact

    ranked = sorted(
        fused.items(),
        key=lambda item: (
            -((1.0 / (rrf_constant + item[1]["semantic_rank"])
               if item[1]["semantic_rank"] else 0.0)
              + (1.0 / (rrf_constant + item[1]["lexical_rank"])
                 if item[1]["lexical_rank"] else 0.0)),
            item[0][0], item[0][1],
        ),
    )

    hits: list[dict] = []
    for key, ranks in ranked[:k]:
        if key in semantic_keys:
            similarity, meta, doc = semantic_keys[key]
        else:
            index = next(i for i, m in enumerate(corpus_metas)
                         if (m["source_file"], m["slide_number"]) == key)
            meta, doc = corpus_metas[index], corpus_docs[index]
            similarity = None
        hits.append(_row_to_hit(doc, meta, similarity, ranks["lexical_exact"]))
    return hits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Ingest course slides into Chroma")
    ap.add_argument("--curriculum", default="01_data/curriculum",
                    help="root folder containing week_NN/ subfolders")
    ap.add_argument("--db", default="./vectorstore", help="Chroma persistence path")
    ap.add_argument("--query", help="skip ingestion, just search the existing store")
    ap.add_argument("--week", type=int, help="restrict a query to one week")
    ap.add_argument("--type", choices=["slides", "lab"], dest="content_type",
                    help="restrict a query to slides or lab notebooks")
    ap.add_argument("-k", type=int, default=3, help="results to return")
    ap.add_argument("--no-ocr", action="store_true",
                    help="skip Tesseract OCR of sparse/diagram PDF pages "
                         "(same as CURRICULUM_OCR=0); native text only")
    args = ap.parse_args()

    if args.query:
        hits = query(args.db, args.query, args.k, week=args.week,
                     content_type=args.content_type)
        if not hits:
            print("no results -- has anything been ingested?")
            return
        for h in hits:
            # similarity is None for hits found only by the lexical pass; that is
            # "exact" only when a literal identifier matched, else keyword overlap
            if h["similarity"] is not None:
                score = f"{h['similarity']}"
            else:
                score = "exact" if h["exact_match"] else "lexical"
            tag = f"  <- literal '{h['exact_match']}'" if h["exact_match"] else ""
            print(f"\n[{score}] {h['citation']}{tag}")
            print(f"  {h['text'][:200]}...")
    else:
        if args.no_ocr:
            global OCR_ENABLED
            OCR_ENABLED = False
        ingest(args.curriculum, args.db)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# EMBEDDING NOTE
# ---------------------------------------------------------------------------
# Chroma's default embedding model runs locally and costs nothing -- fine for
# development. For the final demo, OpenAI embeddings retrieve noticeably better
# on technical text. To switch:
#
#   from chromadb.utils import embedding_functions
#   ef = embedding_functions.OpenAIEmbeddingFunction(
#           api_key=os.environ["OPENAI_API_KEY"],
#           model_name="text-embedding-3-small")
#   client.get_or_create_collection(name=COLLECTION, embedding_function=ef)
#
# Pass the SAME embedding_function on both ingest and query, or results are
# meaningless. If you switch models, DELETE the vectorstore folder and
# re-ingest -- otherwise Chroma errors on mismatched embedding dimensions.
#
# ---------------------------------------------------------------------------
# IMAGE-ONLY SLIDES
# ---------------------------------------------------------------------------
# If ingestion reports "NO TEXT EXTRACTED" for a deck, its content lives in
# diagrams that text extraction cannot see. Options, cheapest first:
#   1. Accept it and document the gap (fine for an MVP).
#   2. Hand-write a short description per diagram slide into a .txt beside it.
#   3. Send slide images to a vision model and store the description.
# Do NOT build option 3 first -- get text RAG working, then decide whether the
# missing diagrams actually hurt retrieval quality.
