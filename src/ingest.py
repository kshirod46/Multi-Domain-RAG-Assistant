"""
PART 1 — Ingestion & Chunking
=============================
Loads documents from each of the three collections (novel, ml_papers,
podcast_transcripts), cleans them, and splits them into overlapping
chunks ready for embedding in Part 2.

Design choice worth explaining in an interview: we keep each collection
SEPARATE (three chunk lists, not one merged list) so that Part 2 can
build three independent FAISS indexes -- this is what lets the final
app offer a "choose your knowledge base" dropdown instead of one messy
combined index.
"""
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - fallback if the package is missing.
    RecursiveCharacterTextSplitter = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
COLLECTIONS = ("novel", "ml_papers", "podcast_transcripts", "others")

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


@dataclass
class Chunk:
    collection: str
    source_file: str
    chunk_id: int
    text: str
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(path)
    return [
        (page_number, page.extract_text() or "")
        for page_number, page in enumerate(reader.pages, start=1)
    ]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Keep chunking enabled while making each chunk more semantically coherent."""
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 30]

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c for c in chunks if len(c.strip()) > 30]


def ingest_collection(collection: str) -> list[Chunk]:
    folder = DATA_DIR / collection
    all_chunks = []
    if not folder.is_dir():
        return all_chunks
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            sections = read_pdf_pages(path)
        else:
            sections = [(None, read_txt(path))]

        chunk_id = 0
        for page_number, raw_section in sections:
            cleaned = clean_text(raw_section)
            pieces = chunk_text(cleaned)
            for piece in pieces:
                all_chunks.append(
                    Chunk(
                        collection=collection,
                        source_file=path.name,
                        chunk_id=chunk_id,
                        text=piece,
                        page_number=page_number,
                    )
                )
                chunk_id += 1
    return all_chunks


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for collection in COLLECTIONS:
        chunks = ingest_collection(collection)
        out_path = OUTPUT_DIR / f"{collection}_chunks.json"
        with out_path.open("w", encoding="utf-8") as file:
            json.dump([asdict(c) for c in chunks], file, indent=2, ensure_ascii=False)
        summary[collection] = len(chunks)
        print(f"[{collection}] {len(chunks)} chunks -> {out_path}")
    print("\nSummary:", summary)
    return summary


if __name__ == "__main__":
    run()
