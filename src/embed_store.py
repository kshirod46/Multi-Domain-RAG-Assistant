"""Build one FAISS vector store for each document collection."""

import json
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

COLLECTIONS = ("novel", "ml_papers", "podcast_transcripts", "others")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
INDEX_DIR = PROJECT_ROOT / "faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"


class SentenceTransformerEmbeddings(Embeddings):
    """Adapt sentence-transformers to LangChain's embeddings interface."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> SentenceTransformerEmbeddings:
    """Load the embedding model once per process."""
    return SentenceTransformerEmbeddings()


def _load_documents(collection: str) -> list[Document]:
    path = OUTPUT_DIR / f"{collection}_chunks.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run `python src/ingest.py` before building indexes."
        )

    with path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"{path} does not contain any chunks.")

    return [
        Document(
            page_content=chunk["text"],
            metadata={
                "collection": chunk["collection"],
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk.get("page_number"),
                "line_start": chunk.get("line_start"),
                "line_end": chunk.get("line_end"),
            },
        )
        for chunk in chunks
    ]


def build_index(collection: str, embeddings: Embeddings | None = None) -> Path:
    """Embed one collection and persist its FAISS index."""
    if collection not in COLLECTIONS:
        raise ValueError(f"Unknown collection {collection!r}. Choose from {COLLECTIONS}.")

    embeddings = embeddings or SentenceTransformerEmbeddings()
    index = FAISS.from_documents(_load_documents(collection), embeddings)
    destination = INDEX_DIR / collection
    destination.mkdir(parents=True, exist_ok=True)
    index.save_local(str(destination))
    print(f"[{collection}] index saved to {destination}")
    return destination


def build_all() -> None:
    """Build all three indexes with one shared embedding model."""
    embeddings = get_embeddings()
    for collection in COLLECTIONS:
        try:
            build_index(collection, embeddings)
        except ValueError as error:
            if "does not contain any chunks" not in str(error):
                raise
            print(f"[{collection}] skipped because it has no documents yet")


if __name__ == "__main__":
    build_all()
