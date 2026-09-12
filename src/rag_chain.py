"""Retrieval and Groq generation for the three collection-specific indexes."""

import os
import re
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from groq import APIError
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.embed_store import COLLECTIONS, INDEX_DIR, SentenceTransformerEmbeddings

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_TOKENS = max(256, int(os.getenv("RAG_MAX_TOKENS", "2500")))
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the provided context. "
            "Use the retrieved chunks as evidence. Give a complete, detailed answer in "
            "roughly 6-12 sentences when the context supports it. Cover the main points, "
            "relevant caveats, and important details instead of stopping after the first fact. "
            "Use short paragraphs or bullet points when that makes the answer clearer. "
            "If the documents do not state the answer directly, provide a cautious practical "
            "synthesis from the evidence and label it as an inference. Only say "
            "\"I don't know based on the provided documents\" when the retrieved chunks contain "
            "no useful information at all. Do not use outside knowledge.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


@lru_cache(maxsize=3)
def _load_index(collection: str) -> FAISS:
    if collection not in COLLECTIONS:
        raise ValueError(f"Unknown collection {collection!r}. Choose from {COLLECTIONS}.")
    path = INDEX_DIR / collection
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing FAISS index at {path}. Run `python src/embed_store.py` first."
        )
    return FAISS.load_local(
        str(path),
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def clear_index_cache() -> None:
    """Clear loaded indexes after an upload rebuilds one of them."""
    _load_index.cache_clear()


@lru_cache(maxsize=1)
def _get_embeddings() -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings()


@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    return ChatGroq(model=MODEL_NAME, temperature=0.1, max_tokens=MAX_TOKENS)


def _retrieval_count(index: FAISS) -> int:
    """Return the configured count, or a bounded count based on index size."""
    configured = os.getenv("RAG_TOP_K")
    if configured:
        try:
            requested = int(configured)
        except ValueError as error:
            raise ValueError("RAG_TOP_K must be a positive integer.") from error
        if requested < 1:
            raise ValueError("RAG_TOP_K must be a positive integer.")
        return min(requested, index.index.ntotal)

    chunk_count = index.index.ntotal
    return min(chunk_count, max(8, min(24, chunk_count // 4 + 1)))


def _keyword_overlap(question: str, text: str) -> int:
    words = {word.lower() for word in re.findall(r"[a-zA-Z0-9]+", question) if len(word) > 2}
    if not words:
        return 0
    text_words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    overlap = sum(1 for word in text_words if word in words)
    return overlap


def _fallback_answer(question: str, documents: list[Any]) -> str:
    if not documents:
        return "I don't know based on the provided documents"

    ranked = sorted(
        documents,
        key=lambda doc: _keyword_overlap(question, doc.page_content),
        reverse=True,
    )
    top_texts = [doc.page_content for doc in ranked[:2]]
    clean_text = " ".join(top_texts)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if s.strip()]
    answer_bits = []
    for sentence in sentences:
        if len(sentence) < 40:
            continue
        if _keyword_overlap(question, sentence) > 0:
            answer_bits.append(sentence)
    if not answer_bits:
        answer_bits = (
            [sentences[0]]
            if sentences
            else ["The retrieved documents contain no readable text for this question."]
        )

    answer = " ".join(answer_bits[:6])
    if len(answer) > 1200:
        answer = answer[:1197].rsplit(" ", 1)[0] + "..."
    return answer


def answer_question(question: str, collection: str) -> dict[str, Any]:
    """Return a grounded answer and the retrieved chunks used to produce it."""
    if not question.strip():
        raise ValueError("Question must not be empty.")

    index = _load_index(collection)
    retrieval_count = _retrieval_count(index)
    documents_with_scores = index.similarity_search_with_score(
        question, k=retrieval_count
    )
    documents = [document for document, _ in documents_with_scores]
    if not documents:
        raise RuntimeError("No relevant chunks were returned for this query.")

    sources = [
        {
            "text": document.page_content,
            "source_file": document.metadata.get("source_file", "unknown"),
            "chunk_id": document.metadata.get("chunk_id", "unknown"),
            "page_number": document.metadata.get("page_number"),
            "score": round(score, 4),
        }
        for document, score in documents_with_scores
    ]

    if not os.getenv("GROQ_API_KEY"):
        return {
            "answer": _fallback_answer(question, documents),
            "sources": sources,
            "retrieval_k": retrieval_count,
            "total_chunks": index.index.ntotal,
        }

    try:
        context = "\n\n".join(document.page_content for document in documents)
        response = (PROMPT | _get_llm()).invoke({"context": context, "question": question})
        answer = str(response.content).strip()
        if not answer:
            answer = _fallback_answer(question, documents)
    except (APIError, ValueError):
        answer = _fallback_answer(question, documents)

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_k": retrieval_count,
        "total_chunks": index.index.ntotal,
    }
