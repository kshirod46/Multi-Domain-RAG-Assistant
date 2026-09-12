"""Streamlit chat interface for the collection-specific RAG chain."""

import os
import re
from pathlib import Path

import streamlit as st
from groq import APIError

from src.embed_store import COLLECTIONS, build_index, get_embeddings
from src.ingest import run as run_ingest
from src.rag_chain import answer_question, clear_index_cache

st.set_page_config(page_title="Multi-Domain RAG", page_icon="📚")
st.title("Multi-Domain RAG")
st.caption("Ask questions grounded in one selected collection.")

if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False

controls = st.sidebar if st.session_state.setup_complete else st
collection = controls.selectbox(
    "Knowledge base",
    COLLECTIONS,
    format_func=lambda name: name.replace("_", " ").title(),
    key="knowledge_base",
)
controls.info(
    "Choose a knowledge base above, then ask a question below. "
    "Answers use only the documents indexed for that collection."
)

if not os.getenv("GROQ_API_KEY"):
    st.info(
        "GROQ_API_KEY is not configured. Add it to .env for full generated answers. "
        "The app will use the grounded retrieved chunks until then."
    )

with controls.expander("Upload documents to this knowledge base", expanded=not st.session_state.setup_complete):
    controls.caption("Supported formats: TXT and PDF. Uploaded files are stored locally.")
    uploads = controls.file_uploader(
        "Select one or more documents",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        key=f"uploads_{collection}",
    )
    if controls.button("Add documents and rebuild index", disabled=not uploads):
        data_dir = Path(__file__).resolve().parent / "data" / collection
        data_dir.mkdir(parents=True, exist_ok=True)
        for upload in uploads:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(upload.name).name)
            if not safe_name:
                st.error(f"Skipped invalid filename: {upload.name}")
                continue
            (data_dir / safe_name).write_bytes(upload.getvalue())

        with st.spinner("Processing documents and rebuilding this index..."):
            run_ingest()
            build_index(collection, get_embeddings())
            clear_index_cache()
        st.session_state.setup_complete = True
        st.session_state.upload_message = (
            f"Added {len(uploads)} document(s) to "
            f"{collection.replace('_', ' ')}. Upload controls are now in the sidebar."
        )
        st.rerun()

if st.session_state.get("upload_message"):
    st.success(st.session_state.pop("upload_message"))

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the selected collection")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            result = answer_question(question, collection)
        except (FileNotFoundError, RuntimeError, ValueError, APIError) as error:
            st.error(f"Could not answer the question: {error}")
        else:
            st.markdown(result["answer"])
            with st.expander("View retrieved chunks", expanded=False):
                st.caption(
                    f"Retrieved {result.get('retrieval_k', len(result.get('sources', [])))} "
                    f"of {result.get('total_chunks', 'unknown')} indexed chunks for this question."
                )
                for index, source in enumerate(result.get("sources", []), start=1):
                    st.markdown(f"**Chunk {index}** — `{source.get('source_file', 'unknown')}`")
                    page = source.get("page_number")
                    page_label = f" | page: {page}" if page else ""
                    st.caption(
                        f"Chunk ID: {source.get('chunk_id', 'unknown')}{page_label} "
                        f"| score: {source.get('score', 'n/a')}"
                    )
                    st.write(source.get("text", ""))
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                }
            )
