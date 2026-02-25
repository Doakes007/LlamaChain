# app.py

import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.rag.summarizer import (
    get_base_summaries,
    combined_from_base,
    per_doc_from_base,
    topic_from_base,
)
from src.rag.rag_query import build_retrieval_chain, ask_question
from src.core.loader import load_documents
from src.core.text_splitter import split_documents
from src.core.embed_store import embed_and_store


st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("📚 Local RAG Assistant")
st.caption("Powered by Chroma + Llama-3 (Ollama)")


# =====================================================
# SESSION STATE
# =====================================================
for k, v in {
    "docs": None,
    "doc_paths": None,
    "base_summaries": None,
    "summaries": {},
    "messages": [],
    "busy": False,
}.items():
    st.session_state.setdefault(k, v)


# =====================================================
# VECTOR STORE
# =====================================================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="LlamaChainDocs",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
chain = build_retrieval_chain(vectorstore)


# =====================================================
# UPLOAD
# =====================================================
st.sidebar.title("📂 Document Upload")

uploaded = st.sidebar.file_uploader(
    "Upload PDF / PPT",
    type=["pdf", "pptx"],
    accept_multiple_files=True
)

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

saved = []
if uploaded:
    for f in uploaded:
        p = os.path.join(UPLOAD_DIR, f.name)
        with open(p, "wb") as out:
            out.write(f.read())
        saved.append(p)
    st.sidebar.success(f"✅ {len(saved)} files uploaded")


# =====================================================
# INDEX + BASE SUMMARY
# =====================================================
if st.sidebar.button("📥 Index Documents"):
    if not saved:
        st.sidebar.warning("Upload files first.")
    else:
        st.session_state.busy = True
        with st.spinner("Indexing & summarizing…"):
            st.session_state.docs = load_documents(saved)
            st.session_state.doc_paths = tuple(saved)

            embed_and_store(
                split_documents(st.session_state.docs)
            )

            st.session_state.base_summaries = get_base_summaries(
                st.session_state.doc_paths
            )
            st.session_state.summaries = {}

        st.session_state.busy = False
        st.sidebar.success("✅ Indexed & summarized")


# =====================================================
# SUMMARY BUTTONS
# =====================================================
st.sidebar.divider()

if st.sidebar.button("🧾 Combined Summary"):
    st.session_state.summaries["combined"] = combined_from_base(
        st.session_state.base_summaries
    )

if st.sidebar.button("🧾 Per-Document Summary"):
    st.session_state.summaries["per_doc"] = per_doc_from_base(
        st.session_state.base_summaries
    )

if st.sidebar.button("🧾 Topic-wise Summary"):
    st.session_state.summaries["topic"] = topic_from_base(
        st.session_state.base_summaries
    )


# =====================================================
# DISPLAY
# =====================================================
if "combined" in st.session_state.summaries:
    st.subheader("📄 Combined Summary")
    st.markdown(st.session_state.summaries["combined"])

if "per_doc" in st.session_state.summaries:
    st.subheader("📄 Per-Document Summaries")
    for src, txt in st.session_state.summaries["per_doc"].items():
        st.markdown(f"### 📘 {src}")
        st.markdown(txt)

if "topic" in st.session_state.summaries:
    st.subheader("📚 Topic-wise Summaries")
    for src, txt in st.session_state.summaries["topic"].items():
        st.markdown(f"### 🔹 {src}")
        st.markdown(txt)


# =====================================================
# CHAT
# =====================================================

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.busy:
    prompt = st.chat_input("Ask something from your documents…")

    if prompt:
        # Save + show user message
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            ans = ask_question(chain, prompt)
            st.markdown(ans)

        st.session_state.messages.append(
            {"role": "assistant", "content": ans}
        )