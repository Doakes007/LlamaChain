# =====================================================
# ENVIRONMENT SETTINGS
# =====================================================
import os

# Disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# PDF export
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

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


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("📚 LlamaChain - Local RAG model")
st.caption("Powered by ChromaDB + Mistral")


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
    "uploaded_paths": [],
}.items():
    st.session_state.setdefault(k, v)


# =====================================================
# LOAD EMBEDDINGS (GPU)
# =====================================================
@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_kwargs={
            "device": "cuda"   # 🔥 embeddings now use GPU
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32    # faster embedding batches
        },
    )


embeddings = load_embeddings()


# =====================================================
# VECTOR STORE
# =====================================================
@st.cache_resource
def load_vectorstore(_embeddings):

    return Chroma(
        collection_name="LlamaChainDocs",
        persist_directory="./chroma_db",
        embedding_function=_embeddings
    )


vectorstore = load_vectorstore(embeddings)

chain = build_retrieval_chain(vectorstore)


# =====================================================
# FILE UPLOAD
# =====================================================
st.sidebar.title("📂 Upload Document")

uploaded = st.sidebar.file_uploader(
    "Upload PDF / PPT",
    type=["pdf", "pptx"],
    accept_multiple_files=True
)

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if uploaded:

    saved_paths = []

    for f in uploaded:

        path = os.path.join(UPLOAD_DIR, f.name)

        with open(path, "wb") as out:
            out.write(f.read())

        saved_paths.append(path)

    st.session_state.uploaded_paths = saved_paths

    st.sidebar.success(f"✅ {len(saved_paths)} files uploaded")


# =====================================================
# INDEX DOCUMENTS
# =====================================================
if st.sidebar.button("📥 Index Documents"):

    if not st.session_state.uploaded_paths:
        st.sidebar.warning("Upload files first.")

    else:

        st.session_state.busy = True

        with st.spinner("Indexing documents..."):

            st.session_state.docs = load_documents(
                st.session_state.uploaded_paths
            )

            st.session_state.doc_paths = tuple(
                st.session_state.uploaded_paths
            )

            chunks = split_documents(st.session_state.docs)

            embed_and_store(chunks, vectorstore)

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

    if st.session_state.base_summaries:

        st.session_state.summaries["combined"] = combined_from_base(
            st.session_state.base_summaries
        )


if st.sidebar.button("🧾 Per-Document Summary"):

    if st.session_state.base_summaries:

        st.session_state.summaries["per_doc"] = per_doc_from_base(
            st.session_state.base_summaries
        )


if st.sidebar.button("🧾 Topic-wise Summary"):

    if st.session_state.base_summaries:

        st.session_state.summaries["topic"] = topic_from_base(
            st.session_state.base_summaries
        )


# =====================================================
# DISPLAY SUMMARIES
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
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if not st.session_state.busy:

    prompt = st.chat_input("Ask something from your documents…")

    if prompt:

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            answer = ask_question(chain, prompt)
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


# =====================================================
# DOWNLOAD CHAT PDF
# =====================================================
st.sidebar.divider()
st.sidebar.subheader("💾 Download Chat")


def generate_chat_pdf(messages):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph("LlamaChain - Chat history", styles["Title"])
    )

    elements.append(Spacer(1, 20))

    for msg in messages:

        role = "User" if msg["role"] == "user" else "Assistant"

        safe_text = msg["content"].replace("<", "&lt;").replace(">", "&gt;")

        elements.append(
            Paragraph(f"<b>{role}:</b> {safe_text}", styles["Normal"])
        )

        elements.append(Spacer(1, 10))

    doc.build(elements)

    buffer.seek(0)

    return buffer


if st.session_state.messages:

    pdf_file = generate_chat_pdf(st.session_state.messages)

    st.sidebar.download_button(
        label="⬇️ Download Chat PDF",
        data=pdf_file,
        file_name="LlamaChain_ChatHistory.pdf",
        mime="application/pdf",
    )