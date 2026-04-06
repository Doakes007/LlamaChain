import os
import re
import shutil
import uuid
import streamlit as st
import numpy as np

# Suppress Telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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
# ✅ UPDATED IMPORT
from src.rag import (
    build_retrieval_chain,
    ask_question,
    get_indexed_documents,
)
from src.core.loader import load_documents
from src.core.text_splitter import split_documents
from src.core.embed_store import embed_and_store


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="LlamaChain RAG", layout="wide")
st.title("LlamaChain — Local RAG Assistant")
st.caption("Powered by ChromaDB + Mistral + CLIP")


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
    "chain": None, # Added to track retriever state
}.items():
    st.session_state.setdefault(k, v)


# =====================================================
# EMBEDDINGS
# =====================================================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
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
        embedding_function=_embeddings,
    )


vectorstore = load_vectorstore(embeddings)

# Initialize chain in state if it doesn't exist
if st.session_state.chain is None:
    st.session_state.chain = build_retrieval_chain(vectorstore)


# =====================================================
# SIDEBAR: SYSTEM MANAGEMENT
# =====================================================
st.sidebar.title("System Management")

# ✅ FIX 1: CLEAR CACHE ON RESET
if st.sidebar.button("🗑️ Reset Database"):
    shutil.rmtree("./chroma_db", ignore_errors=True)
    st.cache_resource.clear()  # Purge stale embeddings/vectorstore connections
    st.session_state.clear()
    st.sidebar.success("Database cleared. Please refresh page.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("Upload Documents")

uploaded = st.sidebar.file_uploader(
    "Upload PDF or PPTX",
    type=["pdf", "pptx"],
    accept_multiple_files=True,
)

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if uploaded:
    saved_paths = []
    for f in uploaded:
        # ✅ FIX 2: PREVENT OVERWRITES WITH UUID
        unique_name = f"{uuid.uuid4().hex}_{f.name}"
        path = os.path.join(UPLOAD_DIR, unique_name)
        with open(path, "wb") as out:
            out.write(f.read())
        saved_paths.append(path)

    st.session_state.uploaded_paths = saved_paths
    st.sidebar.success(f"{len(saved_paths)} file(s) uploaded")


# =====================================================
# INDEX DOCUMENTS
# =====================================================
if st.sidebar.button("Index Documents"):
    if not st.session_state.uploaded_paths:
        st.sidebar.warning("Upload files first.")
    else:
        current_paths = tuple(sorted(
            os.path.realpath(p) for p in st.session_state.uploaded_paths
        ))

        if st.session_state.doc_paths == current_paths:
            st.sidebar.info("Already indexed. No changes detected.")
        else:
            st.session_state.busy = True

            with st.spinner("Indexing documents…"):
                try:
                    docs = load_documents(list(current_paths))
                    chunks = split_documents(docs)
                    embed_and_store(chunks, vectorstore)

                    # Update the retrieval chain after indexing new data
                    st.session_state.chain = build_retrieval_chain(vectorstore)

                    st.session_state.docs = docs
                    st.session_state.doc_paths = current_paths
                    st.session_state.base_summaries = get_base_summaries(current_paths)
                    st.session_state.summaries = {}

                    st.sidebar.success(f"Indexed {len(chunks)} chunks from {len(docs)} pages")

                except Exception as e:
                    st.sidebar.error(f"Indexing failed: {e}")
            st.session_state.busy = False


# =====================================================
# SUMMARY BUTTONS
# =====================================================
st.sidebar.divider()
st.sidebar.subheader("Summarization")

if st.sidebar.button("Combined Summary"):
    if st.session_state.base_summaries:
        with st.spinner("Generating combined summary…"):
            st.session_state.summaries["combined"] = combined_from_base(
                st.session_state.base_summaries
            )
    else:
        st.sidebar.warning("Index documents first.")


if st.sidebar.button("Per-Document Summary"):
    if st.session_state.base_summaries:
        with st.spinner("Generating per-document summaries…"):
            st.session_state.summaries["per_doc"] = per_doc_from_base(
                st.session_state.base_summaries
            )
    else:
        st.sidebar.warning("Index documents first.")


if st.sidebar.button("Topic-wise Summary"):
    if st.session_state.base_summaries:
        with st.spinner("Generating topic summaries…"):
            st.session_state.summaries["topic"] = topic_from_base(
                st.session_state.base_summaries
            )
    else:
        st.sidebar.warning("Index documents first.")


# =====================================================
# DOCUMENT COMPARISON
# =====================================================
st.sidebar.divider()
st.sidebar.subheader("Compare Documents")

indexed_docs = get_indexed_documents(vectorstore)

if not indexed_docs:
    st.sidebar.caption("Index documents first to enable comparison.")
else:
    selected_docs = st.sidebar.multiselect(
        "Select documents to compare (min 2):",
        options=indexed_docs,
        key="compare_select"
    )

    compare_aspect = st.sidebar.text_input(
        "Aspect to compare (optional):",
        placeholder="e.g. methodology, architecture",
        key="compare_aspect"
    )

    if st.sidebar.button("Compare Selected Documents"):
        if len(selected_docs) < 2:
            st.sidebar.warning("Select at least 2 documents.")
        else:
            with st.spinner(f"Comparing documents..."):
                # ✅ REUSE THE ELITE PIPELINE
                comparison_query = f"Compare these documents: {', '.join(selected_docs)}"
                if compare_aspect:
                    comparison_query += f" focusing specifically on {compare_aspect}"
                
                # This triggers reranking, CLIP, and grounding automatically
                comparison_result, _ = ask_question(st.session_state.chain, comparison_query)

            st.session_state.summaries["comparison"] = {
                "result": comparison_result,
                "docs": selected_docs
            }


# =====================================================
# DISPLAY RESULTS (COMPARISON & SUMMARIES)
# =====================================================
if "comparison" in st.session_state.summaries:
    comp = st.session_state.summaries["comparison"]
    st.subheader(f"Document Comparison: {' vs '.join(comp['docs'])}")
    st.markdown(comp["result"])
    st.divider()

if "combined" in st.session_state.summaries:
    st.subheader("Combined Summary")
    st.markdown(st.session_state.summaries["combined"])

if "per_doc" in st.session_state.summaries:
    st.subheader("Per-Document Summaries")
    for src, txt in st.session_state.summaries["per_doc"].items():
        with st.expander(src):
            st.markdown(txt)

if "topic" in st.session_state.summaries:
    st.subheader("Topic-wise Summaries")
    for src, txt in st.session_state.summaries["topic"].items():
        with st.expander(src):
            st.markdown(txt)


# =====================================================
# CHAT DISPLAY
# =====================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image_paths"):
            st.markdown("**Retrieved figures:**")
            cols = st.columns(min(len(msg["image_paths"]), 2))
            unique_paths = list(dict.fromkeys(msg["image_paths"]))
            for i, img_path in enumerate(unique_paths):
                if os.path.exists(img_path):
                    with cols[i % 2]:
                        st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)


# =====================================================
# CHAT INPUT
# =====================================================
if not st.session_state.busy:
    prompt = st.chat_input("Ask a question about your documents…")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents…"):
                try:
                    answer, image_paths = ask_question(st.session_state.chain, prompt)
                    st.markdown(answer, unsafe_allow_html=True)

                    if image_paths:
                        st.markdown("**Retrieved figures:**")
                        cols = st.columns(min(len(image_paths), 2))
                        unique_paths = list(dict.fromkeys(image_paths))
                        for i, img_path in enumerate(unique_paths):
                            if os.path.exists(img_path):
                                with cols[i % 2]:
                                    st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "image_paths": image_paths
                    })
                except Exception as e:
                    st.error(f"Error generating answer: {e}")


# =====================================================
# DOWNLOAD CHAT PDF
# =====================================================
st.sidebar.divider()
st.sidebar.subheader("Download Chat")

def generate_chat_pdf(messages):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [Paragraph("LlamaChain — Chat History", styles["Title"]), Spacer(1, 20)]

    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        safe_text = re.sub(r'!\[image\]\(.*?\)', '[image]', msg.get("content", ""))
        safe_text = safe_text.replace("<", "&lt;").replace(">", "&gt;")
        elements.append(Paragraph(f"<b>{role}:</b> {safe_text}", styles["Normal"]))
        elements.append(Spacer(1, 10))

    doc.build(elements)
    buffer.seek(0)
    return buffer

if st.session_state.messages:
    st.sidebar.download_button(
        label="Download Chat PDF",
        data=generate_chat_pdf(st.session_state.messages),
        file_name="LlamaChain_Chat.pdf",
        mime="application/pdf",
    )