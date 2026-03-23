import os

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

import streamlit as st
import re

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
from src.rag.rag_query import build_retrieval_chain, ask_question
from src.core.loader import load_documents
from src.core.text_splitter import split_documents
from src.core.embed_store import embed_and_store


# =====================================================
# RESPONSE RENDERER
# =====================================================
def render_response(answer):
    image_paths = re.findall(r'!\[image\]\((.*?)\)', answer)
    clean_text = re.sub(r'!\[image\]\(.*?\)\n?', '', answer).strip()

    st.markdown(clean_text, unsafe_allow_html=True)

    if image_paths:
        st.markdown("**Retrieved figures:**")
        cols = st.columns(min(len(image_paths), 2))
        for i, img_path in enumerate(image_paths):
            img_path = img_path.strip()
            abs_path = img_path if os.path.isabs(img_path) else os.path.abspath(img_path)
            if os.path.exists(abs_path):
                with cols[i % 2]:
                    st.image(abs_path, caption="Retrieved figure", width=650)
            else:
                st.caption(f"Image not found: {abs_path}")


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="LlamaChain RAG", layout="wide")
st.title("LlamaChain — Local RAG Assistant")
st.caption("Powered by ChromaDB + Phi3 + CLIP")


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
chain = build_retrieval_chain(vectorstore)


# =====================================================
# FILE UPLOAD
# =====================================================
st.sidebar.title("Upload Documents")
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
        path = os.path.join(UPLOAD_DIR, f.name)
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

                    st.session_state.docs = docs
                    st.session_state.doc_paths = current_paths
                    st.session_state.base_summaries = get_base_summaries(current_paths)
                    st.session_state.summaries = {}

                    st.sidebar.success(f"Indexed {len(chunks)} chunks from {len(docs)} pages")
                except Exception as e:
                    st.sidebar.error(f"Indexing failed: {e}")
                    print(f"Indexing error: {e}")

            st.session_state.busy = False


# =====================================================
# SUMMARY BUTTONS
# =====================================================
st.sidebar.divider()
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
# DISPLAY SUMMARIES
# =====================================================
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
        render_response(msg["content"])


# =====================================================
# CHAT INPUT
# =====================================================
if not st.session_state.busy:
    prompt = st.chat_input("Ask a question about your documents…")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents… (this may take 30-40 seconds)"):
                answer = ask_question(chain, prompt)
            render_response(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


# =====================================================
# DOWNLOAD CHAT PDF
# =====================================================
st.sidebar.divider()
st.sidebar.subheader("Download Chat")


def generate_chat_pdf(messages):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("LlamaChain — Chat History", styles["Title"]),
        Spacer(1, 20),
    ]
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        safe_text = re.sub(r'!\[image\]\(.*?\)', '[image]', msg["content"])
        safe_text = safe_text.replace("<", "&lt;").replace(">", "&gt;")
        elements.append(Paragraph(f"<b>{role}:</b> {safe_text}", styles["Normal"]))
        elements.append(Spacer(1, 10))
    doc.build(elements)
    buffer.seek(0)
    return buffer


if st.session_state.messages:
    pdf_file = generate_chat_pdf(st.session_state.messages)
    st.sidebar.download_button(
        label="Download Chat PDF",
        data=pdf_file,
        file_name="LlamaChain_Chat.pdf",
        mime="application/pdf",
    )