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
# ✅ UPDATED IMPORT
from src.rag import (
    build_retrieval_chain,
    ask_question,
    compare_documents,
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
        default=indexed_docs[:2] if len(indexed_docs) >= 2 else indexed_docs,
        key="compare_select"
    )

    compare_aspect = st.sidebar.text_input(
        "Aspect to compare (optional):",
        placeholder="e.g. methodology, accuracy, architecture",
        key="compare_aspect"
    )

    if st.sidebar.button("Compare Selected Documents"):
        if len(selected_docs) < 2:
            st.sidebar.warning("Select at least 2 documents.")
        else:
            with st.spinner(f"Comparing {', '.join(selected_docs)}..."):
                comparison_result = compare_documents(
                    vectorstore,
                    selected_docs,
                    aspect=compare_aspect.strip() if compare_aspect else ""
                )

            st.session_state.summaries["comparison"] = {
                "result": comparison_result,
                "docs": selected_docs
            }


if "comparison" in st.session_state.summaries:
    comp = st.session_state.summaries["comparison"]
    st.subheader(f"Document Comparison: {' vs '.join(comp['docs'])}")
    st.markdown(comp["result"])
    st.divider()


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
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            # ✅ UPDATED: Handle answer and image_paths from assistant
            st.markdown(msg["content"])
            if "image_paths" in msg and msg["image_paths"]:
                st.markdown("**Retrieved figures:**")
                cols = st.columns(min(len(msg["image_paths"]), 2))
                for i, img_path in enumerate(msg["image_paths"]):
                    img_path = img_path.strip()
                    abs_path = img_path if os.path.isabs(img_path) else os.path.abspath(img_path)

                    if os.path.exists(abs_path):
                        with cols[i % 2]:
                            st.image(
                                abs_path,
                                caption=os.path.basename(abs_path),
                                use_container_width=True
                            )


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
            # ✅ UPDATED: Handle return tuple (answer, image_paths)
            with st.spinner("Analyzing documents… (15–20s)"):
                answer, image_paths = ask_question(chain, prompt)

            st.markdown(answer, unsafe_allow_html=True)

            # ✅ UPDATED: Render images from pipeline
            if image_paths:
                st.markdown("**Retrieved figures:**")
                cols = st.columns(min(len(image_paths), 2))
                for i, img_path in enumerate(image_paths):
                    img_path = img_path.strip()
                    abs_path = img_path if os.path.isabs(img_path) else os.path.abspath(img_path)

                    if os.path.exists(abs_path):
                        with cols[i % 2]:
                            st.image(
                                abs_path,
                                caption=os.path.basename(abs_path),
                                use_container_width=True
                            )
                    else:
                        st.caption(f"Image not found: {abs_path}")

        # Store both answer and image paths in session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "image_paths": image_paths
        })


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
        content = msg.get("content", "")

        safe_text = re.sub(r'!\[image\]\(.*?\)', '[image]', content)
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