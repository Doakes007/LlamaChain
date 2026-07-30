import os
import re
import shutil
import hashlib
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

# Suppress Telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch

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
# ONE-TIME WORKSPACE INITIALIZATION
# =====================================================

@st.cache_resource
def initialize_workspace():
    """
    Start every new application process with a clean workspace.

    Runs once for the lifetime of the Streamlit server process,
    not on every Streamlit script rerun.
    """

    cleanup_dirs = [
        "./chroma_db",
        "./uploaded_docs",
        "./extracted_images",
    ]

    for directory in cleanup_dirs:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"[STARTUP] Cleared: {directory}")
            except Exception as e:
                print(
                    f"[STARTUP WARNING] "
                    f"Could not clear {directory}: {e}"
                )

    # Recreate directories needed during the session
    os.makedirs("./uploaded_docs", exist_ok=True)
    os.makedirs("./extracted_images", exist_ok=True)

    print("[STARTUP] Fresh workspace initialized.")

    return True


initialize_workspace()


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
    "indexed_file_hashes": set(),
    "chain": None,
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
# HELPER FUNCTIONS
# =====================================================
def calculate_file_hash(file_bytes: bytes) -> str:
    """
    Generate a stable SHA-256 hash from file contents.

    Identical files always produce the same hash,
    regardless of filename or upload time.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def clean_filename(filename):
    """Remove UUID prefix and file extension from filename"""
    # Remove UUID prefix (everything before the first underscore)
    if '_' in filename:
        filename = filename.split('_', 1)[1]
    # Remove file extensions
    filename = filename.replace('.pdf', '').replace('.pptx', '')
    return filename


def clean_image_caption(caption):
    """
    Clean image captions for UI.
    Removes UUID prefixes and unnecessary file extensions.
    """

    filename = os.path.basename(caption)

    # Remove UUID prefix
    if "_" in filename:
        filename = filename.split("_", 1)[1]

    # Remove extension
    filename = os.path.splitext(filename)[0]

    # Replace underscores with spaces
    filename = filename.replace("_", " ")

    return filename


def remove_key_point_wording(text):
    """Remove 'key point' and 'key points' wording from text"""
    text = re.sub(r'[•*\-–]\s*Key points?:\s*', '- ', text, flags=re.IGNORECASE)
    text = re.sub(r'Key points?:\s*', '', text, flags=re.IGNORECASE)
    return text


def clean_comparison_text(comparison_text):
    """Remove sources and confidence metadata from comparison text"""
    lines = comparison_text.split('\n')
    filtered_lines = []
    for line in lines:
        # Skip sources and confidence lines
        if not line.strip().lower().startswith('sources:') and not line.strip().lower().startswith('confidence:'):
            filtered_lines.append(line)
    
    cleaned_text = '\n'.join(filtered_lines).strip()
    return cleaned_text


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

    uploaded_files = []

    for f in uploaded:

        # Read uploaded file once
        file_bytes = f.getvalue()

        # Stable identity based on CONTENT
        file_hash = calculate_file_hash(file_bytes)

        # Keep only a short hash for the internal filename
        short_hash = file_hash[:16]

        # Safe original filename
        original_name = os.path.basename(f.name)

        # Internal storage filename
        stored_name = f"{short_hash}_{original_name}"

        path = os.path.join(
            UPLOAD_DIR,
            stored_name,
        )

        # Avoid rewriting an identical uploaded file
        if not os.path.exists(path):
            with open(path, "wb") as out:
                out.write(file_bytes)

        uploaded_files.append(
            {
                "path": path,
                "hash": file_hash,
                "name": original_name,
            }
        )

    st.session_state.uploaded_paths = uploaded_files

    st.sidebar.success(
        f"{len(uploaded_files)} file(s) ready"
    )


# =====================================================
# INDEX DOCUMENTS
# =====================================================

if st.sidebar.button("Index Documents"):

    if not st.session_state.uploaded_paths:

        st.sidebar.warning("Upload files first.")

    else:

        # -------------------------------------------------
        # FIND FILES THAT HAVE NOT BEEN INDEXED YET
        # -------------------------------------------------

        new_files = [
            item
            for item in st.session_state.uploaded_paths
            if item["hash"]
            not in st.session_state.indexed_file_hashes
        ]

        if not new_files:

            st.sidebar.info(
                "All selected documents are already indexed."
            )

        else:

            st.session_state.busy = True

            with st.spinner("Indexing documents…"):

                try:

                    # -----------------------------------------
                    # PATHS FOR ONLY NEW DOCUMENTS
                    # -----------------------------------------

                    new_paths = [
                        os.path.realpath(item["path"])
                        for item in new_files
                    ]

                    # -----------------------------------------
                    # LOAD
                    # -----------------------------------------

                    docs = load_documents(new_paths)

                    # -----------------------------------------
                    # SPLIT
                    # -----------------------------------------

                    chunks = split_documents(docs)

                    # -----------------------------------------
                    # ADD TO EXISTING VECTOR STORE
                    # -----------------------------------------

                    embed_and_store(
                        chunks,
                        vectorstore,
                    )

                    # -----------------------------------------
                    # MARK FILES AS INDEXED
                    # -----------------------------------------

                    for item in new_files:
                        st.session_state.indexed_file_hashes.add(
                            item["hash"]
                        )

                    # -----------------------------------------
                    # KEEP ALL INDEXED PATHS
                    # -----------------------------------------

                    previous_paths = list(
                        st.session_state.doc_paths or ()
                    )

                    all_paths = list(
                        dict.fromkeys(
                            previous_paths + new_paths
                        )
                    )

                    st.session_state.doc_paths = tuple(
                        all_paths
                    )

                    # -----------------------------------------
                    # KEEP DOCUMENT OBJECTS
                    # -----------------------------------------

                    previous_docs = (
                        st.session_state.docs or []
                    )

                    st.session_state.docs = (
                        previous_docs + docs
                    )

                    # -----------------------------------------
                    # REBUILD RETRIEVER
                    # -----------------------------------------

                    st.session_state.chain = (
                        build_retrieval_chain(
                            vectorstore
                        )
                    )

                    # -----------------------------------------
                    # REBUILD SUMMARIZATION BASE
                    # FOR ALL CURRENT DOCUMENTS
                    # -----------------------------------------

                    st.session_state.base_summaries = (
                        get_base_summaries(
                            st.session_state.doc_paths
                        )
                    )

                    # Old summaries may no longer represent
                    # the complete document collection.
                    st.session_state.summaries = {}

                    # -----------------------------------------
                    # SUCCESS
                    # -----------------------------------------

                    names = [
                        item["name"]
                        for item in new_files
                    ]

                    st.sidebar.success(
                        f"Indexed {len(new_files)} new document(s): "
                        + ", ".join(names)
                    )

                except Exception as e:

                    st.sidebar.error(
                        f"Indexing failed: {e}"
                    )

                finally:

                    st.session_state.busy = False


# =====================================================
# SUMMARY BUTTONS
# =====================================================
st.sidebar.divider()
st.sidebar.subheader("Summarization")

if st.sidebar.button("Combined Summary"):
    if st.session_state.base_summaries:
        with st.spinner("Generating combined summary…"):
            summary = combined_from_base(st.session_state.base_summaries)
            st.session_state.summaries["combined"] = remove_key_point_wording(summary)
    else:
        st.sidebar.warning("Index documents first.")


if st.sidebar.button("Per-Document Summary"):
    if st.session_state.base_summaries:
        with st.spinner("Generating per-document summaries…"):
            per_doc_summary = per_doc_from_base(st.session_state.base_summaries)
            cleaned_summaries = {clean_filename(k): remove_key_point_wording(v) for k, v in per_doc_summary.items()}
            st.session_state.summaries["per_doc"] = cleaned_summaries
    else:
        st.sidebar.warning("Index documents first.")


if st.sidebar.button("Topic-wise Summary"):
    if st.session_state.base_summaries:
        with st.spinner("Generating topic summaries…"):
            topic_summary = topic_from_base(st.session_state.base_summaries)
            cleaned_topics = {clean_filename(k): remove_key_point_wording(v) for k, v in topic_summary.items()}
            st.session_state.summaries["topic"] = cleaned_topics
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
            with st.spinner("Comparing documents..."):

                # Clean internal/hash-prefixed filenames before
                # sending document names to the LLM
                cleaned_docs = [
                    clean_filename(d)
                    for d in selected_docs
                ]

                comparison_query = (
                    f"Compare these documents: "
                    f"{', '.join(cleaned_docs)}"
                )

                if compare_aspect:
                    comparison_query += (
                        f" focusing specifically on {compare_aspect}"
                    )

                comparison_result, _ = ask_question(
                    st.session_state.chain,
                    comparison_query,
                )

            st.session_state.summaries["comparison"] = {
                "result": comparison_result,
                "docs": selected_docs,
            }


# =====================================================
# DISPLAY RESULTS (COMPARISON & SUMMARIES)
# =====================================================
if "comparison" in st.session_state.summaries:
    comp = st.session_state.summaries["comparison"]
    # Clean filenames for display
    cleaned_docs = [clean_filename(doc) for doc in comp['docs']]
    st.subheader(f"Document Comparison: {' vs '.join(cleaned_docs)}")
    
    # Extract and clean comparison text (remove sources and confidence)
    cleaned_comparison = clean_comparison_text(comp["result"])
    
    # Display cleaned comparison
    st.markdown(cleaned_comparison)
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
                        clean_caption = clean_image_caption(os.path.basename(img_path))
                        st.image(img_path, caption=clean_caption, use_container_width=True)


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
                                    clean_caption = clean_image_caption(os.path.basename(img_path))
                                    st.image(img_path, caption=clean_caption, use_container_width=True)
                    
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

def generate_chat_pdf(messages, include_summaries=False, include_diagrams=False):
    """Generate comprehensive PDF with chat history, summaries, and diagrams"""

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()
    elements = []

    # =====================================================
    # TITLE
    # =====================================================

    title_style = styles["Heading1"]
    title_style.fontSize = 24
    title_style.textColor = colors.HexColor("#1f77b4")

    elements.append(
        Paragraph(
            "LlamaChain — Chat History",
            title_style,
        )
    )

    elements.append(Spacer(1, 0.3 * inch))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    elements.append(
        Paragraph(
            f"<i>Generated on {timestamp}</i>",
            styles["Normal"],
        )
    )

    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("_" * 80, styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    # =====================================================
    # CHAT HISTORY
    # =====================================================

    elements.append(
        Paragraph(
            "Chat Conversation",
            styles["Heading2"],
        )
    )

    elements.append(Spacer(1, 0.15 * inch))

    for msg in messages:

        role = (
            "User"
            if msg["role"] == "user"
            else "Assistant"
        )

        safe_text = re.sub(
            r'!\[image\]\(.*?\)',
            '[diagram]',
            msg.get("content", ""),
        )

        safe_text = (
            safe_text
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        safe_text = safe_text[:1000]

        elements.append(
            Paragraph(
                f"<b>{role}:</b>",
                styles["Heading3"],
            )
        )

        elements.append(
            Paragraph(
                safe_text,
                styles["Normal"],
            )
        )

        elements.append(Spacer(1, 0.1 * inch))

        # =====================================================
        # ADD RETRIEVED IMAGES
        # =====================================================

        if include_diagrams and msg.get("image_paths"):

            unique_paths = list(
                dict.fromkeys(
                    msg.get("image_paths", [])
                )
            )

            for img_path in unique_paths:

                if not os.path.exists(img_path):
                    continue

                try:

                    img = RLImage(
                        img_path,
                        width=3.5 * inch,
                        height=2.5 * inch,
                    )

                    elements.append(img)

                    # ✅ CLEAN CAPTION
                    caption = clean_image_caption(
                        os.path.basename(img_path)
                    )

                    elements.append(
                        Paragraph(
                            f"<i>Figure: {caption}</i>",
                            styles["Normal"],
                        )
                    )

                    elements.append(Spacer(1, 0.15 * inch))

                except Exception:

                    caption = clean_image_caption(
                        os.path.basename(img_path)
                    )

                    elements.append(
                        Paragraph(
                            f"<i>[Diagram: {caption}]</i>",
                            styles["Normal"],
                        )
                    )

                    elements.append(Spacer(1, 0.1 * inch))

    # =====================================================
    # SUMMARIES
    # =====================================================

    if include_summaries and st.session_state.summaries:

        elements.append(PageBreak())

        elements.append(
            Paragraph(
                "Summaries & Analysis",
                styles["Heading1"],
            )
        )

        elements.append(Spacer(1, 0.2 * inch))

        if "combined" in st.session_state.summaries:

            elements.append(
                Paragraph(
                    "Combined Summary",
                    styles["Heading2"],
                )
            )

            summary_text = (
                st.session_state.summaries["combined"][:2000]
            )

            elements.append(
                Paragraph(
                    summary_text,
                    styles["Normal"],
                )
            )

            elements.append(Spacer(1, 0.15 * inch))

        if "per_doc" in st.session_state.summaries:

            elements.append(
                Paragraph(
                    "Per-Document Summaries",
                    styles["Heading2"],
                )
            )

            for src, txt in st.session_state.summaries["per_doc"].items():

                elements.append(
                    Paragraph(
                        f"<b>{src}</b>",
                        styles["Heading3"],
                    )
                )

                elements.append(
                    Paragraph(
                        txt[:1000],
                        styles["Normal"],
                    )
                )

                elements.append(Spacer(1, 0.1 * inch))

        if "topic" in st.session_state.summaries:

            elements.append(
                Paragraph(
                    "Topic-wise Summaries",
                    styles["Heading2"],
                )
            )

            for src, txt in st.session_state.summaries["topic"].items():

                elements.append(
                    Paragraph(
                        f"<b>{src}</b>",
                        styles["Heading3"],
                    )
                )

                elements.append(
                    Paragraph(
                        txt[:1000],
                        styles["Normal"],
                    )
                )

                elements.append(Spacer(1, 0.1 * inch))

    # =====================================================
    # BUILD PDF
    # =====================================================

    doc.build(elements)

    buffer.seek(0)

    return buffer


# PDF download options
col1, col2, col3 = st.sidebar.columns(3)

with col1:
    include_summaries = st.checkbox("Include Summaries", value=False, key="pdf_summaries")

with col2:
    include_diagrams = st.checkbox("Include Diagrams", value=False, key="pdf_diagrams")

if st.session_state.messages:
    st.sidebar.download_button(
        label="📥 Download Chat PDF",
        data=generate_chat_pdf(
            st.session_state.messages,
            include_summaries=include_summaries,
            include_diagrams=include_diagrams
        ),
        file_name=f"LlamaChain_Chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
    )