# app.py

import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# PDF export imports
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

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
st.caption("Powered by ChromaDB + Llama-3")


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
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    collection_name="LlamaChainDocs",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

if "db_reset" not in st.session_state:
    try:
        vectorstore.delete_collection()
    except:
        pass
    st.session_state.db_reset = True

chain = build_retrieval_chain(vectorstore)


# =====================================================
# UPLOAD
# =====================================================
st.sidebar.title("📂 Upload Document")

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
# INDEX + SUMMARY
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
                split_documents(st.session_state.docs),
                vectorstore
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
# PDF GENERATION (STRUCTURED)
# =====================================================
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

    user_style = ParagraphStyle(
        "UserStyle",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#1f4e79"),
        spaceAfter=6,
    )

    assistant_style = ParagraphStyle(
        "AssistantStyle",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        spaceAfter=6,
    )

    source_style = ParagraphStyle(
        "SourceStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        leftIndent=15,
        textColor=colors.HexColor("#555555"),
    )

    elements = []
    elements.append(
        Paragraph("LlamaChain - Chat history", styles["Title"])
    )
    elements.append(Spacer(1, 20))

    for msg in messages:

        if msg["role"] == "user":
            elements.append(
                Paragraph(f"<b>🧑 User:</b> {msg['content']}", user_style)
            )

        else:
            content = msg["content"]

            if "### 📌 Sources" in content:
                answer, sources = content.split("### 📌 Sources", 1)

                elements.append(
                    Paragraph(
                        f"<b>🤖 Assistant:</b><br/>{answer.strip()}",
                        assistant_style
                    )
                )

                elements.append(
                    Paragraph("<b>📌 Sources:</b>", assistant_style)
                )

                for line in sources.split("\n"):
                    line = line.strip()
                    if line:
                        elements.append(
                            Paragraph(line, source_style)
                        )

            else:
                elements.append(
                    Paragraph(
                        f"<b>🤖 Assistant:</b><br/>{content}",
                        assistant_style
                    )
                )

        elements.append(Spacer(1, 8))

    doc.build(elements)
    buffer.seek(0)
    return buffer


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
            ans = ask_question(chain, prompt)
            st.markdown(ans)

        st.session_state.messages.append(
            {"role": "assistant", "content": ans}
        )


# =====================================================
# DOWNLOAD CHAT PDF
# =====================================================
st.sidebar.divider()
st.sidebar.subheader("💾 Download Chat")

if st.session_state.messages:
    pdf_file = generate_chat_pdf(st.session_state.messages)

    st.sidebar.download_button(
        label="⬇️ Download Chat PDF",
        data=pdf_file,
        file_name="LlamaChain - ChatHistory.pdf",
        mime="application/pdf",
    )