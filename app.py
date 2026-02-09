# =====================================================
# 🔒 CRITICAL: Disable telemetry BEFORE any imports
# =====================================================
import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from llm import get_llm
from src.rag.rag_query import build_retrieval_chain, ask_question
from src.rag.summarizer import (
    summarize_documents,
    summarize_per_document,
    summarize_by_cluster
)
from src.core.loader import load_documents
from src.core.text_splitter import split_documents
from src.core.embed_store import embed_and_store


# =====================================================
# Streamlit Page Config
# =====================================================
st.set_page_config(
    page_title="RAG Assistant",
    layout="wide"
)

st.title("📚 Local RAG Assistant")
st.caption("Powered by Chroma + Llama-3 (Ollama, GPU)")


# =====================================================
# Session State Initialization
# =====================================================
if "docs" not in st.session_state:
    st.session_state.docs = None

if "doc_paths" not in st.session_state:
    st.session_state.doc_paths = None  # used for cache keys

if "summaries" not in st.session_state:
    st.session_state.summaries = {
        "combined": None,
        "per_doc": None,
        "topic": None
    }

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# Initialize LLM ONCE
# =====================================================
llm = get_llm()


# =====================================================
# Load embeddings
# =====================================================
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# =====================================================
# Load vector store
# =====================================================
vectorstore = Chroma(
    collection_name="LlamaChainDocs",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


# =====================================================
# Build RAG chain
# =====================================================
chain = build_retrieval_chain(vectorstore)


# =====================================================
# 🔹 CACHED SUMMARY FUNCTIONS (MOST IMPORTANT PART)
# =====================================================

@st.cache_data(show_spinner=False)
def cached_combined_summary(doc_paths):
    docs = load_documents(doc_paths)
    return summarize_documents(docs)


@st.cache_data(show_spinner=False)
def cached_per_doc_summary(doc_paths):
    docs = load_documents(doc_paths)
    return summarize_per_document(docs)


@st.cache_data(show_spinner=False)
def cached_topic_summary(doc_paths, topic_clusters):
    docs = load_documents(doc_paths)
    return summarize_by_cluster(docs, topic_clusters)


# =====================================================
# 📂 SIDEBAR — FILE UPLOAD + INDEXING
# =====================================================
st.sidebar.title("📂 Document Upload")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or PPT files",
    type=["pdf", "pptx"],
    accept_multiple_files=True
)

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

saved_files = []

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        saved_files.append(file_path)

    st.sidebar.success(f"✅ {len(saved_files)} file(s) saved successfully")


# =====================================================
# 📥 INDEX BUTTON
# =====================================================
if st.sidebar.button("📥 Index Uploaded Documents"):
    if not saved_files:
        st.sidebar.warning("No files uploaded yet.")
    else:
        with st.spinner("Indexing documents..."):
            st.session_state.docs = load_documents(saved_files)
            st.session_state.doc_paths = tuple(saved_files)

            chunks = split_documents(st.session_state.docs)
            embed_and_store(chunks)

            # Reset visible summaries (cache remains)
            st.session_state.summaries = {
                "combined": None,
                "per_doc": None,
                "topic": None
            }

        st.sidebar.success("✅ Documents indexed successfully!")


# =====================================================
# 🧾 SUMMARIZATION BUTTONS (CACHED)
# =====================================================
st.sidebar.divider()
st.sidebar.subheader("🧾 Summarization")

TOPIC_CLUSTERS = {
    "Overview": ["overview", "introduction", "summary"],
    "Key Concepts": ["concept", "definition", "principle"],
    "Process / Workflow": ["process", "workflow", "steps", "pipeline"],
    "Methods / Techniques": ["method", "approach", "technique"],
    "Challenges / Limitations": ["challenge", "limitation", "issue"],
    "Applications / Use Cases": ["application", "use case", "example"],
    "Conclusion / Future Work": ["conclusion", "future", "next"]
}

if st.sidebar.button("🧾 Summarize All Documents"):
    if not st.session_state.doc_paths:
        st.warning("Please index documents first.")
    else:
        with st.spinner("Generating combined summary…"):
            st.session_state.summaries["combined"] = cached_combined_summary(
                st.session_state.doc_paths
            )

if st.sidebar.button("🧾 Summarize Per Document"):
    if not st.session_state.doc_paths:
        st.warning("Please index documents first.")
    else:
        with st.spinner("Generating per-document summaries…"):
            st.session_state.summaries["per_doc"] = cached_per_doc_summary(
                st.session_state.doc_paths
            )

if st.sidebar.button("🧾 Summarize by Topic"):
    if not st.session_state.doc_paths:
        st.warning("Please index documents first.")
    else:
        with st.spinner("Generating topic-wise summaries…"):
            st.session_state.summaries["topic"] = cached_topic_summary(
                st.session_state.doc_paths,
                TOPIC_CLUSTERS
            )


# =====================================================
# 📄 DISPLAY STORED SUMMARIES
# =====================================================
if st.session_state.summaries["combined"]:
    st.subheader("📄 Combined Document Summary")
    st.markdown(st.session_state.summaries["combined"])
    st.divider()

if st.session_state.summaries["per_doc"]:
    st.subheader("📄 Per-Document Summaries")
    for filename, summary in st.session_state.summaries["per_doc"].items():
        st.markdown(f"### 📘 {os.path.basename(filename)}")
        st.markdown(summary)
        st.divider()

if st.session_state.summaries["topic"]:
    st.subheader("📚 Topic-wise Summaries")
    for topic, summary in st.session_state.summaries["topic"].items():
        st.markdown(f"### 🔹 {topic}")
        st.markdown(summary)
        st.divider()


# =====================================================
# 💬 CHAT UI
# =====================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask something from your documents...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking (GPU)…"):
            answer = ask_question(chain, prompt)
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
