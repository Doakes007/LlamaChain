import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from llm import get_llm  # ✅ Central Ollama LLM (GPU)
from src.rag.rag_query import build_retrieval_chain, ask_question
from src.rag.summarizer import summarize_documents
from src.core.loader import load_documents
from src.core.text_splitter import split_documents
from src.core.embed_store import embed_and_store


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="RAG Assistant",
    layout="wide"
)

st.title("📚 Local RAG Assistant")
st.caption("Powered by Chroma + Llama-3 (Ollama, GPU)")


# -----------------------------
# Initialize LLM ONCE (GPU)
# -----------------------------
# NOTE:
# - This ensures Ollama is warmed up
# - Actual usage happens inside summarizer / RAG modules
llm = get_llm()


# -----------------------------
# Load embeddings (MUST match indexing)
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# -----------------------------
# Load existing vector store
# -----------------------------
vectorstore = Chroma(
    collection_name="LlamaChainDocs",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


# -----------------------------
# Build RAG chain ONCE
# -----------------------------
chain = build_retrieval_chain(vectorstore)


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


# -----------------------------
# 📥 INDEX BUTTON
# -----------------------------
if st.sidebar.button("📥 Index Uploaded Documents"):
    if not saved_files:
        st.sidebar.warning("No files uploaded yet.")
    else:
        with st.spinner("Indexing documents..."):
            docs = load_documents(saved_files)

            # 🔒 Store docs in session (used for summary)
            st.session_state.docs = docs

            chunks = split_documents(docs)
            embed_and_store(chunks)

        st.sidebar.success("✅ Documents indexed successfully!")


# =====================================================
# 🧾 SUMMARIZATION UI
# =====================================================
st.sidebar.divider()
st.sidebar.subheader("🧾 Summarization")

if st.sidebar.button("🧾 Summarize All Documents"):
    if "docs" not in st.session_state or not st.session_state.docs:
        st.sidebar.warning("Please index documents first.")
    else:
        with st.spinner("Generating summary (GPU)…"):
            summary = summarize_documents(st.session_state.docs)

        st.subheader("📄 Combined Document Summary")
        st.markdown(summary)


# =====================================================
# 💬 CHAT UI
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask something from your documents...")

if prompt:
    # User message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking (GPU)…"):
            answer = ask_question(chain, prompt)
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
