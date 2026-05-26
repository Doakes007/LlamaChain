# LlamaChain – AI Knowledge Assistant for Multi-Document Analysis

LlamaChain is a locally deployed Retrieval-Augmented Generation (RAG) system designed for intelligent document understanding, summarization, comparison, and conversational question answering across multiple documents.

Supports:

- PDF documents
- PowerPoint presentations
- Text extraction
- Image retrieval
- Multi-document comparison
- Structured summarization

---

## Architecture

<img width="504" height="762" alt="image" src="https://github.com/user-attachments/assets/727b5bd7-4de7-4298-b12a-b37d8d3b2a01" />


---

## RAG Pipeline

### Document Processing

Supported formats:

- PDF
- PPT

Extracted information:

- Text
- Tables
- Images
- Metadata

---

## Chunking Strategy

Implemented:

- Hierarchical chunking
- Overlapping chunks
- Metadata preservation

Metadata stored:

- Document name
- Page number
- Content type

---

## Retrieval Pipeline

### Semantic Search

Uses sentence embeddings for semantic understanding

### Keyword Retrieval

Improves domain-specific retrieval accuracy

### Reranking

Improves relevance of retrieved context

---

## ChromaDB Integration

Stored:

- Embeddings
- Metadata
- Source information

Supports:

- Similarity search
- Source attribution

---

## Local LLM

Model serving:

- Ollama

Models:

- Llama
- Mistral

Benefits:

- Local execution
- Privacy preservation
- No external API dependency

---

## Features

- Conversational Question Answering
- Multi-document comparison
- Topic-wise summarization
- Combined summaries
- Source attribution
- PDF report export
- Retrieval of text and image content

---

## Screenshots

- Upload Interface
- Chat Interface
- Summary Output
- Multi-document Comparison
- Retrieval Output
- Export PDF Report

---

## Demo

Example workflow:

Upload PDF → Ask Question → Retrieve Sources → Generate Grounded Response

---

## How To Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Tech Stack

- Python
- LangChain
- ChromaDB
- Sentence Transformers
- Ollama
- Streamlit
- PyMuPDF
- Unstructured

---

## Future Improvements

- Faster retrieval optimization
- More document formats
- Better reranking models
- Advanced evaluation metrics

---

## Contributors

- Chirag N
- Rhiya Giridhara Bhat
- Tarun G P
- Ghana Shyam D

