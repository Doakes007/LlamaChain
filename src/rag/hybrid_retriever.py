from typing import List, Any, Optional
from langchain.schema import Document, BaseRetriever
from langchain_community.retrievers import BM25Retriever
from pydantic import model_validator


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever: MMR vector search (ChromaDB) + BM25 keyword search.
    Compatible with LangChain 0.3.x / Pydantic v2.
    """

    # Pydantic v2: all fields declared at class level with types
    vectorstore: Any
    vector_retriever: Any = None
    bm25_retriever: Any = None
    k: int = 15
    _doc_count: int = 0

    # model_validator replaces __init__ for post-construction setup
    @model_validator(mode="after")
    def _setup(self) -> "HybridRetriever":
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.k,
                "fetch_k": 80,
                "lambda_mult": 0.55,
            },
        )
        self._rebuild_bm25()
        return self

    # --------------------------------------------------
    def _load_all_documents(self) -> List[Document]:
        results = self.vectorstore.get()
        docs = []
        for content, metadata in zip(
            results.get("documents", []),
            results.get("metadatas", []),
        ):
            if content and content.strip():
                docs.append(Document(page_content=content, metadata=metadata or {}))
        return docs

    def _rebuild_bm25(self):
        docs = self._load_all_documents()
        self._doc_count = len(docs)
        if docs:
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = self.k
        else:
            self.bm25_retriever = None

    def _maybe_rebuild_bm25(self):
        """Rebuild BM25 index if new documents were added since last build."""
        try:
            results = self.vectorstore.get()
            current_count = len(results.get("documents", []))
            if current_count != self._doc_count:
                print(f"BM25: rebuilding ({self._doc_count} → {current_count} docs)")
                self._rebuild_bm25()
        except Exception as e:
            print(f"BM25 rebuild check failed: {e}")

    def _merge_results(self, vector_docs, keyword_docs) -> List[Document]:
        seen = set()
        merged = []
        for doc in vector_docs:
            key = doc.page_content[:120]
            if key not in seen:
                merged.append(doc)
                seen.add(key)
        for doc in keyword_docs:
            key = doc.page_content[:120]
            if key not in seen:
                merged.append(doc)
                seen.add(key)
        return merged[: self.k]

    # LangChain 0.3.x requires _get_relevant_documents
    def _get_relevant_documents(self, query: str) -> List[Document]:
        self._maybe_rebuild_bm25()

        try:
            vector_docs = self.vector_retriever.invoke(query)
        except Exception as e:
            print(f"Vector retrieval failed: {e}")
            vector_docs = []

        if self.bm25_retriever is None:
            return vector_docs[: self.k]

        try:
            keyword_docs = self.bm25_retriever.invoke(query)
        except Exception as e:
            print(f"BM25 retrieval failed: {e}")
            keyword_docs = []

        return self._merge_results(vector_docs, keyword_docs)