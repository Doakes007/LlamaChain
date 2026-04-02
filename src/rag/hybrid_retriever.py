from typing import List, Any
from langchain.schema import Document, BaseRetriever
from langchain_community.retrievers import BM25Retriever
from pydantic import model_validator


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever:
    - Dense vector search (MMR via ChromaDB)
    - BM25 keyword search
    Optimized for speed + relevance.
    """

    # ===================================================
    # CONFIG (OPTIMIZED)
    # ===================================================
    vectorstore: Any
    vector_retriever: Any = None
    bm25_retriever: Any = None

    k: int = 8            # 🔥 reduced from 15 → faster
    _doc_count: int = 0


    # ===================================================
    # INITIAL SETUP
    # ===================================================
    @model_validator(mode="after")
    def _setup(self) -> "HybridRetriever":

        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.k,
                "fetch_k": 30,        # 🔥 reduced from 80 → major speed gain
                "lambda_mult": 0.55,
            },
        )

        self._rebuild_bm25()
        return self


    # ===================================================
    # LOAD DOCUMENTS FROM VECTORSTORE
    # ===================================================
    def _load_all_documents(self) -> List[Document]:
        results = self.vectorstore.get()

        docs = []

        for content, metadata in zip(
            results.get("documents", []),
            results.get("metadatas", []),
        ):
            if content and content.strip():
                docs.append(
                    Document(
                        page_content=content,
                        metadata=metadata or {}
                    )
                )

        return docs


    # ===================================================
    # BUILD BM25 INDEX
    # ===================================================
    def _rebuild_bm25(self):
        docs = self._load_all_documents()

        self._doc_count = len(docs)

        if docs:
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = self.k
        else:
            self.bm25_retriever = None


    # ===================================================
    # AUTO-REBUILD IF NEW DOCS ADDED
    # ===================================================
    def _maybe_rebuild_bm25(self):
        try:
            results = self.vectorstore.get()
            current_count = len(results.get("documents", []))

            if current_count != self._doc_count:
                print(f"BM25: rebuilding ({self._doc_count} → {current_count})")
                self._rebuild_bm25()

        except Exception as e:
            print(f"BM25 rebuild check failed: {e}")


    # ===================================================
    # MERGE VECTOR + BM25 RESULTS
    # ===================================================
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


    # ===================================================
    # MAIN RETRIEVAL FUNCTION
    # ===================================================
    def _get_relevant_documents(self, query: str) -> List[Document]:
        self._maybe_rebuild_bm25()

        # ---------- VECTOR SEARCH ----------
        try:
            vector_docs = self.vector_retriever.invoke(query)
        except Exception as e:
            print(f"Vector retrieval failed: {e}")
            vector_docs = []

        # ---------- BM25 SEARCH ----------
        if self.bm25_retriever is None:
            return vector_docs[: self.k]

        try:
            keyword_docs = self.bm25_retriever.invoke(query)
        except Exception as e:
            print(f"BM25 retrieval failed: {e}")
            keyword_docs = []

        # ---------- MERGE ----------
        return self._merge_results(vector_docs, keyword_docs)