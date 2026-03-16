from typing import List, Any
from langchain.schema import Document, BaseRetriever
from langchain.retrievers import BM25Retriever


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining:
    - Vector search (ChromaDB)
    - Keyword search (BM25)
    """

    vectorstore: Any
    vector_retriever: Any
    bm25_retriever: Any
    k: int = 12


    def __init__(self, vectorstore, k=12):

        super().__init__()

        self.vectorstore = vectorstore
        self.k = k

        # =====================================================
        # VECTOR RETRIEVER
        # =====================================================
        # Using MMR for diverse results + larger candidate pool
        self.vector_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,               # final returned docs
                "fetch_k": 200,       # 🔥 larger candidate pool
                "lambda_mult": 0.7    # diversity balance
            }
        )

        # =====================================================
        # LOAD ALL DOCUMENTS FOR BM25
        # =====================================================
        docs = self._load_all_documents()

        if len(docs) == 0:
            self.bm25_retriever = None
        else:
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = 12


    # =====================================================
    # LOAD DOCUMENTS FROM VECTOR STORE
    # =====================================================
    def _load_all_documents(self):

        results = self.vectorstore.get()

        docs = []

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        for content, metadata in zip(documents, metadatas):

            docs.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )

        return docs


    # =====================================================
    # MERGE VECTOR + BM25 RESULTS
    # =====================================================
    def _merge_results(self, vector_docs, keyword_docs):

        seen = set()
        merged = []

        # Prioritize semantic (vector) results first
        for doc in vector_docs:

            if doc.page_content not in seen:
                merged.append(doc)
                seen.add(doc.page_content)

        # Add keyword results if not duplicate
        for doc in keyword_docs:

            if doc.page_content not in seen:
                merged.append(doc)
                seen.add(doc.page_content)

        return merged[:self.k]


    # =====================================================
    # MAIN RETRIEVAL FUNCTION
    # =====================================================
    def _get_relevant_documents(self, query: str) -> List[Document]:

        # Vector search
        vector_docs = self.vector_retriever.get_relevant_documents(query)

        if self.bm25_retriever is None:
            return vector_docs[:self.k]

        # Keyword search
        keyword_docs = self.bm25_retriever.get_relevant_documents(query)

        # Merge both retrieval results
        return self._merge_results(vector_docs, keyword_docs)