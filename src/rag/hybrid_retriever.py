from typing import List, Any, Dict, Tuple

from langchain.schema import Document, BaseRetriever
from langchain_community.retrievers import BM25Retriever
from pydantic import model_validator


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever using:

        Dense vector retrieval
                +
        BM25 keyword retrieval
                ↓
        Reciprocal Rank Fusion (RRF)
                ↓
        final candidate pool

    The retriever keeps detailed diagnostic information about
    Dense, BM25, and fused candidates so that the 39-case
    forensic evaluation can later calculate stage-wise recall.

    IMPORTANT:
        This class does NOT calculate gold Recall@K directly because
        the retriever does not know the gold evidence for a question.

        Instead, it records the exact retrieval candidates/ranks.
        The benchmark diagnostic can compare these candidates
        against the gold evidence afterward.
    """

    # ===================================================
    # CONFIGURATION
    # ===================================================

    vectorstore: Any

    vector_retriever: Any = None
    bm25_retriever: Any = None

    # Final number of candidates returned to downstream pipeline.
    #
    # Tried raising this to 12 (2026-08-15) to let borderline chunks (e.g.
    # strong on BM25 only, cut by MMR-diversified dense ranking) survive
    # RRF fusion long enough to reach the cross-encoder reranker. Regression
    # test on 6 questions showed it fixed CLIP-09's page localization but
    # caused cross-document contamination that broke two previously-correct
    # answers (SR-05, L7) by diluting final context with off-topic
    # candidates. Reverted to 8 pending a more targeted fix (e.g. scope-
    # aware contamination filtering) rather than a blanket widening.
    k: int = 8

    # Larger candidate pool used BEFORE RRF.
    #
    # This is important because we want to test whether the
    # correct evidence exists deeper in the retrieval results.
    candidate_k: int = 25

    # RRF constant.
    rrf_k: int = 60

    _doc_count: int = 0

    # ===================================================
    # INITIAL SETUP
    # ===================================================

    @model_validator(mode="after")
    def _setup(self) -> "HybridRetriever":

        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.candidate_k,
                "fetch_k": 50,
                "lambda_mult": 0.55,
            },
        )

        self._rebuild_bm25()

        return self

    # ===================================================
    # LOAD ALL DOCUMENTS
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
                        metadata=metadata or {},
                    )
                )

        return docs

    # ===================================================
    # DOCUMENT IDENTITY
    # ===================================================

    @staticmethod
    def _document_key(doc: Document) -> Tuple:

        metadata = doc.metadata or {}

        image_path = metadata.get("image_path")

        if image_path:
            return (
                "image",
                image_path,
            )

        return (
            "text",
            metadata.get("source"),
            metadata.get("page"),
            (doc.page_content or "")[:200],
        )

    # ===================================================
    # BUILD BM25 INDEX
    # ===================================================

    def _rebuild_bm25(self):

        docs = self._load_all_documents()

        self._doc_count = len(docs)

        if docs:

            self.bm25_retriever = (
                BM25Retriever.from_documents(docs)
            )

            self.bm25_retriever.k = self.candidate_k

        else:

            self.bm25_retriever = None

    # ===================================================
    # AUTO-REBUILD BM25
    # ===================================================

    def _maybe_rebuild_bm25(self):

        try:

            results = self.vectorstore.get()

            current_count = len(
                results.get("documents", [])
            )

            if current_count != self._doc_count:

                print(
                    f"BM25: rebuilding "
                    f"({self._doc_count} → {current_count})"
                )

                self._rebuild_bm25()

        except Exception as e:

            print(
                f"BM25 rebuild check failed: {e}"
            )

    # ===================================================
    # DIAGNOSTIC DISPLAY
    # ===================================================

    def _print_retrieval_candidates(
        self,
        label: str,
        docs: List[Document],
    ):

        print("\n" + "-" * 90)

        print(
            f"[{label}] {len(docs)} candidates"
        )

        print("-" * 90)

        for rank, doc in enumerate(
            docs,
            start=1,
        ):

            metadata = doc.metadata or {}

            print(
                f"{rank:02d}. "
                f"Source={metadata.get('source')} | "
                f"Page={metadata.get('page')} | "
                f"Type={metadata.get('chunk_type', 'text')} | "
                f"Image={metadata.get('image_path', 'N/A')}"
            )

            preview = (
                (doc.page_content or "")
                .replace("\n", " ")
                .strip()
            )

            print(
                f"    Content={preview[:180]}"
            )

    # ===================================================
    # RRF FUSION
    # ===================================================

    def _rrf_fusion(
        self,
        vector_docs: List[Document],
        keyword_docs: List[Document],
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion.

        For each document:

            RRF score =
                1 / (rrf_k + dense_rank)
                +
                1 / (rrf_k + bm25_rank)

        A document appearing in both retrieval systems
        therefore receives evidence from both rankings.

        Documents appearing in only one system still remain
        candidates.

        This is deliberately rank-based rather than score-based
        because Dense similarity and BM25 scores are not directly
        comparable.
        """

        scores: Dict[Tuple, float] = {}

        documents: Dict[Tuple, Document] = {}

        dense_ranks: Dict[Tuple, int] = {}

        bm25_ranks: Dict[Tuple, int] = {}

        # -------------------------------------------------
        # Dense ranking
        # -------------------------------------------------

        for rank, doc in enumerate(
            vector_docs,
            start=1,
        ):

            key = self._document_key(doc)

            dense_ranks[key] = rank

            documents[key] = doc

            scores[key] = (
                scores.get(key, 0.0)
                + 1.0 / (
                    self.rrf_k + rank
                )
            )

        # -------------------------------------------------
        # BM25 ranking
        # -------------------------------------------------

        for rank, doc in enumerate(
            keyword_docs,
            start=1,
        ):

            key = self._document_key(doc)

            bm25_ranks[key] = rank

            documents[key] = doc

            scores[key] = (
                scores.get(key, 0.0)
                + 1.0 / (
                    self.rrf_k + rank
                )
            )

        # -------------------------------------------------
        # Sort by RRF score
        # -------------------------------------------------

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        # -------------------------------------------------
        # Diagnostic table
        # -------------------------------------------------

        print("\n" + "-" * 90)

        print(
            "[RRF FUSION RANKING]"
        )

        print("-" * 90)

        for final_rank, (
            key,
            score,
        ) in enumerate(
            ranked,
            start=1,
        ):

            doc = documents[key]

            dense_rank = dense_ranks.get(
                key,
                None,
            )

            bm25_rank = bm25_ranks.get(
                key,
                None,
            )

            dense_display = (
                str(dense_rank)
                if dense_rank is not None
                else "-"
            )

            bm25_display = (
                str(bm25_rank)
                if bm25_rank is not None
                else "-"
            )

            metadata = doc.metadata or {}

            print(
                f"{final_rank:02d}. "
                f"RRF={score:.6f} | "
                f"DenseRank={dense_display} | "
                f"BM25Rank={bm25_display} | "
                f"Source={metadata.get('source')} | "
                f"Page={metadata.get('page')} | "
                f"Type={metadata.get('chunk_type', 'text')}"
            )

        print("-" * 90)

        # -------------------------------------------------
        # Return documents in fused ranking order
        # -------------------------------------------------

        return [
            documents[key]
            for key, _ in ranked
        ]

    # ===================================================
    # RETRIEVAL STAGE DIAGNOSTIC SUMMARY
    # ===================================================

    def _print_stage_summary(
        self,
        vector_docs: List[Document],
        keyword_docs: List[Document],
        hybrid_docs: List[Document],
    ):

        dense_keys = {
            self._document_key(doc)
            for doc in vector_docs
        }

        bm25_keys = {
            self._document_key(doc)
            for doc in keyword_docs
        }

        hybrid_keys = {
            self._document_key(doc)
            for doc in hybrid_docs
        }

        overlap = (
            dense_keys
            & bm25_keys
        )

        dense_only = (
            dense_keys
            - bm25_keys
        )

        bm25_only = (
            bm25_keys
            - dense_keys
        )

        print("\n" + "=" * 90)

        print(
            "STEP 3A — RETRIEVAL STAGE SUMMARY"
        )

        print("=" * 90)

        print(
            f"Dense candidates       : "
            f"{len(vector_docs)}"
        )

        print(
            f"BM25 candidates        : "
            f"{len(keyword_docs)}"
        )

        print(
            f"Dense ∩ BM25           : "
            f"{len(overlap)}"
        )

        print(
            f"Dense-only candidates  : "
            f"{len(dense_only)}"
        )

        print(
            f"BM25-only candidates   : "
            f"{len(bm25_only)}"
        )

        print(
            f"Unique candidate pool  : "
            f"{len(dense_keys | bm25_keys)}"
        )

        print(
            f"RRF candidates         : "
            f"{len(hybrid_keys)}"
        )

        print(
            f"Returned to pipeline   : "
            f"{min(len(hybrid_docs), self.k)}"
        )

        print("=" * 90)

    # ===================================================
    # MERGE API
    # ===================================================

    def _merge_results(
        self,
        vector_docs: List[Document],
        keyword_docs: List[Document],
    ) -> List[Document]:

        # Proper rank fusion.
        fused_docs = self._rrf_fusion(
            vector_docs,
            keyword_docs,
        )

        # Return only the downstream candidate count.
        return fused_docs[:self.k]

    # ===================================================
    # MAIN RETRIEVAL
    # ===================================================

    def _get_relevant_documents(
        self,
        query: str,
    ) -> List[Document]:

        self._maybe_rebuild_bm25()

        print("\n" + "=" * 90)

        print(
            "STEP 3A — GLOBAL HYBRID RETRIEVAL"
        )

        print("=" * 90)

        print(
            f"Query              : {query}"
        )

        print(
            f"Dense candidate K  : {self.candidate_k}"
        )

        print(
            f"Final hybrid K     : {self.k}"
        )

        print(
            f"RRF constant       : {self.rrf_k}"
        )

        # =================================================
        # 1. DENSE RETRIEVAL
        # =================================================

        try:

            vector_docs = (
                self.vector_retriever.invoke(
                    query
                )
            )

        except Exception as e:

            print(
                f"Vector retrieval failed: {e}"
            )

            vector_docs = []

        self._print_retrieval_candidates(
            "DENSE / VECTOR RETRIEVAL",
            vector_docs,
        )

        # =================================================
        # 2. BM25 RETRIEVAL
        # =================================================

        if self.bm25_retriever is None:

            print(
                "\n[BM25] Retriever unavailable"
            )

            merged = vector_docs[:self.k]

            self._print_stage_summary(
                vector_docs,
                [],
                merged,
            )

            self._print_retrieval_candidates(
                "FINAL HYBRID RESULT",
                merged,
            )

            print("=" * 90)

            return merged

        try:

            keyword_docs = (
                self.bm25_retriever.invoke(
                    query
                )
            )

        except Exception as e:

            print(
                f"BM25 retrieval failed: {e}"
            )

            keyword_docs = []

        self._print_retrieval_candidates(
            "BM25 RETRIEVAL",
            keyword_docs,
        )

        # =================================================
        # 3. CREATE FULL RRF CANDIDATE POOL
        # =================================================

        fused_docs = self._rrf_fusion(
            vector_docs,
            keyword_docs,
        )

        # =================================================
        # 4. DIAGNOSTIC SUMMARY
        # =================================================

        self._print_stage_summary(
            vector_docs,
            keyword_docs,
            fused_docs,
        )

        # =================================================
        # 5. FINAL HYBRID RESULT
        # =================================================

        final_docs = fused_docs[:self.k]

        self._print_retrieval_candidates(
            "FINAL HYBRID RESULT",
            final_docs,
        )

        print("=" * 90)

        return final_docs

    # ===================================================
    # DOCUMENT-SCOPED RETRIEVAL
    # ===================================================

    def invoke_scoped(
        self,
        query: str,
        sources: List[str],
    ) -> List[Document]:

        """
        Retrieve only from requested source documents.

        Dense retrieval is filtered directly in Chroma.

        BM25 retrieves a larger global candidate pool and is
        subsequently restricted to the requested sources.

        RRF is then applied to the scoped Dense + BM25 candidates.
        """

        if not sources:

            return self.invoke(query)

        self._maybe_rebuild_bm25()

        source_set = set(sources)

        print("\n" + "=" * 90)

        print(
            "STEP 3A — SCOPED HYBRID RETRIEVAL"
        )

        print("=" * 90)

        print(
            f"Query              : {query}"
        )

        print(
            f"Sources            : {sources}"
        )

        print(
            f"Dense candidate K  : {self.candidate_k}"
        )

        print(
            f"Final hybrid K     : {self.k}"
        )

        # =================================================
        # 1. SCOPED DENSE RETRIEVAL
        # =================================================

        try:

            if len(sources) == 1:

                chroma_filter = {
                    "source": sources[0]
                }

            else:

                chroma_filter = {
                    "source": {
                        "$in": sources
                    }
                }

            scoped_vector_retriever = (
                self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": self.candidate_k,
                        "fetch_k": 50,
                        "lambda_mult": 0.55,
                        "filter": chroma_filter,
                    },
                )
            )

            vector_docs = (
                scoped_vector_retriever.invoke(
                    query
                )
            )

        except Exception as e:

            print(
                f"Scoped vector retrieval failed: {e}"
            )

            vector_docs = []

        self._print_retrieval_candidates(
            "SCOPED DENSE / VECTOR RETRIEVAL",
            vector_docs,
        )

        # =================================================
        # 2. SCOPED BM25 RETRIEVAL
        # =================================================

        keyword_docs = []

        if self.bm25_retriever is not None:

            try:

                original_k = (
                    self.bm25_retriever.k
                )

                try:

                    self.bm25_retriever.k = (
                        self.candidate_k * 2
                    )

                    global_keyword_docs = (
                        self.bm25_retriever.invoke(
                            query
                        )
                    )

                finally:

                    self.bm25_retriever.k = (
                        original_k
                    )

                keyword_docs = [
                    doc
                    for doc in global_keyword_docs
                    if doc.metadata.get("source")
                    in source_set
                ]

                keyword_docs = (
                    keyword_docs[
                        :self.candidate_k
                    ]
                )

            except Exception as e:

                print(
                    f"Scoped BM25 retrieval failed: {e}"
                )

                keyword_docs = []

        self._print_retrieval_candidates(
            "SCOPED BM25 RETRIEVAL",
            keyword_docs,
        )

        # =================================================
        # 3. RRF
        # =================================================

        fused_docs = self._rrf_fusion(
            vector_docs,
            keyword_docs,
        )

        # Defensive source restriction.
        fused_docs = [
            doc
            for doc in fused_docs
            if doc.metadata.get("source")
            in source_set
        ]

        # =================================================
        # 4. DIAGNOSTIC SUMMARY
        # =================================================

        self._print_stage_summary(
            vector_docs,
            keyword_docs,
            fused_docs,
        )

        # =================================================
        # 5. FINAL SCOPED HYBRID
        # =================================================

        final_docs = fused_docs[:self.k]

        self._print_retrieval_candidates(
            "SCOPED FINAL HYBRID RESULT",
            final_docs,
        )

        print("=" * 90)

        return final_docs