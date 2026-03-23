from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain.schema import Document

from src.core.table_extractor import extract_tables_from_pdf
from src.core.image_extractor import extract_images_from_pdf

import os
import uuid


def load_documents(file_paths):

    documents = []

    # FIX: deduplicate paths so the same file is never processed twice
    # (this was causing the double-indexing seen in the logs)
    seen_paths = set()
    unique_paths = []
    for p in file_paths:
        real = os.path.realpath(p)
        if real not in seen_paths:
            seen_paths.add(real)
            unique_paths.append(p)

    for path in unique_paths:

        ext = os.path.splitext(path)[1].lower()
        filename = os.path.basename(path)
        doc_id = str(uuid.uuid4())

        # =====================================================
        # PDF DOCUMENTS
        # =====================================================
        if ext == ".pdf":

            # -------------------------
            # TEXT EXTRACTION
            # -------------------------
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()

                for page_number, d in enumerate(docs):
                    if not d.page_content.strip():
                        continue            # skip blank pages

                    documents.append(Document(
                        page_content=d.page_content,
                        metadata={
                            "source": filename,
                            "doc_id": doc_id,
                            "page": page_number + 1,
                            "chunk_type": "text",
                            "preview": d.page_content[:200],
                        }
                    ))

            except Exception as e:
                print(f"Text extraction failed for {filename}: {e}")

            # -------------------------
            # TABLE EXTRACTION
            # -------------------------
            try:
                table_docs = extract_tables_from_pdf(path)

                for table_doc in table_docs:
                    # Merge metadata — do NOT mutate the original object's dict
                    meta = dict(table_doc.metadata)
                    meta["source"] = filename
                    meta["doc_id"] = doc_id
                    meta.setdefault("preview", table_doc.page_content[:200])

                    documents.append(Document(
                        page_content=table_doc.page_content,
                        metadata=meta,
                    ))

            except Exception as e:
                print(f"Table extraction failed for {filename}: {e}")

            # -------------------------
            # IMAGE EXTRACTION
            # -------------------------
            try:
                image_docs = extract_images_from_pdf(path)

                for img_doc in image_docs:
                    # FIX: copy metadata dict before mutating so we don't
                    # accidentally modify the object returned by the extractor
                    meta = dict(img_doc.metadata)
                    meta["source"] = filename
                    meta["doc_id"] = doc_id
                    meta.setdefault("preview", img_doc.page_content[:200])

                    documents.append(Document(
                        page_content=img_doc.page_content,
                        metadata=meta,
                    ))

            except Exception as e:
                print(f"Image extraction failed for {filename}: {e}")

        # =====================================================
        # POWERPOINT DOCUMENTS
        # =====================================================
        elif ext == ".pptx":

            try:
                loader = UnstructuredPowerPointLoader(path)
                docs = loader.load()

                for slide_number, d in enumerate(docs):
                    if not d.page_content.strip():
                        continue            # skip blank slides

                    documents.append(Document(
                        page_content=d.page_content,
                        metadata={
                            "source": filename,
                            "doc_id": doc_id,
                            "page": slide_number + 1,
                            "chunk_type": "text",
                            "preview": d.page_content[:200],
                        }
                    ))

            except Exception as e:
                print(f"PPTX extraction failed for {filename}: {e}")

        else:
            print(f"Unsupported file type skipped: {filename}")

    print(f"\nload_documents complete: {len(documents)} total chunks from {len(unique_paths)} file(s)")
    return documents