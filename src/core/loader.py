from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain.schema import Document

from src.core.table_extractor import extract_tables_from_pdf
from src.core.image_extractor import extract_images_from_pdf

import os
import uuid


def load_documents(file_paths):

    documents = []

    for path in file_paths:

        ext = os.path.splitext(path)[1].lower()
        filename = os.path.basename(path)

        # Unique ID per document
        doc_id = str(uuid.uuid4())

        # =====================================================
        # PDF DOCUMENTS
        # =====================================================
        if ext == ".pdf":

            loader = PyPDFLoader(path)
            docs = loader.load()

            # -------------------------
            # TEXT EXTRACTION
            # -------------------------
            for page_number, d in enumerate(docs):

                metadata = {
                    "source": filename,
                    "doc_id": doc_id,
                    "page": page_number + 1,
                    "chunk_type": "text",
                    "preview": d.page_content[:200]
                }

                documents.append(
                    Document(
                        page_content=d.page_content,
                        metadata=metadata
                    )
                )

            # -------------------------
            # TABLE EXTRACTION
            # -------------------------
            table_docs = extract_tables_from_pdf(path)

            for table_doc in table_docs:

                metadata = table_doc.metadata

                metadata["source"] = filename
                metadata["doc_id"] = doc_id

                if "preview" not in metadata:
                    metadata["preview"] = table_doc.page_content[:200]

                documents.append(
                    Document(
                        page_content=table_doc.page_content,
                        metadata=metadata
                    )
                )

            # -------------------------
            # IMAGE EXTRACTION
            # -------------------------
            image_docs = extract_images_from_pdf(path)

            for img_doc in image_docs:

                metadata = img_doc.metadata

                metadata["source"] = filename
                metadata["doc_id"] = doc_id

                if "preview" not in metadata:
                    metadata["preview"] = img_doc.page_content[:200]

                documents.append(img_doc)

        # =====================================================
        # POWERPOINT DOCUMENTS
        # =====================================================
        elif ext == ".pptx":

            loader = UnstructuredPowerPointLoader(path)
            docs = loader.load()

            for slide_number, d in enumerate(docs):

                metadata = {
                    "source": filename,
                    "doc_id": doc_id,
                    "page": slide_number + 1,
                    "chunk_type": "text",
                    "preview": d.page_content[:200]
                }

                documents.append(
                    Document(
                        page_content=d.page_content,
                        metadata=metadata
                    )
                )

    return documents