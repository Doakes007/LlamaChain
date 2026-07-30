import os
import uuid

from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPowerPointLoader,
)

from src.core.image_extractor import extract_images_from_pdf
from src.core.table_extractor import extract_tables_from_pdf


# =====================================================
# HELPERS
# =====================================================

def clean_filename(path: str) -> str:
    """
    Remove UUID prefix from uploaded filenames.

    Example:
        82f14dc3dd2ddcca_Paper.pdf
            ->
        Paper.pdf
    """
    filename = os.path.basename(path)

    if "_" in filename:
        filename = filename.split("_", 1)[1]

    return filename


# =====================================================
# DOCUMENT LOADER
# =====================================================

def load_documents(file_paths):

    documents = []

    # -------------------------------------------------
    # Remove duplicate paths
    # -------------------------------------------------

    seen_paths = set()
    unique_paths = []

    for path in file_paths:

        real_path = os.path.realpath(path)

        if real_path not in seen_paths:
            seen_paths.add(real_path)
            unique_paths.append(path)

    # -------------------------------------------------
    # Process each document
    # -------------------------------------------------

    for path in unique_paths:

        extension = os.path.splitext(path)[1].lower()

        filename = clean_filename(path)

        doc_id = str(uuid.uuid4())

        # =====================================================
        # PDF
        # =====================================================

        if extension == ".pdf":

            # -------------------------------------------------
            # TEXT
            # -------------------------------------------------

            try:

                loader = PyPDFLoader(path)

                docs = loader.load()

                for page_number, doc in enumerate(docs):

                    if not doc.page_content.strip():
                        continue

                    documents.append(
                        Document(
                            page_content=doc.page_content,
                            metadata={
                                "source": filename,
                                "doc_id": doc_id,
                                "page": page_number + 1,
                                "chunk_type": "text",
                                "preview": doc.page_content[:200],
                            },
                        )
                    )

            except Exception as e:

                print(
                    f"Text extraction failed for "
                    f"{filename}: {e}"
                )

            # -------------------------------------------------
            # TABLES
            # -------------------------------------------------

            try:

                table_docs = extract_tables_from_pdf(path)

                for table_doc in table_docs:

                    metadata = dict(table_doc.metadata)

                    metadata["source"] = filename
                    metadata["doc_id"] = doc_id

                    metadata.setdefault(
                        "preview",
                        table_doc.page_content[:200],
                    )

                    documents.append(
                        Document(
                            page_content=table_doc.page_content,
                            metadata=metadata,
                        )
                    )

            except Exception as e:

                print(
                    f"Table extraction failed for "
                    f"{filename}: {e}"
                )

            # -------------------------------------------------
            # IMAGES
            # -------------------------------------------------

            try:

                image_docs = extract_images_from_pdf(path)

                for image_doc in image_docs:

                    metadata = dict(image_doc.metadata)

                    metadata["source"] = filename
                    metadata["doc_id"] = doc_id

                    metadata.setdefault(
                        "preview",
                        image_doc.page_content[:200],
                    )

                    documents.append(
                        Document(
                            page_content=image_doc.page_content,
                            metadata=metadata,
                        )
                    )

            except Exception as e:

                print(
                    f"Image extraction failed for "
                    f"{filename}: {e}"
                )

        # =====================================================
        # POWERPOINT
        # =====================================================

        elif extension == ".pptx":

            try:

                loader = UnstructuredPowerPointLoader(path)

                docs = loader.load()

                for slide_number, doc in enumerate(docs):

                    if not doc.page_content.strip():
                        continue

                    documents.append(
                        Document(
                            page_content=doc.page_content,
                            metadata={
                                "source": filename,
                                "doc_id": doc_id,
                                "page": slide_number + 1,
                                "chunk_type": "text",
                                "preview": doc.page_content[:200],
                            },
                        )
                    )

            except Exception as e:

                print(
                    f"PPTX extraction failed for "
                    f"{filename}: {e}"
                )

        # =====================================================
        # UNSUPPORTED
        # =====================================================

        else:

            print(
                f"Unsupported file type skipped: "
                f"{filename}"
            )

    print(
        f"\nload_documents complete: "
        f"{len(documents)} total chunks "
        f"from {len(unique_paths)} file(s)"
    )

    return documents