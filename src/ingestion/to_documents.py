
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.chunk_schema import Chunk
from src.ingestion.vision_handler import describe_image_with_vision # 🚨 NEW: Import vision handler



INDEX_CHUNK_SIZE = 500 
INDEX_CHUNK_OVERLAP = 50


def _merge_titles_with_body(chunks: List[Chunk]) -> List[Chunk]:
    merged: List[Chunk] = []
    active_title: Chunk | None = None
    buffer_text = []

    chunks_sorted = sorted(
        chunks,
        key=lambda c: (c.file_name, c.page_number or 0, c.id),
    )

    for ch in chunks_sorted:
        category = ch.extra.get("category") if ch.extra else None

        
        if ch.modality == "text" and category == "Title":
            if active_title:
                # Combine title with all accumulated text
                full_content = f"{active_title.content.strip()}\n\n" + "\n".join(buffer_text)
                merged.append(
                    Chunk(
                        id=active_title.id,
                        content=full_content,
                        modality="text",
                        file_name=active_title.file_name,
                        file_type=active_title.file_type,
                        page_number=active_title.page_number,
                        extra=active_title.extra,
                    )
                )
            active_title = ch
            buffer_text = []
            continue

        # If we have an active title, keep adding text under it
        if active_title and ch.modality == "text":
            buffer_text.append(ch.content.strip())
            continue

        # Otherwise, it's a standalone chunk (table/image)
        merged.append(ch)

   
    if active_title:
        full_content = f"{active_title.content.strip()}\n\n" + "\n".join(buffer_text)
        merged.append(
            Chunk(
                id=active_title.id,
                content=full_content,
                modality="text",
                file_name=active_title.file_name,
                file_type=active_title.file_type,
                page_number=active_title.page_number,
                extra=active_title.extra,
            )
        )

    return merged


def process_to_hierarchical_documents(chunks: List[Chunk]) -> List[Document]:
    """
    Processes raw chunks into two document types for Hierarchical RAG:
    1. Small/Summary chunks (indexed for retrieval: is_summary=True)
    2. Large/Original chunks (stored but NOT indexed: is_summary=False)
    """
    merged_chunks = _merge_titles_with_body(chunks)
    final_docs: List[Document] = []
    
    # Text splitter for splitting large ORIGINAL text chunks (Large chunks)
    large_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,          # Larger size for original context
        chunk_overlap=200, 
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    # Text splitter for splitting small INDEXED text chunks (Small chunks)
    # The small chunks are often the summaries/first few sentences
    small_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=INDEX_CHUNK_SIZE,
        chunk_overlap=INDEX_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


    for ch in merged_chunks:
        # Create a consistent, original ID for the entire chunk/section
        # This ID links the small (summary) doc to the large (original) doc
        original_chunk_id = f"original-{ch.id}"

        # ----------------------------------------------------------------------
        # A. Create the LARGE Document (The full context, not indexed)
        # ----------------------------------------------------------------------
        large_metadata = ch.to_langchain_document().metadata.copy()
        large_metadata['chunk_id'] = original_chunk_id  # Link ID
        large_metadata['is_summary'] = False           # Flag as the original data
        
        # Original text content
        original_content = ch.content
        
        # Splitting the original text into large chunks if too big
        if ch.modality == "text" and len(original_content) > 2000:
            original_parts = large_text_splitter.split_text(original_content)
            for i, part in enumerate(original_parts):
                 part_metadata = large_metadata.copy()
                 # Create a unique ID for the sub-part of the large chunk
                 part_metadata['id'] = f"{original_chunk_id}-{i}" 
                 
                 final_docs.append(Document(page_content=part, metadata=part_metadata))
        else:
            large_metadata['id'] = original_chunk_id
            final_docs.append(Document(page_content=original_content, metadata=large_metadata))
            
        
        # ----------------------------------------------------------------------
        # B. Create the SMALL Document (The summary/description, indexed)
        # ----------------------------------------------------------------------
        
        # Base metadata for the indexed small chunk
        small_metadata = ch.to_langchain_document().metadata.copy()
        small_metadata['chunk_id'] = original_chunk_id # Link to the large document
        small_metadata['is_summary'] = True           # Flag as the indexable summary

        indexed_content = ""
        
        if ch.modality == "text":
            # For text, the summary is the first part of the text.
            # Use the small splitter to get the indexable portion.
            indexed_content = small_text_splitter.split_text(ch.content)[0]
            
        elif ch.modality in ["table", "image"]:
            # For multimodal content, the summary is the LLM-generated description.
            # The original path is stored in the chunk's content/extra field.
            image_path = ch.content # Assuming the path is in the content field for image/table elements
            context_snippet = original_content[:200]
            
            description = describe_image_with_vision(image_path, context=context_snippet)
            
            indexed_content = description if description else f"Description missing for {ch.modality} on page {ch.page_number}."
        
        # Check if we have indexable content
        if indexed_content:
            small_metadata['id'] = f"summary-{original_chunk_id}" # Unique ID for the indexed summary
            final_docs.append(Document(page_content=indexed_content, metadata=small_metadata))
            
    return final_docs


def chunks_to_documents(chunks: List[Chunk]) -> List[Document]:
    """
    Backward-compatible function. The multimodal RAG now uses the hierarchical version.
    """
    # Simply call the new function to be used by multimodal_rag.py
    return process_to_hierarchical_documents(chunks)