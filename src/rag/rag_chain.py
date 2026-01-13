
from typing import Dict, Any, List


from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.core.vectorstore import get_vectorstore # We use this to access the Chroma Client

from src.core.llm import get_llm 



RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant answering questions based on the provided document context.

**Instructions:**
1. Answer directly and clearly based on the context below
2. Use bullet points or numbered lists when listing multiple items
3. If the context doesn't contain enough information, say so clearly
4. Cite specific sections when possible (e.g., "According to page 3...")
5. Be concise but comprehensive

**Context from Documents:**
{context}

**Question:** {question}

**Answer:**"""

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)




def retrieve_original_context(chunk_ids: List[str], question: str) -> List[Document]:
    """
    Retrieves the full, original text (large chunks) for a list of chunk IDs 
    that were matched by the small (summary) chunks.
    """
    vs = get_vectorstore()
    
    if not chunk_ids:
        return []
    
   
    unique_ids = [cid for cid in set(chunk_ids) if cid is not None]

   
    chroma_filter = {
        "$and": [
            {"chunk_id": {"$in": unique_ids}},
            {"is_summary": False} 
        ]
    }
    
   
    original_docs = vs.similarity_search(
        query=question, 
        k=20, # Higher K to ensure we capture all relevant originals
        filter=chroma_filter
    )
    
    print(f"[DEBUG] Retrieved {len(original_docs)} original documents based on {len(unique_ids)} summary IDs.")
    return original_docs


def custom_re_ranker(docs: List[Document], question: str) -> List[Document]:
    """
    Custom re-ranking logic applied to the final set of original context documents.
    (Your original implementation, kept here for continuity)
    """
    
    scored_docs = []
    question_lower = question.lower()
    question_keywords = set(question_lower.split())
    
    for doc in docs:
        score = 0
        
       
        if not doc.page_content: continue 
        
        content_lower = doc.page_content.lower()
        meta = doc.metadata or {}
        
      
        
        # 1. Keyword overlap
        content_words = set(content_lower.split())
        overlap = len(question_keywords & content_words)
        score += overlap * 2
        
        # 2. Length bonus/penalty
        doc_len = len(doc.page_content)
        if 100 < doc_len < 2000: score += 5
        if doc_len < 50: score -= 10
        if doc_len > 3000: score -= 5
        
        # 3. Modality bonus
        modality = meta.get('modality', '')
        if modality == 'text': score += 3
        elif modality == 'table':
            if any(word in question_lower for word in ['data', 'table', 'numbers', 'statistics']):
                score += 5
        
        scored_docs.append((score, doc))
        
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for score, doc in scored_docs[:8]]
    
    print(f"[DEBUG] Re-ranked documents (Top 3 Scores): {', '.join([str(s) for s, d in scored_docs[:3]])}")
    
    return top_docs


def format_docs(docs: List[Document]) -> str:
    context_parts = []
    total_chars = 0
    max_chars = 6000 # Max context size
    
    for i, doc in enumerate(docs):
        meta = doc.metadata or {}
        page_num = meta.get('page_number', '?')
        modality = meta.get('modality', '?')
        
        # Score is hard to retrieve here, so we remove the placeholder for now
        header = f"\n--- Source {i+1} (Page {page_num}, {modality}) ---\n"
        body = doc.page_content.strip()
        
        if not body: continue
        
        # Truncate very long chunks
        if len(body) > 1500:
            body = body[:1500] + "... [truncated]"
        
        text = header + body
        
        if total_chars + len(text) > max_chars and context_parts:
            break
        
        context_parts.append(text)
        total_chars += len(text)

    if not context_parts:
        return "No relevant context found in the documents."
    
    print(f"[DEBUG] Final context built from {len(context_parts)} chunks.")
    return "\n".join(context_parts)



def full_retrieval_process(inputs: Dict[str, Any]) -> Dict[str, Any]:
    vs = get_vectorstore()
    question = inputs["question"]
    # 1. RETRIEVE SMALL CHUNKS
    summary_retriever = vs.as_retriever(search_kwargs={"k": 10, "filter": {"is_summary": True}})
    retrieved_summaries = summary_retriever.invoke(question)
    # 2. EXTRACT LINKING ID
    chunk_ids = [doc.metadata.get("chunk_id") for doc in retrieved_summaries]
    # 3. RETRIEVE LARGE CHUNKS
    original_docs = retrieve_original_context(chunk_ids, question)
    # 4. RE-RANK AND FORMAT
    ranked_docs = custom_re_ranker(original_docs, question)
    
    return {
        "context": format_docs(ranked_docs),
        "question": question,
        "source_documents": retrieved_summaries 
    }


def build_rag_chain():
    """Builds and returns the final Hierarchical RAG chain."""
    llm = get_llm()
    
    # 1. Retrieval: Custom logic extracts context and source documents
    retrieval_step = RunnablePassthrough.assign(retrieved_data=full_retrieval_process)
    
    # 2. Context Extraction: Flatten the retrieved_data structure so the prompt can access 'context'
    context_extraction = RunnablePassthrough.assign(
        context=lambda x: x["retrieved_data"]["context"],
        question=lambda x: x["retrieved_data"]["question"],
        source_documents=lambda x: x["retrieved_data"]["source_documents"]
    )
    
    # 3. Answer Generation: Now the prompt and LLM can be piped together easily
    answer_generation = RAG_PROMPT | llm | StrOutputParser()
    
    # 4. Final Formatting: Combine the extracted sources with the generated answer
    rag_chain = (
        retrieval_step
        | context_extraction # Flatten the dictionary structure
        | RunnablePassthrough.assign(answer=answer_generation)
        | (lambda x: {
            "answer": x["answer"],
            "source_documents": x["source_documents"] # Use the source_documents extracted in step 2
        })
    )
    
    return rag_chain