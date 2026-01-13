
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Optional, List

from langchain_core.documents import Document 


@dataclass
class Chunk:

    id: str
    content: str               # The actual content (text, table description, image description)
    modality: str              # "text" | "table" | "image"
    file_name: str
    file_type: str             # "pdf" | "pptx" | "image"
    page_number: Optional[int]
    extra: Dict[str, Any]
    
    
    
    def to_langchain_document(self) -> Document:
        
        data_dict = asdict(self)
        
        page_content = data_dict.pop('content')
        
        
        metadata = data_dict
        
        
        extra_data = metadata.pop('extra', {})
        metadata.update(extra_data)
        
        return Document(page_content=page_content, metadata=metadata)
    
    
    @classmethod
    def from_langchain_document(cls, doc: Document) -> 'Chunk':
        
        metadata = doc.metadata.copy()
        
       
        chunk_data = {
            'id': metadata.pop('id', 'N/A'),
            'content': doc.page_content,
            'modality': metadata.pop('modality', 'N/A'),
            'file_name': metadata.pop('file_name', 'N/A'),
            'file_type': metadata.pop('file_type', 'N/A'),
            'page_number': metadata.pop('page_number', None),
        }
        
        
        chunk_data['extra'] = metadata 

        
        return cls(**chunk_data)