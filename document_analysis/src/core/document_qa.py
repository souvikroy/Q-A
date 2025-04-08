from typing import List, Dict, Any
import spacy
import nltk
from textblob import TextBlob

class DocumentChunker:
    def __init__(self):
        """Initialize the document chunker with NLP models."""
        self.nlp = spacy.load("en_core_web_sm")
        nltk.download('punkt', quiet=True)
    
    def chunk_document(self, text: str) -> List[Dict]:
        """Split document into chunks with metadata."""
        chunks = []
        doc = self.nlp(text)
        
        # Process document in sections
        current_section = ""
        current_chunk = []
        chunk_size = 0
        
        for para in text.split('\n\n'):
            if not para.strip():
                continue
                
            # Detect if this is a new section
            blob = TextBlob(para)
            if blob.sentiment.subjectivity < 0.3 and len(para.split()) < 15:
                current_section = para.strip()
                continue
            
            sentences = nltk.sent_tokenize(para)
            
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                    
                # Add to current chunk
                current_chunk.append(sent)
                chunk_size += len(sent.split())
                
                # Create new chunk if size limit reached
                if chunk_size >= 100:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'content': chunk_text,
                        'metadata': {
                            'section': current_section,
                            'size': chunk_size,
                            'has_numbers': any(c.isdigit() for c in chunk_text)
                        }
                    })
                    current_chunk = []
                    chunk_size = 0
        
        # Add remaining text as final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'metadata': {
                    'section': current_section,
                    'size': chunk_size,
                    'has_numbers': any(c.isdigit() for c in chunk_text)
                }
            })
        
        return chunks

def load_document(file_path: str) -> str:
    """Load document from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
