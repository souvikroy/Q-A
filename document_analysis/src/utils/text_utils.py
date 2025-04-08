"""Text processing utilities for document analysis."""
import re
from typing import List, Dict, Any
from textblob import TextBlob

def extract_numerical_values(text: str) -> List[str]:
    """Extract numerical values with their units from text."""
    # Match numbers with optional units
    pattern = r'\b\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:Rs\.?|INR|USD|EUR|%|years?|months?|days?|crores?|lakhs?))?\b'
    return re.findall(pattern, text)

def is_section_header(text: str) -> bool:
    """Determine if text is likely a section header."""
    # Check characteristics of section headers
    return (
        len(text.split()) < 15 and  # Short length
        text.strip().endswith((':')) or  # Ends with colon
        text.strip().isupper() or  # All caps
        bool(re.match(r'^(?:\d+\.)+\s', text))  # Numbered section
    )

def get_text_statistics(text: str) -> Dict[str, Any]:
    """Get statistical information about text."""
    blob = TextBlob(text)
    return {
        'word_count': len(text.split()),
        'sentence_count': len(blob.sentences),
        'has_numbers': bool(re.search(r'\d', text)),
        'sentiment': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize quotes
    text = re.sub(r'[''′]', "'", text)
    text = re.sub(r'[""″]', '"', text)
    # Normalize dashes
    text = re.sub(r'[‐‑‒–—―]', '-', text)
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32)
    return text.strip()

def extract_key_phrases(text: str) -> List[str]:
    """Extract likely key phrases from text."""
    blob = TextBlob(text)
    noun_phrases = blob.noun_phrases
    
    # Filter for relevant phrases
    relevant_phrases = []
    for phrase in noun_phrases:
        # Check if phrase is likely a requirement
        if any(word in phrase for word in ['requirement', 'qualification', 'criteria', 'experience', 'minimum']):
            relevant_phrases.append(phrase)
    
    return relevant_phrases
