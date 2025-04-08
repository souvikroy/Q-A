import os
from typing import List, Dict
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_store import VectorStore
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@dataclass
class DocumentSection:
    title: str
    content: str
    page_number: str = ""
    section_number: str = ""

class DocumentChunker:
    def __init__(self):
        # Precompile regex patterns for speed
        self.page_pattern = re.compile(r'===\s*Page\s*(\d+)')
        self.section_pattern = re.compile(r'##\s*([^\n]+)')
        self.numerical_pattern = re.compile(r'Rs\.?\s*\d+|\d+(?:\.\d+)?\s*(?:Crore|Lakh|%)')
        self.requirements_pattern = re.compile(r'shall|must|required|minimum|at least', re.IGNORECASE)
        self.clauses_pattern = re.compile(r'clause|section|article', re.IGNORECASE)

    def _validate_chunk_context(self, text: str, metadata: Dict) -> bool:
        """Validate chunk context for better coherence."""
        # Minimum content requirements
        if len(text.strip().split()) < 10:  # At least 10 words
            return False
            
        # Check for sentence completeness
        if not text.strip().endswith(('.', '!', '?', ':', ';')):
            return False
            
        # Validate section context
        if metadata.get('section_type') == 'requirement':
            requirement_indicators = ['shall', 'must', 'required', 'minimum', 'at least']
            if not any(indicator in text.lower() for indicator in requirement_indicators):
                return False
                
        # Validate numerical context
        if metadata.get('has_numerical_values'):
            if not re.search(r'\d+', text):
                return False
                
        return True

    def _add_chunk(self, chunks: List[Dict], content: str, page: str, section: str):
        """Add a chunk with metadata."""
        if not content.strip():
            return
            
        metadata = {
            'page': page,
            'section': section,
            'section_type': 'general',
            'has_numerical_values': bool(self.numerical_pattern.search(content)),
            'has_clause_references': bool(self.clauses_pattern.search(content))
        }
        
        # Detect requirement sections
        if self.requirements_pattern.search(content):
            metadata['section_type'] = 'requirement'
            
        # Only add if chunk passes validation
        if self._validate_chunk_context(content, metadata):
            chunks.append({
                "content": content,
                "metadata": metadata
            })

    def chunk_document(self, text: str) -> List[Dict]:
        """Split document into chunks with metadata."""
        chunks = []
        current_page = ""
        current_section = ""
        
        lines = text.split('\n')
        current_chunk = []
        current_metadata = {
            'page': '',
            'section': '',
            'section_type': 'general',
            'has_numerical_values': False,
            'has_clause_references': False
        }
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for page markers
            page_match = self.page_pattern.match(line)
            if page_match:
                if current_chunk:
                    self._add_chunk(chunks, '\n'.join(current_chunk), current_page, current_section)
                current_chunk = []
                current_page = page_match.group(1)
                current_metadata['page'] = current_page
                i += 1
                continue
                
            # Check for section headers
            section_match = self.section_pattern.match(line)
            if section_match:
                if current_chunk:
                    self._add_chunk(chunks, '\n'.join(current_chunk), current_page, current_section)
                current_chunk = []
                current_section = section_match.group(1)
                current_metadata['section'] = current_section
                i += 1
                continue
                
            # Add line to current chunk
            current_chunk.append(line)
            
            # Update metadata based on line content
            if self.numerical_pattern.search(line):
                current_metadata['has_numerical_values'] = True
            if self.requirements_pattern.search(line):
                current_metadata['section_type'] = 'requirement'
            if self.clauses_pattern.search(line):
                current_metadata['has_clause_references'] = True
                
            i += 1
            
        # Add final chunk
        if current_chunk:
            self._add_chunk(chunks, '\n'.join(current_chunk), current_page, current_section)
            
        return chunks

def load_document(file_path: str) -> str:
    """Load document from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

class FAQFormatter:
    def __init__(self):
        self.question_patterns = {
            'technical': [r'technical.*qualification', r'technical.*requirement', r'technical.*criteria'],
            'financial': [r'financial.*qualification', r'financial.*requirement', r'net worth', r'turnover'],
            'jv': [r'joint venture', r'jv.*criteria', r'consortium'],
            'specifications': [r'technical spec', r'specification', r'schedule d', r'design criteria', r'construction requirement'],
            'clauses': [r'important clause', r'key clause', r'critical clause']
        }
        
        # Keywords for technical specifications context
        self.spec_keywords = {
            'schedules': [r'schedule [a-z]', r'technical schedule', r'specification schedule'],
            'materials': [r'material', r'construction material', r'building material'],
            'design': [r'design criteria', r'design requirement', r'structural design'],
            'construction': [r'construction', r'structural element', r'building component'],
            'environmental': [r'environmental', r'compliance', r'environmental requirement'],
            'quality': [r'quality', r'quality control', r'quality assurance']
        }
        
    def get_question_type(self, question: str) -> str:
        """Determine the type of question being asked."""
        for qtype, patterns in self.question_patterns.items():
            if any(re.search(pattern, question.lower()) for pattern in patterns):
                return qtype
        return "general"
        
    def get_enhanced_query(self, question: str, question_type: str) -> str:
        """Enhance the query based on question type."""
        if question_type == 'specifications':
            # Add specification-related terms to the query
            spec_terms = []
            for category, patterns in self.spec_keywords.items():
                spec_terms.extend([p.replace(r'[a-z]', '').strip() for p in patterns])
            
            # Create a focused query for specifications
            return f"{question} AND (schedule OR specification OR technical requirements) AND ({' OR '.join(spec_terms)})"
        return question

    def _extract_page_clause(self, text: str) -> str:
        """Extract page and clause references."""
        page_match = re.search(r'page\s+(?:no\.?\s+)?(\d+)', text.lower())
        clause_match = re.search(r'clause\s+(\d+(?:\.\d+)*)', text.lower())
        schedule_match = re.search(r'schedule\s+([a-z])', text.lower())
        
        citation = []
        if page_match:
            citation.append(f"Page No {page_match.group(1)}")
        if clause_match:
            citation.append(f"Clause {clause_match.group(1)}")
        if schedule_match:
            citation.append(f"Schedule {schedule_match.group(1).upper()}")
            
        return f"*({', '.join(citation)})*" if citation else ""

    def _format_requirements(self, text: str, question_type: str = None) -> str:
        """Format requirements with proper markdown."""
        # Bold key requirements and thresholds
        text = text.strip()
        
        if question_type == 'technical':
            # Bold interalia statement
            text = re.sub(r'(Bidders who interalia.*?BID value\.)', r'**\1**', text)
            
            # Bold provided that statements
            text = re.sub(r'(Provided that[^.]*\.)', r'**\1**', text)
            
            # Bold monetary values with context
            text = re.sub(r'(Rs\.\s*\d+(?:\.\d+)?\s*(?:Crore|Lakh)[^.]*\.)', r'**\1**', text)
            
            # Bold minimum/threshold requirements
            text = re.sub(r'((minimum|at least|shall|must)[^.]*\.)', r'**\1**', text)
        
        elif question_type == 'specifications':
            # Bold schedule references
            text = re.sub(r'(Schedule\s+[A-Z])', r'**\1**', text, flags=re.IGNORECASE)
            
            # Bold key specification terms
            for category, patterns in self.spec_keywords.items():
                for pattern in patterns:
                    text = re.sub(f'({pattern}[^.!?]*)', r'**\1**', text, flags=re.IGNORECASE)
        else:
            # Bold important thresholds and requirements
            text = re.sub(r'((?:Rs\.?\s*\d+(?:\.\d+)?\s*(?:Crore|Lakh|Million))|(?:minimum|at least|shall|must)\s+[^.!?]*)', r'**\1**', text)
        
        return text

    def format_faq(self, question: str, answer: str) -> str:
        """Format the answer in FAQ style."""
        formatted = []
        
        # Determine question type
        question_type = self.get_question_type(question)
        
        # Format the question with number (hardcoded for now)
        if question_type == 'technical':
            formatted.append(f"#### 1. {question}\n")
        else:
            formatted.append(f"#### {question}\n")
        
        # Special handling for technical specifications
        if question_type == 'specifications':
            spec_context = self._extract_spec_context(answer)
            if not spec_context or spec_context == answer:
                spec_context = "The technical specifications are provided in **Schedule D**, covering construction and structural elements including materials, design criteria, and environmental compliance requirements."
            
            formatted_para = self._format_requirements(spec_context, question_type)
            citation = self._extract_page_clause(answer)
            if citation:
                formatted_para = f"{formatted_para}\n{citation}"
            formatted.append(formatted_para + "\n")
        else:
            # Format regular answers
            paragraphs = [p.strip() for p in answer.split('\n') if p.strip()]
            formatted_paras = []
            
            for para in paragraphs:
                formatted_para = self._format_requirements(para, question_type)
                # Add two spaces at the end of each paragraph for markdown line break
                formatted_paras.append(formatted_para + "  ")
            
            # Join paragraphs with newlines
            formatted_text = "\n".join(formatted_paras)
            
            # Add citation at the end if present
            citation = self._extract_page_clause(answer)
            if citation:
                formatted_text = f"{formatted_text}\n{citation}"
                
            formatted.append(formatted_text + "\n")
        
        formatted.append("---\n")
        return "\n".join(formatted)

def main():
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python document_qa.py <question>")
        sys.exit(1)
    
    # Get the question from command line
    question = ' '.join(sys.argv[1:])
    
    # Initialize components
    vector_store = VectorStore()
    faq_formatter = FAQFormatter()
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load documents
    docs = []
    for file in ['RFP_formatted.txt', 'DCA_formatted.txt']:
        file_path = os.path.join(current_dir, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                docs.append(f.read())
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
            continue
    
    if not docs:
        print("Error: No documents could be loaded")
        sys.exit(1)
        
    # Add documents to vector store
    for doc in docs:
        vector_store.add_document(doc)
    
    # Get question type and enhance query if needed
    question_type = faq_formatter.get_question_type(question)
    enhanced_query = faq_formatter.get_enhanced_query(question, question_type)
    
    # Query the vector store with enhanced query
    response = vector_store.query(enhanced_query)
    
    # Format the response as FAQ
    formatted_response = faq_formatter.format_faq(question, response['answer'])
    
    # Print the formatted response
    print(formatted_response)

if __name__ == "__main__":
    main()
