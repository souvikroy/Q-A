import os
from typing import List, Dict
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_store import VectorStore
import re


@dataclass
class DocumentSection:
    title: str
    content: str
    page_number: str = ""
    section_number: str = ""


class DocumentChunker:
    def __init__(self):
        # Precompile regex patterns for speed
        self.page_pattern = re.compile(r"===\s*Page\s*(\d+)")
        self.section_pattern = re.compile(r"##\s*([^\n]+)")
        self.numerical_pattern = re.compile(
            r"Rs\.?\s*\d+|\d+(?:\.\d+)?\s*(?:Crore|Lakh|%)"
        )
        self.requirements_pattern = re.compile(
            r"shall|must|required|minimum|at least|mandatory|necessary|essential",
            re.IGNORECASE,
        )
        self.prohibitions_pattern = re.compile(
            r"(?:"
            r"shall not|must not|cannot|not permitted|not allowed|"
            r"prohibited|restricted|forbidden|disallowed|"
            r"no[t\s]+(?:be|to|acceptable|eligible|qualified)|"
            r"will(?:\s+not|\s+be\s+rejected)|"
            r"(?:dis)?qualify.*if|"
            r"subject\s+to\s+(?:rejection|disqualification)|"
            r"not\s+(?:consider|acceptable|allowed|permitted|eligible)|"
            r"breach|violation|penalty|terminate|blacklist|"
            r"under\s+no\s+circumstances|"
            r"fail(?:ure)?\s+to\s+comply|"
            r"non[-\s]?compliance|"
            r"strict(?:ly)?\s+prohibited"
            r")",
            re.IGNORECASE,
        )
        self.clauses_pattern = re.compile(
            r"clause|section|article|schedule|appendix|annexure", re.IGNORECASE
        )

    def _validate_chunk_context(self, text: str, metadata: Dict) -> bool:
        """Validate chunk context for better coherence."""
        # Minimum content requirements
        if len(text.strip().split()) < 10:  # At least 10 words
            return False

        # Validate section context
        if metadata.get("section_type") == "requirement":
            has_requirement = any(
                pattern in text.lower()
                for pattern in ["shall", "must", "required", "minimum", "at least"]
            )
            has_prohibition = any(
                pattern in text.lower()
                for pattern in [
                    "shall not",
                    "must not",
                    "cannot",
                    "prohibited",
                    "restricted",
                ]
            )
            if not (has_requirement or has_prohibition):
                return False

        # Validate numerical context
        if metadata.get("has_numerical_values"):
            if not re.search(r"\d+", text):
                return False

        return True

    def _add_chunk(
        self, chunks: List[Dict], content: str, page: str, section: str, title: str = ""
    ):
        """Add a chunk with metadata."""
        if not content.strip():
            return

        metadata = {
            "page": page,
            "section": section,
            "title": title,  # Add document title to metadata
            "section_type": "general",
            "has_numerical_values": bool(self.numerical_pattern.search(content)),
            "has_clause_references": bool(self.clauses_pattern.search(content)),
            "has_requirements": bool(self.requirements_pattern.search(content)),
            "has_prohibitions": bool(self.prohibitions_pattern.search(content)),
        }

        # Detect requirement sections
        if self.requirements_pattern.search(
            content
        ) or self.prohibitions_pattern.search(content):
            metadata["section_type"] = "requirement"

        # Only add if chunk passes validation
        if self._validate_chunk_context(content, metadata):
            chunks.append({"content": content, "metadata": metadata})

    def chunk_document(
        self, text: str, document_title: str = "Unknown Document", metadata: dict = {}
    ) -> List[Dict]:
        """Split document into validated, metadata-rich chunks."""
        raw_chunks = []

        lines = text.split("\n")
        current_chunk = []
        current_metadata = {
            "section": "",
            "section_type": "general",
            "has_numerical_values": False,
            "has_clause_references": False,
            "has_requirements": False,
            "has_prohibitions": False,
            **metadata,
        }
        
        current_page = current_metadata.get("page", "")
        current_section = current_metadata.get("section", "")

        i = 0
        while i < len(lines):
            line = lines[i]

            page_match = self.page_pattern.match(line)
            if page_match:
                if current_chunk:
                    raw_chunks.append(
                        ("\n".join(current_chunk), current_page, current_section)
                    )
                current_chunk = []
                current_page = page_match.group(1)
                current_metadata["page"] = current_page
                i += 1
                continue

            section_match = self.section_pattern.match(line)
            if section_match:
                if current_chunk:
                    raw_chunks.append(
                        ("\n".join(current_chunk), current_page, current_section)
                    )
                current_chunk = []
                current_section = section_match.group(1)
                current_metadata["section"] = current_section
                i += 1
                continue

            current_chunk.append(line)
            i += 1

        if current_chunk:
            raw_chunks.append(("\n".join(current_chunk), current_page, current_section))

        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

        final_chunks = []
        for content, page, section in raw_chunks:
            split_chunks = splitter.split_text(content)
            for chunk in split_chunks:
                self._add_chunk(
                    final_chunks, chunk, page, section, title=document_title
                )

        return final_chunks


def load_document(file_path: str) -> str:
    """Load document from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


class FAQFormatter:
    def __init__(self):
        self.question_patterns = {
            "technical": [
                r"technical.*qualification",
                r"technical.*requirement",
                r"technical.*criteria",
            ],
            "financial": [
                r"financial.*requirement",
                r"financial.*criteria",
                r"budget",
                r"cost",
            ],
            "project": [r"project.*scope", r"deliverable", r"timeline"],
            "compliance": [r"compliance", r"regulation", r"standard", r"certification"],
            "restrictions": [
                r"restriction",
                r"limitation",
                r"prohibited",
                r"not allowed",
                r"cannot",
            ],
        }

        self.section_prompts = {
            "technical": "What are the technical requirements and restrictions?",
            "financial": "What are the financial requirements and limitations?",
            "project": "What are the project scope requirements and constraints?",
            "compliance": "What are the compliance requirements and prohibited actions?",
            "restrictions": "What specific actions or conditions are prohibited or restricted?",
        }

        # Keywords for technical specifications context
        self.spec_keywords = {
            "schedules": [
                r"schedule [a-z]",
                r"technical schedule",
                r"specification schedule",
            ],
            "materials": [r"material", r"construction material", r"building material"],
            "design": [r"design criteria", r"design requirement", r"structural design"],
            "construction": [
                r"construction",
                r"structural element",
                r"building component",
            ],
            "environmental": [
                r"environmental",
                r"compliance",
                r"environmental requirement",
            ],
            "quality": [r"quality", r"quality control", r"quality assurance"],
        }

    def get_question_type(self, question: str) -> str:
        """Determine the type of question being asked."""
        for qtype, patterns in self.question_patterns.items():
            if any(re.search(pattern, question.lower()) for pattern in patterns):
                return qtype
        return "general"

    def get_enhanced_query(self, question: str, question_type: str) -> str:
        """Enhance the query based on question type."""
        if question_type == "specifications":
            # Add specification-related terms to the query
            spec_terms = []
            for category, patterns in self.spec_keywords.items():
                spec_terms.extend([p.replace(r"[a-z]", "").strip() for p in patterns])

            # Create a focused query for specifications
            return f"{question} AND (schedule OR specification OR technical requirements) AND ({' OR '.join(spec_terms)})"
        return question

    def _extract_page_clause(self, text: str) -> str:
        """Extract page and clause references."""
        page_match = re.search(r"page\s+(?:no\.?\s+)?(\d+)", text.lower())
        clause_match = re.search(r"clause\s+(\d+(?:\.\d+)*)", text.lower())
        schedule_match = re.search(r"schedule\s+([a-z])", text.lower())

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

        if question_type == "technical":
            # Bold interalia statement
            text = re.sub(r"(Bidders who interalia.*?BID value\.)", r"**\1**", text)

            # Bold provided that statements
            text = re.sub(r"(Provided that[^.]*\.)", r"**\1**", text)

            # Bold monetary values with context
            text = re.sub(
                r"(Rs\.\s*\d+(?:\.\d+)?\s*(?:Crore|Lakh)[^.]*\.)", r"**\1**", text
            )

            # Bold minimum/threshold requirements
            text = re.sub(r"((minimum|at least|shall|must)[^.]*\.)", r"**\1**", text)

        elif question_type == "specifications":
            # Bold schedule references
            text = re.sub(r"(Schedule\s+[A-Z])", r"**\1**", text, flags=re.IGNORECASE)

            # Bold key specification terms
            for category, patterns in self.spec_keywords.items():
                for pattern in patterns:
                    text = re.sub(
                        f"({pattern}[^.!?]*)", r"**\1**", text, flags=re.IGNORECASE
                    )
        else:
            # Bold important thresholds and requirements
            text = re.sub(
                r"((?:Rs\.?\s*\d+(?:\.\d+)?\s*(?:Crore|Lakh|Million))|(?:minimum|at least|shall|must)\s+[^.!?]*)",
                r"**\1**",
                text,
            )

        return text

    def format_faq(self, question: str, answer: str) -> str:
        """Format the answer in FAQ style."""
        formatted = []

        # Determine question type
        question_type = self.get_question_type(question)

        # Format the question with number (hardcoded for now)
        formatted.append(f"#### {question}\n")

        # Extract requirements and prohibitions
        requirements = []
        prohibitions = []
        paragraphs = [p.strip() for p in answer.split("\n") if p.strip()]

        for para in paragraphs:
            if any(
                pattern in para.lower()
                for pattern in [
                    "shall not",
                    "must not",
                    "cannot",
                    "prohibited",
                    "restricted",
                    "avoid",
                ]
            ):
                prohibitions.append(para)
            elif any(
                pattern in para.lower()
                for pattern in ["shall", "must", "required", "minimum", "at least"]
            ):
                requirements.append(para)

        # Format requirements
        if requirements:
            formatted.append("*Requirements:*")
            for req in requirements:
                formatted_req = self._format_requirements(req, question_type)
                formatted.append(f"- {formatted_req}")
            formatted.append("")

        # Format prohibitions
        if prohibitions:
            formatted.append("*Restrictions and Prohibitions:*")
            for prob in prohibitions:
                formatted_prob = self._format_requirements(prob, question_type)
                formatted.append(f"- {formatted_prob}")
            formatted.append("")

        # Add any remaining content that's not requirements or prohibitions
        other_content = [
            p for p in paragraphs if p not in requirements and p not in prohibitions
        ]
        if other_content:
            formatted.append("*Additional Information:*")
            for content in other_content:
                formatted_content = self._format_requirements(content, question_type)
                formatted.append(formatted_content)

        # Add citation at the end if present
        citation = self._extract_page_clause(answer)
        if citation:
            formatted.append(f"\n{citation}")

        formatted.append("\n---\n")
        return "\n".join(formatted)


def main():
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python document_qa.py <question>")
        sys.exit(1)

    # Get the question from command line
    question = " ".join(sys.argv[1:])

    # Initialize components
    vector_store = VectorStore()
    faq_formatter = FAQFormatter()

    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Load documents
    docs = []
    for file in ["RFP_formatted.txt", "DCA_formatted.txt"]:
        file_path = os.path.join(current_dir, file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
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
    formatted_response = faq_formatter.format_faq(question, response["answer"])

    # Print the formatted response
    print(formatted_response)


if __name__ == "__main__":
    main()
