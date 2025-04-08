"""Tests for document processing and QA functionality."""
import unittest
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.document_qa import DocumentChunker, load_document
from src.utils.text_utils import is_section_header, extract_numerical_values

class TestDocumentChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = DocumentChunker()
        self.test_text = """
Section 1: Technical Requirements
The minimum experience required is 5 years in similar projects.
The contractor must have completed at least 3 projects worth Rs. 10 crores each.

Section 2: Financial Requirements
Annual turnover should be minimum Rs. 20 crores.
Net worth should be positive for the last 3 years.
"""

    def test_chunk_document(self):
        chunks = self.chunker.chunk_document(self.test_text)
        self.assertTrue(len(chunks) > 0)
        self.assertTrue(all('content' in chunk for chunk in chunks))
        self.assertTrue(all('metadata' in chunk for chunk in chunks))

    def test_chunk_metadata(self):
        chunks = self.chunker.chunk_document(self.test_text)
        for chunk in chunks:
            metadata = chunk['metadata']
            self.assertIn('section', metadata)
            self.assertIn('size', metadata)
            self.assertIn('has_numbers', metadata)

    def test_section_detection(self):
        lines = self.test_text.split('\n')
        section_headers = [line for line in lines if is_section_header(line.strip())]
        self.assertEqual(len(section_headers), 2)
        self.assertTrue('Technical Requirements' in section_headers[0])
        self.assertTrue('Financial Requirements' in section_headers[1])

    def test_numerical_extraction(self):
        numbers = extract_numerical_values(self.test_text)
        expected = ['5', '3', '10 crores', '20 crores', '3']
        self.assertEqual(len(numbers), len(expected))
        for num in expected:
            self.assertTrue(any(n in num for n in numbers))

if __name__ == '__main__':
    unittest.main()
