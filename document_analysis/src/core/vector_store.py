from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document 
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import re
import numpy as np
from pathlib import Path
import faiss
import os

class VectorStore:
    def __init__(self):
        # Initialize embeddings with aggressive caching and batching
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            chunk_size=1000,  # Process more text at once
            max_retries=1,    # Minimize retries
            request_timeout=10 # Short timeout
        )
        self.vector_store = None
        self.qa_chain = None

    def create_vector_store(self, chunks: List[Dict]) -> None:
        """Create a FAISS vector store from document chunks."""
        # Prepare documents in batch
        documents = [
            Document(
                page_content=chunk["content"],
                metadata=chunk["metadata"]
            )
            for chunk in chunks
        ]

        # Create vector store with optimized settings
        self.vector_store = FAISS.from_documents(
            documents, 
            self.embeddings,
            normalize_L2=True,  # Faster similarity search
        )

    def setup_qa_chain(self):
        """Setup the QA chain with RFP analysis specialization."""
        if not self.vector_store:
            raise ValueError("Vector store must be created before setting up QA chain")

        # Custom prompt template for FAQ format
        template = """You are analyzing a tender document to answer specific questions about requirements and specifications.
Use the following context to provide a detailed answer in the required FAQ format.

Context: {context}

Question: {question}

Format your answer exactly as follows:

#### {question}

*Key Requirements:*
- List each main requirement with exact values
- Include mandatory criteria
*(Include page and clause references for each point)*

*Supporting Details:*
- List additional specifications
- Include any clarifications or conditions
- Add cross-references to related requirements

*Important Notes:*
- Highlight any critical deadlines or thresholds
- Note any special conditions or exceptions
- Flag any ambiguous or unclear points

Remember to:
1. Only include information explicitly stated in the context
2. Always cite page numbers and clause references
3. Use bold for key numerical values and thresholds
4. Maintain proper formatting with bullet points
5. Cross-reference related requirements when relevant

---"""

        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Initialize chain with RFP-specific settings
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                model="gpt-3.5-turbo-16k",
                temperature=0.0,  # Exact extraction
                max_tokens=10000,  # Increased for comprehensive answers
                presence_penalty=0.0,
                frequency_penalty=0.0
            ),
            retriever=self.vector_store.as_retriever(
                search_kwargs={
                    "k": 6,  # Increased for split clause coverage
                    "fetch_k": 10,
                    "score_threshold": 0.5
                }
            ),
            chain_type_kwargs={
                "prompt": QA_PROMPT,
                "verbose": False
            },
            return_source_documents=True
        )

    def format_citations(self, metadata: Dict) -> str:
        """Format metadata into a citation string."""
        citations = []
        
        if metadata.get("page_ref"):
            citations.append(f"Page: {metadata['page_ref']}")
        if metadata.get("section"):
            citations.append(f"Section: {metadata['section']}")
        if metadata.get("source"):
            citations.append(f"Source: {metadata['source']}")
            
        return " | ".join(citations) if citations else "No citation available"
