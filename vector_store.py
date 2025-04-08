from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document 
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import re
import pickle
from pathlib import Path
import faiss
import os
import numpy as np
from pymongo import MongoClient
from datetime import datetime

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
        self.index_path = None

    def create_vector_store(self, chunks: List[Document]) -> None:
        """Create a FAISS vector store from document chunks."""
        # Prepare documents in batch
        # documents = [
        #     Document(
        #         page_content=chunk["content"],
        #         metadata=chunk["metadata"]
        #     )
        #     for chunk in chunks
        # ]

        # # Create vector store with optimized settings
        # self.vector_store = FAISS.from_documents(
        #     documents, 
        #     self.embeddings,
        #     normalize_L2=True,  # Faster similarity search
        # )
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)

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
        if metadata.get("title"):
            citations.append(f"Document: {metadata['title']}")
            
        return " | ".join(citations) if citations else "No citation available"

    def format_response(self, answer: str, question: str, source_documents: List[Document]) -> Dict[str, Any]:
        """Format response with the required simple format."""
        # Remove any extra whitespace and ensure proper line endings
        answer = answer.strip()
        
        # Format in the required pattern
        formatted_response = f"Question : {question}\nAnswer : {answer}"
        
        # Extract citations from source documents
        citations = [self.format_citations(doc.metadata) for doc in source_documents]
        
        return {
            "answer": formatted_response,
            "citations": citations,
            "context_quality": "good"
        }

    def validate_and_filter_context(self, documents: List[Document], question: str) -> str:
        """Validate and filter context based on relevance to question."""
        filtered_contexts = []
        
        # Extract key terms from question
        question_terms = set(re.findall(r'\w+', question.lower()))
        
        for doc in documents:
            # Calculate relevance score
            content_terms = set(re.findall(r'\w+', doc.page_content.lower()))
            relevance_score = len(question_terms.intersection(content_terms)) / len(question_terms)
            
            # Check for numerical values if question asks for them
            has_numbers = bool(re.search(r'\d+', doc.page_content)) if any(term in question.lower() for term in ['how many', 'how much', 'value', 'amount']) else True
            
            # Check for temporal information if question asks for timing
            has_temporal = bool(re.search(r'\b(date|time|year|month|schedule|deadline)\b', doc.page_content, re.I)) if any(term in question.lower() for term in ['when', 'date', 'time', 'schedule']) else True
            
            # Validate metadata
            metadata = doc.metadata
            valid_metadata = (
                metadata.get('page_ref', '').strip() != '' and
                metadata.get('section', '').strip() != ''
            )
            
            # Only include if context is relevant and valid
            if relevance_score > 0.3 and valid_metadata and has_numbers and has_temporal:
                # Format context with metadata
                formatted_context = f"{doc.page_content} [{metadata.get('page_ref', '')}] [{metadata.get('section', '')}]"
                filtered_contexts.append((relevance_score, formatted_context))
        
        # Sort by relevance and combine
        filtered_contexts.sort(reverse=True)
        return "\n\n".join(context for score, context in filtered_contexts[:3])

    def get_requirement_context(self, question_type: str) -> Dict[str, Any]:
        """Get specific requirement context based on question type."""
        # Define search parameters for each requirement type
        search_params = {
            'technical_qualification': {
                'keywords': ['technical qualification', 'experience', 'expertise', 'similar works'],
                'required_patterns': [r'\d+\s*years?', r'similar\s+works?', r'projects?'],
                'metadata_filters': {'section_type': ['technical', 'qualification', 'eligibility']}
            },
            'financial_qualification': {
                'keywords': ['financial', 'turnover', 'net worth', 'revenue'],
                'required_patterns': [r'Rs\.?\s*\d+(?:\.\d+)?\s*crores?', r'\d+\s*crores?', r'\d+\s*lakhs?'],
                'metadata_filters': {'section_type': ['financial', 'qualification', 'eligibility']}
            },
            'jv_criteria': {
                'keywords': ['joint venture', 'JV', 'consortium', 'partnership'],
                'required_patterns': [r'\d+\s*%', r'lead\s+member', r'partner\s+share'],
                'metadata_filters': {'section_type': ['jv', 'partnership', 'eligibility']}
            },
            'technical_specs': {
                'keywords': ['technical specification', 'technical requirement', 'specification'],
                'required_patterns': [r'shall', r'must', r'specifications?'],
                'metadata_filters': {'section_type': ['technical', 'specification', 'requirement']}
            },
            'important_clauses': {
                'keywords': ['clause', 'article', 'section', 'provision'],
                'required_patterns': [r'clause\s+\d+', r'article\s+\d+', r'section\s+\d+'],
                'metadata_filters': {'section_type': ['clause', 'legal', 'condition']}
            }
        }

        # Get search parameters for the question type
        params = search_params.get(question_type, {})
        if not params:
            return {'context': '', 'confidence': 0}

        # Search for relevant documents
        relevant_docs = self.vector_store.similarity_search(
            ' '.join(params['keywords']),
            k=8  # Increased for better coverage
        )

        # Filter and validate documents
        validated_contexts = []
        for doc in relevant_docs:
            content = doc.page_content
            metadata = doc.metadata

            # Calculate match confidence
            pattern_matches = sum(1 for pattern in params['required_patterns'] 
                                if re.search(pattern, content, re.IGNORECASE))
            pattern_score = pattern_matches / len(params['required_patterns'])

            section_match = any(stype in metadata.get('section_type', '').lower() 
                              for stype in params['metadata_filters'].get('section_type', []))
            
            confidence = (pattern_score + (1 if section_match else 0)) / 2

            if confidence > 0.3:  # Minimum confidence threshold
                # Format context with metadata
                formatted_context = f"{content} [{metadata.get('page_ref', '')}] [{metadata.get('section', '')}]"
                validated_contexts.append({
                    'content': formatted_context,
                    'confidence': confidence
                })

        # Sort by confidence and combine
        validated_contexts.sort(key=lambda x: x['confidence'], reverse=True)
        if not validated_contexts:
            return {'context': '', 'confidence': 0}

        # Format combined context with citations
        formatted_contexts = []
        for ctx in validated_contexts[:3]:  # Top 3 most confident contexts
            citation = f"[{ctx['content'].split('[')[1].split(']')[0]}]"
            formatted_contexts.append(f"{ctx['content'].split('[')[0]} {citation}")

        return {
            'context': '\n\n'.join(formatted_contexts),
            'confidence': sum(ctx['confidence'] for ctx in validated_contexts[:3]) / 3
        }

    def query(self, question: str) -> Dict[str, Any]:
        """Query with requirement-specific context handling."""
        try:
            # Check if this is a technical qualification query
            if re.search(r'technical\s+qualification|technical\s+requirements?', question, re.IGNORECASE):
                tech_quals = self.extract_technical_qualifications()
                formatted_response = self.format_technical_response(tech_quals)
                return {
                    "answer": formatted_response,
                    "citations": [tech_quals.get('source_reference', '')],
                    "context_quality": "good" if tech_quals.get('full_text') else "partial",
                    "raw_data": tech_quals
                }
            
            # Check if this is a financial qualification query
            elif re.search(r'financial\s+qualification|financial\s+requirements?|net\s+worth|turnover', question, re.IGNORECASE):
                fin_quals = self.extract_financial_qualifications()
                formatted_response = self.format_financial_response(fin_quals)
                return {
                    "answer": formatted_response,
                    "citations": [fin_quals.get('source_reference', '')],
                    "context_quality": "good" if fin_quals.get('financial_qualification_text') else "partial",
                    "raw_data": fin_quals
                }

            # Check if this is a joint venture query
            elif re.search(r'joint\s+venture|JV|consortium|lead\s+member', question, re.IGNORECASE):
                jv_criteria = self.extract_joint_venture_criteria()
                formatted_response = self.format_jv_response(jv_criteria)
                return {
                    "answer": formatted_response,
                    "citations": [jv_criteria.get('source_reference', '')],
                    "context_quality": "good" if jv_criteria.get('joint_venture_clause_text') else "partial",
                    "raw_data": jv_criteria
                }

            # Check if this is a technical specifications query
            elif re.search(r'technical\s+specifications?|specifications?|schedule\s+d', question, re.IGNORECASE):
                tech_specs = self.extract_technical_specifications()
                formatted_response = self.format_technical_specs_response(tech_specs)
                return {
                    "answer": formatted_response,
                    "citations": [tech_specs.get('source_reference', '')],
                    "context_quality": "good" if tech_specs.get('technical_specifications_text') else "partial",
                    "raw_data": tech_specs
                }

            # Check if this is an important clauses query
            elif re.search(r'important\s+clauses|key\s+clauses', question, re.IGNORECASE):
                imp_clauses = self.extract_important_clauses()
                formatted_response = self.format_important_clauses_response(imp_clauses)
                return {
                    "answer": formatted_response,
                    "citations": [imp_clauses.get('source_reference', '')],
                    "context_quality": "good",
                    "raw_data": imp_clauses
                }

            # Check if this is a legal clauses query
            elif re.search(r'legal\s+clauses', question, re.IGNORECASE):
                legal_clauses = self.extract_legal_clauses()
                formatted_response = self.format_legal_clauses_response(legal_clauses)
                return {
                    "answer": formatted_response,
                    "citations": [],
                    "context_quality": "good",
                    "raw_data": legal_clauses
                }

            # Handle other question types
            question_type_patterns = {
                'important_clauses': r'important\s+clauses|key\s+clauses'
            }

            # Determine question type
            question_type = None
            for qtype, pattern in question_type_patterns.items():
                if re.search(pattern, question, re.IGNORECASE):
                    question_type = qtype
                    break

            if question_type:
                # Get specific requirement context
                context_data = self.get_requirement_context(question_type)
                if context_data['confidence'] > 0:
                    validated_context = context_data['context']
                else:
                    # Fallback to general search if specific context not found
                    relevant_docs = self.vector_store.similarity_search(question, k=6)
                    validated_context = self.validate_and_filter_context(relevant_docs, question)
            else:
                # General question handling
                relevant_docs = self.vector_store.similarity_search(question, k=6)
                validated_context = self.validate_and_filter_context(relevant_docs, question)

            # Get answer using validated context
            result = self.qa_chain({
                "query": question,
                "context": validated_context
            })

            # Extract answer and sources
            answer = result.get('result', '')
            source_documents = result.get('source_documents', [])

            # Format response
            return self.format_response(answer, question, source_documents)

        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "citations": [],
                "context_quality": "error"
            }

    def extract_technical_qualifications(self) -> Dict[str, str]:
        """Extract technical qualification details in the required format."""
        # Search parameters for each component
        search_configs = {
            'full_text': {
                'keywords': ['technical qualification', 'qualification criteria', 'minimum qualification'],
                'patterns': [r'bidders?\s+who\s+meet', r'qualification\s+criteria', r'eligible\s+bidders?'],
                'section_types': ['qualification', 'eligibility']
            },
            'bid_capacity_condition': {
                'keywords': ['bid capacity', 'bidding capacity', 'available capacity'],
                'patterns': [r'bid\s+capacity', r'available\s+capacity', r'total\s+bid\s+value'],
                'section_types': ['financial', 'capacity']
            },
            'technical_capacity': {
                'keywords': ['technical capacity', 'construction experience', 'eligible projects'],
                'patterns': [r'Rs\.?\s*\d+(?:\.\d+)?\s*crores?', r'past\s+\d+\s+(?:financial\s+)?years?'],
                'section_types': ['technical', 'experience']
            },
            'project_completion_requirement': {
                'keywords': ['similar work', 'project completion', 'completed work'],
                'patterns': [r'similar\s+work\s+worth', r'\d+%\s*of\s*estimated', r'Rs\.?\s*\d+(?:\.\d+)?\s*crores?'],
                'section_types': ['completion', 'experience']
            },
            'span_condition': {
                'keywords': ['span', 'bridge', 'ROB', 'flyover'],
                'patterns': [r'span\s*[<>=]+\s*\d+\s*m', r'bridge|ROB|flyover', r'demonstrate\s+experience'],
                'section_types': ['technical', 'bridge']
            }
        }

        result = {}
        source_ref = None

        for component, config in search_configs.items():
            # Search for relevant documents
            docs = self.vector_store.similarity_search(
                ' '.join(config['keywords']),
                k=5
            )

            best_match = None
            best_confidence = 0

            for doc in docs:
                content = doc.page_content
                metadata = doc.metadata

                # Calculate match confidence
                pattern_matches = sum(1 for pattern in config['patterns'] 
                                   if re.search(pattern, content, re.IGNORECASE))
                pattern_score = pattern_matches / len(config['patterns'])

                section_match = any(stype in metadata.get('section_type', '').lower() 
                                  for stype in config['section_types'])
                
                confidence = (pattern_score + (1 if section_match else 0)) / 2

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = content
                    
                    # Update source reference if this is the best match
                    if confidence > 0.5:  # Only update if confidence is high enough
                        page_ref = metadata.get('page_ref', '').replace('Page ', '')
                        section = metadata.get('section', '')
                        if page_ref and section:
                            source_ref = f"Page {page_ref}, {section}"

            if best_match and best_confidence > 0.3:
                # Clean and format the content
                content = best_match.strip()
                content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
                content = re.sub(r'\[.*?\]', '', content)  # Remove existing citations
                result[component] = content

        # Add source reference if found
        if source_ref:
            result['source_reference'] = source_ref
        else:
            result['source_reference'] = "Source reference not found"

        return result

    def format_technical_response(self, tech_quals: Dict[str, str]) -> str:
        """Format technical qualifications into a readable response."""
        response = """#### Technical Qualification Requirements

*Key Requirements:*\n"""
        
        if tech_quals.get('full_text'):
            response += f"- **Overview**: {tech_quals['full_text']}\n"
            
        if tech_quals.get('bid_capacity_condition'):
            response += f"- **Bid Capacity**: {tech_quals['bid_capacity_condition']}\n"
            
        if tech_quals.get('technical_capacity'):
            response += f"- **Technical Capacity**: {tech_quals['technical_capacity']}\n"
            
        if tech_quals.get('project_completion_requirement'):
            response += f"- **Project Completion**: {tech_quals['project_completion_requirement']}\n"
            
        if tech_quals.get('span_condition'):
            response += f"- **Span Requirements**: {tech_quals['span_condition']}\n"
            
        response += f"\n*Source: {tech_quals.get('source_reference', 'Reference not found')}*\n\n---"
        
        return response

    def extract_financial_qualifications(self) -> Dict[str, str]:
        """Extract financial qualification details in the required format."""
        # Search parameters for financial components
        search_configs = {
            'financial_qualification_text': {
                'keywords': ['financial capacity', 'net worth', 'annual turnover'],
                'patterns': [
                    r'net\s+worth.*Rs\.?\s*\d+(?:\.\d+)?\s*crores?',
                    r'annual\s+turnover.*Rs\.?\s*\d+(?:\.\d+)?\s*crores?'
                ],
                'section_types': ['financial', 'qualification']
            },
            'minimum_net_worth': {
                'keywords': ['net worth', 'financial capacity'],
                'patterns': [
                    r'minimum\s+net\s+worth',
                    r'Rs\.?\s*\d+(?:\.\d+)?\s*crores?',
                    r'preceding\s+financial\s+year'
                ],
                'section_types': ['financial', 'net worth']
            },
            'average_annual_turnover': {
                'keywords': ['annual turnover', 'average turnover'],
                'patterns': [
                    r'minimum\s+average\s+annual\s+turnover',
                    r'Rs\.?\s*\d+(?:\.\d+)?\s*crores?',
                    r'last\s+\d+\s+financial\s+years?'
                ],
                'section_types': ['financial', 'turnover']
            }
        }

        result = {}
        source_ref = None

        for component, config in search_configs.items():
            # Search for relevant documents
            docs = self.vector_store.similarity_search(
                ' '.join(config['keywords']),
                k=5
            )

            best_match = None
            best_confidence = 0

            for doc in docs:
                content = doc.page_content
                metadata = doc.metadata

                # Calculate match confidence
                pattern_matches = sum(1 for pattern in config['patterns'] 
                                   if re.search(pattern, content, re.IGNORECASE))
                pattern_score = pattern_matches / len(config['patterns'])

                section_match = any(stype in metadata.get('section_type', '').lower() 
                                  for stype in config['section_types'])
                
                confidence = (pattern_score + (1 if section_match else 0)) / 2

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = content
                    
                    # Update source reference if this is the best match
                    if confidence > 0.5:  # Only update if confidence is high enough
                        page_ref = metadata.get('page_ref', '').replace('Page ', '')
                        section = metadata.get('section', '')
                        if page_ref and section:
                            source_ref = f"Page {page_ref}, {section}"

            if best_match and best_confidence > 0.3:
                # Clean and format the content
                content = best_match.strip()
                content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
                content = re.sub(r'\[.*?\]', '', content)  # Remove existing citations
                
                # Extract specific financial values
                if component == 'minimum_net_worth':
                    net_worth_match = re.search(r'net\s+worth.*Rs\.?\s*\d+(?:\.\d+)?\s*crores?', content, re.IGNORECASE)
                    if net_worth_match:
                        content = f"Minimum Net Worth (Financial Capacity) of {net_worth_match.group()} at the close of the preceding financial year."
                
                elif component == 'average_annual_turnover':
                    turnover_match = re.search(r'Rs\.?\s*\d+(?:\.\d+)?\s*crores?', content, re.IGNORECASE)
                    years_match = re.search(r'last\s+(\d+)\s+financial\s+years?', content, re.IGNORECASE)
                    if turnover_match and years_match:
                        content = f"Minimum Average Annual Turnover of {turnover_match.group()} (updated as per price level indices) for the last {years_match.group(1)} financial years."
                
                result[component] = content

        # Add source reference if found
        if source_ref:
            result['source_reference'] = source_ref
        else:
            result['source_reference'] = "Source reference not found"

        return result

    def format_financial_response(self, fin_quals: Dict[str, str]) -> str:
        """Format financial qualifications into a readable response."""
        response = """#### Financial Qualification Requirements

*Key Requirements:*\n"""
        
        if fin_quals.get('minimum_net_worth'):
            response += f"- **Net Worth**: {fin_quals['minimum_net_worth']}\n"
            
        if fin_quals.get('average_annual_turnover'):
            response += f"- **Annual Turnover**: {fin_quals['average_annual_turnover']}\n"
            
        if fin_quals.get('financial_qualification_text'):
            response += f"\n*Additional Details:*\n{fin_quals['financial_qualification_text']}\n"
            
        response += f"\n*Source: {fin_quals.get('source_reference', 'Reference not found')}*\n\n---"
        
        return response

    def extract_joint_venture_criteria(self) -> Dict[str, str]:
        """Extract joint venture criteria details in the required format."""
        # Search parameters for JV components
        search_configs = {
            'joint_venture_clause_text': {
                'keywords': ['joint venture', 'JV', 'lead member', 'consortium'],
                'patterns': [
                    r'members?\s+of\s+the\s+joint\s+venture',
                    r'lead\s+member',
                    r'\d+%\s*(?:requirement|of)',
                ],
                'section_types': ['jv', 'eligibility', 'qualification']
            },
            'lead_member_criteria': {
                'keywords': ['lead member', 'lead partner'],
                'patterns': [
                    r'lead\s+member\s+must\s+meet',
                    r'\d+%\s*(?:requirement|of)',
                    r'(?:bid|technical|financial)\s+capacity'
                ],
                'section_types': ['jv', 'lead member']
            },
            'other_members_criteria': {
                'keywords': ['other members', 'JV members'],
                'patterns': [
                    r'other\s+members?\s+must\s+meet',
                    r'\d+%\s*(?:requirement|of)',
                    r'(?:bid|technical|financial)\s+capacity'
                ],
                'section_types': ['jv', 'members']
            },
            'collective_jv_criteria': {
                'keywords': ['JV as a whole', 'collectively'],
                'patterns': [
                    r'(?:JV|joint\s+venture)\s+as\s+a\s+whole',
                    r'collectively\s+meet',
                    r'100%'
                ],
                'section_types': ['jv', 'collective']
            },
            'project_execution_criteria': {
                'keywords': ['project length', 'undertake', 'lead member'],
                'patterns': [
                    r'lead\s+member\s+shall',
                    r'\d+%\s*of\s*(?:the\s+)?project',
                    r'project\s+length'
                ],
                'section_types': ['jv', 'execution']
            },
            'joint_bidding_agreement': {
                'keywords': ['joint bidding agreement', 'JV agreement'],
                'patterns': [
                    r'joint\s+bidding\s+agreement',
                    r'must\s+be\s+executed',
                    r'uploaded'
                ],
                'section_types': ['jv', 'agreement']
            }
        }

        result = {}
        source_ref = None

        for component, config in search_configs.items():
            docs = self.vector_store.similarity_search(
                ' '.join(config['keywords']),
                k=5
            )

            best_match = None
            best_confidence = 0

            for doc in docs:
                content = doc.page_content
                metadata = doc.metadata

                # Calculate match confidence
                pattern_matches = sum(1 for pattern in config['patterns'] 
                                   if re.search(pattern, content, re.IGNORECASE))
                pattern_score = pattern_matches / len(config['patterns'])

                section_match = any(stype in metadata.get('section_type', '').lower() 
                                  for stype in config['section_types'])
                
                confidence = (pattern_score + (1 if section_match else 0)) / 2

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = content
                    
                    # Update source reference if this is the best match
                    if confidence > 0.5:  # Only update if confidence is high enough
                        page_ref = metadata.get('page_ref', '').replace('Page ', '')
                        section = metadata.get('section', '')
                        if page_ref and section:
                            source_ref = f"Page {page_ref}, {section}"

            if best_match and best_confidence > 0.3:
                # Clean and format the content
                content = best_match.strip()
                content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
                content = re.sub(r'\[.*?\]', '', content)  # Remove existing citations
                
                # Format specific components
                if component == 'lead_member_criteria':
                    lead_match = re.search(r'lead\s+member\s+must\s+meet\s+.*?(?=\.|$)', content, re.IGNORECASE)
                    if lead_match:
                        content = lead_match.group()
                elif component == 'other_members_criteria':
                    other_match = re.search(r'other\s+members?\s+must\s+meet\s+.*?(?=\.|$)', content, re.IGNORECASE)
                    if other_match:
                        content = other_match.group()
                elif component == 'collective_jv_criteria':
                    collective_match = re.search(r'(?:JV|joint\s+venture)\s+as\s+a\s+whole\s+.*?(?=\.|$)', content, re.IGNORECASE)
                    if collective_match:
                        content = collective_match.group()
                
                result[component] = content

        # Add source reference if found
        if source_ref:
            result['source_reference'] = source_ref
        else:
            result['source_reference'] = "Source reference not found"

        return result

    def format_jv_response(self, jv_criteria: Dict[str, str]) -> str:
        """Format joint venture criteria into a readable response."""
        response = """#### Joint Venture Requirements

*Key Requirements:*\n"""
        
        if jv_criteria.get('lead_member_criteria'):
            response += f"- **Lead Member**: {jv_criteria['lead_member_criteria']}\n"
            
        if jv_criteria.get('other_members_criteria'):
            response += f"- **Other Members**: {jv_criteria['other_members_criteria']}\n"
            
        if jv_criteria.get('collective_jv_criteria'):
            response += f"- **Collective Requirements**: {jv_criteria['collective_jv_criteria']}\n"
            
        if jv_criteria.get('project_execution_criteria'):
            response += f"- **Project Execution**: {jv_criteria['project_execution_criteria']}\n"
            
        if jv_criteria.get('joint_bidding_agreement'):
            response += f"- **Agreement**: {jv_criteria['joint_bidding_agreement']}\n"
            
        if jv_criteria.get('joint_venture_clause_text'):
            response += f"\n*Additional Details:*\n{jv_criteria['joint_venture_clause_text']}\n"
            
        response += f"\n*Source: {jv_criteria.get('source_reference', 'Reference not found')}*\n\n---"
        
        return response

    def extract_technical_specifications(self) -> Dict[str, str]:
        """Extract technical specifications details in the required format."""
        # Search parameters for technical specification components
        search_configs = {
            'technical_specifications_text': {
                'keywords': ['technical specifications', 'technical requirements', 'specifications'],
                'patterns': [
                    r'technical\s+specifications?',
                    r'schedule\s+[a-d]',
                    r'(?:construction|structural|design)\s+(?:elements?|criteria)'
                ],
                'section_types': ['technical', 'specification', 'schedule']
            },
            'specification_scope': {
                'keywords': ['scope', 'coverage', 'specifications include'],
                'patterns': [
                    r'cover(?:s|ing)\s+(?:construction|structural|design)',
                    r'materials?',
                    r'(?:environmental|compliance)\s+requirements?'
                ],
                'section_types': ['scope', 'specification']
            },
            'specification_location': {
                'keywords': ['schedule', 'technical schedules', 'location'],
                'patterns': [
                    r'schedule\s+[a-d]',
                    r'technical\s+schedules?',
                    r'(?:section|part|volume)\s+\d+'
                ],
                'section_types': ['schedule', 'location']
            }
        }

        result = {}
        source_ref = None

        for component, config in search_configs.items():
            docs = self.vector_store.similarity_search(
                ' '.join(config['keywords']),
                k=5
            )

            best_match = None
            best_confidence = 0

            for doc in docs:
                content = doc.page_content
                metadata = doc.metadata

                # Calculate match confidence
                pattern_matches = sum(1 for pattern in config['patterns'] 
                                   if re.search(pattern, content, re.IGNORECASE))
                pattern_score = pattern_matches / len(config['patterns'])

                section_match = any(stype in metadata.get('section_type', '').lower() 
                                  for stype in config['section_types'])
                
                confidence = (pattern_score + (1 if section_match else 0)) / 2

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = content
                    
                    # Update source reference if this is the best match
                    if confidence > 0.5:
                        page_ref = metadata.get('page_ref', '').replace('Page ', '')
                        section = metadata.get('section', '')
                        if page_ref and section:
                            source_ref = f"Page {page_ref}, {section}"

            if best_match and best_confidence > 0.3:
                # Clean and format the content
                content = best_match.strip()
                content = re.sub(r'\s+', ' ', content)
                content = re.sub(r'\[.*?\]', '', content)
                
                # Format specific components
                if component == 'specification_scope':
                    scope_match = re.search(r'cover(?:s|ing)\s+.*?(?=\.|$)', content, re.IGNORECASE)
                    if scope_match:
                        content = scope_match.group()
                elif component == 'specification_location':
                    location_match = re.search(r'(?:schedule|section|part)\s+.*?(?=\.|$)', content, re.IGNORECASE)
                    if location_match:
                        content = f"Schedule D of Technical Schedules (A to D)."
                
                result[component] = content

        # Add source reference
        if source_ref:
            result['source_reference'] = source_ref
        else:
            result['source_reference'] = "Source reference not found"

        return result

    def format_technical_specs_response(self, tech_specs: Dict[str, str]) -> str:
        """Format technical specifications into a readable response."""
        response = """#### Technical Specifications

*Key Requirements:*\n"""
        
        if tech_specs.get('specification_location'):
            response += f"- **Location**: {tech_specs['specification_location']}\n"
            
        if tech_specs.get('specification_scope'):
            response += f"- **Scope**: {tech_specs['specification_scope']}\n"
            
        if tech_specs.get('technical_specifications_text'):
            response += f"\n*Additional Details:*\n{tech_specs['technical_specifications_text']}\n"
            
        response += f"\n*Source: {tech_specs.get('source_reference', 'Reference not found')}*\n\n---"
        
        return response

    def extract_important_clauses(self) -> Dict[str, Any]:
        """Extract important clauses details in the required format."""
        # Comprehensive clause patterns organized by category
        clause_patterns = {
            # General & Administrative Clauses
            'Bid Validity Period': [r'bid\s+validity', r'tender\s+validity', r'offer\s+validity'],
            'Pre-Bid Meeting': [r'pre[-\s]bid\s+meeting', r'pre[-\s]tender\s+meeting', r'bidders?\s+conference'],
            'Addendum / Corrigendum': [r'addendum', r'corrigendum', r'amendment', r'modification\s+to\s+tender'],
            'Joint Venture / Consortium': [r'joint\s+venture', r'consortium', r'JV\s+agreement', r'partnership'],
            'Subcontracting': [r'sub[-\s]contract', r'sublet', r'outsource', r'third[-\s]party\s+work'],
            'Debarment / Blacklisting': [r'debar', r'blacklist', r'ban', r'exclude\s+from\s+bidding'],
            
            # Financial & Security Clauses
            'Earnest Money Deposit': [r'earnest\s+money\s+deposit', r'EMD', r'bid\s+security'],
            'Mode of EMD Submission': [r'mode\s+of\s+(?:EMD|earnest\s+money)', r'EMD\s+(?:form|submission|payment)'],
            'Security Deposit': [r'security\s+deposit', r'retention\s+money', r'retention\s+amount'],
            'Performance Security': [r'performance\s+(?:security|guarantee|bond)', r'contract\s+performance\s+guarantee'],
            'Mode of Performance Security': [r'mode\s+of\s+performance\s+security', r'form\s+of\s+security', r'BG\s+submission'],
            'Bank Guarantee': [r'bank\s+guarantee', r'BG', r'financial\s+security'],
            'Payment Terms': [r'payment\s+terms', r'milestone\s+payment', r'stage\s+payment', r'running\s+account\s+bill'],
            
            # Project Timeline Clauses
            'Completion Period': [r'completion\s+period', r'contract\s+duration', r'project\s+timeline', r'time\s+of\s+completion'],
            'Mobilization Advance': [r'mobilization\s+advance', r'mobilisation\s+payment', r'advance\s+payment'],
            'Extension of Time': [r'extension\s+of\s+time', r'EOT', r'time\s+extension'],
            'Liquidated Damages': [r'liquidated\s+damages', r'LD', r'delay\s+damages', r'compensation\s+for\s+delay'],
            
            # Technical & Quality Clauses
            'Technical Specifications': [r'technical\s+specifications?', r'technical\s+requirements?', r'specifications?'],
            'Quality Assurance': [r'quality\s+(?:assurance|control|plan)', r'QA[/\s]QC', r'testing\s+requirements?'],
            'Material Standards': [r'material\s+(?:standards?|specifications?)', r'IS\s+codes?', r'BIS\s+standards?'],
            
            # Legal & Compliance Clauses
            'Dispute Resolution': [r'dispute\s+resolution', r'arbitration', r'conciliation', r'settlement\s+of\s+disputes'],
            'Force Majeure': [r'force\s+majeure', r'act\s+of\s+god', r'unforeseen\s+circumstances'],
            'Labour Laws': [r'labour\s+laws?', r'workers?\s+compensation', r'minimum\s+wages?'],
            'Insurance': [r'insurance', r'CAR\s+policy', r'contractor\'?s?\s+all\s+risk'],
            
            # Price & Cost Clauses
            'Price Variation': [r'price\s+variation', r'cost\s+adjustment', r'escalation', r'price\s+adjustment'],
            'Taxes and Duties': [r'taxes?\s+and\s+duties?', r'GST', r'income\s+tax', r'statutory\s+(?:levy|payment)'],
            
            # Safety & Environmental Clauses
            'Safety Requirements': [r'safety\s+(?:requirements?|measures?|protocols?)', r'HSE', r'EHS'],
            'Environmental Compliance': [r'environmental\s+(?:compliance|protection)', r'pollution\s+control', r'eco[-\s]friendly']
        }

        # Search parameters for general clause information
        search_configs = {
            'important_clauses_text': {
                'keywords': ['important clauses', 'key clauses', 'tender conditions'],
                'patterns': [
                    r'section\s+\d+',
                    r'general\s+conditions?',
                    r'special\s+conditions?',
                    r'instructions?\s+to\s+bidders?'
                ],
                'section_types': ['clause', 'condition', 'instruction', 'legal']
            },
            'clause_location_reference': {
                'keywords': ['section', 'clause', 'article'],
                'patterns': [
                    r'section\s+\d+(?:\.\d+)*',
                    r'clause\s+\d+(?:\.\d+)*',
                    r'article\s+\d+(?:\.\d+)*',
                    r'pages?\s+\d+(?:\s*[-–]\s*\d+)?'
                ],
                'section_types': ['section', 'reference', 'contents']
            }
        }

        result = {
            'important_clauses_list': [],
            'clause_categories': {
                'general_administrative': [],
                'financial_security': [],
                'project_timeline': [],
                'technical_quality': [],
                'legal_compliance': [],
                'price_cost': [],
                'safety_environmental': []
            }
        }
        source_ref = None

        # First get the general clauses text and location
        for component, config in search_configs.items():
            docs = self.vector_store.similarity_search(
                ' '.join(config['keywords']),
                k=5
            )

            best_match = None
            best_confidence = 0

            for doc in docs:
                content = doc.page_content
                metadata = doc.metadata

                # Calculate match confidence
                pattern_matches = sum(1 for pattern in config['patterns'] 
                                   if re.search(pattern, content, re.IGNORECASE))
                pattern_score = pattern_matches / len(config['patterns'])

                section_match = any(stype in metadata.get('section_type', '').lower() 
                                  for stype in config['section_types'])
                
                confidence = (pattern_score + (1 if section_match else 0)) / 2

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = content
                    
                    # Update source reference if this is the best match
                    if confidence > 0.5:
                        page_ref = metadata.get('page_ref', '').replace('Page ', '')
                        section = metadata.get('section', '')
                        if page_ref and section:
                            source_ref = f"Page {page_ref}, {section}"

            if best_match and best_confidence > 0.3:
                # Clean and format the content
                content = best_match.strip()
                content = re.sub(r'\s+', ' ', content)
                content = re.sub(r'\[.*?\]', '', content)
                result[component] = content

        # Now search for specific clauses
        for clause_name, patterns in clause_patterns.items():
            docs = self.vector_store.similarity_search(
                clause_name,
                k=3
            )

            for doc in docs:
                content = doc.page_content
                if any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns):
                    result['important_clauses_list'].append(clause_name)
                    
                    # Categorize the clause
                    if any(term in clause_name.lower() for term in ['bid', 'pre-bid', 'addendum', 'joint venture', 'subcontract', 'debar']):
                        result['clause_categories']['general_administrative'].append(clause_name)
                    elif any(term in clause_name.lower() for term in ['earnest', 'security', 'performance', 'bank', 'payment']):
                        result['clause_categories']['financial_security'].append(clause_name)
                    elif any(term in clause_name.lower() for term in ['completion', 'mobilization', 'extension', 'liquidated']):
                        result['clause_categories']['project_timeline'].append(clause_name)
                    elif any(term in clause_name.lower() for term in ['technical', 'quality', 'material']):
                        result['clause_categories']['technical_quality'].append(clause_name)
                    elif any(term in clause_name.lower() for term in ['dispute', 'force', 'labour', 'insurance']):
                        result['clause_categories']['legal_compliance'].append(clause_name)
                    elif any(term in clause_name.lower() for term in ['price', 'tax', 'cost']):
                        result['clause_categories']['price_cost'].append(clause_name)
                    elif any(term in clause_name.lower() for term in ['safety', 'environmental']):
                        result['clause_categories']['safety_environmental'].append(clause_name)
                    break

        # Add source reference
        if source_ref:
            result['source_reference'] = source_ref
        else:
            result['source_reference'] = "Source reference not found"

        return result

    def format_important_clauses_response(self, clauses: Dict[str, Any]) -> str:
        """Format important clauses into a readable response."""
        # Category display names for better readability
        category_names = {
            'priority_clauses': '🔍 High Priority Clauses',
            'general_administrative': 'General & Administrative',
            'financial_security': 'Financial & Security',
            'project_timeline': 'Project Timeline',
            'technical_quality': 'Technical & Quality',
            'legal_compliance': 'Legal & Compliance',
            'price_cost': 'Price & Cost',
            'safety_environmental': 'Safety & Environmental'
        }
        
        response = []
        
        # Add summary of findings
        total_clauses = sum(len(clauses) for clauses in clauses.values())
        priority_count = len(clauses.get('priority_clauses', []))
        
        if priority_count > 0:
            response.append(f"Found {priority_count} high-priority clauses and {total_clauses - priority_count} other clauses.\n")
        else:
            response.append(f"Found {total_clauses} clauses in the document.\n")
            response.append("\n⚠️ Note: The following high-priority clauses were not found:")
            missing_clauses = [
                "Earnest Money Deposit (EMD)",
                "Mode of EMD Submission",
                "Completion Period",
                "Mobilization Advance",
                "Security Deposit / Retention Money",
                "Defect Liability Period",
                "Performance Security",
                "Mode of Performance Security",
                "Price Variation Clause",
                "Incentive / Bonus Clause"
            ]
            for clause in missing_clauses:
                response.append(f"- {clause}")
            response.append("")
        
        # Start with priority clauses if any
        if clauses.get('priority_clauses'):
            response.append(f"\n### {category_names['priority_clauses']}")
            
            # Sort priority clauses by confidence
            priority_clauses = sorted(clauses['priority_clauses'], 
                                   key=lambda x: x['confidence'], 
                                   reverse=True)
            
            for clause in priority_clauses:
                # Format clause name with emphasis
                clause_title = clause['clause_name'].replace('_', ' ').title()
                response.append(f"\n#### ⭐ {clause_title}")
                
                # Add section type if available
                if clause['section_type']:
                    section_type = clause['section_type'].replace('_', ' ').title()
                    response.append(f"*Type: {section_type}*")
                
                # Add the matched text with strong emphasis
                response.append(f"\n**Key Requirement:** {clause['matched_text']}")
                
                if clause['values']:
                    response.append("\n*Specified Values:*")
                    for value in clause['values']:
                        response.append(f"- **{value}**")
                
                # Add relevant context (truncated for readability)
                context = clause['context']
                if len(context) > 200:
                    context = context[:200] + "..."
                response.append(f"\n*Context:*\n> {context}\n")
        
        # Then add other categories
        for category, clauses in clauses.items():
            if category != 'priority_clauses' and clauses:
                response.append(f"\n### {category_names[category]}")
                
                # Sort clauses by confidence
                clauses.sort(key=lambda x: x['confidence'], reverse=True)
                
                for clause in clauses:
                    # Format clause name
                    clause_title = clause['clause_name'].replace('_', ' ').title()
                    response.append(f"\n#### {clause_title}")
                    
                    # Add section type if available
                    if clause['section_type']:
                        section_type = clause['section_type'].replace('_', ' ').title()
                        response.append(f"*Type: {section_type}*")
                    
                    # Add the matched text
                    response.append(f"\n**Key Requirement:** {clause['matched_text']}")
                    
                    if clause['values']:
                        response.append("\n*Specified Values:*")
                        for value in clause['values']:
                            response.append(f"- {value}")
                    
                    # Add relevant context (truncated for readability)
                    context = clause['context']
                    if len(context) > 200:
                        context = context[:200] + "..."
                    response.append(f"\n*Context:*\n> {context}\n")
        
        return "\n".join(response)

    def extract_legal_clauses(self) -> Dict[str, List[Dict[str, Any]]]:
        """Extract important legal clauses from the document using pattern matching.
        
        Returns:
            Dictionary mapping clause categories to lists of found clauses with their details
        """
        # Get the pattern store instance
        pattern_store = LocalPatternStore()
        
        # Get all text from documents
        combined_text = ""
        for doc in self.docs:
            combined_text += doc.page_content + "\n\n"
            
        # Find legal clauses using pattern store
        legal_clauses = pattern_store.find_legal_clauses(combined_text)
        
        return legal_clauses
        
    def format_legal_clauses_response(self, legal_clauses: Dict[str, List[Dict[str, Any]]]) -> str:
        """Format the extracted legal clauses into a readable response.
        
        Args:
            legal_clauses: Dictionary of legal clauses by category
            
        Returns:
            Formatted markdown string
        """
        # Category display names for better readability
        category_names = {
            'priority_clauses': '🔍 High Priority Clauses',
            'general_administrative': 'General & Administrative',
            'financial_security': 'Financial & Security',
            'project_timeline': 'Project Timeline',
            'technical_quality': 'Technical & Quality',
            'legal_compliance': 'Legal & Compliance',
            'price_cost': 'Price & Cost',
            'safety_environmental': 'Safety & Environmental'
        }
        
        response = []
        
        # Add summary of findings
        total_clauses = sum(len(clauses) for clauses in legal_clauses.values())
        priority_count = len(legal_clauses.get('priority_clauses', []))
        
        if priority_count > 0:
            response.append(f"Found {priority_count} high-priority clauses and {total_clauses - priority_count} other clauses.\n")
        else:
            response.append(f"Found {total_clauses} clauses in the document.\n")
            response.append("\n⚠️ Note: The following high-priority clauses were not found:")
            missing_clauses = [
                "Earnest Money Deposit (EMD)",
                "Mode of EMD Submission",
                "Completion Period",
                "Mobilization Advance",
                "Security Deposit / Retention Money",
                "Defect Liability Period",
                "Performance Security",
                "Mode of Performance Security",
                "Price Variation Clause",
                "Incentive / Bonus Clause"
            ]
            for clause in missing_clauses:
                response.append(f"- {clause}")
            response.append("")
        
        # Start with priority clauses if any
        if legal_clauses.get('priority_clauses'):
            response.append(f"\n### {category_names['priority_clauses']}")
            
            # Sort priority clauses by confidence
            priority_clauses = sorted(legal_clauses['priority_clauses'], 
                                   key=lambda x: x['confidence'], 
                                   reverse=True)
            
            for clause in priority_clauses:
                # Format clause name with emphasis
                clause_title = clause['clause_name'].replace('_', ' ').title()
                response.append(f"\n#### ⭐ {clause_title}")
                
                # Add section type if available
                if clause['section_type']:
                    section_type = clause['section_type'].replace('_', ' ').title()
                    response.append(f"*Type: {section_type}*")
                
                # Add the matched text with strong emphasis
                response.append(f"\n**Key Requirement:** {clause['matched_text']}")
                
                if clause['values']:
                    response.append("\n*Specified Values:*")
                    for value in clause['values']:
                        response.append(f"- **{value}**")
                
                # Add relevant context (truncated for readability)
                context = clause['context']
                if len(context) > 200:
                    context = context[:200] + "..."
                response.append(f"\n*Context:*\n> {context}\n")
        
        # Then add other categories
        for category, clauses in legal_clauses.items():
            if category != 'priority_clauses' and clauses:
                response.append(f"\n### {category_names[category]}")
                
                # Sort clauses by confidence
                clauses.sort(key=lambda x: x['confidence'], reverse=True)
                
                for clause in clauses:
                    # Format clause name
                    clause_title = clause['clause_name'].replace('_', ' ').title()
                    response.append(f"\n#### {clause_title}")
                    
                    # Add section type if available
                    if clause['section_type']:
                        section_type = clause['section_type'].replace('_', ' ').title()
                        response.append(f"*Type: {section_type}*")
                    
                    # Add the matched text
                    response.append(f"\n**Key Requirement:** {clause['matched_text']}")
                    
                    if clause['values']:
                        response.append("\n*Specified Values:*")
                        for value in clause['values']:
                            response.append(f"- {value}")
                    
                    # Add relevant context (truncated for readability)
                    context = clause['context']
                    if len(context) > 200:
                        context = context[:200] + "..."
                    response.append(f"\n*Context:*\n> {context}\n")
        
        return "\n".join(response)

    def save_vector_store(self, index_path: str = None) -> None:
        """Save the vector store to disk."""
        if self.vector_store:
            save_path = index_path or self.index_path or "vectors.faiss"
            try:
                self.vector_store.save_local(save_path)
                print(f"Vector store saved to {save_path}")
            except Exception as e:
                print(f"Error saving vector store: {str(e)}")
                # Try saving in the current directory as a fallback
                fallback_path = os.path.join(os.getcwd(), "vectors.faiss")
                print(f"Attempting to save in current directory: {fallback_path}")
                self.vector_store.save_local(fallback_path)
                print(f"Vector store saved to fallback location: {fallback_path}")

    def train_with_documents(self, document_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Train the vector store with additional documents.
        
        Args:
            document_paths: List of paths to documents or directories to train on
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks to maintain context
        """
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        import os

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        documents = []
        for path in document_paths:
            if os.path.isdir(path):
                # Load all text files from directory
                loader = DirectoryLoader(
                    path,
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                    loader_kwargs={'encoding': 'utf-8'}
                )
                dir_docs = loader.load()
                documents.extend(dir_docs)
            elif os.path.isfile(path) and path.endswith('.txt'):
                # Load single text file
                loader = TextLoader(path, encoding='utf-8')
                file_docs = loader.load()
                documents.extend(file_docs)

        if not documents:
            raise ValueError("No valid documents found in the provided paths")

        print(f"Loaded {len(documents)} documents")

        # Split documents into chunks
        splits = text_splitter.split_documents(documents)
        print(f"Created {len(splits)} chunks")

        # Add metadata to chunks
        for split in splits:
            # Extract section information from content
            section_match = re.search(r'Section\s+\d+(?:\.\d+)*', split.page_content)
            section = section_match.group() if section_match else ''
            
            # Extract page reference if available
            page_match = re.search(r'Page\s+\d+', split.page_content)
            page_ref = page_match.group() if page_match else ''
            
            # Determine section type based on content
            section_type = self._determine_section_type(split.page_content)
            
            # Update metadata
            split.metadata.update({
                'section': section,
                'page_ref': page_ref,
                'section_type': section_type,
                'source': os.path.basename(split.metadata.get('source', ''))
            })

        print("Added metadata to chunks")

        # Add to vector store
        if not hasattr(self, 'vector_store') or not self.vector_store:
            print("Creating new vector store")
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
        else:
            print("Adding documents to existing vector store")
            self.vector_store.add_documents(splits)

        # Save the updated index
        self.save_vector_store()
        print("Vector store updated and saved")

    def _determine_section_type(self, content: str) -> str:
        """Determine the type of section based on content analysis."""
        section_indicators = {
            'technical': [
                r'technical\s+(?:qualification|requirement|specification)',
                r'experience\s+requirement',
                r'similar\s+works?',
                r'project\s+experience'
            ],
            'financial': [
                r'financial\s+(?:qualification|requirement|capacity)',
                r'net\s+worth',
                r'turnover',
                r'monetary\s+value'
            ],
            'eligibility': [
                r'eligibility\s+criteria',
                r'qualifying\s+requirement',
                r'minimum\s+requirement',
                r'basic\s+qualification'
            ],
            'jv': [
                r'joint\s+venture',
                r'consortium',
                r'partnership',
                r'associate'
            ],
            'clause': [
                r'clause\s+\d+',
                r'article\s+\d+',
                r'section\s+\d+',
                r'provision'
            ],
            'schedule': [
                r'schedule\s+[a-z]',
                r'appendix',
                r'annexure',
                r'attachment'
            ]
        }

        content_lower = content.lower()
        section_types = []

        for section_type, patterns in section_indicators.items():
            if any(re.search(pattern, content_lower) for pattern in patterns):
                section_types.append(section_type)

        return ','.join(section_types) if section_types else 'general'

    def format_common_question_response(self, answer_info: Dict[str, Any]) -> str:
        """Format the answer to a common question into a readable response.
        
        Args:
            answer_info: Dictionary containing answer details
            
        Returns:
            Formatted markdown string
        """
        response = []
        
        # Format based on question type
        if answer_info['question_type'] == 'technical_qualification':
            response.append("### Technical Qualification Requirements\n")
            
            # Add bid capacity requirement
            if 'bid_capacity' in answer_info['matches']:
                response.append("**Bid Capacity Requirement:**")
                response.append(answer_info['matches']['bid_capacity']['matched_text'])
                response.append("")
            
            # Add experience requirement
            if 'experience_value' in answer_info['matches']:
                response.append("**Experience Requirement:**")
                response.append(answer_info['matches']['experience_value']['matched_text'])
                response.append("")
            
            # Add similar work requirement
            if 'similar_work' in answer_info['matches']:
                response.append("**Similar Work Requirement:**")
                response.append(answer_info['matches']['similar_work']['matched_text'])
                response.append("")
            
            # Add span requirement if present
            if 'span_requirement' in answer_info['matches']:
                response.append("**Span Requirement:**")
                response.append(answer_info['matches']['span_requirement']['matched_text'])
                response.append("")
                
        elif answer_info['question_type'] == 'financial_qualification':
            response.append("### Financial Qualification Requirements\n")
            
            # Add net worth requirement
            if 'net_worth' in answer_info['matches']:
                response.append("**Net Worth Requirement:**")
                response.append(answer_info['matches']['net_worth']['matched_text'])
                response.append("")
            
            # Add turnover requirement
            if 'turnover' in answer_info['matches']:
                response.append("**Turnover Requirement:**")
                response.append(answer_info['matches']['turnover']['matched_text'])
                response.append("")
                
        elif answer_info['question_type'] == 'joint_venture':
            response.append("### Joint Venture (JV) Criteria\n")
            
            response.append("**Key Requirements:**")
            
            # Add lead member requirement
            if 'lead_member' in answer_info['matches']:
                response.append(f"- {answer_info['matches']['lead_member']['matched_text']}")
            
            # Add other members requirement
            if 'other_members' in answer_info['matches']:
                response.append(f"- {answer_info['matches']['other_members']['matched_text']}")
            
            # Add collective requirement
            if 'collective' in answer_info['matches']:
                response.append(f"- {answer_info['matches']['collective']['matched_text']}")
            
            # Add project length requirement
            if 'project_length' in answer_info['matches']:
                response.append(f"- {answer_info['matches']['project_length']['matched_text']}")
            
            response.append("")
                
        elif answer_info['question_type'] == 'technical_specifications':
            response.append("### Technical Specifications\n")
            
            # Add schedule information
            if 'schedule_d' in answer_info['matches']:
                response.append("**Schedule D Details:**")
                response.append(answer_info['matches']['schedule_d']['matched_text'])
                response.append("")
            
            if 'schedules' in answer_info['matches']:
                response.append("**Technical Schedules:**")
                response.append(answer_info['matches']['schedules']['matched_text'])
                response.append("")
        
        # Add reference if available
        if answer_info.get('reference'):
            response.append(f"\n*{answer_info['reference']}*")
        
        return "\n".join(response)

    def query(self, query: str) -> Dict[str, Any]:
        """Query the vector store for relevant content."""
        # Get embeddings for the query
        query_embedding = self._get_embedding(query)
        
        # Find most similar chunks
        similar_chunks = []
        for chunk in self.chunks:
            similarity = self._cosine_similarity(query_embedding, chunk['embedding'])
            if similarity > 0.7:  # Minimum similarity threshold
                similar_chunks.append({
                    'content': chunk['content'],
                    'similarity': similarity
                })
        
        # Sort by similarity
        similar_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Take top chunks
        top_chunks = similar_chunks[:3]
        
        if not top_chunks:
            return {
                'answer': 'No relevant information found.',
                'context_quality': 'low'
            }
            
        # Combine content from top chunks
        combined_content = []
        
        for chunk in top_chunks:
            content = chunk['content'].strip()
            # Add content if not duplicate
            if content not in combined_content:
                combined_content.append(content)
        
        # Determine context quality
        context_quality = 'high' if len(top_chunks) >= 2 else 'medium'
        if max(c['similarity'] for c in top_chunks) < 0.8:
            context_quality = 'medium'
        
        # Format answer
        answer = ' '.join(combined_content)
        
        return {
            'answer': answer,
            'context_quality': context_quality
        }

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get the embedding for a piece of text."""
        return self.embeddings.embed_query(text)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class LocalVectorStore:
    def __init__(self, dimension: int = 768, index_path: str = "vectors.faiss", metadata_path: str = "metadata.pkl"):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        # Use IndexFlatIP for cosine similarity with normalized vectors
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: Dict[int, Dict] = {}
        self.load_if_exists()

    def load_if_exists(self):
        """Load existing index and metadata if they exist"""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.metadata_path.exists():
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)

    def save(self):
        """Save index and metadata to disk"""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def add_vectors(self, vectors: np.ndarray, metadata_list: List[Dict]):
        """Add vectors and their metadata to the store"""
        if len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        
        start_idx = self.index.ntotal
        self.index.add(vectors.astype(np.float32))
        
        # Add metadata
        for i, metadata in enumerate(metadata_list):
            self.metadata[start_idx + i] = metadata

        self.save()

    def search(self, query_vector: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors"""
        # Normalize query vector
        faiss.normalize_L2(query_vector)
        
        # Search
        distances, indices = self.index.search(query_vector.astype(np.float32), k)
        return distances, indices

    def clear(self):
        """Clear the index and metadata"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata.clear()
        self.save()

    def __len__(self):
        return self.index.ntotal


class MongoVectorStore:
    def __init__(self, uri: str, db_name: str, collection_name: str = "vector_store"):
        """Initialize MongoDB vector store.
        
        Args:
            uri: MongoDB connection URI
            db_name: Database name
            collection_name: Collection name for storing vectors
        """
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            chunk_size=1000,
            max_retries=1,
            request_timeout=10
        )
        
        # Create indexes for efficient querying
        self.collection.create_index([("vector", 1)])  # Regular index for vector field
        self.collection.create_index([("metadata.source", 1)])
        self.collection.create_index([("created_at", -1)])
        
    def normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return (np.array(vector) / norm).tolist()
        
    def add_documents(self, chunks: List[Dict]) -> None:
        """Add document chunks to MongoDB vector store.
        
        Args:
            chunks: List of dictionaries containing content and metadata
        """
        documents = []
        for chunk in chunks:
            # Generate embedding and normalize
            vector = self.embeddings.embed_query(chunk["content"])
            normalized_vector = self.normalize_vector(vector)
            
            # Prepare document for MongoDB
            doc = {
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "vector": normalized_vector,
                "created_at": datetime.utcnow()
            }
            documents.append(doc)
        
        # Batch insert documents
        if documents:
            self.collection.insert_many(documents)
            
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents using vector similarity.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        # Generate query vector and normalize
        query_vector = self.embeddings.embed_query(query)
        normalized_query = self.normalize_vector(query_vector)
        
        # Perform vector similarity search using dot product (cosine similarity since vectors are normalized)
        pipeline = [
            {
                "$addFields": {
                    "similarity": {
                        "$reduce": {
                            "input": {"$range": [0, {"$size": "$vector"}]},
                            "initialValue": 0,
                            "in": {
                                "$add": [
                                    "$$value",
                                    {"$multiply": [
                                        {"$arrayElemAt": ["$vector", "$$this"]},
                                        {"$arrayElemAt": [normalized_query, "$$this"]}
                                    ]}
                                ]
                            }
                        }
                    }
                }
            },
            {"$match": {"similarity": {"$gt": 0.5}}},  # Filter low similarity matches
            {"$sort": {"similarity": -1}},
            {"$limit": k}
        ]
        
        results = list(self.collection.aggregate(pipeline))
        
        # Convert to Document objects
        documents = []
        for result in results:
            doc = Document(
                page_content=result["content"],
                metadata=result["metadata"]
            )
            documents.append(doc)
            
        return documents
    
    def setup_qa_chain(self, llm=None):
        """Setup QA chain using the vector store."""
        if not llm:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo-16k",
                temperature=0.2,
                max_tokens=800,
                presence_penalty=0.3,
                frequency_penalty=0.3
            )
            
        # Custom prompt template
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer concise and focused on the question.
        Always cite the page and section numbers in your answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer in this format:
        1. Direct answer to the question
        2. Page and section references
        3. Any relevant numerical values or requirements"""
        
        qa_prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self,
            chain_type_kwargs={
                "prompt": qa_prompt,
                "verbose": False
            },
            return_source_documents=True
        )
        
        return qa_chain
        
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Required method for RetrievalQA compatibility."""
        return self.similarity_search(query, k)

if __name__ == "__main__":
    import sys
    import os
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize vector store with specific paths in Downloads directory
    downloads_dir = os.path.dirname(os.path.dirname(__file__))  # Parent of [UCH] FAQ
    index_path = os.path.join(downloads_dir, "vector_store.faiss")
    print(f"Will save vector store to: {index_path}")
    
    vector_store = VectorStore()
    vector_store.index_path = index_path  # Set the path in the instance

    # Train with new documents
    document_paths = [
        r"C:\Users\Souvik Roy\Downloads\[UCH] FAQ\DCA_formatted.txt",
        r"C:\Users\Souvik Roy\Downloads\UCH\[UCH] OCR"
    ]
    
    print("Starting training with new documents...")
    try:
        vector_store.train_with_documents(document_paths)
        print("Training completed successfully!")
        print(f"Vector store saved to: {index_path}")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)
