import json
from typing import Dict, List, Tuple, Set
import re
from collections import Counter
from datetime import datetime
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
from textblob import TextBlob
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Pattern
import re
from collections import defaultdict

class PatternStore:
    def __init__(self, json_path: str = None):
        if json_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.json_path = os.path.join(current_dir, "patterns.json")
        else:
            self.json_path = json_path
            
        # Load spaCy model with only necessary components
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])
        
        # Initialize or load the JSON store
        self.setup_store()
        
    def setup_store(self):
        """Initialize the JSON store if it doesn't exist."""
        if not os.path.exists(self.json_path):
            initial_data = {
                "patterns": [],
                "pattern_clusters": []
            }
            with open(self.json_path, 'w') as f:
                json.dump(initial_data, f, indent=4)

    def _load_store(self):
        """Load the JSON store."""
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                return json.load(f)
        return {"patterns": [], "pattern_clusters": []}

    def _save_store(self, data):
        """Save to the JSON store."""
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=4)

    def _extract_semantic_patterns(self, text: str) -> List[Tuple[str, float, str]]:
        """Extract semantic patterns using NLP."""
        doc = self.nlp(text)
        patterns = []
        
        # Entity-based patterns (optimized)
        entity_labels = {'ORG', 'PERSON', 'GPE', 'MONEY', 'PERCENT', 'DATE'}
        patterns.extend(
            (f'\\b{re.escape(ent.text)}\\b', 0.7, ent.label_)
            for ent in doc.ents
            if ent.label_ in entity_labels
        )
        
        # Use compiled regex patterns for better performance
        numerical_pattern = re.compile(r'\b\d+(?:[\.,]\d+)?(?:\s*(?:Crore|Lakh|Million|Billion|%|Rs\.?))?\b')
        date_pattern = re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b')
        requirement_pattern = re.compile(r'\b(?:shall|must|required|minimum|at least|not more than|maximum)\s+(?:\w+\s+){0,3}(?:\d+(?:\.\d+)?(?:\s*%)?|\w+)\b')
        title_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*(?::|\.)')
        
        # Add matches from compiled patterns
        patterns.extend((m.group(), 0.8, 'NUMERICAL') for m in numerical_pattern.finditer(text))
        patterns.extend((m.group(), 0.8, 'DATE') for m in date_pattern.finditer(text))
        patterns.extend((m.group(), 0.9, 'REQUIREMENT') for m in requirement_pattern.finditer(text))
        patterns.extend((m.group(), 0.7, 'TITLE') for m in title_pattern.finditer(text))

        return patterns

    def _find_recurring_phrases(self, text: str) -> List[Tuple[str, float]]:
        """Find recurring phrases using regex instead of TextBlob for better performance."""
        # Simple phrase extraction using regex
        phrase_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+(?:of|in|for|to|from|with|by|at|on|the|a|an)\s+)?[A-Z][a-z]+\b')
        phrases = phrase_pattern.findall(text)
        
        # Count phrases
        phrase_counts = {}
        for phrase in phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # Convert to patterns
        return [
            (f'\\b{re.escape(phrase)}\\b', min(0.5 + (count * 0.1), 0.9))
            for phrase, count in phrase_counts.items()
            if count > 1
        ]

    def learn_patterns(self, text: str, existing_matches: Dict[str, List[str]]) -> None:
        """Learn new patterns from document text with optimized processing."""
        store_data = self._load_store()
        
        # Process existing matches in batches
        batch_size = 5
        for section_type, matches in existing_matches.items():
            for i in range(0, len(matches), batch_size):
                batch = matches[i:i + batch_size]
                for match in batch:
                    pattern_pos = text.find(match)
                    if pattern_pos != -1:
                        # Get smaller context window
                        start = max(0, pattern_pos - 50)
                        end = min(len(text), pattern_pos + len(match) + 50)
                        context = text[start:end]
                        
                        # Extract patterns
                        semantic_patterns = self._extract_semantic_patterns(context)
                        recurring_patterns = self._find_recurring_phrases(context)
                        
                        # Store patterns
                        self._store_patterns(
                            section_type, 
                            semantic_patterns + [(p, c, 'PHRASE') for p, c in recurring_patterns],
                            store_data
                        )
        
        # Save changes
        self._save_store(store_data)

    def _determine_section_type(self, text: str, existing_matches: Dict[str, List[str]]) -> str:
        """Determine the most likely section type for a piece of text."""
        max_score = 0
        best_type = None
        
        for section_type, matches in existing_matches.items():
            score = 0
            for match in matches:
                if match.lower() in text.lower():
                    score += 1
            if score > max_score:
                max_score = score
                best_type = section_type
        
        return best_type if max_score > 0 else "general"

    def _store_patterns(self, section_type: str, patterns: List[Tuple[str, float, str]], store_data: dict) -> None:
        """Store patterns in the JSON store."""
        now = datetime.now().isoformat()
        
        for pattern, confidence, semantic_type in patterns:
            # Extract common context words
            doc = self.nlp(pattern)
            context_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
            
            # Create pattern object
            pattern_obj = {
                "section_type": section_type,
                "pattern": pattern,
                "frequency": 1,
                "confidence": confidence,
                "context_words": context_words,
                "semantic_type": semantic_type,
                "last_used": now,
                "created_at": now
            }
            
            # Check if pattern already exists
            existing_pattern = next(
                (p for p in store_data["patterns"] 
                 if p["section_type"] == section_type and p["pattern"] == pattern),
                None
            )
            
            if existing_pattern:
                existing_pattern["frequency"] += 1
                existing_pattern["last_used"] = now
                existing_pattern["confidence"] = max(existing_pattern["confidence"], confidence)
            else:
                store_data["patterns"].append(pattern_obj)

    def _update_pattern_clusters(self, store_data: dict) -> None:
        """Update pattern clusters in the JSON store."""
        # Get all patterns
        patterns = [p["pattern"] for p in store_data["patterns"]]
        
        # Cluster patterns
        clusters = self._cluster_similar_patterns(patterns)
        
        # Update clusters in store
        store_data["pattern_clusters"] = []
        for i, cluster in enumerate(clusters):
            cluster_obj = {
                "cluster_name": f"cluster_{i}",
                "patterns": cluster,
                "confidence": 0.7,
                "created_at": datetime.now().isoformat()
            }
            store_data["pattern_clusters"].append(cluster_obj)

    def get_patterns(self, section_type: str = None) -> List[Dict]:
        """Get patterns from the JSON store."""
        store_data = self._load_store()
        if section_type:
            return [p for p in store_data["patterns"] if p["section_type"] == section_type]
        return store_data["patterns"]

    def get_clusters(self) -> List[Dict]:
        """Get pattern clusters from the JSON store."""
        store_data = self._load_store()
        return store_data["pattern_clusters"]

    def merge_with_default_patterns(self, default_patterns: Dict[str, str]) -> Dict[str, str]:
        """Merge learned patterns with default patterns."""
        learned_patterns = self.get_patterns()
        merged_patterns = default_patterns.copy()
        
        for pattern in learned_patterns:
            section_type = pattern["section_type"]
            if section_type in merged_patterns:
                # Combine patterns using regex OR operator
                existing_pattern = merged_patterns[section_type].strip('(?i)()')
                new_pattern = pattern["pattern"]
                merged_patterns[section_type] = f'(?i)({existing_pattern}|{new_pattern})'
            else:
                merged_patterns[section_type] = f'(?i)({new_pattern})'
        
        return merged_patterns

    def _cluster_similar_patterns(self, patterns: List[str]) -> List[List[str]]:
        """Cluster similar patterns using DBSCAN."""
        if not patterns:
            return []
            
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
        pattern_vectors = vectorizer.fit_transform([p for p in patterns])
        
        # Cluster patterns
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        clusters = clustering.fit_predict(pattern_vectors.toarray())
        
        # Group patterns by cluster
        clustered_patterns = {}
        for pattern, cluster_id in zip(patterns, clusters):
            if cluster_id != -1:  # Ignore noise
                if cluster_id not in clustered_patterns:
                    clustered_patterns[cluster_id] = []
                clustered_patterns[cluster_id].append(pattern)
        
        return list(clustered_patterns.values())

    def validate_pattern_context(self, pattern: Dict[str, Any], context: str) -> bool:
        """Validate if a pattern appears in the correct context."""
        # Check if pattern appears with expected surrounding terms
        expected_terms = pattern.get('expected_context', [])
        if expected_terms:
            context_terms = set(re.findall(r'\w+', context.lower()))
            if not any(term in context_terms for term in expected_terms):
                return False
        
        # Validate numerical values if pattern requires them
        if pattern.get('requires_numbers', False):
            if not re.search(r'\d+', context):
                return False
        
        # Check for requirement language
        if pattern.get('is_requirement', False):
            requirement_terms = ['shall', 'must', 'required', 'mandatory', 'essential']
            if not any(term in context.lower() for term in requirement_terms):
                return False
        
        return True

    def extract_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract validated patterns from text."""
        patterns = []
        
        # Split text into sentences for better context analysis
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Look for key patterns
            for pattern_type, pattern_info in self.patterns.items():
                matches = pattern_info['regex'].finditer(sentence)
                
                for match in matches:
                    # Get surrounding context (up to 100 chars before and after)
                    start = max(0, match.start() - 100)
                    end = min(len(sentence), match.end() + 100)
                    context = sentence[start:end]
                    
                    # Validate pattern in context
                    if self.validate_pattern_context(pattern_info, context):
                        pattern = {
                            'type': pattern_type,
                            'value': match.group(),
                            'context': context.strip(),
                            'confidence': pattern_info.get('confidence', 0.5)
                        }
                        patterns.append(pattern)
        
        return patterns

    def _update_pattern_clusters(self, new_patterns: List[Dict[str, Any]]):
        """Update pattern clusters with validation."""
        for pattern in new_patterns:
            pattern_type = pattern['type']
            pattern_value = pattern['value']
            
            # Skip patterns with low confidence
            if pattern.get('confidence', 0) < 0.3:
                continue
                
            if pattern_type not in self.pattern_clusters:
                self.pattern_clusters[pattern_type] = {}
                
            if pattern_value not in self.pattern_clusters[pattern_type]:
                self.pattern_clusters[pattern_type][pattern_value] = {
                    'count': 0,
                    'contexts': [],
                    'confidence': 0
                }
                
            cluster = self.pattern_clusters[pattern_type][pattern_value]
            cluster['count'] += 1
            
            # Add new context if it's different enough from existing ones
            new_context = pattern['context']
            if not any(self._context_similarity(new_context, ctx) > 0.8 
                      for ctx in cluster['contexts']):
                cluster['contexts'].append(new_context)
            
            # Update confidence based on frequency and context diversity
            cluster['confidence'] = min(1.0, (
                cluster['count'] / 10.0 +  # Frequency factor
                len(cluster['contexts']) / 5.0 +  # Context diversity factor
                pattern.get('confidence', 0.5)  # Individual pattern confidence
            ) / 3)

    def _context_similarity(self, context1: str, context2: str) -> float:
        """Calculate similarity between two contexts."""
        # Simple implementation using Jaccard similarity
        set1 = set(re.findall(r'\w+', context1.lower()))
        set2 = set(re.findall(r'\w+', context2.lower()))
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union)

class LocalPatternStore:
    LEGAL_CLAUSE_PATTERNS = {
        # General & Administrative
        'bid_validity': {
            'patterns': [
                r'bid\s+validity\s+(?:period|duration)',
                r'tender\s+valid(?:ity)?\s+for\s+\d+\s+days?',
                r'offer\s+shall?\s+remain\s+valid'
            ],
            'context_window': 100
        },
        'pre_bid_meeting': {
            'patterns': [
                r'pre[-\s]bid\s+meeting\s+(?:shall|will)',
                r'pre[-\s]tender\s+conference',
                r'bidders?\s+conference\s+(?:shall|will)'
            ],
            'context_window': 150
        },
        'addendum_corrigendum': {
            'patterns': [
                r'addendum(?:s)?\s+(?:shall|will|may)',
                r'corrigendum(?:s)?\s+(?:shall|will|may)',
                r'amendment(?:s)?\s+to\s+(?:tender|bid)'
            ],
            'context_window': 100
        },

        # Financial & Security
        'earnest_money': {
            'patterns': [
                r'earnest\s+money\s+deposit\s+(?:\(EMD\))?\s+of\s+Rs\.?\s*[\d,.]+',
                r'EMD\s+(?:amount|sum|payment)\s+of\s+Rs\.?\s*[\d,.]+',
                r'bid\s+security\s+of\s+Rs\.?\s*[\d,.]+'
            ],
            'context_window': 200
        },
        'performance_security': {
            'patterns': [
                r'performance\s+(?:security|guarantee|bond)\s+of\s+(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)',
                r'contract\s+performance\s+guarantee\s+(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)',
                r'performance\s+bank\s+guarantee\s+(?:PBG)'
            ],
            'context_window': 200
        },
        'security_deposit': {
            'patterns': [
                r'security\s+deposit\s+(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)',
                r'retention\s+money\s+(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)',
                r'retention\s+amount\s+(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)'
            ],
            'context_window': 150
        },

        # Project Timeline
        'completion_period': {
            'patterns': [
                r'completion\s+period\s+(?:of\s+)?\d+\s+(?:days?|months?|years?)',
                r'contract\s+duration\s+(?:of\s+)?\d+\s+(?:days?|months?|years?)',
                r'time\s+for\s+completion\s+(?:of\s+)?\d+\s+(?:days?|months?|years?)'
            ],
            'context_window': 150
        },
        'mobilization_advance': {
            'patterns': [
                r'mobilization\s+advance\s+(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)',
                r'advance\s+payment\s+(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)',
                r'initial\s+advance\s+(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)'
            ],
            'context_window': 200
        },
        'liquidated_damages': {
            'patterns': [
                r'liquidated\s+damages?\s+(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)',
                r'delay\s+damages?\s+(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)',
                r'compensation\s+for\s+delay\s+(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)'
            ],
            'context_window': 150
        },

        # Legal & Compliance
        'dispute_resolution': {
            'patterns': [
                r'dispute\s+resolution\s+(?:mechanism|procedure|process)',
                r'arbitration\s+(?:clause|procedure|process)',
                r'settlement\s+of\s+disputes?\s+(?:shall|will)'
            ],
            'context_window': 300
        },
        'force_majeure': {
            'patterns': [
                r'force\s+majeure\s+(?:clause|event|condition)',
                r'act\s+of\s+god',
                r'unforeseen\s+circumstances?\s+beyond\s+(?:the\s+)?control'
            ],
            'context_window': 250
        },
        'insurance': {
            'patterns': [
                r'contractor\'?s?\s+all\s+risk\s+(?:insurance|policy)',
                r'CAR\s+policy',
                r'workmen\'?s?\s+compensation\s+insurance'
            ],
            'context_window': 200
        },

        # Price & Cost
        'price_variation': {
            'patterns': [
                r'price\s+variation\s+(?:clause|formula)',
                r'cost\s+adjustment\s+(?:clause|formula)',
                r'escalation\s+(?:clause|formula)'
            ],
            'context_window': 300
        },
        'taxes_duties': {
            'patterns': [
                r'taxes?\s+and\s+duties?\s+(?:shall|will)',
                r'GST\s+(?:shall|will)',
                r'statutory\s+(?:levy|payment|deduction)'
            ],
            'context_window': 200
        },

        # Safety & Environmental
        'safety_requirements': {
            'patterns': [
                r'safety\s+(?:requirements?|measures?|protocols?)',
                r'health\s+and\s+safety\s+(?:requirements?|measures?)',
                r'EHS\s+(?:requirements?|guidelines?)'
            ],
            'context_window': 250
        },
        'environmental_compliance': {
            'patterns': [
                r'environmental\s+(?:compliance|protection|safeguards?)',
                r'pollution\s+control\s+measures?',
                r'eco[-\s]friendly\s+practices?'
            ],
            'context_window': 250
        }
    }

    PRIORITY_CLAUSES = {
        'earnest_money': {
            'patterns': [
                r'earnest\s+money\s+deposit\s*(?:\(EMD\))?\s*(?:of\s+)?(?:Rs\.?\s*[\d,.]+|\d+(?:\.\d+)?%)',
                r'EMD\s+(?:amount|sum|payment)\s*(?:of\s+)?(?:Rs\.?\s*[\d,.]+|\d+(?:\.\d+)?%)',
                r'bid\s+security\s*(?:of\s+)?(?:Rs\.?\s*[\d,.]+|\d+(?:\.\d+)?%)'
            ],
            'context_window': 200,
            'must_highlight': True
        },
        'emd_submission_mode': {
            'patterns': [
                r'EMD\s+(?:shall|should|must|to)\s+be\s+submitted\s+(?:through|via|by|in\s+form\s+of)',
                r'mode\s+of\s+EMD\s+submission',
                r'earnest\s+money\s+(?:shall|should|must|to)\s+be\s+(?:paid|submitted|deposited)'
            ],
            'context_window': 200,
            'must_highlight': True
        },
        'completion_period': {
            'patterns': [
                r'completion\s+period\s*(?:of\s+)?\d+\s+(?:days?|months?|years?)',
                r'contract\s+duration\s*(?:of\s+)?\d+\s+(?:days?|months?|years?)',
                r'time\s+for\s+completion\s*(?:of\s+)?\d+\s+(?:days?|months?|years?)'
            ],
            'context_window': 150,
            'must_highlight': True
        },
        'mobilization_advance': {
            'patterns': [
                r'mobilization\s+advance\s*(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)',
                r'advance\s+payment\s*(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)',
                r'initial\s+advance\s*(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)'
            ],
            'context_window': 200,
            'must_highlight': True
        },
        'security_deposit': {
            'patterns': [
                r'security\s+deposit\s*(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)',
                r'retention\s+money\s*(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)',
                r'retention\s+amount\s*(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)'
            ],
            'context_window': 150,
            'must_highlight': True
        },
        'defect_liability': {
            'patterns': [
                r'defect\s+liability\s+period\s*(?:of\s+)?\d+\s+(?:days?|months?|years?)',
                r'DLP\s*(?:of\s+)?\d+\s+(?:days?|months?|years?)',
                r'maintenance\s+period\s*(?:of\s+)?\d+\s+(?:days?|months?|years?)'
            ],
            'context_window': 200,
            'must_highlight': True
        },
        'performance_security': {
            'patterns': [
                r'performance\s+(?:security|guarantee|bond)\s*(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)',
                r'contract\s+performance\s+guarantee\s*(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)',
                r'performance\s+bank\s+guarantee\s*(?:PBG)?\s*(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)'
            ],
            'context_window': 200,
            'must_highlight': True
        },
        'performance_security_mode': {
            'patterns': [
                r'performance\s+(?:security|guarantee)\s+(?:shall|should|must|to)\s+be\s+(?:submitted|provided)\s+(?:through|via|by|in\s+form\s+of)',
                r'mode\s+of\s+performance\s+(?:security|guarantee)\s+submission',
                r'performance\s+bank\s+guarantee\s+(?:shall|should|must|to)\s+be\s+(?:submitted|provided)'
            ],
            'context_window': 200,
            'must_highlight': True
        },
        'price_variation': {
            'patterns': [
                r'price\s+variation\s+(?:clause|formula)',
                r'cost\s+adjustment\s+(?:clause|formula)',
                r'escalation\s+(?:clause|formula)',
                r'price\s+adjustment\s+(?:clause|formula)'
            ],
            'context_window': 300,
            'must_highlight': True
        },
        'incentive_bonus': {
            'patterns': [
                r'incentive\s+(?:clause|payment)\s*(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)',
                r'bonus\s+(?:clause|payment)\s*(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)',
                r'early\s+completion\s+(?:bonus|incentive)\s*(?:of\s+)?(?:\d+(?:\.\d+)?%|Rs\.?\s*[\d,.]+)'
            ],
            'context_window': 200,
            'must_highlight': True
        }
    }

    CLAUSE_CATEGORIES = {
        'general_administrative': [
            'bid_validity', 'pre_bid_meeting', 'addendum_corrigendum',
            'joint_venture', 'subcontracting', 'debarment'
        ],
        'financial_security': [
            'earnest_money', 'performance_security', 'security_deposit',
            'bank_guarantee', 'payment_terms'
        ],
        'project_timeline': [
            'completion_period', 'mobilization_advance', 'extension_time',
            'liquidated_damages'
        ],
        'technical_quality': [
            'technical_specifications', 'quality_assurance', 'material_standards'
        ],
        'legal_compliance': [
            'dispute_resolution', 'force_majeure', 'labour_laws', 'insurance'
        ],
        'price_cost': [
            'price_variation', 'taxes_duties'
        ],
        'safety_environmental': [
            'safety_requirements', 'environmental_compliance'
        ]
    }

    SECTION_TYPE_INDICATORS = {
        'clause': [
            r'clause\s+\d+',
            r'article\s+\d+',
            r'section\s+\d+'
        ],
        'condition': [
            r'general\s+conditions?',
            r'special\s+conditions?',
            r'contract\s+conditions?'
        ],
        'instruction': [
            r'instructions?\s+to\s+bidders?',
            r'bid\s+instructions?',
            r'tender\s+instructions?'
        ],
        'legal': [
            r'legal\s+(?:requirements?|obligations?)',
            r'statutory\s+(?:requirements?|compliance)',
            r'regulatory\s+(?:requirements?|compliance)'
        ]
    }

    COMMON_QUESTIONS = {
        'technical_qualification': {
            'patterns': [
                r'technical\s+qualification\s+requirements?',
                r'technical\s+(?:eligibility|criteria)',
                r'what\s+(?:are|is)\s+(?:the\s+)?technical\s+(?:qualification|requirements?)'
            ],
            'key_patterns': {
                'bid_capacity': r'BID\s+capacity\s+(?:is|must\s+be)\s+more\s+than\s+(?:the\s+)?total\s+BID\s+value',
                'experience_value': r'sum\s+total\s+thereof.*?is\s+more\s+than\s+Rs\.\s*[\d.,]+\s+(?:Crore|Lakh|Only)',
                'similar_work': r'at\s+least\s+one\s+similar\s+work\s+of\s+20%.*?Rs\.\s*[\d.,]+\s+(?:Crore|Lakh|Only)',
                'span_requirement': r'longest\s+span\s+(?:of\s+)?Bridge/ROB/flyover.*?(?:\d+\s*m)'
            },
            'context_window': 300,
            'reference': '(Page No 24, Clause 2.2.2.2)'
        },
        'financial_qualification': {
            'patterns': [
                r'financial\s+qualification\s+requirements?',
                r'financial\s+(?:eligibility|criteria)',
                r'what\s+(?:are|is)\s+(?:the\s+)?financial\s+(?:qualification|requirements?)'
            ],
            'key_patterns': {
                'net_worth': r'minimum\s+Net\s+Worth.*?Rs\.\s*[\d.,]+\s+(?:Crore|Lakh|Only)',
                'turnover': r'minimum\s+Average\s+Annual\s+Turnover.*?Rs\.\s*[\d.,]+\s+(?:Crore|Lakh|Only)'
            },
            'context_window': 250,
            'reference': '(Page No 25, Clause 2.2.2.3)'
        },
        'joint_venture': {
            'patterns': [
                r'(?:criteria|requirements?)\s+for\s+(?:a\s+)?Joint\s+Venture',
                r'JV\s+(?:criteria|requirements?)',
                r'what\s+(?:are|is)\s+(?:the\s+)?(?:criteria|requirements?)\s+for\s+(?:a\s+)?(?:JV|Joint\s+Venture)'
            ],
            'key_patterns': {
                'lead_member': r'Lead\s+Member\s+must\s+meet\s+at\s+least\s+60%',
                'other_members': r'Other\s+Members\s+must\s+meet\s+at\s+least\s+20%',
                'collective': r'collectively\s+meet\s+100%',
                'project_length': r'Lead\s+Member\s+shall\s+undertake\s+at\s+least\s+51%'
            },
            'context_window': 300,
            'reference': '(Page No 15, Clause 2.1.11)'
        },
        'technical_specifications': {
            'patterns': [
                r'technical\s+specifications?',
                r'what\s+(?:are|is)\s+(?:the\s+)?technical\s+specifications?'
            ],
            'key_patterns': {
                'schedule_d': r'Schedule\s+D.*?(?:construction|structural|materials|design|environmental)',
                'schedules': r'Technical\s+Schedules\s+A\s+to\s+D'
            },
            'context_window': 200,
            'reference': '(Page No 46, Technical Schedules A to D, Schedule D)'
        }
    }

    def find_common_question_answer(self, text: str, question: str) -> Optional[Dict[str, Any]]:
        """Find answers to common questions using predefined patterns.
        
        Args:
            text: The document text to search in
            question: The question being asked
            
        Returns:
            Dictionary with answer details if found, None otherwise
        """
        for q_type, q_info in self.COMMON_QUESTIONS.items():
            # Check if question matches any of the patterns
            if any(re.search(pattern, question, re.IGNORECASE) for pattern in q_info['patterns']):
                matches = {}
                
                # Find matches for each key pattern
                for key, pattern in q_info['key_patterns'].items():
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        # Get surrounding context
                        start = max(0, match.start() - q_info['context_window'])
                        end = min(len(text), match.end() + q_info['context_window'])
                        context = text[start:end]
                        
                        matches[key] = {
                            'matched_text': match.group(),
                            'context': context
                        }
                
                if matches:
                    return {
                        'question_type': q_type,
                        'matches': matches,
                        'reference': q_info.get('reference', ''),
                        'confidence': self._calculate_answer_confidence(matches, q_info['key_patterns'])
                    }
        
        return None

    def _calculate_answer_confidence(self, matches: Dict[str, Dict], key_patterns: Dict[str, str]) -> float:
        """Calculate confidence score for question answer based on pattern matches.
        
        Args:
            matches: Dictionary of matched patterns and their contexts
            key_patterns: Dictionary of key patterns that define the answer
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base score from number of matched patterns
        pattern_matches = len(matches)
        total_patterns = len(key_patterns)
        base_score = pattern_matches / total_patterns
        
        # Boost score based on presence of numerical values
        has_numbers = any(
            bool(re.search(r'(?:Rs\.?\s*[\d,.]+|\d+(?:\.\d+)?%|\d+\s+(?:days?|months?|years?))', 
                         match['matched_text'])) 
            for match in matches.values()
        )
        number_boost = 0.2 if has_numbers else 0
        
        # Boost score based on presence of reference markers
        reference_keywords = ['page', 'clause', 'section', 'schedule']
        has_reference = any(
            any(kw in match['context'].lower() for kw in reference_keywords)
            for match in matches.values()
        )
        reference_boost = 0.1 if has_reference else 0
        
        # Calculate final score with boosts
        confidence = min(1.0, base_score + number_boost + reference_boost)
        
        return confidence

    def __init__(self, patterns_path: str = "patterns.pkl"):
        self.patterns_path = Path(patterns_path)
        self.patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.compiled_patterns: Dict[str, Pattern] = {}
        self.load_if_exists()
        self.patterns = {
            'technical_qualification': {
                'regex': re.compile(
                    r'(?:technical|qualification|experience|expertise|capability)'
                    r'(?:[^.]*?(?:required|must|shall|should|minimum))[^.]*?'
                    r'(?:\d+(?:\s*years?|\s*projects?|\s*works?)|'
                    r'similar\s+works?|equivalent\s+works?)',
                    re.IGNORECASE
                ),
                'expected_context': ['experience', 'technical', 'qualification', 'requirement'],
                'requires_numbers': True,
                'is_requirement': True,
                'confidence': 0.8
            },
            'financial_qualification': {
                'regex': re.compile(
                    r'(?:financial|turnover|net\s*worth|revenue|profit)'
                    r'(?:[^.]*?(?:required|must|shall|should|minimum))[^.]*?'
                    r'(?:Rs\.?\s*\d+(?:\s*crores?|\s*lakhs?)?|\d+(?:\s*crores?|\s*lakhs?)?)',
                    re.IGNORECASE
                ),
                'expected_context': ['financial', 'turnover', 'worth', 'revenue'],
                'requires_numbers': True,
                'is_requirement': True,
                'confidence': 0.8
            },
            'jv_criteria': {
                'regex': re.compile(
                    r'(?:joint\s*venture|JV|consortium)'
                    r'(?:[^.]*?(?:partner|member|share|lead|responsibility))[^.]*?'
                    r'(?:\d+\s*%|\d+\s*percent|lead\s*member)',
                    re.IGNORECASE
                ),
                'expected_context': ['joint venture', 'JV', 'consortium', 'partner'],
                'requires_numbers': True,
                'is_requirement': True,
                'confidence': 0.8
            },
            'technical_specs': {
                'regex': re.compile(
                    r'(?:technical\s*specifications?|specifications?|technical\s*requirements?)'
                    r'(?:[^.]*?(?:shall|must|should|will))[^.]*?'
                    r'(?:\d+(?:\s*\w+)?|standards?|requirements?)',
                    re.IGNORECASE
                ),
                'expected_context': ['specification', 'technical', 'requirement', 'standard'],
                'requires_numbers': False,
                'is_requirement': True,
                'confidence': 0.7
            },
            'important_clauses': {
                'regex': re.compile(
                    r'(?:clause|article|section)\s*\d+(?:\.\d+)*'
                    r'[^.]*?(?:shall|must|should|will|mandatory|essential)',
                    re.IGNORECASE
                ),
                'expected_context': ['clause', 'article', 'section', 'provision'],
                'requires_numbers': True,
                'is_requirement': True,
                'confidence': 0.7
            }
        }
        self._compile_patterns()

    def load_if_exists(self):
        """Load existing patterns if they exist"""
        if self.patterns_path.exists():
            with open(self.patterns_path, 'rb') as f:
                self.patterns = pickle.load(f)
                # Recompile patterns after loading
                self._compile_patterns()

    def save(self):
        """Save patterns to disk"""
        with open(self.patterns_path, 'wb') as f:
            pickle.dump(self.patterns, f)

    def _compile_patterns(self):
        """Compile regex patterns for efficient matching"""
        self.compiled_patterns = {
            category: re.compile('|'.join(f'(?:{p["pattern"]})' for p in patterns), 
                               re.IGNORECASE | re.MULTILINE)
            for category, patterns in self.patterns.items()
        }

    def add_pattern(self, category: str, pattern: str, metadata: Optional[Dict] = None):
        """Add a new pattern with optional metadata"""
        pattern_dict = {"pattern": pattern, "metadata": metadata or {}}
        self.patterns[category].append(pattern_dict)
        self._compile_patterns()
        self.save()

    def remove_pattern(self, category: str, pattern: str):
        """Remove a pattern from a category"""
        if category in self.patterns:
            self.patterns[category] = [
                p for p in self.patterns[category] 
                if p["pattern"] != pattern
            ]
            if not self.patterns[category]:
                del self.patterns[category]
            self._compile_patterns()
            self.save()

    def find_matches(self, text: str, categories: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """Find all pattern matches in the text for specified categories"""
        results = defaultdict(list)
        search_categories = categories or list(self.patterns.keys())
        
        for category in search_categories:
            if category not in self.compiled_patterns:
                continue
                
            matches = self.compiled_patterns[category].finditer(text)
            for match in matches:
                # Find which pattern matched
                matched_text = match.group(0)
                for pattern_dict in self.patterns[category]:
                    if re.search(pattern_dict["pattern"], matched_text, re.IGNORECASE):
                        results[category].append({
                            "text": matched_text,
                            "span": match.span(),
                            "pattern": pattern_dict["pattern"],
                            "metadata": pattern_dict["metadata"]
                        })
                        break

        return dict(results)

    def clear(self):
        """Clear all patterns"""
        self.patterns.clear()
        self.compiled_patterns.clear()
        self.save()

    def __len__(self):
        return sum(len(patterns) for patterns in self.patterns.values())

    def find_legal_clauses(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Find legal clauses in the text using the predefined patterns.
        
        Args:
            text: The text to search for legal clauses
            
        Returns:
            Dictionary mapping clause categories to lists of found clauses with their details
        """
        results = {
            'priority_clauses': [],  # New category for must-highlight clauses
            'general_administrative': [],
            'financial_security': [],
            'project_timeline': [],
            'technical_quality': [],
            'legal_compliance': [],
            'price_cost': [],
            'safety_environmental': []
        }
        
        # First check priority clauses
        for clause_name, clause_info in self.PRIORITY_CLAUSES.items():
            patterns = clause_info['patterns']
            context_window = clause_info.get('context_window', 200)
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    # Extract the matched text and surrounding context
                    start = max(0, match.start() - context_window)
                    end = min(len(text), match.end() + context_window)
                    context = text[start:end]
                    
                    # Extract any numerical values or requirements
                    values = re.findall(r'(?:Rs\.?\s*[\d,.]+|\d+(?:\.\d+)?%|\d+\s+(?:days?|months?|years?))', context)
                    
                    # Determine section type
                    section_type = None
                    for stype, patterns in self.SECTION_TYPE_INDICATORS.items():
                        if any(re.search(p, context, re.IGNORECASE) for p in patterns):
                            section_type = stype
                            break
                    
                    results['priority_clauses'].append({
                        'clause_name': clause_name,
                        'matched_text': match.group(),
                        'context': context,
                        'values': values,
                        'section_type': section_type,
                        'confidence': self._calculate_clause_confidence(context, patterns),
                        'must_highlight': True
                    })
        
        # Then check other clauses
        for clause_name, clause_info in self.LEGAL_CLAUSE_PATTERNS.items():
            # Skip if this is already a priority clause
            if clause_name in self.PRIORITY_CLAUSES:
                continue
                
            patterns = clause_info['patterns']
            context_window = clause_info.get('context_window', 200)
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    # Extract the matched text and surrounding context
                    start = max(0, match.start() - context_window)
                    end = min(len(text), match.end() + context_window)
                    context = text[start:end]
                    
                    # Determine the category
                    category = None
                    for cat, clauses in self.CLAUSE_CATEGORIES.items():
                        if any(c in clause_name.lower() for c in clauses):
                            category = cat
                            break
                    
                    if category:
                        # Extract any numerical values or requirements
                        values = re.findall(r'(?:Rs\.?\s*[\d,.]+|\d+(?:\.\d+)?%|\d+\s+(?:days?|months?|years?))', context)
                        
                        # Determine section type
                        section_type = None
                        for stype, patterns in self.SECTION_TYPE_INDICATORS.items():
                            if any(re.search(p, context, re.IGNORECASE) for p in patterns):
                                section_type = stype
                                break
                        
                        results[category].append({
                            'clause_name': clause_name,
                            'matched_text': match.group(),
                            'context': context,
                            'values': values,
                            'section_type': section_type,
                            'confidence': self._calculate_clause_confidence(context, patterns),
                            'must_highlight': False
                        })
        
        return results

    def _calculate_clause_confidence(self, context: str, patterns: List[str]) -> float:
        """Calculate confidence score for a clause match based on pattern strength and context.
        
        Args:
            context: The extracted context around the match
            patterns: List of patterns that define this clause type
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base score from pattern matches
        pattern_matches = sum(1 for p in patterns if re.search(p, context, re.IGNORECASE))
        base_score = pattern_matches / len(patterns)
        
        # Boost score based on presence of numerical values
        has_numbers = bool(re.search(r'(?:Rs\.?\s*[\d,.]+|\d+(?:\.\d+)?%|\d+\s+(?:days?|months?|years?))', context))
        number_boost = 0.2 if has_numbers else 0
        
        # Boost score based on presence of legal keywords
        legal_keywords = ['shall', 'will', 'must', 'required', 'mandatory', 'subject to']
        keyword_matches = sum(1 for kw in legal_keywords if re.search(r'\b' + kw + r'\b', context, re.IGNORECASE))
        keyword_boost = min(0.3, keyword_matches * 0.1)
        
        # Calculate final score with boosts
        confidence = min(1.0, base_score + number_boost + keyword_boost)
        
        return confidence
