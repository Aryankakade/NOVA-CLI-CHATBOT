# ml/models/nlp_pipeline.py
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re
import ast
from typing import Dict, List, Optional, Tuple
import spacy
from collections import Counter

class CustomNLPPipeline:
    def __init__(self):
        """
        Specialized NLP pipeline for code analysis and advanced text processing
        Purpose: Better understanding of code-related queries and user sentiment
        """
        print("🔄 Initializing Custom NLP Pipeline...")
        
        # Sentiment analysis for emotional support
        try:
            self.sentiment_model = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            print("✅ Sentiment analysis model loaded")
        except Exception as e:
            print(f"⚠️ Using basic sentiment model: {e}")
            self.sentiment_model = pipeline("sentiment-analysis")
        
        # Load spacy model for advanced NLP (optional)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_available = True
            print("✅ Spacy model loaded for advanced NLP")
        except:
            print("⚠️ Spacy model not available (optional)")
            self.spacy_available = False
        
        # Code analysis patterns
        self.code_patterns = {
            'python': {
                'functions': r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
                'classes': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]',
                'variables': r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=',
                'imports': r'(?:from\s+(\w+(?:\.\w+)*)\s+import|import\s+(\w+(?:\.\w+)*))',
                'decorators': r'@([a-zA-Z_][a-zA-Z0-9_]*)',
                'comments': r'#(.+)',
                'docstrings': r'"""(.*?)"""'
            },
            'javascript': {
                'functions': r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(|([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\(',
                'variables': r'(?:let|const|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                'classes': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*{',
                'imports': r'import\s+.*from\s+[\'"](.+)[\'"]|require\([\'"](.+)[\'"]\)',
                'comments': r'//(.+)|/\*(.*?)\*/'
            }
        }
        
        # Technical term detection
        self.technical_terms = {
            'frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'express', 'spring'],
            'languages': ['python', 'javascript', 'java', 'cpp', 'c++', 'go', 'rust', 'php'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle'],
            'tools': ['docker', 'kubernetes', 'git', 'jenkins', 'terraform', 'aws', 'azure'],
            'concepts': ['api', 'rest', 'graphql', 'microservices', 'devops', 'cicd', 'ml', 'ai']
        }
        
        print("🚀 Custom NLP Pipeline ready!")
    
    def extract_code_entities(self, code: str, language: str = 'python') -> Dict[str, List[str]]:
        """
        Extract functions, classes, variables from code
        """
        if language not in self.code_patterns:
            language = 'python'  # Default to Python
        
        patterns = self.code_patterns[language]
        entities = {key: [] for key in patterns.keys()}
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, code, re.MULTILINE | re.DOTALL)
            
            if entity_type == 'imports':
                # Handle import pattern complexity
                imports = []
                for match in matches:
                    if isinstance(match, tuple):
                        imports.extend([m for m in match if m])
                    else:
                        imports.append(match)
                entities[entity_type] = list(set(imports))
            elif entity_type in ['comments', 'docstrings']:
                entities[entity_type] = [match.strip() for match in matches if match.strip()]
            else:
                # Handle tuple matches (from complex regex)
                if matches and isinstance(matches[0], tuple):
                    flat_matches = [item for sublist in matches for item in sublist if item]
                    entities[entity_type] = list(set(flat_matches))
                else:
                    entities[entity_type] = list(set(matches))
        
        return entities
    
    def analyze_code_complexity(self, code: str, language: str = 'python') -> Dict[str, any]:
        """
        Analyze code complexity and quality metrics
        """
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        comment_lines = [line for line in lines if line.strip().startswith('#' if language == 'python' else '//')]
        
        complexity_metrics = {
            'total_lines': len(lines),
            'code_lines': len(non_empty_lines),
            'comment_lines': len(comment_lines),
            'comment_ratio': len(comment_lines) / max(len(non_empty_lines), 1),
            'avg_line_length': sum(len(line) for line in non_empty_lines) / max(len(non_empty_lines), 1)
        }
        
        if language == 'python':
            try:
                # Parse AST for deeper analysis
                tree = ast.parse(code)
                
                # Count different node types
                node_counts = {}
                for node in ast.walk(tree):
                    node_type = type(node).__name__
                    node_counts[node_type] = node_counts.get(node_type, 0) + 1
                
                # Calculate cyclomatic complexity
                complexity_score = (
                    node_counts.get('For', 0) + 
                    node_counts.get('While', 0) + 
                    node_counts.get('If', 0) + 
                    node_counts.get('Try', 0) + 
                    node_counts.get('With', 0)
                )
                
                complexity_metrics.update({
                    'functions_count': node_counts.get('FunctionDef', 0),
                    'classes_count': node_counts.get('ClassDef', 0),
                    'complexity_score': complexity_score,
                    'ast_nodes': len(list(ast.walk(tree))),
                    'quality_rating': self._get_quality_rating(complexity_score, len(non_empty_lines))
                })
                
            except Exception as e:
                complexity_metrics.update({
                    'complexity_score': 0,
                    'quality_rating': 'unknown',
                    'parse_error': str(e)
                })
        
        return complexity_metrics
    
    def _get_quality_rating(self, complexity: int, lines: int) -> str:
        """Determine code quality based on complexity and size"""
        if lines < 20:
            return 'simple'
        elif complexity < 5:
            return 'clean'
        elif complexity < 10:
            return 'moderate'
        elif complexity < 20:
            return 'complex'
        else:
            return 'very_complex'
    
    def analyze_query_sentiment(self, text: str) -> Dict[str, any]:
        """
        Analyze emotional tone of user query with enhanced insights
        """
        # Basic sentiment analysis
        result = self.sentiment_model(text)[0]
        
        # Enhanced emotion mapping
        emotion_mapping = {
            'POSITIVE': 'confident',
            'NEGATIVE': 'frustrated',
            'NEUTRAL': 'neutral'
        }
        
        # Detect specific emotional indicators
        frustration_indicators = ['stuck', 'error', 'broken', 'help', 'issue', 'problem', 'fail']
        confidence_indicators = ['excited', 'great', 'awesome', 'perfect', 'excellent']
        urgency_indicators = ['urgent', 'asap', 'quickly', 'immediately', 'deadline']
        
        text_lower = text.lower()
        frustration_score = sum(1 for word in frustration_indicators if word in text_lower)
        confidence_score = sum(1 for word in confidence_indicators if word in text_lower)
        urgency_score = sum(1 for word in urgency_indicators if word in text_lower)
        
        # Advanced analysis with spacy if available
        advanced_analysis = {}
        if self.spacy_available:
            doc = self.nlp(text)
            advanced_analysis = {
                'entities': [(ent.text, ent.label_) for ent in doc.ents],
                'key_phrases': [chunk.text for chunk in doc.noun_chunks][:5],
                'technical_terms': self._extract_technical_terms(text)
            }
        
        return {
            'sentiment': result['label'],
            'confidence': result['score'],
            'emotion': emotion_mapping.get(result['label'], 'neutral'),
            'frustration_level': frustration_score,
            'confidence_level': confidence_score,
            'urgency_level': urgency_score,
            'needs_emotional_support': (
                result['label'] == 'NEGATIVE' and result['score'] > 0.7
            ) or frustration_score > 2,
            'needs_urgent_help': urgency_score > 0,
            'advanced_analysis': advanced_analysis
        }
    
    def _extract_technical_terms(self, text: str) -> Dict[str, List[str]]:
        """Extract technical terms from text"""
        text_lower = text.lower()
        found_terms = {}
        
        for category, terms in self.technical_terms.items():
            found = [term for term in terms if term in text_lower]
            if found:
                found_terms[category] = found
        
        return found_terms
    
    def extract_query_intent_keywords(self, text: str) -> Dict[str, any]:
        """
        Extract keywords that help with intent classification
        """
        # Action words for different intents
        action_patterns = {
            'coding': ['write', 'code', 'implement', 'debug', 'fix', 'create', 'build', 'develop'],
            'learning': ['learn', 'understand', 'explain', 'teach', 'tutorial', 'guide', 'how'],
            'career': ['interview', 'job', 'resume', 'career', 'salary', 'promotion', 'hire'],
            'help': ['help', 'assist', 'support', 'stuck', 'problem', 'issue', 'error'],
            'analysis': ['analyze', 'review', 'check', 'evaluate', 'assess', 'examine']
        }
        
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, keywords in action_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Extract main subject/object of the query
        subject_patterns = [
            r'(?:write|create|build|make)\s+(?:a\s+)?(\w+(?:\s+\w+){0,2})',
            r'(?:help|assist).*?with\s+(\w+(?:\s+\w+){0,2})',
            r'(?:about|regarding|concerning)\s+(\w+(?:\s+\w+){0,2})'
        ]
        
        subjects = []
        for pattern in subject_patterns:
            matches = re.findall(pattern, text_lower)
            subjects.extend(matches)
        
        return {
            'intent_keywords': intent_scores,
            'main_subjects': list(set(subjects)),
            'technical_context': self._extract_technical_terms(text),
            'query_type': max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else 'general'
        }
    
    def get_pipeline_stats(self) -> Dict:
        """Get NLP pipeline statistics and capabilities"""
        return {
            'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'spacy_available': self.spacy_available,
            'supported_languages': list(self.code_patterns.keys()),
            'technical_categories': list(self.technical_terms.keys()),
            'capabilities': [
                'code_entity_extraction',
                'complexity_analysis', 
                'sentiment_analysis',
                'technical_term_detection',
                'intent_keyword_extraction'
            ]
        }
