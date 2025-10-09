# src/agents/agent_manager.py
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import logging
from collections import Counter
import re

class AgentManager:
    """Intelligent Agent Manager for NOVA - Routes queries to specialized agents"""
    
    def __init__(self):
        # Specialized agents configuration
        self.agents = self.setup_specialized_agents()
        
        # Agent performance tracking
        self.agent_performance = {}
        self.agent_usage_stats = {}
        self.context_patterns = {}
        
        # Advanced selection algorithms
        self.selection_weights = self.setup_selection_weights()
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        # Learning system
        self.interaction_history = []  # For learning user patterns
        self.agent_success_rates = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        print("✅ Agent Manager initialized with intelligent routing")
    
    def setup_logging(self):
        """Setup performance logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/logs/agent_manager.log'),
                logging.StreamHandler()
            ]
        )
    
    def setup_specialized_agents(self) -> Dict[str, Dict]:
        """Setup configuration for specialized agents"""
        
        return {
            # HIGH-VALUE AGENTS (Core functionality)
            "coding_expert": {
                "description": "Advanced programming and software engineering assistance",
                "specialization": ["programming", "debugging", "architecture", "algorithms"],
                "keywords": {
                    "primary": ["code", "programming", "python", "javascript", "function", "debug", "error", "bug", "algorithm"],
                    "secondary": ["react", "node", "django", "fastapi", "database", "api", "framework"],
                    "context": ["write", "create", "build", "fix", "optimize", "implement", "develop"]
                },
                "complexity_preference": [0.4, 1.0],  # Handles medium to high complexity
                "value_score": 9.8,  # Highest value
                "enhanced_capabilities": [
                    "multi_file_project_generation",
                    "architecture_recommendations", 
                    "performance_optimization",
                    "code_review_and_refactoring",
                    "complex_algorithm_implementation"
                ]
            },
            
            "career_coach": {
                "description": "Professional career development and job search guidance",
                "specialization": ["career_development", "resume_building", "interview_prep", "salary_negotiation"],
                "keywords": {
                    "primary": ["resume", "interview", "job", "career", "salary", "cv", "portfolio"],
                    "secondary": ["promotion", "skills", "experience", "hiring", "application", "linkedin"],
                    "context": ["improve", "help", "review", "prepare", "build", "optimize", "advice"]
                },
                "complexity_preference": [0.2, 0.8],  # Low to medium-high complexity
                "value_score": 9.5,  # High market value
                "enhanced_capabilities": [
                    "industry_specific_guidance",
                    "ats_optimization", 
                    "behavioral_interview_prep",
                    "negotiation_strategies",
                    "career_path_planning"
                ]
            },
            
            "medical_advisor": {
                "description": "Health information and medical guidance",
                "specialization": ["health_analysis", "symptom_assessment", "medical_information"],
                "keywords": {
                    "primary": ["health", "medical", "doctor", "symptoms", "medicine", "treatment", "diagnosis"],
                    "secondary": ["pain", "fever", "headache", "prescription", "hospital", "clinic"],
                    "context": ["feel", "hurt", "sick", "concern", "worried", "check", "analyze"]
                },
                "complexity_preference": [0.3, 0.9],  # Medium to high complexity
                "value_score": 9.0,  # High value with disclaimers
                "enhanced_capabilities": [
                    "symptom_analysis",
                    "health_report_interpretation",
                    "medication_information",
                    "wellness_recommendations",
                    "medical_research_summaries"
                ],
                "disclaimers": ["not_medical_advice", "consult_professional"]
            },
            
            "business_intelligence": {
                "description": "Business analysis and strategic insights",
                "specialization": ["business_analysis", "data_interpretation", "strategic_planning"],
                "keywords": {
                    "primary": ["business", "analytics", "data", "strategy", "kpi", "metrics", "revenue"],
                    "secondary": ["growth", "market", "competition", "roi", "sales", "marketing"],
                    "context": ["analyze", "insights", "trends", "optimize", "improve", "strategy"]
                },
                "complexity_preference": [0.5, 1.0],  # Medium-high to high complexity
                "value_score": 8.5,  # Enterprise value
                "enhanced_capabilities": [
                    "data_visualization_insights",
                    "market_trend_analysis",
                    "competitive_intelligence",
                    "financial_modeling_guidance",
                    "business_process_optimization"
                ]
            },
            
            "emotional_counselor": {
                "description": "Emotional support and mental wellness guidance", 
                "specialization": ["emotional_support", "stress_management", "mental_wellness"],
                "keywords": {
                    "primary": ["stress", "anxiety", "depression", "emotional", "feelings", "mental", "therapy"],
                    "secondary": ["overwhelmed", "sad", "worried", "angry", "lonely", "burnout"],
                    "context": ["feel", "emotional", "help", "support", "cope", "manage", "better"]
                },
                "complexity_preference": [0.2, 0.7],  # Low to medium complexity
                "value_score": 8.0,  # Important for user well-being
                "enhanced_capabilities": [
                    "emotional_intelligence_assessment",
                    "stress_reduction_strategies",
                    "mindfulness_techniques",
                    "cognitive_behavioral_insights",
                    "crisis_support_protocols"
                ],
                "disclaimers": ["not_therapy", "seek_professional_help"]
            },
            
            "technical_architect": {
                "description": "System architecture and technical design guidance",
                "specialization": ["system_design", "architecture", "scalability", "performance"],
                "keywords": {
                    "primary": ["architecture", "system", "design", "scalability", "performance", "infrastructure"],
                    "secondary": ["microservices", "database", "cloud", "security", "deployment"],
                    "context": ["design", "architect", "scale", "optimize", "structure", "plan"]
                },
                "complexity_preference": [0.7, 1.0],  # High complexity only
                "value_score": 9.0,  # High technical value
                "enhanced_capabilities": [
                    "system_architecture_design",
                    "technology_stack_recommendations",
                    "scalability_planning",
                    "security_architecture",
                    "performance_optimization_strategies"
                ]
            },
            
            # GENERAL AGENT (Fallback)
            "general_assistant": {
                "description": "General-purpose AI assistant for diverse queries",
                "specialization": ["general_knowledge", "conversational_ai", "information_retrieval"],
                "keywords": {
                    "primary": ["hello", "hi", "help", "question", "information", "general"],
                    "secondary": ["explain", "what", "how", "why", "tell", "about"],
                    "context": ["help", "assist", "explain", "understand", "learn", "know"]
                },
                "complexity_preference": [0.0, 0.6],  # Low to medium complexity
                "value_score": 6.0,  # Basic but necessary
                "enhanced_capabilities": [
                    "conversational_engagement",
                    "general_information_retrieval",
                    "task_routing_assistance",
                    "basic_problem_solving",
                    "user_guidance"
                ]
            }
        }
    
    def setup_selection_weights(self) -> Dict[str, float]:
        """Setup weights for agent selection algorithm"""
        
        return {
            "keyword_match": 0.35,      # Primary keyword matching
            "context_relevance": 0.25,  # Context understanding
            "complexity_fit": 0.20,     # Query complexity alignment
            "performance_history": 0.15, # Historical performance
            "user_preference": 0.05     # User preference learning
        }
    
    async def select_optimal_agent(self, user_input: str, context: Dict[str, Any] = None,
                                 evolution_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Advanced agent selection with multiple factors"""
        
        start_time = datetime.now()
        
        # Analyze user input
        query_analysis = self.analyze_query_comprehensive(user_input)
        
        # Calculate agent scores
        agent_scores = {}
        selection_details = {}
        
        for agent_name, agent_config in self.agents.items():
            # Calculate multi-factor score
            scores = await self.calculate_agent_score(
                agent_name, agent_config, user_input, query_analysis, context, evolution_data
            )
            
            # Weighted final score
            final_score = sum(
                scores[factor] * self.selection_weights[factor] 
                for factor in self.selection_weights.keys()
                if factor in scores
            )
            
            agent_scores[agent_name] = final_score
            selection_details[agent_name] = scores
        
        # Select best agent
        best_agent = max(agent_scores, key=agent_scores.get)
        confidence = agent_scores[best_agent]
        
        # Determine confidence level
        confidence_level = self.get_confidence_level(confidence)
        
        # Log selection
        selection_time = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"Agent selected: {best_agent} (confidence: {confidence:.3f}, time: {selection_time:.3f}s)")
        
        # Update usage statistics
        self.update_agent_usage_stats(best_agent, confidence, user_input)
        
        return {
            "agent": best_agent,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "agent_description": self.agents[best_agent]["description"],
            "selection_time": selection_time,
            "all_scores": agent_scores,
            "selection_details": selection_details[best_agent],
            "reasoning": self.generate_selection_reasoning(best_agent, selection_details[best_agent])
        }
    
    def analyze_query_comprehensive(self, user_input: str) -> Dict[str, Any]:
        """Comprehensive query analysis for better agent selection"""
        
        text_lower = user_input.lower()
        
        return {
            "query_text": user_input,
            "length": len(user_input),
            "word_count": len(user_input.split()),
            "complexity": self.calculate_query_complexity(user_input),
            "sentiment": self.detect_query_sentiment(user_input),
            "intent": self.classify_query_intent(user_input),
            "urgency": self.assess_query_urgency(user_input),
            "technical_level": self.assess_technical_level(user_input),
            "domain_indicators": self.extract_domain_indicators(user_input)
        }
    
    def calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score (0-1)"""
        
        complexity_factors = {
            'length': min(len(query) / 200, 1.0),  # Longer queries tend to be more complex
            'technical_terms': self.count_technical_terms(query) / 15,  # Technical terminology
            'question_depth': self.assess_question_depth(query),
            'multi_part': 1.0 if self.is_multi_part_query(query) else 0.3,
            'specificity': self.assess_query_specificity(query)
        }
        
        return min(sum(complexity_factors.values()) / len(complexity_factors), 1.0)
    
    def count_technical_terms(self, query: str) -> int:
        """Count technical terms in query"""
        
        technical_terms = [
            # Programming
            'algorithm', 'function', 'variable', 'array', 'object', 'class', 'method',
            'database', 'server', 'api', 'framework', 'library', 'debugging', 'optimization',
            # Career
            'resume', 'portfolio', 'linkedin', 'networking', 'negotiation', 'interview',
            # Medical
            'symptoms', 'diagnosis', 'treatment', 'medication', 'prescription',
            # Business
            'analytics', 'metrics', 'kpi', 'roi', 'strategy', 'optimization',
            # Architecture
            'scalability', 'microservices', 'infrastructure', 'deployment', 'architecture'
        ]
        
        return sum(1 for term in technical_terms if term in query.lower())
    
    def assess_question_depth(self, query: str) -> float:
        """Assess the depth/complexity of the question"""
        
        depth_indicators = {
            'basic': ['what is', 'how to', 'can you', 'simple'],
            'intermediate': ['explain', 'compare', 'analyze', 'implement'],
            'advanced': ['optimize', 'architect', 'strategy', 'complex', 'advanced']
        }
        
        query_lower = query.lower()
        
        if any(indicator in query_lower for indicator in depth_indicators['advanced']):
            return 1.0
        elif any(indicator in query_lower for indicator in depth_indicators['intermediate']):
            return 0.6
        elif any(indicator in query_lower for indicator in depth_indicators['basic']):
            return 0.3
        else:
            return 0.5  # Default
    
    def is_multi_part_query(self, query: str) -> bool:
        """Check if query has multiple parts/requests"""
        
        multi_part_indicators = ['and', 'also', 'plus', 'additionally', 'furthermore', 'moreover']
        return any(indicator in query.lower() for indicator in multi_part_indicators)
    
    def assess_query_specificity(self, query: str) -> float:
        """Assess how specific the query is"""
        
        specific_indicators = [
            'specific', 'exactly', 'precisely', 'detailed', 'step by step',
            'particular', 'exact', 'specific example', 'in detail'
        ]
        
        specificity_score = sum(1 for indicator in specific_indicators if indicator in query.lower())
        return min(specificity_score / 3, 1.0)
    
    def detect_query_sentiment(self, query: str) -> str:
        """Detect emotional tone of query"""
        
        positive_words = ['help', 'please', 'thank', 'appreciate', 'great', 'good']
        negative_words = ['problem', 'issue', 'error', 'bug', 'broken', 'wrong', 'frustrated']
        urgent_words = ['urgent', 'asap', 'quickly', 'immediately', 'emergency']
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in urgent_words):
            return 'urgent'
        elif any(word in query_lower for word in negative_words):
            return 'negative'
        elif any(word in query_lower for word in positive_words):
            return 'positive'
        else:
            return 'neutral'
    
    def classify_query_intent(self, query: str) -> str:
        """Classify the intent behind the query"""
        
        intent_patterns = {
            'seeking_help': ['help', 'can you', 'please', 'assist', 'support'],
            'asking_question': ['what', 'why', 'how', 'when', 'where', 'which', '?'],
            'requesting_creation': ['create', 'build', 'make', 'generate', 'write', 'develop'],
            'seeking_explanation': ['explain', 'clarify', 'understand', 'mean', 'elaborate'],
            'problem_solving': ['error', 'bug', 'issue', 'problem', 'fix', 'solve'],
            'seeking_advice': ['should', 'recommend', 'suggest', 'advice', 'opinion'],
            'comparison': ['compare', 'difference', 'better', 'vs', 'versus', 'contrast'],
            'optimization': ['improve', 'optimize', 'better', 'enhance', 'upgrade']
        }
        
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        return max(intent_scores, key=intent_scores.get) if intent_scores else 'general_inquiry'
    
    def assess_query_urgency(self, query: str) -> str:
        """Assess urgency level of query"""
        
        high_urgency = ['urgent', 'emergency', 'asap', 'immediately', 'critical', 'breaking']
        medium_urgency = ['soon', 'quickly', 'fast', 'rush', 'deadline']
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in high_urgency):
            return 'high'
        elif any(word in query_lower for word in medium_urgency):
            return 'medium'
        else:
            return 'low'
    
    def assess_technical_level(self, query: str) -> str:
        """Assess technical sophistication level"""
        
        beginner_indicators = ['basic', 'simple', 'beginner', 'start', 'new to', 'learn']
        intermediate_indicators = ['implement', 'use', 'apply', 'integrate', 'setup']
        advanced_indicators = ['optimize', 'architect', 'scalable', 'enterprise', 'advanced', 'complex']
        
        query_lower = query.lower()
        
        if any(indicator in query_lower for indicator in advanced_indicators):
            return 'advanced'
        elif any(indicator in query_lower for indicator in intermediate_indicators):
            return 'intermediate'
        elif any(indicator in query_lower for indicator in beginner_indicators):
            return 'beginner'
        else:
            return 'intermediate'  # Default
    
    def extract_domain_indicators(self, query: str) -> List[str]:
        """Extract domain-specific indicators from query"""
        
        domain_keywords = {
            'programming': ['code', 'programming', 'python', 'javascript', 'function'],
            'career': ['resume', 'job', 'interview', 'career', 'salary'],
            'medical': ['health', 'medical', 'symptoms', 'doctor'],
            'business': ['business', 'analytics', 'strategy', 'revenue'],
            'architecture': ['architecture', 'system', 'scalability', 'design']
        }
        
        found_domains = []
        query_lower = query.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                found_domains.append(domain)
        
        return found_domains
    
    async def calculate_agent_score(self, agent_name: str, agent_config: Dict, 
                                  user_input: str, query_analysis: Dict,
                                  context: Dict = None, evolution_data: Dict = None) -> Dict[str, float]:
        """Calculate comprehensive agent score"""
        
        scores = {}
        
        # 1. Keyword matching score
        scores['keyword_match'] = self.calculate_keyword_match_score(
            user_input, agent_config['keywords']
        )
        
        # 2. Context relevance score
        scores['context_relevance'] = self.calculate_context_relevance(
            query_analysis, agent_config, context
        )
        
        # 3. Complexity alignment score
        scores['complexity_fit'] = self.calculate_complexity_fit(
            query_analysis['complexity'], agent_config['complexity_preference']
        )
        
        # 4. Performance history score
        scores['performance_history'] = self.get_performance_history_score(agent_name)
        
        # 5. User preference score (if available)
        scores['user_preference'] = self.calculate_user_preference_score(
            agent_name, evolution_data
        )
        
        return scores
    
    def calculate_keyword_match_score(self, user_input: str, keywords: Dict) -> float:
        """Calculate keyword matching score"""
        
        text_lower = user_input.lower()
        
        # Primary keywords (high weight)
        primary_hits = sum(3 for keyword in keywords['primary'] if keyword in text_lower)
        
        # Secondary keywords (medium weight)
        secondary_hits = sum(2 for keyword in keywords['secondary'] if keyword in text_lower)
        
        # Context keywords (low weight)
        context_hits = sum(1 for keyword in keywords['context'] if keyword in text_lower)
        
        total_score = primary_hits + secondary_hits + context_hits
        max_possible = len(keywords['primary']) * 3 + len(keywords['secondary']) * 2 + len(keywords['context'])
        
        return min(total_score / max(max_possible, 1), 1.0)
    
    def calculate_context_relevance(self, query_analysis: Dict, agent_config: Dict, 
                                  context: Dict = None) -> float:
        """Calculate context relevance score"""
        
        relevance_score = 0.0
        
        # Domain alignment
        if query_analysis.get('domain_indicators'):
            agent_specializations = agent_config.get('specialization', [])
            domain_overlap = len(set(query_analysis['domain_indicators']) & set(agent_specializations))
            relevance_score += domain_overlap * 0.3
        
        # Intent alignment
        intent = query_analysis.get('intent', '')
        if intent in ['seeking_help', 'problem_solving'] and 'coding_expert' in agent_config.get('specialization', []):
            relevance_score += 0.4
        elif intent == 'seeking_advice' and 'career_coach' in agent_config.get('specialization', []):
            relevance_score += 0.4
        
        # Context from previous conversations
        if context and context.get('has_context'):
            # If continuing a conversation, prefer the same agent type
            previous_topics = context.get('relevant_topics', [])
            if any(topic in agent_config.get('specialization', []) for topic in previous_topics):
                relevance_score += 0.3
        
        return min(relevance_score, 1.0)
    
    def calculate_complexity_fit(self, query_complexity: float, agent_preference: List[float]) -> float:
        """Calculate how well query complexity fits agent preference"""
        
        min_pref, max_pref = agent_preference
        
        if min_pref <= query_complexity <= max_pref:
            # Perfect fit
            return 1.0
        elif query_complexity < min_pref:
            # Query too simple for agent
            return max(0.2, 1.0 - (min_pref - query_complexity))
        else:
            # Query too complex for agent
            return max(0.2, 1.0 - (query_complexity - max_pref))
    
    def get_performance_history_score(self, agent_name: str) -> float:
        """Get performance score based on historical data"""
        
        if agent_name not in self.agent_performance:
            return 0.5  # Neutral for new agents
        
        performance_data = self.agent_performance[agent_name]
        
        # Calculate average satisfaction
        avg_satisfaction = np.mean(performance_data.get('satisfaction_scores', [3.0]))
        
        # Normalize to 0-1 scale (satisfaction is 1-5)
        return (avg_satisfaction - 1) / 4
    
    def calculate_user_preference_score(self, agent_name: str, 
                                      evolution_data: Dict = None) -> float:
        """Calculate user preference score"""
        
        if not evolution_data:
            return 0.5  # Neutral
        
        # Check if user has shown preference for this agent type
        preferred_agents = evolution_data.get('preferred_agents', {})
        
        if agent_name in preferred_agents:
            usage_frequency = preferred_agents[agent_name]
            return min(usage_frequency / 10, 1.0)  # Normalize
        
        return 0.5  # Neutral
    
    def get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to level"""
        
        if confidence >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def generate_selection_reasoning(self, selected_agent: str, selection_details: Dict) -> str:
        """Generate human-readable reasoning for agent selection"""
        
        reasons = []
        
        # Keyword match reasoning
        if selection_details.get('keyword_match', 0) > 0.7:
            reasons.append("strong keyword match with agent specialization")
        elif selection_details.get('keyword_match', 0) > 0.4:
            reasons.append("good keyword alignment")
        
        # Complexity reasoning
        if selection_details.get('complexity_fit', 0) > 0.8:
            reasons.append("query complexity matches agent capabilities")
        
        # Context reasoning
        if selection_details.get('context_relevance', 0) > 0.6:
            reasons.append("relevant to conversation context")
        
        # Performance reasoning
        if selection_details.get('performance_history', 0) > 0.7:
            reasons.append("strong historical performance")
        
        if not reasons:
            reasons.append("best available match for query")
        
        return f"Selected {selected_agent} based on: " + ", ".join(reasons)
    
    def update_agent_usage_stats(self, agent_name: str, confidence: float, user_input: str):
        """Update agent usage statistics"""
        
        timestamp = datetime.now()
        
        # Update usage count
        if agent_name not in self.agent_usage_stats:
            self.agent_usage_stats[agent_name] = {
                'total_uses': 0,
                'confidence_scores': [],
                'last_used': None,
                'query_types': []
            }
        
        self.agent_usage_stats[agent_name]['total_uses'] += 1
        self.agent_usage_stats[agent_name]['confidence_scores'].append(confidence)
        self.agent_usage_stats[agent_name]['last_used'] = timestamp.isoformat()
        
        # Store query type for analysis
        query_type = self.classify_query_intent(user_input)
        self.agent_usage_stats[agent_name]['query_types'].append(query_type)
        
        # Keep only recent data (last 100 entries)
        for key in ['confidence_scores', 'query_types']:
            if len(self.agent_usage_stats[agent_name][key]) > 100:
                self.agent_usage_stats[agent_name][key] = self.agent_usage_stats[agent_name][key][-100:]
    
    async def enhance_response(self, agent_name: str, base_response: str, 
                             context: Dict = None) -> str:
        """Enhance response based on agent specialization"""
        
        agent_config = self.agents.get(agent_name, {})
        enhanced_capabilities = agent_config.get('enhanced_capabilities', [])
        
        # Add agent-specific enhancements
        enhancements = []
        
        if agent_name == "coding_expert":
            enhancements.extend([
                "\n💡 **Code Quality Tips**: Consider adding comments and error handling",
                "📚 **Best Practices**: Follow PEP 8 for Python or relevant style guides",
                "🔧 **Testing**: Don't forget to write unit tests for your functions"
            ])
        
        elif agent_name == "career_coach":
            enhancements.extend([
                "\n🎯 **Pro Tip**: Quantify your achievements with specific numbers",
                "📈 **Next Steps**: Update your LinkedIn profile with this information",
                "💼 **Follow-up**: Practice explaining this in interview scenarios"
            ])
        
        elif agent_name == "medical_advisor":
            enhancements.extend([
                "\n⚠️ **Important**: This information is for educational purposes only",
                "🏥 **Recommendation**: Consult with a healthcare professional",
                "📞 **Emergency**: Seek immediate medical attention if symptoms worsen"
            ])
        
        elif agent_name == "business_intelligence":
            enhancements.extend([
                "\n📊 **Data Insight**: Consider tracking this metric over time",
                "💡 **Strategic Thinking**: How does this align with business objectives?",
                "🎯 **Action Items**: Define specific KPIs to measure success"
            ])
        
        # Add contextual enhancements
        if context and context.get('has_context'):
            enhancements.append(f"\n🔗 **Context**: Building on our previous discussion about {context.get('recent_topics', ['general topics'])[0] if context.get('recent_topics') else 'your question'}")
        
        # Combine base response with enhancements
        enhanced_response = base_response
        
        if enhancements:
            enhanced_response += "\n\n---" + "".join(enhancements[:2])  # Limit to 2 enhancements
        
        return enhanced_response
    
    async def record_agent_performance(self, agent_name: str, user_satisfaction: float,
                                     response_time: float, success: bool):
        """Record agent performance for learning"""
        
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = {
                'satisfaction_scores': [],
                'response_times': [],
                'success_rate': [],
                'total_interactions': 0
            }
        
        perf = self.agent_performance[agent_name]
        perf['satisfaction_scores'].append(user_satisfaction)
        perf['response_times'].append(response_time)
        perf['success_rate'].append(1 if success else 0)
        perf['total_interactions'] += 1
        
        # Keep only recent data (last 50 interactions)
        for key in ['satisfaction_scores', 'response_times', 'success_rate']:
            if len(perf[key]) > 50:
                perf[key] = perf[key][-50:]
        
        self.logger.info(f"Recorded performance for {agent_name}: satisfaction={user_satisfaction}, success={success}")
    
    async def get_usage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        
        stats = {
            'total_agents': len(self.agents),
            'active_agents': len(self.agent_usage_stats),
            'agent_usage': {},
            'performance_summary': {},
            'most_used_agent': None,
            'best_performing_agent': None
        }
        
        # Process usage stats
        for agent, usage_data in self.agent_usage_stats.items():
            avg_confidence = np.mean(usage_data['confidence_scores']) if usage_data['confidence_scores'] else 0
            
            stats['agent_usage'][agent] = {
                'total_uses': usage_data['total_uses'],
                'avg_confidence': round(avg_confidence, 3),
                'last_used': usage_data['last_used'],
                'most_common_query_type': Counter(usage_data['query_types']).most_common(1)[0][0] if usage_data['query_types'] else 'unknown'
            }
        
        # Process performance stats
        for agent, perf_data in self.agent_performance.items():
            if perf_data['satisfaction_scores']:
                avg_satisfaction = np.mean(perf_data['satisfaction_scores'])
                avg_response_time = np.mean(perf_data['response_times'])
                success_rate = np.mean(perf_data['success_rate'])
                
                stats['performance_summary'][agent] = {
                    'avg_satisfaction': round(avg_satisfaction, 2),
                    'avg_response_time': round(avg_response_time, 2),
                    'success_rate': round(success_rate * 100, 1),
                    'total_interactions': perf_data['total_interactions']
                }
        
        # Find most used and best performing agents
        if stats['agent_usage']:
            stats['most_used_agent'] = max(stats['agent_usage'], 
                                         key=lambda x: stats['agent_usage'][x]['total_uses'])
        
        if stats['performance_summary']:
            stats['best_performing_agent'] = max(stats['performance_summary'],
                                               key=lambda x: stats['performance_summary'][x]['avg_satisfaction'])
        
        return stats
    
    def get_agent_capabilities(self, agent_name: str) -> Dict[str, Any]:
        """Get detailed agent capabilities"""
        
        agent_config = self.agents.get(agent_name, {})
        
        return {
            'agent_name': agent_name,
            'description': agent_config.get('description', ''),
            'specialization': agent_config.get('specialization', []),
            'enhanced_capabilities': agent_config.get('enhanced_capabilities', []),
            'value_score': agent_config.get('value_score', 0),
            'complexity_range': agent_config.get('complexity_preference', [0, 1]),
            'disclaimers': agent_config.get('disclaimers', [])
        }
    
    async def get_recommended_agent_for_context(self, context: Dict) -> str:
        """Recommend agent based on conversation context"""
        
        if not context or not context.get('has_context'):
            return 'general_assistant'
        
        # Analyze context topics
        recent_topics = context.get('relevant_topics', [])
        
        # Map topics to agents
        topic_agent_mapping = {
            'programming': 'coding_expert',
            'career': 'career_coach', 
            'health': 'medical_advisor',
            'business': 'business_intelligence',
            'personal': 'emotional_counselor'
        }
        
        for topic in recent_topics:
            if topic in topic_agent_mapping:
                return topic_agent_mapping[topic]
        
        return 'general_assistant'

# Quick test function
async def test_agent_manager():
    """Test the agent manager system"""
    
    print("🧪 Testing Agent Manager...")
    
    manager = AgentManager()
    
    # Test queries for different agents
    test_cases = [
        ("How do I write a Python function to calculate factorial?", "coding_expert"),
        ("Help me improve my resume for a software engineer position", "career_coach"),
        ("I have a headache and feel dizzy, what could it be?", "medical_advisor"),
        ("How can I analyze our sales performance data?", "business_intelligence"),
        ("I'm feeling stressed about work, any advice?", "emotional_counselor"),
        ("Hello, how are you today?", "general_assistant")
    ]
    
    for query, expected_agent in test_cases:
        print(f"\n🔍 Query: {query}")
        
        result = await manager.select_optimal_agent(query)
        
        print(f"✅ Selected: {result['agent']} (confidence: {result['confidence']:.3f})")
        print(f"💡 Reasoning: {result['reasoning']}")
        
        # Test if selection matches expectation
        match_status = "✅ CORRECT" if result['agent'] == expected_agent else "⚠️ DIFFERENT"
        print(f"Expected: {expected_agent} - {match_status}")
    
    # Test usage statistics
    stats = await manager.get_usage_statistics()
    print(f"\n📊 Usage Statistics:")
    print(f"Most used agent: {stats['most_used_agent']}")
    print(f"Total agents: {stats['total_agents']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent_manager())
