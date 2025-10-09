"""
Advanced Reasoning Engine for Next-Gen NOVA Chatbot
Implements chain-of-thought reasoning, multi-step problem solving, and advanced context understanding
Integrates with ProductionNovaSystem
"""

import asyncio
import re
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import networkx as nx
import logging
import time
from datetime import datetime
import hashlib
from collections import deque

logger = logging.getLogger(__name__)

class ReasoningType(Enum):
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    LOGICAL = "logical"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    STRATEGIC = "strategic"

@dataclass
class ReasoningStep:
    step_id: int
    description: str
    input_data: Any
    output_data: Any
    confidence: float
    reasoning_type: ReasoningType
    dependencies: List[int]

class AdvancedReasoningEngine:
    """
    Production-ready reasoning engine that provides Claude-level thinking capabilities
    Integrates with ProductionNovaSystem for real AI responses
    """
    
    def __init__(self, nova_system=None):
        self.nova_system = nova_system
        self.reasoning_patterns = self._initialize_reasoning_patterns()
        self.knowledge_graph = nx.DiGraph()
        self.reasoning_history = []
        self.context_windows = {}
        self.reasoning_cache = {}  # Cache for expensive reasoning operations
        
    def _initialize_reasoning_patterns(self) -> Dict[str, Dict]:
        """Initialize sophisticated reasoning patterns"""
        return {
            "problem_decomposition": {
                "pattern": r"(how to|how can|what steps|explain how|solve|implement|build|create)",
                "steps": [
                    "Understand the core problem and requirements",
                    "Break into manageable sub-problems", 
                    "Identify dependencies and constraints",
                    "Plan solution approach and methodology",
                    "Consider edge cases and potential issues",
                    "Outline implementation strategy and timeline"
                ],
                "reasoning_type": ReasoningType.ANALYTICAL,
                "complexity_threshold": "medium"
            },
            
            "comparative_analysis": {
                "pattern": r"(compare|vs|versus|difference|better|best|choose between|pros and cons)",
                "steps": [
                    "Identify comparison criteria and factors",
                    "Gather relevant data points for each option",
                    "Analyze strengths and weaknesses systematically",
                    "Consider context, use cases, and requirements",
                    "Weigh trade-offs and implications",
                    "Provide reasoned recommendation with justification"
                ],
                "reasoning_type": ReasoningType.COMPARATIVE,
                "complexity_threshold": "medium"
            },
            
            "causal_reasoning": {
                "pattern": r"(why|cause|reason|result|impact|effect|consequence|leads to)",
                "steps": [
                    "Identify the phenomenon or outcome",
                    "Trace potential root causes and factors",
                    "Analyze causal relationships and mechanisms",
                    "Consider multiple contributing factors",
                    "Evaluate evidence strength and reliability",
                    "Present comprehensive causal explanation"
                ],
                "reasoning_type": ReasoningType.CAUSAL,
                "complexity_threshold": "high"
            },
            
            "strategic_planning": {
                "pattern": r"(strategy|plan|approach|roadmap|framework|methodology|achieve|goal)",
                "steps": [
                    "Define objectives and success criteria clearly",
                    "Assess current situation and baseline",
                    "Identify constraints, resources, and opportunities",
                    "Generate strategic options and alternatives",
                    "Evaluate risks, benefits, and implementation challenges",
                    "Create detailed actionable plan with milestones"
                ],
                "reasoning_type": ReasoningType.STRATEGIC,
                "complexity_threshold": "high"
            },
            
            "creative_synthesis": {
                "pattern": r"(creative|innovative|design|brainstorm|generate ideas|alternative|novel)",
                "steps": [
                    "Understand creative requirements and constraints",
                    "Gather diverse perspectives and inspiration",
                    "Generate multiple creative concepts and ideas",
                    "Combine ideas in novel and unexpected ways",
                    "Evaluate feasibility and implementation potential",
                    "Refine and present most promising solutions"
                ],
                "reasoning_type": ReasoningType.CREATIVE,
                "complexity_threshold": "medium"
            },
            
            "logical_analysis": {
                "pattern": r"(logic|logical|reasoning|argument|proof|evidence|conclusion)",
                "steps": [
                    "Identify premises and assumptions",
                    "Evaluate logical structure and validity",
                    "Check for logical fallacies or errors",
                    "Assess evidence quality and sources",
                    "Test conclusions against alternative explanations",
                    "Present strengthened logical argument"
                ],
                "reasoning_type": ReasoningType.LOGICAL,
                "complexity_threshold": "high"
            }
        }
    
    async def analyze_query_complexity(self, user_query: str) -> Dict[str, Any]:
        """Analyze query complexity and reasoning requirements with smarter detection"""
        
        # Check cache first
        cache_key = hashlib.md5(user_query.encode()).hexdigest()
        if cache_key in self.reasoning_cache:
            cached_result = self.reasoning_cache[cache_key]
            if time.time() - cached_result['timestamp'] < 3600:  # 1 hour cache
                return cached_result['analysis']
        
        complexity_indicators = {
            "high": [
                "multiple steps", "complex", "comprehensive", "detailed", "in-depth",
            "thorough", "extensive", "strategic", "architecture", "design", 
            "scalability", "optimization", "trade-off", "pros and cons",
            "roadmap", "implementation plan", "framework", "long-term"
            ],
            "medium": [
            "explain", "compare", "analyze", "evaluate", "recommend",
            "suggest", "guide", "walkthrough", "tutorial", "best practices",
            "methodology", "approach", "advantages", "disadvantages",
            "why", "how to"
            ],
            "low": [
            "what is", "define", "quick", "simple", "brief", "short",
            "yes or no", "true or false", "basic", "summary", "example"
        ]
        }
        
        query_lower = user_query.lower()
        word_count = len(user_query.split())
        
        # Calculate complexity scores
        complexity_scores = {
        level: sum(1 for indicator in indicators if indicator in query_lower)
        for level, indicators in complexity_indicators.items()
    }
        
        # Enhanced complexity determination
        if complexity_scores["high"] > 0 or word_count > 20:
           complexity = "high"
        elif complexity_scores["medium"] > 0 or word_count > 8:
           complexity = "medium"
        else:
           complexity = "low"
        
        # Identify reasoning patterns needed
        reasoning_patterns = []
        for pattern_name, pattern_data in self.reasoning_patterns.items():
            if re.search(pattern_data["pattern"], query_lower):
                reasoning_patterns.append({
                    "pattern": pattern_name,
                    "type": pattern_data["reasoning_type"].value,
                    "steps": pattern_data["steps"],
                    "complexity_threshold": pattern_data["complexity_threshold"]
                })
        
        # Calculate reasoning requirements
        estimated_steps = 0
        for pattern in reasoning_patterns:
            if pattern["complexity_threshold"] == "high":
                estimated_steps += 5
            elif pattern["complexity_threshold"] == "medium":
                estimated_steps += 3
            else:
                estimated_steps += 2
        
        if complexity == "high":
            estimated_steps += 3

        requires_multi_step = (
            complexity == "high"
            or len(reasoning_patterns) > 0
            or any(word in query_lower for word in ["why", "how", "compare", "analyze", "steps", "process"])
        )
        
        analysis = {
            "complexity": complexity,
            "word_count": word_count,
            "reasoning_patterns": reasoning_patterns,
            "estimated_reasoning_steps": estimated_steps,
            "requires_multi_step": requires_multi_step,
            "reasoning_confidence": min(0.98, 0.5 + 0.2 * len(reasoning_patterns)),
            "processing_time_estimate": max(2.0, estimated_steps * 2.0),
        }
        
        # Cache the result
        self.reasoning_cache[cache_key] = {
            'analysis': analysis,
            'timestamp': time.time()
        }
        
        return analysis
    
    async def execute_advanced_reasoning(self, user_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute advanced multi-step reasoning process with real AI integration"""
        
        start_time = time.time()
        
        # Step 1: Analyze query complexity
        complexity_analysis = await self.analyze_query_complexity(user_query)
        
        # Step 2: Initialize reasoning chain
        reasoning_context = {
            "original_query": user_query,
            "user_context": context,
            "complexity": complexity_analysis["complexity"],
            "identified_patterns": complexity_analysis["reasoning_patterns"],
            "nova_system": self.nova_system,
            "reasoning_start_time": start_time
        }
        
        # Step 3: Execute reasoning based on complexity
        if complexity_analysis["complexity"] == "high":
            return await self._execute_complex_reasoning(user_query, reasoning_context)
        elif complexity_analysis["complexity"] == "medium":
            return await self._execute_moderate_reasoning(user_query, reasoning_context)
        else:
            return await self._execute_simple_reasoning(user_query, reasoning_context)
    
    async def _execute_complex_reasoning(self, query: str, context: Dict) -> Dict[str, Any]:
        """Execute complex multi-step reasoning chain with real AI calls"""
        reasoning_chain = []
        
        # Step 1: Deep Problem Understanding
        understanding_step = ReasoningStep(
            step_id=1,
            description="Deep problem understanding and context analysis",
            input_data=query,
            output_data=await self._understand_problem_deeply(query, context),
            confidence=0.9,
            reasoning_type=ReasoningType.ANALYTICAL,
            dependencies=[]
        )
        reasoning_chain.append(understanding_step)
        
        # Step 2: Knowledge Retrieval and Integration
        knowledge_step = ReasoningStep(
            step_id=2,
            description="Relevant knowledge retrieval and integration",
            input_data=understanding_step.output_data,
            output_data=await self._retrieve_relevant_knowledge(understanding_step.output_data, context),
            confidence=0.85,
            reasoning_type=ReasoningType.ANALYTICAL,
            dependencies=[1]
        )
        reasoning_chain.append(knowledge_step)
        
        # Step 3: Multi-perspective Analysis (using AI)
        analysis_step = ReasoningStep(
            step_id=3,
            description="Multi-perspective analysis and solution generation",
            input_data=knowledge_step.output_data,
            output_data=await self._analyze_multiple_perspectives(knowledge_step.output_data, context),
            confidence=0.88,
            reasoning_type=ReasoningType.COMPARATIVE,
            dependencies=[1, 2]
        )
        reasoning_chain.append(analysis_step)
        
        # Step 4: Solution Synthesis (using AI)
        synthesis_step = ReasoningStep(
            step_id=4,
            description="Solution synthesis and optimization",
            input_data=analysis_step.output_data,
            output_data=await self._synthesize_solution(analysis_step.output_data, context),
            confidence=0.92,
            reasoning_type=ReasoningType.STRATEGIC,
            dependencies=[1, 2, 3]
        )
        reasoning_chain.append(synthesis_step)
        
        # Step 5: Validation and Refinement (using AI)
        validation_step = ReasoningStep(
            step_id=5,
            description="Solution validation and refinement",
            input_data=synthesis_step.output_data,
            output_data=await self._validate_and_refine(synthesis_step.output_data, query, context),
            confidence=0.94,
            reasoning_type=ReasoningType.LOGICAL,
            dependencies=[4]
        )
        reasoning_chain.append(validation_step)
        
        # Compile final reasoning output
        return {
            "reasoning_chain": reasoning_chain,
            "final_solution": validation_step.output_data,
            "complexity_level": "high",
            "confidence_score": sum(step.confidence for step in reasoning_chain) / len(reasoning_chain),
            "reasoning_summary": self._generate_reasoning_summary(reasoning_chain),
            "implementation_ready": True,
            "processing_time": time.time() - context["reasoning_start_time"],
            "ai_calls_made": 5  # Track AI usage
        }
    
    async def _understand_problem_deeply(self, query: str, context: Dict) -> Dict[str, Any]:
        """Deep problem understanding using real AI analysis"""
        
        # Use NOVA system for AI-powered understanding if available
        if self.nova_system:
            try:
                understanding_prompt = f"""Analyze this problem deeply and provide structured understanding:

Problem: {query}

Context: {json.dumps(context.get('user_context', {}), indent=2)}

Please provide:
1. Core problem identification
2. Key entities and concepts involved
3. Problem type and domain classification
4. Complexity factors and dependencies
5. Success criteria and expected outcomes

Respond in JSON format with detailed analysis."""

                ai_response = await self.nova_system.api_manager.get_ai_response(
                    understanding_prompt,
                    "You are an expert problem analyst. Provide structured, comprehensive problem analysis.",
                    "analysis"
                )
                
                if ai_response:
                    # Try to parse AI response as structured data
                    try:
                        if "{" in ai_response and "}" in ai_response:
                            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                            if json_match:
                                ai_analysis = json.loads(json_match.group())
                                if isinstance(ai_analysis, dict):
                                    return self._enhance_ai_understanding(ai_analysis, query, context)
                    except:
                        pass
                    
                    # Fallback: extract insights from AI text response
                    return self._extract_understanding_from_text(ai_response, query, context)
                
            except Exception as e:
                logger.error(f"AI-powered understanding failed: {e}")
        
        # Fallback to rule-based understanding
        return self._rule_based_understanding(query, context)
    
    def _enhance_ai_understanding(self, ai_analysis: Dict, query: str, context: Dict) -> Dict[str, Any]:
        """Enhance AI analysis with additional structured data"""
        
        # Extract technical terms and entities
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        technical_terms = re.findall(r'\b(?:API|database|algorithm|framework|system|model|architecture|server|client|network|security|performance|scalability|deployment|testing|monitoring)\b', query.lower())
        
        enhanced_understanding = {
            "ai_analysis": ai_analysis,
            "core_problem": ai_analysis.get("core_problem", query),
            "entities": list(set(entities)),
            "technical_terms": list(set(technical_terms)),
            "problem_domain": self._identify_problem_domain(query, technical_terms),
            "complexity_factors": self._analyze_complexity_factors(query, context),
            "success_criteria": ai_analysis.get("success_criteria", self._infer_success_criteria(query)),
            "context_integration": context,
            "understanding_confidence": 0.9,
            "ai_enhanced": True
        }
        
        return enhanced_understanding
    
    def _extract_understanding_from_text(self, ai_response: str, query: str, context: Dict) -> Dict[str, Any]:
        """Extract understanding insights from AI text response"""
        
        # Extract key insights using pattern matching
        insights = {
            "core_insights": re.findall(r'(?:problem|issue|challenge|goal)[:\s]*([^.\n]+)', ai_response.lower()),
            "key_factors": re.findall(r'(?:factor|aspect|element|component)[:\s]*([^.\n]+)', ai_response.lower()),
            "recommendations": re.findall(r'(?:recommend|suggest|should|need to)[:\s]*([^.\n]+)', ai_response.lower()),
            "considerations": re.findall(r'(?:consider|important|note|remember)[:\s]*([^.\n]+)', ai_response.lower())
        }
        
        return {
            "ai_insights": insights,
            "core_problem": query,
            "ai_response_summary": ai_response[:200] + "..." if len(ai_response) > 200 else ai_response,
            "problem_domain": self._identify_problem_domain(query, []),
            "complexity_factors": self._analyze_complexity_factors(query, context),
            "understanding_confidence": 0.85,
            "ai_enhanced": True
        }
    
    def _rule_based_understanding(self, query: str, context: Dict) -> Dict[str, Any]:
        """Fallback rule-based understanding when AI is unavailable"""
        
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        technical_terms = re.findall(r'\b(?:API|database|algorithm|framework|system|model|architecture)\b', query.lower())
        
        # Identify question type
        question_types = {
            "how": "procedural",
            "why": "explanatory", 
            "what": "definitional",
            "when": "temporal",
            "where": "locational",
            "which": "selective"
        }
        
        question_type = "procedural"  # default
        for question_word, q_type in question_types.items():
            if question_word in query.lower()[:20]:
                question_type = q_type
                break
        
        complexity_factors = {
            "multi_domain": len(set(technical_terms)) > 2,
            "requires_planning": any(word in query.lower() for word in ["plan", "strategy", "approach", "implement"]),
            "needs_comparison": any(word in query.lower() for word in ["compare", "vs", "better", "best"]),
            "time_sensitive": any(word in query.lower() for word in ["urgent", "asap", "quickly", "fast"]),
            "context_dependent": len(context.get("conversation_history", "")) > 100
        }
        
        return {
            "core_problem": query,
            "entities": entities,
            "technical_terms": technical_terms,
            "question_type": question_type,
            "problem_domain": self._identify_problem_domain(query, technical_terms),
            "complexity_factors": complexity_factors,
            "context_integration": context,
            "understanding_confidence": 0.75,
            "ai_enhanced": False
        }
    
    def _identify_problem_domain(self, query: str, technical_terms: List[str]) -> str:
        """Identify the primary domain of the problem"""
        
        domain_indicators = {
            "software_engineering": ["code", "programming", "development", "api", "framework", "library"],
            "data_science": ["data", "analysis", "machine learning", "ai", "model", "algorithm"],
            "business_strategy": ["business", "strategy", "market", "revenue", "growth", "customer"],
            "system_design": ["architecture", "system", "scalability", "infrastructure", "deployment"],
            "project_management": ["project", "timeline", "milestone", "team", "resource", "planning"],
            "security": ["security", "authentication", "encryption", "vulnerability", "threat"],
            "performance": ["performance", "optimization", "speed", "efficiency", "bottleneck"]
        }
        
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, indicators in domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return "general"
    
    def _analyze_complexity_factors(self, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze factors that contribute to problem complexity"""
        
        query_lower = query.lower()
        
        return {
            "multi_domain": len(self._extract_domains(query)) > 1,
            "requires_planning": any(word in query_lower for word in ["plan", "strategy", "roadmap", "timeline"]),
            "needs_comparison": any(word in query_lower for word in ["compare", "vs", "versus", "better", "best"]),
            "technical_depth": len(re.findall(r'\b(?:API|database|algorithm|framework|architecture|microservice|cloud|docker|kubernetes)\b', query_lower)) > 2,
            "time_sensitive": any(word in query_lower for word in ["urgent", "asap", "deadline", "quickly"]),
            "context_dependent": len(context.get("conversation_history", "")) > 100,
            "requires_research": any(word in query_lower for word in ["research", "investigate", "explore", "analyze"]),
            "involves_tradeoffs": any(word in query_lower for word in ["tradeoff", "balance", "optimize", "choose"])
        }
    
    def _extract_domains(self, query: str) -> List[str]:
        """Extract knowledge domains mentioned in query"""
        
        domain_keywords = {
            "technology": ["tech", "software", "hardware", "digital", "ai", "ml"],
            "business": ["business", "company", "startup", "enterprise", "market"],
            "finance": ["money", "cost", "budget", "investment", "revenue", "profit"],
            "design": ["design", "ui", "ux", "interface", "visual", "layout"],
            "data": ["data", "analytics", "metrics", "statistics", "database"],
            "security": ["security", "privacy", "encryption", "authentication"],
            "operations": ["operations", "deployment", "infrastructure", "monitoring"]
        }
        
        query_lower = query.lower()
        detected_domains = []
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_domains.append(domain)
        
        return detected_domains
    
    def _infer_success_criteria(self, query: str) -> List[str]:
        """Infer success criteria from the query"""
        
        success_patterns = {
            "implementation": ["working", "functional", "operational", "deployed"],
            "performance": ["fast", "efficient", "optimized", "scalable"],
            "quality": ["high-quality", "reliable", "robust", "maintainable"],
            "usability": ["user-friendly", "intuitive", "accessible", "easy"],
            "completeness": ["comprehensive", "complete", "thorough", "detailed"]
        }
        
        query_lower = query.lower()
        criteria = []
        
        for criterion_type, indicators in success_patterns.items():
            if any(indicator in query_lower for indicator in indicators):
                criteria.append(f"Solution should be {criterion_type}")
        
        if not criteria:
            criteria = [
                "Solution should be practical and implementable",
                "Solution should address the core problem effectively",
                "Solution should be well-documented and clear"
            ]
        
        return criteria
    
    async def _retrieve_relevant_knowledge(self, problem_understanding: Dict, context: Dict) -> Dict[str, Any]:
        """Retrieve relevant knowledge using AI-powered analysis"""
        
        if self.nova_system:
            try:
                knowledge_prompt = f"""Based on this problem analysis, identify relevant knowledge areas and best practices:

Problem Domain: {problem_understanding.get('problem_domain', 'general')}
Technical Terms: {problem_understanding.get('technical_terms', [])}
Complexity Factors: {problem_understanding.get('complexity_factors', {})}

Provide:
1. Key knowledge areas to research
2. Relevant methodologies and frameworks
3. Best practices and patterns
4. Common challenges and solutions
5. Expert recommendations

Format as structured JSON with detailed explanations."""

                ai_knowledge = await self.nova_system.api_manager.get_ai_response(
                    knowledge_prompt,
                    "You are a knowledge expert. Provide comprehensive, well-organized knowledge guidance.",
                    "analysis"
                )
                
                if ai_knowledge:
                    return self._process_ai_knowledge(ai_knowledge, problem_understanding)
                
            except Exception as e:
                logger.error(f"AI knowledge retrieval failed: {e}")
        
        # Fallback to structured knowledge mapping
        return self._map_domain_knowledge(problem_understanding)
    
    def _process_ai_knowledge(self, ai_knowledge: str, problem_understanding: Dict) -> Dict[str, Any]:
        """Process AI-generated knowledge into structured format"""
        
        # Try to extract structured data from AI response
        knowledge_structure = {
            "ai_knowledge_summary": ai_knowledge[:300] + "..." if len(ai_knowledge) > 300 else ai_knowledge,
            "knowledge_areas": self._extract_knowledge_areas(ai_knowledge),
            "methodologies": self._extract_methodologies(ai_knowledge),
            "best_practices": self._extract_best_practices(ai_knowledge),
            "challenges": self._extract_challenges(ai_knowledge),
            "recommendations": self._extract_recommendations(ai_knowledge),
            "confidence_level": 0.9,
            "ai_enhanced": True
        }
        
        return knowledge_structure
    
    def _extract_knowledge_areas(self, text: str) -> List[str]:
        """Extract knowledge areas from AI response"""
        
        # Look for numbered lists and bullet points
        knowledge_areas = []
        
        # Pattern for numbered items
        numbered_items = re.findall(r'\d+\.\s*([^.\n]+(?:\.[^.\n]*)?)', text)
        knowledge_areas.extend(numbered_items)
        
        # Pattern for bullet points
        bullet_items = re.findall(r'[-•*]\s*([^.\n]+)', text)
        knowledge_areas.extend(bullet_items)
        
        # Pattern for "knowledge areas" section
        knowledge_section = re.search(r'knowledge areas?[:\s]*\n?(.*?)(?:\n\n|\n[A-Z]|\Z)', text, re.IGNORECASE | re.DOTALL)
        if knowledge_section:
            section_text = knowledge_section.group(1)
            areas = re.findall(r'[-•*]?\s*([A-Z][^.\n]+)', section_text)
            knowledge_areas.extend(areas)
        
        # Clean and deduplicate
        cleaned_areas = [area.strip().rstrip('.') for area in knowledge_areas if len(area.strip()) > 5]
        return list(set(cleaned_areas[:10]))  # Return top 10 unique areas
    
    def _extract_methodologies(self, text: str) -> List[str]:
        """Extract methodologies from AI response"""
        
        methodology_keywords = [
            "methodology", "framework", "approach", "method", "technique", 
            "pattern", "practice", "strategy", "process", "workflow"
        ]
        
        methodologies = []
        
        for keyword in methodology_keywords:
            pattern = rf'{keyword}[:\s]*([^.\n]+)'
            matches = re.findall(pattern, text, re.IGNORECASE)
            methodologies.extend(matches)
        
        # Clean and deduplicate
        cleaned_methodologies = [method.strip().rstrip('.') for method in methodologies if len(method.strip()) > 5]
        return list(set(cleaned_methodologies[:8]))
    
    def _extract_best_practices(self, text: str) -> List[str]:
        """Extract best practices from AI response"""
        
        practice_patterns = [
            r'best practice[:\s]*([^.\n]+)',
            r'should[:\s]*([^.\n]+)',
            r'recommended[:\s]*([^.\n]+)',
            r'important to[:\s]*([^.\n]+)',
            r'ensure[:\s]*([^.\n]+)'
        ]
        
        practices = []
        
        for pattern in practice_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            practices.extend(matches)
        
        # Clean and deduplicate
        cleaned_practices = [practice.strip().rstrip('.') for practice in practices if len(practice.strip()) > 10]
        return list(set(cleaned_practices[:10]))
    
    def _extract_challenges(self, text: str) -> List[str]:
        """Extract challenges from AI response"""
        
        challenge_patterns = [
            r'challenge[:\s]*([^.\n]+)',
            r'problem[:\s]*([^.\n]+)',
            r'difficulty[:\s]*([^.\n]+)',
            r'issue[:\s]*([^.\n]+)',
            r'risk[:\s]*([^.\n]+)'
        ]
        
        challenges = []
        
        for pattern in challenge_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            challenges.extend(matches)
        
        # Clean and deduplicate
        cleaned_challenges = [challenge.strip().rstrip('.') for challenge in challenges if len(challenge.strip()) > 10]
        return list(set(cleaned_challenges[:8]))
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from AI response"""
        
        recommendation_patterns = [
            r'recommend[:\s]*([^.\n]+)',
            r'suggest[:\s]*([^.\n]+)',
            r'advice[:\s]*([^.\n]+)',
            r'tip[:\s]*([^.\n]+)',
            r'guidance[:\s]*([^.\n]+)'
        ]
        
        recommendations = []
        
        for pattern in recommendation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            recommendations.extend(matches)
        
        # Clean and deduplicate
        cleaned_recommendations = [rec.strip().rstrip('.') for rec in recommendations if len(rec.strip()) > 10]
        return list(set(cleaned_recommendations[:10]))
    
    def _map_domain_knowledge(self, problem_understanding: Dict) -> Dict[str, Any]:
        """Map domain-specific knowledge when AI is unavailable"""
        
        domain = problem_understanding.get('problem_domain', 'general')
        technical_terms = problem_understanding.get('technical_terms', [])
        
        domain_knowledge_map = {
            "software_engineering": {
                "knowledge_areas": ["Software Architecture", "Design Patterns", "Code Quality", "Testing Strategies"],
                "methodologies": ["Agile Development", "Test-Driven Development", "Continuous Integration", "Code Review"],
                "best_practices": ["SOLID Principles", "Clean Code", "Documentation", "Version Control"],
                "tools": ["Git", "Docker", "CI/CD Pipelines", "Testing Frameworks"]
            },
            "data_science": {
                "knowledge_areas": ["Statistical Analysis", "Machine Learning", "Data Visualization", "Feature Engineering"],
                "methodologies": ["CRISP-DM", "Data Mining", "Experimental Design", "Model Validation"],
                "best_practices": ["Data Quality", "Reproducible Research", "Model Documentation", "Ethical AI"],
                "tools": ["Python", "R", "SQL", "Jupyter", "Pandas", "Scikit-learn"]
            },
            "business_strategy": {
                "knowledge_areas": ["Market Analysis", "Competitive Intelligence", "Financial Modeling", "Strategic Planning"],
                "methodologies": ["SWOT Analysis", "Porter's Five Forces", "Business Model Canvas", "OKR Framework"],
                "best_practices": ["Customer Focus", "Data-Driven Decisions", "Agile Strategy", "Risk Management"],
                "tools": ["Analytics Platforms", "CRM Systems", "Financial Software", "Project Management"]
            },
            "system_design": {
                "knowledge_areas": ["Distributed Systems", "Scalability Patterns", "Database Design", "API Design"],
                "methodologies": ["Microservices Architecture", "Event-Driven Design", "Domain-Driven Design"],
                "best_practices": ["High Availability", "Fault Tolerance", "Performance Monitoring", "Security by Design"],
                "tools": ["Load Balancers", "Message Queues", "Monitoring Tools", "Container Orchestration"]
            }
        }
        
        knowledge = domain_knowledge_map.get(domain, domain_knowledge_map["software_engineering"])
        
        return {
            "primary_domain": domain,
            "knowledge_areas": knowledge["knowledge_areas"],
            "methodologies": knowledge["methodologies"],
            "best_practices": knowledge["best_practices"],
            "recommended_tools": knowledge["tools"],
            "confidence_level": 0.8,
            "ai_enhanced": False
        }
    
    async def _analyze_multiple_perspectives(self, knowledge: Dict, context: Dict) -> Dict[str, Any]:
        """Analyze problem from multiple perspectives using AI"""
        
        if self.nova_system:
            try:
                perspective_prompt = f"""Analyze this problem from multiple professional perspectives:

Knowledge Context: {json.dumps(knowledge, indent=2)}
Problem Domain: {knowledge.get('primary_domain', 'general')}

Analyze from these perspectives:
1. Technical Implementation (feasibility, architecture, performance)
2. Business Impact (cost, timeline, ROI, strategic value)
3. User Experience (usability, accessibility, user satisfaction)
4. Risk Management (security, reliability, maintenance)
5. Operational Considerations (deployment, monitoring, support)

For each perspective, provide:
- Key considerations
- Potential solutions
- Challenges and risks
- Success metrics

Format as structured analysis with clear sections."""

                ai_analysis = await self.nova_system.api_manager.get_ai_response(
                    perspective_prompt,
                    "You are a senior consultant. Provide multi-perspective strategic analysis.",
                    "analysis"
                )
                
                if ai_analysis:
                    return self._structure_perspective_analysis(ai_analysis, knowledge)
                
            except Exception as e:
                logger.error(f"AI perspective analysis failed: {e}")
        
        # Fallback to structured perspective analysis
        return self._generate_structured_perspectives(knowledge, context)
    
    def _structure_perspective_analysis(self, ai_analysis: str, knowledge: Dict) -> Dict[str, Any]:
        """Structure AI-generated perspective analysis"""
        
        perspectives = {
            "technical": {
                "focus": "Implementation details, technical feasibility, performance optimization",
                "ai_insights": self._extract_perspective_insights(ai_analysis, "technical"),
                "considerations": ["Architecture scalability", "Performance requirements", "Security implementation", "Technology stack"],
                "solutions": self._extract_solutions(ai_analysis, "technical"),
                "confidence": 0.9
            },
            "business": {
                "focus": "Business value, cost-benefit analysis, strategic alignment",
                "ai_insights": self._extract_perspective_insights(ai_analysis, "business"),
                "considerations": ["Return on investment", "Time to market", "Resource requirements", "Strategic impact"],
                "solutions": self._extract_solutions(ai_analysis, "business"),
                "confidence": 0.85
            },
            "user_experience": {
                "focus": "User satisfaction, usability, accessibility, adoption",
                "ai_insights": self._extract_perspective_insights(ai_analysis, "user"),
                "considerations": ["User interface design", "User journey optimization", "Accessibility compliance", "User feedback"],
                "solutions": self._extract_solutions(ai_analysis, "user"),
                "confidence": 0.8
            },
            "operational": {
                "focus": "Implementation, maintenance, support, monitoring",
                "ai_insights": self._extract_perspective_insights(ai_analysis, "operational"),
                "considerations": ["Deployment strategy", "Monitoring setup", "Support procedures", "Maintenance planning"],
                "solutions": self._extract_solutions(ai_analysis, "operational"),
                "confidence": 0.85
            }
        }
        
        return {
        "multi_perspective_analysis": perspectives,
        "ai_analysis_summary": ai_analysis[:400] + "..." if len(ai_analysis) > 400 else ai_analysis,
        "synthesized_insights": self._synthesize_perspectives(perspectives),
        "recommended_approach": self._recommend_balanced_approach(perspectives),
        "ai_enhanced": True
    }
    
    def _extract_perspective_insights(self, text: str, perspective: str) -> List[str]:
        """Extract insights for specific perspective from AI response"""
        
        perspective_keywords = {
            "technical": ["technical", "implementation", "architecture", "performance", "scalability"],
            "business": ["business", "cost", "revenue", "market", "strategic", "roi"],
            "user": ["user", "experience", "usability", "interface", "customer", "ux"],
            "operational": ["operational", "deployment", "maintenance", "monitoring", "support"]
        }
        
        keywords = perspective_keywords.get(perspective, [])
        insights = []
        
        # Find sentences containing perspective keywords
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords) and len(sentence.strip()) > 20:
                insights.append(sentence.strip())
        
        return insights[:5]  # Return top 5 insights
    
    def _extract_solutions(self, text: str, perspective: str) -> List[Dict[str, Any]]:
        """Extract solutions for specific perspective"""
        
        # Look for solution indicators
        solution_patterns = [
            r'solution[:\s]*([^.\n]+)',
            r'approach[:\s]*([^.\n]+)',
            r'strategy[:\s]*([^.\n]+)',
            r'implement[:\s]*([^.\n]+)'
        ]
        
        solutions = []
        for pattern in solution_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 15:
                    solutions.append({
                        "solution_id": f"{perspective}_{len(solutions)}",
                        "description": match.strip(),
                        "implementation_complexity": "medium",
                        "estimated_effort": "2-4 weeks",
                        "confidence": 0.8
                    })
        
        return solutions[:3]  # Return top 3 solutions
    
    async def _synthesize_perspectives(self, perspectives: Dict) -> Dict[str, Any]:
        """Synthesize insights from all perspectives"""
        
        all_insights = []
        common_themes = []
        
        for perspective_name, perspective_data in perspectives.items():
            insights = perspective_data.get("ai_insights", [])
            all_insights.extend(insights)
        
        # Find common themes across perspectives
        word_frequency = {}
        for insight in all_insights:
            words = insight.lower().split()
            for word in words:
                if len(word) > 4:  # Only meaningful words
                    word_frequency[word] = word_frequency.get(word, 0) + 1
        
        # Identify most common themes
        common_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        common_themes = [word for word, count in common_words if count > 1]
        
        return {
            "common_themes": common_themes,
            "cross_perspective_insights": all_insights[:5],
            "conflicting_priorities": self._identify_conflicts(perspectives),
            "balanced_approach": "Implement solution with phased approach prioritizing core functionality while addressing all perspectives",
            "synthesis_confidence": 0.85
        }
    
    def _identify_conflicts(self, perspectives: Dict) -> List[str]:
        """Identify conflicting priorities across perspectives"""
        
        conflicts = []
        
        # Common business vs technical conflicts
        if "business" in perspectives and "technical" in perspectives:
            conflicts.append("Speed to market vs Technical excellence")
            conflicts.append("Cost optimization vs Feature completeness")
        
        # User experience vs operational conflicts
        if "user_experience" in perspectives and "operational" in perspectives:
            conflicts.append("User convenience vs Security requirements")
            conflicts.append("Feature richness vs System simplicity")
        
        return conflicts
    
    async def _recommend_balanced_approach(self, perspectives: Dict) -> Dict[str, str]:
        """Recommend balanced approach considering all perspectives"""
        
        return {
            "primary_recommendation": "Implement MVP with strong foundation for iterative enhancement",
            "rationale": "Balances technical excellence with business delivery timelines while ensuring user satisfaction",
            "key_principles": "User-focused design, technically sound architecture, business-viable approach, operationally maintainable",
            "implementation_strategy": "Phased delivery with continuous user feedback and technical optimization"
        }
    
    async def _synthesize_solution(self, analysis: Dict, context: Dict) -> Dict[str, Any]:
        """Synthesize comprehensive solution using AI"""
        
        if self.nova_system:
            try:
                synthesis_prompt = f"""Create a comprehensive solution based on this analysis:

Multi-Perspective Analysis: {json.dumps(analysis.get('multi_perspective_analysis', {}), indent=2)}
Synthesized Insights: {json.dumps(analysis.get('synthesized_insights', {}), indent=2)}

Generate:
1. Primary recommended solution with clear approach
2. Alternative approaches for different scenarios
3. Detailed implementation plan with phases
4. Risk mitigation strategies
5. Success metrics and validation criteria
6. Resource requirements and timeline

Provide structured, actionable solution framework."""

                ai_solution = await self.nova_system.api_manager.get_ai_response(
                    synthesis_prompt,
                    "You are a solution architect. Create comprehensive, implementable solutions.",
                    "strategic"
                )
                
                if ai_solution:
                    return self._structure_ai_solution(ai_solution, analysis, context)
                
            except Exception as e:
                logger.error(f"AI solution synthesis failed: {e}")
        
        # Fallback to structured solution generation
        return self._generate_structured_solution(analysis, context)
    
    def _structure_ai_solution(self, ai_solution: str, analysis: Dict, context: Dict) -> Dict[str, Any]:
        """Structure AI-generated solution into organized format"""
        
        solution_components = {
            "primary_solution": {
                "solution_title": "AI-Generated Comprehensive Solution",
                "core_approach": self._extract_core_approach(ai_solution),
                "key_components": self._extract_key_components(ai_solution),
                "unique_advantages": self._extract_advantages(ai_solution),
                "ai_rationale": ai_solution[:300] + "..." if len(ai_solution) > 300 else ai_solution
            },
            "alternative_approaches": self._extract_alternatives(ai_solution),
            "implementation_plan": self._extract_implementation_plan(ai_solution),
            "risk_mitigation": self._extract_risk_mitigation(ai_solution),
            "success_metrics": self._extract_success_metrics(ai_solution),
            "next_steps": self._extract_next_steps(ai_solution),
            "ai_enhanced": True,
            "solution_confidence": 0.92
        }
        
        return solution_components
    
    def _extract_core_approach(self, text: str) -> str:
        """Extract core solution approach from AI response"""
        
        approach_patterns = [
            r'approach[:\s]*([^.\n]+(?:\.[^.\n]*)?)',
            r'strategy[:\s]*([^.\n]+(?:\.[^.\n]*)?)',
            r'solution[:\s]*([^.\n]+(?:\.[^.\n]*)?)',
            r'recommendation[:\s]*([^.\n]+(?:\.[^.\n]*)?)'
        ]
        
        for pattern in approach_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and len(match.group(1).strip()) > 20:
                return match.group(1).strip()
        
        # Fallback: extract first substantial sentence
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if len(sentence.strip()) > 30:
                return sentence.strip()
        
        return "Multi-phase implementation with iterative refinement and continuous optimization"
    
    def _extract_key_components(self, text: str) -> List[str]:
        """Extract key solution components"""
        
        component_patterns = [
            r'component[:\s]*([^.\n]+)',
            r'element[:\s]*([^.\n]+)',
            r'module[:\s]*([^.\n]+)',
            r'layer[:\s]*([^.\n]+)',
            r'phase[:\s]*([^.\n]+)'
        ]
        
        components = []
        for pattern in component_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            components.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        # Add default components if none found
        if not components:
            components = [
                "Foundation layer with core functionality",
                "Enhancement layer with advanced features",
                "Integration layer with external systems",
                "Monitoring and optimization layer"
            ]
        
        return components[:6]  # Return top 6 components
    
    def _extract_advantages(self, text: str) -> List[str]:
        """Extract solution advantages"""
        
        advantage_patterns = [
            r'advantage[:\s]*([^.\n]+)',
            r'benefit[:\s]*([^.\n]+)',
            r'strength[:\s]*([^.\n]+)',
            r'value[:\s]*([^.\n]+)'
        ]
        
        advantages = []
        for pattern in advantage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            advantages.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        # Add default advantages if none found
        if not advantages:
            advantages = [
                "Scalable and maintainable architecture",
                "User-centric design and experience",
                "Performance optimized implementation",
                "Future-proof and extensible solution"
            ]
        
        return advantages[:5]
    
    def _extract_alternatives(self, text: str) -> List[Dict]:
        """Extract alternative solution approaches"""
        
        # Look for alternative mentions
        alternatives = []
        
        alternative_indicators = ["alternative", "option", "approach", "method", "strategy"]
        
        for indicator in alternative_indicators:
            pattern = rf'{indicator}[:\s]*([^.\n]+(?:\.[^.\n]*)?)'
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for i, match in enumerate(matches[:2]):  # Max 2 per indicator
                alternatives.append({
                    "alternative_id": len(alternatives) + 1,
                    "title": f"Alternative {indicator.title()} {i+1}",
                    "description": match.strip(),
                    "timeline": "4-8 weeks",
                    "pros": ["Focused approach", "Lower complexity"],
                    "cons": ["Limited scope", "May require future expansion"]
                })
        
        # Default alternatives if none found
        if not alternatives:
            alternatives = [
                {
                    "alternative_id": 1,
                    "title": "Rapid MVP Approach",
                    "description": "Quick implementation focusing on core features with minimal complexity",
                    "timeline": "2-4 weeks",
                    "pros": ["Fast delivery", "Quick validation", "Lower initial cost"],
                    "cons": ["Limited features", "Technical debt risk", "May need refactoring"]
                },
                {
                    "alternative_id": 2,
                    "title": "Comprehensive Enterprise Solution",
                    "description": "Full-featured implementation with enterprise-grade considerations",
                    "timeline": "8-16 weeks",
                    "pros": ["Complete solution", "Enterprise-ready", "Highly scalable"],
                    "cons": ["Longer timeline", "Higher complexity", "Higher initial cost"]
                }
            ]
        
        return alternatives
    
    def _extract_implementation_plan(self, text: str) -> Dict[str, Any]:
        """Extract implementation plan from AI response"""
        
        # Look for phases and steps
        phases = []
        
        phase_pattern = r'phase\s*(\d+)[:\s]*([^.\n]+(?:\.[^.\n]*)?)'
        phase_matches = re.findall(phase_pattern, text, re.IGNORECASE)
        
        for phase_num, phase_desc in phase_matches:
            phases.append({
                "phase": int(phase_num),
                "title": f"Phase {phase_num}: {phase_desc[:50]}",
                "duration": f"{phase_num}-{int(phase_num)+1} weeks",
                "deliverables": self._extract_deliverables_for_phase(text, phase_num),
                "success_criteria": [f"Phase {phase_num} objectives met", "Quality standards achieved"]
            })
        
        # Default phases if none found
        if not phases:
            phases = [
                {
                    "phase": 1,
                    "title": "Foundation & Core Setup",
                    "duration": "2-3 weeks",
                    "deliverables": ["Architecture design", "Core functionality", "Basic testing", "Initial deployment"],
                    "success_criteria": ["Core system operational", "Basic features working", "Tests passing"]
                },
                {
                    "phase": 2,
                    "title": "Feature Enhancement & Integration",
                    "duration": "3-4 weeks",
                    "deliverables": ["Advanced features", "Third-party integrations", "Performance optimization", "Security implementation"],
                    "success_criteria": ["All features operational", "Performance targets met", "Security validated"]
                },
                {
                    "phase": 3,
                    "title": "Optimization & Production Deployment",
                    "duration": "2-3 weeks",
                    "deliverables": ["Production deployment", "Monitoring setup", "Documentation", "User training"],
                    "success_criteria": ["Production ready", "Monitoring operational", "Users trained"]
                }
            ]
        
        return {
            "phases": phases,
            "critical_path": ["Foundation setup", "Core functionality", "Testing and validation", "Production deployment"],
            "resource_requirements": {
                "technical": "2-3 developers",
                "timeline": f"{len(phases)*2}-{len(phases)*4} weeks total",
                "infrastructure": "Cloud hosting with CI/CD pipeline",
                "budget": "Medium to high depending on scope"
            },
            "success_dependencies": [
                "Clear requirement specifications",
                "Adequate technical resources",
                "Regular stakeholder feedback",
                "Continuous quality monitoring"
            ]
        }
    
    def _extract_deliverables_for_phase(self, text: str, phase_num: str) -> List[str]:
        """Extract deliverables for specific phase"""
        
        deliverable_patterns = [
            r'deliverable[:\s]*([^.\n]+)',
            r'output[:\s]*([^.\n]+)',
            r'result[:\s]*([^.\n]+)',
            r'milestone[:\s]*([^.\n]+)'
        ]
        
        deliverables = []
        for pattern in deliverable_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            deliverables.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        return deliverables[:4] if deliverables else ["Core deliverables", "Quality validation", "Documentation", "Testing"]
    
    async def _validate_and_refine(self, solution: Dict, original_query: str, context: Dict) -> Dict[str, Any]:
        """Validate solution against original query and refine using AI"""
        
        if self.nova_system:
            try:
                validation_prompt = f"""Validate and refine this solution:

Original Query: {original_query}
Proposed Solution: {json.dumps(solution, indent=2)}

Validation Criteria:
1. Does it fully address the original question?
2. Is the solution complete and actionable?
3. Is the approach feasible and realistic?
4. Are the implementation steps clear?
5. Are potential risks adequately addressed?

Provide:
- Validation assessment for each criterion
- Specific improvement recommendations
- Refined solution with enhancements
- Overall confidence rating"""

                ai_validation = await self.nova_system.api_manager.get_ai_response(
                    validation_prompt,
                    "You are a solution validator. Provide thorough validation and improvement recommendations.",
                    "analysis"
                )
                
                if ai_validation:
                    return self._process_ai_validation(ai_validation, solution, original_query)
                
            except Exception as e:
                logger.error(f"AI validation failed: {e}")
        
        # Fallback to rule-based validation
        return self._rule_based_validation(solution, original_query, context)
    
    def _process_ai_validation(self, ai_validation: str, solution: Dict, query: str) -> Dict[str, Any]:
        """Process AI validation results"""
        
        validation_checks = {
            "addresses_core_question": self._extract_validation_score(ai_validation, "address"),
            "completeness": self._extract_validation_score(ai_validation, "complete"),
            "feasibility": self._extract_validation_score(ai_validation, "feasible"),
            "clarity": self._extract_validation_score(ai_validation, "clear"),
            "actionability": self._extract_validation_score(ai_validation, "actionable")
        }
        
        # Extract refinements from AI response
        refinements = self._extract_refinements(ai_validation)
        
        # Calculate overall confidence
        scores = [check["score"] for check in validation_checks.values()]
        overall_confidence = sum(scores) / len(scores)
        
        # Apply AI-suggested refinements
        refined_solution = self._apply_ai_refinements(solution, refinements, ai_validation)
        
        return {
            "validated_solution": refined_solution,
            "validation_results": validation_checks,
            "ai_validation_summary": ai_validation[:300] + "..." if len(ai_validation) > 300 else ai_validation,
            "refinements_applied": refinements,
            "final_confidence": overall_confidence,
            "solution_quality": "high" if overall_confidence > 0.8 else "medium" if overall_confidence > 0.6 else "needs_improvement",
            "ai_enhanced": True
        }
    
    def _extract_validation_score(self, text: str, criterion: str) -> Dict[str, Any]:
        """Extract validation score for specific criterion"""
        
        # Look for criterion mentions with positive/negative indicators
        criterion_section = re.search(rf'{criterion}[^.\n]*([^.\n]+)', text, re.IGNORECASE)
        
        if criterion_section:
            section_text = criterion_section.group(1).lower()
            
            positive_indicators = ["yes", "good", "excellent", "strong", "adequate", "sufficient"]
            negative_indicators = ["no", "poor", "weak", "inadequate", "insufficient", "needs improvement"]
            
            positive_score = sum(1 for indicator in positive_indicators if indicator in section_text)
            negative_score = sum(1 for indicator in negative_indicators if indicator in section_text)
            
            if positive_score > negative_score:
                score = 0.8 + (positive_score * 0.05)
            elif negative_score > positive_score:
                score = 0.4 - (negative_score * 0.1)
            else:
                score = 0.7
            
            return {
                "passed": score > 0.6,
                "score": min(0.95, max(0.1, score)),
                "details": section_text[:100]
            }
        
        return {"passed": True, "score": 0.75, "details": f"{criterion} assessment completed"}
    
    def _extract_refinements(self, text: str) -> List[Dict[str, str]]:
        """Extract refinement suggestions from AI validation"""
        
        refinement_patterns = [
            r'improve[:\s]*([^.\n]+)',
            r'enhance[:\s]*([^.\n]+)',
            r'refine[:\s]*([^.\n]+)',
            r'add[:\s]*([^.\n]+)',
            r'consider[:\s]*([^.\n]+)'
        ]
        
        refinements = []
        for pattern in refinement_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 15:
                    refinements.append({
                        "area": "general_improvement",
                        "improvement": match.strip(),
                        "priority": "medium"
                    })
        
        return refinements[:5]  # Return top 5 refinements
    
    async def _apply_ai_refinements(self, solution: Dict, refinements: List[Dict], ai_validation: str) -> Dict[str, Any]:
        """Apply AI-suggested refinements to solution"""
        
        refined_solution = solution.copy()
        
        # Add AI-suggested improvements
        if refinements:
            refined_solution["ai_improvements"] = {
                "suggested_enhancements": [r["improvement"] for r in refinements],
                "validation_insights": ai_validation[:200] + "..." if len(ai_validation) > 200 else ai_validation,
                "improvement_priority": "high" if len(refinements) > 3 else "medium"
            }
        
        # Add additional considerations based on validation
        refined_solution["additional_considerations"] = [
            "Regular progress monitoring and adjustment",
            "Stakeholder communication and feedback loops",
            "Risk monitoring and contingency planning",
            "Performance measurement and optimization",
            "Documentation and knowledge transfer"
        ]
        
        # Add quick start guide
        refined_solution["quick_start_guide"] = self._generate_quick_start_guide(solution)
        
        return refined_solution
    
    def _generate_quick_start_guide(self, solution: Dict) -> List[str]:
        """Generate quick start guide for solution implementation"""
        
        return [
            "1. Set up development environment and required tools",
            "2. Create project structure and initialize repositories",
            "3. Implement core foundation components first",
            "4. Set up continuous integration and testing pipeline",
            "5. Begin iterative development with regular testing",
            "6. Deploy to staging environment for validation",
            "7. Collect feedback and iterate on implementation",
            "8. Prepare production deployment with monitoring"
        ]
    
    def _rule_based_validation(self, solution: Dict, query: str, context: Dict) -> Dict[str, Any]:
        """Fallback rule-based validation"""
        
        validation_checks = {
            "addresses_core_question": {"passed": True, "score": 0.8, "details": "Solution addresses main query components"},
            "completeness": {"passed": True, "score": 0.85, "details": "Solution includes key implementation components"},
            "feasibility": {"passed": True, "score": 0.8, "details": "Solution appears technically feasible"},
            "clarity": {"passed": True, "score": 0.85, "details": "Solution is clearly structured"},
            "actionability": {"passed": True, "score": 0.9, "details": "Solution provides actionable steps"}
        }
        
        overall_confidence = sum(check["score"] for check in validation_checks.values()) / len(validation_checks)
        
        return {
            "validated_solution": solution,
            "validation_results": validation_checks,
            "refinements_applied": [],
            "final_confidence": overall_confidence,
            "solution_quality": "good",
            "ai_enhanced": False
        }
    
    def _generate_reasoning_summary(self, reasoning_chain: List[ReasoningStep]) -> str:
        """Generate human-readable reasoning summary"""
        summary = "## Reasoning Process Analysis\n\n"
        
        for step in reasoning_chain:
            summary += f"**Step {step.step_id}: {step.description}**\n"
            summary += f"- Type: {step.reasoning_type.value.title()}\n"
            summary += f"- Confidence: {step.confidence:.1%}\n"
            if step.dependencies:
                summary += f"- Dependencies: Steps {', '.join(map(str, step.dependencies))}\n"
            summary += f"- Output: {str(step.output_data)[:100]}...\n\n"
        
        return summary
    
    # Simplified reasoning for medium complexity
    async def _execute_moderate_reasoning(self, query: str, context: Dict) -> Dict[str, Any]:
        """Execute moderate complexity reasoning with AI enhancement"""
        
        reasoning_steps = []
        
        # Step 1: Problem understanding
        understanding = await self._understand_problem_deeply(query, context)
        reasoning_steps.append(understanding)
        
        # Step 2: Direct solution generation with AI
        if self.nova_system:
            try:
                solution_prompt = f"""Provide a structured solution for this problem:

Problem: {query}
Understanding: {json.dumps(understanding, indent=2)}

Provide:
1. Direct solution approach
2. Implementation steps
3. Key considerations
4. Expected outcomes"""

                ai_solution = await self.nova_system.api_manager.get_ai_response(
                    solution_prompt,
                    "You are a problem solver. Provide clear, structured solutions.",
                    "general"
                )
                
                if ai_solution:
                    direct_solution = {
                        "ai_solution": ai_solution,
                        "structured_approach": self._extract_approach_from_ai(ai_solution),
                        "implementation_steps": self._extract_steps_from_ai(ai_solution),
                        "confidence": 0.88
                    }
                else:
                    direct_solution = await self._generate_direct_solution(query, context)
                    
            except Exception as e:
                logger.error(f"AI solution generation failed: {e}")
                direct_solution = await self._generate_direct_solution(query, context)
        else:
            direct_solution = await self._generate_direct_solution(query, context)
        
        reasoning_steps.append(direct_solution)
        
        # Step 3: Practical considerations
        practical_considerations = await self._add_practical_considerations(query, context)
        reasoning_steps.append(practical_considerations)
        
        return {
            "reasoning_chain": reasoning_steps,
            "final_solution": reasoning_steps[-1],
            "complexity_level": "medium",
            "confidence_score": 0.85,
            "implementation_ready": True,
            "ai_enhanced": bool(self.nova_system)
        }
    
    def _extract_approach_from_ai(self, ai_response: str) -> str:
        """Extract structured approach from AI response"""
        
        approach_patterns = [
            r'approach[:\s]*([^.\n]+(?:\.[^.\n]*)?)',
            r'method[:\s]*([^.\n]+(?:\.[^.\n]*)?)',
            r'strategy[:\s]*([^.\n]+(?:\.[^.\n]*)?)'
        ]
        
        for pattern in approach_patterns:
            match = re.search(pattern, ai_response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: use first substantial sentence
        sentences = re.split(r'[.!?]+', ai_response)
        for sentence in sentences:
            if len(sentence.strip()) > 25:
                return sentence.strip()
        
        return "Systematic implementation with iterative improvement"
    
    def _extract_steps_from_ai(self, ai_response: str) -> List[str]:
        """Extract implementation steps from AI response"""
        
        # Look for numbered steps
        numbered_steps = re.findall(r'\d+\.\s*([^.\n]+)', ai_response)
        
        if numbered_steps:
            return [step.strip() for step in numbered_steps[:8]]
        
        # Look for bullet points
        bullet_steps = re.findall(r'[-•*]\s*([^.\n]+)', ai_response)
        
        if bullet_steps:
            return [step.strip() for step in bullet_steps[:8]]
        
        # Default steps
        return [
            "Define requirements and objectives clearly",
            "Research and select appropriate tools/methods",
            "Create implementation plan with milestones",
            "Begin development with core functionality",
            "Test and validate each component",
            "Integrate and optimize overall solution",
            "Deploy and monitor performance",
            "Iterate based on feedback and results"
        ]
    
    async def _execute_simple_reasoning(self, query: str, context: Dict) -> Dict[str, Any]:
        """Execute simple reasoning for straightforward queries"""
        
        # Use AI for direct solution if available
        if self.nova_system:
            try:
                simple_prompt = f"""Provide a clear, direct answer to this question:

Question: {query}

Provide:
1. Direct answer
2. Key points to remember
3. Practical tips
4. Next steps if applicable

Keep response concise but comprehensive."""

                ai_response = await self.nova_system.api_manager.get_ai_response(
                    simple_prompt,
                    "You are a helpful assistant. Provide clear, direct answers.",
                    "general"
                )
                
                if ai_response:
                    direct_solution = {
                        "direct_answer": ai_response,
                        "solution_type": "ai_generated",
                        "confidence": 0.9,
                        "ai_enhanced": True
                    }
                else:
                    direct_solution = await self._generate_direct_solution(query, context)
                    
            except Exception as e:
                logger.error(f"Simple AI reasoning failed: {e}")
                direct_solution = await self._generate_direct_solution(query, context)
        else:
            direct_solution = await self._generate_direct_solution(query, context)
        
        return {
            "reasoning_chain": [direct_solution],
            "final_solution": direct_solution,
            "complexity_level": "simple",
            "confidence_score": 0.9,
            "implementation_ready": True,
            "ai_enhanced": bool(self.nova_system)
        }
    
    async def _generate_direct_solution(self, query: str, context: Dict) -> Dict[str, Any]:
        """Generate direct solution for straightforward problems"""
        
        return {
            "direct_answer": f"For your question about: {query[:100]}...",
            "solution_approach": "Direct implementation with best practices",
            "key_points": [
                "Identify core requirements and constraints",
                "Choose appropriate tools and methodologies",
                "Implement using proven patterns and practices",
                "Test thoroughly with real-world scenarios",
                "Monitor and optimize based on performance data"
            ],
            "practical_tips": [
                "Start with proven, well-documented approaches",
                "Break complex problems into smaller, manageable pieces",
                "Document your process and decisions for future reference",
                "Test incrementally and gather feedback early",
                "Plan for maintenance and future enhancements"
            ],
            "confidence": 0.85,
            "ai_enhanced": False
        }
    
    async def _add_practical_considerations(self, query: str, context: Dict) -> Dict[str, Any]:
        """Add practical implementation considerations"""
        
        return {
            "practical_considerations": [
                "Resource requirements and availability assessment",
                "Timeline constraints and milestone planning",
                "Integration requirements with existing systems",
                "Security and compliance considerations",
                "Maintenance and support planning",
                "Scalability and performance requirements"
            ],
            "success_factors": [
                "Clear objective definition and stakeholder alignment",
                "Adequate resource allocation and team expertise",
                "Regular progress monitoring and course correction",
                "Continuous improvement and optimization mindset",
                "Strong testing and quality assurance processes"
            ],
            "risk_mitigation": [
                "Identify potential risks early in planning phase",
                "Create contingency plans for critical dependencies",
                "Implement monitoring and alerting systems",
                "Plan for rollback procedures if needed",
                "Maintain clear communication channels"
            ]
        }
    async def _identify_risks_and_mitigations(self, analysis: Dict[str, Any]) -> list[dict]:
        """Return key risks + mitigations derived from analysis (simple heuristic)."""
        risks = []

        # Common buckets – adjust per your product/domain
        if analysis.get("assumptions"):
            risks.append({"risk": "Unvalidated assumptions may be wrong",
                          "mitigation": "Run quick experiments / A-B tests to validate"})

        if analysis.get("data_gaps"):
            risks.append({"risk": "Missing or low-quality data",
                          "mitigation": "Add telemetry, logging, and quality checks"})

        if analysis.get("scalability_concerns"):
            risks.append({"risk": "Solution may not scale under peak load",
                          "mitigation": "Add caching, queueing, load tests, autoscaling"})

        if analysis.get("security_privacy"):
            risks.append({"risk": "Security/privacy obligations not fully addressed",
                          "mitigation": "Threat model review, secrets management, PII minimization"})

        # Always return at least one generic item so UI never looks empty
        if not risks:
            risks.append({"risk": "Unknown risks",
                          "mitigation": "Ship in stages, monitor, and roll back quickly if needed"})

        return risks

    async def _define_success_metrics(self, analysis: Dict[str, Any]) -> dict[str, list[str]]:
        """Propose success metrics in three buckets."""
        return {
            "technical": [
                "p95 latency < 1.5s", "error rate < 0.5%", "timeout rate < 1%"
            ],
            "business": [
                "task completion rate +X%", "retention +Y%", "conversion +Z%"
            ],
            "user": [
                "CSAT ≥ 4.3/5", "thumbs-up rate ≥ 80%", "complaints per 1k chats ↓"
            ],
        }

    async def _prioritize_next_steps(self, analysis: Dict[str, Any]) -> list[str]:
        """Turn the synthesis into concrete, ordered steps."""
        steps = [
            "Clarify goal & constraints with 3-bullet brief",
            "Design target workflow (diagram + responsibilities)",
            "Build MVP path (happy path only) behind feature flag",
            "Add guardrails: input validation, quotas, fallbacks",
            "Ship to small beta cohort, collect metrics",
            "Iterate on top 2 issues from telemetry & feedback"
        ]
        return steps

    def _generate_structured_perspectives(self, knowledge: Dict, context: Dict) -> Dict[str, Any]:
     """
     Fallback structured perspective generator when AI analysis fails.
     Creates dummy but structured perspectives so reasoning never breaks.
     """
     query = knowledge.get("query", "the given problem")

     return {
        "technical": {
            "considerations": [f"Scalability issues in {query}", "Performance optimization"],
            "solutions": ["Use caching", "Optimize database queries"],
            "risks": ["Technical debt", "Integration complexity"],
            "metrics": ["Latency < 200ms", "99.9% uptime"]
        },
        "business": {
            "considerations": ["ROI", "Market timing", "Cost efficiency"],
            "solutions": ["Subscription model", "Tiered pricing"],
            "risks": ["High competition", "Budget overruns"],
            "metrics": ["Customer acquisition rate", "Gross margin"]
        },
        "user_experience": {
            "considerations": ["Ease of use", "Accessibility"],
            "solutions": ["Responsive UI", "Onboarding tutorial"],
            "risks": ["Confusing UX", "Accessibility compliance"],
            "metrics": ["User retention", "NPS score"]
        },
        "operational": {
            "considerations": ["Deployment", "Monitoring"],
            "solutions": ["CI/CD pipeline", "Automated alerts"],
            "risks": ["Downtime", "Alert fatigue"],
            "metrics": ["MTTR", "Error rate"]
        }
    }

    def _generate_structured_solution(self, knowledge: Dict, context: Dict) -> Dict[str, Any]:
     """
     Fallback structured solution generator when AI synthesis fails.
     Ensures reasoning pipeline always returns a solution structure.
     """
     query = knowledge.get("query", "the given problem")

     return {
        "primary_solution": {
            "solution_title": f"Baseline strategy for {query}",
            "core_approach": "Step-by-step structured resolution",
            "key_components": [
                "Define requirements",
                "Break problem into smaller parts",
                "Assign responsibilities",
                "Validate with tests"
            ],
            "unique_advantages": [
                "Clear stepwise plan",
                "Low initial risk",
                "Easy to expand"
            ],
            "ai_rationale": "This is a fallback structured plan synthesized locally."
        },
        "implementation_plan": {
            "phase_1": ["Requirement gathering", "Resource allocation"],
            "phase_2": ["Design architecture", "Prototype key modules"],
            "phase_3": ["Testing", "Deployment", "Monitoring"]
        },
        "risks": [
            "Incomplete requirements",
            "Scalability bottlenecks",
            "Operational overhead"
        ],
        "mitigations": [
            "Iterative reviews",
            "Automated monitoring",
            "Scalable infra from start"
        ]
    }

    async def _generate_structured_solution(self, knowledge: Dict, perspectives: Dict, context: Dict) -> Dict[str, Any]:
        """Fallback solution synthesis if AI call fails"""
        
        try:
            primary_solution = {
                "solution_title": "Integrated Strategic Solution",
                "core_approach": "Balanced implementation combining technical feasibility, business impact, and user experience.",
                "key_components": [
                    "Technical architecture aligned with scalability goals",
                    "Business model considerations with ROI analysis",
                    "User-centric design for accessibility and satisfaction",
                    "Risk mitigation and compliance measures",
                    "Operational monitoring and continuous improvement"
                ],
                "unique_advantages": [
                    "Holistic approach covering multiple perspectives",
                    "Designed for long-term maintainability",
                    "Scalable across different domains and use cases"
                ],
                "ai_rationale": "Generated via structured fallback synthesis using embedded knowledge and predefined reasoning patterns."
            }

            implementation_plan = {
                "short_term": [
                    "Establish technical proof of concept",
                    "Conduct stakeholder workshops",
                    "Baseline current system performance"
                ],
                "medium_term": [
                    "Develop scalable system architecture",
                    "Implement monitoring and alerting",
                    "Iterate with user feedback loops"
                ],
                "long_term": [
                    "Expand features with predictive analytics",
                    "Introduce advanced personalization",
                    "Optimize for cost and performance efficiency"
                ]
            }

            return {
                "primary_solution": primary_solution,
                "implementation_plan": implementation_plan,
                "ai_enhanced": False,
                "generated_with": "structured_fallback",
                "confidence": 0.7
            }

        except Exception as e:
            logger.error(f"Structured solution generation failed: {e}")
            return {
                "primary_solution": {
                    "solution_title": "Basic Solution",
                    "core_approach": "Fallback approach with limited detail"
                },
                "implementation_plan": {},
                "ai_enhanced": False,
                "generated_with": "error_fallback",
                "confidence": 0.3,
                "error": str(e)
            }

# Usage Integration class for NOVA system
class ReasoningIntegration:
    """Production-ready integration layer for reasoning engine with NOVA chatbot"""
    
    def __init__(self, nova_system):
        self.nova_system = nova_system
        self.reasoning_engine = AdvancedReasoningEngine(nova_system)
        self.integration_cache = {}
        
    async def enhance_response_with_reasoning(self, user_query: str, context: Dict, 
                                           base_response: str, timeout: int = 30) -> Tuple[str, Dict]:
        """Enhance AI response with advanced reasoning within timeout"""
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = hashlib.md5(f"{user_query}_{context.get('user_id', 'default')}".encode()).hexdigest()
            if cache_key in self.integration_cache:
                cached_result = self.integration_cache[cache_key]
                if time.time() - cached_result['timestamp'] < 1800:  # 30 minutes cache
                    logger.info("Using cached reasoning result")
                    return cached_result['response'], cached_result['reasoning']
            
            # Analyze if reasoning enhancement would add value
            complexity_analysis = await self.reasoning_engine.analyze_query_complexity(user_query)
            
            # Apply timeout check
            if time.time() - start_time > timeout * 0.8:  # 80% of timeout used
                logger.warning("Reasoning timeout approaching, returning base response")
                return base_response, {"reasoning_applied": False, "timeout": True}
            
            if complexity_analysis["requires_multi_step"]:
                # Apply advanced reasoning with timeout
                reasoning_task = asyncio.create_task(
                    self.reasoning_engine.execute_advanced_reasoning(user_query, context)
                )
                
                try:
                    reasoning_result = await asyncio.wait_for(reasoning_task, timeout=timeout-5)
                    
                    # Enhance the base response with reasoning insights
                    enhanced_response = await self._integrate_reasoning_with_response(
                        base_response, reasoning_result, user_query
                    )
                    
                    # Cache successful result
                    self.integration_cache[cache_key] = {
                        'response': enhanced_response,
                        'reasoning': reasoning_result,
                        'timestamp': time.time()
                    }
                    
                    return enhanced_response, reasoning_result
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Reasoning timeout after {timeout}s, returning base response")
                    return base_response, {"reasoning_applied": False, "timeout": True}
            
            return base_response, {"reasoning_applied": False, "complexity": "low"}
            
        except Exception as e:
            logger.error(f"Reasoning enhancement error: {e}")
            return base_response, {"reasoning_applied": False, "error": str(e)}
    
    async def _integrate_reasoning_with_response(self, base_response: str, 
                                                 reasoning_result: Dict, query: str) -> str:
        """Integrate reasoning insights with AI response"""
        
        complexity_level = reasoning_result.get("complexity_level", "medium")
        
        if complexity_level == "high":
            # For complex queries, structure response with comprehensive reasoning
            reasoning_summary = reasoning_result.get("reasoning_summary", "")
            final_solution = reasoning_result.get("final_solution", {})
            
            enhanced_response = f"""{base_response}

---

## 🧠 Advanced Analysis & Strategic Framework

{reasoning_summary}

### 🎯 Comprehensive Solution Strategy:
{self._format_solution_details(final_solution)}

### 📋 Implementation Roadmap:
{self._format_implementation_plan(final_solution.get("implementation_plan", {}))}

### ⚡ Next Steps & Action Items:
{self._format_next_steps(final_solution)}

### 📊 Quality Assurance:
- **Solution Confidence:** {reasoning_result.get('confidence_score', 0.85):.1%}
- **Implementation Ready:** {'✅ Yes' if reasoning_result.get('implementation_ready') else '⚠️ Needs refinement'}
- **Processing Time:** {reasoning_result.get('processing_time', 0):.1f}s

---
*🔬 Enhanced with Advanced Reasoning Engine - {reasoning_result.get('ai_calls_made', 0)} AI analysis calls*
"""
            
        elif complexity_level == "medium":
            # For medium complexity, add structured insights
            final_solution = reasoning_result.get("final_solution", {})
            
            enhanced_response = f"""{base_response}

## 💡 Strategic Analysis & Recommendations

### 🎯 Recommended Approach:
{self._format_recommended_approach(final_solution)}

### 📋 Implementation Guide:
{self._format_implementation_steps(final_solution)}

### ⚠️ Key Considerations:
{self._format_considerations(final_solution)}

---
*Enhanced with Strategic Reasoning Analysis*
"""
        else:
            # For simple queries, minimal enhancement
            enhanced_response = f"""{base_response}

### 💡 Additional Insights:
{self._format_simple_insights(reasoning_result.get("final_solution", {}))}
"""
        
        return enhanced_response
    
    def _format_solution_details(self, solution: Dict) -> str:
        """Format solution details for display"""
        if not solution:
            return "Solution analysis in progress..."
        
        details = ""
        
        # Primary solution
        if "primary_solution" in solution:
            primary = solution["primary_solution"]
            details += f"**🎯 Primary Solution:** {primary.get('solution_title', 'Integrated Solution')}\n"
            details += f"**📈 Core Approach:** {primary.get('core_approach', 'Comprehensive implementation')}\n\n"
            
            if "key_components" in primary:
                details += "**🔧 Key Components:**\n"
                for i, component in enumerate(primary["key_components"], 1):
                    details += f"   {i}. {component}\n"
                details += "\n"
            
            if "unique_advantages" in primary:
                details += "**⭐ Unique Advantages:**\n"
                for advantage in primary["unique_advantages"]:
                    details += f"   • {advantage}\n"
        
        # AI insights if available
        if solution.get("ai_enhanced") and "ai_rationale" in solution.get("primary_solution", {}):
            details += f"\n**🤖 AI Analysis:** {solution['primary_solution']['ai_rationale'][:200]}...\n"
        
        return details

    # ✅ Safe wrapper with timeout + fallback
    async def enhance_response_with_reasoning(self, user_query: str, context: dict, base_response: str):
        complexity = await self.reasoning_engine.analyze_query_complexity(user_query)

        if not complexity.get("requires_multi_step"):
            return base_response, {"reasoning_applied": False}

        try:
            reasoning_result = await asyncio.wait_for(
                self.reasoning_engine.execute_advanced_reasoning(user_query, context),
                timeout=6.0
            )
            enhanced = await self._integrate_reasoning_with_response(base_response, reasoning_result, user_query)
            return enhanced, reasoning_result
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Reasoning fallback: %s", e)
            return base_response, {"reasoning_applied": False, "error": str(e)}
    
    def _format_implementation_plan(self, plan: Dict) -> str:
        """Format implementation plan for display"""
        if not plan or "phases" not in plan:
            return "Implementation roadmap being optimized..."
        
        plan_text = ""
        
        for phase in plan["phases"]:
            phase_num = phase.get('phase', 1)
            phase_title = phase.get('title', f'Phase {phase_num}')
            duration = phase.get('duration', 'TBD')
            
            plan_text += f"**📅 {phase_title}** ({duration})\n"
            
            deliverables = phase.get("deliverables", [])
            for deliverable in deliverables:
                plan_text += f"   ✓ {deliverable}\n"
            
            success_criteria = phase.get("success_criteria", [])
            if success_criteria:
                plan_text += f"   🎯 Success: {', '.join(success_criteria)}\n"
            
            plan_text += "\n"
        
        # Add resource requirements
        if "resource_requirements" in plan:
            resources = plan["resource_requirements"]
            plan_text += "**📊 Resource Requirements:**\n"
            for key, value in resources.items():
                plan_text += f"   • {key.title()}: {value}\n"
        
        return plan_text
    
    def _format_next_steps(self, solution: Dict) -> str:
        """Format next steps for display"""
        
        next_steps = []
        
        # Extract from various solution parts
        if "next_steps" in solution:
            next_steps.extend(solution["next_steps"])
        
        if "quick_start_guide" in solution:
            next_steps.extend(solution["quick_start_guide"][:3])
        
        if not next_steps:
            next_steps = [
                "1. 🔍 Analyze current situation and define clear objectives",
                "2. 📋 Create detailed implementation plan with milestones",
                "3. 🛠️ Set up development environment and required tools",
                "4. 🚀 Begin implementation with core functionality first",
                "5. 🧪 Test thoroughly and gather feedback continuously",
                "6. 📈 Deploy, monitor, and optimize based on real-world usage"
            ]
        
        steps_text = ""
        for step in next_steps[:6]:
            if not step.startswith(('1.', '2.', '3.', '4.', '5.', '6.')):
                step_num = len([s for s in steps_text.split('\n') if s.strip()]) + 1
                step = f"{step_num}. {step}"
            steps_text += f"{step}\n"
        
        return steps_text
    
    def _format_recommended_approach(self, solution: Dict) -> str:
        """Format recommended approach for medium complexity"""
        
        if "ai_solution" in solution:
            # Extract key recommendations from AI response
            ai_text = solution["ai_solution"]
            
            # Look for recommendations
            recommendations = re.findall(r'recommend[:\s]*([^.\n]+)', ai_text, re.IGNORECASE)
            
            if recommendations:
                approach_text = "**AI-Recommended Strategy:**\n"
                for i, rec in enumerate(recommendations[:3], 1):
                    approach_text += f"{i}. {rec.strip()}\n"
                return approach_text
        
        # Fallback approach
        return """**Systematic Implementation Strategy:**
1. **Assess** current situation and define requirements clearly
2. **Design** solution architecture with scalability considerations  
3. **Implement** using proven methodologies and best practices
4. **Test** thoroughly with comprehensive validation scenarios
5. **Deploy** with proper monitoring and feedback mechanisms
6. **Optimize** based on performance data and user feedback"""
    
    def _format_implementation_steps(self, solution: Dict) -> str:
        """Format implementation steps for medium complexity"""
        
        if "structured_approach" in solution:
            approach = solution["structured_approach"]
            return f"**Primary Approach:** {approach}\n\n**Implementation Focus:**\n• Systematic execution\n• Quality validation\n• Continuous improvement"
        
        return """**Implementation Framework:**
• **Foundation:** Set up core infrastructure and tools
• **Development:** Implement features using iterative approach
• **Validation:** Test functionality and performance thoroughly
• **Deployment:** Deploy with monitoring and support systems
• **Optimization:** Continuously improve based on usage data"""
    
    def _format_considerations(self, solution: Dict) -> str:
        """Format key considerations"""
        
        considerations = []
        
        if "practical_considerations" in solution:
            considerations.extend(solution["practical_considerations"][:3])
        
        if not considerations:
            considerations = [
                "⚠️ **Resource Planning:** Ensure adequate time and technical resources",
                "🔒 **Security & Compliance:** Implement security best practices throughout",
                "📈 **Performance Monitoring:** Set up metrics and monitoring from the start"
            ]
        
        return '\n'.join(considerations)
    
    def _format_simple_insights(self, solution: Dict) -> str:
        """Format insights for simple queries"""
        
        if "ai_solution" in solution or "direct_answer" in solution:
            return "✨ **Pro Tip:** Consider documenting your approach for future reference and continuous improvement."
        
        return "✨ **Additional Guidance:** Break down complex tasks into smaller steps for better results."


# Advanced Reasoning Engine Factory
class ReasoningEngineFactory:
    """Factory for creating reasoning engines with different configurations"""
    
    @staticmethod
    def create_production_engine(nova_system) -> AdvancedReasoningEngine:
        """Create production-ready reasoning engine"""
        engine = AdvancedReasoningEngine(nova_system)
        
        # Configure for production
        engine.reasoning_cache = {}  # Fresh cache
        engine.context_windows = {}
        
        logger.info("Production reasoning engine created")
        return engine
    
    @staticmethod
    def create_reasoning_integration(nova_system) -> ReasoningIntegration:
        """Create reasoning integration for NOVA system"""
        integration = ReasoningIntegration(nova_system)
        
        logger.info("Reasoning integration created for NOVA system")
        return integration


# Performance Monitor for Reasoning Engine
class ReasoningPerformanceMonitor:
    """Monitor reasoning engine performance and optimization"""
    
    def __init__(self):
        self.performance_metrics = {
            "reasoning_requests": 0,
            "successful_reasoning": 0,
            "average_reasoning_time": 0.0,
            "cache_hits": 0,
            "timeout_incidents": 0,
            "ai_calls_made": 0,
            "complexity_distribution": {"high": 0, "medium": 0, "low": 0}
        }
        
        self.recent_performance = deque(maxlen=100)
    
    def record_reasoning_performance(self, complexity: str, processing_time: float, 
                                   success: bool, ai_calls: int = 0, cached: bool = False):
        """Record reasoning performance metrics"""
        
        self.performance_metrics["reasoning_requests"] += 1
        
        if success:
            self.performance_metrics["successful_reasoning"] += 1
        
        if cached:
            self.performance_metrics["cache_hits"] += 1
        
        self.performance_metrics["ai_calls_made"] += ai_calls
        self.performance_metrics["complexity_distribution"][complexity] += 1
        
        # Update average processing time
        if self.performance_metrics["average_reasoning_time"] == 0:
            self.performance_metrics["average_reasoning_time"] = processing_time
        else:
            total_time = self.performance_metrics["average_reasoning_time"] * (self.performance_metrics["reasoning_requests"] - 1)
            self.performance_metrics["average_reasoning_time"] = (total_time + processing_time) / self.performance_metrics["reasoning_requests"]
        
        # Store recent performance
        self.recent_performance.append({
            "timestamp": time.time(),
            "complexity": complexity,
            "processing_time": processing_time,
            "success": success,
            "ai_calls": ai_calls,
            "cached": cached
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        total_requests = self.performance_metrics["reasoning_requests"]
        
        if total_requests == 0:
            return {"status": "no_data", "message": "No reasoning requests processed yet"}
        
        success_rate = self.performance_metrics["successful_reasoning"] / total_requests
        cache_hit_rate = self.performance_metrics["cache_hits"] / total_requests
        
        return {
            "total_reasoning_requests": total_requests,
            "success_rate": f"{success_rate:.2%}",
            "average_processing_time": f"{self.performance_metrics['average_reasoning_time']:.2f}s",
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
            "complexity_distribution": self.performance_metrics["complexity_distribution"],
            "total_ai_calls": self.performance_metrics["ai_calls_made"],
            "timeout_incidents": self.performance_metrics["timeout_incidents"],
            "recent_trend": self._calculate_recent_trend(),
            "optimization_suggestions": self._generate_optimization_suggestions()
        }
    
    def _calculate_recent_trend(self) -> str:
        """Calculate recent performance trend"""
        
        if len(self.recent_performance) < 10:
            return "insufficient_data"
        
        recent_times = [p["processing_time"] for p in list(self.recent_performance)[-5:]]
        older_times = [p["processing_time"] for p in list(self.recent_performance)[-10:-5]]
        
        recent_avg = sum(recent_times) / len(recent_times)
        older_avg = sum(older_times) / len(older_times)
        
        if recent_avg < older_avg * 0.9:
            return "improving"
        elif recent_avg > older_avg * 1.1:
            return "degrading"
        else:
            return "stable"
    
    def _generate_optimization_suggestions(self) -> List[str]:
        """Generate optimization suggestions based on performance"""
        
        suggestions = []
        
        avg_time = self.performance_metrics["average_reasoning_time"]
        cache_rate = self.performance_metrics["cache_hits"] / max(1, self.performance_metrics["reasoning_requests"])
        
        if avg_time > 15:
            suggestions.append("Consider increasing cache TTL to reduce processing time")
        
        if cache_rate < 0.2:
            suggestions.append("Implement more aggressive caching for repeated query patterns")
        
        if self.performance_metrics["timeout_incidents"] > 5:
            suggestions.append("Consider reducing default timeout or optimizing AI call efficiency")
        
        high_complexity_ratio = self.performance_metrics["complexity_distribution"]["high"] / max(1, self.performance_metrics["reasoning_requests"])
        if high_complexity_ratio > 0.5:
            suggestions.append("Consider pre-processing complex queries to reduce real-time processing")
        
        return suggestions


# Export classes for integration
__all__ = [
    "AdvancedReasoningEngine",
    "ReasoningIntegration", 
    "ReasoningType",
    "ReasoningStep",
    "ReasoningEngineFactory",
    "ReasoningPerformanceMonitor"
]