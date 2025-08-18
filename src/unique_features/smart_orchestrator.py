# src/unique_features/smart_orchestrator.py
import asyncio
import random
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests
import os
from .api_drift_detector import APIPerformanceDrifter

class IntelligentAPIOrchestrator:
    """Smart API orchestration with performance-based selection"""
    
    def __init__(self):
        self.api_pool = [
            {
                'name': 'groq',
                'url': 'https://api.groq.com/openai/v1/chat/completions',
                'models': ['llama-3.1-8b-instant', 'mixtral-8x7b-32768'],
                'speed': 'fast',
                'quality': 'good',
                'cost': 'free',
                'daily_limit': 14400,
                'priority': 1
            },
            {
                'name': 'openrouter',
                'url': 'https://openrouter.ai/api/v1/chat/completions',
                'models': ['microsoft/phi-3-medium-128k-instruct:free', 'mistralai/mistral-7b-instruct:free'],
                'speed': 'medium',
                'quality': 'excellent',
                'cost': 'free',
                'daily_limit': 200,
                'priority': 2
            },
            {
                'name': 'huggingface',
                'url': 'https://api-inference.huggingface.co/models/',
                'models': ['microsoft/DialoGPT-large', 'facebook/blenderbot-400M-distill'],
                'speed': 'slow',
                'quality': 'basic',
                'cost': 'free',
                'daily_limit': 1000,
                'priority': 3
            },
            {
                'name': 'local',
                'url': None,
                'models': ['fallback_model'],
                'speed': 'medium',
                'quality': 'basic',
                'cost': 'free',
                'daily_limit': float('inf'),
                'priority': 4
            }
        ]
        
        self.api_performance_scores = {}
        self.usage_counts = {api['name']: 0 for api in self.api_pool}
        self.performance_monitor = APIPerformanceDrifter()
        
        # Context-based selection weights
        self.selection_weights = {
            'quality': {'groq': 0.7, 'openrouter': 0.9, 'huggingface': 0.5, 'local': 0.3},
            'speed': {'groq': 0.9, 'openrouter': 0.7, 'huggingface': 0.4, 'local': 0.6},
            'reliability': {'groq': 0.8, 'openrouter': 0.7, 'huggingface': 0.6, 'local': 0.9}
        }
        
        self.load_performance_history()
    
    def load_performance_history(self):
        """Load historical performance data"""
        try:
            with open('data/performance_logs/api_performance_history.json', 'r') as f:
                self.api_performance_scores = json.load(f)
        except FileNotFoundError:
            self.api_performance_scores = {api['name']: 0.5 for api in self.api_pool}
    
    def save_performance_history(self):
        """Save performance data"""
        os.makedirs('data/performance_logs', exist_ok=True)
        with open('data/performance_logs/api_performance_history.json', 'w') as f:
            json.dump(self.api_performance_scores, f, indent=2)
    
    def calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score (0-1)"""
        
        complexity_factors = {
            'length': min(len(query) / 500, 1.0),
            'technical_terms': self.count_technical_terms(query) / 10,
            'multi_part': 1.0 if any(word in query.lower() for word in 
                                   ['and', 'also', 'furthermore', 'additionally']) else 0.3,
            'specificity': self.assess_specificity(query)
        }
        
        return min(sum(complexity_factors.values()) / len(complexity_factors), 1.0)
    
    def count_technical_terms(self, query: str) -> int:
        """Count technical terms in query"""
        technical_terms = [
            'algorithm', 'function', 'variable', 'array', 'object', 'class',
            'database', 'server', 'api', 'framework', 'library', 'debugging',
            'optimization', 'performance', 'scalability', 'architecture'
        ]
        
        return sum(1 for term in technical_terms if term in query.lower())
    
    def assess_specificity(self, query: str) -> float:
        """Assess query specificity"""
        
        specific_indicators = [
            'how to', 'step by step', 'example of', 'implement', 'create',
            'build', 'design', 'optimize', 'fix', 'debug', 'troubleshoot'
        ]
        
        specificity_score = sum(1 for indicator in specific_indicators 
                               if indicator in query.lower()) / len(specific_indicators)
        
        return min(specificity_score * 2, 1.0)  # Amplify the score
    
    async def smart_api_selection(self, query: str, user_priority: str = 'balanced',
                                 context: Dict = None) -> str:
        """Intelligently select best API based on multiple factors"""
        
        query_complexity = self.calculate_query_complexity(query)
        
        # Get API health statuses
        api_scores = {}
        
        for api in self.api_pool:
            api_name = api['name']
            
            # Base score from static weights
            if user_priority == 'quality':
                base_score = self.selection_weights['quality'][api_name]
            elif user_priority == 'speed':
                base_score = self.selection_weights['speed'][api_name]
            elif user_priority == 'reliability':
                base_score = self.selection_weights['reliability'][api_name]
            else:  # balanced
                base_score = (
                    self.selection_weights['quality'][api_name] * 0.4 +
                    self.selection_weights['speed'][api_name] * 0.3 +
                    self.selection_weights['reliability'][api_name] * 0.3
                )
            
            # Adjust based on performance history
            performance_score = self.api_performance_scores.get(api_name, 0.5)
            
            # Adjust based on usage limits
            usage_factor = self.calculate_usage_factor(api_name)
            
            # Adjust based on query complexity
            complexity_factor = self.calculate_complexity_factor(api_name, query_complexity)
            
            # Get real-time health
            health = self.performance_monitor.get_api_health(api_name)
            health_factor = self.health_status_to_factor(health['status'])
            
            # Combine all factors
            final_score = (
                base_score * 0.3 +
                performance_score * 0.25 +
                usage_factor * 0.15 +
                complexity_factor * 0.15 +
                health_factor * 0.15
            )
            
            api_scores[api_name] = final_score
        
        # Select best API
        best_api = max(api_scores, key=api_scores.get)
        
        print(f"🎯 Smart API Selection: {best_api} (score: {api_scores[best_api]:.2f})")
        print(f"   Query complexity: {query_complexity:.2f}")
        print(f"   User priority: {user_priority}")
        print(f"   All scores: {api_scores}")
        
        return best_api
    
    def calculate_usage_factor(self, api_name: str) -> float:
        """Calculate usage factor based on daily limits"""
        
        api_info = next(api for api in self.api_pool if api['name'] == api_name)
        daily_limit = api_info['daily_limit']
        
        if daily_limit == float('inf'):
            return 1.0
        
        current_usage = self.usage_counts.get(api_name, 0)
        usage_ratio = current_usage / daily_limit
        
        if usage_ratio >= 0.9:
            return 0.1  # Almost at limit
        elif usage_ratio >= 0.7:
            return 0.5  # Getting close to limit
        else:
            return 1.0  # Plenty of usage left
    
    def calculate_complexity_factor(self, api_name: str, complexity: float) -> float:
        """Adjust score based on query complexity and API capabilities"""
        
        # High-quality APIs handle complex queries better
        if complexity > 0.7:  # High complexity
            if api_name in ['openrouter', 'groq']:
                return 1.0  # These APIs handle complex queries well
            else:
                return 0.6  # Lower-quality APIs struggle with complex queries
        
        elif complexity > 0.4:  # Medium complexity
            if api_name in ['groq', 'openrouter']:
                return 1.0
            elif api_name == 'huggingface':
                return 0.8
            else:
                return 0.7
        
        else:  # Low complexity
            return 1.0  # All APIs can handle simple queries
    
    def health_status_to_factor(self, health_status: str) -> float:
        """Convert health status to numerical factor"""
        
        health_factors = {
            'healthy': 1.0,
            'degraded': 0.6,
            'unhealthy': 0.2,
            'unknown': 0.5
        }
        
        return health_factors.get(health_status, 0.5)
    
    async def try_api_cascade(self, query: str, api_priority_list: List[str],
                             max_tokens: int = 500) -> Dict[str, Any]:
        """Try APIs in priority order with intelligent fallback"""
        
        for api_name in api_priority_list:
            try:
                start_time = time.time()
                
                print(f"🔄 Trying API: {api_name}")
                
                result = await self.call_api(api_name, query, max_tokens)
                
                response_time = time.time() - start_time
                
                if result and self.validate_response_quality(result.get('content', '')):
                    # Monitor performance
                    await self.performance_monitor.monitor_api_call(
                        api_name, result['content'], response_time, True
                    )
                    
                    # Update usage count
                    self.usage_counts[api_name] += 1
                    
                    # Update performance score
                    self.update_performance_score(api_name, True, response_time)
                    
                    print(f"✅ Success with {api_name} in {response_time:.2f}s")
                    
                    return {
                        'success': True,
                        'api_used': api_name,
                        'response': result['content'],
                        'response_time': response_time,
                        'metadata': result.get('metadata', {})
                    }
                
            except Exception as e:
                response_time = time.time() - start_time
                
                print(f"❌ {api_name} failed: {str(e)}")
                
                # Log failure
                await self.performance_monitor.monitor_api_call(
                    api_name, '', response_time, False
                )
                
                # Update performance score negatively
                self.update_performance_score(api_name, False, response_time)
                
                self.log_api_failure(api_name, e)
                continue  # Try next API
        
        # All APIs failed - return fallback response
        return {
            'success': False,
            'api_used': 'fallback',
            'response': self.get_fallback_response(query),
            'response_time': 0.1,
            'metadata': {'fallback': True}
        }
    
    async def call_api(self, api_name: str, query: str, max_tokens: int = 500) -> Dict[str, Any]:
        """Call specific API"""
        
        if api_name == 'groq':
            return await self.call_groq_api(query, max_tokens)
        elif api_name == 'openrouter':
            return await self.call_openrouter_api(query, max_tokens)
        elif api_name == 'huggingface':
            return await self.call_huggingface_api(query, max_tokens)
        elif api_name == 'local':
            return await self.call_local_model(query, max_tokens)
        else:
            raise ValueError(f"Unknown API: {api_name}")
    
    async def call_groq_api(self, query: str, max_tokens: int) -> Dict[str, Any]:
        """Call Groq API"""
        
        if not os.getenv('GROQ_API_KEY'):
            raise Exception("Groq API key not available")
        
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Authorization': f"Bearer {os.getenv('GROQ_API_KEY')}",
                'Content-Type': 'application/json'
            },
            json={
                'model': 'llama-3.1-8b-instant',
                'messages': [
                    {'role': 'system', 'content': 'You are a helpful AI assistant.'},
                    {'role': 'user', 'content': query}
                ],
                'max_tokens': max_tokens,
                'temperature': 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                'content': result['choices'][0]['message']['content'],
                'metadata': {'model': 'llama-3.1-8b-instant', 'provider': 'groq'}
            }
        else:
            raise Exception(f"Groq API error: {response.status_code}")
    
    async def call_openrouter_api(self, query: str, max_tokens: int) -> Dict[str, Any]:
        """Call OpenRouter API"""
        
        if not os.getenv('OPENROUTER_API_KEY'):
            raise Exception("OpenRouter API key not available")
        
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://nova2027ultra.ai',
                'X-Title': 'NOVA 2027 ULTRA PRO'
            },
            json={
                'model': 'microsoft/phi-3-medium-128k-instruct:free',
                'messages': [
                    {'role': 'system', 'content': 'You are a helpful AI assistant.'},
                    {'role': 'user', 'content': query}
                ],
                'max_tokens': max_tokens,
                'temperature': 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                'content': result['choices'][0]['message']['content'],
                'metadata': {'model': 'phi-3-medium', 'provider': 'openrouter'}
            }
        else:
            raise Exception(f"OpenRouter API error: {response.status_code}")
    
    async def call_huggingface_api(self, query: str, max_tokens: int) -> Dict[str, Any]:
        """Call HuggingFace API"""
        
        if not os.getenv('HUGGINGFACE_API_KEY'):
            raise Exception("HuggingFace API key not available")
        
        response = requests.post(
            'https://api-inference.huggingface.co/models/microsoft/DialoGPT-large',
            headers={
                'Authorization': f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}",
                'Content-Type': 'application/json'
            },
            json={
                'inputs': query,
                'parameters': {
                    'max_new_tokens': max_tokens,
                    'temperature': 0.7,
                    'return_full_text': False
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return {
                    'content': result[0].get('generated_text', ''),
                    'metadata': {'model': 'DialoGPT-large', 'provider': 'huggingface'}
                }
        
        raise Exception(f"HuggingFace API error: {response.status_code}")
    
    async def call_local_model(self, query: str, max_tokens: int) -> Dict[str, Any]:
        """Call local fallback model (rule-based)"""
        
        # Simple rule-based responses
        local_responses = {
            'greeting': "Hello! I'm NOVA, your AI assistant. How can I help you today?",
            'coding': "I can help you with coding questions. Please provide more details about your programming challenge.",
            'career': "I offer career guidance including resume improvement and interview preparation. What specific career help do you need?",
            'general': "I'm here to assist you with various questions. Could you please provide more specific details about what you're looking for?"
        }
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            response = local_responses['greeting']
        elif any(word in query_lower for word in ['code', 'programming', 'function']):
            response = local_responses['coding']
        elif any(word in query_lower for word in ['resume', 'job', 'career', 'interview']):
            response = local_responses['career']
        else:
            response = local_responses['general']
        
        return {
            'content': response,
            'metadata': {'model': 'local_fallback', 'provider': 'local'}
        }
    
    def validate_response_quality(self, response: str) -> bool:
        """Validate response quality"""
        
        if not response or len(response.strip()) < 10:
            return False
        
        # Check for common error indicators
        error_indicators = [
            'error', 'failed', 'unavailable', 'timeout',
            'try again', 'service unavailable', 'rate limit'
        ]
        
        if any(indicator in response.lower() for indicator in error_indicators):
            return False
        
        return True
    
    def update_performance_score(self, api_name: str, success: bool, response_time: float):
        """Update API performance score"""
        
        current_score = self.api_performance_scores.get(api_name, 0.5)
        
        if success:
            # Positive adjustment based on speed
            speed_factor = max(0.1, 1.0 - (response_time / 10.0))  # Faster = better
            adjustment = 0.1 * speed_factor
        else:
            # Negative adjustment for failure
            adjustment = -0.15
        
        # Update score with momentum (don't change too quickly)
        new_score = current_score + (adjustment * 0.3)
        self.api_performance_scores[api_name] = max(0.0, min(1.0, new_score))
        
        # Save updated scores
        self.save_performance_history()
    
    def log_api_failure(self, api_name: str, error: Exception):
        """Log API failure details"""
        
        failure_log = {
            'timestamp': datetime.now().isoformat(),
            'api_name': api_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'usage_count': self.usage_counts.get(api_name, 0)
        }
        
        # Log to file
        os.makedirs('data/performance_logs', exist_ok=True)
        with open('data/performance_logs/api_failures.jsonl', 'a') as f:
            f.write(json.dumps(failure_log) + '\n')
    
    def get_fallback_response(self, query: str) -> str:
        """Generate fallback response when all APIs fail"""
        
        fallback_responses = [
            "I apologize, but I'm experiencing technical difficulties connecting to my AI services. Please try again in a moment.",
            "I'm currently having trouble accessing my advanced AI capabilities. Let me provide a basic response based on your query.",
            "Due to temporary service issues, I'm operating in limited mode. I can still help with basic questions."
        ]
        
        return random.choice(fallback_responses)
    
    async def process_query_with_orchestration(self, query: str, user_priority: str = 'balanced',
                                             context: Dict = None) -> Dict[str, Any]:
        """Main method to process query with intelligent orchestration"""
        
        # Select best API
        best_api = await self.smart_api_selection(query, user_priority, context)
        
        # Create priority list (best API first, then fallbacks)
        other_apis = [api['name'] for api in self.api_pool if api['name'] != best_api]
        api_priority_list = [best_api] + other_apis
        
        # Try API cascade
        result = await self.try_api_cascade(query, api_priority_list)
        
        # Add orchestration metadata
        result['orchestration_metadata'] = {
            'selected_api': best_api,
            'priority_list': api_priority_list,
            'query_complexity': self.calculate_query_complexity(query),
            'user_priority': user_priority,
            'performance_scores': self.api_performance_scores.copy(),
            'usage_counts': self.usage_counts.copy()
        }
        
        return result

# Usage Example
async def example_orchestration():
    """Example of using intelligent API orchestration"""
    
    orchestrator = IntelligentAPIOrchestrator()
    
    # Example queries with different priorities
    queries = [
        ("Write a Python function to calculate fibonacci numbers", "quality"),
        ("Hello, how are you today?", "speed"),
        ("Explain machine learning in detail", "balanced")
    ]
    
    for query, priority in queries:
        print(f"\n🔍 Processing: {query}")
        print(f"🎯 Priority: {priority}")
        
        result = await orchestrator.process_query_with_orchestration(query, priority)
        
        print(f"✅ Result: {result['success']}")
        print(f"🤖 API Used: {result['api_used']}")
        print(f"⏱️ Response Time: {result['response_time']:.2f}s")
        print(f"📝 Response: {result['response'][:100]}...")

if __name__ == "__main__":
    asyncio.run(example_orchestration())
