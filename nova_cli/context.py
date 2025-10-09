import time
import random
import hashlib
import re
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple


class AdvancedContextualMemorySystem:
    """Smart contextual memory that remembers conversation patterns and user preferences"""
    
    def __init__(self, base_memory_system):
        self.base_memory = base_memory_system
        self.conversation_patterns = defaultdict(list)
        self.user_context_graph = defaultdict(dict)
        self.smart_retrieval_cache = {}
        self.context_importance_scores = defaultdict(float)
        
        # Enhanced conversation threading
        self.conversation_threads = defaultdict(list)
        self.thread_summaries = {}
        self.cross_session_memory = defaultdict(list)
        
        print("✅ Advanced Contextual Memory initialized")
    
    async def remember_with_context(self, user_id: str, session_id: str, 
                                  user_input: str, ai_response: str, 
                                  context_metadata: Dict[str, Any]):
        """Remember conversation with rich contextual metadata"""
        
        # Extract conversation patterns
        patterns = self._extract_conversation_patterns(user_input, ai_response)
        self.conversation_patterns[user_id].extend(patterns)
        
        # Build context graph
        self._build_context_graph(user_id, user_input, ai_response, context_metadata)
        
        # Thread management
        thread_id = self._get_or_create_thread(user_id, user_input, context_metadata)
        self.conversation_threads[thread_id].append({
            'user_input': user_input,
            'ai_response': ai_response,
            'timestamp': datetime.now(),
            'metadata': context_metadata
        })
        
        # Store in base memory system
        await self.base_memory.remember_conversation(
            user_id, session_id, user_input, ai_response,
            context_metadata.get('agent_type', 'general'),
            context_metadata.get('language', 'english'),
            context_metadata.get('emotion', 'neutral'),
            context_metadata.get('confidence', 0.8),
            context_metadata.get('intent'),
            context_metadata.get('response_time', 0.0),
            context_metadata.get('voice_used', False),
            context_metadata.get('location'),
            context_metadata.get('weather_context'),
            context_metadata.get('search_queries'),
            context_metadata.get('file_analyzed')
        )
    
    def get_smart_context(self, user_id: str, current_query: str, limit: int = 10) -> str:
        """Get intelligent context based on query similarity and importance"""
        
        # Quick cache check
        cache_key = f"{user_id}:{hashlib.md5(current_query.encode()).hexdigest()[:8]}"
        if cache_key in self.smart_retrieval_cache:
            cached_result = self.smart_retrieval_cache[cache_key]
            if time.time() - cached_result['timestamp'] < 300:  # 5 minutes
                return cached_result['context']
        
        # Get conversation patterns for user
        user_patterns = self.conversation_patterns.get(user_id, [])
        user_context = self.user_context_graph.get(user_id, {})
        
        # Find relevant conversation threads
        relevant_threads = self._find_relevant_threads(user_id, current_query)
        
        # Score and rank context pieces
        context_pieces = []
        
        # Add thread contexts
        for thread_id, relevance_score in relevant_threads[:3]:
            thread_conversations = self.conversation_threads[thread_id][-3:]  # Last 3 from thread
            for conv in thread_conversations:
                context_pieces.append({
                    'content': f"User: {conv['user_input']}\nNOVA: {conv['ai_response'][:200]}...",
                    'score': relevance_score + 0.5,  # Thread bonus
                    'timestamp': conv['timestamp'],
                    'type': 'thread_context'
                })
        
        # Add recent important interactions
        recent_important = self._get_recent_important_context(user_id, current_query)
        context_pieces.extend(recent_important)
        
        # Sort by relevance score and recency
        context_pieces.sort(key=lambda x: (x['score'], x['timestamp']), reverse=True)
        
        # Build final context
        final_context = "RELEVANT CONVERSATION CONTEXT:\n"
        for piece in context_pieces[:limit]:
            final_context += f"\n--- {piece['type'].upper()} (Score: {piece['score']:.2f}) ---\n"
            final_context += piece['content'] + "\n"
        
        # Add user preferences summary
        if user_context:
            final_context += f"\nUSER PREFERENCES:\n{self._format_user_context(user_context)}\n"
        
        # Cache the result
        self.smart_retrieval_cache[cache_key] = {
            'context': final_context,
            'timestamp': time.time()
        }
        
        return final_context
    
    def _extract_conversation_patterns(self, user_input: str, ai_response: str) -> List[Dict]:
        """Extract patterns from conversation for future reference"""
        patterns = []
        
        # Query type patterns
        if any(keyword in user_input.lower() for keyword in ['how to', 'explain', 'what is']):
            patterns.append({'type': 'learning_query', 'content': user_input[:100]})
        
        # Problem-solving patterns
        if any(keyword in user_input.lower() for keyword in ['help', 'issue', 'problem', 'error']):
            patterns.append({'type': 'problem_solving', 'content': user_input[:100]})
        
        # Preference indicators
        preference_indicators = ['I like', 'I prefer', 'I usually', 'I always', 'I never']
        for indicator in preference_indicators:
            if indicator.lower() in user_input.lower():
                patterns.append({'type': 'preference', 'content': user_input})
        
        # Project context
        project_keywords = ['my project', 'working on', 'building', 'developing', 'creating']
        for keyword in project_keywords:
            if keyword in user_input.lower():
                patterns.append({'type': 'project_context', 'content': user_input})
        
        return patterns
    
    def _build_context_graph(self, user_id: str, user_input: str, ai_response: str, metadata: Dict):
        """Build knowledge graph of user context"""
        context = self.user_context_graph[user_id]
        
        # Extract and store user information
        user_lower = user_input.lower()
        
        # Technology stack
        tech_keywords = ['python', 'javascript', 'react', 'node', 'django', 'flask', 'aws', 'docker']
        for tech in tech_keywords:
            if tech in user_lower:
                if 'tech_stack' not in context:
                    context['tech_stack'] = []
                if tech not in context['tech_stack']:
                    context['tech_stack'].append(tech)
        
        # Goals and objectives
        if any(phrase in user_lower for phrase in ['my goal', 'want to', 'trying to', 'planning to']):
            if 'goals' not in context:
                context['goals'] = []
            context['goals'].append({
                'goal': user_input,
                'mentioned_at': datetime.now(),
                'context': metadata.get('agent_type', 'general')
            })
        
        # Experience level indicators
        experience_indicators = {
            'beginner': ['new to', 'just started', 'beginner', 'learning'],
            'intermediate': ['some experience', 'familiar with', 'working with'],
            'advanced': ['expert in', 'specialized', 'years of experience', 'professional']
        }
        
        for level, keywords in experience_indicators.items():
            if any(keyword in user_lower for keyword in keywords):
                context['experience_level'] = level
                break
        
        # Role and industry
        role_keywords = ['developer', 'engineer', 'manager', 'student', 'founder', 'consultant']
        for role in role_keywords:
            if role in user_lower:
                context['role'] = role
                break
    
    def _get_or_create_thread(self, user_id: str, user_input: str, metadata: Dict) -> str:
        """Get existing thread or create new one based on topic similarity"""
        
        # Simple topic extraction
        current_topics = set()
        topic_keywords = ['coding', 'business', 'career', 'health', 'project', 'problem']
        
        for keyword in topic_keywords:
            if keyword in user_input.lower():
                current_topics.add(keyword)
        
        # Add agent type as topic
        agent_type = metadata.get('agent_type', 'general')
        current_topics.add(agent_type)
        
        # Find existing similar thread
        for thread_id, conversations in self.conversation_threads.items():
            if thread_id.startswith(user_id) and conversations:
                last_conv = conversations[-1]
                last_agent = last_conv['metadata'].get('agent_type', 'general')
                
                # Same agent type and recent (within 30 minutes)
                time_diff = (datetime.now() - last_conv['timestamp']).seconds
                if last_agent in current_topics and time_diff < 1800:
                    return thread_id
        
        # Create new thread
        thread_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(100, 999)}"
        return thread_id
    
    def _find_relevant_threads(self, user_id: str, current_query: str) -> List[Tuple[str, float]]:
        """Find conversation threads relevant to current query"""
        relevant_threads = []
        
        # Extract keywords from current query
        current_keywords = set(current_query.lower().split())
        
        for thread_id, conversations in self.conversation_threads.items():
            if not thread_id.startswith(user_id):
                continue
                
            # Calculate relevance score
            thread_score = 0.0
            thread_keywords = set()
            
            for conv in conversations:
                conv_keywords = set(conv['user_input'].lower().split())
                thread_keywords.update(conv_keywords)
            
            # Keyword overlap score
            keyword_overlap = len(current_keywords.intersection(thread_keywords))
            thread_score += keyword_overlap * 0.2
            
            # Recency bonus
            if conversations:
                last_conv_time = conversations[-1]['timestamp']
                hours_ago = (datetime.now() - last_conv_time).seconds / 3600
                if hours_ago < 24:  # Within last 24 hours
                    thread_score += max(0, 1.0 - hours_ago / 24)
            
            # Thread length bonus (more interactions = more context)
            thread_score += min(len(conversations) * 0.1, 0.5)
            
            if thread_score > 0.3:  # Minimum relevance threshold
                relevant_threads.append((thread_id, thread_score))
        
        # Sort by relevance
        relevant_threads.sort(key=lambda x: x[1], reverse=True)
        return relevant_threads
    
    def _get_recent_important_context(self, user_id: str, current_query: str) -> List[Dict]:
        """Get recent important interactions"""
        important_context = []
        
        # Get recent conversations from base memory
        try:
            recent_context = self.base_memory.get_relevant_context(current_query, user_id, limit=5)
            if recent_context:
                # Parse and score recent context
                context_parts = recent_context.split('---')
                for i, part in enumerate(context_parts[-3:]):  # Last 3 interactions
                    if 'User' in part and 'NOVA' in part:
                        score = 0.7 - (i * 0.1)  # More recent = higher score
                        important_context.append({
                            'content': part.strip(),
                            'score': score,
                            'timestamp': datetime.now() - timedelta(hours=i),
                            'type': 'recent_interaction'
                        })
        except Exception as e:
            print(f"Error getting recent context: {e}")
        
        return important_context
    
    def _format_user_context(self, user_context: Dict) -> str:
        """Format user context for display"""
        formatted = []
        
        if 'tech_stack' in user_context:
            formatted.append(f"Technologies: {', '.join(user_context['tech_stack'])}")
        
        if 'role' in user_context:
            formatted.append(f"Role: {user_context['role']}")
        
        if 'experience_level' in user_context:
            formatted.append(f"Experience: {user_context['experience_level']}")
        
        if 'goals' in user_context and user_context['goals']:
            recent_goals = user_context['goals'][-2:]  # Last 2 goals
            goal_text = '; '.join([g['goal'][:100] for g in recent_goals])
            formatted.append(f"Recent Goals: {goal_text}")
        
        return ' | '.join(formatted)
    
    def get_conversation_summary(self, user_id: str, thread_id: str = None) -> str:
        """Get summary of conversation thread or all conversations"""
        if thread_id and thread_id in self.conversation_threads:
            conversations = self.conversation_threads[thread_id]
            summary = f"Thread Summary ({len(conversations)} interactions):\n"
            
            # Get key topics
            all_text = ' '.join([c['user_input'] + ' ' + c['ai_response'] for c in conversations])
            key_topics = self._extract_key_topics(all_text)
            summary += f"Key Topics: {', '.join(key_topics)}\n"
            
            # Get timeline
            if conversations:
                start_time = conversations[0]['timestamp']
                end_time = conversations[-1]['timestamp']
                duration = end_time - start_time
                summary += f"Duration: {duration}\n"
            
            return summary
        
        # Overall user summary
        user_patterns = self.conversation_patterns.get(user_id, [])
        user_context = self.user_context_graph.get(user_id, {})
        
        summary = f"User Summary:\n"
        summary += f"Total Patterns: {len(user_patterns)}\n"
        summary += f"Context Graph: {len(user_context)} attributes\n"
        summary += f"Active Threads: {len([t for t in self.conversation_threads.keys() if t.startswith(user_id)])}\n"
        
        return summary
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from text"""
        # Simple keyword frequency analysis
        words = re.findall(r'\b\w{4,}\b', text.lower())  # Words with 4+ characters
        
        # Filter out common words
        stop_words = {'that', 'this', 'with', 'from', 'they', 'been', 'have', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'would', 'there', 'could', 'other', 'after', 'first', 'well', 'many', 'some', 'like', 'into', 'them', 'make', 'than', 'then', 'what', 'your'}
        
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Count frequency
        word_counts = Counter(filtered_words)
        
        # Return top topics
        return [word for word, count in word_counts.most_common(5) if count > 1]