# src/memory/sharp_memory.py

import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
import os

class SharpMemorySystem:
    """ChatGPT-level Sharp Memory with Context Awareness"""
    
    def __init__(self, db_path: str = "data/memory/nova_memory.db"):
        # Multi-layered memory system
        self.db_path = db_path
        self.short_term_memory = deque(maxlen=100)  # Recent conversations
        self.working_memory = {}  # Current session context
        self.conversation_threads = {}  # Thread tracking
        self.user_profiles = {}  # User preferences & patterns
        
        # Vector-based semantic memory
        self.setup_vector_memory()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize database
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database for persistent memory"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    agent_used TEXT,
                    context_summary TEXT,
                    satisfaction_score REAL,
                    conversation_thread_id TEXT
                )
            ''')
            
            # User profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT,
                    conversation_patterns TEXT,
                    expertise_level TEXT,
                    topics_of_interest TEXT,
                    last_interaction TEXT,
                    total_conversations INTEGER DEFAULT 0
                )
            ''')
            
            # Conversation threads table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_threads (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    topic TEXT,
                    start_time TEXT,
                    last_activity TEXT,
                    message_count INTEGER DEFAULT 0,
                    thread_summary TEXT
                )
            ''')
            
            conn.commit()
        print("✅ Sharp Memory database initialized")
    
    def setup_vector_memory(self):
        """Setup vector-based semantic memory"""
        os.makedirs("data/memory/vector_db", exist_ok=True)
        self.vector_client = chromadb.PersistentClient(path="data/memory/vector_db")
        
        try:
            self.semantic_collection = self.vector_client.get_collection("conversations")
        except:
            self.semantic_collection = self.vector_client.create_collection(
                name="conversations",
                metadata={"description": "Semantic conversation memory"}
            )
        print("✅ Vector memory collection created")
    
    async def remember_conversation_advanced(self, user_input: str, ai_response: str,
                                           metadata: Dict[str, Any], user_id: str = "default",
                                           session_id: str = "default"):
        """Advanced conversation memory with context understanding"""
        
        # Generate conversation essence and context
        conversation_essence = self.extract_conversation_essence(user_input, ai_response)
        context_summary = self.generate_context_summary(user_input, ai_response, metadata)
        
        # Detect conversation thread
        thread_id = self.detect_or_create_thread(user_input, user_id, context_summary)
        
        # Create memory entry
        memory_entry = {
            'user_id': user_id,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'ai_response': ai_response,
            'agent_used': metadata.get('agent_used', 'general'),
            'context_summary': context_summary,
            'satisfaction_score': metadata.get('satisfaction_estimate', 3.0),
            'conversation_thread_id': thread_id,
            'essence': conversation_essence
        }
        
        # Store in multiple memory layers
        # 1. Short-term memory (immediate access)
        self.short_term_memory.append(memory_entry)
        
        # 2. Working memory (session context)
        if session_id not in self.working_memory:
            self.working_memory[session_id] = []
        self.working_memory[session_id].append(memory_entry)
        
        # 3. Long-term database storage
        self.store_in_database(memory_entry)
        
        # 4. Semantic vector memory
        self.store_in_vector_memory(memory_entry)
        
        # 5. Update user profile
        await self.update_user_profile(user_id, user_input, ai_response, metadata)
        
        # 6. Update conversation thread
        self.update_conversation_thread(thread_id, memory_entry)
        
        print(f"✅ Conversation remembered with thread: {thread_id}")
    
    def extract_conversation_essence(self, user_input: str, ai_response: str) -> Dict[str, Any]:
        """Extract the essential meaning from conversation"""
        # Analyze user intent
        user_intent = self.analyze_user_intent(user_input)
        
        # Extract key topics
        topics = self.extract_topics(user_input + " " + ai_response)
        
        # Determine conversation type
        conversation_type = self.classify_conversation_type(user_input)
        
        # Extract actionable items
        action_items = self.extract_action_items(ai_response)
        
        return {
            'user_intent': user_intent,
            'topics': topics,
            'conversation_type': conversation_type,
            'action_items': action_items,
            'complexity_level': self.assess_query_complexity(user_input),
            'emotional_tone': self.detect_emotional_tone(user_input)
        }
    
    def analyze_user_intent(self, user_input: str) -> str:
        """Analyze what the user really wants"""
        intent_patterns = {
            'seeking_help': ['help', 'how to', 'can you', 'please', 'need'],
            'asking_question': ['what', 'why', 'when', 'where', 'which', '?'],
            'requesting_action': ['create', 'build', 'make', 'generate', 'write'],
            'seeking_advice': ['should i', 'recommend', 'suggest', 'advice', 'opinion'],
            'clarification': ['explain', 'clarify', 'understand', 'mean', 'elaborate'],
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good evening'],
            'problem_solving': ['error', 'bug', 'issue', 'problem', 'not working', 'fix']
        }
        
        user_lower = user_input.lower()
        intent_scores = {}
        
        for intent, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in user_lower)
            if score > 0:
                intent_scores[intent] = score
        
        return max(intent_scores, key=intent_scores.get) if intent_scores else 'general_inquiry'
    
    def classify_conversation_type(self, user_input: str) -> str:
        """Classify the type of conversation based on content"""
        text_lower = user_input.lower()
        
        conversation_types = {
            'technical': ['code', 'programming', 'bug', 'error', 'debug', 'algorithm', 'function', 'syntax', 'database', 'api'],
            'career': ['job', 'career', 'resume', 'interview', 'salary', 'promotion', 'work', 'professional', 'skills'],
            'business': ['business', 'strategy', 'marketing', 'sales', 'revenue', 'profit', 'customer', 'market'],
            'medical': ['health', 'doctor', 'symptoms', 'treatment', 'medicine', 'pain', 'medical', 'hospital'],
            'educational': ['learn', 'study', 'course', 'tutorial', 'education', 'teaching', 'knowledge', 'explain'],
            'emotional': ['feeling', 'emotion', 'stress', 'anxiety', 'sad', 'happy', 'worried', 'depressed'],
            'creative': ['design', 'art', 'creative', 'music', 'writing', 'drawing', 'photography', 'story'],
            'finance': ['money', 'investment', 'budget', 'savings', 'loan', 'bank', 'financial', 'income']
        }
        
        type_scores = {}
        for conv_type, keywords in conversation_types.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                type_scores[conv_type] = score
        
        return max(type_scores, key=type_scores.get) if type_scores else 'general'
    
    def assess_query_complexity(self, user_input: str) -> str:
        """Assess the complexity level of user query"""
        text_length = len(user_input.split())
        
        # Simple complexity indicators
        complex_indicators = ['multiple', 'several', 'various', 'complex', 'advanced', 'detailed']
        simple_indicators = ['simple', 'basic', 'easy', 'quick', 'just']
        
        text_lower = user_input.lower()
        
        # Check for complexity indicators
        if any(indicator in text_lower for indicator in complex_indicators):
            return 'high'
        elif any(indicator in text_lower for indicator in simple_indicators):
            return 'low'
        elif text_length > 20:
            return 'medium'
        elif text_length > 10:
            return 'medium'
        else:
            return 'low'
    
    def detect_emotional_tone(self, user_input: str) -> str:
        """Detect emotional tone in user input"""
        text_lower = user_input.lower()
        
        emotion_patterns = {
            'positive': ['happy', 'excited', 'great', 'awesome', 'love', 'amazing', 'wonderful', 'excellent'],
            'negative': ['sad', 'frustrated', 'angry', 'disappointed', 'hate', 'terrible', 'awful', 'bad'],
            'anxious': ['worried', 'nervous', 'anxious', 'stressed', 'concerned', 'afraid', 'scared'],
            'urgent': ['urgent', 'asap', 'quickly', 'immediately', 'emergency', 'help!', 'now']
        }
        
        emotion_scores = {}
        for emotion, keywords in emotion_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        return max(emotion_scores, key=emotion_scores.get) if emotion_scores else 'neutral'
    
    def extract_action_items(self, ai_response: str) -> List[str]:
        """Extract actionable items from AI response"""
        action_words = ['create', 'build', 'make', 'setup', 'install', 'configure', 'implement', 'design', 'write', 'develop']
        
        sentences = ai_response.split('.')
        action_items = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(action in sentence_lower for action in action_words):
                if len(sentence.strip()) > 10:  # Avoid very short sentences
                    action_items.append(sentence.strip())
        
        return action_items[:3]  # Return top 3 action items
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract main topics from conversation"""
        topic_keywords = {
            'programming': ['code', 'programming', 'python', 'javascript', 'function', 'algorithm'],
            'career': ['job', 'career', 'resume', 'interview', 'salary', 'promotion'],
            'health': ['health', 'medical', 'doctor', 'symptoms', 'treatment'],
            'business': ['business', 'strategy', 'marketing', 'sales', 'revenue'],
            'education': ['learn', 'study', 'course', 'tutorial', 'education'],
            'technology': ['ai', 'machine learning', 'data science', 'cloud', 'api'],
            'personal': ['feeling', 'emotion', 'stress', 'anxiety', 'relationship']
        }
        
        text_lower = text.lower()
        found_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics[:3]  # Return top 3 topics
    
    def generate_context_summary(self, user_input: str, ai_response: str, metadata: Dict[str, Any]) -> str:
        """Generate context summary for the conversation"""
        intent = self.analyze_user_intent(user_input)
        topics = self.extract_topics(user_input + " " + ai_response)
        conv_type = self.classify_conversation_type(user_input)
        
        summary_parts = [
            f"Intent: {intent}",
            f"Type: {conv_type}",
            f"Topics: {', '.join(topics) if topics else 'general'}"
        ]
        
        return "; ".join(summary_parts)
    
    def detect_or_create_thread(self, user_input: str, user_id: str, context_summary: str) -> str:
        """Detect existing conversation thread or create new one"""
        # Check if this continues a recent conversation
        recent_conversations = self.get_recent_conversations(user_id, limit=5)
        
        if recent_conversations:
            # Check for thread continuation indicators
            continuation_indicators = [
                'also', 'and', 'furthermore', 'additionally', 'but', 'however',
                'that', 'this', 'it', 'the same', 'similar'
            ]
            
            user_lower = user_input.lower()
            
            # Check for pronoun references or continuation words in first 30 characters
            if any(indicator in user_lower[:30] for indicator in continuation_indicators):
                # Likely continuation of previous thread
                last_conversation = recent_conversations[0]
                return last_conversation.get('conversation_thread_id', self.generate_thread_id())
            
            # Check topic similarity with recent conversations
            for conv in recent_conversations:
                if self.calculate_topic_similarity(context_summary, conv.get('context_summary', '')) > 0.6:
                    return conv.get('conversation_thread_id', self.generate_thread_id())
        
        # Create new thread
        return self.generate_thread_id()
    
    def generate_thread_id(self) -> str:
        """Generate unique thread ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"thread_{timestamp}_{np.random.randint(1000, 9999)}"
    
    def calculate_topic_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text topics"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def recall_sharp_context(self, user_input: str, user_id: str = "default",
                                 depth: str = "deep") -> Dict[str, Any]:
        """ChatGPT-level sharp context recall"""
        print(f"🧠 Recalling context for: {user_input[:50]}...")
        
        context_layers = {}
        
        # 1. Immediate context (short-term memory)
        immediate_context = self.get_immediate_context(user_id)
        context_layers['immediate'] = immediate_context
        
        # 2. Semantic similarity context (vector search)
        semantic_context = self.search_semantic_memory(user_input, user_id)
        context_layers['semantic'] = semantic_context
        
        # 3. Thread continuity context
        thread_context = self.get_thread_continuity_context(user_input, user_id)
        context_layers['thread'] = thread_context
        
        # 4. User profile context
        profile_context = self.get_user_profile_context(user_id)
        context_layers['profile'] = profile_context
        
        # 5. Working memory context (current session)
        session_context = self.get_session_context(user_id)
        context_layers['session'] = session_context
        
        # Synthesize unified context
        unified_context = self.synthesize_context_layers(context_layers, user_input)
        
        print(f"✅ Context recalled: {len(unified_context.get('summary', ''))} chars")
        return unified_context
    
    def get_immediate_context(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Get immediate conversation context"""
        recent_memories = []
        
        # Get from short-term memory
        for memory in reversed(list(self.short_term_memory)):
            if memory.get('user_id') == user_id:
                recent_memories.append({
                    'user_input': memory['user_input'],
                    'ai_response': memory['ai_response'],
                    'timestamp': memory['timestamp'],
                    'agent_used': memory.get('agent_used', 'general')
                })
                if len(recent_memories) >= limit:
                    break
        
        return recent_memories
    
    def search_semantic_memory(self, query: str, user_id: str, top_k: int = 3) -> List[Dict]:
        """Search semantically similar conversations"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Search in vector collection
            results = self.semantic_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where={"user_id": user_id} if user_id != "default" else None
            )
            
            semantic_matches = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    semantic_matches.append({
                        'content': doc,
                        'relevance_score': 1 - distance,  # Convert distance to similarity
                        'metadata': metadata,
                        'rank': i + 1
                    })
            
            return semantic_matches
        
        except Exception as e:
            print(f"⚠️ Semantic search error: {e}")
            return []
    
    def get_thread_continuity_context(self, user_input: str, user_id: str) -> Dict[str, Any]:
        """Get conversation thread continuity context"""
        # Detect if this might continue a thread
        thread_indicators = ['also', 'and', 'furthermore', 'that', 'this', 'it']
        
        if any(indicator in user_input.lower()[:20] for indicator in thread_indicators):
            # Look for recent thread
            recent_conversations = self.get_recent_conversations(user_id, limit=3)
            
            if recent_conversations:
                latest_thread = recent_conversations[0].get('conversation_thread_id')
                # Get thread history
                thread_history = self.get_thread_history(latest_thread)
                
                return {
                    'thread_continuation': True,
                    'thread_id': latest_thread,
                    'thread_history': thread_history[-3:],  # Last 3 messages
                    'thread_topic': thread_history[0].get('context_summary', '') if thread_history else ''
                }
        
        return {'thread_continuation': False}
    
    def get_user_profile_context(self, user_id: str) -> Dict[str, Any]:
        """Get user profile context"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT preferences, conversation_patterns, expertise_level, 
                           topics_of_interest, total_conversations
                    FROM user_profiles 
                    WHERE user_id = ?
                ''', (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'preferences': json.loads(row[0]) if row[0] else {},
                        'patterns': json.loads(row[1]) if row[1] else {},
                        'expertise_level': row[2],
                        'interests': json.loads(row[3]) if row[3] else [],
                        'total_conversations': row[4]
                    }
        except Exception as e:
            print(f"⚠️ Profile retrieval error: {e}")
        
        return {}
    
    def get_session_context(self, user_id: str) -> List[Dict]:
        """Get current session context"""
        session_memories = []
        
        for session_id, memories in self.working_memory.items():
            for memory in memories:
                if memory.get('user_id') == user_id:
                    session_memories.append(memory)
        
        return session_memories[-5:]  # Last 5 in session
    
    def get_thread_history(self, thread_id: str) -> List[Dict]:
        """Get conversation thread history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_input, ai_response, timestamp, context_summary
                    FROM conversations 
                    WHERE conversation_thread_id = ?
                    ORDER BY timestamp ASC
                ''', (thread_id,))
                
                rows = cursor.fetchall()
                return [{
                    'user_input': row[0],
                    'ai_response': row[1],
                    'timestamp': row[2],
                    'context_summary': row[3]
                } for row in rows]
        except Exception as e:
            print(f"⚠️ Thread history error: {e}")
            return []
    
    def synthesize_context_layers(self, context_layers: Dict, user_input: str) -> Dict[str, Any]:
        """Synthesize all context layers into unified understanding"""
        synthesis = {
            'has_context': False,
            'summary': '',
            'conversation_flow': [],
            'user_patterns': {},
            'relevant_topics': [],
            'context_strength': 0.0
        }
        
        # Process immediate context
        immediate = context_layers.get('immediate', [])
        if immediate:
            synthesis['has_context'] = True
            synthesis['conversation_flow'] = immediate[-3:]  # Last 3 exchanges
            synthesis['context_strength'] += 0.3
        
        # Process semantic context
        semantic = context_layers.get('semantic', [])
        if semantic:
            synthesis['has_context'] = True
            high_relevance = [s for s in semantic if s['relevance_score'] > 0.7]
            if high_relevance:
                synthesis['context_strength'] += 0.3
        
        # Process thread context
        thread = context_layers.get('thread', {})
        if thread.get('thread_continuation'):
            synthesis['has_context'] = True
            synthesis['context_strength'] += 0.2
        
        # Process user profile
        profile = context_layers.get('profile', {})
        if profile:
            synthesis['user_patterns'] = profile
            synthesis['context_strength'] += 0.2
        
        # Generate context summary
        synthesis['summary'] = self.generate_context_summary_text(context_layers, user_input)
        
        return synthesis
    
    def generate_context_summary_text(self, context_layers: Dict, user_input: str) -> str:
        """Generate human-readable context summary"""
        summary_parts = []
        
        # Recent conversation summary
        immediate = context_layers.get('immediate', [])
        if immediate:
            recent_topic = self.extract_main_topic(immediate[0]['user_input'])
            summary_parts.append(f"Recently discussed: {recent_topic}")
        
        # Thread continuation
        thread = context_layers.get('thread', {})
        if thread.get('thread_continuation'):
            summary_parts.append(f"Continuing conversation thread")
        
        # User expertise level
        profile = context_layers.get('profile', {})
        if profile.get('expertise_level'):
            summary_parts.append(f"User expertise: {profile['expertise_level']}")
        
        # Semantic context
        semantic = context_layers.get('semantic', [])
        if semantic and semantic[0]['relevance_score'] > 0.8:
            summary_parts.append(f"Highly relevant: similar question asked before")
        
        return "; ".join(summary_parts) if summary_parts else "No previous context available"
    
    def extract_main_topic(self, text: str) -> str:
        """Extract main topic from text"""
        topics = self.extract_topics(text)
        return topics[0] if topics else "general discussion"
    
    def store_in_database(self, memory_entry: Dict):
        """Store conversation in SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations
                    (user_id, session_id, timestamp, user_input, ai_response,
                     agent_used, context_summary, satisfaction_score, conversation_thread_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    memory_entry['user_id'],
                    memory_entry['session_id'],
                    memory_entry['timestamp'],
                    memory_entry['user_input'],
                    memory_entry['ai_response'],
                    memory_entry['agent_used'],
                    memory_entry['context_summary'],
                    memory_entry['satisfaction_score'],
                    memory_entry['conversation_thread_id']
                ))
                conn.commit()
        except Exception as e:
            print(f"⚠️ Database storage error: {e}")
    
    def store_in_vector_memory(self, memory_entry: Dict):
        """Store conversation in vector memory for semantic search"""
        try:
            # Create searchable content
            searchable_content = f"{memory_entry['user_input']} {memory_entry['ai_response']}"
            
            # Generate embedding
            embedding = self.embedding_model.encode(searchable_content)
            
            # Create unique ID
            vector_id = f"{memory_entry['user_id']}_{memory_entry['timestamp']}_{hash(searchable_content) % 10000}"
            
            # Store in vector collection
            self.semantic_collection.add(
                documents=[searchable_content],
                embeddings=[embedding.tolist()],
                metadatas=[{
                    'user_id': memory_entry['user_id'],
                    'timestamp': memory_entry['timestamp'],
                    'agent_used': memory_entry['agent_used'],
                    'satisfaction_score': memory_entry['satisfaction_score'],
                    'thread_id': memory_entry['conversation_thread_id']
                }],
                ids=[vector_id]
            )
        except Exception as e:
            print(f"⚠️ Vector storage error: {e}")
    
    async def update_user_profile(self, user_id: str, user_input: str, ai_response: str, metadata: Dict):
        """Update user profile based on conversation"""
        try:
            # Extract patterns from current conversation
            intent = self.analyze_user_intent(user_input)
            topics = self.extract_topics(user_input + " " + ai_response)
            complexity = self.assess_query_complexity(user_input)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if profile exists
                cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing profile
                    cursor.execute('''
                        UPDATE user_profiles 
                        SET last_interaction = ?, total_conversations = total_conversations + 1
                        WHERE user_id = ?
                    ''', (datetime.now().isoformat(), user_id))
                else:
                    # Create new profile
                    cursor.execute('''
                        INSERT INTO user_profiles 
                        (user_id, preferences, conversation_patterns, expertise_level, 
                         topics_of_interest, last_interaction, total_conversations)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        user_id,
                        json.dumps({}),
                        json.dumps({'common_intents': [intent]}),
                        complexity,
                        json.dumps(topics),
                        datetime.now().isoformat(),
                        1
                    ))
                
                conn.commit()
        except Exception as e:
            print(f"⚠️ Profile update error: {e}")
    
    def update_conversation_thread(self, thread_id: str, memory_entry: Dict):
        """Update conversation thread information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if thread exists
                cursor.execute('SELECT * FROM conversation_threads WHERE thread_id = ?', (thread_id,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing thread
                    cursor.execute('''
                        UPDATE conversation_threads 
                        SET last_activity = ?, message_count = message_count + 1
                        WHERE thread_id = ?
                    ''', (datetime.now().isoformat(), thread_id))
                else:
                    # Create new thread
                    main_topic = self.extract_main_topic(memory_entry['user_input'])
                    cursor.execute('''
                        INSERT INTO conversation_threads 
                        (thread_id, user_id, topic, start_time, last_activity, message_count, thread_summary)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        thread_id,
                        memory_entry['user_id'],
                        main_topic,
                        memory_entry['timestamp'],
                        memory_entry['timestamp'],
                        1,
                        memory_entry['context_summary']
                    ))
                
                conn.commit()
        except Exception as e:
            print(f"⚠️ Thread update error: {e}")
    
    def get_recent_conversations(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversations from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_input, ai_response, agent_used, timestamp,
                           context_summary, conversation_thread_id
                    FROM conversations 
                    WHERE user_id = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (user_id, limit))
                
                rows = cursor.fetchall()
                return [{
                    'user_input': row[0],
                    'ai_response': row[1],
                    'agent_used': row[2],
                    'timestamp': row[3],
                    'context_summary': row[4],
                    'conversation_thread_id': row[5]
                } for row in rows]
        except Exception as e:
            print(f"⚠️ Database retrieval error: {e}")
            return []
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total conversations
                cursor.execute('SELECT COUNT(*) FROM conversations')
                row = cursor.fetchone()
                total_conversations = row[0] if row else 0
                
                # Unique users
                cursor.execute('SELECT COUNT(DISTINCT user_id) FROM conversations')
                row = cursor.fetchone()
                unique_users = row[0] if row else 0
                
                # Recent activity
                week_ago = (datetime.now() - timedelta(days=7)).isoformat()
                cursor.execute('SELECT COUNT(*) FROM conversations WHERE timestamp > ?', (week_ago,))
                row = cursor.fetchone()
                recent_conversations = row[0] if row else 0
                
                return {
                    'total_conversations': total_conversations,
                    'unique_users': unique_users,
                    'recent_conversations_7d': recent_conversations,
                    'short_term_memory_size': len(self.short_term_memory),
                    'working_memory_sessions': len(self.working_memory),
                    'vector_memory_size': self.semantic_collection.count()
                }
        except Exception as e:
            print(f"⚠️ Stats retrieval error: {e}")
            return {}

# Quick test function
async def test_sharp_memory():
    """Test the sharp memory system"""
    print("🧪 Testing Sharp Memory System...")
    
    memory = SharpMemorySystem()
    
    # Test conversation 1
    await memory.remember_conversation_advanced(
        user_input="How do I write a Python function?",
        ai_response="To write a Python function, use 'def' keyword followed by function name...",
        metadata={'agent_used': 'coding', 'satisfaction_estimate': 4.5},
        user_id="test_user",
        session_id="session_1"
    )
    
    # Test conversation 2 (continuation)
    await memory.remember_conversation_advanced(
        user_input="Can you also show me an example?",
        ai_response="Sure! Here's an example: def greet(name): return f'Hello {name}'",
        metadata={'agent_used': 'coding', 'satisfaction_estimate': 4.8},
        user_id="test_user",
        session_id="session_1"
    )
    
    # Test context recall
    context = await memory.recall_sharp_context("What about error handling in functions?", "test_user")
    print(f"✅ Context recalled: {context['summary']}")
    print(f"✅ Has context: {context['has_context']}")
    print(f"✅ Context strength: {context['context_strength']:.2f}")
    
    # Test stats
    stats = await memory.get_memory_stats()
    print(f"✅ Memory stats: {stats}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_sharp_memory())
