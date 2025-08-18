from fastapi import FastAPI
import sqlite3, os, random
from datetime import datetime
from collections import deque, defaultdict

app = FastAPI()

ADVANCED_SYSTEMS = False

class SharpMemorySystem:
    async def remember_conversation_advanced(self, *args, **kwargs):
        # Placeholder, future me embeddings/vector memory yahan implement karenge
        return
class UltraHybridMemorySystem:
    """Ultra Advanced Hybrid Memory with ALL previous features - FROM enhanced_cli.py"""
    def __init__(self, db_path="nova_ultra_professional_memory.db"):
        # FIXED: Proper path handling
        if not os.path.isabs(db_path):
            self.db_path = os.path.join(os.getcwd(), db_path)
        else:
            self.db_path = db_path
        self.setup_database()
        
        # Memory layers from enhanced_cli.py (great for conversation flow)
        self.conversation_context = deque(maxlen=100)  # Increased capacity
        self.user_profile = {}
        self.emotional_state = "neutral"
        self.learning_patterns = defaultdict(list)
        self.personality_insights = {}
        self.user_preferences = {}
        self.conversation_history = []
        
        # Memory layers from cli.py (great for technical queries)
        self.short_term_memory = deque(maxlen=200)  # Increased capacity
        self.working_memory = {}
        self.conversation_threads = {}
        self.context_memory = {}
        
        # Premium memory features
        self.voice_memory = deque(maxlen=50)
        self.file_memory = {}
        self.search_memory = deque(maxlen=30)
        self.image_memory = deque(maxlen=20)
        
        # Semantic memory for technical queries
        self.setup_semantic_memory()
        print("✅ Ultra Hybrid Memory System initialized")

    def setup_database(self):
        """Setup ultra comprehensive database schema"""
        try:
            # Ensure database directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Enhanced conversations table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    session_id TEXT,
                    user_input TEXT,
                    bot_response TEXT,
                    agent_type TEXT,
                    language TEXT,
                    emotion TEXT,
                    confidence REAL,
                    timestamp DATETIME,
                    feedback INTEGER DEFAULT 0,
                    context_summary TEXT,
                    learned_facts TEXT,
                    satisfaction_rating INTEGER,
                    conversation_thread_id TEXT,
                    intent_detected TEXT,
                    response_time REAL,
                    voice_used BOOLEAN DEFAULT 0,
                    location TEXT,
                    weather_context TEXT,
                    search_queries TEXT
                )
                ''')
                
                # Enhanced user profiles
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    career_goals TEXT,
                    current_role TEXT,
                    experience_years INTEGER,
                    skills TEXT,
                    preferences TEXT,
                    communication_style TEXT,
                    emotional_patterns TEXT,
                    conversation_patterns TEXT,
                    expertise_level TEXT,
                    topics_of_interest TEXT,
                    last_updated DATETIME,
                    total_conversations INTEGER DEFAULT 0,
                    preferred_voice TEXT,
                    location TEXT,
                    timezone TEXT,
                    personality_type TEXT,
                    learning_style TEXT
                )
                ''')
                
                # GitHub repositories
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS github_repos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_url TEXT UNIQUE,
                    repo_name TEXT,
                    analysis_date DATETIME,
                    file_count INTEGER,
                    languages_detected TEXT,
                    issues_found TEXT,
                    suggestions TEXT,
                    vector_db_path TEXT
                )
                ''')
                
                # Voice interactions
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS voice_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    voice_input TEXT,
                    voice_response TEXT,
                    language_detected TEXT,
                    emotion_detected TEXT,
                    voice_engine TEXT,
                    timestamp DATETIME
                )
                ''')
                
                # File processing history
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_processing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    file_path TEXT,
                    file_type TEXT,
                    processing_result TEXT,
                    timestamp DATETIME,
                    success BOOLEAN
                )
                ''')
                
                # Search history
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    search_query TEXT,
                    search_type TEXT,
                    results_count INTEGER,
                    timestamp DATETIME
                )
                ''')
                
                conn.commit()
            print("✅ Ultra Database initialized with premium schema")
        except Exception as e:
            print(f"⚠️ Database setup error: {e}")

    def setup_semantic_memory(self):
        """Setup semantic memory for technical queries"""
        try:
            if ADVANCED_SYSTEMS:
                self.semantic_memory = SharpMemorySystem()
            else:
                self.semantic_memory = None
        except Exception as e:
            print(f"⚠️ Semantic memory setup error: {e}")
            self.semantic_memory = None

    async def remember_conversation(self, user_id: str, session_id: str,
                                  user_input: str, bot_response: str,
                                  agent_type: str, language: str,
                                  emotion: str, confidence: float,
                                  intent: str = None, response_time: float = 0.0,
                                  voice_used: bool = False, location: str = None,
                                  weather_context: str = None, search_queries: str = None,
                                  file_analyzed: str = None):
        """Ultra enhanced conversation memory storage"""
        try:
            # Extract learning points
            learned_facts = self.extract_learning_points(user_input, bot_response)
            context_summary = self.generate_context_summary()
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO conversations
                (user_id, session_id, user_input, bot_response, agent_type,
                 language, emotion, confidence, timestamp, context_summary,
                 learned_facts, conversation_thread_id, intent_detected, response_time,
                 voice_used, location, weather_context, search_queries)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, session_id, user_input, bot_response, agent_type,
                      language, emotion, confidence, datetime.now(), context_summary,
                      learned_facts, self.generate_thread_id(), intent, response_time,
                      voice_used, location, weather_context, search_queries))
                conn.commit()
            
            # Store in conversation context
            self.conversation_context.append({
                'user': user_input,
                'bot': bot_response,
                'emotion': emotion,
                'agent': agent_type,
                'timestamp': datetime.now(),
                'voice_used': voice_used,
                'location': location,
                'file_analyzed': file_analyzed
            })
            
            # Store in short-term memory
            memory_entry = {
                'user_id': user_id,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'user_input': user_input,
                'ai_response': bot_response,
                'agent_used': agent_type,
                'emotion': emotion,
                'intent': intent,
                'voice_used': voice_used,
                'file_analyzed': file_analyzed
            }
            self.short_term_memory.append(memory_entry)
            
            # Store in semantic memory for technical queries
            if self.semantic_memory and agent_type in ['coding', 'business', 'technical']:
                try:
                    await self.semantic_memory.remember_conversation_advanced(
                        user_input, bot_response,
                        {'agent_used': agent_type, 'emotion': emotion},
                        user_id, session_id
                    )
                except Exception as e:
                    print(f"⚠️ Semantic memory storage error: {e}")
            
        except Exception as e:
            print(f"⚠️ Memory storage error: {e}")

    def get_relevant_context(self, user_input: str, user_id: str, limit: int = 15) -> str:
        """Get ultra comprehensive relevant context"""
        try:
            # Get database context
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT user_input, bot_response, emotion, learned_facts, agent_type,
                       voice_used, location, weather_context
                FROM conversations
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                ''', (user_id, limit))
                conversations = cursor.fetchall()
            
            if not conversations:
                return ""
            
            # Build ultra context summary
            context = "Previous conversation context:\n"
            for conv in conversations:
                context += f"[{conv[4].upper()}] User ({conv[2]}): {conv[0][:80]}...\n"
                context += f"NOVA: {conv[1][:80]}...\n"
                if conv[3]:
                    context += f"Learned: {conv[3]}\n"
                if conv[5]:  # voice_used
                    context += f"[VOICE MODE]\n"
                if conv[6]:  # location
                    context += f"Location: {conv[6]}\n"
                if conv[7]:  # weather_context
                    context += f"Weather: {conv[7]}\n"
                context += "---\n"
            
            return context
        except Exception as e:
            print(f"⚠️ Context retrieval error: {e}")
            return ""

    def extract_learning_points(self, user_input: str, bot_response: str) -> str:
        """Extract learning points from conversation"""
        learning_keywords = [
            "my name is", "i am", "i work", "i like", "i don't like",
            "my preference", "remember that", "important", "my goal",
            "my project", "my problem", "i need help with", "my role",
            "my company", "my experience", "my skills", "career goal",
            "i live in", "my location", "my city", "my country",
            "i prefer", "i want", "i need", "i use", "my favorite"
        ]
        
        learned = []
        user_lower = user_input.lower()
        for keyword in learning_keywords:
            if keyword in user_lower:
                sentences = user_input.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        learned.append(sentence.strip())
        
        return "; ".join(learned)

    def generate_context_summary(self) -> str:
        """Generate ultra context summary from recent conversations"""
        if not self.conversation_context:
            return ""
        
        recent_topics = []
        emotions = []
        agents = []
        voice_usage = []
        locations = []
        
        for conv in list(self.conversation_context)[-10:]:
            recent_topics.append(conv['user'][:50])
            emotions.append(conv['emotion'])
            agents.append(conv['agent'])
            if conv.get('voice_used'):
                voice_usage.append(True)
            if conv.get('location'):
                locations.append(conv['location'])
        
        dominant_emotion = max(set(emotions), key=emotions.count) if emotions else "neutral"
        most_used_agent = max(set(agents), key=agents.count) if agents else "general"
        voice_percentage = (len(voice_usage) / len(emotions)) * 100 if emotions else 0
        
        summary = f"Recent topics: {'; '.join(recent_topics)}. "
        summary += f"Emotion: {dominant_emotion}. Agent: {most_used_agent}. "
        if voice_percentage > 0:
            summary += f"Voice usage: {voice_percentage:.0f}%. "
        if locations:
            summary += f"Location context: {locations[-1]}."
        
        return summary

    def generate_thread_id(self) -> str:
        """Generate conversation thread ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"thread_{timestamp}_{random.randint(1000, 9999)}"

    def remember_file_processing(self, user_id: str, file_path: str,
                                file_type: str, result: str, success: bool):
        """Remember file processing"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO file_processing
                (user_id, file_path, file_type, processing_result, timestamp, success)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_id, file_path, file_type, result, datetime.now(), success))
                conn.commit()
            
            self.file_memory[file_path] = {
                'type': file_type,
                'result': result,
                'success': success,
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"⚠️ File memory error: {e}")