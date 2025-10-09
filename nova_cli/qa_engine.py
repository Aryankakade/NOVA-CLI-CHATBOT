"""
🚀 ULTIMATE Repository-Aware QA Engine with Smart Model Routing & Enhanced Features
🧠 Ultra-Smart Model Routing System - Auto-selects best free models for coding, reasoning, speed, and general tasks
"""

import os
import sys
import time
import json
import warnings
import re
import shutil
import tempfile
import hashlib
from typing import Optional, Dict, List, Any, Tuple, Union
from datetime import datetime
from pathlib import Path
import logging
import requests
from dotenv import load_dotenv
from collections import defaultdict, deque
import logging
import importlib
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== INGEST.PY INTEGRATION (PRESERVED) ==========
try:
    from ingest import list_processed_repositories, smart_ingest_repository
    REPOSITORY_REGISTRY_AVAILABLE = True
    print("✅ Repository registry functions loaded")
except ImportError as e:
    REPOSITORY_REGISTRY_AVAILABLE = False
    print(f"⚠️ Repository registry not available: {e}")
    
    # Define fallback functions
    def list_processed_repositories():
        return {"repositories": {}, "total_repos": 0}
    
    def smart_ingest_repository(repo_url):
        return {"success": False, "error": "Repository ingestion not available"}

try:
    from ingest import (
        get_retriever_for_qa_engine, get_file_mapping_for_rewrites, 
        is_repository_loaded, get_repository_info
    )
    INGEST_AVAILABLE = True
except ImportError:
    INGEST_AVAILABLE = False
    print("❌ CRITICAL: Ingest functions not available - Repository Q&A will not work!")


    GITHUB_INTEGRATION = True
    print("✅ GitHub QA engine loaded!")

    # Define fallback classes
    class UltimateQAEngine:
        def __init__(self, **kwargs):
            pass
        def ask(self, question):
            return "QA Engine not available"
        def set_repository_context(self, *args, **kwargs):
            return False
    
    def create_qa_engine(**kwargs):
        return UltimateQAEngine()
    
    EnhancedQAEngine = UltimateQAEngine

# Safe import with fallbacks for ingest functions
try:
    from ingest import main as ingest_repo
    print("✅ Ingest main function loaded")
except ImportError:
    def ingest_repo(repo_url, **kwargs):
        return {"success": False, "error": "Repository ingestion not available"}
    print("⚠️ Using fallback ingest function")

try:
    from ingest import process_and_store_documents
    print("✅ Document processing function loaded")
except ImportError:
    def process_and_store_documents(*args, **kwargs):
        return {"success": False, "error": "Document processing not available"}
    print("⚠️ Using fallback document processing")

# Load environment variables
load_dotenv()

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

try:
    # FIXED: Updated imports for LangChain 0.2+ (PRESERVED)
    from langchain_community.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
    from langchain.agents import initialize_agent, Tool, AgentType
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.chains import LLMChain, ConversationChain
    from langchain_community.utilities import GoogleSearchAPIWrapper, WikipediaAPIWrapper
    from langchain_community.tools import DuckDuckGoSearchRun
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"⚠️ LangChain not available: {e}")

try:
    import chromadb
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    VECTOR_SEARCH_AVAILABLE = True
    print("✅ Vector search capabilities loaded!")
except ImportError as e:
    VECTOR_SEARCH_AVAILABLE = False
    print(f"⚠️ Vector search not available: {e}")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ Requests library not available")

# ========== 🔥 NEW: ULTRA-SMART MODEL ROUTING SYSTEM ==========

class UltraSmartModelConfig:
    """🧠 Ultra-Smart Model Configuration - Best Free Models 2025"""
    
    @staticmethod
    def get_premium_models():
        """Get TOP 1% FREE MODELS optimized for different tasks"""
        return {
            # ============ CODING SPECIALISTS ============
            "coding": {
                "primary": {
                    "name": "DeepSeek_R1_Coder", 
                    "url": "https://openrouter.ai/api/v1/chat/completions",
                    "model": "deepseek/deepseek-r1-0528:free",
                    "provider": "OPENROUTER",
                    "speed": 8, "quality": 10, "context": 32000,
                    "specialty": "Best free coding model - complex algorithms, debugging, architecture"
                },
                "secondary": {
                    "name": "Groq_Coding_Pro",
                    "url": "https://api.groq.com/openai/v1/chat/completions", 
                    "model": "llama-3-groq-70b-8192-tool-use-preview",
                    "provider": "GROQ",
                    "speed": 10, "quality": 9, "context": 8192,
                    "specialty": "Fast coding with tool integration"
                }
            },
            
            # ============ REASONING CHAMPIONS ============
            "reasoning": {
                "primary": {
                    "name": "Cerebras_Reasoning_Beast",
                    "url": "https://api.cerebras.ai/v1/chat/completions",
                    "model": "llama3.3-70b",
                    "provider": "CEREBRAS", 
                    "speed": 10, "quality": 9.5, "context": 128000,
                    "specialty": "Ultra-fast reasoning - 450 tokens/sec for 70B model"
                },
                "secondary": {
                    "name": "Groq_Reasoning_Pro",
                    "url": "https://api.groq.com/openai/v1/chat/completions",
                    "model": "llama-3.3-70b-versatile",
                    "provider": "GROQ",
                    "speed": 9, "quality": 9, "context": 131072, 
                    "specialty": "Fast complex analysis and problem solving"
                }
            },
            
            # ============ SPEED DEMONS ============
            "speed": {
                "primary": {
                    "name": "Cerebras_Lightning",
                    "url": "https://api.cerebras.ai/v1/chat/completions",
                    "model": "llama3.1-8b", 
                    "provider": "CEREBRAS",
                    "speed": 10, "quality": 8, "context": 128000,
                    "specialty": "Fastest AI on planet - 2600 tokens/sec"
                },
                "secondary": {
                    "name": "Groq_Speed_King", 
                    "url": "https://api.groq.com/openai/v1/chat/completions",
                    "model": "llama-3.1-8b-instant",
                    "provider": "GROQ",
                    "speed": 9, "quality": 8, "context": 131072,
                    "specialty": "Ultra-fast responses - 1800+ tokens/sec"
                }
            },
            
            # ============ GENERAL PURPOSE ============
            "general": {
                "primary": {
                    "name": "NVIDIA_Llama_General",
                    "url": "https://integrate.api.nvidia.com/v1/chat/completions",
                    "model": "meta/llama-3.1-8b-instruct",
                    "provider": "NVIDIA",
                    "speed": 9, "quality": 8, "context": 4096, 
                    "specialty": "Excellent general assistant with broad knowledge"
                },
                 "secondary": {
                     "name": "Cerebras_Backup", 
                        "url": "https://api.cerebras.ai/v1/chat/completions",
                     "model": "llama3.1-8b",
                     "provider": "CEREBRAS",
                     "speed": 8, "quality": 7,
                     "specialty": "Backup general assistant"
                 },
                 "tertiary": {
                     "name": "HuggingFace_Free",
                      "url": "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct",
                      "model": "Qwen/Qwen2.5-7B-Instruct",
                      "provider": "HUGGINGFACE"
            }
        }
        }

class IntelligentQueryRouter:
    """🎯 Intelligent Query Router - Auto-selects best model for each task"""
    
    def __init__(self):
        self.models = UltraSmartModelConfig.get_premium_models()
        
        # Enhanced query patterns for ultra-smart routing
        self.patterns = {
            "coding": {
                "keywords": [
                    "code", "programming", "debug", "python", "javascript", "java", "c++", 
                    "react", "nodejs", "api", "function", "algorithm", "database", "sql",
                    "git", "github", "deployment", "testing", "frontend", "backend",
                    "fullstack", "django", "flask", "express", "vue", "angular", "docker", 
                    "kubernetes", "aws", "error", "exception", "syntax", "variable", "class",
                    "method", "loop", "array", "object", "json", "xml", "rewrite", "fix code",
                    "create function", "implement", "develop", "build app", "optimize code",
                    "refactor", "write script", "create api", "solve bug"
                ],
                "patterns": [
                    r"write.*code", r"create.*function", r"debug.*error", r"fix.*bug",
                    r"implement.*feature", r"optimize.*code", r"refactor.*", r"build.*app",
                    r"develop.*system", r"create.*api", r"solve.*problem", r"code.*review"
                ],
                "weight": 2.5
            },
            
            "reasoning": {
                "keywords": [
                    "analyze", "analysis", "compare", "evaluate", "assess", "examine",
                    "strategy", "plan", "solution", "approach", "methodology", "framework",
                    "pros and cons", "advantages", "disadvantages", "trade-offs", "research",
                    "study", "investigate", "explore", "understand", "complex", "complicated", 
                    "detailed", "comprehensive", "explain", "reasoning", "logic", "thinking",
                    "problem solving", "decision", "choice", "option", "calculate"
                ],
                "patterns": [
                    r"how to.*", r"what.*best way", r"explain.*why", r"analyze.*",
                    r"compare.*with", r"what.*difference", r"help.*understand", 
                    r"reason.*", r"logic.*", r"think.*about", r"solve.*problem"
                ],
                "weight": 2.0
            },
            
            "speed": {
                "keywords": [
                    "quick", "fast", "simple", "brief", "short", "summary", "overview",
                    "basic", "simple question", "yes or no", "definition", "meaning", 
                    "what is", "who is", "when", "where", "how much", "how many",
                    "quick answer", "briefly", "instant", "immediate"
                ],
                "patterns": [
                    r"^(what|who|when|where|how|why).*\?$", r"define.*", r"meaning.*",
                    r"quick.*question", r"simple.*", r"briefly.*", r"in short.*",
                    r"tell me.*", r"just.*"
                ], 
                "weight": 1.8
            }
        }
    
    def route_query(self, query: str) -> Tuple[str, float, Dict]:
     """🧠 Intelligent query routing with repository awareness"""
     query_lower = query.lower()
     scores = {}
    
     # 🔥 NEW: Enhanced repository keywords
     repo_keywords = [
        'file', 'files', 'repository', 'repo', 'code', 'how many', 'what is',
        'what does', 'structure', 'contains', 'written in', 'language',
        'dependencies', 'main', 'functions', 'classes', 'imports', 'project',
        'codebase', 'application', 'system', 'module', 'component', 'package',
        'directory', 'folder', 'source', 'implementation', 'architecture'
     ]
    
     # Check if this is a repository query first
     repo_score = sum(2 for kw in repo_keywords if kw in query_lower)
    
     # Score each category
     for category, config in self.patterns.items():
        score = 0
        
        # Keyword matching
        keyword_matches = sum(1 for kw in config["keywords"] if kw in query_lower)
        score += keyword_matches * 2
        
        # Pattern matching
        pattern_matches = sum(1 for pattern in config["patterns"] if re.search(pattern, query_lower))
        score += pattern_matches * 3
        
        # Apply category weight
        score *= config.get("weight", 1.0)
        
        if score > 0:
            scores[category] = score
    
     # 🔥 NEW: Force repository routing for repo queries
     if repo_score >= 3:  # Strong repository indicators
        best_category = "coding"  # Use coding model for repo queries
        confidence = min(0.85, 0.6 + (repo_score * 0.05))  # Lower but realistic confidence
        selected_model = self.models["coding"]["primary"]
        print(f"🔥 Repository query detected! Using coding model with {confidence:.1%} confidence")
        return best_category, confidence, selected_model
    
     # Determine best category for non-repo queries
     if not scores:
        return "general", 0.5, self.models["general"]["primary"]
    
     best_category = max(scores, key=scores.get)
    
     # 🔥 FIXED: Lower confidence threshold for realistic routing
     confidence = min(0.85, scores[best_category] / 10 + 0.4)  # More achievable confidence
    
     # Get best model for category
     category_models = self.models.get(best_category, {})
     selected_model = category_models.get("primary", self.models["general"]["primary"])
    
     return best_category, confidence, selected_model

class MultiKeyManager:
    """🔑 Enhanced Multi-Key Manager with 10+ keys per provider"""
    
    def __init__(self):
        self.provider_keys = self._load_keys()
        self.key_stats = {}
        self.initialize_tracking()
    
    def _load_keys(self) -> Dict[str, List[str]]:
        """Load multiple API keys for each provider"""
        keys = {}
        providers = ['GROQ', 'OPENROUTER', 'TOGETHER', 'CEREBRAS', 'DEEPSEEK', 
                    'HUGGINGFACE', 'AI21', 'NVIDIA', 'FIREWORKS']
        
        for provider in providers:
            provider_keys = []
            
            # Load up to 10 keys per provider
            for i in range(1, 11):
                key = os.getenv(f"{provider}_API_KEY_{i}")
                if key:
                    provider_keys.append(key)
            
            # Single key fallback
            single_key = os.getenv(f"{provider}_API_KEY") 
            if single_key and single_key not in provider_keys:
                provider_keys.insert(0, single_key)
            
            if provider_keys:
                keys[provider] = provider_keys
                logger.info(f"✅ Loaded {len(provider_keys)} keys for {provider}")
        
        return keys
    
    def initialize_tracking(self):
        """Initialize key usage tracking"""
        for provider, keys in self.provider_keys.items():
            self.key_stats[provider] = {}
            for idx, key in enumerate(keys):
                self.key_stats[provider][idx] = {
                    'requests': 0,
                    'failures': 0, 
                    'last_used': 0,
                    'rate_limited': False,
                    'quota_exceeded': False
                }
    
    def get_best_key(self, provider: str) -> Optional[Tuple[str, int]]:
        """Get best available key for provider"""
        if provider not in self.provider_keys:
            return None
        
        keys = self.provider_keys[provider]
        current_time = time.time()
        
        # Find best available key
        best_key_idx = None
        best_score = -1
        
        for idx, key in enumerate(keys):
            stats = self.key_stats[provider][idx]
            
            # Skip if rate limited or quota exceeded
            if stats['rate_limited'] or stats['quota_exceeded']:
                # Reset after 1 hour for rate limits
                if stats['rate_limited'] and current_time - stats['last_used'] > 3600:
                    stats['rate_limited'] = False
                # Reset after 24 hours for quota
                if stats['quota_exceeded'] and current_time - stats['last_used'] > 86400:
                    stats['quota_exceeded'] = False
                else:
                    continue
            
            # Calculate score (prefer less used, more successful keys)
            success_rate = 1.0
            if stats['requests'] > 0:
                success_rate = (stats['requests'] - stats['failures']) / stats['requests']
            
            time_factor = max(0, 3600 - (current_time - stats['last_used'])) / 3600
            score = success_rate * 0.7 + time_factor * 0.3
            
            if score > best_score:
                best_score = score
                best_key_idx = idx
        
        if best_key_idx is not None:
            return keys[best_key_idx], best_key_idx
        
        return None
    
    def record_usage(self, provider: str, key_idx: int, success: bool, error_type: str = None):
        """Record key usage statistics"""
        if provider in self.key_stats and key_idx in self.key_stats[provider]:
            stats = self.key_stats[provider][key_idx]
            stats['requests'] += 1
            stats['last_used'] = time.time()
            
            if not success:
                stats['failures'] += 1
                if error_type == 'rate_limit':
                    stats['rate_limited'] = True
                elif error_type == 'quota':
                    stats['quota_exceeded'] = True

# ========== PRESERVED: Enhanced Callback Handler (ORIGINAL) ==========

class EnhancedRepoAwareCallbackHandler(BaseCallbackHandler):
    """🎯 Ultra-enhanced callback handler with smart repo awareness"""
    
    def __init__(self):
        self.start_time = None
        self.tokens_used = 0
        self.repo_context_used = False
        self.file_operations = []
        self.query_type = "general"
        # 🔥 NEW: Smart model tracking
        self.selected_model = None
        self.query_confidence = 0.0
        self.routing_info = {}
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        self.start_time = time.time()
        
        # Enhanced detection of query types (PRESERVED)
        for prompt in prompts:
            if "REPOSITORY_CONTEXT" in prompt or "FILE_CONTENT" in prompt:
                self.repo_context_used = True
            if "REPOSITORY_ANALYSIS" in prompt:
                self.query_type = "repo_analysis"
            elif "FILE_DEBUG" in prompt:
                self.query_type = "file_debug"
            elif "CODE_ONLY" in prompt:
                self.query_type = "code_generation"
            elif "EXPLANATION" in prompt:
                self.query_type = "explanation"
        
        # 🔥 NEW: Smart model routing status
        status_messages = {
            "repo_analysis": f"🔍 Analyzing repository with {self.selected_model or 'smart model'}...",
            "file_debug": f"🐛 Debugging file with {self.selected_model or 'smart model'}...",
            "code_generation": f"💻 Generating code with {self.selected_model or 'smart model'}...",
            "explanation": f"📚 Preparing explanation with {self.selected_model or 'smart model'}...",
            "general": f"🧠 Processing with {self.selected_model or 'smart model'}..." + 
                      (" (with repo context)" if self.repo_context_used else "")
        }
        
        print(status_messages.get(self.query_type, status_messages["general"]))
        
        # Show routing confidence if available
        if self.query_confidence > 0:
            print(f"🎯 Routing confidence: {self.query_confidence:.1%}")
    
    def on_llm_end(self, response, **kwargs) -> None:
        if self.start_time:
            duration = time.time() - self.start_time
            context_note = " (with repo context)" if self.repo_context_used else ""
            type_note = f" [{self.query_type}]" if self.query_type != "general" else ""
            model_note = f" via {self.selected_model}" if self.selected_model else ""
            
            print(f"✅ Response generated in {duration:.2f}s{context_note}{type_note}{model_note}")
    
    def on_llm_error(self, error, **kwargs) -> None:
        model_note = f" (Model: {self.selected_model})" if self.selected_model else ""
        print(f"❌ LLM Error{model_note}: {error}")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        tool_name = serialized.get("name", "Unknown")
        print(f"🔧 Using tool: {tool_name}")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        print("🔧 Tool execution completed")
    
    def set_routing_info(self, model_name: str, confidence: float, routing_info: Dict):
        """🔥 NEW: Set smart routing information"""
        self.selected_model = model_name
        self.query_confidence = confidence
        self.routing_info = routing_info

# ========== PRESERVED: Ultimate File Rewrite Detector (ORIGINAL + ENHANCED) ==========

class UltimateFileRewriteDetector:
    """🧠 ULTIMATE File Rewrite & Analysis Intent Detection"""
    
    def __init__(self):
        # ALL ORIGINAL PATTERNS PRESERVED + NEW ONES
        self.rewrite_patterns = [
            # Direct file rewrite commands (PRESERVED)
            r'(?:rewrite|modify|update|change|fix|improve)\s+(?:the\s+)?(?:file\s+)?([a-zA-Z0-9_.-]+\.(?:py|js|html|css|json|md|txt|java|cpp|c|go|rs|php|rb|swift|kt|sql|yaml|yml|xml))',
            r'(?:give\s+me|show\s+me|create|generate)\s+(?:the\s+)?(?:whole|complete|entire|full|updated|upgraded|modified)\s+([a-zA-Z0-9_.-]+\.(?:py|js|html|css|json|md|txt|java|cpp|c|go|rs|php|rb|swift|kt|sql|yaml|yml|xml))',
            r'(?:fix|debug|resolve|correct)\s+(?:issues?\s+in\s+)?([a-zA-Z0-9_.-]+\.(?:py|js|html|css|json|md|txt|java|cpp|c|go|rs|php|rb|swift|kt|sql|yaml|yml|xml))',
            r'([a-zA-Z0-9_.-]+\.(?:py|js|html|css|json|md|txt|java|cpp|c|go|rs|php|rb|swift|kt|sql|yaml|yml|xml))\s+(?:file\s+)?(?:ko\s+)?(?:rewrite|modify|update|change)',
            
            # Intent-based patterns (PRESERVED)
            r'make\s+(?:changes\s+to\s+)?([a-zA-Z0-9_.-]+\.(?:py|js|html|css|json|md|txt|java|cpp|c|go|rs|php|rb|swift|kt|sql|yaml|yml|xml))',
            r'improve\s+(?:the\s+)?([a-zA-Z0-9_.-]+\.(?:py|js|html|css|json|md|txt|java|cpp|c|go|rs|php|rb|swift|kt|sql|yaml|yml|xml))',
            r'add\s+(?:features?\s+to\s+)?([a-zA-Z0-9_.-]+\.(?:py|js|html|css|json|md|txt|java|cpp|c|go|rs|php|rb|swift|kt|sql|yaml|yml|xml))',
            
            # Hindi/Hinglish patterns (PRESERVED)
            r'([a-zA-Z0-9_.-]+\.(?:py|js|html|css|json|md|txt|java|cpp|c|go|rs|php|rb|swift|kt|sql|yaml|yml|xml))\s+(?:file\s+)?ko\s+(?:rewrite|modify|update|change|fix|improve)\s+kar',
            r'iss?\s+([a-zA-Z0-9_.-]+\.(?:py|js|html|css|json|md|txt|java|cpp|c|go|rs|php|rb|swift|kt|sql|yaml|yml|xml))\s+(?:file\s+)?ko\s+(?:fix|improve|update)',
        ]

        # ✅ PRESERVED: Response type detection patterns
        self.response_type_patterns = {
            "CODE_ONLY": [
                r"give\s+me\s+(?:the\s+)?(?:whole|complete|entire|full)\s+(?:file|code)",
                r"(?:only|just)\s+(?:the\s+)?(?:file|code)\s+content",
                r"raw\s+file",
                r"sirf\s+(?:file|code)\s+(?:chahiye|dedo|content)",
                r"bas\s+(?:code|file)\s+(?:chahiye|dedo)",
                r"without\s+(?:explanation|comments?)",
                r"no\s+(?:comments?|explanation)",
                r"plain\s+code",
                r"clean\s+code\s+only"
            ],
            "EXPLANATION_PLUS_CODE": [
                r"explain\s+(?:and\s+)?(?:rewrite|modify|fix|improve)",
                r"(?:tell|show)\s+me\s+what\s+(?:changes|modifications|improvements)",
                r"explain\s+(?:the\s+)?(?:improvements|fixes|changes)",
                r"with\s+(?:comments|explanation|details)",
                r"describe\s+(?:the\s+)?(?:changes|improvements|modifications)",
                r"(?:comments|explanation)\s+(?:ke\s+)?saath",
                r"samjhao\s+(?:aur\s+)?(?:rewrite|modify|fix)",
                r"explanation\s+(?:chahiye|needed|required)",
                r"detail\s+(?:mein|me)\s+(?:batao|explain)",
                r"steps\s+(?:ke\s+saath|with)",
                r"detailed\s+(?:analysis|explanation)",
            ]
        }

        # ✅ PRESERVED: Repository analysis patterns
        self.repo_analysis_patterns = [
            r"check\s+(?:my\s+)?(?:repo|repository)\s+(?:for\s+)?(?:issues?|problems?|bugs?)",
            r"analyze\s+(?:my\s+)?(?:repo|repository)",
            r"(?:any|what)\s+(?:issues?|problems?)\s+(?:in\s+)?(?:my\s+)?(?:repo|repository)",
            r"scan\s+(?:my\s+)?(?:repo|repository)",
            r"review\s+(?:my\s+)?(?:repo|repository)",
            r"(?:repo|repository)\s+(?:analysis|review|check)",
            r"meri\s+repo\s+(?:check|analyze)\s+kar",
            r"repo\s+(?:mein|me)\s+kya\s+(?:issues?|problems?)\s+(?:hain|hai)",
        ]

        # ✅ PRESERVED: File-specific analysis patterns
        self.file_analysis_patterns = [
            r"what\s+(?:is\s+)?(?:issue|problem|wrong)\s+(?:in\s+|with\s+)?([a-zA-Z0-9_.-]+\.(?:py|js|html|css|json|md|txt|java|cpp|c|go|rs|php|rb|swift|kt|sql|yaml|yml|xml))",
            r"(?:debug|analyze|check)\s+([a-zA-Z0-9_.-]+\.(?:py|js|html|css|json|md|txt|java|cpp|c|go|rs|php|rb|swift|kt|sql|yaml|yml|xml))",
            r"([a-zA-Z0-9_.-]+\.(?:py|js|html|css|json|md|txt|java|cpp|c|go|rs|php|rb|swift|kt|sql|yaml|yml|xml))\s+(?:mein|me)\s+kya\s+(?:issue|problem|galti)\s+hai",
            r"problems?\s+(?:in|with)\s+([a-zA-Z0-9_.-]+\.(?:py|js|html|css|json|md|txt|java|cpp|c|go|rs|php|rb|swift|kt|sql|yaml|yml|xml))",
            r"(?:issues?|bugs?)\s+(?:in|with)\s+([a-zA-Z0-9_.-]+\.(?:py|js|html|css|json|md|txt|java|cpp|c|go|rs|php|rb|swift|kt|sql|yaml|yml|xml))",
        ]

        self.intent_keywords = [
            'rewrite', 'modify', 'update', 'change', 'fix', 'improve', 'enhance',
            'upgrade', 'optimize', 'refactor', 'debug', 'resolve', 'correct',
            'add features', 'make changes', 'give me whole', 'show complete',
            'create updated', 'generate modified', 'analyze', 'check', 'review'
        ]

    def detect_intent(self, question: str) -> Dict[str, Any]:
        """🎯 Ultimate intent detection for all query types (PRESERVED + ENHANCED)"""
        question_lower = question.lower()
        
        # ✅ PRESERVED: All original steps with same logic
        
        # Step 1: Check for repository analysis intent
        for pattern in self.repo_analysis_patterns:
            if re.search(pattern, question_lower):
                return {
                    'intent_type': 'repo_analysis',
                    'is_rewrite': False,
                    'needs_explanation': True,
                    'response_type': 'detailed_analysis',
                    'filename': None,
                    'original_query': question,
                    'confidence': 0.9
                }

        # Step 2: Check for file-specific analysis
        for pattern in self.file_analysis_patterns:
            match = re.search(pattern, question_lower)
            if match:
                filename = match.group(1)
                return {
                    'intent_type': 'file_analysis',
                    'is_rewrite': False,
                    'needs_explanation': True,
                    'response_type': 'file_debug_analysis',
                    'filename': filename,
                    'original_query': question,
                    'confidence': 0.9
                }

        # Step 3: Check for file rewrite patterns
        for pattern in self.rewrite_patterns:
            match = re.search(pattern, question_lower)
            if match:
                filename = match.group(1)
                
                # Determine response type preference
                wants_code_only = False
                wants_explanation = False
                
                # Check for CODE_ONLY patterns
                for code_pattern in self.response_type_patterns["CODE_ONLY"]:
                    if re.search(code_pattern, question_lower):
                        wants_code_only = True
                        break

                # Check for EXPLANATION_PLUS_CODE patterns
                for exp_pattern in self.response_type_patterns["EXPLANATION_PLUS_CODE"]:
                    if re.search(exp_pattern, question_lower):
                        wants_explanation = True
                        break

                # ✅ PRESERVED: Default to explanation for better UX
                if not wants_code_only and not wants_explanation:
                    wants_explanation = True  # Safe default

                return {
                    'intent_type': 'file_rewrite',
                    'is_rewrite': True,
                    'filename': filename,
                    'response_type': 'code_only' if wants_code_only else 'explanation_plus_code',
                    'needs_explanation': wants_explanation,
                    'wants_code_only': wants_code_only,
                    'original_query': question,
                    'confidence': 0.9
                }

        # Step 4: Check for general intent keywords with file detection
        intent_score = 0
        for keyword in self.intent_keywords:
            if keyword in question_lower:
                intent_score += 1

        if intent_score >= 2:
            # Look for any file extensions in the question
            file_pattern = r'([a-zA-Z0-9_.-]+\.(?:py|js|html|css|json|md|txt|java|cpp|c|go|rs|php|rb|swift|kt|sql|yaml|yml|xml))'
            match = re.search(file_pattern, question_lower)
            if match:
                # Check if it's analysis or rewrite
                analysis_keywords = ['issue', 'problem', 'bug', 'error', 'debug', 'analyze', 'check']
                is_analysis = any(keyword in question_lower for keyword in analysis_keywords)

                if is_analysis:
                    return {
                        'intent_type': 'file_analysis',
                        'is_rewrite': False,
                        'needs_explanation': True,
                        'response_type': 'file_debug_analysis',
                        'filename': match.group(1),
                        'original_query': question,
                        'confidence': 0.7
                    }
                else:
                    return {
                        'intent_type': 'file_rewrite',
                        'is_rewrite': True,
                        'filename': match.group(1),
                        'response_type': 'explanation_plus_code',  # Default safe
                        'needs_explanation': True,
                        'wants_code_only': False,
                        'original_query': question,
                        'confidence': 0.7
                    }

        # Step 5: Default case - general question
        return {
            'intent_type': 'general_question',
            'is_rewrite': False,
            'needs_explanation': True,
            'response_type': 'general_answer',
            'filename': None,
            'original_query': question,
            'confidence': 1.0
        }

# ========== PRESERVED: Repository Context Manager (ORIGINAL + ENHANCED) ==========

class RepositoryContextManager:
    """🗂️ Advanced repository context management with enhanced search (PRESERVED + ENHANCED)"""
    
    def __init__(self):
        self.vector_store = None
        self.retriever = None
        self.active_repo = None
        self.file_mappings = {}
        self.embeddings = None
        # 🔥 NEW: Enhanced repository stats
        self.repo_stats = {
            'total_files': 0,
            'total_chunks': 0,
            'last_updated': None,
            'supported_extensions': []
        }

        # Initialize embeddings if available (PRESERVED)
        if VECTOR_SEARCH_AVAILABLE:
            try:
                self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                print("✅ Repository embeddings initialized")
            except Exception as e:
                print(f"⚠️ Embeddings initialization failed: {e}")

    def set_repository_context(self, vector_db_path: str = "./chroma_db", repo_url: str = None):
     """✅ ENHANCED: Set repository context with SMART PATH DETECTION + cross-process support"""
    
     # 🔥 STEP 1: Smart path detection - try to get actual database path first
     actual_db_path = vector_db_path
     smart_path_found = False
    
     print(f"🔍 Initial path: {vector_db_path}")
    
     # 🔥 NEW: Try to get actual DB path from smart ingestion registry
     try:
        from ingest import SmartMultiRepoManager
        manager = SmartMultiRepoManager()
        registry = manager.get_all_repositories()
        
        if repo_url and repo_url in registry.get("repositories", {}):
            actual_db_path = registry["repositories"][repo_url]["db_path"]
            smart_path_found = True
            print(f"🎯 Found smart DB path for {repo_url}: {actual_db_path}")
        elif registry.get("last_active"):
            last_repo = registry["last_active"]
            if last_repo in registry.get("repositories", {}):
                actual_db_path = registry["repositories"][last_repo]["db_path"]
                smart_path_found = True
                print(f"🎯 Using last active repo DB: {actual_db_path}")
        else:
            print("ℹ️ No smart paths found in registry")
            
     except Exception as e:
        print(f"⚠️ Could not access smart DB registry: {e}")
    
     # 🔥 STEP 2: Try direct database loading with SMART PATH first
     paths_to_try = []
     if smart_path_found:
        paths_to_try.append(actual_db_path)
     paths_to_try.append(vector_db_path)  # Original path as fallback
    
     for db_path in paths_to_try:
        if os.path.exists(db_path):
            print(f"📂 Found existing database at {db_path}")
            try:
                success = self._original_set_repository_context(db_path, repo_url)
                if success:
                    print("✅ Repository context loaded from database")
                    print(f"📊 Files: {len(self.file_mappings)}, Database: Ready")
                    
                    # 🔥 NEW: Verify the connection works
                    if hasattr(self, 'retriever') and self.retriever:
                        try:
                            # Test a simple search to verify connection
                            test_docs = self.retriever.get_relevant_documents("test")
                            print(f"✅ Database connection verified: {len(test_docs)} docs accessible")
                        except Exception as verify_error:
                            print(f"⚠️ Database connection verification failed: {verify_error}")
                    
                    return True
            except Exception as e:
                print(f"⚠️ Direct database loading failed for {db_path}: {e}")
                continue  # Try next path
    
     # ✅ PRIORITY 2: Try ingest.py integration as backup (same process only)
     if INGEST_AVAILABLE:
        if is_repository_loaded():
            print("🔗 Repository detected from ingest.py")
            repo_info = get_repository_info()
            print(f"📋 Repository: {repo_info.get('full_name', 'Unknown')}")
            
            # Get components from ingest.py
            self.retriever = get_retriever_for_qa_engine()
            self.file_mappings = get_file_mapping_for_rewrites()
            self.active_repo = repo_info.get('url', repo_url)
            
            # Enhanced repo stats
            self._update_repo_stats(repo_info)
            
            if self.retriever and self.file_mappings:
                print(f"✅ Repository context loaded via ingest.py integration")
                print(f"📊 Files: {len(self.file_mappings)}, Repo: {repo_info.get('full_name', 'Unknown')}")
                return True
            else:
                print("⚠️ Ingest components not ready, falling back to direct method")
        else:
            print("ℹ️ No repository in memory, checking database...")
     
     # 🔥 STEP 3: Enhanced fallback with multiple path attempts
     fallback_paths = [
        actual_db_path if smart_path_found else None,
        vector_db_path,
        "./chroma_db",
        "./chroma_repos"
     ]
    
     # Remove None and duplicates
     fallback_paths = list(dict.fromkeys([p for p in fallback_paths if p]))
    
     print(f"🔄 Trying fallback paths: {fallback_paths}")
    
     for fallback_path in fallback_paths:
        if os.path.exists(fallback_path):
            print(f"🔄 Attempting fallback: {fallback_path}")
            try:
                success = self._original_set_repository_context(fallback_path, repo_url)
                if success:
                    print(f"✅ Fallback successful with: {fallback_path}")
                    return True
            except Exception as e:
                print(f"⚠️ Fallback failed for {fallback_path}: {e}")
                continue
        else:
            print(f"❌ Path doesn't exist: {fallback_path}")
    
     # 🔥 STEP 4: Final attempt - scan for any chroma repositories
     try:
        print("🔍 Scanning for any available repositories...")
        base_paths = ["./chroma_repos", "./chroma_db", "."]
        
        for base_path in base_paths:
            if os.path.exists(base_path):
                for item in os.listdir(base_path):
                    item_path = os.path.join(base_path, item)
                    if os.path.isdir(item_path):
                        # Check if it looks like a chroma database
                        chroma_files = ["chroma.sqlite3", "index"]
                        if any(os.path.exists(os.path.join(item_path, cf)) for cf in chroma_files):
                            print(f"🔍 Found potential database: {item_path}")
                            try:
                                success = self._original_set_repository_context(item_path, repo_url)
                                if success:
                                    print(f"✅ Connected to discovered database: {item_path}")
                                    return True
                            except Exception as e:
                                print(f"⚠️ Discovered database connection failed: {e}")
                                continue
     except Exception as e:
        print(f"⚠️ Repository scanning failed: {e}")
    
     print("❌ All repository context attempts failed")
     return False
    
    def is_repository_loaded(self) -> bool:
     """Check if a repository is properly loaded and has documents"""
     try:
        if not self.repo_context or not self.repo_context.vector_store:
            return False
        collection = self.repo_context.vector_store._collection
        return collection is not None and collection.count() > 0
     except Exception:
        return False


    def _update_repo_stats(self, repo_info: Dict):
        """🔥 NEW: Update repository statistics"""
        self.repo_stats.update({
            'total_files': len(self.file_mappings),
            'last_updated': datetime.now().isoformat(),
            'supported_extensions': list(set(
                os.path.splitext(f)[1] for f in self.file_mappings.keys() 
                if os.path.splitext(f)[1]
            ))
        })

    def _original_set_repository_context(self, vector_db_path: str = "./chroma_db", repo_url: str = None):
        """PRESERVED: Original repository context setting method"""
        if not VECTOR_SEARCH_AVAILABLE or not self.embeddings:
            print("⚠️ Vector search not available")
            return False

        try:
            if os.path.exists(vector_db_path):
                self.vector_store = Chroma(
                    persist_directory=vector_db_path,
                    embedding_function=self.repo_context.embeddings,
                    collection_name="repository_docs"
                )
                self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 8})
                self.active_repo = repo_url

                self._load_ingest_state(vector_db_path)

                if not self.file_mappings:
                    self._build_file_mappings()

                print(f"✅ Repository context loaded from {vector_db_path}")
                return True
            else:
                print(f"⚠️ Vector database not found at {vector_db_path}")
                return False
        except Exception as e:
            print(f"❌ Failed to set repository context: {e}")
            return False
        
    def _original_set_repository_context(self, vector_db_path: str = "./chroma_db", repo_url: str = None):
     """PRESERVED: Original repository context setting method"""
     if not VECTOR_SEARCH_AVAILABLE or not self.embeddings:
        print("⚠️ Vector search not available")
        return False

     try:
        if os.path.exists(vector_db_path):
            self.vector_store = Chroma(
                persist_directory=vector_db_path,
                embedding_function=self.repo_context.embeddings,
                collection_name="repository_docs"
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 8})
            self.active_repo = repo_url

            # 🔧 NEW: Try to load persistent ingest state
            self._load_ingest_state(vector_db_path)
            
            # Fallback: Build file mappings from vector store if state not found
            if not self.file_mappings:
                self._build_file_mappings()

            print(f"✅ Repository context loaded from {vector_db_path}")
            return True
        else:
            print(f"⚠️ Vector database not found at {vector_db_path}")
            return False
     except Exception as e:
        print(f"❌ Failed to set repository context: {e}")
        return False

    def _build_file_mappings(self):
        """PRESERVED: Build file path mappings from vector store"""
        if not self.vector_store:
            return

        try:
            # Get all documents to build file mappings
            collection = self.vector_store._collection
            all_docs = collection.get()

            if all_docs and 'metadatas' in all_docs:
                for metadata in all_docs['metadatas']:
                    if metadata and 'source' in metadata:
                        source = metadata['source']
                        filename = os.path.basename(source)
                        self.file_mappings[filename.lower()] = source

            print(f"✅ Built mappings for {len(self.file_mappings)} files")
        except Exception as e:
            print(f"⚠️ Failed to build file mappings: {e}")

    def search_repository(self, query: str, max_results: int = 8) -> List[Dict[str, Any]]:
        """PRESERVED: Enhanced repository search with better results"""
        if not self.retriever:
            return []

        try:
            docs = self.retriever.get_relevant_documents(query)
            results = []

            for doc in docs[:max_results]:
                results.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'filename': os.path.basename(doc.metadata.get('source', '')),
                    'relevance_score': 1.0
                })

            return results
        except Exception as e:
            print(f"❌ Repository search failed: {e}")
            return []

    def get_file_content(self, filename: str) -> Optional[str]:
        """PRESERVED: Enhanced file content retrieval"""
        if not self.retriever:
            return None

        try:
            # Search for the specific file with multiple strategies (PRESERVED logic)
            filename_lower = filename.lower()

            # Strategy 1: Direct filename mapping
            if filename_lower in self.file_mappings:
                source_path = self.file_mappings[filename_lower]
                
                # Search for content from this specific file with multiple queries
                search_queries = [
                    f"file:{filename}",
                    filename,
                    filename_lower,
                    os.path.splitext(filename)[0]  # filename without extension
                ]

                file_chunks = []
                for query in search_queries:
                    docs = self.retriever.get_relevant_documents(query)
                    for doc in docs:
                        if doc.metadata.get('source', '').lower().endswith(filename_lower):
                            file_chunks.append(doc.page_content)

                if file_chunks:
                    # Remove duplicates while preserving order (PRESERVED)
                    unique_chunks = []
                    seen = set()
                    for chunk in file_chunks:
                        if chunk not in seen:
                            unique_chunks.append(chunk)
                            seen.add(chunk)
                    return '\n\n'.join(unique_chunks)

            return None
        except Exception as e:
            print(f"❌ Failed to get file content for {filename}: {e}")
            return None

    def analyze_repository_issues(self) -> Dict[str, Any]:
        """PRESERVED: Analyze repository for common issues and problems"""
        if not self.retriever:
            return {"error": "No repository context available"}

        try:
            # Search for common issue patterns (PRESERVED)
            issue_queries = [
                "error exception bug",
                "TODO FIXME",
                "deprecated warning",
                "missing import",
                "undefined variable",
                "syntax error",
                "security vulnerability"
            ]

            issues_found = {}
            for query in issue_queries:
                docs = self.retriever.get_relevant_documents(query)
                if docs:
                    issues_found[query] = [
                        {
                            'filename': os.path.basename(doc.metadata.get('source', '')),
                            'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        }
                        for doc in docs[:3]
                    ]

            return {
                "total_issues": len(issues_found),
                "issues_by_type": issues_found,
                "files_scanned": len(self.file_mappings),
                "repo_name": self.active_repo.split('/')[-1] if self.active_repo else "Unknown",
                # 🔥 NEW: Enhanced analysis data
                "repo_stats": self.repo_stats
            }
        except Exception as e:
            return {"error": f"Repository analysis failed: {e}"}

    def get_repository_summary(self) -> str:
        """PRESERVED + ENHANCED: Get a summary of the repository"""
        if not self.active_repo:
            return "No repository context available."
        
        file_count = len(self.file_mappings)
        repo_name = self.active_repo.split('/')[-1] if '/' in self.active_repo else self.active_repo
        
        # 🔥 NEW: Enhanced summary with stats
        extensions = ', '.join(self.repo_stats.get('supported_extensions', [])[:5])
        
        return (f"Repository: {repo_name} | Files: {file_count} | "
                f"Extensions: {extensions} | Vector DB: ✅ | Smart Routing: ✅")
    
    def _load_ingest_state(self, vector_db_path: str):
     """🔧 NEW: Load persistent ingest state if available"""
     try:
        state_file = os.path.join(vector_db_path, "nova_ingest_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            self.file_mappings = state.get("file_mappings", {})
            repo_metadata = state.get("repo_metadata", {})
            
            if self.file_mappings:
                print(f"✅ Loaded persistent state: {len(self.file_mappings)} file mappings")
                self._update_repo_stats(repo_metadata)
            
     except Exception as e:
        print(f"⚠️ Could not load ingest state: {e}")

# ========== 🔥 NEW: Enhanced UltimateQAEngine with Smart Model Integration ==========

class UltimateQAEngine:
    """🚀 ULTIMATE Repository-Aware QA Engine with Smart Model Routing (PRESERVED + ENHANCED)"""
    
    def __init__(self, model_name: str = "anthropic/claude-3.5-sonnet", temperature: float = 0.7, max_tokens: int = 700,
            enable_memory: bool = True, enable_tools: bool = True, memory_type: str = "buffer", 
            max_memory_tokens: int = 900, enable_smart_routing: bool = True, fallback_to_original: bool = True):
     """Initialize the ULTIMATE QA Engine with Smart Model Routing"""
    
     self.model_name = model_name
     self.temperature = temperature
     self.provider_token_limits = {
        'GROQ': 1000,
        'CEREBRAS': 1200,
        'AI21': 600,
        'DEEPSEEK': 500,
        'NVIDIA': 800,
        'OPENROUTER': 400
     }
     self.max_tokens = max_tokens
     self.enable_memory = enable_memory
     self.enable_tools = enable_tools
     self.memory_type = memory_type
     self.max_memory_tokens = max_memory_tokens
     self.enable_smart_routing = enable_smart_routing
     self.fallback_to_original = fallback_to_original
    
     # ✅ FIXED: Multi-key OpenRouter API handling
     # Try multi-key system first
     api_keys = []
     for i in range(1, 11):
        key = os.getenv(f"OPENROUTER_API_KEY_{i}")
        if key:
            api_keys.append(key)
            
     # Fallback to single key
     single_key = os.getenv("OPENROUTER_API_KEY")
     if single_key and single_key not in api_keys:
        api_keys.insert(0, single_key)
        
     if api_keys:
        self.openrouter_api_key = api_keys[0]  # Use first available key
        logger.info(f"✅ QA Engine using OpenRouter key (found {len(api_keys)} keys)")
     else:
        logger.warning("❌ No OpenRouter API keys found for QA Engine")
        self.openrouter_api_key = None
    
     # Initialize smart routing components
     if self.enable_smart_routing:
        self.query_router = IntelligentQueryRouter()
        self.key_manager = MultiKeyManager()
        print("✅ Smart model routing enabled!")
     else:
        self.query_router = None
        self.key_manager = None
    
     # Initialize core components
     self.llm = None
     self.chat_llm = None
     self.memory = None
     self.tools = []
     self.agent = None
     self.conversation_history = []
    
     # Repository-aware components
     self.repo_context = RepositoryContextManager()
     self.intent_detector = UltimateFileRewriteDetector()
     self.coding_agent = None
    
     # Enhanced system prompt
     self.system_prompt = """You are NOVA, an advanced AI assistant with comprehensive repository analysis and development capabilities.
    
Your Core Abilities:
1. Repository Analysis - Scan and analyze entire codebases for issues, patterns, and improvements
2. Debug & Fix - Identify bugs, errors, and problems in specific files with detailed solutions
3. Code Generation - Create complete, production-ready code files (1000+ lines)
4. Detailed Explanations - Provide comprehensive explanations of code, changes, and solutions
5. File Modifications - Rewrite, update, and improve existing files with enhancements
6. Smart Responses - Detect user intent and provide appropriate response type

Response Guidelines:
- For "whole/complete file" requests: Provide ONLY clean code
- For analysis/debugging requests: Provide detailed explanations + solutions + code
- For general questions: Provide comprehensive explanations with examples
- Always use repository context when available
- Maintain code quality, best practices, and proper documentation
- Include error handling and security considerations
- Support both English and Hindi/Hinglish queries

Be thorough, accurate, and user-friendly in all responses."""
    
     # Enhanced performance tracking
     self.performance_metrics = {
        'total_queries': 0,
        'repo_queries': 0,
        'file_rewrites': 0,
        'file_analysis': 0,
        'repo_analysis': 0,
        'avg_response_time': 0.0,
        'success_rate': 1.0,
        'explanation_requests': 0,
        'code_only_requests': 0,
        'smart_routing_queries': 0,
        'routing_accuracy': 1.0,
        'model_usage_stats': defaultdict(int)
     }
    
     # Initialize logger
     logging.basicConfig(level=logging.INFO)
     self.logger = logging.getLogger(__name__)
    
     # Initialize all components
     self._initialize_llm()
     if self.enable_memory and self.llm:
        self._initialize_memory()
     if self.enable_tools:
        self._initialize_tools()
     if LANGCHAIN_AVAILABLE and self.chat_llm and self.tools:
        self._initialize_agent()
    
     print("🚀 ULTIMATE QA Engine initialized with Smart Model Routing + All Original Features!")

    def _initialize_llm(self):
        """PRESERVED + ENHANCED: Initialize ChatOpenAI with enhanced configuration"""
        try:
            api_key = self.openrouter_api_key
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment variables")

            # Initialize enhanced callback handler (PRESERVED + ENHANCED)
            self.callback_handler = EnhancedRepoAwareCallbackHandler()

            # ✅ PRESERVED: Enhanced ChatOpenAI configuration
            self.chat_llm = ChatOpenAI(
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                request_timeout=180,  # PRESERVED: Long timeout for comprehensive responses
                max_retries=3,
                callbacks=[self.callback_handler]
            )

            # Keep both references for compatibility (PRESERVED)
            self.llm = self.chat_llm

            print(f"✅ Enhanced LLM initialized: {self.model_name} (max_tokens: {self.max_tokens})")
            
            # 🔥 NEW: Show smart routing status
            if self.enable_smart_routing:
                available_providers = list(self.key_manager.provider_keys.keys()) if self.key_manager else []
                print(f"🧠 Smart routing ready with: {available_providers}")
                
        except Exception as e:
            print(f"❌ LLM initialization failed: {e}")
            self.llm = None
            self.chat_llm = None

    def _initialize_memory(self):
        """PRESERVED: Initialize enhanced conversation memory"""
        if not self.llm:
            return

        try:
            if self.memory_type == "summary":
                self.memory = ConversationSummaryBufferMemory(
                    llm=self.llm,
                    max_token_limit=self.max_memory_tokens,
                    return_messages=True,
                    memory_key="history"
                )
            else:
                self.memory = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="history"
                )

            print(f"✅ Enhanced memory initialized: {self.memory_type}")
        except Exception as e:
            print(f"❌ Memory initialization failed: {e}")
            self.memory = None

    def _initialize_tools(self):
        """PRESERVED: Initialize enhanced tools with repository awareness"""
        self.tools = []

        # ✅ PRESERVED: Repository search tool
        self.tools.append(Tool(
            name="RepositorySearch",
            func=self._search_repository,
            description="Search the repository for relevant code, files, and information. Use when user asks about repository content."
        ))

        # ✅ PRESERVED: File content retrieval tool
        self.tools.append(Tool(
            name="GetFileContent",
            func=self._get_file_content,
            description="Get the complete content of a specific file from the repository. Use when user asks about a specific file."
        ))

        # ✅ PRESERVED: Repository analysis tool
        self.tools.append(Tool(
            name="AnalyzeRepository",
            func=self._analyze_repository,
            description="Analyze the entire repository for issues, bugs, and problems. Use when user asks to check or analyze the repository."
        ))

        # ✅ PRESERVED: File debugging tool
        self.tools.append(Tool(
            name="DebugFile",
            func=self._debug_file,
            description="Debug and analyze issues in a specific file. Use when user asks about problems in a particular file."
        ))

        # PRESERVED: Standard tools
        self.tools.append(Tool(
            name="Calculator",
            func=self._calculate,
            description="Useful for mathematical calculations. Input should be a mathematical expression."
        ))

        self.tools.append(Tool(
            name="DateTime",
            func=self._get_datetime_info,
            description="Get current date, time, or day of week information."
        ))

        # PRESERVED: Search tool (DuckDuckGo - no API key required)
        try:
            search = DuckDuckGoSearchRun()
            self.tools.append(Tool(
                name="WebSearch",
                func=search.run,
                description="Search the web for current information. Use when you need recent data or facts."
            ))
            print("✅ Web search tool initialized")
        except Exception as e:
            print(f"⚠️ Web search tool failed: {e}")

        # PRESERVED: Wikipedia tool
        try:
            wikipedia = WikipediaAPIWrapper()
            self.tools.append(Tool(
                name="Wikipedia",
                func=wikipedia.run,
                description="Search Wikipedia for encyclopedic information about people, places, events, etc."
            ))
            print("✅ Wikipedia tool initialized")
        except Exception as e:
            print(f"⚠️ Wikipedia tool failed: {e}")

        print(f"✅ {len(self.tools)} enhanced tools initialized")

    def _initialize_agent(self):
        """PRESERVED: Initialize the agent with enhanced tools"""
        if not self.llm or not self.tools:
            return

        try:
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.chat_llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=6,  # PRESERVED: More iterations for complex analysis
                early_stopping_method="generate"
            )

            print("✅ Enhanced agent initialized with comprehensive repository tools")
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            self.agent = None

    # ========== PRESERVED: ENHANCED TOOL IMPLEMENTATIONS ==========

    def _search_repository(self, query: str) -> str:
        """PRESERVED: Enhanced repository search with better formatting"""
        try:
            results = self.repo_context.search_repository(query, max_results=8)
            if not results:
                return "No repository context available or no results found."

            formatted_results = f"🔍 Repository Search Results for '{query}':\n\n"
            for i, result in enumerate(results, 1):
                formatted_results += f"**Result {i} - {result['filename']}:**\n"
                formatted_results += f"{result['content'][:300]}...\n\n"

            return formatted_results
        except Exception as e:
            return f"Repository search error: {e}"

    def _get_file_content(self, filename: str) -> str:
        """PRESERVED: Enhanced file content retrieval"""
        try:
            content = self.repo_context.get_file_content(filename)
            if not content:
                return f"File '{filename}' not found in repository or no content available."

            return f"📁 Complete Content of {filename}:\n\n{content}"
        except Exception as e:
            return f"File retrieval error: {e}"

    def _analyze_repository(self, query: str = "") -> str:
        """PRESERVED: Comprehensive repository analysis"""
        try:
            analysis = self.repo_context.analyze_repository_issues()
            if "error" in analysis:
                return f"Repository analysis failed: {analysis['error']}"

            result = f"🔍 Repository Analysis Report:\n\n"
            result += f"**Repository:** {analysis.get('repo_name', 'Unknown')}\n"
            result += f"**Files Scanned:** {analysis.get('files_scanned', 0)}\n"
            result += f"**Issues Found:** {analysis.get('total_issues', 0)}\n\n"

            if analysis.get('issues_by_type'):
                result += "**Detailed Issues:**\n"
                for issue_type, issues in analysis['issues_by_type'].items():
                    result += f"\n• **{issue_type.upper()}:**\n"
                    for issue in issues[:2]:  # Show top 2 issues per type
                        result += f"  - File: {issue['filename']}\n"
                        result += f"    Issue: {issue['content'][:100]}...\n"
            else:
                result += "✅ No major issues detected in the repository!"

            return result
        except Exception as e:
            return f"Repository analysis error: {e}"

    def _debug_file(self, filename: str) -> str:
        """PRESERVED: Debug and analyze specific file"""
        try:
            content = self.repo_context.get_file_content(filename)
            if not content:
                return f"File '{filename}' not found for debugging."

            result = f"🐛 Debug Analysis for {filename}:\n\n"
            result += f"**File Size:** {len(content)} characters\n"
            result += f"**Lines:** {len(content.splitlines())}\n\n"

            # PRESERVED: Look for common issues
            issues = []
            if "TODO" in content:
                issues.append("Contains TODO comments that need attention")
            if "FIXME" in content:
                issues.append("Contains FIXME comments indicating known issues")
            if "print(" in content and filename.endswith('.py'):
                issues.append("Contains print statements (consider using logging)")
            if "console.log" in content and filename.endswith('.js'):
                issues.append("Contains console.log statements (consider removing for production)")

            if issues:
                result += "**Issues Found:**\n"
                for issue in issues:
                    result += f"• {issue}\n"
            else:
                result += "✅ No obvious issues detected in this file.\n"

            result += f"\n**File Content Preview:**\n{content[:500]}..."

            return result
        except Exception as e:
            return f"File debugging error: {e}"

    # PRESERVED: Standard tool implementations (unchanged)
    def _calculate(self, expression: str) -> str:
        """PRESERVED: Calculator tool implementation"""
        try:
            allowed_chars = set('0123456789+-*/().% ')
            if not all(c in allowed_chars for c in expression.replace(' ', '')):
                return "Error: Invalid characters in expression"
            
            result = eval(expression)
            return f"The result of {expression} is {result}"
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: Unable to calculate '{expression}' - {str(e)}"

    def _get_datetime_info(self, query: str = "") -> str:
        """PRESERVED: Get current date/time information"""
        now = datetime.now()
        query_lower = query.lower()
        
        if "date" in query_lower:
            return f"Today's date is {now.strftime('%A, %B %d, %Y')}"
        elif "time" in query_lower:
            return f"Current time is {now.strftime('%I:%M:%S %p')}"
        elif "day" in query_lower:
            return f"Today is {now.strftime('%A')}"
        else:
            return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M:%S %p')}"

    # ========== 🔥 NEW: Smart Model Integration Methods ==========

    async def _get_smart_ai_response(self, model_config: Dict, prompt: str) -> Optional[str]:
        """🔥 NEW: Get response from smart selected AI model"""
        if not self.enable_smart_routing or not self.key_manager:
            return None
            
        provider = model_config["provider"]
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Get API key
                key_data = self.key_manager.get_best_key(provider)
                if not key_data:
                    logger.error(f"❌ No available keys for {provider}")
                    continue
                
                api_key, key_idx = key_data
                
                # Prepare request
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "NOVA-Ultimate-QA/3.0"
                }
                
                # Provider-specific headers
                if provider == "OPENROUTER":
                    headers.update({
                        "HTTP-Referer": "https://nova-ultimate-qa.app",
                        "X-Title": "NOVA Ultimate QA Engine"
                    })
                
                # Prepare payload
                temperature = 0.3 if "coding" in model_config.get("specialty", "").lower() else 0.7
                
                payload = {
                    "model": model_config["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": min(model_config.get("context", 4000) // 2, self.max_tokens),
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stream": False
                }
                
                # Make request
                timeout = 40 if model_config.get("speed", 0) < 8 else 25
                response = requests.post(
                    model_config["url"],
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                
                # Handle response
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"].strip()
                        self.key_manager.record_usage(provider, key_idx, True)
                        return content
                
                elif response.status_code == 429:
                    self.key_manager.record_usage(provider, key_idx, False, 'rate_limit')
                    logger.warning(f"⚠️ Rate limited for {provider}, trying next key...")
                    continue
                    
                elif response.status_code in [402, 403]:
                    self.key_manager.record_usage(provider, key_idx, False, 'quota')  
                    logger.warning(f"⚠️ Quota exceeded for {provider}, trying next key...")
                    continue
                
                else:
                    self.key_manager.record_usage(provider, key_idx, False)
                    logger.error(f"❌ API error {response.status_code}: {response.text[:200]}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"⚠️ Request timeout for {provider}, attempt {attempt + 1}")
                continue
                
            except Exception as e:
                logger.error(f"❌ Request failed for {provider}: {e}")
                if key_data:
                    self.key_manager.record_usage(provider, key_data[1], False)
                continue
        
        return None

    # ========== PRESERVED + ENHANCED: ULTIMATE ASK METHOD ==========

    def set_repository_context(self, db_path: str = "./chroma_db", repo_url: str = None) -> bool:
     """Enhanced repository context with state sharing"""
     try:
        print(f"🔍 Setting repository context: {db_path}")
        
        # Try shared state first
        try:
            import sys
            sys.path.append('.')  # Add current directory to path
            from shared_state import repo_state
            
            saved_repo = repo_state.get_repository()
            if saved_repo.get('db_path') and os.path.exists(saved_repo['db_path']):
                db_path = saved_repo['db_path']
                repo_url = saved_repo.get('active_repo', repo_url)
                print(f"✅ Using shared state: {db_path}")
        except ImportError:
            print("⚠️ Shared state not available, using provided path")
        
        # Search for database if path doesn't exist
        if not os.path.exists(db_path):
            found_path = self._find_repository_database()
            if found_path:
                db_path = found_path
                print(f"✅ Found database at: {db_path}")
            else:
                print(f"❌ No database found at {db_path}")
                return False
        
        # Load the database
        if VECTOR_SEARCH_AVAILABLE and self.repo_context.embeddings:
            from langchain_community.vectorstores import Chroma
            
            self.repo_context.vector_store = Chroma(
                persist_directory=db_path,
                embedding_function=self.repo_context.embeddings
            )
            
            self.repo_context.retriever = self.repo_context.vector_store.as_retriever(
                search_kwargs={"k": 8}
            )
            self.repo_context.active_repo = repo_url
            
            # Verify connection
            try:
                test_results = self.repo_context.retriever.get_relevant_documents("test")
                print(f"✅ Repository context set successfully: {len(test_results)} docs accessible")
                return True
            except Exception as e:
                print(f"❌ Database verification failed: {e}")
                return False
        else:
            print("❌ Vector search not available")
            return False
            
     except Exception as e:
        print(f"❌ Repository context failed: {e}")
        return False
    
    def debug_repository_status(self):
     """🔥 Debug repository loading status"""
     print(f"\n🔍 ===== REPOSITORY DEBUG STATUS =====")
     print(f"repo_context exists: {hasattr(self, 'repo_context')}")
    
     if hasattr(self, 'repo_context'):
        print(f"active_repo: {self.repo_context.active_repo}")
        print(f"retriever: {type(self.repo_context.retriever)}")
        print(f"vector_store: {type(self.repo_context.vector_store)}")
        print(f"file_mappings: {len(self.repo_context.file_mappings)} files")
        
        if self.repo_context.retriever:
            try:
                test_docs = self.repo_context.retriever.get_relevant_documents("test")
                print(f"retriever test: {len(test_docs)} docs found")
                if test_docs:
                    print(f"sample doc: {test_docs[0].page_content[:100]}...")
            except Exception as e:
                print(f"retriever error: {e}")
     print(f"===== END REPOSITORY DEBUG =====\n")
     
    def _find_repository_database(self):
     """Search for repository databases"""
     search_locations = [
        "./chroma_db",
        "./chroma_repos", 
        "."
     ]
    
     for location in search_locations:
        if os.path.exists(location):
            # Check if it's a database directory
            if os.path.isdir(location):
                if any(os.path.exists(os.path.join(location, f)) 
                      for f in ["chroma.sqlite3", "index"]):
                    return location
                
                # Check subdirectories
                try:
                    for item in os.listdir(location):
                        item_path = os.path.join(location, item)
                        if (os.path.isdir(item_path) and 
                            any(os.path.exists(os.path.join(item_path, f)) 
                                for f in ["chroma.sqlite3", "index"])):
                            return item_path
                except:
                    pass
    
     return None

    async def ask(self, question: str, context: Optional[str] = None, force_long_response: bool = False) -> Dict[str, Any]:
     """
     🚀 ULTIMATE ask method with Smart Model Routing + Repository Vector Search + Enhanced Debugging
     """
     start_time = time.time()
     self.performance_metrics['total_queries'] += 1

     if not question.strip():
        return {
            "response": "Please ask me a question.",
            "error": "Empty question",
            "timestamp": datetime.now().isoformat(),
            "model_used": self.model_name,
            "repo_context_used": False,
            "smart_routing_used": False,
            "vector_search_used": False
        }

     try:
        # 🔥 ENHANCED DEBUG PRINTS - Repository Vector Search Check
        print(f"\n🔍 ===== REPOSITORY DEBUG ANALYSIS =====")
        print(f"🔍 Question: '{question}'")
        print(f"🔍 repo_context exists: {hasattr(self, 'repo_context')}")
        
        if hasattr(self, 'repo_context'):
            print(f"🔍 Active repo: {self.repo_context.active_repo}")
            print(f"🔍 Active repo (bool): {self.repo_context.active_repo is not None}")
            print(f"🔍 Retriever type: {type(self.repo_context.retriever) if self.repo_context.retriever else None}")
            print(f"🔍 Retriever available: {self.repo_context.retriever is not None}")
        else:
            print(f"❌ repo_context attribute not found!")
            
        vector_search_used = False
        vector_search_context = ""
        
        # 🔥 SIMPLIFIED: More aggressive repository detection
        repo_keywords = ['file', 'files', 'repository', 'repo', 'code', 'how many', 'what is',
                        'what does', 'structure', 'contains', 'written in', 'language',
                        'dependencies', 'main', 'functions', 'classes', 'imports', 'project',
                        'codebase', 'application', 'system', 'module', 'component', 'package',
                        'directory', 'folder', 'source', 'implementation', 'architecture']
        
        is_repo_query = any(keyword in question.lower() for keyword in repo_keywords)
        print(f"🔍 Is repo query: {is_repo_query}")
        print(f"🔍 Matching keywords: {[kw for kw in repo_keywords if kw in question.lower()]}")
        
        # 🔥 RELAXED CONDITIONS: Only need retriever + repo query (removed active_repo requirement)
        condition_1 = hasattr(self, 'repo_context') and self.repo_context.retriever is not None
        condition_2 = is_repo_query
        
        print(f"🔍 Vector search conditions:")
        print(f"   • Has retriever: {condition_1}")
        print(f"   • Is repo query: {condition_2}")
        print(f"   • SIMPLIFIED CONDITIONS MET: {condition_1 and condition_2}")
        
        # 🔥 SIMPLIFIED TRIGGER: Only need retriever + repo query
        if (condition_1 and condition_2):
            print(f"\n🔥 ===== VECTOR SEARCH ACTIVATED! =====")
            
            try:
                # Use retriever to search repository for relevant documents
                print(f"🔥 Calling get_relevant_documents...")
                repo_docs = self.repo_context.retriever.get_relevant_documents(question)
                print(f"🔥 Retrieved {len(repo_docs)} documents")
                
                if repo_docs:
                    print(f"🔥 Building vector search context from {len(repo_docs)} documents...")
                    
                    # Build enhanced context from repository documents
                    vector_search_context = f"\n\n🔥 REPOSITORY CONTEXT:\n"
                    
                    for i, doc in enumerate(repo_docs[:5]):  # Limit to top 5 results
                        content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                        
                        # Add file path if available in metadata
                        file_info = ""
                        if hasattr(doc, 'metadata') and doc.metadata:
                            source = doc.metadata.get('source', '')
                            if source:
                                file_info = f"File: {source}\n"
                        
                        vector_search_context += f"\n--- Document {i+1} ---\n{file_info}{content_preview}\n"
                        print(f"🔥 Document {i+1}: {len(doc.page_content)} chars")
                    
                    vector_search_used = True
                    self.performance_metrics['vector_searches'] = self.performance_metrics.get('vector_searches', 0) + 1
                    
                    # 🔥 FORCE REPOSITORY CONTEXT: Create repository-aware enhanced question
                    original_question = question
                    question = f"""Based on the repository context provided below, answer this question: {question}

{vector_search_context}

IMPORTANT: Use the repository data above to give specific, accurate answers. If the question asks about file counts, structure, dependencies, or specific content, reference the actual repository files shown above. Do not give generic programming advice - give repository-specific answers."""
                    
                    print(f"🔥 Enhanced question created:")
                    print(f"   • Original length: {len(original_question)} chars")
                    print(f"   • Enhanced length: {len(question)} chars")
                    print(f"   • Context length: {len(vector_search_context)} chars")
                    
                else:
                    print(f"⚠️ No documents found in vector search - will try traditional repo context")
                    
            except Exception as e:
                print(f"❌ Vector search error: {e}")
                print(f"❌ Error type: {type(e).__name__}")
                import traceback
                print(f"❌ Traceback: {traceback.format_exc()}")
                vector_search_used = False
        else:
            print(f"\n⚠️ Vector search conditions not met - using regular processing")
            if not condition_1:
                print(f"   ❌ Missing: retriever not available")
            if not condition_2:
                print(f"   ❌ Missing: not detected as repository query")

        print(f"🔍 ===== END REPOSITORY DEBUG =====\n")
        
        # ✅ Step 1: Enhanced Intent Detection (PRESERVED)
        intent = self.intent_detector.detect_intent(question)

        # 🔥 Step 2: Smart Model Selection (NEW)
        selected_model = None
        routing_confidence = 0.0
        smart_routing_used = False
        
        if self.enable_smart_routing and self.query_router:
            query_type, routing_confidence, selected_model = self.query_router.route_query(question)
            smart_routing_used = True
            self.performance_metrics['smart_routing_queries'] += 1
            self.performance_metrics['model_usage_stats'][selected_model['name']] += 1
            
            if hasattr(self.callback_handler, 'set_routing_info'):
                self.callback_handler.set_routing_info(
                    selected_model['name'], 
                    routing_confidence,
                    {'query_type': query_type, 'model_config': selected_model}
                )
            
            print(f"🎯 Smart routing: {query_type} → {selected_model['name']} (confidence: {routing_confidence:.1%})")

        # Update performance metrics based on intent (PRESERVED)
        if intent['intent_type'] == 'file_rewrite':
            self.performance_metrics['file_rewrites'] += 1
            if intent.get('wants_code_only'):
                self.performance_metrics['code_only_requests'] += 1
            else:
                self.performance_metrics['explanation_requests'] += 1
        elif intent['intent_type'] == 'file_analysis':
            self.performance_metrics['file_analysis'] += 1
            self.performance_metrics['explanation_requests'] += 1
        elif intent['intent_type'] == 'repo_analysis':
            self.performance_metrics['repo_analysis'] += 1
            self.performance_metrics['explanation_requests'] += 1

        # ✅ Step 3: Prepare Repository Context (PRESERVED - but now enhanced with vector search)
        repo_context = ""
        repo_context_used = vector_search_used  # Vector search counts as repo context usage

        # 🔥 FALLBACK: Only do traditional repo context if vector search wasn't used
        if not vector_search_used and hasattr(self, 'repo_context') and self.repo_context.active_repo:
            print(f"🔍 Using traditional repo context as fallback...")
            
            if intent['intent_type'] == 'file_rewrite' and intent['filename']:
                # Get specific file content for rewrite
                file_content = self.repo_context.get_file_content(intent['filename'])
                if file_content:
                    repo_context = f"\n\nREPOSITORY_CONTEXT:\nFile: {intent['filename']}\nContent:\n{file_content}"
                    repo_context_used = True

            elif intent['intent_type'] == 'file_analysis' and intent['filename']:
                # Get specific file content for analysis
                file_content = self.repo_context.get_file_content(intent['filename'])
                if file_content:
                    repo_context = f"\n\nFILE_DEBUG:\nFile: {intent['filename']}\nContent:\n{file_content}"
                    repo_context_used = True

            elif intent['intent_type'] == 'repo_analysis':
                # Get repository overview for analysis
                search_results = self.repo_context.search_repository("overview structure files", max_results=5)
                if search_results:
                    repo_context = "\n\nREPOSITORY_ANALYSIS:\n"
                    for result in search_results:
                        repo_context += f"File: {result['filename']}\n{result['content'][:300]}...\n\n"
                    repo_context_used = True

            else:
                # Search repository for relevant context
                search_results = self.repo_context.search_repository(question, max_results=5)
                if search_results:
                    repo_context = "\n\nREPOSITORY_CONTEXT:\n"
                    for result in search_results:
                        repo_context += f"File: {result['filename']}\n{result['content'][:300]}...\n\n"
                    repo_context_used = True

        if repo_context_used:
            self.performance_metrics['repo_queries'] += 1

        # ✅ Step 4: Build Enhanced Prompts Based on Intent Type (PRESERVED)
        enhanced_prompt = ""

        # 🔥 NEW: If vector search was used, use simpler prompt since question already has context
        if vector_search_used:
            enhanced_prompt = question  # Question already contains repository context
            print(f"🔥 Using vector search enhanced question as prompt")
        else:
            # Use original complex prompts for non-vector-search queries
            print(f"🔍 Using traditional prompt for intent: {intent['intent_type']}")
            
            # ... rest of your existing prompt building logic remains the same ...
            if intent['intent_type'] == 'repo_analysis':
                enhanced_prompt = f"""🔍 REPOSITORY_ANALYSIS REQUEST: {question}

You are analyzing an entire repository for issues, problems, and improvements.

ANALYSIS INSTRUCTIONS:
1. Scan the repository context for potential issues
2. Look for common problems like:
   - Code quality issues
   - Security vulnerabilities  
   - Performance problems
   - Missing error handling
   - TODO/FIXME comments
   - Deprecated code
   - Unused imports/variables
3. Provide detailed explanations of each issue found
4. Suggest specific improvements and fixes
5. Prioritize issues by severity
6. Give an overall repository health assessment

{repo_context}

Provide a comprehensive repository analysis with detailed findings and recommendations:"""

            elif intent['intent_type'] == 'file_analysis':
                enhanced_prompt = f"""🐛 FILE_DEBUG REQUEST: {question}

You are debugging and analyzing a specific file for issues and problems.

FILE ANALYSIS INSTRUCTIONS:
1. Thoroughly examine the file content
2. Identify specific issues, bugs, or problems
3. Explain each issue in detail
4. Provide step-by-step solutions
5. Suggest improvements and best practices
6. Include code examples where helpful
7. Address the user's specific concerns

{repo_context}

Provide detailed file analysis with explanations and solutions:"""

            elif intent['intent_type'] == 'file_rewrite':
                if intent['response_type'] == 'code_only':
                    enhanced_prompt = f"""CODE_ONLY FILE REWRITE: {question}

CRITICAL INSTRUCTIONS:
1. Generate ONLY the complete file content
2. NO explanations, NO comments about changes
3. Just clean, working, production-ready code
4. Implement all requested changes/improvements
5. Maintain existing functionality while adding new features
6. Follow best coding practices and conventions
7. Include proper error handling
8. Return ONLY the file content, nothing else

Original File Context:
{repo_context}

Generate ONLY the complete {intent['filename']} file content:"""

                else:
                    enhanced_prompt = f"""EXPLANATION FILE REWRITE: {question}

COMPREHENSIVE REWRITE INSTRUCTIONS:
1. First, provide a detailed explanation of what you're changing and why
2. List all improvements and new features being added
3. Explain any issues being fixed
4. Then provide the complete updated file content
5. Use clear sections: EXPLANATION first, then UPDATED FILE
6. Make the explanation educational and detailed

Original File Context:
{repo_context}

Provide detailed explanation of changes, then the complete updated {intent['filename']} file:"""

            else:
                # General question (PRESERVED)
                enhanced_prompt = f"""GENERAL QUESTION: {question}

You are answering a general question about the repository or code.

RESPONSE INSTRUCTIONS:
1. Provide comprehensive, detailed explanations
2. Use repository context when available
3. Include examples and code snippets where helpful
4. Be educational and thorough
5. Address all aspects of the user's question

{repo_context}

{context if context else ''}

Provide a comprehensive answer to the user's question:"""

        # ... rest of your method remains exactly the same ...
        # (Steps 5-6: Generate Response and Post-Processing)
        
        # ✅ Step 5: Generate Response Using Appropriate Method (PRESERVED + ENHANCED)
        response = None
        model_used = self.model_name

        # 🔥 NEW: Try smart routing first if available
        if smart_routing_used and selected_model:
            try:
                response = await self._get_smart_ai_response(selected_model, enhanced_prompt)
                if response:
                    model_used = selected_model['name']
                    print(f"✅ Response generated via smart routing: {model_used}")
                else:
                    print(f"⚠️ Smart routing failed, falling back to original method")
            except Exception as e:
                print(f"⚠️ Smart routing error: {e}, falling back to original method")

        # PRESERVED: Fallback to original method
        if not response:
            print(f"🔍 Using fallback response method...")
            if hasattr(self, 'agent') and self.agent and repo_context_used:
                print(f"🔍 Using agent for complex repository query")
                response = self.agent.run(enhanced_prompt)
            elif hasattr(self, 'chat_llm') and self.chat_llm and hasattr(self, 'memory') and self.memory:
                print(f"🔍 Using chat model with memory")
                conversation = ConversationChain(
                    llm=self.chat_llm,
                    memory=self.memory,
                    verbose=False
                )
                response = conversation.predict(input=enhanced_prompt)
            elif hasattr(self, 'chat_llm') and self.chat_llm:
                print(f"🔍 Using basic chat model")
                messages = [
                    SystemMessage(content=getattr(self, 'system_prompt', 'You are a helpful AI assistant.')),
                    HumanMessage(content=enhanced_prompt)
                ]
                result = self.chat_llm.invoke(messages)
                response = result.content
            else:
                print(f"❌ No response method available")
                response = "I'm sorry, but I'm currently unable to process your question due to configuration issues."

        print(f"🔍 Final response generated: {len(response) if response else 0} chars")

        # ✅ Step 6: Smart Post-Processing Based on Intent (PRESERVED)
        should_clean_response = (
            intent['intent_type'] == 'file_rewrite' and 
            intent.get('response_type') == 'code_only'
        )

        if should_clean_response and response:
            # Clean response to extract only file content (PRESERVED)
            response = self._clean_file_response(response, intent.get('filename', ''))

        # Calculate performance metrics (PRESERVED + ENHANCED)
        response_time = time.time() - start_time
        self.performance_metrics['avg_response_time'] = (
            (self.performance_metrics['avg_response_time'] * (self.performance_metrics['total_queries'] - 1) + response_time) /
            self.performance_metrics['total_queries']
        )
        
        # Update routing accuracy if smart routing was used
        if smart_routing_used and response and len(response.strip()) > 50:
            current_accuracy = self.performance_metrics['routing_accuracy']
            total_smart_queries = self.performance_metrics['smart_routing_queries']
            self.performance_metrics['routing_accuracy'] = (
                (current_accuracy * (total_smart_queries - 1) + 1.0) / total_smart_queries
            )

        # Store in conversation history (PRESERVED + ENHANCED)
        conversation_entry = {
            "question": question,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "response_time": response_time,
            "repo_context_used": repo_context_used,
            "intent": intent,
            "response_cleaned": should_clean_response,
            # 🔥 NEW: Smart routing info
            "smart_routing_used": smart_routing_used,
            "routing_confidence": routing_confidence,
            "selected_model": selected_model['name'] if selected_model else None,
            "model_used": model_used,
            # 🔥 NEW: Vector search info
            "vector_search_used": vector_search_used
        }
        self.conversation_history.append(conversation_entry)

        print(f"🔍 ===== FINAL RESULT SUMMARY =====")
        print(f"✅ Vector search used: {vector_search_used}")
        print(f"✅ Repo context used: {repo_context_used}")
        print(f"✅ Response length: {len(response) if response else 0} chars")
        print(f"✅ Model used: {model_used}")
        print(f"✅ Response time: {response_time:.2f}s")

        return {
            "response": response,
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "response_time": response_time,
            "model_used": model_used,
            "tools_available": getattr(self, 'tools', []) and len(self.tools),
            "memory_enabled": hasattr(self, 'memory') and self.memory is not None,
            "conversation_length": len(self.conversation_history),
            "repo_context_used": repo_context_used,
            "intent": intent,
            "response_type": intent.get('response_type', 'general'),
            "needs_explanation": intent.get('needs_explanation', True),
            "repo_summary": self.repo_context.get_repository_summary() if hasattr(self, 'repo_context') and self.repo_context.active_repo else None,
            # 🔥 NEW: Enhanced response metadata
            "smart_routing_used": smart_routing_used,
            "routing_confidence": routing_confidence,
            "selected_model_info": selected_model,
            "vector_search_used": vector_search_used,
            "performance_metrics": self.performance_metrics.copy()
        }

     except Exception as e:
        print(f"❌ CRITICAL ERROR in ask() method: {e}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        
        self.logger.error(f"Error processing question: {e}")
        return {
            "response": f"I encountered an error while processing your question: {str(e)}",
            "error": str(e),
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "model_used": self.model_name,
            "repo_context_used": False,
            "smart_routing_used": smart_routing_used if 'smart_routing_used' in locals() else False,
            "vector_search_used": False
        }

    def _clean_file_response(self, response: str, filename: str) -> str:
        """PRESERVED: Enhanced response cleaning to extract only file content"""
        try:
            # Remove markdown code blocks if present (PRESERVED)
            if "```" in response:
                parts = response.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # Code block content
                        # Remove language identifier if present (PRESERVED)
                        lines = part.split('\n')
                        if lines and lines[0].strip() in ['python', 'javascript', 'html', 'css', 'json', 'java', 'cpp', 'c', 'go', 'rust', 'php', 'ruby', 'swift', 'kotlin']:
                            lines = lines[1:]
                        clean_content = '\n'.join(lines)
                        
                        # Additional cleaning - remove any leading/trailing explanatory text (PRESERVED)
                        if len(clean_content.strip()) > 50:  # Ensure it's substantial code
                            return clean_content.strip()

            # If no code blocks found, look for substantial code content (PRESERVED)
            lines = response.split('\n')
            code_lines = []
            in_code_section = False

            for line in lines:
                # Skip explanatory text lines (PRESERVED)
                if any(keyword in line.lower() for keyword in ['here is', 'here\'s', 'explanation', 'changes made', 'i\'ve', 'this code', 'the code']):
                    continue
                
                # Look for code patterns (PRESERVED)
                if any(pattern in line for pattern in ['import ', 'def ', 'class ', 'function ', 'const ', 'var ', 'let ', '#!/']):
                    in_code_section = True
                
                if in_code_section:
                    code_lines.append(line)

            if code_lines and len('\n'.join(code_lines).strip()) > 50:
                return '\n'.join(code_lines).strip()

            # Fallback: return original response (PRESERVED)
            return response.strip()

        except Exception as e:
            print(f"⚠️ Response cleaning failed: {e}")
            return response

    # ========== PRESERVED + ENHANCED: UTILITY METHODS ==========

    def get_performance_metrics(self) -> Dict[str, Any]:
        """PRESERVED + ENHANCED: Get comprehensive performance metrics"""
        total_queries = max(1, self.performance_metrics['total_queries'])
        smart_queries = self.performance_metrics['smart_routing_queries']
        
        return {
            **self.performance_metrics,
            "repo_query_percentage": (self.performance_metrics['repo_queries'] / total_queries) * 100,
            "file_rewrite_percentage": (self.performance_metrics['file_rewrites'] / total_queries) * 100,
            "explanation_percentage": (self.performance_metrics['explanation_requests'] / total_queries) * 100,
            "avg_response_time_formatted": f"{self.performance_metrics['avg_response_time']:.2f}s",
            "tools_count": len(self.tools),
            "memory_type": self.memory_type,
            "max_tokens": self.max_tokens,
            "repository_active": self.repo_context.active_repo is not None,
            # 🔥 NEW: Smart routing metrics
            "smart_routing_percentage": (smart_queries / total_queries) * 100 if smart_queries > 0 else 0,
            "routing_accuracy_percentage": self.performance_metrics['routing_accuracy'] * 100,
            "available_providers": list(self.key_manager.provider_keys.keys()) if self.key_manager else [],
            "total_capabilities": [
                "Repository Analysis",
                "File Debugging", 
                "Code Generation",
                "Explanation Mode",
                "Multi-language Support",
                "Vector Search",
                "Smart Intent Detection",
                "🔥 Smart Model Routing",
                "🔥 Multi-Key Management",
                "🔥 DeepSeek R1 Integration",
                "🔥 Cerebras Ultra-Fast",
            ]
        }

    def get_repository_stats(self) -> Dict[str, Any]:
        """PRESERVED + ENHANCED: Get comprehensive repository statistics"""
        if not self.repo_context.active_repo:
            return {"status": "No repository loaded"}

        return {
            "active_repo": self.repo_context.active_repo,
            "file_mappings": len(self.repo_context.file_mappings),
            "vector_store_available": self.repo_context.vector_store is not None,
            "retriever_available": self.repo_context.retriever is not None,
            "summary": self.repo_context.get_repository_summary(),
            # 🔥 NEW: Enhanced repository stats
            "repo_stats": self.repo_context.repo_stats,
            "analysis_capabilities": [
                "Issue Detection",
                "Code Quality Review", 
                "Security Scanning",
                "Performance Analysis",
                "File-specific Debugging",
                "Repository Overview",
                "🔥 Smart Model Selection",
                "🔥 Intelligent Query Routing"
            ]
        }

    def get_smart_routing_stats(self) -> Dict[str, Any]:
        """🔥 NEW: Get smart routing statistics"""
        if not self.enable_smart_routing:
            return {"status": "Smart routing disabled"}
        
        return {
            "enabled": self.enable_smart_routing,
            "total_smart_queries": self.performance_metrics['smart_routing_queries'],
            "routing_accuracy": f"{self.performance_metrics['routing_accuracy']:.1%}",
            "model_usage_distribution": dict(self.performance_metrics['model_usage_stats']),
            "available_providers": list(self.key_manager.provider_keys.keys()) if self.key_manager else [],
            "provider_key_counts": {
                provider: len(keys) 
                for provider, keys in self.key_manager.provider_keys.items()
            } if self.key_manager else {},
            "smart_models_available": [
                "DeepSeek R1 - Best Free Coder",
                "Cerebras Lightning - 2600 tokens/sec", 
                "Groq Ultra-Fast - 1800+ tokens/sec",
                "WizardLM Creative Pro",
                "DeepSeek Chat V3"
            ]
        }

    # ========== PRESERVED: ALL ORIGINAL METHODS ==========

    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict]:
        """PRESERVED: Get conversation history"""
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history

    def clear_conversation_history(self):
        """PRESERVED: Clear conversation history"""
        self.conversation_history.clear()
        if self.memory:
            self.memory.clear()
        print("🧹 Conversation history cleared")

    def save_conversation(self, filename: str):
        """PRESERVED: Save conversation history to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            print(f"💾 Conversation saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save conversation: {e}")

    def load_conversation(self, filename: str):
        """PRESERVED: Load conversation history from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            print(f"📂 Conversation loaded from {filename}")
        except Exception as e:
            print(f"❌ Failed to load conversation: {e}")

    def get_memory_summary(self) -> str:
        """PRESERVED: Get a summary of the conversation memory"""
        if not self.memory:
            return "Memory not enabled"
        
        try:
            if hasattr(self.memory, 'buffer'):
                return f"Memory contains {len(self.memory.buffer)} messages"
            else:
                return "Memory is active but details unavailable"
        except Exception as e:
            return f"Memory status unknown: {e}"

    def set_system_prompt(self, prompt: str):
        """PRESERVED: Set a custom system prompt for the AI"""
        self.system_prompt = prompt
        print(f"🎯 System prompt updated")

    def get_model_info(self) -> Dict[str, Any]:
        """PRESERVED + ENHANCED: Get comprehensive model configuration information"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "memory_enabled": self.memory is not None,
            "memory_type": self.memory_type,
            "tools_count": len(self.tools),
            "agent_enabled": self.agent is not None,
            "conversation_count": len(self.conversation_history),
            "repository_context": self.repo_context.active_repo is not None,
            "performance_metrics": self.performance_metrics,
            # 🔥 NEW: Smart routing info
            "smart_routing_enabled": self.enable_smart_routing,
            "available_models": self.query_router.models if self.query_router else None,
            "enhanced_features": [
                "Smart Intent Detection",
                "Code-only vs Explanation Mode", 
                "Repository Analysis",
                "File Debugging",
                "Multi-language Support",
                "Vector Search Integration", 
                "Comprehensive Error Handling",
                "🔥 Intelligent Model Routing",
                "🔥 Multi-Provider Key Management",
                "🔥 Ultra-Fast Model Selection",
                "🔥 Confidence-Based Routing"
            ]
        }

# ========== PRESERVED: FACTORY FUNCTIONS ==========

def create_qa_engine(simple: bool = False, **kwargs) -> Union[UltimateQAEngine, 'SimpleQAEngine']:
    """
    PRESERVED: Factory function to create QA engine

    Args:
        simple: If True, creates SimpleQAEngine, otherwise UltimateQAEngine
        **kwargs: Arguments passed to UltimateQAEngine

    Returns:
        QA Engine instance
    """
    if simple:
        return SimpleQAEngine(kwargs.get('api_key'))
    else:
        return UltimateQAEngine(**kwargs)

class EnhancedQAEngine(UltimateQAEngine):
    """PRESERVED: Alias for backward compatibility"""
    pass

# ========== PRESERVED: Simple QA Engine ==========

class SimpleQAEngine:
    """PRESERVED: Simplified QA Engine for basic use cases"""
    
    def __init__(self, api_key: Optional[str] = None):
        # ✅ FIXED: Multi-key GROQ API handling
        # Try multi-key system first
        api_keys = []
        for i in range(1, 11):
            key = os.getenv(f"GROQ_API_KEY_{i}")
            if key:
                api_keys.append(key)
                
        # Fallback to single key
        single_key = os.getenv('GROQ_API_KEY')
        if single_key and single_key not in api_keys:
            api_keys.insert(0, single_key)
            
        if api_keys:
            self.api_key = api_key or api_keys[0]  # Use provided key or first available
            print(f"✅ SimpleQA using GROQ key (found {len(api_keys)} keys)")
        else:
            print("❌ No GROQ API keys found for SimpleQA")
            self.api_key = api_key  # Use provided key if any
        
        self.conversation = []

    def ask(self, question: str) -> str:
        """PRESERVED: Simple question asking without advanced features"""
        if not self.api_key:
            return "GROQ API key not configured"

        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://your-site.com',
                'X-Title': 'NOVA Simple QA'
            }

            self.conversation.append({"role": "user", "content": question})

            data = {
                'model': 'llama-3.1-8b-instant',
                'messages': [
                    {"role": "system", "content": "You are NOVA, a helpful and intelligent AI assistant."},
                    *self.conversation[-10:]  # Keep last 10 messages
                ],
                'max_tokens': 700,  # PRESERVED: Increased for comprehensive responses
                'temperature': 0.7
            }

            response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                answer = response.json()['choices'][0]['message']['content']
                self.conversation.append({"role": "assistant", "content": answer})
                return answer
            else:
                return f"GROQ API Error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"Error: {str(e)}"


async def main():
    """🔥 Main function to test repository QA functionality"""
    print(f"\n🚀 ===== NOVA ULTIMATE QA ENGINE TEST =====")
    
    try:
        # Initialize QA engine
        qa_engine = UltimateQAEngine()
        
        # 🔥 FIXED: Use multi-key system for GROQ
        # Try to get GROQ keys
        groq_keys = []
        for i in range(1, 11):
            key = os.getenv(f"GROQ_API_KEY_{i}")
            if key:
                groq_keys.append(key)
        
        if groq_keys:
            qa_engine.api_key = groq_keys[0]  # Use first available GROQ key
            print(f"✅ Main using GROQ key (found {len(groq_keys)} keys)")
        else:
            fallback_key = os.getenv("GROQ_API_KEY")
            if fallback_key:
                qa_engine.api_key = fallback_key
                print("✅ Main using fallback GROQ key")
            else:
                print("❌ No GROQ keys found for Main")
        
        # 🔥 FORCE GROQ MODEL (FREE)
        qa_engine.model_name = "llama-3.1-8b-instant"  # Free GROQ model
        qa_engine.api_base_url = "https://api.groq.com/openai/v1"
        
        # Set repository context
        repo_loaded = qa_engine.set_repository_context("./chroma_db")
        print(f"Repository loaded: {repo_loaded}")
        
        # Debug repository status
        qa_engine.debug_repository_status()
        
        # Test with simple question
        result = await qa_engine.ask("What files are in this repository?")
        
        if result and result.get('response'):
            print(f"✅ Response: {result['response'][:200]}...")
            print(f"✅ Vector search used: {result.get('vector_search_used', False)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ========== ENTRY POINT ==========
if __name__ == "__main__":
    """🔥 Entry point for testing"""
    print(f"🚀 Starting NOVA Ultimate QA Engine...")
    asyncio.run(main())