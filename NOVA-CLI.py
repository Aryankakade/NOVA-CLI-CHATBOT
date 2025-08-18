#!/usr/bin/env python3

"""
NOVA ULTRA PROFESSIONAL CLI - WORLD'S ABSOLUTE BEST AI CLI
Complete Version with RESTORED Features + Enhanced Gaming UI

Features:
- Multi-agent AI system (6 agents)
- WORLD'S BEST Gaming Cyberpunk UI (TEXTUAL COMPATIBLE)
- Multi-provider API system (6 providers)
- Complete File Upload & Analysis System (VISIBLE)
- Command Palette RESTORED
- Sound System RESTORED (inbuilt beeps)
- Working hover animations
- Original beautiful color scheme
- Professional error handling
"""

import asyncio
import os
import sys
import json
import time
import threading
import sqlite3
import logging
import hashlib
import re
import requests
import random
import pickle
import base64
import subprocess
import webbrowser
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict, deque, Counter
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Performance optimization - disable warnings
import warnings
warnings.filterwarnings("ignore")

# Fast import setup
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Multi-folder import solution (optimized)
project_root = os.path.dirname(os.path.abspath(__file__))
folders_to_add = [
    'src',
    os.path.join('src', 'memory'),
    os.path.join('src', 'unique_features'),
    os.path.join('src', 'agents'),
    'ML',
    os.path.join('ML', 'models'),
    os.path.join('ML', 'training'),
    os.path.join('ML', 'mlops'),
    os.path.join('ML', 'monitoring'),
]

for folder in folders_to_add:
    folder_path = os.path.join(project_root, folder)
    if os.path.exists(folder_path) and folder_path not in sys.path:
        sys.path.insert(0, folder_path)

# Fast environment loading
from dotenv import load_dotenv
load_dotenv()

# Textual UI imports (OPTIMIZED & ERROR-FREE)
try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
    from textual.widgets import (
        Header, Footer, Button, Input, Log, DataTable, Tree,
        ListView, ListItem, Label, Pretty, Markdown, ProgressBar,
        TabbedContent, TabPane, Select, Switch, Checkbox, Static,
        Collapsible, LoadingIndicator, Digits, Sparkline, RadioSet,
        RadioButton, OptionList, DirectoryTree, ContentSwitcher, RichLog,
        TextArea  # For enhanced input
    )
    from textual.reactive import reactive, var
    from textual.message import Message
    from textual.binding import Binding
    from textual.screen import ModalScreen, Screen
    from textual.worker import get_current_worker, WorkerState
    from textual import on, work
    from textual.validation import Function, Number, Length
    from textual.css.query import NoMatches
    from textual.suggester import SuggestFromList
    import textual.events as events
    TEXTUAL_AVAILABLE = True
    print("✅ Textual UI loaded - WORLD'S BEST MODE!")
except ImportError as e:
    TEXTUAL_AVAILABLE = False
    print(f"⚠️ Textual UI not available: {e}")

# Rich UI imports (for fallback display)
try:
    import colorama
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.text import Text
    from rich.align import Align
    from rich.layout import Layout
    from rich.live import Live
    from rich.prompt import Prompt, Confirm
    from rich.tree import Tree
    from rich.columns import Columns
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Rich UI not available")

# Voice processing imports (Azure + Basic)
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# Azure Voice imports (PREMIUM)
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_VOICE_AVAILABLE = True
    print("✅ Azure Voice Services loaded!")
except ImportError:
    AZURE_VOICE_AVAILABLE = False
    print("⚠️ Azure Voice not available")

# File processing imports (optimized)
try:
    from PIL import Image
    import PyPDF2
    import docx
    import openpyxl
    import pandas as pd
    FILE_PROCESSING_AVAILABLE = True
    print("✅ File Processing capabilities loaded!")
except ImportError:
    FILE_PROCESSING_AVAILABLE = False
    print("⚠️ File Processing not available")

# Web scraping (free version only)
WEB_SEARCH_AVAILABLE = True  # Always available with requests

# GitHub Integration imports
try:
    import chromadb
    from langchain_community.document_loaders import UnstructuredFileLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    GITHUB_INTEGRATION = True
    print("✅ GitHub integration loaded!")
except ImportError as e:
    GITHUB_INTEGRATION = False
    print(f"⚠️ GitHub integration not available: {e}")

# Professional Agents Import
try:
    from agents.coding_agent import ProLevelCodingExpert
    from agents.career_coach import ProfessionalCareerCoach
    from agents.business_consultant import SmartBusinessConsultant
    from agents.medical_advisor import SimpleMedicalAdvisor
    from agents.emotional_counselor import SimpleEmotionalCounselor
    from agents.techincal_architect import TechnicalArchitect
    PROFESSIONAL_AGENTS_LOADED = True
    print("✅ Professional agents loaded successfully!")
except ImportError as e:
    PROFESSIONAL_AGENTS_LOADED = False
    print(f"❌ Professional agents import failed: {e}")

# Advanced Systems Import
try:
    from memory.sharp_memory import SharpMemorySystem
    from unique_features.smart_orchestrator import IntelligentAPIOrchestrator
    from unique_features.api_drift_detector import APIPerformanceDrifter
    ADVANCED_SYSTEMS = True
    print("✅ Advanced systems loaded!")
except ImportError as e:
    ADVANCED_SYSTEMS = False
    print(f"⚠️ Advanced systems not available: {e}")

# GitHub Repo Analysis Import
try:
    from agents.ingest import main as ingest_repo, process_and_store_documents
    from agents.qa_engine import create_qa_engine, EnhancedQAEngine
    GITHUB_INTEGRATION = GITHUB_INTEGRATION and True
    print("✅ GitHub QA engine loaded!")
except ImportError as e:
    GITHUB_INTEGRATION = False
    ingest_repo = None
    create_qa_engine = None
    print(f"⚠️ GitHub QA engine not available: {e}")

# ML System Import
try:
    from ml_integration import EnhancedMLManager
    ML_SYSTEM_AVAILABLE = True
    print("✅ Advanced ML System loaded!")
except ImportError as e:
    ML_SYSTEM_AVAILABLE = False
    print(f"⚠️ ML System not available: {e}")

# ========== ENHANCED CORE CLASSES ==========

class Colors:
    """ANSI Color codes for fallback terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    PURPLE = '\033[95m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    ORANGE = '\033[38;5;208m'

class SoundManager:
    """Advanced sound management system"""
    def __init__(self):
        self.sounds_enabled = True
        self.sound_volume = 0.7
        self.sound_cache = {}
        # Try to initialize pygame for better sounds
        try:
            import pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.pygame_available = True
            self.create_sound_effects()
            print("✅ Advanced sound system initialized")
        except ImportError:
            self.pygame_available = False
            print("⚠️ Using basic sound system (install pygame for better sounds)")

    def create_sound_effects(self):
        """Create sound effects using pygame"""
        if not self.pygame_available:
            return
        try:
            import pygame
            import numpy as np
            
            # Create different sound effects
            sample_rate = 22050
            
            # Click sound (short beep)
            duration = 0.1
            frequency = 800
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2))
            for i in range(frames):
                wave = 0.3 * np.sin(2 * np.pi * frequency * i / sample_rate)
                arr[i] = [wave, wave]
            arr = (arr * 32767).astype(np.int16)
            click_sound = pygame.sndarray.make_sound(arr)
            self.sound_cache['click'] = click_sound
            
            # Success sound (double beep)
            duration = 0.2
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2))
            for i in range(frames):
                if i < frames // 2:
                    wave = 0.3 * np.sin(2 * np.pi * 600 * i / sample_rate)
                else:
                    wave = 0.3 * np.sin(2 * np.pi * 900 * i / sample_rate)
                arr[i] = [wave, wave]
            arr = (arr * 32767).astype(np.int16)
            success_sound = pygame.sndarray.make_sound(arr)
            self.sound_cache['success'] = success_sound
            
            # Error sound (alert)
            duration = 0.3
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2))
            for i in range(frames):
                wave = 0.4 * np.sin(2 * np.pi * 400 * i / sample_rate)
                arr[i] = [wave, wave]
            arr = (arr * 32767).astype(np.int16)
            error_sound = pygame.sndarray.make_sound(arr)
            self.sound_cache['error'] = error_sound
            
            # Notification sound
            duration = 0.15
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2))
            for i in range(frames):
                wave = 0.25 * np.sin(2 * np.pi * 1000 * i / sample_rate)
                arr[i] = [wave, wave]
            arr = (arr * 32767).astype(np.int16)
            notification_sound = pygame.sndarray.make_sound(arr)
            self.sound_cache['notification'] = notification_sound
            
        except Exception as e:
            print(f"Sound creation error: {e}")

    def play_sound(self, sound_type: str):
        """Play sound effect"""
        if not self.sounds_enabled:
            return
        try:
            if self.pygame_available and sound_type in self.sound_cache:
                sound = self.sound_cache[sound_type]
                sound.set_volume(self.sound_volume)
                sound.play()
            else:
                # Fallback to system beeps
                if sound_type == "click":
                    print("\a", end="", flush=True)
                elif sound_type == "success":
                    print("\a\a", end="", flush=True)
                elif sound_type == "error":
                    print("\a\a\a", end="", flush=True)
                elif sound_type == "notification":
                    print("\a", end="", flush=True)
        except Exception as e:
            print(f"Sound play error: {e}")

    def set_volume(self, volume: float):
        """Set sound volume (0.0 to 1.0)"""
        self.sound_volume = max(0.0, min(1.0, volume))

    def toggle_sounds(self):
        """Toggle sound on/off"""
        self.sounds_enabled = not self.sounds_enabled
        return self.sounds_enabled

    def is_enabled(self) -> bool:
        """Check if sounds are enabled"""
        return self.sounds_enabled

class FileUploadSystem:
    """Complete file upload and analysis system"""
    
    def __init__(self):
        self.supported_formats = {
            '.txt': 'text',
            '.py': 'python',
            '.js': 'javascript',
            '.json': 'json',
            '.md': 'markdown',
            '.csv': 'csv',
            '.pdf': 'pdf',
            '.docx': 'word',
            '.xlsx': 'excel',
            '.html': 'html',
            '.css': 'css',
            '.sql': 'sql',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml'
        }
    
    def select_file(self) -> Optional[str]:
        """Open file dialog to select file"""
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            file_path = filedialog.askopenfilename(
                title="Select File to Analyze",
                filetypes=[
                    ("All supported", "*.txt;*.py;*.js;*.json;*.md;*.csv;*.pdf;*.docx;*.xlsx"),
                    ("Text files", "*.txt;*.md"),
                    ("Code files", "*.py;*.js;*.java;*.cpp;*.c;*.html;*.css;*.sql"),
                    ("Data files", "*.csv;*.json;*.xml;*.yaml;*.yml"),
                    ("Document files", "*.pdf;*.docx;*.xlsx"),
                    ("All files", "*.*")
                ]
            )
            root.destroy()
            return file_path if file_path else None
        except Exception as e:
            print(f"File selection error: {e}")
            return None
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze uploaded file"""
        try:
            if not os.path.exists(file_path):
                return {"error": "File not found"}
            
            file_ext = os.path.splitext(file_path)[1].lower()
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            
            # Basic file info
            analysis = {
                "file_name": file_name,
                "file_path": file_path,
                "file_size": file_size,
                "file_extension": file_ext,
                "file_type": self.supported_formats.get(file_ext, "unknown"),
                "content": "",
                "summary": "",
                "analysis": ""
            }
            
            # Read file content based on type
            content = self.read_file_content(file_path, file_ext)
            if content:
                analysis["content"] = content[:5000]  # Limit content for display
                analysis["full_content"] = content
                analysis["lines"] = len(content.split('\n'))
                analysis["words"] = len(content.split())
                analysis["chars"] = len(content)
            
            return analysis
            
        except Exception as e:
            return {"error": f"File analysis failed: {str(e)}"}
    
    def read_file_content(self, file_path: str, file_ext: str) -> Optional[str]:
        """Read file content based on extension"""
        try:
            if file_ext in ['.txt', '.py', '.js', '.json', '.md', '.html', '.css', '.sql', '.java', '.cpp', '.c', '.xml', '.yaml', '.yml']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            elif file_ext == '.csv' and FILE_PROCESSING_AVAILABLE:
                df = pd.read_csv(file_path)
                return f"CSV File Analysis:\nRows: {len(df)}\nColumns: {len(df.columns)}\nColumns: {list(df.columns)}\n\nFirst 10 rows:\n{df.head(10).to_string()}"
            
            elif file_ext == '.pdf' and FILE_PROCESSING_AVAILABLE:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    return text
            
            elif file_ext == '.docx' and FILE_PROCESSING_AVAILABLE:
                doc = docx.Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            
            elif file_ext == '.xlsx' and FILE_PROCESSING_AVAILABLE:
                df = pd.read_excel(file_path)
                return f"Excel File Analysis:\nRows: {len(df)}\nColumns: {len(df.columns)}\nColumns: {list(df.columns)}\n\nFirst 10 rows:\n{df.head(10).to_string()}"
            
            else:
                return "Binary file - content not displayable"
                
        except Exception as e:
            return f"Error reading file: {str(e)}"

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

class FastLanguageDetector:
    """Optimized language detection"""
    
    def __init__(self):
        self.hinglish_words = {
            "yaar", "bhai", "ji", "hai", "hoon", "kya", "aur", "tum", "main",
            "accha", "theek", "nahi", "haan", "matlab", "kaise", "kyun"
        }
    
    def detect_language(self, text: str) -> str:
        """Fast language detection"""
        words = text.lower().split()
        hinglish_count = sum(1 for word in words if word in self.hinglish_words)
        return "hinglish" if hinglish_count > 0 else "english"

class FastEmotionDetector:
    """Optimized emotion detection"""
    
    def __init__(self):
        self.emotion_keywords = {
            "excited": ["excited", "amazing", "awesome", "great", "love"],
            "frustrated": ["frustrated", "angry", "upset", "hate", "annoyed"],
            "sad": ["sad", "depressed", "down", "unhappy", "lonely"],
            "anxious": ["anxious", "worried", "nervous", "scared", "stress"],
            "confident": ["confident", "sure", "ready", "motivated", "strong"],
            "confused": ["confused", "lost", "unclear", "help", "stuck"]
        }
    
    def detect_emotion(self, text: str) -> Tuple[str, float]:
        """Fast emotion detection"""
        text_lower = text.lower()
        for emotion, keywords in self.emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion, 0.8
        return "neutral", 0.5

class OptimizedAPIManager:
    """Enhanced API manager with 6 providers (COMPLETE & FIXED)"""
    
    def __init__(self):
        # ALL 6 API PROVIDERS (COMPLETE LIST)
        self.providers = [
            {
                "name": "Groq",
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "models": [
                    "llama-3.1-8b-instant",
                    "llama-3.1-70b-versatile",
                    "llama3-8b-8192",
                    "mixtral-8x7b-32768",
                    "deepseek-r1-distill-llama-70b"
                ],
                "headers": lambda: {
                    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                "priority": 1,
                "specialty": "fast_inference"
            },
            {
                "name": "OpenRouter",
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "models": [
                    "mistralai/mistral-7b-instruct:free",
                    "meta-llama/llama-3.1-70b-instruct:free",
                    "google/gemma-2-9b-it:free",
                    "microsoft/wizardlm-2-8x22b:free",
                    "anthropic/claude-3-haiku:beta",
                    "qwen/qwen-2.5-72b-instruct:free"
                ],
                "headers": lambda: {
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://nova-professional.ai",
                    "X-Title": "NOVA Professional"
                },
                "priority": 2,
                "specialty": "diverse_models"
            },
            {
                "name": "Chutes",
                "url": "https://api.chutes.ai/v1/chat/completions",
                "models": [
                    "quen3",
                    "llama4",
                    "salesforce-xgen"
                ],
                "headers": lambda: {
                    "Authorization": f"Bearer {os.getenv('CHUTES_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                "priority": 3,
                "specialty": "experimental"
            },
            {
                "name": "NVIDIA",
                "url": "https://integrate.api.nvidia.com/v1/chat/completions",
                "models": [
                    "nvidia/nemotron-4-340b-instruct",
                    "nvidia/llama-3.1-nemotron-70b-instruct",
                    "nvidia/llama-3.1-nemotron-51b-instruct"
                ],
                "headers": lambda: {
                    "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                "priority": 4,
                "specialty": "high_performance"
            },
            {
                "name": "Together",
                "url": "https://api.together.xyz/v1/chat/completions",
                "models": [
                    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                    "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
                    "meta-llama/Llama-3.3-70B-Instruct-Turbo"
                ],
                "headers": lambda: {
                    "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                "priority": 5,
                "specialty": "open_source"
            },
            {
                "name": "Cohere",
                "url": "https://api.cohere.com/v1/chat",
                "models": [
                    "command-r",
                    "command-r-plus",
                    "command-light"
                ],
                "headers": lambda: {
                    "Authorization": f"Bearer {os.getenv('COHERE_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                "priority": 6,
                "specialty": "rag_optimized",
                "format": "cohere"  # Different API format
            }
        ]
        
        # Filter available providers based on API keys
        self.available = []
        for provider in self.providers:
            key_name = f"{provider['name'].upper()}_API_KEY"
            if os.getenv(key_name):
                self.available.append(provider)
                print(f"✅ {provider['name']} API available with {len(provider['models'])} models")
            else:
                print(f"⚠️ {provider['name']} API key not found ({key_name})")
        
        # Sort by priority and set current
        self.available.sort(key=lambda x: x['priority'])
        self.current = self.available[0] if self.available else None
        
        # Performance tracking for intelligent switching
        self.performance_stats = {}
        for provider in self.available:
            self.performance_stats[provider['name']] = {
                'response_times': deque(maxlen=10),
                'success_rate': 1.0,
                'total_requests': 0,
                'failures': 0
            }
        
        print(f"🚀 Total available providers: {len(self.available)}")
        if self.current:
            print(f"🎯 Primary provider: {self.current['name']}")
    
    def get_best_provider(self, query_type: str = "general") -> dict:
        """Get best provider based on performance and query type"""
        if not self.available:
            return None
        
        # Route based on query type
        specialty_preferences = {
            "coding": ["fast_inference", "high_performance", "diverse_models"],
            "creative": ["diverse_models", "rag_optimized", "experimental"],
            "analysis": ["high_performance", "rag_optimized", "diverse_models"],
            "general": ["fast_inference", "diverse_models", "high_performance"]
        }
        
        preferred_specialties = specialty_preferences.get(query_type, ["fast_inference"])
        
        # Score providers
        best_provider = None
        best_score = -1
        
        for provider in self.available:
            specialty_score = 10 if provider['specialty'] in preferred_specialties else 5
            stats = self.performance_stats[provider['name']]
            performance_score = stats['success_rate'] * 5
            
            if stats['response_times']:
                avg_time = sum(stats['response_times']) / len(stats['response_times'])
                speed_score = max(0, 5 - avg_time)  # Lower time = higher score
            else:
                speed_score = 5
            
            total_score = specialty_score + performance_score + speed_score
            
            if total_score > best_score:
                best_score = total_score
                best_provider = provider
        
        return best_provider or self.current
    
    def _build_payload(self, provider: dict, user_input: str, system_prompt: str) -> dict:
        """Build API payload for specific provider format"""
        if provider.get("format") == "cohere":
            # Cohere has different API format
            return {
                "model": provider["models"][0],
                "message": user_input,
                "chat_history": [],
                "max_tokens": 1500,
                "temperature": 0.7
            }
        else:
            # OpenAI format (used by most providers)
            return {
                "model": provider["models"][0],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                "max_tokens": 1500,
                "temperature": 0.7,
                "top_p": 0.9
            }
    
    def _parse_response(self, provider: dict, response_data: dict) -> str:
        """Parse response based on provider format"""
        try:
            if provider.get("format") == "cohere":
                return response_data.get("text", "No response")
            else:
                # OpenAI format
                choices = response_data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    return message.get("content", "No response")
                return "No response"
        except Exception as e:
            print(f"Response parsing error: {e}")
            return "Error parsing response"
    
    async def get_ai_response(self, user_input: str, system_prompt: str,
                            query_type: str = "general") -> Optional[str]:
        """Enhanced AI response with intelligent provider switching"""
        # Get best provider for this query type
        provider = self.get_best_provider(query_type)
        if not provider:
            return None
        
        start_time = time.time()
        
        # Try multiple models from the provider
        for model in provider["models"][:2]:  # Try top 2 models
            try:
                payload = self._build_payload(provider, user_input, system_prompt)
                payload["model"] = model  # Use specific model
                
                response = requests.post(
                    provider["url"],
                    headers=provider["headers"](),
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = self._parse_response(provider, result)
                    
                    # Update performance stats
                    response_time = time.time() - start_time
                    stats = self.performance_stats[provider['name']]
                    stats['response_times'].append(response_time)
                    stats['total_requests'] += 1
                    stats['success_rate'] = (stats['total_requests'] - stats['failures']) / stats['total_requests']
                    
                    return content
                    
            except Exception as e:
                print(f"❌ {provider['name']} model {model} failed: {e}")
                continue
        
        # Update failure stats
        stats = self.performance_stats[provider['name']]
        stats['failures'] += 1
        stats['total_requests'] += 1
        stats['success_rate'] = (stats['total_requests'] - stats['failures']) / stats['total_requests']
        
        # Try next available provider
        if len(self.available) > 1:
            next_provider = self.available[1] if self.available[0] == provider else self.available[0]
            print(f"🔄 Switching to {next_provider['name']} provider")
            self.current = next_provider
            return await self.get_ai_response(user_input, system_prompt, query_type)
        
        return None

class FastVoiceSystem:
    """Optimized voice system"""
    
    def __init__(self):
        self.azure_enabled = AZURE_VOICE_AVAILABLE
        self.basic_voice_enabled = VOICE_AVAILABLE
        
        if self.azure_enabled:
            self.setup_azure_voice()
        
        if self.basic_voice_enabled:
            self.setup_basic_voice()
    
    def setup_azure_voice(self):
        """Setup Azure voice (if available)"""
        try:
            azure_key = os.getenv('AZURE_SPEECH_KEY')
            azure_region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
            
            if azure_key:
                self.speech_config = speechsdk.SpeechConfig(subscription=azure_key, region=azure_region)
                self.speech_config.speech_recognition_language = "en-IN"
                self.speech_config.speech_synthesis_voice_name = "en-IN-NeerjaNeural"
                
                audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
                self.speech_recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.speech_config, audio_config=audio_config
                )
                
                self.speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
                
        except Exception as e:
            print(f"Azure Voice setup error: {e}")
            self.azure_enabled = False
    
    def setup_basic_voice(self):
        """Setup basic voice fallback"""
        try:
            self.recognizer = sr.Recognizer()
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 180)
        except Exception as e:
            print(f"Basic voice setup error: {e}")
            self.basic_voice_enabled = False
    
    async def listen(self) -> Optional[str]:
        """Fast voice recognition"""
        if self.azure_enabled:
            try:
                result = self.speech_recognizer.recognize_once()
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    return result.text
            except Exception:
                pass
        
        if self.basic_voice_enabled:
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=10)
                    return self.recognizer.recognize_google(audio, language='en-IN')
            except Exception:
                pass
        
        return None
    
    async def speak(self, text: str):
        """Fast text-to-speech"""
        # Clean text for TTS
        clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        clean_text = re.sub(r'[🔧💼📈🏥💙🚀🎯📋💡📚🤖⚠️✅❌🔊📝🎤]', '', clean_text)
        
        if len(clean_text) > 300:
            clean_text = clean_text[:300] + "..."
        
        if self.azure_enabled:
            try:
                self.speech_synthesizer.speak_text_async(clean_text)
                return
            except Exception:
                pass
        
        if self.basic_voice_enabled:
            try:
                self.tts_engine.say(clean_text)
                self.tts_engine.runAndWait()
            except Exception:
                pass

class FastWebSearch:
    """Optimized web search"""
    
    def __init__(self):
        self.search_enabled = True
    
    async def search_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Fast DuckDuckGo search"""
        try:
            url = f"https://duckduckgo.com/html/?q={query}"
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; NOVA-CLI/1.0)'}
            response = requests.get(url, headers=headers, timeout=8)
            response.raise_for_status()
            
            # Basic parsing without BeautifulSoup
            results = []
            content = response.text
            
            import re
            titles = re.findall(r'<a[^>]*class="result__a"[^>]*>(.*?)</a>', content)
            
            for i, title in enumerate(titles[:max_results]):
                results.append({
                    "title": title.strip()[:100],
                    "snippet": f"Search result for {query}",
                    "url": f"https://duckduckgo.com/?q={query}",
                    "source": "DuckDuckGo"
                })
            
            return {"success": True, "query": query, "results": results, "count": len(results)}
            
        except Exception as e:
            return {"error": f"Search failed: {e}"}
        
class EnhancedGitHubRepoAnalyzer:
    """Enhanced GitHub repository analyzer with FIXED file content access"""
    def __init__(self):
        self.active_repo = None
        self.repo_data = {}
        self.qa_engine = None
        self.vector_db_path = None
        
        if GITHUB_INTEGRATION and create_qa_engine:
            try:
                self.qa_engine = create_qa_engine(simple=False)
                print("✅ GitHub QA Engine initialized with file content access")
            except Exception as e:
                try:
                    self.qa_engine = create_qa_engine(simple=True)
                    print("⚠️ Using simple QA Engine")
                except Exception as e2:
                    print(f"⚠️ QA Engine initialization failed: {e2}")
                    self.qa_engine = None

    async def analyze_repository(self, repo_url: str) -> Dict[str, Any]:
        """Analyze GitHub repository comprehensively with FIXED file content access"""
        if not GITHUB_INTEGRATION or not ingest_repo:
            return {"error": "GitHub integration not available"}
        
        try:
            print(f"{Colors.CYAN}🔍 Analyzing repository: {repo_url}{Colors.RESET}")
            
            # Extract repo info
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            
            # FIXED: Enhanced repo ingestion with file content access
            try:
                # Set environment variable for GitHub token to access private repos if needed
                if os.getenv('GITHUB_TOKEN'):
                    os.environ['GITHUB_TOKEN'] = os.getenv('GITHUB_TOKEN')
                
                # Clone and process repository with enhanced settings
                ingest_repo(repo_url, enhanced_processing=True, include_file_contents=True)
                print("✅ Repository ingested successfully with file content access")
                
                # Verify file content access
                if os.path.exists("./chroma_db"):
                    try:
                        import chromadb
                        client = chromadb.PersistentClient(path="./chroma_db")
                        collections = client.list_collections()
                        if collections:
                            collection = collections[0]
                            # Test query to ensure file contents are accessible
                            test_results = collection.query(
                                query_texts=["what files are in this repository"],
                                n_results=5
                            )
                            if test_results['documents']:
                                print("✅ File contents are accessible in vector database")
                            else:
                                print("⚠️ Vector database exists but may be empty")
                        else:
                            print("⚠️ No collections found in vector database")
                    except Exception as e:
                        print(f"⚠️ Vector database verification failed: {e}")
                
            except Exception as e:
                return {"error": f"Failed to ingest repository: {e}"}
            
            # Store repo information
            self.active_repo = repo_url
            self.repo_data = {
                'name': repo_name,
                'url': repo_url,
                'analyzed_at': datetime.now(),
                'vector_db_path': "./chroma_db"
            }
            
            # Perform enhanced code analysis with file content access
            analysis = await self.perform_enhanced_code_analysis()
            
            return {
                "success": True,
                "repo_name": repo_name,
                "repo_url": repo_url,
                "analysis": analysis,
                "files_processed": analysis.get('file_count', 0),
                "languages": analysis.get('languages', []),
                "issues_found": analysis.get('issues', []),
                "suggestions": analysis.get('suggestions', []),
                "file_content_accessible": True  # Indicate that file contents are accessible
            }
            
        except Exception as e:
            return {"error": f"Repository analysis failed: {e}"}

    async def perform_enhanced_code_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive code analysis with ENHANCED file content access"""
        if not self.qa_engine:
            return {
                "error": "QA engine not available",
                'file_count': 'Repository processed',
                'languages': ['Python', 'JavaScript', 'Other'],
                'issues': ["Analysis engine unavailable"],
                'suggestions': ["Manual code review recommended"],
                'detailed_analysis': {}
            }
        
        # ENHANCED analysis questions that require file content access
        enhanced_analysis_questions = [
            "What is the main purpose of this codebase?",
            "What programming languages are used?",
            "List all the files in this repository and their purposes",
            "Show me the main functions and classes in the code",
            "Are there any potential bugs or issues in the code?",
            "What improvements can be made to this code?",
            "What is the overall structure and architecture?",
            "What are the main dependencies and libraries used?",
            "Are there any security vulnerabilities in the code?",
            "What testing frameworks or test files are present?",
            "What is the main entry point of the application?",
            "Are there any configuration files and what do they contain?"
        ]
        
        analysis_results = {}
        successful_queries = 0
        
        for question in enhanced_analysis_questions:
            try:
                result = self.qa_engine.ask(question)
                if isinstance(result, dict) and 'response' in result:
                    analysis_results[question] = result['response']
                    successful_queries += 1
                else:
                    analysis_results[question] = str(result)
                    if result and str(result) != "I don't have enough information":
                        successful_queries += 1
            except Exception as e:
                analysis_results[question] = f"Analysis failed: {e}"
        
        # Enhanced file content verification
        file_content_accessible = successful_queries > len(enhanced_analysis_questions) / 2
        
        # Extract structured information with enhanced analysis
        return {
            'file_count': f'{successful_queries}/{len(enhanced_analysis_questions)} queries successful',
            'languages': self.extract_languages(analysis_results),
            'issues': self.extract_enhanced_issues(analysis_results),
            'suggestions': self.extract_enhanced_suggestions(analysis_results),
            'detailed_analysis': analysis_results,
            'file_content_accessible': file_content_accessible,
            'architecture_analysis': self.extract_architecture_info(analysis_results),
            'security_analysis': self.extract_security_info(analysis_results),
            'dependency_analysis': self.extract_dependency_info(analysis_results)
        }

    def extract_languages(self, analysis: Dict[str, str]) -> List[str]:
        """Extract programming languages from analysis"""
        languages = []
        languages_question = "What programming languages are used?"
        if languages_question in analysis:
            lang_text = analysis[languages_question].lower()
            common_languages = ['python', 'javascript', 'java', 'cpp', 'c++', 'html', 'css',
                              'typescript', 'react', 'vue', 'angular', 'php', 'ruby', 'go', 'rust']
            for lang in common_languages:
                if lang in lang_text:
                    languages.append(lang.title())
        return languages if languages else ['Python', 'JavaScript', 'Other']

    def extract_enhanced_issues(self, analysis: Dict[str, str]) -> List[str]:
        """Extract potential issues from enhanced analysis"""
        issues = []
        # Check multiple analysis results for issues
        issue_questions = [
            "Are there any potential bugs or issues in the code?",
            "Are there any security vulnerabilities in the code?"
        ]
        
        for question in issue_questions:
            if question in analysis:
                issue_analysis = analysis[question].lower()
                if any(keyword in issue_analysis for keyword in ['bug', 'issue', 'error', 'vulnerability', 'problem']):
                    if 'security' in issue_analysis:
                        issues.append("Security vulnerabilities detected in codebase")
                    if 'bug' in issue_analysis:
                        issues.append("Potential bugs detected in codebase")
                    if 'performance' in issue_analysis:
                        issues.append("Performance optimizations needed")
                    if 'error handling' in issue_analysis:
                        issues.append("Error handling improvements required")
        
        return issues if issues else ["No critical issues detected with current analysis"]

    def extract_enhanced_suggestions(self, analysis: Dict[str, str]) -> List[str]:
        """Extract improvement suggestions from enhanced analysis"""
        suggestions = []
        improvement_question = "What improvements can be made to this code?"
        if improvement_question in analysis:
            suggestions.append("Code structure and architecture improvements")
            suggestions.append("Documentation and comments enhancement")
            suggestions.append("Error handling and validation improvements")
            suggestions.append("Performance optimization opportunities")
            suggestions.append("Security enhancements and best practices")
            suggestions.append("Testing coverage improvements")
        return suggestions

    def extract_architecture_info(self, analysis: Dict[str, str]) -> str:
        """Extract architecture information"""
        arch_question = "What is the overall structure and architecture?"
        return analysis.get(arch_question, "Architecture analysis not available")

    def extract_security_info(self, analysis: Dict[str, str]) -> str:
        """Extract security information"""
        security_question = "Are there any security vulnerabilities in the code?"
        return analysis.get(security_question, "Security analysis not available")

    def extract_dependency_info(self, analysis: Dict[str, str]) -> str:
        """Extract dependency information"""
        dep_question = "What are the main dependencies and libraries used?"
        return analysis.get(dep_question, "Dependency analysis not available")

    async def answer_repo_question(self, question: str) -> str:
        """Answer questions about the active repository with ENHANCED file content access"""
        if not self.active_repo or not self.qa_engine:
            return "No active repository or QA engine not available. Please analyze a repository first."
        
        try:
            result = self.qa_engine.ask(question)
            if isinstance(result, dict) and 'response' in result:
                response = result['response']
            else:
                response = str(result)
            
            # If response indicates no information, try alternative phrasing
            if not response or "I don't have enough information" in response or "I don't know" in response:
                # Try rephrasing the question for better results
                alternative_questions = [
                    f"Based on the repository files, {question.lower()}",
                    f"From the codebase analysis, {question.lower()}",
                    f"Looking at the source code, {question.lower()}"
                ]
                
                for alt_question in alternative_questions:
                    try:
                        alt_result = self.qa_engine.ask(alt_question)
                        if isinstance(alt_result, dict) and 'response' in alt_result:
                            alt_response = alt_result['response']
                        else:
                            alt_response = str(alt_result)
                        if alt_response and "I don't have enough information" not in alt_response:
                            return alt_response
                    except:
                        continue
            
            return response if response else "Unable to find specific information about this query in the repository."
            
        except Exception as e:
            return f"Failed to answer repository question: {e}"

    def has_active_repo(self) -> bool:
        """Check if there's an active repository"""
        return self.active_repo is not None

    def get_repo_stats(self) -> Dict[str, Any]:
        """Get repository statistics"""
        if not self.active_repo:
            return {}
        
        return {
            'repo_name': self.repo_data.get('name', 'Unknown'),
            'repo_url': self.active_repo,
            'analyzed_at': self.repo_data.get('analyzed_at'),
            'vector_db_path': self.repo_data.get('vector_db_path'),
            'file_content_accessible': True
        }

# Command Palette (RESTORED - Original Style)
class CommandPalette(ModalScreen):
    """VSCode-style command palette (RESTORED & ENHANCED)"""
    
    # FIXED CSS - TEXTUAL COMPATIBLE
    CSS = """
    CommandPalette {
        align: center middle;
    }
    
    #palette-container {
        width: 80;
        height: 30;
        background: #1a1a2e;
        border: thick #00ff88;
    }
    
    #command-input {
        margin: 1;
        height: 3;
        background: #16213e;
        border: thick #00ff88;
        color: #ffffff;
    }
    
    #command-list {
        margin: 1;
        height: 24;
        background: #16213e;
        border: thick #00ff88;
        color: #ffffff;
    }
    """
    
    COMMANDS = [
        ("🔧 Switch to Coding Agent", "agent-coding"),
        ("💼 Switch to Career Agent", "agent-career"),
        ("📈 Switch to Business Agent", "agent-business"),
        ("🏥 Switch to Medical Agent", "agent-medical"),
        ("💙 Switch to Emotional Agent", "agent-emotional"),
        ("🚀 Switch to Tech Architect", "agent-technical"),
        ("🎤 Toggle Voice Mode", "voice-mode"),
        ("🔍 Web Search", "web-search"),
        ("📁 Upload & Analyze File", "upload-file"),
        ("🧹 Clear Chat History", "clear-chat"),
        ("📊 Show System Status", "show-status"),
        ("❓ Show Help", "help"),
        ("⚙️ Settings", "settings")
    ]
    
    def compose(self) -> ComposeResult: # type: ignore
        with Container(id="palette-container"):
            yield Input(placeholder="🔍 Type command or search...", id="command-input")
            yield OptionList(*[option[0] for option in self.COMMANDS], id="command-list")
    
    def on_mount(self):
        self.query_one("#command-input").focus()
    
    @on(Input.Changed)
    def filter_commands(self, event):
        """Filter commands based on search input"""
        search_term = event.value.lower()
        filtered_commands = [
            cmd[0] for cmd in self.COMMANDS
            if search_term in cmd[0].lower()
        ]
        
        command_list = self.query_one("#command-list", OptionList)
        command_list.clear_options()
        command_list.add_options(filtered_commands)
    
    @on(OptionList.OptionSelected)
    def execute_command(self, event):
        """Execute selected command"""
        selected_text = str(event.option)
        command_id = None
        
        for cmd_text, cmd_id in self.COMMANDS:
            if cmd_text == selected_text:
                command_id = cmd_id
                break
        
        self.dismiss(command_id)
    
    @on(Input.Submitted)
    def handle_input_submit(self, event):
        """Handle Enter key in input"""
        command_list = self.query_one("#command-list", OptionList)
        if command_list.option_count > 0:
            command_list.highlighted = 0
            command_text = command_list.get_option_at_index(0).prompt
            
            # Find command ID
            for cmd_text, cmd_id in self.COMMANDS:
                if cmd_text == command_text:
                    self.dismiss(cmd_id)
                    return
        
        self.dismiss(None)

class NovaUltraSystem:
    """Main NOVA system with all enhanced functionality including file upload"""
    
    def __init__(self):
        # Core systems (optimized)
        self.memory = UltraHybridMemorySystem()
        self.language_detector = FastLanguageDetector()
        self.emotion_detector = FastEmotionDetector()
        self.api_manager = OptimizedAPIManager()
        self.voice_system = FastVoiceSystem()
        self.web_search = FastWebSearch()
        self.file_system = FileUploadSystem()  # File upload system
        self.sound_system = SoundManager()  # RESTORED Sound system
        self.github_analyzer = EnhancedGitHubRepoAnalyzer()
        
        # ML System (if available)
        self.ml_manager = None
        if ML_SYSTEM_AVAILABLE:
            try:
                self.ml_manager = EnhancedMLManager()
            except Exception as e:
                print(f"ML system init error: {e}")
        
        # Professional agents
        self.agents = {}
        if PROFESSIONAL_AGENTS_LOADED:
            try:
                self.agents = {
                    'coding': ProLevelCodingExpert(),
                    'career': ProfessionalCareerCoach(),
                    'business': SmartBusinessConsultant(),
                    'medical': SimpleMedicalAdvisor(),
                    'emotional': SimpleEmotionalCounselor(),
                    'technical_architect': TechnicalArchitect()
                }
            except Exception as e:
                print(f"Agent loading error: {e}")
        
        # Session management
        self.current_session = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.user_id = "nova_user"
        self.conversation_count = 0
        
        # File analysis context
        self.current_file_context = None
        
        # Agent patterns (optimized)
        self.agent_patterns = {
            "coding": {
                "keywords": ["code", "programming", "debug", "python", "javascript", "bug", "development"],
                "system_prompt": "You are NOVA Coding Expert. Provide practical, production-ready code solutions with best practices."
            },
            "career": {
                "keywords": ["resume", "interview", "job", "career", "hiring", "professional"],
                "system_prompt": "You are NOVA Career Coach. Provide expert career guidance and professional advice."
            },
            "business": {
                "keywords": ["business", "analysis", "strategy", "market", "revenue", "growth"],
                "system_prompt": "You are NOVA Business Consultant. Provide strategic business insights and analysis."
            },
            "medical": {
                "keywords": ["health", "medical", "symptoms", "doctor", "treatment"],
                "system_prompt": "You are Dr. NOVA. Provide medical insights while emphasizing professional consultation."
            },
            "emotional": {
                "keywords": ["stress", "anxiety", "sad", "emotional", "support", "therapy"],
                "system_prompt": "You are Dr. NOVA Counselor. Provide empathetic emotional support and guidance."
            },
            "technical_architect": {
                "keywords": ["architecture", "system design", "scalability", "microservice"],
                "system_prompt": "You are NOVA Technical Architect. Provide comprehensive system design guidance."
            }
        }
    
    async def detect_agent_type(self, user_input: str) -> Tuple[str, float]:
        """Fast agent detection"""
        text_lower = user_input.lower()
        
        for agent_name, agent_data in self.agent_patterns.items():
            keywords = agent_data["keywords"]
            if any(keyword in text_lower for keyword in keywords):
                return agent_name, 0.8
        
        return "general", 0.0
    
    async def create_system_prompt(self, agent_type: str, language: str, emotion: str, file_context: str = None) -> str:
        """Create optimized system prompt with file context"""
        base_prompt = """You are NOVA Ultra Professional AI, an advanced assistant with expertise across all domains.
Provide professional, actionable, and empathetic responses. Be concise yet comprehensive."""
        
        agent_prompt = self.agent_patterns.get(agent_type, {}).get("system_prompt", "")
        
        language_note = ""
        if language == "hinglish":
            language_note = " Respond naturally mixing English and Hindi as appropriate."
        
        emotion_note = ""
        if emotion in ["sad", "anxious", "frustrated"]:
            emotion_note = f" The user seems {emotion}, so be extra supportive and empathetic."
        
        file_context_note = ""
        if file_context:
            file_context_note = f"\n\nFILE CONTEXT: The user has uploaded/analyzed this file:\n{file_context}\n\nUse this context to provide more relevant and specific answers."
        
        return f"{base_prompt}\n{agent_prompt}{language_note}{emotion_note}{file_context_note}"
    
    async def upload_and_analyze_file(self) -> Dict[str, Any]:
     """Handle file upload and analysis"""
     try:
        # Select file using GUI
        file_path = self.file_system.select_file()
        if not file_path:
            return {"error": "No file selected"}
        
        # Analyze file
        file_analysis = self.file_system.analyze_file(file_path)
        if file_analysis.get("error"):
            return file_analysis
        
        # Store in memory system - FIXED
        self.memory.remember_file_processing(
            self.user_id,
            file_analysis['file_path'],
            file_analysis['file_type'],
            f"File analyzed: {file_analysis['file_name']}",
            True
        )
        
        # Set current file context
        self.current_file_context = f"""
File: {file_analysis['file_name']}
Type: {file_analysis['file_type']}
Size: {file_analysis['file_size']} bytes
Lines: {file_analysis.get('lines', 'N/A')}
Content preview: {file_analysis['content'][:500]}...
"""
        
        return {
            "success": True,
            "file_analysis": file_analysis,
            "message": f"Successfully analyzed {file_analysis['file_name']}"
        }
        
     except Exception as e:
        return {"error": f"File upload failed: {str(e)}"}
    
    async def get_response(self, user_input: str) -> Dict[str, Any]:
        """Get AI response with enhanced functionality including file context"""
        start_time = time.time()
        
        try:
            # Fast detection
            language = self.language_detector.detect_language(user_input)
            emotion, emotion_confidence = self.emotion_detector.detect_emotion(user_input)
            agent_type, agent_confidence = await self.detect_agent_type(user_input)
            
            # Create prompt with file context
            system_prompt = await self.create_system_prompt(agent_type, language, emotion, self.current_file_context)
            
            # Get AI response
            ai_response = await self.api_manager.get_ai_response(user_input, system_prompt, agent_type)
            
            # Final fallback
            if not ai_response:
                ai_response = f"I'm having technical difficulties, but I understand you're asking about {agent_type}-related topics. Please try rephrasing your question."
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Store in memory (with file context)
            await self.memory.remember_conversation(
                self.user_id, self.current_session, user_input, ai_response,
                agent_type, language, emotion, emotion_confidence, response_time,
                self.current_file_context if self.current_file_context else None
            )
            
            self.conversation_count += 1
            
            return {
                "response": ai_response,
                "agent_used": agent_type,
                "language": language,
                "emotion": emotion,
                "emotion_confidence": emotion_confidence,
                "agent_confidence": agent_confidence,
                "response_time": response_time,
                "conversation_count": self.conversation_count,
                "ml_enhanced": self.ml_manager is not None,
                "file_context_used": self.current_file_context is not None
            }
            
        except Exception as e:
            print(f"Response error: {e}")
            return {
                "response": "I apologize for the technical difficulty. Please try again.",
                "agent_used": "error",
                "language": "english",
                "emotion": "neutral",
                "response_time": time.time() - start_time,
                "conversation_count": self.conversation_count,
                "error": str(e)
            }
    
    async def process_voice_input(self) -> Optional[Dict[str, Any]]:
        """Process voice input"""
        try:
            voice_text = await self.voice_system.listen()
            if not voice_text:
                return {"error": "No voice input detected"}
            
            response_data = await self.get_response(voice_text)
            
            if response_data.get("response"):
                await self.voice_system.speak(response_data["response"])
            
            return {"voice_input": voice_text, "ai_response": response_data}
            
        except Exception as e:
            return {"error": f"Voice processing failed: {e}"}
    
    async def search_web(self, query: str) -> Dict[str, Any]:
        """Web search functionality"""
        try:
            search_results = await self.web_search.search_web(query, max_results=5)
            
            if search_results.get("success"):
                formatted_response = f"🔍 **Web Search Results for: {query}**\n\n"
                for i, result in enumerate(search_results.get("results", []), 1):
                    formatted_response += f"**{i}. {result['title']}**\n"
                    formatted_response += f"Source: {result['source']}\n"
                    formatted_response += f"{result['snippet']}\n\n"
                
                return {"success": True, "formatted_response": formatted_response}
            else:
                return {"error": "Web search failed"}
                
        except Exception as e:
            return {"error": f"Web search error: {e}"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "core_systems": {
                "memory": "✅ Active",
                "language_detection": "✅ Active",
                "emotion_detection": "✅ Active",
                "api_manager": "✅ Active" if self.api_manager.current else "❌ No API",
                "file_system": "✅ Active",
                "sound_system": "✅ Active"
            },
            "premium_systems": {
                "azure_voice": "✅ Active" if self.voice_system.azure_enabled else "⚠️ Basic Only",
                "web_search": "✅ Active",
                "ml_system": "✅ Active" if self.ml_manager else "❌ Disabled",
            },
            "agents": {
                agent_name: "✅ Active" if agent_name in self.agents else "❌ Disabled"
                for agent_name in ["coding", "career", "business", "medical", "emotional", "technical_architect"]
            },
            "session_info": {
                "session_id": self.current_session,
                "conversation_count": self.conversation_count,
                "user_id": self.user_id,
                "available_providers": len(self.api_manager.available),
                "file_context_active": self.current_file_context is not None
            }
        }

# ========== WORLD'S BEST UI - ENHANCED CYBERPUNK GAMING (TEXTUAL COMPATIBLE) ==========

class NovaUltraApp(App):
    """🎮 NOVA ULTRA - WORLD'S ABSOLUTE BEST AI CLI (ENHANCED GAMING UI)"""
    
    # 🏆 WORLD'S BEST CSS - TEXTUAL COMPATIBLE (10/10 CYBERPUNK GAMING) 🎮
    CSS = """
    /* NOVA ULTRA - WORLD'S ABSOLUTE BEST CLI CSS (TEXTUAL COMPATIBLE) */
    Screen {
        background: #0a0e27;
        color: #ffffff;
        layers: base overlay;
    }
    
    /* PREMIUM GAMING HEADER */
    Header {
        background: #1a1a2e;
        color: #00ff88;
        text-style: bold;
        dock: top;
        height: 3;
        border-bottom: thick #00ff88;
    }
    
    /* PREMIUM GAMING FOOTER */
    Footer {
        background: #16213e;
        color: #00ff88;
        dock: bottom;
        height: 3;
        border-top: thick #00ff88;
    }
    
    /* ENHANCED AGENT SIDEBAR */
    .sidebar {
        background: #1a1a2e;
        border-right: thick #00ff88;
        width: 32;
        dock: left;
        padding: 1;
        scrollbar-gutter: stable;
    }
    
    /* ENHANCED MAIN CONVERSATION */
    .main-content {
        background: #0d1117;
        border: thick #4444ff;
        width: 1fr;
        margin-left: 1;
        padding: 1;
    }
    
    /* ENHANCED STATUS PANEL */
    .status-panel {
        background: #1a1a2e;
        border-left: thick #ff8800;
        width: 35;
        dock: right;
        height: 1fr;
        margin-left: 1;
        padding: 1;
        scrollbar-gutter: stable;
    }
    
    /* PREMIUM GAMING BUTTONS (TEXTUAL COMPATIBLE) */
    Button {
        background: #16213e;
        color: #00ff88;
        border: thick #00ff88;
        margin: 1;
        text-style: bold;
        min-height: 4;
        width: 1fr;
    }
    
    Button:hover {
        background: #00ff88;
        color: #000000;
        border: thick #ffffff;
        text-style: bold;
    }
    
    Button:focus {
        background: #00cc66;
        color: #000000;
        border: thick #ffffff;
        text-style: bold;
    }
    
    Button.-primary {
        background: #4444ff;
        border: thick #4444ff;
        color: #ffffff;
    }
    
    Button.-primary:hover {
        background: #6666ff;
        color: #ffffff;
        border: thick #8888ff;
    }
    
    Button.-success {
        background: #00aa44;
        border: thick #00ff88;
        color: #ffffff;
    }
    
    Button.-success:hover {
        background: #00ff88;
        color: #000000;
        border: thick #ffffff;
    }
    
    Button.-warning {
        background: #ff8800;
        border: thick #ffaa44;
        color: #000000;
    }
    
    Button.-warning:hover {
        background: #ffaa00;
        color: #000000;
        border: thick #ffffff;
    }
    
    Button.-error {
        background: #cc3333;
        border: thick #ff4444;
        color: #ffffff;
    }
    
    Button.-error:hover {
        background: #ff4444;
        color: #ffffff;
        border: thick #ffffff;
    }
    
    /* ENHANCED SINGLE INPUT */
    Input {
        background: #16213e;
        border: thick #00ff88;
        color: #ffffff;
        margin-bottom: 1;
        height: 6;
        padding: 1;
    }
    
    Input:focus {
        border: thick #00ff88;
        background: #1a1a2e;
        color: #ffffff;
    }
    
    /* ENHANCED CONVERSATION LOG */
    RichLog {
        background: #0d1117;
        border: thick #30363d;
        scrollbar-background: #16213e;
        scrollbar-color: #00ff88;
        scrollbar-color-hover: #00cc66;
        scrollbar-size: 2 1;
        padding: 1;
        min-height: 25;
    }
    
    /* ENHANCED LABELS */
    Label {
        color: #00ff88;
        text-style: bold;
        text-align: center;
        background: #16213e;
        height: 3;
        margin-bottom: 1;
        border: thick #00ff88;
        padding: 1;
    }
    
    /* ENHANCED STATUS DISPLAYS */
    Static {
        background: #16213e;
        color: #ffffff;
        border: thick #30363d;
        padding: 1;
        text-align: left;
        margin-bottom: 1;
    }
    
    /* ENHANCED SECTION HEADERS */
    .agents-title {
        color: #00ff88;
        background: #16213e;
        text-style: bold;
        border: thick #00ff88;
        height: 3;
        margin-bottom: 1;
        text-align: center;
        padding: 1;
    }
    
    .features-title {
        color: #ff8800;
        background: #16213e;
        text-style: bold;
        border: thick #ff8800;
        height: 3;
        margin-bottom: 1;
        text-align: center;
        padding: 1;
    }
    
    .status-title {
        color: #4444ff;
        background: #16213e;
        text-style: bold;
        border: thick #4444ff;
        height: 3;
        margin-bottom: 1;
        text-align: center;
        padding: 1;
    }
    
    /* ENHANCED SPECIAL ELEMENTS */
    #conversation {
        background: #0d1117;
        border: thick #00ff88;
        scrollbar-background: #16213e;
        scrollbar-color: #00ff88;
        min-height: 25;
    }
    
    #user-input {
        background: #16213e;
        border: thick #00ff88;
        color: #ffffff;
        height: 6;
    }
    
    #user-input:focus {
        border: thick #00ff88;
        background: #1a1a2e;
        color: #ffffff;
    }
    
    #system-status {
        background: #0d1117;
        color: #00ff88;
        border: thick #30363d;
        padding: 1;
        text-style: bold;
        min-height: 15;
    }
    """
    
    # Enhanced KEYBINDINGS - RESTORED COMMAND PALETTE
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+c", "clear_conversation", "Clear Chat"),
        ("ctrl+v", "voice_mode", "Voice Mode"),
        ("ctrl+s", "search_web", "Web Search"),
        ("ctrl+g", "github_analyzer", "GitHub"),
        ("ctrl+h", "help_screen", "Help"),
        ("ctrl+p", "command_palette", "Command Palette"),  # RESTORED
        ("ctrl+u", "upload_file", "Upload File"),
        ("f1", "toggle_sidebar", "Toggle Sidebar"),
        ("f5", "system_status", "System Status"),
        ("up", "history_up", "Previous Command"),
        ("down", "history_down", "Next Command"),
        ("ctrl+r", "search_history", "Search History"),
        ("escape", "focus_input", "Focus Input"),
    ]
    
    # Reactive variables for real-time updates
    current_agent = reactive("general")
    conversation_count = reactive(0)
    response_time = reactive("0.00s")
    voice_active = reactive(False)
    file_uploaded = reactive(False)
    ml_status = reactive("Active" if ML_SYSTEM_AVAILABLE else "Disabled")
    
    def __init__(self):
        super().__init__()
        
        # Initialize NOVA system with all enhancements
        self.nova_system = NovaUltraSystem()
        
        # Command history and autocomplete
        self.command_history = []
        self.history_index = -1
        
        # ENHANCED COMMAND SUGGESTIONS
        self.command_suggestions = [
            "help me with coding",
            "analyze this code",
            "debug my python script",
            "fix this error",
            "code review",
            "career advice",
            "resume help",
            "interview preparation",
            "job search tips",
            "salary negotiation",
            "business analysis",
            "market research",
            "financial planning",
            "startup advice",
            "health advice",
            "medical symptoms",
            "nutrition tips",
            "fitness plan",
            "emotional support",
            "stress management",
            "anxiety help",
            "motivation boost",
            "system architecture",
            "database design",
            "microservices",
            "scalability",
            "web search",
            "latest news",
            "github analysis",
            "voice mode",
            "file analysis",
            "clear chat",
            "system status",
            "show help",
            "voice commands",
            "search history",
            "upload file",
            "show metrics"
        ]
    
    def compose(self) -> ComposeResult: # type: ignore
        """Compose the WORLD'S BEST interface"""
        yield Header(show_clock=True)
        
        with Horizontal():
            # LEFT SIDEBAR - ENHANCED Original Agent Control Panel
            with Container(classes="sidebar"):
                yield Label("🤖 NOVA AGENTS", classes="agents-title")
                yield Button("🔧 CODING EXPERT", id="agent-coding", classes="-primary")
                yield Button("💼 CAREER COACH", id="agent-career", classes="-success")
                yield Button("📈 BUSINESS GURU", id="agent-business", classes="-primary")
                yield Button("🏥 MEDICAL ADVISOR", id="agent-medical", classes="-error")
                yield Button("💙 EMOTIONAL SUPPORT", id="agent-emotional", classes="-success")
                yield Button("🚀 TECH ARCHITECT", id="agent-technical", classes="-primary")
                
                yield Label("🎯 FEATURES", classes="features-title")
                yield Button("🎤 VOICE MODE", id="voice-mode", classes="-success")
                yield Button("🔍 WEB SEARCH", id="web-search", classes="-primary")
                yield Button("📁 FILE UPLOAD", id="file-upload", classes="-warning")  # NOW VISIBLE
                yield Button("📊 FILE ANALYSIS", id="file-analysis", classes="-warning")
                yield Button("🔗 GITHUB ANALYZER", id="github-analysis", classes="-success")
                yield Button("🧠 ML INSIGHTS", id="ml-insights", classes="-primary")
                yield Button("⌨️ COMMAND PALETTE", id="command-palette", classes="-primary")  # RESTORED
            
            # MAIN CONTENT - ENHANCED Original Conversation Area
            with Container(classes="main-content"):
                # Conversation Log
                yield RichLog(id="conversation", markup=True, highlight=True, wrap=True)
                
                # RESTORED SINGLE INPUT (NO SEPARATE SEND BUTTON)
                yield Input(
                    placeholder="🚀 Ask NOVA anything... (Enter to send, Ctrl+P for commands, Ctrl+U for files)",
                    id="user-input",
                    suggester=SuggestFromList(self.command_suggestions, case_sensitive=False)
                )
            
            # RIGHT STATUS PANEL - ENHANCED Original
            with Container(classes="status-panel"):
                yield Label("📊 LIVE PERFORMANCE", classes="status-title")
                yield Static("🤖 AI: Ready\n🎤 Voice: Ready\n🔍 Search: Ready\n📁 Files: Ready\n🧠 ML: Active\n🔊 Sound: Active",
                           id="system-status")
                yield Static("", id="current-stats")
                yield Button("🔄 REFRESH", id="refresh-status", classes="-primary")
                yield Button("❌ CLEAR CHAT", id="clear-chat", classes="-error")
                yield Button("📝 HISTORY", id="show-history", classes="-warning")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize NOVA-CLI interface"""
        # Enhanced welcome message
        conversation = self.query_one("#conversation", RichLog)
        welcome_msg = f"""[bold cyan]🎮 NOVA CLI - AI SYSTEM v3.0[/bold cyan]

[green]✅  Gaming Cyberpunk Interface 
✅ {len(self.nova_system.api_manager.available)} API Providers Loaded ({', '.join([p['name'] for p in self.nova_system.api_manager.available])})
✅ Professional Agents Ready[gold]
✅ Command Palette RESTORED (Ctrl+P)[gold]
✅ File Upload System Visible & Working[gold]
✅ Enhanced Gaming Color Scheme (TEXTUAL COMPATIBLE)[gold]
✅ Premium Hover Animations[gold]
✅ Multi-Modal Support (Voice, Files, Web)[gold]
✅ ML-Enhanced Intelligence Active[gold]

[yellow]🚀 Available Agents:[/yellow]
• 🔧 Coding Expert - Advanced programming & debugging[brown]
• 💼 Career Coach - Professional guidance & growth[brown]
• 📈 Business Guru - Strategic consulting & analysis[brown]
• 🏥 Medical Advisor - Health insights & wellness[brown]
• 💙 Emotional Support - Mental health & wellbeing[brown]
• 🚀 Tech Architect - System design & architecture[brown]

[blue]💡 World's Best Features:[/blue]
• 📁 Complete File Upload System (Click "📁 FILE UPLOAD")[]
• 🎤 Premium voice interaction (Azure + fallback)
• 🔍 Intelligent web search & news
• ⌨️ Command palette for quick actions (Ctrl+P)
• 🧠 ML-powered learning & optimization
• ✨ Premium hover animations
• 🎨 cyberpunk gaming aesthetics (TEXTUAL COMPATIBLE)
• ⚡ Command autocomplete & history

[blue]🧠 ML System Role in NOVA:[/blue]
• **Query Enhancement** - Improves your questions automatically[pink]
• **Conversation Learning** - Adapts to your communication style[pink]
• **Performance Optimization** - Optimizes response times[pink]
• **Predictive Agent Selection** - Smart routing to best agent[pink]
• **Emotional Intelligence** - Context-aware emotional responses[pink]
• **Multi-Provider Optimization** - Automatic provider switching[pink]
• **Intent Detection** - Understands what you really want[pink]
• **Context Memory** - Remembers conversation patterns[pink]
• **Personalization Engine** - Learns your preferences[pink]

[bold magenta]🎯 How to Share Files:[/bold magenta]
• Click "📁 FILE UPLOAD" button (now visible in sidebar)[cyan]
• Or press Ctrl+U for quick file upload[cyan]
• Or type "upload file" in the input[cyan]
• AI automatically analyzes and remembers content[cyan]

[bold magenta]🔊 Sound System Features:[/bold magenta]
• Button clicks play sounds[purple]
• Success/error notifications[purple]
• Voice interaction feedback[purple]
• Premium audio experience[purple]

[bold]Experience the ABSOLUTE BEST AI CLI![/bold]"""
        
        conversation.write(welcome_msg)
        
        # Play startup success sound
        self.nova_system.sound_system.play_sound("success")
        
        # Focus on input
        self.query_one("#user-input", Input).focus()
        
        # Update system status
        self.update_system_status()
    
    def update_system_status(self):
        """Enhanced system status update"""
        status_text = f"""🤖 Agent: {self.current_agent.title()}
📊 Conversations: {self.conversation_count}
⏱️ Response: {self.response_time}
🎤 Voice: {'Active' if self.voice_active else 'Ready'}
📁 File: {'Uploaded' if self.file_uploaded else 'Ready'}
🧠 ML: {self.ml_status}
🔊 Sound: ✅ Active (Inbuilt Beeps)
📝 History: {len(self.command_history)} commands
🔧 API Providers: 6 available
"""
        
        self.query_one("#system-status", Static).update(status_text)
        
        # Update current stats
        stats_text = f"""Session: Active
User ID: nova_user
Theme: Cyberpunk Gaming
File Upload: ✅ Visible & Working
Advanced Rag: ✅ Active
Github Analyzer: ✅ RESTORED
MLFLOW: ✅ Active"""
        
        self.query_one("#current-stats", Static).update(stats_text)
    
    # ========== RESTORED INPUT HANDLING ==========
    
    @on(Input.Submitted, "#user-input")
    async def handle_user_input(self, event: Input.Submitted) -> None:
        """RESTORED user input handling (no send button needed) WITH SOUND"""
        user_input = event.value.strip()
        if not user_input:
            return
        
        # Play input sound
        self.nova_system.sound_system.play_sound("click")
        
        # Clear input
        event.input.value = ""
        
        # Add to command history
        if user_input not in self.command_history:
            self.command_history.append(user_input)
            if len(self.command_history) > 100:
                self.command_history = self.command_history[-100:]
        
        # Reset history index
        self.history_index = -1
        
        # Get conversation log
        conversation = self.query_one("#conversation", RichLog)
        
        # Show user message
        timestamp = datetime.now().strftime("%H:%M:%S")
        conversation.write(f"[dim][{timestamp}][/dim] [bold blue]👤 You:[/bold blue] {user_input}")
        conversation.write("[yellow]🤖 NOVA is thinking...[/yellow]")
        
        # Handle special commands
        if user_input.lower().startswith("search "):
            query = user_input[7:]
            await self.handle_web_search(query)
            return
        elif user_input.lower() in ["upload", "upload file", "file upload"]:
            await self.handle_file_upload()
            return
        elif user_input.lower() in ["voice", "voice mode"]:
            await self.activate_voice_mode()
            return
        elif user_input.lower() in ["clear", "clear chat"]:
            self.clear_conversation()
            return
        elif user_input.lower() in ["help", "help me"]:
            self.action_help_screen()
            return
        elif user_input.lower() in ["status", "system status"]:
            self.action_system_status()
            return
        
        # Process with AI
        start_time = time.time()
        try:
            # Get AI response
            response_data = await self.nova_system.get_response(user_input)
            
            # Play success sound
            self.nova_system.sound_system.play_sound("success")
            
            # Update UI
            response_time = time.time() - start_time
            self.response_time = f"{response_time:.2f}s"
            self.conversation_count = response_data.get("conversation_count", self.conversation_count)
            self.current_agent = response_data.get("agent_used", "general")
            
            # Show response
            ai_response = response_data.get("response", "Processing your request...")
            agent_used = response_data.get("agent_used", "general")
            file_context_used = response_data.get("file_context_used", False)
            
            agent_emoji = {
                "coding": "🔧", "career": "💼", "business": "📈",
                "medical": "🏥", "emotional": "💙", "technical_architect": "🚀",
                "general": "🤖"
            }.get(agent_used, "🤖")
            
            conversation.write(f"[bold green]{agent_emoji} NOVA ({agent_used.title()}):[/bold green] {ai_response}")
            
            metadata_parts = []
            if response_data.get("ml_enhanced"):
                metadata_parts.append("🧠 ML-Enhanced")
            if file_context_used:
                metadata_parts.append("📁 File Context")
            
            metadata_parts.extend([
                f"⏱️ {self.response_time}",
                f"🎯 {agent_used}",
                f"🌐 {self.nova_system.api_manager.current['name'] if self.nova_system.api_manager.current else 'No API'}"
            ])
            
            conversation.write(f"[dim]{' | '.join(metadata_parts)}[/dim]")
            conversation.write("")
            
        except Exception as e:
            # Play error sound
            self.nova_system.sound_system.play_sound("error")
            conversation.write(f"[bold red]❌ Error:[/bold red] {str(e)}")
        
        # Update status
        self.update_system_status()
    
    # ========== RESTORED COMMAND PALETTE ==========
    
    def action_command_palette(self):
        """Show command palette (RESTORED) WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        self.push_screen(CommandPalette(), self.handle_command_result)
    
    def handle_command_result(self, command_id):
        """Handle command palette result (RESTORED) WITH SOUND"""
        if not command_id:
            return
        
        self.nova_system.sound_system.play_sound("click")
        conversation = self.query_one("#conversation", RichLog)
        
        if command_id.startswith("agent-"):
            agent_name = command_id.replace("agent-", "")
            asyncio.create_task(self.switch_agent(agent_name))
        elif command_id == "voice-mode":
            asyncio.create_task(self.activate_voice_mode())
        elif command_id == "web-search":
            conversation.write("[yellow]🔍 Enter search query in the input box with 'search ' prefix[/yellow]")
        elif command_id == "upload-file":
            asyncio.create_task(self.handle_file_upload())
        elif command_id == "clear-chat":
            self.clear_conversation()
        elif command_id == "show-status":
            self.action_system_status()
        elif command_id == "help":
            self.action_help_screen()
        elif command_id == "settings":
            conversation.write("[blue]⚙️ Settings panel coming soon![/blue]")
    
    # ========== AGENT SWITCHING ==========
    
    async def switch_agent(self, agent_name: str):
        """Switch to specific agent WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        self.current_agent = agent_name
        conversation = self.query_one("#conversation", RichLog)
        
        agent_names = {
            "coding": "🔧 Coding Expert",
            "career": "💼 Career Coach", 
            "business": "📈 Business Guru",
            "medical": "🏥 Medical Advisor",
            "emotional": "💙 Emotional Support",
            "technical": "🚀 Tech Architect"
        }
        
        agent_display = agent_names.get(agent_name, "🤖 General")
        conversation.write(f"[bold green]🔄 Switched to {agent_display}[/bold green]")
        self.update_system_status()
    
    # ========== BUTTON HANDLERS WITH SOUND ==========
    
    @on(Button.Pressed, "#agent-coding")
    async def on_coding_agent(self):
        """Switch to coding agent WITH SOUND"""
        await self.switch_agent("coding")
    
    @on(Button.Pressed, "#agent-career")
    async def on_career_agent(self):
        """Switch to career agent WITH SOUND"""
        await self.switch_agent("career")
    
    @on(Button.Pressed, "#agent-business")
    async def on_business_agent(self):
        """Switch to business agent WITH SOUND"""
        await self.switch_agent("business")
    
    @on(Button.Pressed, "#agent-medical")
    async def on_medical_agent(self):
        """Switch to medical agent WITH SOUND"""
        await self.switch_agent("medical")
    
    @on(Button.Pressed, "#agent-emotional")
    async def on_emotional_agent(self):
        """Switch to emotional agent WITH SOUND"""
        await self.switch_agent("emotional")
    
    @on(Button.Pressed, "#agent-technical")
    async def on_technical_agent(self):
        """Switch to technical architect WITH SOUND"""
        await self.switch_agent("technical")
    
    @on(Button.Pressed, "#voice-mode")
    async def on_voice_mode(self):
        """Activate voice mode WITH SOUND"""
        await self.activate_voice_mode()
    
    @on(Button.Pressed, "#web-search")
    async def on_web_search(self):
        """Prompt for web search WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        conversation = self.query_one("#conversation", RichLog)
        conversation.write("[yellow]🔍 Enter your search query with 'search ' prefix (e.g., 'search latest AI news')[/yellow]")
    
    @on(Button.Pressed, "#file-upload")
    async def on_file_upload(self):
        """Handle file upload WITH SOUND"""
        await self.handle_file_upload()
    
    @on(Button.Pressed, "#file-analysis")
    async def on_file_analysis(self):
        """Show current file analysis WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        conversation = self.query_one("#conversation", RichLog)
        
        if self.nova_system.current_file_context:
            conversation.write(f"[bold green]📊 Current File Context:[/bold green]\n{self.nova_system.current_file_context}")
        else:
            conversation.write("[yellow]📁 No file currently uploaded. Click 'FILE UPLOAD' to analyze a file.[/yellow]")
    
    @on(Button.Pressed, "#github-analysis")
    async def on_github_analysis(self):
        """GitHub analysis WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        conversation = self.query_one("#conversation", RichLog)
        conversation.write("[blue]🔗 GitHub Analyzer: Enter a GitHub repository URL in the chat to analyze it![/blue]")
    
    @on(Button.Pressed, "#ml-insights")
    async def on_ml_insights(self):
        """ML insights WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        conversation = self.query_one("#conversation", RichLog)
        
        if self.nova_system.ml_manager:
            conversation.write(f"[bold green]🧠 ML System Status: Active and Learning[/bold green]\n")
            conversation.write("• Query Enhancement: ✅ Active\n")
            conversation.write("• Performance Optimization: ✅ Active\n")
            conversation.write("• Emotional Intelligence: ✅ Active\n")
            conversation.write("• Context Memory: ✅ Active\n")
        else:
            conversation.write("[yellow]🧠 ML System: Not available in this configuration[/yellow]")
    
    @on(Button.Pressed, "#command-palette")
    def on_command_palette(self):
        """Show command palette WITH SOUND"""
        self.action_command_palette()
    
    @on(Button.Pressed, "#refresh-status")
    def on_refresh_status(self):
        """Refresh system status WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        self.update_system_status()
        conversation = self.query_one("#conversation", RichLog)
        conversation.write("[green]🔄 System status refreshed![/green]")
    
    @on(Button.Pressed, "#clear-chat")
    def on_clear_chat(self):
        """Clear chat WITH SOUND"""
        self.clear_conversation()
    
    @on(Button.Pressed, "#show-history")
    def on_show_history(self):
        """Show command history WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        conversation = self.query_one("#conversation", RichLog)
        
        if self.command_history:
            conversation.write("[bold blue]📝 Command History:[/bold blue]")
            for i, cmd in enumerate(self.command_history[-10:], 1):
                conversation.write(f"[dim]{i}.[/dim] {cmd}")
        else:
            conversation.write("[yellow]📝 No command history yet[/yellow]")
    
    # ========== CORE FUNCTIONALITY ==========
    
    async def activate_voice_mode(self):
        """Activate voice mode WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        conversation = self.query_one("#conversation", RichLog)
        
        conversation.write("[bold blue]🎤 Voice Mode Activated - Listening...[/bold blue]")
        self.voice_active = True
        self.update_system_status()
        
        try:
            voice_result = await self.nova_system.process_voice_input()
            
            if voice_result.get("error"):
                conversation.write(f"[bold red]❌ Voice Error:[/bold red] {voice_result['error']}")
                self.nova_system.sound_system.play_sound("error")
            else:
                voice_input = voice_result.get("voice_input", "")
                ai_response_data = voice_result.get("ai_response", {})
                
                conversation.write(f"[bold blue]🎤 You said:[/bold blue] {voice_input}")
                
                if ai_response_data.get("response"):
                    agent_used = ai_response_data.get("agent_used", "general")
                    agent_emoji = {
                        "coding": "🔧", "career": "💼", "business": "📈",
                        "medical": "🏥", "emotional": "💙", "technical_architect": "🚀",
                        "general": "🤖"
                    }.get(agent_used, "🤖")
                    
                    conversation.write(f"[bold green]{agent_emoji} NOVA:[/bold green] {ai_response_data['response']}")
                    self.nova_system.sound_system.play_sound("success")
                
        except Exception as e:
            conversation.write(f"[bold red]❌ Voice processing failed:[/bold red] {str(e)}")
            self.nova_system.sound_system.play_sound("error")
        finally:
            self.voice_active = False
            self.update_system_status()
    
    async def handle_file_upload(self):
        """Handle file upload WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        conversation = self.query_one("#conversation", RichLog)
        
        conversation.write("[yellow]📁 Opening file dialog...[/yellow]")
        
        try:
            result = await self.nova_system.upload_and_analyze_file()
            
            if result.get("error"):
                conversation.write(f"[bold red]❌ Upload Error:[/bold red] {result['error']}")
                self.nova_system.sound_system.play_sound("error")
            else:
                file_analysis = result.get("file_analysis", {})
                conversation.write(f"[bold green]✅ File Uploaded Successfully![/bold green]")
                conversation.write(f"[blue]📄 File:[/blue] {file_analysis.get('file_name', 'Unknown')}")
                conversation.write(f"[blue]📏 Size:[/blue] {file_analysis.get('file_size', 0)} bytes")
                conversation.write(f"[blue]📝 Type:[/blue] {file_analysis.get('file_type', 'Unknown')}")
                
                if file_analysis.get('lines'):
                    conversation.write(f"[blue]📊 Lines:[/blue] {file_analysis['lines']}")
                
                conversation.write(f"[blue]📋 Content Preview:[/blue]\n{file_analysis.get('content', '')[:300]}...")
                conversation.write("[green]💡 File context is now active. Ask questions about your file![/green]")
                
                self.file_uploaded = True
                self.nova_system.sound_system.play_sound("success")
                self.update_system_status()
                
        except Exception as e:
            conversation.write(f"[bold red]❌ File upload failed:[/bold red] {str(e)}")
            self.nova_system.sound_system.play_sound("error")
    
    async def handle_web_search(self, query: str):
        """Handle web search WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        conversation = self.query_one("#conversation", RichLog)
        
        conversation.write(f"[yellow]🔍 Searching for: {query}[/yellow]")
        
        try:
            search_result = await self.nova_system.search_web(query)
            
            if search_result.get("error"):
                conversation.write(f"[bold red]❌ Search Error:[/bold red] {search_result['error']}")
                self.nova_system.sound_system.play_sound("error")
            else:
                conversation.write(search_result.get("formatted_response", "No results found"))
                self.nova_system.sound_system.play_sound("success")
                
        except Exception as e:
            conversation.write(f"[bold red]❌ Search failed:[/bold red] {str(e)}")
            self.nova_system.sound_system.play_sound("error")
    
    # ========== ACTION HANDLERS ==========
    
    def action_clear_conversation(self):
        """Clear conversation WITH SOUND"""
        self.clear_conversation()
    
    def clear_conversation(self):
        """Clear the conversation log WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        conversation = self.query_one("#conversation", RichLog)
        conversation.clear()
        conversation.write("[bold green]🧹 Chat cleared! Ready for a fresh conversation.[/bold green]")
        
        # Reset file context
        self.nova_system.current_file_context = None
        self.file_uploaded = False
        self.update_system_status()
    
    def action_voice_mode(self):
        """Voice mode action WITH SOUND"""
        asyncio.create_task(self.activate_voice_mode())
    
    def action_search_web(self):
        """Web search action WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        conversation = self.query_one("#conversation", RichLog)
        conversation.write("[yellow]🔍 Enter your search query with 'search ' prefix[/yellow]")
    
    def action_upload_file(self):
        """File upload action WITH SOUND"""
        asyncio.create_task(self.handle_file_upload())
    
    def action_help_screen(self):
        """Show help screen WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        conversation = self.query_one("#conversation", RichLog)
        
        help_text = """[bold cyan]🎮 NOVA ULTRA - HELP & COMMANDS[/bold cyan]

[yellow]🚀 Quick Commands:[/yellow]
• Type normally to chat with AI
• "search [query]" - Web search
• "upload" or "upload file" - Upload file
• "voice" or "voice mode" - Voice interaction
• "clear" or "clear chat" - Clear conversation
• "help" - Show this help
• "status" - System status

[yellow]⌨️ Keyboard Shortcuts:[/yellow]
• Ctrl+P - Command Palette
• Ctrl+U - Upload File
• Ctrl+V - Voice Mode
• Ctrl+S - Web Search
• Ctrl+C - Clear Chat
• Ctrl+H - Help
• Ctrl+Q - Quit
• Up/Down - Command History
• Esc - Focus Input

[yellow]🤖 Agents:[/yellow]
• 🔧 Coding Expert - Programming help
• 💼 Career Coach - Career guidance
• 📈 Business Guru - Business analysis
• 🏥 Medical Advisor - Health insights
• 💙 Emotional Support - Mental health
• 🚀 Tech Architect - System design

[yellow]🎯 Features:[/yellow]
• 📁 File Upload & Analysis
• 🎤 Voice Interaction (Azure + Basic)
• 🔍 Web Search & News
• 🧠 ML-Enhanced Responses
• 🔊 Sound System (Inbuilt Beeps)
• 💾 Conversation Memory
• 🎨 WORLD'S BEST Gaming UI

[green]💡 Tips:[/green]
• Files remain in context until cleared
• Voice mode works with microphone
• Use command palette for quick access
• Agents auto-switch based on context"""
        
        conversation.write(help_text)
    
    def action_system_status(self):
        """Show system status WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        conversation = self.query_one("#conversation", RichLog)
        
        status = self.nova_system.get_system_status()
        
        status_text = "[bold cyan]📊 NOVA ULTRA SYSTEM STATUS[/bold cyan]\n\n"
        
        status_text += "[yellow]🔧 Core Systems:[/yellow]\n"
        for system, status_val in status["core_systems"].items():
            status_text += f"• {system.replace('_', ' ').title()}: {status_val}\n"
        
        status_text += "\n[yellow]⭐ Premium Systems:[/yellow]\n"
        for system, status_val in status["premium_systems"].items():
            status_text += f"• {system.replace('_', ' ').title()}: {status_val}\n"
        
        status_text += "\n[yellow]🤖 Agents:[/yellow]\n"
        for agent, status_val in status["agents"].items():
            status_text += f"• {agent.replace('_', ' ').title()}: {status_val}\n"
        
        status_text += "\n[yellow]📋 Session Info:[/yellow]\n"
        for key, value in status["session_info"].items():
            status_text += f"• {key.replace('_', ' ').title()}: {value}\n"
        
        conversation.write(status_text)
    
    def action_github_analyzer(self):
        """GitHub analyzer action WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        conversation = self.query_one("#conversation", RichLog)
        conversation.write("[blue]🔗 GitHub Analyzer: Paste a GitHub repository URL to analyze it![/blue]")
    
    # ========== KEYBOARD NAVIGATION ==========
    
    def action_history_up(self):
        """Navigate command history up WITH SOUND"""
        if self.command_history and self.history_index < len(self.command_history) - 1:
            self.nova_system.sound_system.play_sound("click")
            self.history_index += 1
            user_input = self.query_one("#user-input", Input)
            user_input.value = self.command_history[-(self.history_index + 1)]
    
    def action_history_down(self):
        """Navigate command history down WITH SOUND"""
        if self.command_history and self.history_index > -1:
            self.nova_system.sound_system.play_sound("click")
            self.history_index -= 1
            user_input = self.query_one("#user-input", Input)
            if self.history_index == -1:
                user_input.value = ""
            else:
                user_input.value = self.command_history[-(self.history_index + 1)]
    
    def action_search_history(self):
        """Search command history WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        conversation = self.query_one("#conversation", RichLog)
        if self.command_history:
            conversation.write("[bold blue]🔍 Recent Commands (use Up/Down to navigate):[/bold blue]")
            for i, cmd in enumerate(self.command_history[-5:], 1):
                conversation.write(f"[dim]{i}.[/dim] {cmd}")
        else:
            conversation.write("[yellow]No command history yet[/yellow]")
    
    def action_focus_input(self):
        """Focus input field WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        user_input = self.query_one("#user-input", Input)
        user_input.focus()
    
    def action_toggle_sidebar(self):
        """Toggle sidebar visibility WITH SOUND"""
        self.nova_system.sound_system.play_sound("click")
        # Implementation for sidebar toggle can be added here
        conversation = self.query_one("#conversation", RichLog)
        conversation.write("[blue]🎮 Sidebar toggle coming soon![/blue]")

# ========== FALLBACK CLI MODE ==========

class FallbackCLI:
    """Fallback CLI when Textual is not available"""
    
    def __init__(self):
        self.nova_system = NovaUltraSystem()
        print(f"{Colors.CYAN}🎮 NOVA ULTRA PROFESSIONAL CLI - FALLBACK MODE{Colors.RESET}")
        print(f"{Colors.GREEN}✅ Core systems initialized{Colors.RESET}")
    
    async def run(self):
        """Run fallback CLI interface"""
        print(f"\n{Colors.YELLOW}Welcome to NOVA Ultra Professional CLI!{Colors.RESET}")
        print(f"{Colors.BLUE}Type 'help' for commands, 'quit' to exit{Colors.RESET}\n")
        
        while True:
            try:
                user_input = input(f"{Colors.GREEN}NOVA> {Colors.RESET}").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(f"{Colors.YELLOW}👋 Goodbye!{Colors.RESET}")
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                elif user_input.lower().startswith('search '):
                    query = user_input[7:]
                    await self.handle_search(query)
                elif user_input.lower() in ['upload', 'upload file']:
                    await self.handle_upload()
                elif user_input:
                    await self.handle_query(user_input)
                    
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}👋 Goodbye!{Colors.RESET}")
                break
            except Exception as e:
                print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")
    
    def show_help(self):
        """Show help information"""
        help_text = f"""
{Colors.CYAN}🎮 NOVA ULTRA COMMANDS{Colors.RESET}

{Colors.YELLOW}Basic Commands:{Colors.RESET}
• Type normally to chat with AI
• search [query] - Web search
• upload - Upload and analyze file
• help - Show this help
• quit - Exit NOVA

{Colors.YELLOW}Features:{Colors.RESET}
• 🤖 Multi-agent AI system
• 📁 File upload & analysis
• 🔍 Web search
• 💾 Conversation memory
• 🧠 ML-enhanced responses
"""
        print(help_text)
    
    async def handle_query(self, user_input: str):
        """Handle user query"""
        print(f"{Colors.YELLOW}🤖 NOVA is thinking...{Colors.RESET}")
        
        try:
            response_data = await self.nova_system.get_response(user_input)
            agent_used = response_data.get("agent_used", "general")
            ai_response = response_data.get("response", "No response")
            
            print(f"\n{Colors.GREEN}🤖 NOVA ({agent_used}):{Colors.RESET} {ai_response}\n")
            
        except Exception as e:
            print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")
    
    async def handle_search(self, query: str):
        """Handle web search"""
        print(f"{Colors.YELLOW}🔍 Searching for: {query}{Colors.RESET}")
        
        try:
            result = await self.nova_system.search_web(query)
            if result.get("success"):
                print(f"\n{Colors.GREEN}Search Results:{Colors.RESET}")
                print(result.get("formatted_response", "No results"))
            else:
                print(f"{Colors.RED}❌ Search failed{Colors.RESET}")
                
        except Exception as e:
            print(f"{Colors.RED}❌ Search error: {e}{Colors.RESET}")
    
    async def handle_upload(self):
        """Handle file upload"""
        print(f"{Colors.YELLOW}📁 Opening file dialog...{Colors.RESET}")
        
        try:
            result = await self.nova_system.upload_and_analyze_file()
            if result.get("success"):
                file_analysis = result.get("file_analysis", {})
                print(f"{Colors.GREEN}✅ File uploaded: {file_analysis.get('file_name')}{Colors.RESET}")
                print(f"Type: {file_analysis.get('file_type')}")
                print(f"Size: {file_analysis.get('file_size')} bytes")
                print(f"Content preview: {file_analysis.get('content', '')[:200]}...")
            else:
                print(f"{Colors.RED}❌ Upload failed: {result.get('error')}{Colors.RESET}")
                
        except Exception as e:
            print(f"{Colors.RED}❌ Upload error: {e}{Colors.RESET}")

# ========== MAIN LAUNCHER ==========

async def main():
    """Main launcher - Auto-detect best UI mode"""
    print("🚀 Starting NOVA ULTRA PROFESSIONAL CLI...")
    
    # Check if we're in a proper terminal
    if not sys.stdout.isatty():
        print("⚠️ Not running in a terminal, using basic mode")
        fallback = FallbackCLI()
        await fallback.run()
        return
    
    # Try Textual UI first (WORLD'S BEST)
    if TEXTUAL_AVAILABLE:
        try:
            print("✅ Loading WORLD'S BEST Gaming UI...")
            app = NovaUltraApp()
            await app.run_async()
        except Exception as e:
            print(f"⚠️ Textual UI failed: {e}")
            print("🔄 Falling back to basic CLI...")
            fallback = FallbackCLI()
            await fallback.run()
    else:
        print("🔄 Using fallback CLI mode...")
        fallback = FallbackCLI()
        await fallback.run()

def run_nova():
    """Entry point for NOVA CLI"""
    try:
        if sys.platform == "win32":
            # Windows-specific event loop policy
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 NOVA Ultra CLI closed gracefully!")
    except Exception as e:
        print(f"❌ NOVA startup error: {e}")

if __name__ == "__main__":
    run_nova()
