"""
🚀 ULTIMATE SMART REPOSITORY INGESTION ENGINE
"""

import os
import sys
import time
import json
import logging
import warnings
import hashlib
import tempfile
import shutil
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Union
from collections import defaultdict, deque
from urllib.parse import urlparse
import asyncio
import aiohttp
import concurrent.futures
from dotenv import load_dotenv
import codecs
# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# if sys.platform.startswith('win'):
#     # Use UTF-8 encoding for Windows console
#     sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
#     sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Enhanced logging setup
def setup_logging():
    """Setup logging with Windows compatibility"""
    try:
        if sys.platform.startswith('win'):
            # Don't detach buffers on Windows - just configure properly
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
                sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
        
        # Clear existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler('ingest.log', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Configure root logger
        logging.root.addHandler(console_handler)
        logging.root.addHandler(file_handler)
        logging.root.setLevel(logging.INFO)
        
    except Exception as e:
        # Fallback - just use print statements
        print(f"⚠️ Logging setup failed: {e}")

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
# ========== SMART MODEL CONFIGURATION ==========

class SmartIngestionModelConfig:
    """🧠 Smart Model Configuration for Repository Processing"""
    
    @staticmethod
    def get_processing_models():
        """Get best free models for different ingestion tasks"""
        return {
            # ============ CODE ANALYSIS MODELS ============
            "code_analysis": {
                "primary": {
                    "name": "DeepSeek_Coder_V3", 
                    "url": "https://openrouter.ai/api/v1/chat/completions",
                    "model": "deepseek/deepseek-coder-v2.5:free",
                    "provider": "OPENROUTER",
                    "context": 32000,
                    "specialty": "Code understanding, structure analysis, documentation extraction"
                },
                "secondary": {
                    "name": "Groq_Code_Analysis",
                    "url": "https://api.groq.com/openai/v1/chat/completions", 
                    "model": "llama-3-groq-70b-8192-tool-use-preview",
                    "provider": "GROQ",
                    "context": 8192,
                    "specialty": "Fast code processing and metadata extraction"
                }
            },
            
            # ============ TEXT PROCESSING MODELS ============
            "text_processing": {
                "primary": {
                    "name": "Cerebras_Text_Processor",
                    "url": "https://api.cerebras.ai/v1/chat/completions",
                    "model": "llama3.3-70b",
                    "provider": "CEREBRAS",
                    "context": 128000,
                    "specialty": "Ultra-fast text analysis and summarization"
                },
                "secondary": {
                    "name": "Groq_Lightning",
                    "url": "https://api.groq.com/openai/v1/chat/completions",
                    "model": "llama-3.1-8b-instant",
                    "provider": "GROQ",
                    "context": 131072,
                    "specialty": "Fast text processing for large documents"
                }
            },
            
            # ============ METADATA EXTRACTION ============
            "metadata_extraction": {
                "primary": {
                    "name": "DeepSeek_Chat_V3",
                    "url": "https://openrouter.ai/api/v1/chat/completions",
                    "model": "deepseek/deepseek-chat-v3:free",
                    "provider": "OPENROUTER",
                    "context": 64000,
                    "specialty": "Intelligent metadata and structure extraction"
                }
            }
        }

class IntelligentFileProcessor:
    """🎯 Intelligent File Processing with Smart Model Selection"""
    
    def __init__(self):
        self.models = SmartIngestionModelConfig.get_processing_models()
        self.processing_stats = defaultdict(int)
        
    def select_model_for_file(self, file_path: str, content_preview: str = "") -> Dict:
        """Select best model based on file type and content"""
        ext = os.path.splitext(file_path)[1].lower()
        
        # Code files need specialized analysis
        code_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.php', '.rb', '.swift', '.kt']
        if ext in code_extensions:
            return self.models["code_analysis"]["primary"]
        
        # Large text files need fast processing
        text_extensions = ['.md', '.txt', '.html', '.css', '.json', '.yaml', '.yml']
        if ext in text_extensions:
            if len(content_preview) > 5000:  # Large files
                return self.models["text_processing"]["primary"]  # Cerebras for speed
            else:
                return self.models["metadata_extraction"]["primary"]  # DeepSeek for quality
        
        # Default to metadata extraction
        return self.models["metadata_extraction"]["primary"]

# ========== DEPENDENCIES AND IMPORTS ==========

try:
    import git
    GIT_AVAILABLE = True
    logger.info("✅ Git support available")
except ImportError:
    GIT_AVAILABLE = False
    logger.warning("⚠️ Git not available - install with: pip install gitpython")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
    logger.info("✅ ChromaDB available")
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.error("❌ ChromaDB not available - install with: pip install chromadb")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
    from langchain_community.document_loaders import (
        TextLoader, UnstructuredFileLoader, PyPDFLoader, 
        UnstructuredHTMLLoader, CSVLoader, JSONLoader
    )
    from langchain.schema import Document

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info("✅ Using langchain-huggingface (latest)")
        HUGGINGFACE_AVAILABLE = True
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.info("⚠️ Using langchain-community (fallback)")
        HUGGINGFACE_AVAILABLE = False

    try:
        from langchain_chroma import Chroma
        logger.info("✅ Using langchain-chroma (latest)")
    except ImportError:
        from langchain_community.vectorstores import Chroma
        logger.info("⚠️ Using langchain-community Chroma (fallback)")

    LANGCHAIN_AVAILABLE = True
    logger.info("✅ LangChain available")

except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    logger.error(f"❌ LangChain not available: {e}")

# ========== ENHANCED FILE LOADER SYSTEM ==========

class UltimateFileLoader:
    """🔧 Ultimate File Loader with Smart Type Detection"""
    
    def __init__(self):
        self.supported_extensions = {
            # Text-based files (use TextLoader)
            'text_files': [
                 # JavaScript/TypeScript - FIXED
                 '.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs',
                 # Stylesheets  
                 '.css', '.scss', '.sass', '.less', '.styl',
                 # Frontend frameworks
                  '.vue', '.svelte',
                 # Config and data files - EXPANDED
                 '.json', '.yaml', '.yml', '.txt', '.md', '.toml', '.ini', '.cfg',
                 '.conf', '.env', '.gitignore', '.gitattributes', '.dockerignore',
                 # Build/config files
                 '.dockerfile', '.sh', '.bash', '.zsh', '.fish',
                 # Python and other langs
                 '.py', '.java', '.cpp', '.c', '.h', '.hpp', '.go', '.rs', '.php', '.rb',
                 '.swift', '.kt', '.scala', '.clj', '.hs', '.lua', '.r', '.m',
                 '.vb', '.cs', '.fs', '.ml', '.elm', '.dart', '.sol', '.move',
                 # SQL and lock files
                 '.sql', '.lock'
           ],
           # Document files (use UnstructuredFileLoader)
           'document_files': [
                 '.html', '.htm', '.xml', '.pdf', '.docx', '.doc', '.rtf',
                 '.odt', '.pages', '.epub', '.mobi'
            ],
            # Special loaders
             'special_files': {
                 '.csv': CSVLoader,
                 '.json': JSONLoader,
                  '.pdf': PyPDFLoader,
                  '.html': UnstructuredHTMLLoader,
                   '.htm': UnstructuredHTMLLoader
            }
      }
        
        self.load_stats = {
            'successful': 0,
            'failed': 0,
            'by_type': defaultdict(int),
            'errors': []
        }
    
    def get_loader(self, file_path: str):
        """🎯 Intelligent loader selection based on file type"""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            # Special cases first
            if ext in self.supported_extensions['special_files']:
                if ext == '.json':
                    return JSONLoader(file_path, jq_schema='.', text_content=False)
                elif ext == '.csv':
                    return CSVLoader(file_path, encoding='utf-8')
                elif ext == '.pdf':
                    return PyPDFLoader(file_path)
                elif ext in ['.html', '.htm']:
                    return UnstructuredHTMLLoader(file_path)
            
            # Text files - use TextLoader with UTF-8
            elif ext in self.supported_extensions['text_files']:
                return TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
            
            # Document files - use UnstructuredFileLoader
            elif ext in self.supported_extensions['document_files']:
                return UnstructuredFileLoader(file_path)
            
            # Default fallback - try TextLoader first
            else:
                logger.warning(f"⚠️ Unknown file type {ext}, trying TextLoader")
                return TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
                
        except Exception as e:
            logger.error(f"❌ Failed to create loader for {file_path}: {e}")
            return None
    
    def load_file(self, file_path: str) -> List[Document]:
        """🔄 Load file with intelligent error handling"""
        try:
            # Get appropriate loader
            loader = self.get_loader(file_path)
            if not loader:
                return []
            
            # Load documents
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'filename': os.path.basename(file_path),
                    'file_size': os.path.getsize(file_path),
                    'file_type': os.path.splitext(file_path)[1].lower(),
                    'processed_at': datetime.now().isoformat()
                })
            
            # Update stats
            ext = os.path.splitext(file_path)[1].lower()
            self.load_stats['successful'] += 1
            self.load_stats['by_type'][ext] += 1
            
            logger.debug(f"✅ Loaded {len(documents)} chunks from {os.path.basename(file_path)}")
            return documents
            
        except Exception as e:
            error_msg = f"Failed to load {file_path}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            self.load_stats['failed'] += 1
            self.load_stats['errors'].append(error_msg)
            
            # Try fallback method
            return self._fallback_load(file_path)
    
    def _fallback_load(self, file_path: str) -> List[Document]:
        """🔄 Fallback loading method"""
        try:
            # Try simple text reading as last resort
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if content.strip():
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': file_path,
                        'filename': os.path.basename(file_path),
                        'file_size': len(content),
                        'file_type': os.path.splitext(file_path)[1].lower(),
                        'processed_at': datetime.now().isoformat(),
                        'fallback_loaded': True
                    }
                )
                logger.info(f"✅ Fallback loaded {os.path.basename(file_path)}")
                return [doc]
            
        except Exception as e:
            logger.error(f"❌ Fallback loading also failed for {file_path}: {e}")
        
        return []
    
    def get_stats(self) -> Dict:
        """Get loading statistics"""
        return {
            'successful_files': self.load_stats['successful'],
            'failed_files': self.load_stats['failed'],
            'success_rate': (self.load_stats['successful'] / max(1, self.load_stats['successful'] + self.load_stats['failed'])) * 100,
            'files_by_type': dict(self.load_stats['by_type']),
            'total_errors': len(self.load_stats['errors']),
            'recent_errors': self.load_stats['errors'][-5:]  # Last 5 errors
        }

# ========== ENHANCED REPOSITORY PROCESSOR ==========

class UltimateRepositoryProcessor:
    """🚀 Ultimate Repository Processing Engine"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 max_files: int = 1000,
                 batch_size: int = 50):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap  
        self.max_files = max_files
        self.batch_size = batch_size
        
        # Initialize components
        self.file_loader = UltimateFileLoader()
        self.file_processor = IntelligentFileProcessor()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Repository metadata
        self.repo_metadata = {
            'url': None,
            'full_name': None,
            'description': None,
            'language': None,
            'stars': 0,
            'forks': 0,
            'size': 0,
            'processed_files': 0,
            'total_chunks': 0,
            'processing_time': 0,
            'last_updated': None
        }
        
        # Processing statistics
        self.processing_stats = {
            'start_time': None,
            'end_time': None,
            'total_files_found': 0,
            'files_processed': 0,
            'files_skipped': 0,
            'total_chunks_created': 0,
            'processing_errors': [],
            'memory_usage': 0
        }
        
        logger.info("🚀 Ultimate Repository Processor initialized")
    
    def should_process_file(self, file_path: str) -> bool:
     """🎯 Fixed intelligent file filtering"""
    
     # Get file parts for analysis
     path_parts = Path(file_path).parts
     filename = os.path.basename(file_path)
     ext = os.path.splitext(file_path)[1].lower()
    
     # FIRST: Check if supported file type (do this early)
     all_supported = (
        self.file_loader.supported_extensions['text_files'] +
        self.file_loader.supported_extensions['document_files'] +
        list(self.file_loader.supported_extensions['special_files'].keys())
     )
    
     if ext not in all_supported:
        logger.debug(f"❌ Skipping {file_path}: unsupported extension '{ext}'")
        return False
    
     # SECOND: Always allow important config files (even if hidden)
     important_files = {
        '.gitignore', '.gitattributes', '.env.example', '.dockerignore',
        '.babelrc', '.eslintrc', '.prettierrc', '.env.local', '.env.development',
        '.env.production', '.eslintrc.js', '.eslintrc.json', '.prettierrc.js',
        '.prettierrc.json', '.babelrc.js', '.babelrc.json'
    }
    
     if filename in important_files:
        logger.debug(f"✅ Processing important config: {file_path}")
        return True
    
     # THIRD: Skip .git directory and other version control
     if '.git' in path_parts:
        return False
    
     # FOURTH: Skip node_modules and build directories
     skip_dirs = ['node_modules', 'dist', 'build', '__pycache__', '.pytest_cache']
     if any(skip_dir in path_parts for skip_dir in skip_dirs):
        return False
    
     # FIFTH: Skip hidden directories (but not hidden files in root)
     for part in path_parts[:-1]:  # Check all parts except filename
        if part.startswith('.') and part not in {'.github', '.vscode'}:
            return False
    
     # SIXTH: Check file size (skip very large files)
     try:
        file_size = os.path.getsize(file_path)
        if file_size > 10 * 1024 * 1024:  # Skip files > 10MB
            logger.warning(f"⚠️ Skipping large file: {file_path} ({file_size/1024/1024:.1f}MB)")
            return False
     except OSError:
        return False
    
     # If we reach here, the file should be processed
     logger.debug(f"✅ Will process: {file_path}")
     return True
    
    def is_user_visible_file(self, filepath: str) -> bool:
        """Check if file is visible to user (like GitHub UI shows)"""
        filename = os.path.basename(filepath)
        path_parts = Path(filepath).parts
        
        # Skip hidden files in root directory
        if filename.startswith('.') and len(path_parts) <= 2:
            # Allow important config files
            important_configs = ['.gitignore', '.env.example', '.dockerignore']
            if filename not in important_configs:
                return False
        
        # Skip system directories completely
        skip_dirs = ['.git', '__pycache__', 'node_modules', '.vscode', '.idea', 
                    'dist', 'build', '.pytest_cache', '.mypy_cache']
        if any(skip_dir in path_parts for skip_dir in skip_dirs):
            return False
        
        # Skip system files
        system_files = ['.DS_Store', 'Thumbs.db', 'desktop.ini']
        if filename in system_files:
            return False
            
        return True
    
    def clone_repository(self, repo_url: str, temp_dir: str) -> Optional[str]:
        """📥 Clone repository with enhanced error handling"""
        if not GIT_AVAILABLE:
            logger.error("❌ Git not available - cannot clone repository")
            return None
        
        try:
            logger.info(f"📥 Cloning repository: {repo_url}")
            
            # Parse repo info
            parsed = urlparse(repo_url)
            if 'github.com' in parsed.netloc:
                parts = parsed.path.strip('/').split('/')
                if len(parts) >= 2:
                    self.repo_metadata['full_name'] = f"{parts[0]}/{parts[1]}"
            
            # Clone with optimizations
            repo = git.Repo.clone_from(
                repo_url, 
                temp_dir,
                depth=1,  # Shallow clone
                single_branch=True,  # Only default branch
            )
            
            self.repo_metadata['url'] = repo_url
            logger.info(f"✅ Repository cloned to {temp_dir}")
            
            return temp_dir
            
        except Exception as e:
            logger.error(f"❌ Failed to clone repository: {e}")
            return None
    
    def get_repository_metadata(self, repo_url: str) -> Dict:
        """📊 Extract repository metadata from GitHub API"""
        try:
            if 'github.com' not in repo_url:
                return {}
            
            # Parse GitHub URL
            parts = repo_url.replace('https://github.com/', '').replace('.git', '').split('/')
            if len(parts) < 2:
                return {}
            
            owner, repo = parts[0], parts[1]
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            
            # Make API request with timeout
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                return {
                    'full_name': data.get('full_name', ''),
                    'description': data.get('description', ''),
                    'language': data.get('language', ''),
                    'stars': data.get('stargazers_count', 0),
                    'forks': data.get('forks_count', 0),
                    'size': data.get('size', 0),
                    'default_branch': data.get('default_branch', 'main'),
                    'topics': data.get('topics', []),
                    'license': data.get('license', {}).get('name') if data.get('license') else None
                }
            
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch repository metadata: {e}")
        
        return {}
    
    def process_files(self, directory: str) -> List[Document]:
        """🔄 Process all files in directory with progress tracking"""
        all_documents = []
        
        logger.info(f"🔍 Scanning directory: {directory}")
        
        # Find all processable files
        processable_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if self.should_process_file(file_path):
                    processable_files.append(file_path)
        
        self.processing_stats['total_files_found'] = len(processable_files)
        user_visible_count = 0
        for filepath in processable_files:
            if self.is_user_visible_file(filepath):
                user_visible_count += 1

        self.processing_stats["user_visible_files"] = user_visible_count
        logger.info(f"Found {user_visible_count} user-visible files ({len(processable_files)} total)")
        
        # Limit files if too many
        if len(processable_files) > self.max_files:
            logger.warning(f"⚠️ Too many files ({len(processable_files)}), limiting to {self.max_files}")
            processable_files = processable_files[:self.max_files]
        
        logger.info(f"📂 Processing {len(processable_files)} files")

        if len(processable_files) == 0:
           logger.error("❌ No processable files found! Checking first 10 files in directory:")
           all_files = []
           for root, dirs, files in os.walk(directory):
                for file in files[:10]:  # Show first 10 files for debugging
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file_path)[1].lower()
                    logger.error(f"   📄 {file_path} (ext: '{ext}') - Would process: {self.should_process_file(file_path)}")
                if len(all_files) >= 10:
                  break
              
        
        # Process files in batches
        for i in range(0, len(processable_files), self.batch_size):
            batch = processable_files[i:i + self.batch_size]
            logger.info(f"🔄 Processing batch {i//self.batch_size + 1}/{(len(processable_files)-1)//self.batch_size + 1}")
            
            batch_documents = []
            for file_path in batch:
                try:
                    # Load documents from file
                    documents = self.file_loader.load_file(file_path)
                    
                    if documents:
                        # Process each document
                        for doc in documents:
                            # Split into chunks if necessary
                            if len(doc.page_content) > self.chunk_size:
                                chunks = self.text_splitter.split_documents([doc])
                                batch_documents.extend(chunks)
                            else:
                                batch_documents.append(doc)
                        
                        self.processing_stats['files_processed'] += 1
                    else:
                        self.processing_stats['files_skipped'] += 1
                        
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {str(e)}"
                    logger.error(f"❌ {error_msg}")
                    self.processing_stats['processing_errors'].append(error_msg)
                    self.processing_stats['files_skipped'] += 1
            
            all_documents.extend(batch_documents)
            self.processing_stats['total_chunks_created'] = len(all_documents)
            
            # Progress update
            progress = ((i + len(batch)) / len(processable_files)) * 100
            logger.info(f"📊 Progress: {progress:.1f}% ({len(all_documents)} chunks created)")
        
        logger.info(f"✅ File processing complete: {len(all_documents)} chunks from {self.processing_stats['files_processed']} files")
        return all_documents
    
    def get_processing_summary(self) -> Dict:
        """📊 Get comprehensive processing summary"""
        file_stats = self.file_loader.get_stats()
        
        return {
            'repository': {
                'url': self.repo_metadata.get('url'),
                'name': self.repo_metadata.get('full_name'),
                'description': self.repo_metadata.get('description'),
                'language': self.repo_metadata.get('language'),
                'stars': self.repo_metadata.get('stars', 0),
                'size_kb': self.repo_metadata.get('size', 0)
            },
            'processing': {
                'files_found': self.processing_stats['total_files_found'],
                'files_processed': self.processing_stats['files_processed'],
                'files_skipped': self.processing_stats['files_skipped'],
                'success_rate': f"{(self.processing_stats['files_processed'] / max(1, self.processing_stats['total_files_found'])) * 100:.1f}%",
                'total_chunks': self.processing_stats['total_chunks_created'],
                'processing_time': f"{(time.time() - self.processing_stats.get('start_time', time.time())):.2f}s",
                'user_visible_files': self.processing_stats.get("user_visible_files", 0)
            },
            'file_types': file_stats['files_by_type'],
            'errors': {
                'total_errors': len(self.processing_stats['processing_errors']),
                'recent_errors': self.processing_stats['processing_errors'][-3:]
            }
        }

# ========== ENHANCED VECTOR DATABASE MANAGER ==========

class UltimateVectorDBManager:
    """🗃️ Ultimate Vector Database Manager with Cloud Support"""
    
    def __init__(self, 
                 db_path: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 collection_name: str = "repository_docs"):
        
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.vector_store = None
        self.embeddings = None
        
        # Initialize embeddings
        self._initialize_embeddings()
        
        # Database statistics
        self.db_stats = {
            'total_documents': 0,
            'total_embeddings': 0,
            'db_size_mb': 0,
            'last_update': None
        }
    
    def _initialize_embeddings(self):
        """🔄 Initialize embedding model"""
        try:
            logger.info(f"🧠 Initializing embeddings: {self.embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✅ Embeddings initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {e}")
            self.embeddings = None
    
    def create_vector_store(self, documents: List[Document]) -> bool:
     """🏗️ Create vector store with NEW ChromaDB API"""
     if not self.embeddings:
        logger.error("❌ Cannot create vector store without embeddings")
        return False

     try:
        logger.info(f"🏗️ Creating vector store with {len(documents)} documents")

        # Ensure documents are not empty
        valid_documents = [doc for doc in documents if doc.page_content.strip()]
        if not valid_documents:
            logger.error("❌ No valid documents to process")
            return False

        logger.info(f"📝 Processing {len(valid_documents)} valid documents")

        # ✅ NEW ChromaDB API - No more Settings!
        if len(valid_documents) > 1000:
            logger.info("📦 Processing documents in batches for better memory management")
            
            # Process first batch to create the store
            first_batch = valid_documents[:500]
            self.vector_store = Chroma.from_documents(
                documents=first_batch,
                embedding=self.embeddings,
                persist_directory=self.db_path,
                collection_name=self.collection_name
                # ❌ REMOVED: client_settings=chroma_settings
            )

            # Add remaining documents in batches
            remaining_docs = valid_documents[500:]
            batch_size = 500
            for i in range(0, len(remaining_docs), batch_size):
                batch = remaining_docs[i:i + batch_size]
                logger.info(f"📦 Adding batch {i//batch_size + 2}: {len(batch)} documents")
                self.vector_store.add_documents(batch)
        else:
            # ✅ NEW: Simple creation without deprecated settings
            self.vector_store = Chroma.from_documents(
                documents=valid_documents,
                embedding=self.embeddings,
                persist_directory=self.db_path,
                collection_name=self.collection_name
                # ❌ REMOVED: client_settings=chroma_settings
            )

        # ✅ NEW: No need to call persist() manually
        # Documents are automatically persisted in new versions

        # Update statistics
        self._update_db_stats()
        logger.info("✅ Vector store created and persisted successfully")
        return True

     except Exception as e:
        logger.error(f"❌ Failed to create vector store: {e}")
        return False
    
    def _update_db_stats(self):
        """📊 Update database statistics"""
        try:
            if self.vector_store:
                # Get collection info
                collection = self.vector_store._collection
                count = collection.count()
                
                self.db_stats.update({
                    'total_documents': count,
                    'total_embeddings': count,
                    'last_update': datetime.now().isoformat()
                })
                
                # Calculate database size
                if os.path.exists(self.db_path):
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(self.db_path):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            if os.path.exists(fp):
                                total_size += os.path.getsize(fp)
                    
                    self.db_stats['db_size_mb'] = total_size / (1024 * 1024)
                
        except Exception as e:
            logger.warning(f"⚠️ Could not update database stats: {e}")
    
    def load_existing_store(self) -> bool:
        """📂 Load existing vector store"""
        try:
            if os.path.exists(self.db_path) and self.embeddings:
                logger.info(f"📂 Loading existing vector store from {self.db_path}")
                
                self.vector_store = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                
                self._update_db_stats()
                
                logger.info(f"✅ Loaded vector store with {self.db_stats['total_documents']} documents")
                return True
            else:
                logger.info("ℹ️ No existing vector store found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to load existing vector store: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """📊 Get comprehensive database statistics"""
        return {
            'database': {
                'path': self.db_path,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model,
                'exists': os.path.exists(self.db_path),
                'size_mb': round(self.db_stats.get('db_size_mb', 0), 2)
            },
            'documents': {
                'total_count': self.db_stats.get('total_documents', 0),
                'total_embeddings': self.db_stats.get('total_embeddings', 0),
                'last_updated': self.db_stats.get('last_update')
            },
            'status': 'Ready' if self.vector_store else 'Not initialized'
        }

# ========== INTEGRATION FUNCTIONS FOR QA_ENGINE ==========

# Global variables for qa_engine.py integration
_repository_loaded = False
_repository_info = {}
_retriever = None
_file_mappings = {}

def get_retriever_for_qa_engine():
     """🔗 Get retriever for qa_engine.py integration"""
     global _retriever
     return _retriever

def get_file_mapping_for_rewrites():
    """🔗 Get file mappings for qa_engine.py integration"""
    global _file_mappings
    return _file_mappings

def is_repository_loaded():
    """🔗 Check if repository is loaded for qa_engine.py integration"""
    global _repository_loaded
    return _repository_loaded

def load_repository_metadata():
    """🔗 Load repository metadata for qa_engine.py integration"""
    global _repository_info
    return _repository_info

def get_repository_info():
    """🔗 Get repository info for qa_engine.py integration"""
    global _repository_info
    return _repository_info

def _write_ingest_state(db_path: str, file_mappings: Dict, repo_metadata: Dict):
     """Write ingest state to persistent storage for cross-process access"""
     try:
        state = {
            "file_mappings": file_mappings,
            "repo_metadata": repo_metadata,
            "timestamp": datetime.now().isoformat()
        }
        os.makedirs(db_path, exist_ok=True)
        state_file = os.path.join(db_path, "nova_ingest_state.json")
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        logger.info(f"✅ Saved ingest state to {state_file}")
     except Exception as e:
        logger.warning(f"⚠️ Could not write ingest state: {e}")

def _update_global_state(db_manager: UltimateVectorDBManager, processor: UltimateRepositoryProcessor):
    """🔄 Update global state for qa_engine integration - ENHANCED VERSION"""
    global _repository_loaded, _repository_info, _retriever, _file_mappings
    
    try:
        if db_manager.vector_store:
            _retriever = db_manager.vector_store.as_retriever(search_kwargs={"k": 8})
            
            # Build file mappings from processed documents
            collection = db_manager.vector_store._collection
            all_docs = collection.get()
            
            if all_docs and 'metadatas' in all_docs:
                for metadata in all_docs['metadatas']:
                    if metadata and 'source' in metadata:
                        source = metadata['source']
                        filename = os.path.basename(source)
                        _file_mappings[filename.lower()] = source
            
            # Handle case when processor is None (existing database)
            if processor is not None:
                _repository_info = processor.repo_metadata.copy()
            else:
                _repository_info = {}
                
            _repository_loaded = True

            # Write to persistent storage
            _write_ingest_state(db_manager.db_path, _file_mappings, _repository_info)
            
            # ✅ NEW: Update shared state for CLI-QA communication
            try:
                from shared_state import nova_repo_state
                nova_repo_state.set_active_repository(
                    _repository_info.get('url', 'unknown'),
                    db_manager.db_path,
                    _file_mappings,
                    _repository_info
                )
                logger.info("✅ Shared state updated for CLI-QA communication")
            except ImportError:
                logger.warning("⚠️ Shared state not available")
            
            logger.info(f"🔗 Global state updated: {len(_file_mappings)} file mappings created")
            
    except Exception as e:
        logger.error(f"❌ Failed to update global state: {e}")

class SmartMultiRepoManager:
    """🧠 Smart multi-repository manager"""
    
    def __init__(self, base_db_path: str = "./chroma_repos"):
        self.base_db_path = base_db_path
        self.current_repo_db = None
        self.repo_registry = os.path.join(base_db_path, "repo_registry.json")
        self.setup_registry()
    
    def setup_registry(self):
        """Setup repository registry"""
        os.makedirs(self.base_db_path, exist_ok=True)
        
        if not os.path.exists(self.repo_registry):
            registry_data = {
                "repositories": {},
                "last_active": None,
                "total_repos": 0
            }
            with open(self.repo_registry, 'w') as f:
                json.dump(registry_data, f, indent=2)
        
        print(f"📋 Registry initialized: {self.repo_registry}")
    
    def get_repo_db_path(self, repo_url: str) -> str:
        """Get unique database path for repository"""
        # Create unique hash for repo URL
        repo_hash = hashlib.md5(repo_url.encode()).hexdigest()[:8]
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        return os.path.join(self.base_db_path, f"{repo_name}_{repo_hash}")
    
    def is_repo_already_processed(self, repo_url: str) -> tuple[bool, str]:
        """Check if repository is already processed"""
        try:
            with open(self.repo_registry, 'r') as f:
                registry = json.load(f)
            
            repo_data = registry["repositories"].get(repo_url)
            if repo_data:
                db_path = repo_data["db_path"]
                if os.path.exists(db_path):
                    print(f"✅ Found existing DB for repo: {db_path}")
                    return True, db_path
                else:
                    print(f"⚠️ DB path doesn't exist, will recreate: {db_path}")
            
            return False, self.get_repo_db_path(repo_url)
        except Exception as e:
            print(f"Registry read error: {e}")
            return False, self.get_repo_db_path(repo_url)
    
    def register_repository(self, repo_url: str, db_path: str, metadata: Dict):
        """Register processed repository"""
        try:
            with open(self.repo_registry, 'r') as f:
                registry = json.load(f)
            
            registry["repositories"][repo_url] = {
                "db_path": db_path,
                "processed_at": datetime.now().isoformat(),
                "metadata": metadata,
                "repo_name": repo_url.split('/')[-1].replace('.git', ''),
                "files_count": metadata.get('files_processed', 0)
            }
            registry["last_active"] = repo_url
            registry["total_repos"] = len(registry["repositories"])
            
            with open(self.repo_registry, 'w') as f:
                json.dump(registry, f, indent=2)
            
            print(f"📋 Registry updated: {repo_url} registered")
                
        except Exception as e:
            print(f"Registry update error: {e}")
    
    def get_all_repositories(self) -> Dict:
        """Get all registered repositories"""
        try:
            with open(self.repo_registry, 'r') as f:
                return json.load(f)
        except:
            return {"repositories": {}, "last_active": None, "total_repos": 0}

def smart_ingest_repository(repo_url: str) -> Dict[str, Any]:
    """🧠 SMART repository ingestion with multi-repo support"""
    
    print(f"🚀 Smart ingestion starting for: {repo_url}")
    manager = SmartMultiRepoManager()
    
    # Check if already processed
    already_processed, db_path = manager.is_repo_already_processed(repo_url)
    
    if already_processed:
        print(f"✅ Repository already processed, loading existing database...")
        
        # Load existing database
        db_manager = UltimateVectorDBManager(db_path)
        if db_manager.load_existing_store():
            _update_global_state(db_manager, None)
            
            # Get registry info for better response
            registry = manager.get_all_repositories()
            repo_info = registry["repositories"].get(repo_url, {})
            
            return {
                "success": True,
                "message": "Using existing database for repository",
                "database_path": db_path,
                "repository_url": repo_url,
                "reprocessed": False,
                "files_processed": repo_info.get("files_count", "N/A"),
                "processed_at": repo_info.get("processed_at", "Unknown"),
                "repo_name": repo_info.get("repo_name", repo_url.split('/')[-1])
            }
    
    # Process new repository
    print(f"🔄 Processing new repository: {repo_url}")
    
    processor = UltimateRepositoryProcessor()
    db_manager = UltimateVectorDBManager(db_path)
    processor.processing_stats['start_time'] = time.time()
    
    # Get repository metadata
    repo_metadata = processor.get_repository_metadata(repo_url)
    processor.repo_metadata.update(repo_metadata)
    
    # Clone and process
    temp_dir = tempfile.mkdtemp(prefix="nova_repo_")
    try:
        cloned_path = processor.clone_repository(repo_url, temp_dir)
        if not cloned_path:
            return {"success": False, "error": "Failed to clone repository"}
        
        documents = processor.process_files(cloned_path)
        if not documents:
            return {"success": False, "error": "No documents processed"}
        
        if not db_manager.create_vector_store(documents):
            return {"success": False, "error": "Failed to create vector database"}
        
        # Register repository with enhanced metadata
        enhanced_metadata = {**repo_metadata, "files_processed": len(documents)}
        manager.register_repository(repo_url, db_path, enhanced_metadata)
        
        # Update global state
        _update_global_state(db_manager, processor)
        
        return {
            "success": True,
            "message": "Repository processed successfully",
            "database_path": db_path,
            "repository_url": repo_url,
            "reprocessed": True,
            "files_processed": len(documents),
            "repo_name": repo_url.split('/')[-1].replace('.git', ''),
            "processing_time": f"{time.time() - processor.processing_stats['start_time']:.2f}s"
        }
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"🧹 Cleaned up temp directory")
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")

# Helper function for CLI integration
def list_processed_repositories() -> Dict[str, Any]:
    """List all processed repositories"""
    manager = SmartMultiRepoManager()
    return manager.get_all_repositories()

# ========== MAIN INGESTION FUNCTION ==========

def ingest_repository(repo_url: str,
                     db_path: str = "./chroma_db",
                     chunk_size: int = 1000,
                     chunk_overlap: int = 200,
                     max_files: int = 1000,
                     force_rebuild: bool = False) -> Dict[str, Any]:
    """
    🚀 ULTIMATE REPOSITORY INGESTION with Smart Processing

    Args:
        repo_url: GitHub repository URL
        db_path: Vector database path
        chunk_size: Text chunk size for splitting
        chunk_overlap: Overlap between chunks
        max_files: Maximum files to process
        force_rebuild: Force rebuild even if database exists

    Returns:
        Comprehensive ingestion results
    """
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 NOVA ULTIMATE REPOSITORY INGESTION ENGINE STARTING")
    logger.info("=" * 80)

    # Initialize components
    processor = UltimateRepositoryProcessor(chunk_size, chunk_overlap, max_files)
    db_manager = UltimateVectorDBManager(db_path)
    processor.processing_stats['start_time'] = start_time

    # Check if we should use existing database
    if not force_rebuild and db_manager.load_existing_store():
        logger.info("✅ Using existing vector database")
        _update_global_state(db_manager, processor)
        
        fake_summary = {
            'repository': {
                'url': repo_url,
                'name': repo_url.split('/')[-1] if repo_url else 'Unknown',
                'description': 'Loaded from existing database',
                'language': 'Multiple',
                'stars': 0,
                'size_kb': 0
            },
            'processing': {
                'files_found': len(_file_mappings) if _file_mappings else 0,
                'files_processed': len(_file_mappings) if _file_mappings else 0,
                'files_skipped': 0,
                'success_rate': '100.0%',
                'total_chunks': db_manager.db_stats.get('total_documents', 0),
                'processing_time': f"{time.time() - start_time:.2f}s"
            },
            'file_types': {'existing': 1},
            'errors': {
                'total_errors': 0,
                'recent_errors': []
            }
        }
        
        # 🔧 NEW: Persist state even for existing database
        _write_ingest_state(db_path, _file_mappings, processor.repo_metadata) # type: ignore
        
        return {
            "success": True,
            "message": "Using existing vector database",
            "processing_summary": fake_summary,
            "database_stats": db_manager.get_stats(),
            "processing_time": f"{time.time() - start_time:.2f}s",
            "repository_url": repo_url,
            "database_path": db_path,  # 🔧 ADDED: Include database path
            "integration_status": {
                "qa_engine_ready": _repository_loaded,
                "file_mappings_count": len(_file_mappings),
                "retriever_available": _retriever is not None
            }
        }

    # Get repository metadata
    logger.info(f"📊 Fetching repository metadata: {repo_url}")
    repo_metadata = processor.get_repository_metadata(repo_url)
    processor.repo_metadata.update(repo_metadata)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="nova_repo_")
    
    try:
        # Clone repository
        cloned_path = processor.clone_repository(repo_url, temp_dir)
        if not cloned_path:
            return {
                "success": False,
                "error": "Failed to clone repository",
                "processing_time": f"{time.time() - start_time:.2f}s"
            }

        # Process files
        logger.info("🔄 Starting file processing...")
        documents = processor.process_files(cloned_path)
        
        if not documents:
            return {
                "success": False,
                "error": "No documents were processed",
                "processing_summary": processor.get_processing_summary(),
                "processing_time": f"{time.time() - start_time:.2f}s"
            }

        # Create vector database
        logger.info("🏗️ Creating vector database...")
        if not db_manager.create_vector_store(documents):
            return {
                "success": False,
                "error": "Failed to create vector database",
                "processing_summary": processor.get_processing_summary(),
                "processing_time": f"{time.time() - start_time:.2f}s"
            }

        # Update global state for qa_engine integration
        _update_global_state(db_manager, processor)
        
        # 🔧 NEW: Persist ingest state for cross-process access
        _write_ingest_state(db_path, _file_mappings, processor.repo_metadata) # type: ignore

        # Final statistics
        processing_time = time.time() - start_time
        processor.processing_stats['end_time'] = time.time()

        # Success result
        result = {
            "success": True,
            "message": "Repository ingested successfully",
            "processing_summary": processor.get_processing_summary(),
            "database_stats": db_manager.get_stats(),
            "file_loader_stats": processor.file_loader.get_stats(),
            "processing_time": f"{processing_time:.2f}s",
            "repository_url": repo_url,
            "database_path": db_path,  # 🔧 ADDED: Include database path
            "integration_status": {
                "qa_engine_ready": _repository_loaded,
                "file_mappings_count": len(_file_mappings),
                "retriever_available": _retriever is not None
            }
        }

        logger.info("=" * 80)
        logger.info("✅ NOVA REPOSITORY INGESTION COMPLETED SUCCESSFULLY")
        logger.info(f"📊 Processed: {result['processing_summary']['processing']['files_processed']} files")
        logger.info(f"📦 Created: {result['processing_summary']['processing']['total_chunks']} chunks")
        logger.info(f"⏱️ Time: {result['processing_time']}")
        logger.info(f"🗃️ Database: {result['database_stats']['database']['size_mb']:.1f} MB")
        logger.info("=" * 80)

        return result

    except Exception as e:
        logger.error(f"❌ Ingestion failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "processing_summary": processor.get_processing_summary(),
            "processing_time": f"{time.time() - start_time:.2f}s"
        }
    
    finally:
        # Enhanced cleanup for Windows
        try:
            if os.path.exists(temp_dir):
                import stat
                def remove_readonly(func, path, _):
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                shutil.rmtree(temp_dir, onerror=remove_readonly)
                logger.info(f"🧹 Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Could not cleanup temp directory: {e}")
# ========== COMMAND LINE INTERFACE ==========

def main():
    """🎯 Command line interface for repository ingestion"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🚀 NOVA Ultimate Repository Ingestion Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py https://github.com/username/repo
  python ingest.py https://github.com/username/repo --force-rebuild
  python ingest.py https://github.com/username/repo --max-files 500 --chunk-size 1500
        """
    )
    
    parser.add_argument(
        'repo_url',
        help='GitHub repository URL to ingest'
    )
    
    parser.add_argument(
        '--db-path',
        default='./chroma_db',
        help='Vector database path (default: ./chroma_db)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Text chunk size for splitting (default: 1000)'
    )
    
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help='Overlap between chunks (default: 200)'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        default=1000,
        help='Maximum files to process (default: 1000)'
    )
    
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild even if database exists'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check dependencies
    if not CHROMADB_AVAILABLE:
        print("❌ ChromaDB not available. Install with: pip install chromadb")
        sys.exit(1)
    
    if not LANGCHAIN_AVAILABLE:
        print("❌ LangChain not available. Install with: pip install langchain langchain-community")
        sys.exit(1)
    
    # Run ingestion
    try:
        result = ingest_repository(
            repo_url=args.repo_url,
            db_path=args.db_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_files=args.max_files,
            force_rebuild=args.force_rebuild
        )
        
        if result['success']:
            print("\n🎉 SUCCESS! Repository ingested successfully!")
            print(f"📊 Files processed: {result['processing_summary']['processing']['files_processed']}")
            print(f"📦 Chunks created: {result['processing_summary']['processing']['total_chunks']}")
            print(f"⏱️ Processing time: {result['processing_time']}")
            print(f"🗃️ Database size: {result['database_stats']['database']['size_mb']:.1f} MB")
            print(f"\n🔗 QA Engine integration: {'✅ Ready' if result['integration_status']['qa_engine_ready'] else '❌ Not ready'}")
            
        else:
            print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}")
            if 'processing_summary' in result:
                print(f"📊 Files found: {result['processing_summary']['processing']['files_found']}")
                print(f"📊 Files processed: {result['processing_summary']['processing']['files_processed']}")
            
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Ingestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

def process_and_store_documents(documents, db_path="./chroma_db", collection_name="repository_docs"):
     """Process and store documents in vector database"""
     try:
        db_manager = UltimateVectorDBManager(db_path, collection_name=collection_name)
        success = db_manager.create_vector_store(documents)
        
        if success:
            # Update global state for QA engine integration
            _update_global_state(db_manager, None)
            
            return {
                "success": True,
                "message": f"Successfully stored {len(documents)} documents",
                "database_path": db_path,
                "document_count": len(documents),
                "collection_name": collection_name
            }
        else:
            return {
                "success": False,
                "error": "Failed to create vector store",
                "database_path": db_path
            }
            
     except Exception as e:
        return {
            "success": False,
            "error": f"Document processing failed: {str(e)}",
            "database_path": db_path
        }

def main_func_alias(repo_url):
     """Main function alias for feature.py compatibility"""
     return ingest_repository(repo_url)

def ingest_repo(repo_url, db_path="./chroma_db"):
     """Legacy function name for compatibility"""
     return ingest_repository(repo_url, db_path)

# Export functions for external imports
__all__ = [
    'ingest_repository',
    'process_and_store_documents', 
    'main_func_alias',
    'ingest_repo',
    'smart_ingest_repository',
    'list_processed_repositories',
    'get_retriever_for_qa_engine',
    'get_file_mapping_for_rewrites',
    'is_repository_loaded',
    'load_repository_metadata',
    'get_repository_info'
]

if __name__ == "__main__":
    main()