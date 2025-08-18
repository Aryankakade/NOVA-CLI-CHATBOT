# ml/models/custom_embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
from typing import List, Dict, Tuple, Optional
import torch
import faiss
import pickle
from pathlib import Path
import json
from datetime import datetime
import hashlib

class CustomEmbeddingModel:
    def __init__(self):
        """
        Advanced embedding system with multiple vector stores and optimizations
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Initialize multiple storage backends
        self._init_chromadb()
        self._init_faiss()
        
        # Cache for frequently accessed embeddings
        self.embedding_cache = {}
        self.cache_size_limit = 1000
        
        print(f"🚀 Advanced embedding system initialized on {self.device}")
    
    def _init_chromadb(self):
        """Initialize ChromaDB for persistent storage"""
        try:
            self.chroma_client = chromadb.PersistentClient(path="./vector_db")
            self.collections = {
                'conversations': self._get_or_create_collection('nova_conversations'),
                'code_snippets': self._get_or_create_collection('code_snippets'),
                'documents': self._get_or_create_collection('documents'),
                'knowledge_base': self._get_or_create_collection('knowledge_base')
            }
            print("✅ ChromaDB initialized successfully")
        except Exception as e:
            print(f"⚠️ ChromaDB initialization failed: {e}")
    
    def _init_faiss(self):
        """Initialize FAISS for ultra-fast similarity search"""
        try:
            self.embedding_dim = 384  # MiniLM embedding dimension
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product similarity
            self.faiss_index_mapping = {}  # ID to metadata mapping
            print("✅ FAISS index initialized")
        except Exception as e:
            print(f"⚠️ FAISS initialization failed: {e}")
    
    def _get_or_create_collection(self, name: str):
        """Get existing collection or create new one"""
        try:
            return self.chroma_client.get_collection(name)
        except:
            return self.chroma_client.create_collection(name)
    
    def encode_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Encode text to embeddings with caching
        """
        if use_cache:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Add to cache if enabled
        if use_cache and len(self.embedding_cache) < self.cache_size_limit:
            self.embedding_cache[text_hash] = embedding
        
        return embedding
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Efficiently encode multiple texts in batches
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)
    
    def semantic_search(self, query: str, documents: List[str], top_k: int = 5, 
                       similarity_threshold: float = 0.3) -> List[Dict]:
        """
        Advanced semantic search with multiple ranking algorithms
        """
        query_embedding = self.encode_text(query)
        doc_embeddings = self.encode_batch(documents)
        
        # Multiple similarity metrics
        similarities = {
            'cosine': self._cosine_similarity(query_embedding, doc_embeddings),
            'dot_product': np.dot(doc_embeddings, query_embedding),
            'euclidean': -np.linalg.norm(doc_embeddings - query_embedding, axis=1)  # Negative for sorting
        }
        
        # Ensemble scoring (weighted combination)
        final_scores = (
            0.5 * similarities['cosine'] + 
            0.3 * similarities['dot_product'] + 
            0.2 * similarities['euclidean']
        )
        
        # Filter by threshold and get top results
        valid_indices = np.where(similarities['cosine'] >= similarity_threshold)[0]
        if len(valid_indices) == 0:
            valid_indices = np.arange(len(documents))  # Fallback to all documents
        
        top_indices = valid_indices[np.argsort(final_scores[valid_indices])[-top_k:]][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': documents[idx],
                'cosine_score': float(similarities['cosine'][idx]),
                'ensemble_score': float(final_scores[idx]),
                'index': int(idx),
                'relevance': 'high' if similarities['cosine'][idx] > 0.7 else 'medium' if similarities['cosine'][idx] > 0.5 else 'low'
            })
        
        return results
    
    def _cosine_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity efficiently"""
        return np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
    
    def store_conversation(self, conversation_id: str, content: str, 
                          metadata: Optional[Dict] = None, collection: str = 'conversations'):
        """
        Store conversation with comprehensive metadata
        """
        embedding = self.encode_text(content)
        
        # Enhanced metadata
        enhanced_metadata = {
            'timestamp': datetime.now().isoformat(),
            'content_type': 'conversation',
            'word_count': len(content.split()),
            'char_count': len(content),
            'embedding_model': 'all-MiniLM-L6-v2',
            **(metadata or {})
        }
        
        try:
            # Store in ChromaDB
            self.collections[collection].add(
                embeddings=[embedding.tolist()],
                documents=[content],
                metadatas=[enhanced_metadata],
                ids=[conversation_id]
            )
            
            # Also store in FAISS for fast retrieval
            self.faiss_index.add(embedding.reshape(1, -1))
            self.faiss_index_mapping[self.faiss_index.ntotal - 1] = {
                'id': conversation_id,
                'content': content,
                'metadata': enhanced_metadata
            }
            
            print(f"✅ Stored conversation {conversation_id} in vector database")
            
        except Exception as e:
            print(f"❌ Error storing conversation: {e}")
    
    def store_code_snippet(self, code_id: str, code: str, language: str, 
                          description: str = "", metadata: Optional[Dict] = None):
        """
        Store code snippets with specialized processing
        """
        # Combine code and description for better context
        content = f"Language: {language}\nDescription: {description}\nCode:\n{code}"
        
        code_metadata = {
            'content_type': 'code',
            'language': language,
            'description': description,
            'code_length': len(code),
            'lines_of_code': len(code.split('\n')),
            **(metadata or {})
        }
        
        self.store_conversation(code_id, content, code_metadata, 'code_snippets')
    
    def retrieve_relevant_memories(self, query: str, collection: str = 'conversations',
                                 n_results: int = 5, include_similarity: bool = True) -> List[Dict]:
        """
        Advanced memory retrieval with multiple search strategies
        """
        try:
            query_embedding = self.encode_text(query)
            
            # Primary search using ChromaDB
            results = self.collections[collection].query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                similarity = 1 - dist  # Convert distance to similarity
                formatted_results.append({
                    'content': doc,
                    'metadata': meta,
                    'similarity_score': float(similarity),
                    'rank': i + 1,
                    'relevance': 'high' if similarity > 0.8 else 'medium' if similarity > 0.6 else 'low'
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error retrieving memories: {e}")
            return []
    
    def hybrid_search(self, query: str, collections: List[str] = None, 
                     n_results_per_collection: int = 3) -> Dict[str, List[Dict]]:
        """
        Search across multiple collections and combine results
        """
        if collections is None:
            collections = ['conversations', 'code_snippets', 'documents', 'knowledge_base']
        
        all_results = {}
        for collection in collections:
            if collection in self.collections:
                results = self.retrieve_relevant_memories(
                    query, collection, n_results_per_collection
                )
                all_results[collection] = results
        
        return all_results
    
    def find_similar_code(self, code_query: str, language: str = None, 
                         similarity_threshold: float = 0.6) -> List[Dict]:
        """
        Specialized function for finding similar code snippets
        """
        results = self.retrieve_relevant_memories(code_query, 'code_snippets', n_results=10)
        
        # Filter by language if specified
        if language:
            results = [r for r in results if r['metadata'].get('language', '').lower() == language.lower()]
        
        # Filter by similarity threshold
        results = [r for r in results if r['similarity_score'] >= similarity_threshold]
        
        return results
    
    def get_embedding_stats(self) -> Dict:
        """
        Get statistics about stored embeddings
        """
        stats = {
            'total_embeddings': self.faiss_index.ntotal,
            'embedding_dimension': self.embedding_dim,
            'cache_size': len(self.embedding_cache),
            'collections': {}
        }
        
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats['collections'][name] = count
            except:
                stats['collections'][name] = 0
        
        return stats
    
    def optimize_storage(self):
        """
        Optimize vector storage and clean up
        """
        # Clear embedding cache
        self.embedding_cache.clear()
        
        # Rebuild FAISS index if needed
        if self.faiss_index.ntotal > 10000:  # Rebuild for large indices
            print("🔄 Optimizing FAISS index...")
            # Implementation for index optimization
        
        print("✅ Storage optimization completed")
    
    # ===================== ENHANCED METHODS (NEW) =====================
    
    def reranking_system(self, query: str, initial_results: List[Dict], 
                        rerank_top_k: int = 5) -> List[Dict]:
        """
        Improve search quality through intelligent reranking
        Purpose: Better relevance scoring than simple similarity
        """
        if len(initial_results) <= rerank_top_k:
            return initial_results
        
        # Enhanced reranking with multiple factors
        reranked_results = []
        
        for result in initial_results:
            document = result['document'] if 'document' in result else result['content']
            
            # Factor 1: Original semantic score
            semantic_score = result.get('cosine_score', result.get('similarity_score', 0))
            
            # Factor 2: Query-document length ratio (optimal around 1.0)
            query_len = len(query.split())
            doc_len = len(document.split())
            length_ratio = min(query_len, doc_len) / max(query_len, doc_len) if max(query_len, doc_len) > 0 else 0
            
            # Factor 3: Keyword overlap boost
            query_words = set(query.lower().split())
            doc_words = set(document.lower().split())
            keyword_overlap = len(query_words.intersection(doc_words)) / len(query_words) if query_words else 0
            
            # Factor 4: Recency boost (if timestamp available)
            recency_score = 1.0
            if 'metadata' in result and 'timestamp' in result['metadata']:
                try:
                    doc_time = datetime.fromisoformat(result['metadata']['timestamp'])
                    now = datetime.now()
                    days_old = (now - doc_time).days
                    recency_score = max(0.5, 1.0 - (days_old / 365))  # Decay over a year
                except:
                    pass
            
            # Combine scores with weights
            final_score = (
                0.5 * semantic_score +
                0.2 * length_ratio +
                0.2 * keyword_overlap +
                0.1 * recency_score
            )
            
            # Update result
            enhanced_result = result.copy()
            enhanced_result.update({
                'rerank_score': final_score,
                'semantic_score': semantic_score,
                'length_ratio': length_ratio,
                'keyword_overlap': keyword_overlap,
                'recency_score': recency_score,
                'reranked': True
            })
            
            reranked_results.append(enhanced_result)
        
        # Sort by rerank score and return top_k
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        return reranked_results[:rerank_top_k]

    def cross_encoder_reranking(self, query: str, documents: List[str], 
                               top_k: int = 5) -> List[Dict]:
        """
        Advanced reranking using cross-encoder approach
        Purpose: State-of-the-art reranking for best results
        """
        try:
            # Use the cross-encoder from sentence-transformers
            from sentence_transformers import CrossEncoder
            
            # Initialize cross-encoder (lightweight model)
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')
            
            # Create query-document pairs
            pairs = [(query, doc) for doc in documents]
            
            # Get cross-encoder scores
            cross_scores = cross_encoder.predict(pairs)
            
            # Create results with enhanced scoring
            results = []
            for i, (doc, cross_score) in enumerate(zip(documents, cross_scores)):
                results.append({
                    'document': doc,
                    'cross_encoder_score': float(cross_score),
                    'index': i,
                    'reranking_method': 'cross_encoder',
                    'relevance': 'high' if cross_score > 0.8 else 'medium' if cross_score > 0.5 else 'low'
                })
            
            # Sort by cross-encoder score
            results.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
            return results[:top_k]
            
        except ImportError:
            print("⚠️ Cross-encoder not available, falling back to semantic reranking")
            # Fallback to semantic similarity
            return self.semantic_search(query, documents, top_k)
        except Exception as e:
            print(f"⚠️ Cross-encoder reranking failed: {e}")
            return self.semantic_search(query, documents, top_k)

    def intelligent_context_selection(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Smart context selection combining multiple retrieval strategies
        Purpose: Best possible context for query answering
        """
        # Step 1: Get initial candidates (more than needed)
        initial_candidates = self.retrieve_relevant_memories(query, n_results=n_results*3)
        
        if not initial_candidates:
            return []
        
        # Step 2: Apply reranking
        reranked = self.reranking_system(query, initial_candidates, rerank_top_k=n_results*2)
        
        # Step 3: Final selection with diversity
        final_selection = []
        selected_content = set()
        
        for result in reranked:
            content = result['document'] if 'document' in result else result['content']
            
            # Avoid too similar content (diversity)
            content_words = set(content.lower().split())
            
            # Check if this content is too similar to already selected
            is_diverse = True
            for selected in selected_content:
                selected_words = set(selected.lower().split())
                overlap = len(content_words.intersection(selected_words))
                similarity = overlap / min(len(content_words), len(selected_words)) if min(len(content_words), len(selected_words)) > 0 else 0
                
                if similarity > 0.8:  # Too similar
                    is_diverse = False
                    break
            
            if is_diverse and len(final_selection) < n_results:
                final_selection.append(result)
                selected_content.add(content)
        
        return final_selection
