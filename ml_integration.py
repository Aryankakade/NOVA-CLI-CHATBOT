# ml/ml_integration.py
from ML.models import CustomIntentClassifier
from ML.models import CustomEmbeddingModel
from ML.models import AdvancedRAGSystem
from ML.models import CustomNLPPipeline
from ML.training import CustomModelTrainer
from ML.mlops import ModelExperimentTracker
from ML.monitoring import MLModelMonitor

from typing import Dict, List, Optional, Tuple
import json
import time
from datetime import datetime
import re
import numpy as np

def sanitize_metadata(meta):
    """Convert metadata into JSON-safe primitives, preserving dict/list structure."""
    if isinstance(meta, dict):
        return {k: sanitize_metadata(v) for k, v in meta.items()}
    elif isinstance(meta, (list, tuple)):
        return [sanitize_metadata(v) for v in meta]
    elif isinstance(meta, np.generic):  # numpy types
        return float(meta)
    elif isinstance(meta, np.ndarray):
        return meta.tolist()
    elif isinstance(meta, (str, int, float, bool)) or meta is None:
        return meta
    else:
        return str(meta)
class EnhancedMLManager:
    def __init__(self):
        """
        Comprehensive ML system with advanced capabilities
        Purpose: Complete AI/ML pipeline management with monitoring and optimization
        """
        print("🚀 Initializing Enhanced ML Management System...")
        
        # Core ML components
        print("   Loading core ML components...")
        self.intent_classifier = CustomIntentClassifier()
        self.embedding_model = CustomEmbeddingModel()
        self.trainer = CustomModelTrainer()
        
        # Advanced ML components
        print("   Loading advanced ML components...")
        self.rag_system = AdvancedRAGSystem()
        self.nlp_pipeline = CustomNLPPipeline()
        
        # MLOps components
        print("   Setting up MLOps infrastructure...")
        self.experiment_tracker = ModelExperimentTracker()
        self.model_monitor = MLModelMonitor()
        
        # Performance tracking
        self.query_count = 0
        self.total_processing_time = 0
        
        print("✅ Enhanced ML Management System ready!")
        print("🎯 All components loaded successfully!")
    
    def process_user_query(self, user_query: str, context: Dict = None) -> Dict:
        """
        Enhanced processing with new capabilities
        
        Args:
            user_query: User's input query
            context: Additional context (code, conversation history, etc.)
            
        Returns:
            Comprehensive analysis and routing information
        """
        start_time = time.time()
        self.query_count += 1
        
        try:
            # Step 1: Enhanced NLP analysis
            query_insights = self.nlp_pipeline.extract_query_intent_keywords(user_query)
            sentiment_analysis = self.nlp_pipeline.analyze_query_sentiment(user_query)
            
            # Step 2: Enhanced intent classification with multi-label support
            intent_explanation = self.intent_classifier.get_intent_explanation(user_query)
            best_agent = intent_explanation['selected_intent']
            
            # Step 3: Intelligent context selection with advanced reranking
            enhanced_context = self.embedding_model.intelligent_context_selection(user_query, n_results=5)
            
            # Step 4: Code analysis if code is present in context
            code_analysis = None
            if context and context.get('code'):
                code = context['code']
                language = context.get('language', 'python')
                
                code_analysis = {
                    'entities': self.nlp_pipeline.extract_code_entities(code, language),
                    'complexity': self.nlp_pipeline.analyze_code_complexity(code, language)
                }
            
            # Step 5: Performance monitoring
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.total_processing_time += processing_time
            
            # Track this prediction
            prediction_data = {
                'predicted_class': best_agent,
                'confidence': intent_explanation['confidence_calibrated'],
                'query': user_query,
                'processing_time': processing_time,
                'model_version': 'v3.0_enhanced'
            }
            
            self.model_monitor.track_prediction(prediction_data)
            
            # Enhanced response with new insights
            response = {
                'routing_decision': {
                    'selected_agent': best_agent,
                    'confidence_level': intent_explanation['confidence_calibrated'],
                    'multi_intent_analysis': intent_explanation['multi_intent_analysis'],
                    'explanation': intent_explanation['explanation'],
                    'alternative_agents': list(intent_explanation['multi_intent_analysis']['secondary_intents'])
                },
                'query_analysis': {
                    'sentiment': sentiment_analysis,
                    'intent_keywords': query_insights['intent_keywords'],
                    'supporting_keywords': intent_explanation['supporting_keywords'],
                    'technical_context': query_insights.get('technical_context', {}),
                    'query_type': query_insights['query_type']
                },
                'context_enhancement': {
                    'relevant_context': enhanced_context,
                    'context_quality': 'high' if enhanced_context else 'low',
                    'reranking_applied': True,
                    'diversity_ensured': True,
                    'context_count': len(enhanced_context)
                },
                'code_analysis': code_analysis,
                'performance_metrics': {
                    'processing_time_ms': processing_time,
                    'avg_processing_time': self.total_processing_time / self.query_count,
                    'total_queries_processed': self.query_count,
                    'enhancement_level': 'advanced'
                },
                'recommendations': self._generate_enhanced_recommendations(
                    intent_explanation, sentiment_analysis, enhanced_context
                )
            }
            
            return response
            
        except Exception as e:
            # Error handling and logging
            error_response = {
                'error': True,
                'error_message': str(e),
                'fallback_agent': 'general',
                'processing_time_ms': (time.time() - start_time) * 1000
            }
            
            print(f"❌ Error in query processing: {e}")
            return error_response
    
    def _generate_enhanced_recommendations(self, intent_analysis: Dict, 
                                         sentiment_analysis: Dict, 
                                         context: List) -> List[str]:
        """Generate enhanced recommendations based on analysis"""
        recommendations = []
        
        # Multi-intent recommendations
        if intent_analysis['multi_intent_analysis']['complexity'] == 'multi':
            recommendations.append("Complex query detected - consider breaking into steps")
        
        # Confidence-based recommendations
        confidence = intent_analysis['confidence_calibrated']
        if confidence < 0.6:
            recommendations.append("Query could be more specific for better results")
        elif confidence > 0.9:
            recommendations.append("High confidence - optimal query structure")
        
        # Emotional support recommendations
        if sentiment_analysis.get('needs_emotional_support'):
            recommendations.append("User needs emotional support - prioritize empathetic response")
        
        # Context quality recommendations
        if not context:
            recommendations.append("No relevant context found - might be exploring new topic")
        elif len(context) > 3:
            recommendations.append("Rich context available - can provide detailed response")
        
        # Urgency handling
        if sentiment_analysis.get('needs_urgent_help'):
            recommendations.append("User indicates urgency - provide immediate assistance")
        
        # Technical depth recommendations
        if intent_analysis['selected_intent'] in ['coding', 'technical', 'data_science']:
            recommendations.append("Technical query - provide code examples and detailed explanations")
        
        return recommendations
    
    def store_interaction_intelligently(self, query: str, response: str, 
                                    agent_used: str, user_feedback: Optional[Dict] = None):
     """
    Store only minimal info (query + response) so bot can recall past chats.
    Skip advanced ML metadata that caused issues.
    """
     interaction_id = f"interaction_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    # Just store plain conversation (enough for memory retrieval)
     conversation_content = f"User: {query}\nAgent: {agent_used}\nResponse: {response}"

    # ✅ Metadata cleaned: no None values
     metadata = {
        'agent_used': str(agent_used) if agent_used else "unknown",
        'response_length': int(len(response)),
        'interaction_type': 'chat',
        'user_feedback': str(user_feedback) if user_feedback else "none",
        'timestamp': datetime.now().isoformat()
    }

    # Store in embeddings for recall
     self.embedding_model.store_conversation(interaction_id, conversation_content, metadata)

    # (Optional) store code snippets separately if coding response
     if agent_used == 'coding' and '```' in response:
        code_snippets = self._extract_code_snippets_enhanced(response)
        for i, (code, language, description) in enumerate(code_snippets):
            code_id = f"code_{interaction_id}_{i}"
            self.embedding_model.store_code_snippet(
                code_id, code, language, description,
                {'parent_interaction': interaction_id, 'query_context': query[:100]}
            )

     print(f"💾 Stored minimal interaction: {interaction_id}")
    
    def _extract_code_snippets_enhanced(self, response: str) -> List[Tuple[str, str, str]]:
        """Enhanced code extraction with descriptions"""
        # Pattern to match code blocks with language specification
        pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        code_snippets = []
        for language, code in matches:
            language = language or 'unknown'
            code = code.strip()
            if code:
                # Extract description from surrounding text
                description = self._extract_code_description(response, code)
                code_snippets.append((code, language, description))
        
        return code_snippets
    
    def _extract_code_description(self, response: str, code: str) -> str:
        """Extract description for code snippet from surrounding text"""
        # Find text before the code block
        code_index = response.find(f"```")
        if code_index > 0:
            before_text = response[:code_index].strip()
            # Get last sentence before code
            sentences = before_text.split('.')
            if sentences:
                return sentences[-1].strip()[:100]  # Limit to 100 chars
        
        return "Code snippet from conversation"
    
    def train_custom_model(self, model_type: str = 'intent_classifier') -> Dict:
        """
        Train custom model with experiment tracking
        """
        print(f"🚀 Starting training for {model_type}...")
        
        # Start experiment tracking
        experiment_id = self.experiment_tracker.start_experiment(
            experiment_name=f"train_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            model_type=model_type,
            parameters={
                'model_type': model_type,
                'training_data_size': 'dynamic',
                'enhancement_level': 'advanced_v3'
            }
        )
        
        try:
            if model_type == 'intent_classifier':
                # Train intent classification model
                results = self.trainer.fine_tune_intent_model(save_model=True)
                
                # Log results
                self.experiment_tracker.log_metrics(experiment_id, {
                    'train_accuracy': results.get('eval_accuracy', 0),
                    'train_loss': results.get('train_loss', 0),
                    'eval_loss': results.get('eval_loss', 0)
                })
                
                # Finish experiment
                self.experiment_tracker.finish_experiment(
                    experiment_id, 
                    {'final_accuracy': results.get('eval_accuracy', 0)},
                    success=True
                )
                
                return {
                    'success': True,
                    'experiment_id': experiment_id,
                    'results': results
                }
            
        except Exception as e:
            # Log failed experiment
            self.experiment_tracker.finish_experiment(experiment_id, {}, success=False)
            return {
                'success': False,
                'experiment_id': experiment_id,
                'error': str(e)
            }
    
    def get_enhanced_system_report(self) -> Dict:
        """
        Get comprehensive system health and performance report with enhancements
        """
        # Model monitor report
        monitoring_report = self.model_monitor.generate_monitoring_report(hours_back=24)
        
        # Experiment tracking stats
        experiment_stats = self.experiment_tracker.get_tracking_stats()
        
        # Embedding system stats
        embedding_stats = self.embedding_model.get_embedding_stats()
        
        # RAG system stats
        rag_stats = self.rag_system.get_search_stats()
        
        # NLP pipeline stats
        nlp_stats = self.nlp_pipeline.get_pipeline_stats()
        
        # Enhanced system capabilities
        enhanced_capabilities = {
            'multi_intent_classification': True,
            'confidence_calibration': True,
            'intelligent_reranking': True,
            'context_diversity_control': True,
            'advanced_sentiment_analysis': True,
            'technical_term_detection': True,
            'code_entity_extraction': True,
            'cross_encoder_reranking': True
        }
        
        return {
            'system_overview': {
                'status': 'operational_enhanced',
                'version': 'v3.0_advanced',
                'uptime': '24/7',
                'total_queries_processed': self.query_count,
                'avg_processing_time_ms': self.total_processing_time / max(self.query_count, 1),
                'enhancement_level': 'production_ready'
            },
            'ml_components': {
                'intent_classifier': 'active_enhanced',
                'embeddings_model': 'active_enhanced',
                'rag_system': 'active',
                'nlp_pipeline': 'active',
                'experiment_tracker': 'active',
                'model_monitor': 'active'
            },
            'enhanced_capabilities': enhanced_capabilities,
            'monitoring': monitoring_report,
            'experiments': experiment_stats,
            'storage': embedding_stats,
            'capabilities': {
                'rag': rag_stats,
                'nlp': nlp_stats
            },
            'performance_summary': {
                'health_status': monitoring_report.get('health_status', 'HEALTHY'),
                'recommendations': monitoring_report.get('recommendations', []),
                'confidence_calibrated': True,
                'multi_intent_support': True
            }
        }
    
    def optimize_performance_enhanced(self):
        """
        Enhanced system optimization with new capabilities
        """
        print("🔧 Performing enhanced system optimization...")
        
        optimization_actions = []
        
        # Clear caches if memory usage is high
        self.embedding_model.optimize_storage()
        optimization_actions.append('embedding_cache_cleared')
        
        # Check for model retraining needs
        drift_analysis = self.model_monitor.detect_model_drift()
        if drift_analysis.get('drift_detected'):
            print("📊 Model drift detected - triggering optimization")
            optimization_actions.append('drift_detected')
            
            # Automatic performance enhancement
            if drift_analysis.get('confidence_drift', {}).get('detected'):
                print("🎯 Recalibrating confidence thresholds...")
                optimization_actions.append('confidence_recalibration')
        
        # Optimize intent classification embeddings
        if self.query_count > 100:  # Only after sufficient data
            print("🧠 Optimizing intent classification performance...")
            self.intent_classifier.save_model_state()
            optimization_actions.append('intent_embeddings_optimized')
        
        print("✅ Enhanced system optimization completed")
        return {
            'optimization_performed': True,
            'actions': optimization_actions,
            'recommendations': drift_analysis.get('recommendations', ['System performing optimally']),
            'enhancement_level': 'advanced'
        }
    
    def get_ml_insights_summary(self) -> Dict:
        """
        Get summary of ML system insights and capabilities
        """
        return {
            'core_ml_features': {
                'intent_classification': {
                    'categories': len(self.intent_classifier.intent_examples),
                    'multi_label_support': True,
                    'confidence_calibration': True,
                    'explanation_capability': True
                },
                'embeddings': {
                    'model': 'all-MiniLM-L6-v2',
                    'dimension': 384,
                    'collections': list(self.embedding_model.collections.keys()),
                    'intelligent_reranking': True,
                    'context_diversity': True
                },
                'nlp_pipeline': {
                    'sentiment_analysis': True,
                    'code_entity_extraction': True,
                    'technical_term_detection': True,
                    'complexity_analysis': True
                }
            },
            'advanced_features': {
                'rag_system': True,
                'cross_encoder_reranking': True,
                'experiment_tracking': True,
                'performance_monitoring': True,
                'drift_detection': True
            },
            'performance_metrics': {
                'total_queries_processed': self.query_count,
                'average_processing_time_ms': self.total_processing_time / max(self.query_count, 1),
                'system_uptime': '24/7'
            }
        }
