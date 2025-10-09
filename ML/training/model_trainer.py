# ml/training/model_trainer.py
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from datasets import Dataset
import mlflow
import wandb
from typing import List, Dict, Tuple, Optional
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class CustomModelTrainer:
    def __init__(self):
        """
        Advanced model training pipeline with MLOps capabilities
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Setup experiment tracking
        self._init_mlflow()
        self._init_wandb()
        
        # Training configuration
        self.config = {
            'model_name': 'distilbert-base-uncased',
            'max_length': 512,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'num_epochs': 5,
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'early_stopping_patience': 3
        }
        
        print(f"🚀 Advanced model trainer initialized on {self.device}")
    
    def _init_mlflow(self):
        """Initialize MLflow for experiment tracking"""
        try:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("nova_ml_experiments")
            print("✅ MLflow initialized")
        except Exception as e:
            print(f"⚠️ MLflow initialization failed: {e}")
    
    def _init_wandb(self):
        """Initialize Weights & Biases (optional)"""
        try:
            # Uncomment if you want to use W&B
            # wandb.init(project="nova-ml", entity="your-username")
            print("📊 W&B integration ready (disabled by default)")
        except Exception as e:
            print(f"⚠️ W&B initialization failed: {e}")
    
    def prepare_comprehensive_training_data(self) -> Tuple[List[str], List[int]]:
        """
        Prepare comprehensive training data for all intent categories
        """
        intent_mapping = {
            'coding': 0, 'career': 1, 'business': 2, 'health': 3,
            'emotional': 4, 'technical': 5, 'learning': 6, 'productivity': 7,
            'finance': 8, 'creative': 9, 'lifestyle': 10, 'technology': 11,
            'data_science': 12, 'project_management': 13, 'research': 14
        }
        
        # Comprehensive training dataset
        training_samples = {
            'coding': [
                "write a python function to sort arrays efficiently",
                "help me debug this JavaScript async function",
                "create a REST API with authentication",
                "optimize SQL query performance",
                "implement binary search algorithm",
                "code review for security vulnerabilities",
                "design pattern implementation in Java",
                "unit testing best practices",
                "refactor legacy codebase",
                "database schema design principles",
                "API rate limiting implementation",
                "microservices architecture design",
                "GraphQL vs REST comparison",
                "containerize application with Docker",
                "CI/CD pipeline setup guide",
                "functional programming concepts",
                "memory leak detection techniques",
                "cross-platform mobile development",
                "web scraping with Python",
                "machine learning model deployment"
            ],
            
            'career': [
                "software engineer interview preparation strategies",
                "resume optimization for tech roles",
                "salary negotiation techniques for developers",
                "transition from junior to senior developer",
                "building a strong GitHub portfolio",
                "networking in the tech industry",
                "remote work job search strategies",
                "freelancing vs full-time employment",
                "technical leadership skills development",
                "career change into software development",
                "startup vs big tech company pros and cons",
                "skill assessment for promotion",
                "building personal brand as developer",
                "mentorship opportunities in tech",
                "work-life balance in tech careers",
                "continuous learning and skill updates",
                "conference speaking and community involvement",
                "side project ideas for career growth",
                "interview coding challenge practice",
                "behavioral interview question preparation"
            ],
            
            'business': [
                "startup business plan development",
                "market research and validation strategies",
                "funding options for tech startups",
                "SaaS business model optimization",
                "customer acquisition cost analysis",
                "revenue growth strategies",
                "competitive analysis framework",
                "product-market fit evaluation",
                "go-to-market strategy planning",
                "user retention and engagement metrics",
                "pricing strategy for software products",
                "partnership and collaboration opportunities",
                "scaling team and operations",
                "legal considerations for startups",
                "brand building and marketing",
                "financial forecasting and budgeting",
                "risk assessment and mitigation",
                "exit strategy planning",
                "international expansion strategies",
                "sustainable business practices"
            ],
            
            'health': [
                "ergonomic workspace setup for developers",
                "eye strain prevention for screen work",
                "exercise routines for desk workers",
                "nutrition planning for programmers",
                "stress management in tech jobs",
                "sleep optimization for better productivity",
                "mental health awareness in tech",
                "burnout prevention strategies",
                "building healthy work habits",
                "managing sedentary lifestyle risks",
                "carpal tunnel syndrome prevention",
                "healthy meal prep for busy schedules",
                "meditation and mindfulness practices",
                "staying hydrated during long coding sessions",
                "back pain relief exercises",
                "building immune system strength",
                "managing anxiety in high-pressure jobs",
                "creating work-life boundaries",
                "social connection and community building",
                "regular health checkup importance"
            ],
            
            'emotional': [
                "dealing with imposter syndrome in tech",
                "building confidence as new developer",
                "handling criticism and feedback constructively",
                "managing workplace stress and pressure",
                "overcoming fear of failure",
                "building resilience in challenging projects",
                "communication skills for introverted developers",
                "conflict resolution in team environments",
                "maintaining motivation during difficult tasks",
                "coping with job rejection and setbacks",
                "building emotional intelligence",
                "managing perfectionism tendencies",
                "dealing with toxic work environments",
                "building supportive professional relationships",
                "handling public speaking anxiety",
                "managing time pressure and deadlines",
                "building self-advocacy skills",
                "developing leadership presence",
                "handling career uncertainty",
                "building healthy competitive mindset"
            ],
            
            'technical': [
                "system architecture design principles",
                "scalable database design patterns",
                "cloud infrastructure optimization",
                "security implementation best practices",
                "performance monitoring and optimization",
                "disaster recovery planning",
                "API design and documentation",
                "load balancing strategies",
                "caching mechanisms and strategies",
                "network security implementation",
                "container orchestration with Kubernetes",
                "infrastructure as code practices",
                "monitoring and alerting systems",
                "backup and recovery procedures",
                "capacity planning and scaling",
                "service mesh architecture",
                "event-driven architecture design",
                "data pipeline optimization",
                "real-time system design",
                "compliance and governance frameworks"
            ]
            
            # Add more categories following the same pattern...
        }
        
        # Prepare final training data
        texts, labels = [], []
        for intent, examples in training_samples.items():
            if intent in intent_mapping:
                for example in examples:
                    texts.append(example)
                    labels.append(intent_mapping[intent])
        
        print(f"✅ Prepared {len(texts)} training samples across {len(intent_mapping)} categories")
        return texts, labels, intent_mapping
    
    def create_train_test_split(self, texts: List[str], labels: List[int], 
                               test_size: float = 0.2) -> Tuple[Dataset, Dataset]:
        """
        Create stratified train-test split
        """
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        train_dataset = Dataset.from_dict({
            'text': X_train,
            'labels': y_train
        })
        
        test_dataset = Dataset.from_dict({
            'text': X_test,
            'labels': y_test
        })
        
        return train_dataset, test_dataset
    
    def fine_tune_intent_model(self, save_model: bool = True) -> Dict:
        """
        Fine-tune model with comprehensive training pipeline
        """
        with mlflow.start_run():
            # Prepare data
            texts, labels, intent_mapping = self.prepare_comprehensive_training_data()
            train_dataset, test_dataset = self.create_train_test_split(texts, labels)
            
            # Log parameters
            mlflow.log_params(self.config)
            mlflow.log_param("num_classes", len(intent_mapping))
            mlflow.log_param("train_samples", len(train_dataset))
            mlflow.log_param("test_samples", len(test_dataset))
            
            # Initialize tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config['model_name'],
                num_labels=len(intent_mapping)
            )
            
            # Tokenize datasets
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'], 
                    truncation=True, 
                    padding=True,
                    max_length=self.config['max_length']
                )
            
            train_tokenized = train_dataset.map(tokenize_function, batched=True)
            test_tokenized = test_dataset.map(tokenize_function, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir='./models/intent_classifier',
                num_train_epochs=self.config['num_epochs'],
                per_device_train_batch_size=self.config['batch_size'],
                per_device_eval_batch_size=self.config['batch_size'],
                warmup_steps=self.config['warmup_steps'],
                weight_decay=self.config['weight_decay'],
                logging_dir='./logs',
                logging_steps=100,
                evaluation_strategy="steps",
                eval_steps=200,
                save_strategy="steps",
                save_steps=200,
                load_best_model_at_end=True,
                metric_for_best_model="eval_accuracy",
                greater_is_better=True,
                report_to=None  # Disable default reporting
            )
            
            # Initialize trainer with metrics
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_tokenized,
                eval_dataset=test_tokenized,
                tokenizer=tokenizer,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config['early_stopping_patience'])]
            )
            
            # Train model
            print("🚀 Starting model training...")
            train_result = trainer.train()
            
            # Evaluate model
            print("📊 Evaluating model...")
            eval_result = trainer.evaluate()
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_result.training_loss,
                "eval_loss": eval_result["eval_loss"],
                "eval_accuracy": eval_result["eval_accuracy"],
                "eval_f1": eval_result.get("eval_f1", 0)
            })
            
            # Save model
            if save_model:
                model_path = './models/intent_classifier_finetuned'
                trainer.save_model(model_path)
                tokenizer.save_pretrained(model_path)
                
                # Save intent mapping
                with open(f"{model_path}/intent_mapping.json", 'w') as f:
                    json.dump(intent_mapping, f, indent=2)
                
                # Log model to MLflow
                mlflow.pytorch.log_model(model, "intent_model")
            
            results = {
                "train_loss": train_result.training_loss,
                "eval_accuracy": eval_result["eval_accuracy"],
                "eval_loss": eval_result["eval_loss"],
                "model_path": model_path if save_model else None
            }
            
            print(f"✅ Model training completed! Accuracy: {eval_result['eval_accuracy']:.4f}")
            return results
    
    def compute_metrics(self, eval_pred):
        """Compute comprehensive evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def continuous_learning_setup(self):
        """
        Setup continuous learning pipeline for model improvement
        """
        feedback_file = Path("user_feedback.jsonl")
        if feedback_file.exists():
            print("📈 Setting up continuous learning pipeline...")
            
            # Read feedback data
            feedback_data = []
            with open(feedback_file, 'r') as f:
                for line in f:
                    feedback_data.append(json.loads(line))
            
            # Analyze feedback patterns
            accuracy_trend = [item['accuracy'] for item in feedback_data[-100:]]
            recent_accuracy = np.mean(accuracy_trend) if accuracy_trend else 0
            
            print(f"📊 Recent prediction accuracy: {recent_accuracy:.3f}")
            
            # Retrain if accuracy drops below threshold
            if recent_accuracy < 0.7 and len(feedback_data) > 50:
                print("🔄 Triggering model retraining based on feedback...")
                return self.fine_tune_intent_model()
        
        return None
    
    def model_performance_analysis(self, model_path: str = './models/intent_classifier_finetuned'):
        """
        Analyze model performance and generate insights
        """
        if not Path(model_path).exists():
            print("❌ Model not found. Please train the model first.")
            return
        
        # Load model and test
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Load intent mapping
            with open(f"{model_path}/intent_mapping.json", 'r') as f:
                intent_mapping = json.load(f)
            
            # Test with sample queries
            test_queries = [
                "help me write a python function",
                "I need career advice for interviews",
                "create a business plan",
                "I'm feeling stressed at work"
            ]
            
            print("🧪 Testing model performance:")
            for query in test_queries:
                inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_class = predictions.argmax().item()
                    confidence = predictions.max().item()
                
                # Find intent name
                intent_name = [k for k, v in intent_mapping.items() if v == predicted_class][0]
                print(f"Query: '{query}'")
                print(f"Predicted: {intent_name} (confidence: {confidence:.3f})")
                print("-" * 50)
                
        except Exception as e:
            print(f"❌ Error during performance analysis: {e}")
