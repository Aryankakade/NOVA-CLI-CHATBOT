# ml/mlops/experiment_tracker.py
import mlflow
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import pickle
import numpy as np

class ModelExperimentTracker:
    def __init__(self):
        """
        Comprehensive experiment tracking for ML workflows
        Purpose: Professional ML experiment management and model versioning
        """
        print("🔄 Initializing MLOps Experiment Tracker...")
        
        # Setup MLflow
        self._setup_mlflow()
        
        # Experiment history storage
        self.experiment_history = []
        self.models_registry = {}
        
        # Create necessary directories
        self._create_directories()
        
        print("✅ MLOps Experiment Tracker ready!")
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            # Set tracking URI to local sqlite database
            mlflow_dir = Path("./mlflow_experiments")
            mlflow_dir.mkdir(exist_ok=True)
            
            mlflow.set_tracking_uri(f"sqlite:///{mlflow_dir}/mlflow.db")
            
            # Set or create experiment
            experiment_name = "nova_ml_experiments"
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                    print(f"✅ Created new MLflow experiment: {experiment_name}")
                else:
                    experiment_id = experiment.experiment_id
                    print(f"✅ Using existing MLflow experiment: {experiment_name}")
                    
                mlflow.set_experiment(experiment_name)
                
            except Exception as e:
                print(f"⚠️ MLflow experiment setup issue: {e}")
                
        except Exception as e:
            print(f"⚠️ MLflow initialization failed: {e}")
    
    def _create_directories(self):
        """Create necessary directories for model storage"""
        directories = [
            "./models/experiments",
            "./models/production",
            "./models/archived",
            "./logs/experiments"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def start_experiment(self, experiment_name: str, model_type: str, 
                        parameters: Dict[str, Any]) -> str:
        """
        Start a new ML experiment
        
        Args:
            experiment_name: Name of the experiment
            model_type: Type of model (intent_classifier, embeddings, etc.)
            parameters: Model hyperparameters
            
        Returns:
            Experiment ID for tracking
        """
        experiment_info = {
            'experiment_id': f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'name': experiment_name,
            'model_type': model_type,
            'parameters': parameters,
            'start_time': datetime.now().isoformat(),
            'status': 'running'
        }
        
        try:
            with mlflow.start_run(run_name=experiment_name) as run:
                # Log basic info
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("experiment_name", experiment_name)
                
                # Log all parameters
                for key, value in parameters.items():
                    mlflow.log_param(key, value)
                
                # Store run info
                experiment_info['mlflow_run_id'] = run.info.run_id
                
        except Exception as e:
            print(f"⚠️ MLflow logging failed: {e}")
        
        self.experiment_history.append(experiment_info)
        print(f"🚀 Started experiment: {experiment_name} ({experiment_info['experiment_id']})")
        
        return experiment_info['experiment_id']
    
    def log_metrics(self, experiment_id: str, metrics: Dict[str, float], step: int = 0):
        """
        Log metrics for an experiment
        
        Args:
            experiment_id: Experiment identifier
            metrics: Dictionary of metrics to log
            step: Training step (for time series metrics)
        """
        try:
            # Find experiment
            experiment = self._find_experiment(experiment_id)
            if not experiment:
                print(f"⚠️ Experiment {experiment_id} not found")
                return
            
            # Log to MLflow if available
            if 'mlflow_run_id' in experiment:
                with mlflow.start_run(run_id=experiment['mlflow_run_id']):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(metric_name, value, step)
            
            # Store in experiment history
            if 'metrics' not in experiment:
                experiment['metrics'] = []
            
            experiment['metrics'].append({
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'values': metrics
            })
            
            print(f"📊 Logged metrics for {experiment_id}: {metrics}")
            
        except Exception as e:
            print(f"❌ Failed to log metrics: {e}")
    
    def log_model_artifact(self, experiment_id: str, model_object: Any, 
                          model_name: str, metadata: Dict = None):
        """
        Save model artifact with versioning
        
        Args:
            experiment_id: Experiment identifier
            model_object: The trained model
            model_name: Name for the model
            metadata: Additional model metadata
        """
        try:
            experiment = self._find_experiment(experiment_id)
            if not experiment:
                print(f"⚠️ Experiment {experiment_id} not found")
                return
            if metadata:
             metadata = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in metadata.items()}
            
            # Create model path
            model_dir = Path(f"./models/experiments/{experiment_id}")
            model_dir.mkdir(exist_ok=True)
            
            model_path = model_dir / f"{model_name}.pkl"
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model_object, f)
            
            # Save metadata
            model_metadata = {
                'model_name': model_name,
                'model_path': str(model_path),
                'experiment_id': experiment_id,
                'save_time': datetime.now().isoformat(),
                'model_type': experiment.get('model_type', 'unknown'),
                **(metadata or {})
            }
            
            metadata_path = model_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            # Log to MLflow
            if 'mlflow_run_id' in experiment:
                with mlflow.start_run(run_id=experiment['mlflow_run_id']):
                    mlflow.log_artifact(str(model_path))
                    mlflow.log_artifact(str(metadata_path))
            
            # Update experiment
            experiment['model_artifacts'] = experiment.get('model_artifacts', [])
            experiment['model_artifacts'].append(model_metadata)
            
            print(f"💾 Saved model artifact: {model_name}")
            
        except Exception as e:
            print(f"❌ Failed to save model artifact: {e}")
    
    def finish_experiment(self, experiment_id: str, final_metrics: Dict[str, float],
                         success: bool = True):
        """
        Finish an experiment and record final results
        """
        try:
            experiment = self._find_experiment(experiment_id)
            if not experiment:
                print(f"⚠️ Experiment {experiment_id} not found")
                return
            
            # Update experiment status
            experiment['status'] = 'completed' if success else 'failed'
            experiment['end_time'] = datetime.now().isoformat()
            experiment['final_metrics'] = final_metrics
            
            # Calculate duration
            start_time = datetime.fromisoformat(experiment['start_time'])
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            experiment['duration_seconds'] = duration
            
            # Log final metrics
            if final_metrics:
                self.log_metrics(experiment_id, final_metrics)
            
            # Log to MLflow
            if 'mlflow_run_id' in experiment:
                with mlflow.start_run(run_id=experiment['mlflow_run_id']):
                    mlflow.log_metric("duration_seconds", duration)
                    mlflow.log_param("experiment_status", experiment['status'])
            
            print(f"✅ Finished experiment: {experiment['name']} ({experiment_id})")
            print(f"   Status: {experiment['status']}")
            print(f"   Duration: {duration:.2f} seconds")
            if final_metrics:
                print(f"   Final metrics: {final_metrics}")
            
        except Exception as e:
            print(f"❌ Failed to finish experiment: {e}")
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments
        """
        comparison_data = {
            'experiments': [],
            'best_experiment': None,
            'comparison_metrics': {}
        }
        
        for exp_id in experiment_ids:
            experiment = self._find_experiment(exp_id)
            if experiment:
                comparison_data['experiments'].append({
                    'experiment_id': exp_id,
                    'name': experiment.get('name', 'unknown'),
                    'model_type': experiment.get('model_type', 'unknown'),
                    'final_metrics': experiment.get('final_metrics', {}),
                    'duration': experiment.get('duration_seconds', 0),
                    'status': experiment.get('status', 'unknown')
                })
        
        # Find best experiment based on accuracy (if available)
        best_accuracy = 0
        best_exp = None
        
        for exp in comparison_data['experiments']:
            accuracy = exp['final_metrics'].get('accuracy', 0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_exp = exp
        
        if best_exp:
            comparison_data['best_experiment'] = best_exp
        
        return comparison_data
    
    def get_experiment_history(self) -> List[Dict]:
        """Get all experiment history"""
        return self.experiment_history
    
    def _find_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Find experiment by ID"""
        for exp in self.experiment_history:
            if exp['experiment_id'] == experiment_id:
                return exp
        return None
    
    def save_experiment_history(self):
        """Save experiment history to disk"""
        try:
            history_path = Path("./logs/experiments/experiment_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.experiment_history, f, indent=2)
            print("💾 Experiment history saved")
        except Exception as e:
            print(f"⚠️ Failed to save experiment history: {e}")
    
    def load_experiment_history(self):
        """Load experiment history from disk"""
        try:
            history_path = Path("./logs/experiments/experiment_history.json")
            if history_path.exists():
                with open(history_path, 'r') as f:
                    self.experiment_history = json.load(f)
                print("📂 Experiment history loaded")
        except Exception as e:
            print(f"⚠️ Failed to load experiment history: {e}")
    
    def get_tracking_stats(self) -> Dict:
        """Get experiment tracking statistics"""
        completed_experiments = [e for e in self.experiment_history if e['status'] == 'completed']
        
        return {
            'total_experiments': len(self.experiment_history),
            'completed_experiments': len(completed_experiments),
            'avg_experiment_duration': np.mean([e.get('duration_seconds', 0) for e in completed_experiments]) if completed_experiments else 0,
            'model_types': list(set(e.get('model_type', 'unknown') for e in self.experiment_history)),
            'mlflow_integration': True,
            'artifact_storage': './models/experiments'
        }
