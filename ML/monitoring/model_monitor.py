# ml/monitoring/model_monitor.py
import numpy as np
from scipy import stats
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class MLModelMonitor:
    def __init__(self, history_size: int = 1000):
        """
        Monitor ML model performance and detect issues
        Purpose: Production-ready ML system monitoring and alerting
        """
        print("🔄 Initializing ML Model Monitor...")
        
        # Performance tracking
        self.performance_history = deque(maxlen=history_size)
        self.prediction_history = deque(maxlen=history_size)
        self.error_log = deque(maxlen=100)
        
        # Baseline metrics for drift detection
        self.baseline_metrics = {}
        self.drift_thresholds = {
            'confidence_drop': 0.15,  # 15% drop in average confidence
            'accuracy_drop': 0.10,    # 10% drop in accuracy
            'distribution_shift': 0.20  # 20% change in prediction distribution
        }
        
        # Alert settings
        self.alerts = []
        self.alert_cooldown = {}
        
        # Create monitoring directory
        Path("./logs/monitoring").mkdir(parents=True, exist_ok=True)
        
        print("✅ ML Model Monitor ready!")
    
    def track_prediction(self, prediction_data: Dict):
        """
        Track a single prediction with metadata
        
        Args:
            prediction_data: Dictionary containing prediction info
                - predicted_class: The predicted class/intent
                - confidence: Prediction confidence (0-1)
                - query: Original user query
                - processing_time: Time taken for prediction (ms)
                - model_version: Version of model used
        """
        timestamp = datetime.now()
        
        # Enhance prediction data with timestamp
        enhanced_data = {
            'timestamp': timestamp.isoformat(),
            'predicted_class': prediction_data.get('predicted_class', 'unknown'),
            'confidence': float(prediction_data.get('confidence', 0.0)),
            'query_length': len(prediction_data.get('query', '')),
            'processing_time_ms': prediction_data.get('processing_time', 0),
            'model_version': prediction_data.get('model_version', 'v1.0')
        }
        
        self.prediction_history.append(enhanced_data)
        
        # Check for immediate issues
        self._check_immediate_issues(enhanced_data)
    
    def track_performance_batch(self, predictions: List[Dict], 
                              ground_truth: Optional[List] = None):
        """
        Track batch of predictions for performance analysis
        """
        timestamp = datetime.now()
        
        if not predictions:
            return
        
        # Calculate batch metrics
        confidences = [p.get('confidence', 0) for p in predictions]
        processing_times = [p.get('processing_time', 0) for p in predictions]
        
        batch_metrics = {
            'timestamp': timestamp.isoformat(),
            'batch_size': len(predictions),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'confidence_std': np.std(confidences),
            'low_confidence_count': sum(1 for c in confidences if c < 0.5),
            'low_confidence_rate': sum(1 for c in confidences if c < 0.5) / len(confidences),
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'prediction_distribution': self._calculate_class_distribution(predictions)
        }
        
        # Add accuracy if ground truth provided
        if ground_truth and len(ground_truth) == len(predictions):
            correct = sum(1 for p, t in zip(predictions, ground_truth) 
                         if p.get('predicted_class') == t)
            batch_metrics['accuracy'] = correct / len(predictions)
            batch_metrics['has_ground_truth'] = True
        else:
            batch_metrics['has_ground_truth'] = False
        
        self.performance_history.append(batch_metrics)
        
        # Update baseline if this is early data
        if len(self.performance_history) <= 10:
            self._update_baseline_metrics(batch_metrics)
        
        # Check for performance issues
        self._check_performance_drift(batch_metrics)
        
        return batch_metrics
    
    def _calculate_class_distribution(self, predictions: List[Dict]) -> Dict[str, float]:
        """Calculate distribution of predicted classes"""
        class_counts = {}
        total = len(predictions)
        
        for pred in predictions:
            predicted_class = pred.get('predicted_class', 'unknown')
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
        
        # Convert to percentages
        return {cls: count/total for cls, count in class_counts.items()}
    
    def detect_model_drift(self, window_hours: int = 24) -> Dict[str, any]:
        """
        Detect if model performance is degrading
        
        Args:
            window_hours: Hours to look back for drift detection
            
        Returns:
            Dictionary with drift analysis results
        """
        if len(self.performance_history) < 5:
            return {
                'drift_detected': False,
                'reason': 'insufficient_data',
                'recommendations': ['Collect more prediction data']
            }
        
        # Get recent performance data
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_data = [
            data for data in self.performance_history
            if datetime.fromisoformat(data['timestamp']) > cutoff_time
        ]
        
        if len(recent_data) < 3:
            return {
                'drift_detected': False,
                'reason': 'insufficient_recent_data',
                'recommendations': ['Wait for more recent predictions']
            }
        
        # Calculate recent metrics
        recent_confidences = [d['avg_confidence'] for d in recent_data]
        recent_avg_confidence = np.mean(recent_confidences)
        
        # Compare with baseline
        baseline_confidence = self.baseline_metrics.get('avg_confidence', 0.8)
        confidence_drop = baseline_confidence - recent_avg_confidence
        
        # Check accuracy drift if available
        recent_accuracies = [d['accuracy'] for d in recent_data if d.get('has_ground_truth')]
        accuracy_drift = False
        accuracy_drop = 0
        
        if recent_accuracies and self.baseline_metrics.get('accuracy'):
            recent_avg_accuracy = np.mean(recent_accuracies)
            baseline_accuracy = self.baseline_metrics.get('accuracy', 0.9)
            accuracy_drop = baseline_accuracy - recent_avg_accuracy
            accuracy_drift = accuracy_drop > self.drift_thresholds['accuracy_drop']
        
        # Check prediction distribution drift
        if len(recent_data) > 1:
            recent_distribution = recent_data[-1]['prediction_distribution']
            baseline_distribution = self.baseline_metrics.get('prediction_distribution', {})
            distribution_drift = self._calculate_distribution_drift(
                recent_distribution, baseline_distribution
            )
        else:
            distribution_drift = 0
        
        # Determine if drift occurred
        confidence_drift = confidence_drop > self.drift_thresholds['confidence_drop']
        distribution_drift_detected = distribution_drift > self.drift_thresholds['distribution_shift']
        
        drift_detected = confidence_drift or accuracy_drift or distribution_drift_detected
        
        # Generate recommendations
        recommendations = []
        if confidence_drift:
            recommendations.append("Model confidence dropping - consider retraining")
        if accuracy_drift:
            recommendations.append("Accuracy degradation detected - retrain with recent data")
        if distribution_drift_detected:
            recommendations.append("Input distribution changed - update training data")
        
        if not recommendations:
            recommendations.append("Model performance stable")
        
        drift_analysis = {
            'drift_detected': drift_detected,
            'confidence_drift': {
                'detected': confidence_drift,
                'baseline': baseline_confidence,
                'recent': recent_avg_confidence,
                'drop': confidence_drop
            },
            'accuracy_drift': {
                'detected': accuracy_drift,
                'baseline': self.baseline_metrics.get('accuracy', 0),
                'recent': np.mean(recent_accuracies) if recent_accuracies else 0,
                'drop': accuracy_drop
            },
            'distribution_drift': {
                'detected': distribution_drift_detected,
                'shift_score': distribution_drift
            },
            'recommendations': recommendations,
            'window_hours': window_hours,
            'data_points_analyzed': len(recent_data)
        }
        
        # Create alert if drift detected
        if drift_detected:
            self._create_alert('model_drift', drift_analysis)
        
        return drift_analysis
    
    def _calculate_distribution_drift(self, current_dist: Dict, baseline_dist: Dict) -> float:
        """Calculate drift score between two distributions using Jensen-Shannon divergence"""
        if not baseline_dist:
            return 0
        
        # Get all classes
        all_classes = set(list(current_dist.keys()) + list(baseline_dist.keys()))
        
        # Convert to arrays
        current_array = np.array([current_dist.get(cls, 0) for cls in all_classes])
        baseline_array = np.array([baseline_dist.get(cls, 0) for cls in all_classes])
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        current_array += epsilon
        baseline_array += epsilon
        
        # Normalize
        current_array /= current_array.sum()
        baseline_array /= baseline_array.sum()
        
        # Calculate Jensen-Shannon divergence
        m = 0.5 * (current_array + baseline_array)
        js_divergence = 0.5 * stats.entropy(current_array, m) + 0.5 * stats.entropy(baseline_array, m)
        
        return float(js_divergence)
    
    def _update_baseline_metrics(self, metrics: Dict):
        """Update baseline metrics with early performance data"""
        for key in ['avg_confidence', 'accuracy', 'prediction_distribution']:
            if key in metrics:
                if key not in self.baseline_metrics:
                    self.baseline_metrics[key] = metrics[key]
                else:
                    # Running average for numerical values
                    if isinstance(metrics[key], (int, float)):
                        self.baseline_metrics[key] = 0.9 * self.baseline_metrics[key] + 0.1 * metrics[key]
                    else:
                        self.baseline_metrics[key] = metrics[key]
    
    def _check_immediate_issues(self, prediction_data: Dict):
        """Check for immediate issues with single prediction"""
        issues = []
        
        # Very low confidence
        if prediction_data['confidence'] < 0.3:
            issues.append('very_low_confidence')
        
        # Slow processing
        if prediction_data['processing_time_ms'] > 1000:  # 1 second
            issues.append('slow_processing')
        
        # Very short or long query
        query_length = prediction_data['query_length']
        if query_length < 5:
            issues.append('very_short_query')
        elif query_length > 500:
            issues.append('very_long_query')
        
        if issues:
            self._log_issue('immediate_issue', {
                'timestamp': prediction_data['timestamp'],
                'issues': issues,
                'prediction_data': prediction_data
            })
    
    def _check_performance_drift(self, batch_metrics: Dict):
        """Check for performance drift in batch"""
        if batch_metrics['low_confidence_rate'] > 0.5:  # More than 50% low confidence
            self._create_alert('high_low_confidence_rate', {
                'rate': batch_metrics['low_confidence_rate'],
                'timestamp': batch_metrics['timestamp']
            })
    
    def _create_alert(self, alert_type: str, details: Dict):
        """Create and store alert"""
        # Check cooldown to avoid spam
        cooldown_key = alert_type
        now = datetime.now()
        
        if cooldown_key in self.alert_cooldown:
            last_alert = self.alert_cooldown[cooldown_key]
            if (now - last_alert).total_seconds() < 3600:  # 1 hour cooldown
                return
        
        alert = {
            'alert_type': alert_type,
            'timestamp': now.isoformat(),
            'details': details,
            'severity': self._get_alert_severity(alert_type)
        }
        
        self.alerts.append(alert)
        self.alert_cooldown[cooldown_key] = now
        
        print(f"⚠️ ALERT [{alert['severity']}]: {alert_type}")
        print(f"   Time: {alert['timestamp']}")
        if 'recommendations' in details:
            print(f"   Recommendations: {details['recommendations']}")
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Get severity level for alert type"""
        severity_map = {
            'model_drift': 'HIGH',
            'high_low_confidence_rate': 'MEDIUM',
            'immediate_issue': 'LOW'
        }
        return severity_map.get(alert_type, 'MEDIUM')
    
    def _log_issue(self, issue_type: str, details: Dict):
        """Log issue to error log"""
        error_entry = {
            'issue_type': issue_type,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.error_log.append(error_entry)
    
    def generate_monitoring_report(self, hours_back: int = 24) -> Dict:
        """
        Generate comprehensive monitoring report
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter recent data
        recent_performance = [
            p for p in self.performance_history
            if datetime.fromisoformat(p['timestamp']) > cutoff_time
        ]
        
        recent_predictions = [
            p for p in self.prediction_history
            if datetime.fromisoformat(p['timestamp']) > cutoff_time
        ]
        
        if not recent_performance and not recent_predictions:
            return {
                'status': 'no_recent_data',
                'message': f'No data found for the last {hours_back} hours'
            }
        
        # Calculate summary metrics
        total_predictions = sum(p['batch_size'] for p in recent_performance)
        if not total_predictions:
            total_predictions = len(recent_predictions)
        
        avg_confidence = np.mean([p['avg_confidence'] for p in recent_performance]) if recent_performance else 0
        
        # Get recent alerts
        recent_alerts = [
            a for a in self.alerts
            if datetime.fromisoformat(a['timestamp']) > cutoff_time
        ]
        
        # Drift analysis
        drift_analysis = self.detect_model_drift(hours_back)
        
        report = {
            'report_period': f'Last {hours_back} hours',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_predictions': total_predictions,
                'avg_confidence': avg_confidence,
                'performance_batches': len(recent_performance),
                'alerts_count': len(recent_alerts)
            },
            'performance_metrics': {
                'confidence_trend': [p['avg_confidence'] for p in recent_performance[-10:]],
                'low_confidence_rate_trend': [p['low_confidence_rate'] for p in recent_performance[-10:]],
                'processing_time_trend': [p['avg_processing_time'] for p in recent_performance[-10:]]
            },
            'drift_analysis': drift_analysis,
            'recent_alerts': recent_alerts,
            'recommendations': self._generate_recommendations(recent_performance, drift_analysis, recent_alerts),
            'health_status': self._calculate_health_status(drift_analysis, recent_alerts, avg_confidence)
        }
        
        return report
    
    def _generate_recommendations(self, performance_data: List, drift_analysis: Dict, alerts: List) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on drift analysis
        if drift_analysis.get('drift_detected'):
            recommendations.extend(drift_analysis.get('recommendations', []))
        
        # Based on performance trends
        if performance_data:
            recent_confidence = [p['avg_confidence'] for p in performance_data[-5:]]
            if len(recent_confidence) > 2 and np.mean(recent_confidence) < 0.6:
                recommendations.append("Consider model retraining - confidence consistently low")
        
        # Based on alerts
        high_severity_alerts = [a for a in alerts if a.get('severity') == 'HIGH']
        if len(high_severity_alerts) > 2:
            recommendations.append("Multiple high-severity alerts - immediate attention required")
        
        if not recommendations:
            recommendations.append("Model performance appears stable")
        
        return recommendations
    
    def _calculate_health_status(self, drift_analysis: Dict, alerts: List, avg_confidence: float) -> str:
        """Calculate overall health status"""
        if drift_analysis.get('drift_detected'):
            return 'DEGRADED'
        
        high_severity_alerts = [a for a in alerts if a.get('severity') == 'HIGH']
        if high_severity_alerts:
            return 'WARNING'
        
        if avg_confidence < 0.6:
            return 'WARNING'
        
        return 'HEALTHY'
    
    def get_monitoring_stats(self) -> Dict:
        """Get monitoring system statistics"""
        return {
            'performance_entries': len(self.performance_history),
            'prediction_entries': len(self.prediction_history),
            'total_alerts': len(self.alerts),
            'baseline_established': bool(self.baseline_metrics),
            'monitoring_active_hours': 24,  # Assuming always monitoring last 24h
            'drift_thresholds': self.drift_thresholds,
            'alert_types': list(set(a['alert_type'] for a in self.alerts))
        }
