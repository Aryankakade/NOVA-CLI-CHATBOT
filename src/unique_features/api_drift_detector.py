# src/unique_features/api_drift_detector.py
import time
import json
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional
import logging
import asyncio

class APIPerformanceDrifter:
    """Real-time API performance monitoring & automatic switching"""
    
    def __init__(self):
        self.api_response_history = {
            'groq': deque(maxlen=100),
            'openrouter': deque(maxlen=100),
            'huggingface': deque(maxlen=100),
            'local': deque(maxlen=100)
        }
        
        self.quality_baselines = {}
        self.performance_thresholds = {
            'response_time': 5.0,  # seconds
            'quality_score': 0.7,  # 0-1 scale
            'success_rate': 0.9,   # 90% success rate
            'drift_threshold': 0.8  # 80% of baseline
        }
        
        self.api_health_status = {}
        self.last_performance_check = {}
        self.drift_alerts = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup performance logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/performance_logs/api_performance.log'),
                logging.StreamHandler()
            ]
        )
    
    def calculate_response_quality(self, response: str, expected_length: int = 100) -> float:
        """Calculate response quality score (0-1)"""
        
        if not response or len(response.strip()) < 10:
            return 0.0
        
        quality_factors = {
            'length_score': min(len(response) / expected_length, 1.0),
            'coherence_score': self.assess_coherence(response),
            'completeness_score': self.assess_completeness(response),
            'relevance_score': self.assess_relevance(response)
        }
        
        # Weighted average
        weights = {'length_score': 0.2, 'coherence_score': 0.3, 
                  'completeness_score': 0.3, 'relevance_score': 0.2}
        
        total_score = sum(quality_factors[factor] * weights[factor] 
                         for factor in quality_factors)
        
        return min(total_score, 1.0)
    
    def assess_coherence(self, response: str) -> float:
        """Assess response coherence (simplified)"""
        
        # Check for basic coherence indicators
        coherence_indicators = {
            'complete_sentences': len([s for s in response.split('.') if len(s.strip()) > 5]),
            'proper_structure': 1.0 if any(word in response.lower() for word in 
                                         ['however', 'therefore', 'because', 'since', 'although']) else 0.5,
            'no_repetition': 1.0 if not self.has_excessive_repetition(response) else 0.3
        }
        
        return np.mean(list(coherence_indicators.values()))
    
    def assess_completeness(self, response: str) -> float:
        """Assess response completeness"""
        
        completeness_indicators = {
            'sufficient_length': min(len(response) / 100, 1.0),
            'addresses_query': 1.0 if len(response.split()) > 10 else 0.5,
            'provides_details': 1.0 if any(word in response.lower() for word in 
                                         ['because', 'example', 'specifically', 'details']) else 0.6
        }
        
        return np.mean(list(completeness_indicators.values()))
    
    def assess_relevance(self, response: str) -> float:
        """Assess response relevance (simplified)"""
        
        # Basic relevance check
        if len(response.strip()) < 20:
            return 0.2
        
        # Check for generic/template responses
        generic_phrases = ['i can help', 'let me assist', 'here is some information']
        
        if any(phrase in response.lower() for phrase in generic_phrases):
            return 0.6
        
        return 0.8  # Default relevance score
    
    def has_excessive_repetition(self, response: str) -> bool:
        """Check for excessive repetition in response"""
        
        words = response.lower().split()
        word_counts = {}
        
        for word in words:
            if len(word) > 3:  # Skip short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Check if any word appears more than 20% of the time
        max_repetition = max(word_counts.values()) if word_counts else 0
        repetition_rate = max_repetition / len(words) if words else 0
        
        return repetition_rate > 0.2
    
    async def monitor_api_call(self, api_name: str, response: str, 
                              response_time: float, success: bool) -> Dict:
        """Monitor individual API call performance"""
        
        # Calculate quality score
        quality_score = self.calculate_response_quality(response) if success else 0.0
        
        # Create performance record
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'api_name': api_name,
            'response_time': response_time,
            'quality_score': quality_score,
            'success': success,
            'response_length': len(response) if response else 0
        }
        
        # Store in history
        if api_name in self.api_response_history:
            self.api_response_history[api_name].append(performance_record)
        
        # Update baseline if needed
        await self.update_baseline(api_name)
        
        # Check for drift
        drift_detected = await self.detect_drift(api_name)
        
        # Log performance
        self.logger.info(f"API {api_name}: Response time: {response_time:.2f}s, "
                        f"Quality: {quality_score:.2f}, Success: {success}")
        
        return {
            'performance_record': performance_record,
            'drift_detected': drift_detected,
            'api_health': self.get_api_health(api_name)
        }
    
    async def update_baseline(self, api_name: str):
        """Update performance baseline for API"""
        
        history = self.api_response_history.get(api_name, [])
        
        if len(history) >= 10:  # Need minimum data points
            successful_calls = [record for record in history if record['success']]
            
            if successful_calls:
                avg_quality = np.mean([call['quality_score'] for call in successful_calls])
                avg_response_time = np.mean([call['response_time'] for call in successful_calls])
                success_rate = len(successful_calls) / len(history)
                
                self.quality_baselines[api_name] = {
                    'avg_quality': avg_quality,
                    'avg_response_time': avg_response_time,
                    'success_rate': success_rate,
                    'last_updated': datetime.now().isoformat(),
                    'sample_size': len(successful_calls)
                }
    
    async def detect_drift(self, api_name: str) -> bool:
        """Detect performance drift for specific API"""
        
        if api_name not in self.quality_baselines:
            return False
        
        baseline = self.quality_baselines[api_name]
        recent_history = list(self.api_response_history.get(api_name, []))[-10:]
        
        if len(recent_history) < 5:
            return False
        
        # Calculate recent performance
        recent_successful = [record for record in recent_history if record['success']]
        
        if not recent_successful:
            # All recent calls failed - definitely drift
            self.logger.warning(f"API {api_name}: All recent calls failed!")
            return True
        
        recent_avg_quality = np.mean([call['quality_score'] for call in recent_successful])
        recent_avg_response_time = np.mean([call['response_time'] for call in recent_successful])
        recent_success_rate = len(recent_successful) / len(recent_history)
        
        # Check drift conditions
        quality_drift = recent_avg_quality < (baseline['avg_quality'] * self.performance_thresholds['drift_threshold'])
        response_time_drift = recent_avg_response_time > (baseline['avg_response_time'] * 1.5)
        success_rate_drift = recent_success_rate < self.performance_thresholds['success_rate']
        
        drift_detected = quality_drift or response_time_drift or success_rate_drift
        
        if drift_detected:
            drift_alert = {
                'api_name': api_name,
                'drift_type': [],
                'baseline_quality': baseline['avg_quality'],
                'recent_quality': recent_avg_quality,
                'baseline_response_time': baseline['avg_response_time'],
                'recent_response_time': recent_avg_response_time,
                'baseline_success_rate': baseline['success_rate'],
                'recent_success_rate': recent_success_rate,
                'timestamp': datetime.now().isoformat()
            }
            
            if quality_drift:
                drift_alert['drift_type'].append('quality_degradation')
            if response_time_drift:
                drift_alert['drift_type'].append('response_time_increase')
            if success_rate_drift:
                drift_alert['drift_type'].append('success_rate_drop')
            
            self.drift_alerts.append(drift_alert)
            
            self.logger.warning(f"DRIFT DETECTED for {api_name}: {drift_alert['drift_type']}")
        
        return drift_detected
    
    def get_api_health(self, api_name: str) -> Dict:
        """Get current API health status"""
        
        recent_history = list(self.api_response_history.get(api_name, []))[-20:]
        
        if not recent_history:
            return {'status': 'unknown', 'reason': 'no_data'}
        
        recent_successful = [record for record in recent_history if record['success']]
        success_rate = len(recent_successful) / len(recent_history)
        
        if success_rate >= 0.9:
            status = 'healthy'
        elif success_rate >= 0.7:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        avg_quality = np.mean([call['quality_score'] for call in recent_successful]) if recent_successful else 0
        avg_response_time = np.mean([call['response_time'] for call in recent_successful]) if recent_successful else 0
        
        return {
            'status': status,
            'success_rate': success_rate,
            'avg_quality': avg_quality,
            'avg_response_time': avg_response_time,
            'total_calls': len(recent_history),
            'successful_calls': len(recent_successful),
            'last_updated': datetime.now().isoformat()
        }
    
    def trigger_api_switch(self, failed_api: str) -> str:
        """Automatically switch to best available API"""
        
        # Get health status of all APIs
        api_health_scores = {}
        
        for api_name in self.api_response_history.keys():
            if api_name != failed_api:
                health = self.get_api_health(api_name)
                
                # Calculate composite health score
                health_score = (
                    health['success_rate'] * 0.4 +
                    health['avg_quality'] * 0.4 +
                    (1 / (health['avg_response_time'] + 0.1)) * 0.2
                )
                
                api_health_scores[api_name] = health_score
        
        if api_health_scores:
            best_api = max(api_health_scores, key=api_health_scores.get)
            
            self.logger.info(f"Switching from {failed_api} to {best_api} "
                           f"(health score: {api_health_scores[best_api]:.2f})")
            
            return best_api
        
        # Fallback to local model
        return 'local'
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'api_performance': {},
            'drift_alerts': self.drift_alerts[-10:],  # Last 10 alerts
            'recommendations': []
        }
        
        for api_name in self.api_response_history.keys():
            health = self.get_api_health(api_name)
            baseline = self.quality_baselines.get(api_name, {})
            
            report['api_performance'][api_name] = {
                'health': health,
                'baseline': baseline,
                'total_calls': len(self.api_response_history[api_name])
            }
            
            # Generate recommendations
            if health['status'] == 'unhealthy':
                report['recommendations'].append(f"Consider avoiding {api_name} temporarily")
            elif health['status'] == 'degraded':
                report['recommendations'].append(f"Monitor {api_name} closely")
        
        return report

# Usage Example
async def example_usage():
    """Example of how to use API drift detection"""
    
    detector = APIPerformanceDrifter()
    
    # Simulate API call monitoring
    response_time = 2.3
    response_text = "This is a sample API response with good quality content."
    success = True
    
    result = await detector.monitor_api_call('groq', response_text, response_time, success)
    
    print(f"Drift detected: {result['drift_detected']}")
    print(f"API health: {result['api_health']}")
    
    # Get performance report
    report = detector.get_performance_report()
    print(f"Performance report: {json.dumps(report, indent=2)}")

if __name__ == "__main__":
    asyncio.run(example_usage())
