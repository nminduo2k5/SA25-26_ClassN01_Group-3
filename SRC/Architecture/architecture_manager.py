"""
Architecture Manager - Orchestrates all 4 architectures and combines results
"""

from typing import Dict, List, Any
import logging
from .ensemble_architecture import EnsembleArchitecture
from .hierarchical_architecture import HierarchicalArchitecture
from .round_robin_architecture import RoundRobinArchitecture
from .agent_driven_architecture import AgentDrivenArchitecture

logger = logging.getLogger(__name__)

class ArchitectureManager:
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.architectures = {
            'ensemble': EnsembleArchitecture(agents),
            'hierarchical': HierarchicalArchitecture(agents),
            'round_robin': RoundRobinArchitecture(agents),
            'agent_driven': AgentDrivenArchitecture(agents)
        }
    
    def predict_all_architectures(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """Run prediction using all 4 architectures"""
        
        results = {}
        
        for arch_name, architecture in self.architectures.items():
            try:
                logger.info(f"Running {arch_name} architecture...")
                result = architecture.predict_price(symbol, data)
                results[arch_name] = result
                
            except Exception as e:
                logger.error(f"Architecture {arch_name} failed: {e}")
                results[arch_name] = {'error': str(e)}
        
        # Combine all architecture results
        return self._combine_architecture_results(results, symbol)
    
    def _combine_architecture_results(self, results: Dict, symbol: str) -> Dict[str, Any]:
        """Combine results from all architectures into final prediction"""
        
        timeframes = ['short_term', 'medium_term', 'long_term']
        final_predictions = {}
        architecture_summary = {}
        
        # Extract predictions from each architecture
        for timeframe in timeframes:
            arch_predictions = []
            
            for arch_name, result in results.items():
                if 'error' not in result and 'predictions' in result:
                    pred = result['predictions'].get(timeframe)
                    if pred and pred.get('price'):
                        arch_predictions.append({
                            'architecture': arch_name,
                            'price': pred['price'],
                            'confidence': pred.get('confidence', 50),
                            'method': pred.get('method', arch_name)
                        })
            
            if arch_predictions:
                # Meta-ensemble: Combine architecture predictions
                final_predictions[timeframe] = self._meta_ensemble(arch_predictions, timeframe)
        
        # Create architecture summary
        for arch_name, result in results.items():
            if 'error' not in result:
                architecture_summary[arch_name] = {
                    'status': 'success',
                    'predictions_count': len(result.get('predictions', {})),
                    'method': result.get('architecture', arch_name)
                }
            else:
                architecture_summary[arch_name] = {
                    'status': 'failed',
                    'error': result['error']
                }
        
        return {
            'symbol': symbol,
            'final_predictions': final_predictions,
            'architecture_results': results,
            'architecture_summary': architecture_summary,
            'meta_method': 'multi_architecture_ensemble'
        }
    
    def _meta_ensemble(self, predictions: List[Dict], timeframe: str) -> Dict[str, Any]:
        """Meta-ensemble combining different architecture predictions"""
        
        # Architecture weights based on theoretical strengths
        arch_weights = {
            'ensemble': 0.30,      # Good overall performance
            'hierarchical': 0.25,  # Good for complex decisions
            'round_robin': 0.20,   # Good for sequential processing
            'agent_driven': 0.25   # Good for adaptive decisions
        }
        
        # Adjust weights based on timeframe
        if timeframe == 'short_term':
            arch_weights['ensemble'] = 0.35
            arch_weights['agent_driven'] = 0.30
        elif timeframe == 'long_term':
            arch_weights['hierarchical'] = 0.35
            arch_weights['ensemble'] = 0.25
        
        # Calculate weighted prediction
        total_weighted_price = 0
        total_weight = 0
        confidence_scores = []
        
        for pred in predictions:
            arch_name = pred['architecture']
            weight = arch_weights.get(arch_name, 0.2)
            confidence_weight = pred['confidence'] / 100
            final_weight = weight * confidence_weight
            
            total_weighted_price += pred['price'] * final_weight
            total_weight += final_weight
            confidence_scores.append(pred['confidence'])
        
        if total_weight > 0:
            final_price = total_weighted_price / total_weight
            # Meta-confidence based on architecture agreement
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            architecture_bonus = min(20, len(predictions) * 5)
            meta_confidence = min(95, avg_confidence + architecture_bonus)
            
            return {
                'price': round(final_price, 2),
                'confidence': round(meta_confidence, 1),
                'architectures_used': len(predictions),
                'method': f'meta_ensemble_{timeframe}',
                'architecture_breakdown': [
                    {
                        'arch': p['architecture'],
                        'price': p['price'],
                        'confidence': p['confidence']
                    } for p in predictions
                ]
            }
        
        return {'error': 'No valid predictions to combine'}
    
    def get_architecture_comparison(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """Compare performance of different architectures"""
        
        results = self.predict_all_architectures(symbol, data)
        
        comparison = {
            'symbol': symbol,
            'architectures_compared': len(self.architectures),
            'timeframe_analysis': {},
            'architecture_rankings': {}
        }
        
        # Analyze each timeframe
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            timeframe_data = []
            
            for arch_name, result in results['architecture_results'].items():
                if 'predictions' in result and timeframe in result['predictions']:
                    pred = result['predictions'][timeframe]
                    timeframe_data.append({
                        'architecture': arch_name,
                        'price': pred.get('price'),
                        'confidence': pred.get('confidence'),
                        'method': pred.get('method')
                    })
            
            if timeframe_data:
                # Calculate price variance
                prices = [d['price'] for d in timeframe_data if d['price']]
                if len(prices) > 1:
                    price_variance = max(prices) - min(prices)
                    avg_price = sum(prices) / len(prices)
                    variance_percent = (price_variance / avg_price) * 100
                else:
                    variance_percent = 0
                
                comparison['timeframe_analysis'][timeframe] = {
                    'predictions': timeframe_data,
                    'price_variance_percent': round(variance_percent, 2),
                    'avg_confidence': round(sum(d['confidence'] for d in timeframe_data) / len(timeframe_data), 1)
                }
        
        return comparison