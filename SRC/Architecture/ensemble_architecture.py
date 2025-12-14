"""
Ensemble Architecture - Combines all 6 agents predictions using weighted voting
"""

import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class EnsembleArchitecture:
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.weights = {
            'price_predictor': 0.25,
            'lstm_predictor': 0.25,
            'investment_expert': 0.20,
            'risk_expert': 0.15,
            'ticker_news': 0.10,
            'market_news': 0.05
        }
    
    def predict_price(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """Ensemble prediction combining all agents"""
        predictions = {}
        
        # Collect predictions from all agents
        for agent_name, agent in self.agents.items():
            try:
                if hasattr(agent, 'predict_price'):
                    pred = agent.predict_price(symbol, data)
                    predictions[agent_name] = pred
            except Exception as e:
                logger.warning(f"Agent {agent_name} failed: {e}")
        
        # Combine predictions using weighted average
        return self._combine_predictions(predictions, symbol)
    
    def _combine_predictions(self, predictions: Dict, symbol: str) -> Dict[str, Any]:
        """Weighted ensemble combination"""
        timeframes = ['short_term', 'medium_term', 'long_term']
        final_predictions = {}
        
        for timeframe in timeframes:
            weighted_prices = []
            total_weight = 0
            
            for agent_name, pred in predictions.items():
                if timeframe in pred and pred[timeframe].get('price'):
                    weight = self.weights.get(agent_name, 0.1)
                    weighted_prices.append(pred[timeframe]['price'] * weight)
                    total_weight += weight
            
            if weighted_prices and total_weight > 0:
                final_price = sum(weighted_prices) / total_weight
                confidence = min(85, len(weighted_prices) * 15)
                
                final_predictions[timeframe] = {
                    'price': round(final_price, 2),
                    'confidence': confidence,
                    'method': 'ensemble_weighted'
                }
        
        return {
            'architecture': 'ensemble',
            'predictions': final_predictions,
            'agents_used': list(predictions.keys())
        }