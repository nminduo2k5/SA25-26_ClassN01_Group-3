"""
Concrete Agent Implementations
"""

from .base_agent import BaseAgent
from typing import Dict, Any
import random

class TechnicalAgent(BaseAgent):
    def assess_confidence(self, symbol: str, data: Dict[str, Any]) -> float:
        confidence = 0.3
        if data.get('price_history'):
            confidence += 0.4
        if data.get('volume_data'):
            confidence += 0.3
        return min(1.0, confidence)

    def contribute(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        base_price = data.get('current_price', 100000)
        return {
            'short_term': {'price': base_price * random.uniform(0.98, 1.02)},
            'medium_term': {'price': base_price * random.uniform(0.95, 1.08)},
            'long_term': {'price': base_price * random.uniform(0.90, 1.15)}
        }

class LSTMAgent(BaseAgent):
    def assess_confidence(self, symbol: str, data: Dict[str, Any]) -> float:
        confidence = 0.4
        if data.get('price_history') and len(data.get('price_history', [])) > 30:
            confidence += 0.4
        if data.get('technical_indicators'):
            confidence += 0.2
        return min(1.0, confidence)

    def contribute(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        base_price = data.get('current_price', 100000)
        return {
            'short_term': {'price': base_price * random.uniform(0.99, 1.01)},
            'medium_term': {'price': base_price * random.uniform(0.96, 1.06)},
            'long_term': {'price': base_price * random.uniform(0.92, 1.12)}
        }

class InvestmentAgent(BaseAgent):
    def assess_confidence(self, symbol: str, data: Dict[str, Any]) -> float:
        confidence = 0.5
        if data.get('financial_ratios'):
            confidence += 0.3
        if data.get('company_info'):
            confidence += 0.2
        return min(1.0, confidence)

    def contribute(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        base_price = data.get('current_price', 100000)
        return {
            'short_term': {'price': base_price * random.uniform(0.97, 1.03)},
            'medium_term': {'price': base_price * random.uniform(0.94, 1.10)},
            'long_term': {'price': base_price * random.uniform(0.88, 1.20)}
        }

class RiskAgent(BaseAgent):
    def assess_confidence(self, symbol: str, data: Dict[str, Any]) -> float:
        confidence = 0.6
        if data.get('volatility_data'):
            confidence += 0.2
        if data.get('market_conditions'):
            confidence += 0.2
        return min(1.0, confidence)

    def contribute(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        base_price = data.get('current_price', 100000)
        risk_factor = random.uniform(0.95, 1.05)
        return {
            'short_term': {'price': base_price * risk_factor},
            'medium_term': {'price': base_price * risk_factor * random.uniform(0.98, 1.02)},
            'long_term': {'price': base_price * risk_factor * random.uniform(0.96, 1.04)}
        }

class NewsAgent(BaseAgent):
    def assess_confidence(self, symbol: str, data: Dict[str, Any]) -> float:
        confidence = 0.2
        if data.get('news_available'):
            confidence += 0.5
        if data.get('sentiment_data'):
            confidence += 0.3
        return min(1.0, confidence)

    def contribute(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        base_price = data.get('current_price', 100000)
        sentiment_factor = random.uniform(0.98, 1.02)
        return {
            'short_term': {'price': base_price * sentiment_factor},
            'medium_term': {'price': base_price * sentiment_factor * random.uniform(0.99, 1.01)},
            'long_term': {'price': base_price * sentiment_factor * random.uniform(0.98, 1.02)}
        }

class MarketAgent(BaseAgent):
    def assess_confidence(self, symbol: str, data: Dict[str, Any]) -> float:
        confidence = 0.3
        if data.get('market_trends'):
            confidence += 0.4
        if data.get('economic_indicators'):
            confidence += 0.3
        return min(1.0, confidence)

    def contribute(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        base_price = data.get('current_price', 100000)
        market_factor = random.uniform(0.97, 1.03)
        return {
            'short_term': {'price': base_price * market_factor},
            'medium_term': {'price': base_price * market_factor * random.uniform(0.98, 1.02)},
            'long_term': {'price': base_price * market_factor * random.uniform(0.95, 1.05)}
        }