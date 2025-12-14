"""
Agent-Driven Architecture - Agents autonomously decide when and how to contribute
"""

from typing import List, Dict, Any
import logging
from .base_agent import BaseAgent
from .concrete_agents import TechnicalAgent, LSTMAgent, InvestmentAgent, RiskAgent, NewsAgent, MarketAgent

logger = logging.getLogger(__name__)

class AgentDrivenArchitecture:
    def __init__(self, agents: Dict[str, Any] = None):
        # Create standardized agents using BaseAgent interface
        self.agents: List[BaseAgent] = [
            TechnicalAgent('price_predictor'),
            LSTMAgent('lstm_predictor'),
            InvestmentAgent('investment_expert'),
            RiskAgent('risk_expert'),
            NewsAgent('ticker_news'),
            MarketAgent('market_news')
        ]
        self.confidence_threshold = 0.6
    
    def predict_price(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """Agent-driven autonomous prediction"""
        
        # Phase 1: Agent self-assessment (Polymorphism)
        active_agents = []
        for agent in self.agents:
            confidence = agent.assess_confidence(symbol, data)
            if confidence > self.confidence_threshold:
                active_agents.append((agent, confidence))
        
        # Phase 2: Collect contributions (No if/else check names)
        contributions = {}
        for agent, confidence in active_agents:
            result = agent.contribute(symbol, data)
            contributions[agent.name] = {
                'prediction': result,
                'confidence': confidence
            }
        
        # Phase 3: Peer review process
        final_contributions = self._peer_review_process(contributions)
        
        # Phase 4: Autonomous consensus
        return self._autonomous_consensus(final_contributions, symbol)
    
    def _peer_review_process(self, contributions: Dict) -> Dict[str, Any]:
        """Agents review each other's contributions"""
        
        reviewed_contributions = {}
        
        for agent_name, contribution in contributions.items():
            peer_adjustments = []
            
            for peer_name, peer_contribution in contributions.items():
                if peer_name != agent_name:
                    agreement = self._calculate_peer_agreement(contribution, peer_contribution)
                    peer_adjustments.append(agreement)
            
            if peer_adjustments:
                avg_agreement = sum(peer_adjustments) / len(peer_adjustments)
                adjusted_confidence = contribution['confidence'] * (0.7 + 0.3 * avg_agreement)
            else:
                adjusted_confidence = contribution['confidence']
            
            reviewed_contributions[agent_name] = {
                **contribution,
                'peer_reviewed_confidence': adjusted_confidence
            }
        
        return reviewed_contributions
    
    def _calculate_peer_agreement(self, contribution1: Dict, contribution2: Dict) -> float:
        """Calculate agreement between two agent contributions"""
        
        agreement_scores = []
        
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            pred1 = contribution1.get('prediction', {}).get(timeframe, {})
            pred2 = contribution2.get('prediction', {}).get(timeframe, {})
            
            if pred1.get('price') and pred2.get('price'):
                price_diff = abs(pred1['price'] - pred2['price']) / max(pred1['price'], pred2['price'])
                agreement = max(0, 1 - price_diff)
                agreement_scores.append(agreement)
        
        return sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.5
    
    def _autonomous_consensus(self, contributions: Dict, symbol: str) -> Dict[str, Any]:
        """Agents reach autonomous consensus"""
        
        timeframes = ['short_term', 'medium_term', 'long_term']
        final_predictions = {}
        
        for timeframe in timeframes:
            agent_predictions = []
            
            for agent_name, contribution in contributions.items():
                pred = contribution.get('prediction', {}).get(timeframe, {})
                if pred.get('price'):
                    agent_predictions.append({
                        'agent': agent_name,
                        'price': pred['price'],
                        'confidence': contribution['peer_reviewed_confidence']
                    })
            
            if agent_predictions:
                total_weighted_price = 0
                total_weight = 0
                
                for pred in agent_predictions:
                    weight = pred['confidence'] ** 2
                    total_weighted_price += pred['price'] * weight
                    total_weight += weight
                
                if total_weight > 0:
                    consensus_price = total_weighted_price / total_weight
                    consensus_confidence = min(95, len(agent_predictions) * 15 + 20)
                    
                    final_predictions[timeframe] = {
                        'price': round(consensus_price, 2),
                        'confidence': consensus_confidence,
                        'method': 'agent_driven_consensus'
                    }
        
        return {
            'architecture': 'agent_driven',
            'predictions': final_predictions,
            'participating_agents': list(contributions.keys()),
            'consensus_strength': len(contributions)
        }