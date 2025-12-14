"""
Hierarchical Architecture - Master agent coordinates specialized agents
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class HierarchicalArchitecture:
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.hierarchy = {
            'master': 'investment_expert',
            'technical': ['price_predictor', 'lstm_predictor'],
            'fundamental': ['risk_expert'],
            'sentiment': ['ticker_news', 'market_news']
        }
    
    def predict_price(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """Hierarchical prediction with master coordination"""
        
        # Step 1: Technical analysis layer
        technical_results = self._run_technical_layer(symbol, data)
        
        # Step 2: Fundamental analysis layer  
        fundamental_results = self._run_fundamental_layer(symbol, data)
        
        # Step 3: Sentiment analysis layer
        sentiment_results = self._run_sentiment_layer(symbol, data)
        
        # Step 4: Master agent coordination
        return self._master_coordination(technical_results, fundamental_results, sentiment_results, symbol)
    
    def _run_technical_layer(self, symbol: str, data: Dict) -> Dict:
        """Run technical analysis agents"""
        results = {}
        for agent_name in self.hierarchy['technical']:
            if agent_name in self.agents:
                try:
                    agent = self.agents[agent_name]
                    if hasattr(agent, 'predict_price'):
                        results[agent_name] = agent.predict_price(symbol, data)
                except Exception as e:
                    logger.warning(f"Technical agent {agent_name} failed: {e}")
        return results
    
    def _run_fundamental_layer(self, symbol: str, data: Dict) -> Dict:
        """Run fundamental analysis agents"""
        results = {}
        for agent_name in self.hierarchy['fundamental']:
            if agent_name in self.agents:
                try:
                    agent = self.agents[agent_name]
                    if hasattr(agent, 'assess_risk'):
                        results[agent_name] = agent.assess_risk(symbol, data)
                except Exception as e:
                    logger.warning(f"Fundamental agent {agent_name} failed: {e}")
        return results
    
    def _run_sentiment_layer(self, symbol: str, data: Dict) -> Dict:
        """Run sentiment analysis agents"""
        results = {}
        for agent_name in self.hierarchy['sentiment']:
            if agent_name in self.agents:
                try:
                    agent = self.agents[agent_name]
                    if hasattr(agent, 'get_news_sentiment'):
                        results[agent_name] = agent.get_news_sentiment(symbol)
                except Exception as e:
                    logger.warning(f"Sentiment agent {agent_name} failed: {e}")
        return results
    
    def _master_coordination(self, technical: Dict, fundamental: Dict, sentiment: Dict, symbol: str) -> Dict[str, Any]:
        """Master agent coordinates all layer results"""
        master_agent = self.agents.get(self.hierarchy['master'])
        
        # Combine layer results
        combined_data = {
            'technical_analysis': technical,
            'fundamental_analysis': fundamental,
            'sentiment_analysis': sentiment
        }
        
        # Master agent makes final decision
        timeframes = ['short_term', 'medium_term', 'long_term']
        final_predictions = {}
        
        for timeframe in timeframes:
            # Extract technical predictions
            tech_prices = []
            for agent_result in technical.values():
                if timeframe in agent_result and agent_result[timeframe].get('price'):
                    tech_prices.append(agent_result[timeframe]['price'])
            
            # Calculate weighted prediction based on hierarchy
            if tech_prices:
                base_price = np.mean(tech_prices)
                
                # Adjust based on risk (fundamental)
                risk_adjustment = 1.0
                for risk_result in fundamental.values():
                    if 'risk_level' in risk_result:
                        risk_level = risk_result['risk_level'].lower()
                        if risk_level == 'high':
                            risk_adjustment *= 0.95
                        elif risk_level == 'low':
                            risk_adjustment *= 1.05
                
                # Adjust based on sentiment
                sentiment_adjustment = 1.0
                for sent_result in sentiment.values():
                    if 'sentiment_score' in sent_result:
                        score = sent_result['sentiment_score']
                        sentiment_adjustment *= (1 + score * 0.02)
                
                final_price = base_price * risk_adjustment * sentiment_adjustment
                confidence = min(90, len(tech_prices) * 20 + 10)
                
                final_predictions[timeframe] = {
                    'price': round(final_price, 2),
                    'confidence': confidence,
                    'method': 'hierarchical_master'
                }
        
        return {
            'architecture': 'hierarchical',
            'predictions': final_predictions,
            'layer_results': combined_data
        }