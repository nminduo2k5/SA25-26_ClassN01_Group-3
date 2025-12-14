"""
Round Robin Architecture - Sequential agent execution with result passing
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class RoundRobinArchitecture:
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.execution_order = [
            'price_predictor',
            'lstm_predictor', 
            'investment_expert',
            'risk_expert',
            'ticker_news',
            'market_news'
        ]
    
    def predict_price(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """Round robin execution with result accumulation"""
        
        accumulated_results = {'symbol': symbol, 'input_data': data}
        execution_log = []
        
        # Execute agents in round robin order
        for agent_name in self.execution_order:
            if agent_name in self.agents:
                try:
                    agent = self.agents[agent_name]
                    
                    # Pass accumulated results to next agent
                    result = self._execute_agent(agent, agent_name, accumulated_results)
                    
                    # Accumulate results
                    accumulated_results[f'{agent_name}_result'] = result
                    execution_log.append({
                        'agent': agent_name,
                        'status': 'success',
                        'result_keys': list(result.keys()) if result else []
                    })
                    
                except Exception as e:
                    logger.warning(f"Round robin agent {agent_name} failed: {e}")
                    execution_log.append({
                        'agent': agent_name,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        # Generate final predictions from accumulated results
        return self._generate_final_predictions(accumulated_results, execution_log)
    
    def _execute_agent(self, agent: Any, agent_name: str, context: Dict) -> Dict:
        """Execute individual agent with context"""
        
        if agent_name in ['price_predictor', 'lstm_predictor']:
            if hasattr(agent, 'predict_price'):
                return agent.predict_price(context['symbol'], context)
        
        elif agent_name == 'investment_expert':
            if hasattr(agent, 'analyze_investment'):
                return agent.analyze_investment(context['symbol'], context)
        
        elif agent_name == 'risk_expert':
            if hasattr(agent, 'assess_risk'):
                return agent.assess_risk(context['symbol'], context)
        
        elif agent_name in ['ticker_news', 'market_news']:
            if hasattr(agent, 'get_news_sentiment'):
                return agent.get_news_sentiment(context['symbol'])
        
        return {}
    
    def _generate_final_predictions(self, results: Dict, log: List) -> Dict[str, Any]:
        """Generate final predictions from round robin results"""
        
        timeframes = ['short_term', 'medium_term', 'long_term']
        final_predictions = {}
        
        # Extract price predictions from each agent
        price_predictions = {}
        for agent_name in self.execution_order:
            result_key = f'{agent_name}_result'
            if result_key in results:
                agent_result = results[result_key]
                if isinstance(agent_result, dict):
                    for timeframe in timeframes:
                        if timeframe in agent_result and agent_result[timeframe].get('price'):
                            if timeframe not in price_predictions:
                                price_predictions[timeframe] = []
                            price_predictions[timeframe].append({
                                'agent': agent_name,
                                'price': agent_result[timeframe]['price'],
                                'confidence': agent_result[timeframe].get('confidence', 50)
                            })
        
        # Calculate sequential weighted average (later agents have more weight)
        for timeframe in timeframes:
            if timeframe in price_predictions and price_predictions[timeframe]:
                predictions = price_predictions[timeframe]
                
                # Sequential weighting (later agents get higher weight)
                total_weighted_price = 0
                total_weight = 0
                
                for i, pred in enumerate(predictions):
                    weight = (i + 1) / len(predictions)  # Sequential weight
                    confidence_weight = pred['confidence'] / 100
                    final_weight = weight * confidence_weight
                    
                    total_weighted_price += pred['price'] * final_weight
                    total_weight += final_weight
                
                if total_weight > 0:
                    final_price = total_weighted_price / total_weight
                    avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
                    
                    final_predictions[timeframe] = {
                        'price': round(final_price, 2),
                        'confidence': round(avg_confidence, 1),
                        'method': 'round_robin_sequential'
                    }
        
        return {
            'architecture': 'round_robin',
            'predictions': final_predictions,
            'execution_log': log,
            'agents_executed': len([l for l in log if l['status'] == 'success'])
        }