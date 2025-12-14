"""
Single Architecture Runner - Execute individual architectures
"""

from typing import Dict, Any
import logging
from .ensemble_architecture import EnsembleArchitecture
from .hierarchical_architecture import HierarchicalArchitecture
from .round_robin_architecture import RoundRobinArchitecture
from .agent_driven_architecture import AgentDrivenArchitecture

logger = logging.getLogger(__name__)

class SingleArchitectureRunner:
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.architectures = {
            'ensemble': EnsembleArchitecture(agents),
            'hierarchical': HierarchicalArchitecture(agents),
            'round_robin': RoundRobinArchitecture(agents),
            'agent_driven': AgentDrivenArchitecture(agents)
        }
    
    def run_single_architecture(self, architecture_name: str, symbol: str, data: Dict) -> Dict[str, Any]:
        """Run single architecture prediction"""
        
        if architecture_name not in self.architectures:
            return {'error': f'Architecture {architecture_name} not found'}
        
        try:
            logger.info(f"Running {architecture_name} architecture for {symbol}")
            architecture = self.architectures[architecture_name]
            result = architecture.predict_price(symbol, data)
            
            # Add metadata
            result['selected_architecture'] = architecture_name
            result['single_run'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Single architecture {architecture_name} failed: {e}")
            return {'error': f'Architecture {architecture_name} failed: {str(e)}'}
    
    def get_architecture_info(self, architecture_name: str) -> Dict[str, Any]:
        """Get information about specific architecture"""
        
        info = {
            'ensemble': {
                'name': 'Ensemble Architecture',
                'description': 'Combines all 6 agents using weighted voting',
                'strengths': ['Balanced predictions', 'Reduces individual agent bias'],
                'best_for': 'General purpose analysis'
            },
            'hierarchical': {
                'name': 'Hierarchical Architecture', 
                'description': 'Master agent coordinates specialized layers',
                'strengths': ['Structured decision making', 'Clear responsibility'],
                'best_for': 'Complex multi-factor analysis'
            },
            'round_robin': {
                'name': 'Round Robin Architecture',
                'description': 'Sequential agent execution with result passing',
                'strengths': ['Progressive refinement', 'Context building'],
                'best_for': 'Step-by-step analysis'
            },
            'agent_driven': {
                'name': 'Agent-Driven Architecture',
                'description': 'Agents autonomously decide participation',
                'strengths': ['Adaptive behavior', 'Self-organizing'],
                'best_for': 'Dynamic market conditions'
            }
        }
        
        return info.get(architecture_name, {'name': 'Unknown', 'description': 'No information available'})