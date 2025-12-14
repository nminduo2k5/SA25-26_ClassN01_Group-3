"""
Multi-Agent Architecture Package
Implements 4 different architectures for 6 AI agents
"""

from .ensemble_architecture import EnsembleArchitecture
from .hierarchical_architecture import HierarchicalArchitecture
from .round_robin_architecture import RoundRobinArchitecture
from .agent_driven_architecture import AgentDrivenArchitecture
from .base_agent import BaseAgent
from .concrete_agents import TechnicalAgent, LSTMAgent, InvestmentAgent, RiskAgent, NewsAgent, MarketAgent

__all__ = [
    'EnsembleArchitecture',
    'HierarchicalArchitecture', 
    'RoundRobinArchitecture',
    'AgentDrivenArchitecture',
    'BaseAgent',
    'TechnicalAgent',
    'LSTMAgent',
    'InvestmentAgent',
    'RiskAgent',
    'NewsAgent',
    'MarketAgent'
]