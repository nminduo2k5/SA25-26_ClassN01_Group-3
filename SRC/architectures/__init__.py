"""
Architecture Package for Duong AI Trading Pro
3 thiết kế kiến trúc khác nhau cho dự đoán giá cổ phiếu
"""

from .design_1_hierarchical import HierarchicalPricePredictionSystem
from .design_2_round_robin import RoundRobinPricePredictionSystem
from .design_3_ensemble_voting import EnsembleVotingPricePredictionSystem
from .architecture_manager import ArchitectureManager

__all__ = [
    'HierarchicalPricePredictionSystem',
    'RoundRobinPricePredictionSystem', 
    'EnsembleVotingPricePredictionSystem',
    'ArchitectureManager'
]