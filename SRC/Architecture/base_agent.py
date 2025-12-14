"""
Base Agent Interface for Agent-Driven Architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def assess_confidence(self, symbol: str, data: Dict[str, Any]) -> float:
        """Agent tự đánh giá khả năng của mình"""
        pass

    @abstractmethod
    def contribute(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Agent thực hiện nhiệm vụ"""
        pass