from abc import ABC, abstractmethod
from datetime import datetime
from typing import List
from business_logic.models import MarketDataset

# Simulate database (key: symbol, value: list of MarketDataset sorted by date)
market_db = {
    "VNM": [  # Dummy data for Vietnam stock VNM
        MarketDataset(datetime(2025, 12, 1), 100.0, 105.0, 99.0, 102.0, 100000),
        MarketDataset(datetime(2025, 12, 2), 102.0, 106.0, 101.0, 104.0, 120000),
        MarketDataset(datetime(2025, 12, 3), 104.0, 108.0, 103.0, 106.0, 110000),
        MarketDataset(datetime(2025, 12, 4), 106.0, 109.0, 105.0, 107.5, 95000),
        MarketDataset(datetime(2025, 12, 5), 107.5, 110.0, 106.0, 108.0, 105000),
    ],
    "VCB": [  # Dummy data for VietComBank
        MarketDataset(datetime(2025, 12, 1), 85.0, 87.0, 84.0, 86.5, 200000),
        MarketDataset(datetime(2025, 12, 2), 86.5, 88.0, 85.5, 87.0, 180000),
        MarketDataset(datetime(2025, 12, 3), 87.0, 89.0, 86.0, 88.5, 220000),
        MarketDataset(datetime(2025, 12, 4), 88.5, 90.0, 87.5, 89.0, 190000),
        MarketDataset(datetime(2025, 12, 5), 89.0, 91.0, 88.0, 90.5, 210000),
    ],
    "HPG": [  # Dummy data for Hoa Phat Group
        MarketDataset(datetime(2025, 12, 1), 25.0, 26.0, 24.5, 25.8, 300000),
        MarketDataset(datetime(2025, 12, 2), 25.8, 27.0, 25.0, 26.2, 280000),
        MarketDataset(datetime(2025, 12, 3), 26.2, 27.5, 25.8, 27.0, 320000),
        MarketDataset(datetime(2025, 12, 4), 27.0, 28.0, 26.5, 27.5, 290000),
        MarketDataset(datetime(2025, 12, 5), 27.5, 28.5, 27.0, 28.0, 310000),
    ]
}

class MarketDataRepository(ABC):
    @abstractmethod
    def find_by_symbol_and_date_range(
        self, symbol: str, from_date: datetime, to_date: datetime
    ) -> List[MarketDataset]:
        pass

class InMemoryMarketDataRepository(MarketDataRepository):
    """Layer 3: Handles data access operations on the simulated store."""

    def find_by_symbol_and_date_range(
        self, symbol: str, from_date: datetime, to_date: datetime
    ) -> List[MarketDataset]:
        if symbol not in market_db:
            return []
        data = market_db[symbol]
        return [d for d in data if from_date <= d.date <= to_date]

    # Add CRUD if needed, e.g., for updating data
    def save(self, symbol: str, dataset: MarketDataset):
        if symbol not in market_db:
            market_db[symbol] = []
        market_db[symbol].append(dataset)
        market_db[symbol].sort(key=lambda d: d.date)
    
    def get_latest_price(self, symbol: str) -> float:
        """Get the latest closing price for a symbol"""
        if symbol not in market_db or not market_db[symbol]:
            return 0.0
        return market_db[symbol][-1].close
    
    def get_all_symbols(self) -> List[str]:
        """Get all available symbols"""
        return list(market_db.keys())