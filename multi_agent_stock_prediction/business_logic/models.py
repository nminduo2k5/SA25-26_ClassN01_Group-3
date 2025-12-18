from datetime import datetime
from typing import List, Dict

class MarketDataset:
    """Data model for market records."""
    def __init__(self, date: datetime, open: float, high: float, low: float, close: float, volume: int):
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def to_dict(self):
        return {
            "date": self.date.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        }

class PredictionResult:
    """Result model for unified predictions."""
    def __init__(self, symbol: str, horizon: str, predictions: Dict[str, float]):
        self.symbol = symbol
        self.horizon = horizon
        self.predictions = predictions  # e.g., {"1d": 100.0, "3d": 102.5}

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "horizon": self.horizon,
            "predictions": self.predictions
        }