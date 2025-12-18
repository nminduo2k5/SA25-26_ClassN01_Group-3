from typing import Optional, Dict, List
from datetime import datetime, timedelta
import random  # For dummy agent predictions
import numpy as np

from persistence.market_data_repository import InMemoryMarketDataRepository
from business_logic.models import PredictionResult, MarketDataset

class PricePredictor:
    def __init__(
        self,
        name: str = "Price Predictor Agent with LSTM",
        vn_api: Optional[object] = None,  # Placeholder for real APIs
        stock_info: Optional[object] = None,
        ai_agent: Optional[object] = None,
        crewai_collector: Optional[object] = None
    ):
        self.name = name
        self.vn_api = vn_api
        self.stock_info = stock_info
        self.ai_agent = ai_agent
        self.crewai_collector = crewai_collector
        self.repo = InMemoryMarketDataRepository()

        self.prediction_periods: Dict[str, List[int]] = {
            "short-term": [1, 3, 7],
            "medium-term": [14, 30, 60],
            "long-term": [90, 180, 365],
        }

    def get_unified_prediction(self, symbol: str, horizon: str, arch: str = "ensemble") -> PredictionResult:
        """Unified prediction API. Validates input, fetches data, orchestrates agents."""
        if horizon not in self.prediction_periods:
            raise ValueError("Invalid horizon: Must be short-term, medium-term, or long-term.")

        # Business Rule: Fetch last 30 days data for prediction
        to_date = datetime.now()
        from_date = to_date - timedelta(days=30)
        data = self.repo.find_by_symbol_and_date_range(symbol, from_date, to_date)
        if not data:
            raise ValueError(f"No data found for symbol {symbol} in range.")

        # Simulate agents (Technical, Fundamental, Sentiment, LSTM)
        agent_predictions = self._orchestrate_agents(data, self.prediction_periods[horizon], arch)

        # Aggregate (dummy average for simplicity)
        unified = {f"{d}d": sum(p[d] for p in agent_predictions) / len(agent_predictions) for d in self.prediction_periods[horizon]}

        return PredictionResult(symbol, horizon, unified)

    def _orchestrate_agents(self, data: List[MarketDataset], periods: List[int], arch: str) -> List[Dict[int, float]]:
        """Orchestrate based on architecture."""
        agents = ["technical", "fundamental", "sentiment", "lstm"]
        predictions = []

        if arch == "hierarchical":
            # Master agent refines others
            base_preds = [self._dummy_agent_predict(a, data, periods) for a in agents[:-1]]
            master_pred = self._dummy_agent_predict("lstm", data, periods)  # LSTM as master
            predictions = [master_pred]  # Simplified

        elif arch == "round_robin":
            # Sequential
            current_data = data
            for agent in agents:
                pred = self._dummy_agent_predict(agent, current_data, periods)
                predictions.append(pred)
                # Simulate refinement: update data (dummy)

        elif arch == "ensemble":
            # Parallel voting
            predictions = [self._dummy_agent_predict(a, data, periods) for a in agents]

        elif arch == "agent_driven":
            # Agent-driven architecture - agents decide their own participation
            for agent in agents:
                # Each agent decides whether to participate based on market conditions
                if self._should_agent_participate(agent, data):
                    pred = self._dummy_agent_predict(agent, data, periods)
                    predictions.append(pred)

        else:
            raise ValueError("Invalid architecture: hierarchical, round_robin, ensemble, or agent_driven.")

        return predictions

    def _should_agent_participate(self, agent: str, data: List[MarketDataset]) -> bool:
        """Determine if an agent should participate based on market conditions"""
        if not data:
            return False
        
        latest_price = data[-1].close
        volatility = self._calculate_volatility(data)
        
        # Agent-specific participation logic
        if agent == "technical":
            return True  # Technical analysis always participates
        elif agent == "fundamental":
            return len(data) >= 5  # Needs sufficient data
        elif agent == "sentiment":
            return volatility > 0.02  # Participates in volatile markets
        elif agent == "lstm":
            return len(data) >= 10  # Needs more data for neural network
        
        return True

    def _calculate_volatility(self, data: List[MarketDataset]) -> float:
        """Calculate price volatility"""
        if len(data) < 2:
            return 0.0
        
        prices = [d.close for d in data]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        return np.std(returns) if returns else 0.0

    def _dummy_agent_predict(self, agent: str, data: List[MarketDataset], periods: List[int]) -> Dict[int, float]:
        """Dummy prediction logic for agents."""
        last_close = data[-1].close if data else 100.0
        
        # Agent-specific prediction logic
        if agent == "technical":
            # Technical analysis - trend following
            trend = 1.02 if len(data) > 1 and data[-1].close > data[-2].close else 0.98
            return {p: last_close * (trend ** (p/7)) for p in periods}
        
        elif agent == "fundamental":
            # Fundamental analysis - conservative growth
            growth_rate = 1.01
            return {p: last_close * (growth_rate ** (p/30)) for p in periods}
        
        elif agent == "sentiment":
            # Sentiment analysis - more volatile
            sentiment_factor = random.uniform(0.95, 1.05)
            return {p: last_close * sentiment_factor * (1 + random.uniform(-0.02, 0.02) * p) for p in periods}
        
        elif agent == "lstm":
            # LSTM - sophisticated pattern recognition
            base_trend = 1.015 if self._detect_uptrend(data) else 0.985
            noise = random.uniform(0.99, 1.01)
            return {p: last_close * (base_trend ** (p/10)) * noise for p in periods}
        
        else:
            # Default random variation
            return {p: last_close * (1 + random.uniform(-0.05, 0.05) * p/30) for p in periods}

    def _detect_uptrend(self, data: List[MarketDataset]) -> bool:
        """Simple uptrend detection for LSTM agent"""
        if len(data) < 3:
            return True
        
        recent_prices = [d.close for d in data[-3:]]
        return recent_prices[-1] > recent_prices[0]

    def get_market_overview(self) -> Dict:
        """Get market overview with available symbols"""
        symbols = self.repo.get_all_symbols()
        overview = {
            "total_symbols": len(symbols),
            "symbols": symbols,
            "market_status": "Open" if datetime.now().weekday() < 5 else "Closed",
            "last_updated": datetime.now().isoformat()
        }
        
        # Add latest prices for each symbol
        latest_prices = {}
        for symbol in symbols:
            latest_prices[symbol] = self.repo.get_latest_price(symbol)
        
        overview["latest_prices"] = latest_prices
        return overview

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get detailed information for a specific symbol"""
        if symbol not in self.repo.get_all_symbols():
            raise ValueError(f"Symbol {symbol} not found")
        
        # Get recent data
        to_date = datetime.now()
        from_date = to_date - timedelta(days=7)
        recent_data = self.repo.find_by_symbol_and_date_range(symbol, from_date, to_date)
        
        if not recent_data:
            raise ValueError(f"No recent data found for {symbol}")
        
        # Calculate basic statistics
        prices = [d.close for d in recent_data]
        volumes = [d.volume for d in recent_data]
        
        return {
            "symbol": symbol,
            "current_price": prices[-1],
            "price_change": prices[-1] - prices[0] if len(prices) > 1 else 0,
            "price_change_percent": ((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) > 1 and prices[0] > 0 else 0,
            "avg_volume": sum(volumes) / len(volumes),
            "high_7d": max(d.high for d in recent_data),
            "low_7d": min(d.low for d in recent_data),
            "data_points": len(recent_data),
            "last_updated": recent_data[-1].date.isoformat()
        }