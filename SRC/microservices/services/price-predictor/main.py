from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
import requests
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf

app = FastAPI(title="Price Predictor Service", version="1.0.0")

# Redis connection
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

class PredictionRequest(BaseModel):
    symbol: str
    days: int = 30
    risk_tolerance: int = 50
    time_horizon: str = "medium"
    investment_amount: int = 10000000

class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    change_percent: float
    confidence: float
    method_used: str
    predictions: Dict[str, Any]
    technical_indicators: Dict[str, Any]

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "price-predictor",
        "redis_connected": redis_client.ping(),
        "models_available": ["Technical Analysis", "ML Ensemble"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """Price prediction using technical analysis"""
    try:
        # Check cache
        cache_key = f"prediction:{request.symbol}:{request.days}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            return PredictionResponse(**json.loads(cached_result))
        
        # Get stock data
        predictor = SimplePricePredictor()
        result = await predictor.predict(request.symbol, request.days)
        
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Cache result
        redis_client.setex(cache_key, 300, json.dumps(result))
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/predict/{symbol}")
async def predict_symbol(symbol: str, days: int = 30):
    """Quick prediction for a symbol"""
    request = PredictionRequest(symbol=symbol, days=days)
    return await predict_price(request)

@app.get("/technical/{symbol}")
async def technical_analysis(symbol: str):
    """Technical analysis indicators"""
    try:
        predictor = SimplePricePredictor()
        result = await predictor.get_technical_analysis(symbol)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Technical analysis error: {str(e)}")

class SimplePricePredictor:
    """Simplified price predictor for microservice"""
    
    async def predict(self, symbol: str, days: int):
        """Predict stock price"""
        try:
            # Get stock data
            data = await self._get_stock_data(symbol)
            if data.get('error'):
                return data
            
            current_price = data['current_price']
            
            # Calculate technical indicators
            tech_indicators = self._calculate_indicators(data['price_history'])
            
            # Generate prediction
            predicted_price = self._predict_price(current_price, tech_indicators, days)
            change_percent = ((predicted_price - current_price) / current_price) * 100
            
            # Generate timeframe predictions
            predictions = self._generate_predictions(current_price, tech_indicators)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'change_percent': round(change_percent, 2),
                'confidence': self._calculate_confidence(tech_indicators),
                'method_used': 'Technical Analysis',
                'predictions': predictions,
                'technical_indicators': tech_indicators
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    async def _get_stock_data(self, symbol: str):
        """Get stock data"""
        try:
            # Try VN stock first
            if self._is_vn_stock(symbol):
                ticker = yf.Ticker(f"{symbol}.VN")
                hist = ticker.history(period="1y")
                if hist.empty:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1y")
            else:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
            
            if hist.empty:
                return {"error": f"No data found for {symbol}"}
            
            return {
                'current_price': float(hist['Close'].iloc[-1]),
                'price_history': hist
            }
            
        except Exception as e:
            return {"error": f"Data retrieval failed: {str(e)}"}
    
    def _is_vn_stock(self, symbol: str):
        """Check if Vietnamese stock"""
        vn_stocks = ['VCB', 'BID', 'CTG', 'TCB', 'ACB', 'MBB', 'VPB', 'VIC', 'VHM', 'VRE', 
                    'MSN', 'MWG', 'VNM', 'SAB', 'PNJ', 'HPG', 'HSG', 'GAS', 'PLX', 'FPT']
        return symbol.upper() in vn_stocks
    
    def _calculate_indicators(self, price_data):
        """Calculate technical indicators"""
        try:
            df = price_data.copy()
            df.columns = [col.lower() for col in df.columns]
            
            indicators = {}
            
            # Moving averages
            indicators['sma_20'] = df['close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = df['close'].rolling(50).mean().iloc[-1]
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            indicators['macd'] = (ema_12 - ema_26).iloc[-1]
            indicators['macd_signal'] = (ema_12 - ema_26).ewm(span=9).mean().iloc[-1]
            
            # Volatility
            returns = df['close'].pct_change().dropna()
            indicators['volatility'] = returns.std() * np.sqrt(252) * 100
            
            return {k: round(float(v), 4) if not np.isnan(float(v)) else 0 for k, v in indicators.items()}
            
        except Exception as e:
            return {"error": f"Indicator calculation failed: {str(e)}"}
    
    def _predict_price(self, current_price: float, indicators: dict, days: int):
        """Predict price using technical analysis"""
        try:
            change_factor = 0
            
            # RSI influence
            rsi = indicators.get('rsi', 50)
            if rsi > 70:
                change_factor -= 0.02
            elif rsi < 30:
                change_factor += 0.02
            
            # MACD influence
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd > macd_signal:
                change_factor += 0.01
            else:
                change_factor -= 0.01
            
            # Moving average influence
            sma_20 = indicators.get('sma_20', current_price)
            sma_50 = indicators.get('sma_50', current_price)
            
            if current_price > sma_20 > sma_50:
                change_factor += 0.015
            elif current_price < sma_20 < sma_50:
                change_factor -= 0.015
            
            # Time factor
            time_factor = np.sqrt(days / 30)
            
            # Apply volatility
            volatility = indicators.get('volatility', 20) / 100
            random_factor = np.random.uniform(-volatility/2, volatility/2)
            
            total_change = (change_factor + random_factor) * time_factor
            total_change = max(-0.3, min(0.3, total_change))
            
            return round(current_price * (1 + total_change), 2)
            
        except Exception as e:
            return current_price
    
    def _generate_predictions(self, current_price: float, indicators: dict):
        """Generate predictions for different timeframes"""
        try:
            predictions = {}
            
            timeframes = {
                'short_term': [1, 3, 7],
                'medium_term': [14, 30, 60],
                'long_term': [90, 180, 365]
            }
            
            for timeframe, days_list in timeframes.items():
                predictions[timeframe] = {}
                for days in days_list:
                    predicted_price = self._predict_price(current_price, indicators, days)
                    change_percent = ((predicted_price - current_price) / current_price) * 100
                    
                    predictions[timeframe][f"{days}_days"] = {
                        "price": predicted_price,
                        "change_percent": round(change_percent, 2),
                        "change_amount": round(predicted_price - current_price, 2)
                    }
            
            return predictions
            
        except Exception as e:
            return {}
    
    def _calculate_confidence(self, indicators: dict):
        """Calculate prediction confidence"""
        try:
            base_confidence = 60
            
            # RSI confidence
            rsi = indicators.get('rsi', 50)
            if 30 <= rsi <= 70:
                base_confidence += 10
            
            # Volatility adjustment
            volatility = indicators.get('volatility', 20)
            if volatility < 20:
                base_confidence += 10
            elif volatility > 40:
                base_confidence -= 15
            
            return max(20, min(90, base_confidence))
            
        except Exception as e:
            return 50
    
    async def get_technical_analysis(self, symbol: str):
        """Get technical analysis"""
        try:
            data = await self._get_stock_data(symbol)
            if data.get('error'):
                return data
            
            indicators = self._calculate_indicators(data['price_history'])
            
            return {
                "symbol": symbol,
                "current_price": data['current_price'],
                "technical_indicators": indicators
            }
            
        except Exception as e:
            return {"error": f"Technical analysis failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)