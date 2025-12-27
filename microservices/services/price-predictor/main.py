from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
import asyncio
import sys
import os
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    trend_analysis: Dict[str, Any]

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "price-predictor",
        "redis_connected": redis_client.ping(),
        "models_available": ["LSTM", "Technical Analysis", "ML Ensemble"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """Comprehensive price prediction using LSTM + Technical Analysis"""
    try:
        # Check cache first
        cache_key = f"prediction:{request.symbol}:{request.days}:{request.risk_tolerance}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            cached_data = json.loads(cached_result)
            return PredictionResponse(**cached_data)
        
        # Initialize price predictor
        predictor = EnhancedPricePredictor()
        
        # Get comprehensive prediction
        result = await predictor.predict_enhanced(
            symbol=request.symbol,
            days=request.days,
            risk_tolerance=request.risk_tolerance,
            time_horizon=request.time_horizon,
            investment_amount=request.investment_amount
        )
        
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Prepare response
        response_data = {
            "symbol": result['symbol'],
            "current_price": result['current_price'],
            "predicted_price": result.get('predicted_price', result['current_price']),
            "change_percent": result.get('change_percent', 0),
            "confidence": result.get('confidence', 50),
            "method_used": result.get('method_used', 'Technical Analysis'),
            "predictions": result.get('predictions', {}),
            "technical_indicators": result.get('technical_indicators', {}),
            "trend_analysis": result.get('trend_analysis', {})
        }
        
        # Cache the result for 5 minutes
        redis_client.setex(cache_key, 300, json.dumps(response_data))
        
        return PredictionResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/predict/{symbol}")
async def predict_symbol(symbol: str, days: int = 30):
    """Quick prediction for a symbol"""
    request = PredictionRequest(symbol=symbol, days=days)
    return await predict_price(request)

@app.get("/lstm/{symbol}")
async def lstm_prediction(symbol: str, days: int = 30):
    """LSTM-specific prediction"""
    try:
        predictor = EnhancedPricePredictor()
        
        # Try LSTM prediction
        if hasattr(predictor, 'lstm_predictor') and predictor.lstm_predictor:
            result = predictor.lstm_predictor.predict_with_ai_enhancement(symbol, days)
            
            if not result.get('error'):
                return {
                    "symbol": symbol,
                    "method": "LSTM Neural Network",
                    "predictions": result.get('predictions', {}),
                    "confidence": result.get('model_performance', {}).get('confidence', 0),
                    "model_info": result.get('model_performance', {})
                }
        
        raise HTTPException(status_code=503, detail="LSTM model not available")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LSTM prediction error: {str(e)}")

@app.get("/technical/{symbol}")
async def technical_analysis(symbol: str):
    """Technical analysis indicators"""
    try:
        predictor = EnhancedPricePredictor()
        result = await predictor.get_technical_analysis(symbol)
        
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Technical analysis error: {str(e)}")

@app.post("/llm-enhance")
async def llm_enhance_prediction(request: Dict[str, Any]):
    """Enhance prediction with LLM analysis"""
    try:
        # Call LLM Hub service
        llm_response = requests.post(
            "http://llm-hub:8010/generate",
            json={
                "prompt": f"Analyze stock prediction for {request.get('symbol', 'UNKNOWN')}: {json.dumps(request)}",
                "model": "gemini",
                "cache_key": f"prediction_enhance:{request.get('symbol', 'unknown')}"
            },
            timeout=10
        )
        
        if llm_response.status_code == 200:
            llm_data = llm_response.json()
            return {
                "enhanced": True,
                "llm_analysis": llm_data.get('response', ''),
                "model_used": llm_data.get('model_used', 'unknown')
            }
        else:
            return {
                "enhanced": False,
                "error": "LLM service unavailable"
            }
            
    except Exception as e:
        return {
            "enhanced": False,
            "error": f"LLM enhancement failed: {str(e)}"
        }

class EnhancedPricePredictor:
    """Simplified price predictor for microservice"""
    
    def __init__(self):
        self.name = "Enhanced Price Predictor Microservice"
        self.lstm_predictor = None
        
        # Try to initialize LSTM
        try:
            from agents.lstm_price_predictor import LSTMPricePredictor
            self.lstm_predictor = LSTMPricePredictor()
            print("✅ LSTM predictor initialized")
        except ImportError:
            print("⚠️ LSTM predictor not available")
    
    async def predict_enhanced(self, symbol: str, days: int, risk_tolerance: int, 
                             time_horizon: str, investment_amount: int):
        """Enhanced prediction combining multiple methods"""
        try:
            # Try LSTM first if available
            if self.lstm_predictor:
                try:
                    lstm_result = self.lstm_predictor.predict_with_ai_enhancement(symbol, days)
                    if not lstm_result.get('error') and lstm_result['model_performance']['confidence'] > 20:
                        return self._format_lstm_result(lstm_result, days)
                except Exception as e:
                    print(f"LSTM failed: {e}")
            
            # Fallback to technical analysis
            return await self._technical_prediction(symbol, days, risk_tolerance)
            
        except Exception as e:
            return {"error": f"Enhanced prediction failed: {str(e)}"}
    
    def _format_lstm_result(self, lstm_result: dict, days: int):
        """Format LSTM result for API response"""
        try:
            current_price = lstm_result['current_price']
            
            # Find appropriate prediction timeframe
            predictions = lstm_result.get('predictions', {})
            predicted_price = current_price
            
            if days <= 7:
                pred_data = predictions.get('short_term', {}).get('7_days', {})
            elif days <= 30:
                pred_data = predictions.get('medium_term', {}).get('30_days', {})
            else:
                pred_data = predictions.get('long_term', {}).get('60_days', {})
            
            if pred_data:
                predicted_price = pred_data.get('price', current_price)
            
            change_percent = ((predicted_price - current_price) / current_price) * 100
            
            return {
                'symbol': lstm_result['symbol'],
                'current_price': current_price,
                'predicted_price': predicted_price,
                'change_percent': round(change_percent, 2),
                'confidence': lstm_result['model_performance']['confidence'],
                'method_used': 'LSTM Enhanced',
                'predictions': predictions,
                'technical_indicators': {},
                'trend_analysis': {'direction': 'bullish' if change_percent > 0 else 'bearish'}
            }
            
        except Exception as e:
            return {"error": f"LSTM formatting failed: {str(e)}"}
    
    async def _technical_prediction(self, symbol: str, days: int, risk_tolerance: int):
        """Technical analysis prediction"""
        try:
            # Get stock data (simplified for microservice)
            data = await self._get_stock_data(symbol)
            
            if not data or data.get('error'):
                return {"error": f"No data available for {symbol}"}
            
            current_price = data['current_price']
            
            # Calculate technical indicators
            tech_indicators = self._calculate_basic_indicators(data['price_history'])
            
            # Generate prediction
            predicted_price = self._predict_price_technical(current_price, tech_indicators, days)
            change_percent = ((predicted_price - current_price) / current_price) * 100
            
            # Calculate confidence
            confidence = self._calculate_confidence(tech_indicators, days)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'change_percent': round(change_percent, 2),
                'confidence': confidence,
                'method_used': 'Technical Analysis',
                'predictions': self._generate_timeframe_predictions(current_price, tech_indicators),
                'technical_indicators': tech_indicators,
                'trend_analysis': self._analyze_trend(tech_indicators)
            }
            
        except Exception as e:
            return {"error": f"Technical prediction failed: {str(e)}"}
    
    async def _get_stock_data(self, symbol: str):
        """Get stock data (simplified)"""
        try:
            # Try VN stock first
            if self._is_vn_stock(symbol):
                return await self._get_vn_stock_data(symbol)
            else:
                return await self._get_international_data(symbol)
                
        except Exception as e:
            return {"error": f"Data retrieval failed: {str(e)}"}
    
    def _is_vn_stock(self, symbol: str):
        """Check if symbol is Vietnamese stock"""
        vn_stocks = ['VCB', 'BID', 'CTG', 'TCB', 'ACB', 'MBB', 'VPB', 'VIC', 'VHM', 'VRE', 
                    'MSN', 'MWG', 'VNM', 'SAB', 'PNJ', 'HPG', 'HSG', 'GAS', 'PLX', 'FPT']
        return symbol.upper() in vn_stocks
    
    async def _get_vn_stock_data(self, symbol: str):
        """Get Vietnamese stock data"""
        try:
            # Simplified VN stock data retrieval
            import yfinance as yf
            
            # Try with .VN suffix for Vietnamese stocks
            ticker = yf.Ticker(f"{symbol}.VN")
            hist = ticker.history(period="1y")
            
            if hist.empty:
                # Fallback to direct symbol
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
            
            if hist.empty:
                return {"error": f"No data found for {symbol}"}
            
            current_price = float(hist['Close'].iloc[-1])
            
            return {
                'current_price': current_price,
                'price_history': hist,
                'data_source': 'Yahoo Finance VN'
            }
            
        except Exception as e:
            return {"error": f"VN stock data error: {str(e)}"}
    
    async def _get_international_data(self, symbol: str):
        """Get international stock data"""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if hist.empty:
                return {"error": f"No data found for {symbol}"}
            
            current_price = float(hist['Close'].iloc[-1])
            
            return {
                'current_price': current_price,
                'price_history': hist,
                'data_source': 'Yahoo Finance'
            }
            
        except Exception as e:
            return {"error": f"International data error: {str(e)}"}
    
    def _calculate_basic_indicators(self, price_data):
        """Calculate basic technical indicators"""
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
    
    def _predict_price_technical(self, current_price: float, indicators: dict, days: int):
        """Predict price using technical analysis"""
        try:
            # Simple prediction based on technical indicators
            change_factor = 0
            
            # RSI influence
            rsi = indicators.get('rsi', 50)
            if rsi > 70:
                change_factor -= 0.02  # Overbought
            elif rsi < 30:
                change_factor += 0.02  # Oversold
            
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
                change_factor += 0.015  # Bullish alignment
            elif current_price < sma_20 < sma_50:
                change_factor -= 0.015  # Bearish alignment
            
            # Time factor
            time_factor = np.sqrt(days / 30)
            
            # Apply volatility
            volatility = indicators.get('volatility', 20) / 100
            random_factor = np.random.uniform(-volatility/2, volatility/2)
            
            total_change = (change_factor + random_factor) * time_factor
            
            # Limit change to reasonable bounds
            total_change = max(-0.3, min(0.3, total_change))
            
            predicted_price = current_price * (1 + total_change)
            
            return round(predicted_price, 2)
            
        except Exception as e:
            return current_price
    
    def _calculate_confidence(self, indicators: dict, days: int):
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
            
            # Time adjustment
            if days <= 7:
                base_confidence += 5
            elif days > 90:
                base_confidence -= 10
            
            return max(20, min(90, base_confidence))
            
        except Exception as e:
            return 50
    
    def _generate_timeframe_predictions(self, current_price: float, indicators: dict):
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
                    predicted_price = self._predict_price_technical(current_price, indicators, days)
                    change_percent = ((predicted_price - current_price) / current_price) * 100
                    
                    predictions[timeframe][f"{days}_days"] = {
                        "price": predicted_price,
                        "change_percent": round(change_percent, 2),
                        "change_amount": round(predicted_price - current_price, 2)
                    }
            
            return predictions
            
        except Exception as e:
            return {}
    
    def _analyze_trend(self, indicators: dict):
        """Analyze market trend"""
        try:
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            
            bullish_signals = 0
            bearish_signals = 0
            
            # RSI signals
            if rsi > 60:
                bullish_signals += 1
            elif rsi < 40:
                bearish_signals += 1
            
            # MACD signals
            if macd > macd_signal:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            if bullish_signals > bearish_signals:
                direction = "bullish"
                strength = "Strong" if bullish_signals >= 2 else "Moderate"
            elif bearish_signals > bullish_signals:
                direction = "bearish"
                strength = "Strong" if bearish_signals >= 2 else "Moderate"
            else:
                direction = "neutral"
                strength = "Neutral"
            
            return {
                "direction": direction,
                "strength": f"{strength} {direction.title()}",
                "rsi": round(rsi, 1),
                "macd": round(macd, 4)
            }
            
        except Exception as e:
            return {"direction": "neutral", "strength": "Unknown"}
    
    async def get_technical_analysis(self, symbol: str):
        """Get technical analysis for a symbol"""
        try:
            data = await self._get_stock_data(symbol)
            
            if data.get('error'):
                return data
            
            indicators = self._calculate_basic_indicators(data['price_history'])
            trend = self._analyze_trend(indicators)
            
            return {
                "symbol": symbol,
                "current_price": data['current_price'],
                "technical_indicators": indicators,
                "trend_analysis": trend,
                "data_source": data.get('data_source', 'Unknown')
            }
            
        except Exception as e:
            return {"error": f"Technical analysis failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)