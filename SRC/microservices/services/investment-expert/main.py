from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
import requests
from typing import Optional, Dict, Any
import numpy as np

app = FastAPI(title="Investment Expert Service", version="1.0.0")

# Redis connection
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

class InvestmentRequest(BaseModel):
    symbol: str
    risk_tolerance: int = 50
    time_horizon: str = "medium"
    investment_amount: int = 10000000

class InvestmentResponse(BaseModel):
    symbol: str
    recommendation: str
    confidence: float
    target_price: float
    stop_loss: float
    position_size: int
    investment_rationale: str

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "investment-expert",
        "redis_connected": redis_client.ping()
    }

@app.post("/analyze", response_model=InvestmentResponse)
async def analyze_investment(request: InvestmentRequest):
    """Investment analysis with BUY/SELL/HOLD recommendation"""
    try:
        # Check cache
        cache_key = f"investment:{request.symbol}:{request.risk_tolerance}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            return InvestmentResponse(**json.loads(cached_result))
        
        # Get stock data from price predictor
        stock_data = await get_stock_data(request.symbol)
        if not stock_data or stock_data.get('error'):
            raise HTTPException(status_code=400, detail=f"Cannot get data for {request.symbol}")
        
        # Perform analysis
        expert = InvestmentExpert()
        analysis = await expert.analyze(request, stock_data)
        
        # Cache result
        redis_client.setex(cache_key, 600, json.dumps(analysis))
        
        return InvestmentResponse(**analysis)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Investment analysis error: {str(e)}")

@app.get("/analyze/{symbol}")
async def analyze_symbol(symbol: str):
    """Quick analysis for a symbol"""
    request = InvestmentRequest(symbol=symbol)
    return await analyze_investment(request)

@app.get("/recommendation/{symbol}")
async def get_recommendation(symbol: str):
    """Get simple BUY/SELL/HOLD recommendation"""
    try:
        expert = InvestmentExpert()
        recommendation = await expert.get_simple_recommendation(symbol)
        return recommendation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

async def get_stock_data(symbol: str):
    """Get stock data from Price Predictor service"""
    try:
        response = requests.get(f"http://price-predictor:8001/technical/{symbol}", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Cannot get data for {symbol}"}
    except Exception as e:
        return {"error": f"Data retrieval failed: {str(e)}"}

class InvestmentExpert:
    """Investment Expert with fundamental and technical analysis"""
    
    async def analyze(self, request: InvestmentRequest, stock_data: dict):
        """Comprehensive investment analysis"""
        try:
            current_price = stock_data['current_price']
            tech_indicators = stock_data.get('technical_indicators', {})
            
            # Generate recommendation
            recommendation_data = self._generate_recommendation(tech_indicators, request.risk_tolerance)
            
            # Calculate position sizing
            position_size = self._calculate_position_size(
                current_price, request.investment_amount, request.risk_tolerance
            )
            
            # Calculate price targets
            target_price, stop_loss = self._calculate_price_targets(
                current_price, recommendation_data['action'], tech_indicators
            )
            
            # Generate rationale
            rationale = self._generate_rationale(recommendation_data, tech_indicators)
            
            return {
                "symbol": request.symbol,
                "recommendation": recommendation_data['action'],
                "confidence": recommendation_data['confidence'],
                "target_price": target_price,
                "stop_loss": stop_loss,
                "position_size": position_size,
                "investment_rationale": rationale
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _generate_recommendation(self, tech_indicators: dict, risk_tolerance: int):
        """Generate BUY/SELL/HOLD recommendation"""
        try:
            buy_score = 0
            sell_score = 0
            
            # RSI signals
            rsi = tech_indicators.get('rsi', 50)
            if rsi < 30:
                buy_score += 3
            elif rsi < 40:
                buy_score += 1
            elif rsi > 70:
                sell_score += 3
            elif rsi > 60:
                sell_score += 1
            
            # MACD signals
            macd = tech_indicators.get('macd', 0)
            macd_signal = tech_indicators.get('macd_signal', 0)
            if macd > macd_signal:
                buy_score += 2
            else:
                sell_score += 2
            
            # Moving average signals
            sma_20 = tech_indicators.get('sma_20', 0)
            sma_50 = tech_indicators.get('sma_50', 0)
            if sma_20 > sma_50:
                buy_score += 1
            else:
                sell_score += 1
            
            # Generate recommendation
            if buy_score > sell_score + 2:
                if buy_score >= 6:
                    action = "STRONG BUY"
                    confidence = min(90, 70 + buy_score * 3)
                else:
                    action = "BUY"
                    confidence = min(80, 60 + buy_score * 3)
            elif sell_score > buy_score + 2:
                if sell_score >= 6:
                    action = "STRONG SELL"
                    confidence = min(90, 70 + sell_score * 3)
                else:
                    action = "SELL"
                    confidence = min(80, 60 + sell_score * 3)
            else:
                action = "HOLD"
                confidence = 50 + abs(buy_score - sell_score) * 5
            
            return {
                "action": action,
                "confidence": round(confidence, 1),
                "buy_score": buy_score,
                "sell_score": sell_score
            }
            
        except Exception as e:
            return {"action": "HOLD", "confidence": 50}
    
    def _calculate_position_size(self, current_price: float, investment_amount: int, risk_tolerance: int):
        """Calculate appropriate position size"""
        try:
            # Base position size based on risk tolerance
            if risk_tolerance <= 30:
                max_position_pct = 0.05  # 5% max for conservative
            elif risk_tolerance <= 70:
                max_position_pct = 0.10  # 10% max for moderate
            else:
                max_position_pct = 0.20  # 20% max for aggressive
            
            max_investment = investment_amount * max_position_pct
            position_size = int(max_investment / current_price)
            
            return max(1, position_size)
            
        except Exception as e:
            return 1
    
    def _calculate_price_targets(self, current_price: float, recommendation: str, tech_indicators: dict):
        """Calculate target price and stop loss"""
        try:
            volatility = tech_indicators.get('volatility', 20) / 100
            
            if recommendation in ["STRONG BUY", "BUY"]:
                target_price = current_price * (1.10 + volatility * 0.5)
                stop_loss = current_price * (1 - min(0.15, 0.05 + volatility * 0.3))
            elif recommendation in ["STRONG SELL", "SELL"]:
                target_price = current_price * (0.90 - volatility * 0.5)
                stop_loss = current_price * (1 + min(0.15, 0.05 + volatility * 0.3))
            else:  # HOLD
                target_price = current_price * 1.05
                stop_loss = current_price * 0.95
            
            return round(target_price, 2), round(stop_loss, 2)
            
        except Exception as e:
            return current_price * 1.05, current_price * 0.95
    
    def _generate_rationale(self, recommendation_data: dict, tech_indicators: dict):
        """Generate investment rationale"""
        try:
            action = recommendation_data['action']
            confidence = recommendation_data['confidence']
            rsi = tech_indicators.get('rsi', 50)
            
            rationale_parts = []
            
            if action in ["STRONG BUY", "BUY"]:
                rationale_parts.append(f"Khuyến nghị {action} với độ tin cậy {confidence:.1f}%.")
                if rsi < 40:
                    rationale_parts.append("RSI cho thấy cổ phiếu đang oversold, có cơ hội phục hồi.")
            elif action in ["STRONG SELL", "SELL"]:
                rationale_parts.append(f"Khuyến nghị {action} với độ tin cậy {confidence:.1f}%.")
                if rsi > 60:
                    rationale_parts.append("RSI cho thấy cổ phiếu đang overbought, có thể điều chỉnh.")
            else:
                rationale_parts.append("Khuyến nghị HOLD - tín hiệu thị trường chưa rõ ràng.")
            
            return " ".join(rationale_parts)
            
        except Exception as e:
            return f"Phân tích đầu tư với độ tin cậy {confidence:.1f}%."
    
    async def get_simple_recommendation(self, symbol: str):
        """Get simple recommendation"""
        try:
            stock_data = await get_stock_data(symbol)
            if stock_data.get('error'):
                return {"action": "HOLD", "confidence": 30, "reasoning": "Insufficient data"}
            
            tech_indicators = stock_data.get('technical_indicators', {})
            recommendation = self._generate_recommendation(tech_indicators, 50)
            
            return {
                "action": recommendation['action'],
                "confidence": recommendation['confidence'],
                "reasoning": f"Technical analysis based on RSI: {tech_indicators.get('rsi', 50):.1f}"
            }
            
        except Exception as e:
            return {"action": "HOLD", "confidence": 30, "reasoning": f"Analysis error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)