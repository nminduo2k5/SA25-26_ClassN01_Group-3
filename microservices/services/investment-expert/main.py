from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
import requests
from typing import Optional, Dict, Any, List
import numpy as np
from datetime import datetime

app = FastAPI(title="Investment Expert Service", version="1.0.0")

# Redis connection
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

class InvestmentRequest(BaseModel):
    symbol: str
    risk_tolerance: int = 50
    time_horizon: str = "medium"
    investment_amount: int = 10000000
    current_price: Optional[float] = None

class InvestmentResponse(BaseModel):
    symbol: str
    recommendation: str
    confidence: float
    target_price: float
    stop_loss: float
    position_size: int
    investment_rationale: str
    risk_assessment: Dict[str, Any]
    financial_metrics: Dict[str, Any]

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "investment-expert",
        "redis_connected": redis_client.ping(),
        "analysis_types": ["Fundamental", "Technical", "Risk-Adjusted"]
    }

@app.post("/analyze", response_model=InvestmentResponse)
async def analyze_investment(request: InvestmentRequest):
    """Comprehensive investment analysis with BUY/SELL/HOLD recommendation"""
    try:
        # Check cache first
        cache_key = f"investment:{request.symbol}:{request.risk_tolerance}:{request.time_horizon}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            cached_data = json.loads(cached_result)
            return InvestmentResponse(**cached_data)
        
        # Get stock data and technical analysis
        stock_data = await get_stock_data(request.symbol)
        if not stock_data or stock_data.get('error'):
            raise HTTPException(status_code=400, detail=f"Cannot get data for {request.symbol}")
        
        current_price = request.current_price or stock_data['current_price']
        
        # Perform comprehensive analysis
        expert = InvestmentExpert()
        analysis = await expert.analyze_comprehensive(
            symbol=request.symbol,
            current_price=current_price,
            risk_tolerance=request.risk_tolerance,
            time_horizon=request.time_horizon,
            investment_amount=request.investment_amount,
            stock_data=stock_data
        )
        
        if analysis.get('error'):
            raise HTTPException(status_code=400, detail=analysis['error'])
        
        # Cache the result for 10 minutes
        redis_client.setex(cache_key, 600, json.dumps(analysis))
        
        return InvestmentResponse(**analysis)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Investment analysis error: {str(e)}")

@app.get("/analyze/{symbol}")
async def analyze_symbol(symbol: str, risk_tolerance: int = 50, time_horizon: str = "medium"):
    """Quick investment analysis for a symbol"""
    request = InvestmentRequest(
        symbol=symbol, 
        risk_tolerance=risk_tolerance, 
        time_horizon=time_horizon
    )
    return await analyze_investment(request)

@app.get("/recommendation/{symbol}")
async def get_recommendation(symbol: str):
    """Get simple BUY/SELL/HOLD recommendation"""
    try:
        expert = InvestmentExpert()
        recommendation = await expert.get_simple_recommendation(symbol)
        
        return {
            "symbol": symbol,
            "recommendation": recommendation['action'],
            "confidence": recommendation['confidence'],
            "reasoning": recommendation['reasoning'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

@app.post("/portfolio-analysis")
async def analyze_portfolio(symbols: List[str], weights: List[float]):
    """Analyze portfolio of stocks"""
    try:
        if len(symbols) != len(weights) or abs(sum(weights) - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail="Invalid symbols/weights")
        
        expert = InvestmentExpert()
        portfolio_analysis = await expert.analyze_portfolio(symbols, weights)
        
        return portfolio_analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio analysis error: {str(e)}")

async def get_stock_data(symbol: str):
    """Get stock data from Price Predictor service"""
    try:
        response = requests.get(
            f"http://price-predictor:8001/technical/{symbol}",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Cannot get data for {symbol}"}
            
    except Exception as e:
        return {"error": f"Data retrieval failed: {str(e)}"}

class InvestmentExpert:
    """Investment Expert with fundamental and technical analysis"""
    
    def __init__(self):
        self.name = "Investment Expert Microservice"
    
    async def analyze_comprehensive(self, symbol: str, current_price: float, 
                                  risk_tolerance: int, time_horizon: str, 
                                  investment_amount: int, stock_data: dict):
        """Comprehensive investment analysis"""
        try:
            # Extract technical data
            tech_indicators = stock_data.get('technical_indicators', {})
            trend_analysis = stock_data.get('trend_analysis', {})
            
            # Calculate financial metrics
            financial_metrics = self._calculate_financial_metrics(current_price, tech_indicators)
            
            # Risk assessment
            risk_assessment = self._assess_risk(tech_indicators, risk_tolerance)
            
            # Generate recommendation
            recommendation_data = self._generate_recommendation(
                current_price, tech_indicators, trend_analysis, 
                risk_tolerance, time_horizon, financial_metrics
            )
            
            # Calculate position sizing
            position_size = self._calculate_position_size(
                current_price, investment_amount, risk_tolerance, risk_assessment
            )
            
            # Calculate target price and stop loss
            target_price, stop_loss = self._calculate_price_targets(
                current_price, recommendation_data['action'], risk_tolerance, tech_indicators
            )
            
            # Generate investment rationale
            rationale = self._generate_rationale(
                recommendation_data, financial_metrics, risk_assessment, time_horizon
            )
            
            return {
                "symbol": symbol,
                "recommendation": recommendation_data['action'],
                "confidence": recommendation_data['confidence'],
                "target_price": target_price,
                "stop_loss": stop_loss,
                "position_size": position_size,
                "investment_rationale": rationale,
                "risk_assessment": risk_assessment,
                "financial_metrics": financial_metrics
            }
            
        except Exception as e:
            return {"error": f"Comprehensive analysis failed: {str(e)}"}
    
    def _calculate_financial_metrics(self, current_price: float, tech_indicators: dict):
        """Calculate key financial metrics"""
        try:
            volatility = tech_indicators.get('volatility', 20)
            rsi = tech_indicators.get('rsi', 50)
            
            # Estimate P/E ratio based on technical indicators
            estimated_pe = 15 + (rsi - 50) * 0.2  # Simple estimation
            estimated_pe = max(5, min(30, estimated_pe))
            
            # Estimate other ratios
            estimated_pb = 1.5 + (volatility - 20) * 0.05
            estimated_pb = max(0.5, min(5, estimated_pb))
            
            # Calculate momentum indicators
            momentum_score = self._calculate_momentum_score(tech_indicators)
            
            # Value score
            value_score = self._calculate_value_score(estimated_pe, estimated_pb)
            
            return {
                "estimated_pe": round(estimated_pe, 2),
                "estimated_pb": round(estimated_pb, 2),
                "momentum_score": momentum_score,
                "value_score": value_score,
                "volatility": round(volatility, 2),
                "rsi": round(rsi, 1),
                "overall_score": round((momentum_score + value_score) / 2, 1)
            }
            
        except Exception as e:
            return {"error": f"Financial metrics calculation failed: {str(e)}"}
    
    def _calculate_momentum_score(self, tech_indicators: dict):
        """Calculate momentum score (0-100)"""
        try:
            score = 50  # Base score
            
            # RSI contribution
            rsi = tech_indicators.get('rsi', 50)
            if 30 <= rsi <= 70:
                score += 20
            elif rsi > 70:
                score += 10  # Overbought but still positive momentum
            elif rsi < 30:
                score -= 10  # Oversold
            
            # MACD contribution
            macd = tech_indicators.get('macd', 0)
            macd_signal = tech_indicators.get('macd_signal', 0)
            if macd > macd_signal:
                score += 15
            else:
                score -= 15
            
            # Moving average contribution
            sma_20 = tech_indicators.get('sma_20', 0)
            sma_50 = tech_indicators.get('sma_50', 0)
            if sma_20 > sma_50:
                score += 15
            else:
                score -= 15
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 50
    
    def _calculate_value_score(self, pe_ratio: float, pb_ratio: float):
        """Calculate value score based on ratios"""
        try:
            score = 50  # Base score
            
            # P/E ratio scoring
            if pe_ratio < 10:
                score += 25  # Very undervalued
            elif pe_ratio < 15:
                score += 15  # Undervalued
            elif pe_ratio < 20:
                score += 5   # Fair value
            elif pe_ratio < 25:
                score -= 10  # Slightly overvalued
            else:
                score -= 20  # Overvalued
            
            # P/B ratio scoring
            if pb_ratio < 1:
                score += 15  # Trading below book value
            elif pb_ratio < 2:
                score += 10  # Reasonable
            elif pb_ratio < 3:
                score += 0   # Neutral
            else:
                score -= 15  # High P/B
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 50
    
    def _assess_risk(self, tech_indicators: dict, risk_tolerance: int):
        """Assess investment risk"""
        try:
            volatility = tech_indicators.get('volatility', 20)
            rsi = tech_indicators.get('rsi', 50)
            
            # Calculate risk score
            risk_score = 50  # Base risk
            
            # Volatility risk
            if volatility > 40:
                risk_score += 30  # High risk
            elif volatility > 25:
                risk_score += 15  # Medium risk
            elif volatility < 15:
                risk_score -= 10  # Low risk
            
            # RSI extremes add risk
            if rsi > 80 or rsi < 20:
                risk_score += 20
            elif rsi > 70 or rsi < 30:
                risk_score += 10
            
            # Risk level classification
            if risk_score < 30:
                risk_level = "Low"
            elif risk_score < 60:
                risk_level = "Medium"
            elif risk_score < 80:
                risk_level = "High"
            else:
                risk_level = "Very High"
            
            # Risk tolerance match
            risk_match = 100 - abs(risk_score - risk_tolerance)
            
            return {
                "risk_score": max(0, min(100, risk_score)),
                "risk_level": risk_level,
                "volatility": round(volatility, 2),
                "risk_tolerance_match": max(0, min(100, risk_match)),
                "suitable_for_profile": risk_match > 60
            }
            
        except Exception as e:
            return {"error": f"Risk assessment failed: {str(e)}"}
    
    def _generate_recommendation(self, current_price: float, tech_indicators: dict, 
                               trend_analysis: dict, risk_tolerance: int, 
                               time_horizon: str, financial_metrics: dict):
        """Generate BUY/SELL/HOLD recommendation"""
        try:
            # Scoring system
            buy_score = 0
            sell_score = 0
            
            # Technical analysis scoring
            rsi = tech_indicators.get('rsi', 50)
            macd = tech_indicators.get('macd', 0)
            macd_signal = tech_indicators.get('macd_signal', 0)
            
            # RSI signals
            if rsi < 30:
                buy_score += 3  # Oversold
            elif rsi < 40:
                buy_score += 1
            elif rsi > 70:
                sell_score += 3  # Overbought
            elif rsi > 60:
                sell_score += 1
            
            # MACD signals
            if macd > macd_signal:
                buy_score += 2
            else:
                sell_score += 2
            
            # Trend analysis
            trend_direction = trend_analysis.get('direction', 'neutral')
            if trend_direction == 'bullish':
                buy_score += 2
            elif trend_direction == 'bearish':
                sell_score += 2
            
            # Financial metrics
            overall_score = financial_metrics.get('overall_score', 50)
            if overall_score > 70:
                buy_score += 2
            elif overall_score < 30:
                sell_score += 2
            
            # Time horizon adjustment
            if time_horizon == "short":
                # More weight on technical signals
                buy_score *= 1.2
                sell_score *= 1.2
            elif time_horizon == "long":
                # More weight on fundamentals
                if overall_score > 60:
                    buy_score += 1
                elif overall_score < 40:
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
                "sell_score": sell_score,
                "reasoning": f"Technical: {buy_score-sell_score:+d}, Fundamental: {overall_score:.0f}/100"
            }
            
        except Exception as e:
            return {"action": "HOLD", "confidence": 50, "error": str(e)}
    
    def _calculate_position_size(self, current_price: float, investment_amount: int, 
                               risk_tolerance: int, risk_assessment: dict):
        """Calculate appropriate position size"""
        try:
            # Base position size based on risk tolerance
            if risk_tolerance <= 30:
                max_position_pct = 0.05  # 5% max for conservative
            elif risk_tolerance <= 70:
                max_position_pct = 0.10  # 10% max for moderate
            else:
                max_position_pct = 0.20  # 20% max for aggressive
            
            # Adjust based on stock risk
            risk_score = risk_assessment.get('risk_score', 50)
            risk_adjustment = 1 - (risk_score - 50) / 200  # Reduce position for high risk
            risk_adjustment = max(0.3, min(1.5, risk_adjustment))
            
            # Calculate position size
            adjusted_position_pct = max_position_pct * risk_adjustment
            max_investment = investment_amount * adjusted_position_pct
            
            position_size = int(max_investment / current_price)
            
            return max(1, position_size)  # At least 1 share
            
        except Exception as e:
            return 1
    
    def _calculate_price_targets(self, current_price: float, recommendation: str, 
                               risk_tolerance: int, tech_indicators: dict):
        """Calculate target price and stop loss"""
        try:
            volatility = tech_indicators.get('volatility', 20) / 100
            
            if recommendation in ["STRONG BUY", "BUY"]:
                # Target price: 10-25% above current
                target_multiplier = 1.10 + (volatility * 0.5)
                target_price = current_price * min(1.25, target_multiplier)
                
                # Stop loss: 5-15% below current
                stop_loss_pct = 0.05 + (volatility * 0.3)
                stop_loss = current_price * (1 - min(0.15, stop_loss_pct))
                
            elif recommendation in ["STRONG SELL", "SELL"]:
                # Target price: 10-25% below current
                target_multiplier = 0.90 - (volatility * 0.5)
                target_price = current_price * max(0.75, target_multiplier)
                
                # Stop loss: 5-15% above current (for short positions)
                stop_loss_pct = 0.05 + (volatility * 0.3)
                stop_loss = current_price * (1 + min(0.15, stop_loss_pct))
                
            else:  # HOLD
                target_price = current_price * 1.05  # 5% upside
                stop_loss = current_price * 0.95     # 5% downside
            
            # Adjust for risk tolerance
            if risk_tolerance <= 30:  # Conservative
                # Tighter targets
                target_price = current_price + (target_price - current_price) * 0.7
                stop_loss = current_price + (stop_loss - current_price) * 0.7
            
            return round(target_price, 2), round(stop_loss, 2)
            
        except Exception as e:
            return current_price * 1.05, current_price * 0.95
    
    def _generate_rationale(self, recommendation_data: dict, financial_metrics: dict, 
                          risk_assessment: dict, time_horizon: str):
        """Generate investment rationale"""
        try:
            action = recommendation_data['action']
            confidence = recommendation_data['confidence']
            overall_score = financial_metrics.get('overall_score', 50)
            risk_level = risk_assessment.get('risk_level', 'Medium')
            
            rationale_parts = []
            
            # Main recommendation reasoning
            if action in ["STRONG BUY", "BUY"]:
                rationale_parts.append(f"Khuyến nghị {action} với độ tin cậy {confidence:.1f}%.")
                if overall_score > 70:
                    rationale_parts.append("Cổ phiếu có điểm số tổng thể cao, cho thấy tiềm năng tăng trưởng tốt.")
                rationale_parts.append(f"Phù hợp với chiến lược đầu tư {time_horizon}.")
                
            elif action in ["STRONG SELL", "SELL"]:
                rationale_parts.append(f"Khuyến nghị {action} với độ tin cậy {confidence:.1f}%.")
                if overall_score < 30:
                    rationale_parts.append("Cổ phiếu có điểm số thấp, có thể gặp khó khăn trong thời gian tới.")
                rationale_parts.append("Nên cân nhắc thoát vị thế hoặc chờ thời điểm tốt hơn.")
                
            else:  # HOLD
                rationale_parts.append(f"Khuyến nghị {action} - theo dõi thêm.")
                rationale_parts.append("Tín hiệu thị trường chưa rõ ràng, nên duy trì vị thế hiện tại.")
            
            # Risk assessment
            rationale_parts.append(f"Mức độ rủi ro: {risk_level}.")
            
            # Technical insights
            momentum_score = financial_metrics.get('momentum_score', 50)
            if momentum_score > 70:
                rationale_parts.append("Momentum kỹ thuật tích cực.")
            elif momentum_score < 30:
                rationale_parts.append("Momentum kỹ thuật yếu.")
            
            return " ".join(rationale_parts)
            
        except Exception as e:
            return f"Phân tích đầu tư cho {action} với độ tin cậy {confidence:.1f}%."
    
    async def get_simple_recommendation(self, symbol: str):
        """Get simple BUY/SELL/HOLD recommendation"""
        try:
            # Get basic stock data
            stock_data = await get_stock_data(symbol)
            if stock_data.get('error'):
                return {"action": "HOLD", "confidence": 30, "reasoning": "Insufficient data"}
            
            current_price = stock_data['current_price']
            tech_indicators = stock_data.get('technical_indicators', {})
            trend_analysis = stock_data.get('trend_analysis', {})
            
            # Calculate financial metrics
            financial_metrics = self._calculate_financial_metrics(current_price, tech_indicators)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                current_price, tech_indicators, trend_analysis, 
                50, "medium", financial_metrics  # Default values
            )
            
            return {
                "action": recommendation['action'],
                "confidence": recommendation['confidence'],
                "reasoning": recommendation['reasoning']
            }
            
        except Exception as e:
            return {"action": "HOLD", "confidence": 30, "reasoning": f"Analysis error: {str(e)}"}
    
    async def analyze_portfolio(self, symbols: List[str], weights: List[float]):
        """Analyze portfolio of stocks"""
        try:
            portfolio_analysis = {
                "symbols": symbols,
                "weights": weights,
                "individual_analysis": [],
                "portfolio_metrics": {},
                "recommendations": []
            }
            
            total_score = 0
            total_risk = 0
            
            # Analyze each stock
            for symbol, weight in zip(symbols, weights):
                try:
                    stock_data = await get_stock_data(symbol)
                    if stock_data.get('error'):
                        continue
                    
                    current_price = stock_data['current_price']
                    tech_indicators = stock_data.get('technical_indicators', {})
                    
                    # Calculate metrics
                    financial_metrics = self._calculate_financial_metrics(current_price, tech_indicators)
                    risk_assessment = self._assess_risk(tech_indicators, 50)
                    
                    stock_analysis = {
                        "symbol": symbol,
                        "weight": weight,
                        "current_price": current_price,
                        "overall_score": financial_metrics.get('overall_score', 50),
                        "risk_score": risk_assessment.get('risk_score', 50),
                        "volatility": tech_indicators.get('volatility', 20)
                    }
                    
                    portfolio_analysis["individual_analysis"].append(stock_analysis)
                    
                    # Weighted portfolio metrics
                    total_score += financial_metrics.get('overall_score', 50) * weight
                    total_risk += risk_assessment.get('risk_score', 50) * weight
                    
                except Exception as e:
                    print(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Portfolio-level metrics
            portfolio_analysis["portfolio_metrics"] = {
                "weighted_score": round(total_score, 1),
                "weighted_risk": round(total_risk, 1),
                "diversification_score": self._calculate_diversification_score(symbols),
                "overall_rating": self._get_portfolio_rating(total_score, total_risk)
            }
            
            # Portfolio recommendations
            portfolio_analysis["recommendations"] = self._generate_portfolio_recommendations(
                total_score, total_risk, len(symbols)
            )
            
            return portfolio_analysis
            
        except Exception as e:
            return {"error": f"Portfolio analysis failed: {str(e)}"}
    
    def _calculate_diversification_score(self, symbols: List[str]):
        """Calculate portfolio diversification score"""
        # Simple diversification based on number of stocks
        num_stocks = len(symbols)
        if num_stocks >= 10:
            return 90
        elif num_stocks >= 5:
            return 70
        elif num_stocks >= 3:
            return 50
        else:
            return 30
    
    def _get_portfolio_rating(self, score: float, risk: float):
        """Get overall portfolio rating"""
        if score > 70 and risk < 50:
            return "Excellent"
        elif score > 60 and risk < 60:
            return "Good"
        elif score > 50 and risk < 70:
            return "Average"
        elif score > 40:
            return "Below Average"
        else:
            return "Poor"
    
    def _generate_portfolio_recommendations(self, score: float, risk: float, num_stocks: int):
        """Generate portfolio-level recommendations"""
        recommendations = []
        
        if score > 70:
            recommendations.append("Portfolio có tiềm năng tăng trưởng tốt")
        elif score < 40:
            recommendations.append("Cần xem xét điều chỉnh danh mục đầu tư")
        
        if risk > 70:
            recommendations.append("Mức độ rủi ro cao - cân nhắc giảm tỷ trọng cổ phiếu rủi ro")
        elif risk < 30:
            recommendations.append("Danh mục ổn định - có thể tăng tỷ trọng cổ phiếu tăng trưởng")
        
        if num_stocks < 5:
            recommendations.append("Nên đa dạng hóa thêm để giảm rủi ro")
        elif num_stocks > 20:
            recommendations.append("Danh mục có thể quá phân tán - khó quản lý")
        
        return recommendations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)