# Import with error handling
try:
    from agents.price_predictor import PricePredictor
    from agents.ticker_news import TickerNews
    from agents.market_news import MarketNews
    from agents.investment_expert import InvestmentExpert
    from agents.risk_expert import RiskExpert
    from agents.stock_info import StockInfoDisplay
    from agents.international_news import InternationalMarketNews
except ImportError as e:
    print(f"⚠️ Some agents not available: {e}")
    # Create dummy classes for missing agents
    class DummyAgent:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: {"error": "Agent not available"}
    
    PricePredictor = DummyAgent
    TickerNews = DummyAgent
    MarketNews = DummyAgent
    InvestmentExpert = DummyAgent
    RiskExpert = DummyAgent
    StockInfoDisplay = DummyAgent
    InternationalMarketNews = DummyAgent

try:
    from unified_llm_agent import UnifiedLLMAgent
except ImportError:
    print("⚠️ Unified LLM agent not available")
    class UnifiedLLMAgent:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: {"error": "Unified LLM agent not available"}

try:
    from src.data.vn_stock_api import VNStockAPI
except ImportError:
    print("⚠️ VNStockAPI not available")
    class VNStockAPI:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: {"error": "VNStockAPI not available"}

try:
    from src.utils.error_handler import handle_async_errors, AgentErrorHandler, validate_symbol
except ImportError:
    print("⚠️ Error handler not available")
    def handle_async_errors(default_return=None):
        def decorator(func):
            return func
        return decorator
    
    class AgentErrorHandler:
        @staticmethod
        def handle_prediction_error(symbol, error):
            return {"error": f"Prediction error for {symbol}: {error}"}
        @staticmethod
        def handle_news_error(symbol, error):
            return {"error": f"News error for {symbol}: {error}"}
        @staticmethod
        def handle_risk_error(symbol, error):
            return {"error": f"Risk error for {symbol}: {error}"}
    
    def validate_symbol(symbol):
        return bool(symbol and len(symbol.strip()) > 0)

try:
    from fastapi.concurrency import run_in_threadpool
except ImportError:
    print("⚠️ FastAPI not available")
    import asyncio
    async def run_in_threadpool(func, *args, **kwargs):
        return func(*args, **kwargs)

try:
    from architectures import ArchitectureManager
except ImportError:
    print("⚠️ Architecture manager not available")
    class ArchitectureManager:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: {"error": "Architecture manager not available"}
        def get_architecture_info(self):
            return {
                "ensemble_voting": "Bayesian inference từ 6 agents (fallback)",
                "hierarchical": "Big Agent tổng hợp từ 6 agents (fallback)", 
                "round_robin": "6 agents cải thiện tuần tự (fallback)"
            }

import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MainAgent:
    def __init__(self, vn_api: VNStockAPI, gemini_api_key: str = None, openai_api_key: str = None, llama_api_key: str = None, llama_base_url: str = None, serper_api_key: str = None):
        self.vn_api = vn_api
        self.stock_info = StockInfoDisplay(vn_api)
        self.price_predictor = PricePredictor(vn_api, self.stock_info)
        self.ticker_news = TickerNews()
        self.market_news = MarketNews()
        self.investment_expert = InvestmentExpert(vn_api)
        self.risk_expert = RiskExpert(vn_api)
        self.international_news = InternationalMarketNews()
        
        # Initialize Unified LLM Agent with all API keys
        self.llm_agent = None
        if any([gemini_api_key, openai_api_key, llama_api_key]):
            try:
                self.llm_agent = UnifiedLLMAgent(
                    gemini_api_key=gemini_api_key,
                    openai_api_key=openai_api_key,
                    llama_api_key=llama_api_key,
                    llama_base_url=llama_base_url
                )
                status = self.llm_agent.get_agent_status()
                available_agents = [name for name, info in status['agents'].items() if not info['offline_mode']]
                print(f"✅ LLM Agents initialized: {', '.join(available_agents)} (Current: {status['current_agent']})")
            except Exception as e:
                print(f"⚠️ LLM initialization failed: {e}")
                self.llm_agent = None
        
        # Update VN API with Multi-LLM CrewAI keys
        if any([gemini_api_key, openai_api_key, llama_api_key]) or serper_api_key:
            try:
                self.vn_api.set_crewai_keys(gemini_api_key, openai_api_key, llama_api_key, llama_base_url, serper_api_key)
                print("✅ Multi-LLM CrewAI integration enabled for real news")
            except Exception as e:
                print(f"⚠️ Multi-LLM CrewAI setup failed: {e}")
        
        # Pass LLM agent to other agents for enhanced capabilities
        self._integrate_llm_with_agents()
        
        # Initialize Architecture Manager
        self.architecture_manager = ArchitectureManager(vn_api, gemini_api_key or "")
    
    def _integrate_llm_with_agents(self):
        """Integrate LLM capabilities with all agents"""
        if self.llm_agent:
            # Pass LLM agent to agents that can benefit from it
            if hasattr(self.price_predictor, 'set_llm_agent'):
                self.price_predictor.set_llm_agent(self.llm_agent)
            if hasattr(self.investment_expert, 'set_llm_agent'):
                self.investment_expert.set_llm_agent(self.llm_agent)
            if hasattr(self.risk_expert, 'set_llm_agent'):
                self.risk_expert.set_llm_agent(self.llm_agent)
            if hasattr(self.ticker_news, 'set_llm_agent'):
                self.ticker_news.set_llm_agent(self.llm_agent)
            if hasattr(self.market_news, 'set_llm_agent'):
                self.market_news.set_llm_agent(self.llm_agent)
            if hasattr(self.international_news, 'set_llm_agent'):
                self.international_news.set_llm_agent(self.llm_agent)
            
            # Update architecture manager with LLM agent
            if hasattr(self, 'architecture_manager'):
                # Try to get gemini key from agents dict
                gemini_key = ''
                if hasattr(self.llm_agent, 'agents') and isinstance(self.llm_agent.agents, dict):
                    if 'gemini' in self.llm_agent.agents:
                        gemini_agent = self.llm_agent.agents['gemini']
                        gemini_key = getattr(gemini_agent, 'gemini_api_key', '')
                self.architecture_manager = ArchitectureManager(self.vn_api, gemini_key)
    
    def set_llm_keys(self, gemini_api_key: str = None, openai_api_key: str = None, llama_api_key: str = None, llama_base_url: str = None):
        """Set or update LLM API keys"""
        try:
            # Create new unified agent
            self.llm_agent = UnifiedLLMAgent(
                gemini_api_key=gemini_api_key,
                openai_api_key=openai_api_key,
                llama_api_key=llama_api_key,
                llama_base_url=llama_base_url
            )
            
            # Get agent status
            status = self.llm_agent.get_agent_status()
            available_agents = [name for name, info in status['agents'].items() if info.get('truly_available', False)]
            offline_agents = [name for name, info in status['agents'].items() if info.get('has_models', False) and info.get('offline_mode', True)]
            
            if available_agents:
                print(f"✅ LLM keys updated successfully. Online: {', '.join(available_agents)}")
                if offline_agents:
                    print(f"⚠️ Offline: {', '.join(offline_agents)} (quota/rate limit)")
                self._integrate_llm_with_agents()
                return True
            elif offline_agents:
                print(f"⚠️ LLM agents initialized in offline mode: {', '.join(offline_agents)}")
                print("💡 Reasons: API quota exceeded, invalid keys, or rate limits")
                print("✅ System will work with offline fallback responses")
                self._integrate_llm_with_agents()
                return True
            else:
                print("❌ No LLM agents could be initialized")
                print("💡 Please check your API keys and try again")
                return False
        except Exception as e:
            print(f"❌ Failed to set LLM keys: {e}")
            import traceback
            traceback.print_exc()
            return False
    

    
    def set_crewai_keys(self, gemini_api_key: str = None, openai_api_key: str = None, 
                       llama_api_key: str = None, llama_base_url: str = None, serper_api_key: str = None):
        """Set Multi-LLM CrewAI API keys for real news collection"""
        try:
            # Update LLM agent if needed
            if any([gemini_api_key, openai_api_key, llama_api_key]) and not self.llm_agent:
                self.llm_agent = UnifiedLLMAgent(
                    gemini_api_key=gemini_api_key,
                    openai_api_key=openai_api_key,
                    llama_api_key=llama_api_key,
                    llama_base_url=llama_base_url
                )
                self._integrate_llm_with_agents()
            
            # Update VN API Multi-LLM CrewAI integration
            success = self.vn_api.set_crewai_keys(gemini_api_key, openai_api_key, llama_api_key, llama_base_url, serper_api_key)
            
            # Update architecture manager
            if gemini_api_key:
                self.architecture_manager = ArchitectureManager(self.vn_api, gemini_api_key)
            
            if success:
                print("✅ Multi-LLM CrewAI integration updated successfully")
                return True
            else:
                print("⚠️ Multi-LLM CrewAI integration not available")
                return False
                
        except Exception as e:
            print(f"❌ Failed to set Multi-LLM CrewAI keys: {e}")
            return False
    
    @handle_async_errors(default_return={"error": "Lỗi hệ thống khi phân tích cổ phiếu"})
    async def analyze_stock(self, symbol: str, risk_tolerance: int = 50, time_horizon: str = "Trung hạn", investment_amount: int = 100000000):
        """Phân tích toàn diện một mã cổ phiếu với hồ sơ đầu tư"""
        if not symbol or not validate_symbol(symbol):
            return {"error": "Mã cổ phiếu không hợp lệ"}
            
        symbol = symbol.upper().strip()
        logger.info(f"Starting comprehensive analysis for {symbol} with profile: {risk_tolerance}% risk, {time_horizon}, {investment_amount:,} VND")
        
        tasks = {}
        results = {"symbol": symbol}

        try:
            # Check if VN stock first
            if self.vn_api.is_vn_stock(symbol):
                logger.info(f"{symbol} is Vietnamese stock, using VN API")
                tasks['vn_stock_data'] = self.vn_api.get_stock_data(symbol)
                tasks['ticker_news'] = self.vn_api.get_news_sentiment(symbol)
                tasks['detailed_stock_info'] = self.stock_info.get_detailed_stock_data(symbol)
                market_type = 'Vietnam'
            else:
                # Kiểm tra xem có phải là mã hợp lệ cho international market không
                if self._is_valid_international_symbol(symbol):
                    logger.info(f"{symbol} is international stock, using international APIs")
                    tasks['ticker_news'] = run_in_threadpool(self._safe_get_ticker_news, symbol)
                    tasks['investment_analysis'] = run_in_threadpool(self._safe_get_investment_analysis, symbol, risk_tolerance, time_horizon, investment_amount)
                    market_type = 'International'
                else:
                    logger.warning(f"{symbol} is not a valid stock symbol")
                    return {"error": f"Mã {symbol} không hợp lệ hoặc không được hỗ trợ"}

            # Các tác vụ chung cho cả hai thị trường với investment profile
            tasks['price_prediction'] = run_in_threadpool(self._safe_get_price_prediction, symbol)
            tasks['risk_assessment'] = run_in_threadpool(self._safe_get_risk_assessment, symbol, risk_tolerance, time_horizon, investment_amount)
            
            # Add investment analysis for VN stocks too
            if market_type == 'Vietnam':
                tasks['investment_analysis'] = run_in_threadpool(self._safe_get_investment_analysis, symbol, risk_tolerance, time_horizon, investment_amount)

            # Thực thi tất cả các tác vụ song song
            task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

            # Ánh xạ kết quả với error handling
            for key, result in zip(tasks.keys(), task_results):
                if isinstance(result, Exception):
                    logger.error(f"Error in {key} for {symbol}: {result}")
                    results[key] = self._get_error_fallback(key, symbol, result)
                else:
                    results[key] = result
            
            # Add investment profile to results
            results['investment_profile'] = {
                'risk_tolerance': risk_tolerance,
                'time_horizon': time_horizon,
                'investment_amount': investment_amount,
                'risk_profile': self._get_risk_profile_name(risk_tolerance)
            }
            
            results['market_type'] = market_type
            results['analysis_timestamp'] = asyncio.get_event_loop().time()
            
            logger.info(f"Completed analysis for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Critical error in analyze_stock for {symbol}: {e}")
            return {"error": f"Lỗi nghiêm trọng khi phân tích {symbol}: {str(e)}"}
    
    @handle_async_errors(default_return={"error": "Lỗi khi lấy tổng quan thị trường"})
    async def get_market_overview(self):
        """Lấy tổng quan thị trường"""
        logger.info("Getting market overview")
        
        try:
            # Chạy tác vụ đồng bộ và bất đồng bộ song song với error handling
            international_task = run_in_threadpool(self._safe_get_international_market_news)
            vietnam_task = self.vn_api.get_market_overview()

            international_result, vietnam_result = await asyncio.gather(
                international_task, vietnam_task, return_exceptions=True
            )
            
            # Handle results with error checking
            results = {}
            
            if isinstance(international_result, Exception):
                logger.error(f"International market error: {international_result}")
                results['international_market'] = {"error": "Lỗi lấy tin tức thị trường quốc tế"}
            else:
                results['international_market'] = international_result
            
            if isinstance(vietnam_result, Exception):
                logger.error(f"Vietnam market error: {vietnam_result}")
                results['vietnam_market'] = {"error": "Lỗi lấy dữ liệu thị trường Việt Nam"}
            else:
                results['vietnam_market'] = vietnam_result
            
            results['timestamp'] = asyncio.get_event_loop().time()
            return results
            
        except Exception as e:
            logger.error(f"Critical error in get_market_overview: {e}")
            return {"error": f"Lỗi nghiêm trọng khi lấy tổng quan thị trường: {str(e)}"}
    
    @handle_async_errors(default_return={"error": "Lỗi xử lý truy vấn"})
    async def process_query(self, query: str, symbol: str = ""):
        """Xử lý truy vấn từ người dùng với AI response"""
        if not query or not query.strip():
            return {"error": "Vui lòng nhập câu hỏi"}
        
        query = query.strip()
        symbol = symbol.strip().upper() if symbol else ""
        
        logger.info(f"Processing query: '{query}' for symbol: '{symbol}'")
        
        try:
            # Get comprehensive data for AI analysis
            data = None
            if symbol and validate_symbol(symbol):
                if self.vn_api.is_vn_stock(symbol):
                    logger.info(f"Getting comprehensive VN data for {symbol}")
                    # Get all available data for comprehensive analysis
                    tasks = [
                        self.vn_api.get_stock_data(symbol),
                        run_in_threadpool(self._safe_get_price_prediction, symbol),
                        run_in_threadpool(self._safe_get_risk_assessment, symbol, 50, "Trung hạn", 100000000),
                        run_in_threadpool(self._safe_get_investment_analysis, symbol, 50, "Trung hạn", 100000000),
                        self.get_detailed_stock_info(symbol),
                        run_in_threadpool(self._safe_get_ticker_news, symbol, 5)
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    data = {
                        "vn_stock_data": results[0] if not isinstance(results[0], Exception) else None,
                        "price_prediction": results[1] if not isinstance(results[1], Exception) else None,
                        "risk_assessment": results[2] if not isinstance(results[2], Exception) else None,
                        "investment_analysis": results[3] if not isinstance(results[3], Exception) else None,
                        "detailed_stock_info": results[4] if not isinstance(results[4], Exception) else None,
                        "ticker_news": results[5] if not isinstance(results[5], Exception) else None
                    }
                else:
                    logger.info(f"Getting comprehensive international data for {symbol}")
                    tasks = [
                        run_in_threadpool(self._safe_get_price_prediction, symbol),
                        run_in_threadpool(self._safe_get_investment_analysis, symbol, 50, "Trung hạn", 100000000),
                        run_in_threadpool(self._safe_get_risk_assessment, symbol, 50, "Trung hạn", 100000000),
                        run_in_threadpool(self._safe_get_ticker_news, symbol, 5)
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    data = {
                        "price_prediction": results[0] if not isinstance(results[0], Exception) else None,
                        "investment_analysis": results[1] if not isinstance(results[1], Exception) else None,
                        "risk_assessment": results[2] if not isinstance(results[2], Exception) else None,
                        "ticker_news": results[3] if not isinstance(results[3], Exception) else None
                    }
            
            # Use LLM to generate expert advice
            if self.llm_agent:
                try:
                    llm_response = await run_in_threadpool(
                        self.llm_agent.generate_response, 
                        f"CÂU HỎI: {query}\nMÃ CỔ PHIẾU: {symbol}\nDỮ LIỆU: {data}", 
                        "financial_advice"
                    )
                    ai_response = {
                        "expert_advice": llm_response.get('response', 'Không có phân tích từ AI.'),
                        "model_used": llm_response.get('model_used', 'unknown'),
                        "recommendations": ["Phân tích được tạo bởi AI", "Luôn DYOR trước khi đầu tư"]
                    }
                except Exception as e:
                    logger.error(f"LLM error: {e}")
                    ai_response = {
                        "expert_advice": f"Lỗi LLM AI: {str(e)}",
                        "model_used": "error",
                        "recommendations": ["Thử lại sau", "Kiểm tra API keys"]
                    }
            else:
                ai_response = {
                    "expert_advice": "LLM AI chưa được khởi tạo. Vui lòng nhập API keys.",
                    "model_used": "none",
                    "recommendations": ["Nhập Gemini/OpenAI/Llama API keys để sử dụng AI"]
                }

            response = {
                "query": query,
                "symbol": symbol,
                "response_type": "conversational",
                "expert_advice": ai_response.get("expert_advice", "Không có phân tích từ chuyên gia."),
                "model_used": ai_response.get("model_used", "none"),
                "recommendations": ai_response.get("recommendations", []),
                "data": data,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            logger.info(f"Successfully processed query for {symbol}")
            return response
            
        except Exception as e:
            logger.error(f"Critical error in process_query: {e}")
            return {"error": f"Lỗi nghiêm trọng khi xử lý truy vấn: {str(e)}"}

    
    # Helper methods với error handling
    def _safe_get_price_prediction(self, symbol: str):
        """Safely get price prediction with LSTM enhancement and validation"""
        try:
            # Use LSTM-enhanced prediction if available
            if hasattr(self.price_predictor, 'lstm_predictor') and self.price_predictor.lstm_predictor:
                result = self.price_predictor.predict_price_enhanced(symbol)
            else:
                result = self.price_predictor.predict_price(symbol)
            
            # CRITICAL FIX: Validate prediction results
            if not result.get('error'):
                current_price = result.get('current_price', 0)
                predicted_price = result.get('predicted_price', current_price)
                
                # Check if prediction is reasonable (within 50% of current price)
                if current_price > 0 and predicted_price > 0:
                    price_ratio = predicted_price / current_price
                    if price_ratio > 2.0 or price_ratio < 0.5:
                        print(f"⚠️ Unrealistic prediction detected for {symbol}: {predicted_price:.2f} vs {current_price:.2f}")
                        # Adjust to reasonable bounds
                        if price_ratio > 2.0:
                            result['predicted_price'] = round(current_price * 1.1, 2)  # Max 10% increase
                        else:
                            result['predicted_price'] = round(current_price * 0.9, 2)  # Max 10% decrease
                        
                        # Recalculate change percent
                        result['change_percent'] = round(((result['predicted_price'] - current_price) / current_price) * 100, 2)
                        result['validation_adjusted'] = True
                        print(f"🔧 Adjusted prediction to: {result['predicted_price']:.2f} VND")
            
            return result
        except Exception as e:
            return AgentErrorHandler.handle_prediction_error(symbol, e)
    
    def _safe_get_ticker_news(self, symbol: str, limit: int = 10):
        """Safely get ticker news"""
        try:
            return self.ticker_news.get_ticker_news(symbol, limit)
        except Exception as e:
            return AgentErrorHandler.handle_news_error(symbol, e)
    
    @handle_async_errors(default_return={"error": "Lỗi hệ thống khi lấy tin tức cổ phiếu"})
    async def get_ticker_news_enhanced(self, symbol: str, limit: int = 15):
        """Get enhanced ticker news with detailed stats"""
        try:
            result = await run_in_threadpool(self._safe_get_ticker_news, symbol, limit)
            return result
        except Exception as e:
            logger.error(f"Ticker news enhanced error: {e}")
            return {"error": f"Lỗi lấy tin tức cổ phiếu {symbol}: {str(e)}"}
    
    def _safe_get_investment_analysis(self, symbol: str, risk_tolerance: int = 50, time_horizon: str = "Trung hạn", investment_amount: int = 100000000):
        """Safely get investment analysis with profile parameters"""
        try:
            return self.investment_expert.analyze_stock(symbol, risk_tolerance, time_horizon, investment_amount)
        except Exception as e:
            return {"error": f"Lỗi phân tích đầu tư cho {symbol}: {str(e)}"}
    
    def _safe_get_risk_assessment(self, symbol: str, risk_tolerance: int = 50, time_horizon: str = "Trung hạn", investment_amount: int = 100000000):
        """Safely get risk assessment with profile parameters"""
        try:
            return self.risk_expert.assess_risk(symbol, risk_tolerance, time_horizon, investment_amount)
        except Exception as e:
            return AgentErrorHandler.handle_risk_error(symbol, e)
    
    def _safe_get_market_news(self):
        """Safely get market news"""
        try:
            return self.market_news.get_market_news()
        except Exception as e:
            logger.error(f"Market news error: {e}")
            return {"error": f"Lỗi lấy tin tức thị trường: {str(e)}"}
    
    def _safe_get_international_market_news(self):
        """Safely get international market news"""
        try:
            return self.international_news.get_international_news()
        except Exception as e:
            logger.error(f"International market news error: {e}")
            return {"error": f"Lỗi lấy tin tức thị trường quốc tế: {str(e)}"}
    
    @handle_async_errors(default_return={"error": "Lỗi hệ thống khi lấy tin tức quốc tế"})
    async def get_international_news(self):
        """Get international market news"""
        try:
            result = await run_in_threadpool(self._safe_get_international_market_news)
            return result
        except Exception as e:
            logger.error(f"International news async error: {e}")
            return {"error": f"Lỗi lấy tin tức thị trường quốc tế: {str(e)}"}
    
    def _get_error_fallback(self, task_name: str, symbol: str, error: Exception):
        """Get appropriate error fallback based on task type"""
        fallbacks = {
            'price_prediction': AgentErrorHandler.handle_prediction_error(symbol, error),
            'ticker_news': AgentErrorHandler.handle_news_error(symbol, error),
            'risk_assessment': AgentErrorHandler.handle_risk_error(symbol, error),
            'investment_analysis': {"error": f"Lỗi phân tích đầu tư: {str(error)}"},
            'vn_stock_data': {"error": f"Lỗi dữ liệu VN: {str(error)}"},
            'detailed_stock_info': {"error": f"Lỗi thông tin chi tiết: {str(error)}"}
        }
        return fallbacks.get(task_name, {"error": f"Lỗi {task_name}: {str(error)}"})
    
    def _is_valid_international_symbol(self, symbol: str) -> bool:
        """Kiểm tra xem có phải là mã international hợp lệ không"""
        if not symbol or len(symbol) < 1:
            return False
        
        symbol = symbol.upper().strip()
        
        # Loại bỏ các mã không hợp lệ
        invalid_patterns = [
            'X20', 'X21', 'X22', 'X23', 'X24', 'X25',  # Các mã lạ
            'TEST', 'DEMO', 'NULL', 'NONE'
        ]
        
        if symbol in invalid_patterns:
            return False
        
        # Kiểm tra pattern hợp lệ cho US stocks
        if len(symbol) >= 1 and len(symbol) <= 5:
            # Chấp nhận các mã có chữ và số
            if symbol.replace('.', '').replace('-', '').isalnum():
                return True
        
        return False
    
    async def get_detailed_stock_info(self, symbol: str):
        """Lấy thông tin chi tiết cổ phiếu từ stock_info module"""
        try:
            return await self.stock_info.get_detailed_stock_data(symbol)
        except Exception as e:
            logger.error(f"Error getting detailed stock info for {symbol}: {e}")
            return {"error": f"Lỗi lấy thông tin chi tiết: {str(e)}"}
    
    def display_stock_header(self, stock_data, current_time: str):
        """Display stock header - delegate to stock_info"""
        return self.stock_info.display_stock_header(stock_data, current_time)
    
    def display_detailed_metrics(self, detailed_data):
        """Display detailed metrics - delegate to stock_info"""
        return self.stock_info.display_detailed_metrics(detailed_data)
    
    def display_financial_ratios(self, detailed_data):
        """Display financial ratios - delegate to stock_info"""
        return self.stock_info.display_financial_ratios(detailed_data)
    
    def display_price_chart(self, price_history, symbol):
        """Display price chart - delegate to stock_info"""
        return self.stock_info.display_price_chart(price_history, symbol)
    
    def _get_risk_profile_name(self, risk_tolerance: int) -> str:
        """Get risk profile name from tolerance level"""
        if risk_tolerance <= 30:
            return "Thận trọng"
        elif risk_tolerance <= 70:
            return "Cân bằng"
        else:
            return "Mạo hiểm"
    
    @handle_async_errors(default_return={"error": "Lỗi dự đoán giá với kiến trúc"})
    async def predict_price_with_architecture(self, symbol: str, architecture: str = "ensemble_voting", timeframe: str = "1d"):
        """Dự đoán giá sử dụng kiến trúc được chọn"""
        if not symbol or not validate_symbol(symbol):
            return {"error": "Mã cổ phiếu không hợp lệ"}
        
        symbol = symbol.upper().strip()
        logger.info(f"Predicting price for {symbol} using {architecture} architecture")
        
        try:
            result = await self.architecture_manager.predict_price(symbol, architecture, timeframe)
            return result
        except Exception as e:
            logger.error(f"Architecture prediction error for {symbol}: {e}")
            return {"error": f"Lỗi dự đoán với kiến trúc {architecture}: {str(e)}"}
    
    @handle_async_errors(default_return={"error": "Lỗi so sánh kiến trúc"})
    async def compare_architectures(self, symbol: str, timeframe: str = "1d"):
        """So sánh kết quả của cả 3 kiến trúc"""
        if not symbol or not validate_symbol(symbol):
            return {"error": "Mã cổ phiếu không hợp lệ"}
        
        symbol = symbol.upper().strip()
        logger.info(f"Comparing architectures for {symbol}")
        
        try:
            result = await self.architecture_manager.compare_architectures(symbol, timeframe)
            return result
        except Exception as e:
            logger.error(f"Architecture comparison error for {symbol}: {e}")
            return {"error": f"Lỗi so sánh kiến trúc: {str(e)}"}
    
    def get_architecture_info(self):
        """Lấy thông tin về các kiến trúc"""
        return self.architecture_manager.get_architecture_info()
    
    def get_architecture_performance(self):
        """Lấy thống kê hiệu suất các kiến trúc"""
        return self.architecture_manager.get_performance_stats()
    

