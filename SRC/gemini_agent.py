import google.generativeai as genai
from openai import OpenAI
import os
import logging
from typing import Dict, Any, Optional, List
import asyncio
import json
import time
from datetime import datetime

try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    completion = None

logger = logging.getLogger(__name__)
class UnifiedAIAgent:
    def __init__(self, gemini_api_key: str = None, openai_api_key: str = None, preferred_model: str = "auto"):
        """
        Initialize AI Agent with Gemini AI, OpenAI, and Llama
        """
        self.available_models = {}
        self.current_model_name = None
        self.preferred_model = preferred_model
        self.openai_client = None
        self.model_capabilities = {
            'gemini': {
                'strengths': ['analysis', 'vietnamese', 'reasoning', 'financial_advice', 'prediction', 'technical_analysis', 'news_analysis', 'risk_assessment'],
                'speed': 'fast',
                'cost': 'free'
            },
            'openai': {
                'strengths': ['analysis', 'reasoning', 'financial_advice', 'prediction', 'technical_analysis', 'news_analysis', 'english'],
                'speed': 'medium',
                'cost': 'paid'
            },
            'llama': {
                'strengths': ['analysis', 'vietnamese', 'reasoning', 'financial_advice', 'local_processing'],
                'speed': 'medium',
                'cost': 'free_local'
            }
        }
        
        # Initialize AI models with user-provided API keys
        # No hardcoded or environment variables used
        
        # Initialize Llama (local)
        if LITELLM_AVAILABLE:
            try:
                # Test Llama availability
                test_response = completion(
                    model="ollama/llama3.1:8b",
                    messages=[{"role": "user", "content": "test"}],
                    temperature=0.1,
                    max_tokens=10,
                )
                if test_response and test_response.choices:
                    self.available_models['llama'] = 'ollama/llama3.1:8b'
                    logger.info("✅ Llama 3.1:8b initialized (local)")
            except Exception as e:
                logger.warning(f"⚠️ Llama not available: {str(e)}")
        else:
            logger.warning("⚠️ litellm not installed, Llama unavailable")
        
        # Initialize OpenAI
        if openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
                self.openai_api_key = openai_api_key
                
                # Try different OpenAI models
                openai_models = [
                    'gpt-4o',           # Latest GPT-4 Omni
                    'gpt-4-turbo',      # GPT-4 Turbo
                    'gpt-4',            # Standard GPT-4
                    'gpt-3.5-turbo'     # Fallback
                ]
                
                # Just set the first available model without testing to avoid API calls during init
                self.available_models['openai'] = openai_models[0]
                logger.info(f"✅ OpenAI initialized with model: {openai_models[0]}")
                        
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI: {str(e)}")
        
        # Initialize Gemini
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                
                # Try different model names (API v1beta compatible)
                model_names = [
                    'gemini-3-pro-preview',        # Flagship mới nhất, khả năng suy luận cao
                    'gemini-3-flash-preview',      # Tốc độ cao thế hệ 3

                        # --- GEMINI 2.5 SERIES (Tiêu chuẩn - Production Ready) ---
                    'gemini-2.5-pro',              # Bản chuẩn tốt nhất cho mọi tác vụ (thay thế 1.5 Pro)
                    'gemini-2.5-flash',            # Bản chuẩn tốc độ cao (thay thế 1.5 Flash)
                    'gemini-2.5-flash-lite',       # Chi phí cực thấp, tối ưu cho tác vụ đơn giản
                    'gemini-2.5-pro-001',          # Bản snapshot cụ thể (tránh thay đổi ngầm)
                    'gemini-2.5-flash-001',        # Bản snapshot cụ thể

                    # --- GEMINI 2.0 SERIES (Ổn định cũ) ---
                    'gemini-2.0-flash',            # Fallback tin cậy
                    'gemini-2.0-flash-exp',        # Bản thử nghiệm cũ (có thể vẫn hoạt động)

                    # --- LEGACY (Cũ - Hạn chế dùng cho dự án mới) ---
                    'gemini-1.5-pro',
                    'gemini-1.5-flash',
                    'gemini-1.5-flash-8b',         # Bản siêu nhẹ đời cũ

                    # --- SPECIALIZED (Nhúng & Hình ảnh) ---
                    'text-embedding-005',          # Model nhúng văn bản mới nhất (Semantic Search)
                    'imagen-3.0-generate-001',     # Tạo ảnh
                    'aqa'             # Legacy fallback
                ]
                
                model_initialized = False
                for model_name in model_names:
                    try:
                        model = genai.GenerativeModel(model_name)
                        # Initialize without testing to avoid quota usage
                        self.available_models['gemini'] = model
                        self.gemini_api_key = gemini_api_key
                        self.current_model_name = model_name
                        logger.info(f"✅ Gemini AI initialized with model: {model_name}")
                        model_initialized = True
                        break
                    except Exception as e:
                        error_msg = str(e).lower()
                        if 'quota' in error_msg or '429' in error_msg:
                            logger.warning(f"⚠️ Model {model_name} quota exceeded, trying next...")
                        elif '404' in error_msg or 'not found' in error_msg:
                            logger.warning(f"⚠️ Model {model_name} not found, trying next...")
                        else:
                            logger.warning(f"⚠️ Model {model_name} error: {e}")
                        continue
                
                if not model_initialized:
                    # If no model works, still allow offline mode
                    logger.warning("⚠️ No Gemini models available, will use offline mode")
                    self.available_models = {}
                    
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini: {str(e)}")
                # Don't set available_models if initialization failed
                self.available_models = {}
        
        # Set offline mode based on available models
        if not self.available_models:
            logger.warning("⚠️ No AI models available, system will run in offline mode")
            self.offline_mode = True
        else:
            self.offline_mode = False
            logger.info(f"✅ AI models available: {list(self.available_models.keys())}")
    
    def test_connection(self):
        """Test AI API connections without using quota"""
        results = {}
        
        # Test Gemini
        if 'gemini' in self.available_models:
            try:
                # Just check if model exists, don't make API call
                if hasattr(self, 'gemini_api_key') and self.gemini_api_key:
                    results['gemini'] = True
                    logger.info("✅ Gemini model available")
                else:
                    results['gemini'] = False
            except:
                results['gemini'] = False
        else:
            results['gemini'] = False
        
        # Test OpenAI
        if 'openai' in self.available_models:
            try:
                # Just check if client exists, don't make API call
                if hasattr(self, 'openai_client') and self.openai_client:
                    results['openai'] = True
                    logger.info("✅ OpenAI model available")
                else:
                    results['openai'] = False
            except:
                results['openai'] = False
        else:
            results['openai'] = False
        
        # Test Llama
        if 'llama' in self.available_models:
            try:
                # Just check if litellm is available
                if LITELLM_AVAILABLE:
                    results['llama'] = True
                    logger.info("✅ Llama model available")
                else:
                    results['llama'] = False
            except:
                results['llama'] = False
        else:
            results['llama'] = False
        
        return results
    
    def select_best_model(self, task_type: str) -> str:
        """
        Select the best available model for a specific task type based on user preference
        """
        # Respect user preference first
        if self.preferred_model == "gemini" and 'gemini' in self.available_models:
            return 'gemini'
        elif self.preferred_model == "openai" and 'openai' in self.available_models:
            return 'openai'
        elif self.preferred_model == "llama" and 'llama' in self.available_models:
            return 'llama'
        elif self.preferred_model == "auto":
            # Auto mode: prefer Gemini for Vietnamese content and free usage
            if 'gemini' in self.available_models:
                return 'gemini'
            # Fallback to OpenAI for English content
            if 'openai' in self.available_models:
                return 'openai'
            # Final fallback to Llama (local)
            if 'llama' in self.available_models:
                return 'llama'
        
        # Final fallback - use any available model (priority order)
        if 'gemini' in self.available_models:
            return 'gemini'
        if 'openai' in self.available_models:
            return 'openai'
        if 'llama' in self.available_models:
            return 'llama'
        
        raise ValueError("No AI models available")
    
    def generate_with_model(self, prompt: str, model_name: str, max_tokens: int = 2000) -> str:
        """
        Generate response using specified AI model
        """
        try:
            if model_name == 'gemini' and 'gemini' in self.available_models:
                response = self.available_models['gemini'].generate_content(prompt)
                return response.text
            
            elif model_name == 'openai' and 'openai' in self.available_models:
                if not hasattr(self, 'openai_client') or not self.openai_client:
                    raise ValueError("OpenAI client not initialized")
                openai_model = self.available_models['openai']
                response = self.openai_client.chat.completions.create(
                    model=openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content
            
            elif model_name == 'llama' and 'llama' in self.available_models:
                if not LITELLM_AVAILABLE:
                    raise ValueError("litellm not available for Llama")
                
                response = completion(
                    model="ollama/llama3.1:8b",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Bạn là trợ lý tài chính chuyên nghiệp. "
                                "Trả lời ngắn gọn, thực tế, bằng tiếng Việt. "
                                "Không suy đoán số liệu chính xác."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=min(max_tokens, 400),
                )
                return response.choices[0].message.content

            else:
                raise ValueError(f"Model {model_name} not available.")
                
        except Exception as e:
            logger.error(f"Error generating with {model_name}: {str(e)}")
            raise

    def generate_with_fallback(self, prompt: str, task_type: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Generate response with automatic fallback respecting user preference
        """
        # Check if we're already in offline mode
        if getattr(self, 'offline_mode', True) or not self.available_models:
            logger.info("📴 Using offline mode (no AI models available)")
            return self._generate_offline_fallback(prompt, task_type)
        
        try:
            # Use preferred model first
            preferred_model = self.select_best_model(task_type)
            response = self.generate_with_model(prompt, preferred_model, max_tokens)
            return {
                'response': response,
                'model_used': preferred_model,
                'success': True
            }
        except Exception as e:
            logger.error(f"Primary model {preferred_model} failed: {str(e)}")
            # Try fallback model
            try:
                fallback_models = [m for m in self.available_models.keys() if m != preferred_model]
                if fallback_models:
                    fallback_model = fallback_models[0]
                    response = self.generate_with_model(prompt, fallback_model, max_tokens)
                    return {
                        'response': response,
                        'model_used': fallback_model,
                        'success': True
                    }
            except Exception as e2:
                logger.error(f"Fallback model failed: {str(e2)}")
            
            # Use offline fallback when all AI models fail
            return self._generate_offline_fallback(prompt, task_type)
    
    def _generate_offline_fallback(self, prompt: str, task_type: str) -> Dict[str, Any]:
        """
        Generate offline fallback response when API quota is exhausted
        """
        try:
            # Extract key information from prompt
            if 'CÂU HỎI:' in prompt:
                question = prompt.split('CÂU HỎI:')[1].split('MÃ CỔ PHIẾU:')[0].strip()
            else:
                question = prompt[:200] + '...' if len(prompt) > 200 else prompt
            
            # Generate contextual offline response based on task type
            if task_type == 'financial_advice':
                response = self._generate_financial_advice_fallback(question)
            elif task_type == 'general_query':
                response = self._generate_general_fallback(question)
            else:
                response = self._generate_default_fallback(question)
            
            return {
                'response': response,
                'model_used': 'offline_fallback',
                'success': True,
                'quota_exceeded': True
            }
        except Exception as e:
            return {
                'response': f'Offline fallback failed: {str(e)}',
                'model_used': 'offline_fallback',
                'success': False,
                'error': str(e)
            }
    
    def _generate_financial_advice_fallback(self, question: str) -> str:
        """
        Generate financial advice fallback when API quota exceeded
        """
        return f"""
PHÂN TÍCH CHUYÊN SÂU:
Do Gemini API đã hết quota, hệ thống chuyển sang chế độ offline. Đây là phân tích chuyên sâu dựa trên nguyên tắc đầu tư thực tiễn và kinh nghiệm thị trường:

📊 **Phân tích kỹ thuật:**
- Xem xét xu hướng giá 20-50 phiên gần nhất, xác định vùng hỗ trợ/kháng cự, khối lượng giao dịch, chỉ báo RSI/MACD.
- Đánh giá dòng tiền, mức độ biến động, các tín hiệu đảo chiều.

💰 **Phân tích cơ bản:**
- Đọc báo cáo tài chính, chú ý doanh thu, lợi nhuận, biên lợi nhuận, dòng tiền hoạt động.
- So sánh P/E, P/B với trung bình ngành, xem xét tăng trưởng EPS, ROE, ROA.
- Đánh giá ban lãnh đạo, chiến lược phát triển, vị thế cạnh tranh.

📰 **Tin tức & sự kiện:**
- Theo dõi các tin tức ảnh hưởng đến ngành, cổ phiếu, chính sách vĩ mô, lãi suất, tỷ giá.
- Đánh giá tác động của các sự kiện đặc biệt (chia cổ tức, phát hành thêm, M&A).

KẾT LUẬN & KHUYẾN NGHỊ:
- Chỉ đầu tư khi hiểu rõ doanh nghiệp, ngành và xu hướng thị trường.
- Đặt mục tiêu lợi nhuận, điểm cắt lỗ rõ ràng.
- Đa dạng hóa danh mục, không dồn vốn vào một mã.
- Luôn cập nhật thông tin, điều chỉnh chiến lược khi có biến động lớn.

HÀNH ĐỘNG CỤ THỂ:
- Đọc kỹ báo cáo tài chính quý gần nhất.
- Lập bảng so sánh các mã cùng ngành.
- Theo dõi diễn biến thị trường hàng ngày.
- Đặt lệnh stop-loss, take-profit cho từng vị thế.
- Tham khảo ý kiến chuyên gia, cộng đồng đầu tư.

CẢNH BÁO RỦI RO:
⚠️ **QUAN TRỌNG:** Đây là phân tích offline cơ bản do hết quota API. 
Không nên dựa vào đây để đưa ra quyết định đầu tư quan trọng.
Hãy đợi API reset hoặc tham khảo chuyên gia tài chính.
"""
    
    def _generate_general_fallback(self, question: str) -> str:
        """
        Generate general query fallback when API quota exceeded
        """
        return f"""
📈 **PHÂN TÍCH OFFLINE:**

Do Gemini API đã hết quota, tôi không thể phân tích chi tiết câu hỏi của bạn lúc này.

**Câu hỏi của bạn:** {question}

💡 **Gợi ý đầu tư thực tiễn:**
- Nghiên cứu kỹ báo cáo tài chính, so sánh với các doanh nghiệp cùng ngành.
- Đa dạng hóa danh mục để giảm rủi ro, không đầu tư quá 20% vốn vào một mã.
- Đặt mục tiêu lợi nhuận, điểm cắt lỗ rõ ràng cho từng vị thế.
- Theo dõi tin tức, chính sách vĩ mô, các yếu tố ảnh hưởng đến thị trường.
- Tham khảo ý kiến chuyên gia, cộng đồng đầu tư uy tín.
- Luôn kiểm tra lại chiến lược khi thị trường biến động mạnh.

⚠️ **LƯU Ý:** Để nhận được phân tích chi tiết và cá nhân hóa, 
vui lòng thử lại sau khi API quota được reset (thường là 24h).
"""
    
    def _generate_default_fallback(self, question: str) -> str:
        """
        Generate default fallback response
        """
        return f"""
🤖 **HỆ THỐNG OFFLINE:**

Xin lỗi, Gemini API đã hết quota nên tôi không thể phân tích chi tiết lúc này.

**Câu hỏi:** {question}

**Khuyến nghị thực tế:**
- Thử lại sau vài giờ khi quota reset.
- Đọc báo cáo tài chính, phân tích kỹ thuật cơ bản.
- Tham khảo các nguồn thông tin tài chính uy tín, cộng đồng đầu tư.
- Lập kế hoạch đầu tư rõ ràng, kiểm soát rủi ro.
- Liên hệ chuyên gia tài chính nếu cần tư vấn gấp.

⏰ **Quota thường reset sau 24 giờ**
"""
    
    def generate_enhanced_advice(self, context: dict):
        """Generate enhanced advice with comprehensive system data"""
        query = context.get('query', '')
        symbol = context.get('symbol', '')
        system_data = context.get('system_data', {})
        query_type = context.get('query_type', 'general_inquiry')
        
        # Build enhanced context with all system data
        enhanced_context = f"""
Bạn là chuyên gia tài chính AI hàng đầu với khả năng phân tích toàn diện hệ thống trading.

CÂU HỎI: {query}
MÃ CỔ PHIẾU: {symbol if symbol else 'Không có'}
LOẠI TRUY VẤN: {query_type}

DỮ LIỆU HỆ THỐNG TOÀN DIỆN:
{self._format_comprehensive_data(system_data)}

YÊU CẦU PHÂN TÍCH:
1. 📊 PHÂN TÍCH DỮ LIỆU: Sử dụng tất cả dữ liệu có sẵn từ hệ thống, phân tích sâu về xu hướng, chỉ số tài chính, dòng tiền, tin tức, dự đoán giá, rủi ro.
2. 🎯 PHÂN TÍCH THEO LOẠI TRUY VẤN: Tập trung vào {query_type}, đưa ra nhận định thực tiễn, so sánh với các mã cùng ngành, đánh giá triển vọng.
3. 💡 KHUYẾN NGHỊ CỤ THỂ: Đề xuất hành động cụ thể, chiến lược đầu tư, điểm mua/bán, quản trị rủi ro, phù hợp với từng loại nhà đầu tư (ngắn hạn, dài hạn).
4. ⚠️ RỦI RO & LƯU Ý: Đánh giá toàn diện các rủi ro, khuyến nghị kiểm soát vốn, đa dạng hóa danh mục, cảnh báo các yếu tố bất thường.

TRẢ LỜI THEO FORMAT:
PHÂN TÍCH CHUYÊN SÂU:
[Sử dụng dữ liệu cụ thể từ hệ thống, phân tích chi tiết từng yếu tố]

KẾT LUẬN & KHUYẾN NGHỊ:
[Kết luận dựa trên phân tích dữ liệu, đề xuất hành động thực tế]

HÀNH ĐỘNG CỤ THỂ:
- [Danh sách hành động cụ thể, có thể áp dụng ngay]

CẢNH BÁO RỦI RO:
[Rủi ro dựa trên dữ liệu thực tế, khuyến nghị kiểm soát]
"""
        
        try:
            result = self.generate_with_fallback(enhanced_context, 'financial_advice', max_tokens=3000)
            
            if result['success']:
                parsed_response = self._parse_response(result['response'])
                parsed_response['expert_advice'] += f"\n\n🤖 **AI Model:** {result['model_used']}"
                return parsed_response
            else:
                return self._generate_enhanced_offline_response(query, symbol, system_data, query_type)
                
        except Exception as e:
            logger.error(f"Enhanced advice generation failed: {e}")
            return self._generate_enhanced_offline_response(query, symbol, system_data, query_type)
    
    def _format_comprehensive_data(self, system_data: dict) -> str:
        """Format comprehensive system data for AI analysis"""
        if not system_data:
            return "Không có dữ liệu hệ thống"
        
        formatted = []
        
        # Market Overview
        if 'market_overview' in system_data:
            market = system_data['market_overview']
            formatted.append("📈 TỔNG QUAN THỊ TRƯỜNG:")
            if 'vietnam_market' in market:
                vn_market = market['vietnam_market']
                if 'vn_index' in vn_market:
                    vn_idx = vn_market['vn_index']
                    formatted.append(f"- VN-Index: {vn_idx.get('value', 'N/A')} ({vn_idx.get('change_percent', 0):+.2f}%)")
        
        # Stock Data
        if 'stock_data' in system_data and system_data['stock_data']:
            stock = system_data['stock_data']
            formatted.append(f"\n📊 THÔNG TIN CỔ PHIẾU {system_data.get('symbol', '')}:")
            formatted.append(f"- Giá: {stock.price:,} VND ({stock.change_percent:+.2f}%)")
            formatted.append(f"- Khối lượng: {stock.volume:,}")
            formatted.append(f"- P/E: {stock.pe_ratio}, P/B: {stock.pb_ratio}")
            formatted.append(f"- Vốn hóa: {stock.market_cap:,} tỷ VND")
        
        # Price Prediction
        if 'price_prediction' in system_data and system_data['price_prediction']:
            pred = system_data['price_prediction']
            formatted.append(f"\n🔮 DỰ ĐOÁN GIÁ:")
            formatted.append(f"- Giá dự đoán: {pred.get('predicted_price', 'N/A')}")
            formatted.append(f"- Xu hướng: {pred.get('trend', 'N/A')}")
            formatted.append(f"- Độ tin cậy: {pred.get('confidence', 'N/A')}%")
            
            # Multi-timeframe predictions
            if 'predictions' in pred:
                predictions = pred['predictions']
                for timeframe, data in predictions.items():
                    if data:
                        formatted.append(f"- {timeframe}: {list(data.keys())[:3]}")
        
        # Investment Analysis
        if 'investment_analysis' in system_data and system_data['investment_analysis']:
            inv = system_data['investment_analysis']
            formatted.append(f"\n💼 PHÂN TÍCH ĐẦU TƯ:")
            formatted.append(f"- Khuyến nghị: {inv.get('recommendation', 'N/A')}")
            formatted.append(f"- Điểm số: {inv.get('score', 'N/A')}/100")
            formatted.append(f"- Lý do: {inv.get('reason', 'N/A')}")
        
        # Risk Assessment
        if 'risk_assessment' in system_data and system_data['risk_assessment']:
            risk = system_data['risk_assessment']
            formatted.append(f"\n⚠️ ĐÁNH GIÁ RỦI RO:")
            formatted.append(f"- Mức rủi ro: {risk.get('risk_level', 'N/A')}")
            formatted.append(f"- Volatility: {risk.get('volatility', 'N/A')}%")
            formatted.append(f"- Beta: {risk.get('beta', 'N/A')}")
        
        # News Analysis
        if 'ticker_news' in system_data and system_data['ticker_news']:
            news = system_data['ticker_news']
            formatted.append(f"\n📰 TIN TỨC:")
            formatted.append(f"- Số lượng tin: {news.get('news_count', 0)}")
            if 'news_sentiment' in news:
                formatted.append(f"- Sentiment: {news['news_sentiment']}")
        
        # Available Symbols
        if 'available_symbols' in system_data:
            symbols = system_data['available_symbols']
            if symbols:
                symbol_list = [s.get('symbol', '') for s in symbols[:10]]
                formatted.append(f"\n📋 CỔ PHIẾU KHẢ DỤNG: {', '.join(symbol_list)}")
        
        # Analysis History
        if 'analysis_history' in system_data and system_data['analysis_history']:
            history = system_data['analysis_history']
            formatted.append(f"\n📊 LỊCH SỬ PHÂN TÍCH: {len(history)} phân tích gần đây")
        
        # System Stats
        if 'system_stats' in system_data and system_data['system_stats']:
            stats = system_data['system_stats']
            formatted.append(f"\n📈 THỐNG KÊ HỆ THỐNG:")
            formatted.append(f"- Tổng phân tích: {stats.get('total_analyses', 0)}")
            if 'top_symbols' in stats and stats['top_symbols']:
                top_symbol = stats['top_symbols'][0]
                formatted.append(f"- Phổ biến nhất: {top_symbol.get('symbol', 'N/A')}")
        
        return "\n".join(formatted) if formatted else "Dữ liệu hệ thống không đầy đủ"
    
    def _generate_enhanced_offline_response(self, query: str, symbol: str, system_data: dict, query_type: str) -> dict:
        """Generate enhanced offline response with system data"""
        
        # Analyze available data
        available_data = []
        if system_data.get('stock_data'):
            available_data.append("dữ liệu cổ phiếu")
        if system_data.get('price_prediction'):
            available_data.append("dự đoán giá")
        if system_data.get('investment_analysis'):
            available_data.append("phân tích đầu tư")
        if system_data.get('risk_assessment'):
            available_data.append("đánh giá rủi ro")
        if system_data.get('ticker_news'):
            available_data.append("tin tức")
        
        data_summary = ", ".join(available_data) if available_data else "dữ liệu cơ bản"
        
        # Generate response based on query type and available data
        if query_type == 'price_prediction' and system_data.get('price_prediction'):
            pred = system_data['price_prediction']
            advice = f"""📈 DỰ ĐOÁN GIÁ CHO {symbol}:

Dựa trên {data_summary} có sẵn:
- Giá dự đoán: {pred.get('predicted_price', 'N/A')} VND
- Xu hướng: {pred.get('trend', 'N/A')}
- Độ tin cậy: {pred.get('confidence', 50):.1f}%

⚠️ Đây là phân tích offline do hết quota API."""
        
        elif query_type == 'investment_advice' and system_data.get('investment_analysis'):
            inv = system_data['investment_analysis']
            advice = f"""💼 KHUYẾN NGHỊ ĐẦU TƯ CHO {symbol}:

Dựa trên {data_summary} có sẵn:
- Khuyến nghị: {inv.get('recommendation', 'HOLD')}
- Điểm số: {inv.get('score', 50)}/100
- Lý do: {inv.get('reason', 'Phân tích cơ bản')}

⚠️ Đây là phân tích offline do hết quota API."""
        
        elif query_type == 'risk_assessment' and system_data.get('risk_assessment'):
            risk = system_data['risk_assessment']
            advice = f"""⚠️ ĐÁNH GIÁ RỦI RO CHO {symbol}:

Dựa trên {data_summary} có sẵn:
- Mức rủi ro: {risk.get('risk_level', 'MEDIUM')}
- Volatility: {risk.get('volatility', 25):.1f}%
- Beta: {risk.get('beta', 1.0):.2f}

⚠️ Đây là phân tích offline do hết quota API."""
        
        else:
            advice = f"""📊 PHÂN TÍCH OFFLINE:

Câu hỏi: {query}
Mã cổ phiếu: {symbol if symbol else 'Không có'}
Loại truy vấn: {query_type}

Dữ liệu có sẵn: {data_summary}

💡 Khuyến nghị chung:
- Nghiên cứu kỹ báo cáo tài chính
- Theo dõi tin tức ngành
- Đa dạng hóa danh mục
- Chỉ đầu tư tiền nhàn rỗi

⚠️ Đây là phân tích offline do hết quota API."""
        
        return {
            "expert_advice": advice,
            "recommendations": [
                "Đợi quota API reset để có phân tích chi tiết",
                "Tham khảo nhiều nguồn thông tin",
                "Liên hệ chuyên gia tài chính",
                "Chỉ đầu tư số tiền có thể chấp nhận mất"
            ]
        }
    
    def generate_expert_advice(self, query: str, symbol: str = None, data: dict = None):
        """Backward compatibility method"""
        # Convert to enhanced context format
        context = {
            'query': query,
            'symbol': symbol or '',
            'system_data': data or {},
            'query_type': self.detect_query_type(query)
        }
        return self.generate_enhanced_advice(context)
    
    def _parse_response(self, response_text: str):
        """Parse enhanced Gemini response"""
        try:
            # Parse different sections
            sections = {
                'analysis': '',
                'conclusion': '',
                'actions': [],
                'risks': ''
            }
            
            # Split by sections
            if "PHÂN TÍCH CHUYÊN SÂU:" in response_text:
                parts = response_text.split("PHÂN TÍCH CHUYÊN SÂU:")
                if len(parts) > 1:
                    remaining = parts[1]
                    
                    # Extract analysis
                    if "KẾT LUẬN & KHUYẾN NGHỊ:" in remaining:
                        analysis_part = remaining.split("KẾT LUẬN & KHUYẾN NGHỊ:")[0].strip()
                        sections['analysis'] = analysis_part
                        remaining = remaining.split("KẾT LUẬN & KHUYẾN NGHỊ:")[1]
                    
                    # Extract conclusion
                    if "HÀNH ĐỘNG CỤ THỂ:" in remaining:
                        conclusion_part = remaining.split("HÀNH ĐỘNG CỤ THỂ:")[0].strip()
                        sections['conclusion'] = conclusion_part
                        remaining = remaining.split("HÀNH ĐỘNG CỤ THỂ:")[1]
                    
                    # Extract actions
                    if "CẢNH BÁO RỦI RO:" in remaining:
                        actions_part = remaining.split("CẢNH BÁO RỦI RO:")[0].strip()
                        sections['risks'] = remaining.split("CẢNH BÁO RỦI RO:")[1].strip()
                    else:
                        actions_part = remaining.strip()
                    
                    # Parse actions list
                    for line in actions_part.split('\n'):
                        line = line.strip()
                        if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                            sections['actions'].append(line[1:].strip())
                        elif line and len(line) > 15 and not line.startswith('CẢNH BÁO'):
                            sections['actions'].append(line)
            
            # Format comprehensive response
            expert_advice = f"""
📊 **PHÂN TÍCH CHUYÊN SÂU:**
{sections['analysis']}

🎯 **KẾT LUẬN & KHUYẾN NGHỊ:**
{sections['conclusion']}

⚠️ **CẢNH BÁO RỦI RO:**
{sections['risks'] if sections['risks'] else 'Luôn có rủi ro trong đầu tư. Chỉ đầu tư số tiền có thể chấp nhận mất.'}
""".strip()
            
            return {
                "expert_advice": expert_advice,
                "recommendations": sections['actions'][:5] if sections['actions'] else [
                    "Nghiên cứu kỹ báo cáo tài chính",
                    "Theo dõi tin tức ngành", 
                    "Đặt lệnh stop-loss",
                    "Đa dạng hóa danh mục",
                    "Chỉ đầu tư tiền nhàn rỗi"
                ]
            }
                
        except Exception as e:
            # Fallback parsing
            return {
                "expert_advice": f"📈 **PHÂN TÍCH:**\n{response_text}\n\n⚠️ **LƯU Ý:** Đây chỉ là tham khảo, không phải lời khuyên đầu tư tuyệt đối.",
                "recommendations": [
                    "Nghiên cứu thêm từ nhiều nguồn",
                    "Tham khảo chuyên gia tài chính",
                    "Đánh giá khả năng tài chính cá nhân",
                    "Chỉ đầu tư số tiền có thể chấp nhận mất"
                ]
            }
    

    
    def generate_general_response(self, query: str) -> dict:
        """Generate response for general questions using best available AI model"""
        try:
            # Enhanced context for general financial questions
            context = f"""
Bạn là một chuyên gia tài chính và đầu tư hàng đầu tại Việt Nam với 20+ năm kinh nghiệm.
Bạn có thể trả lời mọi câu hỏi về:
- Thị trường chứng khoán Việt Nam và quốc tế
- Phân tích kỹ thuật và cơ bản, đánh giá xu hướng, dòng tiền, chỉ số tài chính
- Chiến lược đầu tư, quản lý rủi ro, đa dạng hóa danh mục, điểm mua/bán
- Kinh tế vĩ mô, vi mô, tác động chính sách, tin tức thị trường
- Các sản phẩm tài chính (cổ phiếu, trái phiếu, quỹ, forex)
- Lập kế hoạch tài chính cá nhân, kiểm soát vốn, quản trị tâm lý đầu tư
- Thuế, pháp lý đầu tư, quy định mới nhất
- Tâm lý học đầu tư, các sai lầm phổ biến, cách kiểm soát cảm xúc
- Fintech, công nghệ tài chính, ứng dụng AI trong đầu tư

CÂU HỎI: {query}

HÃY TRẢ LỜI:
1. 📚 KIẾN THỨC CƠ BẢN: Giải thích khái niệm/vấn đề, liên hệ thực tiễn Việt Nam.
2. 🎯 PHÂN TÍCH THỰC TẾ: Áp dụng vào thị trường VN, so sánh với các trường hợp thực tế, đưa ra nhận định sâu sắc.
3. 💡 KHUYẾN NGHỊ: Lời khuyên cụ thể, chiến lược đầu tư, hành động thiết thực cho từng loại nhà đầu tư.
4. ⚠️ LƯU Ý: Rủi ro, các yếu tố cần chú ý, cách kiểm soát vốn, tránh các sai lầm phổ biến.

Trả lời bằng tiếng Việt, chuyên nghiệp, chi tiết, thực tiễn, dễ hiểu, có thể áp dụng ngay.
"""
            result = self.generate_with_fallback(context, 'general_query', max_tokens=3000)
            
            if result['success']:
                if result.get('quota_exceeded'):
                    # Quota exceeded, return offline response
                    return {
                        "expert_advice": f"📈 **PHÂN TÍCH OFFLINE:**\n{result['response']}\n\n🤖 **AI Model:** Offline Fallback (Quota Exceeded)\n\n⚠️ **LƯU Ý:** Đây là phản hồi offline do hết quota API.",
                        "recommendations": [
                            "Đợi quota reset (24h) để có phân tích chi tiết",
                            "Tham khảo các nguồn tin tức tài chính", 
                            "Liên hệ chuyên gia nếu cần tư vấn gấp",
                            "Chỉ đầu tư số tiền có thể chấp nhận mất"
                        ]
                    }
                else:
                    # Normal AI response
                    return {
                        "expert_advice": f"📈 **PHÂN TÍCH CHUYÊN GIA:**\n{result['response']}\n\n🤖 **AI Model:** {result['model_used']}\n\n⚠️ **LƯU Ý:** Đây là thông tin tham khảo, không phải lời khuyên đầu tư tuyệt đối.",
                        "recommendations": [
                            "Nghiên cứu thêm từ nhiều nguồn",
                            "Tham khảo chuyên gia tài chính", 
                            "Đánh giá khả năng tài chính cá nhân",
                            "Chỉ đầu tư số tiền có thể chấp nhận mất"
                        ]
                    }
            else:
                return self._get_fallback_response(query)
                
        except Exception as e:
            logger.error(f"Error in generate_general_response: {str(e)}")
            return self._get_fallback_response(query)
    
    def _get_fallback_response(self, query: str) -> dict:
        """Fallback response when Gemini fails"""
        return {
            "expert_advice": f"📈 **VỀ CÂU HỎI: {query}**\n\nXin lỗi, tôi không thể xử lý câu hỏi này lúc này. Vui lòng thử lại sau hoặc đặt câu hỏi khác.\n\n⚠️ **GỢI Ý:**\n- Kiểm tra kết nối internet\n- Thử đặt câu hỏi ngắn gọn hơn\n- Liên hệ hỗ trợ nếu vấn đề tiếp tục",
            "recommendations": [
                "Thử đặt câu hỏi khác",
                "Kiểm tra kết nối mạng",
                "Liên hệ hỗ trợ kỹ thuật"
            ]
        }
    
    def detect_query_type(self, query: str) -> str:
        """Detect if query is stock-specific or general"""
        query_lower = query.lower()
        
        # Stock symbols patterns
        stock_patterns = ['vcb', 'bid', 'ctg', 'vic', 'vhm', 'hpg', 'fpt', 'msn', 'mwg', 'gas', 'plx']
        
        # Check for stock symbols
        for pattern in stock_patterns:
            if pattern in query_lower:
                return "stock_specific"
        
        # Check for stock-related keywords
        stock_keywords = ['cổ phiếu', 'mã', 'ticker', 'stock', 'share']
        if any(keyword in query_lower for keyword in stock_keywords):
            return "stock_specific"
        
        return "general"
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get comprehensive API status information"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'available_models': list(self.available_models.keys()),
            'model_count': len(self.available_models),
            'capabilities': self.model_capabilities,
            'api_keys_configured': {}
        }
        
        # Check API key configuration
        status['api_keys_configured']['gemini'] = hasattr(self, 'gemini_api_key') and bool(self.gemini_api_key)
        
        # Test connections
        try:
            connection_results = self.test_connection()
            status['connection_status'] = connection_results
            status['healthy_models'] = [model for model, healthy in connection_results.items() if healthy]
        except Exception as e:
            status['connection_status'] = {'error': str(e)}
            status['healthy_models'] = []
        
        return status
    
    def update_api_key(self, provider: str, api_key: str) -> Dict[str, Any]:
        """Dynamically update Gemini API key"""
        try:
            if provider.lower() == 'gemini':
                genai.configure(api_key=api_key)
                
                # Try different model names (API v1beta compatible)
                model_names = [
                    'gemini-1.5-pro-latest',        # Latest stable pro
                    'gemini-1.5-flash-latest',      # Latest stable flash
                    'gemini-1.5-pro',               # Pro version
                    'gemini-1.5-flash',             # Flash version
                    'gemini-pro',                    # Legacy pro
                    'gemini-1.0-pro-latest',        # Legacy latest
                    'gemini-1.0-pro'                # Legacy fallback
                ]
                
                for model_name in model_names:
                    try:
                        model = genai.GenerativeModel(model_name)
                        # Initialize without testing to avoid quota usage
                        self.available_models['gemini'] = model
                        self.gemini_api_key = api_key
                        self.current_model_name = model_name
                        logger.info(f"✅ Gemini API key updated with model: {model_name}")
                        return {'success': True, 'message': f'Gemini API key updated with model: {model_name}'}
                    except Exception as e:
                        error_msg = str(e).lower()
                        if '404' in error_msg or 'not found' in error_msg:
                            logger.warning(f"⚠️ Model {model_name} not found, trying next...")
                        else:
                            logger.warning(f"⚠️ Model {model_name} error: {e}")
                        continue
                else:
                    # If no model works, return error
                    return {'success': False, 'message': 'No available Gemini models found'}
            else:
                return {'success': False, 'message': f'Only Gemini provider is supported. Got: {provider}'}
                
        except Exception as e:
            logger.error(f"❌ Failed to update {provider} API key: {str(e)}")
            return {'success': False, 'message': f'Failed to update {provider} API key: {str(e)}'}
    
    def get_model_recommendations(self, task_type: str) -> Dict[str, Any]:
        """Get model recommendations for specific task types"""
        try:
            primary_model = self.select_best_model(task_type)
        except ValueError:
            primary_model = None
            
        recommendations = {
            'task_type': task_type,
            'primary_model': primary_model,
            'preferred_model': self.preferred_model,
            'available_alternatives': [],
            'reasoning': ''
        }
        
        # Get all available models except primary
        if primary_model:
            alternatives = [model for model in self.available_models.keys() if model != primary_model]
            recommendations['available_alternatives'] = alternatives
        
        # Add reasoning based on preference and task type
        if self.preferred_model == "gemini":
            recommendations['reasoning'] = 'User prefers Gemini AI for Vietnamese content and free usage'
        elif self.preferred_model == "openai":
            recommendations['reasoning'] = 'User prefers OpenAI GPT for high-quality analysis'
        else:
            task_reasoning = {
                'financial_advice': 'Auto-selecting best model for Vietnamese financial analysis',
                'price_prediction': 'Auto-selecting best model for technical analysis and predictions',
                'risk_assessment': 'Auto-selecting best model for risk calculation and assessment',
                'news_analysis': 'Auto-selecting best model for sentiment analysis',
                'market_analysis': 'Auto-selecting best model for market reasoning',
                'investment_analysis': 'Auto-selecting best model for investment metrics',
                'general_query': 'Auto-selecting best model for general queries'
            }
            recommendations['reasoning'] = task_reasoning.get(task_type, 'Auto model selection based on availability')
        
        return recommendations
    
    async def generate_async(self, prompt: str, task_type: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """Asynchronous generation with fallback support"""
        try:
            # Run the synchronous method in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.generate_with_fallback, 
                prompt, 
                task_type, 
                max_tokens
            )
            return result
        except Exception as e:
            logger.error(f"Async generation failed: {str(e)}")
            return {
                'response': f'Async generation error: {str(e)}',
                'model_used': None,
                'success': False,
                'error': str(e)
            }
    
    def batch_generate(self, prompts: List[Dict[str, Any]], max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts with concurrency control"""
        async def process_batch():
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_single(prompt_data):
                async with semaphore:
                    prompt = prompt_data.get('prompt', '')
                    task_type = prompt_data.get('task_type', 'general_query')
                    max_tokens = prompt_data.get('max_tokens', 1000)
                    
                    result = await self.generate_async(prompt, task_type, max_tokens)
                    result['original_data'] = prompt_data
                    return result
            
            tasks = [process_single(prompt_data) for prompt_data in prompts]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new event loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, process_batch())
                    return future.result()
            else:
                return asyncio.run(process_batch())
        except Exception as e:
            logger.error(f"Batch generation failed: {str(e)}")
            return [{'success': False, 'error': str(e)} for _ in prompts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            'available_models': list(self.available_models.keys()),
            'current_model': self.current_model_name,
            'model_count': len(self.available_models),
            'is_active': len(self.available_models) > 0
        }

# Backward compatibility alias
GeminiAgent = UnifiedAIAgent