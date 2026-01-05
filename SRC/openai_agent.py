import openai as _openai
import logging
from typing import Dict, Any, Optional, List
import asyncio
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class OpenAIAgent:
    def __init__(self, openai_api_key: str = None):
        """
        Initialize OpenAI Agent with GPT models
        Enhanced setup similar to Gemini with smart model testing
        """
        self.available_models = {}
        self.current_model_name = None
        self.client = None
        self.last_error: Optional[str] = None
        self.openai_api_key = openai_api_key
        self.offline_mode = True  # Default to offline until proven otherwise
        self.model_capabilities = {
            'gpt-4o': {
                'strengths': ['analysis', 'reasoning', 'financial_advice', 'prediction', 'technical_analysis', 'multilingual'],
                'speed': 'medium',
                'cost': 'high',
                'tier': 'premium'
            },
            'gpt-4o-mini': {
                'strengths': ['analysis', 'reasoning', 'fast_response', 'cost_effective'],
                'speed': 'fast',
                'cost': 'low',
                'tier': 'standard'
            },
            'gpt-4-turbo': {
                'strengths': ['analysis', 'reasoning', 'long_context', 'technical_analysis'],
                'speed': 'medium',
                'cost': 'high',
                'tier': 'premium'
            },
            'gpt-4': {
                'strengths': ['analysis', 'reasoning', 'financial_advice'],
                'speed': 'slow',
                'cost': 'very_high',
                'tier': 'premium'
            },
            'gpt-3.5-turbo': {
                'strengths': ['general_purpose', 'fast_response', 'cost_effective'],
                'speed': 'very_fast',
                'cost': 'very_low',
                'tier': 'basic'
            },
            'gpt-4-turbo-preview': {
                'strengths': ['analysis', 'reasoning', 'preview_features'],
                'speed': 'medium',
                'cost': 'high',
                'tier': 'premium'
            }
        }

        if self.openai_api_key:
            try:
                self.client = _openai.OpenAI(api_key=self.openai_api_key)

                # Try different GPT models in order of preference (similar to Gemini)
                model_names = [
                    'gpt-4o',                # Latest GPT-4 Omni (best quality)
                    'gpt-4o-mini',           # Cost-effective GPT-4 (RECOMMENDED)
                    'gpt-4-turbo',           # GPT-4 Turbo
                    'gpt-4-turbo-preview',   # Preview version
                    'gpt-4',                 # Standard GPT-4 (expensive)
                    'gpt-3.5-turbo',         # Fallback option (cheap)
                    'gpt-3.5-turbo-16k'      # Extended context fallback
                ]

                model_initialized = False
                for model_name in model_names:
                    try:
                        # Skip test for expensive models to avoid quota usage (like Gemini Pro)
                        if self._is_expensive_model(model_name):
                            # Just initialize without testing to avoid costs
                            self.available_models['openai'] = model_name
                            self.current_model_name = model_name
                            self.offline_mode = False
                            logger.info(f"✅ OpenAI initialized with model: {model_name} (no test - premium model)")
                            model_initialized = True
                            break
                        else:
                            # Test cheaper models with minimal request
                            test_response = self.client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": "Hi"}],
                                max_tokens=5,
                                timeout=10
                            )

                            if test_response and test_response.choices:
                                self.available_models['openai'] = model_name
                                self.current_model_name = model_name
                                self.offline_mode = False
                                logger.info(f"✅ OpenAI initialized with model: {model_name}")
                                model_initialized = True
                                break

                    except Exception as e:
                        error_msg = str(e).lower()
                        if 'quota' in error_msg or 'rate limit' in error_msg or '429' in error_msg:
                            logger.warning(f"⚠️ Model {model_name} quota/rate limit exceeded, trying next...")
                        elif 'insufficient_quota' in error_msg or 'billing' in error_msg:
                            logger.warning(f"⚠️ Model {model_name} billing issue, trying next...")
                        elif 'invalid' in error_msg and 'api' in error_msg:
                            logger.warning(f"⚠️ Invalid API key for {model_name}")
                        elif 'model not found' in error_msg or 'does not exist' in error_msg:
                            logger.warning(f"⚠️ Model {model_name} not available, trying next...")
                        else:
                            logger.warning(f"⚠️ Model {model_name} error: {e}")
                        continue

                if not model_initialized:
                    msg = "No OpenAI models available, will use offline mode"
                    logger.warning(f"⚠️ {msg}")
                    self.last_error = f"❌ LLM có lỗi: {msg}"
                    self.available_models = {}

            except Exception as e:
                err = str(e)
                logger.error(f"❌ Failed to initialize OpenAI: {err}")
                self.last_error = f"❌ LLM có lỗi: {err}"
                self.available_models = {}

        # Allow initialization without models for offline mode (like Gemini)
        if not self.available_models:
            logger.warning("⚠️ No OpenAI models available, system will run in offline mode")
            self.offline_mode = True
        else:
            self.offline_mode = False

    def _is_expensive_model(self, model_name: str) -> bool:
        """Check if model is expensive and should skip testing"""
        expensive_keywords = ['gpt-4o', 'gpt-4-turbo']
        return any(keyword in model_name for keyword in expensive_keywords)
    
    def test_connection(self):
        """Test OpenAI API connection"""
        results = {}
        
        if 'openai' in self.available_models and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.current_model_name,
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=5
                )
                if response and response.choices:
                    results['openai'] = True
                    logger.info("✅ OpenAI connection test passed")
                else:
                    results['openai'] = False
                    msg = "OpenAI returned empty response"
                    logger.error(f"❌ {msg}")
                    self.last_error = f"❌ LLM có lỗi: {msg}"
            except Exception as e:
                results['openai'] = False
                err = str(e)
                logger.error(f"❌ OpenAI connection test failed: {err}")
                self.last_error = f"❌ LLM có lỗi: {err}"
        
        return results
    
    def generate_with_model(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Generate response using OpenAI GPT model
        """
        try:
            if 'openai' in self.available_models and self.client:
                response = self.client.chat.completions.create(
                    model=self.current_model_name,
                    messages=[
                        {"role": "system", "content": "Bạn là chuyên gia phân tích tài chính chuyên nghiệp, trả lời bằng tiếng Việt."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content
            else:
                raise ValueError("OpenAI model not available")
                
        except Exception as e:
            err = str(e)
            logger.error(f"Error generating with OpenAI: {err}")
            self.last_error = f"❌ LLM có lỗi: {err}"
            raise

    def generate_with_fallback(self, prompt: str, task_type: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Generate response with automatic fallback to offline mode
        """
        if getattr(self, 'offline_mode', True) or not self.available_models:
            logger.info("📴 Using offline mode (no OpenAI models available)")
            return self._generate_offline_fallback(prompt, task_type)
        
        try:
            response = self.generate_with_model(prompt, max_tokens)
            return {
                'response': response,
                'model_used': f'openai_{self.current_model_name}',
                'success': True
            }
        except Exception as e:
            err = str(e)
            logger.error(f"OpenAI model failed: {err}")
            fallback = self._generate_offline_fallback(prompt, task_type)
            fallback['response'] = f"❌ LLM có lỗi: {err}.\n\n" + fallback.get('response', '')
            fallback['error'] = err
            return fallback
    
    def _generate_offline_fallback(self, prompt: str, task_type: str) -> Dict[str, Any]:
        """
        Generate offline fallback response when API fails
        """
        try:
            if 'CÂU HỎI:' in prompt:
                question = prompt.split('CÂU HỎI:')[1].split('MÃ CỔ PHIẾU:')[0].strip()
            else:
                question = prompt[:200] + '...' if len(prompt) > 200 else prompt
            
            if task_type == 'financial_advice':
                response = self._generate_financial_advice_fallback(question)
            elif task_type == 'general_query':
                response = self._generate_general_fallback(question)
            else:
                response = self._generate_default_fallback(question)
            
            return {
                'response': response,
                'model_used': 'openai_offline_fallback',
                'success': True,
                'quota_exceeded': True
            }
        except Exception as e:
            return {
                'response': f'OpenAI offline fallback failed: {str(e)}',
                'model_used': 'openai_offline_fallback',
                'success': False,
                'error': str(e)
            }
    
    def _generate_financial_advice_fallback(self, question: str) -> str:
        """Generate financial advice fallback"""
        return f"""
📊 PHÂN TÍCH OPENAI OFFLINE:
Do OpenAI API không khả dụng, hệ thống chuyển sang chế độ offline với phân tích cơ bản:

💡 Nguyên tắc đầu tư GPT:
- Đa dạng hóa danh mục để phân tán rủi ro
- Đầu tư dài hạn thường mang lại lợi nhuận tốt hơn
- Chỉ đầu tư số tiền có thể chấp nhận mất
- Nghiên cứu kỹ trước khi đầu tư (DYOR)

📈 Phân tích kỹ thuật cơ bản:
- Theo dõi xu hướng giá và khối lượng giao dịch
- Sử dụng các chỉ báo như RSI, MACD, MA
- Xem xét mức hỗ trợ và kháng cự

⚠️ Lưu ý: Đây chỉ là thông tin tham khảo, không phải lời khuyên đầu tư.
🔄 API thường reset sau 24 giờ hoặc khi quota được gia hạn.
"""
    
    def _generate_general_fallback(self, question: str) -> str:
        """Generate general fallback"""
        return f"""
🤖 OPENAI OFFLINE MODE:
Câu hỏi của bạn: {question}

Do OpenAI API tạm thời không khả dụng, tôi không thể cung cấp phân tích chi tiết.

💡 Gợi ý:
- Kiểm tra lại API key OpenAI
- Đảm bảo có đủ credits trong tài khoản
- Thử lại sau vài phút

🔄 Hệ thống sẽ tự động chuyển về OpenAI khi API hoạt động trở lại.
"""
    
    def _generate_default_fallback(self, question: str) -> str:
        """Generate default fallback"""
        return f"""
🤖 OPENAI FALLBACK:
Câu hỏi: {question}

Do OpenAI API không khả dụng, hệ thống sử dụng phản hồi cơ bản.

💡 Để có phân tích chi tiết hơn:
- Kiểm tra OpenAI credits tại: https://platform.openai.com/usage
- Hoặc sử dụng Gemini API (miễn phí) thay thế
- Thử lại sau vài phút

🔄 Hệ thống vẫn hoạt động với các tính năng khác.
"""
        return f"""
🤖 OPENAI SYSTEM OFFLINE:
OpenAI API hiện không khả dụng. Vui lòng:

1. Kiểm tra kết nối internet
2. Xác minh API key OpenAI
3. Kiểm tra quota/credits
4. Thử lại sau ít phút

📞 Hỗ trợ: Liên hệ admin nếu vấn đề kéo dài.
"""