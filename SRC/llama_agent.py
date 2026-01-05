import os
import requests
import logging
from typing import Dict, Any, Optional, List
import asyncio
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Load .env into environment if present. Prefer python-dotenv when available,
# otherwise fall back to a tiny parser so `os.getenv` works during local dev.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # manual fallback: check common locations
    try:
        candidates = [
            os.path.join(os.getcwd(), '.env'),
            os.path.join(os.path.dirname(__file__), '.env')
        ]
        for env_path in candidates:
            if os.path.exists(env_path):
                with open(env_path, 'r', encoding='utf-8') as fh:
                    for raw in fh:
                        line = raw.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' not in line:
                            continue
                        k, v = line.split('=', 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and v and k not in os.environ:
                            os.environ[k] = v
                break
    except Exception:
        pass

class LlamaAgent:
    def __init__(self, llama_api_key: str = None, base_url: str = None):
        """
        Initialize Llama Agent with Meta Llama models
        Supports Ollama local + API providers (Together AI, Groq, etc.)
        """
        self.available_models = {}
        self.current_model_name = None
        self.last_error: Optional[str] = None
        self.llama_api_key = llama_api_key
        self.base_url = base_url or os.getenv("LLAMA_BASE_URL") or "http://localhost:11434"  # Ollama root URL
        self.is_ollama = "localhost" in self.base_url or "127.0.0.1" in self.base_url
        
        # If no API key passed explicitly, try common environment variables
        if not self.llama_api_key and not self.is_ollama:
            self.llama_api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('LLAMA_API_KEY') or os.getenv('TOGETHER_API_KEY')
        self.offline_mode = True  # Default to offline until proven otherwise

        # Enhanced model capabilities with tier classification
        self.model_capabilities = {
            'llama-3.1-405b': {
                'strengths': ['analysis', 'reasoning', 'multilingual', 'financial_advice', 'long_context'],
                'speed': 'slow',
                'cost': 'high',
                'tier': 'premium'
            },
            'llama-3.1-70b': {
                'strengths': ['analysis', 'reasoning', 'balanced_performance', 'financial_advice'],
                'speed': 'medium',
                'cost': 'medium',
                'tier': 'standard'
            },
            'llama-3.1-8b': {
                'strengths': ['fast_response', 'cost_effective', 'general_purpose'],
                'speed': 'fast',
                'cost': 'low',
                'tier': 'basic'
            },
            'llama-2-70b': {
                'strengths': ['stable_performance', 'reasoning', 'cost_effective'],
                'speed': 'medium',
                'cost': 'medium',
                'tier': 'standard'
            },
            'llama-2-13b': {
                'strengths': ['fast_response', 'cost_effective'],
                'speed': 'fast',
                'cost': 'low',
                'tier': 'basic'
            }
        }

        # Provider-specific configurations
        self.providers = {
            'ollama': {
                'base_url': 'http://localhost:11434',
                'models': [
                    'llama3.1:8b',      # User's downloaded model
                    'llama3.1:70b',
                    'llama3:8b',
                    'llama2:13b',
                    'llama2:7b'
                ]
            },
            'together': {
                'base_url': 'https://api.together.xyz/v1',
                'models': [
                    'meta-llama/Llama-3.1-70B-Instruct-Turbo',
                    'meta-llama/Llama-3.1-8B-Instruct-Turbo'
                ]
            },
            'groq': {
                'base_url': 'https://api.groq.com/openai/v1',
                'models': [
                    'llama-3.1-70b-versatile',
                    'llama-3.1-8b-instant'
                ]
            }
        }

        # Initialize based on provider type
        try:
            if self.is_ollama:
                # Ollama local setup (no API key needed)
                self.headers = {"Content-Type": "application/json"}
                model_initialized = self._init_ollama_models()
            else:
                # API provider setup (needs API key)
                if not self.llama_api_key:
                    model_initialized = False
                    logger.warning("⚠️ No API key provided for external Llama provider")
                else:
                    self.headers = {
                        "Authorization": f"Bearer {self.llama_api_key}",
                        "Content-Type": "application/json"
                    }
                    model_initialized = self._init_api_models()

            if not model_initialized:
                self.available_models = {}
                self.offline_mode = True

        except Exception as e:
            err = str(e)
            logger.error(f"❌ Failed to initialize Llama: {err}")
            self.last_error = f"❌ LLM có lỗi: {err}"
            self.available_models = {}

        # Allow initialization without models for offline mode (like Gemini)
        if not self.available_models:
            logger.warning("⚠️ No Llama models available, system will run in offline mode")
            self.offline_mode = True
    
    def _init_ollama_models(self) -> bool:
        """Initialize Ollama local models"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [m['name'] for m in models_data.get('models', [])]
                
                # Prioritize user's downloaded model
                priority_models = ['llama3.1:8b', 'llama3.1:70b', 'llama3:8b', 'llama2:13b']
                
                for model in priority_models:
                    if model in available_models:
                        self.available_models['llama'] = model
                        self.current_model_name = model
                        self.offline_mode = False
                        logger.info(f"✅ Ollama initialized with model: {model}")
                        return True
                
                # If no priority model, use first available
                if available_models:
                    model = available_models[0]
                    self.available_models['llama'] = model
                    self.current_model_name = model
                    self.offline_mode = False
                    logger.info(f"✅ Ollama initialized with model: {model} (fallback)")
                    return True
                
                logger.warning("⚠️ No Ollama models found")
                return False
            else:
                logger.warning(f"⚠️ Ollama not responding: {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Ollama connection failed: {e}")
            return False
    
    def _init_api_models(self) -> bool:
        """Initialize API provider models"""
        current_provider = self._detect_provider(self.base_url)
        model_names = self._get_model_list_for_provider(current_provider)
        
        for model_name in model_names:
            try:
                test_payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=test_payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    self.available_models['llama'] = model_name
                    self.current_model_name = model_name
                    self.offline_mode = False
                    logger.info(f"✅ API Llama initialized with model: {model_name}")
                    return True
                    
            except Exception as e:
                logger.warning(f"⚠️ Model {model_name} failed: {e}")
                continue
        
        return False
    
    def _detect_provider(self, url: str) -> str:
        """Detect provider from URL"""
        if "localhost" in url or "127.0.0.1" in url:
            return 'ollama'
        elif "together" in url:
            return 'together'
        elif "groq" in url:
            return 'groq'
        else:
            return 'together'  # default
    
    def _get_model_list_for_provider(self, provider: str) -> List[str]:
        """Get model list for specific provider"""
        return self.providers.get(provider, {}).get('models', [])
    
    def generate_with_model(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate response using Llama model"""
        if not self.available_models or self.offline_mode:
            raise ValueError("No Llama models available")
        
        try:
            if self.is_ollama:
                # Ollama native API format - POST http://127.0.0.1:11434/api/generate
                payload = {
                    "model": self.current_model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": min(max_tokens, 1000),
                        "temperature": 0.7
                    }
                }
                
                # Use base_url directly for Ollama
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '')
                else:
                    raise ValueError(f"Ollama API returned status {response.status_code}: {response.text}")
            else:
                # OpenAI-compatible API format for other providers
                payload = {
                    "model": self.current_model_name,
                    "messages": [
                        {"role": "system", "content": "Bạn là chuyên gia phân tích tài chính chuyên nghiệp. Trả lời bằng tiếng Việt một cách chi tiết và chính xác."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                else:
                    raise ValueError(f"API returned status {response.status_code}: {response.text}")
            
        except Exception as e:
            logger.error(f"Llama generation failed: {e}")
            raise
    
    def test_connection(self) -> Dict[str, bool]:
        """Test Llama connection"""
        results = {}
        
        if 'llama' in self.available_models:
            try:
                if self.is_ollama:
                    # Test Ollama with native API
                    test_payload = {
                        "model": self.current_model_name,
                        "prompt": "Test",
                        "stream": False
                    }
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        headers=self.headers,
                        json=test_payload,
                        timeout=10
                    )
                else:
                    # Test API providers with OpenAI format
                    test_payload = {
                        "model": self.current_model_name,
                        "messages": [{"role": "user", "content": "Test"}],
                        "max_tokens": 5
                    }
                    response = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=test_payload,
                        timeout=10
                    )
                
                if response.status_code == 200:
                    results['llama'] = True
                    logger.info("✅ Llama connection test passed")
                else:
                    results['llama'] = False
                    logger.error(f"❌ Llama returned status {response.status_code}")
            except Exception as e:
                results['llama'] = False
                logger.error(f"❌ Llama connection test failed: {e}")
        
        return results

    def _detect_provider(self, base_url: str) -> str:
        """Detect provider from base URL"""
        if 'together' in base_url:
            return 'together'
        elif 'openrouter' in base_url:
            return 'openrouter'
        elif 'groq' in base_url:
            return 'groq'
        elif 'replicate' in base_url:
            return 'replicate'
        else:
            return 'together'  # Default

    def _get_model_list_for_provider(self, provider: str) -> list:
        """Get model list for specific provider"""
        if provider in self.providers:
            return self.providers[provider]['models']
        else:
            # Default fallback list
            return [
                'meta-llama/Llama-3.1-70B-Instruct-Turbo',
                'meta-llama/Llama-3.1-8B-Instruct-Turbo',
                'meta-llama/Llama-2-70b-chat-hf',
                'llama-3.1-70b-versatile',
                'llama-3.1-8b-instant'
            ]

    def _is_expensive_model(self, model_name: str) -> bool:
        """Check if model is expensive and should skip testing"""
        expensive_keywords = ['405b', '70b', 'versatile']
        return any(keyword in model_name.lower() for keyword in expensive_keywords)


    def generate_with_fallback(self, prompt: str, task_type: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Generate response with automatic fallback to offline mode
        """
        if getattr(self, 'offline_mode', True) or not self.available_models:
            logger.info("📴 Using offline mode (no Llama models available)")
            return self._generate_offline_fallback(prompt, task_type)
        
        try:
            response = self.generate_with_model(prompt, max_tokens)
            return {
                'response': response,
                'model_used': f'llama_{self.current_model_name}',
                'success': True
            }
        except Exception as e:
            err = str(e)
            logger.error(f"Llama model failed: {err}")
            # Include visible LLM error prefix in the fallback response
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
                'model_used': 'llama_offline_fallback',
                'success': True,
                'quota_exceeded': True
            }
        except Exception as e:
            return {
                'response': f'Llama offline fallback failed: {str(e)}',
                'model_used': 'llama_offline_fallback',
                'success': False,
                'error': str(e)
            }
    
    def _generate_financial_advice_fallback(self, question: str) -> str:
        """Generate financial advice fallback"""
        return f"""
🦙 PHÂN TÍCH LLAMA OFFLINE:
Do Llama API không khả dụng, hệ thống chuyển sang chế độ offline với phân tích Meta AI:

🧠 Nguyên tắc đầu tư Llama:
- Phân tích cơ bản (Fundamental Analysis) là nền tảng
- Quản lý rủi ro là yếu tố quan trọng nhất
- Tâm lý thị trường ảnh hưởng lớn đến giá cả
- Đầu tư theo chu kỳ thị trường

📊 Chiến lược đầu tư thông minh:
- Dollar Cost Averaging (DCA) để giảm rủi ro
- Rebalancing danh mục định kỳ
- Theo dõi các chỉ số kinh tế vĩ mô
- Đọc báo cáo tài chính doanh nghiệp

🎯 Mục tiêu đầu tư:
- Xác định rõ mục tiêu tài chính
- Thời gian đầu tư phù hợp
- Mức độ rủi ro có thể chấp nhận

⚠️ Disclaimer: Thông tin chỉ mang tính chất tham khảo.
🔄 Llama API sẽ hoạt động trở lại khi có kết nối ổn định.
"""
    
    def _generate_general_fallback(self, question: str) -> str:
        """Generate general fallback"""
        return f"""
🦙 LLAMA OFFLINE MODE:
Câu hỏi của bạn: {question}

Do Llama API tạm thời không khả dụng, tôi không thể cung cấp phân tích từ Meta AI.

💡 Kiểm tra:
- API key Llama (Together AI, Groq, Replicate)
- Kết nối internet ổn định
- Quota/credits trong tài khoản
- Base URL endpoint đúng

🔧 Các nhà cung cấp Llama API:
- Together AI: https://api.together.xyz/v1
- Groq: https://api.groq.com/openai/v1
- Replicate: https://api.replicate.com/v1

🔄 Hệ thống sẽ tự động kết nối lại khi API khả dụng.
"""
    
    def _generate_default_fallback(self, question: str) -> str:
        """Generate default fallback"""
        return f"""
🦙 LLAMA SYSTEM OFFLINE:
Meta Llama API hiện không khả dụng. Vui lòng:

1. Kiểm tra API key và provider
2. Xác minh base URL endpoint
3. Kiểm tra quota/rate limits
4. Thử đổi sang provider khác

🌐 Supported Providers:
- Together AI (Recommended)
- Groq (Fast inference)
- Replicate (Reliable)
- Hugging Face Inference

📞 Hỗ trợ: Liên hệ admin nếu vấn đề kéo dài.
"""