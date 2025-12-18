import logging
from typing import Dict, Any, Optional
from .gemini_llm import GeminiLLM
from .openai_llm import OpenAILLM
from .llama_llm import LlamaLLM

logger = logging.getLogger(__name__)

class UnifiedLLM:
    def __init__(self, gemini_key: str = None, openai_key: str = None, preferred: str = "auto"):
        self.gemini = GeminiLLM(gemini_key) if gemini_key else None
        self.openai = OpenAILLM(openai_key) if openai_key else None
        self.llama = LlamaLLM()
        self.preferred = preferred
        
        self.available_models = self._get_available_models()
        all_status = self.get_all_status()
        
        # Log detailed status
        for model_name, status in all_status.items():
            if status.get('available'):
                logger.info(f"✅ {model_name.upper()}: {status.get('model', 'ready')}")
            elif status.get('configured', True):
                setup_cmd = status.get('setup_command', 'check configuration')
                logger.warning(f"⚠️ {model_name.upper()}: configured but unavailable - {setup_cmd}")
            else:
                logger.warning(f"❌ {model_name.upper()}: not configured")
        
        logger.info(f"Active models: {list(self.available_models.keys())}")
    
    def _get_available_models(self) -> Dict[str, Any]:
        models = {}
        if self.gemini and self.gemini.is_available:
            models['gemini'] = self.gemini
        if self.openai and self.openai.is_available:
            models['openai'] = self.openai
        if self.llama and self.llama.is_available:
            models['llama'] = self.llama
        return models
    
    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all LLMs including configured but unavailable ones"""
        status = {}
        if self.gemini:
            status['gemini'] = {'available': self.gemini.is_available, 'model': getattr(self.gemini, 'current_model_name', None)}
        if self.openai:
            status['openai'] = {'available': self.openai.is_available, 'model': getattr(self.openai, 'current_model', None)}
        if self.llama:
            status['llama'] = self.llama.get_status()
        return status
    
    def _select_model(self) -> Optional[str]:
        if self.preferred in self.available_models:
            return self.preferred
        
        if self.preferred == "auto":
            if 'gemini' in self.available_models:
                return 'gemini'
            if 'openai' in self.available_models:
                return 'openai'
            if 'llama' in self.available_models:
                return 'llama'
        
        return list(self.available_models.keys())[0] if self.available_models else None
    
    def generate(self, prompt: str, max_tokens: int = 2000, force_model: str = None) -> Dict[str, Any]:
        target_model = force_model if force_model in self.available_models else self._select_model()
        
        if not target_model:
            return self._offline_response(prompt)
        
        try:
            result = self.available_models[target_model].generate(prompt, max_tokens)
            if result['success']:
                return result
        except Exception as e:
            logger.error(f"Model {target_model} failed: {e}")
        
        # Fallback to other models
        for model_name, model_instance in self.available_models.items():
            if model_name != target_model:
                try:
                    result = model_instance.generate(prompt, max_tokens)
                    if result['success']:
                        return result
                except Exception:
                    continue
        
        return self._offline_response(prompt)
    
    def _offline_response(self, prompt: str) -> Dict[str, Any]:
        offline_advice = """
📴 HỆ THỐNG OFFLINE

Do tất cả AI models không khả dụng, hệ thống chuyển sang chế độ offline.

💡 KHUYẾN NGHỊ CƠ BẢN:
- Nghiên cứu báo cáo tài chính công ty
- Theo dõi tin tức ngành và thị trường
- Đa dạng hóa danh mục đầu tư
- Chỉ đầu tư tiền nhàn rỗi
- Đặt lệnh stop-loss để kiểm soát rủi ro

⚠️ LƯU Ý: Đây là phản hồi offline. Vui lòng kiểm tra:
- API keys có đúng không
- Quota còn lại
- Kết nối internet
"""
        return {
            'success': True,
            'response': offline_advice,
            'model': 'offline_mode'
        }
    
    def test_all_connections(self) -> Dict[str, bool]:
        results = {}
        if self.gemini:
            results['gemini'] = self.gemini.test_connection()
        if self.openai:
            results['openai'] = self.openai.test_connection()
        if self.llama:
            results['llama'] = self.llama.test_connection()
        return results
    
    def update_keys(self, gemini_key: str = None, openai_key: str = None):
        if gemini_key:
            self.gemini = GeminiLLM(gemini_key)
        if openai_key:
            self.openai = OpenAILLM(openai_key)
        
        self.available_models = self._get_available_models()
        logger.info(f"Updated models: {list(self.available_models.keys())}")