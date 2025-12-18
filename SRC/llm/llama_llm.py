import logging
from typing import Dict, Any

try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    completion = None

logger = logging.getLogger(__name__)

class LlamaLLM:
    def __init__(self, model_name: str = None):
        self.is_available = False
        self.is_configured = LITELLM_AVAILABLE
        self.available_models = [
            "ollama/llama3.1:8b",
            "ollama/llama3:8b", 
            "ollama/llama2:7b",
            "ollama/codellama:7b"
        ]
        self.current_model = model_name or self.available_models[0]
        
        if self.is_configured:
            self._initialize()
        else:
            logger.warning("⚠️ Llama not configured: pip install litellm")
    
    def _initialize(self):
        # Try each model until one works
        for model in self.available_models:
            try:
                test_response = completion(
                    model=model,
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=0.1,
                    max_tokens=5,
                    timeout=60
                )
                if test_response and test_response.choices:
                    self.current_model = model
                    self.is_available = True
                    logger.info(f"✅ Llama: {model}")
                    return
            except Exception as e:
                logger.debug(f"Model {model} failed: {e}")
                continue
        
        # If no models work, still mark as configured but unavailable
        logger.info("📋 Llama configured but no models available. Run: ollama serve && ollama pull llama3.1:8b")
    
    def generate(self, prompt: str, max_tokens: int = 150) -> Dict[str, Any]:
        if not self.is_configured:
            return {'success': False, 'error': 'Llama not configured: pip install litellm'}
        
        if not self.is_available:
            return {'success': False, 'error': 'Llama unavailable: ollama serve && ollama pull llama3.1:8b'}
        
        try:
            response = completion(
                model=self.current_model,
                messages=[
                    {"role": "system", "content": "Trợ lý tài chính. Trả lời ngắn gọn tiếng Việt."},
                    {"role": "user", "content": prompt[:500]}
                ],
                temperature=0.1,
                max_tokens=min(max_tokens, 150),
                timeout=30
            )
            return {
                'success': True,
                'response': response.choices[0].message.content,
                'model': self.current_model,
                'provider': 'ollama_local'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_connection(self) -> bool:
        if not self.is_configured or not self.is_available:
            return False
        try:
            result = self.generate("test", 5)
            return result['success']
        except:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information"""
        return {
            'configured': self.is_configured,
            'available': self.is_available,
            'current_model': self.current_model,
            'setup_command': 'ollama serve && ollama pull llama3.1:8b' if self.is_configured else 'pip install litellm'
        }