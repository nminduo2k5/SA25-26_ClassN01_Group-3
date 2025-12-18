import google.generativeai as genai
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GeminiLLM:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.model = None
        self.current_model_name = None
        self.is_available = False
        
        if api_key:
            self._initialize()
    
    def _initialize(self):
        try:
            genai.configure(api_key=self.api_key)
            
            models = [
                'gemini-2.0-flash-exp',
                'gemini-1.5-pro-latest', 
                'gemini-1.5-flash-latest',
                'gemini-1.5-pro',
                'gemini-pro'
            ]
            
            for model_name in models:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    self.current_model_name = model_name
                    self.is_available = True
                    logger.info(f"✅ Gemini: {model_name}")
                    return
                except Exception:
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Gemini init failed: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> Dict[str, Any]:
        if not self.is_available:
            return {'success': False, 'error': 'Gemini unavailable'}
        
        try:
            response = self.model.generate_content(prompt)
            return {
                'success': True,
                'response': response.text,
                'model': self.current_model_name
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_connection(self) -> bool:
        if not self.is_available:
            return False
        try:
            result = self.generate("test", 10)
            return result['success']
        except:
            return False