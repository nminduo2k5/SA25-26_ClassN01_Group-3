from openai import OpenAI
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class OpenAILLM:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = None
        self.current_model = None
        self.is_available = False
        
        if api_key:
            self._initialize()
    
    def _initialize(self):
        try:
            self.client = OpenAI(api_key=self.api_key)
            
            models = [
                'gpt-4o',
                'gpt-4-turbo',
                'gpt-4',
                'gpt-3.5-turbo'
            ]
            
            self.current_model = models[0]
            self.is_available = True
            logger.info(f"✅ OpenAI: {self.current_model}")
            
        except Exception as e:
            logger.error(f"❌ OpenAI init failed: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> Dict[str, Any]:
        if not self.is_available:
            return {'success': False, 'error': 'OpenAI unavailable'}
        
        try:
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return {
                'success': True,
                'response': response.choices[0].message.content,
                'model': self.current_model
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