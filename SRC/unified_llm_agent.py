import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import asyncio

# Import các LLM agents - Unified version
from gemini_agent import UnifiedAIAgent as GeminiAgent  # Gemini 2.0 Flash as standard
try:
    from openai_agent import OpenAIAgent
except ImportError:
    OpenAIAgent = None
    logger.warning("⚠️ OpenAI Agent not available")

try:
    from llama_agent import LlamaAgent
except ImportError:
    LlamaAgent = None
    logger.warning("⚠️ Llama Agent not available")

logger = logging.getLogger(__name__)

class UnifiedLLMAgent:
    """
    Unified LLM Agent quản lý tất cả các AI models:
    - Google Gemini (Free)
    - OpenAI GPT (Paid)
    - Meta Llama (Multiple providers)
    
    Tự động chọn model tốt nhất dựa trên task và availability
    """
    
    def __init__(self, 
                 gemini_api_key: str = None,
                 openai_api_key: str = None, 
                 llama_api_key: str = None,
                 llama_base_url: str = None):
        """
        Initialize tất cả LLM agents
        """
        self.agents = {}
        self.model_priority = ['gemini', 'openai', 'llama']  # Priority order
        self.current_agent = None
        
        # Initialize Gemini Agent
        if gemini_api_key:
            try:
                self.agents['gemini'] = GeminiAgent(gemini_api_key)
                logger.info("✅ Gemini Agent initialized")
            except Exception as e:
                logger.error(f"❌ Gemini Agent failed: {e}")
        
        # Initialize OpenAI Agent
        if openai_api_key and OpenAIAgent:
            try:
                self.agents['openai'] = OpenAIAgent(openai_api_key)
                logger.info("✅ OpenAI Agent initialized")
            except Exception as e:
                logger.error(f"❌ OpenAI Agent failed: {e}")
        
        # Initialize Llama Agent
        if llama_api_key and LlamaAgent:
            try:
                self.agents['llama'] = LlamaAgent(llama_api_key, llama_base_url)
                logger.info("✅ Llama Agent initialized")
            except Exception as e:
                logger.error(f"❌ Llama Agent failed: {e}")
        
        # Set current agent based on priority
        self._select_best_agent()
        
        # Count actually available models
        available_count = sum(1 for agent in self.agents.values() if not getattr(agent, 'offline_mode', True))
        logger.info(f"🤖 Unified LLM Agent ready with {available_count}/{len(self.agents)} models online")
    
    def _select_best_agent(self) -> Optional[str]:
        """
        Chọn agent tốt nhất dựa trên priority và availability
        """
        available_agents = []
        
        for agent_name in self.model_priority:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                is_offline = getattr(agent, 'offline_mode', True)
                has_models = bool(getattr(agent, 'available_models', {}))
                
                if not is_offline and has_models:
                    available_agents.append(agent_name)
                    self.current_agent = agent_name
                    logger.info(f"🎯 Selected agent: {agent_name}")
                    return agent_name
        
        # Nếu không có agent nào available, chọn agent đầu tiên (dù offline)
        if self.agents:
            self.current_agent = list(self.agents.keys())[0]
            logger.warning(f"⚠️ All agents offline, fallback to: {self.current_agent}")
            return self.current_agent
        
        logger.error("❌ No agents available")
        return None
    
    def get_available_models(self) -> Dict[str, Any]:
        """
        Lấy danh sách tất cả models available
        """
        available = {}
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'available_models') and agent.available_models:
                available[agent_name] = {
                    'models': agent.available_models,
                    'current_model': getattr(agent, 'current_model_name', None),
                    'offline_mode': getattr(agent, 'offline_mode', True),
                    'capabilities': getattr(agent, 'model_capabilities', {})
                }
        return available
    
    def test_all_connections(self) -> Dict[str, bool]:
        """
        Test connection tất cả agents
        """
        results = {}
        for agent_name, agent in self.agents.items():
            try:
                if hasattr(agent, 'test_connection'):
                    test_result = agent.test_connection()
                    results[agent_name] = any(test_result.values()) if isinstance(test_result, dict) else test_result
                else:
                    results[agent_name] = not getattr(agent, 'offline_mode', True)
            except Exception as e:
                results[agent_name] = False
                logger.error(f"❌ {agent_name} connection test failed: {e}")
        
        return results
    
    def generate_response(self, 
                         prompt: str, 
                         task_type: str = 'general_query',
                         max_tokens: int = 2000,
                         preferred_agent: str = None) -> Dict[str, Any]:
        """
        Generate response - uses current_agent (user selected) or fallback
        """
        # Use current selected agent first
        if self.current_agent and self.current_agent in self.agents:
            try:
                agent = self.agents[self.current_agent]
                result = agent.generate_with_fallback(prompt, task_type, max_tokens)
                
                # If successful, return with current agent info
                if result.get('success') and not result.get('quota_exceeded'):
                    logger.info(f"✅ Response generated by selected {self.current_agent}")
                    return result
                
                # If quota exceeded, try fallback
                elif result.get('quota_exceeded'):
                    logger.warning(f"⚠️ Selected {self.current_agent} quota exceeded, trying fallback...")
                    
            except Exception as e:
                logger.error(f"❌ Selected agent {self.current_agent} failed: {e}")
        
        # Fallback to other available agents if current fails
        for agent_name in self.model_priority:
            if agent_name in self.agents and agent_name != self.current_agent:
                try:
                    agent = self.agents[agent_name]
                    result = agent.generate_with_fallback(prompt, task_type, max_tokens)
                    
                    if result.get('success') and not result.get('quota_exceeded'):
                        logger.info(f"✅ Fallback response generated by {agent_name}")
                        return result
                        
                except Exception as e:
                    logger.error(f"❌ Fallback agent {agent_name} failed: {e}")
                    continue
        
        # If all agents fail, return offline fallback
        return self._generate_unified_offline_fallback(prompt, task_type)
    
    def _generate_unified_offline_fallback(self, prompt: str, task_type: str) -> Dict[str, Any]:
        """
        Unified offline fallback khi tất cả agents fail
        """
        return {
            'response': f"""
🤖 UNIFIED LLM OFFLINE MODE:
Tất cả AI models hiện không khả dụng. Hệ thống đang chạy ở chế độ offline.

📊 Models được hỗ trợ:
- 🧠 Google Gemini (Free tier)
- 🚀 OpenAI GPT (Paid API)
- 🦙 Meta Llama (Multiple providers)

💡 Khắc phục:
1. Kiểm tra API keys
2. Xác minh quota/credits
3. Kiểm tra kết nối internet
4. Thử lại sau vài phút

🔄 Hệ thống sẽ tự động kết nối lại khi có AI model khả dụng.

⚠️ Lưu ý: Đây chỉ là thông tin cơ bản, không thay thế được phân tích AI chuyên sâu.
""",
            'model_used': 'unified_offline_fallback',
            'success': True,
            'all_agents_failed': True
        }
    
    def switch_agent(self, agent_name: str) -> bool:
        """
        Chuyển đổi sang agent cụ thể (user manual selection)
        """
        if agent_name in self.agents:
            # Check if agent is actually available (not offline)
            agent_status = self.get_agent_status()
            if not agent_status['agents'][agent_name]['offline_mode']:
                self.current_agent = agent_name
                logger.info(f"🔄 User switched to agent: {agent_name}")
                return True
            else:
                logger.error(f"❌ Agent {agent_name} is offline")
                return False
        else:
            logger.error(f"❌ Agent {agent_name} not available")
            return False
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Lấy status của tất cả agents
        """
        status = {
            'current_agent': self.current_agent,
            'total_agents': len(self.agents),
            'agents': {}
        }
        
        for agent_name, agent in self.agents.items():
            is_offline = getattr(agent, 'offline_mode', True)
            has_models = bool(getattr(agent, 'available_models', {}))
            current_model = getattr(agent, 'current_model_name', None)
            
            # Agent is truly available if it has models AND is not in offline mode
            is_truly_available = has_models and not is_offline
            
            status['agents'][agent_name] = {
                'available': agent_name in self.agents,
                'offline_mode': is_offline,
                'current_model': current_model,
                'has_models': has_models,
                'truly_available': is_truly_available
            }
        
        return status
    
    def add_agent(self, agent_name: str, agent_instance) -> bool:
        """
        Thêm agent mới vào hệ thống
        """
        try:
            self.agents[agent_name] = agent_instance
            logger.info(f"✅ Added new agent: {agent_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add agent {agent_name}: {e}")
            return False
    
    def remove_agent(self, agent_name: str) -> bool:
        """
        Xóa agent khỏi hệ thống
        """
        if agent_name in self.agents:
            del self.agents[agent_name]
            if self.current_agent == agent_name:
                self._select_best_agent()
            logger.info(f"🗑️ Removed agent: {agent_name}")
            return True
        return False