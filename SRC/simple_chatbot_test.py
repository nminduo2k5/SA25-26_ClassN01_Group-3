#!/usr/bin/env python3
"""
Simple chatbot test
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm.unified_llm import UnifiedLLM
from gemini_agent import UnifiedAIAgent

def test_simple_chatbot():
    """Test simple chatbot without complex dependencies"""
    print("Simple Chatbot Test")
    print("=" * 20)
    
    # Test UnifiedLLM directly
    print("\n1. Testing UnifiedLLM...")
    llm = UnifiedLLM()
    print(f"Available models: {list(llm.available_models.keys())}")
    
    if llm.available_models:
        result = llm.generate("VCB co nen mua khong?", max_tokens=50)
        print(f"Success: {result['success']}")
        print(f"Model: {result.get('model', 'unknown')}")
        if result['success']:
            print(f"Response: {result['response'][:100]}...")
    
    # Test UnifiedAIAgent
    print("\n2. Testing UnifiedAIAgent...")
    agent = UnifiedAIAgent()
    print(f"Available models: {list(agent.available_models.keys())}")
    print(f"Offline mode: {agent.offline_mode}")
    
    if not agent.offline_mode:
        result = agent.generate_with_fallback("VCB analysis", "financial_advice")
        print(f"Success: {result['success']}")
        print(f"Model used: {result.get('model_used', 'unknown')}")
        if result['success']:
            print(f"Response preview: {result['response'][:100]}...")

if __name__ == "__main__":
    test_simple_chatbot()