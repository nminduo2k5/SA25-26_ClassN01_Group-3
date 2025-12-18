#!/usr/bin/env python3
"""
Test new LLM system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm.unified_llm import UnifiedLLM
from gemini_agent import UnifiedAIAgent

def test_individual_llms():
    """Test individual LLM classes"""
    print("Testing Individual LLMs...")
    
    # Test Gemini
    print("\n1. Testing Gemini LLM...")
    from llm.gemini_llm import GeminiLLM
    gemini = GeminiLLM("your-gemini-key")  # Replace with real key
    print(f"Gemini available: {gemini.is_available}")
    
    # Test OpenAI
    print("\n2. Testing OpenAI LLM...")
    from llm.openai_llm import OpenAILLM
    openai = OpenAILLM("your-openai-key")  # Replace with real key
    print(f"OpenAI available: {openai.is_available}")
    
    # Test Llama
    print("\n3. Testing Llama LLM...")
    from llm.llama_llm import LlamaLLM
    llama = LlamaLLM()
    print(f"Llama available: {llama.is_available}")

def test_unified_llm():
    """Test UnifiedLLM"""
    print("\nTesting UnifiedLLM...")
    
    # Test without keys (offline mode)
    unified = UnifiedLLM()
    print(f"Available models: {list(unified.available_models.keys())}")
    
    # Test generation
    result = unified.generate("Test prompt")
    print(f"Generation success: {result['success']}")
    print(f"Model used: {result.get('model', 'unknown')}")
    print(f"Response preview: {result['response'][:100]}...")

def test_unified_ai_agent():
    """Test updated UnifiedAIAgent"""
    print("\nTesting Updated UnifiedAIAgent...")
    
    # Test without keys
    agent = UnifiedAIAgent()
    print(f"Offline mode: {agent.offline_mode}")
    
    # Test generation
    result = agent.generate_with_fallback("Phân tích VCB", "financial_advice")
    print(f"Generation success: {result['success']}")
    print(f"Model used: {result.get('model_used', 'unknown')}")

def main():
    print("Testing New LLM System")
    print("=" * 50)
    
    test_individual_llms()
    test_unified_llm()
    test_unified_ai_agent()
    
    print("\n" + "=" * 50)
    print("✅ Test completed!")
    print("\n💡 To test with real API keys:")
    print("1. Replace 'your-gemini-key' and 'your-openai-key' with real keys")
    print("2. Ensure Ollama is running for Llama testing")
    print("3. Run this script again")

if __name__ == "__main__":
    main()