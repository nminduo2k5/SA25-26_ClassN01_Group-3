#!/usr/bin/env python3
"""
Quick Test for OpenAI Integration
"""

from gemini_agent import UnifiedAIAgent

def test_basic():
    print("Testing basic initialization...")
    
    # Test 1: No keys (should work in offline mode)
    print("\n1. Testing without keys:")
    agent = UnifiedAIAgent()
    print(f"Available models: {list(agent.available_models.keys())}")
    print(f"Offline mode: {getattr(agent, 'offline_mode', True)}")
    
    # Test 2: With fake OpenAI key (should fail gracefully)
    print("\n2. Testing with fake OpenAI key:")
    try:
        agent = UnifiedAIAgent(openai_api_key="sk-fake-key", preferred_model="openai")
        print(f"Available models: {list(agent.available_models.keys())}")
        print(f"OpenAI client exists: {hasattr(agent, 'openai_client') and agent.openai_client is not None}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: With fake Gemini key (should fail gracefully)
    print("\n3. Testing with fake Gemini key:")
    try:
        agent = UnifiedAIAgent(gemini_api_key="AIza-fake-key", preferred_model="gemini")
        print(f"Available models: {list(agent.available_models.keys())}")
        print(f"Gemini key exists: {hasattr(agent, 'gemini_api_key') and agent.gemini_api_key is not None}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Test generation in offline mode
    print("\n4. Testing offline generation:")
    agent = UnifiedAIAgent()
    result = agent.generate_with_fallback("Test question", "general_query")
    print(f"Success: {result.get('success', False)}")
    print(f"Model used: {result.get('model_used', 'None')}")
    print(f"Response length: {len(result.get('response', ''))}")

if __name__ == "__main__":
    test_basic()
    print("\nBasic test completed!")