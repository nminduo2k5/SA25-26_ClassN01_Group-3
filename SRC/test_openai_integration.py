#!/usr/bin/env python3
"""
Test OpenAI Integration
Kiểm tra tích hợp OpenAI với hệ thống
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemini_agent import UnifiedAIAgent

def test_openai_integration():
    """Test OpenAI integration"""
    print("🧪 Testing OpenAI Integration...")
    
    # Test với OpenAI API key (thay bằng key thật để test)
    openai_key = "your-openai-api-key-here"  # Thay bằng key thật
    gemini_key = "your-gemini-api-key-here"  # Thay bằng key thật
    
    print("\n1. Testing OpenAI preference...")
    try:
        agent = UnifiedAIAgent(
            gemini_api_key=gemini_key,
            openai_api_key=openai_key,
            preferred_model="openai"
        )
        
        print(f"Available models: {list(agent.available_models.keys())}")
        print(f"Preferred model: {agent.preferred_model}")
        
        if 'openai' in agent.available_models:
            print("✅ OpenAI model initialized successfully")
            
            # Test generation
            result = agent.generate_with_fallback(
                "Hello, this is a test", 
                "general_query", 
                max_tokens=50
            )
            
            if result['success']:
                print(f"✅ OpenAI generation successful: {result['model_used']}")
                print(f"Response: {result['response'][:100]}...")
            else:
                print(f"❌ OpenAI generation failed: {result}")
        else:
            print("❌ OpenAI model not available")
            
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
    
    print("\n2. Testing Gemini preference...")
    try:
        agent = UnifiedAIAgent(
            gemini_api_key=gemini_key,
            openai_api_key=openai_key,
            preferred_model="gemini"
        )
        
        print(f"Available models: {list(agent.available_models.keys())}")
        print(f"Preferred model: {agent.preferred_model}")
        
        if 'gemini' in agent.available_models:
            print("✅ Gemini model initialized successfully")
        else:
            print("❌ Gemini model not available")
            
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
    
    print("\n3. Testing auto mode...")
    try:
        agent = UnifiedAIAgent(
            gemini_api_key=gemini_key,
            openai_api_key=openai_key,
            preferred_model="auto"
        )
        
        print(f"Available models: {list(agent.available_models.keys())}")
        print(f"Preferred model: {agent.preferred_model}")
        
        # Test model selection
        try:
            selected_model = agent.select_best_model("general_query")
            print(f"✅ Auto mode selected: {selected_model}")
        except Exception as e:
            print(f"❌ Auto mode selection failed: {e}")
            
    except Exception as e:
        print(f"❌ Auto mode test failed: {e}")

def test_without_keys():
    """Test without API keys (offline mode)"""
    print("\n4. Testing offline mode...")
    try:
        agent = UnifiedAIAgent(preferred_model="auto")
        
        print(f"Available models: {list(agent.available_models.keys())}")
        print(f"Offline mode: {getattr(agent, 'offline_mode', True)}")
        
        # Test offline generation
        result = agent.generate_with_fallback(
            "Test offline mode", 
            "general_query"
        )
        
        if result['success']:
            print(f"✅ Offline mode working: {result['model_used']}")
        else:
            print(f"❌ Offline mode failed: {result}")
            
    except Exception as e:
        print(f"❌ Offline mode test failed: {e}")

if __name__ == "__main__":
    print("🚀 OpenAI Integration Test")
    print("=" * 50)
    
    # Test with keys (replace with real keys to test)
    test_openai_integration()
    
    # Test without keys
    test_without_keys()
    
    print("\n" + "=" * 50)
    print("✅ Test completed!")
    print("\n💡 To test with real API keys:")
    print("1. Replace 'your-openai-api-key-here' with real OpenAI API key")
    print("2. Replace 'your-gemini-api-key-here' with real Gemini API key")
    print("3. Run this script again")