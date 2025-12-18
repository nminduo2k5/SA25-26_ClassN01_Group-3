#!/usr/bin/env python3
"""
Test Gemini LLM specifically
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm.gemini_llm import GeminiLLM
from gemini_agent import UnifiedAIAgent

def test_gemini_direct():
    """Test Gemini LLM directly"""
    print("🧪 Testing Gemini LLM Direct...")
    
    # Replace with your actual Gemini API key
    api_key = input("Nhập Gemini API key (hoặc Enter để skip): ").strip()
    
    if not api_key:
        print("⚠️ Skipping direct test - no API key provided")
        return
    
    gemini = GeminiLLM(api_key)
    print(f"Gemini available: {gemini.is_available}")
    print(f"Current model: {gemini.current_model_name}")
    
    if gemini.is_available:
        # Test generation
        result = gemini.generate("Phân tích cổ phiếu VCB ngắn gọn", 100)
        if result['success']:
            print("✅ Generation successful")
            print(f"Response: {result['response'][:200]}...")
        else:
            print(f"❌ Generation failed: {result['error']}")
    
    # Test connection
    connection_ok = gemini.test_connection()
    print(f"Connection test: {'✅ OK' if connection_ok else '❌ Failed'}")

def test_gemini_via_unified():
    """Test Gemini via UnifiedAIAgent"""
    print("\n🧪 Testing Gemini via UnifiedAIAgent...")
    
    api_key = input("Nhập Gemini API key (hoặc Enter để test offline): ").strip()
    
    if api_key:
        agent = UnifiedAIAgent(gemini_api_key=api_key, preferred_model="gemini")
    else:
        agent = UnifiedAIAgent(preferred_model="gemini")
    
    print(f"Available models: {list(agent.available_models.keys())}")
    print(f"Offline mode: {agent.offline_mode}")
    
    # Test generation
    result = agent.generate_with_fallback(
        "Phân tích VCB có nên mua không?", 
        "financial_advice"
    )
    
    print(f"Generation success: {result['success']}")
    print(f"Model used: {result.get('model_used', 'unknown')}")
    print(f"Response preview: {result['response'][:200]}...")

def test_gemini_models():
    """Test different Gemini models"""
    print("\n🧪 Testing Different Gemini Models...")
    
    api_key = input("Nhập Gemini API key để test models (hoặc Enter để skip): ").strip()
    
    if not api_key:
        print("⚠️ Skipping model test - no API key provided")
        return
    
    models_to_test = [
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro-latest", 
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-pro"
    ]
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            
            response = model.generate_content("Hello")
            print(f"✅ {model_name}: Working")
            
        except Exception as e:
            print(f"❌ {model_name}: {str(e)[:100]}")

def main():
    print("🚀 Gemini LLM Test Suite")
    print("=" * 50)
    
    # Test 1: Direct Gemini LLM
    test_gemini_direct()
    
    # Test 2: Via UnifiedAIAgent
    test_gemini_via_unified()
    
    # Test 3: Different models
    test_gemini_models()
    
    print("\n" + "=" * 50)
    print("✅ Gemini test completed!")
    print("\n💡 Lưu ý:")
    print("- Gemini API key miễn phí tại: https://aistudio.google.com/apikey")
    print("- Quota reset sau 24h nếu hết")
    print("- Hệ thống sẽ fallback offline nếu API không khả dụng")

if __name__ == "__main__":
    main()