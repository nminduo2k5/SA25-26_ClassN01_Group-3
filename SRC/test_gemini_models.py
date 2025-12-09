#!/usr/bin/env python3
"""
Test script to check available Gemini models
"""

import google.generativeai as genai
import os

def test_gemini_models():
    """Test all available Gemini models"""
    
    # Get API key from user input
    api_key = input("Enter your Gemini API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("No API key provided. Testing offline mode...")
        return test_offline_mode()
    
    try:
        genai.configure(api_key=api_key)
        print("✅ API key configured successfully")
    except Exception as e:
        print(f"❌ Failed to configure API key: {e}")
        return
    
    # Updated model list (2024-2025)
    model_names = [
        'gemini-2.0-flash-exp',         # Experimental 2.0 (Dec 2024)
        'gemini-2.0-flash-thinking-exp', # Thinking mode (Jan 2025)
        'gemini-1.5-flash',             # Stable production
        'gemini-1.5-flash-8b',          # Lightweight fast
        'gemini-1.5-flash-002',         # Latest 1.5 flash
        'gemini-1.5-pro',               # Pro version
        'gemini-1.5-pro-002',           # Latest 1.5 pro
        'gemini-1.0-pro',               # Legacy fallback
        'gemini-1.0-pro-001'            # Legacy with version
    ]
    
    print(f"\n🧪 Testing {len(model_names)} Gemini models...")
    print("=" * 60)
    
    working_models = []
    failed_models = []
    
    for i, model_name in enumerate(model_names, 1):
        print(f"\n[{i}/{len(model_names)}] Testing: {model_name}")
        
        try:
            model = genai.GenerativeModel(model_name)
            
            # Test with simple prompt
            response = model.generate_content("Hello, test message")
            
            if response and response.text:
                print(f"✅ {model_name}: WORKING")
                print(f"   Response: {response.text[:50]}...")
                working_models.append(model_name)
            else:
                print(f"❌ {model_name}: Empty response")
                failed_models.append((model_name, "Empty response"))
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ {model_name}: {error_msg[:100]}...")
            failed_models.append((model_name, error_msg))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"✅ Working models: {len(working_models)}")
    for model in working_models:
        print(f"   - {model}")
    
    print(f"\n❌ Failed models: {len(failed_models)}")
    for model, error in failed_models:
        print(f"   - {model}: {error[:50]}...")
    
    if working_models:
        print(f"\n🎯 RECOMMENDED: Use {working_models[0]} (first working model)")
    else:
        print("\n⚠️ No models working - check API key or quota")

def test_offline_mode():
    """Test offline mode functionality"""
    print("\n🔄 Testing offline mode...")
    
    try:
        from gemini_agent import UnifiedAIAgent
        
        # Test without API key
        agent = UnifiedAIAgent()
        print("✅ Offline agent created successfully")
        
        # Test offline response
        result = agent.generate_with_fallback(
            "Phân tích cổ phiếu VCB", 
            "financial_advice"
        )
        
        if result['success']:
            print("✅ Offline response generated")
            print(f"📝 Model: {result['model_used']}")
            print(f"📄 Preview: {result['response'][:100]}...")
        else:
            print(f"❌ Offline failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Offline test failed: {e}")

def main():
    """Main test function"""
    print("🚀 Gemini Models Test Suite")
    print("Testing latest Google Gemini models (2024-2025)")
    
    choice = input("\nChoose test mode:\n1. Test with API key\n2. Test offline mode only\nEnter choice (1/2): ").strip()
    
    if choice == "2":
        test_offline_mode()
    else:
        test_gemini_models()

if __name__ == "__main__":
    main()