#!/usr/bin/env python3
"""
Test script to verify Gemini API model compatibility
"""

import google.generativeai as genai
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gemini_models(api_key: str):
    """Test available Gemini models"""
    
    if not api_key:
        print("❌ No API key provided")
        return
    
    try:
        genai.configure(api_key=api_key)
        
        # Updated model list (API v1beta compatible)
        model_names = [
            'gemini-1.5-pro-latest',        # Latest stable pro
            'gemini-1.5-flash-latest',      # Latest stable flash
            'gemini-1.5-pro',               # Pro version
            'gemini-1.5-flash',             # Flash version
            'gemini-pro',                    # Legacy pro
            'gemini-1.0-pro-latest',        # Legacy latest
            'gemini-1.0-pro'                # Legacy fallback
        ]
        
        print("🔍 Testing Gemini models...")
        working_models = []
        
        for model_name in model_names:
            try:
                print(f"\n📋 Testing: {model_name}")
                model = genai.GenerativeModel(model_name)
                
                # Simple test without quota usage
                print(f"✅ Model {model_name} initialized successfully")
                working_models.append(model_name)
                
                # Optional: Test with actual generation (comment out to save quota)
                # response = model.generate_content("Hi")
                # if response and response.text:
                #     print(f"✅ Model {model_name} response: {response.text[:50]}...")
                
            except Exception as e:
                error_msg = str(e).lower()
                if '404' in error_msg or 'not found' in error_msg:
                    print(f"❌ Model {model_name} not found (404)")
                elif 'quota' in error_msg or '429' in error_msg:
                    print(f"⚠️ Model {model_name} quota exceeded")
                    working_models.append(f"{model_name} (quota exceeded)")
                else:
                    print(f"❌ Model {model_name} error: {e}")
        
        print(f"\n📊 Summary:")
        print(f"✅ Working models: {len([m for m in working_models if 'quota' not in m])}")
        print(f"⚠️ Quota exceeded: {len([m for m in working_models if 'quota' in m])}")
        print(f"❌ Failed models: {len(model_names) - len(working_models)}")
        
        if working_models:
            print(f"\n🎯 Recommended model: {working_models[0]}")
            return working_models[0]
        else:
            print("\n❌ No working models found")
            return None
            
    except Exception as e:
        print(f"❌ API configuration failed: {e}")
        return None

if __name__ == "__main__":
    # Test with your API key
    api_key = input("Enter your Gemini API key: ").strip()
    test_gemini_models(api_key)