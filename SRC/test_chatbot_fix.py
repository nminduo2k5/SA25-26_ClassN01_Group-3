#!/usr/bin/env python3
"""
Test chatbot fix - should use Llama instead of offline mode
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_agent import MainAgent
from src.data.vn_stock_api import VNStockAPI
import asyncio

async def test_chatbot_without_api_keys():
    """Test chatbot without API keys - should use Llama"""
    print("Testing Chatbot without API keys...")
    print("=" * 40)
    
    # Initialize without API keys
    vn_api = VNStockAPI()
    main_agent = MainAgent(vn_api)
    
    print(f"Gemini agent created: {main_agent.gemini_agent is not None}")
    
    if main_agent.gemini_agent:
        available_models = main_agent.gemini_agent.available_models
        print(f"Available models: {list(available_models.keys())}")
        
        # Test query processing
        result = await main_agent.process_query("VCB có nên mua không?", "VCB")
        
        print(f"Query success: {not result.get('error')}")
        
        if not result.get('error'):
            advice = result.get('expert_advice', '')
            print(f"Response preview: {advice[:100]}...")
            
            # Check if it's using a real model or offline mode
            if 'OFFLINE' in advice.upper():
                print("❌ Still using offline mode")
            elif 'llama' in advice.lower() or len(advice) > 50:
                print("✅ Using real AI model")
            else:
                print("⚠️ Unclear model usage")
        else:
            print(f"Error: {result['error']}")
    else:
        print("❌ No gemini agent created")

def main():
    print("Chatbot Fix Test")
    print("=" * 20)
    
    asyncio.run(test_chatbot_without_api_keys())
    
    print("\n💡 Expected behavior:")
    print("- Should detect Llama if available")
    print("- Should use Llama for chatbot responses")
    print("- Should NOT show 'OFFLINE_MODE' if Llama works")

if __name__ == "__main__":
    main()