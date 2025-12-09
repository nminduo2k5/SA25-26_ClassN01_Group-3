#!/usr/bin/env python3
"""
Test script for offline mode functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemini_agent import UnifiedAIAgent
from main_agent import MainAgent
from src.data.vn_stock_api import VNStockAPI

def test_offline_gemini():
    """Test Gemini agent in offline mode"""
    print("🧪 Testing Gemini Agent Offline Mode...")
    
    try:
        # Test with no API key (should work in offline mode)
        agent = UnifiedAIAgent()
        print("✅ Gemini agent created successfully in offline mode")
        
        # Test offline response
        result = agent.generate_with_fallback(
            "Phân tích cổ phiếu VCB có nên mua không?", 
            "financial_advice"
        )
        
        if result['success']:
            print("✅ Offline response generated successfully")
            print(f"📝 Model used: {result['model_used']}")
            print(f"📄 Response preview: {result['response'][:100]}...")
        else:
            print(f"❌ Offline response failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Gemini offline test failed: {e}")

def test_main_agent_offline():
    """Test MainAgent in offline mode"""
    print("\n🧪 Testing MainAgent Offline Mode...")
    
    try:
        # Initialize VN API
        vn_api = VNStockAPI()
        
        # Create MainAgent without API key (should work in offline mode)
        main_agent = MainAgent(vn_api)
        print("✅ MainAgent created successfully in offline mode")
        
        # Test query processing
        import asyncio
        
        async def test_query():
            result = await main_agent.process_query("VCB có nên mua không?", "VCB")
            return result
        
        result = asyncio.run(test_query())
        
        if not result.get('error'):
            print("✅ Query processed successfully in offline mode")
            print(f"📝 Response type: {result.get('response_type')}")
            advice = result.get('expert_advice', '')
            print(f"📄 Advice preview: {advice[:100]}...")
        else:
            print(f"❌ Query processing failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ MainAgent offline test failed: {e}")

def test_with_invalid_api_key():
    """Test with invalid API key (should fallback to offline)"""
    print("\n🧪 Testing Invalid API Key Fallback...")
    
    try:
        # Test with invalid API key
        agent = UnifiedAIAgent(gemini_api_key="invalid_key_12345")
        print("✅ Agent created with invalid key (should use offline mode)")
        
        # Test response
        result = agent.generate_with_fallback(
            "Phân tích thị trường hôm nay", 
            "general_query"
        )
        
        if result['success']:
            print("✅ Fallback response generated successfully")
            print(f"📝 Model used: {result['model_used']}")
            if result.get('quota_exceeded'):
                print("📊 Quota exceeded detected - using offline mode")
        else:
            print(f"❌ Fallback failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Invalid API key test failed: {e}")

def main():
    """Run all offline mode tests"""
    print("🚀 Starting Offline Mode Tests...")
    print("=" * 50)
    
    # Test 1: Pure offline mode
    test_offline_gemini()
    
    # Test 2: MainAgent offline mode
    test_main_agent_offline()
    
    # Test 3: Invalid API key fallback
    test_with_invalid_api_key()
    
    print("\n" + "=" * 50)
    print("✅ All offline mode tests completed!")
    print("\n💡 Hệ thống có thể hoạt động offline khi:")
    print("   - Không có API key")
    print("   - API key không hợp lệ")
    print("   - Hết quota Gemini API")
    print("   - Lỗi kết nối mạng")

if __name__ == "__main__":
    main()