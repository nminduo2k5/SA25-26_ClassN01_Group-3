#!/usr/bin/env python3
"""
Optimize Gemini usage for stock prediction system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm.unified_llm import UnifiedLLM

def test_optimized_prompts():
    """Test optimized prompts for stock analysis"""
    print("Testing Optimized Prompts for Stock Analysis")
    print("=" * 50)
    
    # Test with minimal API key
    api_key = input("Enter Gemini API key (or skip): ").strip()
    
    if not api_key:
        print("Skipping - no API key provided")
        return
    
    llm = UnifiedLLM(gemini_key=api_key, preferred="gemini")
    
    # Optimized prompts - short and focused
    test_prompts = [
        {
            "name": "Stock Analysis",
            "prompt": "VCB: BUY/SELL/HOLD? 1 sentence reason.",
            "max_tokens": 50
        },
        {
            "name": "Market Sentiment", 
            "prompt": "VN market today: Positive/Negative/Neutral + why (20 words)",
            "max_tokens": 30
        },
        {
            "name": "Price Prediction",
            "prompt": "VCB price next week: Up/Down/Flat + confidence %",
            "max_tokens": 25
        },
        {
            "name": "Risk Assessment",
            "prompt": "VCB risk level: Low/Medium/High + main risk factor",
            "max_tokens": 30
        }
    ]
    
    total_tokens = 0
    
    for test in test_prompts:
        print(f"\nTesting: {test['name']}")
        print(f"Prompt: {test['prompt']}")
        
        result = llm.generate(test['prompt'], test['max_tokens'])
        
        if result['success']:
            response = result['response']
            estimated_tokens = len(test['prompt'].split()) + len(response.split())
            total_tokens += estimated_tokens
            
            print(f"Response: {response}")
            print(f"Estimated tokens: {estimated_tokens}")
        else:
            print(f"Failed: {result['error']}")
    
    print(f"\nTotal estimated tokens used: {total_tokens}")
    print(f"Average per query: {total_tokens/len(test_prompts):.1f}")

def show_optimization_tips():
    """Show tips for optimizing Gemini usage"""
    print("\nGemini Optimization Tips:")
    print("=" * 30)
    
    print("\n1. Use Short Prompts:")
    print("   ❌ Bad: 'Analyze the comprehensive financial situation...'")
    print("   ✅ Good: 'VCB: BUY/SELL/HOLD?'")
    
    print("\n2. Limit Response Length:")
    print("   ❌ Bad: max_tokens=2000")
    print("   ✅ Good: max_tokens=50")
    
    print("\n3. Use Structured Outputs:")
    print("   ❌ Bad: 'Tell me about the stock'")
    print("   ✅ Good: 'JSON: {\"action\":\"BUY\", \"confidence\":80}'")
    
    print("\n4. Batch Similar Queries:")
    print("   ❌ Bad: 5 separate API calls")
    print("   ✅ Good: 'Analyze VCB,BID,CTG: BUY/SELL/HOLD for each'")
    
    print("\n5. Use Gemini Flash Model:")
    print("   ❌ Bad: gemini-1.5-pro (expensive)")
    print("   ✅ Good: gemini-1.5-flash (cheaper, faster)")
    
    print("\n6. Cache Results:")
    print("   ❌ Bad: Re-analyze same stock every minute")
    print("   ✅ Good: Cache for 15-30 minutes")

def estimate_daily_usage():
    """Estimate daily token usage"""
    print("\nDaily Usage Estimation:")
    print("=" * 25)
    
    scenarios = {
        "Light User": {
            "stock_analyses": 10,
            "market_checks": 5,
            "news_summaries": 3,
            "tokens_per_query": 50
        },
        "Regular User": {
            "stock_analyses": 50,
            "market_checks": 20,
            "news_summaries": 10,
            "tokens_per_query": 75
        },
        "Heavy User": {
            "stock_analyses": 200,
            "market_checks": 50,
            "news_summaries": 30,
            "tokens_per_query": 100
        }
    }
    
    for user_type, usage in scenarios.items():
        total_queries = sum([usage[k] for k in usage if k != 'tokens_per_query'])
        total_tokens = total_queries * usage['tokens_per_query']
        
        print(f"\n{user_type}:")
        print(f"  Queries/day: {total_queries}")
        print(f"  Tokens/day: {total_tokens:,}")
        print(f"  Monthly: {total_tokens * 30:,}")
        
        # Gemini free tier: 15 requests/minute, 1M tokens/month
        if total_tokens * 30 > 1000000:
            print(f"  ⚠️ Exceeds free tier (1M/month)")
        else:
            print(f"  ✅ Within free tier")

def main():
    print("🔧 Gemini Usage Optimizer")
    print("=" * 30)
    
    # Test optimized prompts
    test_optimized_prompts()
    
    # Show optimization tips
    show_optimization_tips()
    
    # Estimate usage
    estimate_daily_usage()
    
    print("\n💡 Key Takeaways:")
    print("- Use gemini-1.5-flash model")
    print("- Keep prompts under 20 words")
    print("- Limit responses to 50 tokens")
    print("- Cache results for 15-30 minutes")
    print("- Batch multiple queries together")

if __name__ == "__main__":
    main()