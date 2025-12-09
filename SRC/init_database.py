#!/usr/bin/env python3
"""
Initialize and populate database for Duong AI Trading Pro
"""

import sqlite3
import json
from datetime import datetime, timedelta
from src.data.sqlite_manager import SQLiteManager

def init_and_populate_database():
    """Initialize database and add sample data"""
    print("Initializing Duong AI Trading Pro Database...")
    
    # Initialize database
    db = SQLiteManager("duong_trading.db")
    
    # Check current state
    stats = db.get_stats()
    print(f"Current database stats:")
    print(f"   - Total analyses: {stats.get('total_analyses', 0)}")
    print(f"   - Active cache entries: {stats.get('active_cache_entries', 0)}")
    print(f"   - Database size: {stats.get('database_size', 0)} bytes")
    
    # Add sample analysis data if empty
    if stats.get('total_analyses', 0) == 0:
        print("Adding sample analysis data...")
        
        sample_analyses = [
            {
                "symbol": "VCB",
                "analysis_type": "comprehensive",
                "result": {
                    "predicted_price": 87500,
                    "current_price": 85000,
                    "change_percent": 2.94,
                    "confidence": 85,
                    "trend": "bullish",
                    "method_used": "LSTM Primary",
                    "risk_level": "MEDIUM"
                },
                "risk_tolerance": 60,
                "time_horizon": "Trung hạn",
                "investment_amount": 100000000
            },
            {
                "symbol": "BID",
                "analysis_type": "price_prediction",
                "result": {
                    "predicted_price": 52300,
                    "current_price": 51000,
                    "change_percent": 2.55,
                    "confidence": 78,
                    "trend": "bullish",
                    "method_used": "Technical Analysis",
                    "risk_level": "LOW"
                },
                "risk_tolerance": 40,
                "time_horizon": "Dài hạn",
                "investment_amount": 50000000
            },
            {
                "symbol": "CTG",
                "analysis_type": "risk_assessment",
                "result": {
                    "predicted_price": 34200,
                    "current_price": 34500,
                    "change_percent": -0.87,
                    "confidence": 72,
                    "trend": "neutral",
                    "method_used": "LSTM Enhanced",
                    "risk_level": "MEDIUM"
                },
                "risk_tolerance": 70,
                "time_horizon": "Ngắn hạn",
                "investment_amount": 200000000
            }
        ]
        
        for analysis in sample_analyses:
            success = db.save_analysis(
                symbol=analysis["symbol"],
                analysis_type=analysis["analysis_type"],
                result=analysis["result"],
                risk_tolerance=analysis["risk_tolerance"],
                time_horizon=analysis["time_horizon"],
                investment_amount=analysis["investment_amount"]
            )
            if success:
                print(f"   Added sample analysis for {analysis['symbol']}")
    
    # Add sample cache data
    print("Adding sample cache data...")
    
    cache_samples = [
        {
            "key": "vn_symbols_list",
            "data": {
                "symbols": ["VCB", "BID", "CTG", "TCB", "ACB", "MBB", "VPB", "VIC", "VHM", "VRE"],
                "last_updated": datetime.now().isoformat(),
                "source": "VNStock"
            },
            "expires_minutes": 1440  # 24 hours
        },
        {
            "key": "market_status",
            "data": {
                "status": "OPEN",
                "session": "afternoon",
                "last_check": datetime.now().isoformat()
            },
            "expires_minutes": 60  # 1 hour
        },
        {
            "key": "lstm_model_cache_VCB",
            "data": {
                "model_trained": True,
                "confidence": 85.2,
                "last_training": datetime.now().isoformat(),
                "data_points": 1095
            },
            "expires_minutes": 1440  # 24 hours
        }
    ]
    
    for cache_item in cache_samples:
        success = db.cache_data(
            cache_key=cache_item["key"],
            data=cache_item["data"],
            expires_minutes=cache_item["expires_minutes"]
        )
        if success:
            print(f"   Cached {cache_item['key']}")
    
    # Set default user preferences
    print("Setting default user preferences...")
    
    default_prefs = {
        "gemini_api_key": None,  # Will be set by user
        "serper_api_key": None,  # Optional
        "default_risk_tolerance": 50,
        "default_time_horizon": "Trung hạn",
        "default_investment_amount": 100000000
    }
    
    if db.save_user_preferences(default_prefs):
        print("   Default preferences saved")
    
    # Final stats
    final_stats = db.get_stats()
    print(f"\nFinal database stats:")
    print(f"   - Total analyses: {final_stats.get('total_analyses', 0)}")
    print(f"   - Top symbols: {final_stats.get('top_symbols', [])}")
    print(f"   - Active cache entries: {final_stats.get('active_cache_entries', 0)}")
    print(f"   - Database size: {final_stats.get('database_size', 0)} bytes")
    
    print("\nDatabase initialization completed!")
    return db

def test_database_operations():
    """Test database operations"""
    print("\nTesting database operations...")
    
    db = SQLiteManager("duong_trading.db")
    
    # Test analysis history
    history = db.get_analysis_history(limit=5)
    print(f"Recent analyses: {len(history)} records")
    
    for record in history[:3]:  # Show first 3
        result = record.get('result', {})
        print(f"   - {record['symbol']}: {result.get('predicted_price', 'N/A')} VND ({result.get('trend', 'N/A')})")
    
    # Test cache retrieval
    cached_symbols = db.get_cached_data("vn_symbols_list")
    if cached_symbols:
        print(f"Cached symbols: {len(cached_symbols.get('symbols', []))} symbols")
    
    # Test user preferences
    prefs = db.get_user_preferences()
    if prefs:
        print(f"User preferences: {prefs.get('default_risk_tolerance', 50)}% risk tolerance")
    
    print("Database operations test completed!")

if __name__ == "__main__":
    # Initialize and populate database
    db = init_and_populate_database()
    
    # Test operations
    test_database_operations()
    
    print("\nDatabase is ready for Duong AI Trading Pro!")