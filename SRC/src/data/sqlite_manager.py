# src/data/sqlite_manager.py
"""
SQLite Database Manager for Duong AI Trading Pro
Minimal implementation for storing analysis history and user preferences
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class SQLiteManager:
    """Minimal SQLite manager for Duong AI Trading Pro"""
    
    def __init__(self, db_path: str = "duong_trading.db"):
        self.db_path = Path(db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Analysis history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        analysis_type TEXT NOT NULL,
                        result TEXT NOT NULL,
                        risk_tolerance INTEGER,
                        time_horizon TEXT,
                        investment_amount INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # User preferences table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        gemini_api_key TEXT,
                        serper_api_key TEXT,
                        default_risk_tolerance INTEGER DEFAULT 50,
                        default_time_horizon TEXT DEFAULT 'Trung hạn',
                        default_investment_amount INTEGER DEFAULT 100000000,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Cache table for API responses
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cache_key TEXT UNIQUE NOT NULL,
                        data TEXT NOT NULL,
                        expires_at DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("✅ SQLite database initialized successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
    
    def save_analysis(self, symbol: str, analysis_type: str, result: Dict[str, Any], 
                     risk_tolerance: int = 50, time_horizon: str = "Trung hạn", 
                     investment_amount: int = 100000000) -> bool:
        """Save analysis result to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO analysis_history 
                    (symbol, analysis_type, result, risk_tolerance, time_horizon, investment_amount)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (symbol, analysis_type, json.dumps(result, default=str), risk_tolerance, time_horizon, investment_amount))
                conn.commit()
                print(f"✅ Saved analysis for {symbol} to database")
                return True
        except Exception as e:
            print(f"❌ Failed to save analysis: {e}")
            print(f"❌ Error type: {type(e)}")
            print(f"❌ Error details: {str(e)}")
            logger.error(f"❌ Failed to save analysis: {e}")
            return False
    
    def get_analysis_history(self, symbol: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get analysis history from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if symbol:
                    cursor.execute("""
                        SELECT * FROM analysis_history 
                        WHERE symbol = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (symbol, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM analysis_history 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                history = []
                for row in rows:
                    record = dict(zip(columns, row))
                    record['result'] = json.loads(record['result'])
                    history.append(record)
                
                return history
        except Exception as e:
            logger.error(f"❌ Failed to get analysis history: {e}")
            return []
    
    def save_user_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Save user preferences to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if preferences exist
                cursor.execute("SELECT COUNT(*) FROM user_preferences")
                count = cursor.fetchone()[0]
                
                if count > 0:
                    # Update existing preferences
                    cursor.execute("""
                        UPDATE user_preferences SET
                        gemini_api_key = ?,
                        serper_api_key = ?,
                        default_risk_tolerance = ?,
                        default_time_horizon = ?,
                        default_investment_amount = ?,
                        updated_at = CURRENT_TIMESTAMP
                        WHERE id = 1
                    """, (
                        preferences.get('gemini_api_key'),
                        preferences.get('serper_api_key'),
                        preferences.get('default_risk_tolerance', 50),
                        preferences.get('default_time_horizon', 'Trung hạn'),
                        preferences.get('default_investment_amount', 100000000)
                    ))
                else:
                    # Insert new preferences
                    cursor.execute("""
                        INSERT INTO user_preferences 
                        (gemini_api_key, serper_api_key, default_risk_tolerance, 
                         default_time_horizon, default_investment_amount)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        preferences.get('gemini_api_key'),
                        preferences.get('serper_api_key'),
                        preferences.get('default_risk_tolerance', 50),
                        preferences.get('default_time_horizon', 'Trung hạn'),
                        preferences.get('default_investment_amount', 100000000)
                    ))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to save user preferences: {e}")
            return False
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM user_preferences ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                else:
                    return {}
        except Exception as e:
            logger.error(f"❌ Failed to get user preferences: {e}")
            return {}
    
    def cache_data(self, cache_key: str, data: Any, expires_minutes: int = 60) -> bool:
        """Cache data with expiration"""
        try:
            from datetime import timedelta
            expires_at = datetime.now() + timedelta(minutes=expires_minutes)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO api_cache (cache_key, data, expires_at)
                    VALUES (?, ?, ?)
                """, (cache_key, json.dumps(data), expires_at))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to cache data: {e}")
            return False
    
    def get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get cached data if not expired"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT data FROM api_cache 
                    WHERE cache_key = ? AND expires_at > CURRENT_TIMESTAMP
                """, (cache_key,))
                row = cursor.fetchone()
                
                if row:
                    return json.loads(row[0])
                return None
        except Exception as e:
            logger.error(f"❌ Failed to get cached data: {e}")
            return None
    
    def clear_expired_cache(self) -> bool:
        """Clear expired cache entries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM api_cache WHERE expires_at <= CURRENT_TIMESTAMP")
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to clear expired cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Analysis count
                cursor.execute("SELECT COUNT(*) FROM analysis_history")
                analysis_count = cursor.fetchone()[0]
                
                # Most analyzed symbols
                cursor.execute("""
                    SELECT symbol, COUNT(*) as count 
                    FROM analysis_history 
                    GROUP BY symbol 
                    ORDER BY count DESC 
                    LIMIT 5
                """)
                top_symbols = cursor.fetchall()
                
                # Cache stats
                cursor.execute("SELECT COUNT(*) FROM api_cache WHERE expires_at > CURRENT_TIMESTAMP")
                active_cache = cursor.fetchone()[0]
                
                return {
                    'total_analyses': analysis_count,
                    'top_symbols': [{'symbol': s[0], 'count': s[1]} for s in top_symbols],
                    'active_cache_entries': active_cache,
                    'database_size': self.db_path.stat().st_size if self.db_path.exists() else 0
                }
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {}