"""
News Dashboard Module - Real-time Market News
Fetches news from system's news agents
"""

import streamlit as st
from datetime import datetime
import asyncio

def display_news_dashboard(vn_api, symbol=None, is_english=False, dark_mode=False):
    """Display real-time news dashboard from news agents"""
    
    current_time = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    # Dark mode colors
    if dark_mode:
        text_color = "#e0e0e0"
        card_bg = "#2d2d2d"
    else:
        text_color = "#333333"
        card_bg = "white"
    
    # Header
    st.markdown(
        f"""
        <h1 style='text-align: center; color: #1e88e5; margin-bottom: 0;'>
            📰 {"Tin tức chứng khoán" if not is_english else "Stock Market News"}
        </h1>
        <p style='text-align: center; color: #757575; font-size: 1.1rem;'>
            {"Phân tích · Góc nhìn · Cập nhật nóng thị trường Việt Nam" if not is_english else "Analysis · Insights · Hot Updates from Vietnam Market"}
        </p>
        """,
        unsafe_allow_html=True
    )
    
    st.caption(f"{"Cập nhật lúc" if not is_english else "Updated at"} {current_time}")
    st.markdown("---")
    
    # Fetch real news data
    with st.spinner("⏳ " + ("Đang tải tin tức..." if not is_english else "Loading news...")):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if symbol:
            # Get news for specific symbol
            news_data = loop.run_until_complete(vn_api.get_news_sentiment(symbol))
            news_items = []
            
            if news_data and not news_data.get('error'):
                headlines = news_data.get('headlines', [])
                summaries = news_data.get('summaries', [])
                sentiment = news_data.get('sentiment', 'Neutral')
                
                for i, headline in enumerate(headlines[:10]):
                    news_items.append({
                        'title': headline,
                        'desc': summaries[i] if i < len(summaries) else "",
                        'category': f"{symbol} - {sentiment}",
                        'time': news_data.get('timestamp', current_time),
                        'sentiment': sentiment
                    })
        else:
            # Get market news
            market_data = loop.run_until_complete(vn_api.get_market_overview())
            market_news = market_data.get('market_news', {})
            
            # Get top gainers/losers for news context
            top_gainers = market_data.get('top_gainers', [])[:5]
            top_losers = market_data.get('top_losers', [])[:5]
            
            news_items = []
            
            # Market overview news
            if market_news:
                news_items.append({
                    'title': "📊 " + ("Tổng quan thị trường hôm nay" if not is_english else "Today's Market Overview"),
                    'desc': market_news.get('overview', 'Thị trường ổn định'),
                    'category': "Tổng quan" if not is_english else "Overview",
                    'time': market_news.get('timestamp', current_time),
                    'highlight': True
                })
            
            # Top gainers news
            for stock in top_gainers:
                news_items.append({
                    'title': f"🚀 {stock.get('symbol', 'N/A')} tăng mạnh {stock.get('change_percent', 0):+.2f}%",
                    'desc': f"Cổ phiếu {stock.get('symbol')} đang có xu hướng tăng tích cực trong phiên giao dịch hôm nay",
                    'category': "Top tăng giá" if not is_english else "Top Gainers",
                    'time': current_time,
                    'sentiment': 'Positive'
                })
            
            # Top losers news
            for stock in top_losers:
                news_items.append({
                    'title': f"📉 {stock.get('symbol', 'N/A')} giảm {stock.get('change_percent', 0):.2f}%",
                    'desc': f"Cổ phiếu {stock.get('symbol')} đang chịu áp lực bán trong phiên giao dịch hôm nay",
                    'category': "Top giảm giá" if not is_english else "Top Losers",
                    'time': current_time,
                    'sentiment': 'Negative'
                })
        
        loop.close()
    
    # Display news grid
    if not news_items:
        st.info("📭 " + ("Không có tin tức mới" if not is_english else "No news available"))
        return
    
    st.subheader("📌 " + ("Tin nổi bật & Góc nhìn mới nhất" if not is_english else "Featured News & Latest Insights"))
    
    # Create responsive grid
    cols_per_row = 5
    for i in range(0, len(news_items), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, news in enumerate(news_items[i:i+cols_per_row]):
            with cols[j]:
                # Determine colors based on sentiment
                if news.get('sentiment') == 'Positive':
                    category_color = "#26a526"
                    gradient = "linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%)"
                elif news.get('sentiment') == 'Negative':
                    category_color = "#d32f2f"
                    gradient = "linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%)"
                else:
                    category_color = "#1e88e5"
                    gradient = "linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%)"
                
                # News card
                card_html = f"""
                <div style="
                    background: {gradient if not dark_mode else card_bg};
                    border-radius: 12px;
                    padding: 16px;
                    margin-bottom: 16px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    min-height: 220px;
                    border: 2px solid {category_color};
                    transition: transform 0.2s;
                ">
                    <div style="
                        background: {category_color};
                        color: white;
                        padding: 4px 10px;
                        border-radius: 12px;
                        font-size: 0.75rem;
                        font-weight: bold;
                        display: inline-block;
                        margin-bottom: 10px;
                    ">
                        {news['category']}
                    </div>
                    
                    {f'<div style="background: #ff9800; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: bold; display: inline-block; margin-left: 8px;">HOT</div>' if news.get('highlight') else ''}
                    
                    <h4 style="
                        margin: 10px 0;
                        font-size: 1.05rem;
                        line-height: 1.3;
                        font-weight: 600;
                        color: {text_color};
                    ">
                        {news['title'][:80]}{'...' if len(news['title']) > 80 else ''}
                    </h4>
                    
                    <p style="
                        margin: 8px 0;
                        font-size: 0.85rem;
                        color: {text_color};
                        opacity: 0.8;
                        line-height: 1.4;
                    ">
                        {news['desc'][:100].replace('**', '').replace('*', '')}{'...' if len(news['desc']) > 100 else ''}
                    </p>
                    
                    <div style="
                        margin-top: 12px;
                        font-size: 0.75rem;
                        color: {text_color};
                        opacity: 0.6;
                    ">
                        🕐 {news.get('time', current_time) if isinstance(news.get('time'), str) and len(news.get('time', '')) < 30 else current_time}
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <p style='text-align: center; color: #616161; font-size: 0.9rem;'>
            © 2025 {"Hệ thống phân tích chứng khoán" if not is_english else "Stock Analysis System"} • 
            {"Dữ liệu từ VNStock API & News Agents" if not is_english else "Data from VNStock API & News Agents"}
        </p>
        """,
        unsafe_allow_html=True
    )
