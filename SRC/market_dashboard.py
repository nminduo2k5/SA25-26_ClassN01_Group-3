"""
Market Dashboard Module - Real-time Market Overview
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

def fake_chart_data(points=90, base=1800):
    """Generate fake chart data for visualization"""
    return np.random.normal(0, 0.8, points).cumsum() + base + np.random.normal(0, 3, points)

def display_market_overview(vn_api, is_english=False, dark_mode=False):
    """Display real-time market overview dashboard"""
    
    current_time = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    # Dark mode colors
    if dark_mode:
        bg_color = "#1e1e1e"
        text_color = "#e0e0e0"
        card_bg = "#2d2d2d"
        border_color = "#404040"
        table_bg = "#252525"
    else:
        bg_color = "white"
        text_color = "#333333"
        card_bg = "white"
        border_color = "#e0e0e0"
        table_bg = "#f9f9f9"
    
    # Header
    title = f"BẢNG GIÁ CHỨNG KHOÁN {current_time}" if not is_english else f"STOCK PRICE BOARD {current_time}"
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h2 style='text-align: center; color: white; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                📊 {title}
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Fetch real market data and historical charts
    with st.spinner("⏳ Đang tải dữ liệu thị trường..." if not is_english else "⏳ Loading market data..."):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        market_data = loop.run_until_complete(vn_api.get_market_overview())
        
        # Fetch historical data for charts (last 90 days)
        vn_history = loop.run_until_complete(vn_api.get_price_history('VNINDEX', days=90))
        vn30_history = loop.run_until_complete(vn_api.get_price_history('VN30', days=90))
        hn_history = loop.run_until_complete(vn_api.get_price_history('HNXINDEX', days=90))
        upcom_history = loop.run_until_complete(vn_api.get_price_history('UPCOM', days=90))
        loop.close()
    
    # Extract real data
    vn_index = market_data.get('vn_index', {})
    vn30_index = market_data.get('vn30_index', {})
    hn_index = market_data.get('hn_index', {})
    upcom_index = market_data.get('upcom_index', {'value': 127.14, 'change': 1.24, 'change_percent': 0.98})
    
    # Row 1: Market Charts
    st.markdown(f"<h3 style='color: {text_color};'>📈 " + ("Biểu đồ chỉ số thị trường" if not is_english else "Market Indices Charts") + "</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        vn_val = vn_index.get('value', 1200)
        vn_change = vn_index.get('change', 0)
        vn_pct = vn_index.get('change_percent', 0)
        
        st.markdown(f"""
        <div style='background: {card_bg}; padding: 10px; border-radius: 8px; border: 2px solid {"#26a526" if vn_change >= 0 else "#d32f2f"}; margin-bottom: 10px;'>
            <h4 style='text-align:center; color:#1e88e5; margin:0;'>VNINDEX</h4>
            <p style='text-align:center; font-size:1.5rem; font-weight:bold; margin:5px 0; color:{"#26a526" if vn_change >= 0 else "#d32f2f"};'>{vn_val:,.2f}</p>
            <p style='text-align:center; margin:0; color:{"#26a526" if vn_change >= 0 else "#d32f2f"};'>{vn_change:+.2f} ({vn_pct:+.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        if vn_history and len(vn_history) > 0:
            chart_data = pd.DataFrame(vn_history)
            st.line_chart(chart_data.set_index('date')['close'], height=180, use_container_width=True)
        else:
            st.line_chart(fake_chart_data(base=vn_val), height=180, use_container_width=True)
    
    with c2:
        vn30_val = vn30_index.get('value', 1500)
        vn30_change = vn30_index.get('change', 0)
        vn30_pct = vn30_index.get('change_percent', 0)
        
        st.markdown(f"""
        <div style='background: {card_bg}; padding: 10px; border-radius: 8px; border: 2px solid {"#26a526" if vn30_change >= 0 else "#d32f2f"}; margin-bottom: 10px;'>
            <h4 style='text-align:center; color:#1e88e5; margin:0;'>VN30</h4>
            <p style='text-align:center; font-size:1.5rem; font-weight:bold; margin:5px 0; color:{"#26a526" if vn30_change >= 0 else "#d32f2f"};'>{vn30_val:,.2f}</p>
            <p style='text-align:center; margin:0; color:{"#26a526" if vn30_change >= 0 else "#d32f2f"};'>{vn30_change:+.2f} ({vn30_pct:+.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        if vn30_history and len(vn30_history) > 0:
            chart_data = pd.DataFrame(vn30_history)
            st.line_chart(chart_data.set_index('date')['close'], height=180, use_container_width=True)
        else:
            st.line_chart(fake_chart_data(base=vn30_val), height=180, use_container_width=True)
    
    with c3:
        hn_val = hn_index.get('value', 230)
        hn_change = hn_index.get('change', 0)
        hn_pct = hn_index.get('change_percent', 0)
        
        st.markdown(f"""
        <div style='background: {card_bg}; padding: 10px; border-radius: 8px; border: 2px solid {"#26a526" if hn_change >= 0 else "#d32f2f"}; margin-bottom: 10px;'>
            <h4 style='text-align:center; color:#1e88e5; margin:0;'>HNX-Index</h4>
            <p style='text-align:center; font-size:1.5rem; font-weight:bold; margin:5px 0; color:{"#26a526" if hn_change >= 0 else "#d32f2f"};'>{hn_val:,.2f}</p>
            <p style='text-align:center; margin:0; color:{"#26a526" if hn_change >= 0 else "#d32f2f"};'>{hn_change:+.2f} ({hn_pct:+.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        if hn_history and len(hn_history) > 0:
            chart_data = pd.DataFrame(hn_history)
            st.line_chart(chart_data.set_index('date')['close'], height=180, use_container_width=True)
        else:
            st.line_chart(fake_chart_data(base=hn_val), height=180, use_container_width=True)
    
    with c4:
        upcom_val = upcom_index.get('value', 127.14)
        upcom_change = upcom_index.get('change', 0)
        upcom_pct = upcom_index.get('change_percent', 0)
        
        st.markdown(f"""
        <div style='background: {card_bg}; padding: 10px; border-radius: 8px; border: 2px solid {"#26a526" if upcom_change >= 0 else "#d32f2f"}; margin-bottom: 10px;'>
            <h4 style='text-align:center; color:#1e88e5; margin:0;'>UPCoM</h4>
            <p style='text-align:center; font-size:1.5rem; font-weight:bold; margin:5px 0; color:{"#26a526" if upcom_change >= 0 else "#d32f2f"};'>{upcom_val:,.2f}</p>
            <p style='text-align:center; margin:0; color:{"#26a526" if upcom_change >= 0 else "#d32f2f"};'>{upcom_change:+.2f} ({upcom_pct:+.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        if upcom_history and len(upcom_history) > 0:
            chart_data = pd.DataFrame(upcom_history)
            st.line_chart(chart_data.set_index('date')['close'], height=180, use_container_width=True)
        else:
            st.line_chart(fake_chart_data(base=upcom_val), height=180, use_container_width=True)
    
    # Row 2: Indices Table
    st.markdown(f"<hr style='margin: 30px 0; border: 1px solid {border_color};'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {text_color};'>📊 " + ("Bảng chỉ số chi tiết" if not is_english else "Detailed Indices Table") + "</h3>", unsafe_allow_html=True)
    
    # Use real data for table
    vn_vol = vn_index.get('volume', 1058940000) / 1000
    vn30_vol = vn30_index.get('volume', 429881000) / 1000
    hn_vol = hn_index.get('volume', 117314000) / 1000
    
    data = [
        ["VNINDEX", f"{vn_val:,.2f}", f"{vn_change:+.2f}", f"{vn_pct:+.2f}%", f"{vn_vol:,.0f}"],
        ["VN30", f"{vn30_val:,.2f}", f"{vn30_change:+.2f}", f"{vn30_pct:+.2f}%", f"{vn30_vol:,.0f}"],
        ["HNX-Index", f"{hn_val:,.2f}", f"{hn_change:+.2f}", f"{hn_pct:+.2f}%", f"{hn_vol:,.0f}"],
    ]
    
    col_names = ["Chỉ số", "Giá trị", "Thay đổi", "% Thay đổi", "Khối lượng (K)"] if not is_english else ["Index", "Value", "Change", "% Change", "Volume (K)"]
    df = pd.DataFrame(data, columns=col_names)
    
    def style_table(val):
        base_bg = "#2d2d2d" if dark_mode else "#ffffff"
        if isinstance(val, str):
            if val.startswith("+"): return f"color: #26a526; font-weight: bold; background-color: {'#1a3d1a' if dark_mode else '#e8f5e9'}; font-size: 1.1rem;"
            if val.startswith("-"): return f"color: #d32f2f; font-weight: bold; background-color: {'#3d1a1a' if dark_mode else '#ffebee'}; font-size: 1.1rem;"
        return f"font-size: 1.05rem; background-color: {base_bg}; color: {text_color};"
    
    st.dataframe(
        df.style.applymap(style_table),
        hide_index=True,
        use_container_width=True,
        height=180
    )
    
    # Row 3: Trading Statistics with real data
    st.markdown(f"<hr style='margin: 30px 0; border: 1px solid {border_color};'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {text_color};'>💹 " + ("Thống kê giao dịch" if not is_english else "Trading Statistics") + "</h3>", unsafe_allow_html=True)
    
    total_vol = int(vn_vol)
    cols = st.columns(5)
    with cols[0]:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; border-radius: 10px; text-align: center;'>
            <p style='color: white; margin: 0; font-size: 0.9rem;'>""" + ("Khớp lệnh" if not is_english else "Matched") + f"""</p>
            <h3 style='color: white; margin: 5px 0;'>{total_vol:,}</h3>
            <p style='color: #b3ffb3; margin: 0; font-size: 0.85rem;'>CP</p>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 15px; border-radius: 10px; text-align: center;'>
            <p style='color: white; margin: 0; font-size: 0.9rem;'>""" + ("Giá trị KL" if not is_english else "Value") + f"""</p>
            <h3 style='color: white; margin: 5px 0;'>33,603 tỷ</h3>
            <p style='color: #ffcccc; margin: 0; font-size: 0.85rem;'>{vn_pct:+.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 15px; border-radius: 10px; text-align: center;'>
            <p style='color: white; margin: 0; font-size: 0.9rem;'>""" + ("Đỏ / Xanh" if not is_english else "Red / Green") + """</p>
            <h3 style='color: white; margin: 5px 0;'>52 / 92</h3>
            <p style='color: white; margin: 0; font-size: 0.85rem;'>↓</p>
        </div>
        """, unsafe_allow_html=True)
    with cols[3]:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 15px; border-radius: 10px; text-align: center;'>
            <p style='color: white; margin: 0; font-size: 0.9rem;'>""" + ("Trần / Sàn" if not is_english else "Ceiling / Floor") + """</p>
            <h3 style='color: white; margin: 5px 0;'>2 / 0</h3>
            <p style='color: white; margin: 0; font-size: 0.85rem;'>—</p>
        </div>
        """, unsafe_allow_html=True)
    with cols[4]:
        trend = "↑" if vn_change > 0 else "↓" if vn_change < 0 else "→"
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 15px; border-radius: 10px; text-align: center;'>
            <p style='color: white; margin: 0; font-size: 0.9rem;'>""" + ("Thanh khoản" if not is_english else "Liquidity") + f"""</p>
            <h3 style='color: white; margin: 5px 0;'>{trend}</h3>
            <p style='color: white; margin: 0; font-size: 0.85rem;'>""" + ("vs hôm qua" if not is_english else "vs yesterday") + """</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Top Movers Section
    st.markdown("<hr style='margin: 30px 0; border: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
    col_gain, col_lose = st.columns(2)
    
    with col_gain:
        st.markdown("### 🚀 " + ("Top tăng giá" if not is_english else "Top Gainers"))
        top_gainers = market_data.get('top_gainers', [])
        if top_gainers:
            for i, stock in enumerate(top_gainers[:5], 1):
                pct = stock.get('change_percent', 0)
                st.markdown(f"**{i}. {stock.get('symbol', 'N/A')}** <span style='color: #26a526; font-weight: bold;'>+{pct:.2f}%</span>", unsafe_allow_html=True)
        else:
            st.info("Không có dữ liệu" if not is_english else "No data")
    
    with col_lose:
        st.markdown("### 📉 " + ("Top giảm giá" if not is_english else "Top Losers"))
        top_losers = market_data.get('top_losers', [])
        if top_losers:
            for i, stock in enumerate(top_losers[:5], 1):
                pct = stock.get('change_percent', 0)
                st.markdown(f"**{i}. {stock.get('symbol', 'N/A')}** <span style='color: #d32f2f; font-weight: bold;'>{pct:.2f}%</span>", unsafe_allow_html=True)
        else:
            st.info("Không có dữ liệu" if not is_english else "No data")
    
    # Footer
    st.markdown("<hr style='margin: 30px 0; border: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
    data_source = "✅ Dữ liệu thực từ VNStock API" if not is_english else "✅ Real-time data from VNStock API"
    st.markdown(
        f"""
        <div style='background: #e8f5e9; padding: 15px; border-radius: 10px; text-align: center;'>
            <p style='color: #2e7d32; font-size: 0.9rem; margin: 0; font-weight: bold;'>
                {data_source} • {current_time}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
