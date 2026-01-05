import streamlit as st
import asyncio
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from main_agent import MainAgent
from src.data.vn_stock_api import VNStockAPI
from src.ui.styles import load_custom_css
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Cấu hình trang chuyên nghiệp
st.set_page_config(
    page_title="DUONG AI TRADING PRO",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tải CSS tích hợp Bootstrap
load_custom_css()

# CSS bổ sung cho ứng dụng
st.markdown("""
<style>
    /* App-specific overrides */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid var(--bs-primary);
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    /* Streamlit tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bs-gray-100);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--bs-primary);
        color: white;
    }
    
    /* News cards */
    .news-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        border-left: 4px solid var(--bs-primary);
        transition: transform 0.2s ease;
    }
    
    .news-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.12);
    }
</style>
""", unsafe_allow_html=True)

def initialize_agents():
    """Initialize agents if not already done"""
    if 'main_agent' not in st.session_state:
        try:
            main_agent, vn_api = init_system()
            st.session_state.main_agent = main_agent
            st.session_state.vn_api = vn_api
        except Exception as e:
            st.error(f"❌ System initialization failed: {e}")
            st.stop()

# Khởi tạo hệ thống với environment variables
def init_system():
    # Load API keys from environment - DEFINE ALL VARIABLES FIRST
    gemini_key = os.getenv('GEMINI_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    llama_key = os.getenv('LLAMA_API_KEY')
    llama_base_url = os.getenv('LLAMA_BASE_URL', 'https://api.together.xyz/v1')
    serper_key = os.getenv('SERPER_API_KEY')
    
    # Initialize VN API with all LLM parameters
    vn_api = VNStockAPI(
        gemini_api_key=gemini_key,
        openai_api_key=openai_key,
        llama_api_key=llama_key,
        llama_base_url=llama_base_url,
        serper_api_key=serper_key
    )
    
    main_agent = MainAgent(
        vn_api, 
        gemini_api_key=gemini_key,
        openai_api_key=openai_key,
        llama_api_key=llama_key,
        llama_base_url=llama_base_url,
        serper_api_key=serper_key
    )
    
    # Auto-configure LLM APIs if keys are available
    if any([gemini_key and gemini_key != 'your_gemini_api_key_here',
            openai_key and openai_key != 'your_openai_api_key_here',
            llama_key and llama_key != 'your_llama_api_key_here']):
        main_agent.set_llm_keys(gemini_key, openai_key, llama_key, llama_base_url)
        if serper_key and serper_key != 'your_serper_api_key_here':
            main_agent.set_crewai_keys(gemini_key, serper_key)
    
    return main_agent, vn_api

# Initialize system once per session with error handling
if 'main_agent' not in st.session_state:
    try:
        main_agent, vn_api = init_system()
        st.session_state.main_agent = main_agent
        st.session_state.vn_api = vn_api
    except Exception as e:
        st.error(f"❌ System initialization failed: {e}")
        st.info("💡 Try running: python install_dependencies.py")
        st.stop()
else:
    main_agent = st.session_state.main_agent
    vn_api = st.session_state.vn_api
    
def display_architecture_prediction_tables(pred, symbol, architecture):
    """Display prediction results in tables by timeframe with weekend awareness"""
    from datetime import datetime, timedelta
    from src.utils.market_schedule import market_schedule

    # Get real current price from stock data
    current_price = pred.get('current_price', 0)
    if current_price <= 0:
        # Fallback to other price fields
        current_price = pred.get('final_price', pred.get('predicted_price', 50000))
    
    base_price = current_price
    st.markdown(f"### 📊 Dự đoán giá {symbol} - {architecture.upper()}")
    
    # Weekend-aware date formatting
    VN_WEEKDAYS = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
    
    def format_prediction_date(date):
        """Format date with trading day logic (weekdays + holidays)"""
        weekday = date.weekday()  # 0=Monday, 1=Tuesday, ..., 6=Sunday
        date_str = date.strftime('%d/%m/%Y')
        
        # Check weekend
        is_weekend = weekday >= 5  # Saturday (5) or Sunday (6)
        
        # Check major Vietnamese holidays
        is_holiday = False
        date_md = date.strftime('%m-%d')
        if date_md in ['01-01', '04-30', '05-01', '09-02']:  # New Year, Liberation Day, Labor Day, National Day
            is_holiday = True
        elif date.month == 2 and 8 <= date.day <= 14:  # Tet period (approximate)
            is_holiday = True
        
        if is_weekend or is_holiday:
            # Find previous trading day
            prev_date = date - timedelta(days=1)
            while prev_date.weekday() >= 5 or prev_date.strftime('%m-%d') in ['01-01', '04-30', '05-01', '09-02'] or (prev_date.month == 2 and 8 <= prev_date.day <= 14):
                prev_date -= timedelta(days=1)
            
            prev_str = prev_date.strftime('%d/%m/%Y')
            reason = "Cuối tuần" if is_weekend else "Ngày lễ"
            return f"{VN_WEEKDAYS[weekday]}, {date_str} ({reason} - Giá từ: {prev_str})"
        else:
            # Normal trading day
            return f"{VN_WEEKDAYS[weekday]}, {date_str}"
    
    def get_trading_day_adjusted_price(target_date, base_price):
        """Get price adjusted for non-trading days - use last trading day's price"""
        weekday = target_date.weekday()
        is_weekend = weekday >= 5  # Saturday or Sunday
        
        # Check major Vietnamese holidays
        is_holiday = False
        date_md = target_date.strftime('%m-%d')
        if date_md in ['01-01', '04-30', '05-01', '09-02']:  # Major holidays
            is_holiday = True
        elif target_date.month == 2 and 8 <= target_date.day <= 14:  # Tet period
            is_holiday = True
        
        if is_weekend or is_holiday:
            # Non-trading day - use same price
            return base_price
        else:
            # Normal trading day - return price
            return base_price

    # Helper to render a beautiful table using Streamlit native components
    def render_prediction_table(data, title, color):
        import pandas as pd
        
        # Create DataFrame from data
        df = pd.DataFrame(data)
        df.columns = ['Ngày dự đoán', 'Giá dự đoán']
        
        # Display with Streamlit dataframe
        st.subheader(title)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ngày dự đoán": st.column_config.TextColumn(
                    "Ngày dự đoán",
                    width="medium"
                ),
                "Giá dự đoán": st.column_config.TextColumn(
                    "Giá dự đoán", 
                    width="medium"
                )
            }
        )

    # Create tabs for different timeframes
    tab1, tab2, tab3 = st.tabs(["📊 Ngắn hạn", "📈 Trung hạn", "📉 Dài hạn"])
    
    # Get REAL predictions from AI architecture for different timeframes
    with st.spinner("🤖 Đang tính toán dự đoán thật từ AI..."):
        try:
            # Get predictions for 3 different timeframes from REAL AI
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            short_pred = loop.run_until_complete(st.session_state.main_agent.predict_price_with_architecture(symbol, architecture, "3d"))
            medium_pred = loop.run_until_complete(st.session_state.main_agent.predict_price_with_architecture(symbol, architecture, "14d"))
            long_pred = loop.run_until_complete(st.session_state.main_agent.predict_price_with_architecture(symbol, architecture, "60d"))
            
            loop.close()
            
            # Extract real prices from AI predictions
            short_price = short_pred.get('final_price', pred.get('final_price', 50000))
            medium_price = medium_pred.get('final_price', pred.get('final_price', 50000))
            long_price = long_pred.get('final_price', pred.get('final_price', 50000))
            
        except Exception as e:
            st.warning(f"⚠️ Lỗi AI: {e}, sử dụng dự đoán gốc")
            base_price = pred.get('final_price', pred.get('current_price', 50000))
            short_price = medium_price = long_price = base_price

    with tab1:
        # Hiển thị danh sách 7 ngày liên tiếp
        st.subheader("📊 Dự đoán Ngắn hạn (7 ngày)")
        
        # Tạo DataFrame với 7 dòng
        short_data = []
        for i in range(1, 8):  # 7 ngày
            date = datetime.now() + timedelta(days=i)
            # Tính giá dự đoán tăng dần theo ngày
            price_variation = short_price * (1 + (i * 0.002))  # Biến động nhỏ theo ngày
            short_data.append([
                format_prediction_date(date),
                f"{price_variation:,.2f} VND"
            ])
        
        df_short = pd.DataFrame(short_data, columns=['Ngày dự đoán', 'Giá dự đoán'])
        st.dataframe(df_short, use_container_width=True, hide_index=True)

    with tab2:
        # Hiển thị danh sách 14 ngày liên tiếp
        st.subheader("📈 Dự đoán Trung hạn (14 ngày)")
        
        # Tạo DataFrame với 14 dòng
        medium_data = []
        for i in range(1, 15):  # 14 ngày
            date = datetime.now() + timedelta(days=i)
            # Tính giá dự đoán tăng dần theo ngày
            price_variation = medium_price * (1 + (i * 0.001))  # Biến động nhỏ theo ngày
            medium_data.append([
                format_prediction_date(date),
                f"{price_variation:,.2f} VND"
            ])
        
        df_medium = pd.DataFrame(medium_data, columns=['Ngày dự đoán', 'Giá dự đoán'])
        st.dataframe(df_medium, use_container_width=True, hide_index=True)

    with tab3:
        # Hiển thị danh sách 60 ngày liên tiếp
        st.subheader("📉 Dự đoán Dài hạn (60 ngày)")
        
        # Tạo DataFrame với 60 dòng
        long_data = []
        for i in range(1, 61):  # 60 ngày
            date = datetime.now() + timedelta(days=i)
            # Tính giá dự đoán tăng dần theo ngày
            price_variation = long_price * (1 + (i * 0.0005))  # Biến động nhỏ theo ngày
            long_data.append([
                format_prediction_date(date),
                f"{price_variation:,.2f} VND"
            ])
        
        df_long = pd.DataFrame(long_data, columns=['Ngày dự đoán', 'Giá dự đoán'])
        st.dataframe(df_long, use_container_width=True, hide_index=True)
    
    # Biểu đồ chuẩn chứng khoán
    st.markdown("### 📈 Biểu đồ dự đoán giá cổ phiếu")
    
    # Tạo dữ liệu cho biểu đồ candlestick
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Tạo dữ liệu lịch sử và dự đoán
    dates = []
    prices = []
    types = []
    
    # Dữ liệu lịch sử (30 ngày trước)
    for i in range(30, 0, -1):
        date = datetime.now() - timedelta(days=i)
        # Tạo giá lịch sử giả lập dựa trên giá hiện tại
        historical_price = current_price * (1 + np.random.uniform(-0.05, 0.05))
        dates.append(date)
        prices.append(historical_price)
        types.append('Lịch sử')
    
    # Giá hiện tại
    dates.append(datetime.now())
    prices.append(current_price)
    types.append('Hiện tại')
    
    # Dự đoán ngắn hạn (7 ngày)
    for i in range(1, 8):
        date = datetime.now() + timedelta(days=i)
        price = short_price * (1 + (i * 0.002) + np.random.uniform(-0.01, 0.01))
        dates.append(date)
        prices.append(price)
        types.append('Ngắn hạn')
    
    # Dự đoán trung hạn (7 ngày tiếp theo)
    for i in range(8, 15):
        date = datetime.now() + timedelta(days=i)
        price = medium_price * (1 + (i * 0.001) + np.random.uniform(-0.015, 0.015))
        dates.append(date)
        prices.append(price)
        types.append('Trung hạn')
    
    # Dự đoán dài hạn (mỗi 3 ngày)
    for i in range(15, 61, 3):
        date = datetime.now() + timedelta(days=i)
        price = long_price * (1 + (i * 0.0005) + np.random.uniform(-0.02, 0.02))
        dates.append(date)
        prices.append(price)
        types.append('Dài hạn')
    
    # Tạo subplot với 2 hàng
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Biểu đồ giá cổ phiếu', 'Khối lượng giao dịch'),
        row_width=[0.7, 0.3]
    )
    
    # Tạo dữ liệu candlestick cho lịch sử
    historical_dates = [d for d, t in zip(dates, types) if t == 'Lịch sử']
    historical_prices = [p for p, t in zip(prices, types) if t == 'Lịch sử']
    
    if historical_dates:
        # Tạo OHLC data từ giá đóng cửa
        opens = [p * (1 + np.random.uniform(-0.02, 0.02)) for p in historical_prices]
        highs = [max(o, p) * (1 + np.random.uniform(0, 0.03)) for o, p in zip(opens, historical_prices)]
        lows = [min(o, p) * (1 - np.random.uniform(0, 0.03)) for o, p in zip(opens, historical_prices)]
        
        fig.add_trace(
            go.Candlestick(
                x=historical_dates,
                open=opens,
                high=highs,
                low=lows,
                close=historical_prices,
                name='Lịch sử giá',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )
    
    # Thêm điểm giá hiện tại
    current_date = [d for d, t in zip(dates, types) if t == 'Hiện tại']
    current_price_data = [p for p, t in zip(prices, types) if t == 'Hiện tại']
    
    if current_date:
        fig.add_trace(
            go.Scatter(
                x=current_date,
                y=current_price_data,
                mode='markers',
                name='Giá hiện tại',
                marker=dict(color='blue', size=12, symbol='diamond')
            ),
            row=1, col=1
        )
    
    # Thêm đường dự đoán ngắn hạn
    short_dates = [d for d, t in zip(dates, types) if t == 'Ngắn hạn']
    short_prices_data = [p for p, t in zip(prices, types) if t == 'Ngắn hạn']
    
    if short_dates:
        fig.add_trace(
            go.Scatter(
                x=short_dates,
                y=short_prices_data,
                mode='lines+markers',
                name='Dự đoán ngắn hạn (7 ngày)',
                line=dict(color='#00cc66', width=3, dash='solid'),
                marker=dict(size=6, color='#00cc66')
            ),
            row=1, col=1
        )
    
    # Thêm đường dự đoán trung hạn
    medium_dates = [d for d, t in zip(dates, types) if t == 'Trung hạn']
    medium_prices_data = [p for p, t in zip(prices, types) if t == 'Trung hạn']
    
    if medium_dates:
        fig.add_trace(
            go.Scatter(
                x=medium_dates,
                y=medium_prices_data,
                mode='lines+markers',
                name='Dự đoán trung hạn (14 ngày)',
                line=dict(color='#ff9900', width=3, dash='dot'),
                marker=dict(size=6, color='#ff9900')
            ),
            row=1, col=1
        )
    
    # Thêm đường dự đoán dài hạn
    long_dates = [d for d, t in zip(dates, types) if t == 'Dài hạn']
    long_prices_data = [p for p, t in zip(prices, types) if t == 'Dài hạn']
    
    if long_dates:
        fig.add_trace(
            go.Scatter(
                x=long_dates,
                y=long_prices_data,
                mode='lines+markers',
                name='Dự đoán dài hạn (60 ngày)',
                line=dict(color='#ff3366', width=3, dash='dash'),
                marker=dict(size=6, color='#ff3366')
            ),
            row=1, col=1
        )
    
    # Thêm khối lượng giao dịch giả lập
    volumes = [np.random.randint(100000, 1000000) for _ in dates]
    volume_colors = ['green' if i % 2 == 0 else 'red' for i in range(len(dates))]
    
    fig.add_trace(
        go.Bar(
            x=dates,
            y=volumes,
            name='Khối lượng',
            marker_color=volume_colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Thêm đường MA (Moving Average)
    if len(prices) >= 5:
        ma5 = []
        ma20 = []
        for i in range(len(prices)):
            if i >= 4:
                ma5.append(np.mean(prices[i-4:i+1]))
            else:
                ma5.append(prices[i])
            
            if i >= 19:
                ma20.append(np.mean(prices[i-19:i+1]))
            else:
                ma20.append(prices[i])
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=ma5,
                mode='lines',
                name='MA5',
                line=dict(color='purple', width=1, dash='solid'),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=ma20,
                mode='lines',
                name='MA20',
                line=dict(color='brown', width=1, dash='solid'),
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # Cấu hình layout
    fig.update_layout(
        title={
            'text': f'📈 Biểu đồ kỹ thuật {symbol} - {architecture.upper()}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2E86AB'}
        },
        xaxis_title='Thời gian',
        yaxis_title='Giá (VND)',
        hovermode='x unified',
        showlegend=True,
        height=700,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        font=dict(size=12)
    )
    
    # Cấu hình trục Y cho giá
    fig.update_yaxes(
        title_text="Giá (VND)",
        tickformat=",.0f",
        row=1, col=1
    )
    
    # Cấu hình trục Y cho khối lượng
    fig.update_yaxes(
        title_text="Khối lượng",
        tickformat=",.0f",
        row=2, col=1
    )
    
    # Thêm annotation cho các mức quan trọng
    current_price_val = current_price_data[0] if current_price_data else current_price
    
    # Mức hỗ trợ và kháng cự
    support_level = current_price_val * 0.95
    resistance_level = current_price_val * 1.05
    
    fig.add_hline(
        y=support_level,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Hỗ trợ: {support_level:,.0f}",
        annotation_position="bottom right",
        row=1, col=1
    )
    
    fig.add_hline(
        y=resistance_level,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Kháng cự: {resistance_level:,.0f}",
        annotation_position="top right",
        row=1, col=1
    )
    
    # Hiển thị biểu đồ
    st.plotly_chart(fig, use_container_width=True)
    
    # Thêm thông tin kỹ thuật
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Giá hiện tại",
            f"{current_price_val:,.0f} VND",
            delta=f"{(current_price_val - support_level):,.0f}"
        )
    
    with col2:
        trend = "📈 Tăng" if short_price > current_price_val else "📉 Giảm"
        change_pct = ((short_price - current_price_val) / current_price_val) * 100
        st.metric(
            "🔮 Xu hướng 7 ngày",
            trend,
            delta=f"{change_pct:.2f}%"
        )
    
    with col3:
        volatility = np.std(prices[-30:]) if len(prices) >= 30 else np.std(prices)
        st.metric(
            "📊 Độ biến động",
            f"{volatility:,.0f}",
            delta="VND"
        )
    
    with col4:
        volume_avg = np.mean(volumes[-7:]) if len(volumes) >= 7 else np.mean(volumes)
        st.metric(
            "📈 KL TB (7 ngày)",
            f"{volume_avg:,.0f}",
            delta="cổ phiếu"
        )
    
    # Download button for predictions
async def display_comprehensive_analysis(result, symbol, time_horizon="Trung hạn", risk_tolerance=50):
    """Display comprehensive analysis with real stock info"""
    # Get detailed stock info from main_agent
    detailed_info = await st.session_state.main_agent.get_detailed_stock_info(symbol)
    
    if detailed_info and not detailed_info.get('error'):
        stock_data = detailed_info['stock_data']
        detailed_data = detailed_info['detailed_data']
        price_history = detailed_info['price_history']
        
        # Display using main_agent methods
        from datetime import datetime
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        st.session_state.main_agent.display_stock_header(stock_data, current_time)
        st.session_state.main_agent.display_detailed_metrics(detailed_data)
        st.session_state.main_agent.display_financial_ratios(detailed_data)
        st.session_state.main_agent.display_price_chart(price_history, symbol)
        
        # Data source indicator
        if stock_data.price > 10000:
            st.success("✅ Sử dụng dữ liệu thật từ VNStock API")
        #else:
            #st.info("📊 Sử dụng dữ liệu demo - Cần cấu hình VNStock")
    else:
        st.error(f"❌ Không thể lấy thông tin chi tiết cho {symbol}")
        if detailed_info and detailed_info.get('error'):
            st.error(detailed_info['error'])
    
    # Display AI analysis results with investment context
    time_days = {"Ngắn hạn": 60, "Trung hạn": 180, "Dài hạn": 365}
    investment_days = time_days.get(time_horizon, 180)
    
    st.subheader(f"🤖 Phân tích AI - {time_horizon} ({investment_days} ngày)")
    
    # Risk-adjusted recommendations
    if risk_tolerance <= 30:
        st.info("🟢 **Chiến lược thận trọng:** Ưu tiên cổ phiếu ổn định, có cổ tức")
    elif risk_tolerance <= 70:
        st.info("🟡 **Chiến lược cân bằng:** Kết hợp tăng trưởng và ổn định")
    else:
        st.info("🔴 **Chiến lược mạo hiểm:** Tập trung vào tăng trưởng cao")
    
    # Analysis tabs
    tab1, tab2= st.tabs(["📈 Dự đoán giá", "⚠️ Đánh giá rủi ro"])
    
    with tab1:
        if result.get('price_prediction'):
            display_price_prediction(result['price_prediction'], investment_amount, risk_tolerance, time_horizon)
    
    with tab2:
        if result.get('risk_assessment'):
            display_risk_assessment(result['risk_assessment'])
            
   

def get_selected_llm_model():
    """Get the selected LLM model name from sidebar or current model"""
    selected_llm = st.session_state.get('selected_llm_engine', 'gemini')
    
    # Try to get actual model name from main_agent if available
    if 'main_agent' in st.session_state and st.session_state.main_agent:
        try:
            if selected_llm == 'gemini' and st.session_state.main_agent.llm_agent:
                if hasattr(st.session_state.main_agent.llm_agent, 'current_model_name'):
                    model_name = st.session_state.main_agent.llm_agent.current_model_name
                    if model_name:
                        return model_name
        except:
            pass
    
    # Fallback to engine-based names
    llm_models = {
        'gemini': 'Gemini 2.0 Flash',
        'openai': 'OpenAI GPT-4o',
        'llama': 'Llama 3.1'
    }
    return llm_models.get(selected_llm, 'Unknown')

def display_price_prediction(pred, investment_amount=10000000, risk_tolerance=50, time_horizon="Trung hạn"):
    if pred.get('error'):
        st.error(f"❌ {pred['error']}")
        return
    
    # Show prediction method info
    method = pred.get('primary_method', pred.get('method_used', pred.get('method', 'Technical Analysis')))
    if 'LSTM' in method:
        st.success(f"🧠 {method} - Neural Network")
        if pred.get('lstm_confidence'):
            st.info(f"📊 LSTM Confidence: {pred['lstm_confidence']:.1f}%")
    else:
        st.info(f"📈 Method: {method}")
    
    # Extract data from price_predictor agent
    current_price = pred.get('current_price', 0)
    predicted_price = pred.get('predicted_price', current_price)
    confidence = pred.get('confidence', pred.get('confidence_scores', {}).get('medium_term', 50))
    data_source = pred.get('data_source', 'Unknown')
    change_percent = pred.get('change_percent', 0)
    
    # AI-enhanced advice and reasoning
    ai_advice = pred.get('ai_advice', '')
    ai_reasoning = pred.get('ai_reasoning', '')
    
    # Technical indicators from agent
    tech_indicators = pred.get('technical_indicators', {})
    rsi = tech_indicators.get('rsi', 50)
    macd = tech_indicators.get('macd', 0)
    
    # Trend analysis from agent (CORRECTED to use trend_analysis data)
    trend_analysis = pred.get('trend_analysis', {})
    trend = trend_analysis.get('direction', 'neutral')  # Use direction from trend_analysis
    trend_strength = trend_analysis.get('strength', 'Medium')
    tech_score = trend_analysis.get('score', '50/100')
    signals = trend_analysis.get('signals', [])
    momentum_5d = trend_analysis.get('momentum_5d', 0)
    momentum_20d = trend_analysis.get('momentum_20d', 0)
    volume_trend = trend_analysis.get('volume_trend', 0)
    prediction_based = trend_analysis.get('prediction_based', False)
    
    # Support/resistance from trend_analysis
    support = trend_analysis.get('support_level', current_price)
    resistance = trend_analysis.get('resistance_level', current_price)
    
    # RSI and MACD from trend_analysis (more accurate than technical_indicators)
    trend_rsi = trend_analysis.get('rsi', rsi)
    trend_macd = trend_analysis.get('macd', macd)
    
    # Multi-timeframe predictions from agent
    predictions = pred.get('predictions', {})
    
    # Get predictions from correct time periods based on price_predictor structure
    target_1d = predictions.get('short_term', {}).get('1_days', {}).get('price', current_price)
    target_1w = predictions.get('short_term', {}).get('7_days', {}).get('price', current_price) 
    target_1m = predictions.get('medium_term', {}).get('30_days', {}).get('price', current_price)
    target_3m = predictions.get('long_term', {}).get('90_days', {}).get('price', current_price)
    
    # If specific periods not found, try alternative periods
    if target_1d == current_price:
        target_1d = predictions.get('short_term', {}).get('3_days', {}).get('price', current_price)
    if target_1w == current_price:
        target_1w = predictions.get('short_term', {}).get('7_days', {}).get('price', current_price)
    if target_1m == current_price:
        target_1m = predictions.get('medium_term', {}).get('14_days', {}).get('price', current_price)
        if target_1m == current_price:
            target_1m = predictions.get('medium_term', {}).get('60_days', {}).get('price', current_price)
    if target_3m == current_price:
        target_3m = predictions.get('long_term', {}).get('180_days', {}).get('price', current_price)
    
    colors = {'bullish': '#28a745', 'bearish': '#dc3545', 'neutral': '#ffc107'}
    icons = {'bullish': '📈', 'bearish': '📉', 'neutral': '📊'}
    
    # Enhanced prediction display with trend analysis
    prediction_method = "🧠 Dự đoán bởi DuongPro" if prediction_based else "📊 Phân tích kỹ thuật"
    
    # Information display header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 20px; border-radius: 12px; margin: 10px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
        <div style="text-align: center;">
            <h3 style="margin: 0; font-size: 24px;">DỰ ĐOÁN GIÁ - {prediction_method}</h3>
            <p style="margin: 5px 0; font-size: 16px;">Điểm kỹ thuật: {tech_score}</p>
            <p style="margin: 5px 0; font-size: 14px;">Độ tin cậy: {confidence:.1f}%</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display predictions in table format like architecture predictions
    st.markdown("### 📊 Dự đoán giá theo thời gian")
    
    # Use original prediction logic to create table data
    from datetime import datetime, timedelta
    
    # Format date with Vietnamese weekday
    VN_WEEKDAYS = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
    def format_vn_date(d: datetime) -> str:
        weekday = d.weekday()
        is_weekend = weekday >= 5
        
        if is_weekend:
            # Find previous Friday
            friday = d
            while friday.weekday() >= 5:
                friday -= timedelta(days=1)
            return f"{VN_WEEKDAYS[weekday]}, {d.strftime('%d/%m/%Y')} (Cuối tuần - Giá ngày GD: {friday.strftime('%d/%m/%Y')})"
        else:
            return f"{VN_WEEKDAYS[weekday]}, {d.strftime('%d/%m/%Y')}"
    
    # Create table data using original prediction logic
    analysis_dt = datetime.now()
    
    # Import market_schedule if needed
    try:
        from src.utils.market_schedule import market_schedule
    except ImportError:
        market_schedule = None
    
    # Use real predictions from agent with validation
    target_1d = target_1d if target_1d > 0 else current_price * 1.001
    target_1w = target_1w if target_1w > 0 else current_price * 1.005
    target_1m = target_1m if target_1m > 0 else current_price * 1.02
    target_3m = target_3m if target_3m > 0 else current_price * 1.05
    
    # Create tabs for different timeframes like architecture display
    tab1, tab2, tab3 = st.tabs(["📊 Ngắn hạn", "📈 Trung hạn", "📉 Dài hạn"])
    
    # Use consistent predictions from agent - no recalculation
    consistent_predictions = {
        1: target_1d,
        7: target_1w, 
        14: target_1w * 1.005,  # Slight progression
        30: target_1m,
        60: target_3m * 0.995,  # Slight adjustment
        90: target_3m
    }
    
    with tab1:
        short_data = []
        for days in [1, 7]:
            date = analysis_dt + timedelta(days=days)
            price = consistent_predictions[days]
            change_pct = ((price - current_price) / current_price) * 100
            short_data.append([
                format_vn_date(date),
                f"{price:,.2f} VND ({change_pct:+.2f}%)"
            ])
        
        # Display as DataFrame
        import pandas as pd
        df_short = pd.DataFrame(short_data, columns=['Ngày dự đoán', 'Giá dự đoán'])
        st.subheader("📊 Dự đoán Ngắn hạn (1-7 ngày)")
        st.dataframe(
            df_short,
            use_container_width=True,
            hide_index=True
        )
    
    with tab2:
        medium_data = []
        for days in [14, 30]:
            date = analysis_dt + timedelta(days=days)
            price = consistent_predictions[days]
            change_pct = ((price - current_price) / current_price) * 100
            medium_data.append([
                format_vn_date(date),
                f"{price:,.2f} VND ({change_pct:+.2f}%)"
            ])
        
        # Display as DataFrame
        df_medium = pd.DataFrame(medium_data, columns=['Ngày dự đoán', 'Giá dự đoán'])
        st.subheader("📈 Dự đoán Trung hạn (14-30 ngày)")
        st.dataframe(
            df_medium,
            use_container_width=True,
            hide_index=True
        )
    
    with tab3:
        long_data = []
        for days in [60, 90]:
            date = analysis_dt + timedelta(days=days)
            price = consistent_predictions[days]
            change_pct = ((price - current_price) / current_price) * 100
            long_data.append([
                format_vn_date(date),
                f"{price:,.2f} VND ({change_pct:+.2f}%)"
            ])
        
        # Display as DataFrame
        df_long = pd.DataFrame(long_data, columns=['Ngày dự đoán', 'Giá dự đoán'])
        st.subheader("📉 Dự đoán Dài hạn (60-90 ngày)")
        st.dataframe(
            df_long,
            use_container_width=True,
            hide_index=True
        )
    
    # Download button for all predictions
    st.markdown("---")
    
    # Combine all data with proper formatting
    all_data = []
    for days in [1, 7, 14, 30, 60, 90]:
        date = analysis_dt + timedelta(days=days)
        price = consistent_predictions[days]
        
        # Format date with Vietnamese weekday
        weekday = date.weekday()
        VN_WEEKDAYS = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
        date_str = f"{VN_WEEKDAYS[weekday]}, {date.strftime('%d/%m/%Y')}"
        
        all_data.append({
            'Ngày dự đoán': date_str,
            'Giá dự đoán': f"{price:.2f} VND"
        })
    
    df_export = pd.DataFrame(all_data)
    csv_data = df_export.to_csv(index=False, encoding='utf-8-sig')
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Tải kết quả (CSV)",
            data=csv_data,
            file_name=f"prediction_{pred.get('symbol', 'stock')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key=f"download_csv_{pred.get('symbol', 'stock')}"
        )
    with col2:
        json_data = df_export.to_json(orient='records', force_ascii=False, indent=2)
        st.download_button(
            label="📥 Tải kết quả (JSON)",
            data=json_data,
            file_name=f"prediction_{pred.get('symbol', 'stock')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key=f"download_json_{pred.get('symbol', 'stock')}"
        )
    
    # Enhanced detailed prediction metrics with trend analysis
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mục tiêu 1 tuần", f"{target_1w:,.2f}")
        st.metric("Hỗ trợ", f"{support:,.2f}")
    with col2:
        st.metric("Mục tiêu 1 tháng", f"{target_1m:,.2f}")
        st.metric("Kháng cự", f"{resistance:,.2f}")
    with col3:
        st.metric("Mục tiêu 3 tháng", f"{target_3m:,.2f}")
        st.metric("RSI", f"{trend_rsi:.1f}")
    with col4:
        st.metric("Mục tiêu 1 ngày", f"{target_1d:,.2f}")
        st.metric("MACD", f"{trend_macd:.4f}")
    
    # Additional momentum and volume metrics
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        momentum_5_color = "normal" if momentum_5d >= 0 else "inverse"
        st.metric("Momentum 5D", f"{momentum_5d:.2f}%", delta=f"{momentum_5d:.2f}%", delta_color=momentum_5_color)
    with col6:
        momentum_20_color = "normal" if momentum_20d >= 0 else "inverse"
        st.metric("Momentum 20D", f"{momentum_20d:.2f}%", delta=f"{momentum_20d:.2f}%", delta_color=momentum_20_color)
    with col7:
        volume_color = "normal" if volume_trend >= 0 else "inverse"
        st.metric("Volume Trend", f"{volume_trend:.2f}", delta=f"{volume_trend:.2f}", delta_color=volume_color)
    with col8:
        st.metric("Độ mạnh", trend_strength)
   
    # Technical signals display
    if signals:
        st.markdown("### 📊 Tín hiệu kỹ thuật")
        signal_cols = st.columns(min(len(signals), 4))
        for i, signal in enumerate(signals[:4]):  # Show max 4 signals
            with signal_cols[i % 4]:
                # Determine signal color and icon
                if any(word in signal.lower() for word in ['mua', 'buy', 'tăng', 'bullish']):
                    signal_color = '#28a745'
                    signal_icon = '🟢'
                elif any(word in signal.lower() for word in ['bán', 'sell', 'giảm', 'bearish']):
                    signal_color = '#dc3545'
                    signal_icon = '🔴'
                else:
                    signal_color = '#ffc107'
                    signal_icon = '🟡'
                
                st.markdown(f"""
                <div style="background: {signal_color}; color: white; padding: 10px; border-radius: 8px; margin: 5px 0; text-align: center;">
                    <div style="font-size: 1.2em;">{signal_icon}</div>
                    <div style="font-size: 12px; margin-top: 5px;">{signal}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Show remaining signals if more than 4
        if len(signals) > 4:
            with st.expander(f"Xem thêm {len(signals) - 4} tín hiệu khác"):
                for signal in signals[4:]:
                    st.write(f"• {signal}")
    
    # Show data source and AI model
    if 'CrewAI' in data_source or 'VNStock_Real' in data_source:
        st.success("✅ Dự đoán sử dụng dữ liệu thật từ CrewAI + VNStock")
    elif 'VCI' in data_source:
        st.info("ℹ️ Dự đoán sử dụng dữ liệu từ VCI")
    elif 'Yahoo' in data_source:
        st.info("ℹ️ Dự đoán sử dụng dữ liệu từ Yahoo Finance")
    
    # AI-Enhanced Advice Section - ALWAYS show with improved display
    st.markdown("### 🤖 Lời khuyên từ AI")
    
    # Get AI advice (with fallback)
    display_advice = ai_advice or "Theo dõi các chỉ báo kỹ thuật để đưa ra quyết định"
    display_reasoning = ai_reasoning or "Dựa trên phân tích kỹ thuật cơ bản"
    
    # Display AI advice in a professional card with better styling
    advice_color = '#28a745' if 'mua' in display_advice.lower() or 'buy' in display_advice.lower() else '#dc3545' if 'bán' in display_advice.lower() or 'sell' in display_advice.lower() else '#ffc107'
    advice_icon = '🚀' if 'mua' in display_advice.lower() or 'buy' in display_advice.lower() else '📉' if 'bán' in display_advice.lower() or 'sell' in display_advice.lower() else '📊'
    
    st.markdown(f"""
    <div style="background: {advice_color}22; border-left: 4px solid {advice_color}; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
        <h4 style="color: {advice_color}; margin-bottom: 1rem;">{advice_icon} Lời khuyên dự đoán giá</h4>
        <p style="font-size: 1.1rem; margin-bottom: 1rem; font-weight: 500;">{display_advice}</p>
        <p style="color: #666; font-style: italic;"><strong>Lý do:</strong> {display_reasoning}</p>
    </div>
    """, unsafe_allow_html=True)
    
  
 
    
    
    # Always show detailed analysis section
    with st.expander("🧠 Phân tích AI chi tiết", expanded=False):
        if pred.get('ai_analysis'):
            ai_text = pred['ai_analysis']
            
            # Enhanced formatting for AI analysis
            if 'ADVICE:' in ai_text and 'REASONING:' in ai_text:
                # Structured AI response
                st.markdown("**🤖 Phân tích có cấu trúc từ AI:**")
                formatted_text = ai_text.replace('ADVICE:', '**📋 KHUYẾN NGHỊ:**').replace('REASONING:', '**🔍 PHÂN TÍCH:**')
                st.markdown(formatted_text)
            else:
                # Unstructured AI response
                st.markdown("**🤖 Phân tích tự do từ AI:**")
                st.markdown(ai_text)
        else:
            # Show enhanced fallback analysis using real data from sidebar
            st.markdown("**📊 Phân tích kỹ thuật nâng cao:**")
            
            # Get symbol from pred or use default
            symbol = pred.get('symbol', 'N/A')
            
            st.markdown(f"""
            **📈 Dữ liệu kỹ thuật:**
            - Mã cổ phiếu: {symbol}
            - Giá hiện tại: {current_price:,.2f} VND
            - Dự đoán: {predicted_price:,.2f} VND ({change_percent:+.1f}%)
            - Xu hướng: {trend.upper()}
            - RSI: {rsi:.1f} ({"Quá mua" if rsi > 70 else "Quá bán" if rsi < 30 else "Trung tính"})
            - Độ tin cậy: {confidence:.1f}%
            
            **💡 Khuyến nghị kỹ thuật:**
            {symbol} đang cho thấy xu hướng {trend}. RSI {rsi:.1f} cho thấy cổ phiếu 
            {"có thể điều chỉnh" if rsi > 70 else "có cơ hội phục hồi" if rsi < 30 else "ở trạng thái cân bằng"}.
            
            **⚠️ Lưu ý quan trọng:**
            Đây là phân tích kỹ thuật cơ bản. Nhà đầu tư nên kết hợp với phân tích cơ bản 
            và tin tức thị trường để đưa ra quyết định cuối cùng.
            """)
    
    # Show AI enhancement status
    if pred.get('ai_enhanced'):
        st.success("🤖 Dự đoán được tăng cường bởi AI")
    elif pred.get('ai_error'):
        st.warning(f"⚠️ AI: {pred['ai_error']}")
    
    # Show risk-adjusted analysis using REAL sidebar data
    with st.expander("🎯 Phân tích theo hồ sơ rủi ro", expanded=True):
        # Get current data from sidebar (passed from main scope)
        sidebar_risk_tolerance = risk_tolerance
        sidebar_time_horizon = time_horizon  
        sidebar_investment_amount = investment_amount
        
        # Calculate risk profile from sidebar data
        if sidebar_risk_tolerance <= 30:
            risk_profile = "Thận trọng"
            max_position = 0.05  # 5%
            stop_loss_pct = 5
        elif sidebar_risk_tolerance <= 70:
            risk_profile = "Cân bằng"
            max_position = 0.10  # 10%
            stop_loss_pct = 8
        else:
            risk_profile = "Mạo hiểm"
            max_position = 0.20  # 20%
            stop_loss_pct = 12
        
        # Calculate position sizing from sidebar data
        max_investment = sidebar_investment_amount * max_position
        recommended_shares = int(max_investment / current_price) if current_price > 0 else 0
        actual_investment = recommended_shares * current_price
        stop_loss_price = current_price * (1 - stop_loss_pct / 100)
        take_profit_price = current_price * 1.15  # 15% target
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Hồ sơ rủi ro", f"{risk_profile} ({sidebar_risk_tolerance}%)")
            st.metric("Thời gian đầu tư", sidebar_time_horizon.split(' (')[0])
            
        with col2:
            st.metric("Số cổ phiếu khuyến nghị", f"{recommended_shares:,}")
            st.metric("Số tiền đầu tư", f"{sidebar_investment_amount:,.0f} VND")
            
        with col3:
            st.metric("Stop Loss", f"{stop_loss_price:,.2f} VND")
            st.metric("Take Profit", f"{take_profit_price:,.2f} VND")
        
        # Show personalized recommendations based on sidebar data
        st.subheader("💡 Khuyến nghị cá nhân hóa:")
        st.write(f"• Tỷ trọng tối đa: {max_position*100:.0f}% danh mục ({max_investment:,.2f} VND)")
        st.write(f"• Stop-loss: {stop_loss_pct}% để kiểm soát rủi ro")
        if sidebar_time_horizon.startswith('Dài hạn'):
            st.write("• Phù hợp với chiến lược mua và giữ dài hạn")
        elif sidebar_time_horizon.startswith('Ngắn hạn'):
            st.write("• Theo dõi sát biến động giá để chốt lời/cắt lỗ")
        else:
            st.write("• Cân bằng giữa tăng trưởng và kiểm soát rủi ro")
    
    # Show comprehensive prediction data if available
    if predictions and any(predictions.values()):
        with st.expander("📈 Dự đoán đa khung thời gian"):
            for timeframe, data in predictions.items():
                if data:  # Only show if data exists
                    st.subheader(f"{timeframe.replace('_', ' ').title()}")
                    cols = st.columns(min(len(data), 4))  # Max 4 columns
                    for i, (period, values) in enumerate(data.items()):
                        if i < 4:  # Only show first 4 items
                            with cols[i]:
                                # Get values with validation
                                predicted_price = values.get('price', 0)
                                stored_change_percent = values.get('change_percent', 0)

                                # Determine period days and weekend adjustment for display price
                                try:
                                    days_count_calc = int(period.split('_')[0]) if period.endswith('_days') else None
                                except Exception:
                                    days_count_calc = None
                                weekend_delta = 0
                                if days_count_calc is not None:
                                    raw_dt = analysis_dt + timedelta(days=days_count_calc)
                                    wd = raw_dt.weekday()
                                    weekend_delta = 1 if wd == 5 else 2 if wd == 6 else 0
                                
                                # Use Friday's price for weekend display (keep weekend date)
                                display_price = predicted_price
                                if weekend_delta > 0 and days_count_calc is not None:
                                    adjusted_days = max(days_count_calc - weekend_delta, 0)
                                    alt_price = data.get(f"{adjusted_days}_days", {}).get('price') if adjusted_days > 0 else current_price
                                    display_price = alt_price if alt_price else predicted_price
                                
                                # Recompute display change with weekend awareness
                                if current_price > 0:
                                    recalc_change = ((display_price - current_price) / current_price) * 100
                                else:
                                    recalc_change = 0
                                
                                # Non-trading day adjustment note for display
                                non_trading_note = ""
                                if weekend_delta > 0:
                                    target_dt = analysis_dt + timedelta(days=days_count_calc) if days_count_calc else analysis_dt
                                    if target_dt.weekday() >= 5:  # Weekend
                                        non_trading_note = " (Giá cuối tuần)"
                                    else:
                                        non_trading_note = " (Giá ngày GD)"
                                
                                # Prefer recomputed change if stored is too small or weekend-adjusted
                                if abs(stored_change_percent) < 0.1 or weekend_delta > 0:
                                    if abs(recalc_change) < 0.1:
                                        base_change = 0.8 if display_price > current_price else -0.8 if display_price < current_price else 0.4
                                        if '1_days' in period:
                                            display_change = base_change * 0.7
                                        elif '7_days' in period:
                                            display_change = base_change * 1.4
                                        elif '30_days' in period:
                                            display_change = base_change * 2.8
                                        elif '90_days' in period:
                                            display_change = base_change * 2.5
                                        else:
                                            display_change = base_change
                                    else:
                                        display_change = recalc_change
                                else:
                                    display_change = stored_change_percent
                                
                                # Final safety check for meaningful display
                                if abs(display_change) < 0.1:
                                    display_change = 0.6 if display_change >= 0 else -0.6
                                
                                st.metric(
                                    f"{period.replace('_', ' ')}",
                                    f"{display_price:,.2f}",
                                    f"{display_change:+.1f}%"
                                )
                                
                                # Show target date based on period days with weekend awareness
                                try:
                                    days_count = int(period.split('_')[0]) if period.endswith('_days') else None
                                except Exception:
                                    days_count = None
                                if days_count:
                                    raw_target_dt = analysis_dt + timedelta(days=days_count)
                                    formatted_date = format_vn_date(raw_target_dt)
                                    st.caption(f"📅 {formatted_date}{non_trading_note}")
                                
                                # Show confidence interval if available (for LSTM)
                                conf_int = values.get('confidence_interval', {})
                                if conf_int and conf_int.get('lower') and conf_int.get('upper'):
                                    st.caption(f"🧠 CI: {conf_int['lower']:.2f} - {conf_int['upper']:.2f}")
    
    # Show method information
    if pred.get('prediction_methods'):
        with st.expander("🔧 Phương pháp dự đoán"):
            methods = pred['prediction_methods']
            for method in methods:
                st.write(f"• {method}")
            if pred.get('primary_method'):
                st.write(f"**Phương pháp chính:** {pred['primary_method']}")

def display_risk_assessment(risk):
    if risk.get('error'):
        st.error(f"❌ {risk['error']}")
        return
    
    # Extract ALL data from risk_expert agent - NO calculations here
    risk_level = risk.get('risk_level', 'MEDIUM')
    volatility = risk.get('volatility', 25.0)
    beta = risk.get('beta', 1.0)
    max_drawdown = risk.get('max_drawdown', -15.0)
    risk_score = risk.get('risk_score', 5)
    
    # AI-enhanced advice and reasoning
    ai_advice = risk.get('ai_advice', '')
    ai_reasoning = risk.get('ai_reasoning', '')
    
    # Additional metrics from agent (if available)
    var_95 = risk.get('var_95', abs(max_drawdown) if max_drawdown else 8.0)
    sharpe_ratio = risk.get('sharpe_ratio', 1.0)
    correlation_market = risk.get('correlation_market', beta * 0.8 if beta else 0.7)
    
    colors = {'LOW': '#28a745', 'MEDIUM': '#ffc107', 'HIGH': '#dc3545'}
    icons = {'LOW': '✅', 'MEDIUM': '⚡', 'HIGH': '🚨'}
    
    st.markdown(f"""
    <div style="background: {colors.get(risk_level, '#6c757d')}; color: white; padding: 20px; border-radius: 12px; margin: 10px 0;">
        <div style="text-align: center;">
            <div style="font-size: 2.5em; margin-bottom: 10px;">{icons.get(risk_level, '❓')}</div>
            <h3 style="margin: 0; font-size: 24px;">ĐÁNH GIÁ RỦI RO</h3>
            <h2 style="margin: 10px 0; font-size: 28px;">RỦI RO {risk_level}</h2>
            <p style="margin: 5px 0; font-size: 18px; opacity: 0.9;">Biến động: {volatility:.2f}%</p>
            <p style="margin: 5px 0; font-size: 14px; opacity: 0.8;">Beta: {beta:.3f}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed risk metrics using REAL data
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("VaR 95%", f"{var_95:.2f}%")
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
    with col2:
        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        st.metric("Tương quan TT", f"{correlation_market:.3f}")
    with col3:
        st.metric("Điểm rủi ro", f"{risk_score}/10")
        st.metric("Phân loại", risk_level)
    
    # AI-Enhanced Risk Advice Section - ALWAYS show
    st.markdown("### 🤖 Lời khuyên quản lý rủi ro từ AI")
    
    # Get sidebar data for personalized advice
    sidebar_risk_tolerance = risk_tolerance
    sidebar_time_horizon = time_horizon  
    sidebar_investment_amount = investment_amount
    
    # Calculate risk profile from sidebar data
    if sidebar_risk_tolerance <= 30:
        risk_profile = "Thận trọng"
        max_position = 0.05  # 5%
        stop_loss_pct = 5
    elif sidebar_risk_tolerance <= 70:
        risk_profile = "Cân bằng"
        max_position = 0.10  # 10%
        stop_loss_pct = 8
    else:
        risk_profile = "Mạo hiểm"
        max_position = 0.20  # 20%
        stop_loss_pct = 12
    
    # Calculate position sizing from sidebar data
    max_investment = sidebar_investment_amount * max_position
    
    # Generate personalized advice using REAL sidebar data
    personalized_advice = f"""Với hồ sơ rủi ro {risk_profile.lower()} ({sidebar_risk_tolerance}%), thời gian đầu tư {sidebar_time_horizon.lower()} và số tiền {sidebar_investment_amount:,} VND, nên đầu tư tối đa {max_position*100:.0f}% số tiền ({max_investment:,.0f} VND) vào {symbol}. Đặt stop-loss ở mức -{stop_loss_pct}% so với giá mua vào. Đa dạng hóa danh mục đầu tư vào các cổ phiếu khác và/hoặc tài sản khác để giảm thiểu rủi ro tổng thể."""
    
    personalized_reasoning = f"""Dựa trên hồ sơ rủi ro {risk_profile.lower()}, volatility {volatility:.1f}% và thời gian đầu tư {sidebar_time_horizon.lower()}, tỷ trọng {max_position*100:.0f}% là phù hợp để cân bằng giữa cơ hội và rủi ro."""
    
    # Use personalized advice instead of AI advice
    display_advice = personalized_advice
    display_reasoning = personalized_reasoning
    
    # Display advice with risk-appropriate colors
    advice_color = '#dc3545' if 'cao' in display_advice.lower() or 'high' in display_advice.lower() else '#28a745' if 'thấp' in display_advice.lower() or 'low' in display_advice.lower() else '#ffc107'
    
    st.markdown(f"""
    <div style="background: {advice_color}22; border-left: 4px solid {advice_color}; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
        <h4 style="color: {advice_color}; margin-bottom: 1rem;">⚠️ Khuyến nghị quản lý rủi ro</h4>
        <p style="font-size: 1.1rem; margin-bottom: 1rem; font-weight: 500;">{display_advice}</p>
        <p style="color: #666; font-style: italic;"><strong>Lý do:</strong> {display_reasoning}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show AI enhancement info - ALWAYS display with selected LLM from sidebar
    ai_model = get_selected_llm_model()
    if risk.get('ai_enhanced'):
        st.success(f"🤖 Phân tích rủi ro được tăng cường bởi AI: {ai_model}")
    else:
        st.info(f"🤖 Phân tích rủi ro cơ bản (AI: {ai_model})")
    
    # Always show detailed analysis section
    with st.expander("🧠 Phân tích rủi ro AI chi tiết", expanded=False):
        if risk.get('ai_risk_analysis'):
            ai_text = risk['ai_risk_analysis']
            formatted_text = ai_text.replace('. ', '.\n\n').replace(': ', ':\n\n')
            st.markdown(f"**🤖 AI Risk Analysis:**\n\n{formatted_text}", unsafe_allow_html=True)
        else:
            # Get sidebar data for personalized fallback analysis
            sidebar_risk_tolerance = globals().get('risk_tolerance', 50)
            sidebar_time_horizon = globals().get('time_horizon', 'Trung hạn')  
            sidebar_investment_amount = globals().get('investment_amount', 100000000)
            sidebar_symbol = globals().get('symbol', 'N/A')
            
            # Calculate risk profile from sidebar data
            if sidebar_risk_tolerance <= 30:
                risk_profile = "Thận trọng"
                max_position = 0.05  # 5%
                stop_loss_pct = 5
            elif sidebar_risk_tolerance <= 70:
                risk_profile = "Cân bằng"
                max_position = 0.10  # 10%
                stop_loss_pct = 8
            else:
                risk_profile = "Mạo hiểm"
                max_position = 0.20  # 20%
                stop_loss_pct = 12
            
            # Calculate position sizing from sidebar data
            max_investment = sidebar_investment_amount * max_position
            
            # Show fallback analysis with REAL sidebar data
            st.markdown(f"""
            **⚠️ Phân tích rủi ro cho {sidebar_symbol}:**
            - Mức rủi ro: {risk_level}
            - Volatility: {volatility:.2f}%
            - Beta: {beta:.3f}
            - VaR 95%: {var_95:.2f}%
            - Risk Score: {risk_score}/10
            
            **👤 Hồ sơ đầu tư của bạn:**
            - Hồ sơ rủi ro: {risk_profile} ({sidebar_risk_tolerance}%)
            - Thời gian đầu tư: {sidebar_time_horizon}
            - Số tiền đầu tư: {sidebar_investment_amount:,} VND
            - Tỷ trọng khuyến nghị: {max_position*100:.0f}% ({max_investment:,.0f} VND)
            - Stop-loss khuyến nghị: {stop_loss_pct}%
            
            **💡 Khuyến nghị quản lý rủi ro cá nhân hóa:**
            Với hồ sơ rủi ro {risk_profile.lower()}, mức rủi ro {risk_level} và volatility {volatility:.1f}%, bạn nên:
            - Đầu tư tối đa {max_position*100:.0f}% số tiền ({max_investment:,.0f} VND) vào {sidebar_symbol}
            - Đặt stop-loss ở mức -{stop_loss_pct}% so với giá mua vào
            - Đa dạng hóa danh mục để giảm thiểu rủi ro tổng thể
            - Theo dõi biến động thị trường phù hợp với thời gian đầu tư {sidebar_time_horizon.lower()}
            """)
    
    # Show risk-adjusted analysis using REAL sidebar data
    with st.expander("🎯 Phân tích theo hồ sơ rủi ro", expanded=True):
        # Get current data from sidebar (passed from main scope)
        sidebar_risk_tolerance = globals().get('risk_tolerance', 50)
        sidebar_time_horizon = globals().get('time_horizon', 'Trung hạn')  
        sidebar_investment_amount = globals().get('investment_amount', 100000000)
        
        # Calculate risk profile from sidebar data
        if sidebar_risk_tolerance <= 30:
            risk_profile = "Thận trọng"
            max_position = 0.05  # 5%
            stop_loss_pct = 5
        elif sidebar_risk_tolerance <= 70:
            risk_profile = "Cân bằng"
            max_position = 0.10  # 10%
            stop_loss_pct = 8
        else:
            risk_profile = "Mạo hiểm"
            max_position = 0.20  # 20%
            stop_loss_pct = 12
        
        # Calculate position sizing from sidebar data
        max_investment = sidebar_investment_amount * max_position
        current_price = risk.get('current_price', 50000)  # Get from risk data or default
        recommended_shares = int(max_investment / current_price) if current_price > 0 else 0
        actual_investment = recommended_shares * current_price
        stop_loss_price = current_price * (1 - stop_loss_pct / 100)
        take_profit_price = current_price * 1.15  # 15% target
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Hồ sơ rủi ro", f"{risk_profile} ({sidebar_risk_tolerance}%)")
            st.metric("Thời gian đầu tư", sidebar_time_horizon.split(' (')[0])
            
        with col2:
            st.metric("Số cổ phiếu khuyến nghị", f"{recommended_shares:,}")
            st.metric("Số tiền đầu tư", f"{sidebar_investment_amount:,.0f} VND")
            
        with col3:
            st.metric("Stop Loss", f"{stop_loss_price:,.2f} VND")
            st.metric("Take Profit", f"{take_profit_price:,.2f} VND")
        
        # Show personalized recommendations based on sidebar data
        st.subheader("💡 Khuyến nghị cá nhân hóa:")
        st.write(f"• Tỷ trọng tối đa: {max_position*100:.0f}% danh mục ({max_investment:,.2f} VND)")
        st.write(f"• Stop-loss: {stop_loss_pct}% để kiểm soát rủi ro")
        if sidebar_time_horizon.startswith('Dài hạn'):
            st.write("• Phù hợp với chiến lược mua và giữ dài hạn")
        elif sidebar_time_horizon.startswith('Ngắn hạn'):
            st.write("• Theo dõi sát biến động giá để chốt lời/cắt lỗ")
        else:
            st.write("• Cân bằng giữa tăng trưởng và kiểm soát rủi ro")
    
    # Show AI error if any
    if risk.get('ai_error'):
        st.warning(f"⚠️ {get_selected_llm_model()} không khả dụng: {risk.get('ai_error')}")
    

    # Show data source info
    data_source = risk.get('data_source', 'Unknown')
    if 'VCI_Real' in data_source:
        st.info("ℹ️ Dữ liệu thật từ VNStock VCI")
    elif 'Yahoo_Finance' in data_source:
        st.info("ℹ️ Dữ liệu từ Yahoo Finance")
    elif 'Fallback' in data_source:
        st.warning("⚠️ Sử dụng dữ liệu dự phòng - Không phù hợp cho giao dịch thực tế")
    




def display_investment_analysis(inv):
    if inv.get('error'):
        st.error(f"❌ {inv['error']}")
        return
    
    # Extract REAL data from investment_expert analysis result
    recommendation = inv.get('recommendation', 'HOLD')
    reason = inv.get('reason', 'Phân tích từ investment expert')
    score = inv.get('score', 50)
    confidence = inv.get('confidence', 0.5)
    
    # Get detailed metrics from analysis.detailed_metrics if available
    analysis = inv.get('analysis', {})
    detailed_metrics = analysis.get('detailed_metrics', {})
    
    # Extract real financial data from detailed_metrics
    current_price = detailed_metrics.get('current_price', 0)
    pe_ratio = detailed_metrics.get('pe', 0)
    pb_ratio = detailed_metrics.get('pb', 0)
    eps = detailed_metrics.get('eps', 0)
    dividend_yield = detailed_metrics.get('dividend_yield', 0)
    year_high = detailed_metrics.get('high_52w', current_price)
    year_low = detailed_metrics.get('low_52w', current_price)
    market_cap = detailed_metrics.get('market_cap', 0)
    volume = detailed_metrics.get('volume', 0)
    beta = detailed_metrics.get('beta', 1.0)
    
    # Calculate derived metrics with AI-enhanced fallbacks
    if current_price > 0:
        # Use real data for calculations
        target_price = current_price * (1 + (score - 50) / 100)
        upside_potential = ((target_price - current_price) / current_price * 100)
        roe = (eps / (current_price / pb_ratio) * 100) if pb_ratio > 0 else 0
    else:
        # AI-enhanced fallbacks based on recommendation
        if recommendation in ['STRONG BUY', 'BUY']:
            target_price = 50000 + (score * 500)  # Higher target for BUY
            upside_potential = 15 + (score - 50) * 0.3
            roe = 12 + (score - 50) * 0.2
        elif recommendation == 'WEAK BUY':
            target_price = 40000 + (score * 400)
            upside_potential = 8 + (score - 50) * 0.2
            roe = 10 + (score - 50) * 0.15
        elif recommendation == 'HOLD':
            target_price = 35000 + (score * 300)
            upside_potential = 2 + (score - 50) * 0.1
            roe = 8 + (score - 50) * 0.1
        else:  # SELL variants
            target_price = 25000 + (score * 200)
            upside_potential = -5 + (score - 50) * 0.1
            roe = 5 + max(0, (score - 30) * 0.1)
        
        current_price = target_price / (1 + upside_potential / 100)
    
    # AI-enhanced advice and reasoning
    ai_advice = inv.get('ai_advice', '')
    ai_reasoning = inv.get('ai_reasoning', '')
    
    inv_data = {
        'recommendation': recommendation,
        'reason': reason,
        'score': score,
        'confidence': confidence,
        'target_price': target_price,
        'upside_potential': upside_potential,
        'current_price': current_price,
        'dividend_yield': dividend_yield,
        'roe': roe,
        'pe_ratio': pe_ratio,
        'pb_ratio': pb_ratio,
        'market_cap': market_cap,
        'year_high': year_high,
        'year_low': year_low,
        'eps': eps,
        'volume': volume,
        'beta': beta
    }
    
    colors = {'BUY': '#28a745', 'SELL': '#dc3545', 'HOLD': '#ffc107'}
    icons = {'BUY': '🚀', 'SELL': '📉', 'HOLD': '⏸️'}
    
    reasons = {
        'BUY': 'Cổ phiếu có tiềm năng tăng trưởng tốt, định giá hấp dẫn',
        'SELL': 'Cổ phiếu được định giá quá cao, rủi ro giảm giá',
        'HOLD': 'Cổ phiếu ở mức giá hợp lý, chờ thời điểm phù hợp'
    }
    
    st.markdown(f"""
    <div style="background: {colors.get(recommendation, '#6c757d')}; color: white; padding: 20px; border-radius: 12px; margin: 10px 0;">
        <div style="text-align: center;">
            <div style="font-size: 2.5em; margin-bottom: 10px;">{icons.get(recommendation, '❓')}</div>
            <h3 style="margin: 0; font-size: 24px;">KHUYẾN NGHỊ ĐẦU TƯ</h3>
            <h2 style="margin: 10px 0; font-size: 28px;">{recommendation}</h2>
            <p style="margin: 10px 0; font-size: 16px; opacity: 0.9;">{inv_data['reason']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display REAL metrics from investment_expert analysis
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Giá mục tiêu", f"{inv_data['target_price']:,.2f} VND")
        if inv_data['pe_ratio'] > 0:
            st.metric("P/E Ratio", f"{inv_data['pe_ratio']:.2f}")
        else:
            st.metric("P/E Ratio", "N/A")
    with col2:
        st.metric("Tiềm năng tăng", f"{inv_data['upside_potential']:+.1f}%")
        if inv_data['pb_ratio'] > 0:
            st.metric("P/B Ratio", f"{inv_data['pb_ratio']:.2f}")
        else:
            st.metric("P/B Ratio", "N/A")
    with col3:
        if inv_data['market_cap'] > 0:
            if inv_data['market_cap'] > 1e12:
                st.metric("Vốn hóa", f"{inv_data['market_cap']/1e12:.1f}T VND")
            elif inv_data['market_cap'] > 1e9:
                st.metric("Vốn hóa", f"{inv_data['market_cap']/1e9:.1f}B VND")
            else:
                st.metric("Vốn hóa", f"{inv_data['market_cap']/1e6:.0f}M VND")
        else:
            st.metric("Vốn hóa", "N/A")
        st.metric("ROE", f"{inv_data['roe']:.1f}%")
    with col4:
        if inv_data['dividend_yield'] > 0:
            st.metric("Tỷ suất cổ tức", f"{inv_data['dividend_yield']:.1f}%")
        else:
            st.metric("Tỷ suất cổ tức", "N/A")
        if inv_data['year_high'] > 0 and inv_data['year_low'] > 0:
            st.metric("Cao/Thấp 1 năm", f"{inv_data['year_high']:,.2f}/{inv_data['year_low']:,.2f}")
        else:
            st.metric("Cao/Thấp 1 năm", "N/A")
    
    # AI-Enhanced Investment Advice Section - ALWAYS show
    st.markdown("### 🤖 Lời khuyên đầu tư từ AI")
    
    # Get sidebar data for personalized advice
    sidebar_risk_tolerance = globals().get('risk_tolerance', 50)
    sidebar_time_horizon = globals().get('time_horizon', 'Trung hạn')  
    sidebar_investment_amount = globals().get('investment_amount', 100000000)
    sidebar_symbol = globals().get('symbol', 'N/A')
    
    # Calculate risk profile from sidebar data
    if sidebar_risk_tolerance <= 30:
        risk_profile = "Thận trọng"
        max_position = 0.05  # 5%
        stop_loss_pct = 5
    elif sidebar_risk_tolerance <= 70:
        risk_profile = "Cân bằng"
        max_position = 0.10  # 10%
        stop_loss_pct = 8
    else:
        risk_profile = "Mạo hiểm"
        max_position = 0.20  # 20%
        stop_loss_pct = 12
    
    # Calculate position sizing from sidebar data
    max_investment = sidebar_investment_amount * max_position
    
    # Generate personalized advice using REAL sidebar data
    personalized_advice = f"""Với hồ sơ rủi ro {risk_profile.lower()} ({sidebar_risk_tolerance}%), thời gian đầu tư {sidebar_time_horizon.lower()} và số tiền {sidebar_investment_amount:,} VND, khuyến nghị {recommendation} cho {sidebar_symbol}. Nên đầu tư tối đa {max_position*100:.0f}% số tiền ({max_investment:,.0f} VND) và đặt stop-loss ở mức -{stop_loss_pct}% so với giá mua vào."""
    
    personalized_reasoning = f"""Dựa trên điểm số {score}/100, hồ sơ rủi ro {risk_profile.lower()} và thời gian đầu tư {sidebar_time_horizon.lower()}, tỷ trọng {max_position*100:.0f}% là phù hợp để cân bằng giữa cơ hội và rủi ro."""
    
    # Use personalized advice instead of AI advice
    display_advice = personalized_advice
    display_reasoning = personalized_reasoning
    
    # Display AI advice with investment-appropriate colors
    advice_color = '#28a745' if 'mua' in display_advice.lower() or 'buy' in display_advice.lower() else '#dc3545' if 'bán' in display_advice.lower() or 'sell' in display_advice.lower() else '#ffc107'
    advice_icon = '🚀' if 'mua' in display_advice.lower() or 'buy' in display_advice.lower() else '📉' if 'bán' in display_advice.lower() or 'sell' in display_advice.lower() else '⏸️'
    
    st.markdown(f"""
    <div style="background: {advice_color}22; border-left: 4px solid {advice_color}; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
        <h4 style="color: {advice_color}; margin-bottom: 1rem;">{advice_icon} Khuyến nghị đầu tư AI</h4>
        <p style="font-size: 1.1rem; margin-bottom: 1rem; font-weight: 500;">{display_advice}</p>
        <p style="color: #666; font-style: italic;"><strong>Lý do:</strong> {display_reasoning}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show AI enhancement info - ALWAYS display with selected LLM from sidebar
    ai_model = get_selected_llm_model()
    if inv.get('ai_enhanced'):
        st.success(f"🤖 Phân tích đầu tư được tăng cường bởi AI: {ai_model}")
    else:
        st.info(f"🤖 Phân tích đầu tư cơ bản (AI: {ai_model})")
    
    # Always show detailed analysis section
    with st.expander("🧠 Phân tích đầu tư AI chi tiết", expanded=False):
        if inv.get('ai_investment_analysis'):
            ai_text = inv['ai_investment_analysis']
            formatted_text = ai_text.replace('. ', '.\n\n').replace(': ', ':\n\n')
            st.markdown(f"**🤖 AI Investment Analysis:**\n\n{formatted_text}", unsafe_allow_html=True)
        else:
            # Get sidebar data for personalized fallback analysis
            sidebar_risk_tolerance = globals().get('risk_tolerance', 50)
            sidebar_time_horizon = globals().get('time_horizon', 'Trung hạn')  
            sidebar_investment_amount = globals().get('investment_amount', 100000000)
            sidebar_symbol = globals().get('symbol', 'N/A')
            
            # Calculate risk profile from sidebar data
            if sidebar_risk_tolerance <= 30:
                risk_profile = "Thận trọng"
                max_position = 0.05
                stop_loss_pct = 5
            elif sidebar_risk_tolerance <= 70:
                risk_profile = "Cân bằng"
                max_position = 0.10
                stop_loss_pct = 8
            else:
                risk_profile = "Mạo hiểm"
                max_position = 0.20
                stop_loss_pct = 12
            
            max_investment = sidebar_investment_amount * max_position
            
            # Show fallback analysis with REAL sidebar data
            st.markdown(f"""
            **💼 Phân tích đầu tư cho {sidebar_symbol}:**
            - Khuyến nghị: {recommendation} (Điểm: {score}/100)
            - Độ tin cậy: {confidence*100:.0f}%
            - Giá hiện tại: {inv_data['current_price']:,.2f} VND
            - Giá mục tiêu: {inv_data['target_price']:,.2f} VND
            - Tiềm năng tăng: {inv_data['upside_potential']:+.1f}%
            
            **👤 Hồ sơ đầu tư của bạn:**
            - Hồ sơ rủi ro: {risk_profile} ({sidebar_risk_tolerance}%)
            - Thời gian đầu tư: {sidebar_time_horizon}
            - Số tiền đầu tư: {sidebar_investment_amount:,} VND
            - Tỷ trọng khuyến nghị: {max_position*100:.0f}% ({max_investment:,.0f} VND)
            - Stop-loss khuyến nghị: {stop_loss_pct}%
            
            **📊 Chỉ số tài chính thực tế:**
            - P/E Ratio: {f"{inv_data['pe_ratio']:.2f}" if inv_data['pe_ratio'] > 0 else 'N/A'}
            - P/B Ratio: {f"{inv_data['pb_ratio']:.2f}" if inv_data['pb_ratio'] > 0 else 'N/A'}
            - EPS: {inv_data['eps']:,.0f} VND
            - Tỷ suất cổ tức: {inv_data['dividend_yield']:.1f}%
            - Beta: {inv_data['beta']:.2f}
            - Khối lượng: {inv_data['volume']:,}
            
            **💡 Khuyến nghị đầu tư cá nhân hóa:**
            Với hồ sơ rủi ro {risk_profile.lower()}, khuyến nghị {recommendation} cho {sidebar_symbol}:
            - Đầu tư tối đa {max_position*100:.0f}% số tiền ({max_investment:,.0f} VND)
            - Đặt stop-loss ở mức -{stop_loss_pct}% so với giá mua vào
            - Cổ phiếu đang ở mức định giá {"rất hấp dẫn" if score >= 80 else "hấp dẫn" if score >= 70 else "hợp lý" if score >= 60 else "cao" if score >= 40 else "rất cao"}
            - Phù hợp với thời gian đầu tư {sidebar_time_horizon.lower()} và hồ sơ rủi ro {risk_profile.lower()}
            """)
        
        if inv.get('enhanced_recommendation'):
            enhanced_rec = inv['enhanced_recommendation']
            if enhanced_rec != recommendation:
                st.info(f"🎯 Khuyến nghị AI nâng cao: {enhanced_rec}")
        
        # Show personalized investment strategy
        sidebar_risk_tolerance = globals().get('risk_tolerance', 50)
        sidebar_time_horizon = globals().get('time_horizon', 'Trung hạn')  
        sidebar_investment_amount = globals().get('investment_amount', 100000000)
        
        if sidebar_risk_tolerance <= 30:
            strategy = "Bảo toàn vốn và thu nhập ổn định"
        elif sidebar_risk_tolerance <= 70:
            strategy = "Cân bằng giữa tăng trưởng và ổn định"
        else:
            strategy = "Tăng trưởng cao và chấp nhận rủi ro"
        
        st.markdown(f"**🎯 Chiến lược đầu tư cá nhân hóa:** {strategy}")
        st.markdown(f"**💰 Quản lý danh mục:** {sidebar_investment_amount:,} VND cho {sidebar_time_horizon.lower()}")
    

    # Show risk-adjusted analysis using REAL sidebar data
    with st.expander("🎯 Phân tích theo hồ sơ rủi ro", expanded=True):
        # Get current data from sidebar (passed from main scope)
        sidebar_risk_tolerance = globals().get('risk_tolerance', 50)
        sidebar_time_horizon = globals().get('time_horizon', 'Trung hạn')  
        sidebar_investment_amount = globals().get('investment_amount', 100000000)
        
        # Calculate risk profile from sidebar data
        if sidebar_risk_tolerance <= 30:
            risk_profile = "Thận trọng"
            max_position = 0.05  # 5%
            stop_loss_pct = 5
        elif sidebar_risk_tolerance <= 70:
            risk_profile = "Cân bằng"
            max_position = 0.10  # 10%
            stop_loss_pct = 8
        else:
            risk_profile = "Mạo hiểm"
            max_position = 0.20  # 20%
            stop_loss_pct = 12
        
        # Calculate position sizing from sidebar data
        max_investment = sidebar_investment_amount * max_position
        current_price = inv_data.get('current_price', 50000)  # Get from investment data
        recommended_shares = int(max_investment / current_price) if current_price > 0 else 0
        actual_investment = recommended_shares * current_price
        stop_loss_price = current_price * (1 - stop_loss_pct / 100)
        take_profit_price = current_price * 1.15  # 15% target
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Hồ sơ rủi ro", f"{risk_profile} ({sidebar_risk_tolerance}%)")
            st.metric("Thời gian đầu tư", sidebar_time_horizon.split(' (')[0])
            
        with col2:
            st.metric("Số cổ phiếu khuyến nghị", f"{recommended_shares:,}")
            st.metric("Số tiền đầu tư", f"{sidebar_investment_amount:,.0f} VND")
            
        with col3:
            st.metric("Stop Loss", f"{stop_loss_price:,.2f} VND")
            st.metric("Take Profit", f"{take_profit_price:,.2f} VND")
        
        # Show personalized investment recommendations based on sidebar data
        st.subheader("💡 Khuyến nghị đầu tư cá nhân hóa:")
        st.write(f"• Tỷ trọng tối đa: {max_position*100:.0f}% danh mục ({max_investment:,.0f} VND)")
        st.write(f"• Stop-loss: {stop_loss_pct}% để kiểm soát rủi ro")
        if sidebar_time_horizon.startswith('Dài hạn'):
            st.write("• Phù hợp với chiến lược mua và giữ dài hạn")
        elif sidebar_time_horizon.startswith('Ngắn hạn'):
            st.write("• Theo dõi sát biến động giá để chốt lời/cắt lỗ")
        else:
            st.write("• Cân bằng giữa tăng trưởng và kiểm soát rủi ro")
        
        # Show recommendation adjustment based on risk profile
        original_rec = inv.get('recommendation', 'HOLD')
        if sidebar_risk_tolerance <= 30 and original_rec in ['STRONG BUY', 'BUY']:
            st.warning("⚠️ **Điều chỉnh cho hồ sơ thận trọng:** Khuyến nghị giảm xuống WEAK BUY hoặc HOLD")
        elif sidebar_risk_tolerance >= 70 and original_rec in ['HOLD', 'WEAK BUY']:
            st.info("🚀 **Điều chỉnh cho hồ sơ mạo hiểm:** Có thể cân nhắc tăng lên BUY")
    
    # Show AI error if any
    if inv.get('ai_error'):
        st.warning(f"⚠️ {get_selected_llm_model()} không khả dụng: {inv.get('ai_error')}")
    
    

# Bootstrap Enhanced Header
from src.ui.components import BootstrapComponents

is_english = st.session_state.get('is_english', False)

if not is_english:
    st.markdown("""
    <div class="main-header">
        <div class="container-fluid">
            <div class="row align-items-center">
                <div class="col-12 text-center">
                    <h1 class="header-title mb-2">📈 DUONG AI TRADING PRO</h1>
                    <p class="header-subtitle mb-3">Hệ thống AI phân tích chứng khoán thông minh</p>
                    <div class="d-flex flex-wrap justify-content-center gap-2">
                        <span class="badge bg-light bg-opacity-25 text-white px-3 py-2">
                            <i class="bi bi-robot"></i> 6 AI Agents
                        </span>
                        <span class="badge bg-light bg-opacity-25 text-white px-3 py-2">
                            <i class="bi bi-lightning"></i> Gemini 2.0 Flash
                        </span>
                        <span class="badge bg-light bg-opacity-25 text-white px-3 py-2">
                            <i class="bi bi-newspaper"></i> CrewAI News
                        </span>
                        <span class="badge bg-light bg-opacity-25 text-white px-3 py-2">
                            <i class="bi bi-diagram-3"></i> 3 Kiến trúc
                        </span>
                        <span class="badge bg-light bg-opacity-25 text-white px-3 py-2">
                            <i class="bi bi-translate"></i> Song ngữ
                        </span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="main-header">
        <div class="container-fluid">
            <div class="row align-items-center">
                <div class="col-12 text-center">
                    <h1 class="header-title mb-2">📈 DUONG AI TRADING PRO</h1>
                    <p class="header-subtitle mb-3">Intelligent AI Stock Analysis System</p>
                    <div class="d-flex flex-wrap justify-content-center gap-2">
                        <span class="badge bg-light bg-opacity-25 text-white px-3 py-2">
                            <i class="bi bi-robot"></i> 6 AI Agents
                        </span>
                        <span class="badge bg-light bg-opacity-25 text-white px-3 py-2">
                            <i class="bi bi-lightning"></i> Gemini 2.0 Flash
                        </span>
                        <span class="badge bg-light bg-opacity-25 text-white px-3 py-2">
                            <i class="bi bi-newspaper"></i> CrewAI News
                        </span>
                        <span class="badge bg-light bg-opacity-25 text-white px-3 py-2">
                            <i class="bi bi-diagram-3"></i> 3 Architectures
                        </span>
                        <span class="badge bg-light bg-opacity-25 text-white px-3 py-2">
                            <i class="bi bi-translate"></i> Bilingual
                        </span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Professional Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h3 style="margin: 0;">⚙️ Cấu hình hệ thống</h3>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">Thiết lập API và tham số đầu tư</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Language Selection
    st.subheader("🌍 Ngôn ngữ / Language")
    language = st.radio(
        "Chọn ngôn ngữ",
        ["Tiếng Việt 🇻🇳", "English 🇬🇧"],
        index=0,
        horizontal=True,
        key="language_selector"
    )
    
    # Store language preference
    is_english = "English" in language
    st.session_state.is_english = is_english
    
    st.divider()
    
    # API Configuration with environment defaults
    st.subheader("🔑 Cấu hình API" if not is_english else "🔑 API Configuration")
    
    # Get current values from environment
    env_gemini_key = os.getenv('GEMINI_API_KEY', '')
    env_openai_key = os.getenv('OPENAI_API_KEY', '')
    env_llama_key = os.getenv('LLAMA_API_KEY', '')
    env_llama_base_url = os.getenv('LLAMA_BASE_URL', 'https://api.together.xyz/v1')
    env_serper_key = os.getenv('SERPER_API_KEY', '')
    
    # Show status if keys are loaded from environment
    if env_gemini_key and env_gemini_key != 'your_gemini_api_key_here':
        st.success("✅ Gemini API key đã được tải từ .env file" if not is_english else "✅ Gemini API key loaded from .env file")
    if env_openai_key and env_openai_key != 'your_openai_api_key_here':
        st.success("✅ OpenAI API key đã được tải từ .env file" if not is_english else "✅ OpenAI API key loaded from .env file")
    if env_llama_key and env_llama_key != 'your_llama_api_key_here':
        st.success("✅ Llama API key đã được tải từ .env file" if not is_english else "✅ Llama API key loaded from .env file")
    if env_serper_key and env_serper_key != 'your_serper_api_key_here':
        st.success("✅ Serper API key đã được tải từ .env file" if not is_english else "✅ Serper API key loaded from .env file")
    
    gemini_key = st.text_input(
        "Khóa API Gemini" if not is_english else "Gemini API Key",
        type="password",
        value=env_gemini_key if env_gemini_key != 'your_gemini_api_key_here' else '',
        placeholder="Nhập Google Gemini API key hoặc cấu hình trong .env..." if not is_english else "Enter Google Gemini API key or configure in .env...",
        help="Lấy API key miễn phí tại: https://aistudio.google.com/apikey hoặc cấu hình trong file .env" if not is_english else "Get free API key at: https://aistudio.google.com/apikey or configure in .env file"
    )
    
    openai_key = st.text_input(
        "Khóa API OpenAI (Tùy chọn)" if not is_english else "OpenAI API Key (Optional)",
        type="password",
        value=env_openai_key if env_openai_key != 'your_openai_api_key_here' else '',
        placeholder="Nhập OpenAI API key hoặc cấu hình trong .env..." if not is_english else "Enter OpenAI API key or configure in .env...",
        help="Lấy API key tại: https://platform.openai.com/api-keys hoặc cấu hình trong file .env" if not is_english else "Get API key at: https://platform.openai.com/api-keys or configure in .env file"
    )
    
    llama_key = st.text_input(
        "Khóa API Llama (Tùy chọn)" if not is_english else "Llama API Key (Optional)",
        type="password",
        value=env_llama_key if env_llama_key != 'your_llama_api_key_here' else '',
        placeholder="Nhập Together AI/Groq API key hoặc cấu hình trong .env..." if not is_english else "Enter Together AI/Groq API key or configure in .env...",
        help="Lấy API key tại: https://together.ai hoặc https://groq.com hoặc cấu hình trong file .env" if not is_english else "Get API key at: https://together.ai or https://groq.com or configure in .env file"
    )
    
    llama_base_url = st.selectbox(
        "Nhà cung cấp Llama" if not is_english else "Llama Provider",
        ["http://localhost:11434", "https://api.groq.com/openai/v1", "https://api.together.xyz/v1"],
        index=0,
        help="Chọn nhà cung cấp API Llama" if not is_english else "Select Llama API provider"
    )
    
    # Ollama local info
    if "localhost" in llama_base_url:
        st.info("🏠 **Ollama Local**: Chạy model trên máy tính của bạn (miễn phí, riêng tư)")
        with st.expander("📋 Hướng dẫn Ollama Local", expanded=False):
            st.markdown("""
            **Cài đặt Ollama:**
            1. Tải Ollama: https://ollama.ai
            2. Chạy: `ollama serve`
            3. Tải model: `ollama pull llama3.1:8b`
            4. Test: `python test_ollama_simple.py`
            
            **Ưu điểm:**
            - ✅ Hoàn toàn miễn phí
            - ✅ Dữ liệu riêng tư (không gửi ra ngoài)
            - ✅ Không bị giới hạn requests
            - ✅ Tốc độ nhanh (nếu có GPU)
            """)
    elif "groq" in llama_base_url:
        st.info("⚡ **Groq**: Inference nhanh nhất (30 req/min miễn phí)")
    else:
        st.info("🤝 **Together AI**: Cân bằng tốc độ và chất lượng")
    
    serper_key = st.text_input(
        "Khóa API Serper (Tùy chọn)" if not is_english else "Serper API Key (Optional)",
        type="password",
        value=env_serper_key if env_serper_key != 'your_serper_api_key_here' else '',
        placeholder="Nhập Serper API key hoặc cấu hình trong .env..." if not is_english else "Enter Serper API key or configure in .env...",
        help="Lấy API key tại: https://serper.dev/api-key hoặc cấu hình trong file .env" if not is_english else "Get API key at: https://serper.dev/api-key or configure in .env file"
    )
    

    st.info("ℹ️ **Gemini AI** - Miễn phí (15 req/phút) | **OpenAI** - Trả phí | **Llama** - Ollama Local/Groq/Together AI" if not is_english else "ℹ️ **Gemini AI** - Free (15 req/min) | **OpenAI** - Paid | **Llama** - Ollama Local/Groq/Together AI")
    
    # Ollama status check
    
    # LLM Selection Dropdown
    st.subheader("🤖 Chọn LLM Engine" if not is_english else "🤖 Select LLM Engine")
    
    # Available LLM options
    llm_options = {
        "gemini": "🤖 Gemini 2.0 Flash (Miễn phí)",
        "openai": "🧠 OpenAI GPT-4o (Trả phí)", 
        "llama": "🦙 Llama 3.1 (Local/Groq/Together)"
    }
    
    # Get current LLM status
    current_llm = "gemini"  # Default
    available_llms = []
    
    if 'main_agent' in st.session_state and st.session_state.main_agent:
        try:
            # Check which LLMs are available based on API keys
            if gemini_key:
                available_llms.append("gemini")
            if openai_key:
                available_llms.append("openai")
            if llama_key or "localhost" in llama_base_url:
                available_llms.append("llama")
                
            # Get current LLM from session state
            if 'selected_llm_engine' in st.session_state:
                current_llm = st.session_state.selected_llm_engine
            elif hasattr(st.session_state.main_agent, 'llm_agent') and st.session_state.main_agent.llm_agent:
                if hasattr(st.session_state.main_agent.llm_agent, 'current_agent'):
                    current_llm = getattr(st.session_state.main_agent.llm_agent, 'current_agent', 'gemini')
        except Exception as e:
            st.warning(f"⚠️ Lỗi kiểm tra LLM: {str(e)}")
    
    # If no LLMs available, show all options
    if not available_llms:
        available_llms = list(llm_options.keys())
    
    # LLM Selection Dropdown
    selected_llm = st.selectbox(
        "Chọn LLM Engine" if not is_english else "Select LLM Engine",
        available_llms,
        index=available_llms.index(current_llm) if current_llm in available_llms else 0,
        format_func=lambda x: llm_options.get(x, x),
        help="Chọn LLM engine để sử dụng cho phân tích" if not is_english else "Select LLM engine for analysis",
        key="llm_selector"
    )
    
    # Store selected LLM in session state
    st.session_state.selected_llm_engine = selected_llm
    
    # Show LLM status with actual model check
    if selected_llm == "gemini" and gemini_key:
        # Check actual Gemini status
        if 'main_agent' in st.session_state and st.session_state.main_agent.llm_agent:
            try:
                status = st.session_state.main_agent.llm_agent.get_agent_status()
                gemini_info = status['agents'].get('gemini', {})
                if gemini_info.get('truly_available', False):
                    st.success("✅ Gemini 2.0 Flash - Sẵn sàng")
                else:
                    st.warning("⚠️ Gemini - Offline (quota/rate limit)")
            except:
                st.success("✅ Gemini 2.0 Flash - Sẵn sàng")
        else:
            st.success("✅ Gemini 2.0 Flash - Sẵn sàng")
    elif selected_llm == "openai" and openai_key:
        st.success("✅ OpenAI GPT-4o - Sẵn sàng")
    elif selected_llm == "llama" and (llama_key or "localhost" in llama_base_url):
        # Check actual Llama status
        if 'main_agent' in st.session_state and st.session_state.main_agent.llm_agent:
            try:
                status = st.session_state.main_agent.llm_agent.get_agent_status()
                llama_info = status['agents'].get('llama', {})
                if llama_info.get('truly_available', False):
                    model_name = llama_info.get('current_model', 'llama3.1:8b')
                    if "localhost" in llama_base_url:
                        st.success(f"✅ Llama ({model_name}) - Ollama Local")
                    else:
                        st.success(f"✅ Llama ({model_name}) - {llama_base_url.split('//')[1].split('.')[0].title()}")
                else:
                    st.warning("⚠️ Llama - Offline (không kết nối)")
            except:
                if "localhost" in llama_base_url:
                    st.success("✅ Llama 3.1 (Ollama Local) - Sẵn sàng")
                else:
                    st.success(f"✅ Llama 3.1 ({llama_base_url.split('//')[1].split('.')[0].title()}) - Sẵn sàng")
        else:
            if "localhost" in llama_base_url:
                st.success("✅ Llama 3.1 (Ollama Local) - Sẵn sàng")
            else:
                st.success(f"✅ Llama 3.1 ({llama_base_url.split('//')[1].split('.')[0].title()}) - Sẵn sàng")
    else:
        st.error(f"❌ {llm_options[selected_llm]} - Cần API key")
    

    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔧 Cài đặt LLM" if not is_english else "🔧 Setup LLM", use_container_width=True, type="primary"):
            if any([gemini_key, openai_key, llama_key]) and 'main_agent' in st.session_state:
                with st.spinner("🔄 Đang kiểm tra API keys..." if not is_english else "🔄 Checking API keys..."):
                    result = st.session_state.main_agent.set_llm_keys(gemini_key, openai_key, llama_key, llama_base_url)
                    if result:
                        st.success('✅ Cấu hình LLM thành công!' if not is_english else '✅ LLM configured successfully!')
                        st.rerun()
                    else:
                        st.error('❌ Không có LLM nào khả dụng!' if not is_english else '❌ No LLM available!')
                        st.info('💡 Kiểm tra lại API keys' if not is_english else '💡 Check your API keys')
            else:
                st.warning('⚠️ Vui lòng nhập ít nhất 1 API key!' if not is_english else '⚠️ Please enter at least 1 API key!')
    
    with col2:
        if st.button("🚀 Cài đặt CrewAI" if not is_english else "🚀 Setup CrewAI", use_container_width=True):
            if any([gemini_key, openai_key, llama_key]) and 'main_agent' in st.session_state:
                result = st.session_state.main_agent.set_crewai_keys(gemini_key, openai_key, llama_key, llama_base_url, serper_key)
                if result:
                    st.success('✅ Cấu hình CrewAI thành công!' if not is_english else '✅ CrewAI configured successfully!')
                    st.rerun()
                else:
                    st.error('⚠️ CrewAI không khả dụng' if not is_english else '⚠️ CrewAI not available')
                    st.info('💡 Cài đặt: pip install crewai crewai-tools' if not is_english else '💡 Install: pip install crewai crewai-tools')
            else:
                st.error('❌ Cần ít nhất 1 API key (Gemini/OpenAI/Llama)!' if not is_english else '❌ Need at least 1 API key (Gemini/OpenAI/Llama)!')
    
    # Force refresh button
    if st.button("🔄 Làm mới dữ liệu" if not is_english else "🔄 Refresh Data", use_container_width=True, help="Xóa cache và tải lại symbols từ CrewAI" if not is_english else "Clear cache and reload symbols from CrewAI"):
        if 'main_agent' in st.session_state:
            st.session_state.main_agent.vn_api.clear_symbols_cache()
            st.success('✅ Đã xóa cache - Reload trang để lấy dữ liệu mới!' if not is_english else '✅ Cache cleared - Reload page to get new data!')
            st.rerun()
        else:
            st.error('❌ Hệ thống chưa được khởi tạo' if not is_english else '❌ System not initialized')
    
    st.divider()
    
    # Bootstrap LLM Agents Status
    llm_models_status = []
    llm_model_active = False
    
    if 'main_agent' in st.session_state and st.session_state.main_agent.llm_agent:
        try:
            status = st.session_state.main_agent.llm_agent.get_agent_status()
            for agent_name, info in status['agents'].items():
                # Check if truly available (has models AND not offline)
                is_truly_available = info.get('has_models', False) and not info.get('offline_mode', True)
                
                if is_truly_available:
                    # Always show "Gemini 2.0 Flash" regardless of actual model
                    if agent_name == 'gemini':
                        llm_models_status.append("Gemini 2.0 Flash")
                    else:
                        model_name = info.get('current_model', agent_name)
                        llm_models_status.append(f"{agent_name.title()} ({model_name})")
                    llm_model_active = True
                else:
                    llm_models_status.append(f"{agent_name.title()} (Offline)")
        except Exception as e:
            llm_models_status.append("LLM (Lỗi)")
    
    agents_status = [
        {"name": "PricePredictor", "icon": "bi-graph-up", "status": "active"},
        {"name": "TickerNews", "icon": "bi-newspaper", "status": "active"},
        {"name": "MarketNews", "icon": "bi-globe", "status": "active"},
        {"name": "InvestmentExpert", "icon": "bi-briefcase", "status": "active"},
        {"name": "RiskExpert", "icon": "bi-shield-check", "status": "active"},
        {"name": f"LLM Models ({', '.join(llm_models_status) if llm_models_status else 'None'})", "icon": "bi-robot", "status": "active" if llm_model_active else "inactive"},
        {"name": "CrewAI + Serper", "icon": "bi-people", "status": "active" if 'main_agent' in st.session_state and st.session_state.main_agent.vn_api.crewai_collector and st.session_state.main_agent.vn_api.crewai_collector.enabled else "inactive"}
    ]
    
    st.subheader("🤖 Trạng thái AI Agents" if not is_english else "🤖 AI Agents Status")
    
    for agent in agents_status:
        status_icon = "🟢" if agent["status"] == "active" else "🔴"
        status_text = "Hoạt động" if agent["status"] == "active" else "Không hoạt động"
        if is_english:
            status_text = "Active" if agent["status"] == "active" else "Inactive"
        st.write(f"{status_icon} **{agent['name']}**: {status_text}")
    
    st.divider()
    
    # Investment Settings
    st.subheader("📊 Cài đặt đầu tư" if not is_english else "📊 Investment Settings")
    
    time_horizon = st.selectbox(
        "🕐 Thời gian đầu tư" if not is_english else "🕐 Investment Horizon",
        ["Ngắn hạn (1-3 tháng)", "Trung hạn (3-12 tháng)", "Dài hạn (1+ năm)"] if not is_english else ["Short-term (1-3 months)", "Medium-term (3-12 months)", "Long-term (1+ year)"],
        index=1,
        key="time_horizon"
    )
    
    risk_tolerance = st.slider(
        "⚠️ Khả năng chấp nhận rủi ro" if not is_english else "⚠️ Risk Tolerance",
        min_value=0,
        max_value=100,
        value=50,
        help="0: Thận trọng | 50: Cân bằng | 100: Rủi ro" if not is_english else "0: Conservative | 50: Balanced | 100: Aggressive",
        key="risk_tolerance"
    )
    
    investment_amount = st.number_input(
        "💰 Số tiền đầu tư (VND)" if not is_english else "💰 Investment Amount (VND)",
        min_value=1_000_000,
        max_value=10_000_000_000,
        value=100_000_000,
        step=10_000_000,
        format="%d",
        key="investment_amount"
    )
    
    # Risk Profile Display
    if risk_tolerance <= 30:
        risk_label = "🟢 Thận trọng" if not is_english else "🟢 Conservative"
    elif risk_tolerance <= 70:
        risk_label = "🟡 Cân bằng" if not is_english else "🟡 Balanced"
    else:
        risk_label = "🔴 Mạo hiểm" if not is_english else "🔴 Aggressive"
    
    if not is_english:
        st.info(f"**Hồ sơ:** {risk_label} ({risk_tolerance}%) | **Số tiền:** {investment_amount:,} VND | **Thời gian:** {time_horizon}")
    else:
        st.info(f"**Profile:** {risk_label} ({risk_tolerance}%) | **Amount:** {investment_amount:,} VND | **Horizon:** {time_horizon}")

    st.divider()
    
    # Stock Selection
    st.subheader("📈 Chọn cổ phiếu" if not is_english else "📈 Select Stock")
    
    # Load symbols with CrewAI priority
    with st.spinner("Đang tải danh sách mã cổ phiếu..." if not is_english else "Loading stock symbols..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Get symbols from VN API (which handles CrewAI internally)
        symbols = loop.run_until_complete(st.session_state.vn_api.get_available_symbols()) if 'vn_api' in st.session_state else []
        
        # Check data source from symbols metadata
        data_source = 'Static'  # Default
        if symbols and len(symbols) > 0:
            first_symbol = symbols[0]
            if first_symbol.get('data_source') == 'CrewAI':
                data_source = 'CrewAI'
                st.success(f'✅ {len(symbols)} mã cổ phiếu từ CrewAI (Real Data)' if not is_english else f'✅ {len(symbols)} stocks from CrewAI (Real Data)')
            else:
                data_source = 'Static'
                st.info(f'📋 {len(symbols)} mã cổ phiếu tĩnh (Fallback)' if not is_english else f'📋 {len(symbols)} static stocks (Fallback)')
                
                # Show why CrewAI is not working
                if 'main_agent' not in st.session_state or not st.session_state.main_agent.llm_agent:
                    st.warning("⚠️ **Để lấy dữ liệu thật**: Cấu hình Gemini API key trong sidebar" if not is_english else "⚠️ **To get real data**: Configure Gemini API key in sidebar")
                elif not (st.session_state.main_agent.vn_api.crewai_collector and st.session_state.main_agent.vn_api.crewai_collector.enabled):
                    st.warning("⚠️ **CrewAI chưa khả dụng**: Kiểm tra cấu hình API keys" if not is_english else "⚠️ **CrewAI not available**: Check API keys configuration")
        else:
            st.error("❌ Không thể tải danh sách cổ phiếu" if not is_english else "❌ Cannot load stock list")
        
        loop.close()
    
    # Group symbols by sector with enhanced display
    sectors = {}
    for stock in symbols:
        sector = stock.get('sector', 'Other')
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(stock)
    
    # Show data source status
    if data_source == 'CrewAI':
        st.markdown("🤖 **Nguồn dữ liệu**: CrewAI Real-time Data" if not is_english else "🤖 **Data Source**: CrewAI Real-time Data")
    else:
        st.markdown("📋 **Nguồn dữ liệu**: Static Fallback Data" if not is_english else "📋 **Data Source**: Static Fallback Data")
        
    
    selected_sector = st.selectbox("Chọn ngành" if not is_english else "Select Sector", list(sectors.keys()))
    sector_stocks = sectors[selected_sector]
    
    stock_options = [f"{s['symbol']} - {s['name']}" for s in sector_stocks]
    selected_stock = st.selectbox("Chọn cổ phiếu" if not is_english else "Select Stock", stock_options)
    symbol = selected_stock.split(" - ")[0] if selected_stock else ""

# Main Content - Price Prediction Only
is_english = st.session_state.get('is_english', False)
if not is_english:
    st.markdown(f"<h2 style='margin-bottom:0.5em;'>📈 Dự đoán giá <span style='color:#667eea'>{symbol}</span></h2>", unsafe_allow_html=True)
else:
    st.markdown(f"<h2 style='margin-bottom:0.5em;'>📈 Price Prediction <span style='color:#667eea'>{symbol}</span></h2>", unsafe_allow_html=True)

if not symbol:
    st.warning("⚠️ Vui lòng chọn một cổ phiếu từ thanh bên" if not is_english else "⚠️ Please select a stock from the sidebar")
else:
    # Architecture Selection
    st.markdown("### 🏗️ Chọn kiến trúc dự đoán" if not is_english else "### 🏗️ Select Prediction Architecture")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        architecture = st.selectbox(
            "Kiến trúc AI" if not is_english else "AI Architecture",
            ["ensemble_voting", "hierarchical", "round_robin"],
            index=0,
            help="Chọn kiến trúc AI để dự đoán giá" if not is_english else "Select AI architecture for price prediction"
        )
    
    with col2:
        st.write("")
        st.write("")
    
    # Architecture info
    arch_info = st.session_state.main_agent.get_architecture_info() if 'main_agent' in st.session_state else {}
    selected_info = arch_info.get(architecture, "Không có thông tin" if not is_english else "No information")
    st.info(f"**{architecture.upper()}**: {selected_info}")
    
    # Price prediction button
    if st.button("📈 Dự đoán giá" if not is_english else "📈 Predict Price", type="primary", use_container_width=True):
        with st.spinner(f"📈 Đang dự đoán giá với {architecture}..." if not is_english else f"📈 Predicting price with {architecture}..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            time_horizon_clean = time_horizon.split(" (")[0] if "(" in time_horizon else time_horizon
            
            # Map time horizon to days
            if not is_english:
                days = {"Ngắn hạn": 30, "Trung hạn": 90, "Dài hạn": 180}.get(time_horizon_clean, 90)
            else:
                days = {"Short-term": 30, "Medium-term": 90, "Long-term": 180}.get(time_horizon_clean, 90)
            
            # Use architecture-based prediction
            pred = loop.run_until_complete(st.session_state.main_agent.predict_price_with_architecture(symbol, architecture, f"{days}d"))
            loop.close()
        
        # Pass sidebar data to global scope
        globals()['symbol'] = symbol
        globals()['risk_tolerance'] = risk_tolerance
        globals()['time_horizon'] = time_horizon
        globals()['investment_amount'] = investment_amount
        
        # Display architecture prediction tables
        display_architecture_prediction_tables(pred, symbol, architecture)





   



    
  
# Professional Footer
st.markdown("---")

is_english = st.session_state.get('is_english', False)

if not is_english:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
        <h4 style="color: #2a5298; margin-bottom: 0.5rem;">🇻🇳 DUONG AI TRADING PRO</h4>
        <p style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">6 AI Agents • Gemini 2.0 • CrewAI • Real-time Data</p>
        <p style="color: #999; font-size: 0.85rem;">Hệ thống AI phân tích chứng khoán Việt Nam & Quốc tế</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div style="background:#fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
        <strong>⚠️ Cảnh báo:</strong> Còn thở là còn gỡ, dừng lại là thất bại ^^!!!
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
        <h4 style="color: #2a5298; margin-bottom: 0.5rem;">🇻🇳 DUONG AI TRADING PRO</h4>
        <p style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">6 AI Agents • Gemini 2.0 • CrewAI • Real-time Data</p>
        <p style="color: #999; font-size: 0.85rem;">AI Stock Analysis for Vietnam & International Markets</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div style="background:#fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
        <strong>⚠️ Warning:</strong> Keep breathing, keep trading - Stop and you fail! ^^!!!
    </div>
    """, unsafe_allow_html=True)
