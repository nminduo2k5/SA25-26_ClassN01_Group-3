import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Configuration
GATEWAY_URL = "http://nginx"  # Nginx gateway
SERVICES = {
    "price_predictor": f"{GATEWAY_URL}/api/price",
    "investment_expert": f"{GATEWAY_URL}/api/investment", 
    "risk_expert": f"{GATEWAY_URL}/api/risk",
    "news_agent": f"{GATEWAY_URL}/api/news",
    "market_news": f"{GATEWAY_URL}/api/market",
    "stock_info": f"{GATEWAY_URL}/api/stock",
    "llm_hub": f"{GATEWAY_URL}/api/llm"
}

# Page config
st.set_page_config(
    page_title="Multi-Agent Stock Analysis System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .service-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🤖 Multi-Agent Stock Analysis System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Service status
        st.subheader("🔧 Service Status")
        check_services()
        
        # Stock selection
        st.subheader("📊 Stock Selection")
        symbol = st.text_input("Stock Symbol", value="VCB", help="Enter stock symbol (e.g., VCB, AAPL)")
        
        # Investment parameters
        st.subheader("💰 Investment Parameters")
        risk_tolerance = st.slider("Risk Tolerance", 0, 100, 50, help="0=Conservative, 100=Aggressive")
        time_horizon = st.selectbox("Time Horizon", ["short", "medium", "long"])
        investment_amount = st.number_input("Investment Amount (VND)", min_value=1000000, value=10000000, step=1000000)
        
        # Analysis button
        if st.button("🚀 Run Analysis", type="primary"):
            st.session_state.run_analysis = True
            st.session_state.symbol = symbol
            st.session_state.risk_tolerance = risk_tolerance
            st.session_state.time_horizon = time_horizon
            st.session_state.investment_amount = investment_amount
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Price Prediction", 
        "💼 Investment Analysis", 
        "⚠️ Risk Assessment",
        "📰 News Analysis",
        "🌍 Market Overview",
        "🤖 AI Chat"
    ])
    
    # Run analysis if requested
    if st.session_state.get('run_analysis', False):
        with st.spinner("🔄 Running comprehensive analysis..."):
            results = run_comprehensive_analysis(
                st.session_state.symbol,
                st.session_state.risk_tolerance,
                st.session_state.time_horizon,
                st.session_state.investment_amount
            )
            st.session_state.analysis_results = results
            st.session_state.run_analysis = False
    
    # Display results in tabs
    with tab1:
        display_price_prediction()
    
    with tab2:
        display_investment_analysis()
    
    with tab3:
        display_risk_assessment()
    
    with tab4:
        display_news_analysis()
    
    with tab5:
        display_market_overview()
    
    with tab6:
        display_ai_chat()

def check_services():
    """Check status of all microservices"""
    service_status = {}
    
    for service_name, base_url in SERVICES.items():
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                service_status[service_name] = "✅ Online"
            else:
                service_status[service_name] = "❌ Error"
        except:
            service_status[service_name] = "🔴 Offline"
    
    for service, status in service_status.items():
        st.write(f"{service}: {status}")

def run_comprehensive_analysis(symbol, risk_tolerance, time_horizon, investment_amount):
    """Run analysis across all microservices"""
    results = {}
    
    # Price prediction
    try:
        response = requests.post(f"{SERVICES['price_predictor']}/predict", 
            json={
                "symbol": symbol,
                "days": 30,
                "risk_tolerance": risk_tolerance,
                "time_horizon": time_horizon,
                "investment_amount": investment_amount
            }, timeout=30)
        
        if response.status_code == 200:
            results['price_prediction'] = response.json()
        else:
            results['price_prediction'] = {"error": f"Service error: {response.status_code}"}
    except Exception as e:
        results['price_prediction'] = {"error": str(e)}
    
    # Investment analysis
    try:
        response = requests.post(f"{SERVICES['investment_expert']}/analyze",
            json={
                "symbol": symbol,
                "risk_tolerance": risk_tolerance,
                "time_horizon": time_horizon,
                "investment_amount": investment_amount
            }, timeout=30)
        
        if response.status_code == 200:
            results['investment_analysis'] = response.json()
        else:
            results['investment_analysis'] = {"error": f"Service error: {response.status_code}"}
    except Exception as e:
        results['investment_analysis'] = {"error": str(e)}
    
    # Add other service calls here...
    
    return results

def display_price_prediction():
    """Display price prediction results"""
    st.header("📈 Price Prediction Analysis")
    
    if 'analysis_results' not in st.session_state:
        st.info("👆 Please run analysis from the sidebar to see results")
        return
    
    prediction_data = st.session_state.analysis_results.get('price_prediction', {})
    
    if prediction_data.get('error'):
        st.error(f"❌ Price Prediction Error: {prediction_data['error']}")
        return
    
    # Current vs Predicted Price
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Price",
            f"{prediction_data.get('current_price', 0):,.0f} VND"
        )
    
    with col2:
        predicted_price = prediction_data.get('predicted_price', 0)
        st.metric(
            "Predicted Price (30d)",
            f"{predicted_price:,.0f} VND"
        )
    
    with col3:
        change_percent = prediction_data.get('change_percent', 0)
        st.metric(
            "Expected Change",
            f"{change_percent:+.2f}%",
            delta=f"{change_percent:+.2f}%"
        )
    
    with col4:
        confidence = prediction_data.get('confidence', 0)
        st.metric(
            "Confidence",
            f"{confidence:.1f}%"
        )
    
    # Prediction chart
    if prediction_data.get('predictions'):
        st.subheader("📊 Multi-Timeframe Predictions")
        
        predictions = prediction_data['predictions']
        chart_data = []
        
        for timeframe, preds in predictions.items():
            for period, data in preds.items():
                days = int(period.split('_')[0])
                chart_data.append({
                    'Days': days,
                    'Price': data['price'],
                    'Timeframe': timeframe.replace('_', ' ').title()
                })
        
        if chart_data:
            df = pd.DataFrame(chart_data)
            
            fig = px.line(df, x='Days', y='Price', color='Timeframe',
                         title="Price Predictions by Timeframe",
                         markers=True)
            
            # Add current price line
            fig.add_hline(y=prediction_data.get('current_price', 0), 
                         line_dash="dash", 
                         annotation_text="Current Price")
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators
    if prediction_data.get('technical_indicators'):
        st.subheader("🔧 Technical Indicators")
        
        tech_data = prediction_data['technical_indicators']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write("**RSI**")
            rsi = tech_data.get('rsi', 50)
            st.write(f"{rsi:.1f}")
            if rsi > 70:
                st.write("🔴 Overbought")
            elif rsi < 30:
                st.write("🟢 Oversold")
            else:
                st.write("🟡 Neutral")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write("**MACD**")
            macd = tech_data.get('macd', 0)
            st.write(f"{macd:.4f}")
            if macd > 0:
                st.write("🟢 Bullish")
            else:
                st.write("🔴 Bearish")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write("**Volatility**")
            volatility = tech_data.get('volatility', 20)
            st.write(f"{volatility:.1f}%")
            if volatility > 30:
                st.write("🔴 High")
            elif volatility < 15:
                st.write("🟢 Low")
            else:
                st.write("🟡 Medium")
            st.markdown('</div>', unsafe_allow_html=True)

def display_investment_analysis():
    """Display investment analysis results"""
    st.header("💼 Investment Analysis")
    
    if 'analysis_results' not in st.session_state:
        st.info("👆 Please run analysis from the sidebar to see results")
        return
    
    investment_data = st.session_state.analysis_results.get('investment_analysis', {})
    
    if investment_data.get('error'):
        st.error(f"❌ Investment Analysis Error: {investment_data['error']}")
        return
    
    # Recommendation
    recommendation = investment_data.get('recommendation', 'HOLD')
    confidence = investment_data.get('confidence', 50)
    
    # Color coding for recommendations
    if recommendation in ['STRONG BUY', 'BUY']:
        rec_color = "🟢"
        rec_style = "success-box"
    elif recommendation in ['STRONG SELL', 'SELL']:
        rec_color = "🔴"
        rec_style = "error-box"
    else:
        rec_color = "🟡"
        rec_style = "metric-card"
    
    st.markdown(f'<div class="{rec_style}">', unsafe_allow_html=True)
    st.markdown(f"## {rec_color} Recommendation: **{recommendation}**")
    st.markdown(f"**Confidence:** {confidence:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Investment metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Target Price",
            f"{investment_data.get('target_price', 0):,.0f} VND"
        )
    
    with col2:
        st.metric(
            "Stop Loss",
            f"{investment_data.get('stop_loss', 0):,.0f} VND"
        )
    
    with col3:
        st.metric(
            "Position Size",
            f"{investment_data.get('position_size', 0):,} shares"
        )
    
    with col4:
        risk_level = investment_data.get('risk_assessment', {}).get('risk_level', 'Medium')
        st.metric(
            "Risk Level",
            risk_level
        )
    
    # Investment rationale
    if investment_data.get('investment_rationale'):
        st.subheader("📝 Investment Rationale")
        st.write(investment_data['investment_rationale'])
    
    # Financial metrics
    if investment_data.get('financial_metrics'):
        st.subheader("📊 Financial Metrics")
        
        metrics = investment_data['financial_metrics']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Estimated P/E Ratio:**", f"{metrics.get('estimated_pe', 0):.2f}")
            st.write("**Estimated P/B Ratio:**", f"{metrics.get('estimated_pb', 0):.2f}")
            st.write("**Momentum Score:**", f"{metrics.get('momentum_score', 0):.1f}/100")
        
        with col2:
            st.write("**Value Score:**", f"{metrics.get('value_score', 0):.1f}/100")
            st.write("**Overall Score:**", f"{metrics.get('overall_score', 0):.1f}/100")
            st.write("**Volatility:**", f"{metrics.get('volatility', 0):.2f}%")

def display_risk_assessment():
    """Display risk assessment results"""
    st.header("⚠️ Risk Assessment")
    
    if 'analysis_results' not in st.session_state:
        st.info("👆 Please run analysis from the sidebar to see results")
        return
    
    # For now, extract risk data from investment analysis
    investment_data = st.session_state.analysis_results.get('investment_analysis', {})
    risk_data = investment_data.get('risk_assessment', {})
    
    if not risk_data:
        st.warning("⚠️ Risk assessment data not available")
        return
    
    # Risk metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_score = risk_data.get('risk_score', 50)
        st.metric("Risk Score", f"{risk_score:.1f}/100")
        
        if risk_score < 30:
            st.success("🟢 Low Risk")
        elif risk_score < 60:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")
    
    with col2:
        risk_level = risk_data.get('risk_level', 'Medium')
        st.metric("Risk Level", risk_level)
    
    with col3:
        risk_match = risk_data.get('risk_tolerance_match', 50)
        st.metric("Profile Match", f"{risk_match:.1f}%")
    
    # Risk suitability
    suitable = risk_data.get('suitable_for_profile', False)
    if suitable:
        st.success("✅ This investment is suitable for your risk profile")
    else:
        st.warning("⚠️ This investment may not match your risk tolerance")
    
    # Risk breakdown chart
    st.subheader("📊 Risk Breakdown")
    
    risk_components = {
        'Volatility Risk': risk_data.get('volatility', 20),
        'Market Risk': risk_score * 0.6,
        'Liquidity Risk': max(10, risk_score * 0.3),
        'Company Risk': max(15, risk_score * 0.4)
    }
    
    fig = go.Figure(data=[
        go.Bar(x=list(risk_components.keys()), 
               y=list(risk_components.values()),
               marker_color=['red' if v > 50 else 'orange' if v > 30 else 'green' for v in risk_components.values()])
    ])
    
    fig.update_layout(
        title="Risk Component Analysis",
        yaxis_title="Risk Level",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_news_analysis():
    """Display news analysis results"""
    st.header("📰 News Analysis")
    
    # Placeholder for news service
    st.info("🔄 News analysis service integration in progress...")
    
    # Mock news data for demonstration
    mock_news = [
        {
            "title": "VCB báo cáo lợi nhuận quý 3 tăng 15%",
            "sentiment": "Positive",
            "impact": "High",
            "source": "CafeF",
            "time": "2 hours ago"
        },
        {
            "title": "Ngân hàng Nhà nước điều chỉnh lãi suất",
            "sentiment": "Neutral",
            "impact": "Medium", 
            "source": "VnEconomy",
            "time": "4 hours ago"
        },
        {
            "title": "Thị trường chứng khoán Việt Nam thu hút vốn ngoại",
            "sentiment": "Positive",
            "impact": "Medium",
            "source": "VietStock",
            "time": "6 hours ago"
        }
    ]
    
    for news in mock_news:
        with st.expander(f"📰 {news['title']}"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sentiment_color = "🟢" if news['sentiment'] == 'Positive' else "🔴" if news['sentiment'] == 'Negative' else "🟡"
                st.write(f"**Sentiment:** {sentiment_color} {news['sentiment']}")
            
            with col2:
                st.write(f"**Impact:** {news['impact']}")
            
            with col3:
                st.write(f"**Source:** {news['source']}")
            
            with col4:
                st.write(f"**Time:** {news['time']}")

def display_market_overview():
    """Display market overview"""
    st.header("🌍 Market Overview")
    
    # Market indices (mock data)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("VN-Index", "1,234.56", "+12.34 (+1.01%)")
    
    with col2:
        st.metric("HNX-Index", "234.56", "-2.34 (-0.99%)")
    
    with col3:
        st.metric("UPCOM-Index", "89.12", "+0.45 (+0.51%)")
    
    # Top movers
    st.subheader("📈 Top Movers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🟢 Top Gainers**")
        gainers_data = {
            'Symbol': ['VCB', 'BID', 'CTG'],
            'Price': [95000, 48500, 32100],
            'Change': ['+2.5%', '+1.8%', '+1.2%']
        }
        st.dataframe(pd.DataFrame(gainers_data), hide_index=True)
    
    with col2:
        st.write("**🔴 Top Losers**")
        losers_data = {
            'Symbol': ['VIC', 'VHM', 'MSN'],
            'Price': [78500, 65200, 125000],
            'Change': ['-1.5%', '-2.1%', '-0.8%']
        }
        st.dataframe(pd.DataFrame(losers_data), hide_index=True)

def display_ai_chat():
    """Display AI chat interface"""
    st.header("🤖 AI Investment Assistant")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your investments..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = get_ai_response(prompt)
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

def get_ai_response(prompt):
    """Get AI response from LLM Hub service"""
    try:
        response = requests.post(f"{SERVICES['llm_hub']}/generate",
            json={
                "prompt": f"Investment question: {prompt}",
                "model": "gemini",
                "temperature": 0.7
            }, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('response', 'Sorry, I could not process your request.')
        else:
            return "🔧 AI service is currently unavailable. Please try again later."
            
    except Exception as e:
        return f"❌ Error connecting to AI service: {str(e)}"

if __name__ == "__main__":
    main()