import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Configuration
GATEWAY_URL = "http://localhost:8080"  # Updated port
SERVICES = {
    "price_predictor": f"{GATEWAY_URL}/api/price",
    "investment_expert": f"{GATEWAY_URL}/api/investment", 
    "llm_hub": f"{GATEWAY_URL}/api/llm"
}

# Page config
st.set_page_config(
    page_title="Multi-Agent Stock Analysis System",
    page_icon="📈",
    layout="wide"
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
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
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
        symbol = st.text_input("Stock Symbol", value="VCB")
        
        # Investment parameters
        st.subheader("💰 Investment Parameters")
        risk_tolerance = st.slider("Risk Tolerance", 0, 100, 50)
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
    tab1, tab2, tab3 = st.tabs(["📈 Price Prediction", "💼 Investment Analysis", "🤖 AI Chat"])
    
    # Run analysis if requested
    if st.session_state.get('run_analysis', False):
        with st.spinner("🔄 Running analysis..."):
            results = run_analysis(
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
        display_ai_chat()

def check_services():
    """Check status of microservices"""
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

def run_analysis(symbol, risk_tolerance, time_horizon, investment_amount):
    """Run analysis across microservices"""
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
    
    return results

def display_price_prediction():
    """Display price prediction results"""
    st.header("📈 Price Prediction Analysis")
    
    if 'analysis_results' not in st.session_state:
        st.info("👆 Please run analysis from the sidebar")
        return
    
    prediction_data = st.session_state.analysis_results.get('price_prediction', {})
    
    if prediction_data.get('error'):
        st.error(f"❌ Error: {prediction_data['error']}")
        return
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"{prediction_data.get('current_price', 0):,.0f} VND")
    
    with col2:
        predicted_price = prediction_data.get('predicted_price', 0)
        st.metric("Predicted Price (30d)", f"{predicted_price:,.0f} VND")
    
    with col3:
        change_percent = prediction_data.get('change_percent', 0)
        st.metric("Expected Change", f"{change_percent:+.2f}%", delta=f"{change_percent:+.2f}%")
    
    with col4:
        confidence = prediction_data.get('confidence', 0)
        st.metric("Confidence", f"{confidence:.1f}%")
    
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
        st.info("👆 Please run analysis from the sidebar")
        return
    
    investment_data = st.session_state.analysis_results.get('investment_analysis', {})
    
    if investment_data.get('error'):
        st.error(f"❌ Error: {investment_data['error']}")
        return
    
    # Recommendation
    recommendation = investment_data.get('recommendation', 'HOLD')
    confidence = investment_data.get('confidence', 50)
    
    # Color coding
    if recommendation in ['STRONG BUY', 'BUY']:
        rec_color = "🟢"
        rec_style = "success"
    elif recommendation in ['STRONG SELL', 'SELL']:
        rec_color = "🔴"
        rec_style = "error"
    else:
        rec_color = "🟡"
        rec_style = "info"
    
    st.markdown(f"## {rec_color} Recommendation: **{recommendation}**")
    st.markdown(f"**Confidence:** {confidence:.1f}%")
    
    # Investment metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Target Price", f"{investment_data.get('target_price', 0):,.0f} VND")
    
    with col2:
        st.metric("Stop Loss", f"{investment_data.get('stop_loss', 0):,.0f} VND")
    
    with col3:
        st.metric("Position Size", f"{investment_data.get('position_size', 0):,} shares")
    
    # Investment rationale
    if investment_data.get('investment_rationale'):
        st.subheader("📝 Investment Rationale")
        st.write(investment_data['investment_rationale'])

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
    if prompt := st.chat_input("Ask me about investments..."):
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