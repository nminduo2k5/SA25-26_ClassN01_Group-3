# Design and Evaluation of Multi-Agent Architectures for Stock Price Prediction: A Vietnam Case Study

<!-- Author information -->
**Author:** Nguyen Minh Duong  
**Student ID:** 23010441  
**Group:** 3

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Google-Gemini-orange.svg)](https://ai.google.dev)
[![CrewAI](https://img.shields.io/badge/CrewAI-0.117+-purple.svg)](https://crewai.com)

> **Intelligent investment analysis system with 6 AI Agents (4 Architectures) + Gemini AI + CrewAI + LSTM Neural Network**

## 🎯 Overview

The Multi-Agent Vietnam Stock system is a complete investment analysis platform that integrates 6 specialized AI agents, Google Gemini, and LSTM neural networks to provide comprehensive analysis for the Vietnamese and international stock markets.

### ✨ Key Features

- 🤖 **Six professional AI agents** providing personalized analysis
- 🧠 **Gemini AI Chatbot** with offline fallback capability
- 🔮 **LSTM Neural Network** for enhanced price prediction
- 📊 **Real-time data** from VNStock API and CrewAI
- 🚀 **FastAPI backend** + **Streamlit frontend** with six professional tabs
- 📈 **Technical & fundamental analysis** with precise metrics
- ⚙️ **Personalized investment settings** (time horizon + risk tolerance)
- 🎨 **Polished UI** with Bootstrap integration

## 🤖 The 6 AI Agents

| Agent | Purpose | Description | Notable Features |
|-------|---------|-------------|------------------|
| 📈 **PricePredictor** | Price prediction | LSTM + technical analysis for forecasting | LSTM models, multi-timeframe predictions |
| 💼 **InvestmentExpert** | Investment adviser | Fundamental analysis and BUY/SELL/HOLD recommendations | Real financial ratios, AI-enhanced recommendations |
| ⚠️ **RiskExpert** | Risk management | Risk assessment using VaR, Beta, Sharpe ratio | Advanced risk metrics and AI guidance |
| 📰 **TickerNews** | Stock news | Crawls news from multiple Vietnamese sources | Multi-source crawling, sentiment analysis |
| 🌍 **MarketNews** | Market news | Risk-based news filtering and summaries | Underground news detection, risk-adjusted content |
| 🏢 **StockInfo** | Company details | Displays metrics and professional charts | Real-time data, interactive charts |

## 🏗️ System Architecture

```
agentvnstock/
├── agents/                           # 6 AI Agents + LSTM predictors
│   ├── price_predictor.py           # LSTM + Technical Analysis
│   ├── lstm_price_predictor.py      # Neural Network predictor
│   ├── investment_expert.py         # BUY/SELL recommendations
│   ├── risk_expert.py               # Risk assessment with VaR
│   ├── ticker_news.py               # Multi-source news crawling
│   ├── market_news.py               # Risk-based market news
│   ├── stock_info.py                # Professional data display
│   └── risk_based_news.py           # Underground news agent
├── src/
│   ├── data/                        # Data layer
│   │   ├── vn_stock_api.py          # VNStock + CrewAI integration
│   │   ├── crewai_collector.py      # Real news collection
│   │   └── company_search_api.py    # Company information
│   ├── ui/                          # UI components
│   │   ├── styles.py                # Bootstrap + custom CSS
│   │   └── components.py            # Reusable UI components
│   └── utils/                       # Utilities
│       ├── error_handler.py         # Error handling utilities
│       ├── market_schedule.py       # Market timing logic
│       ├── performance_monitor.py   # System monitoring
│       └── security_manager.py      # Security utilities
├── static/                          # Web assets
│   ├── index.html                   # Front-end HTML
│   ├── script.js                    # Front-end scripts
│   └── styles.css                   # Styling
├── gemini_agent.py                  # Unified AI with offline fallback
├── main_agent.py                    # Main orchestrator
├── api.py                           # FastAPI backend (20+ endpoints)
└── app.py                           # Streamlit frontend (6 tabs)
```

## 🚀 Quick Start

### 1. Clone the repository
```powershell
git clone https://github.com/nminduo2k5/agentvnstock.git
cd agentvnstock
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the application

#### Streamlit Frontend (recommended)
```powershell
streamlit run app.py
```

### 4. Configure API keys (in the app)
- Open the Streamlit sidebar
- Enter your **Gemini API key** (obtainable at https://aistudio.google.com/apikey)
- Optionally enter a **Serper API key** (https://serper.dev/api-key)
- Click **"🔧 Configure Gemini"** or **"🚀 Configure CrewAI"**

## 📱 Six Professional Tabs

### **Tab 1: 📊 Stock Analysis**
- **Comprehensive analysis**: All six agents + LSTM
- **Price prediction**: LSTM models + technical analysis
- **Investment analysis**: BUY/SELL/HOLD recommendations using real financial ratios
- **Risk assessment**: VaR, Beta, Sharpe ratio, Max Drawdown

### **Tab 2: 💬 AI Chatbot**
- **Gemini AI**: Natural-language analysis and explanations
- **Offline fallback**: Continues to provide useful answers when API quota is exhausted
- **Suggested prompts**: Five common sample questions
- **Smart responses**: Auto-formatted replies with icons and color cues

### **Tab 3: 📈 Vietnam Market**
- **VN-Index real-time**: Data from VNStock API
- **Top movers**: Styled list of biggest gainers/losers
- **37+ Vietnamese stocks**: CrewAI provides live search or static fallback
- **Market overview**: News and sentiment analysis

### **Tab 4: 📰 Stock News**
- **Multi-source crawling**: CafeF, VietStock, VCI
- **AI sentiment analysis**: Market sentiment scoring
- **Priority highlighting**: Important news flagged
- **Real-time updates**: CrewAI integration

### **Tab 5: 🏢 Company Information**
- **Company overview**: Details from CrewAI
- **Financial metrics**: P/E, P/B, EPS, dividend yield
- **Interactive charts**: Price history via Plotly
- **Professional presentation**: Clean layout and styling

### **Tab 6: 🌍 Market News**
- **Risk-based filtering**: News tailored to risk profiles
- **Underground news**: Sources like F319, F247, Facebook groups
- **Official news**: CafeF, VnEconomy, DanTri
- **Smart categorization**: Auto-classifies news by risk profile

## 🧠 LSTM Neural Network

### **Advanced LSTM features:**
- **18 model variants**: From basic LSTM to Transformer-based approaches
- **Multi-timeframe prediction**: Horizons from 1 day up to 1 year
- **Confidence scoring**: Prediction reliability estimates
- **AI enhancement**: Combines model outputs with Gemini analysis
- **Real-time training**: Optional continuous model updates

## ⚙️ Personalized Investment Settings

### **🕐 Investment horizons:**
- **Short-term**: 1–3 months (focus on technical analysis)
- **Medium-term**: 3–12 months (balance technical and fundamental)
- **Long-term**: 1+ years (fundamental-focused)

### **⚠️ Risk tolerance (0–100):**
- **0–30**: 🟢 Conservative (blue-chip, dividend stocks)
- **31–70**: 🟡 Moderate (mixed portfolio)
- **71–100**: 🔴 Aggressive (growth stocks, speculative news)

### **💰 Investment amount:**
- **From ~1 million to 10 billion VND**
- **Position sizing**: Automatic weight calculation
- **Risk management**: Smart stop-loss and take-profit rules

## 🛡️ Offline Fallback Behavior

### **When Gemini API quota is exhausted:**
- ✅ The system does **not crash**
- ✅ It still provides useful, fallback answers
- ✅ Clear status notifications are shown to the user
- ✅ Guidance on next steps is provided

### **Sample offline response:**
```
📈 OFFLINE ANALYSIS:
Gemini API quota has been exhausted; the system is operating in offline mode...

💡 Basic investment principles:
- P/E < 15 is commonly considered attractive
- Diversify your portfolio to reduce risk
- Only invest money you can afford to lose

⏰ Quota typically resets after 24 hours
```

## 📊 Supported Stocks

### 🏦 Banks (7 symbols)
**VCB** • **BID** • **CTG** • **TCB** • **ACB** • **MBB** • **VPB**

### 🏢 Real Estate (5 symbols)
**VIC** • **VHM** • **VRE** • **DXG** • **NVL**

### 🛒 Consumer (5 symbols)
**MSN** • **MWG** • **VNM** • **SAB** • **PNJ**

### 🏭 Industrial (3 symbols)
**HPG** • **HSG** • **NKG**

### ⚡ Utilities (3 symbols)
**GAS** • **PLX** • **POW**

### 💻 Technology (2 symbols)
**FPT** • **CMG**

### 🚁 Transport (2 symbols)
**VJC** • **HVN**

### 💊 Healthcare (2 symbols)
**DHG** • **IMP**

**Total: 37+ Vietnamese stocks supported**

## 💻 Using the API

### FastAPI endpoints (20+ endpoints)

#### Stock analysis
```python
# POST /analyze
{
  "symbol": "VCB",
  "time_horizon": "medium",
  "risk_tolerance": 50,
  "investment_amount": 100000000
}
```

#### AI Chatbot
```python
# POST /query
{
  "query": "Analyze VCB — should I buy?",
  "symbol": "VCB"
}
```

#### Price prediction
```python
# GET /predict/VCB
# Response: LSTM + technical analysis outputs
```

#### Risk assessment
```python
# GET /risk/VCB
# Response: VaR, Beta, Sharpe ratio
```

### Python SDK
```python
from main_agent import MainAgent
from src.data.vn_stock_api import VNStockAPI

# Initialize
vn_api = VNStockAPI()
main_agent = MainAgent(vn_api, gemini_api_key="your_key")

# Comprehensive analysis
result = await main_agent.analyze_stock('VCB')

# AI Chatbot
response = await main_agent.process_query("Analyze VCB", "VCB")
```

## 📋 Core Requirements

```
# Core Framework
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn>=0.24.0

# CrewAI Integration
crewai[tools]>=0.117.0
crewai-tools>=0.12.0

# AI & ML
google-generativeai>=0.3.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Data Sources
vnstock>=3.2.0
yfinance>=0.2.0
requests>=2.31.0
aiohttp>=3.8.0

# Visualization
plotly>=5.17.0
matplotlib>=3.7.0
beautifulsoup4>=4.12.0
```

## 🔧 Advanced Configuration

### Dynamic API Key (no .env required)
```python
# In the Streamlit sidebar
gemini_key = st.text_input("Gemini API Key", type="password")
if st.button("🔧 Configure"):
    main_agent.set_gemini_api_key(gemini_key)
```

### FastAPI Health Check
```powershell
curl http://localhost:8000/health
# Response: system status + agents status
```

### CrewAI Live Data
```python
# Automatically fetch symbols from CrewAI
symbols = await vn_api.get_available_symbols()
# Fallback to static data if CrewAI fails
```

## 🎨 UI Enhancements

### **Bootstrap Integration:**
- **Professional styling**: Card-based layout
- **Responsive design**: Mobile-friendly
- **Color-coded metrics**: Green/Red/Yellow indicators
- **Interactive charts**: Plotly integration
- **Gradient backgrounds**: Modern UI/UX

### **Enhanced features:**
- **Real-time updates**: Auto-refresh data
- **Error handling**: Graceful fallbacks
- **Loading states**: Professional spinners
- **Tooltips**: Helpful explanations
- **Keyboard shortcuts**: Power-user features

## 🔍 Troubleshooting

### Common issues

**1. Gemini API Error:**
```powershell
# Check your API key at: https://aistudio.google.com/apikey
# Ensure the key has access to Gemini 2.0 Flash
```

**2. VNStock Error:**
```powershell
pip install vnstock --upgrade
# Or use fallback static data
```

**3. CrewAI Error:**
```powershell
pip install crewai[tools] --upgrade
# Verify Serper API key (optional)
```

**4. LSTM Errors:**
```powershell
pip install tensorflow scikit-learn --upgrade
# LSTM components will fallback to traditional methods if unavailable
```

## 🚀 Roadmap

### **Version 2.0 (Current)**
- ✅ Six AI agents completed
- ✅ LSTM neural network
- ✅ Gemini AI with offline fallback
- ✅ CrewAI live data integration
- ✅ Support for 37+ Vietnamese stocks

### **Version 2.2 (Planned)**
- 🔄 Transformer models (GPT-style)
- 🔄 Real-time alerts system
- 🔄 Portfolio management
- 🔄 Backtesting engine
- 🔄 Mobile app

### **Version 3.0 (Future)**
- 🔮 Multi-market support (US, EU, Asia)
- 🔮 Options & derivatives analysis
- 🔮 Social sentiment integration
- 🔮 Automated trading signals

## 🤝 Contributing
nminduo2k5 🤖

## 📞 Support

- 🐛 **Issues**: https://github.com/nminduo2k5/agentvnstock/issues
- 💬 **Discussions**: https://github.com/nminduo2k5/agentvnstock/discussions
- 📧 **Email**: duongnguyenminh808@gmail.com or 23010441@st.phenikaa-uni.edu.vn

## 🙏 Acknowledgments

- [Google Gemini](https://ai.google.dev) - AI chatbot
- [CrewAI](https://crewai.com) - Multi-agent framework using Serper.dev
- [Serper.dev](https://serper.dev) - Search engine API
- [Scikit-Learn](https://scikit-learn.org) - Machine learning library
- [vnstock](https://github.com/thinh-vu/vnstock) - Vietnamese stock data
- [Streamlit](https://streamlit.io) - Streamlit web framework
- [FastAPI](https://fastapi.tiangolo.com) - Modern API framework
- [TensorFlow](https://tensorflow.org) - LSTM neural networks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for Vietnamese investors**

[![Star this repo](https://img.shields.io/github/stars/nminduo2k5/agentvnstock?style=social)](https://github.com/nminduo2k5/agentvnstock)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**🚀 Version 2.0 - Professional AI Trading System**

"Invest intelligently with the power of AI and Machine Learning!" 💪
</div>