# 🇻🇳 Duong AI Trading Pro

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Google-Gemini-orange.svg)](https://ai.google.dev)
[![CrewAI](https://img.shields.io/badge/CrewAI-0.117+-purple.svg)](https://crewai.com)

> **Intelligent stock investment analysis system with 6 AI Agents + Gemini AI + CrewAI + LSTM Neural Network**

## 🎯 Overview

**Duong AI Trading Pro** is a comprehensive stock investment analysis system that integrates 6 professional AI Agents, Gemini AI, and LSTM neural networks to provide comprehensive analysis for Vietnamese and international stock markets.

### ✨ Key Features

- 🤖 **6 Professional AI Agents** with personalized analysis
- 🧠 **Gemini AI Chatbot** with offline fallback capability
- 🔮 **LSTM Neural Network** for advanced price prediction
- 📊 **Real-time data** from VNStock API and CrewAI
- 🚀 **FastAPI Backend** + **Streamlit Frontend** with 6 professional tabs
- 📈 **Technical & Fundamental Analysis** with accurate metrics
- ⚙️ **Personal Investment Settings** (time horizon + risk tolerance)
- 🎨 **Beautiful Interface** with Bootstrap integration

## 🔍 CrewAI - 2 Operating Modes

### **Mode 1: Direct Serper Search (Independent - Recommended)**
- **Only needs**: Serper API key
- **No need**: Gemini/OpenAI/Llama API
- **Features**: Direct news search from Serper
- **Advantages**: Fast, simple, no LLM dependency
- **Suitable for**: Users who only need real news

### **Mode 2: LLM-Enhanced Search (Advanced)**
- **Needs**: Serper API + (Gemini or OpenAI or Llama)
- **Features**: AI analysis and news summarization
- **Advantages**: Deeper analysis, sentiment analysis
- **Suitable for**: Users who want AI-assisted analysis

```python
# Chế độ 1: Chỉ Serper (không LLM)
collector = CrewAIDataCollector(serper_api_key="your_key")
# ➡️ CrewAI: Serper only (Direct search mode - no LLM)

# Chế độ 2: Serper + LLM
collector = CrewAIDataCollector(
    gemini_api_key="your_key",
    serper_api_key="your_key"
)
# ➡️ CrewAI: gemini + Serper (LLM mode)
```

## 🤖 Team of 6 AI Agents

| Agent | Function | Description | Special Features |
|-------|----------|-------------|------------------|
| 📈 **PricePredictor** | Price Prediction | LSTM + Technical Analysis for price forecasting | LSTM Neural Network, Multi-timeframe |
| 💼 **InvestmentExpert** | Investment Expert | Fundamental analysis and BUY/SELL/HOLD recommendations | Real financial ratios, AI-enhanced |
| ⚠️ **RiskExpert** | Risk Management | Risk assessment with VaR, Beta, Sharpe ratio | Advanced risk metrics, AI advice |
| 📰 **TickerNews** | Stock News | Crawl news from CafeF, VietStock | Multi-source crawling, Sentiment analysis |
| 🌍 **MarketNews** | Market News | Risk-based news filtering | Underground news, Risk-adjusted content |
| 🏢 **StockInfo** | Detailed Information | Display professional metrics and charts | Real-time data, Interactive charts |

## 🏗️ Kiến trúc hệ thống

```
agentvnstock/
├── agents/                           # 6 AI Agents + LSTM
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
│   │   ├── styles.py                # Bootstrap + Custom CSS
│   │   └── components.py            # Reusable UI components
│   └── utils/                       # Utilities
│       ├── error_handler.py         # Comprehensive error handling
│       ├── market_schedule.py       # Market timing logic
│       ├── performance_monitor.py   # System monitoring
│       └── security_manager.py      # Security utilities
├── deep-learning/                   # LSTM Research & Development
│   ├── 1.lstm.ipynb                # Basic LSTM implementation
│   ├── 16.attention-is-all-you-need.ipynb # Transformer models
│   └── [18 Jupyter notebooks]      # Various ML approaches
├── static/                          # Web interface
│   ├── index.html                   # Professional web UI
│   ├── script.js                    # Interactive features
│   └── styles.css                   # Web styling
├── gemini_agent.py                  # Unified AI with offline fallback
├── main_agent.py                    # Main orchestrator
├── api.py                           # FastAPI backend (20+ endpoints)
└── app.py                           # Streamlit frontend (6 tabs)
```

## 🚀 Quick Setup

### 1. Clone repository
```bash
git clone https://github.com/nminduo2k5/Multi-agentVNstock.git
cd Multi-agentVNstock
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application

#### Main Application (Recommended)
```bash
streamlit run app_predict.py
```

#### Alternative Streamlit Frontend
```bash
streamlit run app.py
```

#### FastAPI Backend (Optional)
```bash
python api.py
# Or
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. API Configuration (in application)
- Open sidebar in Streamlit
- **Gemini API key** (optional, free at [Google AI Studio](https://aistudio.google.com/apikey))
- **Serper API key** (recommended for real news, at [Serper.dev](https://serper.dev/api-key))
- Click **"🔧 Setup Gemini"** or **"🚀 Setup CrewAI"**

**Note**: CrewAI can operate independently with only Serper API key (no need for Gemini/OpenAI/Llama)

## 📱 6 Professional Tabs Interface

### **Tab 1: 📊 Stock Analysis**
- **🚀 Comprehensive Analysis**: All 6 agents + LSTM
- **📈 Price Prediction**: LSTM Neural Network + Technical Analysis
- **💼 Investment Analysis**: BUY/SELL/HOLD with real financial ratios
- **⚠️ Risk Assessment**: VaR, Beta, Sharpe ratio, Max Drawdown

### **Tab 2: 💬 AI Chatbot**
- **Gemini AI**: In-depth analysis with natural language
- **Offline Fallback**: Still works when API quota is exhausted
- **Question Suggestions**: 5 common sample questions
- **Smart Responses**: Auto-formatting with colors and icons

### **Tab 3: 📈 VN Market**
- **VN-Index Real-time**: Data from VNStock API
- **Top movers**: Strong gainers/losers with beautiful styling
- **37+ VN stocks**: CrewAI real-time search or static fallback
- **Market overview**: News and sentiment analysis

### **Tab 4: 📰 Stock News**
- **Multi-source crawling**: CafeF, VietStock, VCI
- **AI sentiment analysis**: Market sentiment analysis
- **Priority highlighting**: Important news highlighted
- **Real-time updates**: CrewAI integration

### **Tab 5: 🏢 Company Information**
- **Company overview**: Detailed information from CrewAI
- **Financial metrics**: P/E, P/B, EPS, Dividend yield
- **Interactive charts**: Price history with Plotly
- **Enhanced display**: Professional styling

### **Tab 6: 🌍 Market News**
- **Risk-based filtering**: News according to risk profile
- **Underground news**: Insider news from F319, F247, FB Groups
- **Official news**: CafeF, VnEconomy, DanTri
- **Smart categorization**: Auto-categorization by risk profile

## 🧠 LSTM Neural Network

### **Advanced LSTM Features:**
- **18 ML models**: From basic LSTM to Transformer
- **Multi-timeframe prediction**: 1 day to 1 year
- **Confidence scoring**: Prediction reliability assessment
- **AI enhancement**: Integration with Gemini AI
- **Real-time training**: Continuous model updates

### **Available Models:**
```
1. LSTM Basic                    11. Bidirectional LSTM Seq2Seq
2. Bidirectional LSTM           12. LSTM Seq2Seq VAE
3. LSTM 2-Path                  13. GRU Seq2Seq
4. GRU                          14. Bidirectional GRU Seq2Seq
5. Bidirectional GRU            15. GRU Seq2Seq VAE
6. GRU 2-Path                   16. Attention (Transformer)
7. Vanilla RNN                  17. CNN Seq2Seq
8. Bidirectional Vanilla        18. Dilated CNN Seq2Seq
9. Vanilla 2-Path
10. LSTM Seq2Seq
```

## ⚙️ Personal Investment Settings

### **🕐 Investment Time Horizon:**
- **Short-term**: 1-3 months (Focus: Technical analysis)
- **Medium-term**: 3-12 months (Balance: Technical + Fundamental)
- **Long-term**: 1+ years (Focus: Fundamental analysis)

### **⚠️ Risk Tolerance (0-100):**
- **0-30**: 🟢 Conservative (Blue-chip, dividend stocks)
- **31-70**: 🟡 Balanced (Mixed portfolio)
- **71-100**: 🔴 Aggressive (Growth stocks, underground news)

### **💰 Investment Amount:**
- **From 1 million to 10 billion VND**
- **Position sizing**: Automatic proportion calculation
- **Risk management**: Smart stop-loss and take-profit

## 🛡️ Offline Fallback Feature

### **When Gemini API quota is exhausted:**
- ✅ System **DOES NOT crash**
- ✅ Still answers questions with useful content
- ✅ Clear status notifications
- ✅ User guidance on how to handle

### **Smart offline responses:**
```
📈 OFFLINE ANALYSIS:
Due to Gemini API quota exhaustion, system switched to offline mode...

💡 Basic investment principles:
- P/E < 15 is usually considered attractive
- Diversify portfolio to reduce risk
- Only invest money you can afford to lose

⏰ Quota usually resets after 24 hours
```

## 📊 Supported Stocks

### 🏦 Banking (7 stocks)
**VCB** • **BID** • **CTG** • **TCB** • **ACB** • **MBB** • **VPB**

### 🏢 Real Estate (5 stocks)
**VIC** • **VHM** • **VRE** • **DXG** • **NVL**

### 🛒 Consumer (5 stocks)
**MSN** • **MWG** • **VNM** • **SAB** • **PNJ**

### 🏭 Industrial (3 stocks)
**HPG** • **HSG** • **NKG**

### ⚡ Utilities (3 stocks)
**GAS** • **PLX** • **POW**

### 💻 Technology (2 stocks)
**FPT** • **CMG**

### 🚁 Transportation (2 stocks)
**VJC** • **HVN**

### 💊 Healthcare (2 stocks)
**DHG** • **IMP**

**Total: 37+ VN stocks**

## 💻 API Usage

### FastAPI Endpoints (20+ endpoints)

#### Stock Analysis
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
  "query": "Should I buy VCB stock?",
  "symbol": "VCB"
}
```

#### Price Prediction
```python
# GET /predict/VCB
# Response: LSTM + Technical analysis
```

#### Risk Assessment
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
response = await main_agent.process_query("Phân tích VCB", "VCB")
```

## 📋 Main Requirements

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

### Dynamic API Key (No .env file needed)
```python
# In Streamlit sidebar
gemini_key = st.text_input("Gemini API Key", type="password")
if st.button("🔧 Setup"):
    main_agent.set_gemini_api_key(gemini_key)
```

### FastAPI Health Check
```bash
curl http://localhost:8000/health
# Response: System status + agents status
```

### CrewAI Real Data
```python
# Auto-fetch symbols from CrewAI
symbols = await vn_api.get_available_symbols()
# Fallback to static if CrewAI fails
```

## 🎨 New Interface

### **Bootstrap Integration:**
- **Professional styling**: Card-based layout
- **Responsive design**: Mobile-friendly
- **Color-coded metrics**: Green/Red/Yellow indicators
- **Interactive charts**: Plotly integration
- **Gradient backgrounds**: Modern UI/UX

### **Enhanced Features:**
- **Real-time updates**: Auto-refresh data
- **Error handling**: Graceful fallbacks
- **Loading states**: Professional spinners
- **Tooltips**: Helpful explanations
- **Keyboard shortcuts**: Power user features

## 🔍 Troubleshooting

### **Common Errors:**

**1. Gemini API Error:**
```bash
# Check API key at: https://aistudio.google.com/apikey
# Ensure API key has access to Gemini 2.0 Flash
```

**2. VNStock Error:**
```bash
pip install vnstock --upgrade
# Or use fallback data
```

**3. CrewAI Error:**
```bash
pip install crewai[tools] --upgrade
# Check Serper API key (optional)
```

**4. LSTM Error:**
```bash
pip install tensorflow scikit-learn --upgrade
# LSTM will fallback to traditional methods
```

## 🚀 Roadmap

### **Version 2.0 (Current)**
- ✅ 6 complete AI Agents
- ✅ LSTM Neural Network
- ✅ Gemini AI with offline fallback
- ✅ CrewAI real data integration
- ✅ 37+ VN stocks support

### **Version 2.2 (Planned)**
- 🔄 Transformer models (GPT-style)
- 🔄 Real-time alerts system
- 🔄 Portfolio management
- 🔄 Backtesting engine
- 🔄 Mobile app

### **Version 3.0 (Future)**
- 🔮 Multi-market support (US, EU, Asia)
- 🔮 Options & Derivatives analysis
- 🔮 Social sentiment integration
- 🔮 Automated trading signals

## 🤝 Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push branch: `git push origin feature/amazing-feature`
5. Create Pull Request

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/nminduo2k5/Multi-agentVNstock/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/nminduo2k5/Multi-agentVNstock/discussions)
- 📧 **Email**: duongnguyenminh808@gmail.com or 23010441@st.phenikaa-uni.edu.vn


## 🙏 Acknowledgments

- [Google Gemini](https://ai.google.dev) - AI chatbot with offline fallback
- [CrewAI](https://crewai.com) - Multi-agent framework
- [vnstock](https://github.com/thinh-vu/vnstock) - Vietnamese stock data
- [Streamlit](https://streamlit.io) - Beautiful web framework
- [FastAPI](https://fastapi.tiangolo.com) - Modern API framework
- [TensorFlow](https://tensorflow.org) - LSTM Neural Networks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**🇻🇳 Made with ❤️ for Vietnamese investors**

[![Star this repo](https://img.shields.io/github/stars/nminduo2k5/Multi-agentVNstock?style=social)](https://github.com/nminduo2k5/Multi-agentVNstock)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**🚀 Version 2.0 - Professional AI Trading System**

*"Smart investing with the power of AI and Machine Learning!"* 💪

### ⚠️ Disclaimer

**Important Warning**: This is an analysis support tool, **NOT absolute investment advice**.

- Data may not be 100% accurate
- Always do your own research (DYOR)
- Only invest money you can afford to lose
- Author is not responsible for financial losses

**"As long as you're breathing, there's still hope to recover!"** 🚀

</div>