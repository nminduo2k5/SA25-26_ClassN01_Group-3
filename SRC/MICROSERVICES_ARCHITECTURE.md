# Microservices Architecture for 6 Agents

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gateway       │    │   Discovery     │    │   Config        │
│   (Nginx)       │    │   (Consul)      │    │   (Vault)       │
│   Port: 80      │    │   Port: 8500    │    │   Port: 8200    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
    ┌────────────────────────────┼────────────────────────────┐
    │                            │                            │
┌───▼────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│Frontend│ │ Agent 1  │ │ Agent 2  │ │ Agent 3  │ │ Agent 4  │ │ Agent 5  │
│Streamlit│ │Price     │ │Investment│ │Risk      │ │News      │ │Market    │
│:8501   │ │:8001     │ │:8002     │ │:8003     │ │:8004     │ │:8005     │
└────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
    │           │            │            │            │            │
    └───────────┼────────────┼────────────┼────────────┼────────────┘
                │            │            │            │
         ┌──────▼────────────▼────────────▼────────────▼──────┐
         │              Shared Services                       │
         │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
         │  │Database │ │ Redis   │ │ RabbitMQ│ │ LLM Hub │  │
         │  │:5432    │ │:6379    │ │:5672    │ │:8010    │  │
         │  └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
         └─────────────────────────────────────────────────────┘
```

## 🎯 6 Microservices

### 1. Price Predictor Service (Port: 8001)
- **Function**: LSTM + Technical Analysis
- **Endpoints**: `/predict/{symbol}`, `/lstm/{symbol}`, `/technical/{symbol}`
- **Dependencies**: TensorFlow, scikit-learn

### 2. Investment Expert Service (Port: 8002)  
- **Function**: BUY/SELL/HOLD recommendations
- **Endpoints**: `/analyze/{symbol}`, `/recommendation/{symbol}`
- **Dependencies**: Financial data APIs

### 3. Risk Expert Service (Port: 8003)
- **Function**: Risk assessment, VaR, Beta calculation
- **Endpoints**: `/risk/{symbol}`, `/var/{symbol}`, `/beta/{symbol}`
- **Dependencies**: Risk calculation libraries

### 4. News Agent Service (Port: 8004)
- **Function**: News crawling + sentiment analysis
- **Endpoints**: `/news/{symbol}`, `/sentiment/{symbol}`
- **Dependencies**: BeautifulSoup, NLP libraries

### 5. Market News Service (Port: 8005)
- **Function**: Market overview + international news
- **Endpoints**: `/market-news`, `/international-news`
- **Dependencies**: News APIs, web scraping

### 6. Stock Info Service (Port: 8006)
- **Function**: Company data + charts
- **Endpoints**: `/info/{symbol}`, `/charts/{symbol}`
- **Dependencies**: VNStock, Plotly

## 🔧 Shared Services

### LLM Hub Service (Port: 8010)
- **Function**: Unified LLM access (Gemini, OpenAI, Llama)
- **Endpoints**: `/generate`, `/models`, `/health`

### Message Queue (RabbitMQ: 5672)
- **Function**: Inter-service communication
- **Queues**: `price.requests`, `news.updates`, `risk.calculations`

### Cache (Redis: 6379)
- **Function**: Response caching, session storage
- **Keys**: `stock:{symbol}`, `news:{symbol}`, `predictions:{symbol}`

### Database (PostgreSQL: 5432)
- **Function**: Persistent data storage
- **Tables**: `analyses`, `predictions`, `news`, `user_sessions`