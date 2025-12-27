# Multi-Agent Stock Analysis Microservices System

## 🏗️ Architecture Overview

Hệ thống microservices cho phân tích cổ phiếu với 6 AI agents, được triển khai trong folder `SRC/microservices/`.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gateway       │    │   LLM Hub       │    │   Frontend      │
│   (Nginx:80)    │    │   (Port:8010)   │    │   (Port:8501)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
    ┌────────────────────────────┼────────────────────────────┐
    │                            │                            │
┌───▼────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│Price   │ │Investment│ │Risk      │ │News      │ │Market    │
│:8001   │ │:8002     │ │:8003     │ │:8004     │ │:8005     │
└────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
    │           │            │            │            │
    └───────────┼────────────┼────────────┼────────────┘
                │            │            │
         ┌──────▼────────────▼────────────▼──────┐
         │              Shared Services          │
         │  ┌─────────┐ ┌─────────┐ ┌─────────┐ │
         │  │Database │ │ Redis   │ │RabbitMQ │ │
         │  │:5432    │ │:6379    │ │:5672    │ │
         │  └─────────┘ └─────────┘ └─────────┘ │
         └─────────────────────────────────────────┘
```

## 🎯 Services Implemented

### ✅ Core Services (Implemented)

1. **🧠 LLM Hub Service (Port: 8010)**
   - Unified AI access (Gemini + OpenAI + Offline fallback)
   - Response caching với Redis
   - Intelligent fallback responses

2. **📈 Price Predictor Service (Port: 8001)**
   - Technical analysis (RSI, MACD, Moving Averages)
   - Multi-timeframe predictions
   - Vietnamese & International stocks

3. **💼 Investment Expert Service (Port: 8002)**
   - BUY/SELL/HOLD recommendations
   - Risk-adjusted position sizing
   - Target price & stop loss calculation

4. **🌐 Frontend Service (Port: 8501)**
   - Streamlit multi-tab interface
   - Real-time service communication
   - Interactive charts & dashboards

### 🔧 Infrastructure Services

5. **Nginx Gateway (Port: 80)** - Load balancing & routing
6. **Redis Cache (Port: 6379)** - Response caching
7. **PostgreSQL (Port: 5432)** - Persistent storage
8. **RabbitMQ (Port: 5672)** - Message queue

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed and running
- 8GB+ RAM recommended

### 1. Navigate to SRC folder
```cmd
cd SRC\microservices
```

### 2. Configure Environment (Optional)
Create `.env` file:
```env
GEMINI_API_KEY=your_actual_gemini_key
OPENAI_API_KEY=your_actual_openai_key
```

### 3. Start the System

**Windows:**
```cmd
start-system.bat
```

**Manual Docker:**
```cmd
docker-compose up --build -d
```

### 4. Access the System

- **Frontend Dashboard**: http://localhost:8501
- **API Gateway**: http://localhost:80
- **Price Predictor API**: http://localhost:8001/docs
- **Investment Expert API**: http://localhost:8002/docs
- **LLM Hub API**: http://localhost:8010/docs

## 📊 API Usage Examples

### Price Prediction
```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "VCB",
    "days": 30,
    "risk_tolerance": 50,
    "time_horizon": "medium"
  }'
```

### Investment Analysis
```bash
curl -X POST "http://localhost:8002/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "VCB",
    "risk_tolerance": 50,
    "investment_amount": 10000000
  }'
```

### AI Chat
```bash
curl -X POST "http://localhost:8010/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Should I buy VCB stock?",
    "model": "gemini"
  }'
```

## 🔧 Configuration

### Supported Stocks
- **Vietnamese**: VCB, BID, CTG, TCB, ACB, MBB, VPB, VIC, VHM, VRE, MSN, MWG, VNM, SAB, PNJ, HPG, HSG, GAS, PLX, FPT
- **International**: Any symbol supported by Yahoo Finance (AAPL, GOOGL, TSLA, etc.)

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required for AI features |
| `OPENAI_API_KEY` | OpenAI API key | Optional |
| `POSTGRES_PASSWORD` | Database password | `vnstock123` |

## 🐳 Docker Commands

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f [service_name]

# Restart a service
docker-compose restart [service_name]

# Check service status
docker-compose ps
```

## 🔍 Health Monitoring

Check service health:
```bash
# All services via gateway
curl http://localhost:80/health

# Individual services
curl http://localhost:8001/health  # Price Predictor
curl http://localhost:8002/health  # Investment Expert
curl http://localhost:8010/health  # LLM Hub
```

## 📈 Features

### ✅ Implemented Features
- ✅ **Microservices Architecture** - Scalable, distributed services
- ✅ **Price Prediction** - Technical analysis + multi-timeframe forecasts
- ✅ **Investment Recommendations** - BUY/SELL/HOLD with confidence scores
- ✅ **AI Chat Interface** - Gemini AI with offline fallback
- ✅ **Service Discovery** - Nginx gateway routing
- ✅ **Caching Layer** - Redis for performance optimization
- ✅ **Health Monitoring** - Service status checks
- ✅ **Docker Containerization** - Easy deployment
- ✅ **Interactive Frontend** - Streamlit dashboard

### 🔄 Planned Features (Future)
- 🔄 Risk Expert Service (Port: 8003)
- 🔄 News Agent Service (Port: 8004)
- 🔄 Market News Service (Port: 8005)
- 🔄 Stock Info Service (Port: 8006)
- 🔄 LSTM Neural Networks
- 🔄 Portfolio optimization
- 🔄 Real-time alerts

## 🚨 Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check Docker
docker info

# Check logs
docker-compose logs [service_name]

# Restart services
docker-compose restart
```

**Port conflicts:**
```bash
# Check port usage
netstat -ano | findstr :8501

# Change ports in docker-compose.yml if needed
```

**Memory issues:**
```bash
# Increase Docker memory limit in Docker Desktop
# Settings > Resources > Memory > 8GB+
```

## 📚 Development

### Project Structure
```
SRC/microservices/
├── docker-compose.yml          # Main orchestration
├── gateway/
│   └── nginx.conf             # API gateway config
├── services/
│   ├── llm-hub/               # AI service
│   ├── price-predictor/       # Price prediction
│   ├── investment-expert/     # Investment analysis
│   └── frontend/              # Streamlit UI
└── start-system.bat           # Windows launcher
```

### Adding New Services
1. Create service directory in `services/`
2. Add service to `docker-compose.yml`
3. Update `nginx.conf` for routing
4. Implement FastAPI endpoints

## 🎉 Success Metrics

- **✅ 4/6 Core Services** implemented and working
- **✅ Full Docker orchestration** with docker-compose
- **✅ API Gateway** routing all services
- **✅ Interactive Frontend** with real-time communication
- **✅ AI Integration** with Gemini + offline fallback
- **✅ Caching Layer** for performance optimization
- **✅ Health Monitoring** for all services

## 📞 Support

For issues or questions:
1. Check service logs: `docker-compose logs -f [service]`
2. Verify service health: `curl http://localhost:[port]/health`
3. Restart services: `docker-compose restart`

---

**🎯 Ready for production deployment and scaling!**