# Multi-Agent Stock Analysis Microservices System

## 🏗️ Architecture Overview

This system implements a microservices architecture for the 6 AI agents stock analysis platform, providing scalable, maintainable, and distributed services.

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

## 🎯 Services

### Core Services

1. **Frontend Service (Port: 8501)**
   - Streamlit web interface
   - Multi-tab dashboard
   - Real-time service communication

2. **Price Predictor Service (Port: 8001)**
   - LSTM neural networks
   - Technical analysis
   - Multi-timeframe predictions

3. **Investment Expert Service (Port: 8002)**
   - BUY/SELL/HOLD recommendations
   - Risk-adjusted analysis
   - Portfolio optimization

4. **LLM Hub Service (Port: 8010)**
   - Unified AI access (Gemini, OpenAI)
   - Response caching
   - Offline fallback

### Infrastructure Services

5. **API Gateway (Nginx - Port: 80)**
   - Load balancing
   - Request routing
   - SSL termination

6. **Redis Cache (Port: 6379)**
   - Response caching
   - Session storage
   - Real-time data

7. **PostgreSQL Database (Port: 5432)**
   - Persistent storage
   - Analysis history
   - User data

## 🚀 Quick Start

### Prerequisites

- Docker Desktop installed and running
- Docker Compose available
- 8GB+ RAM recommended
- Internet connection for AI services

### 1. Clone and Setup

```bash
git clone <repository-url>
cd microservices
```

### 2. Configure Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
POSTGRES_PASSWORD=vnstock123
REDIS_PASSWORD=vnstock123
```

### 3. Start the System

**Windows:**
```cmd
start-system.bat
```

**Linux/Mac:**
```bash
chmod +x start-system.sh
./start-system.sh
```

**Manual Docker Compose:**
```bash
docker-compose up --build -d
```

### 4. Access the System

- **Frontend**: http://localhost:8501
- **API Gateway**: http://localhost:80
- **Price Predictor**: http://localhost:8001/docs
- **Investment Expert**: http://localhost:8002/docs
- **LLM Hub**: http://localhost:8010/docs

## 📊 API Endpoints

### Price Predictor Service

```http
POST /predict
{
  "symbol": "VCB",
  "days": 30,
  "risk_tolerance": 50,
  "time_horizon": "medium",
  "investment_amount": 10000000
}

GET /predict/{symbol}?days=30
GET /lstm/{symbol}?days=30
GET /technical/{symbol}
```

### Investment Expert Service

```http
POST /analyze
{
  "symbol": "VCB",
  "risk_tolerance": 50,
  "time_horizon": "medium",
  "investment_amount": 10000000
}

GET /analyze/{symbol}
GET /recommendation/{symbol}
POST /portfolio-analysis
```

### LLM Hub Service

```http
POST /generate
{
  "prompt": "Analyze VCB stock",
  "model": "gemini",
  "temperature": 0.7,
  "cache_key": "vcb_analysis"
}

GET /models
GET /health
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `OPENAI_API_KEY` | OpenAI API key | Optional |
| `POSTGRES_PASSWORD` | Database password | `vnstock123` |
| `REDIS_PASSWORD` | Redis password | Optional |

### Service Configuration

Each service can be configured via environment variables:

```yaml
# docker-compose.yml
services:
  price-predictor:
    environment:
      - REDIS_URL=redis://redis:6379
      - DB_URL=postgresql://vnstock:vnstock123@postgres:5432/vnstock_db
      - LOG_LEVEL=INFO
```

## 🐳 Docker Commands

### Basic Operations

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f [service_name]

# Restart a service
docker-compose restart [service_name]

# Scale a service
docker-compose up -d --scale price-predictor=3
```

### Development

```bash
# Build without cache
docker-compose build --no-cache

# Start with rebuild
docker-compose up --build

# Remove all containers and volumes
docker-compose down -v --remove-orphans
```

### Monitoring

```bash
# View service status
docker-compose ps

# View resource usage
docker stats

# View service logs
docker-compose logs -f --tail=100 price-predictor
```

## 🔍 Health Checks

Each service provides health check endpoints:

```bash
# Check all services
curl http://localhost:80/health

# Check individual services
curl http://localhost:8001/health  # Price Predictor
curl http://localhost:8002/health  # Investment Expert
curl http://localhost:8010/health  # LLM Hub
```

## 📈 Monitoring and Logging

### Service Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f price-predictor

# Last 100 lines
docker-compose logs --tail=100 investment-expert
```

### Performance Monitoring

```bash
# Resource usage
docker stats

# Service status
docker-compose ps

# Network inspection
docker network ls
docker network inspect microservices_default
```

## 🛠️ Development

### Adding New Services

1. Create service directory:
```bash
mkdir services/new-service
cd services/new-service
```

2. Create service files:
```
new-service/
├── main.py
├── Dockerfile
├── requirements.txt
└── README.md
```

3. Add to docker-compose.yml:
```yaml
new-service:
  build: ./services/new-service
  ports:
    - "8007:8007"
  depends_on:
    - redis
    - postgres
```

4. Update nginx configuration:
```nginx
location /api/new-service/ {
    proxy_pass http://new-service:8007/;
}
```

### Local Development

```bash
# Run single service locally
cd services/price-predictor
pip install -r requirements.txt
python main.py

# Connect to local Redis/Postgres
export REDIS_URL=redis://localhost:6379
export DB_URL=postgresql://localhost:5432/vnstock_db
```

## 🔒 Security

### API Security

- All services communicate through internal Docker network
- External access only through Nginx gateway
- API rate limiting configured
- Input validation on all endpoints

### Data Security

- Database credentials in environment variables
- Redis password protection
- SSL/TLS for external connections
- No sensitive data in logs

## 🚨 Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check Docker daemon
docker info

# Check port conflicts
netstat -tulpn | grep :8501

# Check logs
docker-compose logs [service_name]
```

**Memory issues:**
```bash
# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory

# Check memory usage
docker stats
```

**Network issues:**
```bash
# Recreate network
docker-compose down
docker network prune
docker-compose up
```

### Service-Specific Issues

**Price Predictor fails:**
- Check if TensorFlow/LSTM dependencies are installed
- Verify stock data API access
- Check memory allocation for ML models

**Investment Expert errors:**
- Verify connection to Price Predictor service
- Check financial data availability
- Review calculation logic

**LLM Hub timeout:**
- Check API keys configuration
- Verify internet connection
- Review rate limits

## 📚 API Documentation

Each service provides interactive API documentation:

- Price Predictor: http://localhost:8001/docs
- Investment Expert: http://localhost:8002/docs
- LLM Hub: http://localhost:8010/docs

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-service`
3. Make changes and test locally
4. Update documentation
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FastAPI for microservices framework
- Docker for containerization
- Streamlit for frontend
- Redis for caching
- PostgreSQL for data storage
- Nginx for load balancing

---

**🎉 Happy Trading with AI-Powered Analysis!**