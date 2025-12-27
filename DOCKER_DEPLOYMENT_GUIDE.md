# 🐳 Multi-Agent Stock Analysis - Complete Docker Deployment Guide

## 📍 Hệ thống có 2 Architecture

### **1. Original System (SRC/)** - Monolithic
- **Port**: 8501
- **Architecture**: Single container với tất cả agents
- **Best for**: Quick start, testing, demo

### **2. Microservices System (SRC/microservices/)** - Distributed  
- **Ports**: 8502, 8080, 8001, 8002, 8010
- **Architecture**: 6 services riêng biệt + monitoring
- **Best for**: Production, scaling, enterprise

---

## 🚀 ORIGINAL SYSTEM (SRC/)

### **Quick Commands:**

#### **🎯 Option 1: Quick Start (Khuyến nghị)**
```bash
cd SRC
docker-start.bat          # Windows
make up                   # Linux/Mac
```
- ✅ **Dễ nhất** - 1 lệnh duy nhất
- ✅ **Production-ready**
- ✅ **Truy cập**: http://localhost:8501

#### **🔧 Option 2: Development**
```bash
cd SRC
docker-start.bat dev      # Windows
make dev                  # Linux/Mac
```
- ✅ **Hot reload** - Code tự động cập nhật
- ✅ **Redis UI**: http://localhost:8081
- ✅ **Truy cập**: http://localhost:8501

#### **🏭 Option 3: Production**
```bash
cd SRC
docker-start.bat start    # Windows
make prod                 # Linux/Mac
```
- ✅ **Optimized** performance
- ✅ **Security hardened**
- ✅ **Truy cập**: http://localhost:8501

### **Management Commands:**
```bash
# Status & Monitoring
docker-start.bat status   # System status
docker-start.bat logs     # View logs
make health              # Health check

# Maintenance
docker-start.bat clean    # Cleanup
docker-start.bat update   # Update code
make backup              # Backup data

# Stop
docker-start.bat stop     # Stop services
make down                # Stop services
```

---

## 🏗️ MICROSERVICES SYSTEM (SRC/microservices/)

### **Quick Commands:**

#### **🎯 Option 1: Development (Khuyến nghị)**
```bash
cd SRC/microservices
start-system.bat         # Windows
make dev                 # Linux/Mac
```
- ✅ **Hot reload** cho tất cả services
- ✅ **Development tools**
- ✅ **Truy cập**: http://localhost:8502

#### **🏭 Option 2: Production**
```bash
cd SRC/microservices
make prod                # All platforms
```
- ✅ **Full monitoring stack**
- ✅ **Auto-scaling**
- ✅ **Truy cập**: http://localhost:8502

#### **⚡ Option 3: Basic**
```bash
cd SRC/microservices
make up                  # All platforms
```
- ✅ **Simple setup**
- ✅ **Core services only**
- ✅ **Truy cập**: http://localhost:8502

### **Service URLs:**
```
Frontend:          http://localhost:8502
API Gateway:       http://localhost:8080
Price Predictor:   http://localhost:8001/docs
Investment Expert: http://localhost:8002/docs
LLM Hub:          http://localhost:8010/docs

# Development only
Database Admin:    http://localhost:8081
Redis Admin:       http://localhost:8082

# Production only  
Monitoring:        http://localhost:3000 (admin/admin123)
Metrics:          http://localhost:9090
Logs:             http://localhost:5601
```

### **Management Commands:**
```bash
# Environment Management
make help                # Show all commands
make setup              # Initial setup
make dev                # Development environment
make prod               # Production environment
make up                 # Basic environment

# Monitoring & Health
make health             # Check service health
make logs               # Show all logs
make logs-frontend      # Frontend logs only
make logs-price         # Price predictor logs
make monitor            # Open monitoring dashboard
make metrics            # Show system metrics

# Database Operations
make db-backup          # Backup database
make db-restore         # Restore from backup
make db-shell           # Connect to database

# Cache Operations
make cache-clear        # Clear Redis cache
make cache-info         # Show cache info

# Testing
make test               # Run all tests
make test-unit          # Unit tests only
make test-integration   # Integration tests

# Cleanup
make down               # Stop all services
make clean              # Clean Docker resources
make clean-all          # Clean everything including images
```

---

## 🎯 WHICH ONE TO CHOOSE?

### **🚀 For Quick Start & Demo:**
```bash
cd SRC
docker-start.bat          # Simplest option
# Access: http://localhost:8501
```

### **🔧 For Development:**
```bash
cd SRC/microservices
make dev                  # Best development experience
# Access: http://localhost:8502
```

### **🏭 For Production:**
```bash
cd SRC/microservices
make prod                 # Enterprise-grade
# Access: http://localhost:8502
```

### **📊 Comparison:**

| Feature | Original (SRC/) | Microservices (SRC/microservices/) |
|---------|----------------|-----------------------------------|
| **Setup** | ⭐⭐⭐ Simple | ⭐⭐ Moderate |
| **Performance** | ⭐⭐ Good | ⭐⭐⭐ Excellent |
| **Scalability** | ⭐ Limited | ⭐⭐⭐ High |
| **Monitoring** | ⭐ Basic | ⭐⭐⭐ Advanced |
| **Development** | ⭐⭐ Good | ⭐⭐⭐ Excellent |
| **Production** | ⭐⭐ Good | ⭐⭐⭐ Enterprise |

---

## 🔧 ENVIRONMENT SETUP

### **Prerequisites:**
```bash
# Check Docker installation
docker --version
docker-compose --version

# Minimum requirements
# RAM: 8GB+ (4GB for Original, 6GB for Microservices)
# CPU: 4+ cores
# Disk: 20GB+ free space
```

### **API Keys Setup:**
```bash
# For both systems, create .env file:
cd SRC                    # or SRC/microservices
cp .env.template .env

# Edit .env with your keys:
GEMINI_API_KEY=your_actual_gemini_key
OPENAI_API_KEY=your_actual_openai_key
```

---

## 🚨 TROUBLESHOOTING

### **Common Issues:**

#### **Port Conflicts:**
```bash
# Check port usage
netstat -ano | findstr :8501    # Windows
lsof -i :8501                   # Linux/Mac

# Kill process if needed
taskkill /PID [PID] /F          # Windows
kill -9 [PID]                  # Linux/Mac
```

#### **Docker Issues:**
```bash
# Restart Docker
# Windows: Restart Docker Desktop
sudo systemctl restart docker   # Linux

# Clean up if needed
docker system prune -af --volumes
```

#### **Memory Issues:**
```bash
# Check memory usage
docker stats

# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory > 8GB+
```

### **Health Checks:**
```bash
# Original System
curl http://localhost:8501/_stcore/health

# Microservices System  
curl http://localhost:8080/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8010/health
```

---

## 📊 MONITORING & LOGS

### **Original System:**
```bash
# View logs
docker-start.bat logs     # Windows
make logs                # Linux/Mac

# System status
docker-start.bat status   # Windows
make status              # Linux/Mac
```

### **Microservices System:**
```bash
# View logs
make logs                # All services
make logs-frontend       # Frontend only
make logs-price          # Price predictor only

# Monitoring dashboards
make monitor             # Open Grafana
make metrics             # Prometheus metrics

# System status
make health              # Health checks
make status              # Detailed status
```

---

## 🧹 CLEANUP & MAINTENANCE

### **Regular Cleanup:**
```bash
# Original System
docker-start.bat clean    # Windows
make clean               # Linux/Mac

# Microservices System
make clean               # Clean resources
make clean-all           # Deep clean
```

### **Updates:**
```bash
# Original System
docker-start.bat update   # Windows
make update              # Linux/Mac

# Microservices System
make down && make up     # Restart with latest code
```

### **Backup:**
```bash
# Original System
make backup              # Backup database

# Microservices System
make db-backup           # Backup database
```

---

## 🎉 QUICK REFERENCE

### **Most Common Commands:**

#### **Original System (Simple):**
```bash
cd SRC
docker-start.bat         # Start
docker-start.bat stop    # Stop
docker-start.bat logs    # Logs
```

#### **Microservices System (Advanced):**
```bash
cd SRC/microservices
make dev                 # Development
make prod                # Production
make health              # Check health
make logs                # View logs
make clean               # Cleanup
```

### **Emergency Commands:**
```bash
# Stop everything immediately
docker kill $(docker ps -q)

# Nuclear cleanup
docker system prune -af --volumes

# Restart Docker daemon
sudo systemctl restart docker  # Linux
# Restart Docker Desktop        # Windows/Mac
```

---

## 🌟 BEST PRACTICES

1. **Start Simple**: Use Original System (SRC/) for learning
2. **Scale Up**: Move to Microservices for production
3. **Monitor**: Always check health and logs
4. **Backup**: Regular database backups
5. **Clean**: Regular cleanup to free disk space
6. **Update**: Keep Docker and images updated
7. **Security**: Use proper API keys and environment variables

---

**🚀 Ready to deploy your Multi-Agent Stock Analysis System!**

Choose your path:
- **Quick & Simple**: `cd SRC && docker-start.bat`
- **Advanced & Scalable**: `cd SRC/microservices && make dev`