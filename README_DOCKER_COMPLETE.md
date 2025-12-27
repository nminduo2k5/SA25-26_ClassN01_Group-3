# 🚀 Multi-Agent Stock Analysis - Complete Docker Guide

## 📋 Overview

Hệ thống phân tích cổ phiếu với 6 AI Agents được triển khai theo 2 kiến trúc:

### **🏠 Original System** - Monolithic (SRC/)
- **Single container** chứa tất cả agents
- **Port**: 8501
- **Best for**: Quick start, demo, testing
- **Resource**: ~4GB RAM

### **🏗️ Microservices System** - Distributed (SRC/microservices/)
- **6 separate services** + monitoring stack
- **Ports**: 8502, 8080, 8001, 8002, 8010
- **Best for**: Production, scaling, enterprise
- **Resource**: ~6GB RAM

---

## 🎯 Quick Start Guide

### **🚀 Fastest Way (Recommended for beginners):**
```bash
# Windows
docker-master.bat original start

# Linux/Mac
./docker-master.sh original start

# Access: http://localhost:8501
```

### **🔧 Development (Recommended for developers):**
```bash
# Windows
docker-master.bat micro dev

# Linux/Mac
./docker-master.sh micro dev

# Access: http://localhost:8502
```

### **🏭 Production (Recommended for deployment):**
```bash
# Windows
docker-master.bat micro prod

# Linux/Mac
./docker-master.sh micro prod

# Access: http://localhost:8502
```

---

## 📊 System Comparison

| Feature | Original System | Microservices System |
|---------|----------------|----------------------|
| **Setup Complexity** | ⭐⭐⭐ Simple | ⭐⭐ Moderate |
| **Performance** | ⭐⭐ Good | ⭐⭐⭐ Excellent |
| **Scalability** | ⭐ Limited | ⭐⭐⭐ High |
| **Monitoring** | ⭐ Basic | ⭐⭐⭐ Advanced |
| **Development** | ⭐⭐ Good | ⭐⭐⭐ Excellent |
| **Production Ready** | ⭐⭐ Good | ⭐⭐⭐ Enterprise |
| **Resource Usage** | 4GB RAM | 6GB RAM |
| **Startup Time** | ~30 seconds | ~60 seconds |

---

## 🛠️ Master Commands

### **System Management:**
```bash
# Windows
docker-master.bat <system> <command>

# Linux/Mac  
./docker-master.sh <system> <command>
```

### **Available Systems:**
- `original` - Original monolithic system
- `micro` - Microservices system
- `microservices` - Alias for micro

### **Available Commands:**
- `start` - Start system (default)
- `dev` - Development mode with hot reload
- `prod` - Production mode with monitoring
- `basic` - Basic mode (microservices only)
- `stop` - Stop system
- `logs` - Show logs
- `status` - Show system status
- `health` - Check service health (microservices only)
- `clean` - Clean system resources

### **Global Commands:**
- `status` - Show status of all systems
- `stop-all` - Stop all systems
- `clean-all` - Clean all Docker resources

---

## 📖 Detailed Usage

### **🏠 Original System Commands:**

#### **Start Options:**
```bash
# Quick start (production-ready)
docker-master.bat original start

# Development with hot reload
docker-master.bat original dev

# Production optimized
docker-master.bat original prod
```

#### **Management:**
```bash
# Stop services
docker-master.bat original stop

# View logs
docker-master.bat original logs

# System status
docker-master.bat original status

# Cleanup resources
docker-master.bat original clean
```

### **🏗️ Microservices System Commands:**

#### **Start Options:**
```bash
# Development with hot reload
docker-master.bat micro dev

# Production with full monitoring
docker-master.bat micro prod

# Basic setup
docker-master.bat micro basic
```

#### **Management:**
```bash
# Stop all services
docker-master.bat micro stop

# View logs
docker-master.bat micro logs

# Service status
docker-master.bat micro status

# Health checks
docker-master.bat micro health

# Cleanup resources
docker-master.bat micro clean
```

### **🌐 Global Management:**
```bash
# Show status of all systems
docker-master.bat status

# Stop everything
docker-master.bat stop-all

# Clean all Docker resources
docker-master.bat clean-all
```

---

## 🌐 Service URLs

### **Original System:**
- **Main Application**: http://localhost:8501

### **Microservices System:**

#### **Core Services:**
- **Frontend**: http://localhost:8502
- **API Gateway**: http://localhost:8080
- **Price Predictor API**: http://localhost:8001/docs
- **Investment Expert API**: http://localhost:8002/docs
- **LLM Hub API**: http://localhost:8010/docs

#### **Development Tools:**
- **Database Admin**: http://localhost:8081
- **Redis Admin**: http://localhost:8082

#### **Production Monitoring:**
- **Grafana Dashboard**: http://localhost:3000 (admin/admin123)
- **Prometheus Metrics**: http://localhost:9090
- **Kibana Logs**: http://localhost:5601

---

## ⚙️ Environment Setup

### **Prerequisites:**
```bash
# Check Docker installation
docker --version
docker-compose --version

# Minimum system requirements:
# - RAM: 8GB+ (4GB for Original, 6GB for Microservices)
# - CPU: 4+ cores
# - Disk: 20GB+ free space
# - OS: Windows 10+, macOS 10.14+, Ubuntu 18.04+
```

### **API Keys Configuration:**
```bash
# 1. Navigate to system directory
cd SRC                    # For Original System
cd SRC/microservices      # For Microservices System

# 2. Copy environment template
cp .env.template .env

# 3. Edit .env file with your API keys:
GEMINI_API_KEY=your_actual_gemini_api_key_here
OPENAI_API_KEY=your_actual_openai_api_key_here

# 4. Get API keys from:
# Gemini: https://aistudio.google.com/apikey
# OpenAI: https://platform.openai.com/api-keys
```

---

## 🚨 Troubleshooting

### **Common Issues:**

#### **Port Conflicts:**
```bash
# Check what's using the ports
netstat -ano | findstr :8501    # Windows
lsof -i :8501                   # Linux/Mac

# Kill conflicting processes
taskkill /PID [PID] /F          # Windows
kill -9 [PID]                  # Linux/Mac
```

#### **Docker Issues:**
```bash
# Check Docker status
docker info

# Restart Docker
# Windows: Restart Docker Desktop
sudo systemctl restart docker   # Linux

# Clean Docker if needed
docker system prune -af --volumes
```

#### **Memory Issues:**
```bash
# Check current usage
docker stats

# Increase Docker memory limit:
# Docker Desktop > Settings > Resources > Memory > 8GB+
```

### **Health Checks:**
```bash
# Original System
curl http://localhost:8501/_stcore/health

# Microservices System
docker-master.bat micro health
```

---

## 📊 Monitoring & Logs

### **View Logs:**
```bash
# Original System
docker-master.bat original logs

# Microservices System
docker-master.bat micro logs

# Specific service logs (microservices)
cd SRC/microservices
make logs-frontend       # Frontend logs
make logs-price          # Price predictor logs
make logs-llm           # LLM hub logs
```

### **System Status:**
```bash
# All systems overview
docker-master.bat status

# Specific system status
docker-master.bat original status
docker-master.bat micro status
```

### **Performance Monitoring:**
```bash
# Real-time resource usage
docker stats

# Microservices monitoring dashboards
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

---

## 🧹 Maintenance

### **Regular Cleanup:**
```bash
# Clean specific system
docker-master.bat original clean
docker-master.bat micro clean

# Clean everything
docker-master.bat clean-all
```

### **Updates:**
```bash
# Update Original System
cd SRC
docker-start.bat update

# Update Microservices System
cd SRC/microservices
make down && make up
```

### **Backup:**
```bash
# Original System
cd SRC
make backup

# Microservices System
cd SRC/microservices
make db-backup
```

---

## 🎯 Use Case Recommendations

### **👨‍💻 For Developers:**
```bash
# Start with microservices development mode
docker-master.bat micro dev
# Benefits: Hot reload, debugging tools, Redis UI
```

### **🏢 For Production Deployment:**
```bash
# Use microservices production mode
docker-master.bat micro prod
# Benefits: Full monitoring, auto-scaling, enterprise features
```

### **🎓 For Learning/Demo:**
```bash
# Start with original system
docker-master.bat original start
# Benefits: Simple setup, all features in one container
```

### **⚡ For Quick Testing:**
```bash
# Use original development mode
docker-master.bat original dev
# Benefits: Fast startup, hot reload, minimal resources
```

---

## 🔐 Security Best Practices

1. **API Keys**: Never commit API keys to version control
2. **Environment Files**: Keep `.env` files secure and local
3. **Network**: Use Docker networks for service isolation
4. **Updates**: Regularly update Docker images and dependencies
5. **Monitoring**: Enable logging and monitoring in production
6. **Backup**: Regular database and configuration backups

---

## 📚 Additional Resources

### **Documentation:**
- [Docker Installation Guide](DOCKER_GUIDE.md)
- [Microservices Architecture](SRC/microservices/README.md)
- [API Documentation](http://localhost:8001/docs) (when running)

### **Support:**
- **Issues**: Check logs first with `docker-master.bat <system> logs`
- **Health**: Use `docker-master.bat <system> health` for diagnostics
- **Status**: Use `docker-master.bat status` for overview

### **Performance Tuning:**
- **Memory**: Increase Docker memory limit to 8GB+
- **CPU**: Allocate 4+ CPU cores to Docker
- **Storage**: Ensure 20GB+ free disk space
- **Network**: Use wired connection for better performance

---

## 🎉 Quick Reference Card

### **Most Common Commands:**

#### **First Time Setup:**
```bash
docker-master.bat original start    # Easiest start
```

#### **Development:**
```bash
docker-master.bat micro dev         # Best development experience
```

#### **Production:**
```bash
docker-master.bat micro prod        # Enterprise deployment
```

#### **Troubleshooting:**
```bash
docker-master.bat status            # Check all systems
docker-master.bat <system> logs     # View logs
docker-master.bat clean-all         # Nuclear cleanup
```

#### **Emergency:**
```bash
docker-master.bat stop-all          # Stop everything
docker kill $(docker ps -q)        # Force stop all containers
docker system prune -af --volumes   # Complete cleanup
```

---

**🚀 Ready to analyze stocks with AI-powered insights!**

Choose your deployment strategy and start building the future of financial analysis! 📈