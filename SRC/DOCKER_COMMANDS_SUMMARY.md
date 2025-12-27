# 🐳 Tổng hợp Câu lệnh Docker - Multi-Agent Stock System

## 📍 Vị trí các Docker Systems

### 1. **Original System** (SRC/)
```bash
cd SRC/
# Port: 8501
```

### 2. **Microservices System** (SRC/microservices/)
```bash
cd SRC/microservices/
# Ports: 8502, 8080, 8001, 8002, 8010
```

## 🚀 Câu lệnh cơ bản

### **Original System (SRC/)**
```bash
# Khởi chạy
cd SRC
docker-compose up -d

# Xem logs
docker-compose logs -f

# Dừng
docker-compose down

# Rebuild
docker-compose up --build -d

# Truy cập: http://localhost:8501
```

### **Microservices System (SRC/microservices/)**

#### **Quick Start:**
```bash
cd SRC/microservices

# Windows
start-system.bat

# Linux/Mac  
chmod +x setup.sh
./setup.sh
```

#### **Manual Commands:**
```bash
cd SRC/microservices

# Development environment
make dev
# hoặc
docker-compose -f docker-compose.dev.yml up --build -d

# Production environment  
make prod
# hoặc
docker-compose -f docker-compose.production.yml up --build -d

# Basic environment
make up
# hoặc
docker-compose up --build -d
```

## 🎯 Makefile Commands (SRC/microservices/)

### **Environment Management:**
```bash
make help          # Show all commands
make setup         # Initial setup
make dev           # Development with hot reload
make prod          # Production with monitoring
make up            # Basic environment
make down          # Stop all services
make restart       # Restart all services
```

### **Monitoring & Health:**
```bash
make health        # Check service health
make logs          # Show all logs
make logs-frontend # Frontend logs only
make logs-price    # Price predictor logs
make logs-llm      # LLM hub logs
make monitor       # Open monitoring dashboard
make metrics       # Show system metrics
```

### **Database Operations:**
```bash
make db-backup     # Backup database
make db-restore    # Restore from backup
make db-shell      # Connect to database
```

### **Cache Operations:**
```bash
make cache-clear   # Clear Redis cache
make cache-info    # Show cache info
```

### **Testing:**
```bash
make test          # Run all tests
make test-unit     # Unit tests only
make test-integration # Integration tests
```

### **Cleanup:**
```bash
make clean         # Clean Docker resources
make clean-all     # Clean everything including images
```

## 🌐 Service URLs

### **Original System:**
- Frontend: http://localhost:8501

### **Microservices System:**

#### **Development:**
- Frontend: http://localhost:8502
- API Gateway: http://localhost:8080
- Database Admin: http://localhost:8081
- Redis Admin: http://localhost:8082

#### **Production:**
- Frontend: http://localhost:8502
- API Gateway: http://localhost:8080
- Monitoring: http://localhost:3000 (admin/admin123)
- Metrics: http://localhost:9090
- Logs: http://localhost:5601

#### **API Documentation:**
- Price Predictor: http://localhost:8001/docs
- Investment Expert: http://localhost:8002/docs
- LLM Hub: http://localhost:8010/docs

## 🔧 Docker Compose Commands

### **Basic Operations:**
```bash
# Start services
docker-compose up -d

# Start with rebuild
docker-compose up --build -d

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f [service_name]

# Check service status
docker-compose ps

# Restart service
docker-compose restart [service_name]

# Scale service
docker-compose up -d --scale price-predictor=3
```

### **Multi-file Compose:**
```bash
# Development
docker-compose -f docker-compose.dev.yml up -d

# Production
docker-compose -f docker-compose.production.yml up -d

# Override files
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

## 🐳 Docker Commands

### **Container Management:**
```bash
# List running containers
docker ps

# List all containers
docker ps -a

# Stop container
docker stop [container_name]

# Remove container
docker rm [container_name]

# Execute command in container
docker exec -it [container_name] bash

# View container logs
docker logs -f [container_name]

# Inspect container
docker inspect [container_name]
```

### **Image Management:**
```bash
# List images
docker images

# Build image
docker build -t vnstock/app .

# Remove image
docker rmi [image_name]

# Pull image
docker pull [image_name]

# Push image
docker push [image_name]

# Prune unused images
docker image prune -f
```

### **Volume Management:**
```bash
# List volumes
docker volume ls

# Create volume
docker volume create [volume_name]

# Remove volume
docker volume rm [volume_name]

# Prune unused volumes
docker volume prune -f
```

### **Network Management:**
```bash
# List networks
docker network ls

# Create network
docker network create [network_name]

# Remove network
docker network rm [network_name]

# Inspect network
docker network inspect [network_name]
```

## 🔍 Health Checks & Monitoring

### **Service Health:**
```bash
# Check all services
curl http://localhost:8080/health

# Individual services
curl http://localhost:8001/health  # Price Predictor
curl http://localhost:8002/health  # Investment Expert
curl http://localhost:8010/health  # LLM Hub
```

### **System Monitoring:**
```bash
# Resource usage
docker stats

# System info
docker system df

# System events
docker system events

# Container processes
docker top [container_name]
```

## 🧹 Cleanup Commands

### **Basic Cleanup:**
```bash
# Stop all containers
docker stop $(docker ps -aq)

# Remove all containers
docker rm $(docker ps -aq)

# Remove unused images
docker image prune -f

# Remove unused volumes
docker volume prune -f

# Remove unused networks
docker network prune -f
```

### **Complete Cleanup:**
```bash
# Remove everything
docker system prune -af --volumes

# Reset Docker (nuclear option)
docker system prune -af --volumes
docker builder prune -af
```

## 🚨 Troubleshooting Commands

### **Debug Container:**
```bash
# Enter container shell
docker exec -it [container_name] /bin/bash

# Check container processes
docker exec [container_name] ps aux

# Check container environment
docker exec [container_name] env

# Check container network
docker exec [container_name] netstat -tulpn
```

### **Port Conflicts:**
```bash
# Check port usage (Windows)
netstat -ano | findstr :8501

# Check port usage (Linux/Mac)
lsof -i :8501
netstat -tulpn | grep :8501

# Kill process using port
# Windows
taskkill /PID [PID] /F

# Linux/Mac
kill -9 [PID]
```

### **Memory Issues:**
```bash
# Check Docker memory usage
docker stats --no-stream

# Increase Docker memory (Docker Desktop)
# Settings > Resources > Memory > 8GB+

# Check system memory
free -h  # Linux
vm_stat  # Mac
```

## 📊 Performance Optimization

### **Resource Limits:**
```yaml
# docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### **Build Optimization:**
```bash
# Multi-stage build
docker build --target production -t vnstock/app .

# Build with cache
docker build --cache-from vnstock/app:latest -t vnstock/app .

# Build without cache
docker build --no-cache -t vnstock/app .
```

## 🔐 Security Commands

### **Security Scan:**
```bash
# Scan image for vulnerabilities
docker scout quickview [image_name]

# Detailed security report
docker scout cves [image_name]
```

### **Secrets Management:**
```bash
# Create secret
echo "my_secret" | docker secret create my_secret -

# List secrets
docker secret ls

# Use secret in compose
services:
  app:
    secrets:
      - my_secret
```

## 📋 Quick Reference

### **Most Used Commands:**
```bash
# Start development
cd SRC/microservices && make dev

# Start production
cd SRC/microservices && make prod

# Check health
make health

# View logs
make logs

# Stop everything
make down

# Clean up
make clean
```

### **Emergency Commands:**
```bash
# Stop everything immediately
docker kill $(docker ps -q)

# Remove everything
docker system prune -af --volumes

# Restart Docker daemon
sudo systemctl restart docker  # Linux
# Restart Docker Desktop        # Windows/Mac
```

### **Environment Variables:**
```bash
# Set API keys
export GEMINI_API_KEY="your_key"
export OPENAI_API_KEY="your_key"

# Or use .env file
cp .env.template .env
# Edit .env with your keys
```

## 🎯 Best Practices

1. **Always use specific tags:** `image:v1.0` not `image:latest`
2. **Use multi-stage builds** for smaller images
3. **Set resource limits** to prevent resource exhaustion
4. **Use health checks** for better reliability
5. **Regular cleanup** to free disk space
6. **Monitor logs** for issues
7. **Backup data** before major changes
8. **Use .dockerignore** to exclude unnecessary files

---

**🚀 Ready to deploy and scale your Multi-Agent Stock Analysis System!**