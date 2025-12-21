# Docker Commands - Multi-Agent Vietnam Stock System

## 🚀 Quick Start Commands

### Build & Run
```bash
# Build và start tất cả services
docker-compose up -d --build

# Chỉ build images
docker-compose build

# Start services (đã build)
docker-compose up -d

# Stop tất cả
docker-compose down
```

## 🔄 Update & Maintenance

### Update với code mới
```bash
# Method 1: Auto script
update_docker.bat

# Method 2: Manual
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Clean & Rebuild
```bash
# Remove containers + images
docker-compose down --rmi all

# Clean system
docker system prune -f

# Rebuild from scratch
docker-compose build --no-cache
docker-compose up -d
```

## 📊 Monitoring & Logs

### Container Status
```bash
# Xem tất cả containers
docker-compose ps

# Xem logs real-time
docker-compose logs -f

# Logs specific service
docker-compose logs streamlit
docker-compose logs api
```

### Health Check
```bash
# Streamlit health
curl http://localhost:8501/healthz

# Container health
docker inspect vnstock-streamlit --format='{{.State.Health.Status}}'
```

## 🛠️ Development Commands

### Service Management
```bash
# Start Streamlit
docker-compose up streamlit -d

# Restart Streamlit
docker-compose restart streamlit

# Stop Streamlit
docker-compose stop streamlit
```

### Debug & Access
```bash
# Exec vào container
docker exec -it vnstock-streamlit bash
docker exec -it vnstock-api bash

# Copy files
docker cp file.py vnstock-streamlit:/app/
```

## 🔧 Configuration Commands

### Environment Variables
```bash
# Set custom env
docker-compose up -d --env-file .env.custom

# Override command
docker-compose run streamlit python test.py
```

### Volumes & Data
```bash
# Backup database
docker cp vnstock-streamlit:/app/duong_trading.db ./backup.db

# Restore database
docker cp ./backup.db vnstock-streamlit:/app/duong_trading.db
```

## 🌐 Network & Ports

### Port Management
```bash
# Check port usage
netstat -ano | findstr :8501
netstat -ano | findstr :8000

# Custom ports
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

### Network Debug
```bash
# Test Ollama connection from container
docker exec vnstock-streamlit curl http://host.docker.internal:11434/api/tags

# Network inspect
docker network ls
docker network inspect src_vnstock-network
```

## 🐛 Troubleshooting Commands

### Common Issues
```bash
# Container won't start
docker-compose logs --tail=50

# Port conflicts
docker-compose down
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Permission issues
docker-compose down
docker system prune -f
docker-compose up -d
```

### Reset Everything
```bash
# Nuclear option
docker-compose down --volumes --rmi all
docker system prune -af
docker volume prune -f
docker-compose build --no-cache
docker-compose up -d
```

## 📦 Image Management

### Build Specific Images
```bash
# Build only Streamlit
docker build -t vnstock-streamlit .

# Build with custom tag
docker build -t vnstock:v2.0 .

# Multi-stage build
docker build --target production -t vnstock-prod .
```

### Image Operations
```bash
# List images
docker images | grep vnstock

# Remove images
docker rmi vnstock-streamlit
docker rmi $(docker images -q vnstock*)

# Save/Load images
docker save vnstock:latest > vnstock.tar
docker load < vnstock.tar
```

## 🔄 Production Commands

### Production Deploy
```bash
# Production mode
docker-compose -f docker-compose.prod.yml up -d

# With scaling
docker-compose up -d --scale streamlit=2

# Rolling update
docker-compose up -d --no-deps streamlit
```

### Backup & Restore
```bash
# Backup volumes
docker run --rm -v src_vnstock-data:/data -v $(pwd):/backup alpine tar czf /backup/backup.tar.gz /data

# Restore volumes
docker run --rm -v src_vnstock-data:/data -v $(pwd):/backup alpine tar xzf /backup/backup.tar.gz -C /
```

## 📋 Service-Specific Commands

### Streamlit Service
```bash
# Start Streamlit only
docker-compose up streamlit -d

# Custom Streamlit command
docker-compose run streamlit streamlit run app.py --server.port=8502

# Debug Streamlit
docker-compose exec streamlit python -c "import streamlit; print(streamlit.__version__)"
```

### API Service
```bash
# Start API only
docker-compose up api -d

# Test API endpoints
curl http://localhost:8000/docs
curl http://localhost:8000/health

# API debug
docker-compose exec api python -c "from main_agent import MainAgent; print('OK')"
```

## 🎯 Quick Reference

### Most Used Commands
```bash
# Start everything
docker-compose up -d

# Update code
docker-compose down && docker-compose build --no-cache && docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop everything
docker-compose down

# Clean restart
docker-compose down && docker system prune -f && docker-compose up -d --build
```

### URLs
- **Streamlit**: http://localhost:8501
- **Health**: http://localhost:8501/healthz