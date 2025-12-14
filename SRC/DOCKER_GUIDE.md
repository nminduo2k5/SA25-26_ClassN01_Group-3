# 🐳 Docker Guide - Multi-Agent Vietnam Stock System

## 📋 Tổng quan

Hệ thống Multi-Agent Vietnam Stock được containerized với Docker để dễ dàng triển khai và quản lý. Hướng dẫn này sẽ giúp bạn chạy hệ thống bằng Docker.

## 🛠️ Yêu cầu hệ thống

- **Docker Desktop** (Windows/Mac) hoặc **Docker Engine** (Linux)
- **Docker Compose** v3.8+
- **RAM**: Tối thiểu 4GB, khuyến nghị 8GB+
- **Disk**: Tối thiểu 5GB trống

## 🚀 Cách sử dụng nhanh

### Windows:
```cmd
# Khởi động development
docker-start.bat start dev

# Khởi động production
docker-start.bat start prod

# Dừng tất cả services
docker-start.bat stop
```

### Linux/Mac:
```bash
# Cấp quyền thực thi
chmod +x docker-start.sh

# Khởi động development
./docker-start.sh start dev

# Khởi động production
./docker-start.sh start prod

# Dừng tất cả services
./docker-start.sh stop
```

## 📦 Các môi trường Docker

### 1. Development Environment
```bash
# Khởi động
docker-compose -f docker-compose.dev.yml up -d

# Truy cập
- Streamlit: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/api/docs
```

**Tính năng:**
- Hot reload cho code changes
- Volume mount để development
- Debug mode enabled
- Logs chi tiết

### 2. Production Environment
```bash
# Khởi động
docker-compose -f docker-compose.prod.yml up -d

# Truy cập
- Application: http://localhost (qua Nginx)
- API: http://localhost:8000
- Redis: localhost:6379
```

**Tính năng:**
- Gunicorn với multiple workers
- Nginx reverse proxy
- Redis caching
- Health checks
- Resource limits
- Auto restart

### 3. Simple Environment
```bash
# Khởi động
docker-compose up -d

# Truy cập
- Streamlit: http://localhost:8501
- API: http://localhost:8000
```

**Tính năng:**
- Cấu hình đơn giản
- Phù hợp cho testing nhanh

## 🔧 Cấu hình nâng cao

### Environment Variables

Tạo file `.env` trong thư mục SRC:

```env
# API Keys
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
SERPER_API_KEY=your_serper_key_here

# Database
DATABASE_URL=sqlite:///./duong_trading.db

# Redis (Production)
REDIS_URL=redis://redis:6379/0

# Logging
LOG_LEVEL=INFO
```

### Custom Docker Build

```bash
# Build với custom tag
docker build -t vnstock:custom .

# Build với build args
docker build --build-arg PYTHON_VERSION=3.11 -t vnstock:py311 .
```

### Volume Management

```bash
# Xem volumes
docker volume ls | grep vnstock

# Backup database
docker cp vnstock-api:/app/duong_trading.db ./backup/

# Restore database
docker cp ./backup/duong_trading.db vnstock-api:/app/
```

## 📊 Monitoring & Logs

### Xem logs
```bash
# Tất cả services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f streamlit

# Với timestamp
docker-compose logs -f -t api
```

### Health Checks
```bash
# Check API health
curl http://localhost:8000/health

# Check container status
docker-compose ps

# Resource usage
docker stats
```

### Performance Monitoring
```bash
# Container resource usage
docker stats vnstock-api vnstock-streamlit

# System resource usage
docker system df

# Network inspection
docker network inspect src_vnstock-network
```

## 🔒 Security & Production

### SSL/HTTPS Setup

1. Tạo SSL certificates:
```bash
mkdir -p nginx/ssl
# Copy your SSL certificates to nginx/ssl/
```

2. Cập nhật nginx.conf:
```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    # ... rest of config
}
```

### Firewall Rules
```bash
# Chỉ mở ports cần thiết
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 8000/tcp  # Block direct API access
ufw deny 8501/tcp  # Block direct Streamlit access
```

### Resource Limits
```yaml
# Trong docker-compose.prod.yml
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '0.5'
    reservations:
      memory: 512M
      cpus: '0.25'
```

## 🐛 Troubleshooting

### Common Issues

**1. Port conflicts:**
```bash
# Check port usage
netstat -tulpn | grep :8000
netstat -tulpn | grep :8501

# Kill process using port
sudo kill -9 $(lsof -t -i:8000)
```

**2. Memory issues:**
```bash
# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory

# Check container memory usage
docker stats --no-stream
```

**3. Permission issues (Linux):**
```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod +x docker-start.sh
```

**4. Database locked:**
```bash
# Stop all containers
docker-compose down

# Remove database lock
rm -f duong_trading.db-wal duong_trading.db-shm

# Restart
docker-compose up -d
```

### Debug Commands

```bash
# Enter container shell
docker exec -it vnstock-api bash
docker exec -it vnstock-streamlit bash

# Check container logs
docker logs vnstock-api --tail 100
docker logs vnstock-streamlit --tail 100

# Inspect container
docker inspect vnstock-api

# Check network connectivity
docker exec vnstock-api ping vnstock-streamlit
```

## 📈 Scaling & Performance

### Horizontal Scaling
```yaml
# Trong docker-compose.prod.yml
api:
  deploy:
    replicas: 3
  
streamlit:
  deploy:
    replicas: 2
```

### Load Balancing
```nginx
# Trong nginx.conf
upstream api_backend {
    server api_1:8000;
    server api_2:8000;
    server api_3:8000;
}
```

### Caching Strategy
```yaml
# Redis configuration
redis:
  image: redis:alpine
  command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
```

## 🔄 CI/CD Integration

### GitHub Actions Example
```yaml
name: Docker Build and Deploy

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build Docker image
      run: docker build -t vnstock:latest .
    
    - name: Run tests
      run: docker run --rm vnstock:latest python -m pytest
    
    - name: Deploy to production
      run: |
        docker-compose -f docker-compose.prod.yml down
        docker-compose -f docker-compose.prod.yml up -d
```

## 📚 Useful Commands

```bash
# Cleanup everything
docker system prune -a --volumes

# Update images
docker-compose pull
docker-compose up -d

# Backup entire system
docker run --rm -v src_vnstock-data:/data -v $(pwd):/backup alpine tar czf /backup/vnstock-backup.tar.gz /data

# Restore from backup
docker run --rm -v src_vnstock-data:/data -v $(pwd):/backup alpine tar xzf /backup/vnstock-backup.tar.gz -C /

# Export/Import images
docker save vnstock:latest | gzip > vnstock-image.tar.gz
docker load < vnstock-image.tar.gz
```

## 🆘 Support

Nếu gặp vấn đề:

1. Kiểm tra logs: `docker-compose logs -f`
2. Kiểm tra health: `curl http://localhost:8000/health`
3. Restart services: `docker-compose restart`
4. Clean rebuild: `docker-compose down && docker-compose build --no-cache && docker-compose up -d`

---

**Made with ❤️ for Vietnamese investors**

🚀 **Version 2.0 - Professional Docker Deployment**