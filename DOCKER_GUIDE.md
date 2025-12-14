# 🐳 Docker Guide - Multi-Agent Vietnam Stock System

## 🛠️ Cài đặt Docker

### 1. Windows (Khuyến nghị: Docker Desktop)

#### Cách 1: Docker Desktop (Dễ nhất)
```powershell
# Tải Docker Desktop từ: https://www.docker.com/products/docker-desktop/
# Hoặc dùng winget
winget install Docker.DockerDesktop

# Khởi động lại máy sau khi cài
# Mở Docker Desktop và đăng nhập
```

#### Cách 2: WSL2 + Docker Engine (Advanced)
```powershell
# 1. Cài WSL2
wsl --install
wsl --set-default-version 2

# 2. Cài Ubuntu trong WSL2
wsl --install -d Ubuntu

# 3. Trong WSL2 Ubuntu:
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker $USER
sudo systemctl enable docker
sudo systemctl start docker

# 4. Test
docker --version
docker-compose --version
```

### 2. macOS

#### Cách 1: Docker Desktop
```bash
# Tải từ: https://www.docker.com/products/docker-desktop/
# Hoặc dùng Homebrew
brew install --cask docker

# Mở Docker Desktop từ Applications
```

#### Cách 2: Homebrew + Docker Engine
```bash
# Cài Docker
brew install docker docker-compose

# Cài Docker Machine (nếu cần)
brew install docker-machine

# Test
docker --version
docker-compose --version
```

### 3. Linux (Ubuntu/Debian)

#### Cài đặt Docker Engine
```bash
# 1. Cập nhật system
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release

# 2. Thêm Docker GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 3. Thêm Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 4. Cài Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# 5. Cài Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 6. Thêm user vào docker group
sudo usermod -aG docker $USER
newgrp docker

# 7. Enable Docker service
sudo systemctl enable docker
sudo systemctl start docker
```

#### CentOS/RHEL/Fedora
```bash
# 1. Cài Docker
sudo dnf install -y docker docker-compose

# 2. Start Docker
sudo systemctl enable docker
sudo systemctl start docker

# 3. Thêm user vào group
sudo usermod -aG docker $USER
newgrp docker
```

### 4. Kiểm tra cài đặt

```bash
# Kiểm tra Docker version
docker --version
# Output: Docker version 24.0.x, build xxx

# Kiểm tra Docker Compose
docker-compose --version
# Output: Docker Compose version v2.x.x

# Test chạy container
docker run hello-world

# Kiểm tra Docker info
docker info

# Kiểm tra containers đang chạy
docker ps

# Kiểm tra images
docker images
```

### 5. Cấu hình Docker (Optional)

#### Tăng memory limit (Windows/macOS)
```bash
# Docker Desktop > Settings > Resources > Advanced
# Memory: 8GB+ (khuyến nghị cho hệ thống AI)
# CPU: 4+ cores
# Disk: 100GB+
```

#### Linux: Cấu hình daemon
```bash
# Tạo file cấu hình
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "default-address-pools": [
    {
      "base": "172.17.0.0/16",
      "size": 24
    }
  ]
}
EOF

# Restart Docker
sudo systemctl restart docker
```

### 6. Troubleshooting

#### Windows Issues
```powershell
# WSL2 không hoạt động
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Hyper-V conflicts
bcdedit /set hypervisorlaunchtype auto

# Docker Desktop không start
# Restart Docker Desktop service
Restart-Service -Name "Docker Desktop Service"
```

#### Linux Issues
```bash
# Permission denied
sudo chmod 666 /var/run/docker.sock

# Docker daemon không start
sudo systemctl status docker
sudo journalctl -u docker.service

# Port conflicts
sudo netstat -tulpn | grep :80
sudo netstat -tulpn | grep :443
```

#### macOS Issues
```bash
# Docker Desktop không start
# Reset Docker Desktop
# Docker Desktop > Troubleshoot > Reset to factory defaults

# Permission issues
sudo chown -R $(whoami) ~/.docker
```

### 7. Cài đặt bổ sung cho Development

```bash
# Docker Buildx (multi-platform builds)
docker buildx install

# Docker Scout (security scanning)
docker scout quickview

# Lazydocker (TUI for Docker)
# Linux/macOS
curl https://raw.githubusercontent.com/jesseduffield/lazydocker/master/scripts/install_update_linux.sh | bash

# Windows (Scoop)
scoop install lazydocker
```

### 8. Verification Script

```bash
#!/bin/bash
# docker-check.sh

echo "🐳 Docker Installation Check"
echo "========================="

# Check Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker: $(docker --version)"
else
    echo "❌ Docker not installed"
    exit 1
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose: $(docker-compose --version)"
elif docker compose version &> /dev/null; then
    echo "✅ Docker Compose (plugin): $(docker compose version)"
else
    echo "❌ Docker Compose not installed"
    exit 1
fi

# Check Docker daemon
if docker info &> /dev/null; then
    echo "✅ Docker daemon running"
else
    echo "❌ Docker daemon not running"
    exit 1
fi

# Test container
if docker run --rm hello-world &> /dev/null; then
    echo "✅ Docker container test passed"
else
    echo "❌ Docker container test failed"
    exit 1
fi

echo "🎉 Docker setup is ready for VNStock system!"
```

## 📋 Tổng quan

Tài liệu này hướng dẫn chi tiết cách containerize hệ thống **Multi-Agent Vietnam Stock** với Docker, bao gồm:

- 🤖 6 AI Agents (PricePredictor, InvestmentExpert, RiskExpert, TickerNews, MarketNews, StockInfo)
- 🧠 Gemini AI + OpenAI + Llama 3.1 (UnifiedAIAgent)
- 🚀 FastAPI Backend (20+ endpoints)
- 🎨 Streamlit Frontend (6 tabs)
- 🤖 CrewAI Integration
- 📊 SQLite Database
- 🏗️ 4 Multi-Architecture Systems

## 🏗️ Kiến trúc Docker

```
docker-compose.yml
├── app (Streamlit Frontend)
├── api (FastAPI Backend)  
├── nginx (Reverse Proxy)
└── volumes (Persistent Data)
```

## 📁 Cấu trúc Files Docker

### 1. Dockerfile chính (Multi-stage)

```dockerfile
# Dockerfile
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/cache

# Set permissions
RUN chmod +x /app/*.py

# Production stage
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "api.py"]
```

### 2. Docker Compose (Production)

```yaml
# docker-compose.yml
version: '3.8'

services:
  # FastAPI Backend
  api:
    build:
      context: .
      target: production
    container_name: vnstock_api
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
      - ENV=production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - api_cache:/app/cache
    networks:
      - vnstock_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    depends_on:
      - nginx

  # Streamlit Frontend
  app:
    build:
      context: .
      target: production
    container_name: vnstock_app
    ports:
      - "8501:8501"
    environment:
      - PYTHONPATH=/app
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - app_cache:/app/cache
    networks:
      - vnstock_network
    restart: unless-stopped
    command: ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: vnstock_nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    networks:
      - vnstock_network
    restart: unless-stopped
    depends_on:
      - api
      - app

networks:
  vnstock_network:
    driver: bridge

volumes:
  api_cache:
  app_cache:
```

### 3. Nginx Configuration

```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server api:8000;
    }
    
    upstream app_backend {
        server app:8501;
    }

    # API Server
    server {
        listen 80;
        server_name api.vnstock.local localhost;
        
        location / {
            proxy_pass http://api_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /health {
            proxy_pass http://api_backend/health;
            access_log off;
        }
    }

    # Streamlit App
    server {
        listen 80;
        server_name app.vnstock.local;
        
        location / {
            proxy_pass http://app_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
        
        location /_stcore/stream {
            proxy_pass http://app_backend/_stcore/stream;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

### 4. Development Docker Compose

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  # Development API with hot reload
  api-dev:
    build:
      context: .
      target: base
    container_name: vnstock_api_dev
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
      - ENV=development
      - RELOAD=true
    volumes:
      - .:/app
      - dev_cache:/app/cache
    networks:
      - vnstock_dev
    command: ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

  # Development Streamlit with hot reload
  app-dev:
    build:
      context: .
      target: base
    container_name: vnstock_app_dev
    ports:
      - "8501:8501"
    environment:
      - PYTHONPATH=/app
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    volumes:
      - .:/app
      - dev_cache:/app/cache
    networks:
      - vnstock_dev
    command: ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true"]

networks:
  vnstock_dev:
    driver: bridge

volumes:
  dev_cache:
```

### 5. Environment Files

```bash
# .env.production
ENV=production
PYTHONPATH=/app

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Database
DATABASE_PATH=/app/data/duong_trading.db

# Logging
LOG_LEVEL=INFO
LOG_PATH=/app/logs
```

```bash
# .env.development
ENV=development
PYTHONPATH=/app
RELOAD=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Database
DATABASE_PATH=./duong_trading.db

# Logging
LOG_LEVEL=DEBUG
LOG_PATH=./logs
```

### 6. Docker Ignore

```gitignore
# .dockerignore
.git
.gitignore
README.md
Dockerfile
docker-compose*.yml
.env*
.vscode
.idea
__pycache__
*.pyc
*.pyo
*.pyd
.pytest_cache
.coverage
htmlcov
.tox
.cache
.mypy_cache
*.egg-info
dist
build
node_modules
.DS_Store
Thumbs.db
*.log
logs/
cache/
*.db-journal
*.db-wal
*.db-shm
```

## 🚀 Hướng dẫn Build và Deploy

### 0. Kiểm tra Docker đã cài đặt

```bash
# Chạy script kiểm tra
bash docker-check.sh

# Hoặc kiểm tra thủ công
docker --version && docker-compose --version && docker info
```

### 1. Chuẩn bị môi trường

```bash
# Clone repository
git clone <your-repo>
cd SA25-26_ClassN01_Group-3/SRC

# Tạo thư mục cần thiết
mkdir -p data logs nginx/ssl

# Copy environment files
cp .env.production .env
```

### 2. Build Images

```bash
# Build production image
docker build -t vnstock:latest .

# Hoặc build với tag cụ thể
docker build -t vnstock:v2.0.0 .

# Build development image
docker build --target base -t vnstock:dev .
```

### 3. Development Mode

```bash
# Chạy development environment
docker-compose -f docker-compose.dev.yml up -d

# Xem logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop
docker-compose -f docker-compose.dev.yml down
```

### 4. Production Deployment

```bash
# Deploy production
docker-compose up -d

# Kiểm tra status
docker-compose ps

# Xem logs
docker-compose logs -f api
docker-compose logs -f app

# Scale services (nếu cần)
docker-compose up -d --scale api=3

# Stop
docker-compose down
```

### 5. Health Checks

```bash
# Kiểm tra API health
curl http://localhost:8000/health

# Kiểm tra Streamlit health
curl http://localhost:8501/_stcore/health

# Kiểm tra từng container
docker exec vnstock_api curl -f http://localhost:8000/health
docker exec vnstock_app curl -f http://localhost:8501/_stcore/health
```

## 🔧 Configuration Management

### 1. API Keys trong Docker

```yaml
# docker-compose.yml (thêm vào services)
services:
  api:
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SERPER_API_KEY=${SERPER_API_KEY}
    secrets:
      - gemini_key
      - openai_key
      - serper_key

secrets:
  gemini_key:
    file: ./secrets/gemini_key.txt
  openai_key:
    file: ./secrets/openai_key.txt
  serper_key:
    file: ./secrets/serper_key.txt
```

### 2. Database Persistence

```yaml
# Thêm vào docker-compose.yml
services:
  api:
    volumes:
      - db_data:/app/data
      - ./backups:/app/backups

volumes:
  db_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./data
```

### 3. SSL Configuration

```bash
# Tạo SSL certificates (development)
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/vnstock.key \
  -out nginx/ssl/vnstock.crt \
  -subj "/C=VN/ST=HCM/L=HCM/O=VNStock/CN=localhost"
```

## 📊 Monitoring và Logging

### 1. Logging Configuration

```yaml
# docker-compose.yml (thêm vào services)
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service=api"
  
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service=app"
```

### 2. Monitoring với Prometheus (Optional)

```yaml
# monitoring/docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - vnstock_network

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    networks:
      - vnstock_network

networks:
  vnstock_network:
    external: true
```

## 🔄 CI/CD Pipeline

### 1. GitHub Actions

```yaml
# .github/workflows/docker.yml
name: Docker Build and Deploy

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to DockerHub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: ./SRC
        push: true
        tags: |
          your-username/vnstock:latest
          your-username/vnstock:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      run: |
        # Add your deployment script here
        echo "Deploying to production..."
```

## 🛠️ Troubleshooting

### 1. Common Issues

```bash
# Container không start
docker-compose logs api
docker-compose logs app

# Port conflicts
docker-compose down
sudo lsof -i :8000
sudo lsof -i :8501

# Permission issues
sudo chown -R $USER:$USER ./data ./logs
chmod 755 ./data ./logs

# Memory issues
docker system prune -a
docker volume prune
```

### 2. Debug Commands

```bash
# Vào container để debug
docker exec -it vnstock_api bash
docker exec -it vnstock_app bash

# Kiểm tra logs real-time
docker-compose logs -f --tail=100 api

# Kiểm tra resource usage
docker stats

# Kiểm tra network
docker network ls
docker network inspect vnstock_network
```

### 3. Performance Tuning

```yaml
# docker-compose.yml (thêm resource limits)
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
  
  app:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
```

## 📋 Checklist Deployment

### Pre-deployment
- [ ] Kiểm tra requirements.txt đầy đủ
- [ ] Cấu hình environment variables
- [ ] Tạo SSL certificates (nếu cần)
- [ ] Backup database hiện tại
- [ ] Test build locally

### Deployment
- [ ] Build images thành công
- [ ] All containers start healthy
- [ ] Health checks pass
- [ ] API endpoints accessible
- [ ] Streamlit app loads correctly
- [ ] Database connections work
- [ ] AI agents respond correctly

### Post-deployment
- [ ] Monitor logs for errors
- [ ] Test core functionality
- [ ] Verify data persistence
- [ ] Check performance metrics
- [ ] Setup monitoring alerts

## 🔐 Security Best Practices

### 1. Container Security

```dockerfile
# Dockerfile security improvements
FROM python:3.11-slim

# Don't run as root
RUN useradd --create-home --shell /bin/bash app
USER app

# Remove unnecessary packages
RUN apt-get autoremove -y && apt-get clean

# Use specific versions
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

### 2. Network Security

```yaml
# docker-compose.yml
services:
  api:
    networks:
      - internal
      - external
  
  app:
    networks:
      - internal
  
  nginx:
    networks:
      - external

networks:
  internal:
    internal: true
  external:
```

### 3. Secrets Management

```bash
# Sử dụng Docker secrets thay vì environment variables
echo "your-gemini-key" | docker secret create gemini_key -
echo "your-openai-key" | docker secret create openai_key -
```

## 📈 Scaling và Load Balancing

### 1. Horizontal Scaling

```bash
# Scale API service
docker-compose up -d --scale api=3

# Scale với load balancer
docker-compose -f docker-compose.yml -f docker-compose.scale.yml up -d
```

### 2. Load Balancer Configuration

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  nginx:
    volumes:
      - ./nginx/nginx.scale.conf:/etc/nginx/nginx.conf:ro
```

## 🎯 Kết luận

Hệ thống Multi-Agent Vietnam Stock đã được containerize hoàn chỉnh với:

- ✅ **Production-ready** Docker setup
- ✅ **Development** environment với hot reload
- ✅ **Nginx** reverse proxy và load balancing
- ✅ **Health checks** và monitoring
- ✅ **Security** best practices
- ✅ **CI/CD** pipeline ready
- ✅ **Scaling** capabilities

### Next Steps:
1. Customize environment variables cho production
2. Setup SSL certificates
3. Configure monitoring và alerting
4. Implement backup strategies
5. Setup CI/CD pipeline

### Support:
- 📧 Email: duongnguyenminh808@gmail.com
- 🐛 Issues: GitHub Issues
- 📚 Docs: README.md

---

**Made with ❤️ for Vietnamese investors**