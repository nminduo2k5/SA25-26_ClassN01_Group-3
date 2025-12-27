# Docker Compatibility Guide

## Port Mapping - Tránh xung đột với Docker cũ

### 🔄 Port Changes (Microservices vs Original)

| Service | Original Port | Microservices Port | Reason |
|---------|---------------|-------------------|---------|
| Frontend | 8501 | **8502** | Tránh xung đột với SRC/app.py |
| Gateway | 80 | **8080** | Tránh xung đột với system |
| Redis | 6379 | **6380** | Tránh xung đột nếu Redis local |
| PostgreSQL | 5432 | **5433** | Tránh xung đột nếu PostgreSQL local |

### 🚀 Cách chạy song song:

#### 1. Chạy Docker cũ (SRC/app.py):
```cmd
cd SRC
docker-compose up -d
# Truy cập: http://localhost:8501
```

#### 2. Chạy Microservices mới:
```cmd
cd SRC\microservices  
start-system.bat
# Truy cập: http://localhost:8502
```

### 🌐 URLs sau khi điều chỉnh:

**Microservices System:**
- Frontend: http://localhost:8502
- API Gateway: http://localhost:8080
- Price Predictor: http://localhost:8001/docs
- Investment Expert: http://localhost:8002/docs
- LLM Hub: http://localhost:8010/docs

**Original System:**
- Frontend: http://localhost:8501

### ✅ Compatibility Status:

- ✅ **Ports**: No conflicts
- ✅ **Networks**: Separate Docker networks
- ✅ **Volumes**: Different volume names
- ✅ **Services**: Can run simultaneously
- ✅ **Resources**: Minimal overlap

### 🔧 Internal Service Communication:

Microservices sử dụng internal Docker network, không ảnh hưởng đến hệ thống cũ.

### 📊 Resource Usage:

- **Original**: ~2GB RAM
- **Microservices**: ~4GB RAM  
- **Total**: ~6GB RAM (recommended 8GB+)