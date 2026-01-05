# 🐳 DUONG AI TRADING PRO - Docker Setup

Hướng dẫn chạy ứng dụng DUONG AI TRADING PRO bằng Docker.

## 🚀 Quick Start

### 1. Chuẩn bị API Keys

Tạo file `.env` trong thư mục SRC:

```bash
# Copy từ template
cp .env.example .env

# Hoặc tạo mới
echo "GEMINI_API_KEY=your_actual_gemini_key" > .env
echo "OPENAI_API_KEY=your_actual_openai_key" >> .env
echo "LLAMA_API_KEY=your_actual_llama_key" >> .env
echo "SERPER_API_KEY=your_actual_serper_key" >> .env
```

### 2. Chạy ứng dụng

#### Windows:
```cmd
docker-run.bat run
```

#### Linux/Mac:
```bash
chmod +x docker-run.sh
./docker-run.sh run
```

### 3. Truy cập ứng dụng

Mở trình duyệt: **http://localhost:8501**

## 📋 Các lệnh Docker

### Build image:
```bash
# Windows
docker-run.bat build

# Linux/Mac  
./docker-run.sh build
```

### Start/Stop:
```bash
# Start
docker-run.bat start

# Stop
docker-run.bat stop

# Restart
docker-run.bat restart
```

### Xem logs:
```bash
docker-run.bat logs
```

## 🔧 Manual Docker Commands

### Build:
```bash
docker build -t duong-ai-trading-pro .
```

### Run:
```bash
docker run -d \
  --name duong-ai-trading-pro \
  -p 8501:8501 \
  --env-file .env \
  duong-ai-trading-pro
```

### Docker Compose:
```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Logs
docker-compose logs -f
```

## 📦 Container Info

- **Image**: `duong-ai-trading-pro`
- **Container**: `duong-ai-trading-pro`
- **Port**: `8501`
- **Base**: `python:3.11-slim`

## 🛠️ Troubleshooting

### Port đã được sử dụng:
```bash
# Kiểm tra port 8501
netstat -an | findstr 8501

# Thay đổi port trong docker-compose.yml
ports:
  - "8502:8501"  # Đổi thành port khác
```

### Container không start:
```bash
# Xem logs chi tiết
docker logs duong-ai-trading-pro

# Kiểm tra container
docker ps -a
```

### API Keys không hoạt động:
```bash
# Kiểm tra file .env
cat .env

# Restart container sau khi sửa .env
docker-run.bat restart
```

## 🔒 Security Notes

- Không commit file `.env` vào Git
- Sử dụng API keys thật trong production
- Cân nhắc sử dụng Docker secrets cho production

## 📊 System Requirements

- **Docker**: >= 20.10
- **Docker Compose**: >= 2.0
- **RAM**: >= 2GB
- **Storage**: >= 5GB

## 🌐 Production Deployment

Để deploy production, cập nhật `docker-compose.yml`:

```yaml
services:
  duong-ai-trading:
    build: .
    ports:
      - "80:8501"  # Sử dụng port 80
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_ENABLE_CORS=false
    restart: always
```

## 📞 Support

Nếu gặp vấn đề với Docker setup:

1. Kiểm tra Docker đã cài đặt: `docker --version`
2. Kiểm tra Docker Compose: `docker-compose --version`
3. Xem logs: `docker-run.bat logs`
4. Restart: `docker-run.bat restart`