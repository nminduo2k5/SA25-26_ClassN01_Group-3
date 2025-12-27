@echo off
echo 🚀 Starting Multi-Agent Stock Analysis Microservices System
echo ============================================================

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Check if docker-compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ docker-compose is not installed. Please install Docker Desktop with docker-compose.
    pause
    exit /b 1
)

REM Set environment variables (you can modify these)
if "%GEMINI_API_KEY%"=="" set GEMINI_API_KEY=your_gemini_api_key_here
if "%OPENAI_API_KEY%"=="" set OPENAI_API_KEY=your_openai_api_key_here

echo 🔧 Environment variables set
echo    GEMINI_API_KEY: %GEMINI_API_KEY:~0,10%...
echo    OPENAI_API_KEY: %OPENAI_API_KEY:~0,10%...

REM Build and start services
echo.
echo 🏗️ Building and starting services...
docker-compose up --build -d

REM Wait for services to be ready
echo.
echo ⏳ Waiting for services to be ready...
timeout /t 30 /nobreak >nul

REM Check service health
echo.
echo 🔍 Checking service health...
docker-compose ps

echo.
echo 🌐 Service URLs:
echo    Frontend (Streamlit): http://localhost:8501
echo    API Gateway (Nginx): http://localhost:80
echo    Price Predictor API: http://localhost:8001
echo    Investment Expert API: http://localhost:8002
echo    LLM Hub API: http://localhost:8010
echo    Redis: localhost:6379
echo    PostgreSQL: localhost:5432

echo.
echo 📊 To view logs:
echo    docker-compose logs -f [service_name]
echo.
echo 🛑 To stop all services:
echo    docker-compose down
echo.
echo 🎉 Multi-Agent Stock Analysis System is ready!
echo    Open http://localhost:8501 in your browser to start using the system.
echo.

REM Open browser automatically
start http://localhost:8501

pause