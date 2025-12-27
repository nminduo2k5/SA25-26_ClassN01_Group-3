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

REM Set environment variables
if "%GEMINI_API_KEY%"=="" set GEMINI_API_KEY=your_gemini_api_key_here
if "%OPENAI_API_KEY%"=="" set OPENAI_API_KEY=your_openai_api_key_here

echo 🔧 Environment variables set
echo    GEMINI_API_KEY: %GEMINI_API_KEY:~0,10%...

REM Build and start services
echo.
echo 🏗️ Building and starting services...
docker-compose up --build -d

REM Wait for services
echo.
echo ⏳ Waiting for services to be ready...
timeout /t 30 /nobreak >nul

echo.
echo 🌐 Service URLs:
echo    Frontend: http://localhost:8502
echo    API Gateway: http://localhost:8080
echo    Price Predictor: http://localhost:8001/docs
echo    Investment Expert: http://localhost:8002/docs
echo    LLM Hub: http://localhost:8010/docs

echo.
echo 🎉 System is ready! Opening browser...
start http://localhost:8502

pause