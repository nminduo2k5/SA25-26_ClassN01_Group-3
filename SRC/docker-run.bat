@echo off
REM DUONG AI TRADING PRO - Docker Build & Run Script for Windows

echo 🚀 DUONG AI TRADING PRO - Docker Setup
echo ======================================

if "%1"=="build" goto build
if "%1"=="run" goto run
if "%1"=="start" goto start
if "%1"=="stop" goto stop
if "%1"=="logs" goto logs
if "%1"=="restart" goto restart
goto usage

:build
echo 📦 Building Docker image...
docker build -t duong-ai-trading-pro .
if %errorlevel% equ 0 (
    echo ✅ Docker image built successfully!
) else (
    echo ❌ Failed to build Docker image
    exit /b 1
)
goto end

:run
call :build
if %errorlevel% neq 0 goto end
call :start
goto end

:start
echo 🐳 Starting with docker-compose...
docker-compose up -d
if %errorlevel% equ 0 (
    echo ✅ Application started successfully!
    echo 🌐 Access at: http://localhost:8501
) else (
    echo ❌ Failed to start application
    exit /b 1
)
goto end

:stop
echo 🛑 Stopping containers...
docker-compose down 2>nul
docker stop duong-ai-trading-pro 2>nul
docker rm duong-ai-trading-pro 2>nul
echo ✅ Containers stopped
goto end

:logs
echo 📋 Showing logs...
docker-compose logs -f 2>nul || docker logs -f duong-ai-trading-pro 2>nul
goto end

:restart
call :stop
timeout /t 2 /nobreak >nul
call :start
goto end

:usage
echo Usage: %0 {build^|run^|start^|stop^|logs^|restart}
echo.
echo Commands:
echo   build   - Build Docker image only
echo   run     - Build and start application
echo   start   - Start application (image must exist)
echo   stop    - Stop application
echo   logs    - Show application logs
echo   restart - Restart application
echo.
echo Example: %0 run
exit /b 1

:end