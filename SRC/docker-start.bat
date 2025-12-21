@echo off
REM Multi-Agent Vietnam Stock System Docker Startup Script for Windows

setlocal enabledelayedexpansion

echo 🚀 Starting Multi-Agent Vietnam Stock System...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)
echo ✅ Docker is running

REM Parse command line arguments
set "command=%1"
set "environment=%2"

if "%command%"=="" set "command=help"
if "%environment%"=="" set "environment=dev"

if "%command%"=="start" goto :start
if "%command%"=="stop" goto :stop
if "%command%"=="restart" goto :restart
if "%command%"=="logs" goto :logs
if "%command%"=="status" goto :status
if "%command%"=="clean" goto :clean
if "%command%"=="update" goto :update
goto :help

:start
echo 📦 Building Docker images...

if "%environment%"=="dev" goto :start_dev
if "%environment%"=="development" goto :start_dev
if "%environment%"=="prod" goto :start_prod
if "%environment%"=="production" goto :start_prod

echo ❌ Invalid environment. Use: dev or prod
pause
exit /b 1

:start_dev
echo 🔧 Starting Development Environment...
docker-compose down
docker-compose build
docker-compose up -d
echo ✅ Development services started!
echo 🌐 Streamlit: http://localhost:8501
goto :end

:start_prod
echo 🏭 Starting Production Environment...
docker-compose down
docker-compose build --no-cache
docker-compose up -d
echo ✅ Production services started!
echo 🌐 Streamlit: http://localhost:8501
goto :end

:stop
echo 🛑 Stopping all services...
docker-compose down 2>nul
echo ✅ All services stopped
goto :end

:restart
echo 🔄 Restarting services...
call :stop
call :start %environment%
goto :end

:update
echo 🔄 Updating Docker with latest code changes...
echo ==========================================
echo.
echo 1. Stopping existing containers...
docker-compose down
echo.
echo 2. Removing old images...
docker rmi src-streamlit:latest 2>nul
echo.
echo 3. Rebuilding images with latest code...
docker-compose build --no-cache
echo.
echo 4. Starting updated containers...
docker-compose up -d
echo.
echo 5. Checking container status...
docker-compose ps
echo.
echo ✅ Docker update completed!
echo 🌐 Streamlit: http://localhost:8501
goto :end

:logs
if "%2"=="" (
    echo 📋 Showing all logs...
    docker-compose logs -f
) else (
    echo 📋 Showing logs for %2...
    docker-compose logs -f %2
)
goto :end

:status
echo 📊 Service Status:
docker-compose ps
echo.
echo 🐳 Docker Images:
docker images | findstr vnstock
echo.
echo 📦 Docker Volumes:
docker volume ls | findstr vnstock
goto :end

:clean
echo 🧹 Cleaning up Docker resources...
call :stop
docker system prune -f
docker volume prune -f
echo ✅ Cleanup completed
goto :end

:help
echo 🔧 Multi-Agent Vietnam Stock System Docker Manager
echo.
echo Usage: %0 ^<command^> [options]
echo.
echo Commands:
echo   start [env]    Start services (env: dev, prod)
echo   stop           Stop all services
echo   restart [env]  Restart services
echo   update         Update with latest code changes
echo   logs [service] Show logs
echo   status         Show service status
echo   clean          Clean up Docker resources
echo   help           Show this help
echo.
echo Examples:
echo   %0 start dev       # Start development environment
echo   %0 start prod      # Start production environment
echo   %0 update          # Update with latest code
echo   %0 logs streamlit  # Show Streamlit logs
echo   %0 status          # Show service status
echo.

:end
if "%command%"=="help" pause
endlocal