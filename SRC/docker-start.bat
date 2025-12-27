@echo off
setlocal enabledelayedexpansion

echo 🚀 Multi-Agent Vietnam Stock System - Docker Manager
echo =====================================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    echo 💡 Tip: Make sure Docker Desktop is installed and running
    pause
    exit /b 1
)

REM Parse command line arguments
set "command=%1"
if "%command%"=="" set "command=start"

if "%command%"=="start" goto :start
if "%command%"=="dev" goto :dev
if "%command%"=="stop" goto :stop
if "%command%"=="restart" goto :restart
if "%command%"=="logs" goto :logs
if "%command%"=="status" goto :status
if "%command%"=="clean" goto :clean
if "%command%"=="update" goto :update
if "%command%"=="help" goto :help
goto :help

:start
echo 🏭 Starting Production Environment...
echo ====================================
echo.
echo 1. Stopping existing containers...
docker-compose down 2>nul
echo.
echo 2. Building optimized images...
docker-compose build --no-cache
echo.
echo 3. Starting services...
docker-compose up -d
echo.
echo 4. Waiting for services to be ready...
timeout /t 15 /nobreak >nul
echo.
echo 5. Checking service health...
docker-compose ps
echo.
echo ✅ Production environment ready!
echo 🌐 Access: http://localhost:8501
echo 📊 Redis: localhost:6379
echo.
echo 💡 Use 'docker-start.bat logs' to view logs
echo 💡 Use 'docker-start.bat stop' to stop services
goto :end

:dev
echo 🔧 Starting Development Environment...
echo ====================================
echo.
echo 1. Stopping existing containers...
docker-compose -f docker-compose.dev.yml down 2>nul
echo.
echo 2. Building development images...
docker-compose -f docker-compose.dev.yml build
echo.
echo 3. Starting development services...
docker-compose -f docker-compose.dev.yml up -d
echo.
echo 4. Waiting for services to be ready...
timeout /t 20 /nobreak >nul
echo.
echo 5. Checking service health...
docker-compose -f docker-compose.dev.yml ps
echo.
echo ✅ Development environment ready!
echo 🌐 App: http://localhost:8501 (with hot reload)
echo 🔧 Redis UI: http://localhost:8081
echo 📊 Redis: localhost:6379
echo.
echo 💡 Code changes will auto-reload
echo 💡 Use 'docker-start.bat logs' to view logs
goto :end

:stop
echo 🛑 Stopping all services...
docker-compose down 2>nul
docker-compose -f docker-compose.dev.yml down 2>nul
echo ✅ All services stopped
goto :end

:restart
echo 🔄 Restarting services...
call :stop
timeout /t 3 /nobreak >nul
call :start
goto :end

:update
echo 🔄 Updating with latest code changes...
echo =====================================
echo.
echo 1. Stopping containers...
docker-compose down
docker-compose -f docker-compose.dev.yml down 2>nul
echo.
echo 2. Removing old images...
docker rmi src-app:latest 2>nul
docker rmi src_app:latest 2>nul
echo.
echo 3. Rebuilding with latest code...
docker-compose build --no-cache
echo.
echo 4. Starting updated services...
docker-compose up -d
echo.
echo 5. Verifying deployment...
timeout /t 10 /nobreak >nul
docker-compose ps
echo.
echo ✅ Update completed successfully!
echo 🌐 Access: http://localhost:8501
goto :end

:logs
if "%2"=="" (
    echo 📋 Showing all logs (Ctrl+C to exit)...
    docker-compose logs -f
) else (
    echo 📋 Showing logs for %2...
    docker-compose logs -f %2
)
goto :end

:status
echo 📊 System Status
echo ===============
echo.
echo 🐳 Running Containers:
docker-compose ps
echo.
echo 📦 Docker Images:
docker images | findstr vnstock
echo.
echo 💾 Docker Volumes:
docker volume ls | findstr vnstock
echo.
echo 🌐 Network Status:
docker network ls | findstr vnstock
echo.
echo 📈 Resource Usage:
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
goto :end

:clean
echo 🧹 Cleaning up Docker resources...
echo =================================
echo.
echo 1. Stopping all containers...
call :stop
echo.
echo 2. Removing unused containers...
docker container prune -f
echo.
echo 3. Removing unused images...
docker image prune -f
echo.
echo 4. Removing unused volumes...
docker volume prune -f
echo.
echo 5. Removing unused networks...
docker network prune -f
echo.
echo ✅ Cleanup completed!
echo 💡 Use 'docker system prune -af' for deep clean
goto :end

:help
echo 🔧 Multi-Agent Vietnam Stock System - Docker Commands
echo ===================================================
echo.
echo Usage: docker-start.bat [command]
echo.
echo Commands:
echo   start     Start production environment (default)
echo   dev       Start development environment with hot reload
echo   stop      Stop all services
echo   restart   Restart services
echo   update    Update with latest code changes
echo   logs      Show logs (add service name for specific logs)
echo   status    Show detailed system status
echo   clean     Clean up Docker resources
echo   help      Show this help
echo.
echo Examples:
echo   docker-start.bat              # Start production
echo   docker-start.bat dev          # Start development
echo   docker-start.bat logs app     # Show app logs
echo   docker-start.bat status       # Show system status
echo   docker-start.bat clean        # Clean up resources
echo.
echo 🌐 URLs:
echo   Production:  http://localhost:8501
echo   Development: http://localhost:8501 (hot reload)
echo   Redis UI:    http://localhost:8081 (dev only)
echo.

:end
if "%command%"=="help" pause
endlocal