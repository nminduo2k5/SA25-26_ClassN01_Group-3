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
goto :help

:start
echo 📦 Building Docker images...

if "%environment%"=="dev" goto :start_dev
if "%environment%"=="development" goto :start_dev
if "%environment%"=="prod" goto :start_prod
if "%environment%"=="production" goto :start_prod
if "%environment%"=="simple" goto :start_simple
if "%environment%"=="fast" goto :start_fast

echo ❌ Invalid environment. Use: dev, prod, simple, or fast
pause
exit /b 1

:start_dev
echo 🔧 Starting Development Environment...
docker-compose -f docker-compose.dev.yml down
docker-compose -f docker-compose.dev.yml build
docker-compose -f docker-compose.dev.yml up -d
echo ✅ Development services started!
echo 🌐 Streamlit: http://localhost:8501
echo 🔗 API: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/api/docs
goto :end

:start_prod
echo 🏭 Starting Production Environment...
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
echo ✅ Production services started!
echo 🌐 Application: http://localhost
echo 🔗 API: http://localhost:8000
goto :end

:start_simple
echo 🎯 Starting Simple Environment...
docker-compose down
docker-compose build
docker-compose up -d
echo ✅ Simple services started!
echo 🌐 Streamlit: http://localhost:8501
echo 🔗 API: http://localhost:8000
goto :end

:start_fast
echo ⚡ Starting Fast Environment...
docker-compose -f docker-compose.fast.yml down
docker-compose -f docker-compose.fast.yml build
docker-compose -f docker-compose.fast.yml up -d
echo ✅ Fast services started!
echo 🌐 Streamlit: http://localhost:8501
echo 🔗 API: http://localhost:8000
goto :end

:stop
echo 🛑 Stopping all services...
docker-compose -f docker-compose.dev.yml down 2>nul
docker-compose -f docker-compose.prod.yml down 2>nul
docker-compose down 2>nul
echo ✅ All services stopped
goto :end

:restart
echo 🔄 Restarting services...
call :stop
call :start
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
echo   start [env]    Start services (env: dev, prod, simple)
echo   stop           Stop all services
echo   restart [env]  Restart services
echo   logs [service] Show logs
echo   status         Show service status
echo   clean          Clean up Docker resources
echo   help           Show this help
echo.
echo Examples:
echo   %0 start dev       # Start development environment
echo   %0 start prod      # Start production environment
echo   %0 logs api        # Show API logs
echo   %0 status          # Show all service status
echo.

:end
if "%command%"=="help" pause
endlocal