@echo off
setlocal enabledelayedexpansion

echo 🚀 Multi-Agent Stock Analysis - Master Docker Manager
echo =====================================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Parse command line arguments
set "system=%1"
set "command=%2"

if "%system%"=="" goto :help
if "%command%"=="" set "command=start"

if "%system%"=="original" goto :original
if "%system%"=="micro" goto :microservices
if "%system%"=="microservices" goto :microservices
if "%system%"=="help" goto :help
if "%system%"=="status" goto :status_all
if "%system%"=="stop-all" goto :stop_all
if "%system%"=="clean-all" goto :clean_all
goto :help

:original
echo 🏠 Managing Original System (SRC/)
echo =================================
cd SRC
if "%command%"=="start" (
    echo 🚀 Starting Original System...
    call docker-start.bat
) else if "%command%"=="dev" (
    echo 🔧 Starting Original System in Development mode...
    call docker-start.bat dev
) else if "%command%"=="prod" (
    echo 🏭 Starting Original System in Production mode...
    call docker-start.bat start
) else if "%command%"=="stop" (
    echo 🛑 Stopping Original System...
    call docker-start.bat stop
) else if "%command%"=="logs" (
    echo 📋 Showing Original System logs...
    call docker-start.bat logs
) else if "%command%"=="status" (
    echo 📊 Original System status...
    call docker-start.bat status
) else if "%command%"=="clean" (
    echo 🧹 Cleaning Original System...
    call docker-start.bat clean
) else (
    echo ❌ Unknown command for original system: %command%
    goto :help_original
)
goto :end

:microservices
echo 🏗️ Managing Microservices System (SRC/microservices/)
echo ===================================================
cd SRC\microservices
if "%command%"=="start" (
    echo 🚀 Starting Microservices System...
    call start-system.bat
) else if "%command%"=="dev" (
    echo 🔧 Starting Microservices in Development mode...
    docker-compose -f docker-compose.dev.yml up --build -d
    echo ✅ Microservices Development ready!
    echo 🌐 Frontend: http://localhost:8502
    echo 🔧 Redis UI: http://localhost:8081
) else if "%command%"=="prod" (
    echo 🏭 Starting Microservices in Production mode...
    docker-compose -f docker-compose.production.yml up --build -d
    echo ✅ Microservices Production ready!
    echo 🌐 Frontend: http://localhost:8502
    echo 📊 Monitoring: http://localhost:3000
) else if "%command%"=="basic" (
    echo ⚡ Starting Microservices Basic mode...
    docker-compose up --build -d
    echo ✅ Microservices Basic ready!
    echo 🌐 Frontend: http://localhost:8502
) else if "%command%"=="stop" (
    echo 🛑 Stopping Microservices System...
    docker-compose down 2>nul
    docker-compose -f docker-compose.dev.yml down 2>nul
    docker-compose -f docker-compose.production.yml down 2>nul
    echo ✅ Microservices stopped
) else if "%command%"=="logs" (
    echo 📋 Showing Microservices logs...
    docker-compose logs -f
) else if "%command%"=="status" (
    echo 📊 Microservices status...
    docker-compose ps
) else if "%command%"=="health" (
    echo 🔍 Checking Microservices health...
    curl -s http://localhost:8080/health && echo ✅ Gateway healthy || echo ❌ Gateway not responding
    curl -s http://localhost:8001/health && echo ✅ Price Predictor healthy || echo ❌ Price Predictor not responding
    curl -s http://localhost:8002/health && echo ✅ Investment Expert healthy || echo ❌ Investment Expert not responding
    curl -s http://localhost:8010/health && echo ✅ LLM Hub healthy || echo ❌ LLM Hub not responding
) else if "%command%"=="clean" (
    echo 🧹 Cleaning Microservices System...
    docker-compose down -v 2>nul
    docker-compose -f docker-compose.dev.yml down -v 2>nul
    docker-compose -f docker-compose.production.yml down -v 2>nul
    echo ✅ Microservices cleaned
) else (
    echo ❌ Unknown command for microservices: %command%
    goto :help_microservices
)
goto :end

:status_all
echo 📊 System Status Overview
echo ========================
echo.
echo 🏠 Original System (Port 8501):
cd SRC
docker-compose ps 2>nul || echo "   Not running"
echo.
echo 🏗️ Microservices System (Port 8502):
cd ..\SRC\microservices
docker-compose ps 2>nul || echo "   Not running"
echo.
echo 🐳 All Docker Containers:
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.
echo 💾 Docker Resources:
docker system df
goto :end

:stop_all
echo 🛑 Stopping All Systems
echo ======================
echo.
echo 1. Stopping Original System...
cd SRC
docker-compose down 2>nul
echo.
echo 2. Stopping Microservices System...
cd ..\SRC\microservices
docker-compose down 2>nul
docker-compose -f docker-compose.dev.yml down 2>nul
docker-compose -f docker-compose.production.yml down 2>nul
echo.
echo ✅ All systems stopped
goto :end

:clean_all
echo 🧹 Cleaning All Systems
echo ======================
echo.
echo 1. Stopping all containers...
call :stop_all
echo.
echo 2. Cleaning Docker resources...
docker system prune -f
docker volume prune -f
echo.
echo ✅ Complete cleanup finished
goto :end

:help
echo 🔧 Multi-Agent Stock Analysis - Master Docker Manager
echo ===================================================
echo.
echo Usage: docker-master.bat ^<system^> [command]
echo.
echo Systems:
echo   original       Original monolithic system (SRC/)
echo   micro          Microservices system (SRC/microservices/)
echo   microservices  Alias for micro
echo.
echo Commands:
echo   start          Start system (default)
echo   dev            Start in development mode
echo   prod           Start in production mode
echo   basic          Start basic mode (microservices only)
echo   stop           Stop system
echo   logs           Show logs
echo   status         Show system status
echo   health         Check service health (microservices only)
echo   clean          Clean system resources
echo.
echo Global Commands:
echo   status         Show status of all systems
echo   stop-all       Stop all systems
echo   clean-all      Clean all Docker resources
echo   help           Show this help
echo.
echo Examples:
echo   %0 original start     # Start original system
echo   %0 micro dev          # Start microservices in dev mode
echo   %0 micro prod         # Start microservices in production
echo   %0 original stop      # Stop original system
echo   %0 status             # Show all systems status
echo   %0 stop-all           # Stop everything
echo.
echo 🌐 URLs:
echo   Original System:      http://localhost:8501
echo   Microservices:        http://localhost:8502
echo   Microservices API:    http://localhost:8080
echo.
goto :help_details

:help_original
echo.
echo 🏠 Original System Commands:
echo   %0 original start     # Quick start (production-ready)
echo   %0 original dev       # Development with hot reload
echo   %0 original prod      # Production optimized
echo   %0 original stop      # Stop services
echo   %0 original logs      # View logs
echo   %0 original status    # System status
echo   %0 original clean     # Cleanup resources
echo.
goto :end

:help_microservices
echo.
echo 🏗️ Microservices System Commands:
echo   %0 micro start        # Start with monitoring
echo   %0 micro dev          # Development with hot reload
echo   %0 micro prod         # Production with full monitoring
echo   %0 micro basic        # Basic setup
echo   %0 micro stop         # Stop all services
echo   %0 micro logs         # View logs
echo   %0 micro status       # Service status
echo   %0 micro health       # Health checks
echo   %0 micro clean        # Cleanup resources
echo.
goto :end

:help_details
echo 📋 Detailed Information:
echo.
echo 🏠 Original System (Monolithic):
echo   - Single container with all agents
echo   - Port: 8501
echo   - Best for: Quick start, testing, demo
echo   - Resource usage: ~4GB RAM
echo.
echo 🏗️ Microservices System (Distributed):
echo   - 6 separate services + monitoring
echo   - Ports: 8502, 8080, 8001, 8002, 8010
echo   - Best for: Production, scaling, enterprise
echo   - Resource usage: ~6GB RAM
echo.
echo 💡 Recommendations:
echo   - New users: Start with 'original'
echo   - Developers: Use 'micro dev'
echo   - Production: Use 'micro prod'
echo.

:end
if "%system%"=="help" pause
endlocal