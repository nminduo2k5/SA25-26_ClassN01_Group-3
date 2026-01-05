#!/bin/bash

# DUONG AI TRADING PRO - Docker Build & Run Script

echo "🚀 DUONG AI TRADING PRO - Docker Setup"
echo "======================================"

# Function to build Docker image
build_image() {
    echo "📦 Building Docker image..."
    docker build -t duong-ai-trading-pro .
    if [ $? -eq 0 ]; then
        echo "✅ Docker image built successfully!"
    else
        echo "❌ Failed to build Docker image"
        exit 1
    fi
}

# Function to run with docker-compose
run_compose() {
    echo "🐳 Starting with docker-compose..."
    docker-compose up -d
    if [ $? -eq 0 ]; then
        echo "✅ Application started successfully!"
        echo "🌐 Access at: http://localhost:8501"
    else
        echo "❌ Failed to start application"
        exit 1
    fi
}

# Function to run standalone Docker
run_docker() {
    echo "🐳 Starting Docker container..."
    docker run -d \
        --name duong-ai-trading-pro \
        -p 8501:8501 \
        --env-file .env \
        duong-ai-trading-pro
    
    if [ $? -eq 0 ]; then
        echo "✅ Container started successfully!"
        echo "🌐 Access at: http://localhost:8501"
    else
        echo "❌ Failed to start container"
        exit 1
    fi
}

# Function to stop containers
stop_containers() {
    echo "🛑 Stopping containers..."
    docker-compose down 2>/dev/null || docker stop duong-ai-trading-pro 2>/dev/null
    docker rm duong-ai-trading-pro 2>/dev/null
    echo "✅ Containers stopped"
}

# Function to show logs
show_logs() {
    echo "📋 Showing logs..."
    docker-compose logs -f 2>/dev/null || docker logs -f duong-ai-trading-pro 2>/dev/null
}

# Main menu
case "$1" in
    "build")
        build_image
        ;;
    "run")
        build_image
        run_compose
        ;;
    "start")
        run_compose
        ;;
    "stop")
        stop_containers
        ;;
    "logs")
        show_logs
        ;;
    "restart")
        stop_containers
        sleep 2
        run_compose
        ;;
    *)
        echo "Usage: $0 {build|run|start|stop|logs|restart}"
        echo ""
        echo "Commands:"
        echo "  build   - Build Docker image only"
        echo "  run     - Build and start application"
        echo "  start   - Start application (image must exist)"
        echo "  stop    - Stop application"
        echo "  logs    - Show application logs"
        echo "  restart - Restart application"
        echo ""
        echo "Example: ./docker-run.sh run"
        exit 1
        ;;
esac